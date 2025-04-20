coding = 'utf-8'
import os
from Mnet import Mnet
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
import pytorch_iou
import cv2
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

from QAnet import QAnet

'''
Second Stage 
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IOU = pytorch_iou.IOU(size_average = True)

def bce_loss(pred,target):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')

    return bce


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def visualize_fusion_weights(weights_dict, epoch):
    """动态权重可视化函数"""
    os.makedirs('./vis_weights', exist_ok=True)

    # 转换为numpy数组
    depth_weights = np.concatenate([w[:, 0].cpu().numpy() for w in weights_dict.values()])
    thermal_weights = np.concatenate([w[:, 1].cpu().numpy() for w in weights_dict.values()])

    # 创建画布
    plt.figure(figsize=(15, 5))

    # 子图1：双权重的分布直方图
    plt.subplot(131)
    plt.hist(depth_weights, bins=30, alpha=0.5, label='Depth', color='blue')
    plt.hist(thermal_weights, bins=30, alpha=0.5, label='Thermal', color='red')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title(f'EPOCH {epoch} Weight Distribution')
    plt.legend()

    # 子图2：权重相关性散点图
    plt.subplot(132)
    plt.scatter(depth_weights, thermal_weights, alpha=0.6, c='green')
    plt.plot([0, 1], [1, 0], 'r--')  # 反比例对角线
    plt.xlabel('Depth Weight')
    plt.ylabel('Thermal Weight')
    plt.title('Cross-modal Weight Correlation')

    # 子图3：时间序列趋势（按样本顺序）
    plt.subplot(133)
    plt.plot(depth_weights[:500], 'b-', alpha=0.4, label='Depth')
    plt.plot(thermal_weights[:500], 'r-', alpha=0.4, label='Thermal')
    plt.xlabel('Sample Index')
    plt.ylabel('Weight Value')
    plt.title('Temporal Variation')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./vis_weights/epoch_{epoch}_weights.png')
    plt.close()

if __name__ == '__main__':

    # dataset
    img_root = './VDT-2048 dataset/Train/'
    out_path_PGT1 = './output_PGT1/'
    save_path_QA = './modelQA'
    if not os.path.exists(save_path_QA): os.mkdir(save_path_QA)
    lr = 0.0001
    batch_size = 4
    epoch = 200
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 8)
    net = Mnet().cuda()
    net.load_state_dict(torch.load('./model/final.pth'))
    qnet = QAnet().cuda()
    params = qnet.parameters()
    optimizer = torch.optim.Adam(params, lr, betas=(0.5, 0.999))
    for p in qnet.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    net.train()
    qnet.train()

    for epochi in tqdm(range(1, epoch + 1)):    

        prefetcher = DataPrefetcher(loader)
        rgb, t, d, eg, label = prefetcher.next()

        # 新增权重收集器
        weight_collector = []

        B, C, H, W = label.shape
        r_QA_loss = 0
        epoch_ave_loss = 0
        net.zero_grad()
        qnet.zero_grad()
        i = 0
        while rgb is not None:
            i += 1
            with torch.no_grad():
                x3e_pred, x2e_pred, x1e_pred, x3e_pred_t, x2e_pred_t, x1e_pred_t, x3e_pred_d, x2e_pred_d, x1e_pred_d,PGTD_P,PGTD_N,PGTT_P,PGTT_N = net(
                    rgb, t, d)
            '''
            x0e_pred_vdt_D, x0e_pred_vdt_T = qnet(rgb, t, d)

            #Computing PGTs of D and T branch
            PGTD = PGTD_P * label + PGTD_N * (1 - label)
            PGTT = PGTT_P * label + PGTT_N * (1 - label)

            #Supervised by PGT
            loss2 = bce_loss(x0e_pred_vdt_D, PGTD)
            loss1 = bce_loss(x0e_pred_vdt_T, PGTT)

            QA_loss = loss1 + loss2
            '''
            # 修改：QAnet返回融合后的预测
            # 新增修改（可视化）新增传入label参数
            #fused_pred = qnet(rgb, t, d, label)
            # 修改网络返回（假设qnet返回权重）
            fused_pred, fusion_weights, _, _ = qnet(rgb, t, d, label)  # 需要修改QAnet的forward返回值

            # 收集权重（每100个batch）
            if i % 100 == 0:
                weight_collector.append(fusion_weights.detach().cpu())

            # 计算损失（使用融合结果和真值label）
            QA_loss = bce_loss(fused_pred, label)  # 直接使用真值监督

            r_QA_loss += QA_loss.data
            QA_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if len(weight_collector) > 0:
                weights_dict = {'epoch': torch.cat(weight_collector, dim=0)}
                visualize_fusion_weights(weights_dict, epochi)

            #第三周：融合过程可视化
            # 在训练循环中保存示例图像
            # 修改后的代码段
            if not os.path.exists('./vis_results'):
                os.makedirs('./vis_results')
            if epochi%10==0 and i % 100 == 0:
                # RGB（三通道）
                rgb_sample = rgb[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                cv2.imwrite(f'./vis_results/batch_{i}_rgb.png', (rgb_sample * 255).astype(np.uint8))

                # 热成像（单通道）
                t_sample = t[0].cpu().numpy().squeeze()
                t_single_channel = t_sample.mean(axis=0) if len(t_sample.shape) == 3 else t_sample
                cv2.imwrite(f'./vis_results/batch_{i}_thermal.png', (t_single_channel * 255).astype(np.uint8))

                # 深度图（单通道）
                d_sample = d[0].cpu().numpy().squeeze()
                if len(d_sample.shape) == 3:
                    d_single_channel = d_sample[0]  # 取第一个通道
                else:
                    d_single_channel = d_sample
                cv2.imwrite(f'./vis_results/batch_{i}_depth.png', (d_single_channel * 255).astype(np.uint8))

                # 标签和预测（单通道）
                label_sample = label[0].cpu().numpy().squeeze()
                fused_pred_sample = torch.sigmoid(fused_pred[0]).detach().cpu().numpy().squeeze()
                cv2.imwrite(f'./vis_results/batch_{i}_label.png', (label_sample * 255).astype(np.uint8))
                cv2.imwrite(f'./vis_results/batch_{i}_pred.png', (fused_pred_sample * 255).astype(np.uint8))
            #第三周：保存预测结果

            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_QA_loss / 100, lr,))
                epoch_ave_loss += (r_QA_loss / 100)
                r_QA_loss = 0
            rgb, t, d,eg, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if epochi % 20 == 0:
            model_path = '%s/epoch_QA_%d.pth' % (save_path_QA, epochi)
            torch.save(qnet.state_dict(), '%s/epoch_QA_%d.pth' % (save_path_QA, epochi))


    torch.save(qnet.state_dict(), '%s/final.pth' % (save_path_QA))