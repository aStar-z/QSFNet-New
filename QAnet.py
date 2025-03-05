import torch
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from torchvision.models import resnet34 as resnet

'''
Second Stage   
QAnet = Quality-aware Region Selection Subnet
'''

class QAnet(nn.Module):
    def __init__(self):
        super(QAnet, self).__init__()
        self.resnet_D = resnet()
        self.resnet_T = resnet()
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        ####新增内容
        self.quality_weight_network = nn.Sequential(
            nn.Linear(2, 32),  # 输入：Depth和Thermal的质量评分
            nn.ReLU(),
            nn.Linear(32, 2),  # 输出：Depth和Thermal的权重
            nn.Softmax(dim=1)
        )


##################################################################################################################
        self.conv4_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv1_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv0_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_4_vdt_D = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_3_vdt_D = nn.Sequential(
            Conv(128, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_2_vdt_D = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1_vdt_D = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_0_vdt_D = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )



##########################################################################################################################
        self.conv4_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv1_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv0_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_4_vdt_T = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_3_vdt_T = nn.Sequential(
            Conv(128, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_2_vdt_T = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1_vdt_T = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_0_vdt_T = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.resconvV1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.resconvD1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.resconvT1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    #第二周工作
    # ------------------------- 新增质量评估函数 -------------------------

    def calculate_sharpness_quality(self, image):
        # 输入：单通道图像张量 (B, 1, H, W)
        batch_size = image.size(0)
        sharpness_scores = []
        for i in range(batch_size):
            # 提取单张图像并转换为二维数组
            img = image[i].squeeze().cpu().detach().numpy()  # 形状 (H, W)
            img = (img * 255).astype(np.uint8)  # 归一化到 0-255 并转为 uint8
            # 计算梯度
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            sharpness = np.mean(magnitude)
            sharpness_scores.append(sharpness / 255.0)  # 归一化到 0-1
        # 返回形状为 (B, 1) 的张量
        #return torch.tensor(sharpness_scores, device=image.device).unsqueeze(1)
        return torch.tensor(sharpness_scores, device=image.device, dtype=torch.float32).unsqueeze(1)

    def calculate_contrast_quality(self, image):
        batch_size = image.size(0)
        contrast_scores = []
        for i in range(batch_size):
            img = image[i].squeeze().cpu().detach().numpy()
            img = (img * 255).astype(np.uint8)
            mean = np.mean(img)
            contrast = np.mean(np.abs(img - mean))
            contrast_scores.append(contrast / 255.0)
        #return torch.tensor(contrast_scores, device=image.device).unsqueeze(1)
        return torch.tensor(contrast_scores, device=image.device, dtype=torch.float32).unsqueeze(1)
    #以上为第二周工作：质量感知机制调整

    def forward(self, rgb, t, d,label):#在可视化验证中新加入了label参数

        vdt = torch.cat((rgb, t, d), dim=1)
        #D Branch
        #Encoder
        x_u_D = self.resconvD1(vdt)
        x_u_D = self.resnet_D.bn1(x_u_D)
        x_u_D = self.resnet_D.relu(x_u_D)

        x0_vdt_D = self.resnet_D.maxpool(x_u_D)
        x1_vdt_D = self.resnet_D.layer1(x0_vdt_D)
        x2_vdt_D = self.resnet_D.layer2(x1_vdt_D)
        x3_vdt_D = self.resnet_D.layer3(x2_vdt_D)
        x4_vdt_D = self.resnet_D.layer4(x3_vdt_D)
        #Decoder
        x4_vdt_D = self.up2(x4_vdt_D)
        x4e_vdt_D = self.conv4_vdt_D(x4_vdt_D)

        x3e_vdt_D = self.conv3_vdt_D(torch.cat((x4e_vdt_D, x3_vdt_D), dim=1))
        x3e_vdt_D = self.up2(x3e_vdt_D)

        x2e_vdt_D = self.conv2_vdt_D(torch.cat((x3e_vdt_D, x2_vdt_D), dim=1))
        x2e_vdt_D = self.up2(x2e_vdt_D)

        x1e_vdt_D = self.conv1_vdt_D(torch.cat((x2e_vdt_D, x1_vdt_D), dim=1))
        x1e_vdt_D = self.up2(x1e_vdt_D)

        x0e_vdt_D = self.conv0_vdt_D(torch.cat((x1e_vdt_D, x_u_D), dim=1))
        x0e_vdt_D = self.up2(x0e_vdt_D)
        x0e_pred_vdt_D = self.final_0_vdt_D(x0e_vdt_D)

        #T Branch
        x_u_T = self.resconvT1(vdt)
        x_u_T = self.resnet_T.bn1(x_u_T)
        x_u_T = self.resnet_T.relu(x_u_T)

        x0_vdt_T = self.resnet_T.maxpool(x_u_T)
        x1_vdt_T = self.resnet_T.layer1(x0_vdt_T)
        x2_vdt_T = self.resnet_T.layer2(x1_vdt_T)
        x3_vdt_T = self.resnet_T.layer3(x2_vdt_T)
        x4_vdt_T = self.resnet_T.layer4(x3_vdt_T)

        x4_vdt_T = self.up2(x4_vdt_T)
        x4e_vdt_T = self.conv4_vdt_T(x4_vdt_T)

        x3e_vdt_T = self.conv3_vdt_T(torch.cat((x4e_vdt_T, x3_vdt_T), dim=1))
        x3e_vdt_T = self.up2(x3e_vdt_T)

        x2e_vdt_T = self.conv2_vdt_T(torch.cat((x3e_vdt_T, x2_vdt_T), dim=1))
        x2e_vdt_T = self.up2(x2e_vdt_T)

        x1e_vdt_T = self.conv1_vdt_T(torch.cat((x2e_vdt_T, x1_vdt_T), dim=1))
        x1e_vdt_T = self.up2(x1e_vdt_T)

        x0e_vdt_T = self.conv0_vdt_T(torch.cat((x1e_vdt_T, x_u_T), dim=1))
        x0e_vdt_T = self.up2(x0e_vdt_T)
        x0e_pred_vdt_T = self.final_0_vdt_T(x0e_vdt_T)

        #第二周工作
        # 新增：质量评估
        quality_depth = self.calculate_contrast_quality(d)  # 深度图质量（对比度）
        quality_thermal = self.calculate_sharpness_quality(t)  # 热成像质量（清晰度）

        # 生成融合权重 (Batch_size, 2)
        quality_scores = torch.cat([quality_depth, quality_thermal], dim=1)
        quality_scores = quality_scores.to(torch.float32)  # 强制转换为 Float
        weights = self.quality_weight_network(quality_scores)  # 输出：[weight_D, weight_T]

        # 动态加权融合
        fused_pred = weights[:, 0].view(-1, 1, 1, 1) * x0e_pred_vdt_D +  weights[:, 1].view(-1, 1, 1, 1) * x0e_pred_vdt_T

        #return fused_pred  # 修改返回值为融合后的预测
        #return x0e_pred_vdt_D, x0e_pred_vdt_T
        #第二周工作

        #第三周工作 可视化验证
        # 新增：保存质量评分到CSV文件
        import pandas as pd
        import os

        # 获取当前批次的数据索引（假设每个批次是独立的）
        batch_size = d.size(0)
        indices = np.arange(batch_size)  # 示例索引

        # 计算质量评分
        quality_depth = self.calculate_contrast_quality(d)  # (B, 1)
        quality_thermal = self.calculate_sharpness_quality(t)  # (B, 1)

        # 将数据转换为numpy数组
        #quality_depth_np = quality_depth.cpu().numpy().flatten()
        #quality_thermal_np = quality_thermal.cpu().numpy().flatten()
        quality_depth_np = quality_depth.detach().cpu().numpy().flatten()
        quality_thermal_np = quality_thermal.detach().cpu().numpy().flatten()
        #用于解决：RuntimeError: Can't call numpy() on Tensor that requires grad（以替换）

        # 修改前（报错：ValueError: All arrays must be of the same length）
        #label_np = label.cpu().numpy().flatten()  # 错误：展平后长度为 B*C*H*W
        # 修改后（假设标签为二值掩码，计算每个样本的均值作为简单参考）
        label_np = label.cpu().numpy().mean(axis=(1, 2, 3)).flatten()  # 形状 (B,)

        # 保存到CSV（按批次追加）
        df = pd.DataFrame({
            'Index': indices,
            'Depth_Quality': quality_depth_np,
            'Thermal_Quality': quality_thermal_np,
            'Label_IoU': label_np  # 假设用真值IoU作为质量参考
        })

        # 确保文件存在时追加，否则新建
        csv_path = './quality_scores.csv'
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
        #第三周工作1：以上内容为质量评分分布可视化

        # 新增：保存权重到CSV
        df_weights = pd.DataFrame({
            'Index': indices,
            '''
            'Weight_Depth': weights[:, 0].cpu().numpy().flatten(),
            'Weight_Thermal': weights[:, 1].cpu().numpy().flatten(),
            '''
            'Weight_Depth': weights[:, 0].detach().cpu().numpy().flatten(),  # 添加 detach()
            'Weight_Thermal': weights[:, 1].detach().cpu().numpy().flatten()  # 添加 detach()
            #错误原因:在保存权重数据到CSV时，weights 张量仍关联梯度计算图，直接调用 .numpy() 导致报错：
            #RuntimeError: Can't call numpy() on Tensor that requires grad

            #'Scene_Type': ['Case1', 'Case2', 'Case3']  # 需根据实际场景手动标记（错误，长度固定为3）
        })

        csv_weights_path = './fusion_weights.csv'
        if not os.path.exists(csv_weights_path):
            df_weights.to_csv(csv_weights_path, index=False)
        else:
            df_weights.to_csv(csv_weights_path, mode='a', header=False, index=False)
        #第三周工作2：动态权重可视化
        return fused_pred


    def load_pretrained_model(self):
        self.resnet_D.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        self.resnet_T.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        print('loading pretrained model success!')


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x