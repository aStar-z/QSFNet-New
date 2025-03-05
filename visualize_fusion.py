#第三周：融合过程可视化-生成对比图
import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    return overlay

# 读取示例数据
rgb = cv2.imread('./vis_results/batch_100_rgb.png')
thermal = cv2.imread('./vis_results/batch_100_thermal.png', 0)
depth = cv2.imread('./vis_results/batch_100_depth.png', 0)
label = cv2.imread('./vis_results/batch_100_label.png', 0)
pred = cv2.imread('./vis_results/batch_100_pred.png', 0)

# 生成热力图叠加
rgb_overlay = overlay_heatmap(rgb, pred)

# 绘制对比图
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1), plt.imshow(rgb), plt.title('RGB')
plt.subplot(2, 3, 2), plt.imshow(thermal, cmap='gray'), plt.title('Thermal')
plt.subplot(2, 3, 3), plt.imshow(depth, cmap='gray'), plt.title('Depth')
plt.subplot(2, 3, 4), plt.imshow(label, cmap='gray'), plt.title('Label')
plt.subplot(2, 3, 5), plt.imshow(pred, cmap='gray'), plt.title('Prediction')
plt.subplot(2, 3, 6), plt.imshow(rgb_overlay), plt.title('RGB + Prediction')
plt.savefig('./fusion_visualization.png')