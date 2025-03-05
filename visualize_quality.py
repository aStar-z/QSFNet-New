#此文件为质量评分分布可视化绘图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('./quality_scores.csv')

# 根据标签IoU划分高质量/低质量组（假设IoU>0.7为高质量）
df['Quality_Group'] = ['High' if iou > 0.7 else 'Low' for iou in df['Label_IoU']]

# 绘制深度图清晰度分布
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Depth_Quality', hue='Quality_Group', kde=True)
plt.title('Depth Image Sharpness Distribution')
plt.savefig('./depth_quality_distribution.png')

# 绘制热成像对比度分布
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Thermal_Quality', hue='Quality_Group', kde=True)
plt.title('Thermal Image Contrast Distribution')
plt.savefig('./thermal_quality_distribution.png')