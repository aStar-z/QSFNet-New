#第三周：动态权重可视化
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./fusion_weights.csv')

# 绘制权重分布箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Scene_Type', y='Weight_Depth')
plt.title('Depth Weight Distribution by Scene Type')
plt.savefig('./depth_weight_distribution.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Scene_Type', y='Weight_Thermal')
plt.title('Thermal Weight Distribution by Scene Type')
plt.savefig('./thermal_weight_distribution.png')