# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:07:25 2025

@author: maxim
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取原始数据（无标题模式）
df = pd.read_csv('xy.csv', header=None, names=['x', 'fx', 'y', 'fy'])

# 直接生成笛卡尔积
result = df[['x', 'fx']].merge(df[['y', 'fy']], how='cross')

# 计算总和并保留所需列
result['sum'] = (result['fx'] + result['fy']) / 2
result = result[['x', 'y', 'sum']]

# 保存满足条件的数据到文件（示例范围：x∈[1.8,1.85]，y∈[1.35,1.4]）
condition = (result['x'].between(1.8, 1.85)) & (result['y'].between(1.35, 1.4))
result[condition].to_csv('result_filtered.csv', index=False)
print(f"处理完成，总组合数：{len(result)}，已保存{condition.sum()}条过滤数据")

# 使用完整内存数据生成热力图
# 设置分箱参数
bin_width = 0.05

# 生成完整的分箱范围（确保覆盖所有可能值）
x_min = result['x'].min() // bin_width * bin_width
x_max = (result['x'].max() // bin_width + 1) * bin_width
y_min = result['y'].min() // bin_width * bin_width
y_max = (result['y'].max() // bin_width + 1) * bin_width

# 创建分箱标签
x_bins = pd.interval_range(start=x_min, end=x_max, freq=bin_width, closed='left')
y_bins = pd.interval_range(start=y_min, end=y_max, freq=bin_width, closed='left')

# 添加分箱列（直接在内存数据上操作）
result['x_bin'] = pd.cut(result['x'], bins=x_bins, include_lowest=True).astype("category")
result['y_bin'] = pd.cut(result['y'], bins=y_bins, include_lowest=True).astype("category")

# 生成完整的热力图数据（空值填充为0）
heatmap_data = result.groupby(['x_bin', 'y_bin'])['sum'].mean().unstack(fill_value=0)

# 可视化设置
plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    heatmap_data.T,  # 转置矩阵
    cmap='YlGnBu',
    cbar_kws={'label': 'Average Score'},
    annot=True,
    fmt=".1f",
    square=True
)
ax.invert_yaxis()

# 调整坐标标签
def format_interval(interval):
    return f"[{interval.left:.2f}, {interval.right:.2f})"

x_labels = [format_interval(iv) for iv in x_bins]
y_labels = [format_interval(iv) for iv in y_bins]

ax.set_xticks(ticks=range(len(x_labels))[::2])
ax.set_xticklabels(x_labels[::2], rotation=45)
ax.set_yticks(ticks=range(len(y_labels))[::2])
ax.set_yticklabels(y_labels[::2], rotation=0)

plt.xlabel('X Range')
plt.ylabel('Y Range')
plt.title(f'Score Distribution (Bin={bin_width}, Empty=0)')
plt.tight_layout()
plt.savefig('score_map_full_data.png', dpi=1200)
plt.show()
