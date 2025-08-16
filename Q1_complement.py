import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('Figures/工件1_down_new.xlsx')

# 读取x列数据
x = df['x']

# 选取区间内部的点
points_x=[(51.71590531, 54.62098921),(59.59006041,61.68709433),(66.66597425,68.75222298)]
start, end = points_x[2]
start_idx = np.searchsorted(x, start, side='left')
end_idx = np.searchsorted(x, end, side='right')

# 提取指定区间的z值
z_selected = df['z'].iloc[start_idx:end_idx + 1].values

# 自定义间隔距离（可修改为任意正数）
interval = 0.1

# 计算区间边界（确保覆盖所有数据）
min_z = np.floor(np.min(z_selected) / interval) * interval  # 向下取整到间隔的倍数
max_z = np.ceil(np.max(z_selected) / interval) * interval  # 向上取整到间隔的倍数

# 生成自定义间隔的区间
bins = np.arange(min_z, max_z + interval, interval)

# 统计每个区间的点数量
counts, edges = np.histogram(z_selected, bins=bins)

# 打印结果
print(f"间隔距离: {interval}")
print("z值区间 | 点的个数")
print("-" * 20)
for i in range(len(counts)):
    # 格式化区间显示，根据间隔精度调整小数位数
    decimal_places = max(str(interval).count('.'), 1)  # 自动适配间隔的小数位数
    interval_str = f"[{edges[i]:.{decimal_places}f}, {edges[i + 1]:.{decimal_places}f})"
    print(f"{interval_str} | {counts[i]}")
print("-" * 20)