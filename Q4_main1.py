import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Q1_main import arc_fit
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('Figures', exist_ok=True)

# 读取数据
xls = pd.ExcelFile('第2周C题：接触式轮廓仪的自动标注(2020年D题)/附件3_工件2的局部测量数据（圆）.xlsx')

# 读取所有sheet到字典
sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names[:10]}

# 存储平移后的数据
translated_sheets = {}

# 计算每个sheet的最低点并进行平移
for sheet_name, df in sheets.items():
    # 获取X和Z列数据
    x_data = df.iloc[:, 0]
    z_data = df.iloc[:, 1]

    # 找到Z值最小的点（最低点）
    min_z = z_data.min()

    # 平移数据，使最低点移到Z=0的位置
    translated_z = z_data - min_z

    # 保存平移后的数据
    translated_sheets[sheet_name] = (x_data, translated_z)

# 绘制平移前的数据
plt.figure(figsize=(12, 8))
for sheet_name, df in sheets.items():
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=sheet_name, alpha=0.5)
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('平移前的所有数据')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Figures/Q4(1)_平移前的所有数据.png')
plt.show()

# # 绘制平移后的所有数据（最低点对齐）
# plt.figure(figsize=(12, 8))
# for sheet_name, (x_data, translated_z) in translated_sheets.items():
#     plt.plot(x_data, translated_z, label=sheet_name)
#
# # 标记零点线（所有最低点将对齐到这条线）
# plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='零点（对齐线）')
# plt.legend(loc='best')
# plt.xlabel('X')
# plt.ylabel('Z（平移后）')
# plt.title('最低点对齐后的所有数据')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('Figures/Q4(1)_最低点对齐后的所有数据.png')
# plt.show()

# 标记弧线区间
arc_interval = [37, 45]
global_x_min = arc_interval[0]
global_x_max = arc_interval[1]
# 2. 创建参考x轴（在全局范围内均匀采样）
num_samples = 10000
reference_x = np.linspace(global_x_min, global_x_max, num_samples)

# 3. 为每个参考x点匹配所有工作表的z值并计算平均
average_z = []

for ref_x in reference_x:
    z_matches = []

    # 遍历每个工作表，找到最接近参考x的z值
    for sheet in translated_sheets.keys():
        # 获取当前工作表的x和z数据
        x_vals = translated_sheets[sheet][0]
        z_vals = translated_sheets[sheet][1]

        # 找到最接近参考x的索引
        closest_idx = np.argmin(np.abs(x_vals - ref_x))
        # 收集对应的z值
        z_matches.append(z_vals[closest_idx])

    # 计算当前参考x点的平均z值
    avg_z = np.mean(z_matches)
    average_z.append(avg_z)

# 4. 生成结果数据框
result_df = pd.DataFrame({
    'x': reference_x,
    'z': average_z
})

result_df.to_excel('Figures/Q4(1)_平移并平均后的弧线数据.xlsx', index=False)

arc=arc_fit(arc_interval[0],arc_interval[1],result_df)
print('圆参数:',arc)

def plot_full_circle(arc_params, color, label, name):
    h, k, r = arc_params
    theta = np.linspace(0, 2 * np.pi, 100)
    x = h + r * np.cos(theta)
    y = k + r * np.sin(theta)
    plt.plot(x, y, color=color, linestyle='--', linewidth=2, label=label)
    plt.scatter(h, k, color='salmon', s=20)
    plt.text(h, k + 0.12, name, fontsize=20, ha='center')

# 绘制平移后的所有数据（最低点对齐）
plt.figure(figsize=(12, 8))
for sheet_name, (x_data, translated_z) in translated_sheets.items():
    plt.plot(x_data, translated_z, label=sheet_name)
# 标记弧线区间
plt.plot(result_df['x'], result_df['z'], label='平均后', color='blue',linewidth=3)
plot_full_circle(arc, 'skyblue', '拟合圆', 'O1')
# plt.axvspan(arc_interval[0], arc_interval[1], color='gray', alpha=0.3, label='弧线区间')
# 标记零点线（所有最低点将对齐到这条线）
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='零点（对齐线）')
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Z（平移后）')
plt.title('最低点对齐后的所有数据')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Figures/Q4(1)_最低点对齐后的所有数据.png')
plt.show()




