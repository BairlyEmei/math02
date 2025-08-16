import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.traversal import dfs_successors

# # 创建ExcelFile对象
# xls = pd.ExcelFile('第2周C题：接触式轮廓仪的自动标注(2020年D题)/附件2_工件2的整体测量数据.xlsx')
#
# # 读取所有sheet到字典
# sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names[:10]}
#
# # 存储所有原始数据点，用于后续查找最近点
# all_data = []
# for sheet_name, df in sheets.items():
#     x_data = df.iloc[:, 0].values
#     z_data = df.iloc[:, 1].values
#     all_data.extend(list(zip(x_data, z_data, [sheet_name] * len(x_data))))

df=pd.read_excel("Figures/Q3_Avg_Z_Results.xlsx")
all_data=df.values

# 绘制图形
plt.figure(figsize=(12, 8))
plt.plot(all_data[:,0],all_data[:,1])

# 遍历所有sheet并绘图
# for sheet_name, df in sheets.items():
#     # 绘制x和z列
#     plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=sheet_name)



# 添加图例和标签
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('All Sheets Data')
plt.grid(True)

# 获取当前坐标轴的范围
xlim = plt.xlim()
ylim = plt.ylim()

# 显示图形并等待用户点击
print("请在图上点击以选择点（点击右键结束）...")
points = plt.ginput(n=10, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)

# 处理每个点击的点
for pixel_x, pixel_y in points:
    # 将像素坐标转换为数据坐标
    ax = plt.gca()
    inv = ax.transData.inverted()
    data_coords = inv.transform((pixel_x, pixel_y))
    data_x, data_y = data_coords

    # 在原始数据中查找最接近的点
    min_distance = float('inf')
    closest_point = None
    closest_sheet = None

    for x, z, sheet in all_data:
        distance = np.sqrt((x - data_x) ** 2 + (z - data_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x, z)
            closest_sheet = sheet

    # 标记选中的点及其对应的原始数据点
    plt.plot(data_x, data_y, 'ro', alpha=0.5)  # 半透明红点表示点击位置
    plt.plot(closest_point[0], closest_point[1], 'bo')  # 蓝点表示原始数据点
    plt.text(closest_point[0], closest_point[1],
             f'({closest_point[0]:.2f}, {closest_point[1]:.2f})\n{closest_sheet}',
             color='blue', backgroundcolor='white')

    print(f'点击位置像素坐标: ({pixel_x:.2f}, {pixel_y:.2f})')
    print(f'对应数据坐标: ({data_x:.2f}, {data_y:.2f})')
    print(f'最接近的原始数据点: {closest_point} 来自工作表: {closest_sheet}')
    print('---')

# 显示最终图形
plt.show()
