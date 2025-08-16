import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from Q1_main import data_reader,angle_cal,line_intersection,line_circle_intersection,linear_fit,arc_fit


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('Figures', exist_ok=True)

#区间
#[45.155,61.25]
interval=[45.8,47.4,55.1,57.5,58.9,59.5,59.8,60.3,60.65,61.25]

#读取数据
df=pd.read_excel("Figures/Q3_Avg_Z_Results.xlsx")

# 转折点命名
turning_dict = {}
# 为每个区间点找到最近的x对应的z值
for i, x_val in enumerate(interval):
    # 计算每个x与目标值的绝对差
    differences = np.abs(df['x'] - x_val)

    # 找到最小差值的索引
    min_index = differences.idxmin()

    # 获取对应的z值
    corresponding_z = df.loc[min_index, 'z']

    # 存储到字典中，a为x值，b为对应的z值
    turning_dict[f'a{i + 1}'] = df.loc[min_index, 'x']
    turning_dict[f'b{i + 1}'] = corresponding_z

# 绘图
plt.figure(figsize=(18, 6))
plt.plot(df['x'], df['z'], color='gray', linewidth=3)
# 添加转折点
i=0
for i in range(len(interval)):
    plt.scatter(turning_dict[f'a{i + 1}'], turning_dict[f'b{i + 1}'], s=30, color='black',zorder=10)
    plt.text(turning_dict[f'a{i + 1}'], turning_dict[f'b{i + 1}'] + 0.08, f'a{i + 1}', fontsize=20)
    i+=1
plt.title('工件2的测量数据')
plt.xlabel('x')
plt.ylabel('z')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Figures/Q3_data.png')
plt.show()
plt.close()

# 片段拟合
print('区间[a1,a2]的拟合结果：')
line1=linear_fit(turning_dict['a1'],turning_dict['a2'],df, epsilon=0.1)
print('区间[a2,a3]的拟合结果：')
arc1=arc_fit(turning_dict['a2'],turning_dict['a3'],df,epsilon=0.2)
print('区间[a3,a4]的拟合结果：')
line2=linear_fit(turning_dict['a3'],turning_dict['a4'],df,epsilon=0.2)
print('区间[a4,a5]的拟合结果：')
line3=linear_fit(turning_dict['a4'],turning_dict['a5'],df,epsilon=0.1)
print('区间[a5,a6]的拟合结果：')
line4=linear_fit(turning_dict['a5'],turning_dict['a6'],df,epsilon=0.01)
print('区间[a6,a7]的拟合结果：')
line5=linear_fit(turning_dict['a6'],turning_dict['a7'],df,epsilon=0.01)
print('区间[a7,a8]的拟合结果：')
line6=linear_fit(turning_dict['a7'],turning_dict['a8'],df,epsilon=0.01)
print('区间[a8,a9]的拟合结果：')
line7=linear_fit(turning_dict['a8'],turning_dict['a9'],df,epsilon=0.01)
print('区间[a9,a10]的拟合结果：')
line8=linear_fit(turning_dict['a9'],turning_dict['a10'],df,epsilon=0.01)

#更新转折点
point=line_circle_intersection(line1,arc1)[0]
turning_dict['a2']=point[0]
turning_dict['b2']=point[1]
point=line_circle_intersection(line2,arc1)[1]
turning_dict['a3']=point[0]
turning_dict['b3']=point[1]
point=line_intersection(line2,line3)
turning_dict['a4']=point[0]
turning_dict['b4']=point[1]
point=line_intersection(line3,line4)
turning_dict['a5']=point[0]
turning_dict['b5']=point[1]
point=line_intersection(line4,line5)
turning_dict['a6']=point[0]
turning_dict['b6']=point[1]
point=line_intersection(line5,line6)
turning_dict['a7']=point[0]
turning_dict['b7']=point[1]
point=line_intersection(line6,line7)
turning_dict['a8']=point[0]
turning_dict['b8']=point[1]
point=line_intersection(line7,line8)
turning_dict['a9']=point[0]
turning_dict['b9']=point[1]

# 打印转折点列表
print("转折点列表")
for i in range(10):
    print(f'(a{i + 1},b{i + 1})=({turning_dict[f'a{i + 1}']:.8f},{turning_dict[f'b{i + 1}']:.8f})')
print("圆参数")
h, k, r = arc1
print(f'O1=({h:.8f},{k:.8f}) 半径={r:.8f}')

#绘制总图
def plot_all_results(df):
    plt.figure(figsize=(18, 8))

    # 1. 绘制原始数据
    plt.plot(df['x'], df['z'], color='gray', linewidth=4, label='原始数据')

    # 2. 绘制所有直线段拟合曲线
    def plot_fitted_line(line, start, end, line_name, color='skyblue'):
        slope, _, intercept = line
        x_fit = np.linspace(start, end, 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=3, label=line_name)

    plot_fitted_line(line1, turning_dict['a1'], turning_dict['a2'], 'line1')
    plot_fitted_line(line2, turning_dict['a3'], turning_dict['a4'], 'line2')
    plot_fitted_line(line3, turning_dict['a4'], turning_dict['a5'], 'line3')
    plot_fitted_line(line4, turning_dict['a5'], turning_dict['a6'], 'line4')
    plot_fitted_line(line5, turning_dict['a6'], turning_dict['a7'], 'line5')
    plot_fitted_line(line6, turning_dict['a7'], turning_dict['a8'], 'line6')
    plot_fitted_line(line7, turning_dict['a8'], turning_dict['a9'], 'line7')
    plot_fitted_line(line8, turning_dict['a9'], turning_dict['a10'], 'line8')

    # 3. 绘制所有完整圆
    def plot_full_circle(arc_params, color, label, name):
        h, k, r = arc_params
        theta = np.linspace(0, 2 * np.pi, 100)
        x = h + r * np.cos(theta)
        y = k + r * np.sin(theta)
        plt.plot(x, y, color=color, linestyle='--', linewidth=3, label=label)
        plt.scatter(h, k, color='salmon', s=20)
        plt.text(h, k + 0.12, name, fontsize=20, ha='center')

    # 绘制所有弧线
    plot_full_circle(arc1, 'pink', 'arc1', 'O1')

    # 4. 绘制所有转折点
    for i in range(1, 11):
        if f'a{i}' in turning_dict:
            plt.scatter(turning_dict[f'a{i}'], turning_dict[f'b{i}'], color='black', s=30, zorder=10)
            plt.text(turning_dict[f'a{i}'], turning_dict[f'b{i}'], f'a{i}', fontsize=24, ha='center')

    # 5. 添加图例和标签
    plt.legend(fontsize=20, loc='upper right')
    plt.title('Q3_data_fit', fontsize=24)
    plt.xlabel('x坐标', fontsize=20)
    plt.ylabel('z坐标', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 6. 保存和显示图像
    plt.savefig('Figures/Q3_data_fit.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_all_results(df)

#计算各拟合片段之间的长度
print("轮廓片段之间长度：")
line_length={}
for i in range(1,10):
    line_length[f'line{i}']=np.sqrt((turning_dict[f'a{i+1}']-turning_dict[f'a{i}'])**2+(turning_dict[f'b{i+1}']-turning_dict[f'b{i}'])**2)
    print(f'(a{i },a{i + 1})={line_length[f'line{i}']:.8f}')

# 获取圆弧参数
h, k, r = arc1

# 获取转折点坐标
a2_x, a2_z = turning_dict['a2'], turning_dict['b2']
a3_x, a3_z = turning_dict['a3'], turning_dict['b3']

# 计算两点相对于圆心的角度
theta1 = math.atan2(a2_z - k, a2_x - h)
theta2 = math.atan2(a3_z - k, a3_x - h)

# 计算圆弧角度（确保角度为正）
delta_theta = abs(theta2 - theta1)
if delta_theta > math.pi:
    delta_theta = 2 * math.pi - delta_theta

# 计算圆弧长度
arc_length = r * delta_theta
print(f'圆弧长度(a2,a3)={arc_length:.8f}')
