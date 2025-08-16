from Q1_main import line_intersection, line_circle_intersection, linear_fit, arc_fit, mixed_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

#设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
#设置负号
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('Figures', exist_ok=True)

def angle_search(*lines):
    slopes = [line[0] for line in lines]
    return np.mean([math.atan(s) for s in slopes])

def coordinate_tranaform(df,angle):
    df['x_new'] = df['x'] * np.cos(angle) - df['z'] * np.sin(angle)
    df['z_new'] = df['x'] * np.sin(angle) + df['z'] * np.cos(angle)
    df['x'] = df['x_new'].round(8)
    df['z'] = df['z_new'].round(8)

    df.drop(columns=['x_new', 'z_new'], inplace=True)
    df.to_excel('Figures/工件1_down_new.xlsx', sheet_name='down', index=False)
    return df

def plot_all_results(df):
    plt.figure(figsize=(24, 10))

    # 1. 绘制原始数据
    plt.plot(df['x'], df['z'], color='gray', linewidth=5, label='原始数据')

    # 2. 绘制所有直线段拟合曲线
    def plot_fitted_line(line, start, end, line_name, color='skyblue'):
        slope, _, intercept = line
        x_fit = np.linspace(start, end, 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=3, label=line_name)

    plot_fitted_line(line1, df['x'][0], turning_dict['a1'], 'line1')
    plot_fitted_line(line2, turning_dict['a2'], turning_dict['a3'], 'line2')
    plot_fitted_line(line3, turning_dict['a4'], turning_dict['a5'], 'line3')
    plot_fitted_line(line4, turning_dict['a6'], turning_dict['a7'], 'line4')
    plot_fitted_line(line5, turning_dict['a7'], turning_dict['a8'], 'line5')
    plot_fitted_line(line6, turning_dict['a8'], turning_dict['a9'], 'line6')
    plot_fitted_line(line7, turning_dict['a9'], turning_dict['a10'], 'line7')
    plot_fitted_line(line8, turning_dict['a11'], turning_dict['a12'], 'line8')
    plot_fitted_line(line9, turning_dict['a13'], turning_dict['a14'], 'line9')
    plot_fitted_line(line10, turning_dict['a15'], turning_dict['a16'], 'line10')
    plot_fitted_line(line11, turning_dict['a17'], df['x'].iloc[-1], 'line11')

    plot_fitted_line(mixed_line1, turning_dict['a1'], turning_dict['a18'], 'mixed_line1', color='blue')
    plot_fitted_line(mixed_line2, turning_dict['a19'], turning_dict['a2'], 'mixed_line2', color='blue')
    plot_fitted_line(mixed_line3, turning_dict['a3'], turning_dict['a20'], 'mixed_line3', color='blue')
    plot_fitted_line(mixed_line4, turning_dict['a21'], turning_dict['a4'], 'mixed_line4', color='blue')
    plot_fitted_line(mixed_line5, turning_dict['a5'], turning_dict['a22'], 'mixed_line5', color='blue')
    plot_fitted_line(mixed_line6, turning_dict['a23'], turning_dict['a6'], 'mixed_line6', color='blue')

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
    plot_full_circle(arc1, 'pink', 'arc1', 'O4')
    plot_full_circle(arc2, 'pink', 'arc2', 'O5')
    plot_full_circle(arc3, 'pink', 'arc3', 'O6')
    plot_full_circle(arc4, 'pink', 'arc4', 'O7')

    plot_full_circle(mixed_arc1, 'salmon', 'mixed_arc1', 'O1')
    plot_full_circle(mixed_arc2, 'salmon', 'mixed_arc2', 'O2')
    plot_full_circle(mixed_arc3, 'salmon', 'mixed_arc3', 'O3')

    # 4. 绘制所有转折点
    for i in range(1, 24):
        if f'a{i}' in turning_dict:
            plt.scatter(turning_dict[f'a{i}'], turning_dict[f'b{i}'], color='black', s=20, zorder=10)
            if i in [18, 20, 22]:  # 左侧点向左偏移
                plt.text(turning_dict[f'a{i}'] - 0.1, turning_dict[f'b{i}'],
                         f'a{i}', fontsize=20, ha='right')
            elif i in [19, 21, 23]:  # 右侧点向右偏移
                plt.text(turning_dict[f'a{i}'] + 0.1, turning_dict[f'b{i}'],
                         f'a{i}', fontsize=20, ha='left')
            else:  # 其他点保持原样
                plt.text(turning_dict[f'a{i}'], turning_dict[f'b{i}'] + 0.05,
                         f'a{i}', fontsize=20, ha='center')

    # 5. 添加图例和标签
    plt.legend(fontsize=15, loc='upper right')
    plt.title('工件1_down_fit', fontsize=24)
    plt.xlabel('x坐标', fontsize=20)
    plt.ylabel('z坐标', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 6. 保存和显示图像
    plt.savefig('Figures/工件1_down_fit.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 读取数据
    df = pd.read_excel('第2周C题：接触式轮廓仪的自动标注(2020年D题)/附件1_工件1的测量数据.xlsx', sheet_name='down')
    # 标记转折点
    turning_points = ['(52.2842,0.9468)', '(55.1646,0.5689)', '(60.0920,-0.0733)', '(62.1717,-0.3424)',
                      '(67.1081,-0.9914)', '(69.1762,-1.2660)', '(74.1437,-1.9026)','(79.2127,-1.5718)',
                      '(84.1239,-3.2049)','(87.0161,-3.5982)', '(88.8237,-3.8364)', '(89.7228,-3.9415)',
                      '(91.7218,-4.1919)','(96.5639,-4.8494)', '(103.7796,-5.7907)', '(108.6285,-6.4093)',
                      '(116.0250,-7.3731)'
                      ]
    turning_points_x = []
    turning_points_z = []
    for item in turning_points:
        coord = item.strip('()').split(',')
        turning_points_x.append(float(coord[0]))
        turning_points_z.append(float(coord[1]))
    # 转折点命名
    turning_dict = {}
    for i in range(len(turning_points)):
        turning_dict[f'a{i + 1}'] = turning_points_x[i]
        turning_dict[f'b{i + 1}'] = turning_points_z[i]

    # 绘图
    plt.figure(figsize=(24, 10))
    plt.plot(df['x'], df['z'], color='gray', linewidth=5)
    # 添加转折点
    plt.scatter(turning_points_x, turning_points_z, color='gray')
    for i in range(len(turning_points)):
        plt.text(turning_points_x[i], turning_points_z[i] + 0.08, f'a{i + 1}', fontsize=20)

    # 直线区段拟合
    print("区间[,a1]的拟合结果")
    line1 = linear_fit(df['x'][0], turning_dict['a1'], df)
    print('区间[a2,a3]的拟合结果：')
    line2 = linear_fit(turning_dict['a2'], turning_dict['a3'], df)
    print('区间[a4,a5]的拟合结果：')
    line3 = linear_fit(turning_dict['a4'], turning_dict['a5'], df)
    print('区间[a6,a7]的拟合结果：')
    line4 = linear_fit(turning_dict['a6'], turning_dict['a7'], df)
    print('区间[a9,a10]的拟合结果：')
    line7 = linear_fit(turning_dict['a9'], turning_dict['a10'], df)
    print('区间[a11,a12]的拟合结果：')
    line8 = linear_fit(turning_dict['a11'], turning_dict['a12'], df)
    print('区间[a13,a14]的拟合结果：')
    line9 = linear_fit(turning_dict['a13'], turning_dict['a14'], df)
    print('区间[a15,a16]的拟合结果：')
    line10 = linear_fit(turning_dict['a15'], turning_dict['a16'], df)
    print("区间[a17,]的拟合结果：")
    line11 = linear_fit(turning_dict['a17'], df['x'].iloc[-1], df)

    # 计算角度
    angle = -angle_search(line1, line2, line3, line4, line7, line8, line9, line10, line11)
    print(f'angle= {angle}')

    # 坐标变换
    df = coordinate_tranaform(df, angle)

    # 计算新的转折点坐标
    turning_points_x_new = []
    turning_points_z_new = []
    for i in range(len(turning_points)):
        turning_points_x_new.append(turning_points_x[i] * np.cos(angle) - turning_points_z[i] * np.sin(angle))
        turning_points_z_new.append(turning_points_x[i] * np.sin(angle) + turning_points_z[i] * np.cos(angle))
    turning_points_x=turning_points_x_new
    turning_points_z=turning_points_z_new

    # 转折点更新
    for i in range(len(turning_points)):
        turning_dict[f'a{i + 1}'] = turning_points_x[i]
        turning_dict[f'b{i + 1}'] = turning_points_z[i]

    # 绘图
    plt.plot(df['x'], df['z'], color='black', linewidth=3)
    # 添加转折点
    plt.scatter(turning_points_x, turning_points_z, color='black')
    for i in range(len(turning_points)):
        plt.text(turning_points_x[i], turning_points_z[i] + 0.08, f'a{i + 1}', fontsize=15)
    plt.title('工件1_down')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figures/工件1_down.png')
    plt.show()
    plt.close()

    # 直线区段拟合
    print("区间[,a1]的拟合结果")
    line1 = linear_fit(df['x'][0], turning_dict['a1'], df)
    print('区间[a2,a3]的拟合结果：')
    line2 = linear_fit(turning_dict['a2'], turning_dict['a3'], df)
    print('区间[a4,a5]的拟合结果：')
    line3 = linear_fit(turning_dict['a4'], turning_dict['a5'], df)
    print('区间[a6,a7]的拟合结果：')
    line4 = linear_fit(turning_dict['a6'], turning_dict['a7'],df)
    print('区间[a7,a8]的拟合结果：')
    line5 = linear_fit(turning_dict['a7'], turning_dict['a8'],df)
    print('区间[a8,a9]的拟合结果：')
    line6 = linear_fit(turning_dict['a8'], turning_dict['a9'],df)
    print('区间[a9,a10]的拟合结果：')
    line7 = linear_fit(turning_dict['a9'], turning_dict['a10'],df)
    print('区间[a11,a12]的拟合结果：')
    line8 = linear_fit(turning_dict['a11'], turning_dict['a12'],df)
    print('区间[a13,a14]的拟合结果：')
    line9 = linear_fit(turning_dict['a13'], turning_dict['a14'],df)
    print('区间[a15,a16]的拟合结果：')
    line10 = linear_fit(turning_dict['a15'], turning_dict['a16'],df)
    print("区间[a17,]的拟合结果：")
    line11 = linear_fit(turning_dict['a17'], df['x'].iloc[-1],df)

    # 更新转折点坐标
    point = line_intersection(line4, line5)
    turning_dict['a7'] = point[0]
    turning_dict['b7'] = point[1]
    point = line_intersection(line5, line6)
    turning_dict['a8'] = point[0]
    turning_dict['b8'] = point[1]
    point = line_intersection(line6, line7)
    turning_dict['a9'] = point[0]
    turning_dict['b9'] = point[1]

    # 计算z1
    z1 = turning_dict['b8'] - turning_dict['b9']
    print(f'z1={z1:.8f}\n')

    # 弧线区段拟合
    print('区间[a10,a11]的拟合结果：')
    arc1 = arc_fit(turning_dict['a10'], turning_dict['a11'],df)
    print('区间[a12,a13]的拟合结果：')
    arc2 = arc_fit(turning_dict['a12'], turning_dict['a13'],df)
    print('区间[a14,a15]的拟合结果：')
    arc3 = arc_fit(turning_dict['a14'], turning_dict['a15'],df)
    print('区间[a16,a17]的拟合结果：')
    arc4 = arc_fit(turning_dict['a16'], turning_dict['a17'],df)

    # 混合区段拟合
    print('区间[a1,a2]的拟合结果：')
    mixed_line1, mixed_line2, mixed_arc1 = mixed_fit(turning_dict['a1'], turning_dict['a2'], 7.5, 5.3, 5.2,df)
    print('区间[a3,a4]的拟合结果：')
    mixed_line3, mixed_line4, mixed_arc2 = mixed_fit(turning_dict['a3'], turning_dict['a4'], 7.7, 5.3, 5.2,df)
    print('区间[a5,a6]的拟合结果：')
    mixed_line5, mixed_line6, mixed_arc3 = mixed_fit(turning_dict['a5'], turning_dict['a6'], 7.7, 5.3, 5.3,df)

    # 更新转折点坐标
    point = line_intersection(line1, mixed_line1)
    turning_dict['a1'] = point[0]
    turning_dict['b1'] = point[1]
    point = line_intersection(line2, mixed_line2)
    turning_dict['a2'] = point[0]
    turning_dict['b2'] = point[1]
    point = line_intersection(line2, mixed_line3)
    turning_dict['a3'] = point[0]
    turning_dict['b3'] = point[1]
    point = line_intersection(mixed_line4, line3)
    turning_dict['a4'] = point[0]
    turning_dict['b4'] = point[1]
    point = line_intersection(mixed_line5, line3)
    turning_dict['a5'] = point[0]
    turning_dict['b5'] = point[1]
    point = line_intersection(mixed_line6, line4)
    turning_dict['a6'] = point[0]
    turning_dict['b6'] = point[1]
    point = line_circle_intersection(line7, arc1)[0]
    turning_dict['a10'] = point[0]
    turning_dict['b10'] = point[1]
    point = line_circle_intersection(line8, arc1)[1]
    turning_dict['a11'] = point[0]
    turning_dict['b11'] = point[1]
    point = line_circle_intersection(line8, arc2)[0]
    turning_dict['a12'] = point[0]
    turning_dict['b12'] = point[1]
    point = line_circle_intersection(line9, arc2)[1]
    turning_dict['a13'] = point[0]
    turning_dict['b13'] = point[1]
    point = line_circle_intersection(line9, arc3)[0]
    turning_dict['a14'] = point[0]
    turning_dict['b14'] = point[1]
    point = line_circle_intersection(line10, arc3)[1]
    turning_dict['a15'] = point[0]
    turning_dict['b15'] = point[1]
    point = line_circle_intersection(line10, arc4)[0]
    turning_dict['a16'] = point[0]
    turning_dict['b16'] = point[1]
    point = line_circle_intersection(line11, arc4)[1]
    turning_dict['a17'] = point[0]
    turning_dict['b17'] = point[1]
    point = line_circle_intersection(mixed_line1, mixed_arc1)
    turning_dict['a18'] = point[0]
    turning_dict['b18'] = point[1]
    point = line_circle_intersection(mixed_line2, mixed_arc1)
    turning_dict['a19'] = point[0]
    turning_dict['b19'] = point[1]
    point = line_circle_intersection(mixed_line3, mixed_arc2)
    turning_dict['a20'] = point[0]
    turning_dict['b20'] = point[1]
    point = line_circle_intersection(mixed_line4, mixed_arc2)
    turning_dict['a21'] = point[0]
    turning_dict['b21'] = point[1]
    point = line_circle_intersection(mixed_line5, mixed_arc3)
    turning_dict['a22'] = point[0]
    turning_dict['b22'] = point[1]
    point = line_circle_intersection(mixed_line6, mixed_arc3)
    turning_dict['a23'] = point[0]
    turning_dict['b23'] = point[1]

    # 打印转折点列表
    print("转折点列表")
    for i in range(23):
        print(f'(a{i + 1},b{i + 1})=({turning_dict[f'a{i + 1}']:.8f},{turning_dict[f'b{i + 1}']:.8f})')
    print("圆参数")
    for i, arc in enumerate([mixed_arc1, mixed_arc2, mixed_arc3], 1):
        h, k, r = arc
        print(f'O{i}=({h:.8f},{k:.8f}) 半径={r:.8f}')
    for i, arc in enumerate([arc1, arc2, arc3, arc4], 4):
        h, k, r = arc
        print(f'O{i}=({h:.8f},{k:.8f}) 半径={r:.8f}')

    # 绘制总图
    plot_all_results(df)