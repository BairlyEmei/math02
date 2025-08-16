#This is a file for Q1.
import numpy as np
from numpy import argmin
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.optimize import least_squares

import os

#设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
#设置负号
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('Figures', exist_ok=True)

#区间数据读取器
def data_reader(start,end,df, epsilon):
    # 读取数据
    x = df['x']
    #选取区间内部的点
    start_idx = np.searchsorted(x, start+epsilon, side='left')
    end_idx = np.searchsorted(x, end-epsilon, side='right')
    x = df['x'].iloc[start_idx:end_idx + 1].values
    z = df['z'].iloc[start_idx:end_idx + 1].values
    return x,z

#角度计算器（大角）
def angle_cal(slope):
    if slope >= 0:
        angle_rad = np.pi-math.atan(slope)  # 弧度
        angle_deg = math.degrees(angle_rad)  # 角度
    else:
        angle_rad = np.pi+math.atan(slope)
        angle_deg = math.degrees(angle_rad)
    return angle_rad,angle_deg

#交点计算器
def line_intersection(line1, line2, tol=1e-6):
    """
    计算直线间的交点
    :param: 直线参数 (Ai, Bi, Ci)=(slope,-1,intercept) 表示 Ai*x + Bi*z + Ci = 0
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    # 计算行列式
    det = A1 * B2 - A2 * B1
    # 处理特殊情况
    if abs(det) < tol:
        # 检查是否重合
        if abs(A1 * C2 - A2 * C1) < tol and abs(B1 * C2 - B2 * C1) < tol:
            return "直线重合"
        else:
            return "直线平行"
    # 计算交点坐标
    x = (B1 * C2 - B2 * C1) / det
    z = (A2 * C1 - A1 * C2) / det
    return (x, z)

def line_circle_intersection(line, circle, tol=0.7):
    """
    计算直线与圆的交点
    :param line: 直线参数 (A, B, C)=(slope,-1,intercept) 表示 Ax + Bz + C = 0
    :param circle: 圆参数 (h, k, r) 圆心(h,k)和半径r
    """
    A, B, C = line
    h, k, r = circle
    # 特殊情况处理：垂直线 (B=0)
    if abs(B) < tol:
        # 垂直线方程：Ax + C = 0 → x = -C/A
        x0 = -C / A
        # 代入圆方程：(x0-h)^2 + (z-k)^2 = r^2
        d_sq = r ** 2 - (x0 - h) ** 2
        if d_sq < -tol:
            print("无交点")
            return ()  # 无交点
        elif abs(d_sq) < tol:
            print("相切")
            return (x0, k)  # 相切
        else:
            y1 = k + np.sqrt(d_sq)
            y2 = k - np.sqrt(d_sq)
            print("两个交点")
            return (x0, y1), (x0, y2)
    # 将直线表示为 z = mx + c
    m = -A / B
    c = -C / B
    # 展开为二次方程：ax^2 + bx + c = 0
    a = 1 + m ** 2
    b = 2 * (m * (c - k) - h)
    c_eq = h ** 2 + (c - k) ** 2 - r ** 2
    # 计算判别式
    delta = b ** 2 - 4 * a * c_eq
    print('delta=',delta)
    if delta < -tol:
        print("无交点")
        return ()  # 无交点
    elif abs(delta) < tol:
        print("相切")
        x = -b / (2 * a)
        y = m * x + c
        return (x, y)
    else:
        # 两个交点
        print("两个交点")
        sqrt_disc = np.sqrt(delta)
        x2 = (-b + sqrt_disc) / (2 * a)
        x1 = (-b - sqrt_disc) / (2 * a)
        return (x1, m * x1 + c), (x2, m * x2 + c)

#直线拟合器
def linear_fit(start,end,df, epsilon=0.2):
    # 读取数据
    x,z=data_reader(start,end,df, epsilon)

    # 计算线性拟合参数
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
    angle_rad,angle_deg=angle_cal(slope)

    # 计算评价指标
    y_pred = slope * x + intercept
    residuals = z - y_pred
    mae = np.mean(np.abs(residuals))  # 平均绝对误差
    rmse = np.sqrt(np.mean(residuals ** 2))  # 均方根误差

    print(f"拟合直线方程: z = {slope:.8f}x + {intercept:.8f}\n"
          f"R^2: {r_value ** 2:.8f}\n"
          f"p值: {p_value:.8f}\n"
          f"斜率标准误差: {std_err:.8f}\n"
          f"MAE: {mae:.8f}\n"
          f"RMSE: {rmse:.8f}\n"
          f"直线角度: {angle_deg:.4f}° (弧度: {angle_rad:.4f})\n"
          )
    return (slope,-1,intercept)

#弧线拟合器
def arc_fit(start,end,df, epsilon=0.2):
    # 读取数据
    x,z=data_reader(start,end,df, epsilon)
    # 初始参数估计
    # 使用数据的质心作为圆心初始估计
    h0 = np.mean(x)
    k0 = np.mean(z)
    # 计算到质心的平均距离作为半径初始估计
    r0 = np.sqrt(np.mean((x - h0) ** 2 + (z - k0) ** 2))

    #定义残差
    def residuals(params):
        h, k, r = params
        return np.sqrt((x - h) ** 2 + (z - k) ** 2) - r

    # 使用LM算法最小化残差平方和
    result = least_squares(
        fun=residuals,  # 残差函数
        x0=[h0, k0, r0],  # 初始参数
        method='lm',  # Levenberg-Marquardt方法
        ftol=1e-8,  # 函数容忍度
        xtol=1e-8,  # 参数容忍度
        max_nfev=100  # 最大函数评估次数
    )
    h_fit, k_fit, r_fit = result.x
    # 确保半径为正值
    r_fit = abs(r_fit)

    # 计算评价指标
    res = residuals((h_fit, k_fit, r_fit))
    rss = np.sum(res ** 2)  # 残差平方和
    sst = np.sum((z - np.mean(z)) ** 2)  # 总平方和
    r_squared = 1 - (rss / sst)  # 决定系数
    mae = np.mean(np.abs(res))  # 平均绝对误差
    rmse = np.sqrt(np.mean(res ** 2))  # 均方根误差

    print(f"拟合参数(h, k, r)={(h_fit, k_fit, r_fit)}\n"
          f"残差平方和(RSS): {rss:.8f}\n"
          f"R^2: {r_squared:.8f}\n"
          f"MAE: {mae:.8f}\n"
          f"RMSE: {rmse:.8f}\n")
    return (h_fit, k_fit, r_fit)

#混合拟合器
def mixed_fit(start,end,target_z1,target_z2,target_z3,df, epsilon=0.2):
    #获取区间
    x,z=data_reader(start,(start+end)/2,df, epsilon)
    # 获取原始数据的全局索引
    start_idx = np.searchsorted(df['x'], start+0.2, side='left')
    idx1 = start_idx + np.argmin(np.abs(z - target_z1))
    idx2 = start_idx + np.argmin(np.abs(z - target_z2))
    idx3 = start_idx + np.argmin(np.abs(z - target_z3))

    x, z = data_reader((start + end) / 2,end,df)
    mid_idx = np.searchsorted(df['x'], (start+end)/2+0.2, side='left')
    idx6 = mid_idx + np.argmin(np.abs(z - target_z1))
    idx5 = mid_idx + np.argmin(np.abs(z - target_z2))
    idx4 = mid_idx + np.argmin(np.abs(z - target_z3))

    #左侧直线拟合
    print(f"左侧直线区间的拟合结果：")
    mixed_line1=linear_fit(df['x'][idx1],df['x'][idx2],df)

    #右侧直线拟合
    print(f"右侧直线区间的拟合结果：")
    mixed_line2=linear_fit(df['x'][idx5],df['x'][idx6],df)

    #中间弧线拟合
    print(f"中间弧线区间的拟合结果：")
    mixed_arc=arc_fit(df['x'][idx3],df['x'][idx4],df)

    return mixed_line1,mixed_line2,mixed_arc

def plot_all_results(df):
    plt.figure(figsize=(24, 10))

    # 1. 绘制原始数据
    plt.plot(df['x'], df['z'], color='gray', linewidth=5, label='原始数据')

    # 2. 绘制所有直线段拟合曲线
    def plot_fitted_line(line,start,end,line_name,color='skyblue'):
        slope, _, intercept = line
        x_fit = np.linspace(start, end, 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=3, label=line_name)

    plot_fitted_line(line1,df['x'][0],turning_dict['a1'],'line1')
    plot_fitted_line(line2,turning_dict['a2'],turning_dict['a3'],'line2')
    plot_fitted_line(line3,turning_dict['a4'],turning_dict['a5'],'line3')
    plot_fitted_line(line4,turning_dict['a6'],turning_dict['a7'],'line4')
    plot_fitted_line(line5,turning_dict['a7'],turning_dict['a8'],'line5')
    plot_fitted_line(line6,turning_dict['a8'],turning_dict['a9'],'line6')
    plot_fitted_line(line7,turning_dict['a9'],turning_dict['a10'],'line7')
    plot_fitted_line(line8,turning_dict['a11'],turning_dict['a12'],'line8')
    plot_fitted_line(line9,turning_dict['a13'],turning_dict['a14'],'line9')
    plot_fitted_line(line10,turning_dict['a15'],turning_dict['a16'],'line10')
    plot_fitted_line(line11,turning_dict['a17'],df['x'].iloc[-1],'line11')

    plot_fitted_line(mixed_line1,turning_dict['a1'],turning_dict['a18'],'mixed_line1',color='blue')
    plot_fitted_line(mixed_line2,turning_dict['a19'],turning_dict['a2'],'mixed_line2',color='blue')
    plot_fitted_line(mixed_line3,turning_dict['a3'],turning_dict['a20'],'mixed_line3',color='blue')
    plot_fitted_line(mixed_line4,turning_dict['a21'],turning_dict['a4'],'mixed_line4',color='blue')
    plot_fitted_line(mixed_line5,turning_dict['a5'],turning_dict['a22'],'mixed_line5',color='blue')
    plot_fitted_line(mixed_line6,turning_dict['a23'],turning_dict['a6'],'mixed_line6',color='blue')

    # 3. 绘制所有完整圆
    def plot_full_circle(arc_params, color, label, name):
        h, k, r = arc_params
        theta = np.linspace(0, 2 * np.pi, 100)
        x = h + r * np.cos(theta)
        y = k + r * np.sin(theta)
        plt.plot(x, y, color=color, linestyle='--', linewidth=3, label=label)
        plt.scatter(h, k, color='salmon', s=20)
        plt.text(h, k+0.12, name, fontsize=20, ha='center')

    # 绘制所有弧线
    plot_full_circle(arc1, 'pink', 'arc1','O4')
    plot_full_circle(arc2, 'pink', 'arc2','O5')
    plot_full_circle(arc3, 'pink', 'arc3','O6')
    plot_full_circle(arc4, 'pink', 'arc4','O7')

    plot_full_circle(mixed_arc1, 'salmon', 'mixed_arc1','O1')
    plot_full_circle(mixed_arc2, 'salmon', 'mixed_arc2','O2')
    plot_full_circle(mixed_arc3, 'salmon', 'mixed_arc3','O3')

     # 4. 绘制所有转折点
    for i in range(1, 24):
        if f'a{i}' in turning_dict:
            plt.scatter(turning_dict[f'a{i}'], turning_dict[f'b{i}'], color='black', s=20, zorder=10)
            if i in [18,20,22]:  # 左侧点向左偏移
                plt.text(turning_dict[f'a{i}']-0.1, turning_dict[f'b{i}'],
                         f'a{i}', fontsize=20, ha='right')
            elif i in [19,21,23]:  # 右侧点向右偏移
                plt.text(turning_dict[f'a{i}']+0.1, turning_dict[f'b{i}'],
                         f'a{i}', fontsize=20, ha='left')
            else:  # 其他点保持原样
                plt.text(turning_dict[f'a{i}'], turning_dict[f'b{i}'] + 0.05,
                         f'a{i}', fontsize=20, ha='center')


    # 5. 添加图例和标签
    plt.legend(fontsize=15, loc='upper right')
    plt.title('工件1_level_fit', fontsize=24)
    plt.xlabel('x坐标', fontsize=20)
    plt.ylabel('z坐标', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 6. 保存和显示图像
    plt.savefig('Figures/工件1_level_fit.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #读取数据
    df=pd.read_excel('第2周C题：接触式轮廓仪的自动标注(2020年D题)/附件1_工件1的测量数据.xlsx',sheet_name='level')
    #标记转折点
    turning_points=['(49.7154,-1.7739)','(52.7223,-1.7699)','(57.5009,-1.7676)','(59.7754,-1.7659)',
                    '(64.5999,-1.7690)','(66.8444,-1.7679)','(71.6264,-1.7676)','(76.8240,-0.7815)',
                    '(81.8819,-1.7691)','(84.6165,-1.7795)','(86.6494,-1.7737)','(87.4959,-1.7757)',
                    '(89.5424,-1.7685)','(94.3925,-1.7840)','(101.7909,-1.7828)',
                    '(106.5504,-1.7766)','(114.1814,-1.7795)'
                    ]
    turning_points_x=[]
    turning_points_z=[]
    for item in turning_points:
        coord=item.strip('()').split(',')
        turning_points_x.append(float(coord[0]))
        turning_points_z.append(float(coord[1]))
    #转折点命名
    turning_dict = {}
    for i in range(len(turning_points)):
        turning_dict[f'a{i+1}'] = turning_points_x[i]
        turning_dict[f'b{i+1}'] = turning_points_z[i]

    #绘图
    plt.figure(figsize=(24,10))
    plt.plot(df['x'],df['z'],color='gray',linewidth=5)
    #添加转折点
    plt.scatter(turning_points_x, turning_points_z, color='gray')
    for i in range(len(turning_points)):
        plt.text(turning_points_x[i],turning_points_z[i]+0.08,f'a{i+1}',fontsize=20)
    plt.title('工件1的测量数据')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figures/工件1_level.png')
    plt.show()
    plt.close()

    #直线区段拟合
    print("区间[,a1]的拟合结果")
    line1=linear_fit(df['x'][0],turning_dict['a1'],df)
    print('区间[a2,a3]的拟合结果：')
    line2=linear_fit(turning_dict['a2'],turning_dict['a3'],df)
    print('区间[a4,a5]的拟合结果：')
    line3=linear_fit(turning_dict['a4'],turning_dict['a5'],df)
    print('区间[a6,a7]的拟合结果：')
    line4=linear_fit(turning_dict['a6'],turning_dict['a7'],df)
    print('区间[a7,a8]的拟合结果：')
    line5=linear_fit(turning_dict['a7'],turning_dict['a8'],df)
    print('区间[a8,a9]的拟合结果：')
    line6=linear_fit(turning_dict['a8'],turning_dict['a9'],df)
    print('区间[a9,a10]的拟合结果：')
    line7=linear_fit(turning_dict['a9'],turning_dict['a10'],df)
    print('区间[a11,a12]的拟合结果：')
    line8=linear_fit(turning_dict['a11'],turning_dict['a12'],df)
    print('区间[a13,a14]的拟合结果：')
    line9=linear_fit(turning_dict['a13'],turning_dict['a14'],df)
    print('区间[a15,a16]的拟合结果：')
    line10=linear_fit(turning_dict['a15'],turning_dict['a16'],df)
    print("区间[a17,]的拟合结果：")
    line11=linear_fit(turning_dict['a17'],df['x'].iloc[-1],df)

    #更新转折点坐标
    point=line_intersection(line4, line5)
    turning_dict['a7']=point[0]
    turning_dict['b7']=point[1]
    point=line_intersection(line5, line6)
    turning_dict['a8']=point[0]
    turning_dict['b8']=point[1]
    point = line_intersection(line6, line7)
    turning_dict['a9'] = point[0]
    turning_dict['b9'] = point[1]
    #计算z1
    z1=turning_dict['b8']-turning_dict['b9']
    print(f'z1={z1:.8f}\n')

    #弧线区段拟合
    print('区间[a10,a11]的拟合结果：')
    arc1=arc_fit(turning_dict['a10'],turning_dict['a11'],df)
    print('区间[a12,a13]的拟合结果：')
    arc2=arc_fit(turning_dict['a12'],turning_dict['a13'],df)
    print('区间[a14,a15]的拟合结果：')
    arc3=arc_fit(turning_dict['a14'],turning_dict['a15'],df)
    print('区间[a16,a17]的拟合结果：')
    arc4=arc_fit(turning_dict['a16'],turning_dict['a17'],df)

    #混合区段拟合
    print('区间[a1,a2]的拟合结果：')
    mixed_line1, mixed_line2, mixed_arc1=mixed_fit(turning_dict['a1'],turning_dict['a2'],-1.9,-3.8,-4.3,df)
    print('区间[a3,a4]的拟合结果：')
    mixed_line3, mixed_line4, mixed_arc2=mixed_fit(turning_dict['a3'],turning_dict['a4'],-2.1,-4.2,-4.3,df)
    print('区间[a5,a6]的拟合结果：')
    mixed_line5, mixed_line6, mixed_arc3=mixed_fit(turning_dict['a5'],turning_dict['a6'],-2.1,-4.0,-4.2,df)

    # 更新转折点坐标
    point=line_intersection(line1,mixed_line1 )
    turning_dict['a1']=point[0]
    turning_dict['b1']=point[1]
    point = line_intersection(line2, mixed_line2)
    turning_dict['a2'] = point[0]
    turning_dict['b2'] = point[1]
    point=line_intersection(line2,mixed_line3)
    turning_dict['a3']=point[0]
    turning_dict['b3']=point[1]
    point=line_intersection(mixed_line4,line3)
    turning_dict['a4']=point[0]
    turning_dict['b4']=point[1]
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
    point=line_circle_intersection(mixed_line1,mixed_arc1)
    turning_dict['a18']=point[0]
    turning_dict['b18']=point[1]
    point = line_circle_intersection(mixed_line2, mixed_arc1)
    turning_dict['a19'] = point[0]
    turning_dict['b19'] = point[1]
    point = line_circle_intersection(mixed_line3, mixed_arc2)
    turning_dict['a20'] = point[0]
    turning_dict['b20'] = point[1]
    point = line_circle_intersection(mixed_line4, mixed_arc2)[0]
    turning_dict['a21'] = point[0]
    turning_dict['b21'] = point[1]
    point = line_circle_intersection(mixed_line5, mixed_arc3)
    turning_dict['a22'] = point[0]
    turning_dict['b22'] = point[1]
    point = line_circle_intersection(mixed_line6, mixed_arc3)[0]
    turning_dict['a23'] = point[0]
    turning_dict['b23'] = point[1]

    #打印转折点列表
    print("转折点列表")
    for i in range(23):
        print(f'(a{i+1},b{i+1})=({turning_dict[f'a{i+1}']:.8f},{turning_dict[f'b{i+1}']:.8f})')
    print("圆参数")
    for i, arc in enumerate([mixed_arc1, mixed_arc2, mixed_arc3], 1):
        h, k, r = arc
        print(f'O{i}=({h:.8f},{k:.8f}) 半径={r:.8f}')
    for i, arc in enumerate([arc1, arc2, arc3, arc4], 4):
        h, k, r = arc
        print(f'O{i}=({h:.8f},{k:.8f}) 半径={r:.8f}')

    #绘制总图
    plot_all_results(df)
