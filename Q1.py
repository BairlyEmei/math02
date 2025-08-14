#This is a file for Q1.
import numpy as np
from numpy import argmin
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
import os

#设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
#设置负号
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('Figures', exist_ok=True)

#区间数据读取器
def data_reader(start,end):
    # 读取数据
    x = df['x']
    start_idx = np.searchsorted(x, start, side='left')
    end_idx = np.searchsorted(x, end, side='right')
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

#直线拟合器
def linear_fit(start,end):
    # 读取数据
    x,z=data_reader(start,end)
    # 计算线性拟合参数
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, z)
    angle_rad,angle_deg=angle_cal(slope)
    print(f"拟合直线方程: z = {slope:.8f}x + {intercept:.8f}\n"
          f"R^2: {r_value**2:.8f}\n"
          f"p值: {p_value:.8f}\n"
          f"标准误差: {std_err:.8f}\n"
          f"弧度: {angle_rad:.4f}，角度: {angle_deg:.4f}°\n"
          )
    return slope, intercept

#弧线拟合器
def arc_fit(start,end):
    # 读取数据
    x,z=data_reader(start,end)
    pass

#混合拟合器
def mixed_fit(start,end):
    # 读取数据
    x,z=data_reader(start,end)
    pass


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
    plt.figure(figsize=(32,8))
    plt.plot(df['x'],df['z'],color='gray',linewidth=3)
    #添加转折点
    plt.scatter(turning_points_x, turning_points_z, color='gray')
    for i in range(len(turning_points)):
        plt.text(turning_points_x[i],turning_points_z[i]+0.05,f'a{i+1}',fontsize=12)
    plt.title('工件1的测量数据')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('Figures/工件1_level.png')
    # plt.show()
    plt.close()

    #直线区段拟合
    print('区间[a7,a8]的拟合结果：')
    slope1, intercept1=linear_fit(turning_dict['a7'],turning_dict['a8'])
    print('区间[a8,a9]的拟合结果：')
    slope2, intercept2=linear_fit(turning_dict['a8'],turning_dict['a9'])

    #计算z1
    z=(slope1*intercept2-slope2*intercept1)/(slope1-slope2)
    z1=z-(turning_dict['b7']+turning_dict['b9'])/2
    print(f'交点z坐标={z:.8f},z1={z1:.8f}\n')

    #弧线区段拟合
    print('区间[a8,a9]的拟合结果：')
    arc_fit(turning_dict['a8'],turning_dict['a9'])

    #混合区段拟合
    print('区间[a8,a9]的拟合结果：')
    mixed_fit(turning_dict['a8'],turning_dict['a9'])

    # # 绘制总图
    # plt.figure(figsize=(32, 8))
    # plt.plot(df['x'], df['z'], color='gray', linewidth=3)
    # # 添加转折点
    # plt.scatter(turning_points_x, turning_points_z, color='gray')
    # for i in range(len(turning_points)):
    #     plt.text(turning_points_x[i], turning_points_z[i] + 0.05, f'a{i + 1}', fontsize=12)
    #
    # # 添加拟合直线
    # x1_fit = np.linspace(turning_dict['a7'], turning_dict['a9'], 100)
    # y1_fit = slope1 * x1_fit + intercept1
    # plt.plot(x1_fit, y1_fit, color='skyblue', linewidth=1.5, linestyle='--')
    # x2_fit = np.linspace(turning_dict['a8'], turning_dict['a9'], 100)
    # y2_fit = slope2 * x2_fit + intercept2
    # plt.plot(x2_fit, y2_fit, color='skyblue', linewidth=1.5, linestyle='--')
    #
    # plt.title('工件1测量与拟合数据')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig('Figures/工件1_level_fit.png')
    # # plt.show()
    # plt.close()