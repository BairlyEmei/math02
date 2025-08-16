from Q1_main import linear_fit,line_intersection
from Q2_main import  coordinate_transform
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

if __name__ == '__main__':
    # 读取数据
    # 创建ExcelFile对象
    xls = pd.ExcelFile('第2周C题：接触式轮廓仪的自动标注(2020年D题)/附件2_工件2的整体测量数据.xlsx')

    # 读取所有sheet到字典
    sheets = {sheet: xls.parse(sheet) for sheet in xls.sheet_names[:10]}

    plt.figure(figsize=(12, 8))

    # 遍历所有sheet
    for sheet_name, df in sheets.items():
        # 绘制x和z列
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=sheet_name)

    # 添加图例和标签
    plt.legend(loc='best')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Q3 All Sheets Data')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figures/Q3_All_Sheets_Data.png')
    plt.show()

    #标记5组区间
    interval_axis={
        'p25':(58,59.6),
        'p26':(56.8,58.6),
        'p27':(54.6,56.2),
        'p28':(52.4,54),
        'p29':(55.5,56.8)
        }
    interval_axis_vertical={
        'p25':(60.5,61.2),
        'p26':(59.2,60),
        'p27':(57,57.8),
        'p28':(54.8,55.8),
        'p29':(57.4,58.4)
    }
    # 按顺序处理每两个sheet对应一个区间
    lines={}
    lines_vertical={}
    intersection_points={}
    for i, (key, (start, end)) in enumerate(interval_axis.items()):
        sheet1_idx = i * 2
        sheet2_idx = sheet1_idx + 1

        # 确保索引不越界
        if sheet2_idx >= len(sheets):
            break

        sheet1_name = list(sheets.keys())[sheet1_idx]
        sheet2_name = list(sheets.keys())[sheet2_idx]

        # 获取两个sheet的数据
        df1 = sheets[sheet1_name]
        df2 = sheets[sheet2_name]

        print(f"区间 {key}({start}-{end}) 对应的sheet: {sheet1_name} 和 {sheet2_name}")

        #线性拟合
        line1=linear_fit(start,end,df1)
        line2=linear_fit(start,end,df2)
        lines[sheet1_name] = line1
        lines[sheet2_name] = line2

        line1_vertical=linear_fit(interval_axis_vertical[key][0],interval_axis_vertical[key][1],df1)
        line2_vertical=linear_fit(interval_axis_vertical[key][0],interval_axis_vertical[key][1],df2)
        lines_vertical[sheet1_name]=line1_vertical
        lines_vertical[sheet2_name]=line2_vertical

        #计算交点
        point1=line_intersection(line1,line1_vertical)
        point2=line_intersection(line2,line2_vertical)

        intersection_points[sheet1_name]=point1
        intersection_points[sheet2_name]=point2
    # print(lines)

    #计算倾角
    slopes = [line[0] for line in lines.values()]
    angles={}
    angles_deg={}
    for i in range(len(slopes)):
        angle=math.atan(slopes[i])
        key_name=list(lines.keys())
        angles[key_name[i]]=angle
        angles_deg[key_name[i]]=angle*180/math.pi
    alpha_ref=np.mean(list(angles.values()))
    alphas_i={}
    for i in range(len(angles)):
        alphas_i[key_name[i]]=angles[key_name[i]]-alpha_ref

    print(f"角度(弧度):{angles}")
    print(f"角度(角度):{angles_deg}")
    print(f"标准角度(弧度):{alpha_ref}，标准角度(角度):{alpha_ref * 180 / math.pi}")
    print(f"相对角度(弧度):{alphas_i}")
    print(f"相对角度(角度):{np.array(list(alphas_i.values()))*180/math.pi}\n")

    # 转换交点坐标
    for key, point in intersection_points.items():
        # 获取当前角度差
        alpha = alphas_i[key]
        # 旋转变换
        x = point[0] * math.cos(alpha) - point[1] * math.sin(alpha)
        y = point[0] * math.sin(alpha) + point[1] * math.cos(alpha)
        intersection_points[key] = (x, y)

    #计算转换后的质心
    centroid_x = np.mean([point[0] for point in intersection_points.values()])
    centroid_z = np.mean([point[1] for point in intersection_points.values()])
    print(f"交点的质心: ({centroid_x}, {centroid_z})")

    # 计算每个点的x轴平移量
    translations_x = {key: centroid_x - point[0] for key, point in intersection_points.items()}
    print("每个的x轴平移量:",translations_x)

    # 转换坐标
    data_x={}
    data_z={}
    os.makedirs('Figures/Q3_excels', exist_ok=True)
    alpha_values = list(alphas_i.values())
    plt.figure(figsize=(12, 8))
    for i, (sheet_name, df) in enumerate(sheets.items()):
        df = coordinate_transform(df, alpha_values[i], save_path=f'Figures/Q3_excels/Q3_{sheet_name}_new.xlsx')
        # 平移
        df.iloc[:, 0] += translations_x[sheet_name]
        # 保存x数据
        data_x[sheet_name]=df.iloc[:, 0].values
        # 保存z数据
        data_z[sheet_name]=df.iloc[:, 1].values
        #绘图
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=sheet_name)
    plt.legend(loc='best')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Q3_Transformed')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Figures/Q3_All_Sheets_Data_Transformed.png')
    plt.show()

    ##平均并绘图
    # 1. 确定全局x轴范围（所有工作表的x值合并后的范围）
    all_x_values = []
    for x_array in data_x.values():
        all_x_values.extend(x_array)

    global_x_min = np.min(all_x_values)
    global_x_max = np.max(all_x_values)
    print(f"全局x轴范围: [{global_x_min:.4f}, {global_x_max:.4f}]")

    # 2. 创建参考x轴（在全局范围内均匀采样）
    num_samples = 20000
    reference_x = np.linspace(global_x_min, global_x_max, num_samples)

    # 3. 为每个参考x点匹配所有工作表的z值并计算平均
    average_z = []

    for ref_x in reference_x:
        z_matches = []

        # 遍历每个工作表，找到最接近参考x的z值
        for sheet in data_x.keys():
            # 获取当前工作表的x和z数据
            x_vals = data_x[sheet]
            z_vals = data_z[sheet]

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

    # 5. 保存结果
    result_df.to_excel('Figures/Q3_Avg_Z_Results.xlsx', index=False)
    print(f"结果已保存到 Figures/Q3_Avg_Z_Results.xlsx")

    # 6. 绘制平均曲线
    plt.figure(figsize=(12, 8))
    plt.plot(result_df['x'], result_df['z'], color='blue', linewidth=2.5, label='平均Z值')

    # 可选：叠加显示原始数据曲线（浅灰色）作为参考
    for sheet in data_x.keys():
        plt.plot(data_x[sheet], data_z[sheet], color='lightgray', alpha=0.5)

    plt.xlabel('X坐标', fontsize=12)
    plt.ylabel('平均Z坐标', fontsize=12)
    plt.title('全局X范围内的平均Z值分布', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figures/Q3_Avg_Z_Results.png', dpi=300)
    plt.show()


