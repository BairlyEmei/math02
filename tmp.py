import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.distance import cdist
import pandas as pd


class WorkpieceAnalyzer:
    def __init__(self, data_path):
        """
        初始化工件分析器
        :param data_path: 数据文件路径
        """
        self.raw_data = pd.read_excel(data_path, sheet_name='level')
        self.x = self.raw_data['x'].values
        self.z = self.raw_data['z'].values
        self.turning_points = []
        self.segments = []
        self.params = {}

    def preprocess_data(self, window_size=11, polyorder=3):
        """
        数据预处理：去除离群点和平滑处理
        :param window_size: 平滑窗口大小
        :param polyorder: 多项式阶数
        """
        # 1. 离群点处理 (基于标准差)
        z_mean = np.mean(self.z)
        z_std = np.std(self.z)
        valid_idx = np.abs(self.z - z_mean) < 3 * z_std
        self.x = self.x[valid_idx]
        self.z = self.z[valid_idx]

        # 2. Savitzky-Golay平滑
        from scipy.signal import savgol_filter
        self.z_smooth = savgol_filter(self.z, window_size, polyorder)

        # 3. 展示预处理效果
        plt.figure(figsize=(12, 6))
        plt.plot(self.x, self.z, 'gray', alpha=0.5, label='原始数据')
        plt.plot(self.x, self.z_smooth, 'b-', label='平滑后数据')
        plt.legend()
        plt.title('数据预处理结果')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.grid(True)
        plt.show()

    def detect_turning_points(self, min_curvature=0.1, min_distance=10):
        """
        基于曲率分析自动检测转折点
        :param min_curvature: 最小曲率阈值
        :param min_distance: 转折点最小间距
        """
        # 计算一阶和二阶导数
        dx = np.gradient(self.x)
        dz = np.gradient(self.z_smooth)
        d2z = np.gradient(dz)

        # 计算曲率
        curvature = np.abs(d2z) / (1 + dz ** 2) ** 1.5

        # 检测曲率极大值点
        candidates = []
        for i in range(1, len(curvature) - 1):
            if curvature[i] > curvature[i - 1] and curvature[i] > curvature[i + 1] and curvature[i] > min_curvature:
                candidates.append(i)

        # 按间距过滤点
        self.turning_points = []
        for idx in candidates:
            if not self.turning_points or self.x[idx] - self.x[self.turning_points[-1]] > min_distance:
                self.turning_points.append(idx)

        # 可视化转折点
        plt.figure(figsize=(12, 6))
        plt.plot(self.x, self.z_smooth, 'b-', label='平滑数据')
        plt.scatter(self.x[self.turning_points], self.z_smooth[self.turning_points],
                    c='red', s=100, label='检测转折点')
        plt.title('转折点检测结果')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 输出转折点坐标
        turning_coords = [(self.x[i], self.z_smooth[i]) for i in self.turning_points]
        print(f"检测到转折点坐标: {turning_coords}")
        return turning_coords

    def fit_line_ransac(self, x_segment, z_segment):
        """使用RANSAC拟合直线"""
        model = RANSACRegressor()
        model.fit(x_segment.reshape(-1, 1), z_segment)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        return slope, intercept

    def fit_circle_lm(self, x_segment, z_segment):
        """使用Levenberg-Marquardt算法拟合圆"""

        def circle_residuals(params, x, z):
            a, b, r = params
            return (x - a) ** 2 + (z - b) ** 2 - r ** 2

        # 初始估计
        x_mid = np.mean(x_segment)
        z_mid = np.mean(z_segment)
        r_est = np.sqrt((x_segment - x_mid) ** 2 + (z_segment - z_mid) ** 2).mean()

        # 优化
        result = least_squares(circle_residuals, [x_mid, z_mid, r_est],
                               args=(x_segment, z_segment))
        a, b, r = result.x
        return a, b, r

    def fit_transition_segment(self, x_segment, z_segment):
        """拟合过渡段（直线-圆弧-直线）"""
        # 1. 曲率分析确定过渡点
        dz = np.gradient(z_segment)
        d2z = np.gradient(dz)
        curvature = np.abs(d2z) / (1 + dz ** 2) ** 1.5
        transition_idx = np.argmax(curvature)

        # 2. 分段拟合
        # 左侧直线
        left_idx = max(0, transition_idx - 5)
        slope_left, intercept_left = self.fit_line_ransac(
            x_segment[:left_idx], z_segment[:left_idx])

        # 右侧直线
        right_idx = min(len(x_segment), transition_idx + 5)
        slope_right, intercept_right = self.fit_line_ransac(
            x_segment[right_idx:], z_segment[right_idx:])

        # 中间圆弧
        a, b, r = self.fit_circle_lm(
            x_segment[left_idx:right_idx], z_segment[left_idx:right_idx])

        return {
            'left_line': (slope_left, intercept_left),
            'circle': (a, b, r),
            'right_line': (slope_right, intercept_right),
            'transition_points': (x_segment[left_idx], x_segment[right_idx])
        }

    def fit_segments(self):
        """对检测到的转折点之间的所有区段进行拟合"""
        self.segments = []
        turning_indices = sorted(self.turning_points)

        # 根据文档中的区段分类进行拟合
        for i in range(len(turning_indices) - 1):
            start_idx = turning_indices[i]
            end_idx = turning_indices[i + 1]
            x_seg = self.x[start_idx:end_idx + 1]
            z_seg = self.z_smooth[start_idx:end_idx + 1]

            # 根据文档中的区段类型进行拟合
            seg_type = self.classify_segment(i + 1)  # i+1对应转折点编号

            if seg_type == 'line':
                # 直线拟合 (RANSAC)
                slope, intercept = self.fit_line_ransac(x_seg, z_seg)
                self.segments.append({
                    'type': 'line',
                    'params': (slope, intercept),
                    'x_range': (x_seg[0], x_seg[-1])
                })

            elif seg_type == 'circle':
                # 圆弧拟合 (LM)
                a, b, r = self.fit_circle_lm(x_seg, z_seg)
                self.segments.append({
                    'type': 'circle',
                    'params': (a, b, r),
                    'x_range': (x_seg[0], x_seg[-1])
                })

            else:  # transition
                # 过渡区拟合
                fit_result = self.fit_transition_segment(x_seg, z_seg)
                self.segments.append({
                    'type': 'transition',
                    'params': fit_result,
                    'x_range': (x_seg[0], x_seg[-1])
                })

    def classify_segment(self, segment_idx):
        """根据文档中的区段分类规则判断区段类型"""
        # 文档中的分类规则：
        # 第一类: [a7,a8], [a8,a9] -> 直线
        # 第二类: [a10,a11], [a12,a13], [a14,a15], [a16,a17] -> 圆周
        # 第三类: [a1,a2], [a3,a4], [a5,a6] -> 过渡区

        if segment_idx in [7, 8]:  # [a7,a8]和[a8,a9]
            return 'line'
        elif segment_idx in [10, 12, 14, 16]:  # 文档中的圆周区段
            return 'circle'
        else:
            return 'transition'

    def calculate_parameters(self):
        """计算所有需要的参数"""
        self.params = {}

        # 1. 水平线段长度 (x方向差值)
        for i, seg in enumerate(self.segments):
            if seg['type'] == 'line':
                x_start, x_end = seg['x_range']
                self.params[f'L{i + 1}'] = abs(x_end - x_start)

        # 2. 角度计算 (∠1)
        # 查找第一条斜线区段 (文档中的∠1)
        for seg in self.segments:
            if seg['type'] == 'line':
                slope = seg['params'][0]
                self.params['∠1'] = np.pi - np.arctan(slope)
                break

        # 3. 半径计算
        circle_count = 1
        for seg in self.segments:
            if seg['type'] == 'circle':
                _, _, r = seg['params']
                self.params[f'R{circle_count}'] = r
                circle_count += 1

        # 4. 高度差 (z方向差值)
        z_values = [self.z_smooth[i] for i in self.turning_points]
        self.params['z1'] = max(z_values) - min(z_values)

        # 5. 圆心距离
        circle_centers = [seg['params'][:2] for seg in self.segments if seg['type'] == 'circle']
        if len(circle_centers) >= 2:
            dist = np.sqrt((circle_centers[0][0] - circle_centers[1][0]) ** 2 +
                           (circle_centers[0][1] - circle_centers[1][1]) ** 2)
            self.params['c1'] = dist

        return self.params

    def visualize_results(self):
        """可视化最终拟合结果"""
        plt.figure(figsize=(14, 8))

        # 绘制原始数据和平滑数据
        plt.plot(self.x, self.z, 'gray', alpha=0.3, label='原始数据')
        plt.plot(self.x, self.z_smooth, 'b-', alpha=0.7, label='平滑数据')

        # 绘制转折点
        plt.scatter(self.x[self.turning_points], self.z_smooth[self.turning_points],
                    c='red', s=100, label='转折点')

        # 绘制拟合结果
        for seg in self.segments:
            x_start, x_end = seg['x_range']
            x_vals = np.linspace(x_start, x_end, 100)

            if seg['type'] == 'line':
                slope, intercept = seg['params']
                z_fit = slope * x_vals + intercept
                plt.plot(x_vals, z_fit, 'g-', linewidth=2.5, label='直线拟合')

            elif seg['type'] == 'circle':
                a, b, r = seg['params']
                # 计算圆上的点
                theta = np.linspace(0, 2 * np.pi, 100)
                x_fit = a + r * np.cos(theta)
                z_fit = b + r * np.sin(theta)
                # 只绘制在区间内的部分
                in_segment = (x_fit >= x_start) & (x_fit <= x_end)
                plt.plot(x_fit[in_segment], z_fit[in_segment], 'm-', linewidth=2.5, label='圆弧拟合')

            else:  # transition
                params = seg['params']
                # 绘制左侧直线
                slope_left, intercept_left = params['left_line']
                x_left = np.linspace(x_start, params['transition_points'][0], 30)
                z_left = slope_left * x_left + intercept_left
                plt.plot(x_left, z_left, 'g-', linewidth=2.5)

                # 绘制右侧直线
                slope_right, intercept_right = params['right_line']
                x_right = np.linspace(params['transition_points'][1], x_end, 30)
                z_right = slope_right * x_right + intercept_right
                plt.plot(x_right, z_right, 'g-', linewidth=2.5)

                # 绘制圆弧
                a, b, r = params['circle']
                # 计算圆弧角度范围
                angle_start = np.arctan2(params['transition_points'][0] - a,
                                         z_left[-1] - b)
                angle_end = np.arctan2(params['transition_points'][1] - a,
                                       z_right[0] - b)
                angles = np.linspace(angle_start, angle_end, 50)
                x_arc = a + r * np.cos(angles)
                z_arc = b + r * np.sin(angles)
                plt.plot(x_arc, z_arc, 'm-', linewidth=2.5)

        plt.title('工件轮廓线拟合结果')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.grid(True)

        # 创建图例（避免重复标签）
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.show()

        # 显示参数计算结果
        print("\n计算得到的参数值:")
        for key, value in self.params.items():
            print(f"{key}: {value:.4f}")

    def run_analysis(self):
        """执行完整分析流程"""
        self.preprocess_data()
        self.detect_turning_points()
        self.fit_segments()
        self.calculate_parameters()
        self.visualize_results()


# 使用示例
if __name__ == "__main__":
    # 替换为实际数据文件路径
    analyzer = WorkpieceAnalyzer("第2周C题：接触式轮廓仪的自动标注(2020年D题)/附件1_工件1的测量数据.xlsx")
    analyzer.run_analysis()