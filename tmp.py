# 转折点横坐标列表 (a1到a23)
turning_points_x = [
    51.71612752,  # a1
    54.61731276,  # a2
    59.59042966,  # a3
    61.66283071,  # a4
    66.67629630,  # a5
    68.77197985,  # a6
    73.75861804,  # a7
    78.75129234,  # a8
    83.75279007,  # a9
    86.75512121,  # a10
    88.56307010,  # a11
    89.56389253,  # a12
    91.37253129,  # a13
    96.37349704,  # a14
    103.63023566,  # a15
    108.63030918,  # a16
    115.87780736,  # a17
]

# 圆心横坐标列表 (O1到O7)
circle_centers_x = [
    53.16713554,  # O1
    60.65838899,  # O2
    67.71937448,  # O3
    87.65777711,  # O4
    90.46907017,  # O5
    100.00046694,  # O6
    112.25446438  # O7
]

# 计算转折点之间的差值 (xi)
x_differences = []
for i in range(1, len(turning_points_x)):
    diff = turning_points_x[i] - turning_points_x[i - 1]
    x_differences.append((i, i + 1, diff))  # (前点序号, 后点序号, 差值)

# 计算圆心之间的差值 (ci)
c_differences = []
for i in range(1, len(circle_centers_x)):
    diff = circle_centers_x[i] - circle_centers_x[i - 1]
    c_differences.append((i, i + 1, diff))  # (前圆心序号, 后圆心序号, 差值)

# 输出结果
print("转折点横坐标差值 (xi):")
print("(前一点, 后一点)   差值")
for prev, curr, diff in x_differences:
    print(f"(a{prev}, a{curr})   {diff:.8f}")

print("\n圆心横坐标差值 (ci):")
print("(前一圆心, 后一圆心)   差值")
for prev, curr, diff in c_differences:
    print(f"(O{prev}, O{curr})   {diff:.8f}")
