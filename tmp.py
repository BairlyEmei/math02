# 转折点横坐标列表 (a1到a23)
turning_points_x = [
    49.77244522,  # a1
    52.68599790,  # a2
    57.66175599,  # a3
    59.69210004,  # a4
    64.72112977,  # a5
    66.78017990,  # a6
    71.79527031,  # a7
    76.80736676,  # a8
    81.80912741,  # a9
    84.80644359,  # a10
    86.60833758,  # a11
    87.60683917,  # a12
    89.42532428,  # a13
    94.42512304,  # a14
    101.67439161,  # a15
    106.67559764,  # a16
    113.93170376,  # a17
    50.74531724,  # a18
    51.66930726,  # a19
    58.39396379,  # a20
    58.85808035,  # a21
    65.48377838,  # a22
    65.92796722  # a23
]

# 圆心横坐标列表 (O1到O7)
circle_centers_x = [
    51.21828075,  # O1
    58.70235016,  # O2
    65.77278610,  # O3
    85.70623007,  # O4
    88.51730081,  # O5
    98.05010617,  # O6
    110.30300829  # O7
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
