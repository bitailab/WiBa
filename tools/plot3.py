import numpy as np
import matplotlib.pyplot as plt
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("np_file", type=str, help="Path to the file containing dataset")
args = parser.parse_args()
# 加载文件中的数据
sets = np.load(args.np_file)

# 定义连接关系
next_point = np.array([
    [0, 1], [1, 2], [2, 5], [3, 0], [4, 2], [5, 7],
    [6, 3], [7, 3], [8, 4], [9, 5], [10, 6], [11, 7],
    [12, 9], [13, 11]
])

# 初始化 3D 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'red', 'green']

# 绘制每个人体的骨架
for i, (set_data, color) in enumerate(zip(sets, colors), start=1):
    ax.scatter(set_data[:, 0], set_data[:, 1], set_data[:, 2], color=color, label=f'Person {i}', s=50, marker='o')
    for start, end in next_point:
        x_vals = [set_data[start, 0], set_data[end, 0]]
        y_vals = [set_data[start, 1], set_data[end, 1]]
        z_vals = [set_data[start, 2], set_data[end, 2]]
        ax.plot(x_vals, y_vals, z_vals, color=color)

# 设置坐标轴和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(args.np_file)
ax.legend()

plt.show()
