import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 加载保存的轨迹数据
data = np.load('data/arrive_orbit_kepler.npz')
r = data['r']
t = data['t']

data_2 = np.load('data/arrive_orbit_int.npz')
r_2 = data_2['r']
t_2 = data_2['t']

# 提取位置坐标
x = r[:, 0]
y = r[:, 1]
z = r[:, 2]


# 创建三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制轨迹
ax.plot(r[:, 0], r[:, 1], r[:, 2], label='航天器轨迹')
ax.plot(r_2[:, 0], r_2[:, 1], r_2[:, 2], label='航天器轨迹(int)')


# # 添加太阳(球体)
# R_sun = 695700
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = R_sun * np.outer(np.cos(u), np.sin(v))
# y = R_sun * np.outer(np.sin(u), np.sin(v))
# z = R_sun * np.outer(np.ones(np.size(u)), np.cos(v))

# # 绘制球体
# ax.plot_surface(x, y, z, color='r', alpha=0.5)

ax.set_xlabel('X/km')
ax.set_ylabel('Y/km')
ax.set_zlabel('Z/km')
ax.set_title('航天器日心轨迹')
ax.set_aspect('equal')
ax.legend()

plt.show()
