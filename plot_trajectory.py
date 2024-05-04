import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 加载保存的轨迹数据
data_arrive = np.load('data/arrive_orbit_kepler.npz')
r_arrive = data_arrive['r']
t_arrive = data_arrive['t']

data_leave = np.load('data/leave_orbit_kepler.npz')
r_leave = data_leave['r']
t_leave = data_leave['t']

data_with_2 = np.load('data/with_orbit_int.npz')
r_with_2 = data_with_2['r']
t_with_2 = data_with_2['t']

data_with = np.load('data/with_orbit_kepler.npz')
r_with = data_with['r']
t_with = data_with['t']

data_earth = np.load('data/orbit_earth.npz')
r_earth = data_earth['r']
t_earth = data_earth['t']

data_mars = np.load('data/orbit_mars.npz')
r_mars = data_mars['r']
t_mars = data_mars['t']

r_1 = np.concatenate((r_arrive, r_leave))

###################################################################
# 日心轨道绘制
# 创建三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制轨迹
ax.plot(r_1[:, 0], r_1[:, 1], r_1[:, 2], label='航天器轨迹')
# ax.plot(r_leave[:, 0], r_leave[:, 1], r_leave[:, 2])
ax.plot(r_earth[:, 0], r_earth[:, 1], r_earth[:, 2], label='地球轨迹', linestyle='--')
ax.plot(r_mars[:, 0], r_mars[:, 1], r_mars[:, 2], label='火星轨迹', linestyle='--')

# 太阳（点）
ax.scatter(0, 0, 0, label='太阳', marker='o', color='red')

# # 太阳(球体)
# R_sun = 695700
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = R_sun * np.outer(np.cos(u), np.sin(v))
# y = R_sun * np.outer(np.sin(u), np.sin(v))
# z = R_sun * np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_surface(x, y, z, color='r', alpha=0.5)

ax.set_xlabel('X/km')
ax.set_ylabel('Y/km')
ax.set_zlabel('Z/km')
ax.set_aspect('equal')
ax.legend()

######################################################################
# 借力过程火心轨道
fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111, projection='3d')

# 绘制轨迹
ax_2.plot(r_with[:, 0], r_with[:, 1], r_with[:, 2], label='航天器轨迹')

# 火星（点）
ax_2.scatter(0, 0, 0, label='火星', marker='o', color='red')

ax_2.set_xlabel('X/km')
ax_2.set_ylabel('Y/km')
ax_2.set_zlabel('Z/km')
ax_2.set_aspect('equal')
ax_2.legend()

######################################################################
# 借力过程火心轨道（放大至近心点处）
# 借力过程火心轨道
fig_3 = plt.figure()
ax_3 = fig_3.add_subplot(111, projection='3d')

# 根据距离截取轨道
R_desire = 15000  # 考虑这个球内部的轨道
i_start = -1
i_end = -1
for i in range(len(r_with)):
    if i_start == -1:
        if np.linalg.norm(r_with[i, 0:3]) < R_desire:
            i_start = i
    else:
        if np.linalg.norm(r_with[i, 0:3]) > R_desire:
            i_end = i
            break

# 绘制轨迹
ax_3.plot(r_with[i_start:i_end, 0], r_with[i_start:i_end, 1], r_with[i_start:i_end, 2], label='航天器轨迹')

# 火星(球体)
R_mars = 3396.2  # 火星平均半径，单位 km
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = R_mars * np.outer(np.cos(u), np.sin(v))
y = R_mars * np.outer(np.sin(u), np.sin(v))
z = R_mars * np.outer(np.ones(np.size(u)), np.cos(v))
ax_3.plot_surface(x, y, z, color='r', alpha=0.9, label='火星')

ax_3.set_xlabel('X/km')
ax_3.set_ylabel('Y/km')
ax_3.set_zlabel('Z/km')
ax_3.set_aspect('equal')
ax_3.legend()


plt.tight_layout()
plt.show()
