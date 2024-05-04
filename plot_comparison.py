import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 从数据文件中读取结果数据
data_int = np.load('data/arrive_orbit_int.npz')
data_kepler = np.load('data/arrive_orbit_kepler.npz')
data_lagrange = np.load('data/arrive_orbit_lagrange.npz')
data_lagrange_2 = np.load('data/arrive_orbit_lagrange_2.npz')

# 提取时间数据，换算至天
t_int = data_int['t']
t_kepler = data_kepler['t']
t_lagrange = data_lagrange['t']
t_lagrange_2 = data_lagrange_2['t']

t_int = t_int / 60 / 60 /24 
t_kepler = t_kepler / 60 / 60 / 24
t_lagrange = t_lagrange / 60 / 60 / 24
t_lagrange_2 = t_lagrange_2 / 60 / 60 / 24

# 提取结果数据
r_int = data_int['r']
r_kepler = data_kepler['r']
r_lagrange = data_lagrange['r']
r_lagrange_2 = data_lagrange_2['r']

# 提取六根数数据
A_int = data_int['A']
A_kepler = data_kepler['A']
A_lagrange = data_lagrange['A']
A_lagrange_2 = data_lagrange_2['A']

#########################################################################
# 位置速度绝对误差

absolute_error_int = r_int - r_kepler
absolute_error_lagrange = r_lagrange - r_kepler
absolute_error_lagrange_2 = r_lagrange_2 - r_kepler

fig_1, axs_1 = plt.subplots(3, 2, figsize=(12, 10))
ylabels_1 = ['X(km)', 'Y(km)', 'Z(km)', 'Vx(km/s)', 'Vy(km/s)', 'Vz(km/s)']
for i in range(2):
    for j in range(3):
        axs_1[j, i].plot(t_int, absolute_error_int[:, i * 3 + j], label = '数值积分方法', color = 'blue')
        axs_1[j, i].plot(t_int, absolute_error_lagrange[:, i * 3 + j], label = '拉格朗日系数法（级数）', color = 'orange')
        axs_1[j, i].plot(t_int, absolute_error_lagrange_2[:, i * 3 + j], label = '拉格朗日系数法（闭合）', color = 'green')
        axs_1[j, i].set_xlabel('时间(天)')
        axs_1[j, i].set_ylabel(ylabels_1[i * 3 + j])
        axs_1[j, i].legend()
        axs_1[j, i].grid(True)

plt.tight_layout()

#########################################################################
# 轨道六根数绘制

fig_2, axs_2 = plt.subplots(3, 2, figsize=(12, 10))
ylabels_2 = ['半长轴(km)', '偏心率', '轨道倾角(rad)', '升交点赤经(rad)', '近地点幅角(rad)', '真近点角(rad)']

for i in range(3):
    for j in range(2):
        axs_2[i, j].plot(t_int, A_int[:, i * 2 + j], label='数值积分方法', color = 'blue')
        axs_2[i, j].plot(t_kepler, A_kepler[:, i * 2 + j], label='开普勒方法', color = 'red')
        axs_2[i, j].plot(t_lagrange, A_lagrange[:, i * 2 + j], label='拉格朗日系数法（级数）', color='orange')
        axs_2[i, j].plot(t_lagrange_2, A_lagrange_2[:, i * 2 + j], label='拉格朗日系数法（闭合）', color='green')
        axs_2[i, j].set_xlabel('时间(天)')
        axs_2[i, j].set_ylabel(ylabels_2[i * 2 + j])
        axs_2[i, j].legend()
        axs_2[i, j].grid(True)

plt.tight_layout()

# 显示图形
plt.show()
