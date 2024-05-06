import numpy as np
from orbital_conversion import angle_between_vectors

data_arrive_int = np.load('data/arrive_orbit_int.npz')
data_arrive_kepler = np.load('data/arrive_orbit_kepler.npz')

data_with_int = np.load('data/with_orbit_int.npz')
data_with_kepler = np.load('data/with_orbit_kepler.npz')

data_leave_int = np.load('data/leave_orbit_int.npz')
data_leave_kepler = np.load('data/leave_orbit_kepler.npz')

print(f"到达引力影响球时日心(int)：{data_arrive_int['r'][-1]}")
print(f"到达引力影响球时日心(kepler)：{data_arrive_kepler['r'][-1]}")
print(f"到达引力影响球时日心(int - kepler)：{data_arrive_int['r'][-1] - data_arrive_kepler['r'][-1]}")

print(f"飞入引力影响球时火心：{data_with_kepler['r'][0]}")
print(f"飞出引力影响球时火心：{data_with_kepler['r'][-1]}")

print(f"飞出引力影响球500天后日心：{data_leave_kepler['r'][-1]}")

a = data_with_kepler['A'][0][0]
e = data_with_kepler['A'][0][1]
R_mars = 3396.2  # 火星平均半径，单位 km
v_inf_in = data_with_kepler['r'][0][3:]  
v_inf_out = data_with_kepler['r'][-1][3:]
v_in = data_arrive_kepler['r'][-1][3:]
v_out = data_leave_kepler['r'][0][3:]
print(f"双曲线轨道a、e：{a} , {e}")
print(f"双曲线近地点高度：{a * (1-e) - R_mars}")
print(f"双曲线超速： {np.linalg.norm(v_inf_in)}")
print(f"速度增量：{np.linalg.norm(v_out - v_in)}")
print(f"速度增量（理论）：{2 * np.linalg.norm(v_inf_in) / e}")
print(f"速度偏转角：{angle_between_vectors(v_inf_in, v_inf_out) * 180 / np.pi}")
print(f"速度偏转角（理论）：{2 * np.arcsin(1 / e) * 180 / np.pi}")
