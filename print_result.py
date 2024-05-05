import numpy as np

data_arrive = np.load('data/lastRV/arrive_lastRV_kepler.npz')
data_with = np.load('data/lastRV/with_lastRV_kepler.npz')
data_leave = np.load('data/lastRV/leave_lastRV_kepler.npz')

print(f"到达引力影响球时：r={data_arrive['r']},v={data_arrive['v']}")
print(f"离开引力影响球时：r={data_with['r']},v={data_with['v']}")
print(f"离开引力影响球500天后：r={data_leave['r']},v={data_leave['v']}")