"""
主程序，包含了所有问题的对应函数
"""
import os
import numpy as np
from orbital_solving_algorithms import solve_orbit_integrate, solve_orbit_kepler, solve_orbit_lagrange, solve_orbit_lagrange_2

# 检测数据存储文件夹
if not os.path.isdir('data'):
    os.mkdir('data')

if not os.path.isdir('data/lastRV'):
    os.mkdir('data/lastRV')


#################################################################################
# 
#       给定数据
# 
#################################################################################
# 天文常数
mu_sun = 1.3271244001787e11  # 单位 km^3/s^2
mu_mars = 4.28283762065e4  # 单位 km^3/s^2
R_mars = 3396.2  # 火星平均半径，单位 km

# 时间
# 单位：天
day_arrive_mars = 279.1317802839208  # t_0至到达火星影响球
day_with_mars = 3.834478518347840  # 飞进到飞出火星影响球
day_leave_mars = 500  # 飞出火星影响球后继续递推的时间
time_factor = 60 * 60 * 24  # 转换到秒所需倍数

# 所有距离、速度单位均为km km/s
# 在日心坐标系下航天器运动参数（t_0时刻）
x_0 = -0.370264003660595e8
y_0 = 1.315142470848916e8
z_0 = 0.608322679422336e8
vx_0 = -31.806213625480979
vy_0 = -6.234823833392683
vz_0 = -0.078190790328369
r_0 = np.array([x_0, y_0, z_0, vx_0, vy_0, vz_0])

# 在日心坐标系下地球运动参数（t_0时刻）
x_earth_0 = 0.110937729685236e8
y_earth_0 = 1.346867468212425e8
z_earth_0 = 0.583831802330196e8
vx_earth_0 = -30.200703848645549
vy_earth_0 = 1.956654058695767
vz_earth_0 = 0.847099469360955
r_earth_0 = np.array([x_earth_0, y_earth_0, z_earth_0, vx_earth_0, vy_earth_0, vz_earth_0])

# 在日心坐标系下火星运动参数（航天器到达火星时）
x_mars = 0.598177297152914e8
y_mars = -1.853298132110158e8
z_mars = -0.866200961703781e8
vx_marx = 24.165448742900686
vy_mars = 8.313187618593524
vz_mars = 3.161448109961551
r_mars = np.array([x_mars, y_mars, z_mars, vx_marx, vy_mars, vz_mars])

def main():
    """
    可自由调用封装好的各问题求解函数
    各函数可以使用关键字向对应数值算法传参
    各函数有前后依赖关系（依赖于data文件夹中的已保存数据）
    其中所有方法为了方便进行结果对比，对于数值积分的所有中间点均进行了求解
    """
    # orbit_earth_mars()
    
    # arrive_int(tol=1e-3)
    # arrive_kepler()
    # arrive_lagrange()
    # arrive_lagrange_2()

    # with_int()
    with_kepler()

    # leave_int()
    # leave_kepler()
    
    
def debug_int(func):
    # 伪造数值积分的时间采样点
    def wrapper(*args, **kwargs):
        data = np.load('data/arrive_orbit_int.npz')
        r = data['r']
        A = data['A']
        t = np.arange(0, day_arrive_mars * time_factor, 1e4)
        np.savez('data/arrive_orbit_int.npz', r = r, t = t, A = A)
        return None
    return wrapper
    
def print_function_name(func):
    # 使得函数在运行完毕后向控制台输出信息
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"'{func.__name__}'运行完成")
        return result
    return wrapper

@print_function_name
def orbit_earth_mars(**kwargs):
    #################################################################################
    # 
    #       地球火星轨道求解（开普勒方法）
    # 
    #################################################################################
    # 求解
    t_span_1 = np.linspace(0, 370 * time_factor, 1000)
    r_1, A_1, t_1 = solve_orbit_kepler(mu_sun, r_earth_0, t_span_1, **kwargs)
    t_span_2 = np.linspace(0, 700 * time_factor, 1000)
    r_2, A_2, t_2 = solve_orbit_kepler(mu_sun, r_mars, t_span_2, **kwargs)

    # 保存数据到文件
    np.savez('data/orbit_earth.npz', r = r_1, t = t_1, A = A_1)
    np.savez('data/orbit_mars.npz', r = r_2, t = t_2, A = A_2)

# @debug_int
@print_function_name
def arrive_int(**kwargs):
    #################################################################################
    # 
    #       数值积分递推轨道（t_0至到达火星影响球）
    # 
    #################################################################################
    # 积分递推
    t_span = (0.0, time_factor * day_arrive_mars)
    r_arrive_int, A_arrive_int, t_arrive_int = solve_orbit_integrate(mu_sun, r_0, t_span, **kwargs)

    # 保存数据到文件
    np.savez('data/arrive_orbit_int.npz', r = r_arrive_int, t = t_arrive_int, A = A_arrive_int)
    np.savez('data/lastRV/arrive_lastRV_int.npz', r = r_arrive_int[-1][0: 3], v = r_arrive_int[-1][3: ])

@print_function_name
def arrive_kepler(**kwargs):
    #################################################################################
    # 
    #       开普勒方法求解轨道（t_0至到达火星影响球）
    # 
    #################################################################################
    # 分别对上一次积分的各个时间点求解（便于对比）
    data = np.load('data/arrive_orbit_int.npz')
    t_target = tuple(data['t'].tolist())
    r_arrive_kepler, A_arrive_kepler, t_arrive_kepler = solve_orbit_kepler(mu_sun, r_0, t_target, **kwargs)

    # 保存数据到文件
    np.savez('data/arrive_orbit_kepler.npz', r = r_arrive_kepler, t = t_arrive_kepler, A = A_arrive_kepler)
    np.savez('data/lastRV/arrive_lastRV_kepler.npz', r = r_arrive_kepler[-1][0: 3], v = r_arrive_kepler[-1][3: ])
    
@print_function_name
def arrive_lagrange(**kwargs):
    #################################################################################
    # 
    #       拉格朗日法（级数）求解轨道（t_0至到达火星影响球）
    # 
    #################################################################################
    # 分别对上一次积分的各个时间点求解（便于对比）
    data = np.load('data/arrive_orbit_int.npz')
    t_lagrange = tuple(data['t'].tolist())
    r_arrive_lagrange, A_arrive_lagrange, t_arrive_lagrange = solve_orbit_lagrange(mu_sun, r_0, t_lagrange, **kwargs)

    # 保存数据到文件
    np.savez('data/arrive_orbit_lagrange.npz', r = r_arrive_lagrange, t = t_arrive_lagrange, A = A_arrive_lagrange)
    np.savez('data/lastRV/arrive_lastRV_lagrange.npz', r = r_arrive_lagrange[-1][0: 3], v = r_arrive_lagrange[-1][3: ])

@print_function_name
def arrive_lagrange_2(**kwargs):
    #################################################################################
    # 
    #       拉格朗日法(闭合)求解轨道（t_0至到达火星影响球）
    # 
    #################################################################################
    # 分别对上一次积分的各个时间点求解（便于对比）
    data = np.load('data/arrive_orbit_int.npz')
    t_lagrange = tuple(data['t'].tolist())
    r_arrive_lagrange, A_arrive_lagrange, t_arrive_lagrange = solve_orbit_lagrange_2(mu_sun, r_0, t_lagrange, **kwargs)

    # 保存数据到文件
    np.savez('data/arrive_orbit_lagrange_2.npz', r = r_arrive_lagrange, t = t_arrive_lagrange, A = A_arrive_lagrange)
    np.savez('data/lastRV/arrive_lastRV_lagrange_2.npz', r = r_arrive_lagrange[-1][0: 3], v = r_arrive_lagrange[-1][3: ])

@print_function_name
def with_int(**kwargs):
    #################################################################################
    # 
    #       数值积分递推轨道（飞进飞出火星影响球）
    # 
    #################################################################################
    # 相对于火星的航天器运动参数
    data = np.load('data/lastRV/arrive_lastRV_kepler.npz')
    RV_arrive_SOI = np.concatenate((data['r'], data['v']))
    r_mars_relative = RV_arrive_SOI - r_mars

    # 积分递推
    t_span = (0.0, time_factor * day_with_mars)
    r_with_int, A_with_int, t_with_int = solve_orbit_integrate(mu_mars, r_mars_relative, t_span, **kwargs)

    # 保存数据到文件
    np.savez('data/with_orbit_int.npz', r = r_with_int, t = t_with_int, A = A_with_int)
    np.savez('data/lastRV/with_lastRV_int.npz', r = r_with_int[-1][0: 3], v = r_with_int[-1][3: ])

@print_function_name
def with_kepler(**kwargs):
    #################################################################################
    # 
    #       开普勒方法求解轨道（飞进飞出火星影响球）
    # 
    #################################################################################
    # 相对于火星的航天器运动参数
    data = np.load('data/lastRV/arrive_lastRV_kepler.npz')
    RV_arrive_SOI = np.concatenate((data['r'], data['v']))
    r_mars_relative = RV_arrive_SOI - r_mars

    # 分别对上一次积分的各个时间点求解
    data = np.load('data/with_orbit_int.npz')
    t_target = tuple(data['t'].tolist())
    r_with_kepler, A_with_kepler, t_with_kepler = solve_orbit_kepler(mu_mars, r_mars_relative, t_target, **kwargs)

    # 保存数据到文件
    np.savez('data/with_orbit_kepler.npz', r = r_with_kepler, t = t_with_kepler, A = A_with_kepler)
    np.savez('data/lastRV/with_lastRV_kepler.npz', r = r_with_kepler[-1][0: 3], v = r_with_kepler[-1][3: ])

@print_function_name
def leave_int(**kwargs):
    #################################################################################
    # 
    #       数值积分递推轨道（离开火星影响球）
    # 
    #################################################################################
    # 相对于太阳的航天器运动参数（忽略飞过火星时火星运动）
    data = np.load('data/lastRV/with_lastRV_int.npz')
    RV_leave_SOI = np.concatenate((data['r'], data['v']))
    r_sun_relative = RV_leave_SOI + r_mars

    # 积分递推
    t_span = (0.0, time_factor * day_leave_mars)
    r_leave_int, A_leave_int, t_leave_int = solve_orbit_integrate(mu_sun, r_sun_relative, t_span, **kwargs)

    # 保存数据到文件
    np.savez('data/leave_orbit_int.npz', r = r_leave_int, t = t_leave_int, A = A_leave_int)
    np.savez('data/lastRV/leave_lastRV_int.npz', r = r_leave_int[-1][0: 3], v = r_leave_int[-1][3: ])

@print_function_name
def leave_kepler(**kwargs):
    #################################################################################
    # 
    #       开普勒方法求解轨道（离开火星影响球）
    # 
    #################################################################################
    # 相对于太阳的航天器运动参数（忽略飞过火星时火星运动）
    data = np.load('data/lastRV/with_lastRV_int.npz')
    RV_leave_SOI = np.concatenate((data['r'], data['v']))
    r_sun_relative = RV_leave_SOI + r_mars

    # 分别对上一次积分的各个时间点求解
    data = np.load('data/leave_orbit_int.npz')
    t_target = tuple(data['t'].tolist())
    r_leave_kepler, A_leave_kepler, t_leave_kepler = solve_orbit_kepler(mu_sun, r_sun_relative, t_target, **kwargs)

    # 保存数据到文件
    np.savez('data/leave_orbit_kepler.npz', r = r_leave_kepler, t = t_leave_kepler, A = A_leave_kepler)
    np.savez('data/lastRV/leave_lastRV_kepler.npz', r = r_leave_kepler[-1][0: 3], v = r_leave_kepler[-1][3: ])

if __name__ == "__main__":
    main()