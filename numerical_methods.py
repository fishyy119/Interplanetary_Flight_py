"""
数值计算方法：
    龙格库塔4-5阶方法  
    牛顿迭代法
fishyy  24.04.09 -- 24.04.12
"""
import warnings
import numpy as np

def runge_kutta45(f, r_0, t_span, tol=1e-3, dt_init=1e-4, max_dt=-1, debug=False):
    """
    龙格-库塔4-5阶方法(Dormand-Prince 法)用于数值积分。

    参数:
        f: function
            微分方程右侧的函数。
            此函数应接受两个参数：当前状态 r 和当前时间 t，
            并返回状态相对于时间的导数。
            (dr/dt = f(r, t))
            在处理一阶常微分方程组时，f 应返回一个包含多个导数的数组。
        r_0: array_like
            初始状态向量。
        t_span: tuple
            一个元组 (t_0, ... ,t_end)
            首尾两项指定积分的初始和最终时间，
            中间的不定多项指定期望求解的准确时刻。
        dt_init: float
            积分时间步长（初始值）。
        tol: float, optional
            容许误差。默认值为1e-6。
        max_dt: float, optional
            最大时间步长。默认值为(t_end-t_0)/10。
        debug: bool
            默认False，开启则输出循环信息

    返回值:
        r: array_like
            在积分时间范围内的状态向量数组。
        t: array_like
            对应于每个状态向量的时间值数组。
    """
    dt = dt_init
    t_0 = t_span[0]
    t_end = t_span[-1]
    t_target = t_span[1:]
    if max_dt <= 0:
        max_dt = (t_end - t_0) / 10
    t = [t_0]
    r = [r_0]
    f7 = f(r_0, t_0)  # 前一次循环的f7是下一次循环的f1，进入循环后此值会自然赋给f1

    while t[-1] < t_end:
        f1 = f7  # 和上一次循环的f7是同一个值，减少计算量
        k1 = dt * f1

        f2 = f(r[-1] + 1/5 * k1, t[-1] + 1/5 * dt)
        k2 = dt * f2

        f3 = f(r[-1] + 3/40 * k1 + 9/40 * k2, t[-1] + 3/10 * dt)
        k3 = dt * f3
        
        f4 = f(r[-1] + 44/45 * k1 - 56/15 * k2 + 32/9 * k3, t[-1] + 4/5 * dt)
        k4 = dt * f4

        f5 = f(r[-1] + 19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4, t[-1] + 8/9 * dt)
        k5 = dt * f5

        f6 = f(r[-1] + 9017/3168 * k1 - 355/33 * k2 - 46732/5247  * k3 + 49/176 * k4 - 5103/18656 * k5, t[-1] + dt)
        k6 = dt * f6

        r_next = r[-1] + 35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6
        t_next = t[-1] + dt
        f7 = f(r_next, t_next)
        k7 = dt * f7

        # 估算误差
        r_star = r[-1] + 5179/57600 * k1 + 7571/16695 * k3 + 393/640 * k4 - 92097/339200 * k5 + 187/2100 * k6 + 1/40 * k7
        eps = np.abs(r_next - r_star)

        # 添加结果
        r.append(r_next)
        t.append(t_next)

        # 如果误差不为零，则计算缩放因子
        s = (tol * dt / (2 * eps + 1e-15)) ** (1/5)  # 根据误差和容许误差计算缩放因子，1e-15是在避免除零
        dt *= np.min(s)  # 更新时间步长
        dt = min(max_dt, dt)  # 限制步长

        if dt < np.finfo(float).eps * t_end:
            raise ValueError(f'递推过程出现了过小的步长,dt={dt}')

        # 对期望点的定位(包括了t_span的中间若干位与结束时刻)
        for t_t in t_target:
            if t_next < t_t and t_next + dt > t_t:
                dt = t_t - t_next

        if debug == True:
            print(f"t={t_next},r={r_next},dt={dt}")

    return np.array(r), np.array(t)

def newton_method(func_and_dfunc, x0, tol = 1e-7, max_iter = 100):
    """
    使用牛顿迭代法求解非线性方程。

    参数:
        func_and_dfunc: function
            原函数及其导函数。
            其输出为tuple[f, d_f]
        x0: float
            迭代初值。
        tol: float, optional
            允许误差，默认值为1e-7。
        max_iter: int, optional
            最大迭代次数，默认值为100。

    返回值:
        x: float
            近似解。
        n_iter: int
            迭代次数。
    """
    x = x0
    for n_iter in range(max_iter):
        fx, dfx = func_and_dfunc(x)
        if abs(fx) < tol:
            return x, n_iter
        
        if dfx == 0:
            raise ValueError("导函数为0，无法继续迭代")
        
        x = x - fx / dfx
    fx, _ = func_and_dfunc(x)
    warnings.warn(f"迭代次数达到上限，当前误差{abs(fx)}")
    return x, max_iter

