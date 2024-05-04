"""
二体轨道动力学相关方程
    运动微分方程  
    开普勒方程
    巴克方程求解
    拉格朗日系数计算(级数)
    拉格朗日系数计算(闭合)
fishyy  24.04.09 -- 24.05.02
"""

import numpy as np

def two_body_dynamics(r, t, mu):
    """
    二体动力学方程。

    参数:
        r: array_like
            控制器的状态向量，包含位置向量和速度向量。
            r = [x, y, z, vx, vy, vz]，其中 x、y、z 是位置坐标，vx、vy、vz 是速度分量。
        t: float
            此项未用到。
        mu: float
            中心天体的引力常数，单位为 km^3/s^2。

    返回值:
        drdt: array_like
            状态向量的导数，即速度向量和加速度向量。
            drdt = [vx, vy, vz, ax, ay, az]，其中 ax、ay、az 是加速度分量。
    """
    x, y, z, vx, vy, vz = r

    distance = np.sqrt(x**2 + y**2 + z**2)
    a_gravity = -mu / distance**3 * np.array([x, y, z])
    ax, ay, az = a_gravity

    # 返回状态向量的导数
    return np.array([vx, vy, vz, ax, ay, az])

def keplers_equation_and_derivative(E, M, e):
    """
    开普勒方程与其导数。

    参数：
        E: float
            偏近点角（自变量）。
        M: float
            平近点角（不变参数）。
        e: float
            偏心率（不变参数）。

    返回值：
        result: float
            对应点的函数值。
        d_result: float
            对应点的导数值。

    备注：
        对于抛物线轨道，返回巴克方程及其导数
    """

    if e < 1:  # 椭圆轨道
        result = E - e * np.sin(E) - M
        d_result = 1 - e * np.sin(E)
    elif e > 1:  # 双曲线轨道
        result = e * np.sinh(E) - E - M
        d_result = e * np.cosh(E) - 1
    else:
        result = 1/2 * E + 1/6 * E**3 - M
        d_result = 1/2 * (E**2 + 1)
    
    return result, d_result

def barkers_equation(M):
    """
    抛物线巴克方程直接求解
    
    参数：
        M: float
            平近点角。

    返回值：
        result: float
            对应平近点角的抛物线巴克方程的解。
    """
    X = 3 * M
    Y = np.cbrt(X + np.sqrt(X**2 + 1))
    return 2 * np.arctan(Y - 1 / Y)

def lagrange_coefficient(mu, a, r, dr, t):
    """
    拉格朗日系数计算（级数形式）
    
    参数：
        mu: float
            中心天体引力常数
        a: float
            轨道半长轴
        r: float
            位置标量(t_0时)
        dr: float
            速度标量(t_0时)
        t: float
            经过时间，t_0默认为0
        
    返回值：
        f, g, df, dg: float
            求解出的系数值
    """
    f0 = 1
    f1 = 0
    f2 = -0.5 * mu / r**3
    f3 = 0.5 * mu * dr / r**4
    f4 = 0.5 * mu**2 / r**6 *(1/3 - 1/4 * r/a - 5/4 * r * dr**2/mu)
    f5 = -0.5 * mu**2 * dr / r**7 * (1 - 3/4 * r/a - 7/4 * r * dr**2/mu)
    
    g0 = 0
    g1 = 1
    g2 = 0
    g3 = -1/6 * mu / r**3
    g4 = 1/4 * mu * dr / r**4
    g5 = 1/4 * mu**2 / r**6 * (1/3 - 3/10 * r / a - 3/2 * r * dr**2 / mu)
    
    f = ((((f5 * t + f4) * t + f3) * t + f2) * t + f1) * t + f0
    g = ((((g5 * t + g4) * t + g3) * t + g2) * t + g1) * t + g0
    df = (((5*f5 * t + 4*f4) * t + 3*f3) * t + 2*f2) * t + f1
    dg = (((5*g5 * t + 4*g4) * t + 3*g3) * t + 2*g2) * t + g1
    
    return f, g, df, dg

def lagrange_coefficient_2(mu, r, dr, delta_f):
    """
    拉格朗日系数计算（闭合形式）
    
    参数：
        mu: float
            中心天体引力常数
        r: array
            位置**矢量**(t_0时)
        dr: array
            速度**矢量**(t_0时)
        delta_f: float
            真近点角变化量
        
    返回值：
        f, g, df, dg: float
            求解出的系数值
    """
    h = np.linalg.norm(np.cross(r, dr))
    r_norm = np.linalg.norm(r)
    dr_norm = np.linalg.norm(dr)
    
    # 简化书写的中间变量
    tmp_1 = h**2 / mu / r_norm - 1
    tmp_2 = h * dr_norm / mu
    tmp_cos = np.cos(delta_f)
    tmp_sin = np.sin(delta_f)
    
    f = 1 - (1 - tmp_cos) / (1 + tmp_1 * tmp_cos - tmp_2 * tmp_sin)
    g = h * r_norm * tmp_sin / mu / (1 + tmp_1 * tmp_cos - tmp_2 * tmp_sin)
    dg = 1 - mu * r_norm / h**2 * (1 - tmp_cos)
    if g == 0:
        df = 0
    else:
        df = (f * dg - 1) / g
    
    return f, g, df, dg


