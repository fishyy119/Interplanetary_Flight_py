"""
二体轨道动力学相关方程
    运动微分方程  
    开普勒方程
    巴克方程求解
    拉格朗日系数计算
fishyy  24.04.09 -- 24.04.12
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
            定常，此项未用到。
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

def lagrange_coefficient():
    """
    拉格朗日系数计算（级数形式）
    
    参数：
        

    返回值：
        f, g, df, dg: float
            求解出的系数值
    """
    pass

