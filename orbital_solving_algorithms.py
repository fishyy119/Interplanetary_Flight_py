"""
轨道求解算法（综合调用其他基础函数）
    数值积分方法求解轨道
    开普勒方法求解轨道
fishyy  24.04.13 -- 24.04.13
"""
import numpy as np
from functools import partial
from numerical_methods import newton_method, runge_kutta45
from orbital_conversion import kepler_to_state, state_to_kepler, calculate_M
from two_body_orbit_dynamics import keplers_equation_and_derivative, two_body_dynamics

def solve_orbit_integrate(mu, r_0, t_span, **kwargs):
    """
    调用数值积分求解器求解轨道。
    
    参数:
        mu: float
            中心天体引力常数。
        r_0: array_like
            初始状态向量。
        t_span: tuple
            一个元组 (t_0, ... ,t_end)，
            首尾两项指定积分的初始和最终时间，
            中间的不定多项指定期望求解的准确时刻。
        **kwargs: dict
            传递给 runge_kutta45 函数的其他参数。
    
    返回值:
        r: array_like
            在积分时间范围内的状态向量数组。
        A: array_like
            对应于每个状态向量的轨道六根数数组。
            [a, e, i, Omega, omega, f, E, M]
        t: array_like
            对应于每个状态向量的时间值数组。
    """
    A = []
    two_body_with_mu = partial(two_body_dynamics, mu=mu)
    r, t = runge_kutta45(two_body_with_mu, r_0=r_0, t_span=t_span, **kwargs)
    for r_single in r:
        r_vec_new = r_single[0: 3]
        v_vec_new = r_single[3: ]
        A.append(list(state_to_kepler(r_vec_new, v_vec_new, mu)))
        
    return r, np.array(A), t

def solve_orbit_kepler(mu, r_0, t_target, **kwargs):
    """
    开普勒方法求解轨道。

    参数:
        mu: float
            中心天体引力常数。
        r_0: array_like
            初始状态向量。
        t_target: tuple
            不定多项时间，分别求解对应时间处轨道状态
        **kwargs: dict
            传递给 newton_method 函数的其他参数。

    返回值:
        r: array_like
            对应时间的状态向量数组。
        A: array_like
            对应于每个状态向量的轨道六根数数组（由r再次反解）。
            [a, e, i, Omega, omega, f, E, M]
        t: array_like
            对应于每个状态向量的时间值数组。
    """
    # 速度位置矢量转轨道六根数
    a, e, i, Omega, omega, f, E, M = state_to_kepler(r_0[0:3], r_0[3:], mu)
    A = [np.array([a, e, i, Omega, omega, f, E, M])]
    if t_target[0] == 0:
        r = [r_0]
        t = [0]
        t_new_target = t_target[1:]
    else:
        r = []
        t = []
        t_new_target = t_target

    for t_end in t_new_target:
        # 计算给定时间后平近点角
        M_new = calculate_M(M, t_end, mu, a, e)

        # 开普勒方程求解
        target_func = partial(keplers_equation_and_derivative, M=M_new, e=e)
        if e > 1: # 选取其他初值
            E_new, _ = newton_method(target_func, np.log(2 * M_new / e), **kwargs)
        else:
            E_new, _ = newton_method(target_func, M_new, **kwargs)

        # 轨道六根数转速度位置矢量
        r_vec_new, v_vec_new = kepler_to_state(a, e, i, Omega, omega, E_new, mu, option='E')
        r.append(np.concatenate((r_vec_new,v_vec_new)))
        t.append(t_end)
        A.append(list(state_to_kepler(r_vec_new, v_vec_new, mu)))

    return np.array(r), np.array(A), np.array(t)