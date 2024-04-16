"""
对RK45积分器的简单验证

测试用例：
dy/dx = y - 2 * x / y
y_0 = 1

真实解：y = sqrt(1 + 2 * x)
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from numerical_methods import runge_kutta45

def sample(x, t):
    return x - 2 * t / x

def solution(x):
    return np.sqrt(1 + 2 * x)

t_span = (0, 1.4, 2)
r, t = runge_kutta45(sample, 1, t_span, tol=1e-4, dt_init=1e-4, max_dt=1, debug=True)
error = r - solution(t)
print(f"{error}")
