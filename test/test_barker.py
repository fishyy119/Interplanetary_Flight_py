"""
对barkers_equation()的简单验证

迭代求解与解析求解比较
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from two_body_orbit_dynamics import barkers_equation, keplers_equation_and_derivative
from numerical_methods import newton_method
from functools import partial
import numpy as np

e = 1
for M in np.linspace(0, 1, 100):
    func1 = partial(keplers_equation_and_derivative, e = e, M = M)
    result_1, _ = newton_method(func1, M)
    result_1 = 2 * np.arctan(result_1)
    result_2 = barkers_equation(M)
    print(f"{result_1},{result_2}")
