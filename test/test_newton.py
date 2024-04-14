"""
对牛顿迭代法的简单验证

测试用例：
f(x) = x**2 - 2

理论解：
x = sqrt(2)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from numerical_methods import newton_method

def func(x):
    f = x**2 - 2
    df = 2 * x
    return f, df

x, n = newton_method(func, 1)
print(f"x={x},n={n}")