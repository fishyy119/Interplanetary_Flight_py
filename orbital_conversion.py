"""
轨道要素转换
    位置速度矢量转轨道六根数
    轨道六根数转速度位置矢量
    给定时刻平近点角计算
fishyy  24.04.10 -- 24.04.12
"""
import numpy as np

def state_to_kepler(r, v, mu):
    """
    将位置速度矢量转换为轨道六根数。

    参数:
        r: array_like
            位置矢量，单位为 km。
        v: array_like
            速度矢量，单位为 km/s。
        mu: float
            中心天体的标准引力参数，单位为 km^3/s^2。

    返回值:
        a: float
            半长轴，单位为 km。
        e: float
            偏心率。
        i: float
            轨道的倾角，单位为弧度。
        Omega: float
            升交点赤经，单位为弧度。
        omega: float
            近地点幅角，单位为弧度。
        f: float
            真近点角，单位为弧度。
        E: float
            偏近点角，单位为弧度。
        M: float
            平近点角，单位为弧度。
    """
    # 计算位置向量和速度向量的模
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    # 计算角动量矢量
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    # 计算轨道倾角
    i = np.arccos(h[2] / h_norm)

    # 计算升交点赤经
    N = np.array([-h[1], h[0], 0])
    Omega = np.arctan2(N[1], N[0])
    if Omega < 0:
        Omega += 2 * np.pi

    # 计算偏心率矢量
    e_vec = np.cross(v, h) / mu - r / r_norm
    e = np.linalg.norm(e_vec)
    
    # 计算近地点幅角
    omega = angle_between_vectors(N, e_vec)
    if e_vec[2] < 0:
        omega = 2 * np.pi - omega
    
    # 计算真近点角
    f = angle_between_vectors(r, e_vec)
    if np.dot(r, v) < 0:
        f = 2 * np.pi - f
    
    # 计算半长轴与偏近点角、平近点角
    if e != 1:
        a = 1 / (2 / r_norm - v_norm**2 / mu)
        if e < 1:
            E = np.arccos((a - r_norm) / (a * e))
            if np.sin(f) < 0:
                E = 2 * np.pi - E

            M = E - e * np.sin(E)

        elif e > 1:
            n = np.sqrt(-mu / a**3)
            E = np.arctanh(np.dot(r, v) / (a * n * (r_norm - a)))

            M = e * np.sinh(E) - E
    else:
        a = h_norm**2 / mu  # 没有半长轴，使用半正焦弦代替
        E = np.tan(f / 2)
        M = E / 2 + E**3 / 6
        
    return a, e, i, Omega, omega, f, E, M

def E_to_f(E, e, a):
    """
    将偏近点角转换至真近点角
    
    参数:
        E: float
            偏近点角，弧度
        e: float
            偏心率。
        a: float
            半长轴，km

    返回值:
        f: float
            真近点角，弧度
    """
    if e == 1: 
        f = 2 * np.arctan(E)
    else:
        if e < 1:
            r = a * (1 - e * np.cos(E))
        elif e > 1:
            r = a * (1 - e * np.cosh(E))
        
        cos_f = ((a * (1 - e**2)) / r - 1) / e
        f = np.arccos(cos_f)
        if np.sin(E) < 0:
            f = 2 * np.pi - f
            
    return f
            
def kepler_to_state(a, e, i, Omega, omega, f_or_E, mu, option = 'E'):
    """
    将轨道六根数转换为位置速度矢量。

    参数:
        a: float
            半长轴，单位为 km。
        e: float
            偏心率。
        i: float
            轨道的倾角，单位为弧度。
        Omega: float
            升交点赤经，单位为弧度。
        omega: float
            近地点幅角，单位为弧度。
        f_or_E: float
            真近点角或偏近点角，单位为弧度。
        mu: float
            中心天体的标准引力参数，单位为 km^3/s^2。
        option: 'f' / 'E'(default)
            指示提供参数为真近点角/偏近点角

    返回值:
        r_vec: array_like
            位置矢量，单位为 km。
        v_vec: array_like
            速度矢量，单位为 km/s。
    """
    # 计算距离r与真近点角的三角函数值
    if option == 'f':
        f = f_or_E
        sin_f = np.sin(f)
        cos_f = np.cos(f)
        if e == 1:
            r = a / (1 + e * cos_f)  # 此处a为半正焦弦p
        else:
            r = a * (1 - e**2) / (1 + e * cos_f)
    else:
        E = f_or_E
        if e == 1:
            f = 2 * np.arctan(E)
            sin_f = np.sin(f)
            cos_f = np.cos(f)
            r = a / (1 + e * cos_f)
        else:
            if e < 1:
                r = a * (1 - e * np.cos(E))
            elif e > 1:
                r = a * (1 - e * np.cosh(E))
    
            cos_f = ((a * (1 - e**2)) / r - 1) / e
            f = np.arccos(cos_f)
            if np.sin(E) < 0:
                f = 2 * np.pi - f
            sin_f = np.sin(f)

    # 转换矩阵计算
    sin_Omega = np.sin(Omega)
    cos_Omega = np.cos(Omega)
    sin_i = np.sin(i)
    cos_i = np.cos(i)
    sin_u = np.sin(omega) * cos_f + np.cos(omega) * sin_f
    cos_u = np.cos(omega) * cos_f - np.sin(omega) * sin_f

    rotation_matrix = np.array([
        [cos_Omega, -sin_Omega, 0],
        [sin_Omega, cos_Omega, 0],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i]
    ]) @ np.array([
        [cos_u, -sin_u, 0],
        [sin_u, cos_u, 0],
        [0, 0, 1]
    ])

    # 计算速度分量
    v_perp = np.sqrt(mu * a * (1 - e**2)) / r
    v_r = e * sin_f * np.sqrt(mu / (a * (1 - e**2)))

    r_orbit = np.array([r, 0, 0])
    v_orbit = np.array([v_r, v_perp, 0])
    
    r_vec_I = rotation_matrix @ r_orbit
    v_vec_I = rotation_matrix @ v_orbit

    return r_vec_I, v_vec_I

def calculate_M(M_0, t, mu, a, e):
    """
    计算给定时刻平近点角
    
    参数：
        M_0: float
            初始平近点角。
        t: float
            时刻。
        mu: float
            中心天体的标准引力参数。
        a: float
            半长轴。
        e: float
            偏心率。

    返回值：
        M: float
            对应给定时刻的平近点角。

    备注：
        对于抛物线轨道，a处提供半正焦弦p。
    """
    # 平均角速度
    n = np.sqrt(mu / np.abs(a)**3)
        
    return M_0 + n * t

def angle_between_vectors(v1, v2):
    """
    计算两个向量之间的夹角（弧度）。

    参数：
    v1：第一个向量，numpy 数组。
    v2：第二个向量，numpy 数组。

    返回值：
    夹角的弧度值。
    """
    # 计算向量的范数（模）
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 计算向量的点积
    dot_product = np.dot(v1, v2)

    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_v1 * norm_v2)

    # 计算夹角（弧度）
    angle_rad = np.arccos(cos_angle)
    
    return angle_rad