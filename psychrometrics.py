"""
湿空气热力学计算模块
基于 ASHRAE Handbook of Fundamentals (2013) Chapter 1 - Psychrometrics
及 ASHRAE RP-1485 / RP-1460 计算方法

参考知识库标准:
  - SI_F13_Ch01 = 湿空气学 (ASHRAE Handbook 2013 Fundamental Ch01)
  - air psychrometry 文件夹中的 PsychrLib 文档
"""
import numpy as np
from scipy.optimize import brentq


# ============================================================
# ASHRAE Hyland-Wexler 饱和水蒸气压方程 (ASHRAE Fundamentals 2013, Ch.1, Eq. 5 & 6)
# ============================================================

def saturation_pressure_water(T_celsius: float) -> float:
    """
    计算水面饱和水蒸气压 (Pa)
    ASHRAE Fundamentals 2013 Ch.1, Eq.5 (Hyland-Wexler)
    适用温度范围: -100 ~ 200 °C (低于0°C使用冰面Hyland-Wexler方程)

    Parameters
    ----------
    T_celsius : float
        温度 (°C)

    Returns
    -------
    float
        饱和水蒸气压 (Pa)
    """
    T = T_celsius + 273.15  # 转为开尔文
    if T < 273.15:
        # 冰面饱和蒸气压 (ASHRAE Eq.6)
        C = [-5.6745359e3, 6.3925247, -9.6778430e-3,
             6.2215701e-7, 2.0747825e-9, -9.4840240e-13, 4.1635019]
        ln_pws = (C[0] / T + C[1] + C[2] * T + C[3] * T**2 +
                  C[4] * T**3 + C[5] * T**4 + C[6] * np.log(T))
    else:
        # 水面饱和蒸气压 (ASHRAE Eq.5)
        C = [-5.8002206e3, 1.3914993, -4.8640239e-2,
             4.1764768e-5, -1.4452093e-8, 6.5459673]
        ln_pws = (C[0] / T + C[1] + C[2] * T + C[3] * T**2 +
                  C[4] * T**3 + C[5] * np.log(T))
    return np.exp(ln_pws)


def enhancement_factor(p: float, T_celsius: float) -> float:
    """
    增强因子 f (ASHRAE Fundamentals 2013 Ch.1, Eq.12)
    修正大气压对饱和蒸气压的影响

    Parameters
    ----------
    p : float
        大气压 (Pa)
    T_celsius : float
        温度 (°C) — ASHRAE公式使用摄氏温度, 非开尔文
    """
    # ASHRAE Fundamentals 2013 Ch.1 Eq.12: t 为摄氏温度
    alpha = 1.00062
    beta = 3.14e-8
    gamma = 5.67e-7
    f = alpha + beta * p + gamma * T_celsius**2
    return f


def humidity_ratio(T_db: float, rh: float, p: float = 101325.0) -> float:
    """
    含湿量 W (kg水蒸气/kg干空气)
    ASHRAE Fundamentals 2013 Ch.1, Eq.20 & 22

    Parameters
    ----------
    T_db : float
        干球温度 (°C)
    rh : float
        相对湿度 (0~1)
    p : float
        大气压 (Pa)
    """
    f = enhancement_factor(p, T_db)
    p_ws = saturation_pressure_water(T_db)
    p_w = f * rh * p_ws  # 实际水蒸气分压
    W = 0.621945 * p_w / (p - p_w)
    return W


def humidity_ratio_from_wb(T_db: float, T_wb: float, p: float = 101325.0) -> float:
    """
    由干球温度和湿球温度计算含湿量
    ASHRAE Fundamentals 2013 Ch.1, Eq.33-35

    Parameters
    ----------
    T_db : float
        干球温度 (°C)
    T_wb : float
        湿球温度 (°C)
    p : float
        大气压 (Pa)
    """
    f_db = enhancement_factor(p, T_db)
    f_wb = enhancement_factor(p, T_wb)
    p_ws_db = saturation_pressure_water(T_db)
    p_ws_wb = saturation_pressure_water(T_wb)

    W_s_wb = 0.621945 * f_wb * p_ws_wb / (p - f_wb * p_ws_wb)

    # ASHRAE Eq.35
    W = ((2501 - 2.326 * T_wb) * W_s_wb - 1.006 * (T_db - T_wb)) / \
        (2501 + 1.86 * T_db - 4.186 * T_wb)
    return W


def enthalpy(T_db: float, W: float) -> float:
    """
    湿空气比焓 (kJ/kg干空气)
    ASHRAE Fundamentals 2013 Ch.1, Eq.28

    h = 1.006*t + W*(2501 + 1.86*t)

    Parameters
    ----------
    T_db : float
        干球温度 (°C)
    W : float
        含湿量 (kg/kg)
    """
    return 1.006 * T_db + W * (2501 + 1.86 * T_db)


def specific_volume(T_db: float, W: float, p: float = 101325.0) -> float:
    """
    湿空气比容 (m³/kg干空气)
    ASHRAE Fundamentals 2013 Ch.1, Eq.26

    v = R_da*T*(1 + 1.6078*W) / p
    """
    R_da = 287.042  # 干空气气体常数 (J/kg·K)
    T = T_db + 273.15
    return R_da * T * (1 + 1.6078 * W) / p


def density(T_db: float, W: float, p: float = 101325.0) -> float:
    """湿空气密度 (kg湿空气/m³)"""
    v = specific_volume(T_db, W, p)
    return (1 + W) / v


def relative_humidity(T_db: float, W: float, p: float = 101325.0) -> float:
    """
    相对湿度 φ
    ASHRAE Fundamentals 2013 Ch.1, Eq.22-24
    """
    f = enhancement_factor(p, T_db)
    p_ws = saturation_pressure_water(T_db)
    p_w = W * p / (0.621945 + W)
    rh = p_w / (f * p_ws)
    return min(max(rh, 0.0), 1.0)


def dew_point(T_db: float, W: float, p: float = 101325.0) -> float:
    """
    露点温度 (°C)
    ASHRAE Fundamentals 2013 Ch.1, Eq.39-40
    """
    p_w = W * p / (0.621945 + W)

    # 求解 p_ws(T_dp) = p_w
    def _func(T):
        return saturation_pressure_water(T) - p_w

    try:
        T_dp = brentq(_func, -60, 90)
    except Exception:
        T_dp = T_db
    return T_dp


def wet_bulb_temperature(T_db: float, W: float, p: float = 101325.0) -> float:
    """
    热力学湿球温度 (°C)
    通过迭代求解 ASHRAE Eq.35 的逆问题
    """
    def _func(T_wb):
        W_calc = humidity_ratio_from_wb(T_db, T_wb, p)
        return W_calc - W

    try:
        T_wb = brentq(_func, -60, min(T_db, 50))
    except Exception:
        T_wb = T_db - 2.0
    return T_wb


def saturated_enthalpy(T: float, p: float = 101325.0) -> float:
    """饱和空气的比焓 (kJ/kg干空气) - 用于湿工况计算"""
    W_s = 0.621945 * saturation_pressure_water(T) / (p - saturation_pressure_water(T))
    return enthalpy(T, W_s)


def saturated_humidity_ratio(T: float, p: float = 101325.0) -> float:
    """饱和含湿量 (kg/kg)"""
    p_ws = saturation_pressure_water(T)
    return 0.621945 * p_ws / (p - p_ws)


class MoistAir:
    """湿空气状态点 - ASHRAE Psychrometrics"""

    def __init__(self, T_db: float, W: float, p: float = 101325.0):
        self.T_db = T_db        # 干球温度 (°C)
        self.W = W              # 含湿量 (kg/kg)
        self.p = p              # 大气压 (Pa)
        self._update()

    def _update(self):
        self.h = enthalpy(self.T_db, self.W)
        self.v = specific_volume(self.T_db, self.W, self.p)
        self.rho = density(self.T_db, self.W, self.p)
        self.rh = relative_humidity(self.T_db, self.W, self.p)
        self.T_dp = dew_point(self.T_db, self.W, self.p)
        self.T_wb = wet_bulb_temperature(self.T_db, self.W, self.p)

    @classmethod
    def from_db_rh(cls, T_db: float, rh: float, p: float = 101325.0):
        W = humidity_ratio(T_db, rh, p)
        return cls(T_db, W, p)

    @classmethod
    def from_db_wb(cls, T_db: float, T_wb: float, p: float = 101325.0):
        W = humidity_ratio_from_wb(T_db, T_wb, p)
        return cls(T_db, W, p)

    def __repr__(self):
        return (f"MoistAir(Tdb={self.T_db:.1f}°C, W={self.W*1000:.2f}g/kg, "
                f"RH={self.rh*100:.1f}%, h={self.h:.1f}kJ/kg, "
                f"Tdp={self.T_dp:.1f}°C, Twb={self.T_wb:.1f}°C)")
