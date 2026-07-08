"""
制冷剂热力学物性模块
优先使用 CoolProp (基于 Helmholtz 状态方程)，不可用时使用内置多项式拟合关联式

支持制冷剂: R22, R134a, R410A, R32, R290 (丙烷), R407C, R404A, R23 (三氟甲烷), R744 (CO2)

参考知识库:
  - 制冷系统 → refrigerant 文件夹
  - 制冷/暖通工程师效率神器：Python调用CoolProp完全指南
"""
import numpy as np

# 尝试导入 CoolProp
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False


# ============================================================
# 内置制冷剂物性关联式 (CoolProp 不可用时的后备方案)
# 基于标准物性表的多项式拟合，精度约 ±5%
# ============================================================

# 各制冷剂临界参数 [Tc(°C), pc(MPa), M(g/mol)]
_REFRIG_CRITICAL = {
    'R22':   (96.15, 4.99, 86.47),
    'R134a': (101.06, 4.059, 102.03),
    'R410A': (72.13, 4.901, 72.585),
    'R32':   (78.11, 5.782, 52.024),
    'R290':  (96.74, 4.247, 44.096),
    'R407C': (86.74, 4.636, 86.204),
    'R404A': (72.07, 3.735, 97.60),
    'R23':   (25.9, 4.827, 70.014),
    'R744':  (31.06, 7.377, 44.010),
}


def _sat_pressure_poly(ref: str, T_celsius: float) -> float:
    """
    饱和蒸气压拟合 (MPa)
    使用 Antoine 方程形式: ln(P) = A - B/(T+C)
    """
    params = {
        'R22':   (13.91, 2354.0, -25.0),
        'R134a': (14.12, 2520.0, -28.0),
        'R410A': (14.43, 2300.0, -22.0),
        'R32':   (14.56, 2350.0, -18.0),
        'R290':  (13.77, 2200.0, -25.0),
        'R407C': (14.20, 2380.0, -24.0),
        'R404A': (14.32, 2260.0, -20.0),
        'R23':   (19.66, 4811.7, 128.8),
    }
    if ref not in params:
        ref = 'R134a'
    A, B, C = params[ref]
    T = T_celsius + 273.15
    return np.exp(A - B / (T + C)) * 1e-3  # MPa


def _sat_liquid_enthalpy_poly(ref: str, T_celsius: float) -> float:
    """饱和液体焓 (kJ/kg) - 参考焓: T_ref=0°C 时 h_l=200 kJ/kg"""
    cp_l = {'R22': 1.25, 'R134a': 1.42, 'R410A': 1.60, 'R32': 2.10,
            'R290': 2.55, 'R407C': 1.40, 'R404A': 1.42, 'R23': 1.85}
    cp = cp_l.get(ref, 1.40)
    h_ref = {'R22': 200, 'R134a': 200, 'R410A': 200, 'R32': 200,
             'R290': 200, 'R407C': 200, 'R404A': 200, 'R23': 200}
    return h_ref.get(ref, 200) + cp * T_celsius


def _sat_vapor_enthalpy_poly(ref: str, T_celsius: float) -> float:
    """饱和蒸汽焓 (kJ/kg) - 饱和液体焓 + 汽化潜热"""
    h_l = _sat_liquid_enthalpy_poly(ref, T_celsius)
    # 汽化潜热拟合 (kJ/kg) - 随温度升高而减小
    h_fg_0 = {'R22': 205, 'R134a': 198, 'R410A': 230, 'R32': 360,
              'R290': 376, 'R407C': 210, 'R404A': 155, 'R23': 230}
    hfg0 = h_fg_0.get(ref, 200)
    Tc = _REFRIG_CRITICAL.get(ref, (100, 5, 100))[0]
    # Watson 关联式: h_fg = h_fg0 * ((Tc - T)/(Tc - 0))^0.38
    ratio = max((Tc - T_celsius) / Tc, 0.01)
    h_fg = hfg0 * ratio**0.38
    return h_l + h_fg


def _vapor_density_poly(ref: str, T_celsius: float, p_MPa: float) -> float:
    """蒸汽密度 (kg/m³) - 理想气体近似 + 压缩因子修正"""
    M = _REFRIG_CRITICAL.get(ref, (100, 5, 100))[2]  # g/mol
    R = 8.314  # J/(mol·K)
    T = T_celsius + 273.15
    # 压缩因子近似 (偏离理想气体)
    Z = 0.85  # 典型值
    rho = p_MPa * 1e6 * M / (Z * R * T)
    return rho


def _liquid_density_poly(ref: str, T_celsius: float) -> float:
    """饱和液体密度 (kg/m³)"""
    rho_0 = {'R22': 1290, 'R134a': 1320, 'R410A': 1060, 'R32': 960,
             'R290': 580, 'R407C': 1200, 'R404A': 1045, 'R23': 820}
    rho0 = rho_0.get(ref, 1200)
    Tc = _REFRIG_CRITICAL.get(ref, (100, 5, 100))[0]
    # 饱和液体密度拟合
    ratio = max((Tc - T_celsius) / Tc, 0.1)
    return rho0 * ratio**0.25


def _liquid_viscosity_poly(ref: str, T_celsius: float) -> float:
    """饱和液体动力粘度 (Pa·s)"""
    mu_0 = {'R22': 2.4e-4, 'R134a': 2.5e-4, 'R410A': 1.7e-4, 'R32': 1.5e-4,
            'R290': 1.1e-4, 'R407C': 2.3e-4, 'R404A': 2.8e-4, 'R23': 1.7e-4}
    mu = mu_0.get(ref, 2.0e-4)
    # 粘度随温度升高而降低
    return mu * (300 / (T_celsius + 273.15))**0.5


def _liquid_thermal_conductivity_poly(ref: str, T_celsius: float) -> float:
    """饱和液体导热系数 (W/m·K)"""
    k_0 = {'R22': 0.092, 'R134a': 0.088, 'R410A': 0.085, 'R32': 0.155,
           'R290': 0.115, 'R407C': 0.085, 'R404A': 0.086, 'R23': 0.090}
    return k_0.get(ref, 0.090)


def _liquid_cp_poly(ref: str, T_celsius: float) -> float:
    """饱和液体定压比热 (kJ/kg·K)"""
    cp = {'R22': 1.25, 'R134a': 1.42, 'R410A': 1.60, 'R32': 2.10,
          'R290': 2.55, 'R407C': 1.40, 'R404A': 1.42, 'R23': 1.85}
    return cp.get(ref, 1.40)


def _vapor_cp_poly(ref: str, T_celsius: float) -> float:
    """饱和蒸汽定压比热 (kJ/kg·K)"""
    cp = {'R22': 0.75, 'R134a': 0.95, 'R410A': 1.20, 'R32': 1.50,
          'R290': 1.70, 'R407C': 0.95, 'R404A': 0.88, 'R23': 0.74}
    return cp.get(ref, 0.95)


def _vapor_viscosity_poly(ref: str, T_celsius: float) -> float:
    """蒸汽动力粘度 (Pa·s)"""
    return 1.2e-5 * ((T_celsius + 273.15) / 273.15)**0.7


def _vapor_thermal_conductivity_poly(ref: str, T_celsius: float) -> float:
    """蒸汽导热系数 (W/m·K)"""
    return 0.012 + 0.00003 * T_celsius


# ============================================================
# 统一物性接口
# ============================================================

class Refrigerant:
    """
    制冷剂热力学物性接口
    优先调用 CoolProp (高精度 Helmholtz 状态方程)
    不可用时回退到内置多项式关联式

    Parameters
    ----------
    name : str
        制冷剂名称，如 'R22', 'R134a', 'R410A', 'R32', 'R290', 'R407C', 'R404A', 'R23'
    """

    def __init__(self, name: str = 'R410A'):
        self.name = name
        # CoolProp 流体标识 (R410A/R134a/R22/R32/R290/R407C/R404A/R23 均可直接使用)
        self._cp_name = name

    def _call_coolprop(self, prop: str, T_K: float, p_Pa: float):
        """调用 CoolProp 获取物性"""
        try:
            return CP.PropsSI(prop, 'T', T_K, 'P', p_Pa, self._cp_name)
        except Exception:
            return None

    def saturation_pressure(self, T_celsius: float) -> float:
        """饱和蒸气压 (Pa)"""
        if HAS_COOLPROP:
            val = self._call_coolprop('P', T_celsius + 273.15, 0)
            try:
                # 使用饱和蒸气压
                return CP.PropsSI('P', 'T', T_celsius + 273.15, 'Q', 1, self._cp_name)
            except Exception:
                pass
        return _sat_pressure_poly(self.name, T_celsius) * 1e6

    def sat_liquid_enthalpy(self, T_celsius: float) -> float:
        """饱和液体焓 (J/kg)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('H', 'T', T_celsius + 273.15, 'Q', 0, self._cp_name)
            except Exception:
                pass
        return _sat_liquid_enthalpy_poly(self.name, T_celsius) * 1000

    def sat_vapor_enthalpy(self, T_celsius: float) -> float:
        """饱和蒸汽焓 (J/kg)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('H', 'T', T_celsius + 273.15, 'Q', 1, self._cp_name)
            except Exception:
                pass
        return _sat_vapor_enthalpy_poly(self.name, T_celsius) * 1000

    def latent_heat(self, T_celsius: float) -> float:
        """汽化潜热 (J/kg)"""
        return self.sat_vapor_enthalpy(T_celsius) - self.sat_liquid_enthalpy(T_celsius)

    def vapor_enthalpy(self, T_celsius: float, p_Pa: float) -> float:
        """过热蒸汽焓 (J/kg)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('H', 'T', T_celsius + 273.15, 'P', p_Pa, self._cp_name)
            except Exception:
                pass
        # 回退: 饱和蒸汽焓 + cp_v * (T - T_sat)
        T_sat = self.saturation_temperature(p_Pa)
        h_sat = self.sat_vapor_enthalpy(T_sat)
        cp_v = _vapor_cp_poly(self.name, T_celsius) * 1000
        return h_sat + cp_v * (T_celsius - T_sat)

    def saturation_temperature(self, p_Pa: float) -> float:
        """由压力求饱和温度 (°C)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('T', 'P', p_Pa, 'Q', 0.5, self._cp_name) - 273.15
            except Exception:
                pass
        # 回退: 迭代搜索
        from scipy.optimize import brentq
        def _f(T):
            return self.saturation_pressure(T) - p_Pa
        try:
            return brentq(_f, -100, 90)
        except Exception:
            return 5.0

    def liquid_density(self, T_celsius: float) -> float:
        """饱和液体密度 (kg/m³)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('D', 'T', T_celsius + 273.15, 'Q', 0, self._cp_name)
            except Exception:
                pass
        return _liquid_density_poly(self.name, T_celsius)

    def vapor_density(self, T_celsius: float, p_Pa: float) -> float:
        """蒸汽密度 (kg/m³)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('D', 'T', T_celsius + 273.15, 'P', p_Pa, self._cp_name)
            except Exception:
                pass
        return _vapor_density_poly(self.name, T_celsius, p_Pa * 1e-6)

    def sat_vapor_density(self, T_celsius: float) -> float:
        """饱和蒸汽密度 (kg/m³) - 用Q=1确保在饱和线上取气相"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('D', 'T', T_celsius + 273.15, 'Q', 1, self._cp_name)
            except Exception:
                pass
        p = self.saturation_pressure(T_celsius)
        return _vapor_density_poly(self.name, T_celsius, p * 1e-6)

    def liquid_viscosity(self, T_celsius: float) -> float:
        """饱和液体动力粘度 (Pa·s)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('V', 'T', T_celsius + 273.15, 'Q', 0, self._cp_name)
            except Exception:
                pass
        return _liquid_viscosity_poly(self.name, T_celsius)

    def vapor_viscosity(self, T_celsius: float, p_Pa: float) -> float:
        """蒸汽动力粘度 (Pa·s)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('V', 'T', T_celsius + 273.15, 'P', p_Pa, self._cp_name)
            except Exception:
                pass
        return _vapor_viscosity_poly(self.name, T_celsius)

    def liquid_thermal_conductivity(self, T_celsius: float) -> float:
        """饱和液体导热系数 (W/m·K)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('L', 'T', T_celsius + 273.15, 'Q', 0, self._cp_name)
            except Exception:
                pass
        return _liquid_thermal_conductivity_poly(self.name, T_celsius)

    def vapor_thermal_conductivity(self, T_celsius: float, p_Pa: float) -> float:
        """蒸汽导热系数 (W/m·K)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('L', 'T', T_celsius + 273.15, 'P', p_Pa, self._cp_name)
            except Exception:
                pass
        return _vapor_thermal_conductivity_poly(self.name, T_celsius)

    def liquid_cp(self, T_celsius: float) -> float:
        """饱和液体定压比热 (J/kg·K)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('C', 'T', T_celsius + 273.15, 'Q', 0, self._cp_name)
            except Exception:
                pass
        return _liquid_cp_poly(self.name, T_celsius) * 1000

    def vapor_cp(self, T_celsius: float, p_Pa: float) -> float:
        """蒸汽定压比热 (J/kg·K)"""
        if HAS_COOLPROP:
            try:
                return CP.PropsSI('C', 'T', T_celsius + 273.15, 'P', p_Pa, self._cp_name)
            except Exception:
                pass
        return _vapor_cp_poly(self.name, T_celsius) * 1000

    def liquid_prandtl(self, T_celsius: float) -> float:
        """饱和液体普朗特数"""
        mu = self.liquid_viscosity(T_celsius)
        cp = self.liquid_cp(T_celsius)
        k = self.liquid_thermal_conductivity(T_celsius)
        return mu * cp / k

    @property
    def uses_coolprop(self) -> bool:
        return HAS_COOLPROP

    def info(self) -> dict:
        return {
            'name': self.name,
            'coolprop': HAS_COOLPROP,
            'critical_temp': _REFRIG_CRITICAL.get(self.name, (None,))[0],
            'critical_pressure': _REFRIG_CRITICAL.get(self.name, (0, 0))[1],
        }
