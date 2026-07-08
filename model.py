"""
翅片管式蒸发器数字孪生模型
Finned-Tube Evaporator Digital Twin Model

理论基础:
  - ASHRAE Handbook of Fundamentals (2013) Ch.4 Heat Transfer
    · 翅片效率 (Fin Efficiency) - 圆形翅片 Bessel 函数解
    · 管外对流换热系数 (Gray & Webb 关联式)
  - ASHRAE Handbook of Systems & Equipment (2012) Ch.22 Cooling Coils
    · 湿工况冷却盘管计算 (Lewis 关系)
    · ε-NTU 方法
  - ASHRAE Standard 33-2000 强制循环空气冷却盘管测试方法
  - ASHRAE Fundamentals (2013) Ch.1 Psychrometrics (湿空气)
  - JB/T 7659.5-95 氟利昂制冷装置用翅片式换热器 (结构参数)
  - GB/T 23130-2008 房间空调器用翅片管式换热器 (空气侧换热量)
  - GB/T 47234-2026 数字孪生要求

管内沸腾换热关联式:
  - Shah (1976/1982) 两相流沸腾换热
  - Dittus-Boelter 单相强制对流
  - Kandlikar 两相流关联式 (可选)

模型层次 (GB/T 47234-2026):
  1. 几何模型层 (Geometric Model)
  2. 物理/机理模型层 (Physical/Mechanistic Model)
  3. 性能仿真与可视化层 (Performance Simulation & Visualization)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.special import i0, i1, k0, k1

from psychrometrics import (
    MoistAir, enthalpy, humidity_ratio, saturation_pressure_water,
    saturated_enthalpy, saturated_humidity_ratio, dew_point,
    specific_volume, density as air_density
)
from refrigerant import Refrigerant


# ============================================================
# 第一部分: 几何模型层 (Geometric Model)
# 符合 JB/T 7659.5-95 表5 结构参数
# ============================================================

@dataclass
class FinEvaporatorGeometry:
    """
    翅片管式蒸发器几何结构参数
    参数范围遵循 JB/T 7659.5-95 表5

    单位说明: 管径/翅片参数用 mm, 管长用 mm, 计算时转换为 m
    """
    # ---- 管子参数 (JB/T 7659.5-95 表5) ----
    tube_material: str = "紫铜"          # 管材
    tube_outer_diameter: float = 9.52    # 外径 D0 (mm), 范围 7~16
    tube_wall_thickness: float = 0.35    # 壁厚 δ0 (mm), 范围 0.30~1
    tube_pitch_transverse: float = 25.4  # 横向管距 Pt (mm), 范围 20~38
    tube_pitch_longitudinal: float = 22.0  # 纵向管距 Pl (mm) - 错排
    tube_arrangement: str = "错排"        # 排列方式: 错排/顺排

    # ---- 翅片参数 (JB/T 7659.5-95 表5) ----
    fin_material: str = "铝"             # 翅片材料
    fin_thickness: float = 0.15          # 片厚 δf (mm), 范围 0.10~0.30
    fin_pitch: float = 5.5               # 翅片节距 (mm), 范围 1.3~12.0

    # ---- 宏观尺寸 ----
    num_rows: int = 4                    # 排数 n, 范围 1~6
    num_tubes_per_row: int = 12          # 每排管数
    tube_length: float = 800.0           # 单根管长 (mm)
    num_circuits: int = 6                # 分液路数 (并联回路)

    # ---- 材料导热系数 (W/m·K) ----
    k_tube: float = 398.0    # 紫铜
    k_fin: float = 236.0     # 铝
    # 若为铝管: k_tube=236; 铜翅片: k_fin=398

    def __post_init__(self):
        """参数校验 - JB/T 7659.5-95 表5 范围"""
        assert 7 <= self.tube_outer_diameter <= 16, "管外径需在 7~16mm 范围内"
        assert 0.30 <= self.tube_wall_thickness <= 1.0, "壁厚需在 0.30~1.0mm 范围内"
        assert 20 <= self.tube_pitch_transverse <= 38, "横向管距需在 20~38mm 范围内"
        assert 0.10 <= self.fin_thickness <= 0.30, "片厚需在 0.10~0.30mm 范围内"
        assert 1.3 <= self.fin_pitch <= 12.0, "翅片节距需在 1.3~12.0mm 范围内"
        assert 1 <= self.num_rows <= 12, "排数需在 1~12 范围内"
        assert self.num_tubes_per_row >= 2, "每排管数至少 2"

    # ---- 几何属性计算 ----

    @property
    def d_o(self) -> float:
        """管外径 (m)"""
        return self.tube_outer_diameter / 1000

    @property
    def d_i(self) -> float:
        """管内径 (m)"""
        return (self.tube_outer_diameter - 2 * self.tube_wall_thickness) / 1000

    @property
    def p_t(self) -> float:
        """横向管距 (m)"""
        return self.tube_pitch_transverse / 1000

    @property
    def p_l(self) -> float:
        """纵向管距 (m)"""
        return self.tube_pitch_longitudinal / 1000

    @property
    def f_pitch(self) -> float:
        """翅片节距 (m)"""
        return self.fin_pitch / 1000

    @property
    def f_thickness(self) -> float:
        """翅片厚度 (m)"""
        return self.fin_thickness / 1000

    @property
    def L_tube(self) -> float:
        """单根管长 (m)"""
        return self.tube_length / 1000

    @property
    def total_tubes(self) -> int:
        """总管数"""
        return self.num_rows * self.num_tubes_per_row

    @property
    def tubes_per_circuit(self) -> float:
        """每回路管数"""
        return self.total_tubes / self.num_circuits

    @property
    def fin_height(self) -> float:
        """翅片高度 (沿空气流向) (m) = 纵向排数 × 纵向管距"""
        return self.num_rows * self.p_l

    @property
    def fin_width(self) -> float:
        """翅片宽度 (m) = 横向管数 × 横向管距"""
        return self.num_tubes_per_row * self.p_t

    @property
    def face_area(self) -> float:
        """迎风面积 (m²)"""
        return self.fin_width * self.L_tube

    @property
    def num_fins(self) -> int:
        """翅片数量"""
        return int(self.L_tube / self.f_pitch)

    @property
    def area_tube_external(self) -> float:
        """
        管外表面积 (m²) - 翅片间裸管部分
        每根管: π * D_o * (L - N_fin * δf)
        """
        bare_length_per_tube = self.L_tube - self.num_fins * self.f_thickness
        return np.pi * self.d_o * bare_length_per_tube * self.total_tubes

    @property
    def area_fin(self) -> float:
        """
        翅片总面积 (m²) - 双面换热
        单片翅片面积 = 2 × (翅片宽 × 翅片高 - 管孔面积)
        """
        single_fin_area = 2 * (self.fin_width * self.fin_height)
        # 减去管孔面积 (翅片上被管子占据的孔)
        tube_hole_area = self.total_tubes * np.pi * (self.d_o**2) / 4
        single_fin_net = single_fin_area - 2 * tube_hole_area  # 双面
        return single_fin_net * self.num_fins

    @property
    def area_external_total(self) -> float:
        """空气侧总换热面积 (m²) - GB/T 23130-2008 3.8"""
        return self.area_tube_external + self.area_fin

    @property
    def area_internal_total(self) -> float:
        """管内总换热面积 (m²)"""
        return np.pi * self.d_i * self.L_tube * self.total_tubes

    @property
    def volume_internal(self) -> float:
        """管内容积 (m³) = π/4 × d_i² × L × N_total"""
        return np.pi * (self.d_i ** 2) / 4 * self.L_tube * self.total_tubes

    @property
    def fin_area_ratio(self) -> float:
        """翅片面积占比 = A_fin / A_total"""
        return self.area_fin / self.area_external_total

    @property
    def min_flow_area_ratio(self) -> float:
        """
        最小流通面积 / 迎风面积 (错排管束)
        用于计算最大空气流速
        """
        # 翅片间最小截面
        s_f = self.f_pitch - self.f_thickness  # 翅片净间距
        # 横向最小截面
        sigma = 1 - self.d_o / self.p_t
        return sigma * (s_f / self.f_pitch)

    def summary(self) -> dict:
        """几何参数摘要"""
        return {
            '管外径 (mm)': self.tube_outer_diameter,
            '管内径 (mm)': self.d_i * 1000,
            '管壁厚 (mm)': self.tube_wall_thickness,
            '横向管距 (mm)': self.tube_pitch_transverse,
            '纵向管距 (mm)': self.tube_pitch_longitudinal,
            '排列方式': self.tube_arrangement,
            '翅片材料': self.fin_material,
            '翅片厚 (mm)': self.fin_thickness,
            '翅片节距 (mm)': self.fin_pitch,
            '排数': self.num_rows,
            '每排管数': self.num_tubes_per_row,
            '总管数': self.total_tubes,
            '分液路数': self.num_circuits,
            '管长 (mm)': self.tube_length,
            '迎风面积 (m²)': round(self.face_area, 4),
            '管外表面积 (m²)': round(self.area_tube_external, 3),
            '翅片面积 (m²)': round(self.area_fin, 3),
            '空气侧总面积 (m²)': round(self.area_external_total, 3),
            '管内总面积 (m²)': round(self.area_internal_total, 3),
            '管内容积 (L)': round(self.volume_internal * 1000, 3),
            '翅片面积占比': f"{self.fin_area_ratio*100:.1f}%",
        }


# ============================================================
# 第二部分: 物理/机理模型层 (Physical/Mechanistic Model)
# ============================================================

class FinEfficiency:
    """
    翅片效率计算 - ASHRAE Fundamentals 2013 Ch.4

    圆形(环形)翅片效率使用 Bessel 函数精确解:
      η_f = (2*r1 / (m*(r2c²-r1²))) *
            (I1(m*r2c)*K1(m*r1) - K1(m*r2c)*I1(m*r1)) /
            (I0(m*r2c)*K1(m*r1) + K0(m*r2c)*I1(m*r1))

    其中:
      m = sqrt(2*h_o / (k_fin * δf))   翅片参数
      r1 = D_o/2                       翅片根部半径 (管外半径)
      r2 = P_t/2 (近似)                翅片尖端半径
      r2c = r2 + δf/2                  修正翅片尖端半径
    """

    @staticmethod
    def annular_fin_efficiency(h_o: float, k_fin: float, delta_f: float,
                               r1: float, r2: float) -> float:
        """
        环形翅片效率 (Bessel 函数精确解)

        Parameters
        ----------
        h_o : float - 管外对流换热系数 (W/m²·K)
        k_fin : float - 翅片材料导热系数 (W/m·K)
        delta_f : float - 翅片厚度 (m)
        r1 : float - 翅片根部半径 (管外半径) (m)
        r2 : float - 翅片尖端半径 (m)
        """
        if h_o <= 0 or k_fin <= 0:
            return 1.0
        m = np.sqrt(2 * h_o / (k_fin * delta_f))
        r2c = r2 + delta_f / 2  # 修正翅尖半径
        mr1 = m * r1
        mr2c = m * r2c

        if mr1 < 1e-6:
            return 1.0

        # Bessel 函数
        I0_2, I1_2 = float(i0(mr2c)), float(i1(mr2c))
        I1_1 = float(i1(mr1))
        K0_2, K1_2 = float(k0(mr2c)), float(k1(mr2c))
        K1_1 = float(k1(mr1))

        numerator = I1_2 * K1_1 - K1_2 * I1_1
        denominator = I0_2 * K1_1 + K0_2 * I1_1

        if abs(denominator) < 1e-15:
            return 0.95

        eta = (2 * r1 / (m * (r2c**2 - r1**2))) * (numerator / denominator)
        return max(min(float(eta), 1.0), 0.0)

    @staticmethod
    def straight_fin_efficiency(h_o: float, k_fin: float, delta_f: float,
                                L_fin: float) -> float:
        """
        直翅片效率 (近似公式)
        η_f = tanh(m * Lc) / (m * Lc)
        其中 Lc = L + δf/2, m = sqrt(2h_o/(k_fin*δf))
        """
        m = np.sqrt(2 * h_o / (k_fin * delta_f))
        Lc = L_fin + delta_f / 2
        mLc = m * Lc
        if mLc < 1e-6:
            return 1.0
        return float(np.tanh(mLc) / mLc)

    @staticmethod
    def overall_surface_efficiency(eta_f: float, fin_area_ratio: float) -> float:
        """
        整体表面效率 η_o = 1 - (A_f/A_total) * (1 - η_f)
        ASHRAE Fundamentals 2013 Ch.4, Eq.10
        """
        return 1.0 - fin_area_ratio * (1.0 - eta_f)


class AirSideCorrelation:
    """
    管外空气侧对流换热系数关联式

    1. Gray & Webb (1986) - 错排翅片管束 (ASHRAE 推荐)
       j = 0.14 * Re_D^(-0.328) * (Pt/Pl)^(-0.502) * (Pt/Do)^0.031
       St = j / Pr^(2/3)
       h_o = St * G_c * c_p

    2. McQuiston (1978) - 另一常用关联式
       j = 0.0014 + 0.2618*Re_D^(-0.4) * (A/A_t)^(-0.15)

    3. 简化关联式 - 适用于快速估算
    """

    @staticmethod
    def gray_webb(G_c: float, D_o: float, P_t: float, P_l: float,
                  mu_air: float, rho_air: float, cp_air: float,
                  k_air: float, Pr_air: float) -> Tuple[float, float]:
        """
        Gray & Webb 关联式 (错排翅片管束)

        Parameters
        ----------
        G_c : float - 最小截面质量流速 (kg/m²·s)
        D_o : float - 管外径 (m)
        P_t, P_l : float - 横向/纵向管距 (m)
        mu_air : float - 空气动力粘度 (Pa·s)
        rho_air : float - 空气密度 (kg/m³)
        cp_air : float - 空气比热 (J/kg·K)
        k_air : float - 空气导热系数 (W/m·K)
        Pr_air : float - 空气普朗特数

        Returns
        -------
        (h_o, j_factor) : 对流换热系数 (W/m²·K), Colburn j因子
        """
        Re_D = G_c * D_o / mu_air

        # Gray & Webb 关联式 (4排管束基准)
        j4 = 0.14 * Re_D**(-0.328) * (P_t / P_l)**(-0.502) * (P_t / D_o)**0.031

        # 排数修正 (ASHRAE: N排管 j/j4 修正)
        # 对于 >=4 排, 修正因子 ≈ 1.0
        # 此处取 j4 作为基准

        St = j4 / Pr_air**(2/3)
        h_o = St * G_c * cp_air
        return h_o, j4

    @staticmethod
    def mcquiston(G_c: float, D_o: float, A_ratio: float,
                  mu_air: float, cp_air: float, Pr_air: float,
                  num_rows: int) -> float:
        """
        McQuiston (1978) 关联式

        Parameters
        ----------
        A_ratio : float - A_total/A_tube_external (总面/管面)
        num_rows : int - 排数
        """
        Re_D = G_c * D_o / mu_air
        j = 0.0014 + 0.2618 * Re_D**(-0.4) * A_ratio**(-0.15)
        # 排数修正
        j *= (1.0 if num_rows >= 4 else 0.5 + 0.5 / num_rows)
        St = j / Pr_air**(2/3)
        h_o = St * G_c * cp_air
        return h_o

    @staticmethod
    def air_prandtl(cp: float, mu: float, k: float) -> float:
        """空气普朗特数"""
        return cp * mu / k


class TubeSideBoiling:
    """
    管内沸腾换热关联式

    1. Shah (1976/1982) - 两相流沸腾换热
       h_tp = h_l * F_shah
       F_shah = 1 + 3.8 / Z^0.95  (当 Co > 0, 即对流沸腾为主)
       Z = ((1-x)/x)^0.8 * (ρ_v/ρ_l)^0.5   对流数

    2. Kandlikar (1990) - 两相流关联式
       h_tp = h_l * max(NBD, CBD)
       含核态沸腾和对流沸腾两项

    3. Gungor-Winterton (1986)
       h_tp = E*h_l + S*h_nb

    4. Dittus-Boelter (单相液体)
       Nu = 0.023 * Re^0.8 * Pr^0.4
    """

    @staticmethod
    def dittus_boelter(Re: float, Pr: float, k: float, D: float) -> float:
        """
        Dittus-Boelter 单相对流换热系数
        Nu = 0.023 * Re^0.8 * Pr^0.4  (加热流体)
        """
        Nu = 0.023 * Re**0.8 * Pr**0.4
        return Nu * k / D

    @staticmethod
    def shah(mass_flux: float, x: float, D_i: float,
             rho_l: float, rho_v: float, mu_l: float, k_l: float,
             cp_l: float, Pr_l: float) -> float:
        """
        Shah (1976/1982) 两相流沸腾换热系数

        Parameters
        ----------
        mass_flux : float - 质量流速 (kg/m²·s)
        x : float - 干度 (0~1)
        D_i : float - 管内径 (m)
        rho_l, rho_v : float - 液相/气相密度 (kg/m³)
        mu_l : float - 液相粘度 (Pa·s)
        k_l : float - 液相导热系数 (W/m·K)
        cp_l : float - 液相比热 (J/kg·K)
        Pr_l : float - 液相普朗特数
        """
        if x <= 0.001:
            x = 0.001
        if x >= 0.999:
            x = 0.999

        # 液相单独流动的 Re
        Re_l = mass_flux * (1 - x) * D_i / mu_l
        h_l = TubeSideBoiling.dittus_boelter(Re_l, Pr_l, k_l, D_i)

        # 对流数 Z
        Z = ((1 - x) / x)**0.8 * (rho_v / rho_l)**0.5

        # Shah 修正因子
        if Z <= 10:
            F_shah = 1.0 + 3.8 / Z**0.95
        else:
            F_shah = 1.0 + 3.8 / Z**0.95

        # 核态沸腾贡献 (Shah 1982 修正)
        Bo = 0.0  # 简化: 不考虑热流密度对核态沸腾的影响

        h_tp = h_l * F_shah
        return h_tp

    @staticmethod
    def kandlikar(mass_flux: float, x: float, D_i: float,
                  rho_l: float, rho_v: float, mu_l: float, k_l: float,
                  cp_l: float, Pr_l: float, h_fg: float, q_flux: float = 5000) -> float:
        """
        Kandlikar (1990) 两相流沸腾关联式

        Parameters
        ----------
        q_flux : float - 热流密度 (W/m²)
        h_fg : float - 汽化潜热 (J/kg)
        """
        if x <= 0.001:
            x = 0.001
        if x >= 0.999:
            x = 0.999

        Re_l = mass_flux * (1 - x) * D_i / mu_l
        h_l = TubeSideBoiling.dittus_boelter(Re_l, Pr_l, k_l, D_i)

        # Boiling number
        Bo = q_flux / (mass_flux * h_fg)

        # Convection number
        Co = ((1 - x) / x)**0.8 * (rho_l / rho_v)**0.5

        # Kandlikar 系数 (R134a 典型值)
        F_fl = 1.63  # 制冷剂相关系数

        # NBD (核态沸腾主导)
        h_nbd = (0.6683 * Co**(-0.2) * (1 - x)**0.8 + 1058.0 * Bo**0.7 * F_fl * (1 - x)**0.8) * h_l

        # CBD (对流沸腾主导)
        h_cbd = (1.136 * Co**(-0.9) * (1 - x)**0.8 + 667.2 * Bo**0.7 * F_fl * (1 - x)**0.8) * h_l

        return max(h_nbd, h_cbd)

    @staticmethod
    def average_boiling_h(mass_flux: float, D_i: float,
                          rho_l: float, rho_v: float, mu_l: float, k_l: float,
                          cp_l: float, Pr_l: float, h_fg: float,
                          x_in: float = 0.15, x_out: float = 0.95,
                          method: str = 'shah') -> float:
        """
        沿管长平均沸腾换热系数 (积分平均)

        在 x_in ~ x_out 范围内对干度积分求平均 h
        """
        n_points = 20
        qualities = np.linspace(x_in, x_out, n_points)
        h_values = []

        for x in qualities:
            if method == 'shah':
                h = TubeSideBoiling.shah(mass_flux, x, D_i, rho_l, rho_v,
                                         mu_l, k_l, cp_l, Pr_l)
            elif method == 'kandlikar':
                h = TubeSideBoiling.kandlikar(mass_flux, x, D_i, rho_l, rho_v,
                                              mu_l, k_l, cp_l, Pr_l, h_fg)
            else:
                h = TubeSideBoiling.shah(mass_flux, x, D_i, rho_l, rho_v,
                                         mu_l, k_l, cp_l, Pr_l)
            h_values.append(h)

        # 沿干度积分平均
        from scipy.integrate import trapezoid
        h_avg = trapezoid(h_values, qualities) / (x_out - x_in)
        return float(h_avg)


class AirProperties:
    """干空气热物性 (ASHRAE Fundamentals Ch.1)"""

    @staticmethod
    def cp(T_celsius: float = 20) -> float:
        """定压比热 (J/kg·K)"""
        return 1006.0  # 近似常数

    @staticmethod
    def viscosity(T_celsius: float = 20) -> float:
        """动力粘度 (Pa·s) - Sutherland 公式"""
        T = T_celsius + 273.15
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 111  # Sutherland 常数
        return mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)

    @staticmethod
    def thermal_conductivity(T_celsius: float = 20) -> float:
        """导热系数 (W/m·K)"""
        T = T_celsius + 273.15
        return 0.02418 + 7.7e-5 * T_celsius

    @staticmethod
    def prandtl(T_celsius: float = 20) -> float:
        """普朗特数"""
        cp = AirProperties.cp()
        mu = AirProperties.viscosity(T_celsius)
        k = AirProperties.thermal_conductivity(T_celsius)
        return cp * mu / k


# ============================================================
# 第三部分: 蒸发器数字孪生主模型
# ============================================================

class FinEvaporatorDigitalTwin:
    """
    翅片管式蒸发器数字孪生模型

    符合 GB/T 47234-2026 数字孪生要求:
      - 几何模型 (5.3.7.1)
      - 物理/机理模型 (5.3.7.2)
      - 性能仿真与可视化 (5.3.7.3)

    计算方法: 分段(逐排)计算 + ε-NTU 混合方法
    空气侧: 湿工况 (Lewis 关系) / 干工况自动判别
    制冷剂侧: Shah 两相沸腾 + 过热段单相对流
    """

    def __init__(self, geometry: FinEvaporatorGeometry,
                 refrigerant: Refrigerant = None):
        self.geometry = geometry
        self.refrigerant = refrigerant or Refrigerant('R410A')
        self.version = "2.0.0"
        self.standards = [
            "ASHRAE Fundamentals 2013 Ch.1/Ch.4",
            "ASHRAE Systems & Equipment 2012 Ch.22",
            "ASHRAE Standard 33-2000",
            "JB/T 7659.5-95",
            "GB/T 23130-2008",
            "GB/T 47234-2026",
        ]

    def compute_fin_efficiency(self, h_o: float) -> Tuple[float, float]:
        """
        计算翅片效率和整体表面效率
        ASHRAE Fundamentals 2013 Ch.4

        板式翅片采用等效圆形翅片法 (Sector Method 近似):
          1. 计算面积等效半径 r2_eq = sqrt(r1² + Pt*Pl/π)
          2. 用 Harper-Brown 修正直翅片公式: η = tanh(m*Lc)/(m*Lc)
          3. 同时计算环形翅片 Bessel 精确解作为参考

        Returns
        -------
        (eta_f, eta_o) : 翅片效率, 整体表面效率
        """
        g = self.geometry
        r1 = g.d_o / 2           # 翅片根部半径 = 管外半径

        # 面积等效半径 (板式翅片等效为环形翅片)
        # 每管分摊的翅片面积 = Pt * Pl (单面), 等效为 π(r2²-r1²)
        r2_eq = np.sqrt(r1**2 + g.p_t * g.p_l / np.pi)

        # Harper-Brown 修正直翅片公式 (板式翅片标准方法)
        L_fin = r2_eq - r1       # 等效翅片高度
        eta_f = FinEfficiency.straight_fin_efficiency(
            h_o=h_o,
            k_fin=g.k_fin,
            delta_f=g.f_thickness,
            L_fin=L_fin
        )

        eta_o = FinEfficiency.overall_surface_efficiency(
            eta_f, g.fin_area_ratio
        )
        return eta_f, eta_o

    def compute_air_side_h(self, air_state: MoistAir,
                           face_velocity: float) -> Tuple[float, float, float]:
        """
        计算管外空气侧对流换热系数
        Gray & Webb 关联式 (ASHRAE 推荐)

        Parameters
        ----------
        air_state : MoistAir - 空气状态
        face_velocity : float - 迎面风速 (m/s)

        Returns
        -------
        (h_o, G_c, Re_D) : 对流换热系数, 最大质量流速, 雷诺数
        """
        g = self.geometry
        rho = air_state.rho

        # 最小截面流速 (错排管束)
        sigma = g.min_flow_area_ratio
        V_max = face_velocity / sigma
        G_c = rho * V_max  # 最大质量流速

        # 空气物性 (平均膜温度近似)
        T_mean = air_state.T_db
        mu = AirProperties.viscosity(T_mean)
        k = AirProperties.thermal_conductivity(T_mean)
        cp = AirProperties.cp()
        Pr = AirProperties.prandtl(T_mean)

        h_o, j = AirSideCorrelation.gray_webb(
            G_c, g.d_o, g.p_t, g.p_l,
            mu, rho, cp, k, Pr
        )
        Re_D = G_c * g.d_o / mu
        return h_o, G_c, Re_D

    def compute_tube_side_h(self, T_evap: float, mass_flux: float,
                            x_in: float = 0.15, x_out: float = 0.95,
                            method: str = 'shah') -> Tuple[float, float]:
        """
        计算管内沸腾换热系数 (沿管长平均)
        Shah / Kandlikar 关联式

        Parameters
        ----------
        T_evap : float - 蒸发温度 (°C)
        mass_flux : float - 管内质量流速 (kg/m²·s)
        x_in, x_out : float - 进出口干度
        method : str - 'shah' 或 'kandlikar'

        Returns
        -------
        (h_i_avg, h_l) : 平均沸腾换热系数, 液相单相换热系数
        """
        ref = self.refrigerant
        g = self.geometry

        rho_l = ref.liquid_density(T_evap)
        p_evap = ref.saturation_pressure(T_evap)
        rho_v = ref.sat_vapor_density(T_evap)
        mu_l = ref.liquid_viscosity(T_evap)
        k_l = ref.liquid_thermal_conductivity(T_evap)
        cp_l = ref.liquid_cp(T_evap)
        Pr_l = ref.liquid_prandtl(T_evap)
        h_fg = ref.latent_heat(T_evap)

        h_avg = TubeSideBoiling.average_boiling_h(
            mass_flux, g.d_i, rho_l, rho_v, mu_l, k_l,
            cp_l, Pr_l, h_fg, x_in, x_out, method
        )

        # 液相单相 h_l (参考)
        Re_l = mass_flux * g.d_i / mu_l
        h_l = TubeSideBoiling.dittus_boelter(Re_l, Pr_l, k_l, g.d_i)

        return h_avg, h_l

    def compute_overall_UA(self, h_o: float, h_i: float,
                           eta_o: float, wet: bool = False,
                           cp_air: float = 1006.0) -> float:
        """
        计算总传热系数 UA (基于管外面积)

        热阻网络:
          1/UA = 1/(η_o * h_o * A_o) + ln(Do/Di)/(2π*k_tube*L*N) + 1/(h_i * A_i)

        湿工况: 空气侧使用焓差-温差转换 (Lewis 关系)
          h_o_wet = h_o / (cp_air * Le) ≈ h_o / cp_air (Le≈1)
          有效传热基于焓差

        Returns
        -------
        UA : float - 总传热系数 (W/K), 基于管外面积
        """
        g = self.geometry

        # 管壁热阻
        R_wall = np.log(g.d_o / g.d_i) / (2 * np.pi * g.k_tube * g.L_tube * g.total_tubes)

        # 空气侧热阻 (含翅片效率)
        R_air = 1.0 / (eta_o * h_o * g.area_external_total)

        # 制冷剂侧热阻
        R_ref = 1.0 / (h_i * g.area_internal_total)

        # 总热阻
        R_total = R_air + R_wall + R_ref

        UA = 1.0 / R_total
        return UA

    def compute_wet_UA(self, h_o: float, h_i: float, eta_o: float,
                       cp_air: float = 1006.0) -> float:
        """
        湿工况总传热系数 (基于焓差)
        ASHRAE Systems & Equipment 2012 Ch.22

        湿工况下空气侧换热基于焓差:
          Q = h_o_wet * A_o * (h_air - h_surface)
        其中 h_o_wet = h_o / (cp_air * Le), Le ≈ 1

        有效 UA_enthalpy = 1 / (1/(η_o*h_o*A_o/cp) + R_wall + R_ref)
        传热量 Q = UA_enthalpy * (h_air - h_surf)

        Returns
        -------
        UA_enh : float - 基于焓差的总传热系数 (W/(kJ/kg) = kg/s)
        """
        g = self.geometry
        Le = 1.0  # Lewis 数 (空气-水蒸气系统 ≈ 1)

        R_wall = np.log(g.d_o / g.d_i) / (2 * np.pi * g.k_tube * g.L_tube * g.total_tubes)
        R_air_enthalpy = cp_air / (eta_o * h_o * g.area_external_total * Le)
        R_ref = 1.0 / (h_i * g.area_internal_total)

        # 统一单位: 将焓差 (kJ/kg) 转为等效温差
        UA_enh = 1.0 / (R_air_enthalpy + R_wall + R_ref)
        return UA_enh

    # ================================================================
    # 制冷剂管内压降仿真模型
    # ASHRAE Fundamentals 2013 Ch.5 (Two-Phase Flow)
    # ================================================================

    @staticmethod
    def _blasius_friction_factor(Re: float) -> float:
        """Blasius公式: 湍流摩擦因子 f = 0.079*Re^(-0.25)"""
        if Re < 1:
            return 0.079
        if Re < 2300:
            # 层流: f = 16/Re
            return 16.0 / max(Re, 1.0)
        return 0.079 * Re ** (-0.25)

    def compute_pressure_drop_tp(self, T_evap, p_local, mass_flux, x_avg,
                                 dx, d_i, L_seg, rho_l, rho_v,
                                 mu_l, mu_v, include_bend=True):
        """
        两相流压降计算 - 均匀流模型 + 加速压降
        ASHRAE Fundamentals 2013 Ch.5

        均匀流模型对高密度比制冷剂(R410A, R32等)精度优于Lockhart-Martinelli

        组成:
          1. 摩擦压降: ΔP_f = 2*f_m*G²*L/(ρ_m*d_i)  (均匀流)
          2. 加速压降: ΔP_a = G²*(1/ρ_v-1/ρ_l)*Δx  (蒸发动量变化)
          3. 局部压降(U型弯头): ΔP_b = K*G²/(2*ρ_m)  (均匀流)

        Parameters
        ----------
        x_avg : float - 段平均干度
        dx : float - 段干度变化(出口-入口, 沿制冷剂流向)
        include_bend : bool - 是否计入U型弯头局部压降

        Returns
        -------
        (delta_P_total, delta_P_friction, delta_P_accel, delta_P_bend) : Pa
        """
        x = max(min(x_avg, 0.99), 0.01)

        # ---- 均匀流物性 ----
        v_m = x / rho_v + (1 - x) / rho_l       # 比容 (m³/kg)
        rho_m = 1.0 / max(v_m, 1e-6)             # 均匀流密度
        mu_m = x * mu_v + (1 - x) * mu_l         # 均匀流粘度

        # ---- 1. 摩擦压降 (均匀流模型) ----
        Re_m = mass_flux * d_i / max(mu_m, 1e-8)
        f_m = self._blasius_friction_factor(Re_m)
        dPdz_tp = 2 * f_m * mass_flux**2 / (rho_m * d_i)
        delta_P_friction = dPdz_tp * L_seg

        # ---- 2. 加速压降 (均匀流模型) ----
        # ΔP_acc = G² * (1/ρ_v - 1/ρ_l) * Δx
        delta_P_accel = mass_flux**2 * (1.0 / rho_v - 1.0 / rho_l) * dx

        # ---- 3. 局部压降 (U型弯头, 均匀流模型) ----
        delta_P_bend = 0.0
        if include_bend:
            K_single = 0.9  # 180°弯头
            delta_P_bend = K_single * mass_flux**2 / (2.0 * rho_m)

        delta_P_total = delta_P_friction + delta_P_accel + delta_P_bend
        return (delta_P_total, delta_P_friction, delta_P_accel, delta_P_bend)

    def compute_pressure_drop_sh(self, mass_flux, d_i, L_seg,
                                 rho_v, mu_v, include_bend=True):
        """
        过热段单相(蒸汽)压降 - Darcy-Weisbach
        ΔP = 4*f*G²*L / (2*ρ_v*d_i) + 弯头损失

        Returns
        -------
        (delta_P_total, delta_P_friction, delta_P_bend) : Pa
        """
        Re_v = mass_flux * d_i / max(mu_v, 1e-8)
        f_v = self._blasius_friction_factor(Re_v)

        delta_P_friction = 4 * f_v * mass_flux**2 * L_seg / (2 * rho_v * d_i)

        delta_P_bend = 0.0
        if include_bend:
            K_single = 0.9  # 180°弯头
            delta_P_bend = K_single * mass_flux**2 / (2 * rho_v)

        return (delta_P_friction + delta_P_bend, delta_P_friction, delta_P_bend)

    def solve(self, T_evap: float, x_in: float,
              T_air_in: float, RH_air_in: float,
              face_velocity: float, mass_flow: float,
              p_atm: float = 101325.0,
              num_segments: int = 10,
              boiling_method: str = 'shah') -> dict:
        """
        蒸发器性能求解 - 能量守恒迭代法

        核心改进: 过热度由能量守恒计算得出 (不再是输入参数)
          Q_air = Q_refrigerant  (严格能量守恒)
          superheat = f(Q, mass_flow, x_in)  (输出而非输入)

        迭代逻辑:
          1. 给定进口干度 x_in 和质量流量 mass_flow
          2. 猜测过热度 superheat → 计算空气侧换热量 Q_air
          3. 由能量守恒计算新过热度: superheat_new = (Q_air - Q_tp_max) / (m_dot * cp_v)
          4. 收敛判断: |superheat_new - superheat| < 0.005 °C

        Parameters
        ----------
        T_evap : float - 蒸发温度 (°C)
        x_in : float - 制冷剂进口干度 (0~1), 节流后状态
        T_air_in : float - 进风干球温度 (°C)
        RH_air_in : float - 进风相对湿度 (0~1)
        face_velocity : float - 迎面风速 (m/s)
        mass_flow : float - 制冷剂质量流量 (kg/s)
        p_atm : float - 大气压 (Pa)
        num_segments : int - 分段数
        boiling_method : str - 沸腾换热关联式

        Returns
        -------
        dict : 完整计算结果 (含计算出的过热度 superheat)
        """
        g = self.geometry
        ref = self.refrigerant

        # ---- 进口空气状态 ----
        air_in = MoistAir.from_db_rh(T_air_in, RH_air_in, p_atm)
        rho_air = air_in.rho

        # ---- 空气流量 ----
        V_air = face_velocity * g.face_area  # 体积流量 (m³/s)
        m_air = rho_air * V_air               # 质量流量 (kg/s)

        # ---- 管内质量流速 ----
        A_tube = np.pi * g.d_i**2 / 4  # 单管截面积
        mass_flux = mass_flow / (g.num_circuits * A_tube)  # kg/m²·s

        # ---- 制冷剂物性 ----
        p_evap = ref.saturation_pressure(T_evap)
        rho_l = ref.liquid_density(T_evap)
        rho_v = ref.sat_vapor_density(T_evap)
        h_l = ref.sat_liquid_enthalpy(T_evap)       # J/kg 饱和液体焓
        h_v = ref.sat_vapor_enthalpy(T_evap)         # J/kg 饱和蒸汽焓
        h_fg = ref.latent_heat(T_evap)                # J/kg 汽化潜热
        h_ref_in = h_l + x_in * h_fg                  # J/kg 进口焓 (干度 x_in)

        # 完全蒸发所需热量 (两相段最大吸热量)
        Q_tp_max = mass_flow * (1.0 - x_in) * h_fg   # W

        # ---- 过热段制冷剂物性 ----
        mu_v = ref.vapor_viscosity(T_evap, p_evap)
        k_v = ref.vapor_thermal_conductivity(T_evap, p_evap)
        cp_v = ref.vapor_cp(T_evap, p_evap)

        # ---- 液相物性 (压降计算用) ----
        mu_l = ref.liquid_viscosity(T_evap)

        # ---- 管外换热系数 (基于进口空气状态) ----
        h_o, G_c, Re_D = self.compute_air_side_h(air_in, face_velocity)

        # ---- 翅片效率 ----
        eta_f, eta_o = self.compute_fin_efficiency(h_o)

        # ---- 管内沸腾换热系数 (两相段, 沿干度积分平均) ----
        h_i_tp, h_l_ref = self.compute_tube_side_h(
            T_evap, mass_flux, x_in=x_in, x_out=0.95, method=boiling_method
        )

        # ---- 过热段换热系数 (单相蒸汽强制对流) ----
        Re_v = mass_flux * g.d_i / mu_v
        Pr_v = cp_v * mu_v / k_v
        h_i_sh = TubeSideBoiling.dittus_boelter(Re_v, Pr_v, k_v, g.d_i)

        cp_air = AirProperties.cp()

        # ---- 分段几何 ----
        segment_area_o = g.area_external_total / num_segments
        segment_area_i = g.area_internal_total / num_segments
        R_wall_seg = np.log(g.d_o / g.d_i) / (
            2 * np.pi * g.k_tube * g.L_tube * g.total_tubes / num_segments)

        # ================================================================
        # 能量守恒迭代: 求解过热度 superheat
        # ================================================================
        # 逆流换热器: 制冷剂与空气反向流动
        #   - 过热段在空气进口侧 (段0): 热制冷剂出口遇热空气进口
        #   - 两相段在空气出口侧 (末段): 冷制冷剂进口遇冷空气出口
        #   → 最大化对数平均温差 (LMTD)
        #
        # 核心改进: 壁面温度 T_s 通过热阻网络与制冷剂温度 T_ref 耦合
        #
        # 湿工况 (Threlkeld 焓法, ASHRAE Ch.22):
        #   空气侧: Q = G_enthalpy * (h_air - h_s(T_s))
        #   制冷剂侧: Q = (T_s - T_ref) / R_ref_wall
        #   联立二分法求解 T_s, 使 Q_air = Q_ref
        #
        # 干工况 (温差驱动):
        #   Q = (T_air - T_ref) / (R_air + R_ref_wall)
        #   T_s = T_ref + Q * R_ref_wall
        #
        # 物理约束 (热力学第二定律):
        #   逆流换热: 制冷剂出口温度 ≤ 空气进口温度
        #   即 superheat ≤ T_air_in - T_evap
        #
        # 物理逻辑:
        #   - mass_flow 大 → 过热度小 → T_ref ≈ T_evap → T_s 低 → Δh 大 → Q 大
        #   - mass_flow 小 → 过热度大 → T_ref 升高 → T_s 升高
        #     → h_s(T_s) 升高 → 焓差减小 → Q 降低
        #   - mass_flow 极小 → T_ref_out → T_air_in (制冷剂侧瓶颈, Q由制冷剂侧决定)
        # ================================================================

        # ================================================================
        # 辅助函数: 给定过热度, 计算空气侧换热量和分段详情
        # ================================================================
        def _evaluate(sh: float):
            """
            给定过热度 sh, 计算各段换热量和空气状态变化
            返回: Q_total, Q_sens, Q_lat, cond, T_s_avg, segs, states, T_air_out, W_out

            逆流换热 (过热度SH > 0):
              制冷剂从末段(段n-1)流向首段(段0)
              段0: 制冷剂出口 T_ref = T_evap + SH
              过热段/两相段分界处的排: 制冷剂刚蒸干 T_ref = T_evap
              两相段: T_ref = T_evap
              沿空气流向(段0→n-1), 制冷剂温度从 T_evap+SH 递减到 T_evap
            """
            T_air = T_air_in
            W_air = air_in.W
            Q_total = 0.0
            Q_sens_total = 0.0
            Q_lat_total = 0.0
            cond_total = 0.0
            T_s_sum = 0.0
            segs = []
            states = [air_in]

            # ================================================================
            # 确定过热段与两相段分界点
            # ================================================================
            # 物理原则: 过热段长度 = SH对应的"焓差"占两相段焓差的比例
            # 假设每排吸热能力相同(简化), 则:
            #   f_tp_area = m_dot * h_fg / (m_dot * h_fg + m_dot * cp_v * SH) = h_fg / (h_fg + cp_v*SH)
            #   n_tp = num_segments * f_tp_area
            if sh < 0.01:
                n_tp = num_segments
                n_sh = 0
            else:
                # 基于焓差的比例分配
                h_fg_local = h_fg if h_fg > 0 else 1.0
                f_sh_area = (cp_v * sh) / (h_fg_local + cp_v * sh)
                # SH 沿程分布: 让SH较大时过热段跨越更多排
                # 经验公式: f_sh_area 与 SH 大致成正比
                f_sh_area = max(0.20, min(0.6, sh / (sh + 15.0)))
                n_sh = max(int(round(num_segments * f_sh_area)), 1)
                n_sh = min(n_sh, num_segments - 1)
                n_tp = num_segments - n_sh
                if n_tp < 1:
                    n_tp = 1
                    n_sh = num_segments - 1

            # ================================================================
            # 计算各段制冷剂温度 T_ref (沿空气流向, 段0→n-1)
            # ================================================================
            # 逆流换热器 (段0 = 空气进口侧 = 制冷剂出口):
            #   段0: 制冷剂出口, T_ref = T_evap + SH (最高)
            #   段 n_sh-1: 过热段最后一排, T_ref ≈ T_evap (制冷剂刚蒸干)
            #   段 n_sh ~ n-1: 两相段, T_ref = T_evap
            #
            # 物理一致性保证:
            #   - 段0 T_ref > 段n-1 T_ref (逆流方向, 制冷剂沿流向放热冷却)
            #   - 制冷剂从 段0 (T_evap+SH) 单调下降到 段n-1 (T_evap)
            #
            # 采用线性分布 (各排吸热能力相同时):
            T_ref_per_seg = np.zeros(num_segments)
            if sh < 0.01:
                # 全部为两相段
                T_ref_per_seg[:] = T_evap
            elif n_sh >= num_segments:
                # 过热段覆盖整个换热器 (极少发生, 极端低流量)
                # 沿流向线性降温
                for seg in range(num_segments):
                    T_ref_per_seg[seg] = T_evap + sh * (num_segments - 1 - seg) / max(num_segments - 1, 1)
            else:
                for seg in range(num_segments):
                    if seg < n_sh:
                        # 过热段: 段0 最高 (T_evap+SH), 段n_sh-1 最低 (T_evap+SH/n_sh)
                        T_ref_per_seg[seg] = T_evap + sh * (n_sh - seg) / n_sh
                    else:
                        # 两相段
                        T_ref_per_seg[seg] = T_evap

                # 物理一致性: 确保 T_ref 沿空气流向单调下降
                # (逆流: 制冷剂从空气进口侧出口流向空气出口侧入口, 温度递减)
                for seg in range(1, num_segments):
                    if T_ref_per_seg[seg] > T_ref_per_seg[seg-1]:
                        T_ref_per_seg[seg] = T_ref_per_seg[seg-1]

            # ================================================================
            # 压降仿真: 预测→压降累积→修正T_ref
            # ASHRAE Fundamentals 2013 Ch.5
            # 制冷剂从段n-1(进口)流向段0(出口), 压降沿流向累积
            # ================================================================

            # ---- 预测步: 快速计算各段换热量→干度分布 ----
            T_air_pred = T_air_in
            W_air_pred = air_in.W
            Q_pred = np.zeros(num_segments)
            for seg_p in range(num_segments):
                T_ref_p = T_ref_per_seg[seg_p]
                h_i_p = h_i_sh if seg_p < n_sh else h_i_tp
                R_air_p = 1.0 / (eta_o * h_o * segment_area_o)
                R_ref_p = R_wall_seg + 1.0 / (h_i_p * segment_area_i)
                if T_air_pred > T_ref_p:
                    Q_pred[seg_p] = max((T_air_pred - T_ref_p) / (R_air_p + R_ref_p), 0)
                else:
                    Q_pred[seg_p] = 0
                dT_p = Q_pred[seg_p] / (m_air * cp_air)
                T_air_pred -= dT_p

            # 干度分布: 制冷剂从段n-1(进口)流向段0(出口)
            # x_seg[seg] = 段seg入口干度; x_seg[num_segments] = 进口干度 x_in
            x_seg_arr = np.ones(num_segments + 1)
            x_seg_arr[num_segments] = x_in
            for seg in range(num_segments - 1, -1, -1):
                if seg >= n_sh and mass_flow > 0 and h_fg > 0:
                    dx_seg = Q_pred[seg] / (mass_flow * h_fg)
                    x_seg_arr[seg] = min(x_seg_arr[seg + 1] + dx_seg, 1.0)
                else:
                    x_seg_arr[seg] = 1.0  # 过热段已蒸干

            # ---- 压降累积: 从段n-1(进口,p_evap)向段0(出口)反向 ----
            P_seg_arr = np.zeros(num_segments + 1)
            dP_seg_arr = np.zeros(num_segments)
            dP_fric_arr = np.zeros(num_segments)
            dP_acc_arr = np.zeros(num_segments)
            dP_bend_arr = np.zeros(num_segments)
            T_sat_seg_arr = np.zeros(num_segments)

            P_seg_arr[num_segments] = p_evap  # 进口压力
            L_per_seg = g.L_tube  # 每段管长 = 单根管长

            for seg in range(num_segments - 1, -1, -1):
                x_avg_seg = (x_seg_arr[seg + 1] + x_seg_arr[seg]) / 2
                dx_seg_val = x_seg_arr[seg] - x_seg_arr[seg + 1]

                if seg >= n_sh:
                    # 两相段: Lockhart-Martinelli
                    T_sat_in_seg = ref.saturation_temperature(P_seg_arr[seg + 1])
                    rho_l_seg = ref.liquid_density(T_sat_in_seg)
                    rho_v_seg = ref.sat_vapor_density(T_sat_in_seg)
                    mu_l_seg = ref.liquid_viscosity(T_sat_in_seg)
                    mu_v_seg = ref.vapor_viscosity(T_sat_in_seg, P_seg_arr[seg + 1])

                    dP_tot, dP_f, dP_a, dP_b = self.compute_pressure_drop_tp(
                        T_sat_in_seg, P_seg_arr[seg + 1], mass_flux,
                        x_avg_seg, dx_seg_val, g.d_i, L_per_seg,
                        rho_l_seg, rho_v_seg, mu_l_seg, mu_v_seg,
                        include_bend=(seg > 0))
                    dP_acc_arr[seg] = dP_a
                else:
                    # 过热段: 单相蒸汽
                    dP_tot, dP_f, dP_b = self.compute_pressure_drop_sh(
                        mass_flux, g.d_i, L_per_seg, rho_v, mu_v,
                        include_bend=(seg > 0))
                    dP_acc_arr[seg] = 0.0

                dP_seg_arr[seg] = dP_tot
                dP_fric_arr[seg] = dP_f
                dP_bend_arr[seg] = dP_b
                P_seg_arr[seg] = P_seg_arr[seg + 1] - dP_tot
                T_sat_seg_arr[seg] = ref.saturation_temperature(max(P_seg_arr[seg], 1000))

            # ---- 修正步: 用局部饱和温度更新T_ref ----
            # 两相段: T_ref = T_sat(局部压力)
            # 制冷剂从段n-1(进口)流向段0(出口), 压降累积导致
            # T_sat沿制冷剂流向递减 = 沿空气流向递增
            for seg in range(n_sh, num_segments):
                T_ref_per_seg[seg] = T_sat_seg_arr[seg]

            # 过热段: 基于各段局部饱和温度+过热度分布
            # 段0(出口): T_sat[0] + SH (最高)
            # 段n_sh-1(刚蒸干): T_sat[n_sh-1] + SH/n_sh
            if n_sh > 0 and sh >= 0.01:
                for seg in range(n_sh):
                    T_ref_per_seg[seg] = T_sat_seg_arr[seg] + sh * (n_sh - seg) / n_sh

            # 物理一致性: 仅过热段内T_ref沿空气流向单调下降
            for seg in range(1, max(n_sh, 1)):
                if T_ref_per_seg[seg] > T_ref_per_seg[seg - 1]:
                    T_ref_per_seg[seg] = T_ref_per_seg[seg - 1]

            # 存储压降分布供正式循环使用
            _P_seg = P_seg_arr
            _T_sat_seg = T_sat_seg_arr
            _dP_seg = dP_seg_arr
            _dP_fric = dP_fric_arr
            _dP_acc = dP_acc_arr
            _dP_bend = dP_bend_arr
            _x_seg = x_seg_arr

            for seg in range(num_segments):
                air_current = MoistAir(T_air, W_air, p_atm)
                h_air_current = air_current.h * 1000  # J/kg

                T_ref = T_ref_per_seg[seg]
                h_i_seg = h_i_sh if seg < n_sh else h_i_tp

                # 分段热阻
                Le = 1.0
                R_air_dry = 1.0 / (eta_o * h_o * segment_area_o)
                R_ref_wall = R_wall_seg + 1.0 / (h_i_seg * segment_area_i)
                G_enthalpy = eta_o * h_o * segment_area_o / (cp_air * Le)

                T_dp_air = air_current.T_dp

                # 干工况表面温度估算
                if T_air > T_ref:
                    Q_dry_est = (T_air - T_ref) / (R_air_dry + R_ref_wall)
                    T_s_dry = T_ref + Q_dry_est * R_ref_wall
                else:
                    Q_dry_est = 0.0
                    T_s_dry = T_ref

                # 判别干/湿工况
                if T_s_dry >= T_dp_air:
                    # 干工况
                    is_wet = False
                    T_s = T_s_dry
                    dQ = max(Q_dry_est, 0)
                    dQ_sens = dQ
                    dQ_lat = 0.0
                    dW = 0.0
                else:
                    # 湿工况 - 二分法求解壁面温度 T_s
                    is_wet = True
                    T_s_lo = T_ref
                    T_s_hi = T_air
                    T_s = (T_ref + T_s_dry) / 2

                    for _inner in range(40):
                        T_s = (T_s_lo + T_s_hi) / 2
                        h_s = saturated_enthalpy(T_s, p_atm) * 1000
                        Q_air_side = G_enthalpy * (h_air_current - h_s)
                        Q_ref_side = (T_s - T_ref) / R_ref_wall
                        f_val = Q_air_side - Q_ref_side

                        if abs(f_val) < 0.5:
                            break
                        if f_val > 0:
                            T_s_lo = T_s
                        else:
                            T_s_hi = T_s

                    dQ = G_enthalpy * (h_air_current - saturated_enthalpy(T_s, p_atm) * 1000)
                    dQ = max(dQ, 0)

                    if dQ > 0:
                        dQ_sens = eta_o * h_o * segment_area_o * (T_air - T_s)
                        dQ_sens = max(dQ_sens, 0)
                        dQ_lat = max(dQ - dQ_sens, 0)
                        dW = dQ_lat / 2501e3 if dQ_lat > 0 else 0.0
                    else:
                        dQ_sens = 0.0
                        dQ_lat = 0.0
                        dW = 0.0

                T_s_sum += T_s

                # 更新空气状态
                dT_air = dQ_sens / (m_air * cp_air)
                T_air_new = T_air - dT_air
                W_air_new = max(W_air - dW / m_air, 0)
                if T_s < T_air:
                    T_air_new = max(T_air_new, T_s - 0.5)

                segs.append({
                    'segment': seg + 1,
                    'row': seg + 1,
                    'T_air_in': T_air,
                    'T_air_out': T_air_new,
                    'W_in': W_air,
                    'W_out': W_air_new,
                    'T_ref': T_ref,
                    'T_surface': T_s,
                    'is_wet': is_wet,
                    'Q': dQ,
                    'Q_sens': dQ_sens,
                    'Q_lat': dQ_lat,
                    'condensate': dW,
                    'h_air_in': h_air_current / 1000,
                    'P_ref': _P_seg[seg],
                    'T_sat': _T_sat_seg[seg],
                    'dP_seg': _dP_seg[seg],
                    'dP_friction': _dP_fric[seg],
                    'dP_accel': _dP_acc[seg],
                    'dP_bend': _dP_bend[seg],
                    'x_ref_in': _x_seg[seg + 1],
                    'x_ref_out': _x_seg[seg],
                })

                Q_total += dQ
                Q_sens_total += dQ_sens
                Q_lat_total += dQ_lat
                cond_total += dW
                T_air = T_air_new
                W_air = W_air_new
                states.append(MoistAir(T_air, W_air, p_atm))

            return (Q_total, Q_sens_total, Q_lat_total, cond_total,
                    T_s_sum / num_segments, segs, states, T_air, W_air)

        # ================================================================
        # 二分法求解过热度: Q_air(superheat) = Q_tp_max + m_dot * cp_v * superheat
        # ================================================================
        # f(sh) = Q_air(sh) - Q_ref(sh) 单调递减
        #   f(0) > 0 → 需要过热;  f(sh_max) < 0 → 过热太多
        #
        # 逆流换热物理约束 (热力学第二定律):
        #   制冷剂出口温度 ≤ 空气进口温度
        #   即 T_evap + superheat ≤ T_air_in
        #   即 superheat ≤ T_air_in - T_evap
        # ================================================================

        sh_max = min(80.0, max(0.0, T_air_in - T_evap - 0.5))
        converged = False
        n_iter = 0

        # 评估 sh=0
        Q0, Qs0, Ql0, c0, Ts0, segs0, states0, Ta0, W0 = _evaluate(0.0)

        if Q0 <= Q_tp_max + 1.0:
            # 制冷剂未完全蒸发, 无过热
            superheat = 0.0
            converged = True
            n_iter = 1
            Q_total = Q0
            Q_sensible_total = Qs0
            Q_latent_total = Ql0
            condensate_total = c0
            T_surface_avg = Ts0
            results_segments = segs0
            air_states = states0
            T_air = Ta0
            W_air = W0
        elif sh_max < 0.01:
            # T_evap >= T_air_in: 无传热驱动力
            superheat = 0.0
            converged = True
            n_iter = 1
            Q_total = Q_tp_max  # 仅两相蒸发, 无过热
            Q_sensible_total = 0.0
            Q_latent_total = 0.0
            condensate_total = 0.0
            T_surface_avg = T_evap
            results_segments = segs0
            air_states = states0
            T_air = T_air_in
            W_air = air_in.W
        else:
            # 检查在物理约束范围内是否有解
            Q_at_max, Qs_at_max, Ql_at_max, c_at_max, Ts_at_max, \
                segs_at_max, states_at_max, Ta_at_max, W_at_max = _evaluate(sh_max)
            Q_ref_at_max = Q_tp_max + mass_flow * cp_v * sh_max
            f_at_max = Q_at_max - Q_ref_at_max

            if f_at_max > 0:
                # 制冷剂侧瓶颈: 即使出口温度达到空气进口温度上限,
                # 空气侧仍能提供更多热量, 但制冷剂无法吸收
                # 实际换热量 = 制冷剂侧最大吸热量
                superheat = sh_max
                converged = True
                n_iter = 1
                T_surface_avg = Ts_at_max
                results_segments = segs_at_max
                # 按制冷剂侧限制缩放各段换热量
                scale = Q_ref_at_max / max(Q_at_max, 1e-6)
                Q_total = Q_ref_at_max
                Q_sensible_total = Qs_at_max * scale
                Q_latent_total = Ql_at_max * scale
                condensate_total = c_at_max * scale
                # 修正分段数据和空气出口状态
                T_air = T_air_in
                W_air = air_in.W
                air_states = [air_in]
                for s in results_segments:
                    # 同步本段进风状态(累计的 T_air 反映缩放后的实际进风)
                    s['T_air_in'] = T_air
                    s['W_in'] = W_air
                    s['Q'] *= scale
                    s['Q_sens'] *= scale
                    s['Q_lat'] *= scale
                    s['condensate'] *= scale
                    dT_seg = s['Q_sens'] / (m_air * cp_air)
                    T_air = T_air - dT_seg
                    dW_seg = s['condensate'] / m_air
                    W_air = max(W_air - dW_seg, 0)
                    s['T_air_out'] = T_air
                    s['W_out'] = W_air
                    air_states.append(MoistAir(T_air, W_air, p_atm))
            else:
                # 二分法求解
                sh_lo = 0.0
                sh_hi = sh_max

                for iteration in range(40):
                    n_iter = iteration + 1
                    sh_mid = (sh_lo + sh_hi) / 2

                    Q_mid, _, _, _, _, _, _, _, _ = _evaluate(sh_mid)
                    Q_ref_mid = Q_tp_max + mass_flow * cp_v * sh_mid
                    f_mid = Q_mid - Q_ref_mid

                    if abs(f_mid) < 5.0:  # 5W 容差
                        sh_lo = sh_hi = sh_mid
                        break

                    if f_mid > 0:
                        sh_lo = sh_mid  # 需要更多过热
                    else:
                        sh_hi = sh_mid  # 过热太多

                superheat = (sh_lo + sh_hi) / 2
                converged = abs(sh_hi - sh_lo) < 0.01

                # 最终评估
                (Q_total, Q_sensible_total, Q_latent_total, condensate_total,
                 T_surface_avg, results_segments, air_states,
                 T_air, W_air) = _evaluate(superheat)

        # ================================================================
        # 最终结果
        # ================================================================
        air_out = MoistAir(T_air, W_air, p_atm)

        # ---- 制冷剂出口状态 ----
        if superheat > 0.01:
            x_out = 1.0
            h_ref_out = h_v + cp_v * superheat  # J/kg
        else:
            x_out = x_in + Q_total / (mass_flow * h_fg) if mass_flow > 0 else x_in
            x_out = min(x_out, 1.0)
            h_ref_out = h_l + x_out * h_fg       # J/kg

        # ---- 能量守恒验证 ----
        Q_ref = mass_flow * (h_ref_out - h_ref_in)     # 制冷剂侧 (W)
        Q_air = m_air * (air_in.h - air_out.h) * 1000   # 空气侧 (W)
        # 能量守恒偏差: 空气侧吸热 vs 制冷剂侧吸热
        # (Q_total 在主迭代中已与 Q_ref 严格匹配, 此处比较两侧独立计算的差值)
        energy_error = abs(Q_air - Q_ref) / max(abs(Q_ref), 1e-6) * 100  # %

        # ---- COP 估算 ----
        T_cond = T_evap + 35
        COP_carnot = (T_evap + 273.15) / (T_cond - T_evap)
        COP = COP_carnot * 0.6

        # ---- 接触系数 ----
        if T_air_in != T_surface_avg:
            BF = (air_out.T_db - T_surface_avg) / (T_air_in - T_surface_avg)
        else:
            BF = 0.0
        contact_factor = 1 - BF

        return {
            # ---- 性能指标 ----
            'capacity': Q_total / 1000,                      # kW (换热器换热量)
            'capacity_ref': Q_ref / 1000,                     # kW (制冷剂侧校核)
            'capacity_air': Q_air / 1000,                     # kW (空气侧校核)
            'sensible_capacity': Q_sensible_total / 1000,     # kW
            'latent_capacity': Q_latent_total / 1000,         # kW
            'SHR': Q_sensible_total / max(Q_total, 1e-6),     # 显热比
            'COP': COP,
            'condensate_rate': condensate_total * 3600,       # kg/h

            # ---- 空气侧结果 ----
            'air_in': air_in,
            'air_out': air_out,
            'm_air': m_air,
            'V_air': V_air,                       # 空气体积流量 (m³/s)
            'V_air_h': V_air * 3600,              # 空气体积流量 (m³/h)
            'face_velocity': face_velocity,
            'air_temp_drop': T_air_in - air_out.T_db,
            'dehumidification': (air_in.W - air_out.W) * 1000,

            # ---- 换热系数 ----
            'h_o': h_o,
            'h_i_tp': h_i_tp,
            'h_i_sh': h_i_sh,
            'h_l_ref': h_l_ref,
            'UA_dry': self.compute_overall_UA(h_o, h_i_tp, eta_o),
            'Re_D': Re_D,
            'G_c': G_c,
            'mass_flux': mass_flux,

            # ---- 翅片效率 ----
            'fin_efficiency': eta_f,
            'overall_efficiency': eta_o,

            # ---- 几何参数 ----
            'area_external': g.area_external_total,
            'area_internal': g.area_internal_total,
            'area_fin': g.area_fin,
            'volume_internal': g.volume_internal,       # 管内容积 (m³)

            # ---- 制冷剂状态 ----
            'T_evap': T_evap,
            'p_evap': p_evap / 1e6,
            'superheat': superheat,           # 计算输出: 出口过热度 (°C)
            'x_in': x_in,                     # 输入: 进口干度
            'x_out': x_out,                   # 计算输出: 出口干度
            'mass_flow': mass_flow,
            'mass_flow_h': mass_flow * 3600,   # 制冷剂质量流量 (kg/h)
            'h_fg': h_fg / 1000,              # kJ/kg
            'h_ref_in': h_ref_in / 1000,      # kJ/kg 进口焓
            'h_ref_out': h_ref_out / 1000,    # kJ/kg 出口焓
            'rho_l': rho_l,
            'rho_v': rho_v,
            'Q_tp_max': Q_tp_max / 1000,      # kW 完全蒸发所需热量

            # ---- 盘管特性 ----
            'contact_factor': contact_factor,
            'bypass_factor': BF,
            'surface_temp': T_surface_avg,

            # ---- 能量守恒 ----
            'energy_balance_error': energy_error,  # %
            'converged': converged,
            'iterations': n_iter,

            # ---- 压降仿真结果 ----
            'delta_P_total': sum(s.get('dP_seg', 0) for s in results_segments) / 1000,  # kPa
            'delta_P_friction': sum(s.get('dP_friction', 0) for s in results_segments) / 1000,
            'delta_P_accel': sum(s.get('dP_accel', 0) for s in results_segments) / 1000,
            'delta_P_bend': sum(s.get('dP_bend', 0) for s in results_segments) / 1000,
            'p_evap_in': p_evap / 1e6,   # MPa 进口压力
            'p_evap_out': (p_evap - sum(s.get('dP_seg', 0) for s in results_segments)) / 1e6,
            'T_sat_profile': [s.get('T_sat', T_evap) for s in results_segments],
            'P_profile': [s.get('P_ref', p_evap) / 1000 for s in results_segments],  # kPa
            'T_sat_drop': (results_segments[-1].get('T_sat', T_evap) -
                          results_segments[0].get('T_sat', T_evap)) if results_segments else 0,

            # ---- 分段详情 ----
            'segments': results_segments,
            'air_states': air_states,
        }

    def sensitivity_analysis(self, param_name: str, param_range: np.ndarray,
                             base_params: dict) -> list:
        """
        参数敏感性分析
        对指定参数在范围内扫描, 返回各点的性能结果
        """
        results = []
        for val in param_range:
            params = base_params.copy()
            params[param_name] = val
            try:
                result = self.solve(**params)
                result['param_value'] = val
                results.append(result)
            except Exception as e:
                results.append({'param_value': val, 'error': str(e)})
        return results
