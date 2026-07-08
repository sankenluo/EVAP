"""
翅片管式蒸发器数字孪生 - 交互式仿真应用
Finned-Tube Evaporator Digital Twin - Interactive Simulation App

基于 ASHRAE 知识库标准与理论:
  - ASHRAE Fundamentals 2013 Ch.1/Ch.4 (湿空气/传热)
  - ASHRAE Systems & Equipment 2012 Ch.22 (冷却盘管)
  - ASHRAE Standard 33-2000 (盘管测试方法)
  - JB/T 7659.5-95 / GB/T 23130-2008 / GB/T 47234-2026

运行方式:
  streamlit run app.py
"""
import sys
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    FinEvaporatorGeometry, FinEvaporatorDigitalTwin,
    FinEfficiency, AirSideCorrelation, AirProperties
)
from refrigerant import Refrigerant


# ============================================================
# 模型代码版本检测 (用于自动失效缓存)
# ============================================================
def _get_model_signature() -> str:
    """读取 model.py 的修改时间和大小作为缓存依赖键"""
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.py')
        stat = os.stat(model_path)
        return f"v{int(stat.st_mtime)}_{stat.st_size}"
    except Exception:
        return "default"


_MODEL_SIG = _get_model_signature()
from psychrometrics import (
    MoistAir, enthalpy, humidity_ratio, saturation_pressure_water,
    saturated_enthalpy, saturated_humidity_ratio, dew_point, relative_humidity
)


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="翅片式蒸发器数字孪生仿真",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 14px;
        color: #666;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 13px;
        opacity: 0.9;
    }
    .stMetric {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 侧边栏: 参数设置
# ============================================================

# 缓存清除 + 重置默认参数按钮
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🔄 清除缓存", help="模型修改后点击此按钮刷新图表"):
        st.cache_data.clear()
        st.rerun()
with col_btn2:
    if st.button("♻️ 重置默认", help="恢复室内侧蒸发器默认参数 (R404A, 管径15.88mm, 6排等)"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown("## ❄️ 蒸发器参数设置")

# --- 蒸发器类型预设 ---
st.sidebar.markdown("### 🏠 应用场景")

EVAP_PRESETS = {
    "室内侧蒸发器": {
        'tube_od': 15.88, 'tube_wt': 0.50, 'pt': 38.0, 'pl': 33.0,
        'num_rows': 6, 'fin_pitch': 3.5, 'face_vel': 2.5,
    },
    "室外侧蒸发器": {
        'tube_od': 15.88, 'tube_wt': 0.50, 'pt': 38.0, 'pl': 33.0,
        'num_rows': 8, 'fin_pitch': 8.0, 'face_vel': 2.5,
    },
}

evap_type = st.sidebar.radio(
    "蒸发器类型",
    ["室内侧蒸发器", "室外侧蒸发器"],
    horizontal=True,
    index=0,
    help="选择不同场景自动加载对应默认参数；两者差异为排数(6/8)和翅片间距(3.5/8mm)"
)

# 检测预设切换 → 更新对应 widget 的 session_state
_prev_preset = st.session_state.get('_evap_preset')
if _prev_preset is None:
    st.session_state['_evap_preset'] = evap_type
elif _prev_preset != evap_type:
    st.session_state['_evap_preset'] = evap_type
    for _k, _v in EVAP_PRESETS[evap_type].items():
        st.session_state[_k] = _v

_preset_defaults = EVAP_PRESETS[evap_type]

# 显示当前预设差异提示
if evap_type == "室内侧蒸发器":
    st.sidebar.caption("📋 室内侧: 6排 / 翅片间距3.5mm（可在下方修改）")
else:
    st.sidebar.caption("📋 室外侧: 8排 / 翅片间距8.0mm（可在下方修改）")

# --- 制冷剂选择 ---
st.sidebar.markdown("### 🧪 制冷剂")
refrigerant_name = st.sidebar.selectbox(
    "制冷剂类型",
    ["R404A", "R410A", "R134a", "R22", "R32", "R290", "R407C", "R23"],
    index=0,
    help="CoolProp 提供高精度物性数据 (基于 Helmholtz 状态方程)"
)

# --- 结构参数 ---
st.sidebar.markdown("### 🔩 结构参数 (JB/T 7659.5-95)")

col_g1, col_g2 = st.sidebar.columns(2)
with col_g1:
    tube_od = st.number_input("管外径 (mm)", 7.0, 16.0, _preset_defaults['tube_od'], 0.01,
                              key='tube_od', help="JB/T 7659.5-95 表5: 范围 7~16mm")
    fin_pitch = st.number_input("翅片节距 (mm)", 1.3, 12.0, _preset_defaults['fin_pitch'], 0.1,
                                key='fin_pitch', help="范围 1.3~12.0mm")
    num_rows = st.number_input("排数", 1, 20, _preset_defaults['num_rows'],
                               key='num_rows', help="范围 1~12 排，最大12排")
    tube_length = st.number_input("管长 (mm)", 200, 3000, 1800, 10)

with col_g2:
    tube_wt = st.number_input("壁厚 (mm)", 0.30, 1.0, _preset_defaults['tube_wt'], 0.01,
                              key='tube_wt', help="范围 0.30~1.0mm")
    fin_thk = st.number_input("翅片厚 (mm)", 0.10, 0.30, 0.15, 0.01,
                              help="范围 0.10~0.30mm")
    tubes_per_row = st.number_input("每排管数", 2, 40, 12)
    num_circuits = st.number_input("分液路数", 1, 20, 6)

# 排数校验: 最大12排
if num_rows > 12:
    st.sidebar.error(f"⚠️ 排数不能超过 12 排！当前输入: {num_rows} 排，请修改为 12 排及以下。")
    st.stop()

col_g3, col_g4 = st.sidebar.columns(2)
with col_g3:
    pt = st.number_input("横向管距 (mm)", 20.0, 50.0, _preset_defaults['pt'], 0.1, key='pt')
with col_g4:
    pl = st.number_input("纵向管距 (mm)", 15.0, 45.0, _preset_defaults['pl'], 0.1, key='pl')

fin_material = st.sidebar.selectbox("翅片材料", ["铝 (k=236)", "铜 (k=398)"])
k_fin = 236.0 if "铝" in fin_material else 398.0

# --- 运行工况 ---
st.sidebar.markdown("### 🌡️ 运行工况")

col_o1, col_o2 = st.sidebar.columns(2)
with col_o1:
    T_evap = st.number_input("蒸发温度 (°C)", -60.0, 15.0, 5.0, 0.5)
    T_air_in = st.number_input("进风干球温度 (°C)", -40.0, 50.0, 27.0, 0.5)
    face_vel = st.number_input("迎面风速 (m/s)", 0.5, 5.0, _preset_defaults['face_vel'], 0.1,
                               key='face_vel')

with col_o2:
    x_in = st.number_input("进口干度", 0.05, 0.50, 0.25, 0.01,
                           help="节流后制冷剂干度 (气相比例), 典型值 0.15~0.35")
    RH_in = st.number_input("进风相对湿度 (%)", 10, 100, 50, 5) / 100
    mass_flow_h = st.number_input("制冷剂流量 (kg/h)", 36.0, 3600.0, 540.0, 10.0,
                                   help="制冷剂质量流量, 工程常用单位 kg/h")
mass_flow = mass_flow_h / 3600.0  # 转换为 kg/s 供模型计算

# 实时显示流量换算值
_face_area = (tubes_per_row * pt / 1000) * (tube_length / 1000)  # 迎风面积 (m²)
_V_air_h = face_vel * _face_area * 3600  # 空气体积流量 (m³/h)
st.sidebar.markdown(f"""
<div style='background:#e8f5e9;padding:6px 10px;border-radius:6px;margin:4px 0;font-size:12px;'>
💨 <b>空气体积流量</b>: {_V_air_h:.0f} m³/h<br>
🧪 <b>制冷剂质量流量</b>: {mass_flow_h:.0f} kg/h ({mass_flow:.4f} kg/s)
</div>
""", unsafe_allow_html=True)

# --- 高级选项 ---
with st.sidebar.expander("⚙️ 高级选项"):
    boiling_method = st.selectbox(
        "管内沸腾换热关联式",
        ["Shah (1976)", "Kandlikar (1990)"],
        help="Shah: 通用两相流; Kandlikar: 含核态沸腾贡献"
    )
    method_key = 'shah' if 'Shah' in boiling_method else 'kandlikar'
    # 分段数 = 排数 (每段对应一排管, 逆流换热)
    num_segments = int(num_rows)
    p_atm = st.number_input("大气压 (kPa)", 80.0, 110.0, 101.325, 0.1) * 1000


# ============================================================
# 构建模型并计算
# ============================================================

@st.cache_data
def run_simulation(tube_od, tube_wt, fin_pitch, fin_thk, num_rows,
                   tubes_per_row, tube_length, num_circuits, pt, pl,
                   k_fin, T_evap, x_in, T_air_in, RH_in, face_vel,
                   mass_flow, refrigerant_name, method_key, num_segments, p_atm,
                   _model_sig: str = ""):  # 模型代码签名, 修改 model.py 自动失效缓存
    """运行仿真 (带缓存, _model_sig 自动基于 model.py 修改时间)"""
    geo = FinEvaporatorGeometry(
        tube_outer_diameter=tube_od,
        tube_wall_thickness=tube_wt,
        fin_pitch=fin_pitch,
        fin_thickness=fin_thk,
        num_rows=int(num_rows),
        num_tubes_per_row=int(tubes_per_row),
        tube_length=tube_length,
        num_circuits=int(num_circuits),
        tube_pitch_transverse=pt,
        tube_pitch_longitudinal=pl,
        k_fin=k_fin,
    )
    ref = Refrigerant(refrigerant_name)
    dt = FinEvaporatorDigitalTwin(geo, ref)
    result = dt.solve(
        T_evap=T_evap, x_in=x_in,
        T_air_in=T_air_in, RH_air_in=RH_in,
        face_velocity=face_vel, mass_flow=mass_flow,
        p_atm=p_atm, num_segments=int(num_segments),
        boiling_method=method_key
    )
    return result, geo.summary(), ref.info()


result, geo_summary, ref_info = run_simulation(
    tube_od, tube_wt, fin_pitch, fin_thk, num_rows,
    tubes_per_row, tube_length, num_circuits, pt, pl,
    k_fin, T_evap, x_in, T_air_in, RH_in, face_vel,
    mass_flow, refrigerant_name, method_key, num_segments, p_atm,
    _model_sig=_MODEL_SIG  # 自动检测 model.py 修改
)


# ============================================================
# 主页面: 结果展示
# ============================================================

st.markdown('<div class="main-header">❄️ 翅片管式蒸发器数字孪生仿真</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">'
            f'基于 ASHRAE Handbook / JB/T 7659.5-95 / GB/T 47234-2026 | '
            f'制冷剂: {refrigerant_name} (CoolProp: {"✅" if ref_info["coolprop"] else "❌"}) | '
            f'沸腾关联式: {boiling_method}</div>',
            unsafe_allow_html=True)

# --- 核心性能指标 ---
st.markdown("### 📊 核心性能指标")

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("制冷量", f"{result['capacity']:.2f} kW")
with m2:
    st.metric("出口过热度", f"{result['superheat']:.1f} °C",
              help="由能量守恒计算得出, 非输入参数")
with m3:
    st.metric("显热比 SHR", f"{result['SHR']:.2f}")
with m4:
    st.metric("出风温度", f"{result['air_out'].T_db:.1f} °C")
with m5:
    st.metric("除湿量", f"{result['dehumidification']:.2f} g/kg")
with m6:
    st.metric("COP", f"{result['COP']:.2f}")

# 流量指标 (第二行)
m7, m8, m9, m10 = st.columns(4)
with m7:
    st.metric("空气体积流量", f"{result['V_air_h']:.0f} m³/h",
              help=f"迎风风速 {face_vel} m/s × 迎风面积 {_face_area:.3f} m²")
with m8:
    st.metric("空气质量流量", f"{result['m_air']:.3f} kg/s",
              help=f"空气密度 {result['air_in'].rho:.2f} kg/m³")
with m9:
    st.metric("制冷剂质量流量", f"{result['mass_flow_h']:.0f} kg/h",
              help=f"管内质量流速 {result['mass_flux']:.1f} kg/m²·s")
with m10:
    st.metric("制冷剂/空气质量比", f"{result['mass_flow']/result['m_air']:.3f}",
              help="制冷剂质量流量与空气质量流量之比")

# 能量守恒验证
eb_error = result['energy_balance_error']
eb_color = "green" if eb_error < 5 else "orange" if eb_error < 15 else "red"
conv_status = "✅ 收敛" if result['converged'] else "⚠️ 未收敛"
st.markdown(
    f"<div style='background:#f0f2f6;padding:8px 12px;border-radius:6px;margin:5px 0;font-size:13px;'>"
    f"<b>能量守恒验证</b> &nbsp;|&nbsp; "
    f"制冷剂侧: <b>{result['capacity_ref']:.2f} kW</b> &nbsp;|&nbsp; "
    f"空气侧: <b>{result['capacity_air']:.2f} kW</b> &nbsp;|&nbsp; "
    f"换热器计算: <b>{result['capacity']:.2f} kW</b> &nbsp;|&nbsp; "
    f"偏差: <span style='color:{eb_color}'><b>{eb_error:.1f}%</b></span> &nbsp;|&nbsp; "
    f"迭代: {result['iterations']} 次 {conv_status}"
    f"</div>", unsafe_allow_html=True
)

st.divider()

# --- 蒸发器结构示意图 ---
st.markdown("### 🏗️ 蒸发器结构示意图")

# 结构尺寸 (mm)
_Pt = pt      # 横向管距
_Pl = pl      # 纵向管距
_Do = tube_od # 管外径
_fp = fin_pitch  # 翅片节距
_ft = fin_thk    # 翅片厚
_Nr = int(num_rows)
_Nt = int(tubes_per_row)
_L = tube_length
_W = _Nt * _Pt   # 翅片宽度 (横向)
_D = _Nr * _Pl   # 空气流向深度

col_struct1, col_struct2 = st.columns(2)

with col_struct1:
    st.markdown("#### 管排布置俯视图")

    fig_top = go.Figure()
    shapes_top = []
    annot_top = []

    # 绘制每根管截面 (圆形), 错排布置
    for i in range(_Nr):
        y_pos = (i + 0.5) * _Pl
        x_offset = _Pt / 2 if i % 2 == 1 else 0  # 错排偏移
        for j in range(_Nt):
            x_pos = (j + 0.5) * _Pt + x_offset
            shapes_top.append(dict(
                type="circle", xref="x", yref="y",
                x0=x_pos - _Do/2, y0=y_pos - _Do/2,
                x1=x_pos + _Do/2, y1=y_pos + _Do/2,
                line=dict(color="#e67e22", width=1.5),
                fillcolor="#fef5e7",
            ))

    # 翅片轮廓矩形
    shapes_top.append(dict(
        type="rect", xref="x", yref="y",
        x0=0, y0=0, x1=_W, y1=_D,
        line=dict(color="#3498db", width=1.5, dash="dash"),
        fillcolor="rgba(52, 152, 219, 0.03)",
    ))

    # 空气流向箭头 (从下到上)
    arrow_x = -_Pt * 0.5
    shapes_top.append(dict(
        type="line", xref="x", yref="y",
        x0=arrow_x, y0=0, x1=arrow_x, y1=_D,
        line=dict(color="#e74c3c", width=3),
    ))
    annot_top.append(dict(
        x=arrow_x, y=_D, ax=arrow_x, ay=_D - _Pl * 0.4,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#e74c3c",
    ))
    annot_top.append(dict(
        x=arrow_x - _Pt * 0.15, y=_D / 2, text="空气 →",
        showarrow=False, font=dict(size=13, color="#e74c3c"),
        xref="x", yref="y", textangle=90,
    ))

    # 制冷剂流向箭头 (从上到下, 逆流)
    ref_x = _W + _Pt * 0.5
    shapes_top.append(dict(
        type="line", xref="x", yref="y",
        x0=ref_x, y0=_D, x1=ref_x, y1=0,
        line=dict(color="#2ecc71", width=3),
    ))
    annot_top.append(dict(
        x=ref_x, y=0, ax=ref_x, ay=_Pl * 0.4,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#2ecc71",
    ))
    annot_top.append(dict(
        x=ref_x + _Pt * 0.15, y=_D / 2, text="← 制冷剂",
        showarrow=False, font=dict(size=13, color="#2ecc71"),
        xref="x", yref="y", textangle=-90,
    ))

    # 制冷剂进口/出口标注 (逆流: 末排=进口, 第1排=出口)
    annot_top.append(dict(
        x=ref_x, y=_D + _Pl * 0.35, text="制冷剂进口",
        showarrow=False, font=dict(size=9, color="#2ecc71"),
        xref="x", yref="y",
    ))
    annot_top.append(dict(
        x=ref_x, y=-_Pl * 0.35, text="制冷剂出口",
        showarrow=False, font=dict(size=9, color="#2ecc71"),
        xref="x", yref="y",
    ))

    # 排号标注 (第1排在底部=空气进口侧, 末排在顶部=空气出口侧)
    for i in range(_Nr):
        y_pos = (i + 0.5) * _Pl
        annot_top.append(dict(
            x=ref_x + _Pt * 0.7, y=y_pos,
            text=f"第{i+1}排",
            showarrow=False, font=dict(size=9, color="#555"),
            xref="x", yref="y",
        ))

    # 尺寸标注
    annot_top.append(dict(
        x=_Pt / 2, y=-_Pl * 0.5, text=f"Pt={_Pt}mm",
        showarrow=False, font=dict(size=10, color="#333"),
        xref="x", yref="y",
    ))
    annot_top.append(dict(
        x=-_Pt * 1.0, y=_Pl / 2, text=f"Pl={_Pl}mm",
        showarrow=False, font=dict(size=10, color="#333"),
        xref="x", yref="y", textangle=-90,
    ))
    annot_top.append(dict(
        x=_W / 2, y=_D + _Pl * 0.5,
        text=f"宽度 {_W:.0f}mm ({_Nt}管×{_Pt}mm) | 深度 {_D:.0f}mm ({_Nr}排×{_Pl}mm)",
        showarrow=False, font=dict(size=10, color="#555"),
        xref="x", yref="y",
    ))

    fig_top.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                  marker=dict(size=0), showlegend=False))
    fig_top.update_layout(
        shapes=shapes_top, annotations=annot_top,
        xaxis=dict(title="横向宽度 (mm)", range=[-_Pt*1.5, _W + _Pt*1.8],
                   zeroline=False, showgrid=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(title="空气流向深度 (mm)", range=[-_Pl*1.0, _D + _Pl*0.8],
                   zeroline=False, showgrid=False),
        height=380, margin=dict(l=60, r=20, t=10, b=50),
        plot_bgcolor='white',
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown(f"""
    <div style='font-size:12px;color:#666;'>
    <b>结构参数</b>: 管外径 {_Do}mm | 横向管距 {_Pt}mm | 纵向管距 {_Pl}mm |
    排数 {_Nr} | 每排管数 {_Nt} | 总管数 {_Nr*_Nt} | 管长 {_L}mm |
    翅片节距 {_fp}mm | 翅片厚 {_ft}mm
    </div>
    """, unsafe_allow_html=True)

with col_struct2:
    st.markdown("#### 翅片排列侧视图")

    fig_side = go.Figure()
    shapes_side = []
    annot_side = []

    # 侧视图: X=空气流向深度, Y=管长方向 (局部展示翅片间距)
    # 只展示前 N_show 个翅片节距, 便于看清
    N_fin_show = min(int(_L / _fp), 10)
    Y_show = N_fin_show * _fp  # 展示的管长范围

    # 绘制翅片 (薄矩形)
    for k in range(N_fin_show + 1):
        y_pos = k * _fp
        shapes_side.append(dict(
            type="rect", xref="x", yref="y",
            x0=0, y0=y_pos - _ft/2, x1=_D, y1=y_pos + _ft/2,
            line=dict(color="#3498db", width=0.5),
            fillcolor="rgba(52, 152, 219, 0.15)",
        ))

    # 绘制管子 (垂直矩形, 在每排位置)
    for i in range(_Nr):
        x_pos = (i + 0.5) * _Pl
        shapes_side.append(dict(
            type="rect", xref="x", yref="y",
            x0=x_pos - _Do/2, y0=0, x1=x_pos + _Do/2, y1=Y_show,
            line=dict(color="#e67e22", width=1.5),
            fillcolor="rgba(230, 126, 34, 0.12)",
        ))

    # 空气流向箭头
    shapes_side.append(dict(
        type="line", xref="x", yref="y",
        x0=0, y0=-Y_show*0.08, x1=_D, y1=-Y_show*0.08,
        line=dict(color="#e74c3c", width=3),
    ))
    annot_side.append(dict(
        x=_D, y=-Y_show*0.08, ax=_D - _Pl*0.4, ay=-Y_show*0.08,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#e74c3c",
    ))
    annot_side.append(dict(
        x=_D/2, y=-Y_show*0.15, text="空气流向 →",
        showarrow=False, font=dict(size=12, color="#e74c3c"),
        xref="x", yref="y",
    ))

    # 制冷剂流向箭头 (逆流)
    shapes_side.append(dict(
        type="line", xref="x", yref="y",
        x0=_D, y0=Y_show + Y_show*0.08, x1=0, y1=Y_show + Y_show*0.08,
        line=dict(color="#2ecc71", width=3),
    ))
    annot_side.append(dict(
        x=0, y=Y_show + Y_show*0.08, ax=_Pl*0.4, ay=Y_show + Y_show*0.08,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3, arrowcolor="#2ecc71",
    ))
    annot_side.append(dict(
        x=_D/2, y=Y_show + Y_show*0.15, text="← 制冷剂流向 (逆流)",
        showarrow=False, font=dict(size=12, color="#2ecc71"),
        xref="x", yref="y",
    ))

    # 翅片节距标注
    if N_fin_show >= 2:
        annot_y = _fp
        shapes_side.append(dict(
            type="line", xref="x", yref="y",
            x0=_D + _Pl*0.2, y0=0, x1=_D + _Pl*0.2, y1=_fp,
            line=dict(color="#8e44ad", width=1, dash="dot"),
        ))
        shapes_side.append(dict(
            type="line", xref="x", yref="y",
            x0=_D + _Pl*0.1, y0=0, x1=_D + _Pl*0.3, y1=0,
            line=dict(color="#8e44ad", width=1),
        ))
        shapes_side.append(dict(
            type="line", xref="x", yref="y",
            x0=_D + _Pl*0.1, y0=_fp, x1=_D + _Pl*0.3, y1=_fp,
            line=dict(color="#8e44ad", width=1),
        ))
        annot_side.append(dict(
            x=_D + _Pl*0.55, y=_fp/2,
            text=f"翅片节距<br>={_fp}mm",
            showarrow=False, font=dict(size=10, color="#8e44ad"),
            xref="x", yref="y",
        ))

    # 排号标注
    for i in range(_Nr):
        x_pos = (i + 0.5) * _Pl
        annot_side.append(dict(
            x=x_pos, y=-Y_show*0.25,
            text=f"第{i+1}排",
            showarrow=False, font=dict(size=9, color="#555"),
            xref="x", yref="y",
        ))

    annot_side.append(dict(
        x=_D/2, y=Y_show + Y_show*0.28,
        text=f"(展示管长方向前 {N_fin_show} 个翅片节距 / 共 {int(_L/_fp)} 片)",
        showarrow=False, font=dict(size=9, color="#888"),
        xref="x", yref="y",
    ))

    fig_side.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                   marker=dict(size=0), showlegend=False))
    fig_side.update_layout(
        shapes=shapes_side, annotations=annot_side,
        xaxis=dict(title="空气流向深度 (mm)", range=[-_Pl*0.3, _D + _Pl*1.2],
                   zeroline=False, showgrid=False),
        yaxis=dict(title="管长方向 (mm)", range=[-Y_show*0.35, Y_show*1.35],
                   zeroline=False, showgrid=False),
        height=380, margin=dict(l=60, r=20, t=10, b=50),
        plot_bgcolor='white',
    )
    st.plotly_chart(fig_side, use_container_width=True)

    st.markdown(f"""
    <div style='font-size:12px;color:#666;'>
    <b>面积汇总</b>: 迎风面积 {geo_summary['迎风面积 (m²)']} m² |
    翅片面积 {geo_summary['翅片面积 (m²)']} m² |
    管外表面积 {geo_summary['管外表面积 (m²)']} m² |
    空气侧总面积 {geo_summary['空气侧总面积 (m²)']} m² |
    管内总面积 {geo_summary['管内总面积 (m²)']} m² |
    翅片面积占比 {geo_summary['翅片面积占比']}
    </div>
    """, unsafe_allow_html=True)

st.divider()
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📈 换热性能分解")

    # 显热/潜热饼图
    fig_pie = go.Figure(data=[go.Pie(
        labels=['显热 (降温)', '潜热 (除湿)'],
        values=[result['sensible_capacity'], result['latent_capacity']],
        hole=0.5,
        marker=dict(colors=['#3498db', '#e74c3c']),
        textinfo='label+percent',
        textfont_size=14,
    )])
    fig_pie.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # 换热系数对比
    st.markdown("#### 换热系数")
    fig_h = go.Figure(data=[go.Bar(
        x=['管外空气侧 h_o', '管内沸腾 h_i', '液相单相 h_l'],
        y=[result['h_o'], result['h_i_tp'], result['h_l_ref']],
        marker_color=['#2ecc71', '#e67e22', '#95a5a6'],
        text=[f'{result["h_o"]:.1f}', f'{result["h_i_tp"]:.1f}', f'{result["h_l_ref"]:.1f}'],
        textposition='outside',
    )])
    fig_h.update_layout(
        yaxis_title='换热系数 (W/m²·K)',
        height=280,
        margin=dict(l=40, r=20, t=10, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_h, use_container_width=True)

with col_right:
    st.markdown("### 🌡️ 沿程温度分布")

    segments = result['segments']
    seg_nums = [s['segment'] for s in segments]
    T_air_in_seg = [s['T_air_in'] for s in segments]
    T_air_out_seg = [s['T_air_out'] for s in segments]
    T_ref_seg = [s['T_ref'] for s in segments]
    T_surf_seg = [s.get('T_surface', 0) for s in segments]

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=seg_nums, y=T_air_in_seg,
        mode='lines+markers', name='空气进口温度',
        line=dict(color='#e74c3c', width=2),
    ))
    fig_temp.add_trace(go.Scatter(
        x=seg_nums, y=T_air_out_seg,
        mode='lines+markers', name='空气出口温度',
        line=dict(color='#3498db', width=2),
    ))
    fig_temp.add_trace(go.Scatter(
        x=seg_nums, y=T_ref_seg,
        mode='lines+markers', name='制冷剂温度',
        line=dict(color='#2ecc71', width=2, dash='dash'),
    ))
    fig_temp.add_trace(go.Scatter(
        x=seg_nums, y=T_surf_seg,
        mode='lines+markers', name='壁面温度',
        line=dict(color='#f39c12', width=1.5, dash='dashdot'),
    ))
    # 露点线
    T_dp_seg = []
    for s in segments:
        air = MoistAir(s['T_air_in'], s['W_in'], p_atm)
        T_dp_seg.append(air.T_dp)
    fig_temp.add_trace(go.Scatter(
        x=seg_nums, y=T_dp_seg,
        mode='lines', name='空气露点温度',
        line=dict(color='#9b59b6', width=1.5, dash='dot'),
    ))
    # 逆流流向标注
    n_seg = len(seg_nums)
    y_min = min(min(T_air_out_seg), min(T_ref_seg), min(T_surf_seg), min(T_dp_seg)) - 2
    fig_temp.add_annotation(
        x=n_seg / 2, y=y_min,
        text="空气流向 →",
        showarrow=False, font=dict(size=13, color='#e74c3c'),
        xref='x', yref='y',
    )
    fig_temp.add_annotation(
        x=n_seg / 2, y=y_min - 2.5,
        text="← 制冷剂流向",
        showarrow=False, font=dict(size=13, color='#2ecc71'),
        xref='x', yref='y',
    )
    fig_temp.update_layout(
        xaxis_title='排号 (沿空气流向)',
        yaxis_title='温度 (°C)',
        height=320,
        margin=dict(l=40, r=20, t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # 翅片效率
    st.markdown("#### 翅片效率与整体表面效率")
    fig_eff = go.Figure(data=[go.Bar(
        x=['翅片效率 η_f', '整体表面效率 η_o'],
        y=[result['fin_efficiency'] * 100, result['overall_efficiency'] * 100],
        marker_color=['#1abc9c', '#16a085'],
        text=[f'{result["fin_efficiency"]*100:.1f}%', f'{result["overall_efficiency"]*100:.1f}%'],
        textposition='outside',
    )])
    fig_eff.update_layout(
        yaxis_title='效率 (%)',
        yaxis_range=[0, 100],
        height=250,
        margin=dict(l=40, r=20, t=10, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_eff, use_container_width=True)

st.divider()

# --- 焓湿图 ---
st.markdown("### 📐 焓湿图 (Psychrometric Chart) - ASHRAE Fundamentals Ch.1")

col_chart, col_info = st.columns([3, 1])

with col_chart:
    # 绘制焓湿图
    T_range = np.linspace(-50, 50, 200)
    p = p_atm

    fig_psyc = go.Figure()

    # 饱和线
    W_sat = [saturated_humidity_ratio(T, p) * 1000 for T in T_range]
    fig_psyc.add_trace(go.Scatter(
        x=T_range, y=W_sat,
        mode='lines', name='饱和线 (RH=100%)',
        line=dict(color='blue', width=2),
    ))

    # 等相对湿度线
    for rh in [0.2, 0.4, 0.6, 0.8]:
        W_rh = [humidity_ratio(T, rh, p) * 1000 for T in T_range]
        fig_psyc.add_trace(go.Scatter(
            x=T_range, y=W_rh,
            mode='lines', name=f'RH={int(rh*100)}%',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=True,
        ))

    # 进风状态点
    air_in = result['air_in']
    fig_psyc.add_trace(go.Scatter(
        x=[air_in.T_db], y=[air_in.W * 1000],
        mode='markers+text', name=f'进风 ({air_in.T_db:.1f}°C, {air_in.W*1000:.1f}g/kg)',
        marker=dict(size=14, color='red', symbol='circle'),
        text=['进风'], textposition='top center',
        textfont=dict(size=12, color='red'),
    ))

    # 出风状态点
    air_out = result['air_out']
    fig_psyc.add_trace(go.Scatter(
        x=[air_out.T_db], y=[air_out.W * 1000],
        mode='markers+text', name=f'出风 ({air_out.T_db:.1f}°C, {air_out.W*1000:.1f}g/kg)',
        marker=dict(size=14, color='blue', symbol='circle'),
        text=['出风'], textposition='bottom center',
        textfont=dict(size=12, color='blue'),
    ))

    # 过程线
    fig_psyc.add_trace(go.Scatter(
        x=[air_in.T_db, air_out.T_db],
        y=[air_in.W * 1000, air_out.W * 1000],
        mode='lines', name='冷却除湿过程',
        line=dict(color='green', width=3),
        showlegend=False,
    ))

    fig_psyc.update_layout(
        xaxis_title='干球温度 (°C)',
        yaxis_title='含湿量 (g/kg干空气)',
        height=420,
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_psyc, use_container_width=True)

with col_info:
    st.markdown("#### 空气状态对比")
    st.markdown(f"""
    | 参数 | 进风 | 出风 |
    |------|------|------|
    | 干球温度 | {air_in.T_db:.1f} °C | {air_out.T_db:.1f} °C |
    | 含湿量 | {air_in.W*1000:.2f} g/kg | {air_out.W*1000:.2f} g/kg |
    | 相对湿度 | {air_in.rh*100:.1f} % | {air_out.rh*100:.1f} % |
    | 比焓 | {air_in.h:.1f} kJ/kg | {air_out.h:.1f} kJ/kg |
    | 露点 | {air_in.T_dp:.1f} °C | {air_out.T_dp:.1f} °C |
    | 湿球 | {air_in.T_wb:.1f} °C | {air_out.T_wb:.1f} °C |
    """)
    st.markdown(f"**焓降**: {(air_in.h - air_out.h):.1f} kJ/kg")
    st.markdown(f"**温降**: {(air_in.T_db - air_out.T_db):.1f} °C")
    st.markdown(f"**除湿**: {(air_in.W - air_out.W)*1000:.2f} g/kg")
    st.markdown(f"**接触因子**: {result['contact_factor']:.3f}")

st.divider()

# --- 敏感性分析 ---
st.markdown("### 📉 参数敏感性分析")

sens_param = st.selectbox(
    "选择扫描参数",
    ["制冷剂流量", "蒸发温度", "迎面风速", "进风温度", "进风相对湿度", "翅片节距", "排数", "进口干度"],
    label_visibility="collapsed"
)

# 参数映射
sens_map = {
    "制冷剂流量": ("mass_flow", np.linspace(72, 1800, 25), "kg/h"),
    "蒸发温度": ("T_evap", np.linspace(-40, 12, 53), "°C"),
    "迎面风速": ("face_velocity", np.linspace(0.5, 4.0, 15), "m/s"),
    "进风温度": ("T_air_in", np.linspace(-40, 40, 41), "°C"),
    "进风相对湿度": ("RH_air_in", np.linspace(0.2, 0.9, 15), ""),
    "翅片节距": ("fin_pitch", np.linspace(1.3, 12.0, 12), "mm"),
    "排数": ("num_rows", np.arange(1, 13), "排"),
    "进口干度": ("x_in", np.linspace(0.05, 0.50, 19), ""),
}

param_key, param_range, param_unit = sens_map[sens_param]

base_params = dict(
    T_evap=T_evap, x_in=x_in, T_air_in=T_air_in,
    RH_air_in=RH_in, face_velocity=face_vel, mass_flow=mass_flow,
    p_atm=p_atm, num_segments=int(num_segments), boiling_method=method_key
)

# 构建几何参数 (需要处理排数/翅片节距的变化)
def run_sens(val):
    if param_key in ['fin_pitch', 'num_rows']:
        # 结构参数变化 - 需要重建几何
        new_num_rows = int(num_rows) if param_key != 'num_rows' else int(val)
        kwargs = dict(
            tube_outer_diameter=tube_od, tube_wall_thickness=tube_wt,
            fin_pitch=fin_pitch if param_key != 'fin_pitch' else val,
            fin_thickness=fin_thk,
            num_rows=new_num_rows,
            num_tubes_per_row=int(tubes_per_row),
            tube_length=tube_length, num_circuits=int(num_circuits),
            tube_pitch_transverse=pt, tube_pitch_longitudinal=pl,
            k_fin=k_fin,
        )
        geo = FinEvaporatorGeometry(**kwargs)
        ref = Refrigerant(refrigerant_name)
        dt = FinEvaporatorDigitalTwin(geo, ref)
        # 分段数 = 排数
        params = base_params.copy()
        params['num_segments'] = new_num_rows
        return dt.solve(**params)
    else:
        params = base_params.copy()
        if param_key == 'RH_air_in':
            params[param_key] = val
        elif param_key == 'mass_flow':
            params[param_key] = float(val) / 3600.0  # kg/h → kg/s
        else:
            params[param_key] = float(val)
        geo = FinEvaporatorGeometry(
            tube_outer_diameter=tube_od, tube_wall_thickness=tube_wt,
            fin_pitch=fin_pitch, fin_thickness=fin_thk,
            num_rows=int(num_rows), num_tubes_per_row=int(tubes_per_row),
            tube_length=tube_length, num_circuits=int(num_circuits),
            tube_pitch_transverse=pt, tube_pitch_longitudinal=pl, k_fin=k_fin,
        )
        ref = Refrigerant(refrigerant_name)
        dt = FinEvaporatorDigitalTwin(geo, ref)
        return dt.solve(**params)

sens_results = []
for val in param_range:
    try:
        r = run_sens(val)
        sens_results.append({
            'value': val,
            'capacity': r['capacity'],
            'SHR': r['SHR'],
            'COP': r['COP'],
            'T_out': r['air_out'].T_db,
            'superheat': r['superheat'],
            'fin_eff': r['fin_efficiency'],
        })
    except Exception:
        pass

if sens_results:
    vals = [r['value'] for r in sens_results]
    caps = [r['capacity'] for r in sens_results]
    shrs = [r['SHR'] for r in sens_results]
    cops = [r['COP'] for r in sens_results]
    t_outs = [r['T_out'] for r in sens_results]
    shs = [r['superheat'] for r in sens_results]

    fig_sens = make_subplots(
        rows=2, cols=3,
        subplot_titles=('制冷量', '出口过热度', '显热比 SHR', 'COP', '出风温度', ''),
        vertical_spacing=0.15, horizontal_spacing=0.08,
    )

    x_label = f"{sens_param} ({param_unit})" if param_unit else sens_param

    fig_sens.add_trace(go.Scatter(x=vals, y=caps, mode='lines+markers',
                                  name='制冷量', line=dict(color='#e74c3c', width=2)),
                       row=1, col=1)
    fig_sens.add_trace(go.Scatter(x=vals, y=shs, mode='lines+markers',
                                  name='过热度', line=dict(color='#e67e22', width=2)),
                       row=1, col=2)
    fig_sens.add_trace(go.Scatter(x=vals, y=shrs, mode='lines+markers',
                                  name='SHR', line=dict(color='#3498db', width=2)),
                       row=1, col=3)
    fig_sens.add_trace(go.Scatter(x=vals, y=cops, mode='lines+markers',
                                  name='COP', line=dict(color='#2ecc71', width=2)),
                       row=2, col=1)
    fig_sens.add_trace(go.Scatter(x=vals, y=t_outs, mode='lines+markers',
                                  name='出风温度', line=dict(color='#9b59b6', width=2)),
                       row=2, col=2)

    fig_sens.update_xaxes(title_text=x_label, row=2, col=1)
    fig_sens.update_xaxes(title_text=x_label, row=2, col=2)
    fig_sens.update_yaxes(title_text='kW', row=1, col=1)
    fig_sens.update_yaxes(title_text='°C', row=1, col=2)
    fig_sens.update_yaxes(title_text='SHR', row=1, col=3)
    fig_sens.update_yaxes(title_text='COP', row=2, col=1)
    fig_sens.update_yaxes(title_text='°C', row=2, col=2)

    fig_sens.update_layout(height=400, showlegend=False,
                           margin=dict(l=50, r=20, t=30, b=20))
    st.plotly_chart(fig_sens, use_container_width=True)

st.divider()

# --- 几何参数与详细数据 ---
col_geom, col_detail = st.columns([1, 1])

with col_geom:
    st.markdown("### 📐 几何参数")
    geo_df_data = [(k, v) for k, v in geo_summary.items()]
    for k, v in geo_df_data:
        st.text(f"  {k}: {v}")

with col_detail:
    st.markdown("### 🔬 换热器关键参数")
    st.markdown(f"""
    | 参数 | 值 |
    |------|------|
    | 总传热系数 UA | {result['UA_dry']:.1f} W/K |
    | 空气侧雷诺数 Re_D | {result['Re_D']:.0f} |
    | 最大质量流速 G_c | {result['G_c']:.2f} kg/m²·s |
    | 管内质量流速 | {result['mass_flux']:.1f} kg/m²·s |
    | 蒸发压力(进口) | {result['p_evap_in']:.3f} MPa |
    | 蒸发压力(出口) | {result['p_evap_out']:.3f} MPa |
    | 管内总压降 | {result['delta_P_total']:.2f} kPa |
    | 饱和温度衰减 | {result['T_sat_drop']:.3f} °C |
    | 汽化潜热 | {result['h_fg']:.1f} kJ/kg |
    | 进口干度 | {result['x_in']:.2f} |
    | 出口干度 | {result['x_out']:.2f} |
    | 出口过热度 | {result['superheat']:.1f} °C |
    | 制冷剂进口焓 | {result['h_ref_in']:.1f} kJ/kg |
    | 制冷剂出口焓 | {result['h_ref_out']:.1f} kJ/kg |
    | 完全蒸发需热量 | {result['Q_tp_max']:.2f} kW |
    | 液相密度 | {result['rho_l']:.1f} kg/m³ |
    | 气相密度 | {result['rho_v']:.1f} kg/m³ |
    | 空气侧总面积 | {result['area_external']:.2f} m² |
    | 管内总面积 | {result['area_internal']:.3f} m² |
    | 管内容积 | {result['volume_internal']*1000:.3f} L |
    | 空气质量流量 | {result['m_air']:.3f} kg/s |
    """)

# --- 分段详情表 ---
st.markdown("### 📋 逐排计算详情 (逆流换热)")
st.markdown(f"""
<div style='background:#e8f4f8;padding:8px 12px;border-radius:6px;margin:5px 0;font-size:13px;'>
<b>逆流换热器流向</b>: 空气 → 第1排 → 第2排 → ... → 第{num_segments}排 (空气冷却) &nbsp;|&nbsp;
制冷剂 ← 第{num_segments}排(进口,T_evap) → ... → 第1排(出口,T_evap+过热度) (制冷剂加热)
</div>
""", unsafe_allow_html=True)
seg_data = []
for s in segments:
    seg_data.append({
        '排号': s['segment'],
        '进风温度(°C)': round(s['T_air_in'], 2),
        '出风温度(°C)': round(s['T_air_out'], 2),
        '进风含湿量(g/kg)': round(s['W_in'] * 1000, 2),
        '出风含湿量(g/kg)': round(s['W_out'] * 1000, 2),
        '制冷剂温度(°C)': round(s['T_ref'], 2),
        '壁面温度(°C)': round(s.get('T_surface', 0), 2),
        '饱和温度(°C)': round(s.get('T_sat', 0), 2),
        '压力(kPa)': round(s.get('P_ref', 0) / 1000, 2),
        '段压降(Pa)': round(s.get('dP_seg', 0), 1),
        '湿工况': '是' if s['is_wet'] else '否',
        '换热量(W)': round(s['Q'], 1),
        '显热(W)': round(s['Q_sens'], 1),
        '潜热(W)': round(s['Q_lat'], 1),
    })
st.dataframe(seg_data, use_container_width=True, hide_index=True)

# --- 沿程温度分布与压降曲线 ---
st.markdown("### 📈 沿程温度分布与压降衰减曲线")
st.markdown(f"""
<div style='background:#fff3e0;padding:8px 12px;border-radius:6px;margin:5px 0;font-size:13px;'>
<b>制冷剂流向</b>: 第{num_segments}排(进口, P={result['p_evap_in']:.3f}MPa) → 第1排(出口, P={result['p_evap_out']:.3f}MPa) &nbsp;|&nbsp;
总压降: <b>{result['delta_P_total']:.2f} kPa</b> (摩擦{result['delta_P_friction']:.1f} + 加速{result['delta_P_accel']:.1f} + 弯头{result['delta_P_bend']:.1f}) &nbsp;|&nbsp;
饱和温度衰减: <b>{result['T_sat_drop']:.3f} °C</b>
</div>
""", unsafe_allow_html=True)

# 按制冷剂流向排列(段n→段1 = 进口→出口)
seg_order = list(range(num_segments - 1, -1, -1))
rows_ref = [num_segments - i for i in range(num_segments)]  # 沿制冷剂流向的排号
T_sat_arr = [segments[i].get('T_sat', result['T_evap']) for i in seg_order]
T_ref_arr = [segments[i]['T_ref'] for i in seg_order]
T_air_in_arr = [segments[i]['T_air_in'] for i in seg_order]
T_air_out_arr = [segments[i]['T_air_out'] for i in seg_order]
P_arr = [segments[i].get('P_ref', 0) / 1000 for i in seg_order]  # kPa

fig_prof = make_subplots(rows=1, cols=2, subplot_titles=("沿程温度分布", "沿程压力分布"),
                         horizontal_spacing=0.12)

# 温度分布图
fig_prof.add_trace(go.Scatter(x=rows_ref, y=T_sat_arr, name='制冷剂饱和温度 T_sat',
                               mode='lines+markers', line=dict(color='#d32f2f', width=2.5),
                               marker=dict(size=7)), row=1, col=1)
fig_prof.add_trace(go.Scatter(x=rows_ref, y=T_ref_arr, name='制冷剂温度 T_ref',
                               mode='lines+markers', line=dict(color='#f57c00', width=2, dash='dash'),
                               marker=dict(size=6)), row=1, col=1)
fig_prof.add_trace(go.Scatter(x=rows_ref, y=T_air_in_arr, name='空气进风温度',
                               mode='lines+markers', line=dict(color='#1565c0', width=2),
                               marker=dict(size=6)), row=1, col=1)
fig_prof.add_trace(go.Scatter(x=rows_ref, y=T_air_out_arr, name='空气出风温度',
                               mode='lines+markers', line=dict(color='#42a5f5', width=2, dash='dot'),
                               marker=dict(size=6)), row=1, col=1)

# 无压降基准线(恒定T_evap)
fig_prof.add_trace(go.Scatter(x=rows_ref, y=[result['T_evap']] * num_segments,
                               name='无压降基准(T_evap)', mode='lines',
                               line=dict(color='#9e9e9e', width=1.5, dash='dot'),
                               showlegend=True), row=1, col=1)

# 压力分布图
fig_prof.add_trace(go.Scatter(x=rows_ref, y=P_arr, name='蒸发压力 P',
                               mode='lines+markers', line=dict(color='#7b1fa2', width=2.5),
                               marker=dict(size=7), fill='tozeroy',
                               fillcolor='rgba(123,31,162,0.1)'), row=1, col=2)

fig_prof.update_xaxes(title_text="排号 (沿制冷剂流向: 进口→出口)", row=1, col=1)
fig_prof.update_xaxes(title_text="排号 (沿制冷剂流向: 进口→出口)", row=1, col=2)
fig_prof.update_yaxes(title_text="温度 (°C)", row=1, col=1)
fig_prof.update_yaxes(title_text="压力 (kPa)", row=1, col=2)

fig_prof.update_layout(height=400, margin=dict(l=50, r=20, t=40, b=20),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_prof, use_container_width=True)

st.divider()

# --- 标准合规性 ---
st.markdown("### 📜 标准合规性 (GB/T 47234-2026 数字孪生)")
st.markdown("""
本数字孪生模型符合以下标准要求:

| 标准编号 | 标准名称 | 符合条款 |
|----------|----------|----------|
| ASHRAE Fundamentals 2013 Ch.1 | 湿空气学 (Psychrometrics) | Hyland-Wexler 饱和蒸气压方程、含湿量/焓/露点计算 |
| ASHRAE Fundamentals 2013 Ch.4 | 传热学 (Heat Transfer) | 翅片效率 (Bessel/等效圆形法)、热阻网络 |
| ASHRAE Systems & Equipment 2012 Ch.22 | 冷却盘管 (Cooling Coils) | 湿工况 Lewis 关系、ε-NTU 方法 |
| ASHRAE Standard 33-2000 | 强制循环空气冷却盘管测试方法 | 盘管性能测试基准 |
| JB/T 7659.5-95 | 氟利昂制冷装置用翅片式换热器 | 表5 结构参数范围、名义工况 |
| GB/T 23130-2008 | 房间空调器用翅片管式换热器 | 附录B 空气侧换热量计算 |
| GB/T 47234-2026 | 数字孪生要求 | 5.3.7 几何模型/物理模型/性能仿真 |
""")

# --- 模型架构说明 ---
with st.expander("📖 模型架构与计算方法说明"):
    st.markdown("""
#### 数字孪生三层架构 (GB/T 47234-2026 5.3.7)

**1. 几何模型层 (Geometric Model)**
- 管参数: 外径/内径/壁厚/管距/排数/每排管数/分液路数
- 翅片参数: 节距/厚度/材料导热系数
- 自动计算: 迎风面积、管外表面积、翅片面积、总换热面积
- 参数范围严格遵循 JB/T 7659.5-95 表5

**2. 物理/机理模型层 (Physical/Mechanistic Model)**

*翅片效率* (ASHRAE Fundamentals Ch.4):
- 板式翅片等效圆形法: r₂_eq = √(r₁² + Pt·Pl/π)
- Harper-Brown 修正: η_f = tanh(m·Lc)/(m·Lc), m = √(2h_o/(k_fin·δf))

*管外空气侧换热* (Gray & Webb 1986):
- j = 0.14·Re_D^(-0.328)·(Pt/Pl)^(-0.502)·(Pt/Do)^0.031
- h_o = (j/Pr^(2/3))·G_c·cp

*管内沸腾换热* (Shah 1976 / Kandlikar 1990):
- Shah: h_tp = h_l·(1 + 3.8/Z^0.95), Z = ((1-x)/x)^0.8·(ρv/ρl)^0.5
- h_l: Dittus-Boelter: Nu = 0.023·Re^0.8·Pr^0.4
- 沿干度积分平均

*湿工况* (ASHRAE Systems & Equipment Ch.22, Threlkeld焓法):
- 壁面温度通过热阻网络与制冷剂温度耦合 (非固定值)
- 湿表面: G·(h_air - h_s(T_s)) = (T_s - T_ref)/R_ref_wall, 二分法求解T_s
- 干表面: Q = (T_air - T_ref)/(R_air + R_ref_wall)
- 自动判别干/湿工况 (壁面温度 vs 空气露点)
- T_s 随 T_ref 变化 → 过热段换热量随制冷剂温度升高而降低

**3. 性能仿真层 (Performance Simulation)**
- 逆流换热: 过热段在空气进口侧(第1排), 两相段在空气出口侧(末排) (最大化LMTD)
- 能量守恒迭代: Q_air = Q_refrigerant (严格守恒)
- 物理约束: 制冷剂出口温度 ≤ 空气进口温度 (热力学第二定律)
- 过热度由能量平衡计算输出 (非输入参数), 二分法求解
- 逐排计算: 分段数 = 排数, 每排壁温由热阻网络确定
- 制冷剂温度沿程分布由累积换热量计算 (非线性, 非线性性近似)
- 输出: 制冷量、过热度、SHR、COP、出风状态、除湿量、凝水量、壁面温度
- 敏感性分析: 参数扫描与可视化
- 焓湿图: ASHRAE 标准湿空气过程图

**制冷剂物性**: CoolProp (Helmholtz 状态方程) 高精度计算
- 支持 R22/R134a/R410A/R32/R290/R407C/R404A/R23 等
- 内置多项式关联式作为后备
""")
