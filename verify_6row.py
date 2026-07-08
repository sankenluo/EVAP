"""验证6段排温分布修复效果"""
from model import FinEvaporatorGeometry, FinEvaporatorDigitalTwin
from refrigerant import Refrigerant

geo = FinEvaporatorGeometry(num_rows=6)
ref = Refrigerant('R410A')
dt = FinEvaporatorDigitalTwin(geo, ref)

# 用户原图: T_ref: 5->5->5->5->5->17 (第6排突升) - 错的
# 修复后期望: T_ref沿流向连续渐变

print("=== 6排 mf=0.020 SH=18.6 详细数据 ===")
r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
             face_velocity=2.0, mass_flow=0.020, num_segments=6)
print(f"Q_air={r['capacity']:.2f}kW, Q_ref={r['capacity_ref']:.2f}kW, err={r['energy_balance_error']:.1f}%")
print()
print(f"{'排':>3} {'T_air_in':>9} {'T_air_out':>9} {'T_ref':>7} {'T_surf':>7} {'湿':>4} {'Q(W)':>7} {'潜热(W)':>8}")
for s in r['segments']:
    w = '湿' if s['is_wet'] else '干'
    print(f"{s['segment']:>3} {s['T_air_in']:>9.1f} {s['T_air_out']:>9.1f} {s['T_ref']:>7.1f} {s['T_surface']:>7.1f} {w:>4} {s['Q']:>7.1f} {s['Q_lat']:>8.1f}")

print()
print("=== 6排 mf=0.05 SH=0 详细数据 ===")
r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
             face_velocity=2.0, mass_flow=0.05, num_segments=6)
print(f"Q_air={r['capacity']:.2f}kW, Q_ref={r['capacity_ref']:.2f}kW, err={r['energy_balance_error']:.1f}%")
print()
print(f"{'排':>3} {'T_air_in':>9} {'T_air_out':>9} {'T_ref':>7} {'T_surf':>7} {'湿':>4} {'Q(W)':>7} {'潜热(W)':>8}")
for s in r['segments']:
    w = '湿' if s['is_wet'] else '干'
    print(f"{s['segment']:>3} {s['T_air_in']:>9.1f} {s['T_air_out']:>9.1f} {s['T_ref']:>7.1f} {s['T_surface']:>7.1f} {w:>4} {s['Q']:>7.1f} {s['Q_lat']:>8.1f}")
