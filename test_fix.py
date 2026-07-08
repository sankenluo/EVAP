"""
测试修复后的蒸发器模型: 验证 mass_flow 变化时 Q 和 superheat 正确响应
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FinEvaporatorGeometry, FinEvaporatorDigitalTwin
from refrigerant import Refrigerant

# 构建模型
geo = FinEvaporatorGeometry()
ref = Refrigerant('R410A')
dt = FinEvaporatorDigitalTwin(geo, ref)

print("=" * 90)
print("测试1: 制冷剂流量敏感性 (固定其他参数)")
print("=" * 90)
print(f"{'流量(kg/s)':>10} {'制冷量(kW)':>10} {'过热度(C)':>10} {'出风温度(C)':>12} "
      f"{'Q_ref(kW)':>10} {'Q_air(kW)':>10} {'偏差(%)':>8} {'迭代':>6} {'收敛':>6}")
print("-" * 90)

base_params = dict(
    T_evap=5.0, x_in=0.25,
    T_air_in=27.0, RH_air_in=0.50,
    face_velocity=2.0,
    p_atm=101325.0, num_segments=10,
    boiling_method='shah'
)

for mf in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    try:
        r = dt.solve(mass_flow=mf, **base_params)
        print(f"{mf:>10.3f} {r['capacity']:>10.2f} {r['superheat']:>10.1f} "
              f"{r['air_out'].T_db:>12.1f} {r['capacity_ref']:>10.2f} "
              f"{r['capacity_air']:>10.2f} {r['energy_balance_error']:>8.1f} "
              f"{r['iterations']:>6d} {'Y' if r['converged'] else 'N':>6}")
    except Exception as e:
        print(f"{mf:>10.3f}  ERROR: {e}")

print()
print("=" * 90)
print("测试2: 不同蒸发温度下的流量敏感性")
print("=" * 90)

for T_evap in [0.0, 5.0, 10.0]:
    print(f"\n--- 蒸发温度 = {T_evap}°C ---")
    print(f"{'流量(kg/s)':>10} {'制冷量(kW)':>10} {'过热度(C)':>10} {'出风温度(C)':>12} {'偏差(%)':>8}")
    print("-" * 60)
    for mf in [0.03, 0.05, 0.10, 0.15, 0.20]:
        try:
            r = dt.solve(mass_flow=mf, T_evap=T_evap, **{k:v for k,v in base_params.items() if k != 'T_evap'})
            print(f"{mf:>10.3f} {r['capacity']:>10.2f} {r['superheat']:>10.1f} "
                  f"{r['air_out'].T_db:>12.1f} {r['energy_balance_error']:>8.1f}")
        except Exception as e:
            print(f"{mf:>10.3f}  ERROR: {e}")

print()
print("=" * 90)
print("测试3: 物理一致性验证")
print("=" * 90)

# 验证: 低流量 → 高过热度 + 低冷量; 高流量 → 低/无过热度 + 高冷量
r_low = dt.solve(mass_flow=0.03, **base_params)
r_high = dt.solve(mass_flow=0.30, **base_params)

print(f"低流量 (0.03 kg/s): Q = {r_low['capacity']:.2f} kW, SH = {r_low['superheat']:.1f}°C")
print(f"高流量 (0.30 kg/s): Q = {r_high['capacity']:.2f} kW, SH = {r_high['superheat']:.1f}°C")
print()

checks = []
# 1. 低流量过热度应大于高流量过热度
if r_low['superheat'] > r_high['superheat']:
    print("✅ 低流量过热度 > 高流量过热度")
    checks.append(True)
else:
    print("❌ 低流量过热度应大于高流量过热度")
    checks.append(False)

# 2. 低流量冷量应小于高流量冷量 (或至少有显著变化)
if abs(r_low['capacity'] - r_high['capacity']) > 0.5:
    print(f"✅ 冷量随流量变化: ΔQ = {abs(r_low['capacity'] - r_high['capacity']):.2f} kW")
    checks.append(True)
else:
    print(f"❌ 冷量几乎不变: ΔQ = {abs(r_low['capacity'] - r_high['capacity']):.2f} kW")
    checks.append(False)

# 3. 能量平衡偏差
for label, r in [("低流量", r_low), ("高流量", r_high)]:
    if r['energy_balance_error'] < 10:
        print(f"✅ {label}能量平衡偏差: {r['energy_balance_error']:.1f}%")
        checks.append(True)
    else:
        print(f"❌ {label}能量平衡偏差过大: {r['energy_balance_error']:.1f}%")
        checks.append(False)

print()
if all(checks):
    print("🎉 所有物理一致性检查通过!")
else:
    print(f"⚠️ {checks.count(False)}/{len(checks)} 项检查未通过")
