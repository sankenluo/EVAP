"""
Comprehensive validation: mass_flow sensitivity, superheat transition, energy conservation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FinEvaporatorGeometry, FinEvaporatorDigitalTwin
from refrigerant import Refrigerant

geo = FinEvaporatorGeometry()
ref = Refrigerant('R410A')
dt = FinEvaporatorDigitalTwin(geo, ref)

base = dict(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
            face_velocity=2.0, p_atm=101325.0, num_segments=10, boiling_method='shah')

print("=" * 100)
print("测试A: 宽流量范围扫描 (0.005 ~ 0.50 kg/s) — 观察过热度转变")
print("=" * 100)
print(f"{'流量(kg/s)':>10} {'制冷量(kW)':>10} {'过热度(C)':>10} {'出风温度(C)':>12} "
      f"{'Q_tp_max(kW)':>13} {'x_out':>6} {'Q_ref(kW)':>10} {'Q_air(kW)':>10} {'偏差(%)':>8}")
print("-" * 100)

results = []
for mf in [0.005, 0.008, 0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.050, 0.080, 0.100, 0.150, 0.200, 0.300, 0.500]:
    r = dt.solve(mass_flow=mf, **base)
    results.append((mf, r))
    print(f"{mf:>10.4f} {r['capacity']:>10.2f} {r['superheat']:>10.1f} "
          f"{r['air_out'].T_db:>12.1f} {r['Q_tp_max']:>13.2f} {r['x_out']:>6.2f} "
          f"{r['capacity_ref']:>10.2f} {r['capacity_air']:>10.2f} {r['energy_balance_error']:>8.1f}")

print()
print("=" * 100)
print("测试B: 物理一致性验证")
print("=" * 100)

# 1. Q单调递增 (低流量可能有过热导致Q略降, 允许)
caps = [r['capacity'] for _, r in results]
monotonic = all(caps[i] <= caps[i+1] + 0.5 for i in range(len(caps)-1))
print(f"{'✅' if monotonic else '❌'} Q随流量递增: {caps[0]:.2f} → {caps[-1]:.2f} kW")

# 2. 低流量有过热, 高流量无过热
sh_low = results[0][1]['superheat']  # 最低流量
sh_high = results[-1][1]['superheat']  # 最高流量
if sh_low > 0:
    print(f"✅ 极低流量({results[0][0]} kg/s)出现过热: SH = {sh_low:.1f}°C")
else:
    print(f"⚠️ 极低流量({results[0][0]} kg/s)无过热 (可能蒸发器面积不足)")

if sh_high == 0:
    print(f"✅ 高流量({results[-1][0]} kg/s)无过热: 制冷剂未完全蒸发")

# 3. 能量守恒
all_balanced = all(r['energy_balance_error'] < 5 for _, r in results)
print(f"{'✅' if all_balanced else '❌'} 能量守恒: 所有工况偏差 < 5%")

# 4. Q_air ≈ Q_ref
max_diff = max(abs(r['capacity_ref'] - r['capacity_air']) / max(r['capacity'], 0.001) * 100 
               for _, r in results if r['capacity'] > 0.1)
print(f"{'✅' if max_diff < 10 else '❌'} 制冷剂侧 ≈ 空气侧: 最大偏差 {max_diff:.1f}%")

# 5. 出风温度低于进风温度
all_cooled = all(r['air_out'].T_db < 27.0 for _, r in results)
print(f"{'✅' if all_cooled else '❌'} 出风温度 < 进风温度 (27°C)")

print()
print("=" * 100)
print("测试C: 不同工况下的过热度转变 (T_evap=0°C)")
print("=" * 100)
base2 = base.copy()
base2['T_evap'] = 0.0
print(f"{'流量(kg/s)':>10} {'制冷量(kW)':>10} {'过热度(C)':>10} {'Q_tp_max(kW)':>13} {'x_out':>6}")
print("-" * 60)
for mf in [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]:
    r = dt.solve(mass_flow=mf, **base2)
    print(f"{mf:>10.3f} {r['capacity']:>10.2f} {r['superheat']:>10.1f} "
          f"{r['Q_tp_max']:>13.2f} {r['x_out']:>6.2f}")

print()
print("=" * 100)
print("测试D: 进口干度影响 (T_evap=5°C, mass_flow=0.05)")
print("=" * 100)
base3 = base.copy()
base3['mass_flow'] = 0.05
print(f"{'进口干度':>8} {'制冷量(kW)':>10} {'过热度(C)':>10} {'x_out':>6} {'Q_tp_max(kW)':>13}")
print("-" * 60)
for x_in in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    r = dt.solve(x_in=x_in, **{k:v for k,v in base3.items() if k != 'x_in'})
    print(f"{x_in:>8.2f} {r['capacity']:>10.2f} {r['superheat']:>10.1f} "
          f"{r['x_out']:>6.2f} {r['Q_tp_max']:>13.2f}")
