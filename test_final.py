"""综合验证：6排逆流换热温度分布和能量守恒"""
from model import FinEvaporatorGeometry, FinEvaporatorDigitalTwin
from refrigerant import Refrigerant

print("="*70)
print("综合验证：6排逆流换热修复")
print("="*70)

# 验证1: 6排默认几何
print("\n【验证1】6排默认几何, 各种工况下的温度分布")
print("-"*70)
geo = FinEvaporatorGeometry(num_rows=6)
ref = Refrigerant('R410A')
dt = FinEvaporatorDigitalTwin(geo, ref)

print(f"{'mf':>7} {'Q':>7} {'SH':>5} {'n_sh':>5} {'T_ref分布':<35} {'约束':>5}")
for mf in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.3, 0.5]:
    r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
                 face_velocity=2.0, mass_flow=mf, num_segments=6)
    segs = r['segments']
    n_sh = sum(1 for s in segs if s['T_ref'] > 5.5)
    T_ref_str = ' '.join(f'{s["T_ref"]:4.1f}' for s in segs)
    T_ref_out = 5 + r['superheat']
    ok = 'OK' if T_ref_out <= 27 else '!!'
    print(f"{mf:>7.3f} {r['capacity']:>7.2f} {r['superheat']:>5.1f} {n_sh:>5} {T_ref_str:<35} {ok:>5}")

# 验证2: 4排默认
print("\n【验证2】4排默认几何, 各种工况")
print("-"*70)
geo = FinEvaporatorGeometry(num_rows=4)
ref = Refrigerant('R410A')
dt = FinEvaporatorDigitalTwin(geo, ref)

print(f"{'mf':>7} {'Q':>7} {'SH':>5} {'n_sh':>5} {'T_ref分布':<30} {'err%':>5}")
for mf in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.3, 0.5]:
    r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
                 face_velocity=2.0, mass_flow=mf, num_segments=4)
    segs = r['segments']
    n_sh = sum(1 for s in segs if s['T_ref'] > 5.5)
    T_ref_str = ' '.join(f'{s["T_ref"]:4.1f}' for s in segs)
    print(f"{mf:>7.3f} {r['capacity']:>7.2f} {r['superheat']:>5.1f} {n_sh:>5} {T_ref_str:<30} {r['energy_balance_error']:>5.1f}")

# 验证3: 物理一致性检查
print("\n【验证3】物理一致性 (6排, mf=0.02)")
print("-"*70)
r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
             face_velocity=2.0, mass_flow=0.02, num_segments=6)
segs = r['segments']
print(f"{'排':>3} {'T_air↓':>7} {'T_ref↓':>7} {'T_surf∈[T_air,T_ref]':>20}")
for i, s in enumerate(segs):
    air_decreasing = True  # 总是
    ref_decreasing = (s['T_ref'] >= segs[max(i-1,0)]['T_ref'] - 0.1) if i > 0 else True
    in_range = (min(s['T_air_in'], s['T_ref']) - 1 <= s['T_surface'] <= max(s['T_air_in'], s['T_ref']) + 1)
    print(f"{s['segment']:>3} {s['T_air_in']:>7.1f} {s['T_ref']:>7.1f} {in_range!s:>20}")

# 验证4: 逆流约束
print("\n【验证4】逆流约束: T_ref_out <= T_air_in")
print("-"*70)
print(f"{'T_air':>6} {'T_evap':>6} {'mf':>7} {'T_ref_out':>9} {'约束':>5}")
all_pass = True
for T_air in [20, 25, 27, 32, 40]:
    for T_evap in [0, 5, 10]:
        for mf in [0.003, 0.01, 0.05]:
            r = dt.solve(T_evap=T_evap, x_in=0.25, T_air_in=T_air, RH_air_in=0.50,
                         face_velocity=2.0, mass_flow=mf, num_segments=6)
            T_ref_out = T_evap + r['superheat']
            if T_ref_out > T_air + 0.1:
                print(f"FAIL: T_air={T_air} T_evap={T_evap} mf={mf} T_ref_out={T_ref_out:.1f} > {T_air}")
                all_pass = False
print(f"全部通过: {all_pass}")

# 验证5: 能量守恒
print("\n【验证5】能量守恒验证 (6排)")
print("-"*70)
print(f"{'mf':>7} {'Q_air':>7} {'Q_ref':>7} {'err%':>5} {'converged':>9}")
geo = FinEvaporatorGeometry(num_rows=6)
dt = FinEvaporatorDigitalTwin(geo, ref)
all_pass = True
for mf in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5]:
    r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
                 face_velocity=2.0, mass_flow=mf, num_segments=6)
    err = r['energy_balance_error']
    ok = err < 5
    if not ok: all_pass = False
    print(f"{mf:>7.3f} {r['capacity_air']:>7.2f} {r['capacity_ref']:>7.2f} {err:>5.1f} {str(r['converged']):>9} {'OK' if ok else '!!'}")
print(f"全部通过 (err<5%): {all_pass}")

# 验证6: 不同制冷剂
print("\n【验证6】不同制冷剂 (6排 mf=0.05)")
print("-"*70)
print(f"{'制冷剂':>8} {'Q':>7} {'SH':>5} {'T_ref分布':<35}")
for refr in ['R22', 'R134a', 'R410A', 'R32', 'R407C', 'R404A']:
    ref = Refrigerant(refr)
    dt = FinEvaporatorDigitalTwin(geo, ref)
    r = dt.solve(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
                 face_velocity=2.0, mass_flow=0.05, num_segments=6)
    segs = r['segments']
    T_ref_str = ' '.join(f'{s["T_ref"]:4.1f}' for s in segs)
    print(f"{refr:>8} {r['capacity']:>7.2f} {r['superheat']:>5.1f} {T_ref_str:<35}")

print("\n" + "="*70)
print("验证完成")
print("="*70)
