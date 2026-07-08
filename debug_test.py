"""Debug: print detailed segment info for problematic mass flows"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FinEvaporatorGeometry, FinEvaporatorDigitalTwin
from refrigerant import Refrigerant

geo = FinEvaporatorGeometry()
ref = Refrigerant('R410A')
dt = FinEvaporatorDigitalTwin(geo, ref)

base = dict(T_evap=5.0, x_in=0.25, T_air_in=27.0, RH_air_in=0.50,
            face_velocity=2.0, p_atm=101325.0, num_segments=10, boiling_method='shah')

for mf in [0.03, 0.10, 0.15]:
    print(f"\n{'='*80}")
    print(f"mass_flow = {mf} kg/s")
    print(f"{'='*80}")
    r = dt.solve(mass_flow=mf, **base)
    
    print(f"Q_total = {r['capacity']:.2f} kW, SH = {r['superheat']:.1f}°C")
    print(f"h_o = {r['h_o']:.1f}, h_i_tp = {r['h_i_tp']:.1f}, h_i_sh = {r['h_i_sh']:.1f}")
    print(f"mass_flux = {r['mass_flux']:.1f} kg/m²·s")
    print(f"Q_tp_max = {r['Q_tp_max']:.2f} kW")
    print(f"eta_o = {r['overall_efficiency']:.3f}")
    print(f"area_ext = {r['area_external']:.2f} m², area_int = {r['area_internal']:.3f} m²")
    print(f"m_air = {r['m_air']:.3f} kg/s")
    print()
    print(f"{'Seg':>3} {'T_air_in':>8} {'T_air_out':>9} {'T_ref':>6} {'T_s':>6} {'wet':>4} "
          f"{'Q(W)':>8} {'Q_sens':>8} {'Q_lat':>8} {'h_air':>8}")
    print("-" * 80)
    for s in r['segments']:
        print(f"{s['segment']:>3} {s['T_air_in']:>8.1f} {s['T_air_out']:>9.1f} "
              f"{s['T_ref']:>6.1f} {s['T_surface']:>6.1f} {'W' if s['is_wet'] else 'D':>4} "
              f"{s['Q']:>8.1f} {s['Q_sens']:>8.1f} {s['Q_lat']:>8.1f} "
              f"{s['h_air_in']:>8.1f}")
