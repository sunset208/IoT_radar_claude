import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.dsp.vitals import VitalsAnalyzer
fs=50.0; lam=3e8/3.5e9; rng=np.random.default_rng(1)
for sc in ["empty","breathing"]:
  for drift in [0.0,0.003]:
    x,tr=simulate_slow(sc, 60, fs, lam, rng, snr_db=20, lo_drift_rad=drift)
    va=VitalsAnalyzer(fs, lam)
    out=[]
    for i in range(0, len(x)-1000, 250):
        f,_=va.analyze(x[i:i+1000], want_display=False); out.append((round(f.drift_db,1), round(f.snr_db,1)))
    print(sc, drift, out)
