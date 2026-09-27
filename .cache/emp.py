import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.dsp.vitals import VitalsAnalyzer
fs=50.0; lam=3e8/3.5e9; rng=np.random.default_rng(1)
for drift in [0.0, 0.003, 0.01]:
  for snr in [-10,0,10,20,30]:
    F=[]
    for rep in range(4):
        va=VitalsAnalyzer(fs, lam); x,tr=simulate_slow("empty", 60, fs, lam, rng, snr_db=snr, lo_drift_rad=drift)
        for i in range(0, len(x)-1000, 250):
            f,_=va.analyze(x[i:i+1000], want_display=False); F.append((f.snr_db,f.periodicity,f.concentration,f.acf,f.breath_hz*60))
    F=np.array(F)
    print(f"drift {drift} snr {snr:3d}: snr_db med {np.median(F[:,0]):5.1f} p95 {np.percentile(F[:,0],95):5.1f}  per med {np.median(F[:,1]):.2f} p95 {np.percentile(F[:,1],95):.2f}  conc {np.median(F[:,2]):.2f} acf {np.median(F[:,3]):.2f} bpm {np.median(F[:,4]):.1f}")
print("--- breathing with drift")
for drift in [0.0, 0.003, 0.01]:
  for snr in [-20,-10,0,20]:
    F=[]
    for rep in range(6):
        va=VitalsAnalyzer(fs, lam); x,tr=simulate_slow("breathing", 60, fs, lam, rng, snr_db=snr, lo_drift_rad=drift, breath_depth_mm=rng.uniform(1,8))
        for i in range(0, len(x)-1000, 250):
            f,_=va.analyze(x[i:i+1000], want_display=False); F.append((f.snr_db,f.periodicity,f.drift_db))
    F=np.array(F,dtype=float)
    print(f"drift {drift} snr {snr:3d}: snr_db med {np.median(F[:,0]):5.1f} p10 {np.percentile(F[:,0],10):5.1f}  per med {np.median(F[:,1]):.2f} p10 {np.percentile(F[:,1],10):.2f} drift_db {np.median(F[:,2]):.1f}")
