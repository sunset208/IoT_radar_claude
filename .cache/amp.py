import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.scene import Scene, SceneParams, Breathing
from radar.dsp.vitals import VitalsAnalyzer, fit_circle
fs=50.0; lam=3e8/3.5e9; rng=np.random.default_rng(3)
va=VitalsAnalyzer(fs, lam)
for depth in [2,6,10]:
  for snr in [15,30]:
    x,tr=simulate_slow("breathing", 60, fs, lam, rng, snr_db=snr, breath_depth_mm=depth, heart_depth_mm=0.0, lo_drift_rad=0.0)
    r=[]
    for i in range(0,len(x)-1000,250):
        f,_=va.analyze(x[i:i+1000],want_display=False); r.append(f.disp_mm_pp if f.disp_mm_pp else np.nan)
    print(depth, snr, np.round(r,1))
# true displacement pp of generator
b=Breathing(np.random.default_rng(0),16,6e-3); t=np.arange(0,60,0.02); d=b(t)*1e3
print("true pp 2-98:", np.percentile(d,98)-np.percentile(d,2))
