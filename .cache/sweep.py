import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.dsp.vitals import VitalsAnalyzer
fs=50.0; lam=3e8/3.5e9
data=[]
rng=np.random.default_rng(5)
for sc in ["empty","breathing"]:
  for drift in [0.0,0.003,0.01,0.03]:
    for snr in ([10,20,30] if sc=="empty" else [-5,0,5]):
      for rep in range(5):
        x,_=simulate_slow(sc,60,fs,lam,rng,snr_db=snr,lo_drift_rad=drift,breath_depth_mm=rng.uniform(1,8),breath_rate_bpm=rng.uniform(8,30))
        data.append((sc,drift,snr,x))
for half,q in [(0.15,40),(0.2,35),(0.1,30),(0.15,50),(0.2,50)]:
    E=[];B=[]
    for sc,drift,snr,x in data:
        va=VitalsAnalyzer(fs,lam); va.floor_half_hz=half; va.floor_q=q
        for i in range(0,len(x)-1000,250):
            f,_=va.analyze(x[i:i+1000],want_display=False)
            (E if sc=="empty" else B).append(f.snr_db)
    E=np.array(E);B=np.array(B)
    thr=np.percentile(E,99)
    print(f"half {half} q {q}: empty p99 {thr:5.1f} max {E.max():5.1f} | breathing(low snr) Pd@p99 {np.mean(B>=thr):.2f} med {np.median(B):.1f}")
