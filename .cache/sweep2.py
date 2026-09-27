import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.dsp.vitals import VitalsAnalyzer
from radar.recorder import load_recording
fs=50.0; lam=3e8/3.5e9
data=[]; rng=np.random.default_rng(5)
for sc in ["empty","breathing"]:
  for drift in [0.0,0.003,0.01,0.03]:
    for snr in ([10,20,30] if sc=="empty" else [-5,0,5]):
      for rep in range(5):
        x,_=simulate_slow(sc,60,fs,lam,rng,snr_db=snr,lo_drift_rad=drift,breath_depth_mm=rng.uniform(1,8),breath_rate_bpm=rng.uniform(8,30))
        data.append((sc,x))
real=load_recording(r"F:\Claude\ProjetS7\IoT_radar_v2\data\recordings\20260927_022017_resp_rythme15_50cm_1800.npz").slow
realH0=load_recording(r"F:\Claude\ProjetS7\IoT_radar_v2\data\recordings\20260927_021455_resp_apnee_test_50cm_ram.npz").slow[int(12*fs):int(30*fs)+1000]
for guard,train,q in [(0.12,0.25,50),(0.12,0.4,50),(0.15,0.3,50),(0.12,0.25,60),(0.1,0.25,50)]:
    E=[];B=[]
    for sc,x in data:
        va=VitalsAnalyzer(fs,lam); va.floor_guard_hz=guard; va.floor_train_hz=train; va.floor_q=q
        for i in range(0,len(x)-1000,250):
            f,_=va.analyze(x[i:i+1000],want_display=False); (E if sc=="empty" else B).append(f.snr_db)
    E=np.array(E);B=np.array(B); thr=np.percentile(E,99)
    va=VitalsAnalyzer(fs,3e8/1.8e9); va.floor_guard_hz=guard; va.floor_train_hz=train; va.floor_q=q
    R=[va.analyze(real[i:i+1000],want_display=False)[0] for i in range(0,len(real)-1000,125)]
    rs=np.array([f.snr_db for f in R]); rb=np.array([f.breath_bpm for f in R])
    print(f"g{guard} tr{train} q{q}: sim H0 p99 {thr:5.1f} max {E.max():5.1f} | sim Pd@p99 {np.mean(B>=thr):.2f} | réel snr med {np.median(rs):5.1f} min {rs.min():5.1f} frac>=p99 {np.mean(rs>=thr):.2f} bpm med {np.median(rb):.1f}")
