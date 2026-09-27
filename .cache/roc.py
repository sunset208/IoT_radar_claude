import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.dsp.vitals import VitalsAnalyzer
fs=50.0; lam=3e8/3.5e9; va=VitalsAnalyzer(fs, lam); rng=np.random.default_rng(1)
rows=[]
for rep in range(6):
  for sc in ["empty","breathing","motion","walker","fan","breathing_motion"]:
    for snr in [0,5,10,15,20,30]:
        x,tr=simulate_slow(sc, 60, fs, lam, rng, snr_db=snr, breath_rate_bpm=rng.uniform(8,30), breath_depth_mm=rng.uniform(2,10))
        for i in range(0, len(x)-1000, 500):
            f,_=va.analyze(x[i:i+1000], want_display=False)
            rows.append((sc,snr,f.snr_db,f.periodicity,f.motion,f.concentration,f.acf))
import collections
sc=np.array([r[0] for r in rows]); snr=np.array([r[1] for r in rows]); S=np.array([r[2:] for r in rows])
for thr in [10,12,15,18]:
  for pt in [0.4,0.5,0.6]:
    pos=(S[:,0]>=thr)&(S[:,1]>=pt)&(S[:,2]<6)
    line=f"snr>={thr:2d} per>={pt}: "
    for s in ["empty","walker","motion","fan","breathing","breathing_motion"]:
        m=sc==s; line+=f"{s[:6]} {pos[m].mean():.2f} "
    line+=" | breathing by snr: "+" ".join(f"{v}:{pos[(sc=='breathing')&(snr==v)].mean():.2f}" for v in [0,5,10,15,20,30])
    print(line)
