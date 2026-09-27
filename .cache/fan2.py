import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.config import load_config
from radar.sources.simulation import simulate_slow
from radar.offline import window_features
cfg=load_config(); fs=50.0; rng=np.random.default_rng(2)
for sc in ["fan","breathing"]*3:
    x,_=simulate_slow(sc,180,fs,cfg.sdr.wavelength,rng,snr_db=25)
    r=np.array([f.breath_bpm for f in window_features(x,fs,cfg)])
    cv=[np.std(r[i:i+120])/np.mean(r[i:i+120]) for i in range(0,len(r)-120,20)]
    print(sc, "median bpm %.1f"%np.median(r), "CV(60s):", np.round(cv,3))
