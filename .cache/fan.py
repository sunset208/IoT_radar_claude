import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.config import load_config
from radar.sources.simulation import simulate_slow
from radar.offline import window_features
from radar.dsp.detector import Detector
cfg=load_config(); fs=50.0; rng=np.random.default_rng(2)
for sc in ["fan","breathing","fan","breathing"]:
    x,_=simulate_slow(sc,180,fs,cfg.sdr.wavelength,rng,snr_db=25)
    det=Detector(cfg.detector); w=0; b=0
    for f in window_features(x,fs,cfg):
        d=det.update(f); b+=d.state=="RESPIRATION"; w+=bool(d.warnings)
    print(sc, "windows breathing", b, "with mech warning", w)
