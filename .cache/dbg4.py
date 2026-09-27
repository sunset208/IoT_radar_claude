import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.config import load_config
from radar.scene import SCENARIOS, random_scene_params
from radar.sources.simulation import simulate_slow
from radar.offline import window_features
from radar.dsp.detector import Detector
cfg=load_config(); fs=50.0
rng=np.random.default_rng(1)
target=("breathing",5)
for sc in SCENARIOS:
    for i in range(6):
        p=random_scene_params(rng,cfg.sdr.wavelength,sc)
        x,tr=simulate_slow(sc,120,fs,cfg.sdr.wavelength,rng,params=p)
        if (sc,i)==target or (sc,i)==("empty",5):
            print(sc,i,p)
            F=window_features(x,fs,cfg); det=Detector(cfg.detector)
            for k,f in enumerate(F[::8]):
                d=det.update(f)
                print(f"  t={f.t:5.1f} snr={f.snr_db:5.1f} conc={f.concentration:.2f} acf={f.acf:.2f} mot={f.motion:5.1f} band={f.band_db:5.1f} bpm={f.breath_bpm:5.1f} arc={f.arc_rad} drift={f.drift_db:5.1f}")
