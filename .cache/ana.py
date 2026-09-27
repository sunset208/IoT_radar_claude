import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from scipy import signal
from radar.recorder import load_recording
from radar.config import load_config
from radar.offline import window_features
p=sys.argv[1]; r=load_recording(p); x=r.slow.astype(complex); fs=r.fs_slow
cfg=load_config(r"F:\Claude\ProjetS7\IoT_radar_v2\configs\lab.yaml")
print("dur", r.duration_s, "gaps", r.gaps, "mean dBFS", 20*np.log10(abs(x.mean())))
c=x.mean(); z=(x-c)*np.conj(c/abs(c))
print("std rad/tan (rel to |c|):", np.std(z.real)/abs(c), np.std(z.imag)/abs(c))
for name,v in [("rad",z.real),("tan",z.imag)]:
    f,P=signal.welch(v-v.mean(),fs,nperseg=1500)
    idx=[np.argmin(abs(f-q)) for q in [0.05,0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.7,1,1.5,2,3,5,8,12,20]]
    print(name," ".join(f"{f[i]:.2f}:{10*np.log10(P[i]+1e-30):.0f}" for i in idx))
F=window_features(x,fs,cfg)
for f in F[::10]:
    print(f"t={f.t:5.1f} snr={f.snr_db:5.1f} bpm={f.breath_bpm:5.1f} conc={f.concentration:.2f} acf={f.acf:.2f} mot={f.motion:5.1f} band={f.band_db:5.1f} drift={f.drift_db:5.1f} arc={f.arc_rad} mm={f.disp_mm_pp}")
