import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from scipy import signal
from radar.sources.simulation import simulate_slow
fs=50.0; lam=3e8/3.5e9; rng=np.random.default_rng(1)
x,tr=simulate_slow("empty", 120, fs, lam, rng, snr_db=20, lo_drift_rad=0.003)
c=x.mean(); u=c/abs(c)
z=(x-c)*np.conj(u)
for name,v in [("rad",z.real),("tan",z.imag)]:
    f,P=signal.welch(v,fs,nperseg=2000)
    print(name, " ".join(f"{fr:.2f}:{10*np.log10(p):.0f}" for fr,p in zip(f[[1,2,4,8,12,20,40,100,400]],P[[1,2,4,8,12,20,40,100,400]])))
print("angle(x) std", np.std(np.unwrap(np.angle(x))))
