import sys, time; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.sources.simulation import simulate_slow
from radar.dsp.vitals import VitalsAnalyzer
fs=50.0; lam=3e8/3.5e9
va=VitalsAnalyzer(fs, lam)
rng=np.random.default_rng(0)
t0=time.time()
for sc in ["empty","breathing","apnea","motion","breathing_motion","walker","fan"]:
    for snr in [25, 10]:
        x,tr=simulate_slow(sc, 120, fs, lam, rng, snr_db=snr)
        F=[]
        for i in range(0, len(x)-1000, 250):
            f,_=va.analyze(x[i:i+1000], want_display=False); F.append(f)
        a=lambda k: np.array([getattr(f,k) if getattr(f,k) is not None else np.nan for f in F])
        print(f"{sc:17s} snr{snr:3d}: snr_db {np.median(a('snr_db')):5.1f} [{np.min(a('snr_db')):5.1f},{np.max(a('snr_db')):5.1f}]  conc {np.median(a('concentration')):.2f}  acf {np.median(a('acf')):.2f}  motion {np.median(a('motion')):5.1f}/{np.max(a('motion')):6.1f}  band {np.median(a('band_db')):5.1f}  bpm {np.nanmedian(a('breath_hz'))*60:5.1f}  arc {np.nanmedian(a('arc_rad')) if np.any(~np.isnan(a('arc_rad'))) else float('nan'):.2f} mm {np.nanmedian(a('disp_mm_pp')) if np.any(~np.isnan(a('disp_mm_pp'))) else float('nan'):.1f} hr {np.nanmedian(a('heart_hz'))*60 if np.any(~np.isnan(a('heart_hz'))) else float('nan'):.0f}")
print("time", time.time()-t0)
