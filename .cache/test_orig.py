import sys, time, copy, itertools, logging
sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar-main\IoT_radar-main")
import numpy as np, yaml
import MicroDopplerDetection.main as M
import MicroDopplerDetection.pipeline.acquisition as A
logging.basicConfig(level=logging.WARNING)
cfg = yaml.safe_load(open(r"F:\Claude\ProjetS7\IoT_radar-main\IoT_radar-main\MicroDopplerDetection\configs\config.yaml", encoding="utf-8"))

def realistic(f_c, f_s, buffer_size, fv, D_mm, snr_dB, f_offset=0.0, clutter_amplitude=100.0, breathing=True):
    lam = 3e8/f_c; m = 4*np.pi*D_mm*1e-3/lam
    rng = np.random.default_rng(1); ns = np.sqrt(10**(-snr_dB/10)/2); k=0
    while True:
        t=(np.arange(buffer_size)+k)/f_s
        car = 2*np.pi*f_offset*t
        sig = np.exp(1j*(car - (m*np.sin(2*np.pi*fv*t) if breathing else 0)))
        clut = clutter_amplitude*np.exp(1j*(car+0.7))   # static echo is at f_offset too
        n = ns*(rng.standard_normal(buffer_size)+1j*rng.standard_normal(buffer_size))
        k+=buffer_size; yield (clut+sig+n).astype(np.complex64)

for name, breathing in [("breathing", True), ("empty", False)]:
    A_stream = lambda **kw: realistic(breathing=breathing, **kw)
    M.stream_simulation = A_stream
    g = M._streaming_frame_generator(cfg, simulation=True)
    t0=time.time(); sc=[]; pv=[]; ac=[]
    for fr in itertools.islice(g, 150):
        sc.append(fr["score_presence"]); pv.append(fr["p_value_f"]); ac.append(fr["acf_peak"])
    dt=time.time()-t0
    print(f"{name}: 150 frames in {dt:.1f}s (signal time ~{(150+24)*819/2000:.0f}s)  score mean={np.mean(sc[-100:]):.2f}  frac p<0.01={np.mean(np.array(pv[-100:])<0.01):.2f}  acf={np.mean(ac[-100:]):.2f}")
