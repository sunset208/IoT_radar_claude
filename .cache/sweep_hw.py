import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.config import load_config
from radar.sources.pluto import open_pluto, make_tx_waveform
from radar.hwcheck import analyse_raw
cfg=load_config(r"F:\Claude\ProjetS7\IoT_radar_v2\configs\lab.yaml")
sdr=open_pluto(cfg)
tx=make_tx_waveform(cfg.sdr.f_s,cfg.emission.f_offset,cfg.sdr.rx_buffer_size,cfg.emission.amplitude)
sdr.tx_cyclic_buffer=True; sdr.tx(tx)
for att in [-89,-10,0]:
    sdr.tx_hardwaregain_chan0=att
    for g in [30,50,65]:
        sdr.rx_hardwaregain_chan0=g
        for _ in range(3): sdr.rx()
        raw=np.asarray(sdr.rx(),complex)
        a=analyse_raw(raw,cfg)
        print(f"tx {att:4d} rx {g:3d}: tone {a['tone_dbfs']:6.1f}  noise {a['noise_dbfs_hz']:6.1f} dBFS/Hz  rms {a['rms_dbfs']:6.1f} peak {a['peak_dbfs']:6.1f} clip {a['clip_frac']:.4f}")
# interferers at rx 50, tx -10
sdr.tx_hardwaregain_chan0=-10; sdr.rx_hardwaregain_chan0=50
for _ in range(3): sdr.rx()
raw=np.concatenate([np.asarray(sdr.rx(),complex) for _ in range(5)])/2048
X=np.fft.fftshift(np.abs(np.fft.fft(raw*np.hanning(len(raw))))**2); f=np.fft.fftshift(np.fft.fftfreq(len(raw),1e-6))
Xd=10*np.log10(X/X.max()); 
# top peaks with 2 kHz separation
order=np.argsort(Xd)[::-1]; picked=[]
for i in order:
    if all(abs(f[i]-f[j])>2000 for j in picked): picked.append(i)
    if len(picked)>=8: break
print("raies principales (Hz, dB rel):", [(int(f[i]),round(Xd[i],1)) for i in picked])
# time-domain burstiness
e=np.abs(raw)**2; seg=e[:len(e)//1000*1000].reshape(-1,1000).mean(1); print("énergie par ms: min/med/max dB", np.round(10*np.log10([seg.min(),np.median(seg),seg.max()]),1))
sdr.tx_destroy_buffer()
