import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.config import load_config
from radar.sources.pluto import open_pluto, make_tx_waveform
from radar.hwcheck import analyse_raw
cfg=load_config(r"F:\Claude\ProjetS7\IoT_radar_v2\configs\lab.yaml")
sdr=open_pluto(cfg); sdr.rx_hardwaregain_chan0=30; sdr.tx_hardwaregain_chan0=-10
tx=make_tx_waveform(cfg.sdr.f_s,cfg.emission.f_offset,cfg.sdr.rx_buffer_size,cfg.emission.amplitude)
sdr.tx_cyclic_buffer=True; sdr.tx(tx)
for fc in [433,600,868,915,1000,1200,1500,1700,1800,1900,2000,2200,2350,2600,3000,3300,3600]:
    sdr.rx_lo=int(fc*1e6); sdr.tx_lo=int(fc*1e6)
    for _ in range(3): sdr.rx()
    a=analyse_raw(np.asarray(sdr.rx(),complex),cfg)
    print(f"{fc:5d} MHz  fuite {a['tone_dbfs']:6.1f} dBFS  bruit {a['noise_dbfs_hz']:6.1f}")
sdr.tx_destroy_buffer()
