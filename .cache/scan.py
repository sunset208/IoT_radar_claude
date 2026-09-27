import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np
from radar.config import load_config
from radar.sources.pluto import open_pluto
cfg=load_config(r"F:\Claude\ProjetS7\IoT_radar_v2\configs\lab.yaml")
sdr=open_pluto(cfg); sdr.tx_hardwaregain_chan0=-89; sdr.rx_hardwaregain_chan0=50
res=[]
for fc in list(range(400,3800,100))+[2350,2450,2550,3150,3250,1250,1350,5]:
    if fc<325: continue
    sdr.rx_lo=int(fc*1e6)
    for _ in range(2): sdr.rx()
    e=[]
    for _ in range(5):
        r=np.asarray(sdr.rx(),complex)/2048; p=np.abs(r)**2
        seg=p[:len(p)//1000*1000].reshape(-1,1000).mean(1)
        e.append((np.median(seg),seg.max()))
    med=10*np.log10(np.median([a for a,b in e])); mx=10*np.log10(max(b for a,b in e))
    res.append((fc,med,mx))
for fc,med,mx in sorted(res): print(f"{fc:5d} MHz  médiane {med:6.1f}  max {mx:6.1f} dBFS")
