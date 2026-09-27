import sys; sys.path.insert(0, r"F:\Claude\ProjetS7\IoT_radar_v2")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
from radar.recorder import load_recording
r=load_recording(sys.argv[1]); x=r.slow.astype(complex); fs=r.fs_slow; t=np.arange(len(x))/fs
c=x.mean(); z=(x-c)*np.conj(c/abs(c))
lam=3e8/float(sys.argv[3]) if len(sys.argv)>3 else 3e8/1.8e9
fig,ax=plt.subplots(4,1,figsize=(12,11))
sos=signal.butter(2,[0.05,3],btype='band',fs=fs,output='sos')
ax[0].plot(t,signal.sosfiltfilt(sos,z.real),lw=.7,label='radial');ax[0].plot(t,signal.sosfiltfilt(sos,z.imag),lw=.7,label='tangentiel');ax[0].legend();ax[0].set_title('composantes 0.05–3 Hz');ax[0].set_xlabel('s')
ph=np.unwrap(np.angle(x)); ax[1].plot(t,(ph-ph.mean())*lam/(4*np.pi)*1e3,lw=.7);ax[1].set_title('phase brute → mm (déplacement équivalent, clutter inclus)')
ax[2].plot(x.real,x.imag,'.',ms=1);ax[2].set_aspect('equal');ax[2].set_title('IQ')
f,tt,S=signal.spectrogram(z.real+1j*z.imag,fs,nperseg=500,noverlap=450,return_onesided=False)
m=(np.abs(f)<3); ax[3].pcolormesh(tt,np.fft.fftshift(f)[np.abs(np.fft.fftshift(f))<3],10*np.log10(np.fft.fftshift(S,axes=0)[np.abs(np.fft.fftshift(f))<3]+1e-20),shading='auto');ax[3].set_title('spectrogramme complexe (fenêtre 10 s)')
plt.tight_layout();plt.savefig(sys.argv[2],dpi=80)
