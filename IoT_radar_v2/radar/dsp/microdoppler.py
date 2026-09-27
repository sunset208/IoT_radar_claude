"""Micro-Doppler : analyse temps-fréquence du signal complexe de la cible.

Phase et Doppler portent la même information (f_D = (1/2π)·dφ/dt), mais ne se
traitent pas pareil :

* **phase** : on démodule le déplacement d(t) (projection linéaire ou
  arc-tangente) puis on cherche la raie respiratoire dans son spectre —
  c'est le détecteur principal (:mod:`radar.dsp.vitals`) ;
* **micro-Doppler** : spectrogramme *signé* du signal complexe débarrassé du
  clutter statique ; la vitesse radiale se lit comme un décalage en
  fréquence (centroïde Doppler), l'agitation comme un étalement.

Régimes (modulation de phase d'indice β = 4π·A/λ, règle de Carson :
bande ≈ 2(β+1)·f_resp) :

* β ≪ 1 (respiration à 1.8 GHz : β ≈ 0.4 pour 5 mm, vitesse max ~4 mm/s →
  Doppler ~0.05 Hz, bien plus petit que le rythme lui-même) : le spectre est
  une porteuse + deux raies à ±f_resp ; le spectrogramme ne « voit » pas de
  vitesse → le micro-Doppler n'apporte rien à la détection ;
* β ≫ 1 (mouvements du corps, marche, ventilateur ; respiration à 60 GHz,
  β ≈ 12) : la vitesse instantanée est résolue → signatures micro-Doppler
  (utile pour *reconnaître* le type de mouvement).

Ce module fournit le spectrogramme signé (affichage), le centroïde et
l'étalement Doppler, et la détection de périodicité sur le centroïde
(comparaison « phase vs micro-Doppler », rapport §9.11).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

LN2 = math.log(2.0)


def remove_clutter(x: np.ndarray, fs: float, hp_hz: float = 0.06) -> np.ndarray:
    """Retire l'écho statique (moyenne + tendance linéaire + passe-haut doux)."""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x), dtype=np.float64)
    n -= n.mean()
    xm = x - x.mean()
    xm = xm - (np.dot(n, xm) / np.dot(n, n)) * n
    sos = signal.butter(2, hp_hz, btype="high", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, xm)


def spectrogram(z: np.ndarray, fs: float, win_s: float = 2.0, hop_s: float = 0.1,
                nfft: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectrogramme signé (fréquences négatives = cible qui s'éloigne).

    Renvoie (t centres des trames, f croissantes, S[t, f] puissance).
    """
    nw = max(8, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    w = signal.windows.hann(nw, sym=False)
    nfft = max(nfft, nw)
    starts = np.arange(0, len(z) - nw + 1, hop)
    if starts.size == 0:
        return np.zeros(0), np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)), np.zeros((0, nfft))
    frames = np.stack([z[s:s + nw] * w for s in starts])
    S = np.abs(np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1)) ** 2 / np.sum(w * w)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    t = (starts + nw / 2) / fs
    return t, f, S


def centroid_spread(S: np.ndarray, f: np.ndarray, fmax: float = 5.0,
                    noise_band: tuple[float, float] = (8.0, 12.0)) -> tuple[np.ndarray, np.ndarray]:
    """Centroïde Doppler (Hz, signé) et étalement (Hz) par trame, plancher retiré."""
    af = np.abs(f)
    nb = (af >= noise_band[0]) & (af <= min(noise_band[1], af.max()))
    floor = np.median(S[:, nb], axis=1, keepdims=True) if nb.any() else 0.0
    m = af <= fmax
    P = np.maximum(S[:, m] - floor, 0.0)
    fm = f[m]
    tot = P.sum(axis=1) + 1e-30
    c = (P * fm).sum(axis=1) / tot
    spread = np.sqrt(np.maximum((P * (fm - c[:, None]) ** 2).sum(axis=1) / tot, 0.0))
    return c, spread


def line_snr(s: np.ndarray, fs: float, band=(0.1, 0.8), noise_band: tuple[float, float] | None = None,
             guard_hz: float = 0.10, train_hz: float = 0.25, nfft: int = 8192) -> tuple[float, float]:
    """SNR (dB) du plus fort maximum local dans *band*, rapporté à un plancher
    CFAR en anneau (même principe que :class:`radar.dsp.vitals.VitalsAnalyzer`).
    Renvoie (snr_db, fréquence du pic Hz)."""
    s = np.asarray(s, dtype=np.float64)
    s = s - s.mean()
    n = len(s)
    w = signal.windows.hann(n, sym=False)
    nf = max(nfft, n)
    P = np.abs(np.fft.rfft(s * w, n=nf)) ** 2
    f = np.fft.rfftfreq(nf, 1 / fs)
    df = f[1] - f[0]
    guard = max(guard_hz, 2.2 * fs / n)
    g = int(round(guard / df))
    k = g + max(3, int(round(train_hz / df)))
    lim = int(np.searchsorted(f, min(4.0, 0.45 * fs)))
    foot = np.ones(2 * k + 1, dtype=bool)
    foot[k - g:k + g + 1] = False
    floor = ndimage.percentile_filter(P[:lim], 50, footprint=foot, mode="mirror") / LN2
    if noise_band is not None:
        nbm = (f >= noise_band[0]) & (f <= noise_band[1])
        if nbm.any():
            floor = np.maximum(floor, np.median(P[nbm]) / LN2)
    Pn = P[:lim] / np.maximum(floor, 1e-30)
    idx = np.flatnonzero((f[:lim] >= band[0]) & (f[:lim] <= band[1]))
    if idx.size == 0:
        return 0.0, float("nan")
    loc = idx[(Pn[idx] >= Pn[np.maximum(idx - 1, 0)]) & (Pn[idx] >= Pn[np.minimum(idx + 1, lim - 1)])]
    cand = loc if loc.size else idx
    i = int(cand[np.argmax(Pn[cand])])
    return 10 * math.log10(max(float(Pn[i]), 1e-12)), float(f[i])


def md_breath(x: np.ndarray, fs: float, win_s: float = 2.0, hop_s: float = 0.1,
              band=(0.1, 0.8)) -> dict:
    """Détection « micro-Doppler » de la respiration : périodicité du centroïde
    Doppler (inspiration → vers le radar, expiration → s'éloigne)."""
    z = remove_clutter(x, fs)
    t, f, S = spectrogram(z, fs, win_s, hop_s)
    if len(t) < 16:
        return {"snr_db": 0.0, "rate_hz": float("nan"), "spread_hz": float("nan")}
    c, spread = centroid_spread(S, f, noise_band=(8.0, 0.45 * fs))
    snr, fr = line_snr(c, 1.0 / hop_s, band, noise_band=(2.0, 0.45 / hop_s))
    return {"snr_db": snr, "rate_hz": fr, "spread_hz": float(np.median(spread)),
            "centroid": c, "t": t}


def instantaneous_doppler(x: np.ndarray, fs: float, center: complex | None = None) -> np.ndarray:
    """Doppler instantané f_D(t) = (1/2π)·dφ/dt de la cible (Hz).

    φ = unwrap(angle(x − centre)) ; le centre est l'écho statique (centre du
    cercle IQ si connu, sinon la moyenne — alors φ est dominée par le clutter
    et ne vaut que pour une cible forte)."""
    x = np.asarray(x, dtype=np.complex128)
    c = x.mean() if center is None else center
    ph = np.unwrap(np.angle(x - c))
    return np.gradient(ph) * fs / (2 * np.pi)


def compare_window(x: np.ndarray, fs: float, analyzer) -> dict:
    """Les détecteurs comparés au rapport §9.11, sur une même fenêtre.

    A1 phase linéarisée (voies radiale/tangentielle, détecteur en service),
    A2 vraie phase (arc-tangente autour du centre du cercle IQ),
    B1 micro-Doppler (périodicité du centroïde du spectrogramme, trames de 2 s),
    B2 Doppler instantané (dφ/dt).  Valeurs : (SNR dB, fréquence Hz)."""
    from radar.dsp.vitals import fit_circle
    x = np.asarray(x, dtype=np.complex128)
    f, _ = analyzer.analyze(x, want_display=False)
    out = {"A1": (f.snr_db, f.breath_hz)}
    circ = fit_circle(x)
    c = circ[0] if circ is not None else x.mean()
    ph = np.unwrap(np.angle(x - c))
    out["A2"] = line_snr(ph, fs, noise_band=(4.0, 12.0))
    out["B2"] = line_snr(instantaneous_doppler(x, fs, c), fs)
    r = md_breath(x, fs, win_s=2.0)
    out["B1"] = (r["snr_db"], r["rate_hz"])
    return out
