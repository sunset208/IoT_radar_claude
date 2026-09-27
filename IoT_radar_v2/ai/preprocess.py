"""Pré-traitement commun (entraînement ET inférence temps réel).

Entrée : fenêtre slow-time complexe brute (clutter inclus).
Sortie : tenseur (C, T) float32 invariant aux paramètres de nuisance :

* **phase absolue** — rotation dans la base (radiale, tangentielle) du clutter
  statique C (même base que le détecteur classique, la dérive LO y est
  confinée à la voie tangentielle) ;
* **niveau absolu** (gain RX, distance) — normalisation par l'écart-type du
  bruit estimé dans la bande 4–12 Hz : l'amplitude est donc exprimée en
  « unités de bruit » (≈ SNR), information utile qu'on ne veut pas effacer ;
* **dérive lente** — tendance linéaire + passe-haut 0.06 Hz.

Canaux : 0 radial, 1 tangentiel, 2 |x − C| normalisé (enveloppe), puis
compression douce ``asinh`` pour borner la dynamique (cibles très fortes).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

N_CHANNELS = 3
_SOS_CACHE: dict = {}


def _sos(fs: float):
    if fs not in _SOS_CACHE:
        _SOS_CACHE[fs] = (
            signal.butter(2, 0.06, btype="high", fs=fs, output="sos"),
            signal.butter(4, [4.0, min(12.0, 0.45 * fs)], btype="band", fs=fs, output="sos"),
        )
    return _SOS_CACHE[fs]


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    c = x.mean()
    uc = c / abs(c) if abs(c) > 0 else 1.0
    n = np.arange(len(x), dtype=np.float64) - (len(x) - 1) / 2
    xm = x - c
    xm = xm - (np.dot(n, xm) / np.dot(n, n)) * n
    hp, nb = _sos(fs)
    xh = signal.sosfiltfilt(hp, xm) * np.conj(uc)
    noise = signal.sosfiltfilt(nb, xh)
    sigma = 1.4826 * np.median(np.abs(noise)) / np.sqrt(2) + 1e-12   # MAD robuste
    z = xh / sigma
    env = np.abs(z)
    out = np.stack((z.real, z.imag, env - env.mean()), axis=0)
    return np.arcsinh(out / 4.0).astype(np.float32)
