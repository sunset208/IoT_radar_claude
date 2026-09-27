"""Front-end streaming : IQ brut (f_s) → signal « slow-time » complexe (fs_slow).

Étapes (toutes à état conservé entre buffers) :

1. **NCO** : multiplication par ``exp(-j2π f_offset n / f_s)``.  L'écho utile
   (fuite, murs, personne) passe de f_offset à 0 Hz ; l'offset DC du récepteur
   et la fuite de LO passent à −f_offset.  La table du NCO est précalculée
   (buffer multiple de f_s/f_offset → phase exactement continue).
2. **CIC d'ordre N** de facteur R = f_s/f_offset (moyennes glissantes en
   cascade) : ses zéros tombent *exactement* sur les multiples de f_offset,
   donc sur l'offset DC récepteur et l'image TX.  Coût quasi nul (cumsum).
3. **Étages IIR elliptiques** (SOS, état conservé) de facteurs ≤ 10 jusqu'à
   fs_slow, coupure à 0.4·f_out.

Le signal de sortie est normalisé : 1.0 = pleine échelle ADC.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

ADC_FULL_SCALE = 2048.0


def _split_factors(d: int, max_stage: int = 10) -> list[int]:
    """Décompose *d* en facteurs ≤ max_stage (les plus gros d'abord)."""
    out: list[int] = []
    rem = d
    while rem > 1:
        for q in range(min(max_stage, rem), 1, -1):
            if rem % q == 0:
                out.append(q)
                rem //= q
                break
        else:
            raise ValueError(f"Facteur de décimation {d} non décomposable en étages ≤ {max_stage}.")
    return out


class _CicStage:
    """Décimateur CIC (moyennes glissantes en cascade) streaming, gain DC = 1."""

    def __init__(self, R: int, order: int) -> None:
        self.R = R
        self.order = order
        # Chaque moyenne glissante de longueur R nécessite R-1 échantillons d'historique.
        self.hist = [np.zeros(R - 1, dtype=np.complex128) for _ in range(order)]
        self.primed = False
        self.count = 0  # nombre d'échantillons déjà traités

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.primed and x.size:
            # régime établi : historique = moyenne du 1er bloc (évite un transitoire)
            m = complex(np.mean(x[: max(self.R, 1) * 4]))
            for h in self.hist:
                h[:] = m
            self.primed = True
        y = x
        for i in range(self.order):
            ext = np.concatenate((self.hist[i], y))
            self.hist[i] = ext[len(ext) - (self.R - 1):].copy() if self.R > 1 else self.hist[i]
            c = np.cumsum(ext)
            c = np.concatenate(([0.0], c))
            y = (c[self.R:] - c[:-self.R]) / self.R  # longueur = len(y)
        # on garde les indices globaux i tels que (i + 1) % R == 0
        start = (self.R - 1 - self.count % self.R) % self.R
        self.count += len(y)
        return y[start::self.R]


class _IirStage:
    def __init__(self, q: int, f_in: float) -> None:
        self.q = q
        f_out = f_in / q
        # Elliptique ordre 8 : ondulation 0.05 dB, réjection 80 dB, bord 0.4·f_out
        self.sos = signal.ellip(8, 0.05, 80, 0.4 * f_out, fs=f_in, output="sos")
        self.zi = None
        self.count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self.zi is None:   # état initial en régime établi sur le 1er échantillon
            self.zi = signal.sosfilt_zi(self.sos).astype(np.complex128) * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        start = (self.q - 1 - self.count % self.q) % self.q
        self.count += len(y)
        return y[start::self.q]


class Frontend:
    """NCO + CIC + IIR : IQ brut → slow-time normalisé (pleine échelle = 1)."""

    def __init__(self, f_s: float, f_offset: float, fs_slow: float,
                 buffer_size: int, cic_order: int = 2) -> None:
        self.f_s = float(f_s)
        self.f_offset = float(f_offset)
        self.fs_slow = float(fs_slow)
        R = int(round(f_s / f_offset))
        D = int(round(f_s / fs_slow))
        if D % R:
            raise ValueError("f_s/fs_slow doit être multiple de f_s/f_offset.")
        self.R = R
        self.cic = _CicStage(R, cic_order)
        self.iir: list[_IirStage] = []
        f = f_s / R
        for q in _split_factors(D // R):
            self.iir.append(_IirStage(q, f))
            f /= q
        # Table NCO : une période entière de f_offset tient dans R échantillons.
        n = np.arange(buffer_size + R)
        self._lo = np.exp(-2j * np.pi * f_offset * n / f_s)
        self._lo_len = len(n)
        self._n = 0  # position (mod R) dans la table
        logger.info(
            "Frontend : %.0f Hz → CIC%d(R=%d) → %s → %.1f Hz (f_offset=%.0f Hz)",
            f_s, cic_order, R, "×".join(str(s.q) for s in self.iir) or "∅", fs_slow, f_offset,
        )

    def lo(self, n: int) -> np.ndarray:
        """Échantillons du NCO pour les *n* prochains échantillons (phase continue)."""
        start = self._n % self.R
        if start + n <= self._lo_len:
            out = self._lo[start:start + n]
        else:
            k = np.arange(self._n, self._n + n)
            out = np.exp(-2j * np.pi * self.f_offset * k / self.f_s)
        self._n = (self._n + n) % self.R
        return out

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """*raw* en comptes ADC (±2048) ou déjà normalisé si ``raw.dtype`` flottant ≤ 1."""
        x = np.asarray(raw, dtype=np.complex128) / ADC_FULL_SCALE
        x *= self.lo(len(x))
        y = self.cic(x)
        for st in self.iir:
            y = st(y)
        return y.astype(np.complex64)

    @property
    def group_delay_s(self) -> float:
        """Retard de groupe approximatif (s) — indicatif pour l'UI."""
        d = (self.cic.order * (self.R - 1) / 2) / self.f_s
        f = self.f_s / self.R
        for st in self.iir:
            w, gd = signal.group_delay(signal.sos2tf(st.sos), w=[0.05], fs=f)
            d += float(gd[0]) / f
            f /= st.q
        return d


def adc_level_dbfs(raw: np.ndarray) -> tuple[float, float]:
    """(RMS dBFS, crête dBFS) d'un buffer brut en comptes ADC."""
    a = np.abs(np.asarray(raw)) / ADC_FULL_SCALE
    rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
    peak = float(np.max(a)) if a.size else 0.0
    return 20 * math.log10(rms + 1e-12), 20 * math.log10(peak + 1e-12)
