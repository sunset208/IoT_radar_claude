"""Traitement SFCW (amplitude seule) : balayages → profil distance → respiration par case.

Chaîne, pour chaque balayage (vecteur A_k = |S_k|, k = 0…N−1) :

1. **fond** B_k : moyenne exponentielle des balayages (τ = ``bg_tau_s``) — un
   filtre MTI : ce qui est immobile disparaît, mouvement et respiration restent ;
2. **modulation relative** x_k = (A_k − B_k)/B_k : la réponse en fréquence
   du système (antennes, gains) se simplifie ;
3. **profil** R(r) = FFT_k(w_k · x_k) (fenêtre de Hann, zéro-padding) : x est
   réel, donc on garde les « délais » positifs, 0 … c/(4Δf) ;
4. par case de distance, signal complexe lent R(r, t) (cadence = balayages/s) ;
   toutes les ``hop_s`` : périodogramme sur ``breath_window_s`` → SNR du pic
   respiratoire 0.1–0.8 Hz contre le plancher (bande haute), rythme ;
   énergie de mouvement 0.1–1.5 Hz → carte distance × temps et présence.

Le profil « statique » (FFT de B_k/moy(B) − 1) montre les réflecteurs fixes
(murs, meubles) via leur battement avec la fuite.
"""

from __future__ import annotations

import collections
import dataclasses as dc
import math

import numpy as np

from radar.config import SPEED_OF_LIGHT

EMPTY, PRESENCE, MOTION, BREATHING = "VIDE", "PRESENCE", "MOUVEMENT", "RESPIRATION"


def tone_amplitude(x: np.ndarray, f_s: float, f_tone: float, full_scale: float = 2048.0) -> complex:
    """Amplitude complexe (pleine échelle = 1) de la tonalité f_tone dans *x*.

    Corrélation sur un nombre entier de périodes : orthogonale au DC du
    récepteur et à l'image (−f_tone) si len(x)·f_tone/f_s est entier.
    """
    n = len(x)
    k = np.arange(n)
    return complex(np.dot(np.asarray(x, np.complex128), np.exp(-2j * np.pi * f_tone / f_s * k))
                   / n / full_scale)


@dc.dataclass
class Sweep:
    amps: np.ndarray            # |S_k| (pleine échelle = 1), ordre des fréquences croissantes
    t: float                    # instant (s) du milieu du balayage
    step_t: np.ndarray | None = None
    stats: dict = dc.field(default_factory=dict)


class SfcwProcessor:
    def __init__(self, freqs: np.ndarray, bg_tau_s: float = 30.0, nfft: int = 256,
                 min_range_m: float = 0.3, max_range_m: float = 0.0,
                 breath_window_s: float = 20.0, hop_s: float = 1.0,
                 breath_band=(0.1, 0.8), breath_threshold_db: float = 12.0,
                 presence_threshold_db: float = 8.0, on_count: int = 3,
                 map_span_s: float = 60.0) -> None:
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.n = len(self.freqs)
        df = float(np.median(np.diff(self.freqs)))
        self.df = df
        self.nfft = max(int(nfft), 2 * self.n)
        r_all = SPEED_OF_LIGHT * np.arange(self.nfft // 2 + 1) / (2 * df * self.nfft)
        r_max = max_range_m if max_range_m > 0 else SPEED_OF_LIGHT / (4 * df)
        self.nb = int(np.searchsorted(r_all, r_max, side="right"))
        self.range_m = r_all[: self.nb]
        self.r_min = float(min_range_m)
        self.win = np.hanning(self.n + 2)[1:-1]
        self.win /= self.win.sum()
        self.bg_tau = float(bg_tau_s)
        self.breath_window_s = float(breath_window_s)
        self.hop_s = float(hop_s)
        self.bb = tuple(breath_band)
        self.breath_thr = float(breath_threshold_db)
        self.pres_thr = float(presence_threshold_db)
        self.on_count = int(on_count)
        self.map_span_s = float(map_span_s)
        self.static_ref: np.ndarray | None = None   # fond « salle vide » figé (optionnel)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.B: np.ndarray | None = None
        self.t_last: float | None = None
        self.n_sweeps = 0
        maxlen = 4096
        self._t: collections.deque = collections.deque(maxlen=maxlen)
        self._R: collections.deque = collections.deque(maxlen=maxlen)
        self.map_cols: collections.deque = collections.deque(maxlen=240)
        self.map_t: collections.deque = collections.deque(maxlen=240)
        self._t_hop = -1e9
        self._pos = 0
        self.state = EMPTY
        self.result: dict = {}
        self.profile = np.zeros(self.nb, complex)
        self.static_profile = np.zeros(self.nb)
        self.sweep_hz = 0.0
        self.motion_extent = None

    def freeze_background(self) -> None:
        """Mémorise le fond courant comme référence « salle vide » (profil statique)."""
        if self.B is not None:
            self.static_ref = self.B.copy()

    # ------------------------------------------------------------------
    def range_transform(self, x: np.ndarray) -> np.ndarray:
        """x (…, N) réel → profil complexe (…, nb) sur les délais positifs."""
        X = np.fft.rfft(np.asarray(x) * self.win, n=self.nfft, axis=-1)
        return X[..., : self.nb]

    def push(self, sw: Sweep) -> dict | None:
        """Ajoute un balayage ; renvoie un résultat d'analyse toutes les ``hop_s``."""
        A = np.asarray(sw.amps, dtype=np.float64)
        if A.shape != (self.n,) or not np.all(np.isfinite(A)):
            return None
        if self.B is None:
            self.B = A.copy()
        dt = 0.0 if self.t_last is None else max(1e-3, sw.t - self.t_last)
        if self.t_last is not None:
            self.sweep_hz = 0.9 * self.sweep_hz + 0.1 / dt if self.sweep_hz else 1.0 / dt
        self.t_last = sw.t
        a = 1.0 - math.exp(-dt / self.bg_tau) if dt > 0 else 0.0
        # pendant le 1er quart de τ, fond plus réactif (démarrage)
        if self.n_sweeps < 20:
            a = max(a, 1.0 / (self.n_sweeps + 1))
        x = (A - self.B) / np.maximum(self.B, 1e-12)
        self.B += a * (A - self.B)
        R = self.range_transform(x)
        self.profile = R
        ref = self.static_ref if self.static_ref is not None else self.B
        self.static_profile = np.abs(self.range_transform(ref / ref.mean() - 1.0))
        self._t.append(sw.t)
        self._R.append(R)
        self.n_sweeps += 1
        if sw.t - self._t_hop >= self.hop_s and self.n_sweeps >= 8:
            self._t_hop = sw.t
            return self._analyze(sw.t)
        return None

    # ------------------------------------------------------------------
    def _uniform(self, span_s: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Dernières *span_s* secondes rééchantillonnées sur une grille régulière."""
        t = np.asarray(self._t)
        if len(t) < 8:
            return None
        m = t >= t[-1] - span_s
        t = t[m]
        R = np.asarray(self._R)[m]                    # (n_t, nb)
        fs = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
        n = len(t)
        tu = t[0] + np.arange(n) / fs
        Ru = np.empty_like(R)
        for b in range(R.shape[1]):
            Ru[:, b] = np.interp(tu, t, R[:, b].real) + 1j * np.interp(tu, t, R[:, b].imag)
        return tu, Ru, fs

    def _analyze(self, t_now: float) -> dict:
        u = self._uniform(self.breath_window_s)
        out = {"t": t_now}
        if u is None:
            return out
        tu, Ru, fs = u
        n = len(tu)
        Z = Ru - Ru.mean(axis=0)
        # tendance linéaire retirée (dérive lente résiduelle du fond)
        k = np.arange(n) - (n - 1) / 2
        Z -= np.outer(k, (k @ Z) / (k @ k))
        w = np.hanning(n)[:, None]
        nf = 1 << max(9, int(math.ceil(math.log2(n * 4))))
        F = np.fft.fft(Z * w, n=nf, axis=0)
        f = np.fft.fftfreq(nf, 1.0 / fs)
        P = np.abs(F) ** 2
        # respiration : mouvement de va-et-vient → énergie des deux côtés (±f)
        fpos = f[: nf // 2]
        Ppos = P[: nf // 2] + np.roll(P[::-1], 1, axis=0)[: nf // 2]
        nyq = fs / 2
        noise_m = (fpos >= min(1.0, 0.6 * nyq)) & (fpos <= 0.95 * nyq)
        if not noise_m.any():
            noise_m = fpos >= 0.8 * nyq
        noise_raw = np.median(Ppos[noise_m], axis=0) / math.log(2)
        # plancher global = cases calmes (20e percentile) : un marcheur (Doppler
        # replié, large bande) relève le plancher de *ses* cases, pas celui-ci
        g = float(np.percentile(noise_raw, 20)) + 1e-30
        noise_b = np.maximum(noise_raw, 0.5 * g)
        bm = (fpos >= self.bb[0]) & (fpos <= min(self.bb[1], 0.9 * nyq))
        Pb = Ppos[bm] / noise_b
        i_pk = np.argmax(Pb, axis=0)
        snr = 10 * np.log10(np.maximum(Pb[i_pk, np.arange(Pb.shape[1])], 1e-12))
        rate = fpos[bm][i_pk] * 60.0
        # énergie de mouvement (0.1 Hz → Nyquist) rapportée au plancher global
        mm = (fpos >= 0.1) & (fpos <= 0.95 * nyq)
        motion = 10 * np.log10(np.maximum(Ppos[mm].mean(axis=0) / g, 1e-12))
        valid = self.range_m >= self.r_min
        # Zone « en mouvement » : cases dont l'énergie de mouvement est à moins
        # de 12 dB du maximum.  Une poitrine qui respire reste dans le lobe
        # principal (~4 cellules de résolution) ; un marcheur balaie des mètres.
        res = SPEED_OF_LIGHT / (2 * self.df * self.n)
        mv = valid & (motion >= self.pres_thr) & (motion >= np.max(np.where(valid, motion, -99)) - 12.0)
        extent = float(np.ptp(self.range_m[mv])) if mv.any() else 0.0
        moving = extent > max(0.8, 5 * res)
        snr_v = np.where(valid, snr, -np.inf)
        if moving:
            # respiration candidate seulement hors de la zone en mouvement
            near = np.zeros_like(valid)
            for r in self.range_m[mv]:
                near |= np.abs(self.range_m - r) <= 1.5 * res
            snr_v = np.where(near, -np.inf, snr_v)
        ib = int(np.argmax(snr_v))
        best_snr = float(snr_v[ib]) if np.isfinite(snr_v[ib]) else float(np.max(np.where(valid, snr, -99)))
        pres_any = bool(np.any(motion[valid] >= self.pres_thr))
        if np.isfinite(snr_v[ib]) and best_snr >= self.breath_thr:
            self._pos += 1
        else:
            self._pos = 0
        if self._pos >= self.on_count:
            self.state = BREATHING
        elif moving:
            self.state = MOTION
        elif pres_any or best_snr >= self.breath_thr:
            self.state = PRESENCE
        else:
            self.state = EMPTY
        self.motion_extent = (float(self.range_m[mv].min()), float(self.range_m[mv].max())) if moving else None
        # carte distance × temps : écart récent (~2 s) à la moyenne de la fenêtre
        # longue — une poitrine qui respire y reste visible, pas seulement un marcheur
        k2 = max(2, int(round(max(2.0, self.hop_s) * fs)))
        e = np.mean(np.abs(Z[-k2:]) ** 2, axis=0)
        # référence = cases calmes (20e percentile), comme pour le plancher
        e_db = 10 * np.log10(e / (np.percentile(e, 20) + 1e-30) + 1e-12)
        self.map_cols.append(np.round(e_db, 1))
        self.map_t.append(t_now)
        self.result = out = {
            "t": t_now, "fs": fs, "state": self.state,
            "breath_snr_db": snr, "breath_bpm": rate, "motion_db": motion,
            "best_bin": ib, "best_range_m": float(self.range_m[ib]), "best_snr_db": best_snr,
            "best_bpm": float(rate[ib]), "presence": pres_any, "motion_extent_m": self.motion_extent,
            "slow": Ru[:, ib],
        }
        return out

    # ------------------------------------------------------------------
    def ui_state(self) -> dict:
        """Dictionnaire JSON pour le panneau SFCW du tableau de bord."""
        noise = np.median(np.abs(self.profile)) + 1e-15
        prof_db = 20 * np.log10(np.abs(self.profile) / noise + 1e-12)
        st_db = 20 * np.log10(self.static_profile / (np.median(self.static_profile) + 1e-15) + 1e-12)
        r = self.result
        n_show = max(1, int(self.map_span_s / max(self.hop_s, 1e-3)))
        cols = [c.tolist() for c in list(self.map_cols)[-n_show:]]
        mt = list(self.map_t)[-n_show:]
        span = (mt[-1] - mt[0]) if len(mt) > 1 else 0.0
        return {
            "status": "running" if self.n_sweeps else "starting",
            "sweep_hz": self.sweep_hz, "n_steps": self.n,
            "f_range": [float(self.freqs[0]), float(self.freqs[-1])],
            "range_m": np.round(self.range_m, 3).tolist(),
            "resolution_m": SPEED_OF_LIGHT / (2 * self.df * self.n),
            "profile_db": np.round(prof_db, 1).tolist(),
            "static_db": np.round(st_db, 1).tolist(),
            "static_ref": self.static_ref is not None,
            "rt_map": cols,
            "rt_span_s": span, "rt_vmin": 0.0, "rt_vmax": 25.0,
            "breath_snr_db": (None if r.get("breath_snr_db") is None
                              else np.round(r["breath_snr_db"], 1).tolist()),
            "breath_thr_db": self.breath_thr,
            "best_range_m": r.get("best_range_m"), "best_snr_db": r.get("best_snr_db"),
            "best_bpm": r.get("best_bpm"), "state": self.state,
            "motion_extent_m": self.motion_extent,
        }
