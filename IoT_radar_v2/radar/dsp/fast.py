"""Indices rapides (10 Hz) : activité (~1 s) et présence (~4 s), forme d'onde causale.

Le détecteur de respiration a besoin de ~20 s de signal pour sortir une raie
du bruit (voir rapport §9).  Pour réagir vite, on calcule en parallèle, à
chaque bloc de 0.1 s et avec des filtres **causaux** à état :

* ``activity_db`` : énergie 1.5–15 Hz (les deux voies) sur ``activity_s``
  rapportée au bruit thermique.  Mouvement franc du corps, marche, geste →
  +10 à +25 dB en moins d'une seconde.
* ``presence_db`` : énergie 0.12–1 Hz sur ``presence_s`` de chaque voie
  (radiale / tangentielle au clutter) rapportée à **son** bruit attendu en
  salle vide.  Micro-mouvements lents (respiration, même irrégulière,
  ajustements de posture) → « signe de vie » en ~4 s, avant confirmation de
  la périodicité.

Modèle de bruit d'une voie dans la bande de présence ::

    E0 = N_th · ENBW  +  k · |C|²

``N_th`` : densité du bruit thermique, suivie en continu sur 4–12 Hz (20e
percentile glissant sur 60 s, robuste aux bouffées de mouvement) ;
``k·|C|²`` : bruit **multiplicatif** du clutter statique C (fuite TX→RX) —
bruit d'amplitude sur la voie radiale, bruit de phase LO sur la voie
tangentielle.  ``k`` est une constante matérielle (dBc intégrés dans la bande),
mesurée en salle vide par ``radar calibrate`` ; défauts = Pluto du labo,
27/09/2026.  Sous H0 l'indice vaut donc ≈ 0 dB quel que soit le montage, et
écarter les antennes (|C|² plus faible) rend automatiquement l'indice plus
sensible.
"""

from __future__ import annotations

import collections
import dataclasses as dc
import math

import numpy as np
from scipy import signal


def _enbw(sos: np.ndarray, fs: float) -> float:
    """Bande équivalente de bruit (Hz, unilatérale) d'un filtre SOS."""
    w, h = signal.sosfreqz(sos, worN=1 << 15, fs=fs)
    return float(np.trapezoid(np.abs(h) ** 2, w))


@dc.dataclass
class FastParams:
    activity_s: float = 1.0
    presence_s: float = 4.0
    update_s: float = 0.1
    clutter_tau_s: float = 10.0
    presence_band: tuple[float, float] = (0.12, 1.0)
    activity_band: tuple[float, float] = (1.5, 15.0)
    noise_band: tuple[float, float] = (4.0, 12.0)
    floor_window_s: float = 60.0
    floor_q: float = 20.0
    # bruit multiplicatif du clutter intégré dans la bande de présence (dBc)
    clutter_amp_dbc: float = -84.7      # Pluto du labo, 27/09/2026 (= configs/default.yaml)
    clutter_phase_dbc: float = -65.3
    # la voie tangentielle porte la dérive LO (1/f², non stationnaire) : son
    # plancher est le moins sûr, on lui impose une marge supplémentaire
    tan_margin_db: float = 4.0


@dc.dataclass
class FastTick:
    t: float
    activity_db: float
    presence_db: float
    presence_r_db: float
    presence_t_db: float
    noise_dbfs: float                 # densité thermique × fs (comparable à Features.noise_dbfs)
    wave: np.ndarray                  # projection filtrée (≈10 Hz), en µFS (1e-6 pleine échelle)
    wave_dt: float
    wave_scale_mm: float | None       # mm par µFS si le rayon du cercle IQ est connu
    raw: tuple = ()                   # (E_r, E_t, N_r, N_t, |C|²) — pour la calibration

    def to_dict(self) -> dict:
        return {"t": self.t, "activity_db": self.activity_db, "presence_db": self.presence_db,
                "presence_r_db": self.presence_r_db, "presence_t_db": self.presence_t_db,
                "noise_dbfs": self.noise_dbfs, "wave": self.wave.tolist(),
                "wave_dt": self.wave_dt, "wave_scale_mm": self.wave_scale_mm}


class FastMonitor:
    """Traitement causal par blocs ; ``push`` renvoie un :class:`FastTick` par
    intervalle ``update_s`` écoulé (typiquement un par bloc de 0.1 s)."""

    def __init__(self, fs: float, params: FastParams | None = None,
                 wavelength: float | None = None) -> None:
        self.fs = float(fs)
        self.p = params or FastParams()
        self.lam = wavelength
        p = self.p
        nyq = 0.5 * self.fs
        self._sos_p = np.vstack((
            signal.butter(4, p.presence_band[0], btype="high", fs=fs, output="sos"),
            signal.butter(2, p.presence_band[1], btype="low", fs=fs, output="sos")))
        self._sos_a = signal.butter(4, [p.activity_band[0], min(p.activity_band[1], 0.9 * nyq)],
                                    btype="band", fs=fs, output="sos")
        self._sos_n = signal.butter(4, [p.noise_band[0], min(p.noise_band[1], 0.9 * nyq)],
                                    btype="band", fs=fs, output="sos")
        # densité (par Hz) d'une voie réelle = puissance filtrée / ENBW
        self.enbw_p = _enbw(self._sos_p, fs)
        self.enbw_a = _enbw(self._sos_a, fs)
        self.enbw_n = _enbw(self._sos_n, fs)
        self.alpha = 1.0 - math.exp(-1.0 / (p.clutter_tau_s * fs))
        self.k_amp = 10 ** (p.clutter_amp_dbc / 10)
        self.k_phase = 10 ** (p.clutter_phase_dbc / 10)
        self.tan_margin = 10 ** (p.tan_margin_db / 10)
        self.target_radius: float | None = None     # rayon du cercle IQ (analyse lente)
        self.n_act = max(1, int(round(p.activity_s * fs)))
        self.n_pres = max(1, int(round(p.presence_s * fs)))
        self.n_upd = max(1, int(round(p.update_s * fs)))
        self.n_floor = max(1, int(round(1.0 * fs)))     # plancher : moyennes sur 1 s
        self.wave_dec = max(1, int(round(fs / 10.0)))
        self.floor_bias = self._pct_bias(self._sos_n, self.n_floor)
        self.tan_bias = self._pct_bias(self._sos_p, self.n_pres)
        self.reset()

    def _pct_bias(self, sos, n_avg: int) -> float:
        """moyenne / 20e percentile des moyennes glissantes (n_avg) de bruit blanc filtré.

        Le suivi de plancher par percentile bas est biaisé ; on mesure ce biais
        une fois, numériquement, pour le filtre et la durée de moyennage utilisés.
        """
        rng = np.random.default_rng(12345)
        n = int(20 * self.p.floor_window_s * self.fs)
        y = signal.sosfilt(sos, rng.standard_normal(n))[int(10 * self.fs):] ** 2
        m = np.convolve(y, np.ones(n_avg) / n_avg, mode="valid")[:: self.n_upd]
        return float(np.mean(y) / np.percentile(m, self.p.floor_q))

    # ------------------------------------------------------------------
    def set_clutter_noise(self, amp_dbc: float | None = None, phase_dbc: float | None = None) -> None:
        if amp_dbc is not None:
            self.k_amp = 10 ** (amp_dbc / 10)
        if phase_dbc is not None:
            self.k_phase = 10 ** (phase_dbc / 10)

    def reset(self) -> None:
        self._c: complex | None = None
        self._zi_p = self._zi_a = self._zi_n = None
        n_keep = max(self.n_pres, self.n_act, self.n_upd) + 1
        self._pr = collections.deque(maxlen=n_keep)     # |z_p.real|²
        self._pt = collections.deque(maxlen=n_keep)
        self._pa = collections.deque(maxlen=n_keep)     # |z_a|²
        self._nr = collections.deque(maxlen=self.n_floor)
        self._nt = collections.deque(maxlen=self.n_floor)
        self._floor_r = collections.deque(maxlen=int(self.p.floor_window_s / self.p.update_s))
        self._floor_t = collections.deque(maxlen=int(self.p.floor_window_s / self.p.update_s))
        self._et_hist = collections.deque(maxlen=int(self.p.floor_window_s / self.p.update_s))
        self._since = 0
        self._count = 0
        self._cov = np.zeros((2, 2))
        self._axis = np.array([1.0, 0.0])
        self._wave_buf: list[float] = []
        self._wave_phase = 0
        self._c_abs2 = 0.0

    # ------------------------------------------------------------------
    def _filters(self, z: np.ndarray):
        if self._zi_p is None:
            self._zi_p = np.zeros((self._sos_p.shape[0], 2), complex)
            self._zi_a = np.zeros((self._sos_a.shape[0], 2), complex)
            self._zi_n = np.zeros((self._sos_n.shape[0], 2), complex)
        zp, self._zi_p = signal.sosfilt(self._sos_p, z, zi=self._zi_p)
        za, self._zi_a = signal.sosfilt(self._sos_a, z, zi=self._zi_a)
        zn, self._zi_n = signal.sosfilt(self._sos_n, z, zi=self._zi_n)
        return zp, za, zn

    def push(self, x: np.ndarray, t_end: float) -> list[FastTick]:
        x = np.asarray(x, dtype=np.complex128)
        if x.size == 0:
            return []
        # clutter statique : moyenne exponentielle (τ = clutter_tau_s)
        if self._c is None:
            self._c = complex(np.mean(x))
        a = self.alpha
        c_seq, zf = signal.lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * self._c])
        self._c = complex(c_seq[-1])
        u = c_seq / np.maximum(np.abs(c_seq), 1e-15)
        z = (x - c_seq) * np.conj(u)
        zp, za, zn = self._filters(z)
        self._c_abs2 = float(abs(self._c) ** 2)

        out: list[FastTick] = []
        n = len(x)
        for i in range(n):
            self._pr.append(zp[i].real ** 2)
            self._pt.append(zp[i].imag ** 2)
            self._pa.append(abs(za[i]) ** 2)
            self._nr.append(zn[i].real ** 2)
            self._nt.append(zn[i].imag ** 2)
            # axe principal (covariance exponentielle de la bande de présence)
            v = np.array([zp[i].real, zp[i].imag])
            self._cov += a * (np.outer(v, v) - self._cov)
            self._wave_phase += 1
            if self._wave_phase >= self.wave_dec:
                self._wave_phase = 0
                self._wave_buf.append(float(v @ self._axis))
            self._count += 1
            self._since += 1
            if self._since >= self.n_upd:
                self._since = 0
                ti = t_end - (n - 1 - i) / self.fs
                tick = self._tick(ti)
                if tick is not None:
                    out.append(tick)
        return out

    def _tick(self, t: float) -> FastTick | None:
        self._floor_r.append(float(np.mean(self._nr)))
        self._floor_t.append(float(np.mean(self._nt)))
        # l'axe principal est mis à jour à 10 Hz ; signe continu
        w, vec = np.linalg.eigh(self._cov)
        ax = vec[:, 1]
        if ax @ self._axis < 0:
            ax = -ax
        self._axis = ax
        wave = np.asarray(self._wave_buf, dtype=np.float64) * 1e6       # → µFS
        self._wave_buf = []
        scale = (self.lam / (4 * math.pi * self.target_radius) * 1e3 * 1e-6
                 if (self.target_radius and self.lam) else None)
        if self._count < max(self.n_pres, int(2.0 * self.fs)):
            return None     # filtres et plancher pas encore établis
        q = self.p.floor_q
        Nr = float(np.percentile(self._floor_r, q)) * self.floor_bias / self.enbw_n
        Nt = float(np.percentile(self._floor_t, q)) * self.floor_bias / self.enbw_n
        pr = list(self._pr)[-self.n_pres:]
        pt = list(self._pt)[-self.n_pres:]
        pa = list(self._pa)[-self.n_act:]
        Er, Et = float(np.mean(pr)), float(np.mean(pt))
        self._et_hist.append(Et)
        E0r = Nr * self.enbw_p + self.k_amp * self._c_abs2
        E0t = Nt * self.enbw_p + self.k_phase * self._c_abs2
        # Voie tangentielle : la dérive de phase LO varie (température, montage).
        # Plancher = max(modèle, 20e percentile glissant sur 60 s).  Une cible
        # présente en permanence *et* purement tangentielle serait masquée sur
        # cette voie seulement ; la voie radiale n'a pas ce plancher adaptatif.
        E0t = max(E0t, float(np.percentile(self._et_hist, q)) * self.tan_bias)
        rr = Er / max(E0r, 1e-30)
        rt = Et / max(E0t * self.tan_margin, 1e-30)
        ra = float(np.mean(pa)) / max((Nr + Nt) * self.enbw_a, 1e-30)
        db = lambda v: 10 * math.log10(max(v, 1e-12))
        return FastTick(t=round(t, 3), activity_db=db(ra), presence_db=db(max(rr, rt)),
                        presence_r_db=db(rr), presence_t_db=db(rt),
                        noise_dbfs=db((Nr + Nt) * self.fs * 0.5),
                        wave=wave, wave_dt=self.wave_dec / self.fs, wave_scale_mm=scale,
                        raw=(Er, Et, Nr, Nt, self._c_abs2))

    # ------------------------------------------------------------------
    @property
    def clutter_abs2(self) -> float:
        return self._c_abs2


def measure_clutter_noise(ticks_r: list[float], ticks_t: list[float], Nr: list[float],
                          Nt: list[float], c_abs2: list[float], enbw_p: float) -> tuple[float, float]:
    """(k_amp_dbc, k_phase_dbc) à partir d'énergies de présence mesurées en salle vide.

    Médiane de (E − N·ENBW) / |C|², bornée à −120 dBc (clutter parfaitement stable).
    """
    er, et = np.asarray(ticks_r), np.asarray(ticks_t)
    nr, nt, c2 = np.asarray(Nr), np.asarray(Nt), np.asarray(c_abs2)
    kr = np.median(np.maximum(er - nr * enbw_p, 0.0) / np.maximum(c2, 1e-30))
    kt = np.median(np.maximum(et - nt * enbw_p, 0.0) / np.maximum(c2, 1e-30))
    return 10 * math.log10(max(kr, 1e-12)), 10 * math.log10(max(kt, 1e-12))
