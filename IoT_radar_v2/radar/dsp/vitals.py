"""Analyse d'une fenêtre slow-time : démodulation + features de respiration.

Entrée : ``x`` complexe (fs_slow), **clutter inclus**.  Sortie : :class:`Features`
(scalaires utilisés par le détecteur / la calibration / l'IA) et
:class:`Display` (tableaux pour l'UI).

Démodulation
------------
Le signal vaut ``C + A·exp(j(φ0 − 4π d(t)/λ)) + bruit`` où ``C`` est la somme des
échos statiques.  Deux voies :

* **Projection PCA** (toujours calculée) : on retire la tendance complexe
  (moyenne + dérive linéaire), puis on projette sur l'axe principal du nuage
  IQ.  Linéaire pour les petits arcs, robuste, utilisée pour la détection
  (statistiques identiques en salle vide et avec cible → calibrable).
* **Arc-tangente avec compensation DC** : ajustement d'un cercle (Kåsa) sur le
  nuage IQ ; si l'arc est bien défini, ``d(t) = λ/(4π)·unwrap(angle(x − c))``
  donne le déplacement en **mm** (affichage, amplitude, cœur).

Features
--------
* ``snr_db``        pic respiratoire / plancher de bruit (bande ``noise_band``)
* ``concentration`` part de l'énergie 0.05–3 Hz dans le pic + harmoniques
* ``acf``           pic d'autocorrélation dans [1/f_hi, 1/f_lo]
* ``motion``        max/médiane de l'énergie par tranches de 1 s (bouffées)
* ``band_db``       énergie 0.05–3 Hz / plancher (présence de *quelque chose*)
"""

from __future__ import annotations

import dataclasses as dc
import math

import numpy as np
from scipy import ndimage, optimize, signal

LN2 = math.log(2.0)


@dc.dataclass
class Features:
    t: float = 0.0
    snr_db: float = 0.0
    breath_hz: float | None = None
    concentration: float = 0.0
    acf: float = 0.0
    motion: float = 1.0
    band_db: float = 0.0
    linearity: float = 0.5
    arc_rad: float | None = None
    disp_mm_pp: float | None = None
    heart_hz: float | None = None
    heart_snr_db: float = 0.0
    static_dbfs: float = -200.0
    noise_dbfs: float = -200.0
    drift_db: float = 0.0          # plancher tangentiel/radial à 0.1–0.3 Hz (dérive LO)

    @property
    def breath_bpm(self) -> float | None:
        return None if self.breath_hz is None else 60.0 * self.breath_hz

    @property
    def heart_bpm(self) -> float | None:
        return None if self.heart_hz is None else 60.0 * self.heart_hz

    @property
    def periodicity(self) -> float:
        return max(self.concentration, self.acf)

    # vecteur numérique stable (ordre figé) pour la calibration / l'IA
    VECTOR_KEYS = ("snr_db", "concentration", "acf", "motion_log", "band_db", "linearity")

    def vector(self) -> np.ndarray:
        return np.array([
            self.snr_db, self.concentration, self.acf,
            math.log10(max(self.motion, 1e-3)), self.band_db, self.linearity,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["breath_bpm"] = self.breath_bpm
        d["heart_bpm"] = self.heart_bpm
        d["periodicity"] = self.periodicity
        return d


@dc.dataclass
class Display:
    spec_f: np.ndarray            # Hz (0–3)
    spec_db: np.ndarray           # dB au-dessus du plancher
    wave_t: np.ndarray            # s (relatif à la fin de fenêtre)
    wave: np.ndarray              # mm si arc valide, sinon unités arbitraires
    wave_unit: str
    iq: np.ndarray                # (M, 2) points IQ centrés sur la moyenne
    circle: tuple[float, float, float] | None   # (cx, cy, r) dans le même repère


# ----------------------------------------------------------------------

def _detrend_complex(x: np.ndarray) -> np.ndarray:
    n = np.arange(len(x), dtype=np.float64)
    n -= n.mean()
    xm = x - x.mean()
    b = np.dot(n, xm) / np.dot(n, n)
    return xm - b * n


def _pca_project(xd: np.ndarray) -> tuple[np.ndarray, float, complex]:
    re, im = xd.real, xd.imag
    c = np.cov(np.vstack((re, im)))
    w, v = np.linalg.eigh(c)
    u = v[:, 1]
    lin = float(w[1] / max(w[0] + w[1], 1e-30))
    uc = complex(u[0], u[1])
    return (xd * np.conj(uc)).real, lin, uc


def fit_circle(x: np.ndarray) -> tuple[complex, float, float] | None:
    """Ajustement de cercle : Kåsa (algébrique) puis raffinement géométrique.

    Kåsa seul sous-estime le rayon sur un arc court et bruité (→ amplitude de
    déplacement surestimée) ; quelques itérations de Gauss-Newton sur la
    distance géométrique corrigent ce biais.  Retourne (centre, rayon, résidu_rms).
    """
    m = x.mean()
    z = x - m
    xr, yr = z.real, z.imag
    A = np.column_stack((xr, yr, np.ones_like(xr)))
    b = xr**2 + yr**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx**2 + cy**2
    if not np.isfinite(r2) or r2 <= 0:
        return None
    scale = float(np.sqrt(np.mean(b))) + 1e-30
    xs, ys = xr / scale, yr / scale

    def _res(p):
        return np.hypot(xs - p[0], ys - p[1]) - p[2]

    def _jac(p):
        d = np.maximum(np.hypot(xs - p[0], ys - p[1]), 1e-12)
        return np.column_stack(((p[0] - xs) / d, (p[1] - ys) / d, -np.ones_like(d)))

    try:
        res = optimize.least_squares(
            _res, x0=[cx / scale, cy / scale, math.sqrt(r2) / scale], jac=_jac,
            method="lm", max_nfev=60)
        cx, cy, r = res.x[0] * scale, res.x[1] * scale, abs(res.x[2]) * scale
    except Exception:
        r = math.sqrt(r2)
    resid = float(np.sqrt(np.mean((np.abs(z - complex(cx, cy)) - r) ** 2)))
    return complex(cx, cy) + m, r, resid


def _parabolic(p: np.ndarray, i: int) -> float:
    if 0 < i < len(p) - 1:
        a, b, c = np.log(p[i - 1] + 1e-300), np.log(p[i] + 1e-300), np.log(p[i + 1] + 1e-300)
        den = a - 2 * b + c
        if den < 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


class VitalsAnalyzer:
    def __init__(self, fs: float, wavelength: float, breath_band=(0.1, 0.8),
                 heart_band=(0.8, 2.5), noise_band=(4.0, 12.0), nfft: int = 8192,
                 ) -> None:
        self.fs = float(fs)
        self.lam = float(wavelength)
        self.floor_guard_hz = 0.10
        self.floor_train_hz = 0.25
        self.floor_q = 50
        self.bb = tuple(breath_band)
        self.hb = tuple(heart_band)
        self.nb = tuple(noise_band)
        self.nfft = int(nfft)
        self.f = np.fft.rfftfreq(self.nfft, 1.0 / self.fs)
        self._bp = signal.butter(2, [0.08, 1.5], btype="band", fs=self.fs, output="sos")
        # passe-haut « anti-dérive » (dérive de phase LO, thermique) avant PCA
        self._hp = signal.butter(2, 0.06, btype="high", fs=self.fs, output="sos")
        # passe-haut « mouvement » : la respiration (et ses harmoniques) est < 3 Hz
        self._hp_mot = signal.butter(4, 3.0, btype="high", fs=self.fs, output="sos")
        self._win_cache: dict[int, tuple[np.ndarray, float]] = {}

    def reset(self) -> None:
        pass

    def _floor(self, P: np.ndarray, f: np.ndarray, n: int) -> tuple[float, np.ndarray]:
        """(plancher blanc scalaire, plancher coloré par bin) — même unité que P.

        Plancher coloré « CFAR en anneau » : pour chaque bin, médiane de P sur
        [f−guard−train, f−guard] ∪ [f+guard, f+guard+train].  La bande de garde
        (≥ lobe principal de la fenêtre de Hann, 2/T) empêche la raie
        respiratoire de gonfler son propre plancher — défaut constaté sur le
        vrai Pluto le 27/09 (SNR plafonné à ~6 dB pour une raie à +25 dB).
        Sur une pente de dérive convexe, la moyenne des deux côtés surestime
        légèrement le plancher : biais conservateur.
        """
        nm = (f >= self.nb[0]) & (f <= self.nb[1])
        Nw = max(float(np.median(P[nm]) / LN2), 1e-30)
        lim = int(np.searchsorted(f, 4.0))
        df = f[1] - f[0]
        guard = max(self.floor_guard_hz, 2.2 * self.fs / n)
        g = int(round(guard / df))
        k = g + max(3, int(round(self.floor_train_hz / df)))
        foot = np.ones(2 * k + 1, dtype=bool)
        foot[k - g:k + g + 1] = False
        q = self.floor_q
        med = ndimage.percentile_filter(P[:lim], q, footprint=foot, mode="mirror") / (
            -math.log(1 - q / 100))
        Nf = np.full_like(P, Nw)
        Nf[:lim] = np.maximum(med, Nw)
        return Nw, Nf

    def _psd(self, s: np.ndarray) -> np.ndarray:
        n = len(s)
        if n not in self._win_cache:
            w = signal.windows.hann(n, sym=False)
            self._win_cache[n] = (w, float(np.sum(w * w)))
        w, ws = self._win_cache[n]
        X = np.fft.rfft((s - s.mean()) * w, n=max(self.nfft, n))
        f_len = len(X)
        if f_len != len(self.f):
            self.f = np.fft.rfftfreq(max(self.nfft, n), 1.0 / self.fs)
        return (np.abs(X) ** 2) / (self.fs * ws)

    # ------------------------------------------------------------------
    def analyze(self, x: np.ndarray, t: float = 0.0, want_display: bool = True,
                full: bool = True) -> tuple[Features, Display | None]:
        """*full=False* : features de détection seules (pas de cercle, d'amplitude
        en mm ni de cœur) — échelles courtes du détecteur multi-échelle."""
        x = np.asarray(x, dtype=np.complex128)
        fs = self.fs
        feats = Features(t=t)
        feats.static_dbfs = 20 * math.log10(abs(x.mean()) + 1e-15)

        xd = _detrend_complex(x)
        xhp = signal.sosfiltfilt(self._hp, xd)
        _, lin, _ = _pca_project(xhp)
        feats.linearity = lin

        # Base (radiale, tangentielle) relative au clutter statique C.  Une
        # dérive de phase LO est une rotation commune de la constellation →
        # bruit *tangentiel* coloré (1/f²) ; la voie radiale y est insensible.
        c_mean = x.mean()
        uc = c_mean / abs(c_mean) if abs(c_mean) > 0 else 1.0 + 0j
        z = xhp * np.conj(uc)
        ch_r, ch_t = z.real, z.imag

        # --- démodulation arc-tangente (si un arc est identifiable) -------
        disp = None
        circ = fit_circle(x) if full else None
        if circ is not None:
            c, r, res = circ
            ph = np.unwrap(np.angle(x - c))
            span = float(np.ptp(ph))
            spread = float(np.sqrt(np.mean(np.abs(xd) ** 2)))
            if (res < 0.35 * r and 0.3 < span < 2 * np.pi and r < 20 * spread
                    and lin > 0.6):
                feats.arc_rad = span
                d = -self.lam / (4 * np.pi) * signal.detrend(ph) * 1e3   # mm
                disp = d
            else:
                circ = None

        # --- détecteur spectral 2 voies à plancher coloré ------------------
        # Chaque voie est normalisée par son plancher local (max du plancher
        # blanc 4–12 Hz et d'une médiane glissante ±0.3 Hz) puis les deux
        # voies sont sommées de façon incohérente : invariant à l'orientation
        # de la cible dans le plan IQ, robuste à la dérive (colorée).
        f = None
        Pn = None
        N0s = []
        chans = []
        for ch in (ch_r, ch_t):
            P = self._psd(ch)
            f = self.f
            Nw, Nf = self._floor(P, f, len(ch))
            N0s.append(Nw)
            chans.append((P, Nf))
            Pn = P / Nf if Pn is None else Pn + P / Nf
        Pn = 0.5 * Pn
        P_sum = chans[0][0] + chans[1][0]
        N0 = N0s[0] + N0s[1]              # plancher blanc de la somme des voies
        feats.noise_dbfs = 10 * math.log10(N0 * fs + 1e-30)
        feats.drift_db = 10 * math.log10(
            float(np.mean(chans[1][1][(f >= 0.1) & (f <= 0.3)]))
            / float(np.mean(chans[0][1][(f >= 0.1) & (f <= 0.3)])))

        bm = (f >= self.bb[0]) & (f <= self.bb[1])
        idx_b = np.flatnonzero(bm)
        # vrai maximum local uniquement (un bord de bande sur une pente n'est pas un pic)
        loc = idx_b[(Pn[idx_b] >= Pn[np.maximum(idx_b - 1, 0)]) &
                    (Pn[idx_b] >= Pn[np.minimum(idx_b + 1, len(Pn) - 1)])]
        cand = loc if loc.size else idx_b
        i_pk = int(cand[np.argmax(Pn[cand])])
        df = f[1] - f[0]
        f_pk = f[i_pk] + _parabolic(Pn, i_pk) * df
        # sous-harmonique : si f/2 est dans la bande et porte de l'énergie,
        # c'est la vraie fondamentale (démodulation non linéaire, grand arc)
        half = f_pk / 2
        if half >= self.bb[0]:
            j = int(np.argmin(np.abs(f - half)))
            lo, hi = max(j - 3, 0), j + 4
            jj = lo + int(np.argmax(Pn[lo:hi]))
            if Pn[jj] > 0.3 * Pn[i_pk] and Pn[jj] > 4.0:
                i_pk, f_pk = jj, f[jj] + _parabolic(Pn, jj) * df
        feats.snr_db = 10 * math.log10(max(float(Pn[i_pk]), 1e-12))
        feats.breath_hz = float(f_pk)

        # concentration : part de l'énergie *en excès du plancher* (0.05–3 Hz)
        # dans le pic et ses harmoniques ; régularisée pour valoir ~0 sur du bruit
        tot_m = (f >= 0.05) & (f <= 3.0)
        ex = np.maximum(Pn - 1.0, 0.0)
        half_w = max(0.03, 1.5 * fs / len(ch_r))
        e_pk = 0.0
        for h in (1, 2, 3):
            fh = h * f_pk
            if fh > 3.0:
                break
            e_pk += float(np.sum(ex[(f >= fh - half_w) & (f <= fh + half_w) & tot_m]))
        e_tot = float(np.sum(ex[tot_m])) + 0.5 * float(np.sum(tot_m))
        feats.concentration = float(min(e_pk / e_tot, 1.0))
        # énergie 0.05–3 Hz « anormale » : voie radiale vs plancher blanc (elle
        # est insensible à la dérive LO), voie tangentielle vs plancher coloré
        band_lin = 0.5 * (chans[0][0][tot_m] / N0s[0] + chans[1][0][tot_m] / chans[1][1][tot_m])
        feats.band_db = 10 * math.log10(float(np.mean(band_lin)))

        # voie dominante au pic (pour l'ACF et l'affichage « u.a. »)
        s_best = ch_r if chans[0][0][i_pk] / chans[0][1][i_pk] >= chans[1][0][i_pk] / chans[1][1][i_pk] else ch_t

        # --- ACF (périodicité temporelle) ---------------------------------
        sb = signal.sosfiltfilt(self._bp, s_best)
        feats.acf = self._acf_peak(sb)

        # --- mouvement : bouffées d'énergie > 3 Hz (tranches de 1 s) --------
        xh = signal.sosfiltfilt(self._hp_mot, xd)
        seg = int(fs)
        k = len(xh) // seg
        if k >= 4:
            e = np.mean(np.abs(xh[: k * seg].reshape(k, seg)) ** 2, axis=1)
            feats.motion = float(np.max(e) / max(np.median(e), 1e-30))

        # --- amplitude & cœur (si déplacement en mm disponible) -----------
        if disp is not None:
            db = signal.sosfiltfilt(self._bp, disp)
            feats.disp_mm_pp = float(np.percentile(db, 98) - np.percentile(db, 2))
            feats.heart_hz, feats.heart_snr_db = self._heart(disp, f_pk)

        disp_out = None
        if want_display:
            disp_out = self._display(Pn, 1.0, s_best if disp is None else disp,
                                     "mm" if disp is not None else "u.a.", x, circ)
        return feats, disp_out

    # ------------------------------------------------------------------
    def _acf_peak(self, sb: np.ndarray) -> float:
        n = len(sb)
        sb = sb - sb.mean()
        nfft = 1 << (2 * n - 1).bit_length()
        S = np.fft.rfft(sb, nfft)
        r = np.fft.irfft(np.abs(S) ** 2, nfft)[:n]
        if r[0] <= 0:
            return 0.0
        r = r / r[0]                               # ACF biaisée (conservatrice)
        lo = max(1, int(self.fs / self.bb[1]))
        hi = min(n // 2, int(self.fs / self.bb[0]))
        if hi <= lo:
            return 0.0
        return float(max(0.0, np.max(r[lo:hi + 1])))

    def _heart(self, disp: np.ndarray, f_b: float) -> tuple[float | None, float]:
        """Cœur (expérimental) : retire les harmoniques respiratoires puis cherche un pic."""
        n = len(disp)
        tt = np.arange(n) / self.fs
        cols = [np.ones(n), tt]
        for h in range(1, 7):
            cols += [np.cos(2 * np.pi * h * f_b * tt), np.sin(2 * np.pi * h * f_b * tt)]
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, disp, rcond=None)
        res = disp - A @ coef
        P = self._psd(res)
        f = self.f
        m = (f >= self.hb[0]) & (f <= self.hb[1])
        # exclut les voisinages des harmoniques respiratoires
        for h in range(1, 8):
            m &= np.abs(f - h * f_b) > 0.06
        nm = (f >= self.nb[0]) & (f <= self.nb[1])
        if not np.any(m) or not np.any(nm):
            return None, 0.0
        N0 = float(np.median(P[nm]) / LN2) + 1e-30
        i = np.flatnonzero(m)[np.argmax(P[m])]
        snr = 10 * math.log10(float(P[i]) / N0)
        return (float(f[i]) if snr > 10 else None), snr

    def _display(self, P, N0, wave, unit, x, circ) -> Display:
        f = self.f
        m = f <= 3.0
        fd, pd = f[m], 10 * np.log10(P[m] / N0 + 1e-12)
        step = max(1, len(fd) // 300)
        spec_f = fd[::step]
        spec_db = np.maximum.reduceat(pd, np.arange(0, len(pd), step))[: len(spec_f)]
        wv = signal.sosfiltfilt(self._bp, wave) if unit == "mm" else signal.sosfiltfilt(self._bp, wave)
        ds = max(1, int(self.fs // 10))
        wave_d = wv[::ds]
        wave_t = (np.arange(len(wave_d)) * ds - len(wv)) / self.fs
        m0 = x.mean()
        z = x - m0
        zi = z[:: max(1, len(z) // 300)]
        c = None
        if circ is not None:
            cc, r, _ = circ
            c = (float((cc - m0).real), float((cc - m0).imag), float(r))
        return Display(spec_f=spec_f, spec_db=spec_db, wave_t=wave_t, wave=wave_d,
                       wave_unit=unit, iq=np.column_stack((zi.real, zi.imag)), circle=c)
