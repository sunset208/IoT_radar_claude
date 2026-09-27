"""Simulation SFCW : scène physique + faux Pluto (tests du code matériel sans matériel).

:class:`SfcwScene` donne la réponse complexe S(f, t) d'une scène : fuite TX→RX
(référence), réflecteurs fixes, une personne qui respire, un marcheur ; le
tout vu à travers une réponse d'antenne résonante (antennes Pluto d'origine :
0.9 et ~1.85 GHz).  À chaque changement de LO, une **phase aléatoire**
commune multiplie S (comportement de l'AD936x) — le traitement ne doit pas y
être sensible.

:class:`FakePluto` imite la partie de l'interface ``adi.Pluto`` (pyadi-iio)
utilisée par :mod:`radar.sfcw.sources` et :mod:`radar.sfcw.bench` :
latence d'écriture des LO, buffers noyau périmés après un changement de
fréquence (dont un « à cheval »), tonalité hors bande tant que TX et RX
ne sont pas sur la même fréquence, bruit, offset DC.  Horloge virtuelle :
les tests tournent vite, les durées restent réalistes.
"""

from __future__ import annotations

import dataclasses as dc
import math

import numpy as np

from radar.config import SPEED_OF_LIGHT
from radar.scene import Breathing


@dc.dataclass
class SfcwSceneParams:
    scenario: str = "breathing"            # empty | breathing | walker | breathing_walker
    target_range_m: float = 1.2
    target_rel_db: float = -35.0           # |T|/|L| (écho de la personne / fuite)
    breath_rate_bpm: float = 15.0
    breath_depth_mm: float = 6.0
    statics: tuple = ((0.7, -30.0), (2.4, -28.0), (3.3, -33.0))   # (distance m, dB / fuite)
    walker_rel_db: float = -30.0
    leak_delay_ns: float = 0.4
    amp_noise_db: float = 0.005            # répétabilité d'amplitude par pas (rms)
    common_noise_db: float = 0.01          # variation commune à un balayage (puissance TX…)
    thermal_rel_db: float = -60.0          # bruit thermique sur l'estimation, relatif à la fuite
    antenna: bool = True


def antenna_gain(f: np.ndarray) -> np.ndarray:
    """|G(f)| d'une antenne fouet bi-bande (résonances ~0.9 et ~1.85 GHz), normalisée."""
    f = np.asarray(f, dtype=np.float64) / 1e9
    g = 0.15 + 1.0 / (1 + ((f - 1.85) / 0.22) ** 2) + 0.7 / (1 + ((f - 0.9) / 0.06) ** 2)
    return g / 1.2


class SfcwScene:
    def __init__(self, p: SfcwSceneParams | None = None, rng: np.random.Generator | None = None,
                 wavelength: float = 0.1665) -> None:
        self.p = p or SfcwSceneParams()
        self.rng = rng or np.random.default_rng()
        sc = self.p.scenario
        self.breathing = (Breathing(self.rng, self.p.breath_rate_bpm, self.p.breath_depth_mm * 1e-3)
                          if sc in ("breathing", "breathing_walker") else None)
        self.walker = sc in ("walker", "breathing_walker")
        self._w_phase = float(self.rng.uniform(0, 2 * np.pi))
        self._bt = -1.0
        self._bd = 0.0

    @property
    def truth(self) -> dict:
        return {"scenario": self.p.scenario,
                "target_range_m": self.p.target_range_m if self.breathing is not None else None,
                "breath_rate_bpm": self.p.breath_rate_bpm if self.breathing is not None else None}

    def _disp(self, t: float) -> float:
        if self.breathing is None:
            return 0.0
        if t > self._bt:                      # le générateur exige des instants croissants
            self._bd = float(self.breathing(np.array([t]))[0])
            self._bt = t
        return self._bd

    def response(self, f: np.ndarray, t: float) -> np.ndarray:
        """S(f, t) complexe, fuite = 1 (hors gain d'antenne et phase aléatoire)."""
        f = np.asarray(f, dtype=np.float64)
        p = self.p
        tau_l = p.leak_delay_ns * 1e-9
        s = np.exp(-2j * np.pi * f * tau_l)
        for r, db in p.statics:
            s = s + 10 ** (db / 20) * np.exp(-2j * np.pi * f * (tau_l + 2 * r / SPEED_OF_LIGHT))
        if self.breathing is not None:
            r = p.target_range_m + self._disp(t)
            s = s + 10 ** (p.target_rel_db / 20) * np.exp(-2j * np.pi * f * (tau_l + 2 * r / SPEED_OF_LIGHT))
        if self.walker:
            r = 2.5 + 1.0 * math.sin(2 * np.pi * t / 5.0 + self._w_phase)   # 1.5–3.5 m, ~0.8 m/s
            s = s + 10 ** (p.walker_rel_db / 20) * np.exp(-2j * np.pi * f * (tau_l + 2 * r / SPEED_OF_LIGHT))
        if p.antenna:
            s = s * antenna_gain(f) ** 2
        return s

    def sweep(self, freqs: np.ndarray, t0: float, step_dt: float,
              order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Balayage mesuré en amplitude : (|S_k| dans l'ordre des fréquences, instants)."""
        p = self.p
        n = len(freqs)
        order = np.arange(n) if order is None else order
        amps = np.empty(n)
        tk = np.empty(n)
        common = 1 + (10 ** (p.common_noise_db / 20) - 1) * self.rng.standard_normal()
        for i, k in enumerate(order):
            t = t0 + i * step_dt
            s = self.response(freqs[k:k + 1], t)[0] * np.exp(1j * self.rng.uniform(0, 2 * np.pi))
            s *= common * (1 + (10 ** (p.amp_noise_db / 20) - 1) * self.rng.standard_normal())
            noise = 10 ** (p.thermal_rel_db / 20) * (self.rng.standard_normal()
                                                     + 1j * self.rng.standard_normal()) / math.sqrt(2)
            amps[k] = abs(s + noise * (antenna_gain(freqs[k]) ** 2 if p.antenna else 1.0))
            tk[k] = t
        return amps, tk


# ----------------------------------------------------------------------
# Faux Pluto
# ----------------------------------------------------------------------

class _Attr:
    def __init__(self, value: str) -> None:
        self.value = value


class _Ctrl:
    def __init__(self) -> None:
        self.attrs = {"calib_mode": _Attr("auto")}


class _RxAdc:
    def __init__(self, owner: "FakePluto") -> None:
        self.owner = owner

    def set_kernel_buffers_count(self, n: int) -> None:
        self.owner.kernel_buffers = int(n)


class FakePluto:
    """Imite ``adi.Pluto`` pour le mode SFCW (sous-ensemble utilisé ici)."""

    ADC_FS = 2048.0

    def __init__(self, scene: SfcwScene | None = None, write_latency_s: float = 1.5e-3,
                 vco_cal_s: float = 0.4e-3, usb_latency_s: float = 0.3e-3,
                 leak_dbfs: float = -12.0, noise_dbfs_hz: float = -130.0,
                 rng: np.random.Generator | None = None) -> None:
        self.scene = scene or SfcwScene()
        self.rng = rng or np.random.default_rng()
        self.write_latency_s = write_latency_s
        self.vco_cal_s = vco_cal_s
        self.usb_latency_s = usb_latency_s
        self.leak_amp = 10 ** (leak_dbfs / 20)
        self.noise_dbfs_hz = noise_dbfs_hz
        self.t = 0.0                           # horloge virtuelle (s)
        self.sample_rate = 4_000_000
        self.rx_rf_bandwidth = 4_000_000
        self.tx_rf_bandwidth = 4_000_000
        self.gain_control_mode_chan0 = "manual"
        self.rx_hardwaregain_chan0 = 20.0
        self.tx_hardwaregain_chan0 = 0.0
        self.rx_buffer_size = 1024
        self.tx_cyclic_buffer = True
        self.kernel_buffers = 4
        self._ctrl = _Ctrl()
        self._rxadc = _RxAdc(self)
        self._tx_lo = 1.8e9
        self._rx_lo = 1.8e9
        self._tone = 0.0
        self._tone_amp = 0.0
        self._state = (self._tx_lo, self._rx_lo, self._new_phase())
        self._stale: list = []                 # états « anciens » encore en file
        self._last_rx_t = 0.0
        self.n_writes = 0
        self.n_reads = 0

    def now(self) -> float:
        return self.t

    def _new_phase(self) -> float:
        return float(self.rng.uniform(0, 2 * np.pi))

    # --- LO -----------------------------------------------------------
    def _retune(self, tx: float | None = None, rx: float | None = None) -> None:
        old = self._state
        self.t += self.write_latency_s + self.vco_cal_s
        if tx is not None:
            self._tx_lo = float(tx)
        if rx is not None:
            self._rx_lo = float(rx)
        self._state = (self._tx_lo, self._rx_lo, self._new_phase())
        # file noyau : jusqu'à K buffers remplis avec l'ancien état + 1 à cheval
        buf_s = self.rx_buffer_size / self.sample_rate
        n_old = min(self.kernel_buffers, int((self.t - self._last_rx_t) / buf_s))
        self._stale = [old] * max(n_old, 0) + [("mix", old, self._state)]
        self.n_writes += 1

    @property
    def tx_lo(self) -> int:
        return int(self._tx_lo)

    @tx_lo.setter
    def tx_lo(self, f) -> None:
        self._retune(tx=f)

    @property
    def rx_lo(self) -> int:
        return int(self._rx_lo)

    @rx_lo.setter
    def rx_lo(self, f) -> None:
        self._retune(rx=f)

    # --- IIO bas niveau (désactivation du tracking…) --------------------
    def _set_iio_attr(self, chan, attr, output, value) -> None:
        pass

    # --- TX / RX -------------------------------------------------------
    def tx(self, data: np.ndarray) -> None:
        data = np.asarray(data)
        X = np.fft.fft(data)
        k = int(np.argmax(np.abs(X)))
        f = np.fft.fftfreq(len(data), 1.0 / self.sample_rate)[k]
        self._tone = float(f)
        self._tone_amp = float(np.abs(X[k]) / len(data) / 2 ** 14)

    def _buffer(self, state, n: int, t0: float) -> np.ndarray:
        f_tx, f_rx, ph = state
        fs = self.sample_rate
        k = np.arange(n)
        f_bb = self._tone + (f_tx - f_rx)
        s = self.scene.response(np.array([f_tx + self._tone]), t0)[0]
        amp = self.leak_amp * (self._tone_amp / 0.5) * s * np.exp(1j * ph)
        if abs(f_bb) > 0.45 * self.rx_rf_bandwidth:
            amp *= 1e-4                        # hors bande (filtre analogique + numérique)
            f_bb = math.copysign(0.45 * fs, f_bb)
        x = amp * np.exp(2j * np.pi * f_bb * k / fs)
        sigma = math.sqrt(10 ** (self.noise_dbfs_hz / 10) * fs / 2)
        x = x + sigma * (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n))
        x = x + 10 ** (-40 / 20)               # offset DC du récepteur
        return x

    def rx(self) -> np.ndarray:
        n = int(self.rx_buffer_size)
        buf_s = n / self.sample_rate
        self.t += self.usb_latency_s
        t0 = self.t
        if self._stale:
            st = self._stale.pop(0)
            if st[0] == "mix":                 # buffer à cheval sur le changement
                h = n // 2
                x = np.concatenate((self._buffer(st[1], h, t0), self._buffer(st[2], n - h, t0)))
            else:
                x = self._buffer(st, n, t0)
        else:
            self.t += buf_s
            x = self._buffer(self._state, n, t0)
        self._last_rx_t = self.t
        self.n_reads += 1
        return np.clip(np.round(x.real * self.ADC_FS), -2048, 2047) + 1j * np.clip(
            np.round(x.imag * self.ADC_FS), -2048, 2047)

    def tx_destroy_buffer(self) -> None:
        pass

    def rx_destroy_buffer(self) -> None:
        pass
