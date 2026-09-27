"""Sources simulées.

* :class:`SimulatedRawSource` — simule l'IQ **brut** du Pluto (f_s, comptes ADC,
  quantification, offset DC récepteur, porteuse à f_offset) puis applique le
  vrai front-end : teste toute la chaîne.
* :class:`SimulatedSlowSource` — génère directement le slow-time (rapide ;
  utilisé pour les tests et la génération de données IA).
"""

from __future__ import annotations

import logging
import math
from typing import Iterator

import numpy as np
from scipy import signal

from radar.config import Config
from radar.dsp.frontend import ADC_FULL_SCALE, Frontend, adc_level_dbfs
from radar.scene import BREATHING_SCENARIOS, Scene, SceneParams, dbfs_to_amp, scene_from_config
from radar.sources.base import Chunk, Source, Throttle

logger = logging.getLogger(__name__)

_CTRL_RATE = 1000.0   # la scène est évaluée à 1 kHz puis interpolée (écho ≪ 100 Hz)


def _truth(scene: Scene) -> dict:
    p = scene.p
    return {
        "scenario": p.scenario,
        "breathing": p.scenario in BREATHING_SCENARIOS,
        "breath_rate_bpm": p.breath_rate_bpm if scene.has_breathing else None,
        "heart_rate_bpm": p.heart_rate_bpm if scene.has_breathing else None,
    }


class SimulatedRawSource(Source):
    kind = "simulation-rf"

    def __init__(self, cfg: Config, scenario: str | None = None,
                 realtime: bool | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        s = cfg.simulation
        self.rng = np.random.default_rng(s.seed)
        self.scene = scene_from_config(s, cfg.sdr.wavelength, self.rng, scenario)
        self.fs_slow = cfg.frontend.fs_slow
        self.realtime = s.realtime if realtime is None else realtime
        self.frontend = Frontend(cfg.sdr.f_s, cfg.emission.f_offset, cfg.frontend.fs_slow,
                                 cfg.sdr.rx_buffer_size, cfg.frontend.cic_order)
        # Bruit thermique brut tel que le bruit en slow-time corresponde à snr_db
        # (le filtrage garde ≈ fs_slow·0.8 Hz de bande sur f_s).
        noise_slow = self.scene.noise_std
        self.noise_raw_std = noise_slow * math.sqrt(cfg.sdr.f_s / (0.8 * self.fs_slow))
        self.rx_dc = dbfs_to_amp(s.rx_dc_dbfs) * np.exp(1j * self.rng.uniform(0, 2 * np.pi))
        self.truth = _truth(self.scene)

    def describe(self) -> dict:
        return {"kind": self.kind, "fs_slow": self.fs_slow, "scenario": self.scene.p.scenario}

    def _iter(self) -> Iterator[Chunk]:
        cfg = self.cfg
        f_s, f_off, N = cfg.sdr.f_s, cfg.emission.f_offset, cfg.sdr.rx_buffer_size
        thr = Throttle(self.realtime)
        n0 = 0
        slow_count = 0
        # La scène est évaluée sur une grille globale à 1 kHz (instants
        # strictement croissants d'un appel à l'autre), puis interpolée.
        step = int(round(f_s / _CTRL_RATE))
        k_next = 0
        tc_hist = np.empty(0)
        ec_hist = np.empty(0, dtype=np.complex128)
        while not self._stop:
            n = np.arange(n0, n0 + N)
            t = n / f_s
            k_end = (n0 + N) // step + 2
            tc_new = np.arange(k_next, k_end) * step / f_s
            k_next = k_end
            tc = np.concatenate((tc_hist, tc_new))
            ec = np.concatenate((ec_hist, self.scene.echo(tc_new)))
            tc_hist, ec_hist = tc[-2:], ec[-2:]
            echo = np.interp(t, tc, ec.real) + 1j * np.interp(t, tc, ec.imag)
            carrier = np.exp(2j * np.pi * f_off * (n % int(round(f_s / f_off))) / f_s)
            x = echo * carrier + self.rx_dc
            x += self.noise_raw_std * (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) / math.sqrt(2)
            # ADC 12 bits : quantification + écrêtage
            raw = np.clip(np.round(x.real * ADC_FULL_SCALE), -2048, 2047) + 1j * np.clip(
                np.round(x.imag * ADC_FULL_SCALE), -2048, 2047)
            slow = self.frontend(raw)
            rms, peak = adc_level_dbfs(raw)
            t_slow0 = slow_count / self.fs_slow
            slow_count += len(slow)
            n0 += N
            thr.wait_until(n0 / f_s)
            yield Chunk(slow=slow, t0=t_slow0,
                        raw=raw.astype(np.complex64) if cfg.recording.save_raw else None,
                        stats={"adc_rms_dbfs": rms, "adc_peak_dbfs": peak})


class SimulatedSlowSource(Source):
    """Slow-time direct (sans RF).  Bien plus rapide ; mêmes statistiques."""

    kind = "simulation"

    def __init__(self, fs_slow: float, scene: Scene, chunk_s: float = 0.1,
                 realtime: bool = False, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        self.fs_slow = fs_slow
        self.scene = scene
        self.chunk = max(1, int(round(chunk_s * fs_slow)))
        self.realtime = realtime
        self.rng = rng or np.random.default_rng()
        self.truth = _truth(scene)
        # sur-échantillonnage ×4 puis décimation (évite le repliement du marcheur)
        self._os = 4
        self._sos = signal.ellip(8, 0.05, 80, 0.4 * fs_slow, fs=fs_slow * self._os, output="sos")
        self._zi = None

    def describe(self) -> dict:
        return {"kind": self.kind, "fs_slow": self.fs_slow, "scenario": self.scene.p.scenario}

    def generate(self, n_slow: int, start: int = 0) -> np.ndarray:
        fs_hi = self.fs_slow * self._os
        k = np.arange(start * self._os, (start + n_slow) * self._os)
        echo = self.scene.echo(k / fs_hi)
        noise = self.scene.noise_std * math.sqrt(self._os / 0.8) * (
            self.rng.standard_normal(k.size) + 1j * self.rng.standard_normal(k.size)) / math.sqrt(2)
        sig = echo + noise
        if self._zi is None:
            self._zi = signal.sosfilt_zi(self._sos).astype(np.complex128) * echo[0]
        y, self._zi = signal.sosfilt(self._sos, sig, zi=self._zi)
        return y[self._os - 1::self._os].astype(np.complex64)

    def _iter(self) -> Iterator[Chunk]:
        thr = Throttle(self.realtime)
        i = 0
        while not self._stop:
            slow = self.generate(self.chunk, i)
            t0 = i / self.fs_slow
            i += self.chunk
            thr.wait_until(i / self.fs_slow)
            yield Chunk(slow=slow, t0=t0, stats={})


def simulate_slow(scenario: str, duration_s: float, fs_slow: float, wavelength: float,
                  rng: np.random.Generator, params: SceneParams | None = None,
                  **overrides) -> tuple[np.ndarray, dict]:
    """Génère *duration_s* de slow-time pour un scénario.  Retourne (x, vérité)."""
    p = params or SceneParams(scenario=scenario, wavelength=wavelength)
    for k, v in overrides.items():
        setattr(p, k, v)
    p.scenario = scenario
    scene = Scene(p, rng)
    src = SimulatedSlowSource(fs_slow, scene, rng=rng)
    n = int(round(duration_s * fs_slow))
    # préchauffe du filtre de décimation
    src.generate(int(fs_slow))
    return src.generate(n, int(fs_slow)), src.truth
