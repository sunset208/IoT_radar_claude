"""Sources SFCW : Pluto, simulation, relecture ; enregistrement des balayages.

Mesure d'un pas sur le Pluto
----------------------------
1. écrire LO TX = f_k, puis LO RX = f_k (+ ``tag_offset_hz`` un pas sur deux) ;
2. lire des buffers jusqu'à en obtenir un **frais** ;
3. amplitude de la tonalité par corrélation (bin FFT exact).

Détection des buffers périmés (étape 2) : après un changement de LO, la file
noyau contient des buffers remplis avant le changement, plus un « à cheval ».
Comme le LO RX est décalé de ``tag_offset_hz`` un pas sur deux, la tonalité
tombe alternativement à ``tone_hz`` et ``tone_hz − tag_offset_hz`` : un buffer
n'est accepté que si la tonalité attendue domine l'autre de 26 dB et le
plancher de 30 dB.  Pas de délai fixe « au jugé » : on lit exactement ce
qu'il faut (et ``radar sfcw bench`` mesure combien).  Sans marquage
(``tag_offset_hz = 0``), on jette ``discard`` buffers d'office.
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from radar.config import Config, SfcwCfg
from radar.sfcw.dsp import Sweep
from radar.sources.base import Throttle

logger = logging.getLogger(__name__)

ADC_FS = 2048.0


def sweep_order(n: int, index: int, zigzag: bool) -> np.ndarray:
    o = np.arange(n)
    return o[::-1] if (zigzag and index % 2 == 1) else o


class LoControl:
    """Écriture rapide des LO (canaux IIO en cache ; repli sur les propriétés pyadi)."""

    def __init__(self, sdr) -> None:
        self.sdr = sdr
        self._rx = self._tx = None
        try:
            self._rx = sdr._ctrl.find_channel("altvoltage0", True).attrs["frequency"]
            self._tx = sdr._ctrl.find_channel("altvoltage1", True).attrs["frequency"]
        except Exception:
            pass

    def set_tx(self, f: float) -> None:
        if self._tx is not None:
            self._tx.value = str(int(round(f)))
        else:
            self.sdr.tx_lo = int(round(f))

    def set_rx(self, f: float) -> None:
        if self._rx is not None:
            self._rx.value = str(int(round(f)))
        else:
            self.sdr.rx_lo = int(round(f))


def tone_bins(sc: SfcwCfg) -> tuple[int, int]:
    n = sc.rx_buffer_size
    b0 = int(round(sc.tone_hz / sc.f_s * n)) % n
    b1 = int(round((sc.tone_hz - sc.tag_offset_hz) / sc.f_s * n)) % n
    return b0, b1


def _half_amps(x: np.ndarray, b: int) -> tuple[float, float]:
    """|amplitude| de la tonalité (bin *b*) dans chaque moitié du buffer."""
    n = len(x)
    h = n // 2
    e = np.exp(-2j * np.pi * b * np.arange(h) / n)
    return abs(np.dot(x[:h], e)), abs(np.dot(x[h:2 * h], e))


def is_fresh(x: np.ndarray, b_exp: int, b_oth: int) -> tuple[bool, complex]:
    """Buffer frais ?  (tonalité marquée attendue ≫ l'autre et ≫ plancher, et
    stable entre les deux moitiés du buffer — un buffer « à cheval » sur le
    changement de LO a des moitiés différentes)."""
    X = np.fft.fft(x)
    a = np.abs(X)
    floor = float(np.median(a)) + 1e-9
    if not (a[b_exp] > 20.0 * a[b_oth] and a[b_exp] > 30.0 * floor):
        return False, X[b_exp]
    a1, a2 = _half_amps(x, b_exp)
    return abs(a1 - a2) < 0.1 * max(a1, a2, 1e-9), X[b_exp]


def read_fresh(sdr, sc: SfcwCfg, tagged: bool, bins: tuple[int, int]) -> tuple[complex, int, float]:
    """Lit jusqu'à un buffer frais.  Renvoie (amplitude complexe FS, nb de lectures, crête dBFS)."""
    n = sc.rx_buffer_size
    b_exp, b_oth = (bins[1], bins[0]) if tagged else (bins[0], bins[1])
    if sc.tag_offset_hz <= 0:
        for _ in range(sc.discard):
            sdr.rx()
        x = np.asarray(sdr.rx())
        X = np.fft.fft(x)
        return X[b_exp] / n / ADC_FS, sc.discard + 1, _peak_dbfs(x)
    for i in range(sc.max_reads):
        x = np.asarray(sdr.rx())
        ok, v = is_fresh(x, b_exp, b_oth)
        if ok:
            return v / n / ADC_FS, i + 1, _peak_dbfs(x)
    return complex("nan"), sc.max_reads, _peak_dbfs(x)


def _peak_dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(float(np.max(np.abs(x))) / ADC_FS + 1e-12)


def open_pluto_sfcw(cfg: Config, sdr=None):
    """Ouvre (ou reçoit, pour les tests) le Pluto et le configure pour le SFCW.

    Renvoie ``(sdr, restaurer)`` : ``restaurer()`` remet ``calib_mode``.
    """
    from radar.sources.pluto import (check_lo, connect_pluto, disable_tracking,
                                     set_calib_mode, set_kernel_buffers)
    sc = cfg.sfcw
    if sdr is None:
        sdr = connect_pluto(cfg.sdr.uri)
    freqs = sc.freqs
    check_lo(sdr, [freqs[0], freqs[-1] + sc.tag_offset_hz])
    old_cal = set_calib_mode(sdr, sc.calib_mode)
    sdr.sample_rate = int(sc.f_s)
    sdr.rx_rf_bandwidth = int(sc.rf_bandwidth)
    sdr.tx_rf_bandwidth = int(sc.rf_bandwidth)
    sdr.tx_lo = int(freqs[0])
    sdr.rx_lo = int(freqs[0])
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(sc.rx_gain_db)
    sdr.tx_hardwaregain_chan0 = float(sc.tx_atten_db)
    sdr.rx_buffer_size = int(sc.rx_buffer_size)
    disable_tracking(sdr)
    set_kernel_buffers(sdr, sc.kernel_buffers)

    def restore() -> None:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        if old_cal:
            set_calib_mode(sdr, old_cal)
    return sdr, restore


def start_tone(sdr, sc: SfcwCfg) -> None:
    from radar.sources.pluto import make_tx_waveform
    tx = make_tx_waveform(sc.f_s, sc.tone_hz, 4 * sc.rx_buffer_size, sc.amplitude)
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx)


# ----------------------------------------------------------------------

class SfcwSource:
    kind = "sfcw"
    truth: dict | None = None

    def __init__(self) -> None:
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def sweeps(self) -> Iterator[Sweep]:  # pragma: no cover - abstrait
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind}


class SfcwPlutoSource(SfcwSource):
    kind = "sfcw-pluto"

    def __init__(self, cfg: Config, sdr=None, clock=None) -> None:
        super().__init__()
        self.cfg = cfg
        self._sdr_in = sdr
        self.clock = clock or (getattr(sdr, "now", None) if sdr is not None else None) or time.perf_counter
        self.timeouts = 0

    def describe(self) -> dict:
        sc = self.cfg.sfcw
        return {"kind": self.kind, "uri": self.cfg.sdr.uri, "f_start": sc.f_start,
                "f_stop": sc.f_stop, "n_steps": sc.n_steps, "rx_gain_db": sc.rx_gain_db,
                "tx_atten_db": sc.tx_atten_db}

    def sweeps(self) -> Iterator[Sweep]:
        cfg, sc = self.cfg, self.cfg.sfcw
        sdr, restore = open_pluto_sfcw(cfg, self._sdr_in)
        try:
            start_tone(sdr, sc)
            for _ in range(3):
                sdr.rx()
            lo = LoControl(sdr)
            freqs = sc.freqs
            bins = tone_bins(sc)
            g = 0                        # compteur global de pas (parité du marquage)
            i_sw = 0
            clk = self.clock
            while not self._stop:
                order = sweep_order(sc.n_steps, i_sw, sc.zigzag)
                amps = np.full(sc.n_steps, np.nan)
                tk = np.zeros(sc.n_steps)
                reads = 0
                peak = -200.0
                t_a = clk()
                for k in order:
                    tagged = sc.tag_offset_hz > 0 and g % 2 == 1
                    lo.set_tx(freqs[k])
                    lo.set_rx(freqs[k] + (sc.tag_offset_hz if tagged else 0.0))
                    a, nr, pk = read_fresh(sdr, sc, tagged, bins)
                    amps[k] = abs(a)
                    tk[k] = clk()
                    reads += nr
                    peak = max(peak, pk)
                    g += 1
                t_b = clk()
                n_bad = int(np.sum(~np.isfinite(amps)))
                self.timeouts += n_bad
                if n_bad and n_bad < sc.n_steps:     # trous isolés : interpolation
                    ok = np.isfinite(amps)
                    amps[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), amps[ok])
                yield Sweep(amps=amps, t=0.5 * (t_a + t_b), step_t=tk,
                            stats={"sweep_s": t_b - t_a, "reads_per_step": reads / sc.n_steps,
                                   "timeouts": self.timeouts, "adc_peak_dbfs": peak,
                                   "bad_steps": n_bad})
                i_sw += 1
        finally:
            restore()


class SfcwSimSource(SfcwSource):
    """Balayages simulés directement (sans faux Pluto), cadence réglable."""
    kind = "sfcw-sim"

    def __init__(self, cfg: Config, params=None, sweep_hz: float = 4.0, realtime: bool = True,
                 rng: np.random.Generator | None = None) -> None:
        from radar.sfcw.scene import SfcwScene
        super().__init__()
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.simulation.seed)
        self.scene = SfcwScene(params, self.rng)
        self.sweep_hz = sweep_hz
        self.realtime = realtime
        self.truth = self.scene.truth

    def describe(self) -> dict:
        return {"kind": self.kind, "scenario": self.scene.p.scenario, "sweep_hz": self.sweep_hz}

    def sweeps(self) -> Iterator[Sweep]:
        sc = self.cfg.sfcw
        freqs = sc.freqs
        thr = Throttle(self.realtime)
        T = 1.0 / self.sweep_hz
        step_dt = 0.8 * T / sc.n_steps
        i = 0
        while not self._stop:
            t0 = i * T
            order = sweep_order(sc.n_steps, i, sc.zigzag)
            amps, tk = self.scene.sweep(freqs, t0, step_dt, order)
            i += 1
            thr.wait_until(i * T)
            yield Sweep(amps=amps, t=t0 + 0.4 * T, step_t=tk, stats={"sweep_s": 0.8 * T})


# ----------------------------------------------------------------------
# Enregistrement
# ----------------------------------------------------------------------

@dc.dataclass
class SfcwRecording:
    amps: np.ndarray          # (n_sweeps, n_steps)
    t: np.ndarray             # (n_sweeps,)
    freqs: np.ndarray
    label: int
    events: list
    meta: dict
    path: Path | None = None

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


class SfcwRecorder:
    def __init__(self, directory: Path, freqs: np.ndarray, label: int, meta: dict,
                 tag: str = "", max_s: float | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        lab = {1: "resp", 0: "vide", -1: "inconnu"}.get(label, str(label))
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")
        self.path = directory / f"{stamp}_sfcw_{lab}{'_' + safe if safe else ''}.npz"
        self.freqs = np.asarray(freqs, np.float64)
        self.label = int(label)
        self.meta = {**meta, "kind": "sfcw",
                     "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        self.max_s = max_s
        self._a: list = []
        self._t: list = []
        self.events: list = []
        logger.info("Enregistrement SFCW → %s", self.path)

    @property
    def elapsed_s(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    @property
    def full(self) -> bool:
        return self.max_s is not None and self.elapsed_s >= self.max_s

    def add(self, sw: Sweep) -> None:
        if not self.full:
            self._a.append(np.asarray(sw.amps, np.float32))
            self._t.append(float(sw.t))

    def annotate(self, text: str) -> None:
        self.events.append((round(self.elapsed_s, 3), str(text)))

    def close(self) -> Path:
        amps = np.vstack(self._a) if self._a else np.zeros((0, len(self.freqs)), np.float32)
        self.meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        np.savez_compressed(self.path, amps=amps, t=np.asarray(self._t, np.float64),
                            freqs=self.freqs, label=np.int8(self.label),
                            events=np.asarray(json.dumps(self.events, ensure_ascii=False)),
                            meta=np.asarray(json.dumps(self.meta, ensure_ascii=False, default=str)))
        logger.info("Enregistrement SFCW fermé : %s (%d balayages)", self.path, len(self._t))
        return self.path


def load_sfcw(path) -> SfcwRecording:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        if "amps" not in d.files:
            raise ValueError(f"{path.name} n'est pas un enregistrement SFCW.")
        return SfcwRecording(amps=np.asarray(d["amps"], np.float64), t=np.asarray(d["t"]),
                             freqs=np.asarray(d["freqs"]), label=int(d["label"]),
                             events=json.loads(str(d["events"])), meta=json.loads(str(d["meta"])),
                             path=path)


class SfcwReplaySource(SfcwSource):
    kind = "sfcw-replay"

    def __init__(self, path, realtime: bool = True, loop: bool = True) -> None:
        super().__init__()
        self.rec = load_sfcw(path)
        self.path = str(path)
        self.realtime = realtime
        self.loop = loop
        self.truth = {"label": self.rec.label, **(self.rec.meta.get("truth") or {})}

    def describe(self) -> dict:
        return {"kind": self.kind, "file": self.path, "label": self.rec.label}

    def sweeps(self) -> Iterator[Sweep]:
        r = self.rec
        off = 0.0
        while True:
            thr = Throttle(self.realtime)
            t0 = r.t[0]
            for a, t in zip(r.amps, r.t):
                if self._stop:
                    return
                thr.wait_until(t - t0)
                yield Sweep(amps=a, t=float(t - t0 + off))
            if not self.loop:
                return
            off += r.duration_s + float(np.median(np.diff(r.t))) if len(r.t) > 1 else 1.0
