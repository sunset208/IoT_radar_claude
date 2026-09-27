"""Moteur temps réel : source → tampon circulaire → analyse → détecteur → abonnés.

Le moteur tourne dans un thread.  Toutes les ``hop_s`` secondes de signal, il
analyse la dernière fenêtre ``window_s`` et publie un *snapshot* (dict JSON-
sérialisable) aux abonnés (UI web, CLI).  Il gère aussi l'enregistrement.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable

import numpy as np

from radar.config import Config, resolve_path
from radar.dsp.detector import Decision, Detector
from radar.dsp.vitals import VitalsAnalyzer
from radar.recorder import Recorder
from radar.sources.base import Source

logger = logging.getLogger(__name__)


def make_analyzer(cfg: Config, fs_slow: float | None = None) -> VitalsAnalyzer:
    a = cfg.analysis
    return VitalsAnalyzer(fs_slow or cfg.frontend.fs_slow, cfg.sdr.wavelength,
                          a.breath_band, a.heart_band, a.noise_band, a.nfft)


class SlidingWindow:
    def __init__(self, n: int) -> None:
        self.buf = np.zeros(n, dtype=np.complex64)
        self.n = n
        self.count = 0          # échantillons valides accumulés (≤ n)
        self.total = 0

    def push(self, x: np.ndarray) -> None:
        k = len(x)
        if k >= self.n:
            self.buf[:] = x[-self.n:]
        else:
            self.buf = np.roll(self.buf, -k)
            self.buf[-k:] = x
        self.count = min(self.n, self.count + k)
        self.total += k

    def clear(self) -> None:
        self.count = 0

    def last(self, m: int) -> np.ndarray:
        return self.buf[-min(m, self.count):]


class Engine:
    HISTORY_S = 300.0      # historique des scores / états pour l'UI
    WATERFALL_COLS = 240   # colonnes de spectrogramme conservées

    def __init__(self, cfg: Config, source: Source, calibration: dict | None = None,
                 ml_detector=None) -> None:
        self.cfg = cfg
        self.source = source
        self.fs = source.fs_slow
        a = cfg.analysis
        self.win_n = int(round(a.window_s * self.fs))
        self.hop_n = max(1, int(round(a.hop_s * self.fs)))
        self.min_n = int(round(a.min_window_s * self.fs))
        self.window = SlidingWindow(self.win_n)
        self.analyzer = make_analyzer(cfg, self.fs)
        self.detector = Detector(cfg.detector, calibration)
        self.ml = ml_detector
        self.subscribers: list[Callable[[dict], None]] = []
        self.snapshot: dict = {"status": "starting"}
        self.history: collections.deque = collections.deque(
            maxlen=int(self.HISTORY_S / a.hop_s))
        self.waterfall: collections.deque = collections.deque(maxlen=self.WATERFALL_COLS)
        self.waterfall_f: list[float] | None = None
        self.recorder: Recorder | None = None
        self._rec_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None
        self.last_stats: dict = {}
        self.proc_ms = 0.0
        self.started = time.time()

    # ------------------------------------------------------------------
    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self.subscribers.append(fn)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="radar-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.stop_recording()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    # ------------------------------------------------------------------
    def start_recording(self, label: int, tag: str = "", notes: str = "",
                        max_s: float | None = None) -> str:
        with self._rec_lock:
            if self.recorder is not None:
                self.stop_recording()
            meta = {
                "config": self.cfg.to_dict(),
                "source": self.source.describe(),
                "truth": self.source.truth,
                "notes": notes,
                "tag": tag,
                "session": tag or None,
            }
            raw_fs = self.cfg.sdr.f_s if self.cfg.recording.save_raw else None
            self.recorder = Recorder(resolve_path(self.cfg.recording.dir), self.fs, label,
                                     meta, tag=tag, raw_fs=raw_fs, max_s=max_s)
            return str(self.recorder.path)

    def stop_recording(self) -> str | None:
        with self._rec_lock:
            if self.recorder is None:
                return None
            path = self.recorder.close()
            self.recorder = None
            return str(path)

    def annotate(self, text: str) -> None:
        with self._rec_lock:
            if self.recorder is not None:
                self.recorder.annotate(text)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        since_hop = 0
        try:
            for chunk in self.source.chunks():
                if self._stop.is_set():
                    break
                if chunk.discontinuity:
                    self.window.clear()
                    self.detector.reset()
                    self.analyzer.reset()
                    since_hop = 0
                with self._rec_lock:
                    if self.recorder is not None:
                        self.recorder.add(chunk.slow, chunk.discontinuity, chunk.raw)
                self.window.push(chunk.slow)
                self.last_stats = chunk.stats
                since_hop += len(chunk.slow)
                if since_hop >= self.hop_n and self.window.count >= self.min_n:
                    since_hop = 0
                    self._analyze(chunk.t0 + len(chunk.slow) / self.fs)
                elif self.window.count < self.min_n:
                    self._publish_warmup()
        except Exception as exc:  # affiché dans l'UI
            logger.exception("Erreur moteur")
            self.error = str(exc)
            self.snapshot = {"status": "error", "error": self.error}
            self._emit(self.snapshot)
        finally:
            self.stop_recording()
            logger.info("Moteur arrêté")

    def _publish_warmup(self) -> None:
        snap = {
            "status": "warmup",
            "progress": self.window.count / max(self.min_n, 1),
            "stats": self._stats(),
            "source": self.source.describe(),
        }
        self.snapshot = snap
        self._emit(snap)

    def _stats(self) -> dict:
        st = dict(self.last_stats)
        st["proc_ms"] = round(self.proc_ms, 2)
        st["recording"] = None
        if self.recorder is not None:
            st["recording"] = {"path": str(self.recorder.path.name),
                               "elapsed_s": round(self.recorder.elapsed_s, 1),
                               "label": self.recorder.label}
        return st

    def _analyze(self, t: float) -> None:
        t_a = time.perf_counter()
        x = self.window.last(self.win_n)
        feats, disp = self.analyzer.analyze(x, t=t)
        dec: Decision = self.detector.update(feats)
        ml_p = None
        if self.ml is not None:
            try:
                ml_p = float(self.ml.predict(x, self.fs))
            except Exception as exc:
                logger.warning("Détecteur IA en erreur : %s", exc)
                self.ml = None
        self.proc_ms = 1e3 * (time.perf_counter() - t_a)

        self.history.append({"t": round(t, 2), "score": round(dec.score, 3),
                             "state": dec.state, "snr": round(feats.snr_db, 1),
                             "bpm": None if dec.breath_bpm is None else round(dec.breath_bpm, 1),
                             "ml": None if ml_p is None else round(ml_p, 3)})
        if disp is not None:
            self.waterfall.append(np.round(disp.spec_db.astype(np.float32), 1).tolist())
            self.waterfall_f = np.round(disp.spec_f, 3).tolist()

        snap = {
            "status": "running",
            "t": round(t, 2),
            "decision": _clean(dec.to_dict()),
            "ml_score": ml_p,
            "display": None if disp is None else {
                "spec_f": np.round(disp.spec_f, 3).tolist(),
                "spec_db": np.round(disp.spec_db, 1).tolist(),
                "wave_t": np.round(disp.wave_t, 2).tolist(),
                "wave": np.round(disp.wave, 5).tolist(),
                "wave_unit": disp.wave_unit,
                "iq": (disp.iq / (np.max(np.abs(disp.iq)) + 1e-15)).round(4).tolist(),
                "iq_scale": float(np.max(np.abs(disp.iq))),
                "circle": disp.circle,
            },
            "stats": self._stats(),
            "source": self.source.describe(),
            "truth": self.source.truth,
            "thresholds": {"snr_db": self.detector.snr_thr, "periodicity": self.detector.per_thr,
                           "motion": self.detector.mot_thr},
        }
        self.snapshot = snap
        self._emit(snap)

    def _emit(self, snap: dict) -> None:
        for fn in list(self.subscribers):
            try:
                fn(snap)
            except Exception:
                logger.debug("Abonné en erreur", exc_info=True)


def _clean(obj):
    """Rend un dict JSON-sérialisable (numpy → python, NaN → None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 4)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
