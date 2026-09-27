"""Traitement temps réel : source → tampon → indices rapides + analyses multi-échelle → détecteur.

* :class:`Processor` — logique pure (sans thread) : reçoit les blocs slow-time,
  sort des évènements ``("fast", …)`` à ~10 Hz et ``("analysis", …)`` toutes
  les ``hop_s``.  Utilisé **à l'identique** par le moteur temps réel et par
  l'évaluation hors ligne (``radar evaluate``) : ce qui est mesuré est ce que
  voit l'utilisateur.
* :class:`Engine` — thread autour d'un Processor : lit la source, enregistre,
  publie des *snapshots* JSON aux abonnés (UI web, CLI).
"""

from __future__ import annotations

import collections
import dataclasses as dc
import logging
import threading
import time
from typing import Callable

import numpy as np

from radar.config import Config, resolve_path
from radar.dsp.detector import Decision, make_detector
from radar.dsp.fast import FastMonitor, FastParams, FastTick
from radar.dsp.vitals import Display, VitalsAnalyzer
from radar.recorder import Recorder
from radar.sources.base import Source

logger = logging.getLogger(__name__)


def make_analyzer(cfg: Config, fs_slow: float | None = None) -> VitalsAnalyzer:
    a = cfg.analysis
    return VitalsAnalyzer(fs_slow or cfg.frontend.fs_slow, cfg.sdr.wavelength,
                          a.breath_band, a.heart_band, a.noise_band, a.nfft)


def fast_params(cfg: Config, calibration: dict | None = None) -> FastParams:
    f = cfg.fast
    amp, ph = f.clutter_amp_dbc, f.clutter_phase_dbc
    cal = (calibration or {}).get("fast") or {}
    amp = float(cal.get("clutter_amp_dbc", amp))
    ph = float(cal.get("clutter_phase_dbc", ph))
    return FastParams(activity_s=f.activity_s, presence_s=f.presence_s,
                      presence_band=tuple(f.presence_band), activity_band=tuple(f.activity_band),
                      clutter_amp_dbc=amp, clutter_phase_dbc=ph)


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


@dc.dataclass
class AnalysisResult:
    t: float
    decision: Decision
    display: Display | None
    feats: dict                  # {fenêtre_s: Features}
    x: np.ndarray                # fenêtre la plus longue analysée (pour l'IA)


class Processor:
    """Blocs slow-time → évènements.  Aucune E/S, aucun thread."""

    def __init__(self, cfg: Config, fs: float, calibration: dict | None = None) -> None:
        self.cfg = cfg
        self.fs = float(fs)
        a = cfg.analysis
        self.scales = cfg.scales_s
        self.scale_n = {w: int(round(w * self.fs)) for w in self.scales}
        self.win_n = self.scale_n[self.scales[-1]]
        self.hop_n = max(1, int(round(a.hop_s * self.fs)))
        self.window = SlidingWindow(self.win_n)
        self.analyzer = make_analyzer(cfg, self.fs)
        self.detector = make_detector(cfg, calibration)
        self.fast = (FastMonitor(self.fs, fast_params(cfg, calibration), cfg.sdr.wavelength)
                     if cfg.fast.enabled else None)
        self.since_hop = 0
        self.last_decision: Decision | None = None
        self._radius: float | None = None
        self._radius_t = -1e9

    @property
    def min_n(self) -> int:
        return self.scale_n[self.scales[0]]

    def reset(self) -> None:
        """Perte d'échantillons : la phase n'est plus continue, on repart de zéro."""
        self.window.clear()
        self.detector.reset()
        self.analyzer.reset()
        if self.fast is not None:
            self.fast.reset()
        self.since_hop = 0

    def push(self, slow: np.ndarray, t_end: float, want_display: bool = True) -> list[tuple]:
        """Retourne une liste de ``("fast", FastTick, état)`` et ``("analysis", AnalysisResult)``."""
        out: list[tuple] = []
        slow = np.asarray(slow)
        self.window.push(slow)
        if self.fast is not None:
            for tick in self.fast.push(slow, t_end):
                state = self.detector.update_fast(tick.t, tick.presence_db, tick.activity_db)
                out.append(("fast", tick, state))
        self.since_hop += len(slow)
        if self.since_hop >= self.hop_n and self.window.count >= self.min_n:
            self.since_hop = 0
            out.append(("analysis", self._analyze(t_end, want_display)))
        return out

    def _update_radius(self, d: Display | None, t: float) -> None:
        """Rayon du cercle IQ (amplitude de l'écho de la cible) → échelle mm de la
        forme d'onde rapide.  Lissé et maintenu 30 s quand l'arc n'est plus
        identifiable, pour que l'échelle de la bande défilante ne saute pas."""
        if d is not None and d.circle:
            r = float(d.circle[2])
            self._radius = r if self._radius is None else self._radius * 0.8 + r * 0.2
            self._radius_t = t
        elif t - self._radius_t > 30.0:
            self._radius = None
        self.fast.target_radius = self._radius

    def _analyze(self, t: float, want_display: bool) -> AnalysisResult:
        feats = {}
        disp = None
        avail = [w for w in self.scales if self.window.count >= self.scale_n[w]]
        for w in avail:
            x = self.window.last(self.scale_n[w])
            last = w == avail[-1]
            f, d = self.analyzer.analyze(x, t=t, want_display=want_display and last, full=last)
            feats[w] = f
            if last:
                disp = d
                if self.fast is not None:
                    self._update_radius(d, t)
        dec = self.detector.update(feats)
        self.last_decision = dec
        return AnalysisResult(t=t, decision=dec, display=disp, feats=feats,
                              x=self.window.last(self.scale_n[avail[-1]]))


class Engine:
    HISTORY_S = 300.0      # historique des scores / états pour l'UI
    WATERFALL_COLS = 240   # colonnes de spectrogramme conservées
    WAVE_S = 30.0          # forme d'onde rapide conservée (reconnexion de l'UI)

    def __init__(self, cfg: Config, source: Source, calibration: dict | None = None,
                 ml_detector=None) -> None:
        self.cfg = cfg
        self.source = source
        self.fs = source.fs_slow
        a = cfg.analysis
        self.proc = Processor(cfg, self.fs, calibration)
        self.detector = self.proc.detector
        self.ml = ml_detector
        self.subscribers: list[Callable[[dict], None]] = []
        self.snapshot: dict = {"status": "starting"}
        if source.kind == "idle":
            self.snapshot = {"status": "idle", "message": getattr(source, "message", ""),
                             "source": source.describe()}
        self.snapshot_id = 0
        self.fast: dict | None = None
        self.history: collections.deque = collections.deque(
            maxlen=int(self.HISTORY_S / a.hop_s))
        self.waterfall: collections.deque = collections.deque(maxlen=self.WATERFALL_COLS)
        self.waterfall_f: list[float] | None = None
        self.wave: collections.deque = collections.deque(maxlen=int(self.WAVE_S * 10))
        self.md_cols: collections.deque = collections.deque(maxlen=int(self.WAVE_S * 10))
        self.wave_count = 0          # échantillons de forme d'onde produits depuis le début
        self.wave_end_t = 0.0        # instant (signal) du dernier échantillon
        self.hist_count = 0
        self._act_max = -99.0
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
                self._stop_recording_locked()
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

    def _stop_recording_locked(self) -> str | None:
        if self.recorder is None:
            return None
        path = self.recorder.close()
        self.recorder = None
        return str(path)

    def stop_recording(self) -> str | None:
        with self._rec_lock:
            return self._stop_recording_locked()

    def annotate(self, text: str) -> None:
        with self._rec_lock:
            if self.recorder is not None:
                self.recorder.annotate(text)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            for chunk in self.source.chunks():
                if self._stop.is_set():
                    break
                if chunk.discontinuity:
                    self.proc.reset()
                with self._rec_lock:
                    if self.recorder is not None:
                        self.recorder.add(chunk.slow, chunk.discontinuity, chunk.raw)
                self.last_stats = chunk.stats
                t_end = chunk.t0 + len(chunk.slow) / self.fs
                t_a = time.perf_counter()
                events = self.proc.push(chunk.slow, t_end)
                for ev in events:
                    if ev[0] == "fast":
                        self._on_fast(ev[1], ev[2])
                    else:
                        self._on_analysis(ev[1], t_a)
                if self.proc.window.count < self.proc.min_n:
                    self._publish_warmup()
        except Exception as exc:  # affiché dans l'UI
            logger.exception("Erreur moteur")
            self.error = str(exc)
            self.snapshot = {"status": "error", "error": self.error}
            self.snapshot_id += 1
            self._emit(self.snapshot)
        finally:
            self.stop_recording()
            logger.info("Moteur arrêté")

    def _publish_warmup(self) -> None:
        snap = {
            "status": "warmup",
            "progress": self.proc.window.count / max(self.proc.min_n, 1),
            "stats": self._stats(),
            "source": self.source.describe(),
        }
        self.snapshot = snap
        self.snapshot_id += 1
        self._emit(snap)

    def _stats(self) -> dict:
        st = dict(self.last_stats)
        st["proc_ms"] = round(self.proc_ms, 2)
        st["recording"] = None
        rec = self.recorder
        if rec is not None:
            st["recording"] = {"path": str(rec.path.name),
                               "elapsed_s": round(rec.elapsed_s, 1),
                               "label": rec.label}
        return st

    def _on_fast(self, tick: FastTick, state: str) -> None:
        self._act_max = max(self._act_max, tick.activity_db)
        for v in tick.wave:
            self.wave.append(round(float(v), 3))
        self.wave_count += len(tick.wave)
        self.wave_end_t = tick.t
        det = self.detector
        self.fast = {
            "t": tick.t, "state": state,
            "presence_db": round(tick.presence_db, 1), "activity_db": round(tick.activity_db, 1),
            "presence_r_db": round(tick.presence_r_db, 1), "presence_t_db": round(tick.presence_t_db, 1),
            "presence": det.presence, "activity": det.activity,
            "wave_dt": tick.wave_dt,
            "wave_scale_mm": None if tick.wave_scale_mm is None else float(f"{tick.wave_scale_mm:.4g}"),
        }
        if tick.md_db is not None:
            col = np.round(tick.md_db, 1).tolist()
            self.md_cols.append(col)
            self.fast["md"] = col
        self._emit({"fast": self.fast})

    def _on_analysis(self, res: AnalysisResult, t_a: float) -> None:
        dec, disp, t = res.decision, res.display, res.t
        ml_p = None
        if self.ml is not None:
            try:
                ml_p = float(self.ml.predict(res.x, self.fs))
            except Exception as exc:
                logger.warning("Détecteur IA en erreur : %s", exc)
                self.ml = None
        self.proc_ms = 1e3 * (time.perf_counter() - t_a)
        feats = dec.features
        self.history.append({"t": round(t, 2), "score": round(dec.score, 3),
                             "state": dec.state, "snr": round(feats.snr_db, 1),
                             "bpm": None if dec.breath_bpm is None else round(dec.breath_bpm, 1),
                             "ml": None if ml_p is None else round(ml_p, 3),
                             "pres": None if dec.presence_db is None else round(dec.presence_db, 1),
                             "act": None if self._act_max < -90 else round(self._act_max, 1)})
        self.hist_count += 1
        self._act_max = -99.0
        if disp is not None:
            self.waterfall.append(np.round(disp.spec_db.astype(np.float32), 1).tolist())
            self.waterfall_f = np.round(disp.spec_f, 3).tolist()
        det = self.detector
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
            "thresholds": {"snr_db": det.snr_thr, "periodicity": det.per_thr,
                           "motion": det.mot_thr, "presence_db": det.pres_thr,
                           "activity_db": det.act_thr},
        }
        self.snapshot = snap
        self.snapshot_id += 1
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
        if not np.isfinite(v):
            return None
        # 4 décimales pour les grandeurs ≥ 1 (temps, dB), 4 chiffres
        # significatifs en dessous (niveaux IQ ~1e-3, rayon du cercle…)
        return round(v, 4) if abs(v) >= 1.0 else float(f"{v:.4g}")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
