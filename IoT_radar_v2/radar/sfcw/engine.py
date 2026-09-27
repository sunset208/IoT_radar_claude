"""Moteur SFCW temps réel : source de balayages → processeur → état pour l'UI ; enregistrement.

S'insère dans le tableau de bord comme source « annexe » (``AppState.extras``) :
expose ``state()`` (dict JSON) et ``version`` (entier croissant).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from radar.config import Config, resolve_path
from radar.sfcw.dsp import SfcwProcessor
from radar.sfcw.sources import SfcwRecorder, SfcwSource

logger = logging.getLogger(__name__)


def make_processor(cfg: Config, freqs=None) -> SfcwProcessor:
    s = cfg.sfcw
    return SfcwProcessor(s.freqs if freqs is None else freqs, bg_tau_s=s.bg_tau_s,
                         nfft=s.nfft_range, min_range_m=s.min_range_m,
                         max_range_m=s.max_range_m, breath_window_s=s.breath_window_s,
                         hop_s=s.hop_s, breath_band=cfg.analysis.breath_band,
                         breath_threshold_db=s.breath_threshold_db,
                         presence_threshold_db=s.presence_threshold_db,
                         on_count=s.on_count, map_span_s=s.map_span_s)


class SfcwEngine:
    def __init__(self, cfg: Config, source: SfcwSource) -> None:
        self.cfg = cfg
        self.source = source
        freqs = getattr(getattr(source, "rec", None), "freqs", None)
        self.proc = make_processor(cfg, freqs)
        self.version = 0
        self.error: str | None = None
        self.last_stats: dict = {}
        self.recorder: SfcwRecorder | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state_cache: tuple[int, dict] | None = None
        self.subscribers: list = []

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sfcw-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.stop_recording()

    def _run(self) -> None:
        try:
            for sw in self.source.sweeps():
                if self._stop.is_set():
                    break
                self.last_stats = sw.stats
                with self._lock:
                    if self.recorder is not None:
                        self.recorder.add(sw)
                res = self.proc.push(sw)
                self.version += 1
                for fn in list(self.subscribers):
                    try:
                        fn(sw, res)
                    except Exception:
                        logger.debug("abonné SFCW en erreur", exc_info=True)
        except Exception as exc:
            logger.exception("Erreur SFCW")
            self.error = str(exc)
            self.version += 1
        finally:
            self.stop_recording()

    # ------------------------------------------------------------------
    def start_recording(self, label: int, tag: str = "", notes: str = "",
                        max_s: float | None = None) -> str:
        with self._lock:
            if self.recorder is not None:
                self.recorder.close()
            meta = {"config": self.cfg.to_dict(), "source": self.source.describe(),
                    "truth": self.source.truth, "notes": notes, "tag": tag, "session": tag or None}
            self.recorder = SfcwRecorder(resolve_path(self.cfg.sfcw.record_dir), self.proc.freqs,
                                         label, meta, tag=tag, max_s=max_s)
            return str(self.recorder.path)

    def stop_recording(self) -> str | None:
        with self._lock:
            if self.recorder is None:
                return None
            p = self.recorder.close()
            self.recorder = None
            return str(p)

    def annotate(self, text: str) -> None:
        with self._lock:
            if self.recorder is not None:
                self.recorder.annotate(text)

    def freeze_background(self) -> None:
        self.proc.freeze_background()
        self.version += 1

    # ------------------------------------------------------------------
    def state(self) -> dict:
        if self._state_cache and self._state_cache[0] == self.version:
            return self._state_cache[1]
        st = self.proc.ui_state()
        s = self.last_stats
        st["stats"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in s.items()}
        st["source"] = self.source.describe()
        st["truth"] = self.source.truth
        rec = self.recorder
        st["recording"] = None if rec is None else {
            "path": rec.path.name, "elapsed_s": round(rec.elapsed_s, 1), "label": rec.label}
        if self.error:
            st["status"] = "error"
            st["error"] = self.error
        self._state_cache = (self.version, st)
        return st


def run_headless(cfg: Config, source: SfcwSource, duration_s: float, label: int,
                 tag: str = "", notes: str = "", out: str | None = None) -> str:
    """Enregistrement SFCW sans interface (``radar sfcw record``)."""
    if out:
        cfg.sfcw.record_dir = out
    eng = SfcwEngine(cfg, source)
    path = eng.start_recording(label, tag, notes, max_s=duration_s)
    eng.start()
    t0 = time.time()
    try:
        while True:
            time.sleep(0.5)
            rec = eng.recorder
            st = eng.proc.result
            el = rec.elapsed_s if rec else duration_s
            print(f"\r{el:6.1f}/{duration_s:.0f} s  {eng.proc.sweep_hz:4.1f} balayages/s  "
                  f"état={eng.proc.state:11s} "
                  f"cible={st.get('best_range_m', float('nan')):.2f} m "
                  f"SNR={st.get('best_snr_db', float('nan')):5.1f} dB", end="", flush=True)
            if rec is None or rec.full or eng.error or not eng._thread.is_alive():
                break
            if time.time() - t0 > 3 * duration_s + 30:
                print("\nDélai dépassé.")
                break
    except KeyboardInterrupt:
        print("\nInterrompu — sauvegarde.")
    eng.stop()
    if eng.error:
        print(f"\nErreur : {eng.error}")
    print(f"\nFichier : {path}")
    return path
