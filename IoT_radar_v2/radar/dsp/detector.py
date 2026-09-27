"""Décision : features → état avec hystérésis, à deux cadences.

États (par priorité) :

* **RESPIRATION** — une raie respiratoire est confirmée sur au moins une des
  échelles (fenêtres 8 / 12 / 20 s, seuils propres à chaque échelle) pendant
  ``on_count`` analyses consécutives ; relâchée après ``off_count`` analyses
  sans aucune échelle positive.  Les échelles courtes exigent en plus un
  rythme stable (``rate_consistency_hz``) : une raie de bruit se promène.
* **MOUVEMENT** — indice d'activité rapide (1.5–15 Hz sur 1 s) au-dessus du
  seuil, bouffées d'énergie > 3 Hz dans la fenêtre longue, ou (si calibré)
  plancher de bruit nettement relevé (quelqu'un marche en continu).
* **PRESENCE** — micro-mouvements lents (0.12–1 Hz sur 4 s) au-dessus du
  bruit attendu en salle vide : un « signe de vie » en ~5 s, avant que la
  périodicité ne soit confirmée.
* **VIDE** sinon.

Cadences : :meth:`Detector.update_fast` à ~10 Hz (indices rapides),
:meth:`Detector.update` toutes les ``hop_s`` (analyses spectrales).

Les seuils viennent de la config ; ``radar calibrate`` les recalcule sur des
enregistrements « salle vide » (voir :mod:`radar.offline`).
"""

from __future__ import annotations

import collections
import dataclasses as dc
import json
import math
from pathlib import Path

import numpy as np

from radar.dsp.vitals import Features

EMPTY, PRESENCE, MOTION, BREATHING = "VIDE", "PRESENCE", "MOUVEMENT", "RESPIRATION"
STATES = (EMPTY, PRESENCE, MOTION, BREATHING)


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


@dc.dataclass
class ScaleResult:
    window_s: float
    snr_db: float
    periodicity: float
    breath_bpm: float | None
    snr_thr: float
    positive: bool
    streak: int

    def to_dict(self) -> dict:
        return dc.asdict(self)


@dc.dataclass
class Decision:
    t: float
    state: str
    score: float                  # confiance respiration [0,1] (max sur les échelles)
    raw_positive: bool            # au moins une échelle positive (avant hystérésis)
    breath_bpm: float | None
    heart_bpm: float | None
    features: Features            # features de la fenêtre la plus longue disponible
    warnings: list = dc.field(default_factory=list)
    scales: list = dc.field(default_factory=list)       # [ScaleResult]
    confirmed_by_s: float | None = None                 # échelle ayant confirmé
    presence_db: float | None = None
    activity_db: float | None = None

    def to_dict(self) -> dict:
        return {
            "t": self.t, "state": self.state, "score": self.score, "warnings": self.warnings,
            "raw_positive": self.raw_positive, "breath_bpm": self.breath_bpm,
            "heart_bpm": self.heart_bpm, "features": self.features.to_dict(),
            "scales": [s.to_dict() for s in self.scales], "confirmed_by_s": self.confirmed_by_s,
            "presence_db": self.presence_db, "activity_db": self.activity_db,
        }


@dc.dataclass
class LogisticModel:
    """p = σ(w·((v − μ)/σ) + b) sur ``Features.vector()`` (échelle longue)."""
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray
    threshold: float = 0.5

    def prob(self, v: np.ndarray) -> float:
        z = float(np.dot(self.w, (v - self.mu) / self.sd) + self.b)
        return _sig(z)

    def to_dict(self) -> dict:
        return {"w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "threshold": self.threshold,
                "keys": list(Features.VECTOR_KEYS)}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        return cls(np.asarray(d["w"]), float(d["b"]), np.asarray(d["mu"]),
                   np.asarray(d["sd"]), float(d.get("threshold", 0.5)))


def _fkey(w: float) -> str:
    return f"{float(w):g}"


class Detector:
    def __init__(self, det_cfg, calibration: dict | None = None, fast_cfg=None,
                 window_s: float = 20.0, extra_windows_s=()) -> None:
        self.window_s = float(window_s)
        self.snr_thr = det_cfg.snr_threshold_db
        self.per_thr = det_cfg.periodicity_threshold
        self.mot_thr = det_cfg.motion_threshold
        self.on_count = det_cfg.on_count
        self.off_count = det_cfg.off_count
        self.rate_tol_hz = float(getattr(det_cfg, "rate_consistency_hz", 0.04))
        # seuils par échelle {fenêtre (s): (snr_db, périodicité)}
        self.scale_thr: dict[float, tuple[float, float]] = {}
        for w, snr, per in zip(extra_windows_s,
                               getattr(det_cfg, "scale_snr_threshold_db", ()),
                               getattr(det_cfg, "scale_periodicity_threshold", ())):
            self.scale_thr[float(w)] = (float(snr), float(per))
        # indices rapides
        self.fast_cfg = fast_cfg
        self.pres_thr = getattr(fast_cfg, "presence_threshold_db", 6.0)
        self.act_thr = getattr(fast_cfg, "activity_threshold_db", 8.0)
        self.model: LogisticModel | None = None
        self.noise_ref_dbfs: float | None = None   # plancher « salle vide » calibré
        if calibration:
            self.apply_calibration(calibration)
        self.reset()

    # ------------------------------------------------------------------
    def apply_calibration(self, cal: dict) -> None:
        th = cal.get("thresholds", {})
        self.snr_thr = float(th.get("snr_threshold_db", self.snr_thr))
        self.per_thr = float(th.get("periodicity_threshold", self.per_thr))
        self.mot_thr = float(th.get("motion_threshold", self.mot_thr))
        for key, v in (cal.get("scales") or {}).items():
            w = float(key)
            if w in self.scale_thr or w != self.window_s:
                old = self.scale_thr.get(w, (self.snr_thr, self.per_thr))
                self.scale_thr[w] = (float(v.get("snr_threshold_db", old[0])),
                                     float(v.get("periodicity_threshold", old[1])))
        fast = cal.get("fast") or {}
        self.pres_thr = float(fast.get("presence_threshold_db", self.pres_thr))
        self.act_thr = float(fast.get("activity_threshold_db", self.act_thr))
        if cal.get("noise_ref_dbfs") is not None:
            self.noise_ref_dbfs = float(cal["noise_ref_dbfs"])
        if cal.get("logistic"):
            self.model = LogisticModel.from_dict(cal["logistic"])

    def reset(self) -> None:
        self.state = EMPTY
        self._pos: dict[float, int] = collections.defaultdict(int)
        self._hz: dict[float, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.on_count))
        self._neg = 0
        self._breathing = False
        self.confirmed_by: float | None = None
        self._motion_hold = 0
        self._rates: collections.deque = collections.deque(maxlen=9)
        self._long_rates: collections.deque = collections.deque(maxlen=120)
        # indices rapides (compteurs en secondes)
        self._t_fast: float | None = None
        self._pres_above = 0.0
        self._pres_below = 0.0
        self._presence = False
        self._act_above = 0.0
        self._act_until = -1.0
        self.presence_db: float | None = None
        self.activity_db: float | None = None

    # ------------------------------------------------------------------
    def thresholds_for(self, w: float) -> tuple[float, float]:
        if abs(w - self.window_s) < 1e-6:
            return self.snr_thr, self.per_thr
        return self.scale_thr.get(float(w), (self.snr_thr, self.per_thr))

    def score(self, ft: Features, w: float | None = None) -> float:
        if self.model is not None and (w is None or abs(w - self.window_s) < 1e-6):
            return self.model.prob(ft.vector())
        snr_thr, per_thr = self.thresholds_for(self.window_s if w is None else w)
        return (_sig((ft.snr_db - snr_thr) / 1.5)
                * _sig((ft.periodicity - per_thr) / 0.05)
                * (1.0 - _sig((ft.motion - self.mot_thr) / 0.5)))

    def is_positive(self, ft: Features, w: float | None = None) -> bool:
        long_ = w is None or abs(w - self.window_s) < 1e-6
        if self.model is not None and long_:
            return self.model.prob(ft.vector()) >= self.model.threshold and ft.motion < self.mot_thr
        snr_thr, per_thr = self.thresholds_for(self.window_s if w is None else w)
        return ft.snr_db >= snr_thr and ft.periodicity >= per_thr and ft.motion < self.mot_thr

    def is_motion(self, ft: Features) -> bool:
        # bouffées > 3 Hz, ou (si calibré) plancher de bruit nettement au-dessus
        # de celui de la salle vide (quelqu'un marche en continu : Doppler large bande)
        raised = (self.noise_ref_dbfs is not None
                  and ft.noise_dbfs - self.noise_ref_dbfs > 6.0)
        return ft.motion >= self.mot_thr or raised

    # ------------------------------------------------------------------
    def update_fast(self, t: float, presence_db: float, activity_db: float) -> str:
        """Indices rapides (~10 Hz) → drapeaux présence / activité avec hystérésis temporelle."""
        dt = 0.1 if self._t_fast is None else max(0.0, min(1.0, t - self._t_fast))
        self._t_fast = t
        self.presence_db, self.activity_db = presence_db, activity_db
        fc = self.fast_cfg
        on_s = getattr(fc, "presence_on_s", 1.5)
        off_s = getattr(fc, "presence_off_s", 3.0)
        if presence_db >= self.pres_thr:
            self._pres_above += dt
            self._pres_below = 0.0
        else:
            self._pres_below += dt
            self._pres_above = 0.0
        if not self._presence and self._pres_above >= on_s - 1e-9:
            self._presence = True
        elif self._presence and self._pres_below >= off_s - 1e-9:
            self._presence = False
        if activity_db >= self.act_thr:
            self._act_above += dt
            if self._act_above >= getattr(fc, "activity_on_s", 0.3) - 1e-9:
                self._act_until = t + getattr(fc, "activity_hold_s", 2.0)
        else:
            self._act_above = 0.0
        self._resolve(t)
        return self.state

    @property
    def activity(self) -> bool:
        return self._t_fast is not None and self._t_fast <= self._act_until

    @property
    def presence(self) -> bool:
        return self._presence

    def _resolve(self, t: float | None = None) -> None:
        if self._breathing:
            self.state = BREATHING
        elif self.activity or self._motion_hold:
            self.state = MOTION
        elif self._presence:
            self.state = PRESENCE
        else:
            self.state = EMPTY

    # ------------------------------------------------------------------
    def update(self, feats) -> Decision:
        """Une analyse (toutes les ``hop_s``).  *feats* : Features (échelle
        longue seule) ou dict {fenêtre_s: Features} (multi-échelle)."""
        if isinstance(feats, Features):
            feats = {self.window_s: feats}
        ws = sorted(feats)
        f_long = feats[ws[-1]]
        results: list[ScaleResult] = []
        any_pos = False
        best_pos_w = None
        for w in ws:
            ft = feats[w]
            short = abs(w - self.window_s) > 1e-6
            pos = self.is_positive(ft, w)
            if short and self.activity:
                # mouvement franc en cours (indice rapide) : une fenêtre courte
                # n'a pas assez de recul pour séparer respiration et mouvement
                pos = False
            if pos and ft.breath_hz is not None:
                self._hz[w].append(ft.breath_hz)
            if pos:
                self._pos[w] += 1
            else:
                self._pos[w] = 0
                self._hz[w].clear()
            streak = self._pos[w]
            confirmed = streak >= self.on_count
            if confirmed and short:
                hz = np.asarray(self._hz[w])
                confirmed = hz.size >= self.on_count and float(np.ptp(hz)) <= 2 * self.rate_tol_hz
            snr_thr, _ = self.thresholds_for(w)
            results.append(ScaleResult(w, ft.snr_db, ft.periodicity, ft.breath_bpm, snr_thr,
                                       pos, streak))
            if pos:
                any_pos = True
                best_pos_w = w           # la plus longue fenêtre positive (meilleure résolution)
            if confirmed and not self._breathing:
                self._breathing = True
                self.confirmed_by = w
        if any_pos:
            self._neg = 0
            bf = feats[best_pos_w]
            if bf.breath_bpm is not None:
                self._rates.append(bf.breath_bpm)
                self._long_rates.append(bf.breath_bpm)
        else:
            self._neg += 1
        if self._breathing and self._neg >= self.off_count:
            self._breathing = False
            self.confirmed_by = None
        mot = (not any_pos) and self.is_motion(f_long)
        self._motion_hold = 4 if mot else max(0, self._motion_hold - 1)
        self._resolve(f_long.t)

        bpm = float(np.median(self._rates)) if (self.state == BREATHING and self._rates) else None
        hr = f_long.heart_bpm if self.state == BREATHING else None
        warnings = []
        if self.state == BREATHING and len(self._long_rates) >= 60:
            r = np.asarray(self._long_rates)
            # Une respiration humaine varie de 5–15 % sur une minute ; un
            # ventilateur oscillant ou une machine est quasi parfaitement régulier.
            if np.std(r) / max(np.mean(r), 1e-9) < 0.015:
                warnings.append("rythme anormalement régulier : source mécanique possible")
        score = max(self.score(feats[w], w) for w in ws)
        return Decision(t=f_long.t, state=self.state, score=score, raw_positive=any_pos,
                        breath_bpm=bpm, heart_bpm=hr, features=f_long, warnings=warnings,
                        scales=results, confirmed_by_s=self.confirmed_by,
                        presence_db=self.presence_db, activity_db=self.activity_db)


def make_detector(cfg, calibration: dict | None = None) -> Detector:
    return Detector(cfg.detector, calibration, cfg.fast, cfg.analysis.window_s,
                    cfg.analysis.extra_windows_s)


def load_calibration(path) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
