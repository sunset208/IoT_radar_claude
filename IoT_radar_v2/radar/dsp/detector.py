"""Décision : features → état (VIDE / MOUVEMENT / RESPIRATION) avec hystérésis.

Règle par défaut (seuils dans ``detector`` de la config, recalculables par
``radar calibrate`` sur des enregistrements « salle vide ») :

* RESPIRATION si ``snr_db ≥ snr_thr``, ``periodicity ≥ per_thr`` et
  ``motion < motion_thr`` pendant ``on_count`` fenêtres consécutives ;
  relâchée après ``off_count`` fenêtres négatives.
* MOUVEMENT si bouffées d'énergie (``motion ≥ motion_thr``) ou énergie de
  bande élevée sans périodicité : quelqu'un/quelque chose bouge.
* VIDE sinon.

Si une calibration apprise (régression logistique) est chargée, sa
probabilité remplace le produit de sigmoïdes par défaut.
"""

from __future__ import annotations

import collections
import dataclasses as dc
import json
import math
from pathlib import Path

import numpy as np

from radar.dsp.vitals import Features

EMPTY, MOTION, BREATHING = "VIDE", "MOUVEMENT", "RESPIRATION"


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))


@dc.dataclass
class Decision:
    t: float
    state: str
    score: float                  # probabilité / confiance respiration [0,1]
    raw_positive: bool            # décision instantanée (avant hystérésis)
    breath_bpm: float | None
    heart_bpm: float | None
    features: Features
    warnings: list = dc.field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "t": self.t, "state": self.state, "score": self.score, "warnings": self.warnings,
            "raw_positive": self.raw_positive, "breath_bpm": self.breath_bpm,
            "heart_bpm": self.heart_bpm, "features": self.features.to_dict(),
        }


@dc.dataclass
class LogisticModel:
    """p = σ(w·((v − μ)/σ) + b) sur ``Features.vector()``."""
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


class Detector:
    def __init__(self, det_cfg, calibration: dict | None = None) -> None:
        self.snr_thr = det_cfg.snr_threshold_db
        self.per_thr = det_cfg.periodicity_threshold
        self.mot_thr = det_cfg.motion_threshold
        self.on_count = det_cfg.on_count
        self.off_count = det_cfg.off_count
        self.model: LogisticModel | None = None
        self.noise_ref_dbfs: float | None = None   # plancher « salle vide » calibré
        if calibration:
            self.apply_calibration(calibration)
        self.reset()

    def apply_calibration(self, cal: dict) -> None:
        th = cal.get("thresholds", {})
        self.snr_thr = float(th.get("snr_threshold_db", self.snr_thr))
        self.per_thr = float(th.get("periodicity_threshold", self.per_thr))
        self.mot_thr = float(th.get("motion_threshold", self.mot_thr))
        if cal.get("noise_ref_dbfs") is not None:
            self.noise_ref_dbfs = float(cal["noise_ref_dbfs"])
        if cal.get("logistic"):
            self.model = LogisticModel.from_dict(cal["logistic"])

    def reset(self) -> None:
        self.state = EMPTY
        self._pos = 0
        self._neg = 0
        self._motion_hold = 0
        self._rates: collections.deque = collections.deque(maxlen=9)
        self._long_rates: collections.deque = collections.deque(maxlen=120)

    # ------------------------------------------------------------------
    def score(self, ft: Features) -> float:
        if self.model is not None:
            return self.model.prob(ft.vector())
        return (_sig((ft.snr_db - self.snr_thr) / 1.5)
                * _sig((ft.periodicity - self.per_thr) / 0.05)
                * (1.0 - _sig((ft.motion - self.mot_thr) / 0.5)))

    def is_positive(self, ft: Features) -> bool:
        if self.model is not None:
            return self.model.prob(ft.vector()) >= self.model.threshold and ft.motion < self.mot_thr
        return (ft.snr_db >= self.snr_thr and ft.periodicity >= self.per_thr
                and ft.motion < self.mot_thr)

    def is_motion(self, ft: Features) -> bool:
        # bouffées > 3 Hz, ou énergie lente non périodique, ou (si calibré)
        # plancher de bruit nettement au-dessus de celui de la salle vide
        # (quelqu'un marche en continu : Doppler large bande)
        raised = (self.noise_ref_dbfs is not None
                  and ft.noise_dbfs - self.noise_ref_dbfs > 6.0)
        return (ft.motion >= self.mot_thr or raised or (
            ft.band_db > self.snr_thr and ft.periodicity < 0.6 * self.per_thr))

    def update(self, ft: Features) -> Decision:
        pos = self.is_positive(ft)
        mot = (not pos) and self.is_motion(ft)
        if pos:
            self._pos += 1
            self._neg = 0
            if ft.breath_bpm is not None:
                self._rates.append(ft.breath_bpm)
                self._long_rates.append(ft.breath_bpm)
        else:
            self._neg += 1
            self._pos = 0
        self._motion_hold = 4 if mot else max(0, self._motion_hold - 1)

        if self.state == BREATHING:
            if self._neg >= self.off_count:
                self.state = MOTION if self._motion_hold else EMPTY
        elif self._pos >= self.on_count:
            self.state = BREATHING
        else:
            self.state = MOTION if self._motion_hold else EMPTY

        bpm = float(np.median(self._rates)) if (self.state == BREATHING and self._rates) else None
        hr = ft.heart_bpm if self.state == BREATHING else None
        warnings = []
        if self.state == BREATHING and len(self._long_rates) >= 60:
            r = np.asarray(self._long_rates)
            # Une respiration humaine varie de 5–15 % sur une minute ; un
            # ventilateur oscillant ou une machine est quasi parfaitement régulier.
            if np.std(r) / max(np.mean(r), 1e-9) < 0.015:
                warnings.append("rythme anormalement régulier : source mécanique possible")
        return Decision(t=ft.t, state=self.state, score=self.score(ft), raw_positive=pos,
                        breath_bpm=bpm, heart_bpm=hr, features=ft, warnings=warnings)


def load_calibration(path) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
