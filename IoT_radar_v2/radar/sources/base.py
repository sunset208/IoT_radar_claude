"""Interface commune des sources de données.

Toute source produit des blocs **slow-time** (complexes, fs_slow, pleine
échelle ADC = 1) via :meth:`Source.chunks`.  Les sources « brutes » (Pluto,
simulation RF) appliquent le :class:`~radar.dsp.frontend.Frontend` en interne.
"""

from __future__ import annotations

import dataclasses as dc
import time
from typing import Iterator

import numpy as np


@dc.dataclass
class Chunk:
    slow: np.ndarray                 # complex64, fs_slow
    t0: float                        # temps (s) du 1er échantillon depuis le début
    raw: np.ndarray | None = None    # IQ brut éventuel (debug / enregistrement brut)
    stats: dict = dc.field(default_factory=dict)   # niveaux ADC, retard, pertes…
    discontinuity: bool = False      # perte d'échantillons détectée avant ce bloc


class Source:
    """Classe de base.  Implémenter :meth:`_iter`."""

    kind: str = "base"
    fs_slow: float = 50.0
    #: vérité terrain connue (simulation / relecture étiquetée), sinon None
    truth: dict | None = None

    def __init__(self) -> None:
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def chunks(self) -> Iterator[Chunk]:
        return self._iter()

    def _iter(self) -> Iterator[Chunk]:  # pragma: no cover - abstrait
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": self.kind, "fs_slow": self.fs_slow}


class Throttle:
    """Cadence un producteur sur le temps réel (simulation / relecture)."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.t_start = time.monotonic()

    def wait_until(self, t_signal: float) -> None:
        if not self.enabled:
            return
        dt = self.t_start + t_signal - time.monotonic()
        if dt > 0:
            time.sleep(dt)
