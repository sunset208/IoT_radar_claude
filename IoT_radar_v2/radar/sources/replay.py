"""Relecture d'un enregistrement slow-time (format v2, voir :mod:`radar.recorder`)."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from radar.recorder import load_recording
from radar.sources.base import Chunk, Source, Throttle


class ReplaySource(Source):
    kind = "replay"

    def __init__(self, path, realtime: bool = True, speed: float = 1.0,
                 chunk_s: float = 0.1, loop: bool = False) -> None:
        super().__init__()
        self.rec = load_recording(path)
        self.path = str(path)
        self.fs_slow = float(self.rec.fs_slow)
        self.realtime = realtime
        self.speed = speed
        self.chunk = max(1, int(round(chunk_s * self.fs_slow)))
        self.loop = loop
        self.truth = {"label": self.rec.label, **(self.rec.meta.get("truth") or {})}

    def describe(self) -> dict:
        return {"kind": self.kind, "fs_slow": self.fs_slow, "file": self.path,
                "label": self.rec.label, "duration_s": self.rec.duration_s}

    def _iter(self) -> Iterator[Chunk]:
        x = self.rec.slow
        while True:
            thr = Throttle(self.realtime)
            for i in range(0, len(x), self.chunk):
                if self._stop:
                    return
                thr.wait_until((i + self.chunk) / self.fs_slow / self.speed)
                yield Chunk(slow=np.asarray(x[i:i + self.chunk]), t0=i / self.fs_slow,
                            discontinuity=bool(np.any(
                                (self.rec.gaps >= i) & (self.rec.gaps < i + self.chunk))))
            if not self.loop:
                return
