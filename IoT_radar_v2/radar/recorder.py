"""Enregistrement / chargement au format v2.

Un enregistrement = **le signal slow-time complexe** (avant toute suppression
de clutter) + métadonnées.  C'est la donnée la plus « brute » utile : tout le
traitement aval (détection classique, features, IA) peut être rejoué et
modifié hors ligne.  400 octets/s à 50 Hz → 1 h ≈ 1.4 Mo (contre ~10 Mo par
2 min de spectrogramme dB en v1, qui perdait la phase).

Fichier ``<dir>/<AAAAMMJJ_HHMMSS>_<label>_<tag>.npz`` :

========== ==========================================================
``slow``   complex64 (N,) — slow-time, pleine échelle ADC = 1
``fs_slow`` float
``label``  int8 : 1 = respiration présente, 0 = absente, -1 = inconnu
``gaps``   int64 — indices où une perte d'échantillons a eu lieu
``events`` JSON — [(t_s, texte)] annotations horodatées (ex. « apnée »)
``meta``   JSON — config complète, source, vérité terrain, notes…
========== ==========================================================

L'IQ brut (option ``recording.save_raw``) est écrit en flux dans un ``.c64``
voisin (complex64 little-endian, f_s de la config).
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2


@dc.dataclass
class Recording:
    slow: np.ndarray
    fs_slow: float
    label: int
    gaps: np.ndarray
    events: list
    meta: dict
    path: Path | None = None

    @property
    def duration_s(self) -> float:
        return len(self.slow) / self.fs_slow

    @property
    def group(self) -> str:
        """Identifiant de session (pour les splits train/val sans fuite)."""
        return str(self.meta.get("session") or (self.path.stem if self.path else "?"))


class Recorder:
    def __init__(self, directory: Path, fs_slow: float, label: int, meta: dict,
                 tag: str = "", raw_fs: float | None = None,
                 max_s: float | None = None) -> None:
        self.max_n = None if max_s is None else int(round(max_s * fs_slow))
        directory.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        lab = {1: "resp", 0: "vide", -1: "inconnu"}.get(label, str(label))
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")
        self.path = directory / f"{stamp}_{lab}{'_' + safe if safe else ''}.npz"
        self.fs_slow = fs_slow
        self.label = int(label)
        self.meta = dict(meta)
        self.meta.setdefault("started_utc", dt.datetime.now(dt.timezone.utc).isoformat())
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._gaps: list[int] = []
        self.events: list[tuple[float, str]] = []
        self._raw_fh = None
        if raw_fs is not None:
            self.raw_path = self.path.with_suffix(".c64")
            self._raw_fh = open(self.raw_path, "wb")
            self.meta["raw_file"] = self.raw_path.name
            self.meta["raw_fs"] = raw_fs
        logger.info("Enregistrement → %s", self.path)

    @property
    def elapsed_s(self) -> float:
        return self._n / self.fs_slow

    @property
    def full(self) -> bool:
        return self.max_n is not None and self._n >= self.max_n

    def add(self, slow: np.ndarray, discontinuity: bool = False,
            raw: np.ndarray | None = None) -> None:
        if self.full:
            return
        if self.max_n is not None and self._n + len(slow) > self.max_n:
            keep = self.max_n - self._n
            if raw is not None:
                raw = raw[: int(len(raw) * keep / max(len(slow), 1))]
            slow = slow[:keep]
        if discontinuity:
            self._gaps.append(self._n)
        self._chunks.append(np.asarray(slow, dtype=np.complex64))
        self._n += len(slow)
        if raw is not None and self._raw_fh is not None:
            np.asarray(raw, dtype=np.complex64).tofile(self._raw_fh)

    def annotate(self, text: str) -> None:
        self.events.append((round(self.elapsed_s, 3), str(text)))

    def close(self) -> Path:
        if self._raw_fh is not None:
            self._raw_fh.close()
        slow = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.complex64)
        self.meta["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.meta["format_version"] = FORMAT_VERSION
        np.savez_compressed(
            self.path,
            slow=slow,
            fs_slow=np.float64(self.fs_slow),
            label=np.int8(self.label),
            gaps=np.asarray(self._gaps, dtype=np.int64),
            events=np.asarray(json.dumps(self.events, ensure_ascii=False)),
            meta=np.asarray(json.dumps(self.meta, ensure_ascii=False, default=str)),
        )
        logger.info("Enregistrement fermé : %s (%.1f s)", self.path, len(slow) / self.fs_slow)
        return self.path


def load_recording(path) -> Recording:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        return Recording(
            slow=np.asarray(d["slow"]),
            fs_slow=float(d["fs_slow"]),
            label=int(d["label"]),
            gaps=np.asarray(d["gaps"]) if "gaps" in d.files else np.zeros(0, np.int64),
            events=json.loads(str(d["events"])) if "events" in d.files else [],
            meta=json.loads(str(d["meta"])) if "meta" in d.files else {},
            path=path,
        )


def save_recording(path, slow: np.ndarray, fs_slow: float, label: int, meta: dict,
                   events: list | None = None, gaps=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["format_version"] = FORMAT_VERSION
    np.savez_compressed(
        path, slow=np.asarray(slow, np.complex64), fs_slow=np.float64(fs_slow),
        label=np.int8(label), gaps=np.asarray(gaps if gaps is not None else [], np.int64),
        events=np.asarray(json.dumps(events or [], ensure_ascii=False)),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False, default=str)),
    )
    return path


def list_recordings(root) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.npz") if p.is_file())
