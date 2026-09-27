"""Chemins stables du dépôt (racine IoT_radar ↔ données calibration)."""

from __future__ import annotations

from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent
MICRO_DOPPLER_ROOT: Path = _UTILS_DIR.parent
REPO_ROOT: Path = MICRO_DOPPLER_ROOT.parent


def default_recording_data_root() -> Path:
    """Racine par défaut des enregistrements micro-Doppler (train/test/val)."""
    return (REPO_ROOT / "AICalibration" / "data").resolve()
