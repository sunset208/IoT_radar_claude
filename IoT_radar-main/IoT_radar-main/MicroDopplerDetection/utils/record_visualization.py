#!/usr/bin/env python3
"""Offline replay of a recording — same RX/TX dashboard as ``display.py``,
without the "presence score" panel (the .npz no longer carries those series).

Examples ::

    cd MicroDopplerDetection
    python utils/record_visualization.py --subset train --index 1

    python utils/record_visualization.py --npz ../AICalibration/data/train/3.npz

The ``.npz`` must contain ``spectrogram_db``. YAML path: ``--config``, the
``config_path`` key inside the ``.npz``, or otherwise the ``.json`` of the same index.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_UTILS_DIR = Path(__file__).resolve().parent
_ROOT = _UTILS_DIR.parent
_REPO_ROOT = _ROOT.parent


def _ensure_paths() -> None:
    """Make both ``utils.*`` (via the package dir) and ``MicroDopplerDetection.*``
    (via the repo root, required by ``main.py``) importable, regardless of the
    directory the script is launched from."""
    for p in (_REPO_ROOT, _ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_ensure_paths()

from utils.repo_paths import default_recording_data_root


def _load_main_module():
    """Load the package's ``main.py`` unambiguously w.r.t. a third-party ``main`` module."""
    path = _ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("microdoppler_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {path}")
    mod = importlib.util.module_from_spec(spec)
    _ensure_paths()
    spec.loader.exec_module(mod)
    return mod


def _resolve_npz_path(args: argparse.Namespace) -> Path:
    """Resolve the ``.npz`` path from ``--npz`` or ``--subset``/``--index``."""
    if args.npz is not None:
        return Path(args.npz).expanduser().resolve()
    if args.subset is None or args.index is None:
        raise SystemExit(
            "Fournir --npz CHEMIN.npz ou bien --subset {train,test,val} et --index N (N ≥ 1)."
        )
    if args.index < 1:
        raise SystemExit("--index doit être >= 1.")
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else default_recording_data_root()
    )
    return (data_root / args.subset / f"{int(args.index)}.npz").resolve()


def _load_config_path_from_recording(npz_path: Path) -> str | None:
    """Return the config path stored in the ``.npz`` (or its ``.json`` sidecar), if any."""
    with np.load(npz_path, allow_pickle=False) as z:
        if "config_path" in z.files:
            return str(np.asarray(z["config_path"]).item())
    legacy = npz_path.with_suffix(".json")
    if not legacy.is_file():
        return None
    try:
        with open(legacy, encoding="utf-8") as fh:
            meta: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError:
        logging.warning("JSON illisible — %s", legacy)
        return None
    p = meta.get("config_path")
    return str(p) if isinstance(p, str) and p else None


def _label_from_json_sidecar(npz_path: Path) -> int | None:
    """Read ``label`` from the ``.json`` sidecar next to the ``.npz``."""
    legacy = npz_path.with_suffix(".json")
    if not legacy.is_file():
        return None
    try:
        with open(legacy, encoding="utf-8") as fh:
            meta = json.load(fh)
        v = meta.get("label")
        return int(v) if v is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _frames_from_npz(data: Any) -> tuple[list[dict[str, Any]], int]:
    """Build replay frames from a recording's ``spectrogram_db`` array.

    Returns ``(frames, n_frames)``.
    """
    if "spectrogram_db" not in data:
        raise SystemExit(
            "Ce .npz ne contient pas « spectrogram_db » — "
            "ré-enregistrer sans l’option « --no-spectrogram »."
        )
    spec = np.asarray(data["spectrogram_db"], dtype=np.float64)
    n_spec = spec.shape[0]
    if n_spec == 0:
        raise SystemExit("Aucune trame exploitable dans le .npz.")

    if "n_trame" in data.files:
        ntr = np.asarray(data["n_trame"], dtype=np.int64).ravel()
        n = int(min(n_spec, ntr.size))
    else:
        ntr = None
        n = n_spec

    frames: list[dict[str, Any]] = []
    for i in range(n):
        n_trame_val = int(ntr[i]) if ntr is not None and ntr.size > i else i + 1
        frames.append(
            {
                "spectre_colonne": np.asarray(spec[i], dtype=np.float64),
                "n_trame": n_trame_val,
            },
        )
    return frames, n


def main() -> None:
    """CLI entry point: replay a recorded ``.npz`` in the dashboard."""
    _ensure_paths()
    from utils.display import DashboardRadar

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Visualiser un .npz (record_acquisition) — dashboard comme main.py.",
    )
    p.add_argument("--npz", type=Path, default=None, help="Fichier .npz (prioritaire sur --subset/--index).")
    p.add_argument(
        "--subset",
        choices=("train", "test", "val"),
        default=None,
        metavar="SUBSET",
        help="Sous-dossier sous data/ (avec --index).",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="Indice échantillon (≥1), avec --subset.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Racine data (défaut : {default_recording_data_root()}).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML (prioritaire sur config_path dans le .npz puis .json hérité).",
    )
    p.add_argument(
        "--interval-ms",
        type=float,
        default=30.0,
        help="Intervalle entre trames replay (ms).",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Reboucler la lecture.",
    )
    args = p.parse_args()

    npz_path = _resolve_npz_path(args)
    if not npz_path.is_file():
        raise SystemExit(f"Fichier introuvable : {npz_path}")

    cfg_default = str(_ROOT / "configs" / "config.yaml")
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = _load_config_path_from_recording(npz_path) or cfg_default
    cfg_path = str(Path(cfg_path).expanduser().resolve())

    md = _load_main_module()
    cfg: dict[str, Any] = md._load_config(cfg_path)
    context = md._build_context(cfg)

    with np.load(npz_path, allow_pickle=False) as data:
        rec_label: int | None = None
        if "label" in data.files:
            rec_label = int(np.asarray(data["label"]).item())
        if "f_hz" in data:
            context = {**context, "f_hz": np.asarray(data["f_hz"], dtype=np.float64)}
        frames, n_tr = _frames_from_npz(data)

    if rec_label is None:
        rec_label = _label_from_json_sidecar(npz_path)

    logging.info(
        "Relecture — %d trames — label=%s — %s — config %s",
        n_tr,
        rec_label if rec_label is not None else "?",
        npz_path,
        cfg_path,
    )

    dashboard = DashboardRadar(config=cfg, context=context, show_presence_score=False)
    title = "Radar Micro-Doppler — Relecture enregistrement"
    if rec_label is not None:
        title = f"{title}    ·    label {rec_label}"

    dashboard._fig.suptitle(
        title,
        fontsize=13,
        fontweight="bold",
    )

    def _init() -> tuple:
        return dashboard.update_frame(None)

    def _step(fr: dict[str, Any]) -> tuple:
        return dashboard.update_frame(fr)

    anim = FuncAnimation(
        dashboard._fig,
        _step,
        init_func=_init,
        frames=frames,
        interval=max(1.0, float(args.interval_ms)),
        blit=True,
        cache_frame_data=False,
        repeat=bool(args.loop),
    )
    _ = anim

    plt.show()


if __name__ == "__main__":
    main()
