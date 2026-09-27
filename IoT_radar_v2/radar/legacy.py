"""Conversion des enregistrements v1 (``AICalibration/data/<split>/<n>.iq``) au format v2.

Le ``.iq`` v1 contient l'IQ **décimé à 2 kHz après le filtre de clutter**
(Butterworth passe-haut 0.05 Hz, qui n'affecte que le DC) avec la porteuse
décalée à +f_offset = 500 Hz.  L'écho utile y est donc intact : on le ramène
à 0 Hz (NCO), on filtre et on décime à fs_slow → slow-time v2.

Les ``.npz`` v1 sans ``.iq`` (spectrogramme dB seul) ne sont pas
récupérables : la phase — l'information essentielle — a été jetée.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy import signal

from radar.recorder import save_recording

logger = logging.getLogger(__name__)


def convert_v1_iq(iq_path: Path, npz_path: Path | None, out_dir: Path, fs_slow: float = 50.0,
                  f_offset: float = 500.0, f_c: float = 3.5e9, skip_s: float = 5.0) -> Path | None:
    meta_v1: dict = {}
    label = -1
    fs_in = 2000.0
    if npz_path is not None and npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as d:
            if "label" in d.files:
                label = int(np.asarray(d["label"]).item())
            if "f_s_dec_hz" in d.files:
                fs_in = float(d["f_s_dec_hz"])
            for k in ("env", "subset", "sample_index", "utc_finished", "simulation"):
                if k in d.files:
                    v = np.asarray(d[k]).item()
                    meta_v1[k] = v.item() if hasattr(v, "item") else v
    x = np.fromfile(iq_path, dtype=np.complex64).astype(np.complex128)
    if x.size < fs_in * 30:
        logger.warning("%s trop court (%.1f s) — ignoré", iq_path.name, x.size / fs_in)
        return None
    n = np.arange(x.size)
    x *= np.exp(-2j * np.pi * f_offset * n / fs_in)
    q = fs_in / fs_slow
    if abs(q - round(q)) > 1e-9:
        raise ValueError(f"fs_in/fs_slow = {q} non entier")
    q = int(round(q))
    y = x
    rem = q
    f = fs_in
    for st in (10, 8, 5, 4, 2):
        while rem % st == 0 and rem > 1:
            sos = signal.ellip(8, 0.05, 80, 0.4 * f / st, fs=f, output="sos")
            y = signal.sosfiltfilt(sos, y)[::st]
            f /= st
            rem //= st
    if rem != 1:
        y = signal.resample_poly(y, 1, rem)
    y = y[int(skip_s * fs_slow):]
    wavelength = 299_792_458.0 / f_c
    meta = {
        "converted_from": "v1",
        "v1_iq": str(iq_path),
        "v1_meta": meta_v1,
        "session": f"v1_{meta_v1.get('subset', 'x')}_{iq_path.stem}",
        "config": {"sdr": {"f_c": f_c}, "frontend": {"fs_slow": fs_slow}},
        "wavelength": wavelength,
        "warning": ("v1 : pertes d'échantillons Pluto non détectées à l'époque ; "
                    "f_c supposée"),
    }
    out = out_dir / f"v1_{meta_v1.get('subset', 'x')}_{iq_path.stem}_{'resp' if label == 1 else 'vide' if label == 0 else 'inconnu'}.npz"
    save_recording(out, y.astype(np.complex64), fs_slow, label, meta)
    return out


def convert_tree(root: Path, out_dir: Path, **kw) -> list[Path]:
    outs = []
    for iq in sorted(root.rglob("*.iq")):
        npz = iq.with_suffix(".npz")
        jsn = iq.with_suffix(".json")
        if jsn.is_file():
            try:
                js = json.loads(jsn.read_text(encoding="utf-8"))
                if js.get("simulation"):
                    logger.info("%s : enregistrement simulé v1 — ignoré", iq.name)
                    continue
            except Exception:
                pass
        o = convert_v1_iq(iq, npz, out_dir, **kw)
        if o is not None:
            outs.append(o)
            logger.info("%s → %s", iq, o.name)
    return outs
