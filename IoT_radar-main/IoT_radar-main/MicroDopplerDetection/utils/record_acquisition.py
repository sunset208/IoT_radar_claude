#!/usr/bin/env python3
"""Record the micro-Doppler pipeline outputs for a fixed duration (no GUI).

Example — 2 minutes, training set, next auto index ::

    cd MicroDopplerDetection
    python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120

Explicit sample ``AICalibration/data/train/7.*`` (label 0 = empty, 1 = presence / breathing) ::

    python utils/record_acquisition.py --subset train --env salle --label 0 --index 7 --duration 60

``--env`` must be one of the recognised environments (see ``_VALID_ENVS`` in this module).

Layout (default root: ``AICalibration/data`` in the repo) ::

    AICalibration/data/<train|test|val>/<n>.npz|.json|.iq

The metadata useful for training lives in the ``.npz``; a ``<n>.json`` file is
also written with the same information (human-readable, inventory).
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

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

_VALID_ENVS: tuple[str, ...] = ("salle",)


def _write_iq_complex64(path: Path, iq: np.ndarray) -> None:
    """Write IQ samples to a raw ``.iq`` file (dtype complex64 / float32×2)."""
    if iq.size == 0:
        raise ValueError("Signal IQ vide — impossible d'écrire le .iq.")
    z = np.asarray(iq, dtype=np.complex64)
    z.tofile(path)


def _write_iq_stereo_wav(path: Path, iq: np.ndarray, sample_rate_hz: float) -> None:
    """Write complex IQ to a stereo float32 WAV (I = channel 0, Q = channel 1)."""
    if iq.size == 0:
        raise ValueError("Signal IQ vide — impossible d'écrire le WAV.")
    i = np.asarray(iq.real, dtype=np.float32)
    q = np.asarray(iq.imag, dtype=np.float32)
    stereo = np.column_stack((i, q))
    wavfile.write(path, int(round(float(sample_rate_hz))), stereo)


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


def _next_sample_index(split_dir: Path) -> int:
    """Largest existing index ``n`` (``n.npz`` files) + 1, or 1 if empty."""
    if not split_dir.is_dir():
        return 1
    best = 0
    for p in split_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".npz":
            continue
        try:
            n = int(p.stem)
        except ValueError:
            continue
        if n >= 1:
            best = max(best, n)
    return best + 1


_DEFAULT_DATA_HELP = str(default_recording_data_root())


def build_record_argument_parser(
    *,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Build the CLI parser (reusable by ``auto_record``)."""
    desc = description or (
        "Acquisition micro-Doppler : enregistrer spectrogramme, paramètres, "
        "étiquettes, métadonnées et signal IQ décimé sous "
        "AICalibration/data/<train|test|val>/<index>.*"
    )
    p = argparse.ArgumentParser(description=desc)
    p.add_argument(
        "--subset",
        required=True,
        choices=("train", "test", "val"),
        metavar="SUBSET",
        help="Sous-dossier sous la racine données (train, test ou val).",
    )
    p.add_argument(
        "--env",
        required=True,
        choices=_VALID_ENVS,
        metavar="ENV",
        help=f"Environnement de la mesure ({', '.join(_VALID_ENVS)}).",
    )
    p.add_argument(
        "--label",
        required=True,
        type=int,
        choices=(0, 1),
        metavar="{0,1}",
        help="Étiquette de classe : 0 = sans cible (ex. salle vide), 1 = présence / respiration.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Numéro d’échantillon (≥1). Omis : prochain indice libre sous "
            "<data-root>/<subset>/."
        ),
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Répertoire racine des données (défaut : "
            f"{_DEFAULT_DATA_HELP})."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Durée d'enregistrement en secondes (défaut : 120 = 2 min).",
    )
    p.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "config.yaml"),
        help="Chemin vers le fichier YAML de configuration.",
    )
    p.add_argument(
        "--simulation",
        action="store_true",
        help="Forcer le mode simulation (pas de PlutoSDR).",
    )
    p.add_argument(
        "--no-spectrogram",
        action="store_true",
        help="Ne pas stocker les colonnes du spectrogramme (fichier plus léger).",
    )
    p.add_argument(
        "--no-iq-file",
        action="store_true",
        help="Ne pas écrire le fichier .iq (complex64 brut ; désactive la collecte IQ si sans --wav).",
    )
    p.add_argument(
        "--wav",
        action="store_true",
        help="Écrire en complément un .wav stéréo float32 (I canal 0, Q canal 1).",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Fichier de log (voir main.py). Par défaut selon la config.",
    )
    return p


def parse_record_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the recording CLI arguments (defaults to ``sys.argv``)."""
    return build_record_argument_parser().parse_args(argv)


def run(args: argparse.Namespace | None = None) -> Path:
    """Run one acquisition and write the ``.npz``/``.json`` (+ optional IQ/WAV).

    Returns the path to the written ``.npz``.
    """
    if args is None:
        args = parse_record_args()
    md = _load_main_module()

    cfg: dict[str, Any] = md._load_config(args.config)
    log_path = md._setup_logging(
        cfg,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    logger = logging.getLogger(__name__)
    dur = float(args.duration)
    if dur <= 0:
        raise SystemExit("--duration doit être > 0.")

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else default_recording_data_root()
    )
    split_dir = (data_root / args.subset).resolve()
    split_dir.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        if args.index < 1:
            raise SystemExit("--index doit être >= 1.")
        sample_index = int(args.index)
    else:
        sample_index = _next_sample_index(split_dir)

    stem = str(sample_index)
    npz_path = split_dir / f"{stem}.npz"
    json_path = split_dir / f"{stem}.json"
    iq_path = split_dir / f"{stem}.iq"
    wav_path = split_dir / f"{stem}.wav"

    logger.info(
        "Enregistrement — data_root=%s subset=%s env=%s index=%s label=%d",
        data_root,
        args.subset,
        args.env,
        stem,
        int(args.label),
    )

    context = md._build_context(cfg)
    f_hz = np.asarray(context["f_hz"], dtype=np.float64)
    f_s_dec = float(context["f_s_dec"])

    need_iq_tape = (not args.no_iq_file) or args.wav
    decimated_iq_chunks: list[np.ndarray] | None = [] if need_iq_tape else None
    gen = md._streaming_frame_generator(
        cfg,
        simulation=args.simulation,
        decimated_iq_chunks=decimated_iq_chunks,
    )

    t_wall: list[float] = []
    n_trame: list[int] = []
    spectre_cols: list[np.ndarray] | None = [] if not args.no_spectrogram else None

    _outs = [str(npz_path), str(json_path)]
    if not args.no_iq_file:
        _outs.append(str(iq_path))
    if args.wav:
        _outs.append(str(wav_path))
    logger.info(
        "Enregistrement pendant %.1f s — fichiers : %s",
        dur,
        ", ".join(_outs),
    )
    if log_path:
        logger.info("Journal : %s", log_path)

    t0 = time.monotonic()
    n_frames = 0
    stop_reason: str | None = None

    try:
        for frame in gen:
            elapsed = time.monotonic() - t0
            if elapsed >= dur:
                break

            t_wall.append(elapsed)
            n_trame.append(int(frame["n_trame"]))
            if spectre_cols is not None:
                spectre_cols.append(
                    np.asarray(frame["spectre_colonne"], dtype=np.float64)
                )
            n_frames += 1
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        logger.warning(
            "Interruption clavier — sauvegarde partielle (%d trames).", n_frames,
        )
    except OSError as exc:
        en = getattr(exc, "errno", None)
        link_lost = isinstance(exc, BrokenPipeError) or en in (
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ETIMEDOUT,
            errno.ENOTCONN,
        )
        if link_lost:
            stop_reason = "pluto_link_lost"
            logger.error(
                "Liaison PlutoSDR / libiio interrompue (%s). "
                "Vérifier câble USB ou alimentation, éviter les hubs faibles, "
                "désactiver la veille USB, l’adresse IP (uri dans config), "
                "et qu’aucun autre programme n’accède au Pluto. "
                "Sauvegarde partielle : %d trame(s) STFT.",
                exc,
                n_frames,
            )
        else:
            raise
    finally:
        gen.close()

    has_iq_tape = bool(decimated_iq_chunks and len(decimated_iq_chunks) > 0)
    if n_frames == 0 and not has_iq_tape:
        raise SystemExit(
            "Aucune donnée enregistrée. Augmenter la durée, vérifier le warm-up "
            "ou la connexion au PlutoSDR."
        )
    if n_frames == 0 and has_iq_tape:
        logger.warning(
            "Aucune colonne STFT (warm-up ou coupure très précoce) — "
            "fichiers .npz vides côté détection ; fichier .iq / WAV partiel conservé.",
        )

    label_int = int(args.label)
    recorded_wall = float(t_wall[-1]) if t_wall else 0.0

    save_kw: dict[str, Any] = {
        "t_wall_s": np.asarray(t_wall, dtype=np.float64),
        "n_trame": np.asarray(n_trame, dtype=np.int64),
        "f_hz": f_hz,
        "duration_requested_s": np.array(dur, dtype=np.float64),
        "n_frames": np.array(n_frames, dtype=np.int64),
        "label": np.array(label_int, dtype=np.int8),
        "env": np.asarray(str(args.env), dtype=str),
    }
    if spectre_cols is not None and len(spectre_cols) > 0:
        save_kw["spectrogram_db"] = np.stack(spectre_cols, axis=0)

    wrote_iq = False
    wrote_wav = False
    iq_num_samples = 0

    if decimated_iq_chunks is not None and len(decimated_iq_chunks) > 0:
        iq_full = np.concatenate(decimated_iq_chunks)
        n_cap = int(round(dur * f_s_dec))
        if n_cap > 0 and iq_full.size > n_cap:
            iq_full = iq_full[:n_cap]
        iq_num_samples = int(iq_full.size)

        if not args.no_iq_file:
            _write_iq_complex64(iq_path, iq_full)
            wrote_iq = True
            logger.info(
                "IQ décimé (.iq) — %d échantillons complexes @ %.1f Hz — %s",
                iq_full.size,
                f_s_dec,
                iq_path,
            )

        if args.wav:
            _write_iq_stereo_wav(wav_path, iq_full, f_s_dec)
            wrote_wav = True
            logger.info(
                "WAV IQ décimé — %d échantillons complexes @ %.1f Hz — %s",
                iq_full.size,
                f_s_dec,
                wav_path,
            )
    else:
        if not args.no_iq_file:
            logger.warning("Aucun bloc IQ collecté — fichier .iq non créé.")
        if args.wav:
            logger.warning("Aucun bloc IQ — fichier .wav non créé.")

    cfg_resolved = str(Path(args.config).resolve())
    utc_finished_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    save_kw["subset"] = np.asarray(args.subset, dtype=str)
    save_kw["sample_index"] = np.int64(sample_index)
    save_kw["config_path"] = np.asarray(cfg_resolved, dtype=str)
    save_kw["simulation"] = np.array(bool(args.simulation), dtype=np.bool_)
    save_kw["f_s_dec_hz"] = np.float64(f_s_dec)
    save_kw["n_fft"] = np.int64(f_hz.size)
    save_kw["duration_recorded_wall_s"] = np.float64(recorded_wall)
    save_kw["data_root"] = np.asarray(str(data_root.resolve()), dtype=str)
    save_kw["utc_finished"] = np.asarray(utc_finished_iso, dtype=str)
    save_kw["includes_spectrogram"] = np.array(
        spectre_cols is not None and len(spectre_cols) > 0,
        dtype=np.bool_,
    )
    save_kw["includes_iq_file"] = np.array(wrote_iq, dtype=np.bool_)
    save_kw["includes_wav_file"] = np.array(wrote_wav, dtype=np.bool_)
    save_kw["iq_num_complex_samples"] = np.int64(iq_num_samples)
    if stop_reason is not None:
        save_kw["acquisition_stop_reason"] = np.asarray(stop_reason, dtype=str)

    np.savez_compressed(npz_path, **save_kw)

    meta_json: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "subset": args.subset,
        "env": args.env,
        "label": label_int,
        "sample_index": sample_index,
        "sample_dir": str(split_dir.resolve()),
        "npz_file": str(npz_path.resolve()),
        "config_path": cfg_resolved,
        "simulation": bool(args.simulation),
        "duration_requested_s": dur,
        "duration_recorded_wall_s": recorded_wall,
        "n_frames": n_frames,
        "n_fft": int(f_hz.size),
        "f_s_dec_hz": f_s_dec,
        "includes_spectrogram": bool(
            spectre_cols is not None and len(spectre_cols) > 0,
        ),
        "includes_iq_file": wrote_iq,
        "includes_wav_file": wrote_wav,
        "iq_num_complex_samples": iq_num_samples,
        "utc_finished": utc_finished_iso,
    }
    if wrote_iq:
        meta_json["iq_file"] = str(iq_path.resolve())
    else:
        meta_json["iq_file"] = None
        if args.no_iq_file:
            meta_json["iq_file_disabled"] = True

    if wrote_wav:
        meta_json["wav_file"] = str(wav_path.resolve())
        meta_json["wav_format"] = (
            "stereo float32; channel_0=I (real), channel_1=Q (imag)"
        )
        meta_json["wav_samples"] = iq_num_samples
    else:
        meta_json["wav_file"] = None

    if stop_reason is not None:
        meta_json["acquisition_stop_reason"] = stop_reason

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(meta_json, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    logger.info(
        "%d trames sauvegardées — wall time dernière trame ≈ %.2f s",
        n_frames,
        recorded_wall,
    )
    logger.info("Fichiers : %s, %s", npz_path, json_path)
    return npz_path


def main() -> None:
    """CLI entry point: run a single acquisition."""
    _ensure_paths()
    run()


if __name__ == "__main__":
    main()
