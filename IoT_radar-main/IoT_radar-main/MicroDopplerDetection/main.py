"""CLI entry point for the micro-Doppler radar pipeline.

Usage
-----
Hardware mode (continuous, default config)::
    python -m MicroDopplerDetection.main

Simulation mode (continuous)::
    python -m MicroDopplerDetection.main --simulation

Custom config file::
    python -m MicroDopplerDetection.main --config MicroDopplerDetection/configs/my.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import math
import sys
from pathlib import Path
from typing import Any, Generator

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml
import numpy as np

from MicroDopplerDetection.pipeline.emission import generate_tx_buffer
from MicroDopplerDetection.pipeline.acquisition import (
    stream_pluto,
    stream_simulation,
)
from MicroDopplerDetection.pipeline.decimation import Decimator
from MicroDopplerDetection.pipeline.clutter import ClutterFilter
from MicroDopplerDetection.pipeline.spectrogramme import compute_single_column
from MicroDopplerDetection.pipeline.detection import detect_presence_column
from MicroDopplerDetection.pipeline.windowing import get_window
from MicroDopplerDetection.utils.display import DashboardRadar

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0
_BOLTZMANN: float = 1.380649e-23
_T0: float = 290.0

_PACKAGE_ROOT: Path = Path(__file__).resolve().parent
_DEFAULT_CONFIG: Path = _PACKAGE_ROOT / "configs" / "config.yaml"
_LOGS_DIR: Path = _PACKAGE_ROOT / "logs"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_f_offset(cfg: dict[str, Any]) -> float:
    """Return the effective baseband offset (Hz), 0 if not in cw_offset mode."""
    emi = cfg.get("emission", {})
    if emi.get("mode") == "cw_offset":
        return float(emi.get("f_offset", 0.0))
    return 0.0


def _auto_skip_warmup(
    clu_cfg: dict[str, Any],
    f_s_dec: float,
    hop: int,
) -> int:
    """Compute a sensible warm-up length from the active clutter filter.

    The clutter filter has a transient of ~3·τ.  We translate it into a
    number of STFT hops and round up.  The user can still override via
    ``spectrogramme.skip_warmup`` in the config.
    """
    mode = clu_cfg.get("mode", "butterworth")
    if mode in ("iir", "mean"):
        alpha = float(clu_cfg.get("alpha", 0.9999))
        tau_s = 1.0 / max((1.0 - alpha) * f_s_dec, 1e-12)
    elif mode == "butterworth":
        f_cut = float(clu_cfg.get("butterworth_cutoff", 0.05))
        tau_s = 1.0 / (2.0 * math.pi * max(f_cut, 1e-6))
    else:
        tau_s = 0.0

    hop_s = hop / f_s_dec
    return int(math.ceil(3.0 * tau_s / hop_s)) if hop_s > 0 else 0


# ------------------------------------------------------------------
# Radar range equation
# ------------------------------------------------------------------

def _radar_range(
    params: dict[str, Any],
    wavelength: float,
    B_hz: float,
) -> float:
    """Evaluate the monostatic radar range equation."""
    P_tx = 1e-3 * 10.0 ** (params["P_tx_dBm"] / 10.0)
    G_tx = 10.0 ** (params["G_tx_dBi"] / 10.0)
    G_rx = 10.0 ** (params["G_rx_dBi"] / 10.0)
    sigma = params["sigma_m2"]
    F = 10.0 ** (params["NF_dB"] / 10.0)
    L = 10.0 ** (params["L_sys_dB"] / 10.0)
    SNR_min = 10.0 ** (params["SNR_min_dB"] / 10.0)

    numerator = P_tx * G_tx * G_rx * wavelength**2 * sigma
    denominator = (4.0 * np.pi)**3 * _BOLTZMANN * _T0 * B_hz * F * L * SNR_min
    return float((numerator / denominator) ** 0.25)


def _compute_range(cfg: dict[str, Any]) -> tuple[float, float]:
    """Return ``(R_min, R_max)`` — pessimistic and optimistic ranges (m)."""
    bl = cfg["bilan_liaison"]
    f_c = float(cfg["sdr"]["f_c"])
    wavelength = _SPEED_OF_LIGHT / f_c
    B_hz = float(bl["B_eff_hz"])

    R_max = _radar_range(bl["optimiste"], wavelength, B_hz)
    R_min = _radar_range(bl["pessimiste"], wavelength, B_hz)

    logger.info(
        "Portée effective — R_min = %.1f m (pire-cas) / R_max = %.1f m (optimiste) "
        "(B=%.3f Hz, λ=%.3f m)",
        R_min,
        R_max,
        B_hz,
        wavelength,
    )
    return R_min, R_max


# ------------------------------------------------------------------
# Configuration loading
# ------------------------------------------------------------------

def _load_config(path: str) -> dict[str, Any]:
    """Load and return the YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        print(f"ERREUR : fichier de configuration introuvable : {path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _setup_logging(cfg: dict[str, Any], log_file: Path | None = None) -> Path | None:
    """Configure the root logger from the config ``logging`` section.

    A console handler is always installed.  If *log_file* is provided
    (or if ``logging.to_file`` is true in the config), a parallel
    ``FileHandler`` writes the same records to disk so that runs can be
    audited offline.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary; reads ``logging.level`` and
        optionally ``logging.to_file`` (default: ``True``).
    log_file : pathlib.Path or None, optional
        Explicit log-file path.  When ``None``, a timestamped file is
        created under ``MicroDopplerDetection/logs/``.

    Returns
    -------
    pathlib.Path or None
        Path of the file handler (``None`` if file logging is disabled).
    """
    log_cfg = cfg.get("logging", {})
    level_name = log_cfg.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    write_to_file = bool(log_cfg.get("to_file", True))
    if not write_to_file:
        return None

    if log_file is None:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = _LOGS_DIR / f"radar_{stamp}.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    root.addHandler(file_handler)
    logger.info("Journal écrit dans %s", log_file)
    return log_file


# ------------------------------------------------------------------
# Streaming pipeline
# ------------------------------------------------------------------

def _build_iq_stream(
    cfg: dict[str, Any],
    simulation: bool,
) -> Generator[np.ndarray, None, None]:
    """Return an infinite IQ-buffer generator (hardware or simulation)."""
    sdr = cfg["sdr"]
    sim = cfg["simulation"]
    f_off = _resolve_f_offset(cfg)

    if simulation or sim.get("enable", False):
        logger.info("Mode simulation continu activé (f_offset=%.1f Hz)", f_off)
        return stream_simulation(
            f_c=sdr["f_c"],
            f_s=sdr["f_s"],
            buffer_size=sdr["buffer_size"],
            fv=sim["fv"],
            D_mm=sim["D_mm"],
            snr_dB=sim["snr_dB"],
            f_offset=f_off,
            clutter_amplitude=sim.get("clutter_amplitude", 100.0),
        )

    logger.info("Mode matériel continu — connexion au PlutoSDR")
    tx_buffer = generate_tx_buffer(
        mode=cfg["emission"]["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=sdr["f_s"],
        f_offset=f_off,
    )
    return stream_pluto(
        uri=sdr["uri"],
        f_c=sdr["f_c"],
        f_s=sdr["f_s"],
        rx_gain=sdr["rx_gain"],
        tx_gain=sdr["tx_gain"],
        buffer_size=sdr["buffer_size"],
        tx_buffer=tx_buffer,
    )


def _streaming_frame_generator(
    cfg: dict[str, Any],
    simulation: bool,
    *,
    decimated_iq_chunks: list[np.ndarray] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Acquire → decimate → clutter → FFT → detect → yield, indefinitely.

    Parameters
    ----------
    cfg, simulation
        Identique aux autres appels du pipeline.
    decimated_iq_chunks
        Si fourni, chaque bloc ``iq`` après décimation et filtre clutter est
        recopié dans cette liste (*debug / enregistrement .wav hors ligne*).
    """
    sdr_cfg = cfg["sdr"]
    dec_cfg = cfg["decimation"]
    clu_cfg = cfg["clutter"]
    spec_cfg = cfg["spectrogramme"]
    win_cfg = cfg["windowing"]
    det_cfg = cfg["detection"]

    f_s = float(sdr_cfg["f_s"])
    f_c = float(sdr_cfg["f_c"])
    n_fft = int(spec_cfg["n_fft"])
    overlap = float(spec_cfg["overlap"])
    hop = max(1, int(n_fft * (1.0 - overlap)))

    do_decimate = dec_cfg.get("enable", True)
    D = int(dec_cfg["D"]) if do_decimate else 1
    f_max_utile = float(dec_cfg["f_max_utile"])

    f_off = _resolve_f_offset(cfg)
    bande_resp_bb = tuple(det_cfg["bande_respiration"])
    bande_ref_bb = tuple(det_cfg["bande_reference"])

    f_max_eff = max(f_max_utile, abs(f_off) + bande_ref_bb[1])
    decimator = Decimator(f_s=f_s, D=D, f_max_utile=f_max_eff)
    f_s_dec = decimator.f_s_out

    clutter_filter = ClutterFilter(
        mode=clu_cfg["mode"],
        fs=f_s_dec,
        alpha=float(clu_cfg["alpha"]),
        butterworth_order=int(clu_cfg["butterworth_order"]),
        butterworth_cutoff=float(clu_cfg["butterworth_cutoff"]),
    )
    window = get_window(win_cfg["mode"], n_fft)

    w = float(det_cfg["w"])
    alpha_alert = float(det_cfg["alpha"])
    p_value_decades = float(det_cfg["p_value_decades"])
    acf_floor = float(det_cfg["acf_floor"])
    acf_good = float(det_cfg["acf_good"])

    acf_buffer_seconds = float(det_cfg["acf_buffer_seconds"])
    acf_len = max(1, int(round(acf_buffer_seconds * f_s_dec)))

    user_warmup = spec_cfg.get("skip_warmup")
    auto_warmup = _auto_skip_warmup(clu_cfg, f_s_dec, hop)
    skip_warmup = int(user_warmup) if user_warmup is not None else auto_warmup

    buf = np.empty(0, dtype=np.complex64)
    iq_stream = _build_iq_stream(cfg, simulation)

    logger.info(
        "Pipeline streaming — n_fft=%d, hop=%d, f_s_dec=%.1f Hz, "
        "skip_warmup=%d (auto=%d), f_offset=%.1f Hz",
        n_fft,
        hop,
        f_s_dec,
        skip_warmup,
        auto_warmup,
        f_off,
    )
    logger.info(
        "Bandes (relatives à la porteuse) — respiration=%.2f–%.2f Hz, "
        "référence=%.2f–%.2f Hz, centre spectral f_offset=%.1f Hz",
        bande_resp_bb[0],
        bande_resp_bb[1],
        bande_ref_bb[0],
        bande_ref_bb[1],
        f_off,
    )

    phi_iq_hist = np.empty(0, dtype=np.complex128)
    n_dec_seen = 0

    frame_counter = 0

    for raw_buf in iq_stream:
        iq_dec = decimator(raw_buf)
        iq_filt = clutter_filter(iq_dec)

        if decimated_iq_chunks is not None:
            decimated_iq_chunks.append(
                np.asarray(iq_filt, dtype=np.complex64).copy()
            )

        n_chunk = len(iq_filt)
        if f_off != 0.0:
            k = np.arange(n_chunk, dtype=np.float64) + n_dec_seen
            demod = np.exp(-1j * 2.0 * np.pi * f_off * k / f_s_dec)
            iq_demod = iq_filt.astype(np.complex128) * demod
        else:
            iq_demod = iq_filt.astype(np.complex128)
        n_dec_seen += n_chunk
        phi_iq_hist = np.concatenate((phi_iq_hist, iq_demod))
        if len(phi_iq_hist) > acf_len:
            phi_iq_hist = phi_iq_hist[-acf_len:]

        buf = np.concatenate((buf, np.asarray(iq_filt, dtype=np.complex64)))

        while len(buf) >= n_fft:
            segment = buf[:n_fft].copy()
            buf = buf[hop:]
            frame_counter += 1

            if frame_counter <= skip_warmup:
                logger.debug(
                    "Warm-up : trame %d/%d ignorée", frame_counter, skip_warmup,
                )
                continue

            col = compute_single_column(segment, f_s_dec, f_c, window)

            phi_hist = np.unwrap(np.angle(phi_iq_hist))

            score_presence, p_value_f, acf_peak, fv_estimated = (
                detect_presence_column(
                    col_db=col.col_db,
                    f_hz=col.f_hz,
                    phi_buffer=phi_hist,
                    f_s=f_s_dec,
                    bande_respiration=bande_resp_bb,
                    bande_reference=bande_ref_bb,
                    f_center=f_off,
                    w=w,
                    p_value_decades=p_value_decades,
                    acf_floor=acf_floor,
                    acf_good=acf_good,
                )
            )
            alert = bool(p_value_f < alpha_alert)

            yield {
                "signal_iq_dec":   segment,
                "spectre_colonne": col.col_db,
                "score_presence":  score_presence,
                "p_value_f":       p_value_f,
                "acf_peak":        acf_peak,
                "fv_estimated":    fv_estimated,
                "n_trame":         frame_counter,
                "detection":       alert,
            }


# ------------------------------------------------------------------
# Build context for the dashboard
# ------------------------------------------------------------------

def _build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the dashboard context dict from config alone."""
    sdr = cfg["sdr"]
    emi = cfg["emission"]
    dec_cfg = cfg["decimation"]
    spec_cfg = cfg["spectrogramme"]
    det_cfg = cfg["detection"]

    f_s = float(sdr["f_s"])
    f_c = float(sdr["f_c"])
    D = int(dec_cfg["D"]) if dec_cfg.get("enable", True) else 1
    f_s_dec = f_s / D
    n_fft = int(spec_cfg["n_fft"])

    f_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / f_s_dec)).astype(np.float64)

    f_off = _resolve_f_offset(cfg)
    tx_buffer = generate_tx_buffer(
        mode=emi["mode"],
        buffer_size=sdr["buffer_size"],
        f_s=f_s,
        f_offset=f_off,
    )
    tx_spectrum = np.fft.fftshift(np.fft.fft(tx_buffer, n=len(tx_buffer)))
    eps = 1e-12
    spectre_tx_db = 20.0 * np.log10(np.abs(tx_spectrum) + eps).astype(np.float64)
    f_hz_tx = np.fft.fftshift(
        np.fft.fftfreq(len(tx_buffer), d=1.0 / f_s)
    ).astype(np.float64)

    R_min, R_max = _compute_range(cfg)

    df_hz = f_s_dec / n_fft
    wavelength = _SPEED_OF_LIGHT / f_c
    dv_mps = df_hz * wavelength / 2.0

    clutter_mode = cfg["clutter"]["mode"]
    
    bande_resp_bb = list(det_cfg["bande_respiration"])
    B_eff_hz = float(cfg["bilan_liaison"]["B_eff_hz"])

    return {
        "f_hz": f_hz,
        "f_s_dec": f_s_dec,
        "spectre_tx_db": spectre_tx_db,
        "f_hz_tx": f_hz_tx,
        "R_min_m": R_min,
        "R_max_m": R_max,
        "df_hz": df_hz,
        "dv_mps": dv_mps,
        "n_fft": n_fft,
        "clutter_mode": clutter_mode,
        "bande_resp": bande_resp_bb,
        "B_eff_hz": B_eff_hz,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Radar micro-Doppler — détection de survivants ensevelis",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Chemin vers le fichier de configuration YAML (défaut : %(default)s)",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Forcer le mode simulation (pas de PlutoSDR requis)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Chemin explicite du fichier de log.  Par défaut, "
            "MicroDopplerDetection/logs/radar_<timestamp>.log."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Top-level pipeline orchestration (streaming mode)."""
    args = _parse_args()
    cfg = _load_config(args.config)
    log_path = _setup_logging(
        cfg,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    logger.info("=== Démarrage du pipeline micro-Doppler (mode continu) ===")
    logger.info("Configuration chargée depuis %s", args.config)
    if log_path is not None:
        logger.info("Logs persistants : %s", log_path)

    context = _build_context(cfg)
    dashboard = DashboardRadar(config=cfg, context=context)
    gen = _streaming_frame_generator(cfg, simulation=args.simulation)
    dashboard.run(gen)

    logger.info("=== Pipeline terminé ===")


if __name__ == "__main__":
    main()
