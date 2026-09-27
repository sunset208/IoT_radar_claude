"""Batch-mode detection helpers (full spectrogram input).

Streaming pipeline uses
:func:`MicroDopplerDetection.pipeline.detection.detect_presence_column`
which operates on a single STFT column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from MicroDopplerDetection.pipeline.detection import (
    _acf_peak,
    _fisher_pvalue,
    _fusion_score,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Outcome of a single respiration-detection evaluation."""

    snr_db: float
    alert: bool
    p_signal: float
    p_reference: float
    p_value_f: float = 1.0
    acf_peak: float = 0.0
    fv_estimated: float | None = None
    score_presence: float = 0.0


def detect_respiration(
    S_db: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    seuil_snr_dB: float,
) -> DetectionResult:
    """Evaluate respiration presence by simple SNR thresholding (legacy)."""
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)

    if not np.any(mask_sig):
        raise ValueError(
            f"Masque de la bande respiration vide ({f_lo_sig}–{f_hi_sig} Hz)."
        )
    if not np.any(mask_ref):
        raise ValueError(
            f"Masque de la bande de référence vide ({f_lo_ref}–{f_hi_ref} Hz)."
        )

    S_lin = 10.0 ** (S_db / 10.0)
    p_signal = float(np.mean(S_lin[mask_sig, :]))
    p_reference = float(np.mean(S_lin[mask_ref, :]))

    eps = 1e-30
    snr_db = 10.0 * np.log10(p_signal / (p_reference + eps))
    alert = bool(snr_db >= seuil_snr_dB)

    return DetectionResult(
        snr_db=snr_db,
        alert=alert,
        p_signal=p_signal,
        p_reference=p_reference,
    )


def snr_db_per_column(
    S_db: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-column SNR for a 2-D dB spectrogram (legacy display)."""
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)

    if not np.any(mask_sig) or not np.any(mask_ref):
        raise ValueError("Masque vide — vérifier la résolution fréquentielle.")

    S_lin = 10.0 ** (S_db / 10.0)
    p_sig = np.mean(S_lin[mask_sig, :], axis=0)
    p_ref = np.mean(S_lin[mask_ref, :], axis=0)

    eps = 1e-30
    snr = 10.0 * np.log10(p_sig / (p_ref + eps))
    return snr, p_sig, p_ref


def detect_snr_column(
    col_db: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
) -> float:
    """Plain band-power SNR of one STFT column (kept for legacy plots)."""
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)

    col_lin = 10.0 ** (col_db / 10.0)
    p_sig = float(np.mean(col_lin[mask_sig]))
    p_ref = float(np.mean(col_lin[mask_ref]))
    eps = 1e-30
    return float(10.0 * np.log10(p_sig / (p_ref + eps)))


def detect_presence(
    S_db: np.ndarray,
    f_hz: np.ndarray,
    phi: np.ndarray,
    f_s: float,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    alpha: float,
    w: float = 0.5,
    p_value_decades: float = 3.0,
    acf_floor: float = 0.2,
    acf_good: float = 0.7,
) -> DetectionResult:
    """Fused Fisher / ACF detection on a full 2-D spectrogram (legacy)."""
    S_lin = 10.0 ** (S_db / 10.0)
    p_value_f, _ = _fisher_pvalue(S_lin, f_hz, bande_respiration, bande_reference)
    acf_peak, fv_estimated = _acf_peak(phi, f_s, bande_respiration)
    score_presence = _fusion_score(
        p_value_f,
        acf_peak,
        w=w,
        p_value_decades=p_value_decades,
        acf_floor=acf_floor,
        acf_good=acf_good,
    )

    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference
    abs_f = np.abs(f_hz)
    mask_sig = (abs_f >= f_lo_sig) & (abs_f <= f_hi_sig)
    mask_ref = (abs_f >= f_lo_ref) & (abs_f <= f_hi_ref)
    p_signal = float(np.mean(S_lin[mask_sig]))
    p_reference = float(np.mean(S_lin[mask_ref]))
    eps = 1e-30
    snr_db = float(10.0 * np.log10(p_signal / (p_reference + eps)))

    alert = bool(p_value_f < alpha)

    return DetectionResult(
        snr_db=snr_db,
        alert=alert,
        p_signal=p_signal,
        p_reference=p_reference,
        p_value_f=p_value_f,
        acf_peak=acf_peak,
        fv_estimated=fv_estimated,
        score_presence=score_presence,
    )


__all__ = [
    "DetectionResult",
    "detect_respiration",
    "snr_db_per_column",
    "detect_snr_column",
    "detect_presence",
]
