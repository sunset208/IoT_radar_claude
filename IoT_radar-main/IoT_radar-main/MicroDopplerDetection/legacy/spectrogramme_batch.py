"""Batch STFT (full spectrogram in one call).

Streaming pipeline uses
:func:`MicroDopplerDetection.pipeline.spectrogramme.compute_single_column`
instead, which produces one column at a time.  This module is kept for
offline analysis and reference plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft as _scipy_stft

from MicroDopplerDetection.pipeline.windowing import get_window

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0


@dataclass
class SpectrogramOutput:
    """Container for STFT results and derived physical axes."""

    Z: np.ndarray
    S_db: np.ndarray
    f_hz: np.ndarray
    v_mps: np.ndarray
    t_s: np.ndarray
    df_hz: float
    dv_mps: float


def compute_spectrogram(
    iq: np.ndarray,
    f_s: float,
    f_c: float,
    n_fft: int,
    overlap: float,
    window_mode: str,
) -> SpectrogramOutput:
    """Compute the STFT of a decimated, clutter-suppressed IQ signal."""
    wavelength = _SPEED_OF_LIGHT / f_c
    nperseg = n_fft
    noverlap = int(n_fft * overlap)

    window = get_window(window_mode, nperseg)

    logger.info(
        "STFT — n_fft=%d, overlap=%.0f%%, fenêtre='%s', f_s=%.1f Hz",
        n_fft,
        overlap * 100,
        window_mode,
        f_s,
    )

    f_raw, t_raw, Z_raw = _scipy_stft(
        iq,
        fs=f_s,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=False,
    )

    f_hz = np.fft.fftshift(f_raw)
    Z = np.fft.fftshift(Z_raw, axes=0)
    t_s = np.asarray(t_raw, dtype=np.float64)

    eps = 1e-12
    S_db = 10.0 * np.log10(np.abs(Z) ** 2 + eps)

    v_mps = f_hz * wavelength / 2.0
    df_hz = f_s / n_fft
    dv_mps = df_hz * wavelength / 2.0

    return SpectrogramOutput(
        Z=Z,
        S_db=S_db.astype(np.float64),
        f_hz=f_hz.astype(np.float64),
        v_mps=v_mps.astype(np.float64),
        t_s=t_s,
        df_hz=df_hz,
        dv_mps=dv_mps,
    )


__all__ = ["SpectrogramOutput", "compute_spectrogram"]
