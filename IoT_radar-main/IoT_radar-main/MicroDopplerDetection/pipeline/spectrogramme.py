"""Single-column STFT for the streaming micro-Doppler pipeline.

The full 2-D spectrogram batch helper is preserved in
:mod:`MicroDopplerDetection.legacy.spectrogramme_batch`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0


@dataclass
class ColumnOutput:
    """Container for a single STFT column (one spectral snapshot).

    Attributes
    ----------
    col_complex : numpy.ndarray
        Complex spectrum (complex128), shape ``(n_fft,)``, fftshifted.
    col_db : numpy.ndarray
        Power in dB, shape ``(n_fft,)``.
    f_hz : numpy.ndarray
        Doppler frequency axis (Hz), centred.
    v_mps : numpy.ndarray
        Radial velocity axis (m/s).
    df_hz : float
        Frequency resolution (Hz).
    """

    col_complex: np.ndarray
    col_db: np.ndarray
    f_hz: np.ndarray
    v_mps: np.ndarray
    df_hz: float


def compute_single_column(
    segment: np.ndarray,
    f_s: float,
    f_c: float,
    window: np.ndarray,
) -> ColumnOutput:
    """Compute one STFT column from a windowed time-domain segment.

    Parameters
    ----------
    segment : numpy.ndarray
        Complex IQ segment, shape ``(n_fft,)``.
    f_s : float
        Sampling rate (Hz) of *segment* (decimated rate).
    f_c : float
        Carrier frequency (Hz), for Doppler-to-velocity conversion.
    window : numpy.ndarray
        Pre-computed window, same length as *segment*.

    Returns
    -------
    ColumnOutput
        Single spectrum column with frequency / velocity axes.
    """
    n_fft = len(segment)
    wavelength = _SPEED_OF_LIGHT / f_c

    windowed = segment * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))

    eps = 1e-12
    col_db = 10.0 * np.log10(np.abs(spectrum) ** 2 + eps)

    f_hz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / f_s))
    v_mps = f_hz * wavelength / 2.0
    df_hz = f_s / n_fft

    return ColumnOutput(
        col_complex=spectrum,
        col_db=col_db.astype(np.float64),
        f_hz=f_hz.astype(np.float64),
        v_mps=v_mps.astype(np.float64),
        df_hz=df_hz,
    )
