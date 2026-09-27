"""Batch-mode decimation helpers (full IQ vector input).

Streaming pipeline uses :class:`MicroDopplerDetection.pipeline.decimation.Decimator`
instead, which preserves filter state across buffers.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import decimate as _scipy_decimate

from MicroDopplerDetection.pipeline.decimation import _check_shannon, _factorise

logger = logging.getLogger(__name__)


def _decimate_1d(x: np.ndarray, q: int) -> np.ndarray:
    """Decimate a real 1-D array by factor *q* (single stage, zero-phase)."""
    return _scipy_decimate(x, q, ftype="iir", zero_phase=True)


def decimate_iq(
    iq: np.ndarray,
    f_s: float,
    D: int,
    f_max_utile: float,
) -> tuple[np.ndarray, float]:
    """Decimate a complete IQ vector by factor *D* -- offline use only.

    Applies cascaded zero-phase Chebyshev type-I filtering via
    ``scipy.signal.decimate`` (``filtfilt`` internally). Because ``filtfilt``
    requires the full signal, this function is not suitable for streaming; use
    :class:`MicroDopplerDetection.pipeline.decimation.Decimator` instead.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples at rate *f_s*.
    f_s : float
        Input sampling rate (Hz).
    D : int
        Total decimation factor. Must factor into primes <= 13.
    f_max_utile : float
        Maximum frequency of interest (Hz), used for the Shannon check.

    Returns
    -------
    iq_decimated : numpy.ndarray
        Decimated complex IQ vector, ``dtype=complex64``,
        length approximately ``len(iq) // D``.
    f_s_new : float
        Output sampling rate (Hz) = ``f_s / D``.

    Raises
    ------
    ValueError
        If the Nyquist-Shannon criterion (with filter margin) is violated,
        or if *D* contains a prime factor > 13.
    """
    _check_shannon(f_s, D, f_max_utile)

    stages = _factorise(D)
    f_s_new = f_s / D

    logger.debug(
        "decimate_iq -- D=%d, %d etage(s) %s, f_s : %.0f -> %.1f Hz",
        D,
        len(stages),
        stages,
        f_s,
        f_s_new,
    )

    iq_i = iq.real.astype(np.float64)
    iq_q = iq.imag.astype(np.float64)

    f_cur = f_s
    for idx, q in enumerate(stages):
        f_cut = 0.8 * (f_cur / q) / 2.0
        logger.debug(
            "  Etage %d/%d : x%d -- f_s=%.1f Hz -> %.1f Hz, f_coupure~=%.1f Hz",
            idx + 1,
            len(stages),
            q,
            f_cur,
            f_cur / q,
            f_cut,
        )
        iq_i = _decimate_1d(iq_i, q)
        iq_q = _decimate_1d(iq_q, q)
        f_cur /= q

    iq_decimated = (iq_i + 1j * iq_q).astype(np.complex64)
    logger.debug(
        "decimate_iq termine -- %d -> %d echantillons",
        len(iq),
        len(iq_decimated),
    )
    return iq_decimated, f_s_new


__all__ = ["decimate_iq"]
