"""Spectral window functions for STFT analysis."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import windows

logger = logging.getLogger(__name__)

_VALID_WINDOWS = ("hann", "hamming", "blackman", "flattop")


def get_window(mode: str, n: int) -> np.ndarray:
    """Return a real-valued window of length *n*.

    Parameters
    ----------
    mode : str
        Window type: ``"hann"``, ``"hamming"``, ``"blackman"``,
        ``"flattop"``, or ``"none"`` (rectangular — all ones).
    n : int
        Number of samples in the window.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(n,)`` with values in [0, 1].

    Raises
    ------
    ValueError
        If *mode* is not a recognised window name.

    Notes
    -----
    In STFT-based micro-Doppler analysis each time segment is multiplied
    by a tapering window before the FFT.  This reduces **spectral leakage**
    — energy from strong clutter residuals or harmonics spilling into the
    weak respiratory bins.  The trade-off is a wider main lobe (lower
    frequency resolution).

    * Hann — good general-purpose choice; −31 dB first sidelobe.
    * Hamming — slightly lower sidelobes (−43 dB) at the cost of a
      discontinuity at the edges.
    * Blackman — very low sidelobes (−58 dB), ~50 % wider main lobe.
    * Flat-top — best amplitude accuracy (< 0.01 dB error), useful for
      measuring Bessel-series harmonic magnitudes, but widest main lobe
      (~3.8× Hann) — not recommended for detection where frequency
      resolution is critical.
    * None (rectangular) — maximum resolution, maximum leakage; useful
      only when the signal is well-isolated in frequency.
    """
    if mode == "none":
        logger.debug("Fenêtre rectangulaire (none) — %d points", n)
        return np.ones(n, dtype=np.float64)

    if mode not in _VALID_WINDOWS:
        raise ValueError(
            f"Fenêtre inconnue : '{mode}'. "
            f"Utiliser {', '.join(repr(m) for m in _VALID_WINDOWS)} ou 'none'."
        )

    w = getattr(windows, mode)(n)
    logger.debug("Fenêtre '%s' — %d points", mode, n)
    return np.asarray(w, dtype=np.float64)
