"""Streaming sample-rate reduction for IQ signals.

:class:`Decimator` is stateful and causal for real-time streaming pipelines.
Instantiate once, then call buffer-by-buffer.  Filter state and sub-sampling
phase are preserved across calls so that buffer boundaries do not reset the
anti-aliasing filters or the downsampling phase.

The total decimation factor *D* is decomposed into prime factors ≤ 13
(via :func:`_factorise`) and one Chebyshev type-I order-8 anti-aliasing filter
is applied per stage, following the ``scipy.signal.decimate`` recommendation.

Note
----
``filtfilt`` (zero-phase) is incompatible with streaming because it requires
the complete signal.  :class:`Decimator` uses ``lfilter`` (forward-only) which
introduces a constant group delay — harmless for detection of signals well
below the cutoff (~0.4 · f_out).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import cheby1, lfilter

logger = logging.getLogger(__name__)

_MAX_PRIME = 13
_PRIMES = (2, 3, 5, 7, 11, 13)
_CHEBY_ORDER = 8
_CHEBY_RIPPLE_DB = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _factorise(D: int) -> list[int]:
    """Decompose *D* into prime factors ≤ 13 (sorted ascending).

    Each factor becomes one decimation stage with its own Chebyshev
    anti-aliasing filter, maximising stopband rejection per stage.

    Raises
    ------
    ValueError
        If *D* has a prime factor > 13.
    """
    if D <= 1:
        return []

    factors: list[int] = []
    remaining = D

    for p in _PRIMES:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p

    if remaining > 1:
        raise ValueError(
            f"D={D} contient un facteur premier > {_MAX_PRIME} (résidu={remaining}). "
            f"Choisir un D décomposable en petits facteurs ({', '.join(map(str, _PRIMES))})."
        )

    factors.sort()
    return factors


def _check_shannon(f_s: float, D: int, f_max_utile: float) -> None:
    """Raise ``ValueError`` if the post-decimation Nyquist margin is violated.

    The guard band factor 2.5 (instead of the theoretical 2.0) accounts for
    the transition band of the Chebyshev anti-aliasing filter.
    """
    f_s_out = f_s / D
    if D > 1 and f_s_out <= 2.5 * f_max_utile:
        raise ValueError(
            f"Critère de Shannon (avec marge filtre anti-repliement) violé : "
            f"f_s_out = {f_s_out:.1f} Hz ≤ 2.5 × f_max_utile = "
            f"{2.5 * f_max_utile:.1f} Hz. Réduire D (actuellement {D}) "
            f"ou augmenter f_s."
        )


class Decimator:
    """Stateful cascaded decimator for streaming complex IQ.

    Instantiate once per stream, then call the instance buffer-by-buffer.
    Filter state (``zi``) and sub-sampling phase are preserved across calls
    so that buffer boundaries do not reset the anti-aliasing filters or the
    downsampling phase.

    For offline / batch use, see
    :func:`MicroDopplerDetection.legacy.decimation_batch.decimate_iq`.

    Parameters
    ----------
    f_s : float
        Input sampling rate (Hz).
    D : int
        Total decimation factor.  Must factor into primes ≤ 13.
    f_max_utile : float
        Maximum frequency of interest (Hz), used for the Shannon check.

    Notes
    -----
    Each stage applies a Chebyshev type-I order-8 low-pass filter with
    ``Wn = 0.8 / q``, then downsamples by *q*.  Forward IIR (``lfilter``)
    is used instead of ``filtfilt`` because zero-phase filtering is
    incompatible with streaming.  The induced group delay is constant and
    negligible for signals well below the cutoff (~0.4 · f_out).
    """

    def __init__(self, f_s: float, D: int, f_max_utile: float) -> None:
        self.f_s_in = float(f_s)
        self.D = int(D)
        self.f_s_out = self.f_s_in / self.D

        _check_shannon(self.f_s_in, self.D, f_max_utile)

        self._stages: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._zi_re: list[np.ndarray] = []
        self._zi_im: list[np.ndarray] = []
        self._phase: list[int] = []

        for q in _factorise(self.D):
            b, a = cheby1(_CHEBY_ORDER, _CHEBY_RIPPLE_DB, 0.8 / q)
            n_state = max(len(a), len(b)) - 1
            self._stages.append((b, a, q))
            self._zi_re.append(np.zeros(n_state, dtype=np.float64))
            self._zi_im.append(np.zeros(n_state, dtype=np.float64))
            self._phase.append(0)

        logger.info(
            "Decimator — D=%d, %d étage(s), f_s : %.0f → %.1f Hz",
            self.D, len(self._stages), self.f_s_in, self.f_s_out,
        )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        """Decimate one IQ buffer, preserving filter state.

        Parameters
        ----------
        iq : numpy.ndarray
            Complex IQ samples (1-D) at rate ``f_s_in``.

        Returns
        -------
        numpy.ndarray
            ``complex64`` vector at rate ``f_s_out``.  Length depends on
            buffer size and current sub-sampling phase; for a steady stream
            of equal-sized buffers the average length per call is
            ``len(iq) / D``.
        """
        if self.D == 1:
            return iq.astype(np.complex64, copy=False)

        re = iq.real.astype(np.float64, copy=False)
        im = iq.imag.astype(np.float64, copy=False)

        for i, (b, a, q) in enumerate(self._stages):
            if re.size == 0:
                continue

            re, self._zi_re[i] = lfilter(b, a, re, zi=self._zi_re[i])
            im, self._zi_im[i] = lfilter(b, a, im, zi=self._zi_im[i])

            offset = self._phase[i]
            n = re.size

            if offset >= n:
                self._phase[i] = offset - n
                re = re[:0]
                im = im[:0]
                continue

            keep = np.arange(offset, n, q)
            self._phase[i] = int((keep[-1] + q) - n)
            re = re[keep]
            im = im[keep]

        return (re + 1j * im).astype(np.complex64)

    def reset(self) -> None:
        """Reset filter state and phase counters (start a new stream)."""
        for i in range(len(self._stages)):
            self._zi_re[i][:] = 0.0
            self._zi_im[i][:] = 0.0
            self._phase[i] = 0