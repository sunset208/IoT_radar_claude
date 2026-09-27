"""Static-clutter suppression filters for micro-Doppler radar."""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

logger = logging.getLogger(__name__)

_VALID_MODES = ("mean", "iir", "mti", "butterworth")


class ClutterFilter:
    """Stateful clutter-suppression filter for streaming pipelines.

    Parameters
    ----------
    mode : str
        Clutter-removal strategy: ``"mean"``, ``"iir"``, ``"mti"``,
        or ``"butterworth"``.
    fs : float
        Sampling rate of the input signal (Hz).  Used for cutoff
        frequency computation and logging.
    alpha : float, optional
        EMA memory factor for ``"iir"`` and ``"mean"`` modes.
        Default is 0.9999.
    butterworth_order : int, optional
        Filter order for ``"butterworth"`` mode.  Default is 2.
    butterworth_cutoff : float, optional
        High-pass cutoff frequency (Hz) for ``"butterworth"`` mode.
        Default is 0.05.

    Notes
    -----
    This class retains the internal state (running mean for IIR,
    previous sample for MTI) across successive calls to 
    :meth:`__call__`.  This is essential in a streaming pipeline
    where each buffer must be processed incrementally while
    maintaining filter continuity.
    """

    def __init__(
        self,
        mode: str,
        fs: float,
        alpha: float = 0.9999,
        butterworth_order: int = 2,
        butterworth_cutoff: float = 0.05,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Mode clutter inconnu : '{mode}'. "
                f"Utiliser {', '.join(repr(m) for m in _VALID_MODES)}."
            )
        self._mode = mode
        self._fs = fs
        self._alpha = alpha
        self._mu: complex | None = None
        self._prev: complex | None = None

        if mode in ("iir", "mean"):
            f_cut = (1.0 - alpha) * fs / (2.0 * math.pi)
            logger.info(
                "ClutterFilter initialisé — mode='%s', alpha=%.4f, "
                "f_coupure≈%.3f Hz (fs=%.1f Hz)",
                mode,
                alpha,
                f_cut,
                fs,
            )

        if mode == "mti":
            logger.warning(
                "Mode MTI inadapté à la détection de respiration : "
                "le filtre y[n]=x[n]-x[n-1] atténue de ~60 dB à 0.3 Hz "
                "pour fs_dec=%.0f Hz — il DÉTRUIT le signal respiratoire. "
                "Préférer 'iir' ou 'butterworth'.",
                fs,
            )

        if mode == "butterworth":
            self._sos = butter(
                butterworth_order,
                butterworth_cutoff,
                btype="high",
                fs=fs,
                output="sos",
            )
            self._zi_real = None
            self._zi_imag = None
            logger.info(
                "ClutterFilter initialisé — mode='butterworth', "
                "ordre=%d, f_coupure=%.3f Hz (fs=%.1f Hz)",
                butterworth_order,
                butterworth_cutoff,
                fs,
            )

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        """Filter one buffer of IQ samples.

        Parameters
        ----------
        iq : numpy.ndarray
            Complex IQ samples (1-D).

        Returns
        -------
        numpy.ndarray
            Clutter-suppressed IQ (same length as *iq*).
        """
        if iq.ndim != 1:
            raise ValueError(
                f"iq doit être 1-D, reçu ndim={iq.ndim} shape={iq.shape}."
            )
        if not np.iscomplexobj(iq):
            logger.warning(
                "iq n'est pas complexe (dtype=%s) — cast vers complex128.",
                iq.dtype,
            )
            iq = iq.astype(np.complex128)

        if self._mode == "mean":
            return self._apply_mean(iq)
        if self._mode == "iir":
            return self._apply_iir(iq)
        if self._mode == "butterworth":
            return self._apply_butterworth(iq)
        return self._apply_mti(iq)

    def _apply_mean(self, iq: np.ndarray) -> np.ndarray:
        """Stateful mean subtraction using EMA tracking."""
        out = np.empty_like(iq)
        mu = self._mu if self._mu is not None else complex(np.mean(iq))
        alpha = self._alpha
        for n in range(len(iq)):
            mu = alpha * mu + (1.0 - alpha) * iq[n]
            out[n] = iq[n] - mu
        self._mu = mu
        return out

    def _apply_iir(self, iq: np.ndarray) -> np.ndarray:
        """Vectorised EMA high-pass using lfilter with state carry-over."""
        alpha = self._alpha
        b = np.array([1.0 - alpha])
        a = np.array([1.0, -alpha])

        if not hasattr(self, "_zi_iir"):
            mu0 = complex(np.mean(iq))
            self._zi_iir = np.array([mu0 * alpha])

        mu_filtered, self._zi_iir = lfilter(b, a, iq, zi=self._zi_iir)
        self._mu = complex(mu_filtered[-1])
        return (iq - mu_filtered).astype(iq.dtype)

    def _apply_mti(self, iq: np.ndarray) -> np.ndarray:
        """Single-delay MTI with state carry-over."""
        prev = self._prev if self._prev is not None else complex(iq[0])
        extended = np.concatenate(([prev], iq))
        self._prev = complex(iq[-1])
        return np.diff(extended)

    def _apply_butterworth(self, iq: np.ndarray) -> np.ndarray:
        """High-pass Butterworth with state carry-over (I/Q separate)."""
        if self._zi_real is None:
            zi = sosfilt_zi(self._sos)
            self._zi_real = zi * iq.real[0]
            self._zi_imag = zi * iq.imag[0]

        y_real, self._zi_real = sosfilt(
            self._sos, iq.real.astype(np.float64), zi=self._zi_real,
        )
        y_imag, self._zi_imag = sosfilt(
            self._sos, iq.imag.astype(np.float64), zi=self._zi_imag,
        )
        return (y_real + 1j * y_imag).astype(iq.dtype)
