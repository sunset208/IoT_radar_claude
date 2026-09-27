"""Streaming respiration detection — Fisher F-test fused with phase ACF.

Batch (full-spectrogram) helpers are preserved in
:mod:`MicroDopplerDetection.legacy.detection_batch`.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import correlate
from scipy.stats import f

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Internal building blocks (also re-used by the legacy batch wrappers)
# ----------------------------------------------------------------------

def _fisher_pvalue(
    S_lin: np.ndarray,
    f_hz: np.ndarray,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    f_center: float = 0.0,
) -> tuple[float, float]:
    """Fisher F-test on the band-power ratio.

    Parameters
    ----------
    S_lin : numpy.ndarray
        Linear-scale power spectrum (1-D column or 2-D spectrogram in batch mode).
    f_hz : numpy.ndarray
        Centred frequency axis (Hz).
    bande_respiration, bande_reference : tuple[float, float]
        Signal and reference bands (Hz), expressed **relative to the carrier**
        (i.e. as baseband offsets), as in ``config.yaml``.
    f_center : float, optional
        Spectral location of the carrier (Hz).  Both bands are taken on the
        two sidebands around *f_center*, i.e. on ``|f_hz - f_center|``.  In
        CW-offset mode this is ``f_offset``; in baseband mode it is 0
        (default), which reduces to ``|f_hz|``.

    Returns
    -------
    tuple[float, float]
        ``(p_value, ratio)`` — F-test p-value and band-power ratio.

    Notes
    -----
    Under H₀ of complex white Gaussian noise, the FFT-bin powers are
    proportional to χ²(2) variables, so the ratio of band means follows
    F(2*n_sig, 2*n_ref).
    """
    f_lo_sig, f_hi_sig = bande_respiration
    f_lo_ref, f_hi_ref = bande_reference

    df = np.abs(f_hz - f_center)
    mask_sig = (df >= f_lo_sig) & (df <= f_hi_sig)
    mask_ref = (df >= f_lo_ref) & (df <= f_hi_ref)

    n_sig = int(np.sum(mask_sig))
    n_ref = int(np.sum(mask_ref))

    if n_sig == 0:
        raise ValueError(
            f"Masque bande respiration vide ({f_lo_sig}–{f_hi_sig} Hz)."
        )
    if n_ref == 0:
        raise ValueError(
            f"Masque bande référence vide ({f_lo_ref}–{f_hi_ref} Hz)."
        )

    p_sig = float(np.mean(S_lin[mask_sig]))
    p_ref = float(np.mean(S_lin[mask_ref]))

    eps = 1e-30
    ratio = p_sig / (p_ref + eps)
    p_value = float(f.sf(ratio, dfn=2*n_sig, dfd=2*n_ref))
    return p_value, ratio


def _acf_peak(
    phi: np.ndarray,
    f_s: float,
    bande_respiration: tuple[float, float],
) -> tuple[float, float | None]:
    """Normalised autocorrelation peak inside the respiration delay range.

    Parameters
    ----------
    phi : numpy.ndarray
        Unwrapped instantaneous phase (1-D, radians) of a clutter- and
        carrier-demodulated IQ segment.  **Length should be at least
        ``f_s / f_lo``** to capture one full breathing period.
    f_s : float
        Sampling rate of *phi* (Hz).
    bande_respiration : tuple[float, float]
        ``(f_lo, f_hi)`` — searched delay range is ``[1/f_hi, 1/f_lo]``.

    Returns
    -------
    tuple[float, float or None]
        ``(acf_peak, fv_estimated)`` — normalised ACF value at the best
        lag and corresponding ``1/τ_max`` estimate.
    """
    f_lo, f_hi = bande_respiration
    if f_lo <= 0 or f_hi <= 0 or f_hi <= f_lo:
        return 0.0, None

    phi = np.asarray(phi, dtype=np.float64)
    n = phi.size
    if n < 4:
        return 0.0, None

    phi_zm = phi - np.mean(phi)
    R = correlate(phi_zm, phi_zm, mode="full")
    R0 = R[n - 1]
    if R0 <= 0:
        return 0.0, None

    R_pos = R[n - 1:] / R0

    lag_min = int(np.ceil(f_s / f_hi))
    lag_max = int(np.floor(f_s / f_lo))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, n - 1)

    if lag_max <= lag_min:
        return 0.0, None

    lags = np.arange(lag_min, lag_max + 1)
    segment = R_pos[lag_min: lag_max + 1] * (n / (n - lags))
    idx_local = int(np.argmax(segment))
    tau_max = lag_min + idx_local
    acf_peak = float(segment[idx_local])
    fv_estimated = float(f_s / tau_max) if tau_max > 0 else None
    return acf_peak, fv_estimated


def _fusion_score(
    p_value_f: float,
    acf_peak: float,
    w: float,
    p_value_decades: float,
    acf_floor: float,
    acf_good: float,
) -> float:
    """Fuse the Fisher and ACF scores into a presence score in [0, 1].

    Each raw indicator is mapped to ``[0, 1]`` through an explicit transform
    before the weighted sum, rather than being clipped as-is:

    * spectral — the F-test p-value is mapped on a log scale, so that a
      p-value of ``10**(-p_value_decades)`` (or smaller) saturates to 1 and a
      p-value of 1 maps to 0.  This reflects that confidence grows by orders
      of magnitude, not linearly with ``1 - p``.
    * temporal — the ACF peak is ramped linearly between a noise floor
      (``acf_floor`` → 0) and a "strong periodicity" level (``acf_good`` → 1),
      which is more meaningful than treating the raw peak as a score.

    Parameters
    ----------
    p_value_f : float
        P-value of the Fisher F-test.
    acf_peak : float
        Normalised autocorrelation peak (may be negative).
    w : float
        Weight of the spectral score.  ``w=1`` → purely spectral,
        ``w=0`` → purely time-domain.
    p_value_decades : float
        Number of decades below 1 at which the spectral score saturates to 1
        (e.g. ``3.0`` → ``p ≤ 1e-3`` gives score 1).
    acf_floor, acf_good : float
        ACF peak values mapped to 0 and 1 respectively.
    """
    if p_value_decades <= 0:
        raise ValueError("p_value_decades doit être > 0.")
    if acf_good <= acf_floor:
        raise ValueError("acf_good doit être > acf_floor.")
    if w < 0 or w > 1:
        raise ValueError("w doit être entre 0 et 1.")

    eps = 1e-30
    score_F = float(
        np.clip(-np.log10(p_value_f + eps) / p_value_decades, 0.0, 1.0)
    )
    score_acf = float(
        np.clip((acf_peak - acf_floor) / (acf_good - acf_floor), 0.0, 1.0)
    )
    return float(w * score_F + (1.0 - w) * score_acf)


# ----------------------------------------------------------------------
# Streaming entry point
# ----------------------------------------------------------------------

def detect_presence_column(
    col_db: np.ndarray,
    f_hz: np.ndarray,
    phi_buffer: np.ndarray,
    f_s: float,
    bande_respiration: tuple[float, float],
    bande_reference: tuple[float, float],
    f_center: float,
    w: float,
    p_value_decades: float,
    acf_floor: float,
    acf_good: float,
) -> tuple[float, float, float, float | None]:
    """Streaming detection on a single STFT column + matching phase segment.

    Parameters
    ----------
    col_db : numpy.ndarray
        Power spectrum (dB), 1-D — current STFT column (length n_fft).
    f_hz : numpy.ndarray
        Centred frequency axis (Hz), shape matches *col_db*.
    phi_buffer : numpy.ndarray
        Unwrapped phase of the **same time window** as *col_db*, after
        clutter suppression and (in cw_offset mode) carrier demodulation.
        Length should be ≥ ``f_s / bande_respiration[0]``.
    f_s : float
        Sampling rate of *phi_buffer* (Hz), i.e. the decimated rate.
    bande_respiration, bande_reference : tuple[float, float]
        Signal and reference bands **relative to the carrier** (Hz), e.g.
        ``(0.1, 0.8)`` and ``(2.0, 5.0)``.  Used both for the spectral
        F-test (on the two sidebands around *f_center*) and, for the
        respiration band, for the time-domain ACF on the demodulated phase.
    f_center : float
        Spectral location of the carrier (Hz): ``f_offset`` in CW-offset
        mode, 0 in baseband mode.  Forwarded to :func:`_fisher_pvalue`.
    w : float
        Fusion weight (see :func:`_fusion_score`).
    p_value_decades, acf_floor, acf_good : float
        Score-mapping parameters forwarded to :func:`_fusion_score`.

    Returns
    -------
    tuple[float, float, float, float or None]
        ``(score_presence, p_value_f, acf_peak, fv_estimated)``.
    """
    col_lin = 10.0 ** (col_db / 10.0)
    p_value_f, _ = _fisher_pvalue(
        col_lin, f_hz, bande_respiration, bande_reference, f_center=f_center,
    )

    acf_peak, fv_estimated = _acf_peak(
        phi_buffer, f_s, bande_respiration,
    )
    score_presence = _fusion_score(
        p_value_f,
        acf_peak,
        w=w,
        p_value_decades=p_value_decades,
        acf_floor=acf_floor,
        acf_good=acf_good,
    )

    logger.debug(
        "Détection — p_F=%.2e, ACF=%.2f @ fv=%s Hz, score=%.2f",
        p_value_f,
        acf_peak,
        f"{fv_estimated:.3f}" if fv_estimated is not None else "n/a",
        score_presence,
    )
    return score_presence, p_value_f, acf_peak, fv_estimated
