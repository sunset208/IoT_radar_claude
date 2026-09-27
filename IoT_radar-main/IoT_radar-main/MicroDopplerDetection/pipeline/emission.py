"""TX waveform generation for the micro-Doppler radar pipeline."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


_DAC_FULL_SCALE: int = 2**14
"""PlutoSDR DAC convention used by ``pyadi-iio``.

The ``adi.Pluto.tx()`` API casts ``complex64`` samples directly to
``int16`` without applying any scaling.  The AD9363 DAC is 12-bit but
``pyadi-iio`` aligns its samples to the upper bits of the ``int16`` word,
so unit-amplitude IQ has to be multiplied by ``2**14`` to reach DAC
full-scale.  Without this scaling the carrier sits at ~1 LSB
(≈ −84 dBFS) and is invisible on a spectrum analyser.
"""


def generate_tx_buffer(
    mode: str,
    buffer_size: int,
    f_s: float,
    f_offset: float = 0.0,
) -> np.ndarray:
    """Generate a complex baseband TX waveform of length *buffer_size*.

    Parameters
    ----------
    mode : str
        Emission mode. ``"cw"`` produces a constant-envelope carrier;
        ``"cw_offset"`` produces a complex sinusoid at *f_offset* Hz,
        which shifts the useful signal away from the DC bin.
    buffer_size : int
        Number of IQ samples in the transmit buffer.
    f_s : float
        ADC / DAC sampling rate (Hz).
    f_offset : float, optional
        Frequency offset in baseband (Hz).  Only used when
        *mode* = ``"cw_offset"``.  Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex64 array of shape ``(buffer_size,)`` with samples scaled
        to the PlutoSDR DAC full-scale (``±2**14``).

    Raises
    ------
    ValueError
        If *mode* is not one of ``{"cw", "cw_offset"}``, or if *f_offset*
        violates the Nyquist criterion (``|f_offset| >= f_s / 2``).

    Notes
    -----
    * **CW mode** — baseband samples are a constant ``2**14 + 0j``;
      the RF output is a pure tone at exactly ``f_c``.
    * **CW-offset mode** — baseband samples are
      ``2**14 · exp(j·2π·f_offset·t)``, producing an RF tone at
      ``f_c + f_offset`` and avoiding the DC clutter.

    The ``2**14`` scaling is **mandatory**: without it the DAC effectively
    transmits a zero-amplitude signal (see :data:`_DAC_FULL_SCALE`).
    """
    if mode not in ("cw", "cw_offset"):
        raise ValueError(
            f"Unknown emission mode '{mode}'. Expected 'cw' or 'cw_offset'."
        )

    if mode == "cw":
        logger.info(
            "Génération du buffer TX — mode CW (module constant, scale=%d)",
            _DAC_FULL_SCALE,
        )
        return np.full(buffer_size, _DAC_FULL_SCALE + 0j, dtype=np.complex64)

    if abs(f_offset) >= f_s / 2:
        raise ValueError(
            f"f_offset={f_offset} Hz dépasse la fréquence de Nyquist "
            f"(f_s/2 = {f_s / 2} Hz). Réduire f_offset ou augmenter f_s."
        )

    logger.info(
        "Génération du buffer TX — mode CW-offset (f_offset=%.0f Hz, scale=%d)",
        f_offset,
        _DAC_FULL_SCALE,
    )
    t = np.arange(buffer_size, dtype=np.float64) / f_s
    waveform = _DAC_FULL_SCALE * np.exp(1j * 2 * np.pi * f_offset * t)
    return waveform.astype(np.complex64)
