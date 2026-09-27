"""Batch IQ acquisition (PlutoSDR or numerical synthesis).

These functions collect a fixed-size IQ vector in one go, instead of yielding
buffers as a generator.  They were the original entry points before the
streaming refactor and are kept here for offline analysis and unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from MicroDopplerDetection.pipeline.emission import _DAC_FULL_SCALE

logger = logging.getLogger(__name__)

_SPEED_OF_LIGHT: float = 299_792_458.0
_ADC_FULL_SCALE: int = 2048
_ADC_SATURATION_RATIO: float = 0.80


@dataclass
class AcquisitionResult:
    """Container for a raw IQ acquisition.

    Attributes
    ----------
    iq : numpy.ndarray
        Complex64 vector of concatenated IQ samples.
    f_s : float
        Sampling rate (Hz).
    f_c : float
        Carrier frequency (Hz).
    duration_s : float
        Total acquisition duration (seconds).
    """

    iq: np.ndarray
    f_s: float
    f_c: float
    duration_s: float


def acquire_pluto(
    uri: str,
    f_c: float,
    f_s: float,
    rx_gain: float,
    tx_gain: float,
    buffer_size: int,
    n_frames: int,
    tx_buffer: np.ndarray,
) -> AcquisitionResult:
    """Acquire IQ data from a PlutoSDR transceiver (batch mode).

    Parameters
    ----------
    uri : str
        PlutoSDR address, e.g. ``"ip:192.168.2.1"`` or ``"usb:"``.
    f_c, f_s : float
        Carrier frequency (Hz) and ADC sampling rate (Hz).
    rx_gain, tx_gain : float
        Receiver gain (dB) and transmitter attenuation (dB, negative).
    buffer_size : int
        Samples per RX buffer.
    n_frames : int
        Number of buffers to collect.
    tx_buffer : numpy.ndarray
        Complex64 baseband TX waveform (cyclic).  **Must already be scaled
        to ±2^14** — see ``MicroDopplerDetection.pipeline.emission``.

    Returns
    -------
    AcquisitionResult
        Concatenated IQ vector with metadata.

    Raises
    ------
    RuntimeError
        If the PlutoSDR cannot be reached at *uri*.
    """
    try:
        import adi
    except ImportError as exc:
        raise RuntimeError(
            "pyadi-iio n'est pas installé. Exécuter : pip install pyadi-iio"
        ) from exc

    try:
        sdr = adi.Pluto(uri)
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de se connecter au PlutoSDR à '{uri}'. "
            f"Vérifier l'adresse IP (ex. ip:192.168.2.1) ou la connexion USB (usb:)."
        ) from exc

    sdr.sample_rate = int(f_s)
    sdr.rx_lo = int(f_c)
    sdr.tx_lo = int(f_c)
    sdr.rx_rf_bandwidth = int(f_s)
    sdr.tx_rf_bandwidth = int(f_s)
    sdr.rx_buffer_size = buffer_size
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = rx_gain
    sdr.tx_hardwaregain_chan0 = tx_gain

    sdr.tx_cyclic_buffer = True
    sdr.tx(tx_buffer)

    logger.info(
        "Acquisition PlutoSDR — %d trames × %d échantillons à %.0f Hz",
        n_frames,
        buffer_size,
        f_s,
    )

    frames: list[np.ndarray] = []
    try:
        for i in range(n_frames):
            frame = sdr.rx()
            _check_saturation(frame, i)
            frames.append(frame)
    finally:
        sdr.tx_destroy_buffer()

    iq = np.concatenate(frames).astype(np.complex64)
    duration_s = len(iq) / f_s
    logger.info("Acquisition terminée — %.2f s, %d échantillons", duration_s, len(iq))
    return AcquisitionResult(iq=iq, f_s=f_s, f_c=f_c, duration_s=duration_s)


def synthesize_iq(
    f_c: float,
    f_s: float,
    buffer_size: int,
    n_frames: int,
    fv: float,
    D_mm: float,
    snr_dB: float,
    f_offset: float = 0.0,
    clutter_amplitude: float = 100.0,
) -> AcquisitionResult:
    """Synthesize a simulated IQ signal containing respiration micro-Doppler.

    Notes
    -----
    Used by offline notebooks and tests.  The streaming pipeline relies on
    :func:`MicroDopplerDetection.pipeline.acquisition.stream_simulation`
    instead.
    """
    wavelength = _SPEED_OF_LIGHT / f_c
    D_m = D_mm * 1e-3

    n_total = buffer_size * n_frames
    t = np.arange(n_total, dtype=np.float64) / f_s

    phase_mod = (4.0 * np.pi * D_m / wavelength) * np.sin(2.0 * np.pi * fv * t)
    carrier = 2.0 * np.pi * f_offset * t if f_offset != 0.0 else np.zeros_like(t)
    signal = np.exp(1j * (carrier - phase_mod))

    clutter = clutter_amplitude * np.ones(n_total, dtype=np.complex128)

    noise_power = 10.0 ** (-snr_dB / 10.0)
    noise_std = np.sqrt(noise_power / 2.0)
    rng = np.random.default_rng()
    noise = noise_std * (
        rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)
    )

    iq = (clutter + signal + noise).astype(np.complex64)

    duration_s = n_total / f_s
    logger.info(
        "Signal simulé — fv=%.2f Hz, D=%.1f mm, SNR=%.0f dB, %.2f s",
        fv,
        D_mm,
        snr_dB,
        duration_s,
    )
    return AcquisitionResult(iq=iq, f_s=f_s, f_c=f_c, duration_s=duration_s)


def _check_saturation(frame: np.ndarray, frame_index: int) -> None:
    """Warn if the IQ frame approaches ADC saturation."""
    peak = np.max(np.abs(frame))
    threshold = _ADC_SATURATION_RATIO * _ADC_FULL_SCALE
    if peak > threshold:
        logger.warning(
            "Saturation ADC probable — trame %d : max(|IQ|) = %.0f "
            "(seuil = %.0f, pleine échelle = %d)",
            frame_index,
            peak,
            threshold,
            _ADC_FULL_SCALE,
        )


__all__ = ["AcquisitionResult", "acquire_pluto", "synthesize_iq", "_DAC_FULL_SCALE"]
