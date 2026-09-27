"""Streaming IQ acquisition from PlutoSDR hardware or numerical simulation.

The streaming pipeline consumes one buffer at a time via the generators
defined here.  The legacy batch helpers (full-vector acquisition) live in
:mod:`MicroDopplerDetection.legacy.acquisition_batch`.
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

_ADC_FULL_SCALE: int = 2048
_ADC_SATURATION_RATIO: float = 0.80
_SPEED_OF_LIGHT: float = 299_792_458.0


def stream_pluto(
    uri: str,
    f_c: float,
    f_s: float,
    rx_gain: float,
    tx_gain: float,
    buffer_size: int,
    tx_buffer: np.ndarray,
) -> Generator[np.ndarray, None, None]:
    """Yield IQ buffers from the PlutoSDR indefinitely.

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
    tx_buffer : numpy.ndarray
        Complex64 baseband TX waveform (cyclic).  **Already scaled to
        ±2**14** — see ``MicroDopplerDetection.pipeline.emission``.

    Yields
    ------
    numpy.ndarray
        Complex64 vector of shape ``(buffer_size,)``.

    Raises
    ------
    RuntimeError
        If the PlutoSDR cannot be reached at *uri*.

    Notes
    -----
    * The TX path runs in **cyclic** mode: the SDR re-plays *tx_buffer*
      indefinitely while we keep receiving.
    * On generator close (``gen.close()`` or context exit) the TX buffer
      is destroyed cleanly so the Pluto is left in a known state.
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
        "Streaming PlutoSDR — buffer_size=%d à %.0f Hz (continu)",
        buffer_size,
        f_s,
    )

    frame_idx = 0
    try:
        while True:
            frame = sdr.rx()
            _check_saturation(frame, frame_idx)
            yield np.asarray(frame, dtype=np.complex64)
            frame_idx += 1
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        logger.info("Streaming PlutoSDR arrêté après %d trames", frame_idx)


def stream_simulation(
    f_c: float,
    f_s: float,
    buffer_size: int,
    fv: float,
    D_mm: float,
    snr_dB: float,
    f_offset: float = 0.0,
    clutter_amplitude: float = 100.0,
) -> Generator[np.ndarray, None, None]:
    """Yield simulated IQ buffers indefinitely with continuous phase.

    Parameters
    ----------
    f_c, f_s : float
        Carrier frequency (Hz) and sampling rate (Hz).
    buffer_size : int
        Samples per buffer.
    fv : float
        Simulated breathing frequency (Hz).
    D_mm : float
        Chest displacement amplitude (mm).
    snr_dB : float
        Target micro-Doppler SNR (dB).
    f_offset : float, optional
        Baseband frequency offset (Hz).  Default is 0.
    clutter_amplitude : float, optional
        Amplitude of the static-clutter component relative to the signal.
        Default 100.0 (40 dB above signal — typical CW radar isolation).

    Yields
    ------
    numpy.ndarray
        Complex64 vector of shape ``(buffer_size,)``.

    Notes
    -----
    Each call produces the *next* ``buffer_size`` samples of the same
    continuous waveform, maintaining phase continuity across buffers.
    This mimics the real PlutoSDR streaming behaviour.
    """
    wavelength = _SPEED_OF_LIGHT / f_c
    D_m = D_mm * 1e-3
    mod_index = 4.0 * np.pi * D_m / wavelength

    noise_power = 10.0 ** (-snr_dB / 10.0)
    noise_std = np.sqrt(noise_power / 2.0)
    rng = np.random.default_rng()

    logger.info(
        "Streaming simulation — fv=%.2f Hz, D=%.1f mm, SNR=%.0f dB (continu)",
        fv,
        D_mm,
        snr_dB,
    )

    sample_idx = 0
    while True:
        t = (np.arange(buffer_size, dtype=np.float64) + sample_idx) / f_s

        phase_mod = mod_index * np.sin(2.0 * np.pi * fv * t)
        carrier = (
            2.0 * np.pi * f_offset * t if f_offset != 0.0 else np.zeros_like(t)
        )
        signal = np.exp(1j * (carrier - phase_mod))

        clutter = clutter_amplitude * np.ones(buffer_size, dtype=np.complex128)
        noise = noise_std * (
            rng.standard_normal(buffer_size)
            + 1j * rng.standard_normal(buffer_size)
        )

        buf = (clutter + signal + noise).astype(np.complex64)
        sample_idx += buffer_size
        yield buf


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
