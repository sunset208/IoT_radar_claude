"""Source PlutoSDR : TX cyclique à f_offset, RX en thread dédié.

Points corrigés par rapport à la v1 :

* **Acquisition dans un thread** qui ne fait qu'appeler ``sdr.rx()`` et
  empiler dans une file ; le traitement ne peut plus provoquer de pertes
  d'échantillons (qui cassent la continuité de phase = faux mouvements).
* **Détection des pertes** : si la file déborde, ou si ``rx()`` rend la main
  anormalement vite plusieurs fois de suite (buffers noyau pleins → retard),
  le bloc suivant est marqué ``discontinuity`` et l'analyse repart proprement.
* **Buffer TX à nombre entier de périodes** de f_offset (sinon raie parasite
  à f_s/N et sauts de phase à chaque bouclage du buffer cyclique).
* **Tracking DC / quadrature désactivé** (boucles adaptatives près du DC).
* Mesure continue du niveau ADC (dBFS) pour régler le gain.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from radar.config import PROJECT_ROOT, Config
from radar.dsp.frontend import Frontend, adc_level_dbfs
from radar.sources.base import Chunk, Source

logger = logging.getLogger(__name__)

DAC_SCALE = 2**14   # pyadi-iio : échantillons TX alignés sur les bits de poids fort


def ensure_libiio_on_path() -> None:
    """Ajoute la libiio locale (``../tools/libiio``) au PATH DLL sous Windows."""
    if os.name != "nt":
        return
    candidates = [
        PROJECT_ROOT.parent / "tools" / "libiio" / "Windows-VS-2022-x64",
        PROJECT_ROOT / "tools" / "libiio" / "Windows-VS-2022-x64",
    ]
    for c in candidates:
        if (c / "libiio.dll").is_file():
            os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(c))
            except (AttributeError, OSError):
                pass
            return


def make_tx_waveform(f_s: float, f_offset: float, min_len: int, amplitude: float) -> np.ndarray:
    """Tonalité complexe à f_offset contenant un nombre entier de périodes."""
    period = f_s / f_offset
    if abs(period - round(period)) > 1e-9:
        raise ValueError("f_s/f_offset doit être entier pour un buffer TX sans discontinuité.")
    period = int(round(period))
    n = period * max(1, int(np.ceil(min_len / period)))
    k = np.arange(n)
    return (amplitude * DAC_SCALE * np.exp(2j * np.pi * k / period)).astype(np.complex64)


def _set_attr(sdr, chan: str, attr: str, value, output: bool = False) -> bool:
    try:
        sdr._set_iio_attr(chan, attr, output, value)
        return True
    except Exception as exc:  # attribut absent selon firmware
        logger.debug("Attribut %s/%s non réglé : %s", chan, attr, exc)
        return False


def connect_pluto(uri: str):
    """Connexion pyadi-iio.  Lève RuntimeError avec un message utile."""
    ensure_libiio_on_path()
    try:
        import adi  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(f"pyadi-iio/libiio indisponible : {exc}") from exc
    try:
        return adi.Pluto(uri)
    except Exception as exc:
        raise RuntimeError(
            f"PlutoSDR injoignable à '{uri}' ({exc}).\n"
            "  • Brancher le micro-USB « USB » (données), pas celui marqué « Power ».\n"
            "  • Sous Windows 11 : interface réseau NCM native (usb_ethernet_mode = ncm dans\n"
            "    config.txt du Pluto) ou drivers « PlutoSDR-M2k-USB-Drivers » d'ADI (RNDIS).\n"
            "  • Tester : radar check  (liste les contextes IIO visibles)."
        ) from exc


def lo_range(sdr) -> tuple[float, float] | None:
    """Plage de LO annoncée par le firmware (« [min pas max] »), ou None."""
    try:
        ch = sdr._ctrl.find_channel("altvoltage0", True)
        v = ch.attrs["frequency_available"].value.strip("[] \n").split()
        return float(v[0]), float(v[-1])
    except Exception:
        return None


def check_lo(sdr, freqs_hz) -> None:
    """Vérifie que les fréquences demandées sont accessibles (AD9363 ou AD9364)."""
    rng = lo_range(sdr)
    if rng is None:
        return
    lo, hi = rng
    bad = [f for f in np.atleast_1d(freqs_hz) if not lo <= f <= hi]
    if bad:
        raise RuntimeError(
            f"Fréquence {bad[0] / 1e6:.0f} MHz hors de la plage du Pluto "
            f"({lo / 1e6:.0f}–{hi / 1e6:.0f} MHz).  Pour aller au-delà de 3.8 GHz : déverrouiller "
            "en AD9364 (README, « Passer à 5.8 GHz ») puis sdr.chip: ad9364.")


def disable_tracking(sdr) -> None:
    """Boucles adaptatives DC / quadrature de l'AD936x coupées (elles agissent près du DC)."""
    for attr in ("quadrature_tracking_en", "rf_dc_offset_tracking_en", "bb_dc_offset_tracking_en"):
        _set_attr(sdr, "voltage0", attr, 0)


def set_kernel_buffers(sdr, n: int) -> None:
    try:
        sdr._rxadc.set_kernel_buffers_count(int(n))
    except Exception as exc:
        logger.debug("set_kernel_buffers_count indisponible : %s", exc)


def set_calib_mode(sdr, mode: str) -> str | None:
    """Règle ``calib_mode`` (auto | manual…) ; renvoie l'ancienne valeur."""
    try:
        a = sdr._ctrl.attrs["calib_mode"]
        old = a.value.strip()
        a.value = mode
        return old
    except Exception as exc:
        logger.warning("calib_mode non réglé (%s) : le driver peut recalibrer à chaque saut de LO.", exc)
        return None


def open_pluto(cfg: Config):
    """Ouvre et configure le Pluto (mode CW).  Lève RuntimeError avec un message utile."""
    s = cfg.sdr
    sdr = connect_pluto(s.uri)
    check_lo(sdr, [s.f_c])
    sdr.sample_rate = int(s.f_s)
    sdr.rx_rf_bandwidth = int(s.rf_bandwidth)
    sdr.tx_rf_bandwidth = int(s.rf_bandwidth)
    sdr.rx_lo = int(s.f_c)
    sdr.tx_lo = int(s.f_c)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(s.rx_gain_db)
    sdr.tx_hardwaregain_chan0 = float(s.tx_atten_db)
    sdr.rx_buffer_size = int(s.rx_buffer_size)
    if s.disable_tracking:
        disable_tracking(sdr)
    set_kernel_buffers(sdr, s.kernel_buffers)
    return sdr


class PlutoSource(Source):
    kind = "pluto"

    def __init__(self, cfg: Config, queue_s: float = 5.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.fs_slow = cfg.frontend.fs_slow
        self.frontend = Frontend(cfg.sdr.f_s, cfg.emission.f_offset, cfg.frontend.fs_slow,
                                 cfg.sdr.rx_buffer_size, cfg.frontend.cic_order)
        buf_s = cfg.sdr.rx_buffer_size / cfg.sdr.f_s
        self._q: queue.Queue = queue.Queue(maxsize=max(4, int(queue_s / buf_s)))
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.dropped_buffers = 0
        self.suspect_overflows = 0
        self.sdr = None

    def describe(self) -> dict:
        s = self.cfg.sdr
        return {"kind": self.kind, "fs_slow": self.fs_slow, "uri": s.uri, "f_c": s.f_c,
                "rx_gain_db": s.rx_gain_db, "tx_atten_db": s.tx_atten_db}

    # ------------------------------------------------------------------
    def _rx_loop(self) -> None:
        cfg = self.cfg
        buf_s = cfg.sdr.rx_buffer_size / cfg.sdr.f_s
        fast_streak = 0
        pending_gap = False
        try:
            while not self._stop:
                t_a = time.perf_counter()
                data = self.sdr.rx()
                dt = time.perf_counter() - t_a
                # rx() normal ≈ durée du buffer.  S'il rend la main presque
                # instantanément plusieurs fois de suite, c'est qu'on lit des
                # buffers noyau en retard : risque de débordement.
                fast_streak = fast_streak + 1 if dt < 0.1 * buf_s else 0
                if fast_streak > int(cfg.sdr.kernel_buffers):
                    self.suspect_overflows += 1
                    pending_gap = True
                    fast_streak = 0
                item = (np.asarray(data), pending_gap)
                try:
                    self._q.put_nowait(item)
                    pending_gap = False
                except queue.Full:
                    self.dropped_buffers += 1
                    pending_gap = True
        except BaseException as exc:  # remonté au consommateur
            self._error = exc
        finally:
            self._q.put(None)

    def _iter(self) -> Iterator[Chunk]:
        cfg = self.cfg
        self.sdr = open_pluto(cfg)
        tx = make_tx_waveform(cfg.sdr.f_s, cfg.emission.f_offset, cfg.sdr.rx_buffer_size,
                              cfg.emission.amplitude)
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(tx)
        logger.info("Pluto : TX cyclique %d éch. (%.0f Hz), RX %d éch./buffer",
                    len(tx), cfg.emission.f_offset, cfg.sdr.rx_buffer_size)
        # Purge des premiers buffers (transitoires PLL / AGC / TX)
        for _ in range(3):
            self.sdr.rx()
        self._thread = threading.Thread(target=self._rx_loop, name="pluto-rx", daemon=True)
        self._thread.start()
        slow_count = 0
        try:
            while not self._stop:
                item = self._q.get()
                if item is None:
                    if self._error is not None:
                        raise RuntimeError(f"Erreur acquisition Pluto : {self._error}") from self._error
                    break
                raw, gap = item
                if gap:
                    # la phase n'est plus continue : on repart d'un front-end neuf
                    self.frontend = Frontend(cfg.sdr.f_s, cfg.emission.f_offset, cfg.frontend.fs_slow,
                                             cfg.sdr.rx_buffer_size, cfg.frontend.cic_order)
                    logger.warning("Perte d'échantillons détectée — réinitialisation de l'analyse.")
                slow = self.frontend(raw)
                rms, peak = adc_level_dbfs(raw)
                t0 = slow_count / self.fs_slow
                slow_count += len(slow)
                yield Chunk(
                    slow=slow, t0=t0,
                    raw=raw.astype(np.complex64) if cfg.recording.save_raw else None,
                    discontinuity=gap,
                    stats={"adc_rms_dbfs": rms, "adc_peak_dbfs": peak,
                           "queue": self._q.qsize(), "dropped": self.dropped_buffers,
                           "suspect_overflows": self.suspect_overflows},
                )
        finally:
            self._stop = True
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            try:
                self.sdr.tx_destroy_buffer()
            except Exception:
                pass
            try:
                self.sdr.rx_destroy_buffer()
            except Exception:
                pass
            logger.info("Pluto arrêté (buffers perdus : %d, débordements suspects : %d)",
                        self.dropped_buffers, self.suspect_overflows)


def scan_contexts() -> dict:
    ensure_libiio_on_path()
    import iio
    return iio.scan_contexts()


def libiio_dir() -> Path | None:
    for c in (PROJECT_ROOT.parent / "tools" / "libiio" / "Windows-VS-2022-x64",):
        if c.is_dir():
            return c
    return None
