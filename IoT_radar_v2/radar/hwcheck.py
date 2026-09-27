"""Diagnostic matériel : ``radar check``.

1. libiio chargée ? contextes IIO visibles (USB / réseau) ?
2. Connexion au Pluto, firmware, configuration.
3. Capture courte : niveaux ADC (saturation), raie à f_offset (fuite TX→RX),
   offset DC, image, plancher de bruit → recommandation de gain.
4. Stabilité (``--stability N``) : sur N secondes de slow-time, bruit de
   phase / dérive du clutter statique et **bruit équivalent en déplacement**
   dans la bande respiratoire.  C'est LE chiffre qui borne la sensibilité :
   à faire une fois antennes débranchées (fuite seule) puis branchées.
5. Débit réel vs nominal (pertes d'échantillons).

``--sim`` exécute les mêmes mesures sur le simulateur (démonstration).
"""

from __future__ import annotations

import logging
import math
import os
import socket
import time

import numpy as np
from scipy import signal

from radar.config import Config
from radar.dsp.frontend import ADC_FULL_SCALE, Frontend

logger = logging.getLogger(__name__)


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def _tone_dbfs(raw: np.ndarray, f_s: float, f: float) -> float:
    n = len(raw)
    w = signal.windows.blackmanharris(n)
    X = np.fft.fft(raw / ADC_FULL_SCALE * w) / np.sum(w)
    fr = np.fft.fftfreq(n, 1 / f_s)
    k = int(np.argmin(np.abs(fr - f)))
    lo, hi = max(k - 3, 0), k + 4
    return 20 * math.log10(float(np.max(np.abs(X[lo:hi]))) + 1e-15)


def _noise_density_dbfs_hz(raw: np.ndarray, f_s: float, avoid: list[float]) -> float:
    f, P = signal.welch(raw / ADC_FULL_SCALE, fs=f_s, nperseg=4096, return_onesided=False)
    m = np.ones_like(f, dtype=bool)
    for a in avoid:
        m &= np.abs(f - a) > 5 * f_s / 4096
    m &= np.abs(f) < 0.4 * f_s
    return 10 * math.log10(float(np.median(P[m])) + 1e-30)


def analyse_raw(raw: np.ndarray, cfg: Config) -> dict:
    f_s, f_off = cfg.sdr.f_s, cfg.emission.f_offset
    a = np.abs(raw) / ADC_FULL_SCALE
    return {
        "rms_dbfs": 20 * math.log10(float(np.sqrt(np.mean(a**2))) + 1e-15),
        "peak_dbfs": 20 * math.log10(float(np.max(a)) + 1e-15),
        "clip_frac": float(np.mean((np.abs(raw.real) >= 2047) | (np.abs(raw.imag) >= 2047))),
        "tone_dbfs": _tone_dbfs(raw, f_s, f_off),
        "dc_dbfs": _tone_dbfs(raw, f_s, 0.0),
        "image_dbfs": _tone_dbfs(raw, f_s, -f_off),
        "noise_dbfs_hz": _noise_density_dbfs_hz(raw, f_s, [0.0, f_off, -f_off]),
    }


def stability_report(slow: np.ndarray, fs: float, wavelength: float) -> dict:
    """Bruit de phase du clutter statique → bruit équivalent en déplacement."""
    c = slow.mean()
    ph = np.unwrap(np.angle(slow * np.conj(c) / abs(c)))
    amp = np.abs(slow)
    f, P = signal.welch(signal.detrend(ph), fs=fs, nperseg=min(len(ph), int(20 * fs)))
    band = (f >= 0.1) & (f <= 0.8)
    ph_rms_band = math.sqrt(float(np.trapezoid(P[band], f[band]))) if np.any(band) else float("nan")
    disp_um = ph_rms_band * wavelength / (4 * math.pi) * 1e6
    return {
        "static_dbfs": 20 * math.log10(abs(c) + 1e-15),
        "phase_pp_mrad": 1e3 * float(np.ptp(ph)),
        "phase_rms_band_mrad": 1e3 * ph_rms_band,
        "disp_noise_band_um": disp_um,
        "amp_rel_std_pct": 100 * float(np.std(amp) / (np.mean(amp) + 1e-15)),
        "duration_s": len(slow) / fs,
    }


def _print_stability(r: dict) -> None:
    _p(f"  clutter statique       : {r['static_dbfs']:.1f} dBFS")
    _p(f"  dérive de phase (p-p)  : {r['phase_pp_mrad']:.1f} mrad sur {r['duration_s']:.0f} s")
    _p(f"  bruit de phase 0.1–0.8 Hz : {r['phase_rms_band_mrad']:.2f} mrad rms"
       f"  ≈ {r['disp_noise_band_um']:.1f} µm de déplacement équivalent")
    _p(f"  stabilité d'amplitude  : {r['amp_rel_std_pct']:.2f} %")
    # Le bruit de phase porte sur le clutter statique (fuite + murs).  Pour une
    # cible N dB plus faible que ce clutter, le bruit équivalent en déplacement
    # est multiplié par 10^(N/20) : c'est ce chiffre qui compte.
    for rel in (30, 40, 50):
        d = r["disp_noise_band_um"] * 10 ** (rel / 20)
        _p(f"  → cible {rel} dB sous le clutter : bruit ≈ {d / 1000:.2f} mm "
           f"(respiration 1–10 mm ; il faut ≳ 5× ce bruit)")
    _p("  Réduire la fuite TX→RX (écarter/isoler les antennes) abaisse directement ce plancher.")


def _recommend_gain(an: dict, cfg: Config) -> None:
    pk = an["peak_dbfs"]
    if an["clip_frac"] > 0 or pk > -1:
        _p(f"  ⚠ SATURATION ADC (crête {pk:.1f} dBFS) → baisser rx_gain_db de "
           f"{max(6, pk + 10):.0f} dB ou augmenter l'atténuation TX.")
    elif pk < -30:
        _p(f"  Niveau faible (crête {pk:.1f} dBFS) → on peut monter rx_gain_db de "
           f"~{min(30, -12 - pk):.0f} dB (viser une crête ≈ −12 dBFS).")
    else:
        _p(f"  Niveau correct (crête {pk:.1f} dBFS, cible −20…−6 dBFS).")


def run_check(cfg: Config, stability_s: float = 0.0, sim: bool = False) -> int:
    _p("=== Diagnostic radar ===")
    if sim:
        from radar.sources.simulation import SimulatedRawSource
        src = SimulatedRawSource(cfg, scenario="empty", realtime=False)
        cfg.recording.save_raw = True
        gen = src.chunks()
        chunk = next(gen)
        raw = chunk.raw.astype(np.complex128)
        _p("[simulation] source RF simulée, scénario « empty »")
        _report_raw(raw, cfg)
        if stability_s > 0:
            fe = Frontend(cfg.sdr.f_s, cfg.emission.f_offset, cfg.frontend.fs_slow,
                          cfg.sdr.rx_buffer_size, cfg.frontend.cic_order)
            slows = []
            t_sig = 0.0
            for ch in gen:
                slows.append(fe(ch.raw))
                t_sig += cfg.sdr.rx_buffer_size / cfg.sdr.f_s
                if t_sig >= stability_s + 2:
                    break
            slow = np.concatenate(slows)[int(2 * cfg.frontend.fs_slow):]
            _p(f"\n[4] Stabilité ({stability_s:.0f} s)")
            _print_stability(stability_report(slow, cfg.frontend.fs_slow, cfg.sdr.wavelength))
        return 0

    from radar.sources.pluto import ensure_libiio_on_path, make_tx_waveform, open_pluto
    ensure_libiio_on_path()
    _p("[1] libiio")
    try:
        import iio
        _p(f"  version {'.'.join(str(v) for v in iio.version[:2])} ({iio.version[2]})")
        ctxs = iio.scan_contexts()
        if ctxs:
            for uri, desc in ctxs.items():
                _p(f"  contexte : {uri}  —  {desc}")
        else:
            _p("  aucun contexte USB détecté (normal si le Pluto n'a que le réseau RNDIS).")
    except Exception as exc:
        _p(f"  ✗ libiio indisponible : {exc}")
        return 2
    if cfg.sdr.uri.startswith("ip:"):
        host = cfg.sdr.uri[3:]
        try:
            with socket.create_connection((host, 30431), timeout=1.5):
                _p(f"  serveur IIO joignable sur {host}:30431")
        except OSError:
            _p(f"  ✗ {host}:30431 injoignable (Pluto non branché, driver RNDIS absent, "
               "ou mauvais port USB).")

    _p("\n[2] Connexion")
    try:
        sdr = open_pluto(cfg)
    except RuntimeError as exc:
        _p(f"  ✗ {exc}")
        _hint_windows()
        return 1
    try:
        ctx = sdr.ctx
        attrs = dict(ctx.attrs)
        for k in ("hw_model", "fw_version", "ad9361-phy,model", "hw_serial"):
            if k in attrs:
                _p(f"  {k:18s}: {attrs[k]}")
    except Exception:
        pass
    _p(f"  f_c={cfg.sdr.f_c / 1e9:.3f} GHz  f_s={cfg.sdr.f_s / 1e3:.0f} kS/s  "
       f"RX {cfg.sdr.rx_gain_db} dB  TX {cfg.sdr.tx_atten_db} dB")

    tx = make_tx_waveform(cfg.sdr.f_s, cfg.emission.f_offset, cfg.sdr.rx_buffer_size,
                          cfg.emission.amplitude)
    sdr.tx_cyclic_buffer = True
    sdr.tx(tx)
    try:
        for _ in range(3):
            sdr.rx()
        _p("\n[3] Capture")
        raw = np.asarray(sdr.rx(), dtype=np.complex128)
        _report_raw(raw, cfg)

        _p("\n[5] Débit (3 s)")
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            n += len(sdr.rx())
        rate = n / (time.perf_counter() - t0)
        ok = rate > 0.97 * cfg.sdr.f_s
        _p(f"  {rate / 1e3:.0f} kS/s mesurés pour {cfg.sdr.f_s / 1e3:.0f} nominal → "
           f"{'OK' if ok else '✗ pertes probables (USB lent ? baisser f_s)'}")

        if stability_s > 0:
            _p(f"\n[4] Stabilité ({stability_s:.0f} s) — ne pas bouger près des antennes")
            fe = Frontend(cfg.sdr.f_s, cfg.emission.f_offset, cfg.frontend.fs_slow,
                          cfg.sdr.rx_buffer_size, cfg.frontend.cic_order)
            slows = []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < stability_s + 2:
                slows.append(fe(sdr.rx()))
            slow = np.concatenate(slows)[int(2 * cfg.frontend.fs_slow):]
            _print_stability(stability_report(slow, cfg.frontend.fs_slow, cfg.sdr.wavelength))
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
    return 0


def _report_raw(raw: np.ndarray, cfg: Config) -> None:
    an = analyse_raw(raw, cfg)
    _p(f"  ADC RMS / crête        : {an['rms_dbfs']:.1f} / {an['peak_dbfs']:.1f} dBFS"
       f"  (écrêtage {100 * an['clip_frac']:.3f} %)")
    _p(f"  raie f_offset (écho)   : {an['tone_dbfs']:.1f} dBFS   ← fuite TX→RX + échos statiques")
    _p(f"  DC récepteur           : {an['dc_dbfs']:.1f} dBFS")
    _p(f"  image −f_offset        : {an['image_dbfs']:.1f} dBFS")
    _p(f"  plancher de bruit      : {an['noise_dbfs_hz']:.1f} dBFS/Hz")
    snr_slow = an["tone_dbfs"] - (an["noise_dbfs_hz"] + 10 * math.log10(cfg.frontend.fs_slow))
    _p(f"  dynamique slow-time    : {snr_slow:.0f} dB (raie / bruit dans {cfg.frontend.fs_slow:.0f} Hz)")
    _recommend_gain(an, cfg)


def _hint_windows() -> None:
    if os.name != "nt":
        return
    _p("\n  Pistes Windows :")
    _p("   • Gestionnaire de périphériques : chercher « PlutoSDR » / VID 0456 PID B673.")
    _p("     Rien du tout → câble « charge seule » ou port micro-USB d'alimentation.")
    _p("   • Installer « PlutoSDR-M2k-USB-Drivers.exe » (ADI) : carte réseau RNDIS")
    _p("     192.168.2.1 + accès libusb pour uri « usb: ».")
    _p("   • Le lecteur « PlutoSDR » (clé USB) doit apparaître dans l'explorateur.")


def run_scan(cfg: Config, start: float, stop: float, step: float, tx: bool = False) -> int:
    """``radar scan`` : bruit ambiant (et, avec --tx, fuite TX→RX) sur une plage de fréquences.

    Sans émission : médiane et maximum de la puissance reçue par tranches de
    1 ms (brouilleurs : Wi-Fi, BT, réseau mobile, drones FPV à 5.8 GHz…).
    Avec --tx : niveau de la tonalité reçue = couplage TX→RX, qui suit la
    réponse des antennes.  Pour choisir sdr.f_c (bande calme où les antennes
    marchent) — indispensable avant de passer à 5.8 GHz.
    """
    from radar.sources.pluto import lo_range, make_tx_waveform, open_pluto
    sdr = open_pluto(cfg)
    rng = lo_range(sdr)
    if rng:
        _p(f"Plage LO du firmware : {rng[0] / 1e6:.0f}–{rng[1] / 1e6:.0f} MHz")
        start, stop = max(start, rng[0]), min(stop, rng[1])
    if tx:
        wav = make_tx_waveform(cfg.sdr.f_s, cfg.emission.f_offset, cfg.sdr.rx_buffer_size,
                               cfg.emission.amplitude)
        sdr.tx_cyclic_buffer = True
        sdr.tx(wav)
    else:
        sdr.tx_hardwaregain_chan0 = -89.75          # émetteur au minimum
    _p(f"{'MHz':>7s} {'médiane':>8s} {'max':>7s}" + (f" {'fuite':>7s}" if tx else "") + "  (dBFS)")
    rows = []
    try:
        for f in np.arange(start, stop + 1, step):
            sdr.rx_lo = int(f)
            if tx:
                sdr.tx_lo = int(f)
            for _ in range(2):
                sdr.rx()
            p_med, p_max, tone = [], [], []
            for _ in range(4):
                raw = np.asarray(sdr.rx(), dtype=np.complex128)
                pw = np.abs(raw / ADC_FULL_SCALE) ** 2
                seg = pw[: len(pw) // 1000 * 1000].reshape(-1, 1000).mean(axis=1)
                p_med.append(np.median(seg))
                p_max.append(seg.max())
                if tx:
                    tone.append(_tone_dbfs(raw, cfg.sdr.f_s, cfg.emission.f_offset))
            med = 10 * math.log10(float(np.median(p_med)) + 1e-15)
            mx = 10 * math.log10(float(np.max(p_max)) + 1e-15)
            line = f"{f / 1e6:7.0f} {med:8.1f} {mx:7.1f}"
            if tx:
                line += f" {np.median(tone):7.1f}"
            flag = "  ← brouilleur" if mx - med > 10 else ""
            _p(line + flag)
            rows.append((f, med, mx, float(np.median(tone)) if tx else None))
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
    quiet = [r for r in rows if r[2] - r[1] < 6]
    if tx and quiet:
        best = max(quiet, key=lambda r: r[3])
        _p(f"\nFréquence calme au meilleur couplage d'antennes : {best[0] / 1e6:.0f} MHz "
           f"(fuite {best[3]:.1f} dBFS).  Attention : un fort couplage = forte fuite ; "
           "c'est la réponse des antennes qui compte, pas le niveau absolu.")
    return 0
