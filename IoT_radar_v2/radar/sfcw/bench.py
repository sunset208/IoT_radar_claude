"""``radar sfcw bench`` : ce que le Pluto permet vraiment en fréquence balayée.

Mesures (quelques minutes, antennes branchées, personne devant) :

1. **niveaux par fréquence** — amplitude de la fuite sur la bande balayée,
   crête ADC → recommandation de gain (crête ≤ −6 dBFS) ; montre la réponse
   des antennes ;
2. **latence d'écriture des LO** (TX et RX, petit et grand saut) ;
3. **buffers périmés et temps par pas** pour plusieurs tailles de buffer et
   nombres de buffers noyau → cadence de balayage atteignable ;
4. **répétabilité d'amplitude** sur des balayages répétés : bruit par pas,
   part commune (qui tombe dans la case 0) et **plancher dans le domaine
   distance** → cible la plus faible détectable (relatif à la fuite) ;
5. (``--fastlock``) rappel de profils fastlock (8 max) contre écriture directe.

Rapport imprimé + JSON dans ``data/sfcw/bench_<date>.json``.  ``--sim`` :
même déroulé sur le faux Pluto (vérifie l'outil, pas le matériel).
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import math
import time

import numpy as np

from radar.config import Config, resolve_path
from radar.sfcw.sources import (LoControl, _peak_dbfs, is_fresh, open_pluto_sfcw, start_tone,
                                tone_bins)


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def _db(v: float) -> float:
    return 20 * math.log10(max(v, 1e-12))


def _step(sdr, lo: LoControl, sc, f: float, parity: int, bins, clk, max_reads: int = 20):
    """Un pas de mesure ; renvoie (amplitude FS, lectures, durée s, crête dBFS)."""
    tagged = sc.tag_offset_hz > 0 and parity % 2 == 1
    b_exp, b_oth = (bins[1], bins[0]) if tagged else (bins[0], bins[1])
    t0 = clk()
    lo.set_tx(f)
    lo.set_rx(f + (sc.tag_offset_hz if tagged else 0.0))
    n = sc.rx_buffer_size
    x = None
    for i in range(max_reads):
        x = np.asarray(sdr.rx())
        ok, v = is_fresh(x, b_exp, b_oth)
        if ok:
            return abs(v) / n / 2048.0, i + 1, clk() - t0, _peak_dbfs(x)
    return float("nan"), max_reads, clk() - t0, _peak_dbfs(x)


def _set_buffers(sdr, n: int, k: int) -> None:
    from radar.sources.pluto import set_kernel_buffers
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass
    sdr.rx_buffer_size = int(n)
    set_kernel_buffers(sdr, k)
    for _ in range(2):
        sdr.rx()


def run_bench(cfg: Config, sdr=None, repeats: int = 10, buffer_sizes=(1024, 4096),
              kernel_buffers=(1, 2, 4), fastlock: bool = False, out: str | None = None) -> dict:
    sc = cfg.sfcw
    freqs = sc.freqs
    report: dict = {"date": dt.datetime.now().isoformat(timespec="seconds"),
                    "config": {k: getattr(sc, k) for k in (
                        "f_start", "f_step", "n_steps", "f_s", "tone_hz", "tag_offset_hz",
                        "rx_buffer_size", "kernel_buffers", "rx_gain_db", "tx_atten_db")}}
    _p("=== Banc SFCW (fréquence balayée, amplitude seule) ===")
    if sdr is not None and hasattr(sdr, "now"):
        _p("[SIMULATION] faux Pluto : vérifie l'outil, les chiffres ne disent rien du matériel.")
        report["simulation"] = True
    _p(f"Balayage {freqs[0] / 1e9:.3f}–{freqs[-1] / 1e9:.3f} GHz, {sc.n_steps} pas de "
       f"{sc.f_step / 1e6:.0f} MHz → résolution {sc.resolution_m * 100:.0f} cm, portée non ambiguë "
       f"{sc.unambiguous_range_m:.2f} m")
    sdr, restore = open_pluto_sfcw(cfg, sdr)
    clk = getattr(sdr, "now", None) or time.perf_counter
    try:
        from radar.sources.pluto import lo_range
        rng = lo_range(sdr)
        if rng:
            _p(f"Plage LO du firmware : {rng[0] / 1e6:.0f}–{rng[1] / 1e6:.0f} MHz "
               f"({'AD9364 déverrouillé' if rng[1] > 4e9 else 'AD9363 d origine'})")
            report["lo_range"] = rng
        start_tone(sdr, sc)
        for _ in range(3):
            sdr.rx()
        lo = LoControl(sdr)
        bins = tone_bins(sc)

        # ---------------------------------------------------------- 1. niveaux
        _p("\n[1] Niveau de la fuite par fréquence (réponse des antennes)")
        amps, peaks = [], []
        for k, f in enumerate(freqs):
            a, nr, _, pk = _step(sdr, lo, sc, f, k, bins, clk)
            amps.append(a)
            peaks.append(pk)
        amps_db = np.array([_db(a) for a in amps])
        for k in range(0, len(freqs), max(1, len(freqs) // 10)):
            bar = "#" * int(max(0, amps_db[k] + 60) / 2)
            _p(f"  {freqs[k] / 1e6:7.0f} MHz  {amps_db[k]:6.1f} dBFS  {bar}")
        pk = float(np.nanmax(peaks))
        _p(f"  min {np.nanmin(amps_db):.1f} / max {np.nanmax(amps_db):.1f} dBFS, crête ADC {pk:.1f} dBFS")
        if pk > -3:
            _p(f"  ⚠ proche de la saturation → baisser sfcw.rx_gain_db de {pk + 8:.0f} dB")
        elif pk < -25:
            _p(f"  niveau faible → on peut monter sfcw.rx_gain_db de ~{-8 - pk:.0f} dB")
        else:
            _p("  niveau correct (crête visée ≤ −6 dBFS)")
        report["levels_dbfs"] = amps_db.round(2).tolist()
        report["adc_peak_dbfs"] = pk

        # ---------------------------------------------------------- 2. latence LO
        _p("\n[2] Latence d'écriture des LO")
        lat = {}
        for name, jump in (("petit saut", sc.f_step), ("grand saut", 500e6)):
            ttx, trx = [], []
            f0 = freqs[len(freqs) // 2]
            for i in range(30):
                f = f0 + (jump if i % 2 else 0.0)
                f = min(f, freqs[-1]) if jump > sc.f_step else f
                t0 = clk()
                lo.set_tx(f)
                t1 = clk()
                lo.set_rx(f)
                t2 = clk()
                ttx.append(t1 - t0)
                trx.append(t2 - t1)
            lat[name] = {"tx_ms": 1e3 * float(np.median(ttx)), "rx_ms": 1e3 * float(np.median(trx)),
                         "tx_p95_ms": 1e3 * float(np.percentile(ttx, 95)),
                         "rx_p95_ms": 1e3 * float(np.percentile(trx, 95))}
            _p(f"  {name:10s}: TX {lat[name]['tx_ms']:.2f} ms (p95 {lat[name]['tx_p95_ms']:.2f}), "
               f"RX {lat[name]['rx_ms']:.2f} ms (p95 {lat[name]['rx_p95_ms']:.2f})")
        report["lo_latency"] = lat

        # ---------------------------------------------------------- 3. buffers
        _p("\n[3] Buffers périmés et temps par pas (marquage de la tonalité)")
        _p(f"  {'buffer':>7s} {'noyau':>5s} {'lect./pas':>10s} {'max':>4s} {'ms/pas':>7s} "
           f"{'balayages/s':>12s}")
        combos = []
        for n in buffer_sizes:
            if n % int(round(sc.f_s / max(sc.tone_hz - sc.tag_offset_hz, 1))) or n % int(
                    round(sc.f_s / sc.tone_hz)):
                continue
            for kb in kernel_buffers:
                try:
                    _set_buffers(sdr, n, kb)
                except Exception as exc:
                    _p(f"  {n:7d} {kb:5d}  impossible ({exc})")
                    continue
                sc_n = dc.replace(sc, rx_buffer_size=n)
                bins_n = tone_bins(sc_n)
                reads, times, bad = [], [], 0
                for k in range(40):
                    f = freqs[(k * 7) % len(freqs)]
                    a, nr, dtk, _ = _step(sdr, lo, sc_n, f, k, bins_n, clk)
                    reads.append(nr)
                    times.append(dtk)
                    bad += not np.isfinite(a)
                ms = 1e3 * float(np.median(times))
                rate = 1e3 / (ms * sc.n_steps)
                combos.append({"buffer": n, "kernel_buffers": kb, "reads_median": float(np.median(reads)),
                               "reads_max": int(np.max(reads)), "ms_per_step": ms,
                               "sweeps_per_s": rate, "timeouts": bad})
                _p(f"  {n:7d} {kb:5d} {np.median(reads):10.1f} {np.max(reads):4d} {ms:7.2f} "
                   f"{rate:12.2f}" + (f"   ({bad} échecs)" if bad else ""))
        report["buffers"] = combos
        ok = [c for c in combos if c["timeouts"] == 0]
        if ok:
            best = max(ok, key=lambda c: c["sweeps_per_s"])
            report["recommended"] = best
            _p(f"  → recommandé : rx_buffer_size {best['buffer']}, kernel_buffers "
               f"{best['kernel_buffers']} ≈ {best['sweeps_per_s']:.1f} balayages/s pour "
               f"{sc.n_steps} pas (il en faut ≥ 2 pour la respiration, 4–5 pour le confort)")
            if best["sweeps_per_s"] < 2:
                _p(f"  ⚠ trop lent : réduire n_steps (ex. {int(sc.n_steps * best['sweeps_per_s'] / 3)}) "
                   "ou augmenter f_step (portée non ambiguë c/4Δf)")
        _set_buffers(sdr, sc.rx_buffer_size, sc.kernel_buffers)

        # ---------------------------------------------------------- 4. répétabilité
        _p(f"\n[4] Répétabilité sur {repeats} balayages (ne pas bouger devant les antennes)")
        M = np.full((repeats, sc.n_steps), np.nan)
        t_sw = []
        g = 0
        for r in range(repeats):
            t0 = clk()
            for k, f in enumerate(freqs):
                M[r, k], _, _, _ = _step(sdr, lo, sc, f, g, bins, clk)
                g += 1
            t_sw.append(clk() - t0)
        good = np.all(np.isfinite(M), axis=1)
        M = M[good]
        rep: dict = {"sweep_s_median": float(np.median(t_sw)), "n_sweeps": int(len(M))}
        if len(M) >= 3:
            rel = M / M.mean(axis=0) - 1.0
            common = rel.mean(axis=1, keepdims=True)
            resid = rel - common
            rep["step_noise_db"] = float(np.median(20 * np.log10(1 + np.std(rel, axis=0))))
            rep["common_noise_db"] = float(20 * np.log10(1 + np.std(common)))
            rep["residual_rel_dbc"] = float(20 * np.log10(np.median(np.std(resid, axis=0)) + 1e-12))
            from radar.sfcw.engine import make_processor
            proc = make_processor(cfg)
            R = np.abs(proc.range_transform(resid))
            valid = proc.range_m >= sc.min_range_m
            floor = float(np.sqrt(np.mean(R[:, valid] ** 2)))
            # une cible à −X dB sous la fuite module x_k de 2·10^(−X/20) ; après la
            # FFT fenêtrée (Σw = 1) son pic vaut ≈ 10^(−X/20)
            rep["range_floor_rel_db"] = _db(floor)
            rep["min_target_rel_db_per_sweep"] = _db(floor) + 10.0
            n_int = 20.0 / max(np.median(t_sw), 1e-3)
            rep["min_target_rel_db_20s"] = _db(floor) + 10.0 - 10 * math.log10(max(n_int, 1.0))
            _p(f"  durée d'un balayage : {np.median(t_sw) * 1e3:.0f} ms "
               f"({1 / max(np.median(t_sw), 1e-6):.2f} balayages/s)")
            _p(f"  bruit d'amplitude par pas : {rep['step_noise_db']:.3f} dB rms "
               f"(dont commun au balayage : {rep['common_noise_db']:.3f} dB → case 0, sans effet)")
            _p(f"  résidu relatif par pas : {rep['residual_rel_dbc']:.1f} dBc")
            _p(f"  plancher dans le domaine distance : {rep['range_floor_rel_db']:.1f} dB / fuite")
            _p(f"  → cible détectable jusqu'à ~{-rep['min_target_rel_db_per_sweep']:.0f} dB sous la fuite "
               f"en un balayage, ~{-rep['min_target_rel_db_20s']:.0f} dB après 20 s d'intégration "
               "(repère : la respiration vue en CW le 27/09 était ~40 dB sous la fuite)")
        report["repeatability"] = rep

        # ---------------------------------------------------------- 5. fastlock
        if fastlock:
            _p("\n[5] Fastlock (8 profils) contre écriture directe")
            report["fastlock"] = _bench_fastlock(sdr, freqs, clk)
    finally:
        restore()
    path = resolve_path(out or cfg.sfcw.record_dir) / f"bench_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    _p(f"\nRapport : {path}")
    return report


def _bench_fastlock(sdr, freqs, clk) -> dict:
    """Stocke 8 profils RX/TX puis compare rappel et écriture directe."""
    try:
        rx = sdr._ctrl.find_channel("altvoltage0", True)
        tx = sdr._ctrl.find_channel("altvoltage1", True)
        sel = freqs[np.linspace(0, len(freqs) - 1, 8).astype(int)]
        for p, f in enumerate(sel):
            for ch in (rx, tx):
                ch.attrs["frequency"].value = str(int(f))
                ch.attrs["fastlock_store"].value = str(p)
        t_rec, t_dir = [], []
        for i in range(40):
            p = i % 8
            t0 = clk()
            tx.attrs["fastlock_recall"].value = str(p)
            rx.attrs["fastlock_recall"].value = str(p)
            t1 = clk()
            tx.attrs["frequency"].value = str(int(sel[p]))
            rx.attrs["frequency"].value = str(int(sel[p]))
            t2 = clk()
            t_rec.append(t1 - t0)
            t_dir.append(t2 - t1)
        res = {"recall_ms": 1e3 * float(np.median(t_rec)), "direct_ms": 1e3 * float(np.median(t_dir))}
        _p(f"  rappel fastlock (TX+RX) : {res['recall_ms']:.2f} ms ; écriture directe : "
           f"{res['direct_ms']:.2f} ms")
        return res
    except Exception as exc:
        _p(f"  fastlock indisponible : {exc}")
        return {"error": str(exc)}
