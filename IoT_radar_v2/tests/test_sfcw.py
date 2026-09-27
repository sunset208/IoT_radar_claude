"""Tests du mode SFCW (amplitude seule) — tout tourne sans matériel (simulation + faux Pluto)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar.config import load_config  # noqa: E402
from radar.sfcw.dsp import Sweep, tone_amplitude  # noqa: E402
from radar.sfcw.engine import make_processor  # noqa: E402
from radar.sfcw.scene import FakePluto, SfcwScene, SfcwSceneParams  # noqa: E402
from radar.sfcw.sources import (SfcwPlutoSource, SfcwRecorder, SfcwReplaySource,  # noqa: E402
                                SfcwSimSource, is_fresh, load_sfcw)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _run(cfg, src, until_s: float):
    proc = make_processor(cfg, getattr(getattr(src, "rec", None), "freqs", None))
    res = []
    for sw in src.sweeps():
        r = proc.push(sw)
        if r and "state" in r:
            res.append(r)
        if sw.t > until_s:
            break
    src.stop()
    return proc, res


def test_tone_estimator_rejects_dc_and_image():
    fs, ft, n = 4e6, 250e3, 1024
    k = np.arange(n)
    x = 2048 * (0.1 * np.exp(2j * np.pi * ft * k / fs) + 0.3 + 0.05 * np.exp(-2j * np.pi * ft * k / fs))
    assert abs(abs(tone_amplitude(x, fs, ft)) - 0.1) < 1e-9


def test_fresh_buffer_detection():
    """Le marquage rejette le buffer périmé (autre tonalité) et le buffer à cheval."""
    n, b_exp, b_oth = 1024, 32, 64
    k = np.arange(n)
    tone = lambda b: 1000 * np.exp(2j * np.pi * b * k / n)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert is_fresh(tone(b_exp) + noise, b_exp, b_oth)[0]
    assert not is_fresh(tone(b_oth) + noise, b_exp, b_oth)[0]            # périmé
    half = np.where(k < n // 2, 0.0, 1.0) * tone(b_exp)                    # à cheval (hors bande → frais)
    assert not is_fresh(half + noise, b_exp, b_oth)[0]


@pytest.mark.parametrize("r_m", [0.8, 1.6, 2.8])
def test_range_and_breathing(cfg, r_m):
    src = SfcwSimSource(cfg, SfcwSceneParams(target_range_m=r_m, target_rel_db=-45, breath_rate_bpm=18),
                        sweep_hz=4, realtime=False, rng=np.random.default_rng(int(10 * r_m)))
    _, res = _run(cfg, src, 45)
    last = res[-1]
    assert last["state"] == "RESPIRATION"
    assert abs(last["best_range_m"] - r_m) < cfg.sfcw.resolution_m
    assert abs(last["best_bpm"] - 18) < 3


def test_empty_and_walker(cfg):
    src = SfcwSimSource(cfg, SfcwSceneParams(scenario="empty"), sweep_hz=4, realtime=False,
                        rng=np.random.default_rng(1))
    _, res = _run(cfg, src, 50)
    assert all(r["state"] != "RESPIRATION" for r in res)
    src = SfcwSimSource(cfg, SfcwSceneParams(scenario="walker"), sweep_hz=4, realtime=False,
                        rng=np.random.default_rng(2))
    _, res = _run(cfg, src, 50)
    st = [r["state"] for r in res[10:]]
    assert "RESPIRATION" not in st and st.count("MOUVEMENT") > 0.8 * len(st)


def test_breathing_separated_from_walker(cfg):
    """La distance sépare une victime immobile d'un sauveteur qui marche (impossible en CW)."""
    src = SfcwSimSource(cfg, SfcwSceneParams(scenario="breathing_walker", target_range_m=1.0,
                                             target_rel_db=-40), sweep_hz=4, realtime=False,
                        rng=np.random.default_rng(3))
    _, res = _run(cfg, src, 50)
    last = res[-1]
    assert last["state"] == "RESPIRATION" and abs(last["best_range_m"] - 1.0) < 0.15
    assert last["motion_extent_m"] is not None and last["motion_extent_m"][0] > 1.1


def test_pluto_code_on_fake_pluto(cfg):
    """Code Pluto réel (LO, marquage, buffers périmés) sur le faux Pluto : aucune
    phase aléatoire n'y résiste pas, aucun pas raté, cible localisée."""
    rng = np.random.default_rng(4)
    fake = FakePluto(SfcwScene(SfcwSceneParams(target_range_m=1.4, target_rel_db=-40), rng), rng=rng)
    src = SfcwPlutoSource(cfg, sdr=fake)
    proc, res = _run(cfg, src, 30)
    assert src.timeouts == 0
    assert fake._ctrl.attrs["calib_mode"].value == "auto"          # restauré à la fermeture
    assert res[-1]["state"] == "RESPIRATION"
    assert abs(res[-1]["best_range_m"] - 1.4) < cfg.sfcw.resolution_m
    assert 2.0 < proc.sweep_hz < 8.0


def test_recorder_and_replay(cfg, tmp_path):
    freqs = cfg.sfcw.freqs
    rec = SfcwRecorder(tmp_path, freqs, 1, {"hello": 1}, tag="essai")
    rng = np.random.default_rng(5)
    sim = SfcwScene(SfcwSceneParams(target_range_m=2.0), rng)
    for i in range(160):
        a, _ = sim.sweep(freqs, i * 0.25, 0.004)
        rec.add(Sweep(amps=a, t=i * 0.25))
    rec.annotate("apnée")
    path = rec.close()
    r = load_sfcw(path)
    assert r.amps.shape == (160, len(freqs)) and r.meta["kind"] == "sfcw" and r.events[0][1] == "apnée"
    src = SfcwReplaySource(path, realtime=False, loop=False)
    _, res = _run(cfg, src, 1e9)
    assert abs(res[-1]["best_range_m"] - 2.0) < cfg.sfcw.resolution_m


def test_bench_on_fake(cfg, tmp_path):
    from radar.sfcw.bench import run_bench
    fake = FakePluto(SfcwScene(rng=np.random.default_rng(6)), rng=np.random.default_rng(7))
    rep = run_bench(cfg, sdr=fake, repeats=3, buffer_sizes=(1024,), kernel_buffers=(2,),
                    out=str(tmp_path))
    assert rep["buffers"][0]["timeouts"] == 0
    assert rep["buffers"][0]["reads_median"] == 4        # 2 périmés + 1 à cheval + 1 frais
    assert rep["repeatability"]["n_sweeps"] == 3


def test_frequency_plan_validation():
    with pytest.raises(ValueError):
        load_config(overrides={"sdr": {"f_c": 5.8e9}})                    # AD9363 : 3.8 GHz max
    c = load_config(overrides={"sdr": {"f_c": 5.8e9, "chip": "ad9364"}})
    assert abs(c.sdr.wavelength - 0.0517) < 1e-3
    with pytest.raises(ValueError):
        load_config(overrides={"sfcw": {"f_start": 3.5e9}})               # balayage > 3.8 GHz
    with pytest.raises(ValueError):
        load_config(overrides={"sfcw": {"rx_buffer_size": 1000}})         # tons non orthogonaux
    c = load_config()
    assert abs(c.sfcw.unambiguous_range_m - 3.747) < 0.01                 # c/(4Δf), mesures réelles
