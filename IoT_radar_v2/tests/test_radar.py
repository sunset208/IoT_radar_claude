"""Tests : python -m pytest tests -q   (depuis IoT_radar_v2, venv local)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar.config import load_config  # noqa: E402
from radar.dsp.detector import BREATHING, Detector  # noqa: E402
from radar.dsp.frontend import ADC_FULL_SCALE, Frontend  # noqa: E402
from radar.dsp.vitals import VitalsAnalyzer, fit_circle  # noqa: E402
from radar.offline import evaluate_sequence, fit_logistic, roc_auc  # noqa: E402
from radar.recorder import Recorder, load_recording  # noqa: E402
from radar.sources.pluto import make_tx_waveform  # noqa: E402
from radar.sources.simulation import simulate_slow  # noqa: E402

FS = 50.0
LAM = 299_792_458.0 / 3.5e9


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------- config
def test_config_validation(cfg):
    assert cfg.cic_factor == 100
    assert cfg.total_decimation == 20000
    with pytest.raises(ValueError):
        load_config(overrides={"emission": {"f_offset": 3333.0}})
    with pytest.raises(ValueError):
        load_config(overrides={"sdr": {"rx_buffer_size": 12345}})


# ---------------------------------------------------------------- TX
def test_tx_waveform_integer_periods():
    tx = make_tx_waveform(1e6, 1e4, 100_000, 0.5)
    assert len(tx) % 100 == 0
    # continuité au bouclage du buffer cyclique
    step = tx[1] / tx[0]
    assert abs(tx[0] / tx[-1] - step) < 1e-4


# ---------------------------------------------------------------- frontend
def _tone(n0, n, f, fs, amp=0.5):
    k = np.arange(n0, n0 + n)
    return amp * ADC_FULL_SCALE * np.exp(2j * np.pi * f * k / fs)


def test_frontend_streaming_equals_block():
    fs, foff, N = 1e6, 1e4, 20_000
    rng = np.random.default_rng(0)
    x = _tone(0, 6 * N, foff + 0.3, fs) + 50 * (rng.standard_normal(6 * N) + 1j * rng.standard_normal(6 * N))
    fe1 = Frontend(fs, foff, 50.0, N)
    y1 = np.concatenate([fe1(x[i:i + N]) for i in range(0, 6 * N, N)])
    fe2 = Frontend(fs, foff, 50.0, 6 * N)
    y2 = fe2(x)
    assert len(y1) == len(y2) == 6 * N // 20000
    np.testing.assert_allclose(y1, y2, rtol=1e-4, atol=1e-6)


def test_frontend_passband_and_rejection():
    fs, foff, N = 1e6, 1e4, 100_000
    fe = Frontend(fs, foff, 50.0, N)
    sig = lambda n0: _tone(n0, N, foff + 0.3, fs, 0.25)
    dc = lambda n0: np.full(N, 0.25 * ADC_FULL_SCALE, complex)           # offset DC récepteur
    image = lambda n0: _tone(n0, N, -foff, fs, 0.25)
    out = {"sig": [], "dc": [], "image": []}
    for name, gen in (("sig", sig), ("dc", dc), ("image", image)):
        fe = Frontend(fs, foff, 50.0, N)
        for i in range(40):                      # 4 s
            out[name].append(fe(gen(i * N)))
    lvl = {k: np.sqrt(np.mean(np.abs(np.concatenate(v)[100:]) ** 2)) for k, v in out.items()}
    assert abs(20 * np.log10(lvl["sig"] / 0.25)) < 0.5          # gain unitaire
    assert 20 * np.log10(lvl["dc"] / 0.25 + 1e-30) < -80        # DC récepteur rejeté
    assert 20 * np.log10(lvl["image"] / 0.25 + 1e-30) < -80     # image rejetée


# ---------------------------------------------------------------- vitals
def test_circle_fit():
    rng = np.random.default_rng(1)
    th = rng.uniform(0.2, 1.4, 500)
    x = (3 + 4j) + 2.0 * np.exp(1j * th) + 0.01 * (rng.standard_normal(500) + 1j * rng.standard_normal(500))
    c, r, res = fit_circle(x)
    assert abs(c - (3 + 4j)) < 0.05 and abs(r - 2.0) < 0.05 and res < 0.02


@pytest.mark.parametrize("bpm", [8.0, 15.0, 30.0])
def test_rate_estimation(bpm):
    rng = np.random.default_rng(int(bpm))
    x, _ = simulate_slow("breathing", 60, FS, LAM, rng, snr_db=20, breath_rate_bpm=bpm)
    va = VitalsAnalyzer(FS, LAM)
    est = [va.analyze(x[i:i + 1000], want_display=False)[0].breath_bpm for i in range(0, 2000, 250)]
    assert abs(np.median(est) - bpm) < 0.1 * bpm + 1.0


def test_empty_room_false_alarms_with_lo_drift(cfg):
    """Le détecteur doit rester muet en salle vide, même avec une forte dérive LO."""
    rng = np.random.default_rng(7)
    va = VitalsAnalyzer(FS, LAM)
    snrs = []
    for drift in (0.0, 0.01, 0.03):
        x, _ = simulate_slow("empty", 90, FS, LAM, rng, snr_db=30, lo_drift_rad=drift)
        snrs += [va.analyze(x[i:i + 1000], want_display=False)[0].snr_db for i in range(0, len(x) - 1000, 100)]
    assert np.percentile(snrs, 99) < cfg.detector.snr_threshold_db


def test_state_machine_scenarios(cfg):
    rng = np.random.default_rng(11)
    x, tr = simulate_slow("breathing", 90, FS, LAM, rng, snr_db=15)
    r = evaluate_sequence(x, FS, cfg, 1, "b", true_bpm=tr["breath_rate_bpm"])
    assert r.latency_s is not None and r.latency_s < 30 and r.frac_breathing > 0.7
    x, _ = simulate_slow("empty", 120, FS, LAM, rng, snr_db=15, lo_drift_rad=0.01)
    r = evaluate_sequence(x, FS, cfg, 0, "e")
    assert r.frac_breathing < 0.05
    x, _ = simulate_slow("walker", 120, FS, LAM, rng, snr_db=20)
    r = evaluate_sequence(x, FS, cfg, 0, "w")
    assert r.frac_breathing < 0.1


# ---------------------------------------------------------------- offline utils
def test_roc_auc_and_logistic():
    rng = np.random.default_rng(0)
    X = np.vstack((rng.normal(0, 1, (300, 2)), rng.normal(1.5, 1, (300, 2))))
    y = np.r_[np.zeros(300), np.ones(300)]
    m = fit_logistic(X, y)
    p = np.array([m.prob(v) for v in X])
    assert roc_auc(y, p) > 0.8
    assert abs(roc_auc(y, rng.random(600)) - 0.5) < 0.08


# ---------------------------------------------------------------- recorder
def test_recorder_roundtrip(tmp_path):
    rec = Recorder(tmp_path, FS, 1, {"hello": "monde"}, tag="test session")
    x = (np.arange(500) + 1j).astype(np.complex64)
    rec.add(x[:250])
    rec.annotate("apnée")
    rec.add(x[250:], discontinuity=True)
    path = rec.close()
    r = load_recording(path)
    np.testing.assert_array_equal(r.slow, x)
    assert r.label == 1 and r.gaps.tolist() == [250] and r.events[0][1] == "apnée"
    assert r.meta["hello"] == "monde"


# ---------------------------------------------------------------- pipeline
def test_engine_end_to_end(cfg):
    from radar.pipeline import Engine
    from radar.scene import Scene, SceneParams
    from radar.sources.simulation import SimulatedSlowSource
    rng = np.random.default_rng(5)
    scene = Scene(SceneParams(scenario="breathing", wavelength=LAM, snr_db=20), rng)
    src = SimulatedSlowSource(FS, scene, realtime=False, rng=rng)
    eng = Engine(cfg, src)
    states, fast = [], []
    eng.subscribe(lambda s: states.append(s["decision"]["state"]) if "decision" in s else None)
    eng.subscribe(lambda s: fast.append(s["fast"]) if "fast" in s else None)
    eng.start()
    t0 = time.time()
    while len(states) < 120 and time.time() - t0 < 30:
        time.sleep(0.05)
    eng.stop()
    assert BREATHING in states
    assert eng.error is None
    # indices rapides publiés à ~10 Hz, ~4× plus souvent que les analyses (2 Hz)
    assert len(fast) > 3 * len(states)
    assert fast[-1]["presence_db"] > 6.0


def test_detector_hysteresis(cfg):
    from radar.dsp.vitals import Features
    det = Detector(cfg.detector, window_s=cfg.analysis.window_s)
    pos = Features(snr_db=20, concentration=0.8, acf=0.8, motion=1.2, band_db=20, breath_hz=0.3)
    neg = Features(snr_db=2, concentration=0.05, acf=0.2, motion=1.2, band_db=1, breath_hz=0.3)
    states = [det.update(pos).state for _ in range(cfg.detector.on_count)]
    assert states[-1] == BREATHING and states[0] != BREATHING
    states = [det.update(neg).state for _ in range(cfg.detector.off_count)]
    assert states[-2] == BREATHING and states[-1] != BREATHING


# ---------------------------------------------------------------- legacy v1
def test_convert_v1(tmp_path, cfg):
    from radar.legacy import convert_tree
    from radar.recorder import load_recording
    rng = np.random.default_rng(3)
    slow, _ = simulate_slow("breathing", 90, 2000.0, LAM, rng, snr_db=10)   # « slow-time » à 2 kHz
    n = np.arange(len(slow))
    v1 = (slow * np.exp(2j * np.pi * 500.0 * n / 2000.0)).astype(np.complex64)   # porteuse à +500 Hz
    d = tmp_path / "train"
    d.mkdir()
    v1.tofile(d / "3.iq")
    np.savez(d / "3.npz", label=np.int8(1), f_s_dec_hz=np.float64(2000.0), env=np.asarray("salle"),
             subset=np.asarray("train"))
    outs = convert_tree(tmp_path, tmp_path / "out")
    assert len(outs) == 1
    rec = load_recording(outs[0])
    assert rec.label == 1 and rec.fs_slow == 50.0 and abs(rec.duration_s - 85) < 1
    r = evaluate_sequence(rec.slow, rec.fs_slow, cfg, 1, "v1")
    assert r.frac_breathing > 0.5


# ---------------------------------------------------------------- vrai Pluto
def test_real_pluto_paced_breathing(cfg):
    """Enregistrement réel (27/09/2026) : sujet à 50 cm, 15 resp/min cadencées, 1.8 GHz."""
    from radar.config import load_config
    rec = load_recording(Path(__file__).parent / "data" / "real_pluto_paced15_1800MHz.npz")
    c = load_config(overrides={"sdr": {"f_c": 1.8e9}})
    r = evaluate_sequence(rec.slow, rec.fs_slow, c, 1, "real")
    assert r.latency_s is not None and r.frac_breathing > 0.6
    va = VitalsAnalyzer(rec.fs_slow, c.sdr.wavelength)
    bpm = [va.analyze(rec.slow[i:i + 1000], want_display=False)[0].breath_bpm
           for i in range(1000, len(rec.slow) - 1000, 250)]
    assert abs(np.median(bpm) - 15.0) < 1.0


# ---------------------------------------------------------------- indices rapides / multi-échelle
def _run_fast(x, fs=FS, params=None):
    from radar.dsp.fast import FastMonitor
    fm = FastMonitor(fs, params, LAM)
    ticks = []
    for i in range(0, len(x), 5):
        ticks += fm.push(x[i:i + 5], (i + 5) / fs)
    return ticks


def test_fast_monitor_white_noise_is_0db():
    """Sous H0 (bruit blanc, clutter parfait) les indices valent ≈ 0 dB."""
    from radar.dsp.fast import FastParams
    rng = np.random.default_rng(0)
    x = 0.3 + 1e-4 * (rng.standard_normal(9000) + 1j * rng.standard_normal(9000))
    t = _run_fast(x, params=FastParams(clutter_amp_dbc=-150, clutter_phase_dbc=-150))
    assert len(t) > 800                       # ~10 Hz après 4 s de mise en route
    pres = np.array([k.presence_r_db for k in t])
    act = np.array([k.activity_db for k in t])
    assert abs(np.median(pres)) < 1.5 and abs(np.median(act)) < 1.0
    assert np.percentile(pres, 99.9) < 6.0 and np.percentile(act, 99.9) < 8.0


def test_fast_presence_and_activity(cfg):
    rng = np.random.default_rng(4)
    x, _ = simulate_slow("breathing", 40, FS, LAM, rng, snr_db=15)
    t = _run_fast(x)
    assert np.median([k.presence_db for k in t]) > cfg.fast.presence_threshold_db + 1
    x, _ = simulate_slow("walker", 40, FS, LAM, rng, snr_db=25)
    t = _run_fast(x)
    assert np.percentile([k.activity_db for k in t], 90) > cfg.fast.activity_threshold_db


def test_first_alert_is_fast_on_real_recording(cfg):
    """Vrai Pluto : la présence (signe de vie) est signalée en < 8 s, bien avant la
    confirmation de la respiration (~30 s) ; la salle vide reste muette."""
    from radar.config import load_config
    c = load_config(overrides={"sdr": {"f_c": 1.8e9}})
    rec = load_recording(Path(__file__).parent / "data" / "real_pluto_paced15_1800MHz.npz")
    r = evaluate_sequence(rec.slow, rec.fs_slow, c, 1, "real")
    assert r.alert_s is not None and r.alert_s < 8.0
    assert r.latency_s is not None and r.alert_s < r.latency_s
    rng = np.random.default_rng(9)
    x, _ = simulate_slow("empty", 120, FS, c.sdr.wavelength, rng, snr_db=20, lo_drift_rad=0.01)
    r = evaluate_sequence(x, FS, c, 0, "e")
    assert r.alert_s is None


def test_multiscale_confirms_faster_at_high_snr(cfg):
    """À fort SNR, une fenêtre courte confirme la respiration bien avant 20 s."""
    rng = np.random.default_rng(21)
    x, _ = simulate_slow("breathing", 60, FS, LAM, rng, snr_db=30, breath_depth_mm=6)
    r = evaluate_sequence(x, FS, cfg, 1, "b")
    assert r.latency_s is not None and r.latency_s < 18.0      # 20 s seule : ≥ 22.5 s
    assert r.confirm_scale_s is not None and r.confirm_scale_s < cfg.analysis.window_s


def test_protocol_plans_are_valid():
    from radar.protocol import PLANS_DIR, load_plan
    for p in sorted(PLANS_DIR.glob("*.yaml")):
        name, steps = load_plan(str(p))
        assert steps and all(s.label in (-1, 0, 1) and s.duration_s > 0 for s in steps)
        assert any(s.label == 0 for s in steps)          # chaque plan contient de la salle vide
