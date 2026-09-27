"""Traitements hors ligne : extraction de features, évaluation, calibration.

* ``radar evaluate``  — rejoue le détecteur (machine à états incluse) sur des
  enregistrements ou des scénarios simulés et sort un tableau de métriques.
* ``radar calibrate`` — fixe les seuils à partir d'enregistrements étiquetés
  (au minimum des « salle vide ») pour un taux de fausse alarme visé ;
  ajuste une régression logistique si les deux classes sont disponibles.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import math
from pathlib import Path

import numpy as np

from radar.config import Config
from radar.dsp.detector import BREATHING, Detector, LogisticModel
from radar.dsp.vitals import Features
from radar.pipeline import make_analyzer
from radar.recorder import Recording, list_recordings, load_recording

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------

def window_features(x: np.ndarray, fs: float, cfg: Config) -> list[Features]:
    an = make_analyzer(cfg, fs)
    w = int(round(cfg.analysis.window_s * fs))
    h = max(1, int(round(cfg.analysis.hop_s * fs)))
    out = []
    for end in range(w, len(x) + 1, h):
        f, _ = an.analyze(x[end - w:end], t=end / fs, want_display=False)
        out.append(f)
    return out


def recording_segments(rec: Recording) -> list[np.ndarray]:
    """Découpe aux pertes d'échantillons (la phase n'y est plus continue)."""
    cuts = [0] + sorted(int(g) for g in rec.gaps if 0 < g < len(rec.slow)) + [len(rec.slow)]
    return [rec.slow[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


# ----------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------

def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # rangs moyens pour les ex-aequo
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = sums[inv] / cnt[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 200) -> LogisticModel:
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.column_stack((Z, np.ones(len(Z))))
    w = np.zeros(Z1.shape[1])
    for _ in range(iters):             # Newton / IRLS
        p = 1 / (1 + np.exp(-np.clip(Z1 @ w, -30, 30)))
        g = Z1.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        Hm = (Z1 * (p * (1 - p))[:, None]).T @ Z1 + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        step = np.linalg.solve(Hm + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return LogisticModel(w=w[:-1], b=float(w[-1]), mu=mu, sd=sd)


# ----------------------------------------------------------------------
# Évaluation (machine à états complète)
# ----------------------------------------------------------------------

@dc.dataclass
class EvalRow:
    name: str
    label: int
    duration_s: float
    frac_breathing: float
    frac_motion: float
    latency_s: float | None
    rate_err_bpm: float | None
    extra: str = ""


def evaluate_sequence(x: np.ndarray, fs: float, cfg: Config, label: int, name: str,
                      calibration: dict | None = None, true_bpm: float | None = None) -> EvalRow:
    det = Detector(cfg.detector, calibration)
    feats = window_features(x, fs, cfg)
    states, bpms = [], []
    for f in feats:
        d = det.update(f)
        states.append(d.state)
        bpms.append(d.breath_bpm)
    n = max(len(states), 1)
    fb = sum(s == BREATHING for s in states) / n
    fm = sum(s == "MOUVEMENT" for s in states) / n
    lat = None
    if label == 1:
        for i, s in enumerate(states):
            if s == BREATHING:
                lat = cfg.analysis.window_s + i * cfg.analysis.hop_s
                break
    err = None
    if true_bpm is not None:
        vals = [b for b in bpms if b is not None]
        if vals:
            err = float(np.median(np.abs(np.asarray(vals) - true_bpm)))
    return EvalRow(name, label, len(x) / fs, fb, fm, lat, err)


def print_table(rows: list[EvalRow]) -> None:
    print(f"{'enregistrement':34s} {'lab':>3s} {'durée':>6s} {'%RESP':>6s} {'%MOUV':>6s} "
          f"{'latence':>8s} {'err bpm':>8s}")
    for r in rows:
        lat = "—" if r.latency_s is None else f"{r.latency_s:.1f}s"
        err = "—" if r.rate_err_bpm is None else f"{r.rate_err_bpm:.1f}"
        print(f"{r.name[:34]:34s} {r.label:>3d} {r.duration_s:5.0f}s {100 * r.frac_breathing:5.1f}% "
              f"{100 * r.frac_motion:5.1f}% {lat:>8s} {err:>8s}")
    pos = [r for r in rows if r.label == 1]
    neg = [r for r in rows if r.label == 0]
    print("-" * 80)
    if pos:
        det = np.mean([r.latency_s is not None for r in pos])
        print(f"Positifs : {len(pos)} — détectés {100 * det:.0f} %, temps en RESPIRATION "
              f"{100 * np.mean([r.frac_breathing for r in pos]):.0f} %, latence médiane "
              f"{np.median([r.latency_s for r in pos if r.latency_s is not None] or [np.nan]):.1f} s")
    if neg:
        tot = sum(r.duration_s for r in neg)
        fa = sum(r.frac_breathing * r.duration_s for r in neg)
        print(f"Négatifs : {len(neg)} ({tot / 60:.1f} min) — temps en fausse alarme "
              f"{100 * fa / max(tot, 1e-9):.2f} %")


def evaluate_recordings(paths: list[Path], cfg: Config, calibration: dict | None) -> list[EvalRow]:
    rows = []
    for p in paths:
        rec = load_recording(p)
        for k, seg in enumerate(recording_segments(rec)):
            if len(seg) < cfg.analysis.window_s * rec.fs_slow:
                continue
            truth = rec.meta.get("truth") or {}
            rows.append(evaluate_sequence(
                seg, rec.fs_slow, cfg, rec.label, p.stem + (f"#{k}" if k else ""), calibration,
                truth.get("breath_rate_bpm")))
    return rows


def evaluate_simulation(cfg: Config, n_per: int = 3, duration_s: float = 120.0,
                        seed: int = 0, calibration: dict | None = None) -> list[EvalRow]:
    from radar.scene import BREATHING_SCENARIOS, SCENARIOS, random_scene_params
    from radar.sources.simulation import simulate_slow
    rng = np.random.default_rng(seed)
    fs = cfg.frontend.fs_slow
    rows = []
    for sc in SCENARIOS:
        for i in range(n_per):
            p = random_scene_params(rng, cfg.sdr.wavelength, sc)
            x, truth = simulate_slow(sc, duration_s, fs, cfg.sdr.wavelength, rng, params=p)
            label = int(sc in BREATHING_SCENARIOS)
            row = evaluate_sequence(x, fs, cfg, label, f"sim:{sc}#{i} snr={p.snr_db:.0f}dB",
                                    calibration, truth.get("breath_rate_bpm"))
            rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def calibrate(paths: list[Path], cfg: Config, pfa: float = 0.01) -> dict:
    X, y, groups = [], [], []
    per_thr_neg, snr_neg, motion_neg, noise_neg = [], [], [], []
    for p in paths:
        rec = load_recording(p)
        if rec.label not in (0, 1):
            continue
        for seg in recording_segments(rec):
            if len(seg) < cfg.analysis.window_s * rec.fs_slow:
                continue
            for f in window_features(seg, rec.fs_slow, cfg):
                X.append(f.vector())
                y.append(rec.label)
                groups.append(rec.group)
                if rec.label == 0:
                    snr_neg.append(f.snr_db)
                    per_thr_neg.append(f.periodicity)
                    motion_neg.append(f.motion)
                    noise_neg.append(f.noise_dbfs)
    if not snr_neg:
        raise SystemExit("Calibration impossible : aucun enregistrement étiqueté 0 (salle vide).")
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    q = 100 * (1 - pfa)
    thresholds = {
        "snr_threshold_db": float(np.percentile(snr_neg, q)) + 1.0,
        "periodicity_threshold": float(max(0.2, np.percentile(per_thr_neg, 90))),
        # le seuil de mouvement n'est PAS recalculé : des négatifs peuvent
        # légitimement contenir du mouvement (quelqu'un qui marche) ; on
        # rapporte seulement sa distribution
        "motion_threshold": float(cfg.detector.motion_threshold),
    }
    out = {
        "pfa_target_per_window": pfa,
        "n_windows": int(len(y)),
        "n_neg": int((y == 0).sum()),
        "n_pos": int((y == 1).sum()),
        "thresholds": thresholds,
        "noise_ref_dbfs": float(np.median(noise_neg)),
        "motion_neg_percentiles": {q_: float(np.percentile(motion_neg, q_)) for q_ in (50, 90, 99)},
        "feature_keys": list(Features.VECTOR_KEYS),
    }
    if (y == 1).any() and (y == 0).any():
        # AUC en validation croisée groupée par session (pas de fuite)
        ug = sorted(set(groups))
        g_arr = np.asarray(groups)
        scores = np.full(len(y), np.nan)
        folds = min(5, len(ug))
        for k in range(folds):
            test_g = set(ug[k::folds])
            te = np.isin(g_arr, list(test_g))
            if te.all() or not (~te).any() or len(set(y[~te])) < 2:
                continue
            m = fit_logistic(X[~te], y[~te])
            scores[te] = [m.prob(v) for v in X[te]]
        ok = ~np.isnan(scores)
        out["cv_auc_logistic"] = roc_auc(y[ok], scores[ok]) if ok.any() else None
        out["auc_snr_only"] = roc_auc(y, X[:, 0])
        model = fit_logistic(X, y)
        neg_p = [model.prob(v) for v in X[y == 0]]
        model.threshold = float(np.percentile(neg_p, q))
        out["logistic"] = model.to_dict()
        pos_p = np.asarray([model.prob(v) for v in X[y == 1]])
        out["pd_per_window_logistic"] = float(np.mean(pos_p >= model.threshold))
        snr_pos = X[y == 1, 0]
        out["pd_per_window_snr_rule"] = float(np.mean(snr_pos >= thresholds["snr_threshold_db"]))
    return out


def save_calibration(cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2, ensure_ascii=False)


def find_recordings(root_or_files: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in root_or_files:
        p = Path(item)
        if p.is_dir():
            out += list_recordings(p)
        elif p.suffix == ".npz":
            out.append(p)
    return out


def fmt_pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f} %"
