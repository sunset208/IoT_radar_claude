"""Traitements hors ligne : rejeu, évaluation, calibration.

* ``radar evaluate``  — rejoue la chaîne **temps réel exacte** (:class:`radar.pipeline.Processor` :
  indices rapides 10 Hz + analyses multi-échelle + machine à états) sur des
  enregistrements ou des scénarios simulés et sort un tableau de métriques.
* ``radar calibrate`` — fixe les seuils à partir d'enregistrements étiquetés
  (au minimum des « salle vide ») pour un taux de fausse alarme visé :
  seuils SNR / périodicité **par échelle**, bruit multiplicatif du clutter
  (indice de présence), seuils des indices rapides, plancher de référence.
"""

from __future__ import annotations

import dataclasses as dc
import json
import logging
import math
from pathlib import Path

import numpy as np

from radar.config import Config
from radar.dsp.detector import BREATHING, EMPTY, MOTION, PRESENCE, LogisticModel
from radar.dsp.fast import FastMonitor, measure_clutter_noise
from radar.dsp.vitals import Features
from radar.pipeline import Processor, fast_params, make_analyzer
from radar.recorder import Recording, list_recordings, load_recording

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Features / rejeu
# ----------------------------------------------------------------------

def window_features(x: np.ndarray, fs: float, cfg: Config, window_s: float | None = None,
                    full: bool = True) -> list[Features]:
    """Features d'une seule échelle sur toute la séquence (pas ``hop_s``)."""
    an = make_analyzer(cfg, fs)
    w = int(round((window_s or cfg.analysis.window_s) * fs))
    h = max(1, int(round(cfg.analysis.hop_s * fs)))
    out = []
    for end in range(w, len(x) + 1, h):
        f, _ = an.analyze(x[end - w:end], t=end / fs, want_display=False, full=full)
        out.append(f)
    return out


@dc.dataclass
class Trace:
    """Rejeu d'une séquence par le Processor temps réel."""
    fast_t: np.ndarray
    fast_state: list
    presence_db: np.ndarray
    activity_db: np.ndarray
    fast_raw: np.ndarray            # (N, 5) : E_r, E_t, N_r, N_t, |C|²
    ana_t: np.ndarray
    decisions: list
    feats: list                     # [{fenêtre_s: Features}]

    def states(self) -> tuple[np.ndarray, list]:
        """(temps, états) à la meilleure cadence disponible (10 Hz si indices rapides)."""
        if len(self.fast_t):
            return self.fast_t, self.fast_state
        return self.ana_t, [d.state for d in self.decisions]


def run_sequence(x: np.ndarray, fs: float, cfg: Config, calibration: dict | None = None,
                 chunk_s: float = 0.1) -> Trace:
    proc = Processor(cfg, fs, calibration)
    n = max(1, int(round(chunk_s * fs)))
    ft, fs_, pr, ac, raw, at, dec, feats = [], [], [], [], [], [], [], []
    for i in range(0, len(x), n):
        blk = x[i:i + n]
        for ev in proc.push(blk, (i + len(blk)) / fs, want_display=False):
            if ev[0] == "fast":
                tick, state = ev[1], ev[2]
                ft.append(tick.t)
                fs_.append(state)
                pr.append(tick.presence_db)
                ac.append(tick.activity_db)
                raw.append(tick.raw)
            else:
                r = ev[1]
                at.append(r.t)
                dec.append(r.decision)
                feats.append(r.feats)
    return Trace(np.asarray(ft), fs_, np.asarray(pr), np.asarray(ac),
                 np.asarray(raw).reshape(-1, 5), np.asarray(at), dec, feats)


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
# Évaluation (chaîne temps réel complète)
# ----------------------------------------------------------------------

@dc.dataclass
class EvalRow:
    name: str
    label: int
    duration_s: float
    frac_breathing: float
    frac_motion: float
    latency_s: float | None          # 1re RESPIRATION (s depuis le début)
    rate_err_bpm: float | None
    extra: str = ""
    frac_presence: float = 0.0
    alert_s: float | None = None     # 1er état ≠ VIDE (« signe de vie »)
    confirm_scale_s: float | None = None
    empty_scene: bool = False        # scène sans personne (vide) : toute alerte est fausse


def evaluate_sequence(x: np.ndarray, fs: float, cfg: Config, label: int, name: str,
                      calibration: dict | None = None, true_bpm: float | None = None,
                      empty_scene: bool | None = None) -> EvalRow:
    tr = run_sequence(x, fs, cfg, calibration)
    t, states = tr.states()
    # fractions de temps en régime établi (fenêtre longue remplie) : comparables
    # d'une version à l'autre ; les délais, eux, partent du début
    steady = [s for ti, s in zip(t, states) if ti >= cfg.analysis.window_s]
    n = max(len(steady), 1)
    fb = sum(s == BREATHING for s in steady) / n
    fm = sum(s == MOTION for s in steady) / n
    fp = sum(s == PRESENCE for s in steady) / n
    lat = next((float(ti) for ti, s in zip(t, states) if s == BREATHING), None)
    alert = next((float(ti) for ti, s in zip(t, states) if s != EMPTY), None)
    conf = next((d.confirmed_by_s for d in tr.decisions if d.confirmed_by_s is not None), None)
    err = None
    if true_bpm is not None:
        vals = [d.breath_bpm for d in tr.decisions if d.breath_bpm is not None]
        if vals:
            err = float(np.median(np.abs(np.asarray(vals) - true_bpm)))
    return EvalRow(name, label, len(x) / fs, fb, fm, lat, err, frac_presence=fp, alert_s=alert,
                   confirm_scale_s=conf,
                   empty_scene=(label == 0) if empty_scene is None else empty_scene)


def print_table(rows: list[EvalRow]) -> None:
    print(f"{'enregistrement':34s} {'lab':>3s} {'durée':>6s} {'%RESP':>6s} {'%PRÉS':>6s} "
          f"{'%MOUV':>6s} {'alerte':>7s} {'resp.':>7s} {'éch.':>5s} {'err bpm':>7s}")
    for r in rows:
        lat = "—" if r.latency_s is None else f"{r.latency_s:.1f}s"
        al = "—" if r.alert_s is None else f"{r.alert_s:.1f}s"
        sc = "—" if r.confirm_scale_s is None else f"{r.confirm_scale_s:.0f}s"
        err = "—" if r.rate_err_bpm is None else f"{r.rate_err_bpm:.1f}"
        print(f"{r.name[:34]:34s} {r.label:>3d} {r.duration_s:5.0f}s {100 * r.frac_breathing:5.1f}% "
              f"{100 * r.frac_presence:5.1f}% {100 * r.frac_motion:5.1f}% {al:>7s} {lat:>7s} "
              f"{sc:>5s} {err:>7s}")
    pos = [r for r in rows if r.label == 1]
    neg = [r for r in rows if r.label == 0]
    print("-" * 96)
    if pos:
        det = np.mean([r.latency_s is not None for r in pos])
        lats = [r.latency_s for r in pos if r.latency_s is not None]
        alerts = [r.alert_s for r in pos if r.alert_s is not None]
        print(f"Positifs : {len(pos)} — respiration détectée {100 * det:.0f} %, temps en RESPIRATION "
              f"{100 * np.mean([r.frac_breathing for r in pos]):.0f} %, latence médiane "
              f"{np.median(lats) if lats else float('nan'):.1f} s ; "
              f"1re alerte (signe de vie) médiane {np.median(alerts) if alerts else float('nan'):.1f} s")
    if neg:
        tot = sum(r.duration_s for r in neg)
        fa = sum(r.frac_breathing * r.duration_s for r in neg)
        print(f"Négatifs : {len(neg)} ({tot / 60:.1f} min) — temps en fausse alarme RESPIRATION "
              f"{100 * fa / max(tot, 1e-9):.2f} %")
        emp = [r for r in neg if r.empty_scene]
        if emp:
            te = sum(r.duration_s for r in emp)
            fpres = sum((r.frac_breathing + r.frac_presence + r.frac_motion) * r.duration_s
                        for r in emp)
            print(f"Scènes vides : {len(emp)} ({te / 60:.1f} min) — temps en alerte (tout état ≠ VIDE) "
                  f"{100 * fpres / max(te, 1e-9):.2f} %")


def evaluate_recordings(paths: list[Path], cfg: Config, calibration: dict | None) -> list[EvalRow]:
    rows = []
    for p in paths:
        rec = load_recording(p)
        for k, seg in enumerate(recording_segments(rec)):
            if len(seg) < min(cfg.scales_s) * rec.fs_slow + 20:
                continue
            truth = rec.meta.get("truth") or {}
            empty = None
            if truth.get("scenario"):
                empty = truth["scenario"] == "empty"
            rows.append(evaluate_sequence(
                seg, rec.fs_slow, cfg, rec.label, p.stem + (f"#{k}" if k else ""), calibration,
                truth.get("breath_rate_bpm"), empty_scene=empty))
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
                                    calibration, truth.get("breath_rate_bpm"),
                                    empty_scene=(sc == "empty"))
            rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def _fast_h0(segs: list[np.ndarray], fs: float, cfg: Config, amp_dbc: float,
             phase_dbc: float) -> tuple[np.ndarray, np.ndarray]:
    """(presence_db, activity_db) en salle vide avec des constantes de clutter données."""
    pres, act = [], []
    base = fast_params(cfg)
    for x in segs:
        p = dc.replace(base, clutter_amp_dbc=amp_dbc, clutter_phase_dbc=phase_dbc)
        fm = FastMonitor(fs, p, cfg.sdr.wavelength)
        n = max(1, int(round(0.1 * fs)))
        for i in range(0, len(x), n):
            for tk in fm.push(x[i:i + n], (i + n) / fs):
                pres.append(tk.presence_db)
                act.append(tk.activity_db)
    return np.asarray(pres), np.asarray(act)


def calibrate(paths: list[Path], cfg: Config, pfa: float = 0.01) -> dict:
    scales = cfg.scales_s
    X, y, groups = [], [], []
    per_scale = {w: {"snr0": [], "per0": [], "snr1": []} for w in scales}
    motion_neg, noise_neg = [], []
    raw_neg: list[np.ndarray] = []
    segs_neg: list[np.ndarray] = []
    fs_ref = None
    for p in paths:
        rec = load_recording(p)
        if rec.label not in (0, 1):
            continue
        fs_ref = fs_ref or rec.fs_slow
        for seg in recording_segments(rec):
            if len(seg) < cfg.analysis.window_s * rec.fs_slow:
                continue
            tr = run_sequence(seg, rec.fs_slow, cfg)
            if rec.label == 0:
                segs_neg.append(seg)
                if len(tr.fast_raw):
                    raw_neg.append(tr.fast_raw)
            for fd in tr.feats:
                for w, f in fd.items():
                    if rec.label == 0:
                        per_scale[w]["snr0"].append(f.snr_db)
                        per_scale[w]["per0"].append(f.periodicity)
                    else:
                        per_scale[w]["snr1"].append(f.snr_db)
                w_long = max(fd)
                if w_long != cfg.analysis.window_s:
                    continue
                f = fd[w_long]
                X.append(f.vector())
                y.append(rec.label)
                groups.append(rec.group)
                if rec.label == 0:
                    motion_neg.append(f.motion)
                    noise_neg.append(f.noise_dbfs)
    if not motion_neg:
        raise SystemExit("Calibration impossible : aucun enregistrement étiqueté 0 (salle vide) "
                         f"d'au moins {cfg.analysis.window_s:.0f} s.")
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    # fausse alarme visée par fenêtre, répartie sur les échelles (union)
    q = 100 * (1 - pfa / len(scales))
    main = cfg.analysis.window_s
    h0_s = sum(len(s_) for s_ in segs_neg) / fs_ref
    # Peu de « salle vide » (fenêtres très corrélées) : les percentiles hauts
    # sont mal estimés → on ne descend pas sous les seuils par défaut.
    scarce = h0_s < 600.0
    defaults = {main: (cfg.detector.snr_threshold_db, cfg.detector.periodicity_threshold)}
    for w_, snr_, per_ in zip(cfg.analysis.extra_windows_s, cfg.detector.scale_snr_threshold_db,
                              cfg.detector.scale_periodicity_threshold):
        defaults[float(w_)] = (snr_, per_)

    def thr_for(w):
        d = per_scale[w]
        snr_t = float(np.percentile(d["snr0"], q)) + 1.0
        per_t = float(max(0.2, np.percentile(d["per0"], 90)))
        if scarce:
            snr_t = max(snr_t, defaults[w][0])
            per_t = max(per_t, defaults[w][1])
        return {"snr_threshold_db": snr_t, "periodicity_threshold": per_t,
                "snr_h0_max": float(np.max(d["snr0"])),
                "n_neg": len(d["snr0"]),
                "pd_per_window": (float(np.mean(np.asarray(d["snr1"]) >= snr_t))
                                  if d["snr1"] else None)}

    main_thr = thr_for(main)
    thresholds = {
        "snr_threshold_db": main_thr["snr_threshold_db"],
        "periodicity_threshold": main_thr["periodicity_threshold"],
        # le seuil de mouvement n'est PAS recalculé : des négatifs peuvent
        # légitimement contenir du mouvement (quelqu'un qui marche) ; on
        # rapporte seulement sa distribution
        "motion_threshold": float(cfg.detector.motion_threshold),
    }
    out = {
        "pfa_target_per_window": pfa,
        "h0_duration_s": h0_s,
        "warning": ("moins de 10 min de salle vide : seuils bornés par les valeurs par défaut "
                    "(enregistrer plus de « vide » pour une vraie calibration)") if scarce else None,
        "scales_s": list(scales),
        "n_windows": int(len(y)),
        "n_neg": int((y == 0).sum()),
        "n_pos": int((y == 1).sum()),
        "thresholds": thresholds,
        "scales": {f"{w:g}": thr_for(w) for w in scales if w != main},
        "noise_ref_dbfs": float(np.median(noise_neg)),
        "motion_neg_percentiles": {q_: float(np.percentile(motion_neg, q_)) for q_ in (50, 90, 99)},
        "feature_keys": list(Features.VECTOR_KEYS),
    }
    # --- indices rapides : bruit multiplicatif du clutter + seuils -----------
    if raw_neg and cfg.fast.enabled:
        R = np.vstack(raw_neg)
        enbw_p = FastMonitor(fs_ref, fast_params(cfg), cfg.sdr.wavelength).enbw_p
        amp, ph = measure_clutter_noise(R[:, 0], R[:, 1], R[:, 2], R[:, 3], R[:, 4], enbw_p)
        pres0, act0 = _fast_h0(segs_neg, fs_ref, cfg, amp, ph)
        pres_floor = cfg.fast.presence_threshold_db if scarce else 4.0
        act_floor = cfg.fast.activity_threshold_db if scarce else 6.0
        out["fast"] = {
            "clutter_amp_dbc": amp,
            "clutter_phase_dbc": ph,
            "clutter_phase_mrad_rms": 1e3 * math.sqrt(10 ** (ph / 10)),
            "presence_threshold_db": float(max(pres_floor, np.percentile(pres0, 99.9) + 1.0)),
            "activity_threshold_db": float(max(act_floor, np.percentile(act0, 99.9) + 2.0)),
            "presence_h0_percentiles": {q_: float(np.percentile(pres0, q_)) for q_ in (50, 99, 99.9)},
            "activity_h0_percentiles": {q_: float(np.percentile(act0, q_)) for q_ in (50, 99, 99.9)},
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
        auc = roc_auc(y[ok], scores[ok]) if ok.any() else float("nan")
        out["cv_auc_logistic"] = None if math.isnan(auc) else auc
        out["auc_snr_only"] = roc_auc(y, X[:, 0])
        snr_pos = X[y == 1, 0]
        out["pd_per_window_snr_rule"] = float(np.mean(snr_pos >= thresholds["snr_threshold_db"]))
        # La régression logistique n'est retenue qu'avec assez de sessions de
        # chaque classe (sinon elle apprend les sessions, pas la respiration).
        n_sess = {c: len({g for g, yy in zip(groups, y) if yy == c}) for c in (0, 1)}
        out["n_sessions"] = {"neg": n_sess[0], "pos": n_sess[1]}
        if min(n_sess.values()) >= 3:
            model = fit_logistic(X, y)
            neg_p = [model.prob(v) for v in X[y == 0]]
            model.threshold = float(np.percentile(neg_p, 100 * (1 - pfa)))
            out["logistic"] = model.to_dict()
            pos_p = np.asarray([model.prob(v) for v in X[y == 1]])
            out["pd_per_window_logistic"] = float(np.mean(pos_p >= model.threshold))
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
