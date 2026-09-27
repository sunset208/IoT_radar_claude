"""Entraînement du détecteur IA.

Étapes
------
``pretrain``  simulateur physique seul (randomisation de domaine).
``finetune``  mélange enregistrements réels (sessions de train) +
              semi-synthétique (fonds vides réels + cible simulée) + synthétique,
              validation sur des **sessions** réelles jamais vues.

Exemples (sur la machine d'entraînement, pas ici) ::

    python ai/train.py pretrain
    python ai/train.py finetune --init ai/runs/pretrain_XXXX/best.pt
    python ai/train.py pretrain --check      # vérifie la chaîne sans entraîner

Chaque run écrit ``ai/runs/<étape>_<date>/`` : ``best.pt``, ``history.csv``,
``report.json`` (AUC, Pd @ Pfa, comparaison au détecteur classique).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, IterableDataset, Subset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ai.data import (  # noqa: E402
    RecordingWindows, SemiSyntheticWindows, SyntheticWindows, find_npz, group_split,
)
from ai.models import build_model, count_params  # noqa: E402
from radar.offline import roc_auc  # noqa: E402


# ----------------------------------------------------------------------
def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Mixture(IterableDataset):
    """Échantillonne des sources (itérables ou indexables) selon des poids."""

    def __init__(self, sources: dict, weights: dict, seed: int = 0) -> None:
        self.sources = {k: v for k, v in sources.items() if v is not None and weights.get(k, 0) > 0}
        w = np.asarray([weights[k] for k in self.sources], float)
        self.p = w / w.sum()
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 31337
                                    + int(torch.initial_seed() % 100000))
        its = {k: iter(v) for k, v in self.sources.items() if isinstance(v, IterableDataset)}
        keys = list(self.sources)
        while True:
            k = keys[int(rng.choice(len(keys), p=self.p))]
            src = self.sources[k]
            if k in its:
                yield next(its[k])
            else:
                yield src[int(rng.integers(len(src)))]


def pd_at_pfa(y: np.ndarray, s: np.ndarray, pfa: float) -> tuple[float, float]:
    neg = s[y < 0.5]
    if len(neg) == 0 or (y > 0.5).sum() == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1 - pfa))
    return float(np.mean(s[y > 0.5] > thr)), thr


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    P, Y, R, RT = [], [], [], []
    for x, y, r in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logit, rate = model(x)
        P.append(torch.sigmoid(logit.float()).cpu().numpy())
        R.append(rate.float().cpu().numpy())
        Y.append(y.numpy())
        RT.append(r.numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(R), np.concatenate(RT)


def metrics(p, y, r, rt, pfa) -> dict:
    pd, thr = pd_at_pfa(y, p, pfa)
    m = (rt > 0) & (y > 0.5)
    return {"auc": roc_auc(y, p), "pd_at_pfa": pd, "threshold": thr,
            "rate_mae_bpm": float(np.mean(np.abs(r[m] - rt[m])) * 60) if m.any() else None,
            "n": int(len(y)), "n_pos": int((y > 0.5).sum())}


def classical_baseline(ds: RecordingWindows, idx: list[int], cfg_radar, pfa: float) -> dict:
    """AUC du SNR du détecteur classique sur exactement les mêmes fenêtres."""
    from radar.pipeline import make_analyzer
    an = None
    s, y = [], []
    for i in idx:
        x, fs, lab = ds.raw(i)
        if an is None:
            an = make_analyzer(cfg_radar, fs)
        f, _ = an.analyze(x, want_display=False)
        s.append(f.snr_db)
        y.append(lab)
    s, y = np.asarray(s), np.asarray(y, float)
    pd, _ = pd_at_pfa(y, s, pfa)
    return {"auc": roc_auc(y, s), "pd_at_pfa": pd}


def _nan_to_none(o):
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def loss_fn(logit, rate, y, rt, cfg) -> torch.Tensor:
    ls = cfg["loss"].get("label_smoothing", 0.0)
    y_s = y * (1 - ls) + 0.5 * ls
    l_p = F.binary_cross_entropy_with_logits(logit, y_s)
    m = (rt > 0) & (y > 0.5)
    l_r = F.smooth_l1_loss(rate[m], rt[m], beta=0.05) if m.any() else logit.new_zeros(())
    return l_p + cfg["loss"]["rate_weight"] * l_r


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pretrain", "finetune"))
    ap.add_argument("--config", default=str(_ROOT / "ai" / "config.yaml"))
    ap.add_argument("--init", help="checkpoint de départ (finetune)")
    ap.add_argument("--check", action="store_true",
                    help="construit données + modèle et calcule UNE loss (aucune mise à jour)")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = pick_device(cfg["device"])
    d = cfg["data"]
    fs, win, lam = d["fs"], d["window_s"], d["wavelength"]
    tr = cfg["train"]
    st = cfg[args.stage]

    # ---------------- données
    paths = find_npz([_ROOT / p for p in d["recordings"]])
    real_train = real_val = semi = None
    val_sessions: set = set()
    if paths:
        allrec = RecordingWindows(paths, win, d["hop_s"], fs_expected=fs)
        tr_g, val_sessions = group_split(allrec.groups, d["val_frac"], cfg["seed"])
        idx_tr = [i for i, g in enumerate(allrec.groups) if g in tr_g]
        idx_va = [i for i, g in enumerate(allrec.groups) if g in val_sessions]
        real_train = Subset(allrec, idx_tr) if idx_tr else None
        real_val = Subset(allrec, idx_va) if idx_va else None
        empty_tr = [p for p in paths if _is_empty_train(p, tr_g)]
        if empty_tr:
            semi = SemiSyntheticWindows(empty_tr, win, lam, seed=cfg["seed"])
        print(f"Réel : {len(paths)} enregistrements, {len(allrec)} fenêtres — "
              f"train {len(idx_tr)} / val {len(idx_va)} (sessions val : {sorted(val_sessions)})")
    else:
        print("Aucun enregistrement réel trouvé : pré-entraînement synthétique uniquement.")

    synth = SyntheticWindows(fs, win, lam, seed=cfg["seed"])
    if args.stage == "pretrain":
        train_ds = synth
    else:
        if real_train is None and semi is None:
            raise SystemExit("finetune : il faut des enregistrements réels (voir ai/README.md).")
        train_ds = Mixture({"real": real_train, "semi": semi, "synth": synth}, st["mix"], cfg["seed"])

    nw = 0 if args.check else tr["num_workers"]
    loader = DataLoader(train_ds, batch_size=st["batch_size"], num_workers=nw,
                        pin_memory=dev.type == "cuda", persistent_workers=nw > 0)

    # validation synthétique figée (graine différente)
    vs_n = 64 if args.check else cfg["pretrain"]["val_windows"]
    it = iter(SyntheticWindows(fs, win, lam, seed=cfg["seed"] + 10_000))
    synth_val = [next(it) for _ in range(vs_n)]
    val_loaders = {"synth": DataLoader(synth_val, batch_size=256)}
    if real_val is not None:
        val_loaders["real"] = DataLoader(real_val, batch_size=256, num_workers=0)

    # ---------------- modèle
    model = build_model(dict(cfg["model"])).to(dev)
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Initialisé depuis {args.init}")
    print(f"Modèle {cfg['model']['name']} — {count_params(model) / 1e6:.2f} M paramètres — {dev}")

    if args.check:
        t0 = time.time()
        x, y, r = next(iter(loader))
        t_data = time.time() - t0
        with torch.no_grad():
            logit, rate = model(x.to(dev))
            l = loss_fn(logit, rate, y.to(dev), r.to(dev), cfg)
        print(f"[check] batch x={tuple(x.shape)} y={tuple(y.shape)} (données : {t_data:.1f} s) "
              f"→ logit {tuple(logit.shape)}, loss={float(l):.3f}.  Aucun poids modifié.")
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], weight_decay=tr["weight_decay"])
    total = st["steps"] if args.stage == "pretrain" else st["epochs"] * st["steps_per_epoch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=st["lr"], total_steps=total, pct_start=0.05)
    if tr.get("compile"):
        model = torch.compile(model)

    out = _ROOT / cfg["output"]["dir"] / f"{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    hist_f = open(out / "history.csv", "w", newline="", encoding="utf-8")
    hw = csv.writer(hist_f)
    hw.writerow(["step", "train_loss", "lr", "val_synth_auc", "val_real_auc", "val_real_pd"])

    eval_every = 1000 if args.stage == "pretrain" else st["steps_per_epoch"]
    best, best_state, bad = -1.0, None, 0
    run_loss, step, n_run = 0.0, 0, 0
    t0 = time.time()
    model.train()
    for x, y, r in loader:
        x, y, r = x.to(dev, non_blocking=True), y.to(dev), r.to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=tr["amp"] and dev.type == "cuda"):
            logit, rate = model(x)
        loss = loss_fn(logit.float(), rate.float(), y, r, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += float(loss.detach())
        n_run += 1
        step += 1
        if step % eval_every == 0 or step == total:
            res = {k: metrics(*predict(model, v, dev, tr["amp"]), tr["pfa_target"])
                   for k, v in val_loaders.items()}
            model.train()
            # Sélection du modèle : sur le synthétique en pré-entraînement (le réel
            # y est minuscule) ; sur le réel en affinage, s'il est défini (une
            # validation réelle sans négatif donne une AUC NaN → aucun modèle
            # n'aurait jamais été sauvegardé).
            key = "synth"
            if args.stage == "finetune" and "real" in res and math.isfinite(res["real"]["auc"]):
                key = "real"
            score = res[key]["auc"]
            if not math.isfinite(score):
                score = -0.5
            print(f"step {step:6d}/{total}  loss {run_loss / max(n_run, 1):.4f}  "
                  + "  ".join(f"{k}: AUC {m['auc']:.3f} Pd@{tr['pfa_target']:.0%} {m['pd_at_pfa']:.2f}"
                              for k, m in res.items())
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
            hw.writerow([step, run_loss / max(n_run, 1), sched.get_last_lr()[0],
                         res["synth"]["auc"], res.get("real", {}).get("auc"),
                         res.get("real", {}).get("pd_at_pfa")])
            hist_f.flush()
            run_loss, n_run = 0.0, 0
            if score > best:
                best, bad = score, 0
                best_state = copy.deepcopy(_unwrap(model).state_dict())
                torch.save({"model": best_state, "config": cfg, "step": step, "metrics": res,
                            "selected_on": key,
                            "threshold": res[key]["threshold"],
                            "preprocess": {"fs": fs, "window_s": win}}, out / "best.pt")
            else:
                bad += 1
                if args.stage == "finetune" and bad >= st["patience"]:
                    print("Arrêt anticipé.")
                    break
        if step >= total:
            break
    hist_f.close()

    # ---------------- rapport final (meilleur modèle)
    _unwrap(model).load_state_dict(best_state)
    report = {"stage": args.stage, "best_val_auc": best,
              "final": {k: metrics(*predict(model, v, dev, tr["amp"]), tr["pfa_target"])
                        for k, v in val_loaders.items()}}
    if real_val is not None:
        from radar.config import load_config
        report["classical_baseline_real_val"] = classical_baseline(
            real_val.dataset, list(real_val.indices), load_config(), tr["pfa_target"])
    report = _nan_to_none(report)          # JSON strict (pas de NaN)
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"→ {out}")
    return 0


def _unwrap(m):
    return getattr(m, "_orig_mod", m)


def _is_empty_train(p: Path, train_groups: set) -> bool:
    from radar.recorder import load_recording
    r = load_recording(p)
    return r.label == 0 and r.group in train_groups


if __name__ == "__main__":
    sys.exit(main())
