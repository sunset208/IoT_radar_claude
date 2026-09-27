"""Jeux de données pour l'IA.

Trois sources, combinables :

1. :class:`SyntheticWindows` — génération à la volée par le simulateur physique
   (``radar.scene``) avec **randomisation de domaine** (SNR, fuite, dérive LO,
   rythme, amplitude, confondeurs : marcheur, ventilateur, mouvements, apnées).
   Infini, sert au pré-entraînement.
2. :class:`RecordingWindows` — fenêtres d'enregistrements réels v2 (``.npz``),
   étiquette de l'enregistrement, groupe = session (splits sans fuite).
3. :class:`SemiSyntheticWindows` — **fond réel + cible simulée** : on ajoute à
   un enregistrement « salle vide » réel l'écho d'une personne simulée.  Les
   positifs héritent ainsi du vrai clutter, du vrai bruit de phase du Pluto et
   des vrais parasites de la pièce.  C'est probablement le levier n°1 tant
   qu'on a peu d'enregistrements avec personne.

Toutes renvoient ``(x[C, T] float32, y_presence float32, y_rate_hz float32)``
avec ``y_rate_hz = 0`` si inconnu/absent (masqué dans la loss).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.preprocess import preprocess  # noqa: E402
from radar.recorder import list_recordings, load_recording  # noqa: E402
from radar.scene import (  # noqa: E402
    BREATHING_SCENARIOS, SCENARIOS, Breathing, Heartbeat, Scene, dbfs_to_amp, random_scene_params,
)
from radar.sources.simulation import SimulatedSlowSource  # noqa: E402


def _pick_lam(lam, rng) -> float:
    """λ fixe ou tirée dans une liste (randomisation de domaine sur la porteuse)."""
    if isinstance(lam, (list, tuple)):
        return float(lam[int(rng.integers(len(lam)))])
    return float(lam)


def recording_wavelength(rec, default: float) -> float:
    """λ réelle d'un enregistrement (porteuse sdr.f_c des métadonnées)."""
    try:
        return 299_792_458.0 / float(rec.meta["config"]["sdr"]["f_c"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return default


def _item(x: np.ndarray, fs: float, y: float, rate_hz: float | None):
    return (torch.from_numpy(preprocess(x, fs)), torch.tensor(float(y)),
            torch.tensor(float(rate_hz or 0.0)))


# ----------------------------------------------------------------------
class SyntheticWindows(IterableDataset):
    def __init__(self, fs: float, window_s: float, wavelength, seed: int = 0,
                 scenario_weights: dict[str, float] | None = None, sim_len_s: float = 90.0) -> None:
        self.fs, self.window_s, self.lam, self.seed = fs, window_s, wavelength, seed
        w = scenario_weights or {s: 1.0 for s in SCENARIOS}
        self.scen = list(w)
        p = np.asarray([w[s] for s in self.scen], float)
        self.p = p / p.sum()
        self.sim_len_s = sim_len_s

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 7919 * wid + int(torch.initial_seed() % 100000))
        n_win = int(round(self.window_s * self.fs))
        while True:
            sc = str(rng.choice(self.scen, p=self.p))
            params = random_scene_params(rng, _pick_lam(self.lam, rng), sc)
            scene = Scene(params, rng)
            src = SimulatedSlowSource(self.fs, scene, rng=rng)
            src.generate(int(self.fs))                     # préchauffe
            x = src.generate(int(self.sim_len_s * self.fs), int(self.fs))
            # quelques fenêtres par simulation (coût amorti)
            for _ in range(4):
                a = int(rng.integers(0, len(x) - n_win))
                seg = x[a:a + n_win]
                y = float(sc in BREATHING_SCENARIOS)
                if sc == "apnea":
                    y = float(rng.random() < 0.8)   # étiquette bruitée : la pause peut tomber dans la fenêtre
                rate = params.breath_rate_bpm / 60.0 if y else None
                yield _item(seg, self.fs, y, rate)


# ----------------------------------------------------------------------
def _segments(rec) -> list[np.ndarray]:
    cuts = [0] + sorted(int(g) for g in rec.gaps if 0 < g < len(rec.slow)) + [len(rec.slow)]
    return [rec.slow[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


class RecordingWindows(Dataset):
    def __init__(self, paths: list[Path], window_s: float, hop_s: float,
                 labels: tuple[int, ...] = (0, 1), fs_expected: float | None = None) -> None:
        self.items: list[tuple[int, int, int]] = []   # (rec, seg, start)
        self.recs = []
        self.groups: list[str] = []
        self.labels: list[int] = []
        self.window_s = window_s
        for p in paths:
            try:
                rec = load_recording(p)
            except Exception:           # enregistrement SFCW ou fichier étranger : ignoré
                continue
            if rec.label not in labels:
                continue
            if fs_expected and abs(rec.fs_slow - fs_expected) > 1e-6:
                raise ValueError(f"{p.name}: fs_slow={rec.fs_slow} ≠ {fs_expected}")
            ri = len(self.recs)
            segs = _segments(rec)
            self.recs.append((rec, segs))
            n = int(round(window_s * rec.fs_slow))
            h = max(1, int(round(hop_s * rec.fs_slow)))
            for si, s in enumerate(segs):
                for a in range(0, len(s) - n + 1, h):
                    self.items.append((ri, si, a))
                    self.groups.append(rec.group)
                    self.labels.append(rec.label)

    def __len__(self) -> int:
        return len(self.items)

    def raw(self, i: int) -> tuple[np.ndarray, float, int]:
        ri, si, a = self.items[i]
        rec, segs = self.recs[ri]
        n = int(round(self.window_s * rec.fs_slow))
        return segs[si][a:a + n], rec.fs_slow, rec.label

    def rate_bpm(self, ri: int) -> float | None:
        """Vérité terrain du rythme : simulation, sinon module 60 GHz enregistré à côté."""
        rec = self.recs[ri][0]
        rate = (rec.meta.get("truth") or {}).get("breath_rate_bpm")
        if rate is None and rec.path is not None:
            from radar.offline import reference_bpm
            rate = reference_bpm(rec.path)
        return rate

    def __getitem__(self, i: int):
        x, fs, y = self.raw(i)
        rate = self.rate_bpm(self.items[i][0])
        return _item(x, fs, y, rate / 60.0 if (rate and y) else None)


# ----------------------------------------------------------------------
class SemiSyntheticWindows(IterableDataset):
    """Fonds réels (label 0) + écho humain simulé (positifs) ou rien (négatifs)."""

    def __init__(self, empty_paths: list[Path], window_s: float, wavelength,
                 p_positive: float = 0.5, seed: int = 0) -> None:
        self.bg = RecordingWindows(empty_paths, window_s, hop_s=1.0, labels=(0,))
        if len(self.bg) == 0:
            raise ValueError("Aucune fenêtre « salle vide » disponible pour le semi-synthétique.")
        self.window_s, self.lam, self.p_pos, self.seed = window_s, wavelength, p_positive, seed

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng(self.seed + 104729 * (info.id if info else 0)
                                    + int(torch.initial_seed() % 100000))
        while True:
            i = int(rng.integers(len(self.bg)))
            x, fs, _ = self.bg.raw(i)
            x = x.astype(np.complex128)
            # la cible simulée doit l'être à la porteuse du fond réel (1.8 GHz le 27/09)
            rec = self.bg.recs[self.bg.items[i][0]][0]
            k = 4 * math.pi / recording_wavelength(rec, _pick_lam(self.lam, rng))
            if rng.random() < self.p_pos:
                t = np.arange(len(x)) / fs
                rate = float(rng.uniform(7, 40))
                br = Breathing(rng, rate, float(rng.uniform(0.5, 12)) * 1e-3)
                hb = Heartbeat(rng, float(rng.uniform(50, 120)), float(rng.uniform(0.05, 0.5)) * 1e-3)
                d = br(t) + hb(t)
                # niveau de la cible relatif au bruit réel de la fenêtre (SNR aléatoire)
                noise = np.std(np.diff(x)) / math.sqrt(2) + 1e-12
                amp = noise * 10 ** (rng.uniform(-8, 25) / 20)
                x = x + amp * np.exp(1j * (rng.uniform(0, 2 * np.pi) - k * d))
                yield _item(x, fs, 1.0, rate / 60.0)
            else:
                yield _item(x, fs, 0.0, None)


def find_npz(roots: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        r = Path(r)
        out += list_recordings(r) if r.is_dir() else ([r] if r.suffix == ".npz" else [])
    return out


def group_split(groups: list[str], val_frac: float, seed: int) -> tuple[set, set]:
    """Répartit les *sessions* (pas les fenêtres) entre train et val."""
    ug = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    n_val = max(1, int(round(val_frac * len(ug)))) if len(ug) > 1 else 0
    return set(ug[n_val:]), set(ug[:n_val])
