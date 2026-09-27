"""Inférence temps réel : branché dans ``radar run --ai ai/runs/.../best.pt``.

Le score IA est affiché *à côté* du détecteur classique (courbe orange dans
l'historique) ; il ne pilote pas l'état tant qu'il n'a pas été validé sur des
sessions réelles jamais vues (voir ai/README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.models import build_model  # noqa: E402
from ai.preprocess import preprocess  # noqa: E402


class TorchDetector:
    def __init__(self, ckpt_path: str, device: str | None = None) -> None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        self.model = build_model(dict(ck["config"]["model"])).to(dev).eval()
        self.model.load_state_dict(ck["model"])
        self.fs = float(ck["preprocess"]["fs"])
        self.n = int(round(ck["preprocess"]["window_s"] * self.fs))
        self.threshold = float(ck.get("threshold") or 0.5)
        self.device = dev

    @torch.no_grad()
    def predict(self, x: np.ndarray, fs: float) -> float:
        if abs(fs - self.fs) > 1e-6:
            raise ValueError(f"fs={fs} ≠ fs du modèle ({self.fs})")
        if len(x) < self.n:
            return float("nan")
        t = torch.from_numpy(preprocess(x[-self.n:], fs)).unsqueeze(0).to(self.device)
        logit, _ = self.model(t)
        return float(torch.sigmoid(logit)[0])
