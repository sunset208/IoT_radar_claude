"""Modèles 1-D sur le slow-time pré-traité (C=3 canaux, T = window_s·fs).

* :class:`ResNet1D` — convolutions dilatées résiduelles (~0.5 M paramètres).
  Champ réceptif > 20 s : voit plusieurs cycles respiratoires.  Défaut.
* :class:`ConvTransformer` — tige convolutive (sous-échantillonnage ×8) puis
  encodeur Transformer (~3 M paramètres).  Pour plus tard, quand il y aura
  beaucoup de données (sinon il sur-apprend).

Deux têtes : présence (logit) et rythme respiratoire (Hz, régression, masquée
quand inconnu).  La tête de rythme sert de tâche auxiliaire : elle force le
réseau à « regarder » la périodicité plutôt que l'énergie.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Res(nn.Module):
    def __init__(self, ch: int, dil: int, k: int = 7, p: float = 0.1) -> None:
        super().__init__()
        pad = dil * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dil), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Dropout(p),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dil), nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class _Heads(nn.Module):
    def __init__(self, ch: int, p: float) -> None:
        super().__init__()
        self.presence = nn.Sequential(nn.Dropout(p), nn.Linear(2 * ch, 64), nn.GELU(), nn.Linear(64, 1))
        self.rate = nn.Sequential(nn.Linear(2 * ch, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, h):                      # h : (B, ch, T')
        z = torch.cat((h.mean(-1), h.amax(-1)), dim=1)
        return self.presence(z).squeeze(-1), self.rate(z).squeeze(-1)


class ResNet1D(nn.Module):
    def __init__(self, in_ch: int = 3, ch: int = 64, blocks: int = 8, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, ch, 9, stride=2, padding=4), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.BatchNorm1d(ch), nn.GELU(),
        )                                           # fs 50 → 12.5 Hz
        self.body = nn.Sequential(*[_Res(ch, 2 ** (i % 5)) for i in range(blocks)])
        self.heads = _Heads(ch, dropout)

    def forward(self, x):
        return self.heads(self.body(self.stem(x)))


class ConvTransformer(nn.Module):
    def __init__(self, in_ch: int = 3, d: int = 192, layers: int = 6, heads: int = 6,
                 dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, d // 2, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(d // 2, d, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
        )                                           # ×8 : 50 Hz → 6.25 Hz
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.heads = _Heads(d, dropout)

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)            # (B, T', d)
        h = self.encoder(h + self.pe[: h.shape[1]])
        return self.heads(h.transpose(1, 2))


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("name", "resnet1d")
    kw = {k: v for k, v in cfg.items() if k != "name"}
    if name == "resnet1d":
        return ResNet1D(**kw)
    if name == "convtransformer":
        return ConvTransformer(**kw)
    raise ValueError(f"modèle inconnu : {name}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
