"""Conv2D autoencoder for micro-Doppler spectrograms with a classification head.

The architecture is **fully parameterized** by the constructor (and thus
by ``AICalibration/config.yaml``). Hyperparameters such as channels,
downsampling factors, kernels, and input sizes must be supplied as
constructor arguments.

Input shape (default, aligned with the dataset)
-----------------------------------------------
``(B, 1, n_fft, n_cols)``   e.g.   ``(B, 1, 8192, 32)``
              ↑       ↑
    frequency (H)   time (W)

Forward outputs
---------------
- ``x_hat``  : reconstruction, **same shape** as the input.
- ``logits`` : ``(B, 1)`` — logits *before* sigmoid, to use with
  ``nn.BCEWithLogitsLoss`` (label 1 = presence / breathing).

Latent geometry
---------------
For default values (8192×32 input, 4 downsampling stages of ``(4, 2)``)
→ latent ``(128, 32, 2)``. Compression ratio 32×.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "SpectrogramAutoencoder",
    "Encoder",
    "Decoder",
    "ClassifierHead",
]

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _padding_same(kernel: Sequence[int]) -> Tuple[int, int]:
    """Same padding for a 2D kernel (odd k)."""
    kh, kw = int(kernel[0]), int(kernel[1])
    return (kh // 2, kw // 2)


class _ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + ReLU ("same" padding)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Sequence[int],
    ) -> None:
        super().__init__()
        kh, kw = int(kernel[0]), int(kernel[1])
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kh, kw),
                padding=_padding_same((kh, kw)),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    """Strided conv (``kernel = stride`` → exact division) + 3×3 refine."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: Sequence[int],
        refine_kernel: Sequence[int],
    ) -> None:
        super().__init__()
        kh, kw = int(factor[0]), int(factor[1])
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=(kh, kw),
                stride=(kh, kw),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = _ConvBNAct(out_channels, out_channels, refine_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class _UpBlock(nn.Module):
    """ConvTranspose2d (``kernel = stride`` → exact upsampling) + 3×3 refine."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: Sequence[int],
        refine_kernel: Sequence[int],
    ) -> None:
        super().__init__()
        kh, kw = int(factor[0]), int(factor[1])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=(kh, kw),
                stride=(kh, kw),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = _ConvBNAct(out_channels, out_channels, refine_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Conv2D encoder: ``(B, in_c, H, W)`` → ``(B, latent_c, H', W')``."""

    def __init__(
        self,
        channels: Sequence[int],
        downsample_factors: Sequence[Sequence[int]],
        in_conv_kernel: Sequence[int],
        refine_kernel: Sequence[int],
    ) -> None:
        super().__init__()
        if len(channels) != len(downsample_factors) + 2:
            raise ValueError(
                "channels doit avoir (1 + 1 + n_down) entrées : "
                f"reçu {len(channels)}, attendu {len(downsample_factors) + 2}."
            )

        in_c, first_c = int(channels[0]), int(channels[1])
        self.in_conv = _ConvBNAct(in_c, first_c, kernel=in_conv_kernel)

        self.down_blocks = nn.ModuleList()
        prev = first_c
        for stage_idx, factor in enumerate(downsample_factors):
            out_c = int(channels[2 + stage_idx])
            self.down_blocks.append(_DownBlock(prev, out_c, factor, refine_kernel))
            prev = out_c

        self.out_channels: int = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for block in self.down_blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Conv2D decoder, symmetric to the Encoder."""

    def __init__(
        self,
        channels: Sequence[int],
        downsample_factors: Sequence[Sequence[int]],
        refine_kernel: Sequence[int],
        out_conv_kernel: Sequence[int],
    ) -> None:
        super().__init__()
        if len(channels) != len(downsample_factors) + 2:
            raise ValueError(
                "channels doit avoir la même longueur que pour l'Encoder."
            )

        up_channels = [int(c) for c in channels[1:]]
        rev_factors = list(reversed(list(downsample_factors)))
        rev_channels = list(reversed(up_channels))

        self.up_blocks = nn.ModuleList()
        prev = rev_channels[0]
        for stage_idx, factor in enumerate(rev_factors):
            out_c = rev_channels[stage_idx + 1]
            self.up_blocks.append(_UpBlock(prev, out_c, factor, refine_kernel))
            prev = out_c

        in_c = int(channels[0])
        kh, kw = int(out_conv_kernel[0]), int(out_conv_kernel[1])
        self.out_conv = nn.Conv2d(
            prev, in_c,
            kernel_size=(kh, kw),
            padding=_padding_same((kh, kw)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for block in self.up_blocks:
            z = block(z)
        return self.out_conv(z)


# ---------------------------------------------------------------------------
# Classifier head (sur le latent global-poolé)
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """Lightweight MLP: latent → binary logit (before sigmoid).

    Pipeline: ``AdaptiveAvgPool2d(1) → Flatten → Dropout → Linear →
    ReLU → Dropout → Linear(1)``.
    """

    def __init__(self, in_channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(in_channels), int(hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class SpectrogramAutoencoder(nn.Module):
    """Conv2D autoencoder + binary classification head.

    All parameters come from the constructor (and thus from ``config.yaml``
    via :meth:`from_config`). ``forward`` returns a **tuple**:
    ``(x_hat, logits)``.
    """

    def __init__(
        self,
        expected_input_shape: Sequence[int] = (1, 8192, 32),
        encoder_channels: Sequence[int] = (1, 16, 32, 64, 128, 128),
        downsample_factors: Sequence[Sequence[int]] = ((4, 2), (4, 2), (4, 2), (4, 2)),
        in_conv_kernel: Sequence[int] = (7, 3),
        refine_kernel: Sequence[int] = (3, 3),
        out_conv_kernel: Sequence[int] = (7, 3),
        classifier_hidden: int = 64,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        expected_input_shape = tuple(int(v) for v in expected_input_shape)
        if len(expected_input_shape) != 3:
            raise ValueError(
                "expected_input_shape doit être (C, H, W). "
                f"Reçu {expected_input_shape}."
            )
        if int(encoder_channels[0]) != expected_input_shape[0]:
            raise ValueError(
                "encoder_channels[0] doit valoir C = expected_input_shape[0] "
                f"({expected_input_shape[0]}) ; reçu {encoder_channels[0]}."
            )

        self.expected_input_shape: Tuple[int, int, int] = expected_input_shape

        # Encoder + Decoder (totalement paramétrés)
        self.encoder = Encoder(
            channels=encoder_channels,
            downsample_factors=downsample_factors,
            in_conv_kernel=in_conv_kernel,
            refine_kernel=refine_kernel,
        )
        self.decoder = Decoder(
            channels=encoder_channels,
            downsample_factors=downsample_factors,
            refine_kernel=refine_kernel,
            out_conv_kernel=out_conv_kernel,
        )

        # Calcule la forme latente à partir des facteurs cumulés.
        _, h, w = expected_input_shape
        for kh, kw in downsample_factors:
            kh_i, kw_i = int(kh), int(kw)
            if h % kh_i != 0 or w % kw_i != 0:
                raise ValueError(
                    f"Géométrie incohérente : (H={h}, W={w}) non divisible "
                    f"par ({kh_i}, {kw_i})."
                )
            h //= kh_i
            w //= kw_i
        self.latent_shape: Tuple[int, int, int] = (self.encoder.out_channels, h, w)

        # Tête de classification.
        self.head = ClassifierHead(
            in_channels=self.encoder.out_channels,
            hidden=classifier_hidden,
            dropout=classifier_dropout,
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SpectrogramAutoencoder":
        """Build the model from the config dict (``model`` key)."""
        m = cfg["model"]
        cls_cfg = m["classifier"]
        return cls(
            expected_input_shape=tuple(m["expected_input_shape"]),
            encoder_channels=tuple(m["encoder_channels"]),
            downsample_factors=tuple(tuple(f) for f in m["downsample_factors"]),
            in_conv_kernel=tuple(m["in_conv_kernel"]),
            refine_kernel=tuple(m["refine_kernel"]),
            out_conv_kernel=tuple(m["out_conv_kernel"]),
            classifier_hidden=int(cls_cfg["hidden"]),
            classifier_dropout=float(cls_cfg["dropout"]),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(
                f"Entrée 4D attendue (B, C, H, W), reçu dim={x.dim()} "
                f"shape={tuple(x.shape)}."
            )
        if tuple(x.shape[1:]) != self.expected_input_shape:
            raise ValueError(
                f"Forme par échantillon attendue {self.expected_input_shape}, "
                f"reçu {tuple(x.shape[1:])}."
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits ``(B, 1)`` from a latent tensor."""
        return self.head(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        logits = self.head(z)
        return x_hat, logits


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Smoke-test forward pass and shape consistency (no external config)."""
    model = SpectrogramAutoencoder()
    batch_size = 2
    x = torch.randn(batch_size, *model.expected_input_shape)

    x_hat, logits = model(x)
    z = model.encode(x)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Entrée x       : {tuple(x.shape)}")
    print(f"Latent z       : {tuple(z.shape)}     "
          f"(attendu (B, {model.latent_shape[0]}, "
          f"{model.latent_shape[1]}, {model.latent_shape[2]}))")
    print(f"Reconstruite   : {tuple(x_hat.shape)}")
    print(f"Logits         : {tuple(logits.shape)}  (attendu (B, 1))")
    print(f"Paramètres     : {n_params:,}  (entraînables : {n_train:,})")

    assert x_hat.shape == x.shape, "Forme de sortie != entrée."
    assert tuple(z.shape[1:]) == model.latent_shape, "Forme du latent inattendue."
    assert tuple(logits.shape) == (batch_size, 1), "Forme des logits inattendue."
    print("OK — toutes les formes sont cohérentes.")


if __name__ == "__main__":
    _smoke_test()
