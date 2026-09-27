"""Calibration dataset for the supervised autoencoder.

Source of truth for recordings
------------------------------
``.npz`` files are produced by
``MicroDopplerDetection/utils/record_acquisition.py`` (and its launcher
``auto_record.py``). Typical layout ::

    AICalibration/data/
    ├── train/<n>.npz
    ├── test/<n>.npz
    └── val/<n>.npz

Each ``.npz`` contains at minimum:

    - ``spectrogram_db`` : float64, shape ``(N_frames, n_fft)``
                            — successive STFT columns (in dB)
    - ``label``          : scalar ``int8`` (0 = empty, 1 = breathing)
    - ``env``            : str        (environment tag)
    - ``n_fft``          : int        (frequency width)
    - ``f_s_dec_hz``     : float      (Hz)
    - metadata: ``subset``, ``sample_index``, ``config_path``,
      ``data_root``, ``utc_finished``, ``n_trame``, ``t_wall_s``, etc.

CalibrationDataset
------------------
PyTorch ``Dataset`` that:

1. Recursively loads all ``.npz`` files under ``data_dir``.
2. Splits each recording into sliding windows of ``N_COLS`` STFT columns
   (along the time axis).
3. Optionally normalises each window (zero-mean / unit-variance).
4. Returns ``(tensor[1, n_fft, N_COLS], label)`` — PyTorch convention
   ``(C, H, W)`` with ``H = n_fft`` (frequency) and ``W = N_COLS`` (time).

The ``(1, n_fft, N_COLS)`` format is designed for a **Conv2D** autoencoder
(encoder + decoder), which exploits spectro-temporal locality.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_COLS: int = 32


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CalibrationDataset(Dataset):
    """Sliding-window STFT dataset for autoencoder training.

    Parameters
    ----------
    data_dir : str or Path
        Directory scanned recursively (``rglob("*.npz")``). Point to
        ``AICalibration/data`` to load everything, or to
        ``AICalibration/data/train`` for a single split.
    n_cols : int, optional
        Number of STFT columns per window. Default :data:`N_COLS`.
    stride : int, optional
        Step between successive windows extracted from the same recording.
        Defaults to ``n_cols`` (non-overlapping windows). A smaller
        ``stride`` increases the number of examples at the cost of strong
        correlation between neighbouring windows.
    normalise : bool, optional
        If ``True`` (default), each window is zero-mean and unit-variance
        (over the full ``n_fft × n_cols`` array). This removes dependence
        on absolute power level (distance, RX gain).

    Notes
    -----
    All recordings under ``data_dir`` must share the same ``n_fft``; a
    ``ValueError`` is raised otherwise.

    The dataset is fully loaded into RAM at construction time. For 2-minute
    recordings with ``n_fft=8192`` (~16 MB per file), a few dozen takes
    fit comfortably.
    """

    def __init__(
        self,
        data_dir: str | Path,
        n_cols: int = N_COLS,
        stride: int | None = None,
        normalise: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_cols = int(n_cols)
        self.stride = int(stride) if stride is not None else self.n_cols
        self.normalise = bool(normalise)

        if self.n_cols <= 0:
            raise ValueError("n_cols doit être > 0.")
        if self.stride <= 0:
            raise ValueError("stride doit être > 0.")

        self._windows: list[np.ndarray] = []   # (n_fft, n_cols) float32
        self._labels: list[int] = []

        self._load_all()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Répertoire de données introuvable : {self.data_dir}"
            )

        npz_files = sorted(self.data_dir.rglob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"Aucun fichier .npz trouvé sous {self.data_dir}. "
                "Lancer d'abord MicroDopplerDetection/utils/record_acquisition.py."
            )

        n_fft_ref: int | None = None
        total_windows = 0

        for path in npz_files:
            with np.load(path, allow_pickle=False) as data:
                if "spectrogram_db" not in data.files:
                    logger.warning(
                        "%s ignoré — pas de clé 'spectrogram_db' "
                        "(enregistrement fait avec --no-spectrogram ?).",
                        path.name,
                    )
                    continue
                if "label" not in data.files:
                    logger.warning("%s ignoré — pas de clé 'label'.", path.name)
                    continue

                spec = np.asarray(data["spectrogram_db"], dtype=np.float32)
                label = int(np.asarray(data["label"]).item())
                env = (
                    str(np.asarray(data["env"]).item())
                    if "env" in data.files else "?"
                )

            if spec.ndim != 2:
                logger.warning(
                    "%s ignoré — spectrogram_db de dimension %d (attendu 2).",
                    path.name, spec.ndim,
                )
                continue

            spec_t = spec.T  # (n_fft, N_frames)
            n_fft_file, n_total = spec_t.shape

            if n_fft_ref is None:
                n_fft_ref = n_fft_file
            elif n_fft_file != n_fft_ref:
                raise ValueError(
                    f"n_fft incohérent dans {path.name} : "
                    f"attendu {n_fft_ref}, trouvé {n_fft_file}."
                )

            if n_total < self.n_cols:
                logger.warning(
                    "%s ignoré — trop court (%d colonnes < n_cols=%d)",
                    path.name, n_total, self.n_cols,
                )
                continue

            windows_in_file = 0
            for start in range(0, n_total - self.n_cols + 1, self.stride):
                window = spec_t[:, start: start + self.n_cols].copy()  # (n_fft, n_cols)
                self._windows.append(window)
                self._labels.append(label)
                windows_in_file += 1

            total_windows += windows_in_file
            logger.info(
                "Chargé %s — env='%s', label=%d, %d colonnes → %d fenêtres",
                path.name, env, label, n_total, windows_in_file,
            )

        if not self._windows:
            raise RuntimeError(
                "Aucune fenêtre extraite. Vérifier la durée des enregistrements "
                f"(n_cols={self.n_cols}, stride={self.stride})."
            )

        n1 = sum(1 for lbl in self._labels if lbl == 1)
        n0 = len(self._labels) - n1
        logger.info(
            "Dataset prêt — %d fenêtres total (label=0: %d, label=1: %d, "
            "ratio 1/0=%.2f)",
            total_windows, n0, n1, n1 / max(n0, 1),
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a normalised spectrogram window and its label.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(x, y)`` with ``x`` of shape ``(1, n_fft, n_cols)`` (float32)
            and scalar ``y`` as ``int64`` (0 or 1).
        """
        window = self._windows[idx].astype(np.float32, copy=True)  # (n_fft, n_cols)

        if self.normalise:
            mu = float(window.mean())
            sigma = float(window.std())
            if sigma > 1e-6:
                window = (window - mu) / sigma
            else:
                window = window - mu

        x = torch.from_numpy(window).unsqueeze(0)              # (1, n_fft, n_cols)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def n_fft(self) -> int:
        """Common frequency dimension (``H``) across all windows."""
        return self._windows[0].shape[0]

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights ``[w0, w1]`` for ``CrossEntropyLoss``.

        Pass to ``torch.nn.CrossEntropyLoss(weight=dataset.class_weights)``
        when classes are imbalanced.
        """
        n_total = len(self._labels)
        n1 = sum(self._labels)
        n0 = n_total - n1
        w0 = n_total / (2.0 * max(n0, 1))
        w1 = n_total / (2.0 * max(n1, 1))
        return torch.tensor([w0, w1], dtype=torch.float32)

    def summary(self) -> str:
        """Human-readable dataset summary."""
        n1 = sum(self._labels)
        n0 = len(self._labels) - n1
        return (
            f"CalibrationDataset — {len(self)} fenêtres "
            f"(n_fft={self.n_fft}, n_cols={self.n_cols}, stride={self.stride})\n"
            f"  label=0 (vide)        : {n0}\n"
            f"  label=1 (respiration) : {n1}\n"
            f"  ratio 1/0             : {n1 / max(n0, 1):.2f}\n"
            f"  data_dir              : {self.data_dir}"
        )
