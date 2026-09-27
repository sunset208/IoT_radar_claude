#!/usr/bin/env python3
"""Chain several ``record_acquisition`` captures with the same configuration.

Example — 5 train samples, salle, label 1, 60 s each, 2 min pause in between ::

    cd MicroDopplerDetection
    python utils/auto_record.py -n 5 --interval 120 \\
        --subset train --env salle --label 1 --duration 60

``--interval`` : seconds to wait **after a recording finishes** before starting
the next one (0 by default).

Sample indices are always automatic ("next free"): do not pass ``--index``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent
_ROOT = _UTILS_DIR.parent
_REPO_ROOT = _ROOT.parent


def _ensure_paths() -> None:
    """Make both ``utils.*`` (via the package dir) and ``MicroDopplerDetection.*``
    (via the repo root, required by ``main.py``) importable, regardless of the
    directory the script is launched from."""
    for p in (_REPO_ROOT, _ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def main() -> None:
    """Parse CLI arguments and run ``record_acquisition`` ``--samples`` times."""
    _ensure_paths()
    from utils.record_acquisition import build_record_argument_parser, run as record_run

    p = build_record_argument_parser(
        description=(
            "Enchaîner N enregistrements micro-Doppler (mêmes options que "
            "record_acquisition, plus -n / --interval)."
        ),
    )
    g = p.add_argument_group("enchaînement")
    g.add_argument(
        "--samples",
        "-n",
        type=int,
        required=True,
        metavar="N",
        help="Nombre d’échantillons à enregistrer successivement.",
    )
    g.add_argument(
        "--interval",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "Pause en secondes après la fin d’un .npz avant le suivant "
            "(défaut : 0)."
        ),
    )

    args = p.parse_args()
    if args.samples < 1:
        raise SystemExit("-n / --samples doit être >= 1.")
    if args.interval < 0:
        raise SystemExit("--interval doit être >= 0.")
    if args.index is not None:
        raise SystemExit(
            "--index est incompatible avec auto_record (indices auto : prochain libre).",
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    paths: list[Path] = []
    for k in range(args.samples):
        if k > 0 and args.interval > 0:
            log.info("Pause %.1f s avant l’échantillon %d / %d", args.interval, k + 1, args.samples)
            time.sleep(args.interval)
        log.info(
            "Début échantillon %d / %d (subset=%s env=%s label=%s)",
            k + 1,
            args.samples,
            args.subset,
            args.env,
            args.label,
        )
        paths.append(record_run(args))

    log.info("Terminé — %d fichier(s) .npz : %s", len(paths), ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
