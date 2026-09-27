"""Point d'entrée autonome : ``python radar_cli.py <commande>`` (sans PYTHONPATH)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from radar.sources.pluto import ensure_libiio_on_path  # noqa: E402

ensure_libiio_on_path()

from radar.__main__ import main  # noqa: E402

sys.exit(main())
