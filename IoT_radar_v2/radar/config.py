"""Configuration: YAML → dataclasses typées + validation.

Le YAML par défaut (``configs/default.yaml``) est toujours chargé en premier ;
un fichier utilisateur ne contient que les clés à surcharger.
"""

from __future__ import annotations

import copy
import dataclasses as dc
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
SPEED_OF_LIGHT = 299_792_458.0


@dc.dataclass
class SdrCfg:
    uri: str = "ip:192.168.2.1"
    f_c: float = 3.5e9
    f_s: float = 1.0e6
    rf_bandwidth: float = 1.0e6
    rx_gain_db: float = 40.0
    tx_atten_db: float = -20.0
    rx_buffer_size: int = 100_000
    kernel_buffers: int = 16
    disable_tracking: bool = True

    @property
    def wavelength(self) -> float:
        return SPEED_OF_LIGHT / self.f_c


@dc.dataclass
class EmissionCfg:
    f_offset: float = 10_000.0
    amplitude: float = 0.5


@dc.dataclass
class FrontendCfg:
    fs_slow: float = 50.0
    cic_order: int = 2


@dc.dataclass
class AnalysisCfg:
    window_s: float = 20.0
    hop_s: float = 0.5
    min_window_s: float = 10.0
    breath_band: tuple[float, float] = (0.1, 0.8)
    heart_band: tuple[float, float] = (0.8, 2.5)
    noise_band: tuple[float, float] = (4.0, 12.0)
    nfft: int = 8192


@dc.dataclass
class DetectorCfg:
    snr_threshold_db: float = 10.0
    periodicity_threshold: float = 0.35
    motion_threshold: float = 6.0
    on_count: int = 5
    off_count: int = 8


@dc.dataclass
class SimulationCfg:
    scenario: str = "breathing"
    realtime: bool = True
    seed: int | None = None
    distance_m: float = 2.0
    snr_db: float = 25.0
    leak_dbfs: float = -12.0
    rx_dc_dbfs: float = -30.0
    breath_rate_bpm: float = 16.0
    breath_depth_mm: float = 6.0
    heart_rate_bpm: float = 70.0
    heart_depth_mm: float = 0.25


@dc.dataclass
class RecordingCfg:
    dir: str = "data/recordings"
    save_raw: bool = False


@dc.dataclass
class WebCfg:
    host: str = "127.0.0.1"
    port: int = 8050
    push_hz: float = 10.0


@dc.dataclass
class LoggingCfg:
    level: str = "INFO"
    dir: str = "logs"


@dc.dataclass
class Config:
    sdr: SdrCfg = dc.field(default_factory=SdrCfg)
    emission: EmissionCfg = dc.field(default_factory=EmissionCfg)
    frontend: FrontendCfg = dc.field(default_factory=FrontendCfg)
    analysis: AnalysisCfg = dc.field(default_factory=AnalysisCfg)
    detector: DetectorCfg = dc.field(default_factory=DetectorCfg)
    simulation: SimulationCfg = dc.field(default_factory=SimulationCfg)
    recording: RecordingCfg = dc.field(default_factory=RecordingCfg)
    web: WebCfg = dc.field(default_factory=WebCfg)
    logging: LoggingCfg = dc.field(default_factory=LoggingCfg)

    # ------------------------------------------------------------------
    @property
    def cic_factor(self) -> int:
        return int(round(self.sdr.f_s / self.emission.f_offset))

    @property
    def total_decimation(self) -> int:
        return int(round(self.sdr.f_s / self.frontend.fs_slow))

    def validate(self) -> None:
        s, e, f, a = self.sdr, self.emission, self.frontend, self.analysis
        if s.f_s < 521e3:
            raise ValueError("sdr.f_s doit être ≥ 521 kS/s (limite pyadi-iio).")
        r = s.f_s / e.f_offset
        if abs(r - round(r)) > 1e-9 or round(r) < 4:
            raise ValueError(
                f"f_s/f_offset = {r:g} doit être un entier ≥ 4 (zéros du CIC "
                "alignés sur les multiples de f_offset)."
            )
        d = s.f_s / f.fs_slow
        if abs(d - round(d)) > 1e-9 or round(d) % round(r) != 0:
            raise ValueError(
                f"f_s/fs_slow = {d:g} doit être un entier multiple de "
                f"f_s/f_offset = {round(r)}."
            )
        if s.rx_buffer_size % round(r) != 0:
            raise ValueError(
                f"rx_buffer_size ({s.rx_buffer_size}) doit être un multiple de "
                f"f_s/f_offset = {round(r)} (continuité du NCO)."
            )
        nyq = f.fs_slow / 2
        for name, (lo, hi) in (
            ("breath_band", a.breath_band),
            ("heart_band", a.heart_band),
            ("noise_band", a.noise_band),
        ):
            if not 0 < lo < hi < nyq:
                raise ValueError(f"analysis.{name}={lo, hi} hors ]0, fs_slow/2[.")
        if a.hop_s <= 0 or a.window_s < a.min_window_s:
            raise ValueError("analysis: hop_s > 0 et window_s ≥ min_window_s requis.")
        if not 0 < e.amplitude <= 1:
            raise ValueError("emission.amplitude doit être dans ]0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return dc.asdict(self)


# ----------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls, data: dict[str, Any]):
    fields = {f.name: f for f in dc.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Clés inconnues pour {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, val in data.items():
        ftype = fields[name].type
        sub = _SECTIONS.get(name) if cls is Config else None
        if sub is not None:
            kwargs[name] = _build(sub, val or {})
        elif isinstance(val, list):
            kwargs[name] = tuple(float(x) for x in val)
        elif isinstance(ftype, str) and ftype == "float" and isinstance(val, (int, str)):
            kwargs[name] = float(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


_SECTIONS = {
    "sdr": SdrCfg,
    "emission": EmissionCfg,
    "frontend": FrontendCfg,
    "analysis": AnalysisCfg,
    "detector": DetectorCfg,
    "simulation": SimulationCfg,
    "recording": RecordingCfg,
    "web": WebCfg,
    "logging": LoggingCfg,
}


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Charge ``default.yaml`` puis applique *path* et *overrides* (dicts imbriqués)."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    cfg = _build(Config, data)
    cfg.validate()
    return cfg


def config_from_dict(data: dict[str, Any]) -> Config:
    """Reconstruit une Config depuis ``Config.to_dict()`` (métadonnées d'enregistrement)."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    cfg = _build(Config, _deep_merge(base, data))
    cfg.validate()
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Chemin relatif → relatif à la racine du projet."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
