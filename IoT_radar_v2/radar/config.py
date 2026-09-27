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

# Plages de LO (Hz).  Le Pluto Rev.C est livré en AD9363 (325 MHz–3.8 GHz,
# 20 MHz de bande) ; il se déverrouille en AD9364 (70 MHz–6 GHz, 56 MHz) :
# voir README, « Passer à 5.8 GHz ».
CHIP_LO_RANGE = {
    "ad9363": (325e6, 3.8e9),
    "ad9364": (70e6, 6.0e9),
}


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
    chip: str = "ad9363"               # ad9363 (d'origine) | ad9364 (déverrouillé)

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
    min_window_s: float = 8.0          # obsolète (gardé pour relire d'anciens enregistrements)
    extra_windows_s: tuple = (8.0, 12.0)   # échelles courtes du détecteur multi-échelle
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
    # seuils des échelles courtes (même ordre que analysis.extra_windows_s)
    scale_snr_threshold_db: tuple = (18.0, 15.5)
    scale_periodicity_threshold: tuple = (0.35, 0.35)
    # une échelle courte n'est positive que si le rythme est stable (Hz) sur
    # les on_count dernières fenêtres : une raie de bruit, elle, se promène
    rate_consistency_hz: float = 0.04


@dc.dataclass
class FastCfg:
    """Indices rapides 10 Hz (voir radar/dsp/fast.py)."""
    enabled: bool = True
    activity_s: float = 1.0
    presence_s: float = 4.0
    presence_band: tuple = (0.12, 1.0)
    activity_band: tuple = (1.5, 15.0)
    clutter_amp_dbc: float = -84.7      # (calib) bruit d'amplitude du clutter, bande de présence
    clutter_phase_dbc: float = -65.3    # (calib) bruit de phase LO du clutter, bande de présence
    presence_threshold_db: float = 6.0  # (calib)
    presence_on_s: float = 1.5
    presence_off_s: float = 3.0
    activity_threshold_db: float = 8.0  # (calib)
    activity_on_s: float = 0.3
    activity_hold_s: float = 2.0


@dc.dataclass
class SfcwCfg:
    """Mode à fréquence balayée, amplitude seule (voir radar/sfcw/)."""
    f_start: float = 1.2e9
    f_step: float = 20e6
    n_steps: int = 50
    f_s: float = 4.0e6
    rf_bandwidth: float = 4.0e6
    tone_hz: float = 250e3
    tag_offset_hz: float = 125e3       # LO RX décalé un pas sur deux → buffers périmés repérés
    rx_buffer_size: int = 1024
    kernel_buffers: int = 2
    max_reads: int = 10                # lectures max par pas pour obtenir un buffer frais
    discard: int = 3                   # buffers jetés d'office (seulement si tag_offset_hz = 0)
    rx_gain_db: float = 20.0
    tx_atten_db: float = 0.0
    amplitude: float = 0.5
    calib_mode: str = "manual"         # pas de recalibration à chaque saut de LO > 100 MHz
    zigzag: bool = True                # balayage montant puis descendant (pas de grand saut)
    bg_tau_s: float = 30.0             # fond adaptatif (MTI) : constante de temps
    nfft_range: int = 256
    min_range_m: float = 0.3           # en deçà : fuite / couplage d'antennes
    max_range_m: float = 0.0           # 0 → portée non ambiguë c/(4·f_step)
    breath_window_s: float = 20.0
    hop_s: float = 1.0
    breath_threshold_db: float = 12.0
    presence_threshold_db: float = 8.0
    on_count: int = 3
    map_span_s: float = 60.0
    record_dir: str = "data/sfcw"

    @property
    def freqs(self):
        import numpy as np
        return self.f_start + self.f_step * np.arange(self.n_steps)

    @property
    def f_stop(self) -> float:
        return self.f_start + self.f_step * (self.n_steps - 1)

    @property
    def unambiguous_range_m(self) -> float:
        # mesures réelles (amplitude) : spectre symétrique → c/(4Δf), pas c/(2Δf)
        return SPEED_OF_LIGHT / (4 * self.f_step)

    @property
    def resolution_m(self) -> float:
        return SPEED_OF_LIGHT / (2 * self.f_step * self.n_steps)


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
    push_hz: float = 20.0


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
    fast: FastCfg = dc.field(default_factory=FastCfg)
    sfcw: SfcwCfg = dc.field(default_factory=SfcwCfg)
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

    @property
    def scales_s(self) -> tuple[float, ...]:
        """Fenêtres du détecteur multi-échelle, croissantes ; la dernière = window_s."""
        return tuple(sorted(set(float(w) for w in self.analysis.extra_windows_s))) + (
            float(self.analysis.window_s),)

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
        if a.hop_s <= 0:
            raise ValueError("analysis: hop_s > 0 requis.")
        d_ = self.detector
        if not (len(a.extra_windows_s) == len(d_.scale_snr_threshold_db)
                == len(d_.scale_periodicity_threshold)):
            raise ValueError("analysis.extra_windows_s, detector.scale_snr_threshold_db et "
                             "detector.scale_periodicity_threshold doivent avoir la même longueur.")
        for w in a.extra_windows_s:
            if not 4.0 <= w < a.window_s:
                raise ValueError(f"analysis.extra_windows_s : {w} s hors [4, window_s[.")
        if not 0 < e.amplitude <= 1:
            raise ValueError("emission.amplitude doit être dans ]0, 1].")
        if s.chip not in CHIP_LO_RANGE:
            raise ValueError(f"sdr.chip = {s.chip!r} : choisir parmi {sorted(CHIP_LO_RANGE)}.")
        lo, hi = CHIP_LO_RANGE[s.chip]
        if not lo <= s.f_c <= hi:
            hint = (" Au-delà de 3.8 GHz il faut déverrouiller le Pluto en AD9364 puis mettre "
                    "sdr.chip: ad9364 (README, « Passer à 5.8 GHz »)." if s.chip == "ad9363" else "")
            raise ValueError(f"sdr.f_c = {s.f_c / 1e6:.0f} MHz hors de la plage {s.chip} "
                             f"({lo / 1e6:.0f}–{hi / 1e6:.0f} MHz).{hint}")
        self._validate_sfcw()

    def _validate_sfcw(self) -> None:
        w, s = self.sfcw, self.sdr
        if w.n_steps < 8 or w.f_step <= 0:
            raise ValueError("sfcw : n_steps ≥ 8 et f_step > 0 requis.")
        lo, hi = CHIP_LO_RANGE[s.chip]
        if w.f_start < lo or w.f_stop > hi:
            raise ValueError(f"sfcw : balayage {w.f_start / 1e6:.0f}–{w.f_stop / 1e6:.0f} MHz hors de la "
                             f"plage {s.chip} ({lo / 1e6:.0f}–{hi / 1e6:.0f} MHz).")
        if w.f_s < 521e3:
            raise ValueError("sfcw.f_s doit être ≥ 521 kS/s.")
        for f in (w.tone_hz, w.tone_hz - w.tag_offset_hz):
            if f <= 0 or abs(w.f_s / f - round(w.f_s / f)) > 1e-9:
                raise ValueError(f"sfcw : f_s/{f:g} Hz doit être entier (tonalité et tonalité marquée).")
            per = int(round(w.f_s / f))
            if w.rx_buffer_size % per:
                raise ValueError(f"sfcw.rx_buffer_size doit être multiple de {per} (orthogonalité des tons).")
        if not 0 <= w.tag_offset_hz < w.tone_hz:
            raise ValueError("sfcw.tag_offset_hz doit être dans [0, tone_hz[.")

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
        elif isinstance(val, (list, tuple)):
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
    "fast": FastCfg,
    "sfcw": SfcwCfg,
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
