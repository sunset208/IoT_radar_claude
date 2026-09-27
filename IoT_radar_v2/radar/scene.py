"""Modèle physique de scène pour la simulation (streaming, état conservé).

Chaque diffuseur *k* renvoie un écho complexe ``A_k·exp(j(φ_k − 4π d_k(t)/λ))``
où ``d_k(t)`` est son déplacement radial (m).  La scène fournit la somme de ces
échos **en bande de base de l'écho** (c.-à-d. après retrait de f_offset) ; les
sources ajoutent ensuite la porteuse décalée, l'offset DC récepteur, le bruit.

Pourquoi c'est important : dans le code original, le clutter statique était
simulé à 0 Hz alors qu'en réalité la fuite TX→RX et les murs renvoient la
tonalité émise, donc à **f_offset**.  Ce détail rendait la simulation
trompeusement optimiste (le test de Fisher déclenchait à 100 % en salle vide
avec un clutter réaliste).

Scénarios disponibles (``SCENARIOS``) :

* ``empty``            — fuite + échos statiques + bruit
* ``breathing``        — + une personne immobile qui respire (et cœur)
* ``motion``           — + une personne qui bouge sans respiration régulière visible
* ``breathing_motion`` — respiration + mouvements corporels intermittents
* ``walker``           — personne qui marche dans la pièce (pas de cible immobile)
* ``fan``              — ventilateur oscillant (périodique lent, faux positif classique)
* ``apnea``            — respiration avec pauses (arrêts de 10–25 s)
"""

from __future__ import annotations

import dataclasses as dc
import math

import numpy as np

SCENARIOS = (
    "empty",
    "breathing",
    "motion",
    "breathing_motion",
    "walker",
    "fan",
    "apnea",
)

# Scénarios dont la vérité terrain est « respiration présente ».
BREATHING_SCENARIOS = {"breathing", "breathing_motion", "apnea"}


def dbfs_to_amp(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


class SmoothNoise:
    """Processus aléatoire lisse (Ornstein-Uhlenbeck échantillonné + interp.)

    Évaluable sur des instants croissants fournis par blocs successifs.
    """

    def __init__(self, rng: np.random.Generator, tau_s: float, sigma: float,
                 knot_dt: float = 0.1) -> None:
        self.rng = rng
        self.a = math.exp(-knot_dt / tau_s)
        self.s = sigma * math.sqrt(1.0 - self.a**2)
        self.dt = knot_dt
        self.t_knots = [0.0]
        self.v_knots = [float(rng.normal(0.0, sigma))]

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t_end = float(t[-1]) if t.size else 0.0
        while self.t_knots[-1] < t_end + self.dt:
            v = self.a * self.v_knots[-1] + self.s * self.rng.normal()
            self.t_knots.append(self.t_knots[-1] + self.dt)
            self.v_knots.append(v)
        # garde une petite fenêtre d'historique
        if len(self.t_knots) > 4096:
            keep = 2048
            self.t_knots = self.t_knots[-keep:]
            self.v_knots = self.v_knots[-keep:]
        return np.interp(t, self.t_knots, self.v_knots)


# ----------------------------------------------------------------------
# Générateurs de déplacement (état conservé entre blocs)
# ----------------------------------------------------------------------

class Breathing:
    """Respiration réaliste : forme asymétrique, rythme et amplitude variables."""

    def __init__(self, rng, rate_bpm: float, depth_m: float,
                 apnea: bool = False) -> None:
        self.rng = rng
        self.f0 = rate_bpm / 60.0
        self.depth = depth_m
        self.rate_var = SmoothNoise(rng, tau_s=15.0, sigma=0.08)
        self.amp_var = SmoothNoise(rng, tau_s=10.0, sigma=0.15)
        self.phase = float(rng.uniform(0, 2 * np.pi))
        self.t_last = 0.0
        self.apnea = apnea
        self._gate = 1.0
        self._next_toggle = float(rng.uniform(15, 40)) if apnea else math.inf
        self._gate_on = True

    def _gate_curve(self, t: np.ndarray) -> np.ndarray:
        if not self.apnea:
            return np.ones_like(t)
        out = np.empty_like(t)
        for i, ti in enumerate(t):
            if ti >= self._next_toggle:
                self._gate_on = not self._gate_on
                self._next_toggle = ti + (
                    self.rng.uniform(15, 40) if self._gate_on else self.rng.uniform(10, 25)
                )
            target = 1.0 if self._gate_on else 0.0
            self._gate += (target - self._gate) * 0.02
            out[i] = self._gate
        return out

    def __call__(self, t: np.ndarray) -> np.ndarray:
        if t.size == 0:
            return t
        f_inst = self.f0 * (1.0 + self.rate_var(t))
        dt = np.diff(t, prepend=self.t_last if self.t_last < t[0] else t[0])
        theta = self.phase + 2 * np.pi * np.cumsum(f_inst * dt)
        self.phase = float(theta[-1] % (2 * np.pi))
        self.t_last = float(t[-1])
        # Inspiration ~40 % du cycle, expiration ~60 % (forme non sinusoïdale)
        u = (theta / (2 * np.pi)) % 1.0
        shape = np.where(
            u < 0.4,
            0.5 - 0.5 * np.cos(np.pi * u / 0.4),
            0.5 + 0.5 * np.cos(np.pi * (u - 0.4) / 0.6),
        )
        amp = self.depth * np.clip(1.0 + self.amp_var(t), 0.3, 2.0)
        return amp * shape * self._gate_curve(t)


class Heartbeat:
    """Battement cardiaque (impulsions étroites, ~0.1–0.5 mm à la paroi)."""

    def __init__(self, rng, rate_bpm: float, depth_m: float) -> None:
        self.f0 = rate_bpm / 60.0
        self.depth = depth_m
        self.var = SmoothNoise(rng, tau_s=8.0, sigma=0.05)
        self.phase = float(rng.uniform(0, 2 * np.pi))
        self.t_last = 0.0

    def __call__(self, t: np.ndarray) -> np.ndarray:
        if t.size == 0:
            return t
        f_inst = self.f0 * (1.0 + self.var(t))
        dt = np.diff(t, prepend=self.t_last if self.t_last < t[0] else t[0])
        theta = self.phase + 2 * np.pi * np.cumsum(f_inst * dt)
        self.phase = float(theta[-1] % (2 * np.pi))
        self.t_last = float(t[-1])
        u = (theta / (2 * np.pi)) % 1.0
        return self.depth * np.exp(-0.5 * ((u - 0.2) / 0.06) ** 2)


class BodyMotion:
    """Mouvements corporels intermittents : sauts lisses + agitation."""

    def __init__(self, rng, rate_hz: float = 1 / 12.0, max_step_m: float = 0.05) -> None:
        self.rng = rng
        self.rate = rate_hz
        self.max_step = max_step_m
        self.offset = 0.0
        self.events: list[tuple[float, float, float]] = []   # (t0, durée, amplitude)
        self.next_event = float(rng.exponential(1 / rate_hz))
        self.jitter = SmoothNoise(rng, tau_s=0.3, sigma=1.0)
        self.jitter_env = 0.0

    def __call__(self, t: np.ndarray) -> np.ndarray:
        if t.size == 0:
            return t
        while self.next_event < t[-1]:
            dur = float(self.rng.uniform(0.5, 3.0))
            amp = float(self.rng.uniform(-1, 1) * self.max_step)
            self.events.append((self.next_event, dur, amp))
            self.next_event += float(self.rng.exponential(1 / self.rate))
        d = np.full_like(t, self.offset)
        env = np.zeros_like(t)
        still = []
        for (t0, dur, amp) in self.events:
            x = np.clip((t - t0) / dur, 0.0, 1.0)
            d += amp * (0.5 - 0.5 * np.cos(np.pi * x))
            env += np.where((t >= t0) & (t <= t0 + dur), 1.0, 0.0)
            if t0 + dur < t[-1]:
                self.offset += amp
            else:
                still.append((t0, dur, amp))
        self.events = still
        return d + 0.004 * env * self.jitter(t)


class Walker:
    """Personne qui marche en aller-retour (0.5–1.2 m/s) entre 1.5 et 5 m."""

    def __init__(self, rng) -> None:
        self.rng = rng
        self.r = float(rng.uniform(1.5, 5.0))
        self.v = float(rng.uniform(0.5, 1.2)) * rng.choice([-1, 1])
        self.t_last = 0.0
        self.sway = SmoothNoise(rng, tau_s=0.4, sigma=0.01)

    def __call__(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = np.diff(t, prepend=self.t_last if self.t_last < t[0] else t[0])
        r = np.empty_like(t)
        cur, v = self.r, self.v
        for i, d in enumerate(dt):          # rebonds aux bornes
            cur += v * d
            if cur > 5.0 or cur < 1.5:
                v = -v
                cur = min(max(cur, 1.5), 5.0)
            r[i] = cur
        self.r, self.v, self.t_last = cur, v, float(t[-1])
        amp = (2.0 / np.maximum(r, 0.5)) ** 2    # ~1/R² en amplitude
        return r + self.sway(t), amp


class OscillatingFan:
    """Tête de ventilateur oscillante : déplacement quasi périodique lent."""

    def __init__(self, rng) -> None:
        self.f = float(rng.uniform(0.06, 0.2))
        self.amp = float(rng.uniform(0.01, 0.04))
        self.phase = float(rng.uniform(0, 2 * np.pi))
        self.blade = float(rng.uniform(15, 25))

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return (self.amp * np.sin(2 * np.pi * self.f * t + self.phase)
                + 0.0003 * np.sin(2 * np.pi * self.blade * t))


# ----------------------------------------------------------------------
# Scène
# ----------------------------------------------------------------------

@dc.dataclass
class SceneParams:
    scenario: str = "breathing"
    wavelength: float = 0.0857
    snr_db: float = 25.0              # puissance cible / bruit, à fs_slow
    leak_dbfs: float = -12.0
    breath_rate_bpm: float = 16.0
    breath_depth_mm: float = 6.0
    heart_rate_bpm: float = 70.0
    heart_depth_mm: float = 0.25
    target_dbfs: float = -45.0        # niveau de l'écho humain (avant bruit)
    n_static: int = 4                 # échos statiques supplémentaires
    lo_drift_rad: float = 0.003       # dérive de phase commune TX/RX (à mesurer : radar check)


class Scene:
    """Somme des échos en bande de base (échelle : 1.0 = pleine échelle ADC)."""

    def __init__(self, p: SceneParams, rng: np.random.Generator) -> None:
        if p.scenario not in SCENARIOS:
            raise ValueError(f"Scénario inconnu '{p.scenario}'. Choix : {SCENARIOS}")
        self.p = p
        self.rng = rng
        self.k = 4 * np.pi / p.wavelength

        # Fuite TX→RX + échos statiques (murs, meubles) — tous à f_offset
        leak = dbfs_to_amp(p.leak_dbfs) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        statics = [
            dbfs_to_amp(p.leak_dbfs - rng.uniform(6, 25)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            for _ in range(p.n_static)
        ]
        self.static = complex(leak + sum(statics))
        self.drift = SmoothNoise(rng, tau_s=30.0, sigma=p.lo_drift_rad)

        self.target_amp = dbfs_to_amp(p.target_dbfs)
        self.target_phase = float(rng.uniform(0, 2 * np.pi))
        self.noise_std = self.target_amp / math.sqrt(10 ** (p.snr_db / 10))

        sc = p.scenario
        self.breathing = (
            Breathing(rng, p.breath_rate_bpm, p.breath_depth_mm * 1e-3,
                      apnea=(sc == "apnea"))
            if sc in ("breathing", "breathing_motion", "apnea") else None
        )
        self.heart = (
            Heartbeat(rng, p.heart_rate_bpm, p.heart_depth_mm * 1e-3)
            if self.breathing is not None else None
        )
        self.motion = (
            BodyMotion(rng, rate_hz=1 / 4.0 if sc == "motion" else 1 / 15.0)
            if sc in ("motion", "breathing_motion") else None
        )
        self.walker = Walker(rng) if sc == "walker" else None
        self.fan = OscillatingFan(rng) if sc == "fan" else None

    @property
    def has_breathing(self) -> bool:
        return self.breathing is not None

    def echo(self, t: np.ndarray) -> np.ndarray:
        """Signal complexe noiseless (hors bruit thermique) aux instants *t*."""
        out = np.full(t.shape, self.static, dtype=np.complex128)
        d_person = None
        if self.breathing is not None:
            d_person = self.breathing(t) + self.heart(t)
        if self.motion is not None:
            d_m = self.motion(t)
            d_person = d_m if d_person is None else d_person + d_m
        if d_person is not None:
            out += self.target_amp * np.exp(1j * (self.target_phase - self.k * d_person))
        if self.walker is not None:
            r, amp = self.walker(t)
            out += self.target_amp * amp * np.exp(1j * (self.target_phase - self.k * r))
        if self.fan is not None:
            out += 0.7 * self.target_amp * np.exp(1j * (self.target_phase + 1.0 - self.k * self.fan(t)))
        return out * np.exp(1j * self.drift(t))


def scene_from_config(sim_cfg, wavelength: float, rng: np.random.Generator,
                      scenario: str | None = None) -> Scene:
    p = SceneParams(
        scenario=scenario or sim_cfg.scenario,
        wavelength=wavelength,
        snr_db=sim_cfg.snr_db,
        leak_dbfs=sim_cfg.leak_dbfs,
        breath_rate_bpm=sim_cfg.breath_rate_bpm,
        breath_depth_mm=sim_cfg.breath_depth_mm,
        heart_rate_bpm=sim_cfg.heart_rate_bpm,
        heart_depth_mm=sim_cfg.heart_depth_mm,
    )
    return Scene(p, rng)


def random_scene_params(rng: np.random.Generator, wavelength: float,
                        scenario: str | None = None) -> SceneParams:
    """Randomisation de domaine (pour générer des données IA synthétiques)."""
    return SceneParams(
        scenario=scenario or str(rng.choice(SCENARIOS)),
        wavelength=wavelength,
        snr_db=float(rng.uniform(-5, 35)),
        leak_dbfs=float(rng.uniform(-30, -6)),
        breath_rate_bpm=float(rng.uniform(7, 40)),
        breath_depth_mm=float(rng.uniform(0.5, 12)),
        heart_rate_bpm=float(rng.uniform(50, 120)),
        heart_depth_mm=float(rng.uniform(0.05, 0.5)),
        target_dbfs=float(rng.uniform(-70, -30)),
        n_static=int(rng.integers(0, 8)),
        lo_drift_rad=float(rng.uniform(0.0, 0.05)),
    )
