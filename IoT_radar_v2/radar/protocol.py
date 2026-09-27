"""``radar protocole`` : séance d'enregistrement guidée (CW ou SFCW, module 60 GHz en option).

Les données étiquetées sont le vrai goulot d'étranglement du projet (rapport
§6.2) : cette commande enchaîne des étapes décrites dans ``protocols/*.yaml``
(consigne, étiquette, durée, annotations horodatées), avec **un seul** moteur
qui tourne en continu (pas de réouverture du Pluto entre deux étapes : le fond
et la phase restent continus), puis :

* écrit un manifeste ``data/sessions/<session>.json`` ;
* en CW, calibre sur les étapes « vide » de la séance (``calibration/<session>.json``)
  et évalue toutes les étapes ; en SFCW, analyse les balayages ;
* avec ``--mr60``, sauve les mesures du module 60 GHz à côté de chaque fichier
  (référence de rythme).
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import json
import sys
import time
from pathlib import Path

import yaml

from radar.config import PROJECT_ROOT, Config, resolve_path

PLANS_DIR = PROJECT_ROOT / "protocols"


@dc.dataclass
class Step:
    name: str
    label: int
    duration_s: float
    instruction: str
    annotate: list = dc.field(default_factory=list)


def load_plan(name_or_path: str) -> tuple[str, list[Step]]:
    p = Path(name_or_path)
    if not p.suffix:
        p = PLANS_DIR / f"{name_or_path}.yaml"
    with open(p, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    steps = [Step(name=s["name"], label=int(s["label"]), duration_s=float(s["duration_s"]),
                  instruction=s.get("instruction", ""), annotate=s.get("annotate") or [])
             for s in d["steps"]]
    return d.get("name", p.stem), steps


def _beep() -> None:
    sys.stdout.write("\a")
    sys.stdout.flush()


class _Runner:
    """Adapte le moteur CW ou SFCW à une même interface pour la séance."""

    def __init__(self, cfg: Config, mode: str, source_kind: str, args) -> None:
        self.mode = mode
        self.cfg = cfg
        if mode == "sfcw":
            from radar.__main__ import _sfcw_source
            from radar.sfcw.engine import SfcwEngine
            self.eng = SfcwEngine(cfg, _sfcw_source(cfg, args))
        else:
            from radar.__main__ import _make_source
            from radar.pipeline import Engine
            src = _make_source(cfg, source_kind, getattr(args, "scenario", None), None, realtime=True)
            self.eng = Engine(cfg, src)
            self.last: dict = {}
            self.eng.subscribe(self._on)

    def _on(self, s: dict) -> None:
        self.last.update(s)

    def start(self) -> None:
        self.eng.start()

    def stop(self) -> None:
        self.eng.stop()

    @property
    def error(self) -> str | None:
        return self.eng.error

    def ready(self) -> bool:
        if self.mode == "sfcw":
            return self.eng.proc.n_sweeps > 10
        return bool(self.last.get("fast")) or self.last.get("status") == "running"

    def status_line(self) -> str:
        if self.mode == "sfcw":
            p = self.eng.proc
            r = p.result
            return (f"{p.state:11s} cible {r.get('best_range_m', float('nan')):4.2f} m "
                    f"SNR {r.get('best_snr_db', float('nan')):5.1f} dB  {p.sweep_hz:3.1f} bal./s")
        f = self.last.get("fast") or {}
        d = self.last.get("decision") or {}
        bpm = d.get("breath_bpm")
        return (f"{(f.get('state') or d.get('state') or '…'):11s} présence {f.get('presence_db', 0) or 0:5.1f} dB "
                f"activité {f.get('activity_db', 0) or 0:5.1f} dB  SNR {(d.get('features') or {}).get('snr_db', 0) or 0:5.1f} dB"
                + (f"  {bpm:.1f}/min" if bpm else ""))


def run_protocol(cfg: Config, args) -> int:
    plan_name, steps = load_plan(args.plan)
    session = args.session or f"{plan_name}_{dt.datetime.now():%Y%m%d_%H%M}"
    mode = args.mode
    total = sum(s.duration_s for s in steps)
    print(f"=== Séance « {session} » : plan {plan_name}, {len(steps)} étapes, "
          f"{total / 60:.0f} min d'enregistrement, mode {mode.upper()} ===")
    runner = _Runner(cfg, mode, args.source, args)
    mr = None
    if getattr(args, "mr60", None):
        from radar.mmwave import MmWaveReader
        mr = MmWaveReader(args.mr60, baud=args.mr60_baud)
        mr.start()
    runner.start()
    print("Mise en route (fond, filtres)…", end="", flush=True)
    t0 = time.time()
    while not runner.ready() and time.time() - t0 < 60 and not runner.error:
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()
    if runner.error:
        print(f"Erreur : {runner.error}")
        runner.stop()
        return 1
    files = []
    try:
        for i, st in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] {st.name}  ({st.duration_s:.0f} s, étiquette {st.label})")
            print(f"  ▶ {st.instruction}")
            if args.auto:
                for k in range(int(args.auto), 0, -1):
                    print(f"\r  départ dans {k:2d} s ", end="", flush=True)
                    time.sleep(1)
                print()
            else:
                input("  Entrée quand vous êtes prêt… ")
                for k in (3, 2, 1):
                    print(f"\r  {k}…", end="", flush=True)
                    time.sleep(1)
                print()
            _beep()
            tag = f"{session}_{st.name}"
            path = runner.eng.start_recording(st.label, tag, st.instruction, max_s=st.duration_s)
            t_start = time.time()
            pending = sorted((float(a[0]), str(a[1])) for a in st.annotate)
            while True:
                el = time.time() - t_start
                while pending and el >= pending[0][0]:
                    _, text = pending.pop(0)
                    runner.eng.annotate(text)
                    _beep()
                    print(f"\n  >>> {text.upper()} <<<")
                print(f"\r  {el:5.0f}/{st.duration_s:.0f} s  {runner.status_line()}   ", end="", flush=True)
                if el >= st.duration_s + 0.5 or runner.error:
                    break
                time.sleep(0.5)
            runner.eng.stop_recording()
            _beep()
            print(f"\n  → {Path(path).name}")
            if mr is not None:
                side = Path(path).with_suffix(".mr60.json")
                with open(side, "w", encoding="utf-8") as fh:
                    json.dump({"model": mr.model, "port": mr.port,
                               "rows": mr.window(t_start, time.time())}, fh, ensure_ascii=False)
            files.append({"step": st.name, "label": st.label, "file": str(path),
                          "duration_s": st.duration_s, "instruction": st.instruction})
            if runner.error:
                print(f"Erreur moteur : {runner.error}")
                break
    except KeyboardInterrupt:
        print("\nSéance interrompue — les étapes terminées sont conservées.")
        runner.eng.stop_recording()
    finally:
        runner.stop()
        if mr is not None:
            mr.stop()
    man = resolve_path("data/sessions") / f"{session}.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "w", encoding="utf-8") as fh:
        json.dump({"session": session, "plan": plan_name, "mode": mode,
                   "date": dt.datetime.now().isoformat(timespec="seconds"),
                   "config": cfg.to_dict(), "steps": files}, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nManifeste : {man}")
    _summary(cfg, mode, session, files)
    return 0


def _summary(cfg: Config, mode: str, session: str, files: list) -> None:
    paths = [Path(f["file"]) for f in files if Path(f["file"]).is_file()]
    if not paths:
        return
    if mode == "sfcw":
        from radar.__main__ import _sfcw_analyze
        print("\n=== Analyse SFCW ===")
        _sfcw_analyze(cfg, [str(p) for p in paths])
        return
    from radar.offline import calibrate, evaluate_recordings, print_table, save_calibration
    cal = None
    neg = [f for f in files if f["label"] == 0]
    if neg and any(f["label"] == 1 for f in files):
        try:
            cal = calibrate(paths, cfg)
            out = resolve_path("calibration") / f"{session}.json"
            save_calibration(cal, out)
            print(f"\nCalibration de la séance : {out}"
                  + (f"  ({cal['warning']})" if cal.get("warning") else ""))
        except SystemExit as exc:
            print(f"Calibration impossible : {exc}")
    print("\n=== Évaluation de la séance ===")
    print_table(evaluate_recordings(paths, cfg, cal))
    if cal is not None:
        print("(les étapes « vide » ont servi à la calibration : leur taux de fausse alarme est optimiste)")
