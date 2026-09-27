"""Interface en ligne de commande : ``python -m radar <commande>``.

Commandes
---------
run         tableau de bord temps réel (Pluto, simulation ou relecture)
record      enregistrement sans interface (sessions scriptées)
check       diagnostic matériel (niveaux, fuite, stabilité de phase, débit)
evaluate    métriques du détecteur sur enregistrements ou scénarios simulés
calibrate   seuils (et régression logistique) à partir d'enregistrements étiquetés
convert-v1  conversion des .iq de la v1 vers le format v2
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

from radar.config import PROJECT_ROOT, load_config, resolve_path


def _setup_logging(level: str, log_dir: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        d = resolve_path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(d / f"radar_{time.strftime('%Y%m%d_%H%M%S')}.log",
                                            encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def _overrides(args) -> dict:
    o: dict = {}
    if getattr(args, "uri", None):
        o.setdefault("sdr", {})["uri"] = args.uri
    if getattr(args, "gain", None) is not None:
        o.setdefault("sdr", {})["rx_gain_db"] = args.gain
    if getattr(args, "fc", None) is not None:
        o.setdefault("sdr", {})["f_c"] = args.fc
    if getattr(args, "seed", None) is not None:
        o.setdefault("simulation", {})["seed"] = args.seed
    return o


def _make_source(cfg, kind: str, scenario: str | None, file: str | None, realtime: bool = True):
    if kind == "pluto":
        from radar.sources.pluto import PlutoSource
        return PlutoSource(cfg)
    if kind == "sim-rf":
        from radar.sources.simulation import SimulatedRawSource
        return SimulatedRawSource(cfg, scenario=scenario, realtime=realtime)
    if kind == "sim":
        import numpy as np
        from radar.scene import scene_from_config
        from radar.sources.simulation import SimulatedSlowSource
        rng = np.random.default_rng(cfg.simulation.seed)
        scene = scene_from_config(cfg.simulation, cfg.sdr.wavelength, rng, scenario)
        return SimulatedSlowSource(cfg.frontend.fs_slow, scene, realtime=realtime, rng=rng)
    if kind == "replay":
        from radar.sources.replay import ReplaySource
        if not file:
            raise SystemExit("--file requis pour --source replay")
        return ReplaySource(file, realtime=realtime, loop=True)
    raise SystemExit(f"source inconnue : {kind}")


def _load_ml(path: str | None):
    if not path:
        return None
    sys.path.insert(0, str(PROJECT_ROOT))
    from ai.infer import TorchDetector
    return TorchDetector(path)


# ----------------------------------------------------------------------

def cmd_run(args) -> int:
    from radar.dsp.detector import load_calibration
    from radar.pipeline import Engine
    from radar.scene import SCENARIOS
    from radar.web.server import AppState, serve

    cfg = load_config(args.config, _overrides(args))
    _setup_logging(cfg.logging.level, cfg.logging.dir)
    cal = load_calibration(args.calibration)
    ml = _load_ml(args.ai)
    kind = args.source

    def factory(scenario):
        src = _make_source(cfg, kind, scenario or args.scenario, args.file)
        return Engine(cfg, src, cal, ml)

    state = AppState(factory, cfg.web.push_hz,
                     scenarios=list(SCENARIOS) if kind in ("sim", "sim-rf") else None)
    host = args.host or cfg.web.host
    port = args.port or cfg.web.port
    url = f"http://127.0.0.1:{port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Tableau de bord : {url}   (Ctrl+C pour quitter)")
    serve(state, host, port)
    return 0


def cmd_record(args) -> int:
    from radar.pipeline import Engine
    cfg = load_config(args.config, _overrides(args))
    if args.raw:
        cfg.recording.save_raw = True
    if args.out:
        cfg.recording.dir = args.out
    _setup_logging(cfg.logging.level, cfg.logging.dir)
    realtime = args.source != "sim"
    src = _make_source(cfg, args.source, args.scenario, None, realtime=realtime)
    eng = Engine(cfg, src)
    path = eng.start_recording(args.label, args.tag, args.notes, max_s=args.duration)
    last = {}
    eng.subscribe(lambda s: last.update(s))
    eng.start()
    t0 = time.time()
    try:
        while True:
            time.sleep(0.2 if args.source == "sim" else 1.0)
            el = eng.recorder.elapsed_s if eng.recorder else 0.0
            d = (last.get("decision") or {})
            print(f"\r{el:6.1f}/{args.duration:.0f} s  état={d.get('state', '…'):12s} "
                  f"score={d.get('score', 0) or 0:.2f}", end="", flush=True)
            if (eng.recorder is None or eng.recorder.full or eng.error
                    or not eng._thread.is_alive()):
                break
            if time.time() - t0 > args.duration * 3 + 30:
                print("\nDélai dépassé.")
                break
    except KeyboardInterrupt:
        print("\nInterrompu — sauvegarde.")
    eng.stop()
    print(f"\nFichier : {path}")
    return 1 if eng.error else 0


def cmd_check(args) -> int:
    from radar.hwcheck import run_check
    cfg = load_config(args.config, _overrides(args))
    logging.basicConfig(level=logging.WARNING)
    return run_check(cfg, stability_s=args.stability, sim=args.sim)


def cmd_evaluate(args) -> int:
    from radar.dsp.detector import load_calibration
    from radar.offline import evaluate_recordings, evaluate_simulation, find_recordings, print_table
    cfg = load_config(args.config, _overrides(args))
    logging.basicConfig(level=logging.WARNING)
    cal = load_calibration(args.calibration)
    if args.sim:
        rows = evaluate_simulation(cfg, n_per=args.sim, duration_s=args.duration,
                                   seed=args.seed or 0, calibration=cal)
    else:
        paths = find_recordings(args.paths or [str(resolve_path(cfg.recording.dir))])
        if not paths:
            print("Aucun enregistrement trouvé.")
            return 1
        rows = evaluate_recordings(paths, cfg, cal)
    print_table(rows)
    return 0


def cmd_calibrate(args) -> int:
    import json
    from radar.offline import calibrate, find_recordings, save_calibration
    cfg = load_config(args.config, _overrides(args))
    logging.basicConfig(level=logging.WARNING)
    paths = find_recordings(args.paths or [str(resolve_path(cfg.recording.dir))])
    cal = calibrate(paths, cfg, pfa=args.pfa)
    out = Path(args.out)
    save_calibration(cal, out)
    print(json.dumps({k: v for k, v in cal.items() if k != "logistic"}, indent=2, ensure_ascii=False))
    print(f"\nCalibration écrite : {out}\nUtilisation : radar run --calibration {out}")
    return 0


def cmd_convert(args) -> int:
    from radar.legacy import convert_tree
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outs = convert_tree(Path(args.src), resolve_path(args.out), fs_slow=args.fs_slow,
                        f_offset=args.f_offset, f_c=args.fc or 3.5e9)
    print(f"{len(outs)} enregistrement(s) converti(s) → {resolve_path(args.out)}")
    return 0


# ----------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="radar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML de surcharge (défaut : configs/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--uri", help="ex. ip:192.168.2.1 ou usb:")
        sp.add_argument("--gain", type=float, help="gain RX (dB)")
        sp.add_argument("--fc", type=float, help="porteuse (Hz)")
        sp.add_argument("--seed", type=int)

    r = sub.add_parser("run", help="tableau de bord temps réel")
    common(r)
    r.add_argument("--source", choices=("pluto", "sim", "sim-rf", "replay"), default="pluto")
    r.add_argument("--scenario", help="scénario de simulation")
    r.add_argument("--file", help="enregistrement .npz (replay)")
    r.add_argument("--calibration", help="JSON produit par radar calibrate")
    r.add_argument("--ai", help="modèle IA (.pt) à afficher en parallèle du détecteur classique")
    r.add_argument("--host")
    r.add_argument("--port", type=int)
    r.add_argument("--no-browser", action="store_true")
    r.set_defaults(fn=cmd_run)

    rc = sub.add_parser("record", help="enregistrement sans interface")
    common(rc)
    rc.add_argument("--source", choices=("pluto", "sim", "sim-rf"), default="pluto")
    rc.add_argument("--scenario")
    rc.add_argument("--label", type=int, choices=(-1, 0, 1), required=True,
                    help="1 = respiration présente, 0 = absente, -1 = inconnu")
    rc.add_argument("--duration", type=float, default=120.0)
    rc.add_argument("--tag", default="", help="identifiant de session (ex. salleB_2m_briques)")
    rc.add_argument("--notes", default="")
    rc.add_argument("--raw", action="store_true", help="enregistrer aussi l'IQ brut (.c64)")
    rc.add_argument("--out", help="dossier de sortie")
    rc.set_defaults(fn=cmd_record)

    c = sub.add_parser("check", help="diagnostic matériel")
    common(c)
    c.add_argument("--stability", type=float, default=0.0, help="durée du test de stabilité (s)")
    c.add_argument("--sim", action="store_true", help="démonstration sur simulateur")
    c.set_defaults(fn=cmd_check)

    e = sub.add_parser("evaluate", help="métriques du détecteur")
    common(e)
    e.add_argument("paths", nargs="*", help="fichiers .npz ou dossiers")
    e.add_argument("--sim", type=int, default=0, help="N simulations par scénario (au lieu de fichiers)")
    e.add_argument("--duration", type=float, default=120.0)
    e.add_argument("--calibration")
    e.set_defaults(fn=cmd_evaluate)

    ca = sub.add_parser("calibrate", help="calibration des seuils")
    common(ca)
    ca.add_argument("paths", nargs="*")
    ca.add_argument("--pfa", type=float, default=0.01, help="fausse alarme visée par fenêtre")
    ca.add_argument("--out", default=str(PROJECT_ROOT / "calibration" / "calibration.json"))
    ca.set_defaults(fn=cmd_calibrate)

    cv = sub.add_parser("convert-v1", help="convertir les .iq de la v1")
    cv.add_argument("src", help="dossier AICalibration/data de la v1")
    cv.add_argument("--out", default="data/v1_converted")
    cv.add_argument("--fs-slow", type=float, default=50.0)
    cv.add_argument("--f-offset", type=float, default=500.0)
    cv.add_argument("--fc", type=float, default=None)
    cv.set_defaults(fn=cmd_convert)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
