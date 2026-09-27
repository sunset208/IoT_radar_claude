"""Interface en ligne de commande : ``python -m radar <commande>``.

Commandes
---------
run         tableau de bord temps réel (Pluto, simulation ou relecture)
record      enregistrement sans interface (sessions scriptées)
check       diagnostic matériel (niveaux, fuite, stabilité de phase, débit)
scan        balayage de fréquences : bruit ambiant (et fuite TX→RX avec --tx)
sfcw        mode à fréquence balayée : run | bench | record | analyze
protocole   séance d'enregistrement guidée (plans dans protocols/)
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

    extras: dict = {}
    _add_mr60(cfg, args, extras)
    state = AppState(factory, cfg.web.push_hz,
                     scenarios=list(SCENARIOS) if kind in ("sim", "sim-rf") else None,
                     extras=extras)
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


def _sfcw_cfg(args):
    cfg = load_config(args.config, _overrides(args))
    s = cfg.sfcw
    if getattr(args, "gain", None) is not None:
        s.rx_gain_db = args.gain
    if getattr(args, "start", None) is not None:
        s.f_start = args.start
    if getattr(args, "step", None) is not None:
        s.f_step = args.step
    if getattr(args, "steps", None) is not None:
        s.n_steps = args.steps
    cfg.validate()
    return cfg


def _sfcw_source(cfg, args, realtime: bool = True):
    import numpy as np
    from radar.sfcw.scene import FakePluto, SfcwScene, SfcwSceneParams
    from radar.sfcw.sources import SfcwPlutoSource, SfcwReplaySource, SfcwSimSource
    params = SfcwSceneParams(scenario=args.scenario or "breathing",
                             target_range_m=args.range or 1.2)
    if args.source == "pluto":
        return SfcwPlutoSource(cfg)
    if args.source == "fake":      # vrai code Pluto, matériel simulé
        rng = np.random.default_rng(cfg.simulation.seed)
        return SfcwPlutoSource(cfg, sdr=FakePluto(SfcwScene(params, rng), rng=rng))
    if args.source == "sim":
        return SfcwSimSource(cfg, params, sweep_hz=args.sweep_hz, realtime=realtime)
    if args.source == "replay":
        if not args.file:
            raise SystemExit("--file requis pour --source replay")
        return SfcwReplaySource(args.file, realtime=realtime)
    raise SystemExit(f"source inconnue : {args.source}")


def cmd_sfcw(args) -> int:
    cfg = _sfcw_cfg(args)
    if args.action == "bench":
        logging.basicConfig(level=logging.WARNING)
        from radar.sfcw.bench import run_bench
        sdr = None
        if args.source == "fake" or args.sim:
            import numpy as np
            from radar.sfcw.scene import FakePluto, SfcwScene
            sdr = FakePluto(SfcwScene(rng=np.random.default_rng(0)), rng=np.random.default_rng(1))
        run_bench(cfg, sdr=sdr, repeats=args.repeats, fastlock=args.fastlock)
        return 0
    if args.action == "record":
        _setup_logging(cfg.logging.level, cfg.logging.dir)
        from radar.sfcw.engine import run_headless
        src = _sfcw_source(cfg, args, realtime=args.source != "sim")
        run_headless(cfg, src, args.duration, args.label, args.tag, args.notes, args.out)
        return 0
    if args.action == "analyze":
        logging.basicConfig(level=logging.WARNING)
        return _sfcw_analyze(cfg, args.paths)
    # run : tableau de bord, moteur CW au repos + panneau SFCW
    from radar.pipeline import Engine
    from radar.sfcw.engine import SfcwEngine
    from radar.sources.base import IdleSource
    from radar.web.server import AppState, serve
    _setup_logging(cfg.logging.level, cfg.logging.dir)
    sf = SfcwEngine(cfg, _sfcw_source(cfg, args))
    sf.start()
    extras = {"sfcw": sf}
    _add_mr60(cfg, args, extras)
    state = AppState(lambda _sc: Engine(cfg, IdleSource("MODE SFCW")), cfg.web.push_hz, extras=extras)
    port = args.port or cfg.web.port
    url = f"http://127.0.0.1:{port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Tableau de bord SFCW : {url}   (Ctrl+C pour quitter)")
    serve(state, args.host or cfg.web.host, port)
    return 0


def _sfcw_analyze(cfg, paths) -> int:
    import numpy as np
    from radar.offline import find_recordings
    from radar.sfcw.engine import make_processor
    from radar.sfcw.sources import SfcwReplaySource
    files = [p for p in find_recordings(paths or [str(resolve_path(cfg.sfcw.record_dir))])]
    if not files:
        print("Aucun enregistrement SFCW.")
        return 1
    print(f"{'fichier':40s} {'lab':>3s} {'durée':>6s} {'bal./s':>6s} {'%RESP':>6s} {'%MOUV':>6s} "
          f"{'%PRÉS':>6s} {'dist. (m)':>9s} {'SNR':>6s} {'resp/min':>8s}")
    for p in files:
        try:
            src = SfcwReplaySource(p, realtime=False, loop=False)
        except ValueError:
            continue
        proc = make_processor(cfg, src.rec.freqs)
        res = []
        for sw in src.sweeps():
            r = proc.push(sw)
            if r and "state" in r:
                res.append(r)
        if not res:
            print(f"{p.name[:40]:40s} (trop court)")
            continue
        st = [r["state"] for r in res]
        fr = lambda s: 100 * sum(x == s for x in st) / len(st)
        br = [r for r in res if r["state"] == "RESPIRATION"] or res
        print(f"{p.name[:40]:40s} {src.rec.label:>3d} {src.rec.duration_s:5.0f}s {proc.sweep_hz:6.2f} "
              f"{fr('RESPIRATION'):5.1f}% {fr('MOUVEMENT'):5.1f}% {fr('PRESENCE'):5.1f}% "
              f"{np.median([r['best_range_m'] for r in br]):9.2f} "
              f"{np.median([r['best_snr_db'] for r in br]):6.1f} {np.median([r['best_bpm'] for r in br]):8.1f}")
    return 0


def _add_mr60(cfg, args, extras: dict) -> None:
    """Module radar 60 GHz série (--mr60 PORT), affiché à côté du Pluto."""
    port = getattr(args, "mr60", None)
    if not port:
        return
    from radar.mmwave import MmWaveReader
    rd = MmWaveReader(port, baud=getattr(args, "mr60_baud", 115200))
    rd.start()
    extras["mr60"] = rd


def cmd_protocole(args) -> int:
    from radar.protocol import run_protocol
    cfg = _sfcw_cfg(args) if args.mode == "sfcw" else load_config(args.config, _overrides(args))
    if args.out:
        cfg.recording.dir = args.out
        cfg.sfcw.record_dir = args.out
    logging.basicConfig(level=logging.WARNING)
    return run_protocol(cfg, args)


def cmd_scan(args) -> int:
    from radar.hwcheck import run_scan
    cfg = load_config(args.config, _overrides(args))
    logging.basicConfig(level=logging.WARNING)
    return run_scan(cfg, args.start, args.stop, args.step, tx=args.tx)


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
    r.add_argument("--mr60", help="port série d'un module radar 60 GHz à afficher à côté (ex. COM5)")
    r.add_argument("--mr60-baud", type=int, default=115200)
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

    sn = sub.add_parser("scan", help="balayage de fréquences (bruit ambiant, fuite TX→RX)")
    common(sn)
    sn.add_argument("--start", type=float, default=0.4e9)
    sn.add_argument("--stop", type=float, default=3.7e9)
    sn.add_argument("--step", type=float, default=100e6)
    sn.add_argument("--tx", action="store_true",
                    help="émettre la tonalité : mesure aussi la fuite TX→RX (réponse des antennes)")
    sn.set_defaults(fn=cmd_scan)

    pr = sub.add_parser("protocole", help="séance d'enregistrement guidée")
    common(pr)
    pr.add_argument("--plan", default="labo", help="labo | planche | vide_long | chemin/vers/plan.yaml")
    pr.add_argument("--session", help="nom de séance (défaut : plan_date)")
    pr.add_argument("--mode", choices=("cw", "sfcw"), default="cw")
    pr.add_argument("--source", choices=("pluto", "sim", "sim-rf", "fake"), default="pluto")
    pr.add_argument("--scenario", help="(simulation) scénario")
    pr.add_argument("--range", type=float, help="(simulation SFCW) distance de la cible")
    pr.add_argument("--sweep-hz", type=float, default=4.0)
    pr.add_argument("--file", help=argparse.SUPPRESS)
    pr.add_argument("--start", type=float)
    pr.add_argument("--step", type=float)
    pr.add_argument("--steps", type=int)
    pr.add_argument("--mr60", help="port série du module 60 GHz (mesures sauvées à côté)")
    pr.add_argument("--mr60-baud", type=int, default=115200)
    pr.add_argument("--auto", type=float, default=0,
                    help="enchaîner sans attendre Entrée, avec N s de compte à rebours")
    pr.add_argument("--out", help="dossier des enregistrements")
    pr.set_defaults(fn=cmd_protocole)

    sf = sub.add_parser("sfcw", help="mode à fréquence balayée (amplitude seule)")
    common(sf)
    sf.add_argument("action", choices=("run", "bench", "record", "analyze"))
    sf.add_argument("paths", nargs="*", help="(analyze) fichiers .npz SFCW ou dossiers")
    sf.add_argument("--source", choices=("pluto", "sim", "fake", "replay"), default="pluto",
                    help="fake = code Pluto réel sur matériel simulé")
    sf.add_argument("--file", help="(replay) enregistrement SFCW")
    sf.add_argument("--scenario", help="(sim/fake) breathing | empty | walker | breathing_walker")
    sf.add_argument("--range", type=float, help="(sim/fake) distance de la cible (m)")
    sf.add_argument("--sweep-hz", type=float, default=4.0, help="(sim) balayages par seconde")
    sf.add_argument("--start", type=float, help="1re fréquence (Hz)")
    sf.add_argument("--step", type=float, help="pas de fréquence (Hz)")
    sf.add_argument("--steps", type=int, help="nombre de pas")
    sf.add_argument("--label", type=int, choices=(-1, 0, 1), default=-1)
    sf.add_argument("--duration", type=float, default=60.0)
    sf.add_argument("--tag", default="")
    sf.add_argument("--notes", default="")
    sf.add_argument("--out", help="dossier de sortie (record)")
    sf.add_argument("--repeats", type=int, default=10, help="(bench) balayages de répétabilité")
    sf.add_argument("--fastlock", action="store_true", help="(bench) tester le fastlock")
    sf.add_argument("--sim", action="store_true", help="(bench) sur le faux Pluto")
    sf.add_argument("--mr60", help="(run) port série d'un module radar 60 GHz (ex. COM5)")
    sf.add_argument("--mr60-baud", type=int, default=115200)
    sf.add_argument("--host")
    sf.add_argument("--port", type=int)
    sf.add_argument("--no-browser", action="store_true")
    sf.set_defaults(fn=cmd_sfcw)

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
