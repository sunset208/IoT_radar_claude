"""Tableau de bord web (FastAPI + WebSocket).

Accessible sur http://127.0.0.1:8050 (ou depuis un téléphone/tablette du même
réseau avec ``--host 0.0.0.0``).  Aucune dépendance JS externe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from radar.pipeline import Engine, _clean

logger = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"


class RecordReq(BaseModel):
    label: int
    tag: str = ""
    notes: str = ""


class AnnotateReq(BaseModel):
    text: str


class ScenarioReq(BaseModel):
    scenario: str


class AppState:
    """Détient le moteur courant ; permet de le relancer (changement de scénario)."""

    def __init__(self, engine_factory: Callable[[str | None], Engine], push_hz: float,
                 scenarios: list[str] | None = None, extras: dict | None = None) -> None:
        self.factory = engine_factory
        self.push_hz = push_hz
        self.scenarios = scenarios or []
        self.lock = threading.Lock()
        self.version = 0
        # sources annexes affichées à côté du Pluto (module 60 GHz, SFCW…) :
        # objets avec .state() -> dict JSON et .version (int croissant)
        self.extras: dict = extras or {}
        self.rec_t0: float | None = None
        self.engine = self._new_engine(None)

    def save_aux(self, path: str | None) -> None:
        """Mesures du module 60 GHz pendant l'enregistrement → <fichier>.mr60.json
        (référence de rythme pour `radar evaluate`)."""
        mr = self.extras.get("mr60")
        if not path or mr is None or self.rec_t0 is None:
            return
        try:
            rows = mr.window(self.rec_t0, time.time())
            with open(Path(path).with_suffix(".mr60.json"), "w", encoding="utf-8") as fh:
                json.dump({"model": mr.model, "port": mr.port, "rows": rows}, fh, ensure_ascii=False)
        except Exception:
            logger.warning("Mesures 60 GHz non sauvegardées", exc_info=True)

    def record_target(self):
        """Moteur qui enregistre : le SFCW quand le moteur CW est au repos."""
        if self.engine.source.kind == "idle" and "sfcw" in self.extras:
            return self.extras["sfcw"]
        return self.engine

    def extras_state(self) -> dict:
        out = {}
        for name, ex in self.extras.items():
            try:
                out[name] = ex.state()
            except Exception as exc:  # une annexe en panne ne doit pas couper l'UI
                out[name] = {"status": "error", "error": str(exc)}
        return out

    def extras_version(self) -> int:
        return sum(int(getattr(ex, "version", 0)) for ex in self.extras.values())

    def stop_extras(self) -> None:
        for ex in self.extras.values():
            try:
                ex.stop()
            except Exception:
                pass

    def _new_engine(self, scenario: str | None) -> Engine:
        eng = self.factory(scenario)
        eng.subscribe(self._on_snapshot)
        eng.start()
        return eng

    def _on_snapshot(self, _snap: dict) -> None:
        self.version += 1

    def restart(self, scenario: str) -> None:
        with self.lock:
            old = self.engine
            old.stop()
            self.engine = self._new_engine(scenario)
            self.version += 1


def _safe_list(dq) -> list:
    for _ in range(5):
        try:
            return list(dq)
        except RuntimeError:  # deque modifiée pendant la copie (thread moteur)
            continue
    return []


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="IoT Radar v2")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    def full_state() -> dict:
        eng = state.engine
        return _clean({
            "snapshot": eng.snapshot,
            "fast": eng.fast,
            "history": _safe_list(eng.history),
            "waterfall": _safe_list(eng.waterfall),
            "waterfall_f": eng.waterfall_f,
            "wave": _safe_list(eng.wave),
            "wave_end_t": eng.wave_end_t,
            "md_cols": _safe_list(eng.md_cols),
            "md_f": (None if eng.proc.fast is None else eng.proc.fast.md_f.round(3).tolist()),
            "scenarios": state.scenarios,
            "extras": state.extras_state(),
            "config": {
                "window_s": eng.cfg.analysis.window_s,
                "scales_s": list(eng.cfg.scales_s),
                "hop_s": eng.cfg.analysis.hop_s,
                "breath_band": eng.cfg.analysis.breath_band,
                "f_c": eng.cfg.sdr.f_c,
                "wavelength": eng.cfg.sdr.wavelength,
            },
        })

    @app.get("/api/state")
    def api_state():
        return JSONResponse(full_state())

    @app.post("/api/record/start")
    def rec_start(req: RecordReq):
        path = state.record_target().start_recording(req.label, req.tag, req.notes)
        state.rec_t0 = time.time()
        return {"ok": True, "path": path}

    @app.post("/api/record/stop")
    def rec_stop():
        path = state.record_target().stop_recording()
        state.save_aux(path)
        return {"ok": True, "path": path}

    @app.post("/api/annotate")
    def annotate(req: AnnotateReq):
        state.record_target().annotate(req.text)
        return {"ok": True}

    @app.post("/api/sfcw/background")
    def sfcw_background():
        sf = state.extras.get("sfcw")
        if sf is None:
            return JSONResponse({"ok": False, "error": "mode SFCW inactif"}, status_code=400)
        sf.freeze_background()
        return {"ok": True}

    @app.post("/api/scenario")
    def scenario(req: ScenarioReq):
        if req.scenario not in state.scenarios:
            return JSONResponse({"ok": False, "error": "scénario indisponible"}, status_code=400)
        state.restart(req.scenario)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            eng = state.engine
            await sock.send_text(json.dumps({"type": "full", **full_state()}))
            seen = state.version
            seen_snap, seen_wave, seen_hist = eng.snapshot_id, eng.wave_count, eng.hist_count
            seen_extras = state.extras_version()
            period = 1.0 / max(state.push_hz, 1.0)
            while True:
                await asyncio.sleep(period)
                if state.engine is not eng:          # moteur relancé (scénario) : état complet
                    eng = state.engine
                    await sock.send_text(json.dumps({"type": "full", **full_state()}))
                    seen_snap, seen_wave, seen_hist = eng.snapshot_id, eng.wave_count, eng.hist_count
                    continue
                ev = state.extras_version()
                if state.version == seen and ev == seen_extras:
                    continue
                seen = state.version
                msg: dict = {"type": "update", "fast": eng.fast}
                if eng.snapshot_id != seen_snap:
                    seen_snap = eng.snapshot_id
                    msg["snapshot"] = eng.snapshot
                # toutes les nouveautés depuis le dernier envoi (aucune perte si
                # la boucle prend du retard sur le moteur)
                nw = min(eng.wave_count - seen_wave, len(eng.wave))
                if nw > 0:
                    msg["wave"] = _safe_list(eng.wave)[-nw:]
                    msg["wave_end_t"] = eng.wave_end_t
                seen_wave = eng.wave_count
                nh = min(eng.hist_count - seen_hist, len(eng.history))
                if nh > 0:
                    msg["hist"] = _safe_list(eng.history)[-nh:]
                seen_hist = eng.hist_count
                if ev != seen_extras:
                    seen_extras = ev
                    msg["extras"] = state.extras_state()
                await sock.send_text(json.dumps(_clean(msg)))
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def serve(state: AppState, host: str, port: int) -> None:
    import uvicorn
    app = create_app(state)
    logger.info("Tableau de bord : http://%s:%d", "127.0.0.1" if host in ("0.0.0.0", "") else host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        state.engine.stop()
        state.stop_extras()
