"""Tableau de bord web (FastAPI + WebSocket).

Accessible sur http://127.0.0.1:8050 (ou depuis un téléphone/tablette du même
réseau avec ``--host 0.0.0.0``).  Aucune dépendance JS externe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
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
                 scenarios: list[str] | None = None) -> None:
        self.factory = engine_factory
        self.push_hz = push_hz
        self.scenarios = scenarios or []
        self.lock = threading.Lock()
        self.version = 0
        self.engine = self._new_engine(None)

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
            "history": _safe_list(eng.history),
            "waterfall": _safe_list(eng.waterfall),
            "waterfall_f": eng.waterfall_f,
            "scenarios": state.scenarios,
            "config": {
                "window_s": eng.cfg.analysis.window_s,
                "hop_s": eng.cfg.analysis.hop_s,
                "breath_band": eng.cfg.analysis.breath_band,
                "f_c": eng.cfg.sdr.f_c,
            },
        })

    @app.get("/api/state")
    def api_state():
        return JSONResponse(full_state())

    @app.post("/api/record/start")
    def rec_start(req: RecordReq):
        path = state.engine.start_recording(req.label, req.tag, req.notes)
        return {"ok": True, "path": path}

    @app.post("/api/record/stop")
    def rec_stop():
        return {"ok": True, "path": state.engine.stop_recording()}

    @app.post("/api/annotate")
    def annotate(req: AnnotateReq):
        state.engine.annotate(req.text)
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
            await sock.send_text(json.dumps({"type": "full", **full_state()}))
            seen = state.version
            period = 1.0 / max(state.push_hz, 1.0)
            while True:
                await asyncio.sleep(period)
                if state.version == seen:
                    continue
                seen = state.version
                eng = state.engine
                msg = {"type": "update", "snapshot": eng.snapshot,
                       "hist": eng.history[-1] if eng.history else None}
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
