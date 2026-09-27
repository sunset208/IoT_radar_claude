"""Lecteur série des modules radar 60 GHz « respiration + cœur » (Seeed MR60BHA2, Hi-Link LD6002).

Les deux modules parlent le même protocole binaire, dérivé de TinyFrame ::

    SOF=0x01 | ID (2, gros-boutiste) | LEN (2) | TYPE (2) | HEAD_CKSUM | DATA (LEN) | DATA_CKSUM
    CKSUM = ~(XOR des octets)   (en-tête : les 7 premiers octets ; données : DATA)

Types utiles (DATA en petit-boutiste) :

======== ========================================================== =========
0x0A13   phases : totale, respiration, cœur (3 × float32)            LD6002, MR60BHA2
0x0A14   rythme respiratoire, resp/min (float32)                    idem
0x0A15   rythme cardiaque, bpm (float32)                            idem
0x0A16   distance : drapeau (uint32) + distance m (float32)         idem
0x0F09   présence (uint16)                                          MR60BHA2
0x0A04   nombre de cibles (uint32, + nuage de points)               MR60BHA2
======== ========================================================== =========

Sources : composant ESPHome ``seeed_mr60bha2`` et bibliothèque Rust
``hlk_ld6002`` (même format, mêmes types).  Débits : MR60BHA2 115 200 bauds
(via le croquis de relais ``firmware/mr60bha2_passthrough`` sur le XIAO
ESP32C6) ; LD6002 **1 382 400 bauds** (adaptateur USB-UART rapide type FTDI,
pas CP210x ; module alimenté en 3.3 V / 600 mA).

Le kit XIAO livré avec le firmware ESPHome d'origine n'envoie pas ces trames
sur l'USB mais des journaux texte (« Sending state 15.0 bpm ») : le lecteur
les reconnaît aussi (mode « esphome », best effort).
"""

from __future__ import annotations

import collections
import logging
import math
import re
import struct
import threading
import time

logger = logging.getLogger(__name__)

SOF = 0x01
T_PHASE, T_BREATH, T_HEART, T_DISTANCE = 0x0A13, 0x0A14, 0x0A15, 0x0A16
T_PRESENCE, T_TARGETS = 0x0F09, 0x0A04
KNOWN_TYPES = {T_PHASE, T_BREATH, T_HEART, T_DISTANCE, T_PRESENCE, T_TARGETS}


def checksum(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return (~c) & 0xFF


def encode_frame(ftype: int, payload: bytes, frame_id: int = 0) -> bytes:
    """Construit une trame (tests, simulateur)."""
    head = struct.pack(">BHHH", SOF, frame_id & 0xFFFF, len(payload), ftype)
    return head + bytes([checksum(head)]) + payload + bytes([checksum(payload)])


class TinyFrameParser:
    """Découpe un flux d'octets en trames valides (resynchronisation sur erreur)."""

    MAX_LEN = 1024

    def __init__(self) -> None:
        self.buf = bytearray()
        self.frames_ok = 0
        self.errors = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        """Renvoie [(id, type, payload)]."""
        self.buf += data
        out = []
        b = self.buf
        while True:
            i = b.find(bytes([SOF]))
            if i < 0:
                b.clear()
                break
            if i:
                del b[:i]
            if len(b) < 8:
                break
            fid, ln, ftype = struct.unpack(">HHH", bytes(b[1:7]))
            if checksum(bytes(b[:7])) != b[7] or ln > self.MAX_LEN:
                self.errors += 1
                del b[0]                      # faux départ : on avance d'un octet
                continue
            if len(b) < 8 + ln + 1:
                break
            payload = bytes(b[8:8 + ln])
            if checksum(payload) != b[8 + ln]:
                self.errors += 1
                del b[0]
                continue
            del b[:9 + ln]
            self.frames_ok += 1
            out.append((fid, ftype, payload))
        return out


def decode(ftype: int, p: bytes) -> dict:
    """Trame → grandeurs physiques (dict vide si type inconnu ou trop court)."""
    try:
        if ftype == T_BREATH and len(p) >= 4:
            return {"breath_bpm": struct.unpack("<f", p[:4])[0]}
        if ftype == T_HEART and len(p) >= 4:
            return {"heart_bpm": struct.unpack("<f", p[:4])[0]}
        if ftype == T_DISTANCE and len(p) >= 8:
            flag, dist = struct.unpack("<If", p[:8])
            return {"distance_m": dist if flag else None, "range_flag": bool(flag)}
        if ftype == T_PHASE and len(p) >= 12:
            tot, br, hr = struct.unpack("<fff", p[:12])
            return {"phase_total": tot, "phase_breath": br, "phase_heart": hr}
        if ftype == T_PRESENCE and len(p) >= 2:
            return {"presence": bool(struct.unpack("<H", p[:2])[0])}
        if ftype == T_TARGETS and len(p) >= 4:
            return {"targets": struct.unpack("<I", p[:4])[0]}
    except struct.error:
        pass
    return {}


# Journaux ESPHome (firmware d'origine du kit XIAO) : « 'Nom': Sending state 15.00 bpm »
_ESPH = re.compile(r"'([^']+)'\s*:\s*Sending state\s+(ON|OFF|-?[0-9.]+|nan)", re.I)


def decode_esphome_line(line: str) -> dict:
    m = _ESPH.search(line)
    if not m:
        return {}
    name, val = m.group(1).lower(), m.group(2)
    if val.upper() in ("ON", "OFF"):
        if any(k in name for k in ("person", "presence", "people", "target", "occup")):
            return {"presence": val.upper() == "ON"}
        return {}
    try:
        v = float(val)
    except ValueError:
        return {}
    if math.isnan(v):
        return {}
    if "heart" in name or "cardi" in name:
        return {"heart_bpm": v}
    if "breath" in name or "respir" in name:
        return {"breath_bpm": v}
    if "distance" in name:
        return {"distance_m": v}
    if "target" in name and "number" in name:
        return {"targets": int(v)}
    return {}


class MmWaveReader:
    """Lecture en tâche de fond ; ``state()`` pour le tableau de bord.

    *port* : ``COM5``, ``/dev/ttyUSB0``… ou ``sim`` (trames synthétiques, démo sans module).
    """

    HISTORY_S = 300.0

    def __init__(self, port: str, baud: int = 115200, model: str | None = None,
                 serial_factory=None) -> None:
        self.port = port
        self.baud = int(baud)
        self.model = model or ("LD6002" if self.baud > 1_000_000 else "MR60BHA2 / LD6002")
        self._factory = serial_factory
        self.parser = TinyFrameParser()
        self.values: dict = {}
        self.t_last: float | None = None
        self.mode = "?"
        self.history: collections.deque = collections.deque(maxlen=int(self.HISTORY_S))
        self.version = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.time()
        self._text = bytearray()
        self.log: collections.deque = collections.deque(maxlen=200_000)   # (t_abs, valeurs)

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mmwave", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _open(self):
        if self._factory is not None:
            return self._factory()
        if self.port == "sim":
            return SimulatedModule()
        import serial  # pyserial
        return serial.Serial(self.port, self.baud, timeout=0.1)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ser = self._open()
                self.error = None
                with ser:
                    while not self._stop.is_set():
                        data = ser.read(4096)
                        if data:
                            self.feed(data)
                        self._tick_history()
            except Exception as exc:
                self.error = f"{exc}"
                self.version += 1
                logger.warning("Module 60 GHz (%s) : %s — nouvel essai dans 2 s", self.port, exc)
                self._stop.wait(2.0)

    # ------------------------------------------------------------------
    def feed(self, data: bytes) -> None:
        """Octets bruts → trames binaires ou lignes de journal ESPHome."""
        v0 = self.version
        got = False
        for _fid, ftype, payload in self.parser.feed(data):
            v = decode(ftype, payload)
            if v:
                self._update(v)
                self.mode = "trames"
                got = True
        # repli texte : ESPHome d'origine
        self._text += data
        if len(self._text) > 8192:
            del self._text[:-4096]
        while b"\n" in self._text:
            line, _, rest = bytes(self._text).partition(b"\n")
            self._text = bytearray(rest)
            v = decode_esphome_line(line.decode("utf-8", "replace"))
            if v:
                self._update(v)
                if not got:
                    self.mode = "esphome"
        if self.version != v0:
            self.log.append((self.t_last, dict(self.values)))
            self.version = v0 + 1          # une notification par paquet reçu

    def _update(self, v: dict) -> None:
        # rythme nul = « pas de mesure » (convention du module)
        for k in ("breath_bpm", "heart_bpm"):
            if k in v and (v[k] is None or not math.isfinite(v[k]) or v[k] <= 0):
                v[k] = None
        self.values.update(v)
        self.t_last = time.time()
        self.version += 1

    def _tick_history(self) -> None:
        t = time.time() - self._t0
        if not self.history or t - self.history[-1]["t"] >= 1.0:
            self.history.append({"t": round(t, 1), "breath_bpm": self.values.get("breath_bpm"),
                                 "heart_bpm": self.values.get("heart_bpm"),
                                 "distance_m": self.values.get("distance_m")})

    def window(self, t_start: float, t_stop: float) -> list:
        """Mesures entre deux instants absolus (time.time()), temps relatifs à t_start."""
        return [{"t": round(t - t_start, 2), **{k: v.get(k) for k in
                 ("breath_bpm", "heart_bpm", "distance_m", "presence")}}
                for t, v in list(self.log) if t_start <= t <= t_stop]

    def state(self) -> dict:
        v = self.values
        age = None if self.t_last is None else time.time() - self.t_last
        status = "error" if self.error else ("running" if age is not None and age < 5 else
                                             "en attente de trames")
        return {"status": status, "error": self.error, "port": self.port, "baud": self.baud,
                "model": self.model, "mode": self.mode, "last_rx_s": age,
                "breath_bpm": v.get("breath_bpm"), "heart_bpm": v.get("heart_bpm"),
                "distance_m": v.get("distance_m"), "presence": v.get("presence"),
                "targets": v.get("targets"),
                "frames": self.parser.frames_ok, "errors": self.parser.errors,
                "history": list(self.history)[-120:]}


class SimulatedModule:
    """Faux port série émettant des trames MR60BHA2 plausibles (démo, tests)."""

    def __init__(self, breath_bpm: float = 15.0, heart_bpm: float = 72.0, distance_m: float = 0.8,
                 speed: float = 1.0) -> None:
        self.b, self.h, self.d = breath_bpm, heart_bpm, distance_m
        self.speed = speed
        self.t = 0.0
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = 4096) -> bytes:
        time.sleep(0.2 / self.speed)
        self.t += 0.2
        self.i += 1
        br = self.b + 1.5 * math.sin(self.t / 20)
        hr = self.h + 4 * math.sin(self.t / 13)
        out = encode_frame(T_PRESENCE, struct.pack("<H", 1), self.i)
        out += encode_frame(T_BREATH, struct.pack("<f", br), self.i)
        out += encode_frame(T_HEART, struct.pack("<f", hr), self.i)
        out += encode_frame(T_DISTANCE, struct.pack("<If", 1, self.d), self.i)
        out += encode_frame(T_PHASE, struct.pack("<fff", 0.0, math.sin(self.t), 0.0), self.i)
        return out
