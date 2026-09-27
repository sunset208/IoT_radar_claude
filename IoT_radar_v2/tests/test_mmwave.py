"""Tests du lecteur de modules 60 GHz (MR60BHA2 / LD6002) — sans matériel."""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar.mmwave import (T_BREATH, T_DISTANCE, T_HEART, T_PRESENCE, MmWaveReader,  # noqa: E402
                          SimulatedModule, TinyFrameParser, checksum, decode,
                          decode_esphome_line, encode_frame)


def test_checksum_matches_esphome_definition():
    # ~XOR : exemple calculé à la main
    assert checksum(bytes([0x01, 0x00, 0x01, 0x00, 0x04, 0x0A, 0x14])) == (~(0x01 ^ 0x01 ^ 0x04 ^ 0x0A ^ 0x14)) & 0xFF


def test_parser_frames_and_resync():
    fr = encode_frame(T_BREATH, struct.pack("<f", 14.5), 7) + encode_frame(T_HEART, struct.pack("<f", 71.0))
    bad = bytearray(encode_frame(T_HEART, struct.pack("<f", 99.0)))
    bad[-1] ^= 0xFF                                     # somme de contrôle des données fausse
    stream = b"\x00\xff\x01\x02garbage" + bytes(bad) + fr + encode_frame(T_DISTANCE, struct.pack("<If", 1, 0.9))
    p = TinyFrameParser()
    got = []
    for i in range(0, len(stream), 5):                  # arrivée par petits morceaux
        got += p.feed(stream[i:i + 5])
    vals = {}
    for _, t, pl in got:
        vals.update(decode(t, pl))
    assert abs(vals["breath_bpm"] - 14.5) < 1e-5 and abs(vals["heart_bpm"] - 71.0) < 1e-5
    assert abs(vals["distance_m"] - 0.9) < 1e-6
    assert p.errors >= 1 and len(got) == 3


def test_esphome_log_fallback():
    assert decode_esphome_line("[D][sensor:094]: 'Real-time respiratory rate': Sending state 16.00000 bpm") == {"breath_bpm": 16.0}
    assert decode_esphome_line("[D][sensor:094]: 'Real-time heart rate': Sending state 70.00000 bpm")["heart_bpm"] == 70.0
    assert decode_esphome_line("[D][binary_sensor:036]: 'Person Information': Sending state ON") == {"presence": True}
    assert decode_esphome_line("[I][wifi:123]: connected") == {}


def test_reader_on_simulated_module():
    rd = MmWaveReader("sim", serial_factory=lambda: SimulatedModule(speed=20.0))
    rd.start()
    t0 = time.time()
    while rd.parser.frames_ok < 20 and time.time() - t0 < 5:
        time.sleep(0.05)
    rd.stop()
    st = rd.state()
    assert st["mode"] == "trames" and st["presence"] is True
    assert 10 < st["breath_bpm"] < 20 and 60 < st["heart_bpm"] < 85 and abs(st["distance_m"] - 0.8) < 1e-6


def test_presence_frame():
    assert decode(T_PRESENCE, struct.pack("<H", 0)) == {"presence": False}
