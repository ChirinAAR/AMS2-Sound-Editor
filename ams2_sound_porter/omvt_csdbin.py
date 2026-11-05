"""Codec y envoltorio de alto nivel para archivos .csdbin / .csd de Automobilista 2.

Integra el codec Q02 CSDBIN original de Open Madness Vehicle Tools (OMVT)
(Copyright (C) 2026 ohyeah2389 — GPL-3.0-or-later con Output Exception v1.0).
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# Núcleo original OMVT csdbin.py (Q02 CSDBIN car-sound codec)
# ============================================================================

MAGIC = b"Q\x02\x01\x05"
_HDR = b"\x08\x00\x01\x04"
_FP = bytes.fromhex("01000000ba681a5325006143")
_ROOT = 0x398
_FIX_A, _FIX_B = 0x2E8, 0x2F8

_SOUNDS = {
    0xC0: "mix_snapshot",
    0xC8: "engine_bank",
    0xD0: "engine_ext",
    0xD8: "engine_int",
    0xE0: "exhaust_ext",
    0xE8: "exhaust_int",
    0xF0: "starter",
    0xF8: "starting",
    0x128: "ai_engine_bank",
    0x130: "ai_engine_ext",
    0x138: "ai_engine_int",
    0x140: "ai_engine_live",
    0x148: "turbo_bank",
    0x150: "turbo",
    0x158: "turbo_dump",
    0x160: "transmission_bank",
    0x168: "transmission",
    0x170: "rev_limiter_bank",
    0x178: "rev_limiter",
    0x180: "gearshift_bank",
    0x188: "gearshift_up",
    0x190: "gearshift_down",
    0x198: "gearshift_grind",
    0x1A0: "splutter_ai",
    0x1A8: "splutter_ext",
    0x1B0: "splutter_int",
    0x1B8: "backfire_ai",
    0x1C0: "backfire_ext",
    0x1C8: "backfire_int",
    0x1D0: "brakes",
    0x1D8: "scrape",
    0x1E8: "bumpstop",
    0x1F0: "wind_bank",
    0x1F8: "wind",
    0x200: "suspension",
    0x208: "chassis",
    0x210: "chassis_loop",
    0x218: "shift_pop",
    0x240: "abs",
    0x340: "ignition_on",
    0x348: "ignition_off",
    0x358: "gearbox",
}
_BLOBS = {0xA8: "engine_placement", 0x380: "cam_volumes", 0x390: "surface_params"}
_SOUND_OFF = {name: off for off, name in _SOUNDS.items()}
_BLOB_OFF = {name: off for off, name in _BLOBS.items()}
_OFF_KEY = re.compile(r"^off_([0-9a-fA-F]{4})$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def u32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off : off + 4], "little")


def p32(v: int) -> bytes:
    return (v & 0xFFFFFFFF).to_bytes(4, "little")


def pad16(n: int) -> int:
    return (16 - n % 16) % 16


def fmt_f32(x: float) -> str:
    x = struct.unpack("<f", struct.pack("<f", x))[0]
    for prec in range(8, 18):
        s = format(x, f".{prec}g")
        if "." not in s and "e" not in s.lower():
            s += ".0"
        if struct.pack("<f", float(s)) == struct.pack("<f", x):
            return s
    return format(x, ".17g")


def fmt_slot(raw: bytes) -> str:
    i = int.from_bytes(raw, "little")
    if i == 0xFFFFFFFF:
        return "-1"
    if i < 0x10000:
        return str(i)
    return fmt_f32(struct.unpack("<f", raw)[0])


def parse_slot(s: str) -> bytes:
    if re.search(r"[.eE]", s):
        return struct.pack("<f", float(s))
    v = int(s, 10)
    return struct.pack("<i" if v < 0 else "<I", v)


def fmt_str(s: str) -> str:
    return s if _IDENT.fullmatch(s) else json.dumps(s)


def parse_str(s: str) -> str:
    return json.loads(s) if s[:1] in "\"'" else s


def _cstrs(blob: bytes) -> list[str]:
    out = []
    i = 0
    while i < len(blob):
        end = blob.find(b"\x00", i)
        if end < 0:
            break
        if end > i:
            out.append(blob[i:end].decode("latin-1"))
        i = end + 1
    return out


def parse_container(raw: bytes) -> tuple[bytes, dict[int, bytes]]:
    if raw[:4] != MAGIC:
        raise ValueError("El archivo no es un contenedor CSDBIN (Q02) válido.")
    nsec, extra_n = raw[3], raw[4]
    off = 0x10
    recs = []
    for _ in range(nsec):
        recs.append((u32(raw, off), raw[off + 4], raw[off + 5]))
        off += 8
    cur = extra_n + (nsec + 2) * 8
    secs = {}
    for size, typ, pad in recs:
        secs[typ] = raw[cur : cur + size]
        cur += size + pad
    return raw[8:16], secs


def build_container(extra: bytes, data: bytes, dmap: bytes, pool: bytes) -> bytes:
    fix = (
        p32(_FIX_A)
        + b"\x00" * 12
        + p32(_FIX_B)
        + b"\x00" * 4
        + data[_FIX_B + 8 : _FIX_B + 12]
        + b"\x00" * 4
    )
    blobs = ((1, data), (0x10, dmap), (0x32, pool), (0x33, fix), (0x50, _FP))
    recs, body = bytearray(), bytearray()
    for typ, blob in blobs:
        recs += p32(len(blob)) + bytes((typ, pad16(len(blob)), 0, 0))
        body += blob + b"\x00" * pad16(len(blob))
    return MAGIC + _HDR + extra + bytes(recs) + b"\x00" * _HDR[0] + bytes(body)


def _link_name(off: int, sounds: bool) -> str:
    table = _SOUNDS if sounds else _BLOBS
    return table.get(off) or f"off_{off:04x}"


def _link_off(name: str, sounds: bool) -> int:
    table = _SOUND_OFF if sounds else _BLOB_OFF
    if name in table:
        return table[name]
    m = _OFF_KEY.match(name)
    if not m:
        raise ValueError(f"Enlace desconocido ({'sound' if sounds else 'blob'}): {name}")
    return int(m.group(1), 16)


def _pieces(data: bytes, targets: list[int]) -> list[tuple[int, str, object]]:
    bounds = targets + [len(data)]
    out = []
    for a, b in zip(targets, bounds[1:]):
        chunk = data[a:b]
        nul = chunk.find(b"\x00")
        body = chunk[:nul] if nul > 0 else b""
        if nul > 0 and all(32 <= c < 127 for c in body) and not any(chunk[nul + 1 :]):
            out.append((a, "str", body.decode("ascii")))
        else:
            out.append((a, "raw", chunk.hex()))
    return out


def unpack_bytes(raw: bytes) -> str:
    extra, secs = parse_container(raw)
    data, dmap, pool = secs[1], secs[0x10], secs[0x32]
    if len(data) < _ROOT:
        raise ValueError(f"Datos CSDBIN truncados ({len(data)} bytes)")
    names = _cstrs(pool)
    ptrs = [u32(dmap, i) for i in range(0, len(dmap), 8)]
    targets = sorted({u32(data, off) for off in ptrs})
    kind = {a: (t, v) for a, t, v in _pieces(data, targets)}
    lines = [f"id = {extra.hex()}"]
    if names:
        lines.append(f"sound_class = {fmt_str(names[0])}")
    if len(names) > 1:
        lines.append(f"mix_set = {fmt_str(names[1])}")
    links = []
    for off in sorted(ptrs, key=lambda off: (u32(data, off), off)):
        t, v = kind[u32(data, off)]
        text = fmt_str(v) if t == "str" else "blob:" + v
        links.append(f"{_link_name(off, t == 'str')} = {text}")
    if links:
        lines += ["", "[links]", *links]
    skip = {o for off in ptrs for o in (off, off + 4)}
    fields = []
    for off in range(0, _ROOT, 4):
        if off in skip or data[off : off + 4] == b"\x00\x00\x00\x00":
            continue
        fields.append(f"off_{off:04x} = {fmt_slot(data[off:off + 4])}")
    if fields:
        lines += ["", "[fields]", *fields]
    return "\n".join(lines) + "\n"


def _parse_text(text: str) -> dict:
    meta, section = {}, None
    kv = {"links": {}, "sounds": {}, "blobs": {}, "fields": {}}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if section in kv:
            kv[section][key] = value
        else:
            meta[key] = value
    return {**meta, **kv}


def pack_bytes(text: str) -> bytes:
    t = _parse_text(text)
    data = bytearray(_ROOT)
    for name, value in t["fields"].items():
        m = _OFF_KEY.match(name)
        if not m:
            raise ValueError(f"Campo desconocido: {name}")
        off = int(m.group(1), 16)
        slot = parse_slot(value)
        data[off : off + len(slot)] = slot
    items = []
    for name, value in {**t["sounds"], **t["blobs"], **t["links"]}.items():
        if value.startswith("blob:"):
            items.append((_link_off(name, False), "raw", value[5:]))
        else:
            items.append((_link_off(name, True), "str", parse_str(value)))
    heap = bytearray()
    placed: dict[tuple, int] = {}
    for off, kind, payload in items:
        key = (kind, payload)
        if key not in placed:
            placed[key] = _ROOT + len(heap)
            if kind == "str":
                heap += payload.encode("latin-1") + b"\x00"
            else:
                heap += bytes.fromhex(payload)
        struct.pack_into("<I", data, off, placed[key])
    dmap = b"".join(p32(off) + p32(0) for off, _, _ in sorted(items))
    pool = bytearray()
    for key in ("sound_class", "mix_set"):
        if key in t:
            pool += parse_str(t[key]).encode("latin-1") + b"\x00"
    extra = bytes.fromhex(t["id"]) if t.get("id") else b"\x00" * 8
    if len(extra) != 8:
        raise ValueError("El campo 'id' debe tener 8 bytes (16 caracteres hexadecimales)")
    return build_container(extra, bytes(data + heap), bytes(dmap), bytes(pool))


def unpack_file(src: Path, dst: Path | None = None) -> Path:
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".csd")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(unpack_bytes(src.read_bytes()), encoding="utf-8")
    return dst


def pack_file(src: Path, dst: Path | None = None) -> Path:
    src = Path(src)
    dst = Path(dst) if dst else src.with_suffix(".csdbin")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(pack_bytes(src.read_text(encoding="utf-8")))
    return dst


# ============================================================================
# Modelo estructurado de alto nivel para la Interfaz Gráfica (GUI)
# ============================================================================

SOUND_SLOT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Motor Jugador (Principal)",
        [
            ("engine_bank", "Banco de Motor (.bank)"),
            ("engine_ext", "Evento Motor Exterior"),
            ("engine_int", "Evento Motor Interior"),
            ("exhaust_ext", "Evento Escape Exterior"),
            ("exhaust_int", "Evento Escape Interior"),
            ("starter", "Evento Motor de Arranque (Starter)"),
            ("starting", "Evento Encendido (Starting)"),
            ("mix_snapshot", "Snapshot de Mezcla FMOD"),
        ],
    ),
    (
        "Motor IA (Oponentes)",
        [
            ("ai_engine_bank", "Banco Motor IA (.bank)"),
            ("ai_engine_ext", "Evento IA Exterior"),
            ("ai_engine_int", "Evento IA Interior"),
            ("ai_engine_live", "Evento IA Live"),
        ],
    ),
    (
        "Transmisión, Turbo y Limitador",
        [
            ("transmission_bank", "Banco Transmisión (.bank)"),
            ("transmission", "Evento Transmisión (Whine)"),
            ("gearshift_bank", "Banco Cambios de Marcha (.bank)"),
            ("gearshift_up", "Evento Subir Marcha (Upshift)"),
            ("gearshift_down", "Evento Bajar Marcha (Downshift)"),
            ("gearshift_grind", "Evento Rascar Marcha (Grind)"),
            ("gearbox", "Evento Caja de Cambios"),
            ("shift_pop", "Evento Petardeo al Cambiar (Shift Pop)"),
            ("turbo_bank", "Banco Turbo (.bank)"),
            ("turbo", "Evento Compresor / Turbo"),
            ("turbo_dump", "Evento Válvula Descarga (BOV / Wastegate)"),
            ("rev_limiter_bank", "Banco Limitador RPM (.bank)"),
            ("rev_limiter", "Evento Limitador RPM"),
        ],
    ),
    (
        "Petardeos (Backfires) y Otros Efectos",
        [
            ("backfire_ext", "Backfire Exterior"),
            ("backfire_int", "Backfire Interior"),
            ("backfire_ai", "Backfire IA"),
            ("splutter_ext", "Splutter Exterior"),
            ("splutter_int", "Splutter Interior"),
            ("splutter_ai", "Splutter IA"),
            ("brakes", "Frenos (Chirrido)"),
            ("wind_bank", "Banco Viento (.bank)"),
            ("wind", "Evento Viento"),
            ("suspension", "Suspensión"),
            ("chassis", "Chasis (Impactos)"),
            ("chassis_loop", "Chasis (Loop vibración)"),
            ("scrape", "Roce de Carrocería"),
            ("bumpstop", "Topes de Suspensión"),
            ("abs", "Intervención ABS"),
            ("ignition_on", "Contacto Encendido"),
            ("ignition_off", "Contacto Apagado"),
        ],
    ),
]


@dataclass
class CsdbinDocument:
    """Representación editable en memoria de un archivo .csdbin / .csd."""

    doc_id: str = "0000000000000000"
    sound_class: str = ""
    mix_set: str = ""
    links: dict[str, str] = field(default_factory=dict)  # valores des-escapados o "blob:<hex>"
    fields: dict[str, str] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> "CsdbinDocument":
        path = Path(path)
        raw = path.read_bytes()
        if raw[:4] == MAGIC:
            text = unpack_bytes(raw)
        else:
            text = raw.decode("utf-8")
        doc = cls.from_csd_text(text)
        doc.source_path = path
        return doc

    @classmethod
    def from_csd_text(cls, text: str) -> "CsdbinDocument":
        parsed = _parse_text(text)
        doc = cls(
            doc_id=parsed.get("id", "0000000000000000"),
            sound_class=parse_str(parsed["sound_class"]) if "sound_class" in parsed else "",
            mix_set=parse_str(parsed["mix_set"]) if "mix_set" in parsed else "",
        )
        merged_links = {**parsed["sounds"], **parsed["blobs"], **parsed["links"]}
        for k, v in merged_links.items():
            if v.startswith("blob:"):
                doc.links[k] = v
            else:
                doc.links[k] = parse_str(v)
        doc.fields = dict(parsed["fields"])
        return doc

    def to_csd_text(self) -> str:
        lines = [f"id = {self.doc_id}"]
        if self.sound_class:
            lines.append(f"sound_class = {fmt_str(self.sound_class)}")
        if self.mix_set:
            lines.append(f"mix_set = {fmt_str(self.mix_set)}")
        if self.links:
            lines.append("")
            lines.append("[links]")
            for k, v in self.links.items():
                if v == "":
                    continue
                if v.startswith("blob:"):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f"{k} = {fmt_str(v)}")
        if self.fields:
            lines.append("")
            lines.append("[fields]")
            for k, v in self.fields.items():
                lines.append(f"{k} = {v}")
        return "\n".join(lines) + "\n"

    def to_csdbin_bytes(self) -> bytes:
        return pack_bytes(self.to_csd_text())

    def save_csdbin(self, dst: Path) -> Path:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(self.to_csdbin_bytes())
        return dst

    def save_csd(self, dst: Path) -> Path:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(self.to_csd_text(), encoding="utf-8")
        return dst

    def get_emitter_xyz(self) -> tuple[float, float, float]:
        x = float(self.fields.get("off_0228", "0.0"))
        y = float(self.fields.get("off_022c", "0.0"))
        z = float(self.fields.get("off_0230", "0.0"))
        return (x, y, z)

    def set_emitter_xyz(self, x: float, y: float, z: float) -> None:
        for key, val in (("off_0228", x), ("off_022c", y), ("off_0230", z)):
            if abs(float(val)) < 1e-7 and key not in self.fields:
                continue
            self.fields[key] = fmt_f32(float(val))
