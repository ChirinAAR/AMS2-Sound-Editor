"""Inspector, extractor y constructor de bancos FMOD Studio (.bank / RIFF FEV + FSB5)."""
from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_dsp import (
    AudioBuffer,
    decode_fadpcm,
    encode_fadpcm,
    write_wav_bytes,
    write_wav_file,
)
from .fmod_stdt import parse_stdt_all_paths

# Mapas de constantes FSB5
FSB5_MODE_PCM8 = 1
FSB5_MODE_PCM16 = 2
FSB5_MODE_PCM24 = 3
FSB5_MODE_PCM32 = 4
FSB5_MODE_PCMFLOAT = 5
FSB5_MODE_VORBIS = 15
FSB5_MODE_FADPCM = 16

FSB5_MODE_NAMES = {
    1: "PCM8",
    2: "PCM16",
    3: "PCM24",
    4: "PCM32",
    5: "PCMFLOAT",
    15: "VORBIS",
    16: "FADPCM",
}

FREQ_ID_TO_HZ = {
    1: 8000,
    2: 11000,
    3: 11025,
    4: 16000,
    5: 22050,
    6: 24000,
    7: 32000,
    8: 44100,
    9: 48000,
    10: 96000,
}
HZ_TO_FREQ_ID = {hz: fid for fid, hz in FREQ_ID_TO_HZ.items()}

CHAN_ID_TO_COUNT = {0: 1, 1: 2, 2: 6, 3: 8}
COUNT_TO_CHAN_ID = {cnt: cid for cid, cnt in CHAN_ID_TO_COUNT.items()}

# Buses padre conocidos de Master.bank en Automobilista 2 (Reiza Studios)
AMS2_PARENT_BUS_ROLES = {
    "{c1986fb5-b329-4565-8342-b223857006d5}": "EngineExt",
    "{a8959bf6-0b54-43ea-9d9a-06a8831c40d4}": "EngineInt",
    "{ffe57d5c-46ed-4b06-86d5-8fa6f8ffdf07}": "Exhaust",
    "{ffefac1b-0333-46ad-9d94-c7ae1494451f}": "Startup",
}


def bytes_to_guid_str(b: bytes) -> str:
    """Convierte 16 bytes FMOD_GUID a cadena '{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}'."""
    d1, d2, d3 = struct.unpack_from("<IHH", b, 0)
    return f"{{{d1:08x}-{d2:04x}-{d3:04x}-{b[8:10].hex()}-{b[10:16].hex()}}}"


def guid_str_to_bytes(s: str) -> bytes:
    """Convierte '{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}' a 16 bytes FMOD_GUID."""
    clean = s.strip().strip("{}").replace("-", "")
    if len(clean) != 32:
        raise ValueError(f"GUID inválido: {s}")
    d1 = int(clean[0:8], 16)
    d2 = int(clean[8:12], 16)
    d3 = int(clean[12:16], 16)
    tail = bytes.fromhex(clean[16:32])
    return struct.pack("<IHH", d1, d2, d3) + tail


def iter_riff_chunks(buf: bytes | bytearray, start: int, end: int):
    """Itera sub-chunks RIFF devolviendo (chunk_id_4B, payload_start, payload_end, header_pos)."""
    pos = start
    while pos + 8 <= end:
        cid = bytes(buf[pos : pos + 4])
        csz = struct.unpack_from("<I", buf, pos + 4)[0]
        ps = pos + 8
        pe = min(end, ps + csz)
        yield cid, ps, pe, pos
        pos = pe + (csz & 1)


_GUID_TXT_LINE_RE = re.compile(r"^\s*(\{[0-9a-fA-F\-]{36}\})\s+(.+?)\s*$")
_NON_ENGINE_EVENT_ROLES = frozenset(
    {"Turbo", "Transmission", "Tyres", "Wind", "Gearshift", "GearshiftInt", "GearshiftExt", "FX"}
)
_ENGINE_EVENT_ROLES = frozenset({"EngineInt", "EngineExt", "AuxInt", "Exhaust", "Engine"})


def parse_guids_txt(source: str | Path) -> dict[bytes, str]:
    """Parsea un archivo o texto `GUIDs.txt` de FMOD Studio (`{guid} event:/...`) a `{guid_bytes: ruta}`."""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8-sig", errors="replace")
    else:
        text = str(source)
    out: dict[bytes, str] = {}
    for line in text.splitlines():
        m = _GUID_TXT_LINE_RE.match(line)
        if not m:
            continue
        try:
            gb = guid_str_to_bytes(m.group(1))
        except ValueError:
            continue
        out[gb] = m.group(2).strip()
    return out


def find_companion_guids_txt(
    bank_path: Path,
    known_bank_guids: set[bytes] | None = None,
) -> tuple[Path | None, dict[bytes, str]]:
    """Busca automáticamente un archivo `GUIDs.txt` compañero en la carpeta del `.bank` (o carpeta padre `sfx`)."""
    bank_path = Path(bank_path)
    parent = bank_path.parent
    if not parent.is_dir():
        return None, {}

    stem_low = bank_path.stem.lower()
    candidates: list[Path] = []
    seen_paths: set[Path] = set()

    def _add(p: Path) -> None:
        rp = p.resolve()
        if rp not in seen_paths and p.is_file():
            seen_paths.add(rp)
            candidates.append(p)

    # 1. Nombres específicos del auto en la misma carpeta
    for pat in (f"{bank_path.stem}.guids.txt", f"{bank_path.stem}_guids.txt", f"{bank_path.stem} GUIDs.txt"):
        _add(parent / pat)

    # 2. Cualquier GUIDs.txt o *guid*.txt en la misma carpeta
    try:
        txt_files = sorted(parent.glob("*.txt"), key=lambda f: (0 if f.name.lower() == "guids.txt" else 1, f.name.lower()))
    except OSError:
        txt_files = []
    for tf in txt_files:
        nlow = tf.name.lower()
        if nlow == "guids.txt" or "guid" in nlow or nlow == f"{stem_low}.txt":
            _add(tf)

    # 3. Carpeta padre si estamos en una subcarpeta tipo sfx/
    if parent.parent != parent and parent.parent.is_dir():
        for tf_name in ("GUIDs.txt", "guids.txt", f"{bank_path.stem}.guids.txt"):
            _add(parent.parent / tf_name)

    for cand in candidates:
        try:
            if cand.stat().st_size > 8 * 1024 * 1024:
                continue
            parsed = parse_guids_txt(cand)
        except OSError:
            continue
        if not parsed:
            continue
        if known_bank_guids is not None:
            if not any(g in known_bank_guids for g in parsed):
                continue
        return cand, parsed

    return None, {}


def infer_rpm_from_name(name: str) -> float | None:
    """Extrae una referencia de RPM plausible desde el nombre de muestra (ej. 'z4gt3_on_7500', '6702.75')."""
    # Buscar números entre 800 y 22000 (ignorando prefijos conocidos de modelos de auto como 488, 911, 458, 991, 992, 720)
    candidates = re.findall(r"(?<!\d)(\d{3,5}(?:\.\d+)?)(?!\d)", name)
    valid = []
    for idx, c in enumerate(candidates):
        val = float(c)
        if 750.0 <= val <= 22000.0:
            # Evitar confundir el modelo del auto al inicio (ej. "911_Load_1", "992_Idle") con RPM si < 1050
            if idx == 0 and val < 1000.0 and name.lstrip().startswith(c):
                continue
            valid.append(val)
    return valid[-1] if valid else None


def _classify_event_role_from_path_or_heuristics(
    idx: int,
    ev_path: str,
    bus_path: str,
    plist: list[tuple[str, float, float]],
    snames: str,
    slist_names: list[str],
) -> str:
    """Determina el rol semántico de un evento FMOD usando `GUIDs.txt`/`STDT` o heurística binaria 3D/2D."""
    p_low = ev_path.lower().strip()
    b_low = bus_path.lower().strip()

    if p_low:
        if p_low.startswith(("event:/collisions/", "event:/surfaces/", "event:/common/", "event:/showrooms/")):
            return "FX"
        leaf = p_low.rsplit("/", 1)[-1]
        if leaf in ("engine_int", "int_engine") or ("engine" in leaf and "int" in leaf and "ai" not in leaf):
            return "EngineInt"
        if leaf in ("engine_ext", "ext_engine") or ("engine" in leaf and "ext" in leaf and "ai" not in leaf):
            return "EngineExt"
        if "starter" in leaf or "crank" in leaf:
            return "Starter"
        if "starting" in leaf or "ignition" in leaf or "startup" in leaf:
            return "Starting"
        if "backfire" in leaf or "afterfire" in leaf or "splutter" in leaf or leaf.startswith("pop"):
            return "BackfireInt" if "int" in leaf else "BackfireExt"
        if "limiter" in leaf:
            return "Limiter"
        if "turbo" in leaf or "supercharger" in leaf or "bov" in leaf:
            return "Turbo"
        if "transmission" in leaf or "gearbox" in leaf or "drivetrain" in leaf:
            return "Transmission"
        if leaf.startswith("gear") or "gearshift" in leaf or "gearchange" in leaf:
            if "int" in leaf:
                return "GearshiftInt"
            if "ext" in leaf or "ex" in leaf:
                return "GearshiftExt"
            return "Gearshift"
        if any(k in leaf for k in ("skid", "wheel", "tyre", "tire", "brake")):
            return "Tyres"
        if "wind" in leaf:
            return "Wind"
        if any(
            k in leaf
            for k in ("bodywork", "chassis", "suspension", "door", "horn", "traction", "abs", "dirt", "damage", "hit", "scrape")
        ):
            return "FX"

    if b_low:
        if "engine_int" in b_low:
            return "EngineInt"
        if "engine_ext" in b_low:
            return "EngineExt"
        if "backfire_int" in b_low:
            return "BackfireInt"
        if "backfire_ext" in b_low:
            return "BackfireExt"
        if "turbo" in b_low:
            return "Turbo"
        if "transmission" in b_low:
            return "Transmission"
        if "gear" in b_low:
            if "int" in b_low:
                return "GearshiftInt"
            if "ext" in b_low:
                return "GearshiftExt"
            return "Gearshift"
        if "skid" in b_low or "wheel" in b_low or "surface" in b_low:
            return "Tyres"
        if "wind" in b_low:
            return "Wind"
        if any(k in b_low for k in ("bodywork", "collision", "door", "horn", "traction")):
            return "FX"

    # Heurística binaria cuando no se dispone de GUIDs.txt
    pnames = {p[0].lower() for p in plist}
    has_3d = ("distance" in pnames) or ("event cone angle" in pnames)
    if "boost" in pnames or "bov" in pnames or "bov_decay" in pnames:
        return "Turbo"
    if "drivetrain_speed" in pnames:
        return "Transmission"
    if "air_pressure" in pnames:
        return "Wind"
    if "susp_travel_speed" in pnames:
        return "FX"
    if "inflation" in pnames or "suspension_damage" in pnames:
        return "Tyres"
    if "rpms" in pnames or "rpm" in pnames:
        if has_3d or ("ext" in snames and "int" not in snames):
            return "EngineExt"
        if "int" in snames or not has_3d or any("idle_2" in nm for nm in slist_names):
            return "EngineInt"
        return "Engine"
    if "backfire" in snames or "pop" in snames:
        return "BackfireExt" if (has_3d or "ext" in snames) else "BackfireInt"
    if "limiter" in snames:
        return "Limiter"
    if any(k in snames for k in ("gearup", "geardn", "upshift", "gearshift", "gearchange")):
        if has_3d or ("ext" in snames and "int" not in snames):
            return "GearshiftExt"
        if "int" in snames or not has_3d:
            return "GearshiftInt"
        return "Gearshift"
    if "skid" in snames or "tyre" in snames or "tire" in snames:
        return "Tyres"
    if any(k in snames for k in ("door", "horn", "chassis", "traction", "tc")):
        return "FX"
    return f"Event_{idx:02d}"


def classify_sample_semantics(
    name: str,
    event_roles: list[str],
    rpm_boxes: list[tuple[float, float]],
    thr_boxes: list[tuple[float, float]],
    is_timeline: bool,
    thr_curve_hint: str | None = None,
    autopitch_ref_rpm: float | None = None,
    tcut_boxes: list[tuple[float, float]] | None = None,
    shift_state_boxes: list[tuple[float, float]] | None = None,
    is_looping: bool = False,
) -> tuple[str, str]:
    """Clasifica una muestra en (perspectiva, rol_carga).

    perspectiva in {'int', 'ext', 'startup', 'aux', 'fx'}
    rol_carga   in {'idle', 'on', 'off', 'throttle_cut', 'upshift', 'downshift', 'pops', 'starter', 'starting', 'null', 'fx'}
    """
    low = name.strip().lower()
    if low in ("null", "silence", "empty") or low.startswith("null_"):
        return ("aux", "null")

    role_set = set(event_roles)
    in_engine_evt = bool(role_set & _ENGINE_EVENT_ROLES)

    # 0. Si la muestra está conectada directamente al parámetro ThrottleCut en AMS2
    if tcut_boxes:
        if any(r in ("EngineInt", "AuxInt") for r in role_set) and "EngineExt" not in role_set:
            return ("int", "throttle_cut")
        if "EngineExt" in role_set and not any(r in ("EngineInt", "AuxInt") for r in role_set):
            return ("ext", "throttle_cut")
        return ("int" if ("int" in low or low.startswith("in_")) else "ext", "throttle_cut")

    # 1. Si la muestra pertenece EXCLUSIVAMENTE a eventos de efectos/transmisión/turbo/chasis/cambios
    if role_set and not in_engine_evt:
        if "Starter" in role_set:
            return ("startup", "starter")
        if "Starting" in role_set:
            return ("startup", "starting")
        if "BackfireInt" in role_set and "BackfireExt" not in role_set:
            return ("int", "pops")
        if "BackfireExt" in role_set:
            return ("ext", "pops")
        if "Limiter" in role_set:
            return ("int" if "int" in low else "ext", "pops")
        if any(r in ("GearshiftInt", "GearshiftExt", "Gearshift") for r in role_set):
            if "int" in low or low.startswith("in_"):
                g_persp = "int"
            elif "ext" in low or "_ex" in low or low.endswith(" ex"):
                g_persp = "ext"
            elif "GearshiftInt" in role_set and "GearshiftExt" not in role_set:
                g_persp = "int"
            elif "GearshiftExt" in role_set and "GearshiftInt" not in role_set:
                g_persp = "ext"
            else:
                g_persp = "int"

            is_explicit_down = any(k in low for k in ("geardn", "gear_dn", "downshift"))
            is_explicit_up = any(
                k in low
                for k in (
                    "upshift",
                    "gearup",
                    "gear_up",
                    "throttle_cut",
                    "throttlecut",
                    "shift_cut",
                    "shiftcut",
                    "flatshift",
                    "gearchange",
                    "gearshift",
                )
            )
            has_up_state = (
                bool(shift_state_boxes)
                and any(v1 > 0.55 for _, v1 in shift_state_boxes)
                and not all(v1 <= 0.49 for _, v1 in shift_state_boxes)
            )
            if not is_explicit_down and (is_explicit_up or has_up_state):
                return (g_persp, "throttle_cut")
            if is_explicit_down or (shift_state_boxes and all(v1 <= 0.49 for _, v1 in shift_state_boxes)):
                return (g_persp, "downshift")
            return ("fx", "fx")
        if role_set <= _NON_ENGINE_EVENT_ROLES:
            return ("fx", "fx")

    # 2. Detectar muestras one-shot de Throttle Cut / Upshift o efectos no-motor por nombre (WAVs sueltos / huérfanos)
    if not in_engine_evt:
        if not is_looping and not any(k in low for k in ("geardn", "gear_dn", "downshift")):
            if any(
                k in low
                for k in (
                    "throttle_cut",
                    "throttlecut",
                    "shift_cut",
                    "shiftcut",
                    "flatshift",
                    "upshift",
                    "gearup",
                    "gear_up",
                )
            ):
                g_persp = "int" if ("int" in low or low.startswith("in_")) else "ext"
                return (g_persp, "throttle_cut")
        if any(
            k in low
            for k in (
                "skid",
                "tyre",
                "tire",
                "wind",
                "door",
                "bodywork",
                "chassis",
                "gravel",
                "road noise",
                "brake",
                "traction",
                "_tc",
                "flutter",
                "whistle",
                "turbo",
                "bov",
                "transmission",
                "tranny",
                "gearbox",
                "gearshift",
                "gearchange",
                "whine",
                "horn",
            )
        ):
            return ("fx", "fx")

    # 3. Determinar perspectiva Interior vs Exterior (priorizando el evento real FMOD si es exclusivo)
    if any(r in ("EngineInt", "AuxInt", "BackfireInt") for r in role_set) and not any(
        r in ("EngineExt", "BackfireExt") for r in role_set
    ):
        persp = "int"
    elif any(r in ("EngineExt", "BackfireExt") for r in role_set) and not any(
        r in ("EngineInt", "AuxInt", "BackfireInt") for r in role_set
    ):
        persp = "ext"
    elif ("int" in low and "fill_int" not in low) or low.startswith("in_"):
        persp = "int"
    elif "ext" in low or "_ex_" in low or low.startswith("ex_"):
        persp = "ext"
    else:
        persp = "ext"

    # 4. Arranque / encendido (solo si el evento es de arranque o no está en otro evento no-motor)
    if "Starter" in role_set or ("starter" in low and not role_set):
        return ("startup", "starter")
    if "Starting" in role_set or (("starting" in low or "ignition" in low or "startup" in low) and not in_engine_evt):
        return ("startup", "starting")

    # 5. Limitador / petardeos / cambios dentro de eventos de motor o sueltos
    if "limiter" in low or "_lim" in low or "gt3lim" in low:
        return (persp, "pops")
    if any(k in low for k in ("backfire", "pops", "pop_", "afterfire", "crackle")):
        return (persp, "pops")
    if any(k in low for k in ("throttle_cut", "throttlecut", "shift_cut", "shiftcut", "flatshift")) and not is_looping:
        return (persp, "throttle_cut")
    if "gearup" in low or "upshift" in low:
        if "on_upshift" not in low and not is_looping:
            return (persp, "throttle_cut")
    if "geardn" in low or ("downshift" in low and "off_downshift" not in low):
        return (persp, "downshift")

    # 6. Rol de carga (Idle / Off-Coast / On-Load): priorizar la curva real de acelerador FMOD sobre nombres engañosos
    if "idle" in low or "vlow" in low:
        role = "idle"
    elif thr_curve_hint in ("on", "off"):
        if rpm_boxes:
            avg_start = sum(b[0] for b in rpm_boxes) / len(rpm_boxes)
            avg_end = sum(b[1] for b in rpm_boxes) / len(rpm_boxes)
            has_band_word = any(w in low for w in ("low", "mid", "high", "load", "coast", "on", "off"))
            if (
                not has_band_word
                and avg_start < 500.0
                and avg_end <= 3000.0
                and (autopitch_ref_rpm is None or autopitch_ref_rpm <= 2650.0)
            ):
                role = "idle"
            else:
                role = thr_curve_hint
        else:
            role = thr_curve_hint
    elif any(k in low for k in ("off", "coast", "decel", "lift")) or re.search(r"(?:^|[_\s])c\d(?:$|[_\s])", low):
        role = "off"
    elif any(k in low for k in ("on", "load", "accel", "power")) or re.search(r"(?:^|[_\s])l\d(?:$|[_\s])", low):
        role = "on"
    else:
        if rpm_boxes:
            avg_end = sum(b[1] for b in rpm_boxes) / len(rpm_boxes)
            avg_start = sum(b[0] for b in rpm_boxes) / len(rpm_boxes)
            if avg_start < 500.0 and avg_end <= 3300.0:
                role = "idle"
            elif thr_boxes:
                avg_thr = sum(0.5 * (t[0] + t[1]) for t in thr_boxes) / len(thr_boxes)
                role = "off" if avg_thr < 0.35 else "on"
            else:
                role = "on"
        elif is_timeline and not role_set:
            role = "starter"
        elif not role_set and autopitch_ref_rpm is None and infer_rpm_from_name(name) is None:
            # Muestra huérfana sin evento ni RPM en el banco (ej. efectos residuales en FSB5)
            return ("fx", "fx")
        else:
            role = "on"

    return (persp, role)


def _detect_aux_engine_layer(
    name: str,
    perspective: str,
    load_role: str,
    near_distance_db: float,
    has_rpm_autopitch: bool,
) -> bool:
    """Detecta capas auxiliares/lejanas (Far, Flyby, Passby, Wide, Vibration, Clank, Turbo) dentro de eventos de motor."""
    if perspective not in ("int", "ext") or load_role not in ("on", "off", "idle"):
        return False
    if near_distance_db <= -12.0:
        return True
    low = name.strip().lower().replace("no turbo", "").replace("no_turbo", "").replace("noturbo", "")
    aux_tokens = (
        "far",
        "flyby",
        "passby",
        "wide",
        "bounce",
        "vibration",
        "subwave",
        "clank",
        "grunge",
        "noise test",
        "turbo",
    )
    if any(tok in low for tok in aux_tokens):
        return True
    return False


@dataclass
class FsbSampleSlot:
    """Metadatos completos de una muestra dentro de un sub-banco FSB5 + grafo FMOD."""

    global_index: int
    subbank_index: int
    sample_index: int
    name: str
    mode: int
    sample_rate: int
    channels: int
    pcm_samples: int
    data_offset: int
    data_bytes: bytes
    loop: tuple[int, int] | None = None
    peak_volume: float = 1.0
    wav_guid: bytes | None = None
    event_guids: list[str] = field(default_factory=list)
    event_roles: list[str] = field(default_factory=list)
    event_paths: list[str] = field(default_factory=list)
    rpm_boxes: list[tuple[float, float]] = field(default_factory=list)
    audible_rpm_boxes: list[tuple[float, float]] = field(default_factory=list)
    thr_boxes: list[tuple[float, float]] = field(default_factory=list)
    tcut_boxes: list[tuple[float, float]] = field(default_factory=list)
    shift_state_boxes: list[tuple[float, float]] = field(default_factory=list)
    autopitch_ref_rpm: float | None = None
    has_rpm_autopitch: bool = False
    inst_infinite_loop: bool = False
    near_distance_db: float = 0.0
    thr_curve_hint: str | None = None
    is_aux_layer: bool = False
    is_muted_donor_slot: bool = False
    perspective: str = "ext"
    load_role: str = "on"

    @property
    def codec_name(self) -> str:
        return FSB5_MODE_NAMES.get(self.mode, f"MODE_{self.mode}")

    @property
    def duration_sec(self) -> float:
        return self.pcm_samples / max(1, self.sample_rate)

    @property
    def can_decode_audio(self) -> bool:
        return self.mode in (
            FSB5_MODE_PCM8,
            FSB5_MODE_PCM16,
            FSB5_MODE_PCM24,
            FSB5_MODE_PCM32,
            FSB5_MODE_PCMFLOAT,
            FSB5_MODE_FADPCM,
        )

    @property
    def effective_rpm(self) -> float | None:
        if self.autopitch_ref_rpm and self.autopitch_ref_rpm > 100.0:
            return self.autopitch_ref_rpm
        from_name = infer_rpm_from_name(self.name)
        if from_name:
            return from_name
        boxes = self.audible_rpm_boxes or self.rpm_boxes
        if boxes:
            return sum(0.5 * (a + b) for a, b in boxes) / len(boxes)
        return None

    @property
    def rpm_summary(self) -> str:
        boxes = self.audible_rpm_boxes or self.rpm_boxes
        if boxes:
            lo = min(a for a, _ in boxes)
            hi = max(b for _, b in boxes)
            return f"{lo:.0f}–{hi:.0f}"
        if self.autopitch_ref_rpm and self.autopitch_ref_rpm > 100.0:
            return f"~{self.autopitch_ref_rpm:.0f}"
        fn = infer_rpm_from_name(self.name)
        return f"~{fn:.0f}" if fn else "—"


@dataclass
class RpmBandInfo:
    """Información de una caja/banda de RPM editable dentro de un evento de motor FMOD."""

    band_id: str
    pml_guid: bytes
    event_guid: bytes
    box_index: int
    inst_guid: bytes
    route_bus_guid: bytes
    event_role: str
    perspective: str
    load_role: str
    donor_slot_indices: list[int]
    donor_label: str
    rpm_min: float
    rpm_max: float
    fade_in_end: float
    fade_out_start: float
    fade_in_curv_guid: bytes | None = None
    fade_out_curv_guid: bytes | None = None
    bus_rpm_cutoff: tuple[bytes, float, float] | None = None
    is_muted_bus: bool = False
    is_fill_layer: bool = False


@dataclass
class FmodEventInfo:
    """Información resumida de un evento FMOD dentro del banco."""

    index: int
    guid_bytes: bytes
    guid_str: str
    role: str
    csd_slot_hint: str
    parent_bus_guid: str
    path: str = ""
    parameters: list[tuple[str, float, float]] = field(default_factory=list)
    linked_samples: list[str] = field(default_factory=list)
    ibus_guid_bytes: bytes = b""


@dataclass
class FsbSubBank:
    """Sub-banco FSB5 crudo dentro de un chunk 'SND '."""

    index: int
    version: int
    mode: int
    flags: int
    hash16: bytes
    dummy8: bytes
    samples: list[FsbSampleSlot]
    raw_fsb_bytes: bytes


class FmodBank:
    """Representación en memoria de un archivo FMOD Studio .bank."""

    def __init__(
        self,
        path: Path | None = None,
        raw: bytes | None = None,
        guids_path: Path | None = None,
        external_guids: dict[bytes, str] | None = None,
    ):
        if raw is None and path is not None:
            raw = Path(path).read_bytes()
        if raw is None:
            raise ValueError("Se requiere 'path' o 'raw' para cargar FmodBank.")
        if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"FEV ":
            raise ValueError("El archivo no es un banco FMOD Studio válido (RIFF 'FEV ').")

        self.source_path: Path | None = Path(path) if path else None
        self.guids_source_path: Path | None = Path(guids_path) if guids_path else None
        self._explicit_external_guids: dict[bytes, str] | None = (
            dict(external_guids)
            if external_guids is not None
            else (parse_guids_txt(Path(guids_path)) if guids_path is not None else None)
        )
        self.matched_external_guids_count: int = 0
        self.raw: bytes = raw
        self.file_version: int = 0
        self.compat_version: int = 0
        self.bank_guid: bytes = b"\x00" * 16
        self.sndh_version: int = 0x00080005
        self.stdt_bytes: bytes = b""
        self.subbanks: list[FsbSubBank] = []
        self.slots: list[FsbSampleSlot] = []
        self.events: list[FmodEventInfo] = []
        self.rpm_bands: list[RpmBandInfo] = []
        self.has_active_throttle_cut: bool = False
        self.empty_throttle_cut_events: list[str] = []
        self._parse()

    def _parse(self) -> None:
        raw = self.raw
        proj_bounds: tuple[int, int] | None = None
        snd_chunks: list[tuple[int, int]] = []

        for cid, ps, pe, _ in iter_riff_chunks(raw, 12, len(raw)):
            if cid == b"FMT " and pe - ps >= 8:
                self.file_version, self.compat_version = struct.unpack_from("<II", raw, ps)
            elif cid == b"LIST" and raw[ps : ps + 4] == b"PROJ":
                proj_bounds = (ps + 4, pe)
            elif cid == b"SND ":
                snd_chunks.append((ps, pe))

        # 1. Parsear todos los sub-bancos FSB5
        g_idx = 0
        for sb_idx, (ps, pe) in enumerate(snd_chunks):
            fpos = raw.find(b"FSB5", ps, pe)
            if fpos < 0:
                continue
            sb, g_idx = self._parse_fsb5(raw[fpos:pe], sb_idx, g_idx)
            self.subbanks.append(sb)
            self.slots.extend(sb.samples)

        if proj_bounds is None:
            return

        # 2. Parsear el árbol PROJ para enlazar eventos, parámetros, curvas FMOD, Auto-Pitch y WAVS
        wavs_map: dict[bytes, tuple[int, int]] = {}
        # wais_map: wg_self -> (wg_target, ap_guid, ap_val, inf_loop, route_bus)
        wais_map: dict[bytes, tuple[bytes, bytes, float | None, bool, bytes]] = {}
        # muis_map: mg_self -> (children_guids, ap_guid, ap_val, inf_loop, route_bus)
        muis_map: dict[bytes, tuple[list[bytes], bytes, float | None, bool, bytes]] = {}
        ibus_parent_bytes: dict[bytes, bytes] = {}
        ibus_parent: dict[bytes, str] = {}
        prms_map: dict[bytes, tuple[str, float, float]] = {}
        ctrs_by_target: dict[bytes, list[tuple[bytes, bytes, int]]] = {}
        crvs_map: dict[bytes, list[tuple[float, float]]] = {}
        raw_evts: list[tuple[bytes, bytes, bytes]] = []  # (evt_guid, tln_guid, ibus_guid)
        pmls_list: list[tuple[bytes, bytes, bytes, list[tuple[bytes, float, float]]]] = []
        tlns_list: list[tuple[bytes, bytes, list[tuple[bytes, int, int]]]] = []

        def _parse_inst_chunk(inst_p: bytes) -> tuple[bytes, float | None, bool, bytes]:
            ap_g = b"\x00" * 16
            ap_v: float | None = None
            inf = False
            rbus = b"\x00" * 16
            if len(inst_p) >= 73:
                loop_cnt = struct.unpack_from("<i", inst_p, 24)[0]
                inf = loop_cnt == -1
                ap_g = bytes(inst_p[53:69])
                if ap_g != b"\x00" * 16:
                    cand_v = float(struct.unpack_from("<f", inst_p, 69)[0])
                    if 50.0 <= cand_v <= 30000.0:
                        ap_v = cand_v
            if len(inst_p) >= 99:
                rbus = bytes(inst_p[83:99])
            return ap_g, ap_v, inf, rbus

        def walk_proj(start: int, end: int) -> None:
            for cid, ps, pe, _ in iter_riff_chunks(raw, start, end):
                if cid == b"BNKI" and pe - ps >= 16:
                    self.bank_guid = raw[ps : ps + 16]
                elif cid == b"SNDH" and pe - ps >= 4:
                    self.sndh_version = struct.unpack_from("<I", raw, ps)[0]
                elif cid == b"STDT" and pe > ps:
                    self.stdt_bytes = raw[ps:pe]
                elif cid == b"IBSB" and pe - ps >= 34:
                    ig = raw[ps : ps + 16]
                    pbytes = raw[ps + 18 : ps + 34]
                    ibus_parent_bytes[ig] = pbytes
                    ibus_parent[ig] = bytes_to_guid_str(pbytes)
                elif cid == b"EVTB" and pe - ps >= 64:
                    eg = raw[ps : ps + 16]
                    tg = raw[ps + 32 : ps + 48]
                    ibg = raw[ps + 48 : ps + 64]
                    raw_evts.append((eg, tg, ibg))
                elif cid == b"PRMB" and pe - ps >= 31:
                    p = raw[ps:pe]
                    slen = struct.unpack_from("<H", p, 21)[0]
                    if 23 + slen + 8 <= len(p):
                        pname = p[23 : 23 + slen].decode("utf-8", "replace")
                        pmin, pmax = struct.unpack_from("<ff", p, 23 + slen)
                        prms_map[p[:16]] = (pname, pmin, pmax)
                elif cid == b"PMLB" and pe - ps >= 50:
                    p = raw[ps:pe]
                    nboxes = struct.unpack_from("<H", p, 48)[0] >> 1
                    boxes = []
                    if nboxes > 0 and len(p) >= 52:
                        stride = struct.unpack_from("<H", p, 50)[0]
                        if stride >= 24 and len(p) >= 52 + nboxes * stride:
                            for bi in range(nboxes):
                                boff = 52 + bi * stride
                                ig = p[boff : boff + 16]
                                v0, vlen = struct.unpack_from("<ff", p, boff + 16)
                                boxes.append((ig, float(v0), float(v0 + vlen)))
                    pmls_list.append((p[:16], p[16:32], p[32:48], boxes))
                elif cid == b"TLNB" and pe - ps >= 34:
                    p = raw[ps:pe]
                    nboxes = struct.unpack_from("<H", p, 32)[0] >> 1
                    boxes = []
                    if nboxes > 0 and len(p) >= 36:
                        stride = struct.unpack_from("<H", p, 34)[0]
                        if stride >= 24 and len(p) >= 36 + nboxes * stride:
                            for bi in range(nboxes):
                                boff = 36 + bi * stride
                                ig = p[boff : boff + 16]
                                t0, t1 = struct.unpack_from("<II", p, boff + 16)
                                boxes.append((ig, int(t0), int(t1)))
                    tlns_list.append((p[:16], p[16:32], boxes))
                elif cid == b"CTRL" and pe - ps >= 68:
                    p = raw[ps:pe]
                    prop = struct.unpack_from("<I", p, 64)[0]
                    ctrs_by_target.setdefault(p[16:32], []).append((p[32:48], p[48:64], prop))
                elif cid == b"CURV" and pe - ps >= 34:
                    p = raw[ps:pe]
                    cnt = struct.unpack_from("<H", p, 32)[0] >> 1
                    pts: list[tuple[float, float]] = []
                    if cnt > 0 and len(p) >= 36:
                        stride = struct.unpack_from("<H", p, 34)[0]
                        if stride >= 8 and len(p) >= 36 + cnt * stride:
                            for ki in range(cnt):
                                x, y = struct.unpack_from("<ff", p, 36 + ki * stride)
                                pts.append((float(x), float(y)))
                    crvs_map[p[:16]] = pts
                elif cid == b"WAV " and pe - ps >= 30:
                    p = raw[ps:pe]
                    _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", p, 16)
                    wavs_map[p[:16]] = (sb_i, smp_i)
                elif cid == b"LIST":
                    ltype = raw[ps : ps + 4]
                    if ltype == b"WAIT":
                        wg_self = None
                        wg_target = None
                        ap_g = b"\x00" * 16
                        ref_rpm: float | None = None
                        inf_loop = False
                        rbus = b"\x00" * 16
                        for scid, sps, spe, _ in iter_riff_chunks(raw, ps + 4, pe):
                            if scid == b"WAIB" and spe - sps >= 32:
                                wg_self = raw[sps : sps + 16]
                                wg_target = raw[sps + 16 : sps + 32]
                            elif scid == b"INST":
                                ap_g, ref_rpm, inf_loop, rbus = _parse_inst_chunk(raw[sps:spe])
                        if wg_self and wg_target:
                            wais_map[wg_self] = (wg_target, ap_g, ref_rpm, inf_loop, rbus)
                    elif ltype in (b"MUIT", b"SPIT"):
                        mg = None
                        children: list[bytes] = []
                        ap_g = b"\x00" * 16
                        ref_rpm = None
                        inf_loop = False
                        rbus = b"\x00" * 16
                        for scid, sps, spe, _ in iter_riff_chunks(raw, ps + 4, pe):
                            if scid in (b"MUIB", b"SPIB") and spe - sps >= 16:
                                mg = raw[sps : sps + 16]
                            elif scid == b"PLST" and spe - sps >= 10:
                                sp = raw[sps:spe]
                                nc = struct.unpack_from("<H", sp, 8)[0] >> 1
                                if nc > 0 and len(sp) >= 12:
                                    stride = struct.unpack_from("<H", sp, 10)[0]
                                    if stride >= 16:
                                        for ci in range(nc):
                                            coff = 12 + ci * stride
                                            if coff + 16 <= len(sp):
                                                children.append(sp[coff : coff + 16])
                            elif scid == b"INST":
                                ap_g, ref_rpm, inf_loop, rbus = _parse_inst_chunk(raw[sps:spe])
                        if mg:
                            muis_map[mg] = (children, ap_g, ref_rpm, inf_loop, rbus)
                    else:
                        walk_proj(ps + 4, pe)

        walk_proj(proj_bounds[0], proj_bounds[1])

        def _is_rpm_param(pg: bytes, ap_val: float | None) -> bool:
            if pg == b"\x00" * 16 or ap_val is None:
                return False
            pinfo = prms_map.get(pg)
            if pinfo is not None:
                return pinfo[0].strip().lower() in ("rpm", "rpms")
            # Si el parámetro proviene de un banco externo, exigir rango de RPM de motor plausible
            return ap_val >= 450.0

        # Asignar wav_guid y propiedades directas de WAIT
        slot_lookup: dict[tuple[int, int], FsbSampleSlot] = {
            (s.subbank_index, s.sample_index): s for s in self.slots
        }
        for wg, (sb_i, smp_i) in wavs_map.items():
            sl = slot_lookup.get((sb_i, smp_i))
            if sl is not None:
                sl.wav_guid = wg

        for _, (wg_target, ap_g, ref_rpm, inf_loop, _) in wais_map.items():
            loc = wavs_map.get(wg_target)
            if loc and loc in slot_lookup:
                sl = slot_lookup[loc]
                if ref_rpm is not None and _is_rpm_param(ap_g, ref_rpm) and sl.autopitch_ref_rpm is None:
                    sl.autopitch_ref_rpm = ref_rpm
                    sl.has_rpm_autopitch = True
                if inf_loop:
                    sl.inst_infinite_loop = True

        def _apply_chain_automation(sl: FsbSampleSlot, chain_targets: list[bytes]) -> None:
            near_db_sum = 0.0
            has_dist = False
            for tgt in chain_targets:
                if tgt == b"\x00" * 16:
                    continue
                for c_pg, c_cg, prop in ctrs_by_target.get(tgt, []):
                    if prop != 0:  # 0 = Volume (dB)
                        continue
                    pinfo = prms_map.get(c_pg)
                    if not pinfo:
                        continue
                    pnm = pinfo[0].strip().lower()
                    pts = crvs_map.get(c_cg, [])
                    if not pts:
                        continue
                    if pnm == "distance":
                        has_dist = True
                        near_db_sum += pts[0][1]
                    elif pnm in ("throttle", "load") and sl.thr_curve_hint is None and len(pts) >= 2:
                        y_lo = pts[0][1]
                        y_hi = pts[-1][1]
                        if y_hi - y_lo >= 8.0:
                            sl.thr_curve_hint = "on"
                        elif y_lo - y_hi >= 8.0:
                            sl.thr_curve_hint = "off"
            if has_dist and (sl.near_distance_db == 0.0 or near_db_sum > sl.near_distance_db):
                sl.near_distance_db = near_db_sum

        def resolve_inst_slots(
            ig: bytes,
            inherited_ap: tuple[bytes, float | None] = (b"\x00" * 16, None),
            inherited_inf: bool = False,
            inherited_bus: bytes = b"\x00" * 16,
            parent_chain: tuple[bytes, ...] = (),
        ) -> list[FsbSampleSlot]:
            if ig in wais_map:
                wg_t, ap_g, ap_v, inf_l, rbus = wais_map[ig]
                loc = wavs_map.get(wg_t)
                sl = slot_lookup.get(loc) if loc else None
                if sl is None:
                    return []
                eff_ap_g, eff_ap_v = (ap_g, ap_v) if ap_v is not None else inherited_ap
                if eff_ap_v is not None and _is_rpm_param(eff_ap_g, eff_ap_v) and sl.autopitch_ref_rpm is None:
                    sl.autopitch_ref_rpm = eff_ap_v
                    sl.has_rpm_autopitch = True
                if inf_l or inherited_inf:
                    sl.inst_infinite_loop = True
                eff_bus = rbus if rbus != b"\x00" * 16 else inherited_bus
                _apply_chain_automation(sl, list(parent_chain) + [ig, eff_bus])
                return [sl]
            if ig in muis_map:
                children, ap_g, ap_v, inf_l, rbus = muis_map[ig]
                eff_ap = (ap_g, ap_v) if ap_v is not None else inherited_ap
                eff_inf = inf_l or inherited_inf
                eff_bus = rbus if rbus != b"\x00" * 16 else inherited_bus
                res: list[FsbSampleSlot] = []
                for ch_g in children:
                    res.extend(resolve_inst_slots(ch_g, eff_ap, eff_inf, eff_bus, parent_chain + (ig,)))
                return res
            return []

        def _get_inst_rbus(ig: bytes) -> bytes:
            if ig in wais_map:
                return wais_map[ig][4]
            if ig in muis_map:
                return muis_map[ig][4]
            return b"\x00" * 16

        def _is_chain_muted(ig: bytes, rbus: bytes) -> bool:
            for tgt in (ig, rbus):
                if tgt == b"\x00" * 16:
                    continue
                for _, c_cg, prop in ctrs_by_target.get(tgt, []):
                    if prop != 0:
                        continue
                    pts = crvs_map.get(c_cg, [])
                    if pts and max(y for _, y in pts) <= -36.0:
                        return True
            return False

        def _compute_audible_rpm_window(ig: bytes, rbus: bytes, v0: float, v1: float) -> tuple[float, float] | None:
            if v1 <= v0 + 100.0:
                return None
            grid = np.linspace(v0, v1, 256)
            total_db = np.zeros_like(grid)
            has_rpm_vol = False
            for tgt in (ig, rbus):
                if tgt == b"\x00" * 16:
                    continue
                for c_pg, c_cg, prop in ctrs_by_target.get(tgt, []):
                    if prop != 0:
                        continue
                    pinfo = prms_map.get(c_pg)
                    if not pinfo or pinfo[0].strip().lower() not in ("rpm", "rpms"):
                        continue
                    pts = crvs_map.get(c_cg, [])
                    if len(pts) >= 2:
                        xs = np.array([pt[0] for pt in pts], dtype=np.float64)
                        ys = np.array([pt[1] for pt in pts], dtype=np.float64)
                        total_db += np.interp(grid, xs, ys)
                        has_rpm_vol = True
            if not has_rpm_vol:
                return None
            aud_mask = total_db >= -15.0
            if np.any(aud_mask) and np.any(total_db < -18.0):
                alo = float(grid[aud_mask][0])
                ahi = float(grid[aud_mask][-1])
                if ahi > alo + 80.0 and (ahi - alo) < 0.94 * (v1 - v0):
                    return (alo, ahi)
            return None

        # Construir información de eventos y clasificarlos
        evt_params: dict[bytes, list[tuple[str, float, float]]] = {}
        evt_samples: dict[bytes, list[FsbSampleSlot]] = {}
        evt_is_timeline: set[bytes] = set()
        slot_has_unmuted_ref: set[int] = set()
        slot_has_muted_ref: set[int] = set()

        for _pml_self, pg, eg, boxes in pmls_list:
            pinfo = prms_map.get(pg)
            if pinfo and pinfo not in evt_params.get(eg, []):
                evt_params.setdefault(eg, []).append(pinfo)
            pname_low = pinfo[0].strip().lower() if pinfo else ""
            for ig, v0, v1 in boxes:
                rbus = _get_inst_rbus(ig)
                muted_box = _is_chain_muted(ig, rbus)
                aud_win = (
                    _compute_audible_rpm_window(ig, rbus, v0, v1)
                    if pname_low in ("rpm", "rpms") and not muted_box
                    else None
                )
                for sl in resolve_inst_slots(ig):
                    evt_samples.setdefault(eg, []).append(sl)
                    if muted_box:
                        slot_has_muted_ref.add(sl.global_index)
                    else:
                        slot_has_unmuted_ref.add(sl.global_index)
                    if pname_low in ("rpm", "rpms"):
                        sl.rpm_boxes.append((v0, v1))
                        if aud_win is not None:
                            sl.audible_rpm_boxes.append(aud_win)
                    elif pname_low in ("throttle", "load"):
                        sl.thr_boxes.append((v0, v1))
                    elif pname_low in ("throttlecut", "throttle_cut"):
                        sl.tcut_boxes.append((v0, v1))
                    elif pname_low == "state":
                        sl.shift_state_boxes.append((v0, v1))

        for _, eg, tboxes in tlns_list:
            if tboxes:
                evt_is_timeline.add(eg)
            for ig, _, _ in tboxes:
                rbus = _get_inst_rbus(ig)
                muted_box = _is_chain_muted(ig, rbus)
                for sl in resolve_inst_slots(ig):
                    evt_samples.setdefault(eg, []).append(sl)
                    if muted_box:
                        slot_has_muted_ref.add(sl.global_index)
                    else:
                        slot_has_unmuted_ref.add(sl.global_index)

        # Resolver tabla de rutas combinando STDT embebido + GUIDs.txt externo (explícito o autodetección)
        known_guids_in_bank: set[bytes] = {self.bank_guid} | {eg for eg, _, _ in raw_evts} | set(ibus_parent_bytes.values())
        known_guids_in_bank.discard(b"\x00" * 16)

        stdt_paths = parse_stdt_all_paths(self.stdt_bytes) if self.stdt_bytes else {}
        ext_paths: dict[bytes, str] = {}
        if self._explicit_external_guids is not None:
            ext_paths = self._explicit_external_guids
            self.matched_external_guids_count = sum(1 for g in known_guids_in_bank if g in ext_paths)
        elif not stdt_paths and self.source_path is not None:
            auto_gpath, auto_gmap = find_companion_guids_txt(self.source_path, known_bank_guids=known_guids_in_bank)
            if auto_gpath is not None and auto_gmap:
                self.guids_source_path = auto_gpath
                ext_paths = auto_gmap
                self.matched_external_guids_count = sum(1 for g in known_guids_in_bank if g in ext_paths)

        guid_to_path: dict[bytes, str] = {**ext_paths, **stdt_paths}

        # Determinar el rol de cada evento (especialmente los 6 eventos de motor AMS2 + eventos AC)
        int_events_with_rpm: list[tuple[int, float]] = []
        temp_roles: list[str] = []
        for idx, (eg, _, ibg) in enumerate(raw_evts):
            pbus = ibus_parent.get(ibg, "")
            pbus_bytes = ibus_parent_bytes.get(ibg, b"\x00" * 16)
            bus_role = AMS2_PARENT_BUS_ROLES.get(pbus, "")
            slist = evt_samples.get(eg, [])
            slist_names = [s.name.lower() for s in slist]
            snames = " ".join(slist_names)
            plist = evt_params.get(eg, [])
            ev_path = guid_to_path.get(eg, "")
            bus_path = guid_to_path.get(pbus_bytes, "")

            if ev_path:
                role = _classify_event_role_from_path_or_heuristics(
                    idx=idx,
                    ev_path=ev_path,
                    bus_path=bus_path,
                    plist=plist,
                    snames=snames,
                    slist_names=slist_names,
                )
                if role == "EngineInt":
                    max_rpm = max((p[2] for p in plist if p[0].lower() in ("rpm", "rpms")), default=12000.0)
                    int_events_with_rpm.append((idx, max_rpm))
            elif bus_role == "EngineExt":
                role = "EngineExt"
            elif bus_role == "Exhaust":
                role = "Exhaust"
            elif bus_role == "Startup":
                if "starter" in snames:
                    role = "Starter"
                elif "starting" in snames:
                    role = "Starting"
                else:
                    role = "Starter" if idx == 1 else "Starting"
            elif bus_role == "EngineInt":
                max_rpm = max((p[2] for p in plist if p[0].lower() in ("rpm", "rpms")), default=12000.0)
                int_events_with_rpm.append((idx, max_rpm))
                role = "EngineInt"
            else:
                role = _classify_event_role_from_path_or_heuristics(
                    idx=idx,
                    ev_path=ev_path,
                    bus_path=bus_path,
                    plist=plist,
                    snames=snames,
                    slist_names=slist_names,
                )
            temp_roles.append(role)

        # Si hay 2 eventos EngineInt en AMS2 (uno principal 12000 RPM con más muestras y otro Aux/AI 20000 RPM)
        if len(int_events_with_rpm) >= 2:
            ranked = sorted(
                int_events_with_rpm,
                key=lambda item: len(evt_samples.get(raw_evts[item[0]][0], [])),
                reverse=True,
            )
            primary_idx = ranked[0][0]
            for e_idx, _ in int_events_with_rpm:
                temp_roles[e_idx] = "EngineInt" if e_idx == primary_idx else "AuxInt"

        role_to_csd = {
            "EngineExt": "engine_ext",
            "EngineInt": "engine_int",
            "Exhaust": "exhaust_int",
            "Starter": "starter",
            "Starting": "starting",
            "AuxInt": "ai_engine_int",
        }

        evt_role_by_guid: dict[bytes, str] = {}
        for idx, (eg, _, ibg) in enumerate(raw_evts):
            g_str = bytes_to_guid_str(eg)
            role = temp_roles[idx]
            evt_role_by_guid[eg] = role
            pbus = ibus_parent.get(ibg, "")
            ev_path = guid_to_path.get(eg, "")
            slist = evt_samples.get(eg, [])
            unique_names = list(dict.fromkeys(s.name for s in slist))
            for sl in slist:
                if g_str not in sl.event_guids:
                    sl.event_guids.append(g_str)
                if role not in sl.event_roles:
                    sl.event_roles.append(role)
                if ev_path and ev_path not in sl.event_paths:
                    sl.event_paths.append(ev_path)
            self.events.append(
                FmodEventInfo(
                    index=idx,
                    guid_bytes=eg,
                    guid_str=g_str,
                    role=role,
                    csd_slot_hint=role_to_csd.get(role, ""),
                    parent_bus_guid=pbus,
                    path=ev_path,
                    parameters=evt_params.get(eg, []),
                    linked_samples=unique_names,
                    ibus_guid_bytes=ibg,
                )
            )

        # Clasificar semánticamente todas las muestras del banco
        tln_sample_ids = {
            id(sl)
            for eg in evt_is_timeline
            for sl in evt_samples.get(eg, [])
        }
        for sl in self.slots:
            sl.is_muted_donor_slot = (
                sl.global_index in slot_has_muted_ref and sl.global_index not in slot_has_unmuted_ref
            )
            sl.perspective, sl.load_role = classify_sample_semantics(
                name=sl.name,
                event_roles=sl.event_roles,
                rpm_boxes=sl.rpm_boxes,
                thr_boxes=sl.thr_boxes,
                is_timeline=(id(sl) in tln_sample_ids),
                thr_curve_hint=sl.thr_curve_hint,
                autopitch_ref_rpm=sl.autopitch_ref_rpm,
                tcut_boxes=sl.tcut_boxes,
                shift_state_boxes=sl.shift_state_boxes,
                is_looping=(sl.loop is not None or sl.inst_infinite_loop),
            )
            sl.is_aux_layer = _detect_aux_engine_layer(
                name=sl.name,
                perspective=sl.perspective,
                load_role=sl.load_role,
                near_distance_db=sl.near_distance_db,
                has_rpm_autopitch=sl.has_rpm_autopitch,
            )

        # Construir tabla editable de bandas RPM y estado de ThrottleCut en eventos de motor
        for pml_self, pg, eg, boxes in pmls_list:
            ev_role = evt_role_by_guid.get(eg, "")
            if ev_role not in ("EngineInt", "EngineExt", "AuxInt"):
                continue
            pinfo = prms_map.get(pg)
            pname_low = pinfo[0].strip().lower() if pinfo else ""
            if pname_low in ("throttlecut", "throttle_cut"):
                if boxes:
                    self.has_active_throttle_cut = True
                elif ev_role not in self.empty_throttle_cut_events:
                    self.empty_throttle_cut_events.append(ev_role)
                continue
            if pname_low not in ("rpm", "rpms"):
                continue

            band_persp = "int" if ev_role in ("EngineInt", "AuxInt") else "ext"
            for bi, (ig, v0, v1) in enumerate(boxes):
                box_slots = resolve_inst_slots(ig)
                unique_gidxs = list(dict.fromkeys(s.global_index for s in box_slots))
                rbus = _get_inst_rbus(ig)
                muted_box = _is_chain_muted(ig, rbus)

                fade_in_end = v0
                fade_out_start = v1
                fade_in_cg: bytes | None = None
                fade_out_cg: bytes | None = None
                for c_pg, c_cg, prop in ctrs_by_target.get(ig, []):
                    if prop != 4:
                        continue
                    cp_info = prms_map.get(c_pg)
                    if cp_info and cp_info[0].strip().lower() not in ("rpm", "rpms"):
                        continue
                    pts = crvs_map.get(c_cg, [])
                    if len(pts) >= 2:
                        if pts[0][1] <= 0.1 and pts[-1][1] >= 0.9:
                            fade_in_cg = c_cg
                            fade_in_end = max(v0, min(v1, pts[-1][0]))
                        elif pts[0][1] >= 0.9 and pts[-1][1] <= 0.1:
                            fade_out_cg = c_cg
                            fade_out_start = max(v0, min(v1, pts[0][0]))

                bus_cutoff: tuple[bytes, float, float] | None = None
                if rbus != b"\x00" * 16:
                    for c_pg, c_cg, prop in ctrs_by_target.get(rbus, []):
                        if prop != 0:
                            continue
                        cp_info = prms_map.get(c_pg)
                        if cp_info and cp_info[0].strip().lower() in ("rpm", "rpms"):
                            pts = crvs_map.get(c_cg, [])
                            if len(pts) >= 2 and pts[0][1] > -6.0 and pts[-1][1] <= -30.0:
                                bus_cutoff = (c_cg, pts[0][0], pts[1][0])
                                break

                if box_slots:
                    primary_sl = box_slots[0]
                    b_load = primary_sl.load_role if primary_sl.load_role in ("idle", "on", "off", "pops") else "on"
                    is_fill = any(
                        "fill" in s.name.lower() or s.is_aux_layer for s in box_slots
                    )
                    if len(unique_gidxs) > 1:
                        d_label = f"Playlist ({len(unique_gidxs)} muestras): {primary_sl.name}"
                    else:
                        d_label = primary_sl.name
                else:
                    b_load = "on"
                    is_fill = False
                    d_label = "Vacío"

                self.rpm_bands.append(
                    RpmBandInfo(
                        band_id=f"{ev_role}:{bi:02d}",
                        pml_guid=pml_self,
                        event_guid=eg,
                        box_index=bi,
                        inst_guid=ig,
                        route_bus_guid=rbus,
                        event_role=ev_role,
                        perspective=band_persp,
                        load_role=b_load,
                        donor_slot_indices=unique_gidxs,
                        donor_label=d_label,
                        rpm_min=v0,
                        rpm_max=v1,
                        fade_in_end=fade_in_end,
                        fade_out_start=fade_out_start,
                        fade_in_curv_guid=fade_in_cg,
                        fade_out_curv_guid=fade_out_cg,
                        bus_rpm_cutoff=bus_cutoff,
                        is_muted_bus=muted_box,
                        is_fill_layer=is_fill,
                    )
                )

    def _parse_fsb5(self, fbuf: bytes, sb_idx: int, start_g_idx: int) -> tuple[FsbSubBank, int]:
        magic, version, num_samples, shdr_size, name_size, data_size, mode = struct.unpack_from(
            "<4sIIIIII", fbuf, 0
        )
        if magic != b"FSB5":
            raise ValueError("Sub-banco FSB5 corrupto.")
        flags = struct.unpack_from("<I", fbuf, 28)[0]
        hash16 = fbuf[36:52]
        dummy8 = fbuf[52:60]

        off = 60
        raw_recs = []
        for _ in range(num_samples):
            val = struct.unpack_from("<Q", fbuf, off)[0]
            off += 8
            next_chunk = val & 1
            freq_id = (val >> 1) & 0xF
            ch_id = (val >> 5) & 0x3
            data_off = ((val >> 7) & 0x07FFFFFF) * 32
            pcm_smp = (val >> 34) & 0x3FFFFFFF

            sr = FREQ_ID_TO_HZ.get(freq_id, 48000)
            ch = CHAN_ID_TO_COUNT.get(ch_id, 2)
            loop: tuple[int, int] | None = None
            peak = 1.0

            while next_chunk:
                ehdr = struct.unpack_from("<I", fbuf, off)[0]
                off += 4
                next_chunk = ehdr & 1
                elen = (ehdr >> 1) & 0xFFFFFF
                etype = (ehdr >> 25) & 0x7F
                edata = fbuf[off : off + elen]
                off += elen
                if etype == 1 and len(edata) >= 1:
                    ch = int(edata[0])
                elif etype == 2 and len(edata) >= 4:
                    sr = int(struct.unpack_from("<I", edata, 0)[0])
                elif etype == 3 and len(edata) >= 8:
                    ls, le = struct.unpack_from("<II", edata, 0)
                    if le > ls:
                        loop = (int(ls), int(le))
                elif etype == 13 and len(edata) >= 4:
                    peak = float(struct.unpack_from("<f", edata, 0)[0])

            raw_recs.append((sr, ch, data_off, pcm_smp, loop, peak))

        name_base = 60 + shdr_size
        names: list[str] = []
        if name_size > 0:
            for i in range(num_samples):
                noff = struct.unpack_from("<I", fbuf, name_base + i * 4)[0]
                end = fbuf.find(b"\x00", name_base + noff)
                nm = (
                    fbuf[name_base + noff : end].decode("utf-8", "replace")
                    if end >= 0
                    else f"sample_{i:02d}"
                )
                names.append(nm)
        else:
            names = [f"sample_{i:02d}" for i in range(num_samples)]

        data_base = 60 + shdr_size + name_size
        samples: list[FsbSampleSlot] = []
        g_idx = start_g_idx
        for i, (sr, ch, doff, nsmp, loop, peak) in enumerate(raw_recs):
            next_doff = raw_recs[i + 1][2] if i + 1 < num_samples else data_size
            sbytes = fbuf[data_base + doff : data_base + next_doff]
            samples.append(
                FsbSampleSlot(
                    global_index=g_idx,
                    subbank_index=sb_idx,
                    sample_index=i,
                    name=names[i],
                    mode=mode,
                    sample_rate=sr,
                    channels=ch,
                    pcm_samples=nsmp,
                    data_offset=doff,
                    data_bytes=sbytes,
                    loop=loop,
                    peak_volume=peak,
                )
            )
            g_idx += 1

        return (
            FsbSubBank(
                index=sb_idx,
                version=version,
                mode=mode,
                flags=flags,
                hash16=hash16,
                dummy8=dummy8,
                samples=samples,
                raw_fsb_bytes=fbuf,
            ),
            g_idx,
        )

    def decode_slot_to_buffer(self, slot: FsbSampleSlot) -> AudioBuffer:
        """Decodifica una muestra del banco a AudioBuffer PCM16 listo para reproducir o exportar."""
        ch = slot.channels
        nsmp = slot.pcm_samples
        b = slot.data_bytes

        if slot.mode == FSB5_MODE_PCM16:
            need = nsmp * ch * 2
            arr = np.frombuffer(b[:need], dtype="<i2").copy()
            frames = len(arr) // ch
            pcm = arr[: frames * ch].reshape(frames, ch)
        elif slot.mode == FSB5_MODE_FADPCM:
            pcm = decode_fadpcm(b, ch, nsmp)
        elif slot.mode == FSB5_MODE_PCM8:
            need = nsmp * ch
            u8 = np.frombuffer(b[:need], dtype=np.int8).astype(np.int16) << 8
            frames = len(u8) // ch
            pcm = u8[: frames * ch].reshape(frames, ch)
        elif slot.mode == FSB5_MODE_PCMFLOAT:
            need = nsmp * ch * 4
            f32 = np.frombuffer(b[:need], dtype="<f4")
            s16 = np.clip(np.round(f32 * 32767.0), -32768, 32767).astype(np.int16)
            frames = len(s16) // ch
            pcm = s16[: frames * ch].reshape(frames, ch)
        elif slot.mode == FSB5_MODE_PCM24:
            need = nsmp * ch * 3
            raw3 = np.frombuffer(b[:need], dtype=np.uint8).reshape(-1, 3)
            hi16 = raw3[:, 1].astype(np.uint16) | (raw3[:, 2].astype(np.uint16) << 8)
            s16 = hi16.view(np.int16).copy()
            frames = len(s16) // ch
            pcm = s16[: frames * ch].reshape(frames, ch)
        elif slot.mode == FSB5_MODE_PCM32:
            need = nsmp * ch * 4
            s32 = np.frombuffer(b[:need], dtype="<i4")
            s16 = (s32 >> 16).astype(np.int16)
            frames = len(s16) // ch
            pcm = s16[: frames * ch].reshape(frames, ch)
        else:
            raise ValueError(
                f"La muestra '{slot.name}' usa el códec {slot.codec_name} (solo lectura de metadatos). "
                "Puedes reemplazarla libremente por cualquier muestra WAV o de Assetto Corsa."
            )

        return AudioBuffer(pcm=pcm, sample_rate=slot.sample_rate, loop=slot.loop)

    def extract_slot_wav_bytes(self, slot: FsbSampleSlot) -> bytes:
        return write_wav_bytes(self.decode_slot_to_buffer(slot))

    def extract_all_wavs(self, out_dir: Path) -> tuple[list[Path], list[str]]:
        """Extrae todas las muestras decodificables a archivos .wav en out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        skipped: list[str] = []
        for sl in self.slots:
            if not sl.can_decode_audio:
                skipped.append(f"{sl.name} ({sl.codec_name})")
                continue
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", sl.name).strip() or f"sample_{sl.global_index:02d}"
            dst = out_dir / f"{sl.global_index:02d}_{safe_name}.wav"
            write_wav_file(dst, self.decode_slot_to_buffer(sl))
            written.append(dst)
        return written, skipped


@dataclass
class FsbBuildSample:
    """Entrada para construir un sub-banco FSB5 nuevo."""

    name: str
    audio: AudioBuffer
    peak_override: float | None = None


def build_fsb5_container(
    samples: list[FsbBuildSample],
    mode: int = FSB5_MODE_PCM16,
    template_dummy8: bytes = b"AMS2PORT",
) -> bytes:
    """Construye un contenedor FSB5 binario 100% alineado a 32 bytes (PCM16 o FADPCM)."""
    if mode not in (FSB5_MODE_PCM16, FSB5_MODE_FADPCM):
        raise ValueError(f"Modo de codificación FSB5 no soportado para escritura: {mode}")

    shdr_buf = bytearray()
    data_buf = bytearray()
    names_list: list[bytes] = []

    for item in samples:
        buf = item.audio
        sr = buf.sample_rate
        ch = buf.channels
        loop = buf.loop
        peak = item.peak_override if item.peak_override is not None else buf.peak_linear()

        if mode == FSB5_MODE_PCM16:
            raw_audio = np.ascontiguousarray(buf.pcm, dtype="<i2").tobytes()
            pad32 = (32 - (len(raw_audio) % 32)) % 32
            if pad32:
                raw_audio += b"\x00" * pad32
            pcm_frames = buf.num_samples
        else:
            raw_audio, pcm_frames = encode_fadpcm(buf.pcm)

        cur_data_off = len(data_buf)
        if cur_data_off % 32 != 0:
            raise AssertionError("Desalineación interna de datos FSB5")
        data_buf += raw_audio

        # Construir extensiones de muestra (ext 1=ch, 2=freq si no estándar, 3=loop, 13=peak)
        exts: list[tuple[int, bytes]] = []
        if ch not in COUNT_TO_CHAN_ID:
            ch_id = 0
            exts.append((1, bytes([ch])))
        else:
            ch_id = COUNT_TO_CHAN_ID[ch]

        if sr not in HZ_TO_FREQ_ID:
            freq_id = 9  # 48000 fallback + ext(2)
            exts.append((2, struct.pack("<I", sr)))
        else:
            freq_id = HZ_TO_FREQ_ID[sr]

        if loop is not None and loop[1] > loop[0]:
            ls = max(0, min(pcm_frames - 2, int(loop[0])))
            le = max(ls + 1, min(pcm_frames - 1, int(loop[1])))
            exts.append((3, struct.pack("<II", ls, le)))

        exts.append((13, struct.pack("<f", float(peak))))

        has_next = 1 if exts else 0
        rec_u64 = (
            (has_next & 1)
            | ((freq_id & 0xF) << 1)
            | ((ch_id & 0x3) << 5)
            | (((cur_data_off // 32) & 0x07FFFFFF) << 7)
            | ((pcm_frames & 0x3FFFFFFF) << 34)
        )
        shdr_buf += struct.pack("<Q", rec_u64)

        for idx_e, (etype, edata) in enumerate(exts):
            e_next = 1 if idx_e + 1 < len(exts) else 0
            ehdr = (e_next & 1) | ((len(edata) & 0xFFFFFF) << 1) | ((etype & 0x7F) << 25)
            shdr_buf += struct.pack("<I", ehdr) + edata

        clean_nm = item.name.encode("utf-8", "replace").replace(b"\x00", b"")
        names_list.append(clean_nm + b"\x00")

    # Construir tabla de nombres de forma que (60 + shdr_size + name_size) % 32 == 0
    num_s = len(samples)
    name_tbl = bytearray(num_s * 4)
    str_pool = bytearray()
    for i, nb in enumerate(names_list):
        noff = num_s * 4 + len(str_pool)
        struct.pack_into("<I", name_tbl, i * 4, noff)
        str_pool += nb
    full_names = name_tbl + str_pool

    shdr_size = len(shdr_buf)
    pre_data = 60 + shdr_size + len(full_names)
    pad_names = (32 - (pre_data % 32)) % 32
    full_names += b"\x00" * pad_names
    name_size = len(full_names)
    data_size = len(data_buf)

    hash16 = hashlib.md5(bytes(shdr_buf) + bytes(full_names)).digest()
    dummy8 = (template_dummy8 + b"\x00" * 8)[:8]

    header = struct.pack(
        "<4sIIIIIIQ16s8s",
        b"FSB5",
        1,
        num_s,
        shdr_size,
        name_size,
        data_size,
        mode,
        0,  # flags = 0
        hash16,
        dummy8,
    )
    return header + bytes(shdr_buf) + bytes(full_names) + bytes(data_buf)
