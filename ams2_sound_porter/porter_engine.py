"""Motor de porteo AC -> AMS2: Auto-Mapeo FMOD, Reconstrucción FSB5 y Remapeo Standalone de GUIDs."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_dsp import AudioBuffer, conform_audio, decode_fadpcm, read_wav_file
from .fmod_bank import (
    FSB5_MODE_FADPCM,
    FSB5_MODE_PCM16,
    FSB5_MODE_VORBIS,
    FmodBank,
    FmodEventInfo,
    FsbBuildSample,
    FsbSampleSlot,
    RpmBandInfo,
    build_fsb5_container,
    bytes_to_guid_str,
    classify_sample_semantics,
    guid_str_to_bytes,
    infer_rpm_from_name,
    iter_riff_chunks,
)
from .fmod_stdt import build_stdt_chunk, lookup_stdt_id, parse_stdt_all_paths, parse_stdt_sections
from .omvt_csdbin import CsdbinDocument

# Tags de chunks base en PROJ cuyos primeros 16 bytes definen el GUID propio de un objeto local
LOCAL_OBJECT_BASE_CHUNKS = {
    b"BNKI",
    b"IBSB",
    b"GBSB",
    b"RBSB",
    b"MBSB",
    b"BEFB",
    b"PEFB",
    b"SEFB",
    b"SCFB",
    b"SCEF",
    b"VCAB",
    b"EVTB",
    b"TLNB",
    b"TRNB",
    b"PMLB",
    b"PRMB",
    b"CTRL",
    b"CURV",
    b"MAP ",
    b"MUIB",
    b"SPIB",
    b"PRIB",
    b"EVIB",
    b"WAIB",
    b"EFIB",
    b"WAV ",
    b"SNAB",
    b"MODB",
    b"MODU",
}


@dataclass
class SourceSampleItem:
    """Muestra de audio de origen (proveniente de un .bank de AC o de un archivo .wav suelto)."""

    index: int
    name: str
    sample_rate: int
    channels: int
    duration_sec: float
    loop: tuple[int, int] | None
    effective_rpm: float | None
    rpm_summary: str
    perspective: str
    load_role: str
    codec_name: str
    event_summary: str = "—"
    has_rpm_autopitch: bool = False
    inst_infinite_loop: bool = False
    near_distance_db: float = 0.0
    is_aux_layer: bool = False
    decodable: bool = True
    _bank_ref: FmodBank | None = field(default=None, repr=False)
    _slot_ref: FsbSampleSlot | None = field(default=None, repr=False)
    _wav_path: Path | None = field(default=None, repr=False)

    def load_audio(self) -> AudioBuffer:
        if self._bank_ref is not None and self._slot_ref is not None:
            return self._bank_ref.decode_slot_to_buffer(self._slot_ref)
        if self._wav_path is not None:
            return read_wav_file(self._wav_path)
        raise ValueError(f"Muestra de origen '{self.name}' sin fuente de datos.")


def _summarize_slot_events(sl: FsbSampleSlot) -> str:
    if sl.event_paths:
        leaves = [p.rsplit("/", 1)[-1] for p in sl.event_paths if p]
        if leaves:
            return ", ".join(dict.fromkeys(leaves))
    if sl.event_roles:
        return ", ".join(dict.fromkeys(sl.event_roles))
    return "—"


@dataclass
class SourceAudioLibrary:
    """Colección de muestras de origen cargadas desde un .bank de Assetto Corsa o carpeta WAV."""

    source_label: str
    source_path: Path
    items: list[SourceSampleItem]
    guids_path: Path | None = None
    matched_guids_count: int = 0
    total_events_count: int = 0
    fmod_bank: FmodBank | None = None

    @classmethod
    def from_bank(cls, bank_path: Path, guids_path: Path | None = None) -> "SourceAudioLibrary":
        bank_path = Path(bank_path)
        bank = FmodBank(bank_path, guids_path=guids_path)
        items: list[SourceSampleItem] = []
        for sl in bank.slots:
            items.append(
                SourceSampleItem(
                    index=sl.global_index,
                    name=sl.name,
                    sample_rate=sl.sample_rate,
                    channels=sl.channels,
                    duration_sec=sl.duration_sec,
                    loop=sl.loop,
                    effective_rpm=sl.effective_rpm,
                    rpm_summary=sl.rpm_summary,
                    perspective=sl.perspective,
                    load_role=sl.load_role,
                    codec_name=sl.codec_name,
                    event_summary=_summarize_slot_events(sl),
                    has_rpm_autopitch=sl.has_rpm_autopitch,
                    inst_infinite_loop=sl.inst_infinite_loop,
                    near_distance_db=sl.near_distance_db,
                    is_aux_layer=sl.is_aux_layer,
                    decodable=sl.can_decode_audio,
                    _bank_ref=bank,
                    _slot_ref=sl,
                )
            )
        label = (
            f"{bank_path.name} + {bank.guids_source_path.name}"
            if bank.guids_source_path is not None and bank.matched_external_guids_count > 0
            else bank_path.name
        )
        return cls(
            source_label=label,
            source_path=bank_path,
            items=items,
            guids_path=bank.guids_source_path if bank.matched_external_guids_count > 0 else guids_path,
            matched_guids_count=bank.matched_external_guids_count,
            total_events_count=len(bank.events),
            fmod_bank=bank,
        )

    @classmethod
    def from_wav_folder(cls, folder: Path) -> "SourceAudioLibrary":
        folder = Path(folder)
        wav_files = sorted(folder.rglob("*.wav"))
        if not wav_files:
            raise ValueError(f"No se encontraron archivos .wav dentro de: {folder}")
        items: list[SourceSampleItem] = []
        for idx, wp in enumerate(wav_files):
            buf = read_wav_file(wp)
            stem = wp.stem
            persp, role = classify_sample_semantics(stem, [], [], [], False)
            rpm = infer_rpm_from_name(stem)
            items.append(
                SourceSampleItem(
                    index=idx,
                    name=stem,
                    sample_rate=buf.sample_rate,
                    channels=buf.channels,
                    duration_sec=buf.duration_sec,
                    loop=buf.loop,
                    effective_rpm=rpm,
                    rpm_summary=f"~{rpm:.0f}" if rpm else "—",
                    perspective=persp,
                    load_role=role,
                    codec_name="WAV",
                    event_summary="WAV suelto",
                    has_rpm_autopitch=rpm is not None,
                    inst_infinite_loop=buf.loop is not None,
                    decodable=True,
                    _wav_path=wp,
                )
            )
        return cls(source_label=f"Carpeta: {folder.name}", source_path=folder, items=items)

    def max_engine_rpm(self) -> float:
        autopitch_rpms = [
            it.effective_rpm
            for it in self.items
            if it.effective_rpm
            and it.has_rpm_autopitch
            and it.perspective in ("int", "ext")
            and it.load_role in ("on", "off")
            and not it.is_aux_layer
        ]
        if autopitch_rpms:
            return max(autopitch_rpms)
        primary_rpms = [
            it.effective_rpm
            for it in self.items
            if it.effective_rpm
            and it.perspective in ("int", "ext")
            and it.load_role in ("on", "off")
            and not it.is_aux_layer
        ]
        if primary_rpms:
            return max(primary_rpms)
        fallback_rpms = [
            it.effective_rpm
            for it in self.items
            if it.effective_rpm and it.perspective in ("int", "ext") and it.load_role in ("on", "off")
        ]
        return max(fallback_rpms) if fallback_rpms else 8000.0


@dataclass
class SlotMappingConfig:
    """Configuración de reemplazo para un slot individual del banco donante de AMS2."""

    donor_global_index: int
    action: str = "keep_donor"  # "replace_source" | "replace_wav" | "keep_donor" | "silence"
    source_index: int | None = None
    custom_wav_path: Path | None = None
    override_root_rpm: float | None = None
    gain_db: float = 0.0
    match_score: float = 0.0
    match_reason: str = "Original del donante"


@dataclass
class RpmBandConfig:
    """Configuración editable de una banda de RPM y sus fundidos cruzados (crossfades) en FMOD."""

    band_id: str
    pml_guid: bytes
    event_guid: bytes
    box_index: int
    event_role: str
    perspective: str
    load_role: str
    donor_slot_indices: list[int]
    donor_label: str
    rpm_min: float
    rpm_max: float
    fade_in_end: float
    fade_out_start: float
    orig_rpm_min: float
    orig_rpm_max: float
    orig_fade_in_end: float
    orig_fade_out_start: float
    fade_in_curv_guid: bytes | None = None
    fade_out_curv_guid: bytes | None = None
    bus_rpm_cutoff: tuple[bytes, float, float] | None = None
    is_muted_bus: bool = False
    is_fill_layer: bool = False


@dataclass
class ThrottleCutConfig:
    """Configuración para el efecto de corte de acelerador / cambio de marcha (ThrottleCut)."""

    enabled: bool = True
    int_source_index: int | None = None
    ext_source_index: int | None = None
    int_custom_wav: Path | None = None
    ext_custom_wav: Path | None = None
    gain_db: float = 1.5
    duck_on_throttle_db: float = -13.5
    trigger_min: float = 0.15


def build_default_rpm_bands(donor: FmodBank) -> list[RpmBandConfig]:
    """Crea la lista inicial de bandas de RPM editables a partir de las cajas PMLB del donante."""
    out: list[RpmBandConfig] = []
    for b in donor.rpm_bands:
        out.append(
            RpmBandConfig(
                band_id=b.band_id,
                pml_guid=b.pml_guid,
                event_guid=b.event_guid,
                box_index=b.box_index,
                event_role=b.event_role,
                perspective=b.perspective,
                load_role=b.load_role,
                donor_slot_indices=list(b.donor_slot_indices),
                donor_label=b.donor_label,
                rpm_min=b.rpm_min,
                rpm_max=b.rpm_max,
                fade_in_end=b.fade_in_end,
                fade_out_start=b.fade_out_start,
                orig_rpm_min=b.rpm_min,
                orig_rpm_max=b.rpm_max,
                orig_fade_in_end=b.fade_in_end,
                orig_fade_out_start=b.fade_out_start,
                fade_in_curv_guid=b.fade_in_curv_guid,
                fade_out_curv_guid=b.fade_out_curv_guid,
                bus_rpm_cutoff=b.bus_rpm_cutoff,
                is_muted_bus=b.is_muted_bus,
                is_fill_layer=b.is_fill_layer,
            )
        )
    return out


def estimate_engine_rpm_bounds(
    donor: FmodBank | None,
    library: SourceAudioLibrary | None,
) -> tuple[float, float]:
    """Estima (idle_rpm, max_rpm) plausibles a partir de la librería AC o del banco donante."""
    idle_rpm = 950.0
    max_rpm = 7800.0
    if library is not None and library.items:
        idle_refs = [
            it.effective_rpm
            for it in library.items
            if it.load_role == "idle" and it.effective_rpm and 550.0 <= it.effective_rpm <= 2200.0
        ]
        if idle_refs:
            avg_idle = sum(idle_refs) / len(idle_refs)
            idle_rpm = round(max(700.0, min(1600.0, avg_idle)) / 25.0) * 25.0
        top_ref = library.max_engine_rpm()
        if top_ref > 1500.0:
            max_rpm = round(max(4500.0, min(19500.0, top_ref * 1.11)) / 50.0) * 50.0
    elif donor is not None and donor.slots:
        idle_refs = [
            sl.effective_rpm
            for sl in donor.slots
            if sl.load_role == "idle" and sl.effective_rpm and 550.0 <= sl.effective_rpm <= 2200.0
        ]
        if idle_refs:
            idle_rpm = round(max(700.0, min(1600.0, sum(idle_refs) / len(idle_refs))) / 25.0) * 25.0
        on_refs = [
            sl.effective_rpm
            for sl in donor.slots
            if sl.load_role in ("on", "off") and sl.effective_rpm and not sl.is_muted_donor_slot
        ]
        if on_refs:
            max_rpm = round(max(4500.0, min(19500.0, max(on_refs) * 1.10)) / 50.0) * 50.0
    return (float(idle_rpm), float(max_rpm))


# Términos que identifican micrófonos lejanos / de replay en mods de Assetto Corsa
# para ser excluidos en favor de la Chase Cam (cámara de persecución / escape cercano)
REPLAY_EXCLUDE_TERMS = (
    "front",
    "fron",
    "far",
    "ext_far",
    "mid_far",
    "flyby",
    "passby",
    "pass_by",
    "trackside",
    "side",
    "bounce",
    "test",
    "ambient",
    "dist",
)


def is_chase_cam_sample(name: str) -> bool:
    """Filtra muestras exteriores para conservar exclusivamente las de Chase Cam (cámara de persecución / escape cercano),
    descartando micrófonos frontales de radiador, tomas lejanas, flyby y rebotes de pista."""
    low = name.strip().lower()
    for kw in REPLAY_EXCLUDE_TERMS:
        if kw in low:
            return False
    return True


def cluster_clean_samples(
    items: list[SourceSampleItem],
    max_diff: float = 250.0,
) -> list[SourceSampleItem]:
    """Agrupa muestras por RPM cercana y selecciona la muestra canónica más limpia de cada escalón (eliminando tomas duplicadas)."""
    valid = [it for it in items if it.effective_rpm and it.effective_rpm > 100.0]
    valid.sort(key=lambda x: x.effective_rpm)
    if not valid:
        return items[:1] if items else []

    clusters: list[list[SourceSampleItem]] = []
    current_cluster: list[SourceSampleItem] = [valid[0]]
    for it in valid[1:]:
        if abs(it.effective_rpm - current_cluster[-1].effective_rpm) < max_diff:
            current_cluster.append(it)
        else:
            clusters.append(current_cluster)
            current_cluster = [it]
    if current_cluster:
        clusters.append(current_cluster)

    def _score_cluster_item(it: SourceSampleItem) -> float:
        s = 0.0
        low = it.name.lower()
        if it.is_aux_layer:
            s -= 100.0
        if any(w in low for w in ("gear", "clank", "trans", "turbo", "blowoff", "wastegate", "up", "wide", "test", "_a", "_b", "c1", "c2")):
            s -= 50.0
        if "load" in low or "coast" in low or "idle" in low:
            s += 50.0
        s += min(it.duration_sec, 12.0)
        return s

    out: list[SourceSampleItem] = []
    for cl in clusters:
        best = max(cl, key=_score_cluster_item)
        out.append(best)
    return out


def auto_detect_throttle_cut_config(
    donor: FmodBank | SourceAudioLibrary | None = None,
    library: SourceAudioLibrary | None = None,
) -> ThrottleCutConfig:
    """Detecta automáticamente muestras de Throttle Cut / Upshift en el origen AC y las configura."""
    if isinstance(donor, SourceAudioLibrary) and library is None:
        library = donor
        donor = None

    if library is None:
        return ThrottleCutConfig(enabled=False)

    candidates = [
        it
        for it in library.items
        if it.decodable and it.load_role in ("throttle_cut", "upshift")
    ]
    if not candidates:
        return ThrottleCutConfig(enabled=False)

    def _score_tcut(it: SourceSampleItem, target_persp: str) -> float:
        s = 0.0
        low = it.name.lower()
        if it.perspective == target_persp:
            s += 50.0
        elif it.perspective in ("int", "ext"):
            s += 15.0
        if target_persp == "ext" and not is_chase_cam_sample(it.name):
            s -= 50.0
        if any(k in low for k in ("throttle_cut", "throttlecut", "shift_cut", "shiftcut", "flatshift")):
            s += 30.0
        elif any(k in low for k in ("upshift", "gearup", "gear_up")):
            s += 24.0
        elif "gearchange" in low:
            s += 16.0
        elif "gearshift" in low:
            s += 10.0
        if "1" in low:
            s += 3.0
        if 0.12 <= it.duration_sec <= 1.25:
            s += 15.0
        elif it.duration_sec > 2.2:
            s -= 20.0
        if it.loop is None and not it.inst_infinite_loop:
            s += 8.0
        return s

    best_int = max(candidates, key=lambda it: _score_tcut(it, "int"))
    best_ext = max(candidates, key=lambda it: _score_tcut(it, "ext"))
    return ThrottleCutConfig(
        enabled=True,
        int_source_index=best_int.index,
        ext_source_index=best_ext.index,
        gain_db=1.5,
        duck_on_throttle_db=-13.5,
        trigger_min=0.15,
    )


def auto_distribute_rpm_bands(
    bands: list[RpmBandConfig],
    mappings_or_donor: list[SlotMappingConfig] | FmodBank,
    donor_or_mappings: FmodBank | list[SlotMappingConfig],
    library: SourceAudioLibrary | None = None,
    idle_rpm: float = 950.0,
    max_rpm: float = 7800.0,
) -> list[RpmBandConfig]:
    """Ajusta automáticamente los rangos y crossfades de RPM eliminando solapamientos entre muestras."""
    if isinstance(mappings_or_donor, FmodBank):
        donor: FmodBank = mappings_or_donor
        mappings: list[SlotMappingConfig] = donor_or_mappings  # type: ignore[assignment]
    else:
        mappings = mappings_or_donor
        donor = donor_or_mappings  # type: ignore[assignment]

    idle_rpm = max(500.0, min(3000.0, float(idle_rpm)))
    max_rpm = max(idle_rpm + 2000.0, min(22000.0, float(max_rpm)))
    ceil_rpm = round(max(max_rpm * 1.28, max_rpm + 1500.0))

    map_by_gidx = {m.donor_global_index: m for m in mappings}
    lib_by_idx = {it.index: it for it in library.items} if library else {}

    def _band_center_rpm(b: RpmBandConfig) -> float:
        for gidx in b.donor_slot_indices:
            m = map_by_gidx.get(gidx)
            if m:
                if m.override_root_rpm and m.override_root_rpm > 100.0:
                    return float(m.override_root_rpm)
                if m.action == "replace_source" and m.source_index is not None and m.source_index in lib_by_idx:
                    er = lib_by_idx[m.source_index].effective_rpm
                    if er and er > 100.0:
                        return float(er)
            if 0 <= gidx < len(donor.slots):
                er = donor.slots[gidx].effective_rpm
                if er and er > 100.0:
                    return float(er)
        return 0.5 * (b.orig_rpm_min + min(b.orig_rpm_max, max_rpm))

    def _is_band_silenced(b: RpmBandConfig) -> bool:
        if b.is_muted_bus:
            return True
        if not b.donor_slot_indices:
            return True
        return all(
            map_by_gidx.get(g, SlotMappingConfig(g)).action == "silence"
            for g in b.donor_slot_indices
        )

    idle_hold_end = round(min(max_rpm * 0.24, max(idle_rpm * 1.38, idle_rpm + 350.0)))
    idle_fade_end = round(min(max_rpm * 0.34, max(idle_hold_end + 480.0, idle_rpm * 2.15)))

    for ev_role in ("EngineInt", "EngineExt", "AuxInt"):
        ev_bands = [b for b in bands if b.event_role == ev_role and not _is_band_silenced(b)]
        if not ev_bands:
            continue

        # 1. Bandas IDLE
        for b in [x for x in ev_bands if x.load_role == "idle"]:
            b.rpm_min = 0.0
            b.fade_in_end = 0.0
            b.fade_out_start = float(idle_hold_end)
            b.rpm_max = float(idle_fade_end)

        # 2. Bandas POPS / Petardeos
        pops_start = round(max(idle_fade_end, max_rpm * 0.40))
        for b in [x for x in ev_bands if x.load_role == "pops"]:
            if b.is_muted_bus:
                b.rpm_min = 0.0
                b.fade_in_end = 0.0
                b.fade_out_start = 0.0
                b.rpm_max = 0.0
                continue
            b.rpm_min = float(pops_start)
            b.fade_in_end = float(min(ceil_rpm - 100.0, pops_start + 450.0))
            b.fade_out_start = float(ceil_rpm)
            b.rpm_max = float(ceil_rpm)

        # 3. Escalera limpia sin solapamientos para ON y OFF por separado
        for lrole in ("on", "off"):
            r_bands = [
                x
                for x in ev_bands
                if x.load_role == lrole and not (x.is_fill_layer and ev_role in ("EngineInt", "AuxInt"))
            ]
            if not r_bands:
                continue
            r_bands.sort(key=lambda x: (_band_center_rpm(x), x.orig_rpm_min, x.box_index))
            n = len(r_bands)

            low_entry_min = float(idle_hold_end) if lrole == "on" else float(round(idle_rpm * 1.30))
            low_entry_full = float(idle_fade_end) if lrole == "on" else float(round(idle_fade_end + 120.0))

            if n == 1:
                b0 = r_bands[0]
                b0.rpm_min = low_entry_min
                b0.fade_in_end = low_entry_full
                b0.fade_out_start = float(ceil_rpm)
                b0.rpm_max = float(ceil_rpm)
                continue

            raw_centers = [_band_center_rpm(x) for x in r_bands]
            # Garantizar centros estrictamente crecientes y escalados dentro de [low_entry_full+350 .. max_rpm*0.92]
            span_lo = low_entry_full + 350.0
            span_hi = max(span_lo + 600.0, max_rpm * 0.92)
            c_min, c_max = min(raw_centers), max(raw_centers)
            if c_max - c_min < 450.0:
                centers = [
                    span_lo + (span_hi - span_lo) * (i / float(n - 1))
                    for i in range(n)
                ]
            else:
                centers = []
                for i, rc in enumerate(raw_centers):
                    # Escalar suavemente si las muestras exceden el corte o quedan muy juntas
                    mapped_c = max(span_lo, min(span_hi, rc))
                    if centers and mapped_c < centers[-1] + 380.0:
                        mapped_c = centers[-1] + 380.0
                    centers.append(mapped_c)
                if centers[-1] > span_hi:
                    # Re-comprimir proporcionalmente dentro de [span_lo .. span_hi]
                    lo0, hi0 = centers[0], centers[-1]
                    if hi0 > lo0:
                        centers = [
                            span_lo + (c - lo0) / (hi0 - lo0) * (span_hi - span_lo)
                            for c in centers
                        ]

            # Calcular cruces simétricos exactos entre cada par adyacente (k, k+1)
            crossovers: list[tuple[float, float]] = []
            for k in range(n - 1):
                mid = 0.5 * (centers[k] + centers[k + 1])
                gap = max(300.0, centers[k + 1] - centers[k])
                half_w = max(160.0, min(320.0, 0.27 * gap))
                xf_start = round(mid - half_w)
                xf_end = round(mid + half_w)
                if crossovers and xf_start < crossovers[-1][1] + 60.0:
                    xf_start = crossovers[-1][1] + 60.0
                    xf_end = max(xf_start + 220.0, xf_end)
                crossovers.append((float(xf_start), float(xf_end)))

            for k, bk in enumerate(r_bands):
                if k == 0:
                    bk.rpm_min = low_entry_min
                    bk.fade_in_end = min(crossovers[0][0] - 40.0, low_entry_full)
                    bk.fade_in_end = max(bk.rpm_min + 40.0, bk.fade_in_end)
                else:
                    bk.rpm_min = crossovers[k - 1][0]
                    bk.fade_in_end = crossovers[k - 1][1]

                if k == n - 1:
                    bk.fade_out_start = float(ceil_rpm)
                    bk.rpm_max = float(ceil_rpm)
                else:
                    bk.fade_out_start = crossovers[k][0]
                    bk.rpm_max = crossovers[k][1]

    return bands


def auto_distribute_single_rpm_band(
    target_band: RpmBandConfig,
    bands: list[RpmBandConfig],
    mappings_or_donor: list[SlotMappingConfig] | FmodBank,
    donor_or_mappings: FmodBank | list[SlotMappingConfig],
    library: SourceAudioLibrary | None = None,
    idle_rpm: float = 950.0,
    max_rpm: float = 7800.0,
) -> RpmBandConfig:
    """Ajusta automáticamente únicamente la banda indicada para eliminar solapamientos con sus adyacentes."""
    if isinstance(mappings_or_donor, FmodBank):
        donor: FmodBank = mappings_or_donor
        mappings: list[SlotMappingConfig] = donor_or_mappings  # type: ignore[assignment]
    else:
        mappings = mappings_or_donor
        donor = donor_or_mappings  # type: ignore[assignment]

    idle_rpm = max(500.0, min(3000.0, float(idle_rpm)))
    max_rpm = max(idle_rpm + 2000.0, min(22000.0, float(max_rpm)))
    ceil_rpm = round(max(max_rpm * 1.28, max_rpm + 1500.0))

    map_by_gidx = {m.donor_global_index: m for m in mappings}
    lib_by_idx = {it.index: it for it in library.items} if library else {}

    def _band_center_rpm(b: RpmBandConfig) -> float:
        for gidx in b.donor_slot_indices:
            m = map_by_gidx.get(gidx)
            if m:
                if m.override_root_rpm and m.override_root_rpm > 100.0:
                    return float(m.override_root_rpm)
                if m.action == "replace_source" and m.source_index is not None and m.source_index in lib_by_idx:
                    er = lib_by_idx[m.source_index].effective_rpm
                    if er and er > 100.0:
                        return float(er)
            if 0 <= gidx < len(donor.slots):
                er = donor.slots[gidx].effective_rpm
                if er and er > 100.0:
                    return float(er)
        return 0.5 * (b.rpm_min + min(b.rpm_max, max_rpm))

    idle_hold_end = round(min(max_rpm * 0.24, max(idle_rpm * 1.38, idle_rpm + 350.0)))
    idle_fade_end = round(min(max_rpm * 0.34, max(idle_hold_end + 480.0, idle_rpm * 2.15)))

    # 1. Banda de Ralentí (IDLE)
    if target_band.load_role == "idle":
        target_band.rpm_min = 0.0
        target_band.fade_in_end = 0.0
        target_band.fade_out_start = float(idle_hold_end)
        target_band.rpm_max = float(idle_fade_end)
        return target_band

    # 2. Banda de Petardeos (POPS)
    if target_band.load_role == "pops":
        if target_band.is_muted_bus:
            target_band.rpm_min = 0.0
            target_band.fade_in_end = 0.0
            target_band.fade_out_start = 0.0
            target_band.rpm_max = 0.0
            return target_band
        pops_start = round(max(idle_fade_end, max_rpm * 0.40))
        target_band.rpm_min = float(pops_start)
        target_band.fade_in_end = float(min(ceil_rpm - 100.0, pops_start + 450.0))
        target_band.fade_out_start = float(ceil_rpm)
        target_band.rpm_max = float(ceil_rpm)
        return target_band

    # 3. Bandas ON / OFF
    ev_role = target_band.event_role
    lrole = target_band.load_role
    same_layer_bands = [
        b
        for b in bands
        if b.event_role == ev_role
        and b.load_role == lrole
        and not b.is_muted_bus
        and not (b.is_fill_layer and ev_role in ("EngineInt", "AuxInt"))
    ]
    if target_band not in same_layer_bands:
        same_layer_bands.append(target_band)

    same_layer_bands.sort(key=lambda x: (_band_center_rpm(x), x.rpm_min, x.box_index))
    n = len(same_layer_bands)
    k = same_layer_bands.index(target_band)

    low_entry_min = float(idle_hold_end) if lrole == "on" else float(round(idle_rpm * 1.30))
    low_entry_full = float(idle_fade_end) if lrole == "on" else float(round(idle_fade_end + 120.0))

    if n == 1:
        target_band.rpm_min = low_entry_min
        target_band.fade_in_end = low_entry_full
        target_band.fade_out_start = float(ceil_rpm)
        target_band.rpm_max = float(ceil_rpm)
        return target_band

    c_cur = _band_center_rpm(target_band)

    # Ajuste de entrada (RPM Mín y Fin Fade-In)
    if k == 0:
        target_band.rpm_min = low_entry_min
        next_b = same_layer_bands[1]
        c_next = _band_center_rpm(next_b)
        gap = max(300.0, c_next - c_cur)
        half_w = max(160.0, min(320.0, 0.27 * gap))
        mid = 0.5 * (c_cur + c_next)
        target_band.fade_in_end = min(mid - half_w - 40.0, low_entry_full)
        target_band.fade_in_end = max(target_band.rpm_min + 40.0, target_band.fade_in_end)
        target_band.fade_out_start = float(round(mid - half_w))
        target_band.rpm_max = float(round(mid + half_w))
        next_b.rpm_min = target_band.fade_out_start
        next_b.fade_in_end = target_band.rpm_max
    elif k == n - 1:
        prev_b = same_layer_bands[k - 1]
        c_prev = _band_center_rpm(prev_b)
        gap = max(300.0, c_cur - c_prev)
        half_w = max(160.0, min(320.0, 0.27 * gap))
        mid = 0.5 * (c_prev + c_cur)
        target_band.rpm_min = float(round(mid - half_w))
        target_band.fade_in_end = float(round(mid + half_w))
        prev_b.fade_out_start = target_band.rpm_min
        prev_b.rpm_max = target_band.fade_in_end
        target_band.fade_out_start = float(ceil_rpm)
        target_band.rpm_max = float(ceil_rpm)
    else:
        # Banda intermedia con vecina anterior y vecina posterior
        prev_b = same_layer_bands[k - 1]
        next_b = same_layer_bands[k + 1]
        c_prev = _band_center_rpm(prev_b)
        c_next = _band_center_rpm(next_b)

        gap_prev = max(300.0, c_cur - c_prev)
        half_prev = max(160.0, min(320.0, 0.27 * gap_prev))
        mid_prev = 0.5 * (c_prev + c_cur)
        target_band.rpm_min = float(round(mid_prev - half_prev))
        target_band.fade_in_end = float(round(mid_prev + half_prev))
        prev_b.fade_out_start = target_band.rpm_min
        prev_b.rpm_max = target_band.fade_in_end

        gap_next = max(300.0, c_next - c_cur)
        half_next = max(160.0, min(320.0, 0.27 * gap_next))
        mid_next = 0.5 * (c_cur + c_next)
        target_band.fade_out_start = float(round(mid_next - half_next))
        target_band.rpm_max = float(round(mid_next + half_next))
        next_b.rpm_min = target_band.fade_out_start
        next_b.fade_in_end = target_band.rpm_max

    return target_band


def sync_faithful_assetto_corsa_setup(
    donor: FmodBank,
    library: SourceAudioLibrary,
    mappings: list[SlotMappingConfig],
    bands: list[RpmBandConfig],
) -> tuple[float, float, ThrottleCutConfig, list[RpmBandConfig], list[SlotMappingConfig]]:
    """Copia la configuración acústica del sonido de Assetto Corsa de forma fiel:
    - Ajusta los límites RPM (idle y redline) reales de AC.
    - Sincroniza y activa el efecto Throttle Cut / cambio de marcha.
    - Silencia efectos o capas extras del donante no presentes en AC (ej. fills secundarios, transmisiones extra).
    - Asigna las muestras de AC a los slots principales y redistribuye las bandas sin solapamiento.
    """
    idle_rpm, max_rpm = estimate_engine_rpm_bounds(donor, library)
    tcut_cfg = auto_detect_throttle_cut_config(donor, library)

    # 1. Re-mapear muestras de AC hacia los slots del donante
    new_mappings = auto_map_samples(
        donor,
        library,
        src_redline_override=max_rpm,
        dst_redline_override=max_rpm,
    )

    # 2. Silenciar capas accesorias no presentes en AC para recreación fiel
    has_turbo_in_ac = any(
        it.decodable and any(w in it.name.lower() for w in ("turbo", "blowoff", "wastegate"))
        for it in library.items
    )
    has_trans_in_ac = any(
        it.decodable and any(w in it.name.lower() for w in ("trans", "gear", "gearbox", "differential"))
        for it in library.items
    )

    for m in new_mappings:
        sl = donor.slots[m.donor_global_index]
        s_low = sl.name.lower()
        # Silenciar pistas de transmisión / caja si AC no tiene audio de transmisión
        if not has_trans_in_ac and any(w in s_low for w in ("transmission", "gear", "trans_int", "trans_ext")):
            if sl.mode == FSB5_MODE_VORBIS or m.action != "replace_source":
                m.action = "silence"
                m.match_reason = "Silenciado fiel (Sin transmisión en AC)"
        # Silenciar turbo si AC no tiene sonido de turbo
        if not has_turbo_in_ac and "turbo" in s_low:
            if sl.mode == FSB5_MODE_VORBIS or m.action != "replace_source":
                m.action = "silence"
                m.match_reason = "Silenciado fiel (Sin turbo en AC)"
        # Silenciar capas de vibración secundarias de cabina
        if sl.is_aux_layer or "vibration" in s_low:
            m.action = "silence"
            m.match_reason = "Silenciado fiel (Evita ensuciar mezcla AC)"

    # 3. Restaurar bandas a valores base del donante y luego auto-distribuir según AC
    rebuilt_bands = build_default_rpm_bands(donor)
    distributed_bands = auto_distribute_rpm_bands(
        rebuilt_bands,
        donor,
        new_mappings,
        library=library,
        idle_rpm=idle_rpm,
        max_rpm=max_rpm,
    )

    return (idle_rpm, max_rpm, tcut_cfg, distributed_bands, new_mappings)


def sync_faithful_section_from_source(
    donor: FmodBank,
    library: SourceAudioLibrary,
    mappings: list[SlotMappingConfig],
    bands: list[RpmBandConfig],
    perspective: str,
    idle_rpm: float = 950.0,
    max_rpm: float = 7800.0,
) -> tuple[list[RpmBandConfig], list[SlotMappingConfig], int, int]:
    """Modifica y acomoda TODAS las bandas dentro de la sección elegida (Interior o Exterior)
    para que sea una recreación fiel de los GUIDs y muestras de Assetto Corsa:
    - Conserva las bandas físicas nativas del FMOD donor bank para evitar desincronización y pitch erróneo.
    - En el exterior (EXT), descarta pistas de replay (front, far, flyby, trackside, etc.) para aislar la Chase Cam.
    - Agrupa y deduplica tomas grabadas a RPM idénticas (clusterización limpia).
    - Silencia capas secundarias, fill y bandas donantes excedentes para evitar solapamientos y desfases.
    - Mapea de forma óptima a las cajas nativas del donante y ajusta el Autopitch de cada slot a su RPM real.
    - Re-centra la escalera de RPM con cruces suaves sin desfases de afinación.
    """
    idle_rpm = max(500.0, min(3000.0, float(idle_rpm)))
    max_rpm = max(idle_rpm + 2000.0, min(22000.0, float(max_rpm)))
    target_ev_role = "EngineInt" if perspective == "int" else "EngineExt"

    decodable_items = [it for it in library.items if it.decodable]
    # Filtrar muestras pertenecientes a esta perspectiva (o motor general si no están particionadas)
    persp_items = [it for it in decodable_items if it.perspective == perspective]
    if not persp_items:
        persp_items = [it for it in decodable_items if it.load_role in ("idle", "on", "off", "pops")]

    if perspective == "ext":
        # Filtrar estrictamente solo muestras de Chase Cam para el exterior
        chase_items = [it for it in persp_items if is_chase_cam_sample(it.name)]
        if chase_items:
            persp_items = chase_items

    # Descartar capas auxiliares (engranajes, silbidos, rebotes) para el motor principal
    primary_items = [it for it in persp_items if not it.is_aux_layer]
    if not primary_items:
        primary_items = persp_items

    ac_idle = [it for it in primary_items if it.load_role == "idle"]
    ac_on = [it for it in primary_items if it.load_role == "on"]
    ac_off = [it for it in primary_items if it.load_role == "off"]
    ac_pops = [it for it in primary_items if it.load_role in ("pops", "backfire")]

    # Fallbacks si no hay idle o pops específicos en esa perspectiva
    if not ac_idle:
        ac_idle = [it for it in decodable_items if it.load_role == "idle" and not it.is_aux_layer]
    if not ac_pops:
        ac_pops = [it for it in decodable_items if it.load_role in ("pops", "backfire")]

    # Clusterizar tomas duplicadas a mismas RPMs en ON y OFF
    ac_on = cluster_clean_samples(ac_on, max_diff=250.0)
    ac_off = cluster_clean_samples(ac_off, max_diff=250.0)
    if ac_idle:
        ac_idle = cluster_clean_samples(ac_idle, max_diff=350.0)

    # Ordenar muestras ON y OFF por su RPM efectiva ascendente
    ac_on.sort(key=lambda it: it.effective_rpm if (it.effective_rpm and it.effective_rpm > 100.0) else 4000.0)
    ac_off.sort(key=lambda it: it.effective_rpm if (it.effective_rpm and it.effective_rpm > 100.0) else 4000.0)

    # Reconstruir las bandas de esta perspectiva a partir de las bandas nativas originales del donante
    # para garantizar que correspondan exactamente a los chunks binarios PMLB existentes en el .bank
    other_bands = [b for b in bands if b.perspective != perspective]
    native_donor_bands = [b for b in build_default_rpm_bands(donor) if b.perspective == perspective]

    bands_synced_count = 0
    bands_muted_count = 0

    # Silenciar capas secundarias AuxInt / fill duplicadas de esta perspectiva
    for b in native_donor_bands:
        if b.is_fill_layer or b.event_role == "AuxInt":
            b.is_muted_bus = True
            b.rpm_max = 0.0
            for g in b.donor_slot_indices:
                if 0 <= g < len(mappings):
                    mappings[g].action = "silence"
                    mappings[g].match_reason = "Capa secundaria silenciada (Recreación fiel AC)"
            bands_muted_count += 1

    curr_idle = [b for b in native_donor_bands if b.load_role == "idle" and b.event_role == target_ev_role and not b.is_fill_layer]
    curr_on = [b for b in native_donor_bands if b.load_role == "on" and b.event_role == target_ev_role and not b.is_fill_layer]
    curr_off = [b for b in native_donor_bands if b.load_role == "off" and b.event_role == target_ev_role and not b.is_fill_layer]
    curr_pops = [b for b in native_donor_bands if b.load_role == "pops" and b.event_role == target_ev_role]

    def _sync_role_group(
        curr_role_bands: list[RpmBandConfig],
        ac_role_items: list[SourceSampleItem],
        role_name: str,
    ) -> list[RpmBandConfig]:
        nonlocal bands_synced_count, bands_muted_count
        if not curr_role_bands:
            return []
        if not ac_role_items:
            for b in curr_role_bands:
                b.is_muted_bus = True
                b.rpm_max = 0.0
                for g in b.donor_slot_indices:
                    if 0 <= g < len(mappings):
                        mappings[g].action = "silence"
                        mappings[g].match_reason = f"Banda silenciada (Sin {role_name} en AC)"
                bands_muted_count += 1
            return curr_role_bands

        curr_role_bands.sort(key=lambda b: (b.orig_rpm_min, b.orig_rpm_max, b.box_index))
        K = len(curr_role_bands)
        M = len(ac_role_items)

        if M <= K:
            # AC tiene igual o menos muestras que cajas nativas en el donante
            for i in range(M):
                b = curr_role_bands[i]
                it = ac_role_items[i]
                b.is_muted_bus = False
                b.donor_label = f"AC [{it.index:02d}] {it.name}"
                for g in b.donor_slot_indices:
                    if 0 <= g < len(mappings):
                        m = mappings[g]
                        m.action = "replace_source"
                        m.source_index = it.index
                        m.override_root_rpm = it.effective_rpm
                        m.gain_db = 0.0
                        m.match_reason = f"Recreación Fiel AC ({it.name})"
                bands_synced_count += 1
            # Silenciar cajas donantes excedentes
            for i in range(M, K):
                surplus = curr_role_bands[i]
                surplus.is_muted_bus = True
                surplus.rpm_max = 0.0
                for g in surplus.donor_slot_indices:
                    if 0 <= g < len(mappings):
                        mappings[g].action = "silence"
                        mappings[g].match_reason = "Banda donante silenciada (Sobrante fiel AC)"
                bands_muted_count += 1
        else:
            # AC tiene más muestras que cajas nativas (ej. 5 muestras vs 3 cajas): seleccionar las K mejor distribuidas
            for j in range(K):
                ac_idx = round(j * (M - 1) / (K - 1)) if K > 1 else 0
                it = ac_role_items[ac_idx]
                b = curr_role_bands[j]
                b.is_muted_bus = False
                b.donor_label = f"AC [{it.index:02d}] {it.name}"
                for g in b.donor_slot_indices:
                    if 0 <= g < len(mappings):
                        m = mappings[g]
                        m.action = "replace_source"
                        m.source_index = it.index
                        m.override_root_rpm = it.effective_rpm
                        m.gain_db = 0.0
                        m.match_reason = f"Recreación Fiel AC ({it.name})"
                bands_synced_count += 1

        return curr_role_bands

    # Sincronizar cada grupo de rol con sus muestras de AC
    _sync_role_group(curr_idle, ac_idle, "idle")
    _sync_role_group(curr_on, ac_on, "on")
    _sync_role_group(curr_off, ac_off, "off")
    _sync_role_group(curr_pops, ac_pops, "pops")

    # Silenciar pistas de transmisión / turbo en esta sección si AC no las define
    has_turbo_in_ac = any(
        it.decodable and any(w in it.name.lower() for w in ("turbo", "blowoff", "wastegate"))
        for it in library.items
    )
    has_trans_in_ac = any(
        it.decodable and any(w in it.name.lower() for w in ("trans", "gear", "gearbox", "differential"))
        for it in library.items
    )
    for m in mappings:
        sl = donor.slots[m.donor_global_index]
        if sl.perspective == perspective:
            s_low = sl.name.lower()
            if not has_trans_in_ac and any(w in s_low for w in ("transmission", "gear", "trans_int", "trans_ext")):
                if sl.mode == FSB5_MODE_VORBIS or m.action != "replace_source":
                    m.action = "silence"
                    m.match_reason = "Silenciado fiel (Sin transmisión en AC)"
            if not has_turbo_in_ac and "turbo" in s_low:
                if sl.mode == FSB5_MODE_VORBIS or m.action != "replace_source":
                    m.action = "silence"
                    m.match_reason = "Silenciado fiel (Sin turbo en AC)"

    # Acomodar y auto-distribuir la escalera de RPM de esta sección
    combined_bands = other_bands + native_donor_bands
    auto_distribute_rpm_bands(
        combined_bands,
        donor,
        mappings,
        library=library,
        idle_rpm=idle_rpm,
        max_rpm=max_rpm,
    )

    return (combined_bands, mappings, bands_synced_count, bands_muted_count)




def auto_map_samples(
    donor: FmodBank,
    library: SourceAudioLibrary,
    src_redline_override: float = 0.0,
    dst_redline_override: float = 0.0,
) -> list[SlotMappingConfig]:
    """Asigna automáticamente muestras de origen (AC/WAV) a cada slot del banco donante AMS2."""
    decodable_items = [it for it in library.items if it.decodable]
    src_max_rpm = src_redline_override if src_redline_override > 1000 else library.max_engine_rpm()

    dst_rpms = [
        sl.effective_rpm
        for sl in donor.slots
        if sl.effective_rpm
        and sl.perspective in ("int", "ext")
        and sl.load_role in ("on", "off")
        and not sl.is_aux_layer
        and not sl.is_muted_donor_slot
    ]
    if not dst_rpms:
        dst_rpms = [
            sl.effective_rpm
            for sl in donor.slots
            if sl.effective_rpm and sl.perspective in ("int", "ext") and sl.load_role in ("on", "off")
        ]
    dst_max_rpm = dst_redline_override if dst_redline_override > 1000 else (max(dst_rpms) if dst_rpms else 8500.0)

    has_starter_in_src = any(it.load_role == "starter" for it in decodable_items)
    has_starting_in_src = any(it.load_role == "starting" for it in decodable_items)
    has_tcut_in_src = any(it.load_role in ("throttle_cut", "upshift") for it in decodable_items)
    pops_usage_counter = 0

    # Ordenar primero los slots activos primarios por RPM para repartir muestras distintas sin duplicar
    mappings_by_idx: dict[int, SlotMappingConfig] = {}
    used_src_in_group: dict[tuple[str, str], set[int]] = {}

    def _slot_order_key(sl: FsbSampleSlot) -> tuple[int, int, float, int]:
        muted_rank = 1 if sl.is_muted_donor_slot else 0
        fill_rank = 1 if ("fill" in sl.name.lower() or sl.is_aux_layer) else 0
        return (muted_rank, fill_rank, sl.effective_rpm or 0.0, sl.global_index)

    for sl in sorted(donor.slots, key=_slot_order_key):
        cfg = SlotMappingConfig(donor_global_index=sl.global_index)

        # 1. Slots nulos / mudos (ej. Exhaust 'null')
        if sl.load_role == "null":
            cfg.action = "silence" if sl.mode == FSB5_MODE_VORBIS else "keep_donor"
            cfg.match_reason = "Slot silencioso (Null)"
            mappings_by_idx[sl.global_index] = cfg
            continue

        # 1b. Pistas permanentemente mudas (-42 dB) en el mezclador del donante
        if sl.is_muted_donor_slot and sl.mode == FSB5_MODE_VORBIS:
            cfg.action = "silence"
            cfg.match_reason = "Pista muda (-42 dB) en donante (silenciada)"
            mappings_by_idx[sl.global_index] = cfg
            continue

        # 1c. Capas de vibración de cabina sin Auto-Pitch en el donante (ej. 'Generic_vibration' en SuperV8)
        if (
            not sl.has_rpm_autopitch
            and sl.autopitch_ref_rpm is None
            and "vibration" in sl.name.lower()
            and sl.mode != FSB5_MODE_VORBIS
        ):
            cfg.action = "keep_donor"
            cfg.match_reason = "Vibración auxiliar sin Auto-Pitch (se conserva donante)"
            mappings_by_idx[sl.global_index] = cfg
            continue

        # 1d. Capa secundaria 'fill_int' duplicada en cabina: silenciar para evitar solapamiento de fase
        if (
            not sl.is_muted_donor_slot
            and sl.perspective == "int"
            and sl.load_role in ("on", "off")
            and "fill" in sl.name.lower()
        ):
            cfg.action = "silence"
            cfg.match_reason = "Capa fill secundaria silenciada (evita solapamiento)"
            mappings_by_idx[sl.global_index] = cfg
            continue

        # 2. Preservar siempre Starter / Starting originales de AMS2 (incluso en sub-bancos Vorbis)
        if sl.load_role == "starter":
            cfg.action = "keep_donor"
            cfg.match_reason = "Starter original de AMS2 conservado"
            mappings_by_idx[sl.global_index] = cfg
            continue
        if sl.load_role == "starting":
            cfg.action = "keep_donor"
            cfg.match_reason = "Encendido (Starting) original de AMS2 conservado"
            mappings_by_idx[sl.global_index] = cfg
            continue
        if sl.load_role == "throttle_cut" and not has_tcut_in_src and sl.mode != FSB5_MODE_VORBIS:
            cfg.action = "keep_donor"
            cfg.match_reason = "Sin Throttle Cut en AC (se conserva donante AMS2)"
            mappings_by_idx[sl.global_index] = cfg
            continue

        scored_candidates: list[tuple[float, SourceSampleItem]] = []
        dst_norm_rpm = (sl.effective_rpm / dst_max_rpm) if sl.effective_rpm else 0.5
        need_loop = sl.loop is not None or sl.inst_infinite_loop
        donor_is_limiter = sl.load_role == "pops" and (
            "limiter" in sl.name.lower() or (need_loop and sl.has_rpm_autopitch)
        )
        grp_key = (sl.perspective, sl.load_role)
        already_used = used_src_in_group.get(grp_key, set())

        for it in decodable_items:
            if it.perspective == "fx" or it.load_role in ("fx", "null"):
                continue

            score = 0.0
            # A. Coincidencia de perspectiva (Interior vs Exterior)
            if it.perspective == sl.perspective:
                score += 45.0
            elif it.perspective in ("int", "ext") and sl.perspective in ("int", "ext"):
                score += 12.0  # fallback cruzado si el mod de AC solo tiene 1 perspectiva
            else:
                score -= 40.0

            # Excluir o penalizar fuertemente micrófonos de replay/far/front en el exterior (priorizar Chase Cam)
            if sl.perspective == "ext" and not is_chase_cam_sample(it.name):
                score -= 60.0
            if it.is_aux_layer and not sl.is_aux_layer:
                score -= 40.0

            # B. Coincidencia de rol de carga (Idle / On / Off / ThrottleCut / Pops / Starter)
            if it.load_role == sl.load_role:
                score += 60.0
            elif sl.load_role == "throttle_cut" and it.load_role == "upshift":
                score += 55.0
            elif sl.load_role == "idle" and it.load_role == "off":
                score += 20.0
            elif sl.load_role in ("on", "off") and it.load_role in ("on", "off"):
                score += 10.0
            elif sl.load_role == "pops" and it.load_role in ("downshift", "upshift", "throttle_cut"):
                score += 22.0
            else:
                score -= 50.0

            # C. Proximidad proporcional de RPM + Diversidad en la escalera de RPM (sin duplicar la misma muestra)
            if sl.load_role in ("on", "off", "idle"):
                it_norm_rpm = (it.effective_rpm / src_max_rpm) if it.effective_rpm else 0.5
                rpm_diff = abs(dst_norm_rpm - it_norm_rpm)
                score += max(0.0, 42.0 * (1.0 - min(1.2, rpm_diff * 1.8)))
                if it.has_rpm_autopitch:
                    score += 18.0
                elif it.effective_rpm is None:
                    score -= 22.0
                if it.is_aux_layer and not sl.is_aux_layer:
                    score -= 32.0
                if (
                    not sl.is_muted_donor_slot
                    and sl.load_role in ("on", "off")
                    and it.load_role == sl.load_role
                    and it.perspective == sl.perspective
                    and not it.is_aux_layer
                ):
                    if it.index in already_used:
                        score -= 18.0
                    else:
                        score += 12.0
                if sl.perspective == "ext":
                    donor_low = sl.name.lower()
                    cand_low = it.name.lower()
                    if "front" in donor_low:
                        if "front" in cand_low:
                            score += 10.0
                    else:
                        if "front" in cand_low:
                            score -= 10.0
                        elif "rear" in cand_low or "exhaust" in cand_low or "_ex_" in cand_low or cand_low.startswith("ex_"):
                            score += 6.0

            # D. Preferencia por loop continuo vs one-shot (y limitador en bucle vs petardeo puntual)
            it_has_loop = it.loop is not None or it.inst_infinite_loop
            it_is_limiter = "limiter" in it.name.lower() or "_lim" in it.name.lower() or "gt3lim" in it.name.lower()
            if donor_is_limiter:
                if it_is_limiter:
                    score += 22.0
                if it_has_loop and it.duration_sec >= 0.5:
                    score += 12.0
            elif need_loop:
                if it_has_loop:
                    score += 10.0
                if it.duration_sec >= 0.8:
                    score += 10.0
                elif it.duration_sec < 0.35:
                    score -= 35.0
            else:
                if sl.load_role in ("pops", "throttle_cut"):
                    if not it_is_limiter and it.duration_sec <= 1.8:
                        score += 14.0
                    elif it_is_limiter and it_has_loop:
                        score -= 8.0

            scored_candidates.append((score, it))

        scored_candidates.sort(key=lambda pair: pair[0], reverse=True)
        best_item: SourceSampleItem | None = None
        best_score = -1e9
        if scored_candidates:
            top_score = scored_candidates[0][0]
            if sl.load_role == "pops" and not donor_is_limiter and top_score > 15.0:
                near_ties = [pair for pair in scored_candidates if pair[0] >= top_score - 15.0]
                chosen_score, best_item = near_ties[pops_usage_counter % len(near_ties)]
                best_score = chosen_score
                pops_usage_counter += 1
            else:
                best_score, best_item = scored_candidates[0]

        if best_item is not None and best_score > 15.0:
            cfg.action = "replace_source"
            cfg.source_index = best_item.index
            cfg.override_root_rpm = best_item.effective_rpm
            cfg.match_score = round(best_score, 1)
            muted_suffix = " [Pista muda donante]" if sl.is_muted_donor_slot else ""
            cfg.match_reason = (
                f"Auto ({best_item.perspective.upper()}/{best_item.load_role.upper()}, "
                f"RPM {best_item.rpm_summary}){muted_suffix}"
            )
            if not sl.is_muted_donor_slot and sl.load_role in ("on", "off"):
                used_src_in_group.setdefault(grp_key, set()).add(best_item.index)
        else:
            cfg.action = "keep_donor"
            cfg.match_reason = "Sin equivalente claro (conserva donante)"

        mappings_by_idx[sl.global_index] = cfg

    return [mappings_by_idx[sl.global_index] for sl in donor.slots]


def collect_local_proj_guids(proj_payload: bytes) -> list[bytes]:
    """Recolecta todos los GUIDs de 16 bytes definidos localmente en PROJ, ordenados por memcmp."""
    found: set[bytes] = set()
    zero16 = b"\x00" * 16

    def walk(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(proj_payload, start, end):
            if cid == b"LIST":
                walk(ps + 4, pe)
            elif cid in LOCAL_OBJECT_BASE_CHUNKS and pe - ps >= 16:
                g = bytes(proj_payload[ps : ps + 16])
                if g != zero16:
                    found.add(g)

    walk(0, len(proj_payload))
    return sorted(found)


def build_order_preserving_guid_map(local_guids_sorted: list[bytes], mod_identifier: str) -> dict[bytes, bytes]:
    """Genera nuevos GUIDs UUIDv4 deterministas preservando estrictamente el orden memcmp de FMOD."""
    n = len(local_guids_sorted)
    if n == 0:
        return {}

    generated: set[bytes] = set()
    counter = 0
    seed_prefix = f"ams2-sound-porter:v1:{mod_identifier.strip().lower()}:".encode("utf-8")
    while len(generated) < n:
        digest = bytearray(hashlib.sha256(seed_prefix + str(counter).encode("ascii")).digest()[:16])
        counter += 1
        # Bits estándar RFC 4122 / Microsoft FMOD_GUID v4:
        # Data3 es uint16 LE en bytes [6:8] -> nibble alto en byte[7] = 0x40
        # Data4[0] en byte[8] -> bits altos = 0x80
        digest[7] = (digest[7] & 0x0F) | 0x40
        digest[8] = (digest[8] & 0x3F) | 0x80
        g_bytes = bytes(digest)
        if g_bytes != b"\x00" * 16 and g_bytes not in local_guids_sorted:
            generated.add(g_bytes)

    new_sorted = sorted(generated)
    return dict(zip(local_guids_sorted, new_sorted))


def _riff_chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if (len(payload) & 1) else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _riff_list(ltype: bytes, children_bytes: bytes) -> bytes:
    body = ltype + children_bytes
    return _riff_chunk(b"LIST", body)


def _extract_first_subchunk_guid(list_payload: bytes) -> bytes:
    for scid, sps, spe, _ in iter_riff_chunks(list_payload, 4, len(list_payload)):
        if scid != b"LCNT" and spe - sps >= 16:
            return bytes(list_payload[sps : sps + 16])
    return b"\xff" * 16


def _inject_throttle_cut_into_proj(
    proj_payload: bytes,
    donor: FmodBank,
    tcut_int_loc: tuple[int, int] | None,
    tcut_ext_loc: tuple[int, int] | None,
    duck_on_throttle_db: float = -13.5,
    trigger_min: float = 0.15,
    seed_label: str = "tcut",
) -> tuple[bytes, int]:
    """Inyecta objetos FMOD de ThrottleCut (WAV, GBUS, WAIT, PMLB, CTRL, CURV) en eventos de motor vacíos."""
    existing_guids = set(collect_local_proj_guids(proj_payload))
    existing_guids.add(b"\x00" * 16)
    alloc_counter = 0

    def alloc_local_guid(tag: str) -> bytes:
        nonlocal alloc_counter
        while True:
            raw_h = bytearray(
                hashlib.sha256(f"ams2-tcut:{seed_label}:{tag}:{alloc_counter}".encode("utf-8")).digest()[:16]
            )
            alloc_counter += 1
            raw_h[7] = (raw_h[7] & 0x0F) | 0x40
            raw_h[8] = (raw_h[8] & 0x3F) | 0x80
            gb = bytes(raw_h)
            if gb not in existing_guids:
                existing_guids.add(gb)
                return gb

    # Pasada 1: Recolectar PRMB, EVTB, GBUS, CTRL, CURV y PMLB vacíos de ThrottleCut
    prms_name: dict[bytes, str] = {}
    evtb_mbg: dict[bytes, bytes] = {}
    gbus_parent_mbg: dict[bytes, bytes] = {}
    gbus_has_vol_prop: set[bytes] = set()
    ctrs_by_target: dict[bytes, list[tuple[bytes, bytes, int]]] = {}
    crvs_pts: dict[bytes, list[tuple[float, float]]] = {}
    empty_tcut_pmls: list[tuple[bytes, bytes, bytes]] = []  # (pml_self_guid, param_guid, event_guid)

    def scan_pass(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(proj_payload, start, end):
            if cid == b"PRMB" and pe - ps >= 31:
                p = proj_payload[ps:pe]
                slen = struct.unpack_from("<H", p, 21)[0]
                if 23 + slen <= len(p):
                    prms_name[bytes(p[:16])] = p[23 : 23 + slen].decode("utf-8", "replace").strip().lower()
            elif cid == b"EVTB" and pe - ps >= 80:
                p = proj_payload[ps:pe]
                evtb_mbg[bytes(p[:16])] = bytes(p[64:80])
            elif cid == b"CTRL" and pe - ps >= 68:
                p = proj_payload[ps:pe]
                prop = struct.unpack_from("<I", p, 64)[0]
                ctrs_by_target.setdefault(bytes(p[16:32]), []).append((bytes(p[32:48]), bytes(p[48:64]), prop))
            elif cid == b"CURV" and pe - ps >= 34:
                p = proj_payload[ps:pe]
                cnt = struct.unpack_from("<H", p, 32)[0] >> 1
                pts: list[tuple[float, float]] = []
                if cnt > 0 and len(p) >= 36:
                    stride = struct.unpack_from("<H", p, 34)[0]
                    if stride >= 8 and len(p) >= 36 + cnt * stride:
                        for ki in range(cnt):
                            x, y = struct.unpack_from("<ff", p, 36 + ki * stride)
                            pts.append((float(x), float(y)))
                crvs_pts[bytes(p[:16])] = pts
            elif cid == b"LIST":
                ltype = proj_payload[ps : ps + 4]
                if ltype == b"GBUS":
                    gg: bytes | None = None
                    pmbg: bytes | None = None
                    has_p0 = False
                    for scid, sps, spe, _ in iter_riff_chunks(proj_payload, ps + 4, pe):
                        if scid == b"GBSB" and spe - sps >= 34:
                            gg = bytes(proj_payload[sps : sps + 16])
                            pmbg = bytes(proj_payload[sps + 18 : sps + 34])
                        elif scid == b"LIST" and proj_payload[sps : sps + 4] == b"PRPS":
                            for pcid, pps, ppe, _ in iter_riff_chunks(proj_payload, sps + 4, spe):
                                if pcid == b"PROP" and ppe - pps >= 26:
                                    if struct.unpack_from("<I", proj_payload, pps)[0] == 0:
                                        has_p0 = True
                    if gg and pmbg:
                        gbus_parent_mbg[gg] = pmbg
                        if has_p0:
                            gbus_has_vol_prop.add(gg)
                scan_pass(ps + 4, pe)

    scan_pass(0, len(proj_payload))

    def scan_pmls(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(proj_payload, start, end):
            if cid == b"PMLB" and pe - ps >= 50:
                p = proj_payload[ps:pe]
                pg = bytes(p[16:32])
                eg = bytes(p[32:48])
                nboxes = struct.unpack_from("<H", p, 48)[0] >> 1
                if nboxes == 0 and prms_name.get(pg, "") in ("throttlecut", "throttle_cut"):
                    empty_tcut_pmls.append((bytes(p[:16]), pg, eg))
            elif cid == b"LIST":
                scan_pmls(ps + 4, pe)

    scan_pmls(0, len(proj_payload))
    if not empty_tcut_pmls:
        return proj_payload, 0

    evt_role_map = {ev.guid_bytes: ev.role for ev in donor.events}
    wav_guid_by_loc: dict[tuple[int, int], bytes] = {}
    new_wav_chunks: list[tuple[bytes, bytes]] = []
    new_gbus_chunks: list[tuple[bytes, bytes]] = []
    new_wait_chunks: list[tuple[bytes, bytes]] = []
    new_ctrl_chunks: list[tuple[bytes, bytes]] = []
    new_curv_chunks: list[tuple[bytes, bytes]] = []
    pml_box_patches: dict[tuple[bytes, bytes], tuple[bytes, float, float]] = {}
    pml_ctro_patches: dict[tuple[bytes, bytes], list[bytes]] = {}
    gbus_prop0_extra_ctrls: dict[bytes, list[bytes]] = {}

    trig_lo = max(0.05, min(0.85, float(trigger_min)))
    trig_len = max(0.10, 1.0 - trig_lo)
    injected_events = 0

    for pml_self, tcut_pg, eg in empty_tcut_pmls:
        ev_role = evt_role_map.get(eg, "")
        if ev_role not in ("EngineInt", "AuxInt", "EngineExt"):
            continue
        loc = tcut_int_loc if ev_role in ("EngineInt", "AuxInt") else tcut_ext_loc
        mbg = evtb_mbg.get(eg)
        if loc is None or mbg is None:
            continue

        if loc not in wav_guid_by_loc:
            wg = alloc_local_guid(f"wav_{loc[0]}_{loc[1]}")
            wav_guid_by_loc[loc] = wg
            wav_pay = wg + struct.pack("<HHHII", 0x000C, loc[0], 0, loc[1], loc[0])
            new_wav_chunks.append((wg, _riff_chunk(b"WAV ", wav_pay)))
        else:
            wg = wav_guid_by_loc[loc]

        gbus_g = alloc_local_guid(f"gbus_{ev_role}")
        wait_g = alloc_local_guid(f"wait_{ev_role}")

        gbsb_pay = gbus_g + b"\x18\x00" + mbg + b"\x03\x00\x00\x00\xff\xff\xff\xff"
        bus_pay = bytes.fromhex("00030000000100010008000000000000000000ffffff7f02000000")
        gbus_list_bytes = _riff_list(
            b"GBUS",
            _riff_chunk(b"GBSB", gbsb_pay)
            + _riff_chunk(b"BUS ", bus_pay)
            + _riff_list(b"PRPS", b""),
        )
        new_gbus_chunks.append((gbus_g, gbus_list_bytes))

        waib_pay = wait_g + wg
        inst_pay = (
            b"\x00" * 16
            + struct.pack("<ffii", 0.0, 0.0, 0, 0)
            + b"\x00"
            + struct.pack("<f", 100.0)
            + b"\x00" * 32
            + struct.pack("<fI", 1.0, 0)
            + b"\xff\xff\xff\x7f\x18\x00"
            + gbus_g
            + bytes.fromhex("030000000300000002000000000000000000000000000000")
        )
        wait_list_bytes = _riff_list(
            b"WAIT",
            _riff_chunk(b"WAIB", waib_pay)
            + _riff_chunk(b"INST", inst_pay)
            + _riff_list(b"PRPS", b""),
        )
        new_wait_chunks.append((wait_g, wait_list_bytes))
        pml_box_patches[(pml_self, eg)] = (wait_g, trig_lo, trig_len)

        # Atenuar momentáneamente los buses On-Throttle durante el corte de inyección (ThrottleCut)
        if duck_on_throttle_db < -0.5:
            ev_ctrls: list[bytes] = []
            for cand_gg, cand_mbg in gbus_parent_mbg.items():
                if cand_mbg != mbg or cand_gg not in gbus_has_vol_prop:
                    continue
                is_on_thr_bus = False
                is_muted = False
                for c_pg, c_cg, prop in ctrs_by_target.get(cand_gg, []):
                    if prop != 0:
                        continue
                    pts = crvs_pts.get(c_cg, [])
                    if pts and max(y for _, y in pts) <= -36.0:
                        is_muted = True
                        break
                    if prms_name.get(c_pg, "") in ("throttle", "load") and len(pts) >= 2:
                        if pts[-1][1] - pts[0][1] >= 8.0:
                            is_on_thr_bus = True
                if is_on_thr_bus and not is_muted:
                    cg_ctrl = alloc_local_guid(f"ctrl_{ev_role}_{len(ev_ctrls)}")
                    cg_curv = alloc_local_guid(f"curv_{ev_role}_{len(ev_ctrls)}")
                    ctrl_pay = cg_ctrl + cand_gg + tcut_pg + cg_curv + struct.pack("<I", 0)
                    curv_pay = (
                        cg_curv
                        + cg_ctrl
                        + struct.pack("<HH", 0x0005, 0x0010)
                        + struct.pack("<ffffffff", 0.0, 0.0, 0.0, 0.0, 0.22, float(duck_on_throttle_db), 0.0, 0.0)
                    )
                    new_ctrl_chunks.append((cg_ctrl, _riff_chunk(b"CTRL", ctrl_pay)))
                    new_curv_chunks.append((cg_curv, _riff_chunk(b"CURV", curv_pay)))
                    gbus_prop0_extra_ctrls.setdefault(cand_gg, []).append(cg_ctrl)
                    ev_ctrls.append(cg_ctrl)
            if ev_ctrls:
                pml_ctro_patches[(pml_self, eg)] = sorted(ev_ctrls)

        injected_events += 1

    if injected_events == 0:
        return proj_payload, 0

    def _patch_gbus_list_chunk(gbus_payload: bytes, extra_cgs: list[bytes]) -> bytes:
        out_ch = bytearray()
        for scid, sps, spe, _ in iter_riff_chunks(gbus_payload, 4, len(gbus_payload)):
            sp = gbus_payload[sps:spe]
            if scid == b"LIST" and sp[:4] == b"PRPS":
                prps_ch = bytearray()
                for pcid, pps, ppe, _ in iter_riff_chunks(sp, 4, len(sp)):
                    pp = sp[pps:ppe]
                    if pcid == b"PROP" and len(pp) >= 26 and struct.unpack_from("<I", pp, 0)[0] == 0:
                        hdr = struct.unpack_from("<H", pp, 24)[0]
                        nc = hdr >> 1
                        if nc == 0:
                            cur_cgs: list[bytes] = []
                            tail = pp[26:]
                        else:
                            stride = struct.unpack_from("<H", pp, 26)[0]
                            cur_cgs = [bytes(pp[28 + i * stride : 44 + i * stride]) for i in range(nc)]
                            tail = pp[28 + nc * stride :]
                        merged_cgs = sorted(set(cur_cgs) | set(extra_cgs))
                        new_pp = (
                            pp[:24]
                            + struct.pack("<HH", (len(merged_cgs) << 1) | 1, 0x0010)
                            + b"".join(merged_cgs)
                            + tail
                        )
                        prps_ch += _riff_chunk(b"PROP", new_pp)
                    else:
                        prps_ch += _riff_chunk(pcid, pp)
                out_ch += _riff_list(b"PRPS", bytes(prps_ch))
            else:
                out_ch += _riff_chunk(scid, sp)
        return _riff_list(b"GBUS", bytes(out_ch))

    def _rebuild_sorted_table(list_payload: bytes, ltype: bytes, added_items: list[tuple[bytes, bytes]], is_gbus: bool = False) -> bytes:
        items: list[tuple[bytes, bytes]] = list(added_items)
        for scid, sps, spe, _ in iter_riff_chunks(list_payload, 4, len(list_payload)):
            if scid == b"LCNT":
                continue
            sp = list_payload[sps:spe]
            if scid == b"LIST":
                g_key = _extract_first_subchunk_guid(sp)
                if is_gbus and g_key in gbus_prop0_extra_ctrls:
                    chunk_b = _patch_gbus_list_chunk(sp, gbus_prop0_extra_ctrls[g_key])
                else:
                    chunk_b = _riff_chunk(b"LIST", sp)
            else:
                g_key = bytes(sp[:16])
                chunk_b = _riff_chunk(scid, sp)
            items.append((g_key, chunk_b))
        items.sort(key=lambda pair: pair[0])
        body = _riff_chunk(b"LCNT", struct.pack("<I", len(items))) + b"".join(ch for _, ch in items)
        return _riff_list(ltype, body)

    def _rebuild_proj_range(buf_in: bytes, start: int, end: int) -> bytes:
        out = bytearray()
        for cid, ps, pe, _ in iter_riff_chunks(buf_in, start, end):
            pay = buf_in[ps:pe]
            if cid == b"LIST":
                ltype = pay[:4]
                if ltype == b"WAVS" and new_wav_chunks:
                    out += _rebuild_sorted_table(pay, b"WAVS", new_wav_chunks)
                elif ltype == b"GBSS" and (new_gbus_chunks or gbus_prop0_extra_ctrls):
                    out += _rebuild_sorted_table(pay, b"GBSS", new_gbus_chunks, is_gbus=True)
                elif ltype == b"WAIS" and new_wait_chunks:
                    out += _rebuild_sorted_table(pay, b"WAIS", new_wait_chunks)
                elif ltype == b"CTRS" and new_ctrl_chunks:
                    out += _rebuild_sorted_table(pay, b"CTRS", new_ctrl_chunks)
                elif ltype == b"CRVS" and new_curv_chunks:
                    out += _rebuild_sorted_table(pay, b"CRVS", new_curv_chunks)
                elif ltype == b"PMLO":
                    pml_key: tuple[bytes, bytes] | None = None
                    for scid, sps, spe, _ in iter_riff_chunks(pay, 4, len(pay)):
                        if scid == b"PMLB" and spe - sps >= 48:
                            sp = pay[sps:spe]
                            cand_k = (bytes(sp[:16]), bytes(sp[32:48]))
                            if cand_k in pml_box_patches:
                                pml_key = cand_k
                                break
                    if pml_key is not None:
                        wg_inst, t_lo, t_len = pml_box_patches[pml_key]
                        ctro_cgs = pml_ctro_patches.get(pml_key, [])
                        pmlo_ch = bytearray()
                        for scid, sps, spe, _ in iter_riff_chunks(pay, 4, len(pay)):
                            sp = pay[sps:spe]
                            if scid == b"PMLB":
                                new_pmlb = (
                                    sp[:48]
                                    + struct.pack("<HH", 0x0003, 0x0018)
                                    + wg_inst
                                    + struct.pack("<ff", t_lo, t_len)
                                )
                                pmlo_ch += _riff_chunk(b"PMLB", new_pmlb)
                            elif scid == b"CTRO" and ctro_cgs:
                                new_ctro = (
                                    struct.pack("<HH", (len(ctro_cgs) << 1) | 1, 0x0010)
                                    + b"".join(ctro_cgs)
                                )
                                pmlo_ch += _riff_chunk(b"CTRO", new_ctro)
                            else:
                                pmlo_ch += _riff_chunk(scid, sp)
                        out += _riff_list(b"PMLO", bytes(pmlo_ch))
                    else:
                        out += _riff_chunk(b"LIST", pay)
                else:
                    rebuilt_inner = _rebuild_proj_range(pay, 4, len(pay))
                    out += _riff_list(ltype, rebuilt_inner)
            else:
                out += _riff_chunk(cid, pay)
        return bytes(out)

    return _rebuild_proj_range(proj_payload, 0, len(proj_payload)), injected_events


def _patch_proj_chunks(
    proj_payload: bytes,
    wav_redirects: dict[bytes, tuple[int, int]],
    autopitch_updates: dict[bytes, float],
    guid_map: dict[bytes, bytes] | None,
    rpm_bands: list[RpmBandConfig] | None = None,
    gain_db_deltas: dict[bytes, float] | None = None,
    silence_wav_guids: set[bytes] | None = None,
    unmute_wav_guids: set[bytes] | None = None,
) -> bytes:
    """Retorna los sub-chunks hijos directos de LIST:PROJ ya parcheados (excepto SNDH que se calcula al final)."""
    buf = bytearray(proj_payload)

    # Preparar tablas de parcheo de bandas RPM y curvas de crossfade (antes de remapear GUIDs)
    pmlb_box_updates: dict[tuple[bytes, bytes], list[RpmBandConfig]] = {}
    fade_in_curv_updates: dict[bytes, tuple[float, float, float]] = {}
    fade_out_curv_updates: dict[bytes, tuple[float, float, float]] = {}
    bus_cutoff_max_rpm: dict[bytes, float] = {}
    max_band_rpm = 0.0

    if rpm_bands:
        for b in rpm_bands:
            if b.is_muted_bus:
                rmin = 0.0
                rmax = 0.0
                fi_end = 0.0
                fo_start = 0.0
            else:
                rmin = max(0.0, float(b.rpm_min))
                rmax = max(rmin + 10.0, float(b.rpm_max))
                fi_end = max(rmin, min(rmax, float(b.fade_in_end)))
                fo_start = max(rmin, min(rmax, float(b.fade_out_start)))
                if rmax > max_band_rpm:
                    max_band_rpm = rmax
            pmlb_box_updates.setdefault((b.pml_guid, b.event_guid), []).append(b)
            if b.fade_in_curv_guid is not None:
                fade_in_curv_updates[b.fade_in_curv_guid] = (rmin, fi_end, rmax)
            if b.fade_out_curv_guid is not None:
                fade_out_curv_updates[b.fade_out_curv_guid] = (rmin, fo_start, rmax)
            if b.bus_rpm_cutoff is not None and not b.is_muted_bus:
                c_guid = b.bus_rpm_cutoff[0]
                bus_cutoff_max_rpm[c_guid] = max(bus_cutoff_max_rpm.get(c_guid, 0.0), rmax)

    # Mapear WAIB -> WAV GUID antes de parchear para poder actualizar también MUIT->INST si aplica
    waib_to_wav: dict[bytes, bytes] = {}
    if autopitch_updates or gain_db_deltas or silence_wav_guids or unmute_wav_guids:
        def collect_waibs(start: int, end: int) -> None:
            for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
                if cid == b"WAIB" and pe - ps >= 32:
                    waib_to_wav[bytes(buf[ps : ps + 16])] = bytes(buf[ps + 16 : ps + 32])
                elif cid == b"LIST":
                    collect_waibs(ps + 4, pe)

        collect_waibs(0, len(buf))

    # 1. Aplicar redirecciones WAV, Auto-Pitch RPM, ganancia INST y edición de bandas PMLB / CURV ANTES de remapear GUIDs
    def patch_tree(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"WAV " and pe - ps >= 30:
                wg = bytes(buf[ps : ps + 16])
                if wg in wav_redirects:
                    new_sb, new_smp_idx = wav_redirects[wg]
                    _, _, flags, _, _ = struct.unpack_from("<HHHII", buf, ps + 16)
                    struct.pack_into("<HHHII", buf, ps + 16, 0x000C, new_sb, flags, new_smp_idx, new_sb)
            elif cid == b"PRMB" and max_band_rpm > 2500.0 and pe - ps >= 31:
                slen = struct.unpack_from("<H", buf, ps + 21)[0]
                if 23 + slen + 8 <= pe - ps:
                    pname = buf[ps + 23 : ps + 23 + slen].decode("latin1", "replace").rstrip("\x00").strip().lower()
                    if pname in ("rpm", "rpms"):
                        _pmin, pmax = struct.unpack_from("<ff", buf, ps + 23 + slen)
                        if 2500.0 <= pmax < max_band_rpm:
                            struct.pack_into("<f", buf, ps + 23 + slen + 4, float(max_band_rpm))
            elif cid == b"PMLB" and pmlb_box_updates and pe - ps >= 52:
                pkey = (bytes(buf[ps : ps + 16]), bytes(buf[ps + 32 : ps + 48]))
                if pkey in pmlb_box_updates:
                    nboxes = struct.unpack_from("<H", buf, ps + 48)[0] >> 1
                    stride = struct.unpack_from("<H", buf, ps + 50)[0]
                    if stride >= 24:
                        for b_cfg in pmlb_box_updates[pkey]:
                            if 0 <= b_cfg.box_index < nboxes:
                                boff = ps + 52 + b_cfg.box_index * stride
                                if boff + 24 <= pe:
                                    if b_cfg.is_muted_bus:
                                        struct.pack_into("<ff", buf, boff + 16, 0.0, 0.0)
                                    else:
                                        rmin = max(0.0, float(b_cfg.rpm_min))
                                        rmax = max(rmin + 10.0, float(b_cfg.rpm_max))
                                        struct.pack_into("<ff", buf, boff + 16, rmin, rmax - rmin)
            elif cid == b"CURV" and (fade_in_curv_updates or fade_out_curv_updates or bus_cutoff_max_rpm) and pe - ps >= 52:
                cg = bytes(buf[ps : ps + 16])
                cnt = struct.unpack_from("<H", buf, ps + 32)[0] >> 1
                stride = struct.unpack_from("<H", buf, ps + 34)[0]
                if cnt >= 2 and stride >= 8 and ps + 36 + cnt * stride <= pe:
                    if cg in fade_in_curv_updates:
                        rmin, fi_end, rmax = fade_in_curv_updates[cg]
                        if rmin == 0.0 and rmax == 0.0:
                            struct.pack_into("<ff", buf, ps + 36, 0.0, 0.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, 0.0, 0.0)
                        elif fi_end <= rmin + 1.0:
                            struct.pack_into("<ff", buf, ps + 36, rmin, 1.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, min(rmax, rmin + 1.0), 1.0)
                        else:
                            struct.pack_into("<ff", buf, ps + 36, rmin, 0.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, fi_end, 1.0)
                    elif cg in fade_out_curv_updates:
                        rmin, fo_start, rmax = fade_out_curv_updates[cg]
                        if rmin == 0.0 and rmax == 0.0:
                            struct.pack_into("<ff", buf, ps + 36, 0.0, 0.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, 0.0, 0.0)
                        elif fo_start >= rmax - 1.0:
                            struct.pack_into("<ff", buf, ps + 36, max(rmin, rmax - 1.0), 1.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, rmax, 1.0)
                        else:
                            struct.pack_into("<ff", buf, ps + 36, fo_start, 1.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, rmax, 0.0)
                    elif cg in bus_cutoff_max_rpm:
                        req_max = bus_cutoff_max_rpm[cg]
                        x0, _ = struct.unpack_from("<ff", buf, ps + 36)
                        x1, _ = struct.unpack_from("<ff", buf, ps + 36 + stride)
                        if req_max > x0:
                            slope_w = max(800.0, x1 - x0)
                            new_x0 = req_max
                            new_x1 = req_max + slope_w
                            struct.pack_into("<f", buf, ps + 36, new_x0)
                            struct.pack_into("<f", buf, ps + 36 + stride, new_x1)
                            if cnt >= 3:
                                x2, _ = struct.unpack_from("<ff", buf, ps + 36 + 2 * stride)
                                struct.pack_into("<f", buf, ps + 36 + 2 * stride, max(x2, new_x1 + 1000.0))
            elif cid == b"LIST":
                ltype = bytes(buf[ps : ps + 4])
                if ltype == b"WAIT":
                    target_wg: bytes | None = None
                    inst_span: tuple[int, int] | None = None
                    for scid, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                        if scid == b"WAIB" and spe - sps >= 32:
                            target_wg = bytes(buf[sps + 16 : sps + 32])
                        elif scid == b"INST" and spe - sps >= 20:
                            inst_span = (sps, spe)
                    if target_wg and inst_span:
                        ips, ipe = inst_span
                        if ipe - ips >= 73 and target_wg in autopitch_updates:
                            if bytes(buf[ips + 53 : ips + 69]) != b"\x00" * 16:
                                struct.pack_into("<f", buf, ips + 69, float(autopitch_updates[target_wg]))
                        if silence_wav_guids and target_wg in silence_wav_guids:
                            struct.pack_into("<f", buf, ips + 16, -96.0)
                        else:
                            cur_db = struct.unpack_from("<f", buf, ips + 16)[0]
                            db_changed = False
                            if unmute_wav_guids and target_wg in unmute_wav_guids and cur_db <= -90.0:
                                cur_db = 0.0
                                db_changed = True
                            if gain_db_deltas and target_wg in gain_db_deltas and abs(gain_db_deltas[target_wg]) > 1e-4:
                                cur_db = max(-96.0, min(24.0, cur_db + float(gain_db_deltas[target_wg])))
                                db_changed = True
                            if db_changed:
                                struct.pack_into("<f", buf, ips + 16, float(cur_db))
                    patch_tree(ps + 4, pe)
                elif ltype == b"MUIT" and (autopitch_updates or silence_wav_guids or unmute_wav_guids):
                    child_rpms: list[float] = []
                    child_wgs: list[bytes] = []
                    inst_span = None
                    for scid, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                        if scid == b"PLST" and spe - sps >= 10:
                            sp = buf[sps:spe]
                            nc = struct.unpack_from("<H", sp, 8)[0] >> 1
                            if nc > 0 and len(sp) >= 12:
                                stride = struct.unpack_from("<H", sp, 10)[0]
                                if stride >= 16:
                                    for ci in range(nc):
                                        coff = 12 + ci * stride
                                        if coff + 16 <= len(sp):
                                            ch_g = bytes(sp[coff : coff + 16])
                                            ch_wg = waib_to_wav.get(ch_g)
                                            if ch_wg:
                                                child_wgs.append(ch_wg)
                                                if ch_wg in autopitch_updates:
                                                    child_rpms.append(float(autopitch_updates[ch_wg]))
                        elif scid == b"INST" and spe - sps >= 20:
                            inst_span = (sps, spe)
                    if inst_span:
                        ips, ipe = inst_span
                        if child_rpms and ipe - ips >= 73:
                            if bytes(buf[ips + 53 : ips + 69]) != b"\x00" * 16:
                                avg_rpm = sum(child_rpms) / len(child_rpms)
                                struct.pack_into("<f", buf, ips + 69, float(avg_rpm))
                        if child_wgs:
                            if silence_wav_guids and all(wg in silence_wav_guids for wg in child_wgs):
                                struct.pack_into("<f", buf, ips + 16, -96.0)
                            elif unmute_wav_guids and any(wg in unmute_wav_guids for wg in child_wgs):
                                cur_db = struct.unpack_from("<f", buf, ips + 16)[0]
                                if cur_db <= -90.0:
                                    struct.pack_into("<f", buf, ips + 16, 0.0)
                    patch_tree(ps + 4, pe)
                else:
                    patch_tree(ps + 4, pe)

    patch_tree(0, len(buf))

    # 2. Remapear todos los GUIDs locales dentro de cada chunk hoja de PROJ preservando el orden memcmp
    if guid_map:
        def remap_leaves(start: int, end: int) -> None:
            for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
                if cid == b"LIST":
                    remap_leaves(ps + 4, pe)
                elif cid not in (b"LCNT", b"SNDH", b"STDT", b"STBL") and pe - ps >= 16:
                    i = ps
                    limit = pe - 16
                    while i <= limit:
                        candidate = bytes(buf[i : i + 16])
                        replacement = guid_map.get(candidate)
                        if replacement is not None:
                            buf[i : i + 16] = replacement
                            i += 16
                        else:
                            i += 1

        remap_leaves(0, len(buf))
        buf = _sort_evts_in_proj_payload(buf)

    return bytes(buf)


def _collect_evib_targets(buf: bytes | bytearray) -> set[bytes]:
    """Recolecta los GUIDs de eventos sub-anidados referenciados por LIST:EVIS (EVIB[16:32])."""
    targets: set[bytes] = set()
    for cid, ps, pe, _ in iter_riff_chunks(buf, 0, len(buf)):
        if cid == b"LIST" and buf[ps : ps + 4] == b"EVIS":
            for scid, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                if scid == b"LIST":
                    for ccid, cps, cpe, _ in iter_riff_chunks(buf, sps + 4, spe):
                        if ccid == b"EVIB" and cpe - cps >= 32:
                            targets.add(bytes(buf[cps + 16 : cps + 32]))
            break
    return targets


def _sync_evts_partition_and_bnki(
    buf: bytearray,
    exported_event_guids: set[bytes] | None = None,
) -> bytearray:
    """Reordena LIST:EVTS preservando la partición FMOD [0..exported) + [exported..total) y actualiza BNKI[24:28]."""
    evib_targets = _collect_evib_targets(buf)
    exported_set = set(exported_event_guids) if exported_event_guids is not None else set()

    exported_count: int | None = None
    for cid, ps, pe, pos in iter_riff_chunks(buf, 0, len(buf)):
        if cid == b"LIST" and buf[ps : ps + 4] == b"EVTS":
            children: list[tuple[bytes | None, bytes]] = []
            for ev_cid, ev_ps, ev_pe, ev_pos in iter_riff_chunks(buf, ps + 4, pe):
                ev_pad_end = ev_pe + ((ev_pe - ev_ps) & 1)
                if ev_cid == b"LCNT":
                    continue
                chunk_full = bytes(buf[ev_pos:ev_pad_end])
                ev_guid = None
                for s_cid, s_ps, s_pe, _ in iter_riff_chunks(buf, ev_ps + 4, ev_pe):
                    if s_cid in LOCAL_OBJECT_BASE_CHUNKS and s_pe - s_ps >= 16:
                        ev_guid = bytes(buf[s_ps : s_ps + 16])
                        break
                children.append((ev_guid, chunk_full))
            if any(g is not None for g, _ in children):
                part_exported: list[tuple[bytes, bytes]] = []
                part_nested: list[tuple[bytes, bytes]] = []
                unknown: list[bytes] = []
                for g, c in children:
                    if g is None:
                        unknown.append(c)
                    elif g in evib_targets and g not in exported_set:
                        part_nested.append((g, c))
                    else:
                        part_exported.append((g, c))
                part_exported.sort(key=lambda x: x[0])
                part_nested.sort(key=lambda x: x[0])
                exported_count = len(part_exported)
                total_count = exported_count + len(part_nested) + len(unknown)
                lcnt_chunk = _riff_chunk(b"LCNT", struct.pack("<I", total_count))
                ordered = (
                    [c for _, c in part_exported]
                    + [c for _, c in part_nested]
                    + unknown
                )
                new_payload = b"EVTS" + lcnt_chunk + b"".join(ordered)
                new_chunk = _riff_chunk(b"LIST", new_payload)
                pad_end = pe + ((pe - ps) & 1)
                buf = bytearray(buf[:pos] + new_chunk + buf[pad_end:])
            break

    if exported_count is not None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, 0, len(buf)):
            if cid == b"BNKI" and pe - ps >= 28:
                struct.pack_into("<I", buf, ps + 24, exported_count)
                break
    return buf


def _sort_evts_in_proj_payload(buf: bytearray) -> bytearray:
    """Reordena LIST:EVTS respetando sub-eventos EVIS y sincroniza BNKI[24:28]."""
    return _sync_evts_partition_and_bnki(buf, None)


def _replace_sndh_and_stdt_in_proj(
    proj_payload: bytes,
    new_sndh_payload: bytes,
    new_stdt_payload: bytes | None = None,
) -> bytes:
    """Reconstruye el payload de LIST:PROJ sustituyendo SNDH y opcionalmente incrustando STDT."""
    out = bytearray()
    wrote_stdt = False
    for cid, ps, pe, _ in iter_riff_chunks(proj_payload, 0, len(proj_payload)):
        if cid == b"SNDH":
            payload = new_sndh_payload
        elif cid == b"STDT" and new_stdt_payload is not None:
            payload = new_stdt_payload
            wrote_stdt = True
        else:
            payload = proj_payload[ps:pe]
        out += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) & 1:
            out += b"\x00"

    if new_stdt_payload is not None and not wrote_stdt:
        out += b"STDT" + struct.pack("<I", len(new_stdt_payload)) + new_stdt_payload
        if len(new_stdt_payload) & 1:
            out += b"\x00"
    return bytes(out)


def build_mod_event_paths(safe_mod: str, events: list[FmodEventInfo]) -> dict[str, str]:
    """Genera rutas `event:/AMS2/Vehicles/Mods/<mod>/<rol>` limpias y únicas para cada evento del banco."""
    clean_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in safe_mod.strip()) or "custom_mod"
    role_paths: dict[str, str] = {}
    used_names: set[str] = set()
    for ev in events:
        base_name = ev.role
        candidate = base_name
        suffix = 2
        while candidate.lower() in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(candidate.lower())
        full_path = f"event:/AMS2/Vehicles/Mods/{clean_mod}/{candidate}"
        if ev.role not in role_paths:
            role_paths[ev.role] = full_path
        else:
            role_paths[candidate] = full_path
    return role_paths


@dataclass
class PortedBankResult:
    """Resultado de la compilación de un banco .bank porteado para AMS2."""

    bank_bytes: bytes
    mod_identifier: str
    standalone_guids: bool
    new_bank_guid: str
    event_guid_map: dict[str, str]  # old_guid_str -> new_guid_str
    role_to_event_guid: dict[str, str]  # ej. {'EngineExt': '{...}', 'EngineInt': '{...}', ...}
    role_to_event_path: dict[str, str]  # ej. {'EngineExt': 'event:/AMS2/Vehicles/Mods/.../EngineExt', ...}
    replaced_slots_count: int
    kept_slots_count: int
    codec_used: str
    throttle_cut_injected_events: int = 0
    local_guid_map: dict[bytes, bytes] = field(default_factory=dict, repr=False)
    fsb_blobs_cache: list[bytes] = field(default_factory=list, repr=False)


def build_ported_bank(
    donor: FmodBank,
    mappings: list[SlotMappingConfig],
    library: SourceAudioLibrary | None = None,
    mod_identifier: str = "custom_mod",
    standalone_guids: bool = True,
    sync_autopitch: bool = True,
    output_codec_mode: int = FSB5_MODE_PCM16,
    force_sample_rate: int | None = 48000,
    rpm_bands: list[RpmBandConfig] | None = None,
    throttle_cut: ThrottleCutConfig | None = None,
) -> PortedBankResult:
    """Construye un nuevo archivo .bank para Automobilista 2 a partir del donante, mapeo, bandas RPM y ThrottleCut."""
    map_by_gidx = {m.donor_global_index: m for m in mappings}
    lib_by_idx = {it.index: it for it in library.items} if library else {}

    # Si no se pasó configuración explícita de ThrottleCut, autodetectar desde el origen AC
    effective_tcut = (
        throttle_cut
        if throttle_cut is not None
        else auto_detect_throttle_cut_config(donor, library)
    )

    def _resolve_tcut_audio(src_idx: int | None, custom_wav: Path | None) -> tuple[AudioBuffer, str] | None:
        target_sr = force_sample_rate if force_sample_rate else 48000
        if custom_wav is not None and Path(custom_wav).is_file():
            raw_b = read_wav_file(Path(custom_wav))
            return conform_audio(raw_b, target_sr, 2, effective_tcut.gain_db, False), Path(custom_wav).stem
        if src_idx is not None and src_idx in lib_by_idx:
            it = lib_by_idx[src_idx]
            raw_b = it.load_audio()
            return conform_audio(raw_b, target_sr, 2, effective_tcut.gain_db, False), it.name
        return None

    tcut_int_resolved: tuple[AudioBuffer, str] | None = None
    tcut_ext_resolved: tuple[AudioBuffer, str] | None = None
    need_tcut_inject = bool(effective_tcut.enabled and donor.empty_throttle_cut_events)
    if need_tcut_inject:
        tcut_int_resolved = _resolve_tcut_audio(effective_tcut.int_source_index, effective_tcut.int_custom_wav)
        tcut_ext_resolved = _resolve_tcut_audio(effective_tcut.ext_source_index, effective_tcut.ext_custom_wav)
        if tcut_int_resolved is None and tcut_ext_resolved is not None:
            tcut_int_resolved = tcut_ext_resolved
        elif tcut_ext_resolved is None and tcut_int_resolved is not None:
            tcut_ext_resolved = tcut_int_resolved
        if tcut_int_resolved is None and tcut_ext_resolved is None:
            need_tcut_inject = False

    def resolve_replacement_audio(sl: FsbSampleSlot, cfg: SlotMappingConfig) -> tuple[AudioBuffer, float | None]:
        target_sr = force_sample_rate if force_sample_rate else sl.sample_rate
        need_loop = sl.loop is not None or sl.inst_infinite_loop

        if cfg.action == "silence":
            frames = max(1024, target_sr // 4)
            silent = np.zeros((frames, sl.channels), dtype=np.int16)
            lp = (0, frames - 1) if need_loop else None
            return AudioBuffer(silent, target_sr, lp), None

        if cfg.action == "replace_wav" and cfg.custom_wav_path is not None:
            raw_buf = read_wav_file(cfg.custom_wav_path)
            rpm = cfg.override_root_rpm or infer_rpm_from_name(cfg.custom_wav_path.stem)
            conformed = conform_audio(raw_buf, target_sr, sl.channels, cfg.gain_db, need_loop)
            return conformed, rpm

        if cfg.action == "replace_source" and cfg.source_index is not None:
            src_it = lib_by_idx.get(cfg.source_index)
            if src_it is None:
                raise ValueError(f"Índice de muestra de origen inválido: {cfg.source_index}")
            raw_buf = src_it.load_audio()
            rpm = cfg.override_root_rpm if cfg.override_root_rpm is not None else src_it.effective_rpm
            conformed = conform_audio(raw_buf, target_sr, sl.channels, cfg.gain_db, need_loop)
            return conformed, rpm

        # keep_donor (solo para sub-bancos decodificables como PCM16 o FADPCM)
        orig_buf = donor.decode_slot_to_buffer(sl)
        if abs(cfg.gain_db) > 1e-4 or orig_buf.sample_rate != target_sr:
            orig_buf = conform_audio(orig_buf, target_sr, sl.channels, cfg.gain_db, False)
        return orig_buf, cfg.override_root_rpm

    # 1. Planificar construcción de cada sub-banco FSB5
    wav_redirects: dict[bytes, tuple[int, int]] = {}
    autopitch_updates: dict[bytes, float] = {}
    replaced_count = 0
    kept_count = 0

    # Verificar si existe un sub-banco primario reconstruible (ej. SND[0] FADPCM/PCM16)
    primary_rebuild_sb: int | None = None
    subbank_build_lists: dict[int, list[FsbBuildSample]] = {}
    raw_passthrough_subbanks: dict[int, bytes] = {}

    for sb in donor.subbanks:
        sb_cfgs = [
            map_by_gidx.get(sl.global_index, SlotMappingConfig(sl.global_index))
            for sl in sb.samples
        ]
        any_modified = any(
            c.action != "keep_donor" or abs(c.gain_db) > 1e-4 for c in sb_cfgs
        )
        all_replaced = all(c.action != "keep_donor" for c in sb_cfgs)

        if sb.mode == FSB5_MODE_VORBIS:
            if all_replaced:
                subbank_build_lists[sb.index] = []
                if primary_rebuild_sb is None:
                    primary_rebuild_sb = sb.index
            else:
                raw_passthrough_subbanks[sb.index] = sb.raw_fsb_bytes
        else:
            if any_modified or output_codec_mode != sb.mode:
                subbank_build_lists[sb.index] = []
                if primary_rebuild_sb is None:
                    primary_rebuild_sb = sb.index
            else:
                raw_passthrough_subbanks[sb.index] = sb.raw_fsb_bytes

    # Si hay reemplazos parciales en Vorbis o inyección de ThrottleCut sin sub-banco primario activo, activarlo
    extra_sb_index: int | None = None
    has_vorbis_partial_repl = any(
        sb.index in raw_passthrough_subbanks
        and sb.mode == FSB5_MODE_VORBIS
        and any(
            map_by_gidx.get(sl.global_index, SlotMappingConfig(sl.global_index)).action != "keep_donor"
            for sl in sb.samples
        )
        for sb in donor.subbanks
    )
    if (has_vorbis_partial_repl or need_tcut_inject) and primary_rebuild_sb is None:
        if donor.subbanks and donor.subbanks[0].mode != FSB5_MODE_VORBIS:
            raw_passthrough_subbanks.pop(0, None)
            subbank_build_lists[0] = []
            primary_rebuild_sb = 0
        else:
            extra_sb_index = len(donor.subbanks)
            subbank_build_lists[extra_sb_index] = []
            primary_rebuild_sb = extra_sb_index

    # Poblar muestras en sub-bancos reconstruibles primero (índices nativos 0..N-1)
    for sb in donor.subbanks:
        if sb.index in subbank_build_lists:
            blist = subbank_build_lists[sb.index]
            for sl in sb.samples:
                cfg = map_by_gidx.get(sl.global_index, SlotMappingConfig(sl.global_index))
                if cfg.action == "keep_donor":
                    kept_count += 1
                else:
                    replaced_count += 1
                abuf, new_rpm = resolve_replacement_audio(sl, cfg)
                blist.append(FsbBuildSample(name=sl.name, audio=abuf))
                if sync_autopitch and new_rpm and new_rpm > 100.0 and sl.wav_guid:
                    autopitch_updates[sl.wav_guid] = float(new_rpm)

    # Procesar reemplazos parciales en sub-bancos passthrough (ej. Vorbis parcial)
    for sb in donor.subbanks:
        if sb.index in raw_passthrough_subbanks:
            for sl in sb.samples:
                cfg = map_by_gidx.get(sl.global_index, SlotMappingConfig(sl.global_index))
                if cfg.action == "keep_donor":
                    kept_count += 1
                    continue
                replaced_count += 1
                assert primary_rebuild_sb is not None
                target_list = subbank_build_lists[primary_rebuild_sb]
                new_smp_idx = len(target_list)
                abuf, new_rpm = resolve_replacement_audio(sl, cfg)
                target_list.append(FsbBuildSample(name=sl.name, audio=abuf))
                if sl.wav_guid:
                    wav_redirects[sl.wav_guid] = (primary_rebuild_sb, new_smp_idx)
                    if sync_autopitch and new_rpm and new_rpm > 100.0:
                        autopitch_updates[sl.wav_guid] = float(new_rpm)

    # Anexar muestras one-shot de ThrottleCut al sub-banco primario si el donante tenía ThrottleCut vacío
    tcut_int_loc: tuple[int, int] | None = None
    tcut_ext_loc: tuple[int, int] | None = None
    if need_tcut_inject and primary_rebuild_sb is not None:
        t_list = subbank_build_lists[primary_rebuild_sb]
        if tcut_int_resolved is not None and any(
            r in donor.empty_throttle_cut_events for r in ("EngineInt", "AuxInt")
        ):
            idx_int = len(t_list)
            t_list.append(FsbBuildSample(name=f"tcut_int_{tcut_int_resolved[1][:24]}", audio=tcut_int_resolved[0]))
            tcut_int_loc = (primary_rebuild_sb, idx_int)
        if tcut_ext_resolved is not None and "EngineExt" in donor.empty_throttle_cut_events:
            if (
                tcut_int_loc is not None
                and effective_tcut.int_source_index == effective_tcut.ext_source_index
                and effective_tcut.int_custom_wav == effective_tcut.ext_custom_wav
            ):
                tcut_ext_loc = tcut_int_loc
            else:
                idx_ext = len(t_list)
                t_list.append(FsbBuildSample(name=f"tcut_ext_{tcut_ext_resolved[1][:24]}", audio=tcut_ext_resolved[0]))
                tcut_ext_loc = (primary_rebuild_sb, idx_ext)

    total_subbanks = len(donor.subbanks) + (1 if extra_sb_index is not None else 0)
    final_fsb_blobs: list[bytes] = []
    for sb_i in range(total_subbanks):
        if sb_i in raw_passthrough_subbanks:
            final_fsb_blobs.append(raw_passthrough_subbanks[sb_i])
        else:
            dummy = donor.subbanks[sb_i].dummy8 if sb_i < len(donor.subbanks) else b"AMS2PORT"
            final_fsb_blobs.append(
                build_fsb5_container(
                    samples=subbank_build_lists[sb_i],
                    mode=output_codec_mode,
                    template_dummy8=dummy,
                )
            )

    # 2. Extraer FMT y PROJ originales del donante (e inyectar ThrottleCut antes del remapeo de GUIDs)
    fmt_payload = struct.pack("<II", donor.file_version or 0x50, donor.compat_version or 0x46)
    orig_proj_payload = b""
    for cid, ps, pe, _ in iter_riff_chunks(donor.raw, 12, len(donor.raw)):
        if cid == b"FMT ":
            fmt_payload = donor.raw[ps:pe]
        elif cid == b"LIST" and donor.raw[ps : ps + 4] == b"PROJ":
            orig_proj_payload = donor.raw[ps + 4 : pe]

    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_mod"
    tcut_injected_count = 0
    if need_tcut_inject and (tcut_int_loc is not None or tcut_ext_loc is not None):
        orig_proj_payload, tcut_injected_count = _inject_throttle_cut_into_proj(
            proj_payload=orig_proj_payload,
            donor=donor,
            tcut_int_loc=tcut_int_loc,
            tcut_ext_loc=tcut_ext_loc,
            duck_on_throttle_db=effective_tcut.duck_on_throttle_db,
            trigger_min=effective_tcut.trigger_min,
            seed_label=safe_mod,
        )

    # 3. Generar mapa de GUIDs standalone (preservando orden memcmp) + tabla STDT de rutas FMOD 1.08
    guid_map: dict[bytes, bytes] | None = None
    event_guid_map: dict[str, str] = {}
    role_to_event_guid: dict[str, str] = {}
    role_to_event_path = build_mod_event_paths(safe_mod, donor.events)
    stdt_entries: list[tuple[str, bytes]] = []

    if standalone_guids:
        local_sorted = collect_local_proj_guids(orig_proj_payload)
        guid_map = build_order_preserving_guid_map(local_sorted, safe_mod)
        new_bank_guid_bytes = guid_map.get(donor.bank_guid, donor.bank_guid)
        new_bank_guid = bytes_to_guid_str(new_bank_guid_bytes)
        stdt_entries.append((f"bank:/{safe_mod}_Engine", new_bank_guid_bytes))

        used_paths: set[str] = set()
        for ev in donor.events:
            ng_bytes = guid_map.get(ev.guid_bytes, ev.guid_bytes)
            ng = bytes_to_guid_str(ng_bytes)
            event_guid_map[ev.guid_str] = ng
            role_to_event_guid[ev.role] = ng
            ev_path = role_to_event_path.get(ev.role, f"event:/AMS2/Vehicles/Mods/{safe_mod}/{ev.role}")
            if ev_path.lower() in used_paths:
                ev_path = f"event:/AMS2/Vehicles/Mods/{safe_mod}/{ev.role}_{ev.index}"
            used_paths.add(ev_path.lower())
            stdt_entries.append((ev_path, ng_bytes))
    else:
        new_bank_guid = bytes_to_guid_str(donor.bank_guid)
        for ev in donor.events:
            event_guid_map[ev.guid_str] = ev.guid_str
            role_to_event_guid[ev.role] = ev.guid_str

    new_stdt_payload: bytes | None = None
    if stdt_entries:
        new_stdt_payload = build_stdt_chunk(stdt_entries)
        parsed_stdt = parse_stdt_sections(new_stdt_payload)
        for p_str, expected_g in stdt_entries:
            resolved_g = lookup_stdt_id(new_stdt_payload, p_str, _parsed=parsed_stdt)
            if resolved_g != expected_g:
                raise AssertionError(f"Verificación interna STDT falló para la ruta '{p_str}'")

    patched_proj_body = _patch_proj_chunks(
        proj_payload=orig_proj_payload,
        wav_redirects=wav_redirects,
        autopitch_updates=autopitch_updates,
        guid_map=guid_map,
        rpm_bands=rpm_bands,
    )

    # 4. Calcular offsets exactos de cada FSB5 alineado a 32 bytes y ensamblar SNDH + STDT + RIFF 'FEV '
    sndh_ver = (
        donor.sndh_version
        if len(final_fsb_blobs) == len(donor.subbanks)
        else (0x00080000 | (2 * len(final_fsb_blobs) + 1))
    )
    dummy_sndh = struct.pack("<I", sndh_ver) + b"\x00" * (8 * len(final_fsb_blobs))
    temp_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, dummy_sndh, new_stdt_payload)

    fmt_chunk_bytes = b"FMT " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if len(fmt_payload) & 1:
        fmt_chunk_bytes += b"\x00"

    proj_list_size = 4 + len(temp_proj_body)
    proj_total_bytes = 8 + proj_list_size + (proj_list_size & 1)

    cur_file_pos = 12 + len(fmt_chunk_bytes) + proj_total_bytes
    sndh_entries: list[tuple[int, int]] = []
    snd_chunks_built: list[bytes] = []

    for fsb_blob in final_fsb_blobs:
        snd_payload_start = cur_file_pos + 8
        pad_before = (32 - (snd_payload_start % 32)) % 32
        fsb_pos = snd_payload_start + pad_before
        fsb_len = len(fsb_blob)
        snd_payload = (b"\x00" * pad_before) + fsb_blob
        snd_chunk = b"SND " + struct.pack("<I", len(snd_payload)) + snd_payload
        if len(snd_payload) & 1:
            snd_chunk += b"\x00"
        sndh_entries.append((fsb_pos, fsb_len))
        snd_chunks_built.append(snd_chunk)
        cur_file_pos += len(snd_chunk)

    real_sndh = struct.pack("<I", sndh_ver) + b"".join(
        struct.pack("<II", fpos, flen) for fpos, flen in sndh_entries
    )
    final_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, real_sndh, new_stdt_payload)
    assert len(final_proj_body) == len(temp_proj_body), "Tamaño inesperado al actualizar SNDH/STDT"

    proj_chunk_bytes = (
        b"LIST"
        + struct.pack("<I", 4 + len(final_proj_body))
        + b"PROJ"
        + final_proj_body
        + (b"\x00" if (len(final_proj_body) & 1) else b"")
    )

    body = fmt_chunk_bytes + proj_chunk_bytes + b"".join(snd_chunks_built)
    riff_bytes = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body

    codec_label = "PCM16 (Sin pérdida)" if output_codec_mode == FSB5_MODE_PCM16 else "FADPCM (Comprimido FMOD)"
    return PortedBankResult(
        bank_bytes=riff_bytes,
        mod_identifier=safe_mod,
        standalone_guids=standalone_guids,
        new_bank_guid=new_bank_guid,
        event_guid_map=event_guid_map,
        role_to_event_guid=role_to_event_guid,
        role_to_event_path=role_to_event_path,
        replaced_slots_count=replaced_count,
        kept_slots_count=kept_count,
        codec_used=codec_label,
        throttle_cut_injected_events=tcut_injected_count,
    )


def _patch_prmb_and_ibsb_in_proj(
    proj_payload: bytes,
    renames: dict[str, str],
    ibg_to_target_bus: dict[bytes, bytes],
) -> bytes:
    """Renombra parámetros en chunks PRMB y re-enruta IBSB preservando integridad RIFF."""
    out = bytearray()
    for cid, ps, pe, _ in iter_riff_chunks(proj_payload, 0, len(proj_payload)):
        pay = proj_payload[ps:pe]
        if cid == b"LIST":
            ltype = pay[:4]
            inner = _patch_prmb_and_ibsb_in_proj(pay[4:], renames, ibg_to_target_bus)
            out += _riff_list(ltype, inner)
        elif cid == b"PRMB" and len(pay) >= 23:
            slen = struct.unpack_from("<H", pay, 21)[0]
            name = pay[23 : 23 + slen].decode("latin1", "replace").rstrip("\x00")
            if name in renames:
                new_n = renames[name].encode("latin1")
                new_pay = pay[:21] + struct.pack("<H", len(new_n)) + new_n + pay[23 + slen :]
                out += _riff_chunk(b"PRMB", new_pay)
            else:
                out += _riff_chunk(cid, pay)
        elif cid == b"IBSB" and len(pay) >= 34:
            new_pay = bytearray(pay)
            ig = bytes(pay[:16])
            if ig in ibg_to_target_bus and bytes(pay[18:34]) != b"\x00" * 16:
                new_pay[18:34] = ibg_to_target_bus[ig]
            out += _riff_chunk(b"IBSB", bytes(new_pay))
        else:
            out += _riff_chunk(cid, pay)
    return bytes(out)


def invert_event_cone_curves_in_proj(proj_payload: bytes) -> tuple[bytes, int]:
    """Invierte la orientación espacial de los conos de sonido FMOD (Event Cone Angle)

    exclusivamente en las curvas de volumen (Prop 0) de eventos exteriores, preservando
    intactos los filtros DSP (Prop >= 1000, high-pass, low-pass, EQs y panners 3D)
    y corrigiendo la posición e inversión de tangentes de Hermite para evitar
    mutear el vehículo.
    """
    buf = bytearray(proj_payload)

    # 1. Recolectar GUIDs de parámetros 'Event Cone Angle'
    cone_param_guids: set[bytes] = set()

    def walk_prms(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                walk_prms(ps + 4, pe)
            elif cid == b"PRMB" and pe - ps >= 31:
                slen = struct.unpack_from("<H", buf, ps + 21)[0]
                if ps + 23 + slen <= pe:
                    name = (
                        bytes(buf[ps + 23 : ps + 23 + slen])
                        .decode("latin1", "replace")
                        .rstrip("\x00")
                    )
                    if "cone angle" in name.lower() or "event cone" in name.lower():
                        cone_param_guids.add(bytes(buf[ps : ps + 16]))

    walk_prms(0, len(buf))
    if not cone_param_guids:
        return bytes(buf), 0

    # 2. Recolectar ÚNICAMENTE curvas CURV de volumen (prop == 0) gobernadas por 'Event Cone Angle'.
    # Se ignoran intencionadamente los filtros DSP (prop >= 1000) para evitar damping severo de frecuencias
    # y distorsión de panners 3D que mutearían el vehículo en pista.
    cone_vol_curve_guids: set[bytes] = set()

    def walk_ctrls(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                walk_ctrls(ps + 4, pe)
            elif cid == b"CTRL" and pe - ps >= 68:
                param_g = bytes(buf[ps + 32 : ps + 48])
                prop = struct.unpack_from("<I", buf, ps + 64)[0]
                if param_g in cone_param_guids and prop == 0:
                    curve_g = bytes(buf[ps + 48 : ps + 64])
                    cone_vol_curve_guids.add(curve_g)

    walk_ctrls(0, len(buf))
    if not cone_vol_curve_guids:
        return bytes(buf), 0

    # 3. Invertir las curvas de volumen en su lugar respetando el modelo de segmentos spline de FMOD Studio
    inverted_count = 0

    def walk_curvs(start: int, end: int) -> None:
        nonlocal inverted_count
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                walk_curvs(ps + 4, pe)
            elif cid == b"CURV" and pe - ps >= 36:
                cg = bytes(buf[ps : ps + 16])
                if cg in cone_vol_curve_guids:
                    cnt = struct.unpack_from("<H", buf, ps + 32)[0] >> 1
                    stride = struct.unpack_from("<H", buf, ps + 34)[0]
                    if cnt >= 2 and stride >= 16 and ps + 36 + cnt * stride <= pe:
                        orig_pts = []
                        for ki in range(cnt):
                            off = ps + 36 + ki * stride
                            x, y, t1, t2 = struct.unpack_from("<ffff", buf, off)
                            orig_pts.append((x, y, t1, t2))

                        # Invertir eje horizontal preservando la correspondencia exacta de cada segmento:
                        # El punto ki del nuevo arreglo corresponde a orig_idx = cnt - 1 - ki
                        # x' = 180.0 - x[orig_idx]
                        # y' = y[orig_idx]
                        # La curvatura t1 (forma del segmento que inicia en ki y termina en ki+1)
                        # corresponde al segmento invertido que terminaba en orig_idx,
                        # es decir, el que partía de orig_idx - 1 = cnt - 2 - ki con curvatura -t1.
                        # El último punto ki = cnt - 1 SIEMPRE debe tener t1 = 0.0 (fin de spline).
                        for ki in range(cnt):
                            orig_idx = cnt - 1 - ki
                            new_x = round(max(0.0, min(180.0, 180.0 - orig_pts[orig_idx][0])), 4)
                            new_y = orig_pts[orig_idx][1]
                            if ki < cnt - 1:
                                prev_orig_idx = cnt - 2 - ki
                                new_t1 = -orig_pts[prev_orig_idx][2]
                            else:
                                new_t1 = 0.0
                            new_t2 = 0.0
                            off = ps + 36 + ki * stride
                            struct.pack_into("<ffff", buf, off, new_x, new_y, new_t1, new_t2)
                        inverted_count += 1

    walk_curvs(0, len(buf))
    return bytes(buf), inverted_count


def _make_trigger_evnt(eguid: bytes, ibg: bytes, bus_guid: bytes) -> bytes:
    obj_guid = hashlib.sha256(eguid + b"_obj").digest()[:16]
    prop_guid = hashlib.sha256(eguid + b"_prop").digest()[:16]
    # EVTB de 129 bytes: bit 0x80 marca un disparo one-shot instantáneo sin bucle (termina en 0ms)
    evtb_pay = (
        eguid
        + (b"\x00" * 16)
        + obj_guid
        + ibg
        + bus_guid
        + bytes.fromhex("0100000080000000000100000003001000")
        + prop_guid
        + (b"\x00" * 8)
        + struct.pack("<I", 1)
        + (b"\x00" * 4)
    )
    c_evtb = _riff_chunk(b"EVTB", evtb_pay)
    c_prps = _riff_list(b"PRPS", b"")
    return _riff_list(b"EVNT", c_evtb + c_prps)


def _make_trigger_ibus(ibg: bytes, bus_guid: bytes) -> bytes:
    # IBSB de 42 bytes: bus de instancia para evento disparador conectado al mixer de AMS2
    ibsb_pay = ibg + struct.pack("<H", 0x0018) + bus_guid + (b"\x00" * 8)
    return _riff_list(b"IBS ", _riff_chunk(b"IBSB", ibsb_pay))


def extract_native_starters(
    native_bank_path: Path,
    starter_role_guids: list[bytes],
) -> tuple[list[tuple[bytes, bytes]], list[tuple[bytes, bytes]]]:
    """Extrae los chunks nativos EVNT e IBS de Starter y Starting desde el banco original de AMS2."""
    if not native_bank_path.is_file():
        return [], []
    raw = native_bank_path.read_bytes()
    nat_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(raw, 12, len(raw)):
        if cid == b"LIST" and raw[ps : ps + 4] == b"PROJ":
            nat_proj = raw[ps + 4 : pe]
            break
    if not nat_proj:
        return [], []

    target_g_set = set(starter_role_guids)
    extracted_evnts: list[tuple[bytes, bytes]] = []
    extracted_ibses: list[tuple[bytes, bytes]] = []
    extracted_ibg: set[bytes] = set()

    for cid, ps, pe, _ in iter_riff_chunks(nat_proj, 0, len(nat_proj)):
        if cid == b"LIST" and nat_proj[ps : ps + 4] == b"EVTS":
            lpay = nat_proj[ps + 4 : pe]
            for scid, sps, spe, spos in iter_riff_chunks(lpay, 0, len(lpay)):
                if scid == b"LCNT":
                    continue
                chunk_full = bytes(lpay[spos : spe + ((spe - sps) & 1)])
                for ccid, cps, cpe, _ in iter_riff_chunks(lpay, sps + 4, spe):
                    if ccid == b"EVTB" and cpe - cps >= 64:
                        eguid = bytes(lpay[cps : cps + 16])
                        if eguid in target_g_set:
                            ibg = bytes(lpay[cps + 48 : cps + 64])
                            extracted_ibg.add(ibg)
                            extracted_evnts.append((eguid, chunk_full))
            break

    for cid, ps, pe, _ in iter_riff_chunks(nat_proj, 0, len(nat_proj)):
        if cid == b"LIST" and nat_proj[ps : ps + 4] == b"IBSS":
            lpay = nat_proj[ps + 4 : pe]
            for scid, sps, spe, spos in iter_riff_chunks(lpay, 0, len(lpay)):
                if scid == b"LCNT":
                    continue
                chunk_full = bytes(lpay[spos : spe + ((spe - sps) & 1)])
                for ccid, cps, cpe, _ in iter_riff_chunks(lpay, sps + 4, spe):
                    if ccid == b"IBSB" and cpe - cps >= 16:
                        iguid = bytes(lpay[cps : cps + 16])
                        if iguid in extracted_ibg:
                            extracted_ibses.append((iguid, chunk_full))
            break

    return extracted_evnts, extracted_ibses


def inject_native_chunks_in_proj(
    proj_buf: bytes,
    evnt_items: list[tuple[bytes, bytes]],
    ibs_items: list[tuple[bytes, bytes]],
) -> bytes:
    """Inserta chunks EVNT e IBS nativos de AMS2 en las tablas ordenadas EVTS e IBSS."""
    buf = bytearray(proj_buf)
    if evnt_items:
        for cid, ps, pe, pos in iter_riff_chunks(buf, 0, len(buf)):
            if cid == b"LIST" and buf[ps : ps + 4] == b"EVTS":
                evts_pay = buf[ps + 4 : pe]
                items: list[tuple[bytes | None, bytes]] = []
                for scid, sps, spe, spos in iter_riff_chunks(evts_pay, 0, len(evts_pay)):
                    if scid == b"LCNT":
                        continue
                    sc_full = bytes(evts_pay[spos : spe + ((spe - sps) & 1)])
                    eg = None
                    for sscid, ssps, sspe, _ in iter_riff_chunks(evts_pay, sps + 4, spe):
                        if sscid == b"EVTB" and sspe - ssps >= 16:
                            eg = bytes(evts_pay[ssps : ssps + 16])
                            break
                    items.append((eg, sc_full))
                for eg, raw_item in evnt_items:
                    items.append((eg, raw_item))
                items.sort(key=lambda x: x[0] if x[0] else b"\xff" * 16)
                new_lcnt = _riff_chunk(b"LCNT", struct.pack("<I", len(items)))
                new_evts_body = b"EVTS" + new_lcnt + b"".join(c for _, c in items)
                pad_end = pe + ((pe - ps) & 1)
                buf = bytearray(buf[:pos] + _riff_chunk(b"LIST", new_evts_body) + buf[pad_end:])
                break

    if ibs_items:
        for cid, ps, pe, pos in iter_riff_chunks(buf, 0, len(buf)):
            if cid == b"LIST" and buf[ps : ps + 4] == b"IBSS":
                ibss_pay = buf[ps + 4 : pe]
                items = []
                for scid, sps, spe, spos in iter_riff_chunks(ibss_pay, 0, len(ibss_pay)):
                    if scid == b"LCNT":
                        continue
                    sc_full = bytes(ibss_pay[spos : spe + ((spe - sps) & 1)])
                    ig = None
                    for sscid, ssps, sspe, _ in iter_riff_chunks(ibss_pay, sps + 4, spe):
                        if sscid == b"IBSB" and sspe - ssps >= 16:
                            ig = bytes(ibss_pay[ssps : ssps + 16])
                            break
                    items.append((ig, sc_full))
                for ig, raw_item in ibs_items:
                    items.append((ig, raw_item))
                items.sort(key=lambda x: x[0] if x[0] else b"\xff" * 16)
                new_lcnt = _riff_chunk(b"LCNT", struct.pack("<I", len(items)))
                new_ibss_body = b"IBSS" + new_lcnt + b"".join(c for _, c in items)
                pad_end = pe + ((pe - ps) & 1)
                buf = bytearray(buf[:pos] + _riff_chunk(b"LIST", new_ibss_body) + buf[pad_end:])
                break

    return bytes(buf)


_TC_LC_PARAM_NAMES = frozenset({
    "tc", "traction", "tractioncontrol", "traction_control",
    "lc", "launch", "launchcontrol", "launch_control",
})
_MUTED_SEC_EVENT_ROLES = frozenset({"BackfireExt", "BackfireInt", "Limiter"})
_MUTED_SEC_PATH_KEYWORDS = ("tractioncontrol", "tc_ext", "tc_int", "limiter", "backfire")


def _compute_sparse_pop_locs(
    ac_bank: FmodBank,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float], set[tuple[int, int]]]:
    """Detecta muestras cortas sin loop en pistas de motor/escape con envolvente esparsa de petardeos."""
    sparse_locs: set[tuple[int, int]] = set()
    dip_ratios: dict[tuple[int, int], float] = {}
    non_sustained_locs: set[tuple[int, int]] = set()

    for sl in ac_bank.slots:
        if not sl.can_decode_audio or sl.loop is not None or not (0.02 < sl.duration_sec <= 1.85):
            continue
        if sl.event_roles and not any(
            r in ("EngineExt", "EngineInt", "BackfireExt", "BackfireInt", "Limiter")
            for r in sl.event_roles
        ):
            continue
        loc = (sl.subbank_index, sl.sample_index)
        non_sustained_locs.add(loc)
        try:
            buf = ac_bank.decode_slot_to_buffer(sl)
        except Exception:
            continue
        mono = buf.pcm.astype(np.float32) / 32768.0
        if mono.ndim == 2:
            mono = mono.mean(axis=1)
        win = max(1, int(buf.sample_rate * 0.010))
        nwin = len(mono) // win
        if nwin < 4:
            continue
        rms = np.sqrt(np.mean(mono[: nwin * win].reshape(nwin, win) ** 2, axis=1))
        pk = float(rms.max())
        dip = float((rms < 0.08 * pk).mean()) if pk > 1e-6 else 1.0
        dip_ratios[loc] = dip
        if dip >= 0.55:
            sparse_locs.add(loc)

    return sparse_locs, dip_ratios, non_sustained_locs


def _collect_proj_tc_and_pop_graph(
    proj_buf: bytes | bytearray,
    muted_ev_guids_in: set[bytes] | None = None,
    sparse_locs: set[tuple[int, int]] | None = None,
    loc_dip_ratios: dict[tuple[int, int], float] | None = None,
    non_sustained_locs: set[tuple[int, int]] | None = None,
) -> tuple[set[bytes], set[tuple[int, int]]]:
    """Resuelve todas las cajas/instrumentos (WAIT, MUIT, SPIT, EVIT) y samples de TC/LC y pops esparsos."""
    tc_param_guids: set[bytes] = set()
    muted_ev_guids: set[bytes] = set(muted_ev_guids_in or ())
    pmls_records: list[tuple[bytes, bytes, list[bytes]]] = []
    waib_to_wav: dict[bytes, bytes] = {}
    wav_to_waibs: dict[bytes, list[bytes]] = {}
    wav_to_loc: dict[bytes, tuple[int, int]] = {}
    multi_children: dict[bytes, list[bytes]] = {}
    evit_target: dict[bytes, bytes] = {}

    def scan(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(proj_buf, start, end):
            if cid == b"PRMB" and pe - ps >= 31:
                p = proj_buf[ps:pe]
                slen = struct.unpack_from("<H", p, 21)[0]
                if 23 + slen + 8 <= len(p):
                    pname = bytes(p[23 : 23 + slen]).decode("utf-8", "replace").strip().lower()
                    if pname in _TC_LC_PARAM_NAMES:
                        tc_param_guids.add(bytes(p[:16]))
            elif cid == b"PMLB" and pe - ps >= 52:
                nboxes = struct.unpack_from("<H", proj_buf, ps + 48)[0] >> 1
                stride = struct.unpack_from("<H", proj_buf, ps + 50)[0]
                if stride >= 24 and pe >= ps + 52 + nboxes * stride:
                    boxes = [
                        bytes(proj_buf[ps + 52 + bi * stride : ps + 52 + bi * stride + 16])
                        for bi in range(nboxes)
                    ]
                    pmls_records.append((bytes(proj_buf[ps + 16 : ps + 32]), bytes(proj_buf[ps + 32 : ps + 48]), boxes))
            elif cid == b"WAV " and pe - ps >= 30:
                _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", proj_buf, ps + 16)
                wav_to_loc[bytes(proj_buf[ps : ps + 16])] = (sb_i, smp_i)
            elif cid == b"LIST":
                ltype = bytes(proj_buf[ps : ps + 4])
                if ltype == b"WAIT":
                    wg_self = wg_target = None
                    for scid, sps, spe, _ in iter_riff_chunks(proj_buf, ps + 4, pe):
                        if scid == b"WAIB" and spe - sps >= 32:
                            wg_self = bytes(proj_buf[sps : sps + 16])
                            wg_target = bytes(proj_buf[sps + 16 : sps + 32])
                    if wg_self and wg_target:
                        waib_to_wav[wg_self] = wg_target
                        wav_to_waibs.setdefault(wg_target, []).append(wg_self)
                elif ltype in (b"MUIT", b"SPIT"):
                    mg_self = None
                    ch_guids: list[bytes] = []
                    for scid, sps, spe, _ in iter_riff_chunks(proj_buf, ps + 4, pe):
                        if scid in (b"MUIB", b"SPIB") and spe - sps >= 16:
                            mg_self = bytes(proj_buf[sps : sps + 16])
                        elif scid == b"PLST" and spe - sps >= 10:
                            nc = struct.unpack_from("<H", proj_buf, sps + 8)[0] >> 1
                            stride = struct.unpack_from("<H", proj_buf, sps + 10)[0]
                            if nc > 0 and stride >= 16:
                                for bi in range(nc):
                                    coff = sps + 12 + bi * stride
                                    if coff + 16 <= spe:
                                        ch_guids.append(bytes(proj_buf[coff : coff + 16]))
                    if mg_self:
                        multi_children[mg_self] = ch_guids
                elif ltype == b"EVIT":
                    eg_self = ev_tgt = None
                    for scid, sps, spe, _ in iter_riff_chunks(proj_buf, ps + 4, pe):
                        if scid == b"EVIB" and spe - sps >= 32:
                            eg_self = bytes(proj_buf[sps : sps + 16])
                            ev_tgt = bytes(proj_buf[sps + 16 : sps + 32])
                    if eg_self and ev_tgt:
                        evit_target[eg_self] = ev_tgt
                else:
                    scan(ps + 4, pe)

    scan(0, len(proj_buf))

    muted_inst_guids: set[bytes] = set()
    if sparse_locs:
        for wg, loc in wav_to_loc.items():
            if loc in sparse_locs:
                muted_inst_guids.update(wav_to_waibs.get(wg, ()))

    changed = True
    while changed:
        changed = False
        for param_g, ev_g, boxes in pmls_records:
            if param_g in tc_param_guids or ev_g in muted_ev_guids:
                for bg in boxes:
                    if bg not in muted_inst_guids:
                        muted_inst_guids.add(bg)
                        changed = True
        for ig in list(muted_inst_guids):
            if ig in evit_target and evit_target[ig] not in muted_ev_guids:
                muted_ev_guids.add(evit_target[ig])
                changed = True
            for cg in multi_children.get(ig, ()):
                if cg not in muted_inst_guids:
                    muted_inst_guids.add(cg)
                    changed = True

        # Promoción ascendente: si >=50% de un MUIT/SPIT son pops esparsos y ninguno es una capa sostenida
        for mg_self, children in multi_children.items():
            if not children or mg_self in muted_inst_guids:
                continue
            child_locs = [wav_to_loc.get(waib_to_wav.get(cg, b"")) for cg in children]
            if non_sustained_locs is not None and not all(
                loc in non_sustained_locs for loc in child_locs if loc is not None
            ):
                continue
            n_muted = sum(1 for cg in children if cg in muted_inst_guids)
            if n_muted >= max(1, (len(children) + 1) // 2):
                if loc_dip_ratios is not None:
                    dips = [loc_dip_ratios.get(loc, 0.0) if loc is not None else 0.0 for loc in child_locs]
                    if (sum(dips) / len(dips)) < 0.48:
                        continue
                elif non_sustained_locs is None and n_muted < len(children):
                    continue
                muted_inst_guids.add(mg_self)
                for cg in children:
                    muted_inst_guids.add(cg)
                changed = True

    muted_locs: set[tuple[int, int]] = set(sparse_locs or ())
    for ig in muted_inst_guids:
        wg = waib_to_wav.get(ig)
        if wg and wg in wav_to_loc:
            muted_locs.add(wav_to_loc[wg])
    return muted_inst_guids, muted_locs


def _muted_event_guids_for_bank(ac_bank: FmodBank) -> set[bytes]:
    out: set[bytes] = set()
    for ev in ac_bank.events:
        p_low = (ev.path or "").lower()
        if ev.role in _MUTED_SEC_EVENT_ROLES or any(k in p_low for k in _MUTED_SEC_PATH_KEYWORDS):
            out.add(ev.guid_bytes)
    return out


def _find_tc_and_lc_slots_in_bank(ac_bank: FmodBank) -> list[FsbSampleSlot]:
    """Identifica las ranuras (FsbSampleSlot) asociadas a TC/LC y petardeos esparsos de un banco AC."""
    proj = None
    for cid, ps, pe, _ in iter_riff_chunks(ac_bank.raw, 12, len(ac_bank.raw)):
        if cid == b"LIST" and ac_bank.raw[ps : ps + 4] == b"PROJ":
            proj = ac_bank.raw[ps + 4 : pe]
            break
    if not proj:
        return []

    sparse_locs, dip_ratios, non_sustained_locs = _compute_sparse_pop_locs(ac_bank)
    _, muted_locs = _collect_proj_tc_and_pop_graph(
        proj,
        muted_ev_guids_in=_muted_event_guids_for_bank(ac_bank),
        sparse_locs=sparse_locs,
        loc_dip_ratios=dip_ratios,
        non_sustained_locs=non_sustained_locs,
    )
    if not muted_locs:
        return []
    return [sl for sl in ac_bank.slots if (sl.subbank_index, sl.sample_index) in muted_locs]


def _silence_tc_and_lc_boxes_in_proj(
    proj_payload: bytes,
    muted_ev_guids_in: set[bytes] | None = None,
    sparse_locs: set[tuple[int, int]] | None = None,
    loc_dip_ratios: dict[tuple[int, int], float] | None = None,
    non_sustained_locs: set[tuple[int, int]] | None = None,
    out_muted_locs: set[tuple[int, int]] | None = None,
) -> tuple[bytes, int]:
    """Silencia cajas de TC/LC y capas esparsas de petardeo en pistas PMLB y ganancia INST (-96 dB)."""
    buf = bytearray(proj_payload)
    tc_inst_guids, muted_locs = _collect_proj_tc_and_pop_graph(
        buf,
        muted_ev_guids_in=muted_ev_guids_in,
        sparse_locs=sparse_locs,
        loc_dip_ratios=loc_dip_ratios,
        non_sustained_locs=non_sustained_locs,
    )
    if out_muted_locs is not None:
        out_muted_locs.update(muted_locs)
    if not tc_inst_guids:
        return bytes(buf), 0

    silenced_count = 0
    for cid, ps, pe, _ in iter_riff_chunks(buf, 0, len(buf)):
        if cid == b"LIST" and buf[ps : ps + 4] == b"PMLS":
            for scid, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                for ccid, cps, cpe, _ in iter_riff_chunks(buf, sps + 4, spe):
                    if ccid == b"PMLB" and cpe - cps >= 52:
                        nboxes = struct.unpack_from("<H", buf, cps + 48)[0] >> 1
                        stride = struct.unpack_from("<H", buf, cps + 50)[0]
                        if stride >= 24 and cpe >= cps + 52 + nboxes * stride:
                            for bi in range(nboxes):
                                boff = cps + 52 + bi * stride
                                if bytes(buf[boff : boff + 16]) in tc_inst_guids:
                                    struct.pack_into("<ff", buf, boff + 16, 0.0, 0.0)
                                    silenced_count += 1
            break

    def mute_inst_gain(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                ltype = buf[ps : ps + 4]
                if ltype in (b"WAIT", b"MUIT", b"SPIT", b"EVIT"):
                    self_g = None
                    inst_range = None
                    for scid, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                        if scid in (b"WAIB", b"MUIB", b"SPIB", b"EVIB") and spe - sps >= 16:
                            self_g = bytes(buf[sps : sps + 16])
                        elif scid == b"INST" and spe - sps >= 20:
                            inst_range = (sps, spe)
                    if self_g and self_g in tc_inst_guids and inst_range:
                        sps, _ = inst_range
                        struct.pack_into("<f", buf, sps + 16, -96.0)
                mute_inst_gain(ps + 4, pe)

    mute_inst_gain(0, len(buf))
    return bytes(buf), silenced_count


AMS2_AI_ENGINE_BUS = "{27ec50dd-048b-4f5d-88f7-481c376f3b1c}"  # bus:/.../PLAYER_EngExh_GROUP/3_AI_PLAYER_ENG
AI_EVENT_MAX_INSTANCES = 10  # Límite de instancias simultáneas para eventos de motor de IA porteados desde AC
AI_EVENT_STEAL_VIRTUALIZE = 0x40  # Modo de robo usado por los eventos de IA nativos de AMS2 (32/24 + 0x40)


def _select_primary_engine_events(ac_bank: FmodBank) -> tuple[FmodEventInfo | None, FmodEventInfo | None]:
    """Selecciona los eventos de motor principales (exterior, interior) de un banco de AC.

    Descarta 'engine_extra' y prioriza eventos con ruta real (no sub-eventos anidados sin nombre)
    con parámetro de RPM y el mayor número de muestras enlazadas.
    """

    def _is_engine_cand(ev: FmodEventInfo) -> bool:
        p_text = (ev.path or "").lower()
        if "engine_extra" in p_text:
            return False
        has_rpm = any(p[0].lower() in ("rpm", "rpms") for p in ev.parameters)
        return has_rpm and len(ev.linked_samples) >= 5

    def _pick(role: str, keys: tuple[str, ...]) -> FmodEventInfo | None:
        cands = [
            e for e in ac_bank.events
            if _is_engine_cand(e) and (e.role == role or any(k in (e.path or "").lower() for k in keys))
        ]
        cands.sort(key=lambda x: (0 if x.path else 1, -len(x.linked_samples)))
        if cands:
            return cands[0]
        named = [e for e in ac_bank.events if e.role == role]
        named.sort(key=lambda x: (0 if x.path else 1, -len(x.linked_samples)))
        return named[0] if named else None

    p_ext = _pick("EngineExt", ("_ext", "ext_", "exterior", "chase"))
    p_int = _pick("EngineInt", ("_int", "int_", "cockpit", "cabin"))
    return p_ext, p_int


def _iter_proj_table_items(buf: bytes):
    """Itera las tablas LIST de primer nivel de PROJ devolviendo (tipo, items).

    Cada item es (clave_guid, bytes_completos_con_padding, chunk_base, payload_base).
    """
    for cid, ps, pe, pos in iter_riff_chunks(buf, 0, len(buf)):
        if cid != b"LIST":
            continue
        lt = bytes(buf[ps : ps + 4])
        items: list[tuple[bytes, bytes, bytes, bytes]] = []
        for scid, sps, spe, spos in iter_riff_chunks(buf, ps + 4, pe):
            if scid == b"LCNT":
                continue
            full = bytes(buf[spos : spe + ((spe - sps) & 1)])
            if scid == b"LIST":
                for c2, a2, z2, _ in iter_riff_chunks(buf, sps + 4, spe):
                    if z2 - a2 >= 16:
                        items.append((bytes(buf[a2 : a2 + 16]), full, c2, bytes(buf[a2:z2])))
                    break
            elif spe - sps >= 16:
                items.append((bytes(buf[sps : sps + 16]), full, scid, bytes(buf[sps:spe])))
        yield lt, items


def clone_event_deep_in_proj(proj_buf: bytes, src_event_guid: bytes, dst_event_guid: bytes, salt: bytes) -> bytes:
    """Clona en profundidad un evento FMOD (timeline, pistas, instrumentos, curvas, parámetros locales, buses).

    Calcula el grafo de objetos propiedad del evento siguiendo referencias en ambos sentidos
    (los hijos referencian a su padre), excluyendo objetos compartidos (WAV, MAP, VCA, otros
    eventos anidados y parámetros de otros eventos). Cada objeto clonado recibe una GUID nueva
    y única, de modo que FMOD lo trata como un evento independiente con su propia máquina de estados.
    """
    buf = bytes(proj_buf)
    tables = list(_iter_proj_table_items(buf))

    index: dict[bytes, list[tuple[bytes, bytes, bytes]]] = {}
    for _lt, items in tables:
        for key, full, base, bpay in items:
            index.setdefault(key, []).append((full, base, bpay))
    if src_event_guid not in index:
        return buf

    refs: dict[bytes, set[bytes]] = {}
    for g, parts in index.items():
        s: set[bytes] = set()
        for full, _b, _p in parts:
            for i in range(0, len(full) - 15):
                w = full[i : i + 16]
                if w != g and w in index:
                    s.add(w)
        refs[g] = s
    rev: dict[bytes, set[bytes]] = {g: set() for g in index}
    for g, s in refs.items():
        for w in s:
            rev[w].add(g)

    shared_bases = {b"VCAB", b"BNKI", b"WAV ", b"MAP "}

    def _owned(w: bytes) -> bool:
        for _full, base, bpay in index[w]:
            if base in shared_bases:
                return False
            if base == b"EVTB" and w != src_event_guid:
                return False
            if base == b"PMLB" and bpay[32:48] != src_event_guid:
                return False
            if base == b"EVIB" and len(bpay) >= 32 and bpay[16:32] == src_event_guid:
                return False
        return True

    closure = {src_event_guid}
    frontier = [src_event_guid]
    while frontier:
        g = frontier.pop()
        for w in refs[g] | rev[g]:
            if w not in closure and _owned(w):
                closure.add(w)
                frontier.append(w)

    remap = {g: hashlib.sha256(g + salt + dst_event_guid).digest()[:16] for g in closure}
    remap[src_event_guid] = dst_event_guid

    def _remap_bytes(data: bytes) -> bytes:
        out = bytearray(data)
        i = 8  # saltar cabecera RIFF del item
        limit = len(out) - 16
        while i <= limit:
            rep = remap.get(bytes(out[i : i + 16]))
            if rep is not None:
                out[i : i + 16] = rep
                i += 16
            else:
                i += 1
        return bytes(out)

    out = bytearray()
    for cid, ps, pe, pos in iter_riff_chunks(buf, 0, len(buf)):
        pad_end = pe + ((pe - ps) & 1)
        if cid != b"LIST":
            out += buf[pos:pad_end]
            continue
        lt = buf[ps : ps + 4]
        items = next((it for t, it in tables if t == lt), None)
        cloned = [(remap[k], _remap_bytes(full)) for k, full, _b, _p in (items or []) if k in closure]
        if not cloned:
            out += buf[pos:pad_end]
            continue
        keys = [k for k, _f, _b, _p in items]
        combined = [(k, full) for k, full, _b, _p in items] + cloned
        if keys == sorted(keys):
            combined.sort(key=lambda x: x[0])
        body = lt + _riff_chunk(b"LCNT", struct.pack("<I", len(combined))) + b"".join(f for _, f in combined)
        out += _riff_chunk(b"LIST", body)
    return bytes(out)


def set_event_instance_limits_in_proj(
    proj_buf: bytes,
    event_guids: set[bytes] | None,
    max_instances: int,
    steal_flags: int = AI_EVENT_STEAL_VIRTUALIZE,
) -> bytes:
    """Fija el límite de instancias (EVTB +80) y el modo de robo (EVTB +84, bits 5-7).

    Si `event_guids` es None, limita únicamente los eventos sin límite acotado (<0 o > max_instances).
    """
    buf = bytearray(proj_buf)

    def walk(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                walk(ps + 4, pe)
            elif cid == b"EVTB" and pe - ps >= 88:
                ev_g = bytes(buf[ps : ps + 16])
                cur_max, flags = struct.unpack_from("<iI", buf, ps + 80)
                if event_guids is not None:
                    should_patch = ev_g in event_guids
                else:
                    should_patch = cur_max < 0 or cur_max > max_instances
                if should_patch:
                    struct.pack_into("<iI", buf, ps + 80, int(max_instances), (flags & ~0xE0) | (steal_flags & 0xE0))

    walk(0, len(buf))
    return bytes(buf)


def set_event_output_bus_in_proj(proj_buf: bytes, event_guid: bytes, bus_guid: bytes) -> bytes:
    """Re-enruta el bus de instancia (IBSB) del evento indicado hacia un bus del mixer de AMS2."""
    buf = bytearray(proj_buf)
    ibg: bytes | None = None

    def find_evtb(start: int, end: int) -> None:
        nonlocal ibg
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if ibg is not None:
                return
            if cid == b"LIST":
                find_evtb(ps + 4, pe)
            elif cid == b"EVTB" and pe - ps >= 64 and bytes(buf[ps : ps + 16]) == event_guid:
                ibg = bytes(buf[ps + 48 : ps + 64])

    def patch_ibsb(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                patch_ibsb(ps + 4, pe)
            elif cid == b"IBSB" and pe - ps >= 34 and bytes(buf[ps : ps + 16]) == ibg:
                buf[ps + 18 : ps + 34] = bus_guid

    find_evtb(0, len(buf))
    if ibg is not None:
        patch_ibsb(0, len(buf))
    return bytes(buf)


def _patch_throttle_params_in_proj(
    proj_payload: bytes,
    rename_to_load: bool,
    keep_throttle: bool = False,
) -> bytes:
    """Corrige el parámetro 'throttle' de AC, que AMS2 no alimenta en repeticiones ni en autos de la IA.

    - Su valor por defecto pasa a 1.0 (acelerador a fondo) para que, si el juego no lo alimenta,
      el motor no quede atrapado en las capas de 'pie levantado' (petardeos / crackle en bucle).
    - Por defecto (bancos estándar 'load') renombra 'throttle' -> 'load' cuando no hay 'load'.
    - Si `keep_throttle=True` (bancos nativos que operan con 'throttle', ej. A424, F_Ultimate_2022,
      F_V8_Gen1), conserva 'throttle' y, si el banco de AC sólo trae 'load', lo renombra a 'throttle'.
    """
    has_load = False
    has_throttle = False

    def scan(start: int, end: int) -> None:
        nonlocal has_load, has_throttle
        for cid, ps, pe, _ in iter_riff_chunks(proj_payload, start, end):
            if cid == b"LIST":
                scan(ps + 4, pe)
            elif cid == b"PRMB" and pe - ps >= 23:
                slen = struct.unpack_from("<H", proj_payload, ps + 21)[0]
                pname = proj_payload[ps + 23 : ps + 23 + slen].decode("latin1", "replace").rstrip("\x00")
                if pname == "load":
                    has_load = True
                elif pname == "throttle":
                    has_throttle = True

    scan(0, len(proj_payload))
    do_rename = rename_to_load and not keep_throttle and not has_load
    do_load_to_thr = keep_throttle and not has_throttle

    def rebuild(data: bytes) -> bytes:
        out = bytearray()
        for cid, ps, pe, _ in iter_riff_chunks(data, 0, len(data)):
            pay = data[ps:pe]
            if cid == b"LIST":
                out += _riff_list(pay[:4], rebuild(pay[4:]))
            elif cid == b"PRMB" and len(pay) >= 23:
                slen = struct.unpack_from("<H", pay, 21)[0]
                name = pay[23 : 23 + slen].decode("latin1", "replace").rstrip("\x00")
                tail = bytearray(pay[23 + slen :])
                if (name == "throttle" or (keep_throttle and name == "load")) and len(tail) >= 12:
                    pmin, pmax = struct.unpack_from("<ff", tail, 0)
                    struct.pack_into("<f", tail, 8, float(max(pmin, min(pmax, 1.0))))
                    if name == "throttle" and do_rename and pmin >= -0.0001 and pmax <= 2.0001:
                        new_n = b"load"
                        out += _riff_chunk(b"PRMB", pay[:21] + struct.pack("<H", len(new_n)) + new_n + bytes(tail))
                    elif name == "load" and do_load_to_thr and pmin >= -0.0001 and pmax <= 2.0001:
                        new_n = b"throttle"
                        out += _riff_chunk(b"PRMB", pay[:21] + struct.pack("<H", len(new_n)) + new_n + bytes(tail))
                    else:
                        out += _riff_chunk(b"PRMB", pay[: 23 + slen] + bytes(tail))
                else:
                    out += _riff_chunk(cid, pay)
            else:
                out += _riff_chunk(cid, pay)
        return bytes(out)

    return rebuild(proj_payload)


def _compute_active_direct_sample_locs(
    orig_proj_payload: bytes,
    ac_bank: FmodBank,
    root_event_guids: set[bytes],
) -> set[tuple[int, int]]:
    """Traza hacia adelante las muestras FSB5 alcanzables desde los eventos activos de motor/disparo."""
    if not root_event_guids:
        return {(sl.subbank_index, sl.sample_index) for sl in ac_bank.slots}
    tables = list(_iter_proj_table_items(orig_proj_payload))
    index: dict[bytes, list[tuple[bytes, bytes, bytes]]] = {}
    for _lt, items in tables:
        for k, full, base, bpay in items:
            index.setdefault(k, []).append((full, base, bpay))

    visited: set[bytes] = set(root_event_guids)
    queue: list[bytes] = list(root_event_guids)
    while queue:
        g = queue.pop()
        for full, base, bpay in index.get(g, ()):
            # EVIB enlaza explícitamente un sub-evento anidado en [16:32]
            if base == b"EVIB" and len(bpay) >= 32:
                tgt = bpay[16:32]
                if tgt in index and tgt not in visited:
                    visited.add(tgt)
                    queue.append(tgt)
            for i in range(0, len(full) - 15):
                w = full[i : i + 16]
                if w in index and w not in visited:
                    entries = index[w]
                    # Evitar saltar lateralmente por PRMB compartido hacia PMLBs de otros eventos
                    if all(eb == b"PMLB" and len(ep) >= 48 and ep[32:48] not in visited for _, eb, ep in entries):
                        continue
                    # Evitar saltar lateralmente a otro EVTB raíz si no proviene de EVIB[16:32]
                    if all(eb == b"EVTB" for _, eb, _ in entries):
                        continue
                    visited.add(w)
                    queue.append(w)

    active_wavs = {k for lt, items in tables if lt == b"WAVS" for k, _, _, _ in items if k in visited}
    return {(sl.subbank_index, sl.sample_index) for sl in ac_bank.slots if sl.wav_guid in active_wavs}


def _reencode_pcm16_subbank_to_fadpcm(
    ac_bank: FmodBank,
    subbank_index: int,
    active_locs: set[tuple[int, int]] | None,
    muted_locs: set[tuple[int, int]],
) -> bytes:
    """Re-codifica un sub-banco PCM16 a FADPCM nativo, silenciando pistas inactivas o muteadas."""
    slots_in_sb = sorted(
        (sl for sl in ac_bank.slots if sl.subbank_index == subbank_index),
        key=lambda s: s.sample_index,
    )
    fsb_samples: list[FsbBuildSample] = []
    for sl in slots_in_sb:
        loc = (subbank_index, sl.sample_index)
        is_muted = loc in muted_locs
        is_unused = active_locs is not None and loc not in active_locs
        if is_muted or is_unused:
            ch = max(1, min(2, int(sl.channels or 1)))
            sr = int(sl.sample_rate or 44100)
            buf = AudioBuffer(pcm=np.zeros((256, ch), dtype=np.int16), sample_rate=sr, loop=None)
            fsb_samples.append(FsbBuildSample(name=sl.name, audio=buf, peak_override=0.0))
        else:
            fsb_samples.append(FsbBuildSample(name=sl.name, audio=ac_bank.decode_slot_to_buffer(sl)))
    return build_fsb5_container(fsb_samples, mode=FSB5_MODE_FADPCM)


def _extract_native_engine_max_rpm(native_bank_path: Path | str) -> float | None:
    """Obtiene el tope máximo de RPM cubierto por las cajas PMLB de motor en un .bank nativo."""
    p = Path(native_bank_path)
    if not p.is_file():
        return None
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if len(raw) < 20 or raw[:4] != b"RIFF" or raw[8:12] != b"FEV ":
        return None

    rpm_prm_guids: set[bytes] = set()
    rpm_boxes_by_prm: list[tuple[bytes, float]] = []

    def _scan(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(raw, start, end):
            if cid == b"LIST":
                _scan(ps + 4, pe)
            elif cid == b"PRMB" and pe - ps >= 23:
                slen = struct.unpack_from("<H", raw, ps + 21)[0]
                if 23 + slen <= pe - ps:
                    pname = raw[ps + 23 : ps + 23 + slen].decode("latin1", "replace").rstrip("\x00").strip().lower()
                    if pname in ("rpm", "rpms"):
                        rpm_prm_guids.add(bytes(raw[ps : ps + 16]))
            elif cid == b"PMLB" and pe - ps >= 52:
                pg = bytes(raw[ps + 16 : ps + 32])
                nb = struct.unpack_from("<H", raw, ps + 48)[0] >> 1
                st = struct.unpack_from("<H", raw, ps + 50)[0]
                if st >= 24 and ps + 52 + nb * st <= pe:
                    local_max = 0.0
                    for bi in range(nb):
                        bo = ps + 52 + bi * st
                        v0, vlen = struct.unpack_from("<ff", raw, bo + 16)
                        if vlen > 10.0:
                            local_max = max(local_max, v0 + vlen)
                    if local_max > 2500.0:
                        rpm_boxes_by_prm.append((pg, local_max))

    for cid, ps, pe, _ in iter_riff_chunks(raw, 12, len(raw)):
        if cid == b"LIST" and raw[ps : ps + 4] == b"PROJ":
            _scan(ps + 4, pe)
            break

    max_rpm = 0.0
    for pg, lmax in rpm_boxes_by_prm:
        if pg in rpm_prm_guids:
            max_rpm = max(max_rpm, lmax)
    return max_rpm if max_rpm > 2500.0 else None


def _extend_top_rpm_boxes_in_proj(proj_payload: bytes, target_max_rpm: float) -> bytes:
    """Extiende hasta target_max_rpm las cajas RPM superiores si el banco AC termina antes que el auto AMS2."""
    buf = bytearray(proj_payload)
    rpm_prms: dict[bytes, tuple[float, float]] = {}
    ev_max_rpm: dict[bytes, float] = {}
    ctrls: list[tuple[bytes, bytes, bytes, int]] = []
    evit_prop_to_map: dict[tuple[bytes, int], bytes] = {}
    evit_prop_to_child_prm: dict[tuple[bytes, int], bytes] = {}

    def _scan(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                ltype = buf[ps : ps + 4]
                if ltype == b"EVIT":
                    evib_self: bytes | None = None
                    local_props: list[tuple[int, bytes]] = []
                    local_evib_prms: list[tuple[int, bytes]] = []
                    for scid, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                        if scid == b"EVIB" and spe - sps >= 40:
                            evib_self = bytes(buf[sps : sps + 16])
                            cnt = struct.unpack_from("<H", buf, sps + 36)[0] >> 1
                            st = struct.unpack_from("<H", buf, sps + 38)[0]
                            if st >= 20 and sps + 40 + cnt * st <= spe:
                                for ci in range(cnt):
                                    eo = sps + 40 + ci * st
                                    pid = struct.unpack_from("<I", buf, eo)[0]
                                    cpg = bytes(buf[eo + 4 : eo + 20])
                                    local_evib_prms.append((pid, cpg))
                        elif scid == b"LIST" and buf[sps : sps + 4] == b"PRPS":
                            for pcid, pps, ppe, _ in iter_riff_chunks(buf, sps + 4, spe):
                                if pcid == b"PROP" and ppe - pps >= 24:
                                    pid = struct.unpack_from("<I", buf, pps)[0]
                                    mg = bytes(buf[pps + 8 : pps + 24])
                                    local_props.append((pid, mg))
                    if evib_self is not None:
                        for pid, mg in local_props:
                            evit_prop_to_map[(evib_self, pid)] = mg
                        for pid, cpg in local_evib_prms:
                            evit_prop_to_child_prm[(evib_self, pid)] = cpg
                _scan(ps + 4, pe)
            elif cid == b"PRMB" and pe - ps >= 31:
                slen = struct.unpack_from("<H", buf, ps + 21)[0]
                if 23 + slen + 8 <= pe - ps:
                    pname = buf[ps + 23 : ps + 23 + slen].decode("latin1", "replace").rstrip("\x00").strip().lower()
                    if pname in ("rpm", "rpms"):
                        pmin, pmax = struct.unpack_from("<ff", buf, ps + 23 + slen)
                        rpm_prms[bytes(buf[ps : ps + 16])] = (float(pmin), float(pmax))
            elif cid == b"PMLB" and pe - ps >= 52:
                pg = bytes(buf[ps + 16 : ps + 32])
                eg = bytes(buf[ps + 32 : ps + 48])
                nb = struct.unpack_from("<H", buf, ps + 48)[0] >> 1
                st = struct.unpack_from("<H", buf, ps + 50)[0]
                if st >= 24 and ps + 52 + nb * st <= pe:
                    for bi in range(nb):
                        bo = ps + 52 + bi * st
                        v0, vlen = struct.unpack_from("<ff", buf, bo + 16)
                        if vlen > 10.0:
                            cur = ev_max_rpm.get(eg, 0.0)
                            if v0 + vlen > cur:
                                ev_max_rpm[eg] = v0 + vlen
            elif cid == b"CTRL" and pe - ps >= 68:
                ctrls.append((
                    bytes(buf[ps + 16 : ps + 32]),
                    bytes(buf[ps + 32 : ps + 48]),
                    bytes(buf[ps + 48 : ps + 64]),
                    struct.unpack_from("<I", buf, ps + 64)[0],
                ))

    _scan(0, len(buf))

    extend_evs = {
        eg: mx
        for eg, mx in ev_max_rpm.items()
        if 2500.0 <= mx < target_max_rpm - 100.0
    }
    if not extend_evs:
        return proj_payload

    bumped_prms: dict[bytes, float] = {
        pg: pmax for pg, (_pmin, pmax) in rpm_prms.items() if 2500.0 <= pmax < target_max_rpm
    }

    extended_box_ceiling: dict[bytes, float] = {}

    def _patch_pmlb_and_prmb(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                _patch_pmlb_and_prmb(ps + 4, pe)
            elif cid == b"PRMB" and pe - ps >= 31:
                pg = bytes(buf[ps : ps + 16])
                if pg in rpm_prms:
                    slen = struct.unpack_from("<H", buf, ps + 21)[0]
                    if 23 + slen + 8 <= pe - ps:
                        _pmin, pmax = struct.unpack_from("<ff", buf, ps + 23 + slen)
                        if 2500.0 <= pmax < target_max_rpm:
                            struct.pack_into("<f", buf, ps + 23 + slen + 4, float(target_max_rpm))
            elif cid == b"PMLB" and pe - ps >= 52:
                pg = bytes(buf[ps + 16 : ps + 32])
                eg = bytes(buf[ps + 32 : ps + 48])
                if pg in rpm_prms and eg in extend_evs:
                    ev_mx = extend_evs[eg]
                    nb = struct.unpack_from("<H", buf, ps + 48)[0] >> 1
                    st = struct.unpack_from("<H", buf, ps + 50)[0]
                    if st >= 24 and ps + 52 + nb * st <= pe:
                        for bi in range(nb):
                            bo = ps + 52 + bi * st
                            v0, vlen = struct.unpack_from("<ff", buf, bo + 16)
                            if vlen > 10.0 and v0 + vlen >= ev_mx - 50.0:
                                new_len = max(vlen, float(target_max_rpm) - v0)
                                struct.pack_into("<f", buf, bo + 20, new_len)
                                tgt_g = bytes(buf[bo : bo + 16])
                                extended_box_ceiling[tgt_g] = max(extended_box_ceiling.get(tgt_g, 0.0), ev_mx)

    _patch_pmlb_and_prmb(0, len(buf))

    # Indexar posiciones de chunks MAP en LIST:MPGS para sincronizar atómicamente CURV + MAP en EVIT
    map_spans: dict[bytes, tuple[int, int]] = {}

    def _scan_maps(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                _scan_maps(ps + 4, pe)
            elif cid == b"MAP " and pe - ps >= 28:
                map_spans[bytes(buf[ps : ps + 16])] = (ps, pe)

    _scan_maps(0, len(buf))

    curv_fadeout_ceil: dict[bytes, float] = {}
    curv_fwd_old_max: dict[bytes, float] = {}
    map_fwd_old_max: dict[bytes, float] = {}
    for tgt, prm, cgv, prop in ctrls:
        if prm not in rpm_prms:
            continue
        if prop == 4 and tgt in extended_box_ceiling:
            curv_fadeout_ceil[cgv] = extended_box_ceiling[tgt]
        elif prop > 20 and prm in bumped_prms:
            # Solo extender CURV de reenvío de parámetros a sub-evento EVIT si su MAP asociado en MPGS
            # existe y también se actualiza atómicamente; de lo contrario, MAP(CURV(rpm)) desfasaría el pitch!
            mg = evit_prop_to_map.get((tgt, prop))
            cpg = evit_prop_to_child_prm.get((tgt, prop))
            child_old_mx = bumped_prms.get(cpg, bumped_prms[prm]) if cpg is not None else bumped_prms[prm]
            if mg and mg in map_spans:
                mps, mpe = map_spans[mg]
                mcnt = struct.unpack_from("<H", buf, mps + 16)[0] >> 1
                mst = struct.unpack_from("<H", buf, mps + 18)[0]
                if mcnt >= 2 and mst >= 8 and mps + 20 + mcnt * mst <= mpe:
                    mx_end, my_end = struct.unpack_from("<ff", buf, mps + 20 + (mcnt - 1) * mst)
                    if mx_end >= 0.99 and abs(my_end - child_old_mx) <= 5.0:
                        curv_fwd_old_max[cgv] = bumped_prms[prm]
                        map_fwd_old_max[mg] = child_old_mx

    if curv_fadeout_ceil or curv_fwd_old_max or map_fwd_old_max:
        def _patch_curv_and_map(start: int, end: int) -> None:
            for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
                if cid == b"LIST":
                    _patch_curv_and_map(ps + 4, pe)
                elif cid == b"CURV" and pe - ps >= 36:
                    cg = bytes(buf[ps : ps + 16])
                    cnt = struct.unpack_from("<H", buf, ps + 32)[0] >> 1
                    st = struct.unpack_from("<H", buf, ps + 34)[0]
                    if cnt >= 2 and st >= 8 and ps + 36 + cnt * st <= pe:
                        if cg in curv_fadeout_ceil:
                            ev_mx = curv_fadeout_ceil[cg]
                            x_prev, y_prev = struct.unpack_from("<ff", buf, ps + 36 + (cnt - 2) * st)
                            x_end, y_end = struct.unpack_from("<ff", buf, ps + 36 + (cnt - 1) * st)
                            if x_end >= ev_mx - 50.0 and y_end <= 0.01 < y_prev:
                                struct.pack_into("<f", buf, ps + 36 + (cnt - 2) * st, max(x_prev, ev_mx))
                                struct.pack_into("<f", buf, ps + 36 + (cnt - 1) * st, float(target_max_rpm))
                        elif cg in curv_fwd_old_max:
                            old_mx = curv_fwd_old_max[cg]
                            x_end, y_end = struct.unpack_from("<ff", buf, ps + 36 + (cnt - 1) * st)
                            if abs(x_end - old_mx) <= 5.0 and y_end >= 0.99:
                                struct.pack_into("<f", buf, ps + 36 + (cnt - 1) * st, float(target_max_rpm))
                elif cid == b"MAP " and map_fwd_old_max and pe - ps >= 28:
                    mg = bytes(buf[ps : ps + 16])
                    if mg in map_fwd_old_max:
                        old_mx = map_fwd_old_max[mg]
                        mcnt = struct.unpack_from("<H", buf, ps + 16)[0] >> 1
                        mst = struct.unpack_from("<H", buf, ps + 18)[0]
                        if mcnt >= 2 and mst >= 8 and ps + 20 + mcnt * mst <= pe:
                            mx_end, my_end = struct.unpack_from("<ff", buf, ps + 20 + (mcnt - 1) * mst)
                            if mx_end >= 0.99 and abs(my_end - old_mx) <= 5.0:
                                struct.pack_into("<f", buf, ps + 20 + (mcnt - 1) * mst + 4, float(target_max_rpm))

        _patch_curv_and_map(0, len(buf))

    return bytes(buf)


# Parámetros 3D Spatializer nativos de AMS2 (FMOD_DSP_TYPE_PAN / 0x1b), auditados en los 172 bancos vanilla:
# Tupla: (rolloff, minDist, maxDist, extMode, sndSize, minExt)
# rolloff: 0=LinearSquared (TV Replay / cámaras de pista), 2=Inverse (Chase cam / rivales IA en pista)
AMS2_LIVE_REPLAY_PANNER: tuple[int, float, float, int, float, float] = (0, 1.0, 300.0, 0, 20.0, 120.0)
AMS2_AI_MAIN_PANNER: tuple[int, float, float, int, float, float] = (2, 7.0, 55.0, 0, 20.0, 120.0)
AMS2_PLAYER_EXT_PANNER: tuple[int, float, float, int, float, float] = (2, 7.0, 1000.0, 1, 20.0, 120.0)
AMS2_EXT_TARGET_RMS_DB: float = -9.5
AMS2_EXT_MAX_MBUS_GAIN_DB: float = 9.0
_EXT_RMS_SKIP_KW = (
    "pop", "backfire", "gear", "shift", "start", "limiter", "whine",
    "turbo", "wind", "skid", "null", "blowoff", "wastegate", "cut", "tc_", "lc_",
)


def _parse_pmef_param_offsets(data: bytes) -> dict[int, tuple[int, int, int | float | bool | bytes]]:
    """Parsea un payload PMEF de FMOD y devuelve {idx: (ptype, byte_offset_valor, valor)}."""
    if len(data) < 4:
        return {}
    cnt = struct.unpack_from("<I", data, 0)[0]
    off = 4
    out: dict[int, tuple[int, int, int | float | bool | bytes]] = {}
    for idx in range(cnt):
        if off + 4 > len(data):
            break
        ptype = struct.unpack_from("<I", data, off)[0]
        off += 4
        val_off = off
        if ptype == 0:
            if off + 4 > len(data):
                break
            val: int | float | bool | bytes = struct.unpack_from("<f", data, off)[0]
            off += 4
        elif ptype == 1:
            if off + 4 > len(data):
                break
            val = struct.unpack_from("<i", data, off)[0]
            off += 4
        elif ptype == 2:
            if off + 1 > len(data):
                break
            val = bool(data[off])
            off += 1
        elif ptype == 3:
            if off + 2 > len(data):
                break
            dlen = struct.unpack_from("<H", data, off)[0]
            off += 2
            if off + dlen > len(data):
                break
            val = bytes(data[off : off + dlen])
            off += dlen
        else:
            break
        out[idx] = (ptype, val_off, val)
    return out


def _extract_native_exterior_spatializers(
    native_bank_path: Path | str | None,
) -> dict[bytes, tuple[int, float, float, int, float, float]]:
    """Extrae los parámetros del Spatializer 3D (BEFF 0x1b, panBlend > 0.5) por evento desde un .bank nativo."""
    if not native_bank_path:
        return {}
    p = Path(native_bank_path)
    if not p.is_file():
        return {}
    try:
        raw = p.read_bytes()
    except OSError:
        return {}
    proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(raw, 12, len(raw)):
        if cid == b"LIST" and raw[ps : ps + 4] == b"PROJ":
            proj = raw[ps + 4 : pe]
            break
    if not proj:
        return {}
    tables = dict(_iter_proj_table_items(proj))
    mbss = {g: full for g, full, _, _ in tables.get(b"MBSS", ())}
    befx_3d: dict[bytes, tuple[int, float, float, int, float, float]] = {}
    for g, full, _, bpay in tables.get(b"BEFX", ()):
        if len(bpay) < 24 or struct.unpack_from("<I", bpay, 20)[0] != 27:
            continue
        pmef = next(
            (full[sps:spe] for sc, sps, spe, _ in iter_riff_chunks(full, 12, len(full)) if sc == b"PMEF"),
            b"",
        )
        pm = _parse_pmef_param_offsets(pmef)
        if 17 in pm and pm[17][0] == 0 and float(pm[17][2]) > 0.5:
            if all(k in pm for k in (11, 12, 13, 14, 15, 16)):
                befx_3d[g] = (
                    int(pm[11][2]),
                    float(pm[12][2]),
                    float(pm[13][2]),
                    int(pm[14][2]),
                    float(pm[15][2]),
                    float(pm[16][2]),
                )
    out: dict[bytes, tuple[int, float, float, int, float, float]] = {}
    for g, _full, _, ev_bpay in tables.get(b"EVTS", ()):
        mb_g = next(
            (ev_bpay[i : i + 16] for i in range(16, len(ev_bpay) - 15) if ev_bpay[i : i + 16] in mbss),
            None,
        )
        if not mb_g:
            continue
        mf = mbss[mb_g]
        for i in range(0, len(mf) - 15):
            w = mf[i : i + 16]
            if w in befx_3d:
                out[g] = befx_3d[w]
                break
    return out


def _estimate_ac_exterior_makeup_gain_db(
    ac_bank: FmodBank,
    orig_proj: bytes,
    p_ext_orig_guid: bytes | None,
) -> float:
    """Calcula la compensación limpia en punto flotante (dB) para igualar el nivel exterior de AC al estándar AMS2.

    Mide el RMS mediano de las muestras sostenidas del motor exterior más cualquier atenuación base
    presente en MBUS / FMOD Gain / curva Distance->MBUS, sin modificar ni saturar las muestras PCM16.
    """
    if not p_ext_orig_guid:
        return 0.0
    p_ext_ev = next((e for e in ac_bank.events if e.guid_bytes == p_ext_orig_guid), None)
    if p_ext_ev is None:
        return 0.0

    slot_by_name = {sl.name: sl for sl in ac_bank.slots}
    # Incluir muestras de p_ext y de sus sub-eventos EVIB anidados (ej. RSS Protech P96)
    active_ext_locs = _compute_active_direct_sample_locs(orig_proj, ac_bank, {p_ext_orig_guid})
    if active_ext_locs:
        all_sls = [sl for sl in ac_bank.slots if (sl.subbank_index, sl.sample_index) in active_ext_locs]
    else:
        all_sls = [slot_by_name[n] for n in p_ext_ev.linked_samples if isinstance(n, str) and n in slot_by_name]
    sustained_sls = [
        sl for sl in all_sls
        if sl.mode in (FSB5_MODE_PCM16, FSB5_MODE_FADPCM)
        and not any(kw in sl.name.lower() for kw in _EXT_RMS_SKIP_KW)
    ]
    if not sustained_sls:
        sustained_sls = [sl for sl in all_sls if sl.mode in (FSB5_MODE_PCM16, FSB5_MODE_FADPCM)]
    sustained_sls.sort(key=lambda s: -s.pcm_samples)

    rms_vals: list[float] = []
    for sl in sustained_sls[:24]:
        try:
            if sl.mode == FSB5_MODE_PCM16 and sl.data_bytes:
                nbytes = (len(sl.data_bytes) // 2) * 2
                if nbytes <= 0:
                    continue
                arr = np.frombuffer(sl.data_bytes[:nbytes], dtype="<i2").astype(np.float64) / 32768.0
            elif sl.mode == FSB5_MODE_FADPCM and sl.data_bytes:
                arr = decode_fadpcm(sl.data_bytes, sl.channels, sl.pcm_samples).astype(np.float64) / 32768.0
            else:
                continue
            if arr.size == 0:
                continue
            rms = float(np.sqrt(np.mean(arr * arr)))
            if rms > 1e-6:
                rms_vals.append(20.0 * math.log10(rms))
        except Exception:
            continue
    rms_db = float(np.percentile(rms_vals, 75)) if rms_vals else AMS2_EXT_TARGET_RMS_DB

    # Sumar atenuaciones base en MBUS (BUS static vol + FMOD Gain PEFF + curva Distance->MBUS a 25 m)
    tables = dict(_iter_proj_table_items(orig_proj))
    mbss = {g: full for g, full, _, _ in tables.get(b"MBSS", ())}
    ev_bpay = next((bp for g, _, _, bp in tables.get(b"EVTS", ()) if g == p_ext_orig_guid), b"")
    mb_g = next(
        (ev_bpay[i : i + 16] for i in range(16, len(ev_bpay) - 15) if ev_bpay[i : i + 16] in mbss),
        None,
    )
    mvol_db = 0.0
    gain_fx_db = 0.0
    dist_mbus_db = 0.0
    if mb_g:
        mf = mbss[mb_g]
        bus_b = next(
            (mf[sps:spe] for sc, sps, spe, _ in iter_riff_chunks(mf, 12, len(mf)) if sc == b"BUS "),
            b"",
        )
        if len(bus_b) >= 16 and bus_b[-8:-4] == b"\xff\xff\xff\x7f":
            mvol_db = float(struct.unpack_from("<f", bus_b, len(bus_b) - 16)[0])

        pefx = {g: (full, bpay) for g, full, _, bpay in tables.get(b"PEFX", ())}
        seen_pefx: set[bytes] = set()
        for i in range(0, len(mf) - 15):
            w = mf[i : i + 16]
            if w in pefx and w not in seen_pefx:
                seen_pefx.add(w)
                pfull, pbpay = pefx[w]
                if len(pbpay) >= 22:
                    nlen = struct.unpack_from("<H", pbpay, 20)[0]
                    if 22 + nlen <= len(pbpay):
                        pname = pbpay[22 : 22 + nlen].decode("ascii", errors="replace").lower()
                        if "gain" in pname:
                            pmef = next(
                                (pfull[sps:spe] for sc, sps, spe, _ in iter_riff_chunks(pfull, 12, len(pfull)) if sc == b"PMEF"),
                                b"",
                            )
                            pm = _parse_pmef_param_offsets(pmef)
                            if 0 in pm and pm[0][0] == 0:
                                gain_fx_db += float(pm[0][2])

        # Evaluar cualquier curva Distance -> MBUS (prop==0) a 25 m (distancia típica de cámara de pista)
        dist_prms: set[bytes] = set()
        for _, _, _, pb in tables.get(b"PRMS", ()):
            for idx_n in range(16, min(32, len(pb) - 4)):
                nl = struct.unpack_from("<H", pb, idx_n)[0]
                if 2 <= nl <= 32 and idx_n + 2 + nl + 12 <= len(pb):
                    cand = pb[idx_n + 2 : idx_n + 2 + nl]
                    if all(32 <= c < 127 for c in cand):
                        if cand.decode("ascii").lower() == "distance":
                            dist_prms.add(pb[:16])
                        break
        curvs: dict[bytes, list[tuple[float, float]]] = {}
        for cg, _, _, cb in tables.get(b"CRVS", ()):
            if len(cb) >= 36:
                cnt_r, st = struct.unpack_from("<HH", cb, 32)
                cnt_p = cnt_r >> 1
                if st >= 8:
                    curvs[cg] = [
                        struct.unpack_from("<ff", cb, 36 + pi * st)
                        for pi in range(cnt_p)
                        if 36 + pi * st + 8 <= len(cb)
                    ]
        for _, _, _, cb in tables.get(b"CTRS", ()):
            if len(cb) >= 68 and cb[16:32] == mb_g and cb[32:48] in dist_prms and struct.unpack_from("<I", cb, 64)[0] == 0:
                pts = curvs.get(cb[48:64], ())
                if pts:
                    x_eval = 25.0
                    if x_eval <= pts[0][0]:
                        val_d = pts[0][1]
                    elif x_eval >= pts[-1][0]:
                        val_d = pts[-1][1]
                    else:
                        val_d = pts[-1][1]
                        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                            if x0 <= x_eval <= x1:
                                val_d = y1 if abs(x1 - x0) < 1e-6 else (y0 + (x_eval - x0) / (x1 - x0) * (y1 - y0))
                                break
                    dist_mbus_db += float(val_d)

    effective_db = rms_db + mvol_db + gain_fx_db + dist_mbus_db
    desired_add = AMS2_EXT_TARGET_RMS_DB - effective_db
    # Tope estricto e idempotente sobre el fader MBUS final (<= AMS2_EXT_MAX_MBUS_GAIN_DB)
    max_headroom = max(0.0, AMS2_EXT_MAX_MBUS_GAIN_DB - mvol_db)
    return max(0.0, min(AMS2_EXT_MAX_MBUS_GAIN_DB, max_headroom, desired_add))


def _match_ams2_exterior_loudness_in_proj(
    proj_buf: bytes,
    ac_bank: FmodBank,
    orig_proj_payload: bytes,
    p_ext_orig_guid: bytes | None,
    ext_event_profiles: dict[bytes, tuple[tuple[int, float, float, int, float, float], float]],
) -> bytes:
    """Iguala el volumen y alcance exterior/replay de eventos AC al estándar vanilla de AMS2 sin saturar audio.

    Opera estrictamente in-place (sin cambiar tamaños ni orden de tablas):
    1) Alinea el 3D Spatializer (BEFF 0x1b, panBlend > 0.5) de cada evento exterior al perfil nativo AMS2
       (p.ej. LinearSquared 1..300m en AI_EngineMAIN_LIVE para TV Replays, Inverse 7..1000m en Player EngineExt).
    2) Suaviza el FMOD Distance Filter (PEFF) de AC (ausente en AMS2 vanilla) ampliando su distancia máxima.
    3) Aplica compensación de sonoridad en 32-bit float sobre el fader MBUS (BUS static vol, <= +10 dB)
       antes de los limitadores nativos del MasterBank de AMS2, evitando clipeo de muestras PCM16.
    """
    if not ext_event_profiles:
        return proj_buf

    makeup_db = _estimate_ac_exterior_makeup_gain_db(ac_bank, orig_proj_payload, p_ext_orig_guid)
    buf = bytearray(proj_buf)
    tables = dict(_iter_proj_table_items(proj_buf))
    mbss_items = {g: full for g, full, _, _ in tables.get(b"MBSS", ())}
    evts_bpay = {g: bpay for g, _, _, bpay in tables.get(b"EVTS", ())}

    # Mapear MBUS -> (panner_params, min_df_dist), BEFF -> panner_params, PEFF -> min_df_dist
    mbus_to_cfg: dict[bytes, tuple[tuple[int, float, float, int, float, float], float]] = {}
    befx_set = {g for g, _, _, bp in tables.get(b"BEFX", ()) if len(bp) >= 24 and struct.unpack_from("<I", bp, 20)[0] == 27}
    pefx_df_set: set[bytes] = set()
    for g, _, _, bp in tables.get(b"PEFX", ()):
        if len(bp) >= 22:
            nl = struct.unpack_from("<H", bp, 20)[0]
            if 22 + nl <= len(bp) and "distance filter" in bp[22 : 22 + nl].decode("ascii", errors="replace").lower():
                pefx_df_set.add(g)

    befx_to_pan: dict[bytes, tuple[int, float, float, int, float, float]] = {}
    pefx_to_min_d: dict[bytes, float] = {}
    for ev_g, (pan_cfg, min_df_d) in ext_event_profiles.items():
        eb = evts_bpay.get(ev_g)
        if not eb:
            continue
        mb_g = next(
            (eb[i : i + 16] for i in range(16, len(eb) - 15) if eb[i : i + 16] in mbss_items),
            None,
        )
        if not mb_g or mb_g in mbus_to_cfg:
            continue
        mbus_to_cfg[mb_g] = (pan_cfg, min_df_d)
        mf = mbss_items[mb_g]
        for i in range(0, len(mf) - 15):
            w = mf[i : i + 16]
            if w in befx_set and w not in befx_to_pan:
                befx_to_pan[w] = pan_cfg
            elif w in pefx_df_set and w not in pefx_to_min_d:
                pefx_to_min_d[w] = min_df_d

    if not mbus_to_cfg:
        return proj_buf

    # Parchear in-place MBSS (volumen float), BEFX (Spatializer 3D) y PEFX (Distance Filter)
    for cid, ps, pe, _ in iter_riff_chunks(buf, 0, len(buf)):
        if cid != b"LIST":
            continue
        lt = bytes(buf[ps : ps + 4])
        if lt == b"MBSS" and abs(makeup_db) > 0.05:
            for sc, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                if sc != b"LIST":
                    continue
                mg: bytes | None = None
                bus_span: tuple[int, int] | None = None
                for c2, a2, z2, _ in iter_riff_chunks(buf, sps + 4, spe):
                    if c2 == b"MBSB" and z2 - a2 >= 16:
                        mg = bytes(buf[a2 : a2 + 16])
                    elif c2 == b"BUS " and z2 - a2 >= 16:
                        bus_span = (a2, z2)
                if mg in mbus_to_cfg and bus_span is not None:
                    ba, bz = bus_span
                    if buf[bz - 8 : bz - 4] == b"\xff\xff\xff\x7f":
                        cur_v = struct.unpack_from("<f", buf, bz - 16)[0]
                        new_v = max(-80.0, min(10.0, cur_v + makeup_db))
                        struct.pack_into("<f", buf, bz - 16, float(new_v))
        elif lt == b"BEFX" and befx_to_pan:
            for sc, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                if sc != b"LIST":
                    continue
                bg: bytes | None = None
                pmef_span: tuple[int, int] | None = None
                for c2, a2, z2, _ in iter_riff_chunks(buf, sps + 4, spe):
                    if c2 == b"BEFB" and z2 - a2 >= 24:
                        bg = bytes(buf[a2 : a2 + 16])
                    elif c2 == b"PMEF" and z2 - a2 >= 16:
                        pmef_span = (a2, z2)
                if bg in befx_to_pan and pmef_span is not None:
                    pa, pz = pmef_span
                    pm = _parse_pmef_param_offsets(bytes(buf[pa:pz]))
                    # Solo modificar el Spatializer 3D real (panBlend > 0.5), nunca el panner 2D
                    if 17 in pm and pm[17][0] == 0 and float(pm[17][2]) > 0.5:
                        rolloff, min_d, max_d, ext_mode, snd_size, min_ext = befx_to_pan[bg]
                        if 11 in pm and pm[11][0] == 1:
                            struct.pack_into("<i", buf, pa + pm[11][1], int(rolloff))
                        if 12 in pm and pm[12][0] == 0:
                            struct.pack_into("<f", buf, pa + pm[12][1], float(min_d))
                        if 13 in pm and pm[13][0] == 0:
                            struct.pack_into("<f", buf, pa + pm[13][1], float(max_d))
                        if 14 in pm and pm[14][0] == 1:
                            struct.pack_into("<i", buf, pa + pm[14][1], int(ext_mode))
                        if 15 in pm and pm[15][0] == 0:
                            struct.pack_into("<f", buf, pa + pm[15][1], float(snd_size))
                        if 16 in pm and pm[16][0] == 0:
                            struct.pack_into("<f", buf, pa + pm[16][1], float(min_ext))
        elif lt == b"PEFX" and pefx_to_min_d:
            for sc, sps, spe, _ in iter_riff_chunks(buf, ps + 4, pe):
                if sc != b"LIST":
                    continue
                pg_fx: bytes | None = None
                pmef_span = None
                for c2, a2, z2, _ in iter_riff_chunks(buf, sps + 4, spe):
                    if c2 == b"PEFB" and z2 - a2 >= 22:
                        pg_fx = bytes(buf[a2 : a2 + 16])
                    elif c2 == b"PMEF" and z2 - a2 >= 12:
                        pmef_span = (a2, z2)
                if pg_fx in pefx_to_min_d and pmef_span is not None:
                    pa, pz = pmef_span
                    pm = _parse_pmef_param_offsets(bytes(buf[pa:pz]))
                    if 0 in pm and pm[0][0] == 0:
                        cur_df_d = float(pm[0][2])
                        min_df_d = pefx_to_min_d[pg_fx]
                        if cur_df_d < min_df_d:
                            struct.pack_into("<f", buf, pa + pm[0][1], float(min_df_d))

    return bytes(buf)


def _collect_mapping_patches_for_direct_bank(
    bank: FmodBank,
    mappings: list[SlotMappingConfig] | None,
    rpm_bands: list[RpmBandConfig] | None = None,
    library: SourceAudioLibrary | None = None,
) -> tuple[
    dict[bytes, tuple[int, int]],
    dict[bytes, float],
    dict[bytes, float],
    set[bytes],
    set[bytes],
    dict[int, Path],
]:
    """Traduce mappings + rpm_bands en diccionarios de parcheo para _patch_proj_chunks y reemplazos WAV."""
    wav_redirects: dict[bytes, tuple[int, int]] = {}
    autopitch_updates: dict[bytes, float] = {}
    gain_db_deltas: dict[bytes, float] = {}
    silence_wav_guids: set[bytes] = set()
    unmute_wav_guids: set[bytes] = set()
    custom_wav_replacements: dict[int, Path] = {}

    if rpm_bands:
        band_muted_slots: set[int] = set()
        band_active_slots: set[int] = set()
        for b in rpm_bands:
            for gidx in b.donor_slot_indices:
                if b.is_muted_bus:
                    band_muted_slots.add(gidx)
                else:
                    band_active_slots.add(gidx)
        for sl in bank.slots:
            if sl.wav_guid is None:
                continue
            if sl.global_index in band_active_slots:
                unmute_wav_guids.add(sl.wav_guid)
            elif sl.global_index in band_muted_slots:
                silence_wav_guids.add(sl.wav_guid)

    if mappings:
        map_by_gidx = {c.donor_global_index: c for c in mappings}
        for sl in bank.slots:
            if sl.wav_guid is None:
                continue
            cfg = map_by_gidx.get(sl.global_index)
            if cfg is None:
                continue
            if cfg.override_root_rpm is not None and cfg.override_root_rpm > 100.0:
                if sl.autopitch_ref_rpm is None or abs(cfg.override_root_rpm - sl.autopitch_ref_rpm) > 0.1:
                    autopitch_updates[sl.wav_guid] = float(cfg.override_root_rpm)
            if cfg.action == "silence":
                silence_wav_guids.add(sl.wav_guid)
                unmute_wav_guids.discard(sl.wav_guid)
            else:
                if abs(float(cfg.gain_db or 0.0)) > 1e-4:
                    gain_db_deltas[sl.wav_guid] = float(cfg.gain_db)
                if cfg.action == "replace_source" and cfg.source_index is not None and library is not None:
                    if 0 <= cfg.source_index < len(library.items):
                        src_it = library.items[cfg.source_index]
                        if src_it._slot_ref is not None:
                            new_loc = (src_it._slot_ref.subbank_index, src_it._slot_ref.sample_index)
                            if new_loc != (sl.subbank_index, sl.sample_index):
                                wav_redirects[sl.wav_guid] = new_loc
                        elif src_it._wav_path is not None and Path(src_it._wav_path).is_file():
                            custom_wav_replacements[sl.global_index] = Path(src_it._wav_path)
                elif cfg.action == "replace_wav" and cfg.custom_wav_path is not None:
                    wp = Path(cfg.custom_wav_path)
                    if wp.is_file():
                        custom_wav_replacements[sl.global_index] = wp

    return (
        wav_redirects,
        autopitch_updates,
        gain_db_deltas,
        silence_wav_guids,
        unmute_wav_guids,
        custom_wav_replacements,
    )


def _apply_custom_wav_replacements_to_subbanks(
    bank: FmodBank,
    final_fsb_blobs: list[bytearray],
    custom_wav_replacements: dict[int, Path],
) -> None:
    """Re-codifica únicamente los sub-bancos afectados por reemplazos WAV externos preservando su códec."""
    if not custom_wav_replacements:
        return
    affected_sbs: set[int] = set()
    for sl in bank.slots:
        if sl.global_index in custom_wav_replacements:
            affected_sbs.add(sl.subbank_index)

    for sb_idx in sorted(affected_sbs):
        if not (0 <= sb_idx < len(final_fsb_blobs)):
            continue
        blob = final_fsb_blobs[sb_idx]
        sb_mode = struct.unpack_from("<I", blob, 24)[0] if len(blob) >= 28 else FSB5_MODE_PCM16
        if sb_mode not in (FSB5_MODE_PCM16, FSB5_MODE_FADPCM):
            sb_mode = FSB5_MODE_FADPCM
        slots_in_sb = sorted(
            (sl for sl in bank.slots if sl.subbank_index == sb_idx),
            key=lambda s: s.sample_index,
        )
        fsb_samples: list[FsbBuildSample] = []
        for sl in slots_in_sb:
            wav_p = custom_wav_replacements.get(sl.global_index)
            if wav_p is not None:
                raw_wav = read_wav_file(wav_p)
                buf = conform_audio(
                    raw_wav,
                    target_sr=int(sl.sample_rate or 48000),
                    target_channels=max(1, min(2, int(sl.channels or 1))),
                    gain_db=0.0,
                    need_loop=bool(sl.loop_start is not None or sl.loop_end is not None),
                )
                fsb_samples.append(FsbBuildSample(name=sl.name, audio=buf))
            else:
                fsb_samples.append(FsbBuildSample(name=sl.name, audio=bank.decode_slot_to_buffer(sl)))
        final_fsb_blobs[sb_idx] = bytearray(build_fsb5_container(fsb_samples, mode=sb_mode))


def uninvert_spatial_cone_in_bank_bytes(bank_bytes: bytes) -> bytes:
    """Aplica invert_event_cone_curves_in_proj (involución exacta) sobre PROJ de un .bank sin alterar tamaños ni SNDH."""
    if len(bank_bytes) < 20 or bank_bytes[:4] != b"RIFF" or bank_bytes[8:12] != b"FEV ":
        return bank_bytes
    out = bytearray(bank_bytes)
    for cid, ps, pe, _ in iter_riff_chunks(out, 12, len(out)):
        if cid == b"LIST" and out[ps : ps + 4] == b"PROJ":
            proj_inner = bytes(out[ps + 4 : pe])
            patched, cnt = invert_event_cone_curves_in_proj(proj_inner)
            if cnt > 0 and len(patched) == len(proj_inner):
                out[ps + 4 : pe] = patched
            break
    return bytes(out)


def apply_band_edits_to_bank(
    bank: FmodBank,
    rpm_bands: list[RpmBandConfig] | None = None,
    mappings: list[SlotMappingConfig] | None = None,
    library: SourceAudioLibrary | None = None,
) -> bytes:
    """Aplica ediciones de bandas RPM, pitch, ganancia, silencios y muestras en un .bank ya modificado.

    Preserva al 100% los GUIDs de banco/eventos, tabla STDT, buses IBSB, variantes extra,
    orientación espacial del cono y sub-bancos FSB5 no modificados.
    """
    fmt_payload = struct.pack("<II", bank.file_version or 0x50, bank.compat_version or 0x46)
    orig_proj_payload = b""
    for cid, ps, pe, _ in iter_riff_chunks(bank.raw, 12, len(bank.raw)):
        if cid == b"FMT ":
            fmt_payload = bank.raw[ps:pe]
        elif cid == b"LIST" and bank.raw[ps : ps + 4] == b"PROJ":
            orig_proj_payload = bank.raw[ps + 4 : pe]

    (
        wav_redirects,
        autopitch_updates,
        gain_db_deltas,
        silence_wav_guids,
        unmute_wav_guids,
        custom_wav_replacements,
    ) = _collect_mapping_patches_for_direct_bank(bank, mappings, rpm_bands, library)

    patched_proj_body = _patch_proj_chunks(
        proj_payload=orig_proj_payload,
        wav_redirects=wav_redirects,
        autopitch_updates=autopitch_updates,
        guid_map=None,
        rpm_bands=rpm_bands,
        gain_db_deltas=gain_db_deltas,
        silence_wav_guids=silence_wav_guids,
        unmute_wav_guids=unmute_wav_guids,
    )

    final_fsb_blobs = [bytearray(sb.raw_fsb_bytes) for sb in bank.subbanks]
    _apply_custom_wav_replacements_to_subbanks(bank, final_fsb_blobs, custom_wav_replacements)

    sndh_ver = bank.sndh_version or (0x00080000 | (2 * len(final_fsb_blobs) + 1))
    dummy_sndh = struct.pack("<I", sndh_ver) + b"\x00" * (8 * len(final_fsb_blobs))
    temp_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, dummy_sndh, None)

    fmt_chunk_bytes = b"FMT " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if len(fmt_payload) & 1:
        fmt_chunk_bytes += b"\x00"

    proj_list_size = 4 + len(temp_proj_body)
    proj_total_bytes = 8 + proj_list_size + (proj_list_size & 1)

    cur_file_pos = 12 + len(fmt_chunk_bytes) + proj_total_bytes
    sndh_entries: list[tuple[int, int]] = []
    snd_chunks_built: list[bytes] = []

    for fsb_blob in final_fsb_blobs:
        snd_payload_start = cur_file_pos + 8
        pad_before = (32 - (snd_payload_start % 32)) % 32
        fsb_pos = snd_payload_start + pad_before
        fsb_len = len(fsb_blob)
        snd_payload = (b"\x00" * pad_before) + bytes(fsb_blob)
        snd_chunk = b"SND " + struct.pack("<I", len(snd_payload)) + snd_payload
        if len(snd_payload) & 1:
            snd_chunk += b"\x00"
        sndh_entries.append((fsb_pos, fsb_len))
        snd_chunks_built.append(snd_chunk)
        cur_file_pos += len(snd_chunk)

    real_sndh = struct.pack("<I", sndh_ver) + b"".join(
        struct.pack("<II", fpos, flen) for fpos, flen in sndh_entries
    )
    final_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, real_sndh, None)
    assert len(final_proj_body) == len(temp_proj_body), "Tamaño inesperado al actualizar SNDH"

    proj_chunk_bytes = (
        b"LIST"
        + struct.pack("<I", 4 + len(final_proj_body))
        + b"PROJ"
        + final_proj_body
        + (b"\x00" if (len(final_proj_body) & 1) else b"")
    )

    body = fmt_chunk_bytes + proj_chunk_bytes + b"".join(snd_chunks_built)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body



def _extract_event_subgraph_closure(
    tables: list[tuple[bytes, list[tuple[bytes, bytes, bytes, bytes]]]],
    root_event_guids: set[bytes],
) -> set[bytes]:
    """Calcula el subgrafo completo de objetos PROJ alcanzables desde un conjunto de eventos raíz.

    Sigue referencias hacia adelante (EVTB -> IBSB/MBSB/TLNB/PRMB, EVIB -> sub-evento EVTB,
    WAIB -> WAV/PRMB/GBSB/CURV/MAP, etc.) y únicamente aristas inversas de propiedad real
    (PMLB[32:48], TLNB[16:32], CTRL[16:32], MODS[16:32], GBSB/RBSB[18:34], efectos sobre buses),
    evitando contaminar el subgrafo con otros eventos raíz que compartan parámetros o muestras.
    """
    index: dict[bytes, list[tuple[bytes, bytes, bytes, bytes]]] = {}
    for lt, items in tables:
        for k, full, base, bpay in items:
            index.setdefault(k, []).append((lt, full, base, bpay))

    fwd: dict[bytes, set[bytes]] = {}
    for k, parts in index.items():
        s: set[bytes] = set()
        for _lt, full, _base, _bpay in parts:
            for i in range(0, len(full) - 15):
                w = full[i : i + 16]
                if w != k and w in index:
                    s.add(w)
        fwd[k] = s

    bus_bases = {b"IBSB", b"GBSB", b"MBSB", b"RBSB"}
    fx_bases = {b"BEFB", b"PEFB", b"SEFB", b"SCFB", b"SCEF"}
    rev_owned: dict[bytes, set[bytes]] = {}
    for lt, items in tables:
        for w, full, base, bpay in items:
            if base == b"PMLB" and len(bpay) >= 48:
                rev_owned.setdefault(bpay[32:48], set()).add(w)
            elif base == b"TLNB" and len(bpay) >= 32:
                rev_owned.setdefault(bpay[16:32], set()).add(w)
            elif base == b"CTRL" and len(bpay) >= 32:
                rev_owned.setdefault(bpay[16:32], set()).add(w)
            elif lt == b"MODS" and len(bpay) >= 32:
                rev_owned.setdefault(bpay[16:32], set()).add(w)
            elif base in (b"GBSB", b"RBSB") and len(bpay) >= 34:
                rev_owned.setdefault(bpay[18:34], set()).add(w)
            elif base in fx_bases:
                for ref_g in fwd.get(w, ()):
                    if any(eb in bus_bases for _, _, eb, _ in index.get(ref_g, ())):
                        rev_owned.setdefault(ref_g, set()).add(w)

    closure: set[bytes] = {g for g in root_event_guids if g in index}
    queue: list[bytes] = list(closure)
    while queue:
        g = queue.pop()
        for w in fwd.get(g, ()):
            if w not in closure:
                closure.add(w)
                queue.append(w)
        for w in rev_owned.get(g, ()):
            if w not in closure:
                closure.add(w)
                queue.append(w)

    return closure


def _remap_chunk_guids_leaves(chunk_bytes: bytes, remap: dict[bytes, bytes]) -> bytes:
    """Remapea GUIDs dentro de los chunks hoja de un item RIFF sin alterar cabeceras RIFF ni LCNT."""
    if not remap:
        return chunk_bytes
    buf = bytearray(chunk_bytes)

    def _walk(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"LIST":
                _walk(ps + 4, pe)
            elif cid not in (b"LCNT", b"SNDH", b"STDT", b"STBL") and pe - ps >= 16:
                i = ps
                limit = pe - 16
                while i <= limit:
                    rep = remap.get(bytes(buf[i : i + 16]))
                    if rep is not None:
                        buf[i : i + 16] = rep
                        i += 16
                    else:
                        i += 1

    _walk(0, len(buf))
    return bytes(buf)


def _patch_wav_chunk_subbank(wav_chunk_bytes: bytes, new_sb: int) -> bytes:
    """Actualiza el índice de sub-banco (u16 en WAV[18:20]) preservando flags, sample_index y load_mode."""
    return _patch_wav_chunk_location(wav_chunk_bytes, new_sb, None)


def _patch_wav_chunk_location(
    wav_chunk_bytes: bytes,
    new_sb: int,
    new_smp: int | None = None,
) -> bytes:
    """Actualiza subbank_index (y opcionalmente sample_index) en un chunk WAV preservando flags y load_mode."""
    if len(wav_chunk_bytes) < 38 or wav_chunk_bytes[:4] != b"WAV ":
        return wav_chunk_bytes
    out = bytearray(wav_chunk_bytes)
    f0, _old_sb, flags, old_smp, load_mode = struct.unpack_from("<HHHII", out, 24)
    smp_idx = old_smp if new_smp is None else int(new_smp)
    struct.pack_into("<HHHII", out, 24, f0, int(new_sb), flags, smp_idx, load_mode)
    return bytes(out)


def collect_vanilla_scope_targets(
    vehicle: object,
    ac_bank: FmodBank | None,
    audio_scope: str,
    is_ai: bool = False,
) -> tuple[set[bytes], set[bytes], dict[bytes, str], dict[str, list[tuple]]]:
    """Determina los eventos a reemplazar/silenciar y el mapa de rutas para un alcance parcial ('interior'/'exterior').

    Retorna: (replace_event_guids, silence_native_event_guids, event_paths_map, augmented_extra_variants)
    """
    event_guids: dict[str, str] = dict(getattr(vehicle, "ai_event_guids" if is_ai else "event_guids", {}) or {})
    event_paths: dict[str, str] = dict(getattr(vehicle, "ai_event_paths" if is_ai else "event_paths", {}) or {})
    raw_extras: dict[str, list[tuple]] = {
        k: list(v)
        for k, v in (getattr(vehicle, "extra_ai_event_variants" if is_ai else "extra_event_variants", {}) or {}).items()
    }

    paths_by_bytes: dict[bytes, str] = {}
    for role, g_str in event_guids.items():
        if not g_str:
            continue
        gb = guid_str_to_bytes(g_str)
        if role in event_paths and event_paths[role]:
            paths_by_bytes[gb] = event_paths[role]
    for _role, items in raw_extras.items():
        for it in items:
            if it and it[0]:
                gb = guid_str_to_bytes(it[0])
                if len(it) >= 2 and it[1]:
                    paths_by_bytes.setdefault(gb, it[1])

    replace_guids: set[bytes] = set()
    silence_guids: set[bytes] = set()

    if is_ai:
        if audio_scope == "exterior":
            for role, g_str in event_guids.items():
                if not g_str:
                    continue
                if role.startswith(("ai_engine_main", "ai_engine_live", "ai_engine_ext")):
                    replace_guids.add(guid_str_to_bytes(g_str))
            for role in ("ai_engine_main", "ai_engine_live", "ai_engine_ext"):
                for it in raw_extras.get(role, ()):
                    if it and it[0]:
                        replace_guids.add(guid_str_to_bytes(it[0]))
        return replace_guids, silence_guids, paths_by_bytes, raw_extras

    # Player bank
    if audio_scope == "interior":
        if "EngineInt" in event_guids:
            replace_guids.add(guid_str_to_bytes(event_guids["EngineInt"]))
        if "AuxInt" in event_guids:
            aux_str = event_guids["AuxInt"]
            aux_b = guid_str_to_bytes(aux_str)
            replace_guids.add(aux_b)
            existing_int_ex = {it[0].lower() for it in raw_extras.get("EngineInt", ()) if it and it[0]}
            if aux_str.lower() != event_guids.get("EngineInt", "").lower() and aux_str.lower() not in existing_int_ex:
                raw_extras.setdefault("EngineInt", []).append((aux_str, event_paths.get("AuxInt", "")))
        for it in raw_extras.get("EngineInt", ()):
            if it and it[0]:
                replace_guids.add(guid_str_to_bytes(it[0]))
    elif audio_scope == "exterior":
        if "EngineExt" in event_guids:
            replace_guids.add(guid_str_to_bytes(event_guids["EngineExt"]))
        for it in raw_extras.get("EngineExt", ()):
            if it and it[0]:
                replace_guids.add(guid_str_to_bytes(it[0]))

        # Exhaust en AMS2 es el canal trasero 3D en exterior/chase: reemplazar con AC si AC trae Exhaust,
        # o neutralizarlo para que no se superponga el escape vanilla sobre el exterior de AC.
        exhaust_guids: set[bytes] = set()
        if "Exhaust" in event_guids:
            exhaust_guids.add(guid_str_to_bytes(event_guids["Exhaust"]))
        for it in raw_extras.get("Exhaust", ()):
            if it and it[0]:
                exhaust_guids.add(guid_str_to_bytes(it[0]))
        has_ac_exhaust = bool(
            ac_bank
            and any(
                any(k in (ev.path or "").lower() for k in ("exhaust", "tailpipe", "muffler"))
                for ev in ac_bank.events
            )
        )
        if has_ac_exhaust:
            replace_guids.update(exhaust_guids)
        else:
            silence_guids.update(exhaust_guids)

    return replace_guids, silence_guids, paths_by_bytes, raw_extras


def merge_ported_events_into_native_bank(
    native_bank: FmodBank,
    ported_bank: FmodBank,
    replace_event_guids: set[bytes],
    silence_native_event_guids: set[bytes] | None = None,
    event_paths_map: dict[bytes, str] | None = None,
    mod_identifier: str = "custom_mod",
    seed_tag: str = "hybrid",
    silence_bus_guid: bytes | None = None,
) -> bytes:
    """Combina eventos específicos del banco porteado dentro del banco vanilla original de AMS2.

    Mantiene 100% intactos los eventos vanilla no seleccionados (y sus sub-bancos FSB5 originales),
    mientras trasplanta únicamente el subgrafo y las muestras FSB5 de los eventos seleccionados
    desde `ported_bank` con GUIDs remapeados sin colisiones y tablas FMOD ordenadas.
    """
    fmt_payload = struct.pack("<II", native_bank.file_version or 0x50, native_bank.compat_version or 0x46)
    native_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(native_bank.raw, 12, len(native_bank.raw)):
        if cid == b"FMT ":
            fmt_payload = native_bank.raw[ps:pe]
        elif cid == b"LIST" and native_bank.raw[ps : ps + 4] == b"PROJ":
            native_proj = native_bank.raw[ps + 4 : pe]

    ported_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(ported_bank.raw, 12, len(ported_bank.raw)):
        if cid == b"LIST" and ported_bank.raw[ps : ps + 4] == b"PROJ":
            ported_proj = ported_bank.raw[ps + 4 : pe]
            break

    if not native_proj or not ported_proj:
        raise ValueError("Invalid FMOD bank: missing LIST:PROJ payload for partial merge.")

    native_tables = list(_iter_proj_table_items(native_proj))
    ported_tables = list(_iter_proj_table_items(ported_proj))

    native_evts_all: set[bytes] = {
        k for lt, items in native_tables if lt == b"EVTS" for k, _, _, _ in items
    }
    ported_evts_all: set[bytes] = {
        k for lt, items in ported_tables if lt == b"EVTS" for k, _, _, _ in items
    }

    actual_replace = {g for g in replace_event_guids if g in ported_evts_all}
    if not actual_replace:
        raise ValueError("Selected audio scope has no matching ported events to inject.")

    actual_silence = {
        g for g in (silence_native_event_guids or set()) if g in native_evts_all and g not in actual_replace
    }

    native_exported = _extract_bnki_exported_event_guids(native_proj, dict(native_tables).get(b"EVTS", ()))
    if native_exported:
        kept_native_roots = {
            g for g in native_exported if g not in actual_replace and g not in actual_silence
        }
    else:
        native_evib_targets = _collect_evib_targets(native_proj)
        kept_native_roots = {
            g
            for g in native_evts_all
            if g not in actual_replace and g not in actual_silence and g not in native_evib_targets
        }

    # Si no queda ningún evento vanilla que conservar (ej. un _AI.bank sin ai_engine_int en modo exterior),
    # devolver el banco porteado directamente.
    if not kept_native_roots and not actual_silence:
        return ported_bank.raw

    k_native = _extract_event_subgraph_closure(native_tables, kept_native_roots)
    c_ac = _extract_event_subgraph_closure(ported_tables, actual_replace)

    # Remapear todos los GUIDs internos de C_ac (excepto los eventos raíz destino) garantizando 0 colisiones
    all_native_guids: set[bytes] = set(collect_local_proj_guids(native_proj))
    all_native_guids.add(native_bank.bank_guid)
    reserved_guids: set[bytes] = all_native_guids | actual_replace | actual_silence

    seed_bytes = f"{seed_tag}:{mod_identifier}".encode("utf-8")
    remap_ac: dict[bytes, bytes] = {}
    ctr = 0
    for g in sorted(c_ac):
        if g in actual_replace:
            remap_ac[g] = g
            continue
        while True:
            cand = bytearray(hashlib.sha256(seed_bytes + struct.pack("<I", ctr) + g).digest()[:16])
            ctr += 1
            cand[7] = (cand[7] & 0x0F) | 0x40
            cand[8] = (cand[8] & 0x3F) | 0x80
            cb = bytes(cand)
            if cb not in reserved_guids and cb != b"\x00" * 16:
                remap_ac[g] = cb
                reserved_guids.add(cb)
                break

    # Crear stubs silenciosos para los eventos nativos neutralizados (ej. Exhaust vanilla en modo Exterior)
    ex_bus = silence_bus_guid or guid_str_to_bytes("{ffe57d5c-46ed-4b06-86d5-8fa6f8ffdf07}")
    silent_evnt_items: list[tuple[bytes, bytes]] = []
    silent_ibs_items: list[tuple[bytes, bytes]] = []
    for sg in sorted(actual_silence):
        s_ctr = 0
        while True:
            s_cand = bytearray(hashlib.sha256(seed_bytes + b":sil:" + struct.pack("<I", s_ctr) + sg).digest()[:16])
            s_ctr += 1
            s_cand[7] = (s_cand[7] & 0x0F) | 0x40
            s_cand[8] = (s_cand[8] & 0x3F) | 0x80
            s_ibg = bytes(s_cand)
            if s_ibg not in reserved_guids and s_ibg != b"\x00" * 16:
                reserved_guids.add(s_ibg)
                break
        silent_evnt_items.append((sg, _make_trigger_evnt(sg, s_ibg, ex_bus)))
        silent_ibs_items.append((s_ibg, _make_trigger_ibus(s_ibg, ex_bus)))

    # Recolectar únicamente los sub-bancos FSB5 referenciados por cada lado y construir el remapeo de índices
    used_nat_sbs: set[int] = set()
    for lt, items in native_tables:
        if lt == b"WAVS":
            for k, _full, _base, bpay in items:
                if k in k_native and len(bpay) >= 30:
                    _, sb_i, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
                    if 0 <= sb_i < len(native_bank.subbanks):
                        used_nat_sbs.add(sb_i)

    used_ac_sbs: set[int] = set()
    for lt, items in ported_tables:
        if lt == b"WAVS":
            for k, _full, _base, bpay in items:
                if k in c_ac and len(bpay) >= 30:
                    _, sb_i, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
                    if 0 <= sb_i < len(ported_bank.subbanks):
                        used_ac_sbs.add(sb_i)

    ordered_nat_sbs = sorted(used_nat_sbs)
    ordered_ac_sbs = sorted(used_ac_sbs)
    nat_sb_remap = {old_sb: new_idx for new_idx, old_sb in enumerate(ordered_nat_sbs)}
    ac_sb_remap = {old_sb: len(ordered_nat_sbs) + new_idx for new_idx, old_sb in enumerate(ordered_ac_sbs)}

    final_fsb_blobs: list[bytes] = (
        [bytes(native_bank.subbanks[sb].raw_fsb_bytes) for sb in ordered_nat_sbs]
        + [bytes(ported_bank.subbanks[sb].raw_fsb_bytes) for sb in ordered_ac_sbs]
    )
    if not final_fsb_blobs and native_bank.subbanks:
        final_fsb_blobs = [bytes(native_bank.subbanks[0].raw_fsb_bytes)]

    # Agrupar items de ported_bank por tabla ya remapeados
    ac_items_by_table: dict[bytes, list[tuple[bytes, bytes]]] = {}
    for lt, items in ported_tables:
        out_list: list[tuple[bytes, bytes]] = []
        for k, full, _base, bpay in items:
            if k not in c_ac:
                continue
            new_k = remap_ac.get(k, k)
            new_full = _remap_chunk_guids_leaves(full, remap_ac)
            if lt == b"WAVS" and len(bpay) >= 30:
                _, old_sb, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
                new_sb = ac_sb_remap.get(old_sb, 0)
                new_full = _patch_wav_chunk_subbank(new_full, new_sb)
            out_list.append((new_k, new_full))
        if out_list:
            ac_items_by_table[lt] = out_list

    nat_items_by_table: dict[bytes, list[tuple[bytes, bytes]]] = {}
    for lt, items in native_tables:
        out_list = []
        for k, full, _base, bpay in items:
            if k not in k_native:
                continue
            new_full = full
            if lt == b"WAVS" and len(bpay) >= 30:
                _, old_sb, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
                new_sb = nat_sb_remap.get(old_sb, old_sb)
                if new_sb != old_sb:
                    new_full = _patch_wav_chunk_subbank(new_full, new_sb)
            out_list.append((k, new_full))
        nat_items_by_table[lt] = out_list

    # Reconstruir LIST:PROJ preservando exactamente el orden de chunks del banco nativo
    merged_proj_parts = bytearray()
    seen_tables: set[bytes] = set()
    for cid, ps, pe, pos in iter_riff_chunks(native_proj, 0, len(native_proj)):
        pad_end = pe + ((pe - ps) & 1)
        if cid != b"LIST":
            merged_proj_parts += native_proj[pos:pad_end]
            continue
        lt = bytes(native_proj[ps : ps + 4])
        seen_tables.add(lt)
        combined_map: dict[bytes, bytes] = {}
        for k, f in nat_items_by_table.get(lt, ()):
            combined_map[k] = f
        for k, f in ac_items_by_table.get(lt, ()):
            combined_map[k] = f
        if lt == b"EVTS":
            for k, f in silent_evnt_items:
                combined_map[k] = f
        elif lt == b"IBSS":
            for k, f in silent_ibs_items:
                combined_map[k] = f

        combined_sorted = sorted(combined_map.items(), key=lambda x: x[0])
        if combined_sorted:
            lcnt_ch = _riff_chunk(b"LCNT", struct.pack("<I", len(combined_sorted)))
            body_lt = lt + lcnt_ch + b"".join(f for _, f in combined_sorted)
        else:
            body_lt = lt
        merged_proj_parts += _riff_chunk(b"LIST", body_lt)

    exported_ev_guids = kept_native_roots | actual_replace | actual_silence
    patched_proj_body = bytes(_sync_evts_partition_and_bnki(merged_proj_parts, exported_ev_guids))

    # Construir STDT con las rutas de eventos exportados y el banco
    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_mod"
    stdt_entries: list[tuple[str, bytes]] = []
    if native_bank.bank_guid:
        stdt_entries.append((f"bank:/{safe_mod}", native_bank.bank_guid))
    pmap = event_paths_map or {}
    for ev_i, eg in enumerate(sorted(exported_ev_guids)):
        ep = pmap.get(eg) or f"event:/AMS2/Vehicles/{safe_mod}/Event_{ev_i:02d}"
        stdt_entries.append((ep, eg))
    new_stdt_payload = build_stdt_chunk(stdt_entries) if stdt_entries else None

    sndh_ver = 0x00080000 | (2 * len(final_fsb_blobs) + 1)
    dummy_sndh = struct.pack("<I", sndh_ver) + b"\x00" * (8 * len(final_fsb_blobs))
    temp_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, dummy_sndh, new_stdt_payload)

    fmt_chunk_bytes = b"FMT " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if len(fmt_payload) & 1:
        fmt_chunk_bytes += b"\x00"

    proj_list_size = 4 + len(temp_proj_body)
    proj_total_bytes = 8 + proj_list_size + (proj_list_size & 1)

    cur_file_pos = 12 + len(fmt_chunk_bytes) + proj_total_bytes
    sndh_entries: list[tuple[int, int]] = []
    snd_chunks_built: list[bytes] = []

    for fsb_blob in final_fsb_blobs:
        snd_payload_start = cur_file_pos + 8
        pad_before = (32 - (snd_payload_start % 32)) % 32
        fsb_pos = snd_payload_start + pad_before
        fsb_len = len(fsb_blob)
        snd_payload = (b"\x00" * pad_before) + bytes(fsb_blob)
        snd_chunk = b"SND " + struct.pack("<I", len(snd_payload)) + snd_payload
        if len(snd_payload) & 1:
            snd_chunk += b"\x00"
        sndh_entries.append((fsb_pos, fsb_len))
        snd_chunks_built.append(snd_chunk)
        cur_file_pos += len(snd_chunk)

    real_sndh = struct.pack("<I", sndh_ver) + b"".join(
        struct.pack("<II", fpos, flen) for fpos, flen in sndh_entries
    )
    final_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, real_sndh, new_stdt_payload)
    assert len(final_proj_body) == len(temp_proj_body), "Tamaño inesperado al actualizar SNDH/STDT en banco híbrido"

    proj_chunk_bytes = (
        b"LIST"
        + struct.pack("<I", 4 + len(final_proj_body))
        + b"PROJ"
        + final_proj_body
        + (b"\x00" if (len(final_proj_body) & 1) else b"")
    )

    body = fmt_chunk_bytes + proj_chunk_bytes + b"".join(snd_chunks_built)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body


# ============================================================================
# Opción C: Injerto Quirúrgico por Evento + Enrutador Liviano EVIB Proxy
# ============================================================================

_EVIN_INST_PREFIX = bytes.fromhex(
    "0000000000000000000000000000000000000000000000000000000000000000000000c842"
    "00000000000000000000000000000000000000000000000000000000000000000000803f"
    "00000000ffffff7f1800"
)
_EVIN_INST_SUFFIX = bytes.fromhex("03000000ffffffff02000000000000000000000000000000")
assert len(_EVIN_INST_PREFIX) == 83 and len(_EVIN_INST_SUFFIX) == 24

_BUILTIN_3D_PRMB_HDR5: dict[str, bytes] = {
    "distance": b"\x00\x01\x00\x00\x00",
    "event cone angle": b"\x00\x02\x00\x00\x00",
    "event orientation": b"\x00\x03\x00\x00\x00",
    "direction": b"\x00\x04\x00\x00\x00",
    "elevation": b"\x00\x05\x00\x00\x00",
}


def _extract_bnki_exported_event_guids(
    proj_bytes: bytes,
    evts_items: list[tuple[bytes, bytes, bytes, bytes]] | tuple,
) -> set[bytes]:
    """Devuelve el conjunto exacto de GUIDs exportados según la partición EVTS[0 : BNKI[24:28])."""
    exp_cnt: int | None = None
    for cid, ps, pe, _ in iter_riff_chunks(proj_bytes, 0, len(proj_bytes)):
        if cid == b"BNKI" and pe - ps >= 28:
            exp_cnt = struct.unpack_from("<I", proj_bytes, ps + 24)[0]
            break
    if exp_cnt is None or exp_cnt <= 0 or exp_cnt > len(evts_items):
        return set()
    return {k for k, _, _, _ in list(evts_items)[:exp_cnt]}


def _alloc_unique_guid(reserved: set[bytes], seed: bytes, tag: bytes) -> bytes:
    """Genera un GUID v4 determinista que no colisiona con ningún objeto en `reserved`."""
    ctr = 0
    while True:
        cand = bytearray(hashlib.sha256(seed + b":" + tag + b":" + struct.pack("<I", ctr)).digest()[:16])
        ctr += 1
        cand[7] = (cand[7] & 0x0F) | 0x40
        cand[8] = (cand[8] & 0x3F) | 0x80
        cb = bytes(cand)
        if cb not in reserved and cb != b"\x00" * 16:
            reserved.add(cb)
            return cb


def _inspect_native_event_profiles(
    native_proj: bytes,
    native_tables: list[tuple[bytes, list[tuple[bytes, bytes, bytes, bytes]]]],
    native_panners: dict[bytes, tuple[int, float, float, int, float, float]] | None = None,
) -> dict[bytes, dict]:
    """Analiza cada evento de un banco nativo AMS2 (bus padre, throttle vs load, 3D, límites, panner)."""
    tmap = dict(native_tables)
    ibs_parent: dict[bytes, bytes] = {}
    for k, _, _, bpay in tmap.get(b"IBSS", ()):
        if len(bpay) >= 34:
            ibs_parent[k] = bytes(bpay[18:34])

    prm_name_by_guid: dict[bytes, str] = {}
    for k, _, _, bpay in tmap.get(b"PRMS", ()):
        if len(bpay) >= 23:
            sl = struct.unpack_from("<H", bpay, 21)[0]
            if 23 + sl <= len(bpay):
                prm_name_by_guid[k] = bpay[23 : 23 + sl].decode("latin1", "replace").rstrip("\x00").strip().lower()

    # Detectar BEFX Spatializer 3D (type 27, panBlend > 0.5)
    sp3d_guids: set[bytes] = set()
    for k, full, _, bpay in tmap.get(b"BEFX", ()):
        if len(bpay) >= 24 and struct.unpack_from("<I", bpay, 20)[0] == 27:
            for sc, sps, spe, _ in iter_riff_chunks(full, 12, len(full)):
                if sc == b"PMEF" and spe - sps >= 12:
                    pm = _parse_pmef_param_offsets(full[sps:spe])
                    if 17 in pm and pm[17][0] == 0 and float(pm[17][2]) > 0.5:
                        sp3d_guids.add(k)
                    break

    mbus_is_3d: dict[bytes, bool] = {}
    for k, full, _, _ in tmap.get(b"MBSS", ()):
        mbus_is_3d[k] = any(full[i : i + 16] in sp3d_guids for i in range(0, len(full) - 15))

    ctrs_items = tmap.get(b"CTRS", ())
    pmls_items = tmap.get(b"PMLS", ())

    profiles: dict[bytes, dict] = {}
    for ek, _full, _base, ebpay in tmap.get(b"EVTS", ()):
        ibs_g = bytes(ebpay[48:64]) if len(ebpay) >= 64 else b""
        mbs_g = bytes(ebpay[64:80]) if len(ebpay) >= 80 else b""
        max_i, flg = struct.unpack_from("<iI", ebpay, 80) if len(ebpay) >= 88 else (1, 0)
        dopp = struct.unpack_from("<f", ebpay, len(ebpay) - 12)[0] if len(ebpay) >= 96 else 0.0

        cl = _extract_event_subgraph_closure(native_tables, {ek})
        thr_hits = 0
        load_hits = 0
        for ck, _, _, cbpay in ctrs_items:
            if ck in cl and len(cbpay) >= 48:
                pn = prm_name_by_guid.get(bytes(cbpay[32:48]), "")
                if pn == "throttle":
                    thr_hits += 1
                elif pn == "load":
                    load_hits += 1
        for pmk, _, _, pmbpay in pmls_items:
            if pmk in cl and len(pmbpay) >= 50:
                nb = struct.unpack_from("<H", pmbpay, 48)[0] >> 1
                if nb > 0:
                    pn = prm_name_by_guid.get(bytes(pmbpay[16:32]), "")
                    if pn == "throttle":
                        thr_hits += 2
                    elif pn == "load":
                        load_hits += 2

        pref_thr: bool | None
        if thr_hits > 0 and load_hits == 0:
            pref_thr = True
        elif load_hits > 0 and thr_hits == 0:
            pref_thr = False
        elif thr_hits > load_hits:
            pref_thr = True
        elif load_hits > thr_hits:
            pref_thr = False
        else:
            pref_thr = None

        pbus = ibs_parent.get(ibs_g, b"")
        is_3d = bool(dopp > 0.5 or mbus_is_3d.get(mbs_g, False))
        profiles[ek] = {
            "parent_bus": pbus,
            "prefers_throttle": pref_thr,
            "is_3d": is_3d,
            "max_instances": int(max_i),
            "flags": int(flg),
            "panner_cfg": (native_panners or {}).get(ek),
        }
    return profiles


def _extract_event_prmb_metadata(
    items_by_table: dict[bytes, list[tuple[bytes, bytes]]],
    event_guid: bytes,
) -> list[dict]:
    """Devuelve metadatos completos de cada PRMB propio de `event_guid` (orden estable)."""
    owned_prm_guids: list[bytes] = []
    seen_p: set[bytes] = set()
    for _pk, pml_full in items_by_table.get(b"PMLS", ()):
        for sc, sps, spe, _ in iter_riff_chunks(pml_full, 12, len(pml_full)):
            if sc == b"PMLB" and spe - sps >= 48 and pml_full[sps + 32 : sps + 48] == event_guid:
                pg = bytes(pml_full[sps + 16 : sps + 32])
                if pg not in seen_p:
                    seen_p.add(pg)
                    owned_prm_guids.append(pg)
                break

    prms_lookup: dict[bytes, bytes] = {}
    for k, prm_full in items_by_table.get(b"PRMS", ()):
        for sc, sps, spe, _ in iter_riff_chunks(prm_full, 12, len(prm_full)):
            if sc == b"PRMB" and spe - sps >= 23:
                prms_lookup[k] = bytes(prm_full[sps:spe])
                break

    result: list[dict] = []
    for pg in owned_prm_guids:
        bp = prms_lookup.get(pg)
        if not bp or len(bp) < 23:
            continue
        sl = struct.unpack_from("<H", bp, 21)[0]
        if 23 + sl + 12 > len(bp):
            continue
        pname = bp[23 : 23 + sl].decode("latin1", "replace").rstrip("\x00")
        pmin, pmax, pdef = struct.unpack_from("<fff", bp, 23 + sl)
        result.append(
            {
                "guid": pg,
                "name": pname,
                "name_low": pname.strip().lower(),
                "hdr5": bytes(bp[16:21]),
                "pmin": float(pmin),
                "pmax": float(pmax),
                "pdef": float(pdef),
            }
        )
    return result


def _patch_evtb_doppler_and_limits(
    evnt_full: bytes,
    doppler_scale: float | None = None,
    max_instances: int | None = None,
    flags: int | None = None,
) -> bytes:
    """Actualiza doppler_scale (tail+4) y/o límites de instancias en un chunk LIST:EVNT."""
    out = bytearray()
    for sc, sps, spe, _ in iter_riff_chunks(evnt_full, 12, len(evnt_full)):
        sp = bytearray(evnt_full[sps:spe])
        if sc == b"EVTB" and len(sp) >= 96:
            if max_instances is not None:
                struct.pack_into("<i", sp, 80, int(max_instances))
            if flags is not None:
                cur_f = struct.unpack_from("<I", sp, 84)[0]
                struct.pack_into("<I", sp, 84, (cur_f & ~0xE0) | (int(flags) & 0xE0))
            if doppler_scale is not None:
                struct.pack_into("<f", sp, len(sp) - 12, float(doppler_scale))
        out += _riff_chunk(sc, bytes(sp))
    return _riff_list(b"EVNT", bytes(out))


def _strip_spatializer_from_mbus_chunk(mbus_full: bytes, sp_guid: bytes) -> tuple[bytes, bool]:
    """Elimina un GUID de efecto Spatializer de las listas pre/post-fader de BUS dentro de LIST:MBUS."""
    removed = False
    out = bytearray()
    for sc, sps, spe, _ in iter_riff_chunks(mbus_full, 12, len(mbus_full)):
        sp = mbus_full[sps:spe]
        if sc == b"BUS " and len(sp) >= 27:
            pos = 5
            arrays: list[list[bytes]] = []
            ok_parse = True
            for _ in range(2):
                if pos + 2 > len(sp):
                    ok_parse = False
                    break
                hdr = struct.unpack_from("<H", sp, pos)[0]
                cnt = hdr >> 1
                pos += 2
                if cnt == 0:
                    arrays.append([])
                else:
                    if pos + 2 > len(sp):
                        ok_parse = False
                        break
                    stride = struct.unpack_from("<H", sp, pos)[0]
                    pos += 2
                    if stride < 16 or pos + cnt * stride > len(sp):
                        ok_parse = False
                        break
                    arr = [bytes(sp[pos + i * stride : pos + i * stride + 16]) for i in range(cnt)]
                    pos += cnt * stride
                    arrays.append(arr)
            if ok_parse and len(arrays) == 2:
                tail = sp[pos:]
                new_arrays: list[list[bytes]] = []
                for arr in arrays:
                    filt = [g for g in arr if g != sp_guid]
                    if len(filt) != len(arr):
                        removed = True
                    new_arrays.append(filt)
                if removed:
                    new_bus = bytearray(sp[:5])
                    for arr in new_arrays:
                        if not arr:
                            new_bus += struct.pack("<H", 1)
                        else:
                            new_bus += struct.pack("<HH", (len(arr) << 1) | 1, 0x0010)
                            new_bus += b"".join(arr)
                    new_bus += tail
                    out += _riff_chunk(b"BUS ", bytes(new_bus))
                    continue
        out += _riff_chunk(sc, sp)
    return _riff_list(b"MBUS", bytes(out)), removed


def _patch_befx_spatializer_chunk(
    befx_full: bytes,
    new_guid: bytes,
    pan_cfg: tuple[int, float, float, int, float, float] | None = None,
) -> bytes:
    """Clona un chunk LIST:BEFX de Spatializer 3D con nuevo GUID y perfil de atenuación AMS2 opcional."""
    out = bytearray()
    for sc, sps, spe, _ in iter_riff_chunks(befx_full, 12, len(befx_full)):
        sp = bytearray(befx_full[sps:spe])
        if sc == b"BEFB" and len(sp) >= 16:
            sp[0:16] = new_guid
        elif sc == b"PMEF" and pan_cfg is not None and len(sp) >= 12:
            pm = _parse_pmef_param_offsets(bytes(sp))
            if 17 in pm and pm[17][0] == 0 and float(pm[17][2]) > 0.5:
                rolloff, min_d, max_d, ext_mode, snd_size, min_ext = pan_cfg
                if 11 in pm and pm[11][0] == 1:
                    struct.pack_into("<i", sp, pm[11][1], int(rolloff))
                if 12 in pm and pm[12][0] == 0:
                    struct.pack_into("<f", sp, pm[12][1], float(min_d))
                if 13 in pm and pm[13][0] == 0:
                    struct.pack_into("<f", sp, pm[13][1], float(max_d))
                if 14 in pm and pm[14][0] == 1:
                    struct.pack_into("<i", sp, pm[14][1], int(ext_mode))
                if 15 in pm and pm[15][0] == 0:
                    struct.pack_into("<f", sp, pm[15][1], float(snd_size))
                if 16 in pm and pm[16][0] == 0:
                    struct.pack_into("<f", sp, pm[16][1], float(min_ext))
        elif sc == b"LIST" and sp[:4] == b"PRPS":
            # El Spatializer clonado para proxies no hereda automatizaciones CTRL externas
            out += _riff_list(b"PRPS", b"")
            continue
        out += _riff_chunk(sc, bytes(sp))
    return _riff_list(b"BEFF", bytes(out))


def _hoist_3d_spatializer_from_ac_items(
    ac_items_by_table: dict[bytes, list[tuple[bytes, bytes]]],
    child_ev_guid: bytes,
) -> bytes | None:
    """Extrae el 3D Spatializer (BEFX 27, panBlend>0.5) del MBUS del sub-evento hijo hacia sus proxies padre.

    Sigue el patrón nativo exacto de AMS2 (AI_EngineMAIN / AI_EngineMAIN_LIVE -> AI_EngineExt): el Spatializer 3D
    vive en cada proxy padre (con su atenuación propia) y el sub-evento hijo tiene doppler=0 y sin doble espacialización.
    Si algún CTRL o MODS referencia directamente al Spatializer, se omite el traslado por seguridad.
    """
    evts_list = ac_items_by_table.get(b"EVTS", [])
    ev_entry_idx = next((i for i, (k, _) in enumerate(evts_list) if k == child_ev_guid), None)
    if ev_entry_idx is None:
        return None
    _, ev_full = evts_list[ev_entry_idx]
    mbs_g: bytes | None = None
    for sc, sps, spe, _ in iter_riff_chunks(ev_full, 12, len(ev_full)):
        if sc == b"EVTB" and spe - sps >= 80:
            mbs_g = bytes(ev_full[sps + 64 : sps + 80])
            break
    if not mbs_g or mbs_g == b"\x00" * 16:
        return None

    mbss_list = ac_items_by_table.get(b"MBSS", [])
    mb_idx = next((i for i, (k, _) in enumerate(mbss_list) if k == mbs_g), None)
    if mb_idx is None:
        return None
    _, mb_full = mbss_list[mb_idx]

    befx_list = ac_items_by_table.get(b"BEFX", [])
    sp_idx: int | None = None
    sp_guid: bytes | None = None
    sp_full: bytes | None = None
    for bi, (bk, bf) in enumerate(befx_list):
        if bk not in mb_full:
            continue
        is_pan_27 = False
        is_3d_pan = False
        for sc, sps, spe, _ in iter_riff_chunks(bf, 12, len(bf)):
            if sc == b"BEFB" and spe - sps >= 24 and struct.unpack_from("<I", bf, sps + 20)[0] == 27:
                is_pan_27 = True
            elif sc == b"PMEF" and spe - sps >= 12:
                pm = _parse_pmef_param_offsets(bf[sps:spe])
                if 17 in pm and pm[17][0] == 0 and float(pm[17][2]) > 0.5:
                    is_3d_pan = True
        if is_pan_27 and is_3d_pan:
            sp_idx = bi
            sp_guid = bk
            sp_full = bf
            break

    if sp_guid is None or sp_full is None or sp_idx is None:
        return None

    # Verificar que ningún CTRL ni MODS dependa directamente de este BEFX
    for tbl_tag in (b"CTRS", b"MODS"):
        for _k, cf in ac_items_by_table.get(tbl_tag, ()):
            if sp_guid in cf:
                return None

    new_mb_full, removed = _strip_spatializer_from_mbus_chunk(mb_full, sp_guid)
    if not removed:
        return None

    mbss_list[mb_idx] = (mbs_g, new_mb_full)
    befx_list.pop(sp_idx)
    evts_list[ev_entry_idx] = (child_ev_guid, _patch_evtb_doppler_and_limits(ev_full, doppler_scale=0.0))
    return sp_full


def _align_direct_graft_event(
    ac_items_by_table: dict[bytes, list[tuple[bytes, bytes]]],
    target_ev_guid: bytes,
    target_bus: bytes | None,
    prefer_throttle: bool | None,
    max_instances: int | None = None,
    flags: int | None = None,
) -> None:
    """Alinea bus de salida, límites de instancias y nombre throttle/load de un evento injertado 1:1."""
    evts_list = ac_items_by_table.get(b"EVTS", [])
    ev_idx = next((i for i, (k, _) in enumerate(evts_list) if k == target_ev_guid), None)
    if ev_idx is None:
        return
    _, ev_full = evts_list[ev_idx]
    ibs_g: bytes | None = None
    for sc, sps, spe, _ in iter_riff_chunks(ev_full, 12, len(ev_full)):
        if sc == b"EVTB" and spe - sps >= 64:
            ibs_g = bytes(ev_full[sps + 48 : sps + 64])
            break

    if max_instances is not None or flags is not None:
        evts_list[ev_idx] = (
            target_ev_guid,
            _patch_evtb_doppler_and_limits(ev_full, max_instances=max_instances, flags=flags),
        )

    if target_bus and target_bus != b"\x00" * 16 and ibs_g:
        ibss_list = ac_items_by_table.get(b"IBSS", [])
        for ii, (ik, ifull) in enumerate(ibss_list):
            if ik != ibs_g:
                continue
            out_ib = bytearray()
            for sc, sps, spe, _ in iter_riff_chunks(ifull, 12, len(ifull)):
                sp = bytearray(ifull[sps:spe])
                if sc == b"IBSB" and len(sp) >= 34:
                    sp[18:34] = target_bus
                out_ib += _riff_chunk(sc, bytes(sp))
            ibss_list[ii] = (ik, _riff_list(b"IBUS", bytes(out_ib)))
            break

    if prefer_throttle is None:
        return

    desired_name = "throttle" if prefer_throttle else "load"
    other_name = "load" if prefer_throttle else "throttle"
    ev_prms = _extract_event_prmb_metadata(ac_items_by_table, target_ev_guid)
    names_present = {p["name_low"] for p in ev_prms}
    if desired_name in names_present or other_name not in names_present:
        return

    target_p_guids = {p["guid"] for p in ev_prms if p["name_low"] == other_name}
    prms_list = ac_items_by_table.get(b"PRMS", [])
    desired_b = desired_name.encode("ascii")
    for pi, (pk, pfull) in enumerate(prms_list):
        if pk not in target_p_guids:
            continue
        out_p = bytearray()
        for sc, sps, spe, _ in iter_riff_chunks(pfull, 12, len(pfull)):
            sp = pfull[sps:spe]
            if sc == b"PRMB" and len(sp) >= 23:
                sl = struct.unpack_from("<H", sp, 21)[0]
                if 23 + sl + 12 <= len(sp):
                    tail_p = bytearray(sp[23 + sl :])
                    # Asegurar valor por defecto 1.0 para que jamás caiga en silencio/crackle si no se alimenta
                    struct.pack_into("<f", tail_p, 8, 1.0)
                    new_prmb = sp[:21] + struct.pack("<H", len(desired_b)) + desired_b + bytes(tail_p)
                    out_p += _riff_chunk(b"PRMB", new_prmb)
                    continue
            out_p += _riff_chunk(sc, sp)
        prms_list[pi] = (pk, _riff_list(b"PARM", bytes(out_p)))


def _make_prmb_chunk(
    prm_guid: bytes,
    name: str,
    pmin: float,
    pmax: float,
    pdef: float,
    pml_guid: bytes,
    hdr5: bytes = b"\x00\x00\x00\x00\x00",
) -> bytes:
    """Construye un chunk LIST:PARM (PRMB + LIST:PRPS vacío) idéntico al formato nativo de FMOD Studio."""
    nb = name.encode("ascii")
    prmb_pay = (
        prm_guid
        + hdr5
        + struct.pack("<H", len(nb))
        + nb
        + struct.pack("<fffffB", float(pmin), float(pmax), float(pdef), 0.0, 0.0, 0)
        + struct.pack("<HH", 0x0003, 0x0010)
        + pml_guid
    )
    return _riff_list(b"PARM", _riff_chunk(b"PRMB", prmb_pay) + _riff_list(b"PRPS", b""))


def _make_pmlo_chunk(
    pml_guid: bytes,
    prm_guid: bytes,
    ev_guid: bytes,
    boxes: list[tuple[bytes, float, float]],
    ctrl_guids: list[bytes],
) -> bytes:
    """Construye un chunk LIST:PMLO (PMLB + CTRO) idéntico al formato nativo de FMOD Studio."""
    if boxes:
        box_pay = struct.pack("<HH", (len(boxes) << 1) | 1, 0x0018) + b"".join(
            inst_g + struct.pack("<ff", float(v0), float(vlen)) for inst_g, v0, vlen in boxes
        )
    else:
        box_pay = struct.pack("<H", 1)
    pmlb_pay = pml_guid + prm_guid + ev_guid + box_pay
    if ctrl_guids:
        ctro_pay = struct.pack("<HH", (len(ctrl_guids) << 1) | 1, 0x0010) + b"".join(ctrl_guids)
    else:
        ctro_pay = struct.pack("<H", 1)
    return _riff_list(b"PMLO", _riff_chunk(b"PMLB", pmlb_pay) + _riff_chunk(b"CTRO", ctro_pay))


def _build_evib_proxy_event_chunks(
    proxy_ev_guid: bytes,
    child_ev_guid: bytes,
    child_prms: list[dict],
    target_bus_guid: bytes,
    is_3d: bool,
    prefer_throttle: bool,
    max_instances: int,
    flags: int,
    reserved_guids: set[bytes],
    seed_bytes: bytes,
    spatializer_template: bytes | None = None,
    pan_cfg: tuple[int, float, float, int, float, float] | None = None,
) -> dict[bytes, list[tuple[bytes, bytes]]]:
    """Sintetiza un evento proxy FMOD liviano (~1.1 KB) con Event Instrument (EVIB) hacia un sub-evento AC único.

    Reproduce byte a byte la arquitectura nativa de AMS2 (F_Ultimate_2022 EV[14] y F_Ultimate_2022_AI EV[3]):
    - Dispara permanentemente el sub-evento AC a través de una caja PMLB en todo el rango de `rpm` [0..max_rpm].
    - Reenvía linealmente `rpm` y puentea automáticamente `throttle` o `load` según lo que espera ese evento nativo.
    - En eventos 3D (Exterior / AI / Replay), aloja el 3D Spatializer ajustado a AMS2 en el MBUS del proxy padre.
    """
    tag_base = proxy_ev_guid
    tln_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"tln")
    ibs_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"ibs")
    mbs_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"mbs")
    gbs_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"gbs")
    evib_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"evib")

    out: dict[bytes, list[tuple[bytes, bytes]]] = {}

    def _add(tbl: bytes, k: bytes, ch: bytes) -> None:
        out.setdefault(tbl, []).append((k, ch))

    # 1. Opcional: Spatializer 3D clonado y ajustado a AMS2 en el MBUS del proxy
    sp_fx_g: bytes | None = None
    if is_3d and spatializer_template is not None:
        sp_fx_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"sp3d")
        _add(b"BEFX", sp_fx_g, _patch_befx_spatializer_chunk(spatializer_template, sp_fx_g, pan_cfg))

    # 2. Parámetros del sub-evento hijo -> slots EVIB + puentes PROP/MAP/CTRL/CURV
    child_rpm = next((p for p in child_prms if p["name_low"] in ("rpm", "rpms")), None)
    child_thr = next((p for p in child_prms if p["name_low"] in ("throttle", "load")), None)
    child_rpm_max = max(10000.0, float(child_rpm["pmax"]) if child_rpm else 22000.0)
    parent_rpm_max = max(child_rpm_max, 25000.0)

    used_prop_ids: set[int] = set()

    def _next_prop_id(cg: bytes) -> int:
        h = struct.unpack_from("<I", hashlib.sha256(cg + b":evib_prop").digest(), 0)[0] | 1
        while h in used_prop_ids:
            h = (h + 2) & 0xFFFFFFFF
            if h == 0:
                h = 1
        used_prop_ids.add(h)
        return h

    evib_slots: list[tuple[int, bytes, float]] = []
    child_prop_id_map: dict[bytes, int] = {}
    for cp in child_prms:
        pid = _next_prop_id(cp["guid"])
        child_prop_id_map[cp["guid"]] = pid
        def_v = 1.0 if cp["name_low"] in ("throttle", "load") else float(cp["pdef"])
        evib_slots.append((pid, cp["guid"], def_v))

    parent_pml_guids: list[bytes] = []
    evin_props_chunks: list[tuple[int, bytes]] = []

    # 2a. Parámetro padre `rpm` (siempre presente: aloja la caja EVIB persistente [0..parent_rpm_max])
    rpm_prm_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"prm_rpm")
    parent_pml_guids.append(rpm_prm_g)
    rpm_ctrls: list[bytes] = []
    if child_rpm is not None:
        pid_rpm = child_prop_id_map[child_rpm["guid"]]
        rpm_map_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"map_rpm")
        rpm_ctrl_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"ctrl_rpm")
        rpm_ctrls.append(rpm_ctrl_g)

        c_rmin = float(child_rpm["pmin"])
        c_rmax = max(c_rmin + 100.0, float(child_rpm["pmax"]))
        map_pay = rpm_map_g + struct.pack("<HHffff", 0x0005, 0x0008, 0.0, c_rmin, 1.0, c_rmax)
        _add(b"MPGS", rpm_map_g, _riff_chunk(b"MAP ", map_pay))

        ctrl_pay = rpm_ctrl_g + evib_g + rpm_prm_g + rpm_ctrl_g + struct.pack("<I", pid_rpm)
        _add(b"CTRS", rpm_ctrl_g, _riff_chunk(b"CTRL", ctrl_pay))

        curv_pay = (
            rpm_ctrl_g
            + rpm_ctrl_g
            + struct.pack("<HH", 0x0005, 0x0010)
            + struct.pack("<ffffffff", c_rmin, 0.0, 0.0, 0.0, c_rmax, 1.0, 0.0, 0.0)
        )
        _add(b"CRVS", rpm_ctrl_g, _riff_chunk(b"CURV", curv_pay))

        prop_pay = (
            struct.pack("<II", pid_rpm, 3)
            + rpm_map_g
            + struct.pack("<HH", 0x0003, 0x0010)
            + rpm_ctrl_g
            + struct.pack("<H", 1)
        )
        evin_props_chunks.append((pid_rpm, _riff_chunk(b"PROP", prop_pay)))

    _add(b"PRMS", rpm_prm_g, _make_prmb_chunk(rpm_prm_g, "rpm", 0.0, parent_rpm_max, 0.0, rpm_prm_g))
    _add(
        b"PMLS",
        rpm_prm_g,
        _make_pmlo_chunk(
            rpm_prm_g,
            rpm_prm_g,
            proxy_ev_guid,
            boxes=[(evib_g, 0.0, parent_rpm_max)],
            ctrl_guids=rpm_ctrls,
        ),
    )

    # 2b. Parámetro padre `throttle` o `load` puenteado hacia el parámetro de carga del hijo AC
    if child_thr is not None:
        parent_thr_name = "throttle" if prefer_throttle else "load"
        thr_prm_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"prm_thr")
        parent_pml_guids.append(thr_prm_g)

        pid_thr = child_prop_id_map[child_thr["guid"]]
        thr_map_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"map_thr")
        thr_ctrl_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"ctrl_thr")

        c_tmin = float(child_thr["pmin"])
        c_tmax = float(child_thr["pmax"]) if float(child_thr["pmax"]) > c_tmin else (c_tmin + 1.0)
        map_pay = thr_map_g + struct.pack("<HHffff", 0x0005, 0x0008, 0.0, c_tmin, 1.0, c_tmax)
        _add(b"MPGS", thr_map_g, _riff_chunk(b"MAP ", map_pay))

        ctrl_pay = thr_ctrl_g + evib_g + thr_prm_g + thr_ctrl_g + struct.pack("<I", pid_thr)
        _add(b"CTRS", thr_ctrl_g, _riff_chunk(b"CTRL", ctrl_pay))

        curv_pay = (
            thr_ctrl_g
            + thr_ctrl_g
            + struct.pack("<HH", 0x0005, 0x0010)
            + struct.pack("<ffffffff", c_tmin, 0.0, 0.0, 0.0, c_tmax, 1.0, 0.0, 0.0)
        )
        _add(b"CRVS", thr_ctrl_g, _riff_chunk(b"CURV", curv_pay))

        prop_pay = (
            struct.pack("<II", pid_thr, 3)
            + thr_map_g
            + struct.pack("<HH", 0x0003, 0x0010)
            + thr_ctrl_g
            + struct.pack("<H", 1)
        )
        evin_props_chunks.append((pid_thr, _riff_chunk(b"PROP", prop_pay)))

        _add(b"PRMS", thr_prm_g, _make_prmb_chunk(thr_prm_g, parent_thr_name, c_tmin, c_tmax, 1.0, thr_prm_g))
        _add(
            b"PMLS",
            thr_prm_g,
            _make_pmlo_chunk(thr_prm_g, thr_prm_g, proxy_ev_guid, boxes=[], ctrl_guids=[thr_ctrl_g]),
        )

    # 2c. Parámetros 3D integrados en el padre si el proxy es 3D (espeja F_Ultimate_2022_AI EV[3])
    if is_3d:
        for cp in child_prms:
            if cp["name_low"] in ("distance", "event cone angle", "event orientation", "direction"):
                b3d_g = _alloc_unique_guid(reserved_guids, seed_bytes, tag_base + b"prm3d_" + cp["name_low"].encode())
                parent_pml_guids.append(b3d_g)
                hdr5 = _BUILTIN_3D_PRMB_HDR5.get(cp["name_low"], cp["hdr5"])
                _add(
                    b"PRMS",
                    b3d_g,
                    _make_prmb_chunk(b3d_g, cp["name"], cp["pmin"], cp["pmax"], cp["pdef"], b3d_g, hdr5=hdr5),
                )
                _add(b"PMLS", b3d_g, _make_pmlo_chunk(b3d_g, b3d_g, proxy_ev_guid, boxes=[], ctrl_guids=[]))

    # 3. EVIN (EVIB + INST + LIST:PRPS ordenado por prop_id ascendente para búsqueda binaria de FMOD)
    evib_slots.sort(key=lambda x: x[0])
    evin_props_chunks.sort(key=lambda x: x[0])
    if evib_slots:
        slots_pay = struct.pack("<HH", (len(evib_slots) << 1) | 1, 0x0018) + b"".join(
            struct.pack("<I", pid) + cpg + struct.pack("<f", dv) for pid, cpg, dv in evib_slots
        )
    else:
        slots_pay = struct.pack("<H", 1)
    evib_pay = evib_g + child_ev_guid + struct.pack("<f", 100.0) + slots_pay
    inst_pay = _EVIN_INST_PREFIX + gbs_g + _EVIN_INST_SUFFIX
    if evin_props_chunks:
        prps_body = _riff_chunk(b"LCNT", struct.pack("<I", len(evin_props_chunks))) + b"".join(
            ch for _, ch in evin_props_chunks
        )
    else:
        prps_body = b""
    evin_full = _riff_list(
        b"EVIN",
        _riff_chunk(b"EVIB", evib_pay) + _riff_chunk(b"INST", inst_pay) + _riff_list(b"PRPS", prps_body),
    )
    _add(b"EVIS", evib_g, evin_full)

    # 4. TMLN (TLNB + TRNS vacío + CTRO vacío)
    tlnb_pay = tln_g + proxy_ev_guid + struct.pack("<HHHHH", 1, 1, 1, 0, 1)
    tmln_full = _riff_list(
        b"TMLN",
        _riff_chunk(b"TLNB", tlnb_pay) + _riff_list(b"TRNS", b"") + _riff_chunk(b"CTRO", struct.pack("<H", 1)),
    )
    _add(b"TLNS", tln_g, tmln_full)

    # 5. IBUS (IBSB -> target_bus_guid + BUS + PRPS vacío)
    ibsb_pay = ibs_g + b"\x18\x00" + target_bus_guid + struct.pack("<Ii", 7, -1)
    ibus_bus_pay = bytes.fromhex("01070000000100010008000000000000000000ffffff7f00000000")
    ibus_full = _riff_list(
        b"IBUS",
        _riff_chunk(b"IBSB", ibsb_pay) + _riff_chunk(b"BUS ", ibus_bus_pay) + _riff_list(b"PRPS", b""),
    )
    _add(b"IBSS", ibs_g, ibus_full)

    # 6. MBUS (MBSB + BUS con Spatializer 3D si aplica + PRPS vacío)
    mbs_ch_fmt = 7 if is_3d else 3
    mbsb_pay = mbs_g + b"\x18\x00" + (b"\x00" * 16) + struct.pack("<Ii", mbs_ch_fmt, -1)
    post_fx = (struct.pack("<HH", 0x0003, 0x0010) + sp_fx_g) if sp_fx_g else struct.pack("<H", 1)
    mbus_bus_pay = (
        bytes.fromhex("00030000000100")
        + post_fx
        + bytes.fromhex("08000000000000000000ffffff7f02000000")
    )
    mbus_full = _riff_list(
        b"MBUS",
        _riff_chunk(b"MBSB", mbsb_pay) + _riff_chunk(b"BUS ", mbus_bus_pay) + _riff_list(b"PRPS", b""),
    )
    _add(b"MBSS", mbs_g, mbus_full)

    # 7. GBUS (GBSB -> mbs_g + BUS + PRPS vacío)
    gbsb_pay = gbs_g + b"\x18\x00" + mbs_g + struct.pack("<Ii", 7 if is_3d else 3, -1)
    gbus_bus_pay = bytes.fromhex("00030000000100010008000000000000000000ffffff7f02000000")
    gbus_full = _riff_list(
        b"GBUS",
        _riff_chunk(b"GBSB", gbsb_pay) + _riff_chunk(b"BUS ", gbus_bus_pay) + _riff_list(b"PRPS", b""),
    )
    _add(b"GBSS", gbs_g, gbus_full)

    # 8. EVNT (EVTB + PRPS vacío)
    pml_arr_pay = (
        struct.pack("<HH", (len(parent_pml_guids) << 1) | 1, 0x0010) + b"".join(parent_pml_guids)
        if parent_pml_guids
        else struct.pack("<H", 1)
    )
    evtb_pay = (
        proxy_ev_guid
        + (b"\x00" * 16)
        + tln_g
        + ibs_g
        + mbs_g
        + struct.pack("<iIBI", int(max_instances), int(flags), 0, 1)
        + pml_arr_pay
        + struct.pack("<ffII", 0.0, 100.0 if is_3d else 0.0, 1, 0)
    )
    evnt_full = _riff_list(b"EVNT", _riff_chunk(b"EVTB", evtb_pay) + _riff_list(b"PRPS", b""))
    _add(b"EVTS", proxy_ev_guid, evnt_full)

    return out


def graft_ac_into_native_bank_surgical(
    native_bank: FmodBank,
    ported_ac_bank: FmodBank,
    ac_int_event_guid: bytes | None,
    ac_ext_event_guid: bytes | None,
    target_int_guids: list[bytes] | None = None,
    target_ext_guids: list[bytes] | None = None,
    silence_native_guids: set[bytes] | None = None,
    event_paths_map: dict[bytes, str] | None = None,
    is_ai_bank: bool = False,
    native_panners: dict[bytes, tuple[int, float, float, int, float, float]] | None = None,
    default_prefer_throttle: bool = False,
    mod_identifier: str = "custom_mod",
    seed_tag: str = "surgical",
    ai_live_guids: set[bytes] | None = None,
) -> tuple[bytes, dict[str, bytes | None]]:
    """Opción C: Injerta quirúrgicamente eventos de AC dentro de un .bank nativo de AMS2 con proxies EVIB.

    - Preserva 100% intactos los eventos nativos no seleccionados (otros autos del mismo .bank, Starter,
      Starting, Exhaust y sus sub-bancos FSB5 originales) sin ningún puntero colgante.
    - Cuando un rol (Interior o Exterior) apunta a 1 único evento destino -> injerto directo 1:1 (cero overhead).
    - Cuando un rol apunta a >=2 variantes/eventos (ej. F_Ultimate_2022 con 12 EngineInt y 3 EngineExt,
      o múltiples AI_EngineMAIN/LIVE) -> inserta UNA sola copia del grafo AC como sub-evento interno (NST)
      y crea proxies EVIB livianos (~1.1 KB c/u) con puenteo por-evento de `throttle`/`load` y Spatializer 3D
      en cada proxy exterior.
    - Retorna (bank_bytes, {"ac_int_guid": guid_final_int, "ac_ext_guid": guid_final_ext}).
    """
    EXT_BUS = guid_str_to_bytes("{c1986fb5-b329-4565-8342-b223857006d5}")
    INT_BUS = guid_str_to_bytes("{a8959bf6-0b54-43ea-9d9a-06a8831c40d4}")
    AI_BUS = guid_str_to_bytes(AMS2_AI_ENGINE_BUS)
    EXH_BUS = guid_str_to_bytes("{ffe57d5c-46ed-4b06-86d5-8fa6f8ffdf07}")

    fmt_payload = struct.pack("<II", native_bank.file_version or 0x50, native_bank.compat_version or 0x46)
    native_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(native_bank.raw, 12, len(native_bank.raw)):
        if cid == b"FMT ":
            fmt_payload = native_bank.raw[ps:pe]
        elif cid == b"LIST" and native_bank.raw[ps : ps + 4] == b"PROJ":
            native_proj = native_bank.raw[ps + 4 : pe]

    ported_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(ported_ac_bank.raw, 12, len(ported_ac_bank.raw)):
        if cid == b"LIST" and ported_ac_bank.raw[ps : ps + 4] == b"PROJ":
            ported_proj = ported_ac_bank.raw[ps + 4 : pe]
            break

    if not native_proj or not ported_proj:
        raise ValueError("Invalid FMOD bank: missing LIST:PROJ payload for surgical graft.")

    native_tables = list(_iter_proj_table_items(native_proj))
    ported_tables = list(_iter_proj_table_items(ported_proj))
    native_tmap = dict(native_tables)
    ported_tmap = dict(ported_tables)

    native_evts_all = {k for k, _, _, _ in native_tmap.get(b"EVTS", ())}
    ported_evts_all = {k for k, _, _, _ in ported_tmap.get(b"EVTS", ())}

    # Deduplicar destinos válidos dentro del banco nativo
    int_targets: list[bytes] = []
    seen_targets: set[bytes] = set()
    if ac_int_event_guid and ac_int_event_guid in ported_evts_all:
        for g in (target_int_guids or ()):
            if g in native_evts_all and g not in seen_targets:
                seen_targets.add(g)
                int_targets.append(g)

    ext_targets: list[bytes] = []
    if ac_ext_event_guid and ac_ext_event_guid in ported_evts_all:
        for g in (target_ext_guids or ()):
            if g in native_evts_all and g not in seen_targets:
                seen_targets.add(g)
                ext_targets.append(g)

    all_replaced = set(int_targets) | set(ext_targets)
    if not all_replaced:
        raise ValueError("No matching target events in native bank for surgical AC graft.")

    actual_silence = {
        g for g in (silence_native_guids or set()) if g in native_evts_all and g not in all_replaced
    }

    native_profiles = _inspect_native_event_profiles(native_proj, native_tables, native_panners=native_panners)
    native_exported = _extract_bnki_exported_event_guids(native_proj, native_tmap.get(b"EVTS", ()))
    if not native_exported:
        nat_evib_tgts = _collect_evib_targets(native_proj)
        native_exported = {g for g in native_evts_all if g not in nat_evib_tgts}

    kept_native_roots = {g for g in native_exported if g not in all_replaced and g not in actual_silence}
    k_native = _extract_event_subgraph_closure(native_tables, kept_native_roots)

    # Extraer subgrafos limpios de los 1 ó 2 eventos AC fuente (sin basura: wheel/horn/door/bodywork)
    ac_src_roots: set[bytes] = set()
    if int_targets and ac_int_event_guid:
        ac_src_roots.add(ac_int_event_guid)
    if ext_targets and ac_ext_event_guid:
        ac_src_roots.add(ac_ext_event_guid)
    c_ac = _extract_event_subgraph_closure(ported_tables, ac_src_roots)

    # Reservar GUIDs nativos y destinos para garantizar 0 colisiones
    all_native_guids = set(collect_local_proj_guids(native_proj))
    if native_bank.bank_guid:
        all_native_guids.add(native_bank.bank_guid)
    reserved_guids: set[bytes] = all_native_guids | all_replaced | actual_silence
    seed_bytes = f"{seed_tag}:{mod_identifier}:{'ai' if is_ai_bank else 'p'}".encode("utf-8")

    # Determinar GUID final del evento AC Interior y Exterior (directo si len==1, NST único si len>=2)
    int_uses_proxy = len(int_targets) >= 2
    ext_uses_proxy = len(ext_targets) >= 2
    final_ac_int_guid: bytes | None = None
    final_ac_ext_guid: bytes | None = None

    if int_targets and ac_int_event_guid:
        final_ac_int_guid = (
            _alloc_unique_guid(reserved_guids, seed_bytes, b"ac_int_nst")
            if int_uses_proxy
            else int_targets[0]
        )
    if ext_targets and ac_ext_event_guid:
        if ac_ext_event_guid == ac_int_event_guid:
            # Caso raro: banco AC con un único evento para ambos roles; usar proxy separado si se pide
            final_ac_ext_guid = final_ac_int_guid
        else:
            final_ac_ext_guid = (
                _alloc_unique_guid(reserved_guids, seed_bytes, b"ac_ext_nst")
                if ext_uses_proxy
                else ext_targets[0]
            )

    remap_ac: dict[bytes, bytes] = {}
    if ac_int_event_guid and final_ac_int_guid:
        remap_ac[ac_int_event_guid] = final_ac_int_guid
    if ac_ext_event_guid and final_ac_ext_guid and ac_ext_event_guid not in remap_ac:
        remap_ac[ac_ext_event_guid] = final_ac_ext_guid

    for g in sorted(c_ac):
        if g in remap_ac:
            continue
        remap_ac[g] = _alloc_unique_guid(reserved_guids, seed_bytes, b"ac:" + g)

    # Recolectar únicamente los sub-bancos FSB5 referenciados por cada lado
    used_nat_sbs: set[int] = set()
    for k, _full, _base, bpay in native_tmap.get(b"WAVS", ()):
        if k in k_native and len(bpay) >= 30:
            _, sb_i, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
            if 0 <= sb_i < len(native_bank.subbanks):
                used_nat_sbs.add(sb_i)

    used_ac_sbs: set[int] = set()
    for k, _full, _base, bpay in ported_tmap.get(b"WAVS", ()):
        if k in c_ac and len(bpay) >= 30:
            _, sb_i, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
            if 0 <= sb_i < len(ported_ac_bank.subbanks):
                used_ac_sbs.add(sb_i)

    ordered_nat_sbs = sorted(used_nat_sbs)
    ordered_ac_sbs = sorted(used_ac_sbs)
    nat_sb_remap = {old_sb: new_idx for new_idx, old_sb in enumerate(ordered_nat_sbs)}
    ac_sb_remap = {old_sb: len(ordered_nat_sbs) + new_idx for new_idx, old_sb in enumerate(ordered_ac_sbs)}

    final_fsb_blobs: list[bytes] = (
        [bytes(native_bank.subbanks[sb].raw_fsb_bytes) for sb in ordered_nat_sbs]
        + [bytes(ported_ac_bank.subbanks[sb].raw_fsb_bytes) for sb in ordered_ac_sbs]
    )
    if not final_fsb_blobs and native_bank.subbanks:
        final_fsb_blobs = [bytes(native_bank.subbanks[0].raw_fsb_bytes)]

    # Remapear chunks del subgrafo AC
    ac_items_by_table: dict[bytes, list[tuple[bytes, bytes]]] = {}
    for lt, items in ported_tables:
        out_list: list[tuple[bytes, bytes]] = []
        for k, full, _base, bpay in items:
            if k not in c_ac:
                continue
            new_k = remap_ac.get(k, k)
            new_full = _remap_chunk_guids_leaves(full, remap_ac)
            if lt == b"WAVS" and len(bpay) >= 30:
                _, old_sb, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
                new_sb = ac_sb_remap.get(old_sb, 0)
                new_full = _patch_wav_chunk_subbank(new_full, new_sb)
            out_list.append((new_k, new_full))
        if out_list:
            ac_items_by_table[lt] = out_list

    # Alinear casos directos (1 único evento destino) y elevar Spatializer 3D si Exterior usa proxies EVIB
    default_max_inst = AI_EVENT_MAX_INSTANCES if is_ai_bank else 4

    if int_targets and final_ac_int_guid and not int_uses_proxy:
        prof0 = native_profiles.get(int_targets[0], {})
        bus0 = prof0.get("parent_bus") or INT_BUS
        _align_direct_graft_event(
            ac_items_by_table,
            final_ac_int_guid,
            target_bus=bus0,
            prefer_throttle=prof0.get("prefers_throttle"),
            max_instances=default_max_inst,
            flags=AI_EVENT_STEAL_VIRTUALIZE,
        )
    elif int_targets and final_ac_int_guid and int_uses_proxy:
        # Asegurar que el hijo NST interior tenga bus INT_BUS (para reconocimiento en Edit Modified Sound)
        _align_direct_graft_event(
            ac_items_by_table,
            final_ac_int_guid,
            target_bus=INT_BUS,
            prefer_throttle=None,
            max_instances=default_max_inst,
            flags=AI_EVENT_STEAL_VIRTUALIZE,
        )

    ext_spatializer_template: bytes | None = None
    if ext_targets and final_ac_ext_guid and not ext_uses_proxy:
        prof0 = native_profiles.get(ext_targets[0], {})
        bus0 = prof0.get("parent_bus") or (AI_BUS if is_ai_bank else EXT_BUS)
        _align_direct_graft_event(
            ac_items_by_table,
            final_ac_ext_guid,
            target_bus=bus0,
            prefer_throttle=prof0.get("prefers_throttle"),
            max_instances=default_max_inst,
            flags=AI_EVENT_STEAL_VIRTUALIZE,
        )
    elif ext_targets and final_ac_ext_guid and ext_uses_proxy:
        _align_direct_graft_event(
            ac_items_by_table,
            final_ac_ext_guid,
            target_bus=(AI_BUS if is_ai_bank else EXT_BUS),
            prefer_throttle=None,
            max_instances=default_max_inst,
            flags=AI_EVENT_STEAL_VIRTUALIZE,
        )
        ext_spatializer_template = _hoist_3d_spatializer_from_ac_items(ac_items_by_table, final_ac_ext_guid)

    # Sintetizar proxies EVIB livianos para roles multi-destino
    proxy_items_by_table: dict[bytes, list[tuple[bytes, bytes]]] = {}

    def _merge_proxy_chunks(chunks_map: dict[bytes, list[tuple[bytes, bytes]]]) -> None:
        for tbl, lst in chunks_map.items():
            proxy_items_by_table.setdefault(tbl, []).extend(lst)

    if int_uses_proxy and final_ac_int_guid:
        child_int_prms = _extract_event_prmb_metadata(ac_items_by_table, final_ac_int_guid)
        for tg in int_targets:
            prof = native_profiles.get(tg, {})
            t_bus = prof.get("parent_bus") or INT_BUS
            t_pref_thr = (
                prof["prefers_throttle"]
                if prof.get("prefers_throttle") is not None
                else bool(default_prefer_throttle)
            )
            p_chunks = _build_evib_proxy_event_chunks(
                proxy_ev_guid=tg,
                child_ev_guid=final_ac_int_guid,
                child_prms=child_int_prms,
                target_bus_guid=t_bus,
                is_3d=False,
                prefer_throttle=bool(t_pref_thr),
                max_instances=default_max_inst,
                flags=AI_EVENT_STEAL_VIRTUALIZE,
                reserved_guids=reserved_guids,
                seed_bytes=seed_bytes,
            )
            _merge_proxy_chunks(p_chunks)

    if ext_uses_proxy and final_ac_ext_guid:
        child_ext_prms = _extract_event_prmb_metadata(ac_items_by_table, final_ac_ext_guid)
        live_set = ai_live_guids or set()
        for tg in ext_targets:
            prof = native_profiles.get(tg, {})
            t_bus = prof.get("parent_bus") or (AI_BUS if is_ai_bank else EXT_BUS)
            t_pref_thr = (
                prof["prefers_throttle"]
                if prof.get("prefers_throttle") is not None
                else bool(default_prefer_throttle)
            )
            if prof.get("panner_cfg") is not None:
                pan_cfg = prof["panner_cfg"]
            elif is_ai_bank:
                pan_cfg = AMS2_LIVE_REPLAY_PANNER if tg in live_set else AMS2_AI_MAIN_PANNER
            else:
                pan_cfg = AMS2_PLAYER_EXT_PANNER

            p_chunks = _build_evib_proxy_event_chunks(
                proxy_ev_guid=tg,
                child_ev_guid=final_ac_ext_guid,
                child_prms=child_ext_prms,
                target_bus_guid=t_bus,
                is_3d=True,
                prefer_throttle=bool(t_pref_thr),
                max_instances=default_max_inst,
                flags=AI_EVENT_STEAL_VIRTUALIZE,
                reserved_guids=reserved_guids,
                seed_bytes=seed_bytes,
                spatializer_template=ext_spatializer_template,
                pan_cfg=pan_cfg,
            )
            _merge_proxy_chunks(p_chunks)

    # Crear stubs silenciosos completos (con MBUS y TLN válidos -> 0 punteros colgantes) para eventos neutralizados
    for sg in sorted(actual_silence):
        prof_s = native_profiles.get(sg, {})
        s_bus = prof_s.get("parent_bus") or EXH_BUS
        s_tln = _alloc_unique_guid(reserved_guids, seed_bytes, b"sil_tln:" + sg)
        s_ibs = _alloc_unique_guid(reserved_guids, seed_bytes, b"sil_ibs:" + sg)
        s_mbs = _alloc_unique_guid(reserved_guids, seed_bytes, b"sil_mbs:" + sg)
        tlnb_pay = s_tln + sg + struct.pack("<HHHHH", 1, 1, 1, 0, 1)
        ibsb_pay = s_ibs + b"\x18\x00" + s_bus + struct.pack("<Ii", 3, -1)
        mbsb_pay = s_mbs + b"\x18\x00" + (b"\x00" * 16) + struct.pack("<Ii", 3, -1)
        bus_pay = bytes.fromhex("00030000000100010008000000000000000000ffffff7f02000000")
        ibus_bus_pay = bytes.fromhex("01030000000100010008000000000000000000ffffff7f00000000")
        evtb_pay = (
            sg
            + (b"\x00" * 16)
            + s_tln
            + s_ibs
            + s_mbs
            + struct.pack("<iIBI", 1, 0, 0, 1)
            + struct.pack("<H", 1)
            + struct.pack("<ffII", 0.0, 0.0, 1, 0)
        )
        proxy_items_by_table.setdefault(b"EVTS", []).append(
            (sg, _riff_list(b"EVNT", _riff_chunk(b"EVTB", evtb_pay) + _riff_list(b"PRPS", b"")))
        )
        proxy_items_by_table.setdefault(b"TLNS", []).append(
            (
                s_tln,
                _riff_list(
                    b"TMLN",
                    _riff_chunk(b"TLNB", tlnb_pay) + _riff_list(b"TRNS", b"") + _riff_chunk(b"CTRO", struct.pack("<H", 1)),
                ),
            )
        )
        proxy_items_by_table.setdefault(b"IBSS", []).append(
            (
                s_ibs,
                _riff_list(b"IBUS", _riff_chunk(b"IBSB", ibsb_pay) + _riff_chunk(b"BUS ", ibus_bus_pay) + _riff_list(b"PRPS", b"")),
            )
        )
        proxy_items_by_table.setdefault(b"MBSS", []).append(
            (
                s_mbs,
                _riff_list(b"MBUS", _riff_chunk(b"MBSB", mbsb_pay) + _riff_chunk(b"BUS ", bus_pay) + _riff_list(b"PRPS", b"")),
            )
        )

    # Filtrar y reindexar chunks nativos conservados
    nat_items_by_table: dict[bytes, list[tuple[bytes, bytes]]] = {}
    for lt, items in native_tables:
        out_list = []
        for k, full, _base, bpay in items:
            if k not in k_native:
                continue
            new_full = full
            if lt == b"WAVS" and len(bpay) >= 30:
                _, old_sb, _, _, _ = struct.unpack_from("<HHHII", bpay, 16)
                new_sb = nat_sb_remap.get(old_sb, old_sb)
                if new_sb != old_sb:
                    new_full = _patch_wav_chunk_subbank(new_full, new_sb)
            out_list.append((k, new_full))
        nat_items_by_table[lt] = out_list

    # Reconstruir LIST:PROJ preservando el orden exacto de tablas del banco nativo
    merged_proj_parts = bytearray()
    for cid, ps, pe, pos in iter_riff_chunks(native_proj, 0, len(native_proj)):
        pad_end = pe + ((pe - ps) & 1)
        if cid != b"LIST":
            merged_proj_parts += native_proj[pos:pad_end]
            continue
        lt = bytes(native_proj[ps : ps + 4])
        combined_map: dict[bytes, bytes] = {}
        for k, f in nat_items_by_table.get(lt, ()):
            combined_map[k] = f
        for k, f in ac_items_by_table.get(lt, ()):
            combined_map[k] = f
        for k, f in proxy_items_by_table.get(lt, ()):
            combined_map[k] = f

        combined_sorted = sorted(combined_map.items(), key=lambda x: x[0])
        if combined_sorted:
            lcnt_ch = _riff_chunk(b"LCNT", struct.pack("<I", len(combined_sorted)))
            body_lt = lt + lcnt_ch + b"".join(f for _, f in combined_sorted)
        else:
            body_lt = lt
        merged_proj_parts += _riff_chunk(b"LIST", body_lt)

    # Particionar EVTS [0..exported) + [exported..total) y sincronizar BNKI[24:28]
    exported_ev_guids = kept_native_roots | all_replaced | actual_silence
    patched_proj_body = bytes(_sync_evts_partition_and_bnki(merged_proj_parts, exported_ev_guids))

    # Construir STDT únicamente con el banco y los eventos exportados (nunca sub-eventos NST)
    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_mod"
    stdt_entries: list[tuple[str, bytes]] = []
    if native_bank.bank_guid:
        stdt_entries.append((f"bank:/{safe_mod}", native_bank.bank_guid))
    pmap = event_paths_map or {}
    for ev_i, eg in enumerate(sorted(exported_ev_guids)):
        ep = pmap.get(eg) or f"event:/AMS2/Vehicles/{safe_mod}/Event_{ev_i:02d}"
        stdt_entries.append((ep, eg))
    new_stdt_payload = build_stdt_chunk(stdt_entries) if stdt_entries else None

    sndh_ver = 0x00080000 | (2 * len(final_fsb_blobs) + 1)
    dummy_sndh = struct.pack("<I", sndh_ver) + b"\x00" * (8 * len(final_fsb_blobs))
    temp_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, dummy_sndh, new_stdt_payload)

    fmt_chunk_bytes = b"FMT " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if len(fmt_payload) & 1:
        fmt_chunk_bytes += b"\x00"

    proj_list_size = 4 + len(temp_proj_body)
    proj_total_bytes = 8 + proj_list_size + (proj_list_size & 1)

    cur_file_pos = 12 + len(fmt_chunk_bytes) + proj_total_bytes
    sndh_entries: list[tuple[int, int]] = []
    snd_chunks_built: list[bytes] = []

    for fsb_blob in final_fsb_blobs:
        snd_payload_start = cur_file_pos + 8
        pad_before = (32 - (snd_payload_start % 32)) % 32
        fsb_pos = snd_payload_start + pad_before
        fsb_len = len(fsb_blob)
        snd_payload = (b"\x00" * pad_before) + bytes(fsb_blob)
        snd_chunk = b"SND " + struct.pack("<I", len(snd_payload)) + snd_payload
        if len(snd_payload) & 1:
            snd_chunk += b"\x00"
        sndh_entries.append((fsb_pos, fsb_len))
        snd_chunks_built.append(snd_chunk)
        cur_file_pos += len(snd_chunk)

    real_sndh = struct.pack("<I", sndh_ver) + b"".join(
        struct.pack("<II", fpos, flen) for fpos, flen in sndh_entries
    )
    final_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, real_sndh, new_stdt_payload)
    assert len(final_proj_body) == len(temp_proj_body), "Tamaño inesperado al actualizar SNDH/STDT en injerto quirúrgico"

    proj_chunk_bytes = (
        b"LIST"
        + struct.pack("<I", 4 + len(final_proj_body))
        + b"PROJ"
        + final_proj_body
        + (b"\x00" if (len(final_proj_body) & 1) else b"")
    )
    body = fmt_chunk_bytes + proj_chunk_bytes + b"".join(snd_chunks_built)
    out_bytes = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body
    meta = {
        "ac_int_guid": final_ac_int_guid,
        "ac_ext_guid": final_ac_ext_guid,
    }
    return out_bytes, meta


def verify_fmod_bank_integrity(
    bank_bytes: bytes,
    expected_exported_guids: set[bytes] | None = None,
) -> list[str]:
    """Verifica invariantes binarios críticos de FMOD Studio sobre un .bank compilado.

    Retorna una lista vacía `[]` si el banco es 100% íntegro (tablas ordenadas, partición EVTS/BNKI
    válida para búsqueda binaria, cero referencias colgantes en EVTB/EVIB/PMLB/CTRL/CURV/WAVS y
    alineación 32 bytes de FSB5).
    """
    errors: list[str] = []
    if len(bank_bytes) < 20 or bank_bytes[:4] != b"RIFF" or bank_bytes[8:12] != b"FEV ":
        return ["Cabecera RIFF/FEV inválida"]

    proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(bank_bytes, 12, len(bank_bytes)):
        if cid == b"LIST" and bank_bytes[ps : ps + 4] == b"PROJ":
            proj = bank_bytes[ps + 4 : pe]
            break
    if not proj:
        return ["Falta chunk LIST:PROJ"]

    tables = list(_iter_proj_table_items(proj))
    tmap = dict(tables)
    all_keys: set[bytes] = set()

    for lt, items in tables:
        ks = [k for k, _, _, _ in items]
        if len(ks) != len(set(ks)):
            errors.append(f"Claves duplicadas en tabla {lt.decode('latin1')}")
        if lt != b"EVTS" and ks != sorted(ks):
            errors.append(f"Tabla {lt.decode('latin1')} desordenada")
        all_keys.update(ks)

    bnki_exp = -1
    for cid, ps, pe, _ in iter_riff_chunks(proj, 0, len(proj)):
        if cid == b"BNKI" and pe - ps >= 28:
            bnki_exp = struct.unpack_from("<I", proj, ps + 24)[0]
            break

    ev_items = tmap.get(b"EVTS", [])
    ev_keys = [k for k, _, _, _ in ev_items]
    ev_set = set(ev_keys)
    if not (0 <= bnki_exp <= len(ev_keys)):
        errors.append(f"BNKI_exp={bnki_exp} fuera de rango (EVTS={len(ev_keys)})")
    else:
        exp_slice = ev_keys[:bnki_exp]
        nst_slice = ev_keys[bnki_exp:]
        if exp_slice != sorted(exp_slice):
            errors.append("Partición exportada EVTS[0:BNKI_exp] no ordenada")
        if nst_slice != sorted(nst_slice):
            errors.append("Partición anidada EVTS[BNKI_exp:] no ordenada")
        if expected_exported_guids:
            for g in expected_exported_guids:
                idx = bisect.bisect_left(exp_slice, g)
                if idx >= len(exp_slice) or exp_slice[idx] != g:
                    errors.append(f"Evento exportado no alcanzable por búsqueda binaria: {bytes_to_guid_str(g)}")

    # Verificar referencias críticas EVTB -> TLN / IBS / MBS
    tln_set = {k for k, _, _, _ in tmap.get(b"TLNS", ())}
    ibs_set = {k for k, _, _, _ in tmap.get(b"IBSS", ())}
    mbs_set = {k for k, _, _, _ in tmap.get(b"MBSS", ())}
    for k, _, _, bpay in ev_items:
        if len(bpay) >= 80:
            for nm, ref, valid_set in (
                ("TLN", bytes(bpay[32:48]), tln_set),
                ("IBS", bytes(bpay[48:64]), ibs_set),
                ("MBS", bytes(bpay[64:80]), mbs_set),
            ):
                if ref != b"\x00" * 16 and ref not in valid_set:
                    errors.append(f"EVTB {bytes_to_guid_str(k)} referencia {nm} inexistente")

    # Verificar EVIB -> target EVTS y orden de prop_id en EVIN -> LIST:PRPS
    for k, full, _, bpay in tmap.get(b"EVIS", ()):
        if len(bpay) >= 32 and bytes(bpay[16:32]) not in ev_set:
            errors.append(f"EVIB {bytes_to_guid_str(k)} apunta a evento inexistente")
        for sc, sps, spe, _ in iter_riff_chunks(full, 12, len(full)):
            if sc == b"LIST" and full[sps : sps + 4] == b"PRPS":
                pids = [
                    struct.unpack_from("<I", full, pps)[0]
                    for pc, pps, ppe, _ in iter_riff_chunks(full, sps + 4, spe)
                    if pc == b"PROP" and ppe - pps >= 4
                ]
                if pids != sorted(pids):
                    errors.append(f"EVIN {bytes_to_guid_str(k)} tiene PRPS desordenados por prop_id")

    # Verificar límites de sub-bancos/muestras en WAVS y alineación 32B
    try:
        parsed = FmodBank(raw=bank_bytes)
        sb_counts = [len(sb.samples) for sb in parsed.subbanks]
        for k, _, _, bpay in tmap.get(b"WAVS", ()):
            if len(bpay) >= 30:
                _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", bpay, 16)
                if not (0 <= sb_i < len(sb_counts)) or not (0 <= smp_i < sb_counts[sb_i]):
                    errors.append(f"WAV {bytes_to_guid_str(k)} fuera de rango (sb={sb_i}, smp={smp_i})")
    except Exception as exc:
        errors.append(f"Error al re-parsear FmodBank: {exc}")

    return errors


# ============================================================================
# Injerto completo de eventos nativos AMS2 con audio FSB5 (Starter / Starting / Transmisión)
# ============================================================================

AMS2_STARTUP_BUS_BYTES = guid_str_to_bytes("{ffefac1b-0333-46ad-9d94-c7ae1494451f}")


def _inject_clean_silent_events_in_proj(
    proj_buf: bytes,
    silent_guids: list[bytes],
    bus_guid: bytes,
    seed_bytes: bytes,
) -> bytes:
    """Inserta eventos silenciosos estructuralmente íntegros (EVNT + TMLN + IBUS + MBUS -> 0 punteros colgantes)."""
    if not silent_guids:
        return proj_buf
    tables = list(_iter_proj_table_items(proj_buf))
    existing_evts = {k for lt, items in tables if lt == b"EVTS" for k, _, _, _ in items}
    to_add = [g for g in dict.fromkeys(silent_guids) if g not in existing_evts]
    if not to_add:
        return proj_buf

    reserved = set(collect_local_proj_guids(proj_buf)) | set(to_add)
    extra_by_tbl: dict[bytes, list[tuple[bytes, bytes]]] = {}
    bus_pay = bytes.fromhex("00030000000100010008000000000000000000ffffff7f02000000")
    ibus_bus_pay = bytes.fromhex("01030000000100010008000000000000000000ffffff7f00000000")

    for sg in to_add:
        s_tln = _alloc_unique_guid(reserved, seed_bytes, b"sil_tln:" + sg)
        s_ibs = _alloc_unique_guid(reserved, seed_bytes, b"sil_ibs:" + sg)
        s_mbs = _alloc_unique_guid(reserved, seed_bytes, b"sil_mbs:" + sg)
        tlnb_pay = s_tln + sg + struct.pack("<HHHHH", 1, 1, 1, 0, 1)
        ibsb_pay = s_ibs + b"\x18\x00" + bus_guid + struct.pack("<Ii", 3, -1)
        mbsb_pay = s_mbs + b"\x18\x00" + (b"\x00" * 16) + struct.pack("<Ii", 3, -1)
        evtb_pay = (
            sg
            + (b"\x00" * 16)
            + s_tln
            + s_ibs
            + s_mbs
            + struct.pack("<iIBI", 1, 0, 0, 1)
            + struct.pack("<H", 1)
            + struct.pack("<ffII", 0.0, 0.0, 1, 0)
        )
        extra_by_tbl.setdefault(b"EVTS", []).append(
            (sg, _riff_list(b"EVNT", _riff_chunk(b"EVTB", evtb_pay) + _riff_list(b"PRPS", b"")))
        )
        extra_by_tbl.setdefault(b"TLNS", []).append(
            (
                s_tln,
                _riff_list(
                    b"TMLN",
                    _riff_chunk(b"TLNB", tlnb_pay) + _riff_list(b"TRNS", b"") + _riff_chunk(b"CTRO", struct.pack("<H", 1)),
                ),
            )
        )
        extra_by_tbl.setdefault(b"IBSS", []).append(
            (
                s_ibs,
                _riff_list(b"IBUS", _riff_chunk(b"IBSB", ibsb_pay) + _riff_chunk(b"BUS ", ibus_bus_pay) + _riff_list(b"PRPS", b"")),
            )
        )
        extra_by_tbl.setdefault(b"MBSS", []).append(
            (
                s_mbs,
                _riff_list(b"MBUS", _riff_chunk(b"MBSB", mbsb_pay) + _riff_chunk(b"BUS ", bus_pay) + _riff_list(b"PRPS", b"")),
            )
        )

    existing_map = {lt: items for lt, items in tables}
    out = bytearray()
    for cid, ps, pe, pos in iter_riff_chunks(proj_buf, 0, len(proj_buf)):
        pad_end = pe + ((pe - ps) & 1)
        if cid != b"LIST":
            out += proj_buf[pos:pad_end]
            continue
        lt = bytes(proj_buf[ps : ps + 4])
        if lt not in extra_by_tbl:
            out += proj_buf[pos:pad_end]
            continue
        comb: dict[bytes, bytes] = {k: f for k, f, _, _ in existing_map.get(lt, ())}
        for k, f in extra_by_tbl[lt]:
            comb[k] = f
        comb_sorted = sorted(comb.items(), key=lambda x: x[0])
        lcnt_ch = _riff_chunk(b"LCNT", struct.pack("<I", len(comb_sorted)))
        out += _riff_chunk(b"LIST", lt + lcnt_ch + b"".join(f for _, f in comb_sorted))

    evib_targets = _collect_evib_targets(bytes(out))
    evts_keys_all = existing_evts | set(to_add)
    exported_evs = {g for g in evts_keys_all if g not in evib_targets} | set(to_add)
    return bytes(_sync_evts_partition_and_bnki(out, exported_evs))


_AMS2_MASTER_STRINGS_CACHE: dict[str, dict[str, str]] = {}


def _load_ams2_master_strings_cached(ams2_root: Path | None) -> dict[str, str]:
    """Lee y cachea Audio/MasterBank.strings.bank de AMS2 (guid_str_lower -> path)."""
    eff_root = Path(ams2_root) if ams2_root else None
    if eff_root is None:
        try:
            from .vanilla_manager import detect_ams2_root

            eff_root = detect_ams2_root()
        except Exception:
            eff_root = None
    if eff_root is None:
        return {}
    candidates = [
        eff_root / "Audio" / "MasterBank.strings.bank",
        eff_root / "Audio" / "AMS2" / "MasterBank.strings.bank",
        eff_root / "Audio" / "Master Bank.strings.bank",
    ]
    ms_path = next((c for c in candidates if c.is_file()), None)
    if ms_path is None:
        return {}
    cache_key = str(ms_path.resolve()).lower()
    if cache_key in _AMS2_MASTER_STRINGS_CACHE:
        return _AMS2_MASTER_STRINGS_CACHE[cache_key]
    out: dict[str, str] = {}
    try:
        raw = ms_path.read_bytes()
        for cid, ps, pe, _ in iter_riff_chunks(raw, 12, len(raw)):
            if cid == b"LIST" and raw[ps : ps + 4] == b"PROJ":
                for sc, sps, spe, _ in iter_riff_chunks(raw, ps + 4, pe):
                    if sc == b"STDT":
                        parsed = parse_stdt_all_paths(raw[sps:spe])
                        out = {bytes_to_guid_str(g).lower(): p for g, p in parsed.items()}
                        break
    except Exception:
        out = {}
    _AMS2_MASTER_STRINGS_CACHE[cache_key] = out
    return out


def _find_native_starter_events_in_bank(
    native_bank: FmodBank,
    master_strings: dict[str, str] | None = None,
    preferred_paths: dict[str, str] | None = None,
    explicit_guids: list[bytes] | None = None,
    exclude_guids: set[bytes] | None = None,
) -> tuple[list[bytes], dict[bytes, str], dict[bytes, str]]:
    """Localiza los eventos nativos AMS2 Starter/Starting con audio WAVS válido dentro de un .bank.

    Retorna `(ordered_guids, guid_to_role, guid_to_path)` donde `guid_to_role[g]` es 'Starter' o 'Starting'.
    """
    nat_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(native_bank.raw, 12, len(native_bank.raw)):
        if cid == b"LIST" and native_bank.raw[ps : ps + 4] == b"PROJ":
            nat_proj = native_bank.raw[ps + 4 : pe]
            break
    if not nat_proj:
        return [], {}, {}

    nat_tables = list(_iter_proj_table_items(nat_proj))
    tmap = dict(nat_tables)
    ev_items = tmap.get(b"EVTS", [])
    if not ev_items:
        return [], {}, {}

    exported_ev_guids = _extract_bnki_exported_event_guids(nat_proj, ev_items)
    if not exported_ev_guids:
        exported_ev_guids = {k for k, _, _, _ in ev_items}
    excluded = exclude_guids or set()

    # Resolver bus padre de cada evento (EVTB[48:64] -> IBSB[18:34])
    ibs_parent: dict[bytes, bytes] = {}
    for k, _, _, bpay in tmap.get(b"IBSS", ()):
        if len(bpay) >= 34:
            ibs_parent[k] = bytes(bpay[18:34])
    ev_parent_bus: dict[bytes, bytes] = {}
    for k, _, _, bpay in ev_items:
        if len(bpay) >= 64:
            ev_parent_bus[k] = ibs_parent.get(bytes(bpay[48:64]), b"\x00" * 16)

    # Mapa GUID -> path combinando STDT embebido y Master Bank.strings.bank
    stdt_paths = parse_stdt_all_paths(native_bank.stdt_bytes) if native_bank.stdt_bytes else {}
    ms = master_strings or {}
    ev_path_by_g: dict[bytes, str] = {}
    path_low_to_g: dict[str, bytes] = {}
    for g in exported_ev_guids:
        gs_low = bytes_to_guid_str(g).lower()
        ep = stdt_paths.get(g) or ms.get(gs_low, "")
        if not ep:
            ev_obj = next((e for e in native_bank.events if e.guid_bytes == g), None)
            if ev_obj and ev_obj.path:
                ep = ev_obj.path
        if ep:
            ev_path_by_g[g] = ep
            path_low_to_g[ep.lower()] = g

    ev_role_obj: dict[bytes, str] = {e.guid_bytes: (e.role or "") for e in native_bank.events}
    sb_counts = [len(sb.samples) for sb in native_bank.subbanks]
    slot_names_loc: dict[tuple[int, int], str] = {
        (sl.subbank_index, sl.sample_index): sl.name.lower() for sl in native_bank.slots
    }
    wav_items = tmap.get(b"WAVS", [])
    ev_wav_names_cache: dict[bytes, list[str]] = {}

    def _ev_wav_names(eg: bytes) -> list[str]:
        if eg in ev_wav_names_cache:
            return ev_wav_names_cache[eg]
        if eg not in exported_ev_guids and eg not in ev_parent_bus:
            ev_wav_names_cache[eg] = []
            return []
        clos = _extract_event_subgraph_closure(nat_tables, {eg})
        out_n: list[str] = []
        for wk, _, _, bpay in wav_items:
            if wk in clos and len(bpay) >= 30:
                _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", bpay, 16)
                if 0 <= sb_i < len(sb_counts) and 0 <= smp_i < sb_counts[sb_i]:
                    out_n.append(slot_names_loc.get((int(sb_i), int(smp_i)), ""))
        ev_wav_names_cache[eg] = out_n
        return out_n

    def _has_valid_audio(eg: bytes) -> bool:
        return bool(_ev_wav_names(eg))

    def _infer_st_role(eg: bytes, default_role: str = "Starter") -> str:
        pl = ev_path_by_g.get(eg, "").lower()
        if "starting" in pl:
            return "Starting"
        if "starter" in pl or "crank" in pl:
            return "Starter"
        sn = " ".join(_ev_wav_names(eg))
        if "starting" in sn and "starter" not in sn:
            return "Starting"
        if "starter" in sn and "starting" not in sn:
            return "Starter"
        ro = ev_role_obj.get(eg, "")
        if ro in ("Starter", "Starting"):
            return ro
        return default_role

    selected: list[bytes] = []
    guid_to_role: dict[bytes, str] = {}
    guid_to_path: dict[bytes, str] = {}

    if preferred_paths:
        # Modo Mod (.csdbin): buscar exactamente los eventos pedidos por starter / starting
        for role_k in ("Starter", "Starting"):
            want_p = (preferred_paths.get(role_k) or "").strip()
            if not want_p:
                continue
            want_low = want_p.lower()
            matched_g = path_low_to_g.get(want_low)
            if matched_g is None:
                leaf = want_low.split("/")[-1]
                for g, ep in ev_path_by_g.items():
                    if ep.lower().endswith("/" + leaf) or ep.lower() == leaf:
                        matched_g = g
                        break
            if matched_g is not None and matched_g not in excluded and _has_valid_audio(matched_g):
                if matched_g not in guid_to_role:
                    selected.append(matched_g)
                    guid_to_role[matched_g] = role_k
                    guid_to_path[matched_g] = ev_path_by_g.get(matched_g, want_p)

        # Fallback por si faltó Starter o Starting en preferred_paths o no coincidió por ruta exacta
        have_roles = set(guid_to_role.values())
        if len(have_roles) < 2:
            for g in sorted(exported_ev_guids):
                if g in excluded or g in guid_to_role:
                    continue
                pl = ev_path_by_g.get(g, "").lower()
                is_st_bus = ev_parent_bus.get(g) == AMS2_STARTUP_BUS_BYTES
                is_st_name = any(k in pl for k in ("starter", "starting", "startup")) or ev_role_obj.get(g) in ("Starter", "Starting")
                if not (is_st_bus or is_st_name):
                    continue
                if not _has_valid_audio(g):
                    continue
                inferred = _infer_st_role(g, "Starter" if "Starter" not in have_roles else "Starting")
                if inferred not in have_roles:
                    selected.append(g)
                    guid_to_role[g] = inferred
                    guid_to_path[g] = ev_path_by_g.get(g, "")
                    have_roles.add(inferred)
                    if len(have_roles) >= 2:
                        break
        return selected, guid_to_role, guid_to_path

    # Modo Vanilla in-place: conservar TODOS los eventos Starter/Starting nativos del banco (ej. 963_Starter, 963_Starter_Old, 963_Starting)
    cands_ordered: list[bytes] = []
    seen_c: set[bytes] = set()
    for g in (explicit_guids or ()):
        if g not in excluded and g not in seen_c and g in ev_parent_bus:
            seen_c.add(g)
            cands_ordered.append(g)
    for g in sorted(exported_ev_guids):
        if g in excluded or g in seen_c:
            continue
        pl = ev_path_by_g.get(g, "").lower()
        is_st_bus = ev_parent_bus.get(g) == AMS2_STARTUP_BUS_BYTES
        is_st_name = any(k in pl for k in ("starter", "starting", "startup")) or ev_role_obj.get(g) in ("Starter", "Starting")
        if is_st_bus or is_st_name:
            seen_c.add(g)
            cands_ordered.append(g)

    for g in cands_ordered:
        if not _has_valid_audio(g):
            continue
        selected.append(g)
        guid_to_role[g] = _infer_st_role(g, "Starter" if "Starter" not in guid_to_role.values() else "Starting")
        guid_to_path[g] = ev_path_by_g.get(g, "")

    return selected, guid_to_role, guid_to_path


def graft_native_events_with_audio_into_bank(
    target_bank_bytes: bytes,
    native_bank: FmodBank,
    native_event_guids: list[bytes],
    event_paths_map: dict[bytes, str] | None = None,
    remap_root_guids: bool = False,
    output_codec_mode: int = FSB5_MODE_FADPCM,
    mod_identifier: str = "custom_mod",
    seed_tag: str = "native_graft",
) -> tuple[bytes, dict[bytes, bytes], dict[bytes, str]]:
    """Injerta eventos nativos completos de AMS2 (subgrafo PROJ íntegro + audio FSB5 compacto) en un .bank destino.

    - Preserva 100% del diseño FMOD original del evento nativo (TMLN, PMLS, PRMS, MBUS, GBUS, BEFX,
      CTRS, CRVS, MPGS, MODS, WAIS, MUIS, EVIS, WAVS y enrutamiento IBSB al bus original de AMS2).
    - Empaqueta únicamente las muestras referenciadas por los eventos injertados en un sub-banco FSB5
      compacto (FADPCM por defecto; passthrough intacto para slots Vorbis/no decodificables).
    - Si `remap_root_guids=False` (Vanilla in-place), conserva los GUIDs raíz originales exactos.
    - Si `remap_root_guids=True` (Mod standalone), genera GUIDs únicos deterministas con 0 colisiones.

    Retorna `(new_bank_bytes, orig_to_final_root_guid, final_guid_to_path)`.
    """
    if not native_event_guids:
        return target_bank_bytes, {}, {}

    nat_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(native_bank.raw, 12, len(native_bank.raw)):
        if cid == b"LIST" and native_bank.raw[ps : ps + 4] == b"PROJ":
            nat_proj = native_bank.raw[ps + 4 : pe]
            break
    if not nat_proj:
        return target_bank_bytes, {}, {}

    native_tables = list(_iter_proj_table_items(nat_proj))
    nat_evts_all = {k for lt, items in native_tables if lt == b"EVTS" for k, _, _, _ in items}
    valid_roots = [g for g in dict.fromkeys(native_event_guids) if g in nat_evts_all]
    if not valid_roots:
        return target_bank_bytes, {}, {}

    target_bank = FmodBank(raw=target_bank_bytes)
    fmt_payload = struct.pack("<II", target_bank.file_version or 0x50, target_bank.compat_version or 0x46)
    target_proj = b""
    for cid, ps, pe, _ in iter_riff_chunks(target_bank.raw, 12, len(target_bank.raw)):
        if cid == b"FMT ":
            fmt_payload = target_bank.raw[ps:pe]
        elif cid == b"LIST" and target_bank.raw[ps : ps + 4] == b"PROJ":
            target_proj = target_bank.raw[ps + 4 : pe]
    if not target_proj:
        return target_bank_bytes, {}, {}

    target_tables = list(_iter_proj_table_items(target_proj))
    closure = _extract_event_subgraph_closure(native_tables, set(valid_roots))
    if not closure:
        return target_bank_bytes, {}, {}

    seed_bytes = f"{seed_tag}:{mod_identifier}".encode("utf-8")
    reserved_guids: set[bytes] = set(collect_local_proj_guids(target_proj))
    if target_bank.bank_guid:
        reserved_guids.add(target_bank.bank_guid)
    if not remap_root_guids:
        reserved_guids.update(valid_roots)

    remap_nat: dict[bytes, bytes] = {}
    root_set = set(valid_roots)
    for g in sorted(closure):
        if remap_root_guids:
            remap_nat[g] = _alloc_unique_guid(reserved_guids, seed_bytes, b"nat_mod:" + g)
        else:
            if g in root_set:
                remap_nat[g] = g
            elif g not in reserved_guids:
                remap_nat[g] = g
                reserved_guids.add(g)
            else:
                remap_nat[g] = _alloc_unique_guid(reserved_guids, seed_bytes, b"nat_int:" + g)

    # Extraer y compactar únicamente las muestras FSB5 referenciadas por el subgrafo nativo
    slot_lookup: dict[tuple[int, int], FsbSampleSlot] = {
        (sl.subbank_index, sl.sample_index): sl for sl in native_bank.slots
    }
    needed_locs_ordered: list[tuple[int, int]] = []
    seen_locs: set[tuple[int, int]] = set()
    for lt, items in native_tables:
        if lt == b"WAVS":
            for k, _full, _base, bpay in items:
                if k in closure and len(bpay) >= 30:
                    _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", bpay, 16)
                    loc = (int(sb_i), int(smp_i))
                    if loc in slot_lookup and loc not in seen_locs:
                        seen_locs.add(loc)
                        needed_locs_ordered.append(loc)

    final_fsb_blobs: list[bytes] = [bytes(sb.raw_fsb_bytes) for sb in target_bank.subbanks]
    loc_remap: dict[tuple[int, int], tuple[int, int]] = {}
    decodable_locs: list[tuple[tuple[int, int], FsbSampleSlot]] = []
    passthrough_locs: list[tuple[int, int]] = []

    for loc in needed_locs_ordered:
        sl = slot_lookup[loc]
        if sl.can_decode_audio:
            decodable_locs.append((loc, sl))
        else:
            passthrough_locs.append(loc)

    if decodable_locs:
        fsb_samples = [
            FsbBuildSample(
                name=sl.name,
                audio=native_bank.decode_slot_to_buffer(sl),
                peak_override=sl.peak_volume,
            )
            for _loc, sl in decodable_locs
        ]
        eff_codec = output_codec_mode if output_codec_mode in (FSB5_MODE_PCM16, FSB5_MODE_FADPCM) else FSB5_MODE_FADPCM
        compact_fsb = build_fsb5_container(fsb_samples, mode=eff_codec)
        compact_sb_idx = len(final_fsb_blobs)
        final_fsb_blobs.append(compact_fsb)
        for new_smp_i, (orig_loc, _sl) in enumerate(decodable_locs):
            loc_remap[orig_loc] = (compact_sb_idx, new_smp_i)

    if passthrough_locs:
        pt_sb_map: dict[int, int] = {}
        for orig_sb, orig_smp in passthrough_locs:
            if orig_sb not in pt_sb_map:
                pt_sb_map[orig_sb] = len(final_fsb_blobs)
                final_fsb_blobs.append(bytes(native_bank.subbanks[orig_sb].raw_fsb_bytes))
            loc_remap[(orig_sb, orig_smp)] = (pt_sb_map[orig_sb], orig_smp)

    # Remapear chunks del subgrafo nativo y redirigir WAVS al nuevo sub-banco
    graft_items_by_table: dict[bytes, list[tuple[bytes, bytes]]] = {}
    for lt, items in native_tables:
        out_list: list[tuple[bytes, bytes]] = []
        for k, full, _base, bpay in items:
            if k not in closure:
                continue
            new_k = remap_nat.get(k, k)
            new_full = _remap_chunk_guids_leaves(full, remap_nat)
            if lt == b"WAVS" and len(bpay) >= 30:
                _, old_sb, _, old_smp, _ = struct.unpack_from("<HHHII", bpay, 16)
                dst_loc = loc_remap.get((int(old_sb), int(old_smp)))
                if dst_loc is not None:
                    new_full = _patch_wav_chunk_location(new_full, dst_loc[0], dst_loc[1])
            out_list.append((new_k, new_full))
        if out_list:
            graft_items_by_table[lt] = out_list

    # Fusionar tablas en target_proj preservando orden canónico FMOD
    target_items_map = {lt: items for lt, items in target_tables}
    merged_proj_parts = bytearray()
    seen_tables: set[bytes] = set()
    for cid, ps, pe, pos in iter_riff_chunks(target_proj, 0, len(target_proj)):
        pad_end = pe + ((pe - ps) & 1)
        if cid != b"LIST":
            merged_proj_parts += target_proj[pos:pad_end]
            continue
        lt = bytes(target_proj[ps : ps + 4])
        seen_tables.add(lt)
        combined_map: dict[bytes, bytes] = {k: f for k, f, _, _ in target_items_map.get(lt, ())}
        for k, f in graft_items_by_table.get(lt, ()):
            combined_map[k] = f
        combined_sorted = sorted(combined_map.items(), key=lambda x: x[0])
        if combined_sorted:
            lcnt_ch = _riff_chunk(b"LCNT", struct.pack("<I", len(combined_sorted)))
            body_lt = lt + lcnt_ch + b"".join(f for _, f in combined_sorted)
        else:
            body_lt = lt
        merged_proj_parts += _riff_chunk(b"LIST", body_lt)

    # Anexar cualquier tabla extra presente en el subgrafo nativo y ausente en target_proj
    for lt, items in graft_items_by_table.items():
        if lt in seen_tables or not items:
            continue
        comb_sorted = sorted({k: f for k, f in items}.items(), key=lambda x: x[0])
        lcnt_ch = _riff_chunk(b"LCNT", struct.pack("<I", len(comb_sorted)))
        merged_proj_parts += _riff_chunk(b"LIST", lt + lcnt_ch + b"".join(f for _, f in comb_sorted))

    # Sincronizar partición EVTS [0..BNKI_exp) (exportados) + [BNKI_exp..N) (sub-eventos EVIB)
    all_evib_targets = _collect_evib_targets(bytes(merged_proj_parts))
    target_evts_list = target_items_map.get(b"EVTS", [])
    target_evib_targets = _collect_evib_targets(target_proj)
    target_bnki_exp = _extract_bnki_exported_event_guids(target_proj, target_evts_list)
    target_non_evib = {k for k, _, _, _ in target_evts_list if k not in target_evib_targets}
    grafted_root_final = {remap_nat[g] for g in valid_roots}
    exported_final = (target_bnki_exp - all_evib_targets) | target_non_evib | grafted_root_final
    patched_proj_body = bytes(_sync_evts_partition_and_bnki(merged_proj_parts, exported_final))

    # Actualizar STDT preservando todas las rutas existentes + registrando los eventos injertados
    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_mod"
    existing_stdt = parse_stdt_all_paths(target_bank.stdt_bytes) if target_bank.stdt_bytes else {}
    stdt_entries: list[tuple[str, bytes]] = []
    for g_ex, p_ex in existing_stdt.items():
        if g_ex not in grafted_root_final:
            stdt_entries.append((p_ex, g_ex))
    if not existing_stdt and target_bank.bank_guid:
        stdt_entries.append((f"bank:/{safe_mod}", target_bank.bank_guid))

    pmap = event_paths_map or {}
    used_p_low = {p.lower() for p, _ in stdt_entries}
    final_root_paths: dict[bytes, str] = {}
    for idx_r, orig_g in enumerate(valid_roots):
        fin_g = remap_nat[orig_g]
        cand_p = pmap.get(orig_g) or f"event:/AMS2/Vehicles/Mods/{safe_mod}/Native_{idx_r:02d}"
        if cand_p.lower() in used_p_low:
            k_dup = 2
            while f"{cand_p}_{k_dup}".lower() in used_p_low:
                k_dup += 1
            cand_p = f"{cand_p}_{k_dup}"
        used_p_low.add(cand_p.lower())
        stdt_entries.append((cand_p, fin_g))
        final_root_paths[fin_g] = cand_p

    new_stdt_payload = build_stdt_chunk(stdt_entries) if stdt_entries else None

    # Recalcular siempre sndh_ver con el número actualizado de sub-bancos FSB5
    sndh_ver = 0x00080000 | (2 * len(final_fsb_blobs) + 1)
    dummy_sndh = struct.pack("<I", sndh_ver) + b"\x00" * (8 * len(final_fsb_blobs))
    temp_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, dummy_sndh, new_stdt_payload)

    fmt_chunk_bytes = b"FMT " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if len(fmt_payload) & 1:
        fmt_chunk_bytes += b"\x00"

    proj_list_size = 4 + len(temp_proj_body)
    proj_total_bytes = 8 + proj_list_size + (proj_list_size & 1)

    cur_file_pos = 12 + len(fmt_chunk_bytes) + proj_total_bytes
    sndh_entries: list[tuple[int, int]] = []
    snd_chunks_built: list[bytes] = []

    for fsb_blob in final_fsb_blobs:
        snd_payload_start = cur_file_pos + 8
        pad_before = (32 - (snd_payload_start % 32)) % 32
        fsb_pos = snd_payload_start + pad_before
        fsb_len = len(fsb_blob)
        snd_payload = (b"\x00" * pad_before) + bytes(fsb_blob)
        snd_chunk = b"SND " + struct.pack("<I", len(snd_payload)) + snd_payload
        if len(snd_payload) & 1:
            snd_chunk += b"\x00"
        sndh_entries.append((fsb_pos, fsb_len))
        snd_chunks_built.append(snd_chunk)
        cur_file_pos += len(snd_chunk)

    real_sndh = struct.pack("<I", sndh_ver) + b"".join(
        struct.pack("<II", fpos, flen) for fpos, flen in sndh_entries
    )
    final_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, real_sndh, new_stdt_payload)
    assert len(final_proj_body) == len(temp_proj_body), "Tamaño inesperado al actualizar SNDH/STDT tras injerto nativo"

    proj_chunk_bytes = (
        b"LIST"
        + struct.pack("<I", 4 + len(final_proj_body))
        + b"PROJ"
        + final_proj_body
        + (b"\x00" if (len(final_proj_body) & 1) else b"")
    )
    body = fmt_chunk_bytes + proj_chunk_bytes + b"".join(snd_chunks_built)
    out_bytes = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body
    root_guid_map = {orig_g: remap_nat[orig_g] for orig_g in valid_roots}
    return out_bytes, root_guid_map, final_root_paths


def _resolve_native_starter_source_for_csdbin(
    csd_doc: CsdbinDocument | None,
    orig_engine_bank: str,
    orig_starter_path: str,
    orig_starting_path: str,
    ams2_root: Path | None = None,
    backup_dir: Path | None = None,
    fallback_donor: FmodBank | None = None,
    extra_search_roots: list[Path] | None = None,
) -> tuple[FmodBank | None, list[bytes], dict[bytes, str], dict[bytes, str]]:
    """Resuelve el banco AMS2 original y sus eventos Starter/Starting (con audio válido) para un Mod .csdbin."""
    eff_ams2 = Path(ams2_root) if ams2_root else None
    if eff_ams2 is None:
        try:
            from .vanilla_manager import detect_ams2_root

            eff_ams2 = detect_ams2_root()
        except Exception:
            eff_ams2 = None

    ms = _load_ams2_master_strings_cached(eff_ams2)
    pref_paths: dict[str, str] = {}
    if orig_starter_path:
        pref_paths["Starter"] = orig_starter_path
    if orig_starting_path:
        pref_paths["Starting"] = orig_starting_path

    cand_paths: list[Path] = []
    seen_p: set[str] = set()

    def _push_cand(p: Path | None) -> None:
        if p is None:
            return
        try:
            if not p.is_file():
                return
            key = str(p.resolve()).lower()
        except Exception:
            return
        if key not in seen_p:
            seen_p.add(key)
            cand_paths.append(p)

    clean_eb = orig_engine_bank.replace("\\", "/").removesuffix(".bank").strip("/")
    if clean_eb:
        eb_fname = Path(clean_eb).name + ".bank"
        # 1. Priorizar respaldos limpios de Steam si existen (evita leer bancos vanilla modificados sin starter)
        b_dirs: list[Path] = []
        if backup_dir and Path(backup_dir).is_dir():
            b_dirs.append(Path(backup_dir))
        if eff_ams2:
            ams2_bank_p = eff_ams2 / f"{clean_eb}.bank"
            _push_cand(ams2_bank_p.parent / f"{ams2_bank_p.name}.orig")
        for bdir in b_dirs:
            for hit in bdir.rglob(eb_fname):
                _push_cand(hit)
        # 2. Banco en la instalación de AMS2 y carpetas de búsqueda extra (p.ej. re-exportación en output_root)
        if eff_ams2:
            _push_cand(eff_ams2 / f"{clean_eb}.bank")
        for eroot in (extra_search_roots or ()):
            _push_cand(Path(eroot) / f"{clean_eb}.bank")
        # 3. Banco relativo a la carpeta raíz del mod (.csdbin en <ModRoot>/Vehicles/<car>/<car>.csdbin)
        if csd_doc is not None and csd_doc.source_path is not None:
            cur = csd_doc.source_path.parent
            for _ in range(4):
                _push_cand(cur / f"{clean_eb}.bank")
                if cur.parent == cur:
                    break
                cur = cur.parent

    # 4. Si el .csdbin tiene ruta starter/starting en Sort/<Letra>/<Marca>/..., localizar el banco vanilla allí
    if eff_ams2 and (orig_starter_path or orig_starting_path):
        sort_root = eff_ams2 / "Audio" / "AMS2" / "Vehicles" / "Sort"
        for sp in (orig_starter_path, orig_starting_path):
            if not sp:
                continue
            sp_norm = sp.replace("\\", "/")
            marker = "/Vehicles/Sort/"
            idx_m = sp_norm.lower().find(marker.lower())
            if idx_m >= 0 and sort_root.is_dir():
                tail_parts = [pt for pt in sp_norm[idx_m + len(marker) :].split("/") if pt]
                if len(tail_parts) >= 2:
                    brand_dir = sort_root / tail_parts[0] / tail_parts[1]
                    if brand_dir.is_dir():
                        for bp in sorted(brand_dir.rglob("*.bank")):
                            if bp.stem.lower().endswith("_ai"):
                                continue
                            if backup_dir and Path(backup_dir).is_dir():
                                for hit in Path(backup_dir).rglob(bp.name):
                                    _push_cand(hit)
                            _push_cand(bp.parent / f"{bp.name}.orig")
                            _push_cand(bp)

    for cp in cand_paths:
        try:
            nb = FmodBank(cp)
            sel, g2r, g2p = _find_native_starter_events_in_bank(
                nb,
                master_strings=ms,
                preferred_paths=(pref_paths if pref_paths else None),
            )
            if sel:
                return nb, sel, g2r, g2p
        except Exception:
            continue

    if fallback_donor is not None:
        try:
            sel, g2r, g2p = _find_native_starter_events_in_bank(
                fallback_donor,
                master_strings=ms,
                preferred_paths=(pref_paths if pref_paths else None),
            )
            if sel:
                return fallback_donor, sel, g2r, g2p
        except Exception:
            pass

    return None, [], {}, {}


def build_direct_ported_bank_from_ac(
    ac_bank: FmodBank,
    mod_identifier: str = "custom_mod",
    standalone_guids: bool = True,
    rpm_bands: list[RpmBandConfig] | None = None,
    target_role_guids: dict[str, str | bytes] | None = None,
    target_role_paths: dict[str, str] | None = None,
    target_bank_guid: str | bytes | None = None,
    target_param_names: set[str] | list[str] | None = None,
    target_ctrl_params: set[str] | list[str] | None = None,
    target_role_buses: dict[str, str | bytes] | None = None,
    is_ai_bank: bool = False,
    mute_secondary_pops: bool = False,
    native_bank_path: Path | str | None = None,
    invert_spatial_cone: bool = True,
    mappings: list[SlotMappingConfig] | None = None,
    output_codec_mode: int | None = None,
    prune_unused_samples: bool = False,
    cached_fsb_blobs: list[bytes] | None = None,
    extra_role_variants: dict[str, list[tuple[str, str]] | list[tuple[str, str, str]]] | None = None,
    prefer_throttle_param: bool = False,
    target_max_rpm: float | None = None,
    library: SourceAudioLibrary | None = None,
    active_scope: str = "full",
    explicit_ac_int_guid: bytes | None = None,
    explicit_ac_ext_guid: bytes | None = None,
) -> PortedBankResult:
    """Construye un nuevo archivo .bank para Automobilista 2 directamente desde un banco .bank de Assetto Corsa.

    Preserva el 100% del diseño de sonido original de Assetto Corsa (curvas, timelines, filtros,
    multi-instrumentos, pitch automations y efectos extras) sin verse condicionado ni degradado
    por un banco donante de AMS2.
    Soporta:
      1. Parcheo binario in-place del parámetro de motor 'rpms' -> 'rpm' de AMS2 en chunks PRMB.
      2. Re-enrutamiento de buses de instancia (IBSB) hacia los buses correspondientes en AMS2
         ({c1986fb5-b329-4565-8342-b223857006d5} Exterior y {a8959bf6-0b54-43ea-9d9a-06a8831c40d4} Cockpit).
      3. Inyección de starter de Assetto Corsa si existe, o preservación/inyección de los chunks
         nativos de Starter/Starting de AMS2 desde el banco base del vehículo para evitar bucles mudos.
      4. Asignación 1:1 exacta de GUIDs de eventos (EngineInt, EngineExt) sin clonados artificiales
         que rompan la máquina de estados de FMOD.
      5. Inserción de tabla de rutas FMOD RadixTree (STDT).
      6. Re-ensamblado con alineación de 32 bytes para sub-bancos FSB5 (SNDH y chunks SND ).
    """
    EXT_BUS = guid_str_to_bytes("{c1986fb5-b329-4565-8342-b223857006d5}")
    INT_BUS = guid_str_to_bytes("{a8959bf6-0b54-43ea-9d9a-06a8831c40d4}")
    MUTE_BUS = guid_str_to_bytes("{00000000-0000-0000-0000-000000000000}")

    # 1. Asociar cada evento de AC a su bus correspondiente en AMS2 (Exterior o Cockpit)
    ibg_to_target_bus: dict[bytes, bytes] = {}
    for ev in ac_bank.events:
        if not ev.ibus_guid_bytes:
            continue
        p_low = (ev.path or "").lower()

        # Si se solicita silenciar los petardeos (POPS) en la exportación
        if mute_secondary_pops and (
            ev.role in ("BackfireExt", "BackfireInt", "Limiter")
            or any(k in p_low for k in ("backfire", "limiter", "tractioncontrol", "tc_ext", "tc_int"))
        ):
            ibg_to_target_bus[ev.ibus_guid_bytes] = MUTE_BUS
            continue

        is_ext = (
            ev.role in ("EngineExt", "GearshiftExt", "BackfireExt")
            or "_ext" in p_low
            or "ext_" in p_low
            or p_low.endswith("ext")
            or any(k in p_low for k in ("exterior", "rear", "external", "chase", "skid_ext", "tractioncontrol_ext"))
        )
        ibg_to_target_bus[ev.ibus_guid_bytes] = EXT_BUS if is_ext else INT_BUS

    # 2. Extraer FMT y PROJ originales del banco de AC
    fmt_payload = struct.pack("<II", ac_bank.file_version or 0x50, ac_bank.compat_version or 0x46)
    orig_proj_payload = b""
    for cid, ps, pe, _ in iter_riff_chunks(ac_bank.raw, 12, len(ac_bank.raw)):
        if cid == b"FMT ":
            fmt_payload = ac_bank.raw[ps:pe]
        elif cid == b"LIST" and ac_bank.raw[ps : ps + 4] == b"PROJ":
            orig_proj_payload = ac_bank.raw[ps + 4 : pe]

    # 3. Parcheo binario de PROJ: 'rpms' -> 'rpm' en PRMB e IBSB a buses de AMS2
    proj_buf = bytearray(_patch_prmb_and_ibsb_in_proj(orig_proj_payload, {"rpms": "rpm"}, ibg_to_target_bus))

    # Invertir cono espacial 3D únicamente en el banco del Jugador (AMS2 Jugador usa 0°=Frente/180°=Atrás,
    # opuesto a AC; en cambio _AI.bank —IA y Replay— ya usa 0°=Atrás/180°=Frente igual que AC).
    if invert_spatial_cone and not is_ai_bank:
        proj_buf = bytearray(invert_event_cone_curves_in_proj(bytes(proj_buf))[0])

    # 'throttle' de AC: AMS2 no lo alimenta en repeticiones / IA (usa 'load'). Renombrar a 'load' en bancos
    # estándar; si el banco objetivo opera con 'throttle' (prefer_throttle_param=True), conservarlo intacto.
    proj_buf = bytearray(
        _patch_throttle_params_in_proj(
            bytes(proj_buf),
            rename_to_load=True,
            keep_throttle=bool(prefer_throttle_param),
        )
    )

    p_ext_sel, p_int_sel = _select_primary_engine_events(ac_bank)
    # Clones profundos a generar tras el parcheo: (guid_origen_ya_remapeado, guid_destino, bus_destino | None)
    pending_clones: list[tuple[bytes, bytes, bytes | None]] = []
    # Eventos de IA a los que se aplica límite de instancias y bus de salida: guid_final -> bus | None
    ai_event_outputs: dict[bytes, bytes | None] = {}
    extra_player_clones_count = 0

    def _bus_for(role: str, default: str | None) -> bytes | None:
        raw_b = (target_role_buses or {}).get(role) or default
        if raw_b is None:
            return None
        return guid_str_to_bytes(raw_b) if isinstance(raw_b, str) else bytes(raw_b)

    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_mod"
    guid_map: dict[bytes, bytes] | None = None
    event_guid_map: dict[str, str] = {}
    role_to_event_guid: dict[str, str] = {}
    role_to_event_path: dict[str, str] = {}
    stdt_entries: list[tuple[str, bytes]] = []

    active_root_orig_evs: set[bytes] = set()
    evib_targets_orig = _collect_evib_targets(bytes(proj_buf)) if is_ai_bank else set()
    pending_native_starter_graft: tuple[FmodBank, list[bytes], dict[bytes, str]] | None = None
    pending_clean_silent_exhaust: list[bytes] = []
    pending_clean_silent_starters: list[bytes] = []

    if target_role_guids:
        # Seleccionar el evento primario de exterior e interior de AC
        def _is_engine_cand(ev: FmodEventInfo) -> bool:
            p_text = (ev.path or "").lower()
            if "engine_extra" in p_text:
                return False
            has_rpm = any(p[0].lower() in ("rpm", "rpms") for p in ev.parameters)
            return has_rpm and len(ev.linked_samples) >= 5

        ext_cands = [
            e for e in ac_bank.events
            if _is_engine_cand(e) and (e.role == "EngineExt" or any(k in (e.path or "").lower() for k in ("_ext", "ext_", "exterior", "chase")))
        ]
        ext_cands.sort(key=lambda x: -len(x.linked_samples))
        p_ext = (
            next((e for e in ac_bank.events if e.guid_bytes == explicit_ac_ext_guid), None)
            if explicit_ac_ext_guid
            else None
        ) or (ext_cands[0] if ext_cands else next((e for e in ac_bank.events if e.role == "EngineExt"), None))

        int_cands = [
            e for e in ac_bank.events
            if _is_engine_cand(e) and (e.role == "EngineInt" or any(k in (e.path or "").lower() for k in ("_int", "int_", "cockpit", "cabin")))
        ]
        int_cands.sort(key=lambda x: -len(x.linked_samples))
        p_int = (
            next((e for e in ac_bank.events if e.guid_bytes == explicit_ac_int_guid), None)
            if explicit_ac_int_guid
            else None
        ) or (int_cands[0] if int_cands else next((e for e in ac_bank.events if e.role == "EngineInt"), None))

        if p_ext and active_scope in ("full", "exterior"):
            active_root_orig_evs.add(p_ext.guid_bytes)
        if p_int and active_scope in ("full", "interior"):
            active_root_orig_evs.add(p_int.guid_bytes)

        if is_ai_bank:
            # Solo en _AI.bank: remapear TODOS los GUIDs locales internos respecto al Jugador para evitar
            # colisiones en FMOD Studio al descargar ambos bancos al fin de sesión, dejando intacta la ruta Jugador.
            used_guids: set[bytes] = {
                guid_str_to_bytes(g) if isinstance(g, str) else bytes(g)
                for g in target_role_guids.values()
            }
            if extra_role_variants:
                for _items in extra_role_variants.values():
                    for _it in _items:
                        used_guids.add(guid_str_to_bytes(_it[0]))
            if target_bank_guid:
                used_guids.add(guid_str_to_bytes(target_bank_guid) if isinstance(target_bank_guid, str) else bytes(target_bank_guid))

            local_sorted = collect_local_proj_guids(bytes(proj_buf))
            seed_tag = f"{safe_mod}:ai"
            guid_map = build_order_preserving_guid_map(local_sorted, seed_tag)
            reserved_pool = set(used_guids) | set(guid_map.values())
            fix_ctr = 0
            for orig_k, mapped_v in list(guid_map.items()):
                if mapped_v in used_guids:
                    while True:
                        cand = bytearray(hashlib.sha256(f"{seed_tag}:fix:{fix_ctr}".encode("utf-8") + orig_k).digest()[:16])
                        fix_ctr += 1
                        cand[7] = (cand[7] & 0x0F) | 0x40
                        cand[8] = (cand[8] & 0x3F) | 0x80
                        cand_b = bytes(cand)
                        if cand_b not in reserved_pool and cand_b != b"\x00" * 16:
                            guid_map[orig_k] = cand_b
                            reserved_pool.add(cand_b)
                            break

            assigned_orig_evs: set[bytes] = set()

            def _assign_ai_role(ac_ev: FmodEventInfo, target_g_bytes: bytes) -> None:
                if ac_ev.guid_bytes in evib_targets_orig or ac_ev.guid_bytes in assigned_orig_evs:
                    pending_clones.append((ac_ev.guid_bytes, target_g_bytes, None))
                else:
                    guid_map[ac_ev.guid_bytes] = target_g_bytes
                    assigned_orig_evs.add(ac_ev.guid_bytes)

            ai_default_bus = {
                "ai_engine_main": AMS2_AI_ENGINE_BUS,
                "ai_engine_live": AMS2_AI_ENGINE_BUS,
                "ai_engine_ext": None,
            }
            if p_ext:
                for ai_role in ("ai_engine_main", "ai_engine_live", "ai_engine_ext"):
                    if ai_role not in target_role_guids:
                        continue
                    ai_bg = guid_str_to_bytes(target_role_guids[ai_role])
                    if ai_bg in assigned_orig_evs:
                        continue
                    _assign_ai_role(p_ext, ai_bg)
                    if p_ext.guid_str not in event_guid_map:
                        event_guid_map[p_ext.guid_str] = bytes_to_guid_str(ai_bg)
                    used_guids.add(ai_bg)
                    ai_event_outputs[ai_bg] = _bus_for(ai_role, ai_default_bus[ai_role])
                    role_to_event_guid[ai_role] = bytes_to_guid_str(ai_bg)
                    ai_p = (target_role_paths or {}).get(ai_role) or f"event:/AMS2/Vehicles/{safe_mod}/{ai_role}"
                    role_to_event_path[ai_role] = ai_p
                    stdt_entries.append((ai_p, ai_bg))
            if "ai_engine_int" in target_role_guids and p_int:
                ai_int_bg = guid_str_to_bytes(target_role_guids["ai_engine_int"])
                _assign_ai_role(p_int, ai_int_bg)
                event_guid_map.setdefault(p_int.guid_str, bytes_to_guid_str(ai_int_bg))
                used_guids.add(ai_int_bg)
                ai_event_outputs[ai_int_bg] = _bus_for("ai_engine_int", None)
                role_to_event_guid["ai_engine_int"] = bytes_to_guid_str(ai_int_bg)
                ai_int_p = (target_role_paths or {}).get("ai_engine_int") or f"event:/AMS2/Vehicles/{safe_mod}/ai_engine_int"
                role_to_event_path["ai_engine_int"] = ai_int_p
                stdt_entries.append((ai_int_p, ai_int_bg))

            if extra_role_variants:
                assigned_ai_targets = set(ai_event_outputs.keys())
                for ai_role in ("ai_engine_main", "ai_engine_live", "ai_engine_ext", "ai_engine_int"):
                    src_ai_ev = p_int if ai_role == "ai_engine_int" else p_ext
                    if not src_ai_ev:
                        continue
                    for idx_ex, item in enumerate(extra_role_variants.get(ai_role, ()), start=2):
                        ex_g = guid_str_to_bytes(item[0])
                        if ex_g in assigned_ai_targets:
                            continue
                        assigned_ai_targets.add(ex_g)
                        pending_clones.append((src_ai_ev.guid_bytes, ex_g, None))
                        used_guids.add(ex_g)
                        raw_b = item[2] if len(item) >= 3 and item[2] else None
                        ai_event_outputs[ex_g] = (
                            guid_str_to_bytes(raw_b) if raw_b else _bus_for(ai_role, ai_default_bus.get(ai_role))
                        )
                        ex_p = (item[1] if len(item) >= 2 and item[1] else "") or (
                            f"event:/AMS2/Vehicles/{safe_mod}/{ai_role}_{idx_ex}"
                        )
                        stdt_entries.append((ex_p, ex_g))

            for ev in ac_bank.events:
                if ev.guid_bytes in assigned_orig_evs:
                    continue
                cand_guid = guid_map.get(ev.guid_bytes, ev.guid_bytes)
                if cand_guid in used_guids:
                    cand_guid = hashlib.sha256(cand_guid + seed_tag.encode("utf-8") + struct.pack("<I", ev.index)).digest()[:16]
                    guid_map[ev.guid_bytes] = cand_guid
                used_guids.add(cand_guid)
                event_guid_map[ev.guid_str] = bytes_to_guid_str(cand_guid)
                if ev.guid_bytes not in evib_targets_orig:
                    sec_path = ev.path or f"event:/AMS2/Vehicles/{safe_mod}/Aux_{ev.role}_{ev.index}"
                    role_to_event_path.setdefault(ev.role, sec_path)
                    stdt_entries.append((sec_path, cand_guid))

            if target_bank_guid:
                raw_bg = target_bank_guid
                bg_b = guid_str_to_bytes(raw_bg) if isinstance(raw_bg, str) else bytes(raw_bg)
                guid_map[ac_bank.bank_guid] = bg_b
                new_bank_guid = bytes_to_guid_str(bg_b)
                stdt_entries.append((f"bank:/{safe_mod}", bg_b))
            else:
                new_bank_guid_bytes = guid_map.get(ac_bank.bank_guid, ac_bank.bank_guid)
                new_bank_guid = bytes_to_guid_str(new_bank_guid_bytes)
                stdt_entries.append((f"bank:/{safe_mod}", new_bank_guid_bytes))
        else:
            guid_map = {}
            used_guids = set()
            if extra_role_variants:
                for _items in extra_role_variants.values():
                    for _it in _items:
                        used_guids.add(guid_str_to_bytes(_it[0]))

            if p_ext and "EngineExt" in target_role_guids:
                ee_bg = guid_str_to_bytes(target_role_guids["EngineExt"])
                guid_map[p_ext.guid_bytes] = ee_bg
                used_guids.add(ee_bg)
                event_guid_map[p_ext.guid_str] = bytes_to_guid_str(ee_bg)
                role_to_event_guid["EngineExt"] = bytes_to_guid_str(ee_bg)
                ee_path = (target_role_paths or {}).get("EngineExt") or (p_ext.path or f"event:/AMS2/Vehicles/{safe_mod}/EngineExt")
                role_to_event_path["EngineExt"] = ee_path
                stdt_entries.append((ee_path, ee_bg))

            if p_int and "EngineInt" in target_role_guids:
                ei_bg = guid_str_to_bytes(target_role_guids["EngineInt"])
                guid_map[p_int.guid_bytes] = ei_bg
                used_guids.add(ei_bg)
                event_guid_map[p_int.guid_str] = bytes_to_guid_str(ei_bg)
                role_to_event_guid["EngineInt"] = bytes_to_guid_str(ei_bg)
                ei_path = (target_role_paths or {}).get("EngineInt") or (p_int.path or f"event:/AMS2/Vehicles/{safe_mod}/EngineInt")
                role_to_event_path["EngineInt"] = ei_path
                stdt_entries.append((ei_path, ei_bg))

            if extra_role_variants:
                # Clonar variantes extra de motor únicamente en bancos multi-variante (ej. F_Ultimate_2022, F_V8_Gen1)
                assigned_p_extras = set(used_guids) - {
                    guid_str_to_bytes(_it[0])
                    for _items in extra_role_variants.values()
                    for _it in _items
                }
                for p_role, p_src in (("EngineExt", p_ext), ("EngineInt", p_int)):
                    if not p_src or p_role not in target_role_guids:
                        continue
                    for idx_ex, item in enumerate(extra_role_variants.get(p_role, ()), start=2):
                        ex_g = guid_str_to_bytes(item[0])
                        if ex_g in assigned_p_extras:
                            continue
                        assigned_p_extras.add(ex_g)
                        pending_clones.append((p_src.guid_bytes, ex_g, None))
                        extra_player_clones_count += 1
                        ex_p = (item[1] if len(item) >= 2 and item[1] else "") or (
                            f"event:/AMS2/Vehicles/{safe_mod}/{p_role}_{idx_ex}"
                        )
                        stdt_entries.append((ex_p, ex_g))

            # Preservar SIEMPRE el Starter / Starting nativo de AMS2 (subgrafo PROJ completo + audio FSB5)
            # cuando el banco nativo los posee; solo caer al starter de AC o stub limpio si no existen en AMS2.
            starter_ac_ev = next(
                (ev for ev in ac_bank.events if any(k in (ev.path or "").lower() or k in (ev.role or "").lower() for k in ("starter", "starting", "startup", "start_ext", "start_int", "ignition"))),
                None,
            )
            starter_roles = [r for r in target_role_guids if r in ("Starter", "Starting")]

            nat_st_resolved = False
            if starter_roles and native_bank_path and Path(native_bank_path).is_file():
                try:
                    nat_st_bank = FmodBank(Path(native_bank_path))
                    explicit_st_guids: list[bytes] = [
                        guid_str_to_bytes(target_role_guids[r]) for r in starter_roles
                    ]
                    extra_st_paths_by_g: dict[bytes, str] = {}
                    for r in starter_roles:
                        rg = guid_str_to_bytes(target_role_guids[r])
                        if (target_role_paths or {}).get(r):
                            extra_st_paths_by_g[rg] = str((target_role_paths or {})[r])
                    if extra_role_variants:
                        for s_role in ("Starter", "Starting"):
                            for idx_ex, item in enumerate(extra_role_variants.get(s_role, ()), start=2):
                                ex_g = guid_str_to_bytes(item[0])
                                if ex_g not in explicit_st_guids:
                                    explicit_st_guids.append(ex_g)
                                ex_p = (item[1] if len(item) >= 2 and item[1] else "") or (
                                    f"event:/AMS2/Vehicles/{safe_mod}/{s_role}_{idx_ex}"
                                )
                                extra_st_paths_by_g.setdefault(ex_g, ex_p)

                    excl_for_st: set[bytes] = set()
                    for r_ex in ("EngineInt", "EngineExt", "AuxInt", "Exhaust"):
                        if r_ex in target_role_guids:
                            excl_for_st.add(guid_str_to_bytes(target_role_guids[r_ex]))
                        if extra_role_variants:
                            for _it in extra_role_variants.get(r_ex, ()):
                                if _it and _it[0]:
                                    excl_for_st.add(guid_str_to_bytes(_it[0]))
                    nat_st_guids, nat_st_roles, nat_st_paths = _find_native_starter_events_in_bank(
                        nat_st_bank,
                        explicit_guids=explicit_st_guids,
                        exclude_guids=excl_for_st,
                    )
                    if nat_st_guids:
                        nat_st_resolved = True
                        st_pmap: dict[bytes, str] = {}
                        for idx_g, sg in enumerate(nat_st_guids):
                            sr = nat_st_roles.get(sg, "Starter")
                            st_pmap[sg] = (
                                extra_st_paths_by_g.get(sg)
                                or nat_st_paths.get(sg)
                                or f"event:/AMS2/Vehicles/{safe_mod}/{sr}_{idx_g + 1}"
                            )
                            used_guids.add(sg)
                        pending_native_starter_graft = (nat_st_bank, nat_st_guids, st_pmap)
                        nat_st_set = set(nat_st_guids)
                        for s_role in starter_roles:
                            s_bg = guid_str_to_bytes(target_role_guids[s_role])
                            s_path = st_pmap.get(s_bg) or (target_role_paths or {}).get(s_role) or f"event:/AMS2/Vehicles/{safe_mod}/{s_role}"
                            role_to_event_guid[s_role] = bytes_to_guid_str(s_bg)
                            role_to_event_path[s_role] = s_path
                            if s_bg not in nat_st_set:
                                # Si uno de los roles pedidos no existía con audio en el banco nativo, emitir stub limpio
                                pending_clean_silent_starters.append(s_bg)
                                used_guids.add(s_bg)
                                stdt_entries.append((s_path, s_bg))
                except Exception:
                    nat_st_resolved = False

            if not nat_st_resolved and starter_ac_ev and starter_roles:
                active_root_orig_evs.add(starter_ac_ev.guid_bytes)
                for s_role in starter_roles:
                    s_bg = guid_str_to_bytes(target_role_guids[s_role])
                    guid_map[starter_ac_ev.guid_bytes] = s_bg
                    used_guids.add(s_bg)
                    role_to_event_guid[s_role] = bytes_to_guid_str(s_bg)
                    s_path = (target_role_paths or {}).get(s_role) or (starter_ac_ev.path or f"event:/AMS2/Vehicles/{safe_mod}/{s_role}")
                    role_to_event_path[s_role] = s_path
                    stdt_entries.append((s_path, s_bg))
                if extra_role_variants:
                    seen_st_clones = {guid_str_to_bytes(target_role_guids[r]) for r in starter_roles}
                    for s_role in ("Starter", "Starting"):
                        for idx_ex, item in enumerate(extra_role_variants.get(s_role, ()), start=2):
                            ex_g = guid_str_to_bytes(item[0])
                            if ex_g in seen_st_clones:
                                continue
                            seen_st_clones.add(ex_g)
                            pending_clones.append((starter_ac_ev.guid_bytes, ex_g, None))
                            extra_player_clones_count += 1
                            ex_p = (item[1] if len(item) >= 2 and item[1] else "") or (
                                f"event:/AMS2/Vehicles/{safe_mod}/{s_role}_{idx_ex}"
                            )
                            stdt_entries.append((ex_p, ex_g))
            elif not nat_st_resolved and starter_roles:
                for s_role in starter_roles:
                    s_bg = guid_str_to_bytes(target_role_guids[s_role])
                    pending_clean_silent_starters.append(s_bg)
                    used_guids.add(s_bg)
                    s_path = (target_role_paths or {}).get(s_role) or f"event:/AMS2/Vehicles/{safe_mod}/{s_role}"
                    role_to_event_guid[s_role] = bytes_to_guid_str(s_bg)
                    role_to_event_path[s_role] = s_path
                    stdt_entries.append((s_path, s_bg))
                if extra_role_variants:
                    seen_st_ext = set(pending_clean_silent_starters)
                    for s_role in ("Starter", "Starting"):
                        for idx_ex, item in enumerate(extra_role_variants.get(s_role, ()), start=2):
                            ex_g = guid_str_to_bytes(item[0])
                            if ex_g in seen_st_ext:
                                continue
                            seen_st_ext.add(ex_g)
                            pending_clean_silent_starters.append(ex_g)
                            used_guids.add(ex_g)
                            ex_p = (item[1] if len(item) >= 2 and item[1] else "") or (
                                f"event:/AMS2/Vehicles/{safe_mod}/{s_role}_{idx_ex}"
                            )
                            stdt_entries.append((ex_p, ex_g))

            # Evento de escape (Exhaust) si el objetivo lo requiere
            if "Exhaust" in target_role_guids and "Exhaust" not in role_to_event_guid:
                ex_bg = guid_str_to_bytes(target_role_guids["Exhaust"])
                ex_ev = next(
                    (ev for ev in ac_bank.events if any(k in (ev.path or "").lower() for k in ("exhaust", "tailpipe", "muffler"))),
                    None,
                )
                if ex_ev and ex_ev.guid_bytes not in guid_map:
                    active_root_orig_evs.add(ex_ev.guid_bytes)
                    guid_map[ex_ev.guid_bytes] = ex_bg
                    used_guids.add(ex_bg)
                    role_to_event_guid["Exhaust"] = bytes_to_guid_str(ex_bg)
                    ex_path = (target_role_paths or {}).get("Exhaust") or (ex_ev.path or f"event:/AMS2/Vehicles/{safe_mod}/Exhaust")
                    role_to_event_path["Exhaust"] = ex_path
                    stdt_entries.append((ex_path, ex_bg))
                    if extra_role_variants:
                        seen_ex_clones = {ex_bg}
                        for idx_ex, item in enumerate(extra_role_variants.get("Exhaust", ()), start=2):
                            eg_extra = guid_str_to_bytes(item[0])
                            if eg_extra in seen_ex_clones:
                                continue
                            seen_ex_clones.add(eg_extra)
                            pending_clones.append((ex_ev.guid_bytes, eg_extra, None))
                            extra_player_clones_count += 1
                            ep_extra = (item[1] if len(item) >= 2 and item[1] else "") or (
                                f"event:/AMS2/Vehicles/{safe_mod}/Exhaust_{idx_ex}"
                            )
                            stdt_entries.append((ep_extra, eg_extra))
                else:
                    # AC no trae evento Exhaust separado: emitir stub silencioso íntegro (0 punteros colgantes)
                    # para neutralizar el escape vanilla sin superponerlo al motor de AC.
                    ex_guids_to_silence = [ex_bg]
                    extra_ex_entries: list[tuple[bytes, str]] = []
                    if extra_role_variants:
                        seen_ex_ext = {ex_bg}
                        for idx_ex, item in enumerate(extra_role_variants.get("Exhaust", ()), start=2):
                            eg_extra = guid_str_to_bytes(item[0])
                            if eg_extra in seen_ex_ext:
                                continue
                            seen_ex_ext.add(eg_extra)
                            ex_guids_to_silence.append(eg_extra)
                            ep_extra = (item[1] if len(item) >= 2 and item[1] else "") or (
                                f"event:/AMS2/Vehicles/{safe_mod}/Exhaust_{idx_ex}"
                            )
                            extra_ex_entries.append((eg_extra, ep_extra))
                    pending_clean_silent_exhaust.extend(ex_guids_to_silence)
                    for g_ex in ex_guids_to_silence:
                        used_guids.add(g_ex)
                    role_to_event_guid["Exhaust"] = bytes_to_guid_str(ex_bg)
                    ex_path = (target_role_paths or {}).get("Exhaust") or f"event:/AMS2/Vehicles/{safe_mod}/Exhaust"
                    role_to_event_path["Exhaust"] = ex_path
                    stdt_entries.append((ex_path, ex_bg))
                    for eg_extra, ep_extra in extra_ex_entries:
                        stdt_entries.append((ep_extra, eg_extra))

            # Asignar resto de eventos secundarios de AC con GUIDs únicos garantizados
            for ev in ac_bank.events:
                if ev.guid_bytes in guid_map:
                    continue
                cand_guid = ev.guid_bytes
                if cand_guid in used_guids:
                    cand_guid = hashlib.sha256(cand_guid + safe_mod.encode("utf-8") + struct.pack("<I", ev.index)).digest()[:16]
                guid_map[ev.guid_bytes] = cand_guid
                used_guids.add(cand_guid)
                event_guid_map[ev.guid_str] = bytes_to_guid_str(cand_guid)
                sec_path = ev.path or f"event:/AMS2/Vehicles/{safe_mod}/Aux_{ev.role}_{ev.index}"
                role_to_event_path.setdefault(ev.role, sec_path)
                stdt_entries.append((sec_path, cand_guid))

            if target_bank_guid:
                raw_bg = target_bank_guid
                bg_b = guid_str_to_bytes(raw_bg) if isinstance(raw_bg, str) else bytes(raw_bg)
                guid_map[ac_bank.bank_guid] = bg_b
                new_bank_guid = bytes_to_guid_str(bg_b)
                stdt_entries.append((f"bank:/{safe_mod}", bg_b))
            else:
                new_bank_guid = bytes_to_guid_str(ac_bank.bank_guid)

    elif standalone_guids:
        local_sorted = collect_local_proj_guids(proj_buf)
        guid_map = build_order_preserving_guid_map(local_sorted, safe_mod)
        new_bank_guid_bytes = guid_map.get(ac_bank.bank_guid, ac_bank.bank_guid)
        new_bank_guid = bytes_to_guid_str(new_bank_guid_bytes)

        mod_prefix = safe_mod[:-3] if safe_mod.endswith("_AI") else safe_mod
        if is_ai_bank:
            stdt_entries.append((f"bank:/{mod_prefix}_Engine_AI", new_bank_guid_bytes))
        else:
            stdt_entries.append((f"bank:/{safe_mod}_Engine", new_bank_guid_bytes))

        used_paths: set[str] = set()
        ev_final_path: dict[bytes, str] = {}
        for ev in ac_bank.events:
            ng_bytes = guid_map.get(ev.guid_bytes, ev.guid_bytes)
            ng = bytes_to_guid_str(ng_bytes)
            event_guid_map[ev.guid_str] = ng
            role_to_event_guid.setdefault(ev.role, ng)

            is_main_ext = p_ext_sel is not None and ev.guid_bytes == p_ext_sel.guid_bytes
            is_main_int = p_int_sel is not None and ev.guid_bytes == p_int_sel.guid_bytes

            if is_ai_bank and is_main_ext:
                # Evento real para rivales (MAIN) + clon profundo independiente para TV Replays (MAIN_LIVE),
                # cada uno con su propia GUID (sin alias duplicados que dejen rutas huérfanas en STDT).
                main_path = f"event:/AMS2/Vehicles/Mods/{mod_prefix}/AI_EngineMAIN"
                live_path = f"event:/AMS2/Vehicles/Mods/{mod_prefix}/AI_EngineMAIN_LIVE"
                live_bytes = hashlib.sha256(ng_bytes + b"_ai_engine_live").digest()[:16]
                stdt_entries.append((main_path, ng_bytes))
                stdt_entries.append((live_path, live_bytes))
                pending_clones.append((ev.guid_bytes, live_bytes, None))
                ev_final_path[ev.guid_bytes] = main_path
                ai_event_outputs[ng_bytes] = guid_str_to_bytes(AMS2_AI_ENGINE_BUS)
                ai_event_outputs[live_bytes] = guid_str_to_bytes(AMS2_AI_ENGINE_BUS)
                role_to_event_path["ai_engine_live"] = live_path
                role_to_event_guid["ai_engine_live"] = bytes_to_guid_str(live_bytes)
            elif is_ai_bank and is_main_int:
                int_path = f"event:/AMS2/Vehicles/Mods/{mod_prefix}/AI_EngineInt"
                stdt_entries.append((int_path, ng_bytes))
                ev_final_path[ev.guid_bytes] = int_path
                ai_event_outputs[ng_bytes] = None
            else:
                last_part = ev.path.split("/")[-1] if ev.path else ev.role
                clean_ev_path = f"event:/AMS2/Vehicles/Mods/{safe_mod}/{last_part}"
                if clean_ev_path.lower() in used_paths:
                    clean_ev_path = f"{clean_ev_path}_{ev.index}"
                used_paths.add(clean_ev_path.lower())
                ev_final_path[ev.guid_bytes] = clean_ev_path
                role_to_event_path.setdefault(ev.role, clean_ev_path)
                stdt_entries.append((clean_ev_path, ng_bytes))

        # Los roles de motor SIEMPRE apuntan al evento principal (no a sub-eventos anidados sin nombre)
        if p_ext_sel is not None and p_ext_sel.guid_bytes in ev_final_path:
            ext_p = ev_final_path[p_ext_sel.guid_bytes]
            ext_g = event_guid_map[p_ext_sel.guid_str]
            role_to_event_path["EngineExt"] = ext_p
            role_to_event_guid["EngineExt"] = ext_g
            if is_ai_bank:
                role_to_event_path["ai_engine_ext"] = ext_p
                role_to_event_guid["ai_engine_ext"] = ext_g
        if p_int_sel is not None and p_int_sel.guid_bytes in ev_final_path:
            int_p = ev_final_path[p_int_sel.guid_bytes]
            int_g = event_guid_map[p_int_sel.guid_str]
            role_to_event_path["EngineInt"] = int_p
            role_to_event_guid["EngineInt"] = int_g
            if is_ai_bank:
                role_to_event_path["ai_engine_int"] = int_p
                role_to_event_guid["ai_engine_int"] = int_g

        _keep_mod_roles = {
            "EngineExt", "EngineInt", "Starter", "Starting", "Exhaust",
            "Transmission", "Turbo", "Gearshift", "GearshiftExt", "GearshiftInt",
            "gear_ext", "gear_int", "gear_grind", "GearshiftGrind",
        }
        if not mute_secondary_pops:
            _keep_mod_roles.update({"BackfireExt", "BackfireInt", "Limiter"})
        if p_ext_sel:
            active_root_orig_evs.add(p_ext_sel.guid_bytes)
        if p_int_sel:
            active_root_orig_evs.add(p_int_sel.guid_bytes)
        for ev in ac_bank.events:
            if ev.role in _keep_mod_roles:
                active_root_orig_evs.add(ev.guid_bytes)
    else:
        new_bank_guid = bytes_to_guid_str(ac_bank.bank_guid)
        for ev in ac_bank.events:
            event_guid_map[ev.guid_str] = ev.guid_str
            role_to_event_guid[ev.role] = ev.guid_str
            role_to_event_path[ev.role] = ev.path or f"event:/AMS2/Vehicles/Mods/{safe_mod}/{ev.role}"

    new_stdt_payload = build_stdt_chunk(stdt_entries) if stdt_entries else None
    exported_ev_guids: set[bytes] = {g for p, g in stdt_entries if p.startswith("event:/")}
    if not mute_secondary_pops:
        active_root_orig_evs.update(_muted_event_guids_for_bank(ac_bank))

    # Parchear PROJ
    (
        wav_redirects,
        autopitch_updates,
        gain_db_deltas,
        silence_wav_guids,
        unmute_wav_guids,
        custom_wav_replacements,
    ) = _collect_mapping_patches_for_direct_bank(ac_bank, mappings, rpm_bands, library)

    patched_proj_body = _patch_proj_chunks(
        proj_payload=proj_buf,
        wav_redirects=wav_redirects,
        autopitch_updates=autopitch_updates,
        guid_map=guid_map,
        rpm_bands=rpm_bands,
        gain_db_deltas=gain_db_deltas,
        silence_wav_guids=silence_wav_guids,
        unmute_wav_guids=unmute_wav_guids,
    )

    # Clones profundos de eventos (roles de IA vanilla con GUIDs propias) y límites/buses de IA
    for src_g, dst_g, _bus in pending_clones:
        src_final = guid_map.get(src_g, src_g) if guid_map else src_g
        patched_proj_body = clone_event_deep_in_proj(patched_proj_body, src_final, dst_g, safe_mod.encode("utf-8"))
    if is_ai_bank:
        for ev_g, bus_g in ai_event_outputs.items():
            if bus_g is not None:
                patched_proj_body = set_event_output_bus_in_proj(patched_proj_body, ev_g, bus_g)
        # Limitar instancias simultáneas (modo virtualizar como la IA nativa de AMS2): evita la saturación
        # del mezclador de FMOD con 15-20 rivales, causa del audio entrecortado en carreras contra la IA.
        if ai_event_outputs:
            patched_proj_body = set_event_instance_limits_in_proj(
                patched_proj_body, set(ai_event_outputs.keys()), AI_EVENT_MAX_INSTANCES
            )
        patched_proj_body = set_event_instance_limits_in_proj(
            patched_proj_body, None, 16, AI_EVENT_STEAL_VIRTUALIZE
        )
        patched_proj_body = _sync_evts_partition_and_bnki(
            bytearray(patched_proj_body),
            exported_ev_guids if exported_ev_guids else None,
        )
        patched_proj_body = bytes(patched_proj_body)
    else:
        patched_proj_body = set_event_instance_limits_in_proj(
            patched_proj_body, None, 4, AI_EVENT_STEAL_VIRTUALIZE
        )
        if pending_clean_silent_starters:
            patched_proj_body = _inject_clean_silent_events_in_proj(
                patched_proj_body,
                pending_clean_silent_starters,
                AMS2_STARTUP_BUS_BYTES,
                (safe_mod + ":sil_st").encode("utf-8"),
            )
        if pending_clean_silent_exhaust:
            EXHAUST_BUS = guid_str_to_bytes("{ffe57d5c-46ed-4b06-86d5-8fa6f8ffdf07}")
            patched_proj_body = _inject_clean_silent_events_in_proj(
                patched_proj_body,
                pending_clean_silent_exhaust,
                EXHAUST_BUS,
                (safe_mod + ":sil_exh").encode("utf-8"),
            )
        if extra_player_clones_count > 0:
            proj_mut = bytearray(patched_proj_body)
            for cid, ps, pe, _ in iter_riff_chunks(proj_mut, 0, len(proj_mut)):
                if cid == b"BNKI" and pe - ps >= 28:
                    cur_exp = struct.unpack_from("<I", proj_mut, ps + 24)[0]
                    struct.pack_into("<I", proj_mut, ps + 24, cur_exp + extra_player_clones_count)
                    break
            patched_proj_body = bytes(proj_mut)

    # Silenciar cajas de TC / LC y capas esparsas de petardeo en pistas PMLB y ganancia INST
    # (Se ejecuta DESPUÉS de _patch_proj_chunks para garantizar que rpm_bands no reabra las cajas)
    muted_locs: set[tuple[int, int]] = set()
    if mute_secondary_pops:
        sparse_locs, dip_ratios, non_sustained_locs = _compute_sparse_pop_locs(ac_bank)
        mapped_muted_evs = {
            (guid_map.get(g, g) if guid_map else g)
            for g in _muted_event_guids_for_bank(ac_bank)
        }
        patched_proj_body, _ = _silence_tc_and_lc_boxes_in_proj(
            patched_proj_body,
            muted_ev_guids_in=mapped_muted_evs,
            sparse_locs=sparse_locs,
            loc_dip_ratios=dip_ratios,
            non_sustained_locs=non_sustained_locs,
            out_muted_locs=muted_locs,
        )

    eff_target_max_rpm = target_max_rpm
    if eff_target_max_rpm is None and native_bank_path:
        eff_target_max_rpm = _extract_native_engine_max_rpm(native_bank_path)
    if eff_target_max_rpm and eff_target_max_rpm > 3000.0:
        patched_proj_body = _extend_top_rpm_boxes_in_proj(patched_proj_body, eff_target_max_rpm)

    # Igualar sonoridad y alcance 3D exterior/replay al estándar vanilla de AMS2 sin saturar audio PCM16
    if active_scope != "interior":
        ext_loudness_profiles: dict[bytes, tuple[tuple[int, float, float, int, float, float], float]] = {}
        native_panners = _extract_native_exterior_spatializers(native_bank_path) if native_bank_path else {}
        eff_p_ext_guid: bytes | None = (
            p_ext.guid_bytes if (target_role_guids and "p_ext" in locals() and p_ext is not None)
            else (p_ext_sel.guid_bytes if p_ext_sel is not None else None)
        )
        if target_role_guids:
            if is_ai_bank:
                for role_k, def_pan, df_dist in (
                    ("ai_engine_live", AMS2_LIVE_REPLAY_PANNER, 5000.0),
                    ("ai_engine_main", AMS2_AI_MAIN_PANNER, 3000.0),
                    ("ai_engine_ext", AMS2_LIVE_REPLAY_PANNER, 5000.0),
                ):
                    if role_k in target_role_guids:
                        rg = guid_str_to_bytes(target_role_guids[role_k])
                        ext_loudness_profiles[rg] = (native_panners.get(rg) or def_pan, df_dist)
                    if extra_role_variants:
                        for item in extra_role_variants.get(role_k, ()):
                            eg = guid_str_to_bytes(item[0])
                            ext_loudness_profiles[eg] = (native_panners.get(eg) or def_pan, df_dist)
            else:
                if "EngineExt" in target_role_guids:
                    rg = guid_str_to_bytes(target_role_guids["EngineExt"])
                    ext_loudness_profiles[rg] = (native_panners.get(rg) or AMS2_PLAYER_EXT_PANNER, 3000.0)
                if extra_role_variants:
                    for item in extra_role_variants.get("EngineExt", ()):
                        eg = guid_str_to_bytes(item[0])
                        ext_loudness_profiles[eg] = (native_panners.get(eg) or AMS2_PLAYER_EXT_PANNER, 3000.0)
        elif standalone_guids:
            if is_ai_bank:
                if "ai_engine_live" in role_to_event_guid:
                    ext_loudness_profiles[guid_str_to_bytes(role_to_event_guid["ai_engine_live"])] = (
                        AMS2_LIVE_REPLAY_PANNER, 5000.0
                    )
                if "ai_engine_ext" in role_to_event_guid:
                    ext_loudness_profiles[guid_str_to_bytes(role_to_event_guid["ai_engine_ext"])] = (
                        AMS2_AI_MAIN_PANNER, 3000.0
                    )
            elif "EngineExt" in role_to_event_guid:
                ext_loudness_profiles[guid_str_to_bytes(role_to_event_guid["EngineExt"])] = (
                    AMS2_PLAYER_EXT_PANNER, 3000.0
                )
        elif eff_p_ext_guid is not None:
            ext_loudness_profiles[eff_p_ext_guid] = (
                (AMS2_LIVE_REPLAY_PANNER, 5000.0) if is_ai_bank else (AMS2_PLAYER_EXT_PANNER, 3000.0)
            )

        if ext_loudness_profiles:
            patched_proj_body = _match_ams2_exterior_loudness_in_proj(
                proj_buf=patched_proj_body,
                ac_bank=ac_bank,
                orig_proj_payload=orig_proj_payload,
                p_ext_orig_guid=eff_p_ext_guid,
                ext_event_profiles=ext_loudness_profiles,
            )

    codec_label = "Nativo AC (Transmutación Directa Sin Donante)"
    if cached_fsb_blobs is not None and len(cached_fsb_blobs) == len(ac_bank.subbanks):
        final_fsb_blobs = [bytearray(b) for b in cached_fsb_blobs]
        codec_label = "FADPCM (Transmutación Directa Comprimida)" if output_codec_mode == FSB5_MODE_FADPCM else codec_label
    elif output_codec_mode == FSB5_MODE_FADPCM:
        active_locs: set[tuple[int, int]] | None = None
        if prune_unused_samples and active_root_orig_evs:
            active_locs = _compute_active_direct_sample_locs(orig_proj_payload, ac_bank, active_root_orig_evs)
        final_fsb_blobs = []
        for sb_idx, sb in enumerate(ac_bank.subbanks):
            sb_mode = struct.unpack_from("<I", sb.raw_fsb_bytes, 24)[0] if len(sb.raw_fsb_bytes) >= 28 else FSB5_MODE_PCM16
            if sb_mode == FSB5_MODE_PCM16:
                final_fsb_blobs.append(bytearray(_reencode_pcm16_subbank_to_fadpcm(ac_bank, sb_idx, active_locs, muted_locs)))
            else:
                final_fsb_blobs.append(bytearray(sb.raw_fsb_bytes))
        codec_label = "FADPCM (Transmutación Directa Comprimida)"
    else:
        final_fsb_blobs = [bytearray(sb.raw_fsb_bytes) for sb in ac_bank.subbanks]
        if mute_secondary_pops and muted_locs:
            for sl in ac_bank.slots:
                if (sl.subbank_index, sl.sample_index) not in muted_locs:
                    continue
                # Sólo en PCM16 sin compresión para no corromper flujos Vorbis
                if sl.mode == FSB5_MODE_PCM16 and sl.subbank_index < len(final_fsb_blobs):
                    fbuf = final_fsb_blobs[sl.subbank_index]
                    if len(fbuf) >= 20:
                        shdr_size, name_size = struct.unpack_from("<II", fbuf, 12)
                        data_base = 60 + shdr_size + name_size
                        start = data_base + sl.data_offset
                        end = start + len(sl.data_bytes)
                        if end <= len(fbuf):
                            fbuf[start:end] = b"\x00" * len(sl.data_bytes)

    if cached_fsb_blobs is None and custom_wav_replacements:
        _apply_custom_wav_replacements_to_subbanks(ac_bank, final_fsb_blobs, custom_wav_replacements)

    sndh_ver = ac_bank.sndh_version or (0x00080000 | (2 * len(final_fsb_blobs) + 1))
    dummy_sndh = struct.pack("<I", sndh_ver) + b"\x00" * (8 * len(final_fsb_blobs))
    temp_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, dummy_sndh, new_stdt_payload)

    fmt_chunk_bytes = b"FMT " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    if len(fmt_payload) & 1:
        fmt_chunk_bytes += b"\x00"

    proj_list_size = 4 + len(temp_proj_body)
    proj_total_bytes = 8 + proj_list_size + (proj_list_size & 1)

    cur_file_pos = 12 + len(fmt_chunk_bytes) + proj_total_bytes
    sndh_entries: list[tuple[int, int]] = []
    snd_chunks_built: list[bytes] = []

    for fsb_blob in final_fsb_blobs:
        snd_payload_start = cur_file_pos + 8
        pad_before = (32 - (snd_payload_start % 32)) % 32
        fsb_pos = snd_payload_start + pad_before
        fsb_len = len(fsb_blob)
        snd_payload = (b"\x00" * pad_before) + bytes(fsb_blob)
        snd_chunk = b"SND " + struct.pack("<I", len(snd_payload)) + snd_payload
        if len(snd_payload) & 1:
            snd_chunk += b"\x00"
        sndh_entries.append((fsb_pos, fsb_len))
        snd_chunks_built.append(snd_chunk)
        cur_file_pos += len(snd_chunk)

    real_sndh = struct.pack("<I", sndh_ver) + b"".join(
        struct.pack("<II", fpos, flen) for fpos, flen in sndh_entries
    )
    final_proj_body = _replace_sndh_and_stdt_in_proj(patched_proj_body, real_sndh, new_stdt_payload)
    assert len(final_proj_body) == len(temp_proj_body), "Tamaño inesperado al actualizar SNDH/STDT"

    proj_chunk_bytes = (
        b"LIST"
        + struct.pack("<I", 4 + len(final_proj_body))
        + b"PROJ"
        + final_proj_body
        + (b"\x00" if (len(final_proj_body) & 1) else b"")
    )

    body = fmt_chunk_bytes + proj_chunk_bytes + b"".join(snd_chunks_built)
    full_riff = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body
    ac_fsb_cache = [bytes(b) for b in final_fsb_blobs]

    # Injertar eventos nativos AMS2 Starter/Starting completos (subgrafo PROJ + audio FSB5 compacto)
    if pending_native_starter_graft is not None:
        nat_st_bank, nat_st_guids, st_pmap = pending_native_starter_graft
        full_riff, _st_gmap, _st_pmap_fin = graft_native_events_with_audio_into_bank(
            target_bank_bytes=full_riff,
            native_bank=nat_st_bank,
            native_event_guids=nat_st_guids,
            event_paths_map=st_pmap,
            remap_root_guids=False,
            output_codec_mode=FSB5_MODE_FADPCM,
            mod_identifier=safe_mod,
            seed_tag="vanilla_starter",
        )

    return PortedBankResult(
        bank_bytes=full_riff,
        mod_identifier=safe_mod,
        standalone_guids=standalone_guids,
        new_bank_guid=new_bank_guid,
        event_guid_map=event_guid_map,
        role_to_event_guid=role_to_event_guid,
        role_to_event_path=role_to_event_path,
        replaced_slots_count=len(ac_bank.slots),
        kept_slots_count=0,
        codec_used=codec_label,
        throttle_cut_injected_events=0,
        local_guid_map=dict(guid_map or {}),
        fsb_blobs_cache=ac_fsb_cache,
    )


def apply_ported_bank_to_csdbin(
    csd_doc: CsdbinDocument,
    bank_result: PortedBankResult,
    bank_rel_path_no_ext: str,
    ai_bank_result: PortedBankResult | None = None,
    ai_bank_rel_path_no_ext: str | None = None,
    optimize_chase_cam: bool = True,
    preserve_native_transmission: bool = True,
) -> CsdbinDocument:
    """Actualiza los enlaces del documento .csdbin con las rutas `event:/...` registradas en STDT.

    Con `preserve_native_transmission=True` (por defecto), mantiene intactos los enlaces nativos de
    AMS2 para transmisión (`transmission_bank`/`transmission`/`gearbox`), caja de cambios
    (`gearshift_bank`/`gearshift_up`/`gearshift_down`/`gearshift_grind`), viento y limitador nativos
    para evitar que eventos incompatibles de AC (p.ej. `drivetrain_speed`) los silencien.
    """
    clean_bank_path = bank_rel_path_no_ext.replace("\\", "/").removesuffix(".bank")
    csd_doc.links["engine_bank"] = clean_bank_path

    rpaths = bank_result.role_to_event_path
    if "EngineExt" in rpaths:
        csd_doc.links["engine_ext"] = rpaths["EngineExt"]
    if "EngineInt" in rpaths:
        csd_doc.links["engine_int"] = rpaths["EngineInt"]

    # Enlaces de banco y eventos de IA (rivales en pista + repeticiones LIVE)
    if ai_bank_result is not None and ai_bank_rel_path_no_ext:
        clean_ai_path = ai_bank_rel_path_no_ext.replace("\\", "/").removesuffix(".bank")
        csd_doc.links["ai_engine_bank"] = clean_ai_path

        ai_rpaths = ai_bank_result.role_to_event_path
        if "ai_engine_ext" in ai_rpaths:
            csd_doc.links["ai_engine_ext"] = ai_rpaths["ai_engine_ext"]
        elif "EngineExt" in ai_rpaths:
            csd_doc.links["ai_engine_ext"] = ai_rpaths["EngineExt"]

        if "ai_engine_live" in ai_rpaths:
            csd_doc.links["ai_engine_live"] = ai_rpaths["ai_engine_live"]
        elif "EngineExt" in ai_rpaths:
            csd_doc.links["ai_engine_live"] = ai_rpaths["EngineExt"]

        if "ai_engine_int" in ai_rpaths:
            csd_doc.links["ai_engine_int"] = ai_rpaths["ai_engine_int"]
        elif "EngineInt" in ai_rpaths:
            csd_doc.links["ai_engine_int"] = ai_rpaths["EngineInt"]

    if "Exhaust" in rpaths:
        if "exhaust_int" in csd_doc.links or "exhaust_ext" not in csd_doc.links:
            csd_doc.links["exhaust_int"] = rpaths["Exhaust"]
        if "exhaust_ext" in csd_doc.links:
            csd_doc.links["exhaust_ext"] = rpaths["Exhaust"]
    else:
        # Si no hay evento Exhaust separado, eliminamos cualquier enlace residual
        csd_doc.links.pop("exhaust_int", None)
        csd_doc.links.pop("exhaust_ext", None)

    if "Starter" in rpaths:
        csd_doc.links["starter"] = rpaths["Starter"]
    else:
        csd_doc.links.pop("starter", None)

    if "Starting" in rpaths:
        csd_doc.links["starting"] = rpaths["Starting"]
    else:
        csd_doc.links.pop("starting", None)

    # Enlaces de caja de cambios (gearshift): conservar los nativos de AMS2 si ya existen en el .csdbin
    if "NativeGearshiftUp" in rpaths or "NativeGearshiftDown" in rpaths:
        csd_doc.links["gearshift_bank"] = clean_bank_path
        if "NativeGearshiftUp" in rpaths:
            csd_doc.links["gearshift_up"] = rpaths["NativeGearshiftUp"]
        if "NativeGearshiftDown" in rpaths:
            csd_doc.links["gearshift_down"] = rpaths["NativeGearshiftDown"]
    else:
        has_native_gearshift = preserve_native_transmission and any(
            k in csd_doc.links and csd_doc.links[k]
            for k in ("gearshift_bank", "gearshift_up", "gearshift_down")
        )
        if not has_native_gearshift:
            gear_path = (
                rpaths.get("GearshiftExt")
                or rpaths.get("Gearshift")
                or rpaths.get("GearshiftInt")
                or rpaths.get("gear_ext")
                or rpaths.get("gear_int")
            )
            if gear_path:
                csd_doc.links["gearshift_bank"] = clean_bank_path
                csd_doc.links["gearshift_up"] = gear_path
                csd_doc.links["gearshift_down"] = gear_path
        if not (preserve_native_transmission and csd_doc.links.get("gearshift_grind")):
            grind_path = rpaths.get("gear_grind") or rpaths.get("GearshiftGrind")
            if grind_path:
                csd_doc.links["gearshift_grind"] = grind_path

    # Enlaces de turbo / sobrealimentador
    turbo_path = rpaths.get("Turbo") or rpaths.get("turbo")
    if turbo_path:
        csd_doc.links["turbo_bank"] = clean_bank_path
        csd_doc.links["turbo"] = turbo_path

    # Enlaces de transmisión / diferencial (GearWhine): preservar siempre el sonido original de AMS2
    # (AC usa el parámetro `drivetrain_speed` que AMS2 no alimenta -> quedaría silenciado)
    if "NativeTransmission" in rpaths:
        csd_doc.links["transmission_bank"] = clean_bank_path
        csd_doc.links["transmission"] = rpaths["NativeTransmission"]
    elif not preserve_native_transmission:
        trans_path = rpaths.get("Transmission") or rpaths.get("transmission") or rpaths.get("transmission_ext")
        if trans_path:
            csd_doc.links["transmission_bank"] = clean_bank_path
            csd_doc.links["transmission"] = trans_path

    # Enlaces de limitador de revoluciones: no pisar el RevLimiterTC nativo de AMS2 si ya existe
    has_native_limiter = preserve_native_transmission and any(
        k in csd_doc.links and csd_doc.links[k] for k in ("rev_limiter_bank", "rev_limiter")
    )
    if not has_native_limiter:
        limiter_path = rpaths.get("Limiter") or rpaths.get("limiter")
        if limiter_path:
            csd_doc.links["rev_limiter_bank"] = clean_bank_path
            csd_doc.links["rev_limiter"] = limiter_path

    # Enlaces de petardeos (backfire)
    if "BackfireExt" in rpaths:
        csd_doc.links["backfire_ext"] = rpaths["BackfireExt"]
    elif "backfire_ext" in rpaths:
        csd_doc.links["backfire_ext"] = rpaths["backfire_ext"]

    if "BackfireInt" in rpaths:
        csd_doc.links["backfire_int"] = rpaths["BackfireInt"]
    elif "backfire_int" in rpaths:
        csd_doc.links["backfire_int"] = rpaths["backfire_int"]

    # Enlaces de viento: no pisar el WindBuffeting nativo de AMS2 si ya existe
    has_native_wind = preserve_native_transmission and any(
        k in csd_doc.links and csd_doc.links[k] for k in ("wind_bank", "wind")
    )
    if not has_native_wind:
        wind_path = rpaths.get("Wind") or rpaths.get("wind")
        if wind_path:
            csd_doc.links["wind_bank"] = clean_bank_path
            csd_doc.links["wind"] = wind_path

    # Optimización de colocación 3D para Chase Cam (3ra persona):
    # En AMS2 la cámara trasera se sitúa a 6-8m. Si la distancia mínima es 2.0m, la curva logarítmica
    # atenúa más de 10 dB, haciendo que el auto suene lejano. Aquí expandimos el radio a 8.0m y ajustamos ganancia.
    if optimize_chase_cam and "engine_placement" in csd_doc.links:
        try:
            raw_blob = csd_doc.links["engine_placement"].replace("blob:", "")
            bdata = bytearray(bytes.fromhex(raw_blob))
            num_slots = len(bdata) // 32
            for s in (3, 4):  # Slot 3 = Chase Near, Slot 4 = Chase Far
                if s < num_slots:
                    offset = s * 32
                    vol = struct.unpack_from("<f", bdata, offset)[0]
                    # Refuerzo del volumen para compensar pérdida de presencia
                    struct.pack_into("<f", bdata, offset, min(1.0, max(vol * 1.35, 0.40)))
                    # Expandir distancia mínima a 8.0m para que la cámara de persecución quede en zona 0 dB
                    struct.pack_into("<f", bdata, offset + 20, 8.0)
            csd_doc.links["engine_placement"] = f"blob:{bdata.hex()}"
        except Exception:
            pass

    return csd_doc


@dataclass
class ModExportSummary:
    """Resumen de archivos exportados en el paquete final del Mod para AMS2."""

    output_root: Path
    bank_file: Path
    csdbin_file: Path | None
    csd_text_file: Path | None
    readme_file: Path
    manifest_file: Path
    bank_result: PortedBankResult
    ai_bank_file: Path | None = None
    ai_bank_result: PortedBankResult | None = None


def export_ams2_mod_package(
    output_root: Path,
    mod_identifier: str,
    vehicle_folder_name: str,
    donor: FmodBank | None = None,
    mappings: list[SlotMappingConfig] | None = None,
    library: SourceAudioLibrary | None = None,
    csd_doc: CsdbinDocument | None = None,
    standalone_guids: bool = True,
    sync_autopitch: bool = True,
    output_codec_mode: int = FSB5_MODE_PCM16,
    force_sample_rate: int = 48000,
    auto_link_csdbin: bool = True,
    rpm_bands: list[RpmBandConfig] | None = None,
    throttle_cut: ThrottleCutConfig | None = None,
    direct_ac_bank: FmodBank | None = None,
    build_ai_bank: bool = True,
    optimize_chase_cam: bool = True,
    mute_pops: bool = False,
    invert_spatial_cone: bool = True,
    ams2_root: Path | None = None,
    backup_dir: Path | None = None,
    preserve_native_starter: bool = True,
    preserve_native_transmission: bool = True,
) -> ModExportSummary:
    """Compila el .bank y el .csdbin listos para copiar en Automobilista 2 / AMS2 Content Manager."""
    output_root = Path(output_root)
    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_sound"
    safe_veh = "".join(c if (c.isalnum() or c in "_-") else "_" for c in vehicle_folder_name.strip()) or safe_mod

    # Capturar enlaces originales de AMS2 antes de modificar el .csdbin
    orig_engine_bank = csd_doc.links.get("engine_bank", "") if csd_doc is not None else ""
    orig_starter_path = csd_doc.links.get("starter", "") if csd_doc is not None else ""
    orig_starting_path = csd_doc.links.get("starting", "") if csd_doc is not None else ""
    orig_trans_bank = csd_doc.links.get("transmission_bank", "") if csd_doc is not None else ""
    orig_trans_path = csd_doc.links.get("transmission", "") if csd_doc is not None else ""
    orig_gear_bank = csd_doc.links.get("gearshift_bank", "") if csd_doc is not None else ""
    orig_gear_up = csd_doc.links.get("gearshift_up", "") if csd_doc is not None else ""
    orig_gear_down = csd_doc.links.get("gearshift_down", "") if csd_doc is not None else ""

    if direct_ac_bank is not None:
        has_large_pcm16 = (
            sum(len(sb.raw_fsb_bytes) for sb in direct_ac_bank.subbanks) > 20_000_000
            and any(sl.mode == FSB5_MODE_PCM16 for sl in direct_ac_bank.slots)
        )
        direct_codec = FSB5_MODE_FADPCM if has_large_pcm16 else None
        bank_res = build_direct_ported_bank_from_ac(
            ac_bank=direct_ac_bank,
            mod_identifier=safe_mod,
            standalone_guids=standalone_guids,
            rpm_bands=rpm_bands,
            mute_secondary_pops=mute_pops,
            invert_spatial_cone=invert_spatial_cone,
            mappings=mappings,
            output_codec_mode=direct_codec,
            prune_unused_samples=has_large_pcm16,
        )
    else:
        if donor is None:
            raise ValueError(
                "Se requiere un banco donante de AMS2 cuando no se utiliza el modo directo direct_ac_bank."
            )
        bank_res = build_ported_bank(
            donor=donor,
            mappings=mappings or [],
            library=library,
            mod_identifier=safe_mod,
            standalone_guids=standalone_guids,
            sync_autopitch=sync_autopitch,
            output_codec_mode=output_codec_mode,
            force_sample_rate=force_sample_rate,
            rpm_bands=rpm_bands,
            throttle_cut=throttle_cut,
        )

    # Preservar Starter / Starting (y transmisión/gearshift empaquetados si vivían dentro de engine_bank)
    # desde el banco original de AMS2 al exportar Mods para que nunca queden mudos.
    need_starter_graft = preserve_native_starter and (
        direct_ac_bank is not None
        or ("Starter" not in bank_res.role_to_event_path and "Starting" not in bank_res.role_to_event_path)
    )
    if need_starter_graft and (csd_doc is not None or donor is not None):
        nat_src_bank, st_guids, st_g2r, st_g2p = _resolve_native_starter_source_for_csdbin(
            csd_doc=csd_doc,
            orig_engine_bank=orig_engine_bank,
            orig_starter_path=orig_starter_path,
            orig_starting_path=orig_starting_path,
            ams2_root=ams2_root,
            backup_dir=backup_dir,
            fallback_donor=donor,
            extra_search_roots=[output_root],
        )
        if nat_src_bank is not None:
            all_graft_guids: list[bytes] = list(st_guids)
            graft_roles: dict[bytes, str] = dict(st_g2r)
            graft_orig_paths: dict[bytes, str] = dict(st_g2p)

            # Caso especial: si la transmisión o gearshift nativos venían empaquetados dentro del mismo engine_bank
            eb_norm = orig_engine_bank.replace("\\", "/").removesuffix(".bank").strip("/").lower()
            if preserve_native_transmission and eb_norm:
                nat_stdt = parse_stdt_all_paths(nat_src_bank.stdt_bytes) if nat_src_bank.stdt_bytes else {}
                nat_ms = _load_ams2_master_strings_cached(ams2_root)
                p2g_nat: dict[str, bytes] = {}
                for ev_n in nat_src_bank.events:
                    ep_n = nat_stdt.get(ev_n.guid_bytes) or nat_ms.get(ev_n.guid_str.lower(), "") or (ev_n.path or "")
                    if ep_n:
                        p2g_nat[ep_n.lower()] = ev_n.guid_bytes

                tb_norm = orig_trans_bank.replace("\\", "/").removesuffix(".bank").strip("/").lower()
                if tb_norm == eb_norm and orig_trans_path:
                    tg_b = p2g_nat.get(orig_trans_path.lower())
                    if tg_b and tg_b not in graft_roles:
                        all_graft_guids.append(tg_b)
                        graft_roles[tg_b] = "NativeTransmission"
                        graft_orig_paths[tg_b] = orig_trans_path

                gb_norm = orig_gear_bank.replace("\\", "/").removesuffix(".bank").strip("/").lower()
                if gb_norm == eb_norm:
                    for g_role_k, g_path_v in (("NativeGearshiftUp", orig_gear_up), ("NativeGearshiftDown", orig_gear_down)):
                        if g_path_v:
                            gg_b = p2g_nat.get(g_path_v.lower())
                            if gg_b and gg_b not in graft_roles:
                                all_graft_guids.append(gg_b)
                                graft_roles[gg_b] = g_role_k
                                graft_orig_paths[gg_b] = g_path_v

            if all_graft_guids:
                graft_paths_map: dict[bytes, str] = {}
                role_seq: dict[str, int] = {}
                for og in all_graft_guids:
                    rk = graft_roles.get(og, "Starter")
                    if standalone_guids:
                        role_seq[rk] = role_seq.get(rk, 0) + 1
                        suf = "" if role_seq[rk] == 1 else f"_{role_seq[rk]}"
                        graft_paths_map[og] = f"event:/AMS2/Vehicles/Mods/{safe_mod}/{rk}{suf}"
                    else:
                        graft_paths_map[og] = graft_orig_paths.get(og) or f"event:/AMS2/Vehicles/Mods/{safe_mod}/{rk}"

                grafted_bytes, g_map_nat, fin_paths_nat = graft_native_events_with_audio_into_bank(
                    target_bank_bytes=bank_res.bank_bytes,
                    native_bank=nat_src_bank,
                    native_event_guids=all_graft_guids,
                    event_paths_map=graft_paths_map,
                    remap_root_guids=standalone_guids,
                    output_codec_mode=FSB5_MODE_FADPCM,
                    mod_identifier=safe_mod,
                    seed_tag="mod_native_starter",
                )
                bank_res.bank_bytes = grafted_bytes
                assigned_graft_roles: set[str] = set()
                for og in all_graft_guids:
                    fin_g = g_map_nat.get(og)
                    if fin_g is None:
                        continue
                    rk = graft_roles.get(og, "")
                    if rk in ("Starter", "Starting", "NativeTransmission", "NativeGearshiftUp", "NativeGearshiftDown") and rk not in assigned_graft_roles:
                        assigned_graft_roles.add(rk)
                        fin_p = fin_paths_nat.get(fin_g, graft_paths_map.get(og, ""))
                        bank_res.role_to_event_guid[rk] = bytes_to_guid_str(fin_g)
                        bank_res.role_to_event_path[rk] = fin_p

    bank_rel_no_ext = f"Audio/AMS2/Vehicles/Mods/{safe_mod}/{safe_mod}_Engine"
    if csd_doc is not None and not standalone_guids and csd_doc.links.get("engine_bank"):
        # En modo reemplazo directo conservar la ruta de banco configurada en el .csdbin
        bank_rel_no_ext = csd_doc.links["engine_bank"].replace("\\", "/").removesuffix(".bank")

    bank_file = output_root / f"{bank_rel_no_ext}.bank"
    bank_file.parent.mkdir(parents=True, exist_ok=True)
    bank_file.write_bytes(bank_res.bank_bytes)

    # Compilación simultánea del banco de audio de IA y TV Replays (_AI.bank)
    ai_bank_res: PortedBankResult | None = None
    ai_bank_file: Path | None = None
    ai_bank_rel_no_ext: str | None = None

    if build_ai_bank:
        ai_bank_rel_no_ext = f"Audio/AMS2/Vehicles/Mods/{safe_mod}/{safe_mod}_Engine_AI"
        if direct_ac_bank is not None:
            has_large_pcm16 = (
                sum(len(sb.raw_fsb_bytes) for sb in direct_ac_bank.subbanks) > 20_000_000
                and any(sl.mode == FSB5_MODE_PCM16 for sl in direct_ac_bank.slots)
            )
            direct_codec = FSB5_MODE_FADPCM if has_large_pcm16 else None
            ai_bank_res = build_direct_ported_bank_from_ac(
                ac_bank=direct_ac_bank,
                mod_identifier=f"{safe_mod}_AI",
                standalone_guids=standalone_guids,
                rpm_bands=rpm_bands,
                is_ai_bank=True,
                mute_secondary_pops=mute_pops,
                invert_spatial_cone=False,
                mappings=mappings,
                output_codec_mode=direct_codec,
                prune_unused_samples=has_large_pcm16,
                cached_fsb_blobs=bank_res.fsb_blobs_cache if has_large_pcm16 else None,
            )
        elif donor is not None and donor.source_path is not None:
            companion_ai = donor.source_path.parent / f"{donor.source_path.stem}_AI.bank"
            if companion_ai.is_file():
                ai_donor = FmodBank(companion_ai)
                ai_mappings = auto_map_samples(ai_donor, library) if library else []
                ai_bank_res = build_ported_bank(
                    donor=ai_donor,
                    mappings=ai_mappings,
                    library=library,
                    mod_identifier=f"{safe_mod}_AI",
                    standalone_guids=standalone_guids,
                    sync_autopitch=sync_autopitch,
                    output_codec_mode=output_codec_mode,
                    force_sample_rate=force_sample_rate,
                )
            elif library and library.fmod_bank:
                ai_bank_res = build_direct_ported_bank_from_ac(
                    ac_bank=library.fmod_bank,
                    mod_identifier=f"{safe_mod}_AI",
                    standalone_guids=standalone_guids,
                    rpm_bands=rpm_bands,
                    is_ai_bank=True,
                    mute_secondary_pops=mute_pops,
                    invert_spatial_cone=False,
                )
        elif library and library.fmod_bank:
            ai_bank_res = build_direct_ported_bank_from_ac(
                ac_bank=library.fmod_bank,
                mod_identifier=f"{safe_mod}_AI",
                standalone_guids=standalone_guids,
                rpm_bands=rpm_bands,
                is_ai_bank=True,
                mute_secondary_pops=mute_pops,
                invert_spatial_cone=False,
            )

        if ai_bank_res is not None:
            ai_bank_file = output_root / f"{ai_bank_rel_no_ext}.bank"
            ai_bank_file.parent.mkdir(parents=True, exist_ok=True)
            ai_bank_file.write_bytes(ai_bank_res.bank_bytes)

    csdbin_file: Path | None = None
    csd_text_file: Path | None = None
    if csd_doc is not None:
        if auto_link_csdbin and standalone_guids:
            apply_ported_bank_to_csdbin(
                csd_doc=csd_doc,
                bank_result=bank_res,
                bank_rel_path_no_ext=bank_rel_no_ext,
                ai_bank_result=ai_bank_res,
                ai_bank_rel_path_no_ext=ai_bank_rel_no_ext,
                optimize_chase_cam=optimize_chase_cam,
                preserve_native_transmission=preserve_native_transmission,
            )
        # Conservar el nombre de archivo original del .csdbin si proviene de un vehículo mod existente
        # (p.ej. carpeta 'Vehicles/bmw_z4_gt3' con archivo 'z4_gt3.csdbin')
        orig_stem = ""
        if csd_doc.source_path is not None and not csd_doc.source_path.stem.lower().startswith("archivo"):
            orig_stem = "".join(
                c if (c.isalnum() or c in "_-") else "_" for c in csd_doc.source_path.stem.strip()
            )
        csd_file_stem = orig_stem or safe_veh
        csdbin_file = output_root / "Vehicles" / safe_veh / f"{csd_file_stem}.csdbin"
        csd_text_file = output_root / "Vehicles" / safe_veh / f"{csd_file_stem}.csd"
        csd_doc.save_csdbin(csdbin_file)
        csd_doc.save_csd(csd_text_file)

    manifest_data = {
        "mod_identifier": safe_mod,
        "vehicle_folder": safe_veh,
        "standalone_guids": standalone_guids,
        "codec": bank_res.codec_used,
        "bank_relative_path": f"{bank_rel_no_ext}.bank",
        "bank_guid": bank_res.new_bank_guid,
        "ai_bank_relative_path": f"{ai_bank_rel_no_ext}.bank" if ai_bank_file else None,
        "ai_bank_guid": ai_bank_res.new_bank_guid if ai_bank_res else None,
        "replaced_slots": bank_res.replaced_slots_count,
        "kept_slots": bank_res.kept_slots_count,
        "throttle_cut_injected_events": bank_res.throttle_cut_injected_events,
        "event_paths_by_role": bank_res.role_to_event_path,
        "events_by_role": bank_res.role_to_event_guid,
        "event_guid_remap": bank_res.event_guid_map,
    }
    manifest_file = output_root / "reporte_guid_eventos.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_lines = [
        "======================================================================",
        f"  PAQUETE DE SONIDO GENERADO POR AMS2 SOUND PORTER STUDIO ({safe_mod})",
        "======================================================================",
        "",
        "ARCHIVOS GENERADOS:",
        f"  1. Banco Jugador (STDT):       {bank_file.relative_to(output_root)}",
    ]
    if ai_bank_file:
        readme_lines.append(f"  2. Banco IA + TV Replays:      {ai_bank_file.relative_to(output_root)}")
    if csdbin_file:
        idx_c = 3 if ai_bank_file else 2
        readme_lines.append(f"  {idx_c}. Binario Auto (.csdbin):      {csdbin_file.relative_to(output_root)}")
        readme_lines.append(f"  {idx_c+1}. Texto OMVT (.csd):           {csd_text_file.relative_to(output_root)}")
    readme_lines += [
        "",
        "CÓMO INSTALAR EN UN VEHÍCULO MOD DE AUTOMOBILISTA 2:",
        "  1. Copia las carpetas 'Audio' y 'Vehicles' generadas dentro de la raíz de tu",
        "     Automobilista 2 (o dentro de la carpeta raíz de tu mod antes de activarlo).",
        "  2. Asegúrate de que el nombre del archivo .csdbin coincida con el que usa",
        "     tu vehículo mod en 'Vehicles/<carpeta_del_mod>/'.",
        "",
        "RUTAS Y EVENTOS FMOD REGISTRADOS EN EL BANCO (STDT):",
    ]
    for role, gstr in bank_res.role_to_event_guid.items():
        ev_p = bank_res.role_to_event_path.get(role, "")
        readme_lines.append(f"  - {role:12s} : {ev_p}  ->  {gstr}")
    readme_lines.append("")

    readme_file = output_root / "LEEME_INSTALACION_MOD.txt"
    readme_file.write_text("\n".join(readme_lines), encoding="utf-8")

    return ModExportSummary(
        output_root=output_root,
        bank_file=bank_file,
        csdbin_file=csdbin_file,
        csd_text_file=csd_text_file,
        readme_file=readme_file,
        manifest_file=manifest_file,
        bank_result=bank_res,
        ai_bank_file=ai_bank_file,
        ai_bank_result=ai_bank_res,
    )
