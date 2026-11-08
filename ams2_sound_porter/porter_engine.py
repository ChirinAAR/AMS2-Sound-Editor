"""Motor de porteo AC -> AMS2: Auto-Mapeo FMOD, Reconstrucción FSB5 y Remapeo Standalone de GUIDs."""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_dsp import AudioBuffer, conform_audio, read_wav_file
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
from .fmod_stdt import build_stdt_chunk, lookup_stdt_id, parse_stdt_sections
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

        # 2. Starter / Starting / ThrottleCut existentes en el donante cuando AC no tiene equivalente
        if sl.load_role == "starter" and not has_starter_in_src and sl.mode != FSB5_MODE_VORBIS:
            cfg.action = "keep_donor"
            cfg.match_reason = "Sin starter en AC (se conserva donante AMS2)"
            mappings_by_idx[sl.global_index] = cfg
            continue
        if sl.load_role == "starting" and not has_starting_in_src and sl.mode != FSB5_MODE_VORBIS:
            cfg.action = "keep_donor"
            cfg.match_reason = "Sin encendido en AC (se conserva donante AMS2)"
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
) -> bytes:
    """Retorna los sub-chunks hijos directos de LIST:PROJ ya parcheados (excepto SNDH que se calcula al final)."""
    buf = bytearray(proj_payload)

    # Preparar tablas de parcheo de bandas RPM y curvas de crossfade (antes de remapear GUIDs)
    pmlb_box_updates: dict[tuple[bytes, bytes], list[RpmBandConfig]] = {}
    fade_in_curv_updates: dict[bytes, tuple[float, float, float]] = {}
    fade_out_curv_updates: dict[bytes, tuple[float, float, float]] = {}
    bus_cutoff_max_rpm: dict[bytes, float] = {}

    if rpm_bands:
        for b in rpm_bands:
            rmin = max(0.0, float(b.rpm_min))
            rmax = max(rmin + 10.0, float(b.rpm_max))
            fi_end = max(rmin, min(rmax, float(b.fade_in_end)))
            fo_start = max(rmin, min(rmax, float(b.fade_out_start)))
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
    if autopitch_updates:
        def collect_waibs(start: int, end: int) -> None:
            for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
                if cid == b"WAIB" and pe - ps >= 32:
                    waib_to_wav[bytes(buf[ps : ps + 16])] = bytes(buf[ps + 16 : ps + 32])
                elif cid == b"LIST":
                    collect_waibs(ps + 4, pe)

        collect_waibs(0, len(buf))

    # 1. Aplicar redirecciones WAV, Auto-Pitch RPM y edición de bandas PMLB / CURV ANTES de remapear los GUIDs
    def patch_tree(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(buf, start, end):
            if cid == b"WAV " and pe - ps >= 30:
                wg = bytes(buf[ps : ps + 16])
                if wg in wav_redirects:
                    new_sb, new_smp_idx = wav_redirects[wg]
                    _, _, flags, _, _ = struct.unpack_from("<HHHII", buf, ps + 16)
                    struct.pack_into("<HHHII", buf, ps + 16, 0x000C, new_sb, flags, new_smp_idx, new_sb)
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
                        if fi_end <= rmin + 1.0:
                            struct.pack_into("<ff", buf, ps + 36, rmin, 1.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, min(rmax, rmin + 1.0), 1.0)
                        else:
                            struct.pack_into("<ff", buf, ps + 36, rmin, 0.0)
                            struct.pack_into("<ff", buf, ps + 36 + stride, fi_end, 1.0)
                    elif cg in fade_out_curv_updates:
                        rmin, fo_start, rmax = fade_out_curv_updates[cg]
                        if fo_start >= rmax - 1.0:
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
                        elif scid == b"INST" and spe - sps >= 73:
                            inst_span = (sps, spe)
                    if target_wg and inst_span and target_wg in autopitch_updates:
                        ips, _ = inst_span
                        if bytes(buf[ips + 53 : ips + 69]) != b"\x00" * 16:
                            struct.pack_into("<f", buf, ips + 69, float(autopitch_updates[target_wg]))
                    patch_tree(ps + 4, pe)
                elif ltype == b"MUIT" and autopitch_updates:
                    child_rpms: list[float] = []
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
                                            if ch_wg and ch_wg in autopitch_updates:
                                                child_rpms.append(float(autopitch_updates[ch_wg]))
                        elif scid == b"INST" and spe - sps >= 73:
                            inst_span = (sps, spe)
                    if child_rpms and inst_span:
                        ips, _ = inst_span
                        if bytes(buf[ips + 53 : ips + 69]) != b"\x00" * 16:
                            avg_rpm = sum(child_rpms) / len(child_rpms)
                            struct.pack_into("<f", buf, ips + 69, float(avg_rpm))
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


def _sort_evts_in_proj_payload(buf: bytearray) -> bytearray:
    """Reordena los chunks LIST:EVNT dentro de LIST:EVTS en orden ascendente por su GUID EVTB.

    Garantiza que las tablas FMOD mantengan orden memcmp estricto para binary search (bsearch)
    cuando se reasignan GUIDs nativos de vehículos vanilla o GUIDs personalizados.
    """
    for cid, ps, pe, pos in iter_riff_chunks(buf, 0, len(buf)):
        if cid == b"LIST" and buf[ps : ps + 4] == b"EVTS":
            children: list[tuple[bytes | None, bytes]] = []
            header = bytearray()
            for ev_cid, ev_ps, ev_pe, ev_pos in iter_riff_chunks(buf, ps + 4, pe):
                ev_pad_end = ev_pe + ((ev_pe - ev_ps) & 1)
                if ev_cid == b"LCNT":
                    header.extend(buf[ev_pos:ev_pad_end])
                    continue
                chunk_full = bytes(buf[ev_pos:ev_pad_end])
                ev_guid = None
                for s_cid, s_ps, s_pe, _ in iter_riff_chunks(buf, ev_ps + 4, ev_pe):
                    if s_cid in LOCAL_OBJECT_BASE_CHUNKS and s_pe - s_ps >= 16:
                        ev_guid = bytes(buf[s_ps : s_ps + 16])
                        break
                children.append((ev_guid, chunk_full))
            if any(g is not None for g, _ in children):
                children.sort(key=lambda x: x[0] if x[0] is not None else b"\xff" * 16)
                new_payload = b"EVTS" + header + b"".join(c for _, c in children)
                new_chunk = b"LIST" + struct.pack("<I", len(new_payload)) + new_payload
                if len(new_payload) & 1:
                    new_chunk += b"\x00"
                pad_end = pe + ((pe - ps) & 1)
                return bytearray(buf[:pos] + new_chunk + buf[pad_end:])
    return buf


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
            if ig in ibg_to_target_bus:
                new_pay[18:34] = ibg_to_target_bus[ig]
            out += _riff_chunk(b"IBSB", bytes(new_pay))
        else:
            out += _riff_chunk(cid, pay)
    return bytes(out)


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
    native_bank_path: Path | str | None = None,
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

    # 1. Asociar cada evento de AC a su bus correspondiente en AMS2 (Exterior o Cockpit)
    ibg_to_target_bus: dict[bytes, bytes] = {}
    for ev in ac_bank.events:
        if not ev.ibus_guid_bytes:
            continue
        p_low = (ev.path or "").lower()
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

    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_mod"
    guid_map: dict[bytes, bytes] | None = None
    event_guid_map: dict[str, str] = {}
    role_to_event_guid: dict[str, str] = {}
    role_to_event_path: dict[str, str] = {}
    stdt_entries: list[tuple[str, bytes]] = []

    if target_role_guids:
        guid_map = {}
        used_guids: set[bytes] = set()

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
        p_ext = ext_cands[0] if ext_cands else next((e for e in ac_bank.events if e.role == "EngineExt"), None)

        int_cands = [
            e for e in ac_bank.events
            if _is_engine_cand(e) and (e.role == "EngineInt" or any(k in (e.path or "").lower() for k in ("_int", "int_", "cockpit", "cabin")))
        ]
        int_cands.sort(key=lambda x: -len(x.linked_samples))
        p_int = int_cands[0] if int_cands else next((e for e in ac_bank.events if e.role == "EngineInt"), None)

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

        # Comprobar si AC tiene evento de encendido / starter
        starter_ac_ev = next(
            (ev for ev in ac_bank.events if any(k in (ev.path or "").lower() or k in (ev.role or "").lower() for k in ("starter", "starting", "startup", "start_ext", "start_int", "ignition"))),
            None,
        )
        starter_roles = [r for r in target_role_guids if r in ("Starter", "Starting")]

        if starter_ac_ev and starter_roles:
            for s_role in starter_roles:
                s_bg = guid_str_to_bytes(target_role_guids[s_role])
                guid_map[starter_ac_ev.guid_bytes] = s_bg
                used_guids.add(s_bg)
                role_to_event_guid[s_role] = bytes_to_guid_str(s_bg)
                s_path = (target_role_paths or {}).get(s_role) or (starter_ac_ev.path or f"event:/AMS2/Vehicles/{safe_mod}/{s_role}")
                role_to_event_path[s_role] = s_path
                stdt_entries.append((s_path, s_bg))
        elif starter_roles:
            st_guids_to_extract = [guid_str_to_bytes(target_role_guids[r]) for r in starter_roles]
            evnts = []
            ibses = []
            if native_bank_path and Path(native_bank_path).is_file():
                evnts, ibses = extract_native_starters(Path(native_bank_path), st_guids_to_extract)
            if not evnts:
                MISC_BUS = guid_str_to_bytes("{ffefac1b-0333-46ad-9d94-c7ae1494451f}")
                for s_role in starter_roles:
                    s_bg = guid_str_to_bytes(target_role_guids[s_role])
                    s_ibg = hashlib.sha256(s_bg + b"_ibus_ams2").digest()[:16]
                    evnts.append((s_bg, _make_trigger_evnt(s_bg, s_ibg, MISC_BUS)))
                    ibses.append((s_ibg, _make_trigger_ibus(s_ibg, MISC_BUS)))
            if evnts:
                proj_buf = bytearray(inject_native_chunks_in_proj(bytes(proj_buf), evnts, ibses))
                for s_role in starter_roles:
                    s_bg = guid_str_to_bytes(target_role_guids[s_role])
                    s_path = (target_role_paths or {}).get(s_role) or f"event:/AMS2/Vehicles/{safe_mod}/{s_role}"
                    role_to_event_guid[s_role] = bytes_to_guid_str(s_bg)
                    role_to_event_path[s_role] = s_path
                    stdt_entries.append((s_path, s_bg))

        # Evento de escape (Exhaust) si el objetivo lo requiere
        if "Exhaust" in target_role_guids and "Exhaust" not in role_to_event_guid:
            ex_bg = guid_str_to_bytes(target_role_guids["Exhaust"])
            ex_ev = next(
                (ev for ev in ac_bank.events if any(k in (ev.path or "").lower() for k in ("exhaust", "tailpipe", "muffler"))),
                None,
            )
            if not ex_ev:
                ex_ev = next(
                    (ev for ev in ac_bank.events if ev.role == "BackfireExt" or "backfire_ext" in (ev.path or "").lower()),
                    None,
                )
            if not ex_ev and p_ext:
                ex_ev = next(
                    (ev for ev in ext_cands if ev.index != (p_ext.index if p_ext else -1)),
                    None,
                )
            if ex_ev and ex_ev.guid_bytes not in guid_map:
                guid_map[ex_ev.guid_bytes] = ex_bg
                used_guids.add(ex_bg)
                role_to_event_guid["Exhaust"] = bytes_to_guid_str(ex_bg)
                ex_path = (target_role_paths or {}).get("Exhaust") or (ex_ev.path or f"event:/AMS2/Vehicles/{safe_mod}/Exhaust")
                role_to_event_path["Exhaust"] = ex_path
                stdt_entries.append((ex_path, ex_bg))

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
        for ev in ac_bank.events:
            ng_bytes = guid_map.get(ev.guid_bytes, ev.guid_bytes)
            ng = bytes_to_guid_str(ng_bytes)
            event_guid_map[ev.guid_str] = ng
            role_to_event_guid[ev.role] = ng

            if is_ai_bank and ev.role == "EngineExt":
                live_path = f"event:/AMS2/Vehicles/Mods/{mod_prefix}/AI_EngineMAIN_LIVE"
                main_path = f"event:/AMS2/Vehicles/Mods/{mod_prefix}/AI_EngineMAIN"
                stdt_entries.append((live_path, ng_bytes))
                stdt_entries.append((main_path, ng_bytes))
                role_to_event_path["EngineExt"] = main_path
                role_to_event_path["ai_engine_ext"] = main_path
                role_to_event_path["ai_engine_live"] = live_path
            elif is_ai_bank and ev.role == "EngineInt":
                int_path = f"event:/AMS2/Vehicles/Mods/{mod_prefix}/AI_EngineInt"
                stdt_entries.append((int_path, ng_bytes))
                role_to_event_path["EngineInt"] = int_path
                role_to_event_path["ai_engine_int"] = int_path
            else:
                last_part = ev.path.split("/")[-1] if ev.path else ev.role
                clean_ev_path = f"event:/AMS2/Vehicles/Mods/{safe_mod}/{last_part}"
                if clean_ev_path.lower() in used_paths:
                    clean_ev_path = f"{clean_ev_path}_{ev.index}"
                used_paths.add(clean_ev_path.lower())
                role_to_event_path[ev.role] = clean_ev_path
                stdt_entries.append((clean_ev_path, ng_bytes))
    else:
        new_bank_guid = bytes_to_guid_str(ac_bank.bank_guid)
        for ev in ac_bank.events:
            event_guid_map[ev.guid_str] = ev.guid_str
            role_to_event_guid[ev.role] = ev.guid_str
            role_to_event_path[ev.role] = ev.path or f"event:/AMS2/Vehicles/Mods/{safe_mod}/{ev.role}"

    new_stdt_payload = build_stdt_chunk(stdt_entries) if stdt_entries else None

    # Parchear PROJ
    patched_proj_body = _patch_proj_chunks(
        proj_payload=proj_buf,
        wav_redirects={},
        autopitch_updates={},
        guid_map=guid_map,
        rpm_bands=rpm_bands,
    )

    final_fsb_blobs = [sb.raw_fsb_bytes for sb in ac_bank.subbanks]
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
    full_riff = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"FEV " + body

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
        codec_used="Nativo AC (Transmutación Directa Sin Donante)",
        throttle_cut_injected_events=0,
    )


def apply_ported_bank_to_csdbin(
    csd_doc: CsdbinDocument,
    bank_result: PortedBankResult,
    bank_rel_path_no_ext: str,
    ai_bank_result: PortedBankResult | None = None,
    ai_bank_rel_path_no_ext: str | None = None,
    optimize_chase_cam: bool = True,
) -> CsdbinDocument:
    """Actualiza los enlaces del documento .csdbin con las rutas `event:/...` registradas en STDT."""
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

    # Enlaces de caja de cambios (gearshift)
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
    grind_path = rpaths.get("gear_grind") or rpaths.get("GearshiftGrind")
    if grind_path:
        csd_doc.links["gearshift_grind"] = grind_path

    # Enlaces de turbo / sobrealimentador
    turbo_path = rpaths.get("Turbo") or rpaths.get("turbo")
    if turbo_path:
        csd_doc.links["turbo_bank"] = clean_bank_path
        csd_doc.links["turbo"] = turbo_path

    # Enlaces de transmisión / diferencial
    trans_path = rpaths.get("Transmission") or rpaths.get("transmission") or rpaths.get("transmission_ext")
    if trans_path:
        csd_doc.links["transmission_bank"] = clean_bank_path
        csd_doc.links["transmission"] = trans_path

    # Enlaces de limitador de revoluciones
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

    # Enlaces de viento
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
) -> ModExportSummary:
    """Compila el .bank y el .csdbin listos para copiar en Automobilista 2 / AMS2 Content Manager."""
    output_root = Path(output_root)
    safe_mod = "".join(c if (c.isalnum() or c in "_-") else "_" for c in mod_identifier.strip()) or "custom_sound"
    safe_veh = "".join(c if (c.isalnum() or c in "_-") else "_" for c in vehicle_folder_name.strip()) or safe_mod

    if direct_ac_bank is not None:
        bank_res = build_direct_ported_bank_from_ac(
            ac_bank=direct_ac_bank,
            mod_identifier=safe_mod,
            standalone_guids=standalone_guids,
            rpm_bands=rpm_bands,
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
            ai_bank_res = build_direct_ported_bank_from_ac(
                ac_bank=direct_ac_bank,
                mod_identifier=f"{safe_mod}_AI",
                standalone_guids=standalone_guids,
                rpm_bands=rpm_bands,
                is_ai_bank=True,
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
                )
        elif library and library.fmod_bank:
            ai_bank_res = build_direct_ported_bank_from_ac(
                ac_bank=library.fmod_bank,
                mod_identifier=f"{safe_mod}_AI",
                standalone_guids=standalone_guids,
                rpm_bands=rpm_bands,
                is_ai_bank=True,
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
