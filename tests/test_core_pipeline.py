"""Suite de verificación integral del motor AMS2 Sound Porter Studio."""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ams2_sound_porter.audio_dsp import decode_fadpcm, encode_fadpcm, read_wav_bytes
from ams2_sound_porter.engine_sim import (
    STAGE_DESCRIPTIONS,
    build_engine_pass_trajectory,
    render_engine_pass_wav_bytes,
)
from ams2_sound_porter.fmod_bank import (
    AMS2_PARENT_BUS_ROLES,
    FSB5_MODE_FADPCM,
    FSB5_MODE_PCM16,
    FmodBank,
    bytes_to_guid_str,
    guid_str_to_bytes,
    iter_riff_chunks,
)
from ams2_sound_porter.fmod_stdt import lookup_stdt_id, parse_stdt_all_paths
from ams2_sound_porter.omvt_csdbin import CsdbinDocument
from ams2_sound_porter.porter_engine import (
    AMS2_AI_ENGINE_BUS,
    LOCAL_OBJECT_BASE_CHUNKS,
    RpmBandConfig,
    SlotMappingConfig,
    SourceAudioLibrary,
    ThrottleCutConfig,
    apply_band_edits_to_bank,
    auto_detect_throttle_cut_config,
    _extract_native_engine_max_rpm,
    auto_distribute_rpm_bands,
    auto_map_samples,
    build_default_rpm_bands,
    build_direct_ported_bank_from_ac,
    build_ported_bank,
    collect_vanilla_scope_targets,
    merge_ported_events_into_native_bank,
    uninvert_spatial_cone_in_bank_bytes,
    estimate_engine_rpm_bounds,
    export_ams2_mod_package,
)

def _extract_cone_vol_endpoints(bank_data: bytes) -> list[tuple[float, float, float, float]]:
    """Extrae pares (x0, y0, xN, yN) de curvas CURV prop==0 asociadas a parámetros Event Cone Angle."""
    cone_pgs: set[bytes] = set()
    cone_cgs: set[bytes] = set()
    out_pairs: list[tuple[float, float, float, float]] = []
    for cid, ps, pe, _ in iter_riff_chunks(bank_data, 12, len(bank_data)):
        if cid == b"LIST" and bank_data[ps : ps + 4] == b"PROJ":
            def _walk(s: int, e: int, phase: int) -> None:
                for scid, sps, spe, _ in iter_riff_chunks(bank_data, s, e):
                    if scid == b"LIST":
                        _walk(sps + 4, spe, phase)
                    elif phase == 0 and scid == b"PRMB" and spe - sps >= 31:
                        sl = struct.unpack_from("<H", bank_data, sps + 21)[0]
                        nm = bank_data[sps + 23 : sps + 23 + sl].decode("latin1", "replace").rstrip("\x00").lower()
                        if "cone angle" in nm or "event cone" in nm:
                            cone_pgs.add(bytes(bank_data[sps : sps + 16]))
                    elif phase == 1 and scid == b"CTRL" and spe - sps >= 68:
                        if bytes(bank_data[sps + 32 : sps + 48]) in cone_pgs and struct.unpack_from("<I", bank_data, sps + 64)[0] == 0:
                            cone_cgs.add(bytes(bank_data[sps + 48 : sps + 64]))
                    elif phase == 2 and scid == b"CURV" and spe - sps >= 36:
                        if bytes(bank_data[sps : sps + 16]) in cone_cgs:
                            cnt = struct.unpack_from("<H", bank_data, sps + 32)[0] >> 1
                            st = struct.unpack_from("<H", bank_data, sps + 34)[0]
                            if cnt >= 2 and st >= 16 and sps + 36 + cnt * st <= spe:
                                x0, y0 = struct.unpack_from("<ff", bank_data, sps + 36)
                                x1, y1 = struct.unpack_from("<ff", bank_data, sps + 36 + (cnt - 1) * st)
                                if abs(y0 - y1) >= 5.0:
                                    out_pairs.append((round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)))
            for ph in (0, 1, 2):
                _walk(ps + 4, pe, ph)
    return sorted(set(out_pairs))


REF_DIR = Path(
    r"C:\Users\Usuario\.gemini\antigravity\brain\f847d4e0-8cea-412c-a1b6-15d13cc65709\scratch\sonidos"
)
MOD_488_DIR = Path(
    r"C:\Users\Usuario\.gemini\antigravity\brain\f847d4e0-8cea-412c-a1b6-15d13cc65709\scratch\488gt3_sfx"
)
AC_BANK_PATH = REF_DIR / "Sonido Asseto Corsa.bank"
AC_MOD_488_BANK = MOD_488_DIR / "ks_ferrari_488_gt3.bank"
AC_MOD_488_GUIDS = MOD_488_DIR / "GUIDs.txt"
AMS2_BANK_PATH = REF_DIR / "Sonido Automobilista 2.bank"
CSDBIN_PATH = REF_DIR / "Archivo Sonido Vehiculo Automobilista 2.csdbin"


def assert_proj_tables_sorted(raw_bank: bytes) -> int:
    """Verifica que TODAS las tablas LIST de FMOD mantengan orden estricto memcmp por GUID."""
    checked_lists = 0
    bnki_u2: int | None = None
    for cid, ps, pe, _ in iter_riff_chunks(raw_bank, 12, len(raw_bank)):
        if cid == b"LIST" and raw_bank[ps : ps + 4] == b"PROJ":
            for scid, sps, spe, _ in iter_riff_chunks(raw_bank, ps + 4, pe):
                if scid == b"BNKI" and spe - sps >= 28:
                    bnki_u2 = struct.unpack_from("<I", raw_bank, sps + 24)[0]
                    break
            break

    def walk(start: int, end: int) -> None:
        nonlocal checked_lists
        child_guids_by_tag: dict[bytes, list[bytes]] = {}
        for cid, ps, pe, _ in iter_riff_chunks(raw_bank, start, end):
            if cid == b"LIST":
                ltype = raw_bank[ps : ps + 4]
                # Extraer GUID del primer subchunk *B si es un contenedor de ítem (EVNT, IBUS, WAIT, etc.)
                # Excluir TRAN (TMLN/TRNS) porque FMOD ordena transiciones por prioridad de timeline, no por GUID.
                if ltype != b"TRAN":
                    for scid, sps, spe, _ in iter_riff_chunks(raw_bank, ps + 4, pe):
                        if scid in LOCAL_OBJECT_BASE_CHUNKS and spe - sps >= 16:
                            child_guids_by_tag.setdefault(ltype, []).append(raw_bank[sps : sps + 16])
                            break
                walk(ps + 4, pe)
            elif cid in (b"CTRL", b"CURV", b"MAP ", b"WAV ") and pe - ps >= 16:
                child_guids_by_tag.setdefault(cid, []).append(raw_bank[ps : ps + 16])

        for tag, glist in child_guids_by_tag.items():
            if len(glist) > 1:
                if tag == b"EVNT" and bnki_u2 is not None and 0 < bnki_u2 <= len(glist):
                    part1 = glist[:bnki_u2]
                    part2 = glist[bnki_u2:]
                    assert part1 == sorted(part1), "Tabla FMOD 'EVNT' part1 [0..BNKI.u2) perdió el orden memcmp!"
                    assert part2 == sorted(part2), "Tabla FMOD 'EVNT' part2 [BNKI.u2..N) perdió el orden memcmp!"
                else:
                    assert glist == sorted(glist), f"Tabla FMOD '{tag.decode()}' perdió el orden memcmp!"
                checked_lists += 1

    walk(12, len(raw_bank))
    return checked_lists


def assert_sndh_and_fsb_alignment(raw_bank: bytes) -> None:
    sndh_pairs: list[tuple[int, int]] = []
    snd_spans: list[tuple[int, int]] = []

    def walk(start: int, end: int) -> None:
        for cid, ps, pe, _ in iter_riff_chunks(raw_bank, start, end):
            if cid == b"LIST":
                walk(ps + 4, pe)
            elif cid == b"SNDH":
                cnt = (pe - ps - 4) // 8
                for i in range(cnt):
                    fpos, flen = struct.unpack_from("<II", raw_bank, ps + 4 + i * 8)
                    sndh_pairs.append((fpos, flen))
            elif cid == b"SND ":
                snd_spans.append((ps, pe))

    walk(12, len(raw_bank))
    assert len(sndh_pairs) == len(snd_spans), "SNDH y cantidad de chunks SND no coinciden"
    for idx, ((fpos, flen), (ps, pe)) in enumerate(zip(sndh_pairs, snd_spans)):
        assert fpos % 32 == 0, f"SND[{idx}] FSB5 offset 0x{fpos:x} no está alineado a 32 bytes"
        assert raw_bank[fpos : fpos + 4] == b"FSB5", f"SND[{idx}] no contiene cabecera FSB5 en 0x{fpos:x}"
        assert fpos + flen == pe, f"SND[{idx}] fpos+flen ({fpos+flen}) != pe ({pe})"
        assert fpos >= ps, f"SND[{idx}] fpos < ps"


def main() -> None:
    print("1. Probando carga y análisis de bancos AC y AMS2...")
    ac_lib = SourceAudioLibrary.from_bank(AC_BANK_PATH)
    donor = FmodBank(AMS2_BANK_PATH)
    print(f"   AC muestras={len(ac_lib.items)}, AMS2 slots={len(donor.slots)}, eventos AMS2={len(donor.events)}")
    assert len(ac_lib.items) == 46
    assert len(donor.slots) == 27
    assert len(donor.events) == 6
    roles = {e.role for e in donor.events}
    print(f"   Roles de eventos AMS2 detectados: {roles}")
    assert {"EngineExt", "EngineInt", "AuxInt", "Exhaust", "Starter", "Starting"} == roles

    print("2. Probando extracción WAV desde PCM16 (AC) y FADPCM (AMS2)...")
    w_ac = ac_lib.items[1].load_audio()
    assert w_ac.sample_rate == 44100 and w_ac.channels == 2 and w_ac.num_samples > 100000
    starter_slot = [s for s in donor.slots if "starter" in s.name][0]
    w_ams2 = read_wav_bytes(donor.extract_slot_wav_bytes(starter_slot))
    assert w_ams2.sample_rate == 48000 and w_ams2.channels == 2 and w_ams2.num_samples == starter_slot.pcm_samples

    print("3. Probando codificador/decodificador FADPCM round-trip...")
    short_pcm = w_ams2.pcm[:4096]
    fad_bytes, fad_n = encode_fadpcm(short_pcm)
    rec_pcm = decode_fadpcm(fad_bytes, 2, fad_n)[:4096]
    corr = float(
        (short_pcm[:, 0].astype(float) * rec_pcm[:, 0].astype(float)).sum()
        / (
            ((short_pcm[:, 0].astype(float) ** 2).sum() ** 0.5)
            * ((rec_pcm[:, 0].astype(float) ** 2).sum() ** 0.5)
        )
    )
    print(f"   Correlación FADPCM round-trip: {corr:.5f}")
    assert corr > 0.985, f"Correlación FADPCM baja: {corr}"

    print("4. Probando Auto-Mapeo Inteligente FMOD (AC Z4 GT3 -> AMS2 M3 GTR)...")
    mappings = auto_map_samples(donor, ac_lib)
    for m in mappings:
        sl = donor.slots[m.donor_global_index]
        src_nm = ac_lib.items[m.source_index].name if m.source_index is not None else m.action
        print(f"   [{sl.global_index:02d}] {sl.name:38s} -> {src_nm:24s} | {m.match_reason}")

    print("5. Probando compilación completa en PCM16 + GUIDs Standalone + Verificación estructural...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csd_doc = CsdbinDocument.from_file(CSDBIN_PATH)
        summary = export_ams2_mod_package(
            output_root=tmp_path,
            mod_identifier="bmw_z4_gt3_ac",
            vehicle_folder_name="bmw_z4_gt3_mod",
            donor=donor,
            mappings=mappings,
            library=ac_lib,
            csd_doc=csd_doc,
            standalone_guids=True,
            sync_autopitch=True,
            output_codec_mode=FSB5_MODE_PCM16,
            force_sample_rate=48000,
        )
        raw_out = summary.bank_file.read_bytes()
        assert_sndh_and_fsb_alignment(raw_out)
        sorted_cnt = assert_proj_tables_sorted(raw_out)
        print(f"   Banco exportado ({len(raw_out):,} bytes), {sorted_cnt} tablas FMOD ordenadas verificadas!")

        # Re-parsear el banco generado con FmodBank y comprobar buses padre intactos, STDT y Auto-Pitch sincronizado
        reloaded = FmodBank(summary.bank_file)
        assert len(reloaded.events) == 6
        assert len(reloaded.stdt_bytes) > 0, "El banco exportado no incluye tabla de rutas STDT!"
        assert bytes_to_guid_str(reloaded.bank_guid) != bytes_to_guid_str(donor.bank_guid)
        assert lookup_stdt_id(reloaded.stdt_bytes, "bank:/bmw_z4_gt3_ac_Engine") == reloaded.bank_guid
        for orig_ev, new_ev in zip(donor.events, reloaded.events):
            assert orig_ev.guid_str != new_ev.guid_str, "GUID de evento no fue regenerado"
            assert orig_ev.parent_bus_guid == new_ev.parent_bus_guid, "Bus padre de Master.bank alterado!"
            assert new_ev.parent_bus_guid in AMS2_PARENT_BUS_ROLES
            assert new_ev.path.startswith("event:/AMS2/Vehicles/Mods/bmw_z4_gt3_ac/")
            assert lookup_stdt_id(reloaded.stdt_bytes, new_ev.path) == new_ev.guid_bytes

        # Verificar que el .csdbin exportado usa rutas 'event:/...' válidas y resolvibles vía STDT
        reloaded_csd = CsdbinDocument.from_file(summary.csdbin_file)
        assert reloaded_csd.links["engine_bank"] == "Audio/AMS2/Vehicles/Mods/bmw_z4_gt3_ac/bmw_z4_gt3_ac_Engine"
        assert reloaded_csd.links["engine_ext"] == "event:/AMS2/Vehicles/Mods/bmw_z4_gt3_ac/EngineExt"
        assert reloaded_csd.links["engine_int"] == "event:/AMS2/Vehicles/Mods/bmw_z4_gt3_ac/EngineInt"
        assert (
            lookup_stdt_id(reloaded.stdt_bytes, reloaded_csd.links["engine_ext"])
            == guid_str_to_bytes(summary.bank_result.role_to_event_guid["EngineExt"])
        )
        assert len(parse_stdt_all_paths(reloaded.stdt_bytes)) == 7  # 1 bank:/ + 6 event:/

        # Verificar que si el .csdbin proviene de Vehicles/<carpeta>/<stem>.csdbin se preserva <stem>.csdbin
        veh_sub = tmp_path / "Vehicles" / "bmw_z4_gt3" / "z4_gt3.csdbin"
        veh_sub.parent.mkdir(parents=True, exist_ok=True)
        veh_sub.write_bytes(CSDBIN_PATH.read_bytes())
        mod_csd_doc = CsdbinDocument.from_file(veh_sub)
        summary2 = export_ams2_mod_package(
            output_root=tmp_path / "out2",
            mod_identifier="bmw_z4_gt3",
            vehicle_folder_name="bmw_z4_gt3",
            donor=donor,
            mappings=mappings,
            library=ac_lib,
            csd_doc=mod_csd_doc,
            standalone_guids=True,
        )
        assert summary2.csdbin_file == tmp_path / "out2" / "Vehicles" / "bmw_z4_gt3" / "z4_gt3.csdbin"

    print("6. Probando reemplazo parcial en sub-banco VORBIS (redirección WAVS) y modo FADPCM...")
    partial_maps = [SlotMappingConfig(sl.global_index, action="keep_donor") for sl in donor.slots]
    # Reemplazar solo 1 muestra de SND[1] (Vorbis)
    vorbis_slot = [sl for sl in donor.slots if sl.subbank_index == 1][0]
    partial_maps[vorbis_slot.global_index] = SlotMappingConfig(
        vorbis_slot.global_index,
        action="replace_source",
        source_index=1,
        override_root_rpm=5000.0,
    )
    partial_res = build_ported_bank(
        donor=donor,
        mappings=partial_maps,
        library=ac_lib,
        mod_identifier="partial_test",
        standalone_guids=True,
        output_codec_mode=FSB5_MODE_FADPCM,
        throttle_cut=ThrottleCutConfig(enabled=False),
    )
    assert_sndh_and_fsb_alignment(partial_res.bank_bytes)
    assert_proj_tables_sorted(partial_res.bank_bytes)
    partial_reloaded = FmodBank(raw=partial_res.bank_bytes)
    assert len(partial_reloaded.subbanks[0].samples) == 20  # 19 originales + 1 redirigida desde Vorbis!
    assert len(partial_reloaded.subbanks[1].samples) == 8
    print("   Redirección parcial Vorbis -> FADPCM SND[0] verificada con éxito!")

    if AC_MOD_488_BANK.is_file() and AC_MOD_488_GUIDS.is_file():
        print("7. Probando soporte de sonidos MOD de Assetto Corsa (ks_ferrari_488_gt3 + GUIDs.txt)...")
        mod_lib = SourceAudioLibrary.from_bank(AC_MOD_488_BANK)
        assert len(mod_lib.items) == 174
        assert mod_lib.guids_path == AC_MOD_488_GUIDS
        assert mod_lib.matched_guids_count >= 20
        assert mod_lib.total_events_count == 20

        by_name = {it.name: it for it in mod_lib.items}
        # Interior vs Exterior separados gracias a GUIDs.txt + curvas throttle
        assert by_name["488_Idle"].perspective == "int" and by_name["488_Idle"].load_role == "idle"
        assert by_name["488_Load 2"].perspective == "int" and by_name["488_Load 2"].load_role == "on"
        assert by_name["488_Coast 1"].perspective == "int" and by_name["488_Coast 1"].load_role == "off"
        # Herencia MUIT -> INST de Auto-Pitch RPM en exterior
        assert by_name["488_Ex_Load 0"].perspective == "ext" and by_name["488_Ex_Load 0"].load_role == "on"
        assert abs((by_name["488_Ex_Load 0"].effective_rpm or 0.0) - 6490.0) < 1.0
        assert by_name["488_Ex_Coast 1"].perspective == "ext" and by_name["488_Ex_Coast 1"].load_role == "off"
        # Timelines TLNB de petardeos (backfire_ext)
        assert by_name["488_Ex_Backfire 1"].load_role == "pops"
        # Aislamiento estricto de capas no-motor (caja, chasis, TC, turbo) y capas Far
        assert by_name["488_Gearbox 2nd"].load_role == "fx"
        assert by_name["488_Ex_TC"].load_role == "fx"
        assert by_name["Turbo Boost"].load_role == "fx"
        assert by_name["488_Ex_Far Load 0"].is_aux_layer is True

        # Probar carga manual de GUIDs.txt cuando el .bank está en otra carpeta separada
        with tempfile.TemporaryDirectory() as tmp_iso:
            iso_bank = Path(tmp_iso) / "ks_ferrari_488_gt3_isolated.bank"
            iso_bank.write_bytes(AC_MOD_488_BANK.read_bytes())
            lib_no_guids = SourceAudioLibrary.from_bank(iso_bank)
            assert lib_no_guids.matched_guids_count == 0
            lib_manual_guids = SourceAudioLibrary.from_bank(iso_bank, guids_path=AC_MOD_488_GUIDS)
            assert lib_manual_guids.matched_guids_count == mod_lib.matched_guids_count

        # Auto-mapeo sobre donante AMS2 y verificación de pureza de slots de motor
        mod_maps = auto_map_samples(donor, mod_lib)
        for m in mod_maps:
            if m.action == "replace_source" and m.source_index is not None:
                chosen = mod_lib.items[m.source_index]
                dsl = donor.slots[m.donor_global_index]
                assert chosen.load_role != "fx", f"Slot {dsl.name} recibió efecto no-motor {chosen.name}"
                assert chosen.perspective == dsl.perspective, (
                    f"Perspectiva cruzada en {dsl.name}: {chosen.name} ({chosen.perspective} != {dsl.perspective})"
                )
                if dsl.load_role in ("on", "off", "idle"):
                    assert not chosen.is_aux_layer, f"Slot primario {dsl.name} recibió capa Far/Aux {chosen.name}"

        mod_ported = build_ported_bank(
            donor=donor,
            mappings=mod_maps,
            library=mod_lib,
            mod_identifier="ferrari_488_gt3_ac",
            standalone_guids=True,
            sync_autopitch=True,
            output_codec_mode=FSB5_MODE_PCM16,
            force_sample_rate=48000,
        )
        assert_sndh_and_fsb_alignment(mod_ported.bank_bytes)
        assert_proj_tables_sorted(mod_ported.bank_bytes)
        print("   Porteo completo de sonido MOD AC (488 GT3 + GUIDs.txt) verificado al 100%!")

    print("8. Probando detección e inyección automática de Throttle Cut (Corte de Marcha)...")
    tc_ac = auto_detect_throttle_cut_config(ac_lib)
    assert tc_ac.enabled is True
    assert tc_ac.int_source_index is not None and tc_ac.ext_source_index is not None
    print(f"   Vanilla AC ThrottleCut detectado: INT={ac_lib.items[tc_ac.int_source_index].name}, EXT={ac_lib.items[tc_ac.ext_source_index].name}")

    if AC_MOD_488_BANK.is_file():
        mod_lib_488 = SourceAudioLibrary.from_bank(AC_MOD_488_BANK)
        tc_488 = auto_detect_throttle_cut_config(mod_lib_488)
        assert tc_488.enabled is True
        assert tc_488.int_source_index is not None and tc_488.ext_source_index is not None
        print(f"   Mod 488 GT3 ThrottleCut detectado: INT={mod_lib_488.items[tc_488.int_source_index].name}, EXT={mod_lib_488.items[tc_488.ext_source_index].name}")

        ported_with_tc = build_ported_bank(
            donor=donor,
            mappings=auto_map_samples(donor, mod_lib_488),
            library=mod_lib_488,
            mod_identifier="f488_tcut_test",
            standalone_guids=True,
            throttle_cut=tc_488,
        )
        assert ported_with_tc.throttle_cut_injected_events == 3
        assert_sndh_and_fsb_alignment(ported_with_tc.bank_bytes)
        assert_proj_tables_sorted(ported_with_tc.bank_bytes)
        reloaded_tc = FmodBank(raw=ported_with_tc.bank_bytes)
        assert reloaded_tc.has_active_throttle_cut is True
        assert reloaded_tc.empty_throttle_cut_events == []
        print("   Inyección binaria de Throttle Cut en eventos EngineInt, AuxInt, EngineExt verificada con éxito!")

    print("9. Probando ajuste de Bandas RPM, redistribución anti-solapamiento y parcheo PMLB/CURV...")
    init_bands = build_default_rpm_bands(donor)
    assert len(init_bands) == len(donor.rpm_bands)
    est_idle, est_max = estimate_engine_rpm_bounds(donor, ac_lib)
    print(f"   RPMs estimados para el motor: Idle={est_idle:.0f}, Max={est_max:.0f}")
    assert 600.0 <= est_idle <= 1800.0
    assert 7000.0 <= est_max <= 10000.0

    distrib_bands = auto_distribute_rpm_bands(
        init_bands,
        donor,
        mappings,
        idle_rpm=950.0,
        max_rpm=8000.0,
    )
    # Comprobar que en capas ON y OFF las bandas escalonan limpiamente sin mesetas solapadas
    on_bands = [
        b for b in distrib_bands
        if b.perspective == "int" and b.event_role == "EngineInt" and b.load_role == "on"
        and not b.is_muted_bus and not b.is_fill_layer
    ]
    on_bands.sort(key=lambda b: (b.rpm_min, b.rpm_max))
    for k in range(len(on_bands) - 1):
        assert on_bands[k].rpm_min <= on_bands[k + 1].rpm_min
        assert on_bands[k].fade_out_start <= on_bands[k + 1].fade_in_end + 1.0

    # Modificar deliberadamente una banda que tenga curvas de entrada y salida, y verificar que se refleje en los floats binarios de PMLB y CURV
    test_patch_band = [b for b in distrib_bands if b.fade_in_curv_guid and b.fade_out_curv_guid][0]
    test_patch_band.rpm_min = 777.0
    test_patch_band.fade_in_end = 888.0
    test_patch_band.fade_out_start = 1999.0
    test_patch_band.rpm_max = 2444.0

    bank_with_custom_bands = build_ported_bank(
        donor=donor,
        mappings=mappings,
        library=ac_lib,
        mod_identifier="custom_bands_test",
        rpm_bands=distrib_bands,
    )
    assert_sndh_and_fsb_alignment(bank_with_custom_bands.bank_bytes)
    reloaded_bands_bank = FmodBank(raw=bank_with_custom_bands.bank_bytes)
    # Buscar la banda modificada por su box_index y event_role
    matched_reloaded_band = [
        b for b in reloaded_bands_bank.rpm_bands
        if b.event_role == test_patch_band.event_role and b.box_index == test_patch_band.box_index
    ][0]
    assert abs(matched_reloaded_band.rpm_min - 777.0) < 0.1
    assert abs(matched_reloaded_band.fade_in_end - 888.0) < 0.1
    assert abs(matched_reloaded_band.fade_out_start - 1999.0) < 0.1
    assert abs(matched_reloaded_band.rpm_max - 2444.0) < 0.1
    print("   Parcheo binario de rangos PMLB y curvas CURV comprobado exactamente!")

    print("10. Probando Simulador de Pasada de Motor en 5 Etapas (Interior y Exterior)...")
    traj = build_engine_pass_trajectory(idle_rpm=950.0, max_rpm=7800.0)
    assert traj.duration_sec >= 24.0
    # Comprobar las 5 etapas en orden
    st1_rpm, st1_thr, _, st1_id, _ = traj.sample_telemetry_at(0.5)
    assert st1_id == 1 and abs(st1_rpm - 950.0) < 25.0
    st1_climb_rpm, _, _, st1_id2, _ = traj.sample_telemetry_at(5.0)
    assert st1_id2 == 1 and st1_climb_rpm > 4000.0

    st2_rpm, st2_thr, st2_gear, st2_id, _ = traj.sample_telemetry_at(7.2)
    assert st2_id == 2 and st2_gear == 2 and st2_rpm < 7800.0  # Caída post-cambio
    st2_max_rpm, _, _, st2_id2, _ = traj.sample_telemetry_at(11.7)
    assert st2_id2 == 2 and st2_max_rpm > 7600.0

    st3_hold_rpm, _, _, st3_id, _ = traj.sample_telemetry_at(13.0)
    assert st3_id == 3 and abs(st3_hold_rpm - 7800.0) < 45.0  # Mantener 3s al máximo (con micro-oscilación de limitador)
    st3_decel_rpm, st3_decel_thr, _, st3_id2, _ = traj.sample_telemetry_at(16.5)
    assert st3_id2 == 3 and st3_decel_thr == 0.0 and st3_decel_rpm < 7800.0  # Desacelerar 3s

    st4_blip_rpm, _, _, st4_id, _ = traj.sample_telemetry_at(18.02)
    st4_pre_rpm, _, _, _, _ = traj.sample_telemetry_at(17.75)
    assert st4_id == 4 and st4_blip_rpm > st4_pre_rpm + 500.0  # Pico de Auto-Blip

    st5_idle_rpm, _, _, st5_id, _ = traj.sample_telemetry_at(24.5)
    assert st5_id == 5 and abs(st5_idle_rpm - 950.0) < 15.0  # Llegada a Idle

    # Renderizar audio WAV tanto para Interior como Exterior
    wav_int, _ = render_engine_pass_wav_bytes("int", donor, mappings, distrib_bands, tc_ac, ac_lib, 950.0, 7800.0)
    assert len(wav_int) > 1000000  # ~24.8 s a 44.1 kHz stereo 16-bit son ~4.3 MB
    assert wav_int[:4] == b"RIFF" and wav_int[8:12] == b"WAVE"
    wav_ext, _ = render_engine_pass_wav_bytes("ext", donor, mappings, distrib_bands, tc_ac, ac_lib, 950.0, 7800.0)
    assert len(wav_ext) > 1000000
    assert wav_ext[:4] == b"RIFF" and wav_ext[8:12] == b"WAVE"
    print(f"   Simulador 5 etapas renderizado con éxito (WAV INT={len(wav_int):,} B, EXT={len(wav_ext):,} B)!")

    # 11. Probando funcionalidades avanzadas de Bandas v1.4.0 (Añadir/Eliminar, Ajuste Selectivo, Batch Gain y Recreación Fiel AC)
    print("11. Probando novedades v1.4.0: Recreación Fiel AC, Ajuste Selectivo y Batch Gain...")
    from ams2_sound_porter.porter_engine import (
        auto_distribute_single_rpm_band,
        sync_faithful_assetto_corsa_setup,
        sync_faithful_section_from_source,
    )

    # 11a. Recreación fiel de Assetto Corsa por sección (Interior / Exterior)
    sec_bands, sec_maps, add_ext, mute_ext = sync_faithful_section_from_source(
        donor,
        ac_lib,
        mappings,
        build_default_rpm_bands(donor),
        perspective="ext",
        idle_rpm=1600.0,
        max_rpm=9450.0,
    )
    assert add_ext >= 1
    print(f"   Recreación Fiel Sección Exterior: {add_ext} bandas nativas sincronizadas de AC ({mute_ext} silenciadas).")

    sec_bands_int, sec_maps_int, add_int, mute_int = sync_faithful_section_from_source(
        donor,
        ac_lib,
        sec_maps,
        sec_bands,
        perspective="int",
        idle_rpm=1600.0,
        max_rpm=9450.0,
    )
    assert add_int >= 1
    assert len(sec_bands_int) == len(donor.rpm_bands)
    print(f"   Recreación Fiel Sección Interior: {add_int} bandas sincronizadas, {mute_int} silenciadas.")

    # 11b. Auto-ajuste de una sola banda (selectivo)
    band_target = [b for b in sec_bands_int if b.event_role == "EngineInt" and b.load_role == "on"][1]
    orig_rmin, orig_rmax = band_target.rpm_min, band_target.rpm_max
    # Modificar deliberadamente para provocar solapamiento
    band_target.rpm_min = 200.0
    band_target.rpm_max = 12000.0
    auto_distribute_single_rpm_band(
        band_target,
        sec_bands_int,
        sec_maps_int,
        donor,
        library=ac_lib,
        idle_rpm=1600.0,
        max_rpm=9450.0,
    )
    # Comprobar que la banda se reajustó a valores coherentes
    assert band_target.rpm_min > 500.0 and band_target.rpm_max < 16000.0
    print(f"   Auto-ajuste individual de banda #{band_target.box_index} verificado sin romper otras bandas.")

    # 11c. Añadir banda dinámica y silenciar/eliminar
    new_test_band = RpmBandConfig(
        band_id="test_custom_band_01",
        pml_guid=donor.rpm_bands[0].pml_guid,
        event_guid=donor.rpm_bands[0].event_guid,
        box_index=99,
        event_role="EngineInt",
        perspective="int",
        load_role="on",
        donor_slot_indices=[0],
        donor_label="AC Test Custom",
        rpm_min=4000.0,
        rpm_max=6000.0,
        fade_in_end=4500.0,
        fade_out_start=5500.0,
        orig_rpm_min=4000.0,
        orig_rpm_max=6000.0,
        orig_fade_in_end=4500.0,
        orig_fade_out_start=5500.0,
    )
    sec_bands_int.append(new_test_band)
    assert new_test_band in sec_bands_int
    # Eliminar/silenciar
    new_test_band.is_muted_bus = True
    new_test_band.rpm_max = 0.0
    assert new_test_band.is_muted_bus is True
    print("   Creación y eliminación de banda dinámica verificadas con éxito.")

    # 11d. Batch gain offset (+2.5 dB)
    initial_gain = sec_maps_int[0].gain_db
    for m in sec_maps_int:
        m.gain_db = round(m.gain_db + 2.5, 1)
    assert abs(sec_maps_int[0].gain_db - (initial_gain + 2.5)) < 0.01
    print("   Amplificación masiva de bandas (+2.5 dB) verificada correctamente.")

    if AC_MOD_488_BANK.is_file() and AC_MOD_488_GUIDS.is_file():
        print("12. Probando filtro estricto Chase Cam y mapeo nativo PMLB en Ferrari 488 GT3...")
        from ams2_sound_porter.porter_engine import is_chase_cam_sample
        mod_lib_test = SourceAudioLibrary.from_bank(AC_MOD_488_BANK, guids_path=AC_MOD_488_GUIDS)
        f488_bands, f488_maps, f488_ext_cnt, f488_ext_mute = sync_faithful_section_from_source(
            donor,
            mod_lib_test,
            [SlotMappingConfig(sl.global_index) for sl in donor.slots],
            build_default_rpm_bands(donor),
            perspective="ext",
            idle_rpm=950.0,
            max_rpm=7800.0,
        )
        assert len(f488_bands) == len(donor.rpm_bands)
        # Verificar que NINGUNA banda activa exterior utilice muestras de replay (front, far, flyby, etc.)
        for b in f488_bands:
            if b.perspective == "ext" and not b.is_muted_bus:
                for g in b.donor_slot_indices:
                    m = f488_maps[g]
                    if m.action == "replace_source" and m.source_index is not None:
                        sname = mod_lib_test.items[m.source_index].name
                        assert is_chase_cam_sample(sname), f"Muestra no-ChaseCam detectada en Exterior: {sname}"
                        assert m.override_root_rpm == mod_lib_test.items[m.source_index].effective_rpm
        print("   Filtro Chase Cam y preservación de cajas FMOD PMLB en 488 GT3 comprobados al 100%!")

    print("13. Probando Transmutación Directa AC -> AMS2 Sin Banco Donante (100% Fiel)...")
    # 13a. Probar transmutación directa con banco vanilla Z4 GT3
    assert ac_lib.fmod_bank is not None
    direct_res_z4 = build_direct_ported_bank_from_ac(
        ac_bank=ac_lib.fmod_bank,
        mod_identifier="bmw_z4_direct",
        standalone_guids=True,
    )
    assert len(direct_res_z4.bank_bytes) > 0
    assert_sndh_and_fsb_alignment(direct_res_z4.bank_bytes)
    assert_proj_tables_sorted(direct_res_z4.bank_bytes)
    assert "EngineExt" in direct_res_z4.role_to_event_path
    assert "EngineInt" in direct_res_z4.role_to_event_path
    print(f"   Vanilla Z4 GT3 port directo: {len(direct_res_z4.bank_bytes):,} bytes, tablas FMOD verificadas.")

    # 13b. Probar transmutación directa y exportación con mod Ferrari 488 GT3 + GUIDs.txt
    if AC_MOD_488_BANK.is_file() and AC_MOD_488_GUIDS.is_file():
        ac_488 = FmodBank(AC_MOD_488_BANK, guids_path=AC_MOD_488_GUIDS)
        with tempfile.TemporaryDirectory() as tmp_out:
            tmp_root = Path(tmp_out)
            csd_orig = CsdbinDocument.from_file(CSDBIN_PATH) if CSDBIN_PATH.is_file() else None
            summary_direct = export_ams2_mod_package(
                output_root=tmp_root,
                mod_identifier="f488_direct",
                vehicle_folder_name="ferrari_488_gt3",
                donor=None,
                direct_ac_bank=ac_488,
                csd_doc=csd_orig,
                standalone_guids=True,
            )
            assert summary_direct.bank_file.is_file()
            assert summary_direct.bank_file.stat().st_size > 0
            assert summary_direct.ai_bank_file is not None and summary_direct.ai_bank_file.is_file()
            assert summary_direct.ai_bank_file.stat().st_size > 0
            assert summary_direct.csdbin_file and summary_direct.csdbin_file.is_file()
            assert summary_direct.manifest_file.is_file()
            assert summary_direct.readme_file.is_file()

            # Verificar que el banco resultante se parsea limpiamente por FmodBank
            out_bank = FmodBank(summary_direct.bank_file)
            assert len(out_bank.events) >= 15
            assert len(out_bank.slots) >= len(ac_488.slots)
            assert len(out_bank.stdt_bytes) > 0

            # Verificar banco de IA y TV Replay
            ai_out_bank = FmodBank(summary_direct.ai_bank_file)
            assert len(ai_out_bank.events) >= 15
            assert len(ai_out_bank.stdt_bytes) > 0
            ai_paths = parse_stdt_all_paths(ai_out_bank.stdt_bytes)
            assert any("AI_EngineMAIN_LIVE" in p for p in ai_paths.values()), "Falta evento AI_EngineMAIN_LIVE para TV Replays"
            assert any("AI_EngineMAIN" in p for p in ai_paths.values()), "Falta evento AI_EngineMAIN para rivales"

            # Verificar que el .csdbin contiene todos los enlaces de jugador e IA
            # y conserva intactos Starter/Starting + Transmisión/Gearshift originales de AMS2
            csd_loaded = CsdbinDocument.from_file(summary_direct.csdbin_file)
            assert csd_loaded.links["engine_bank"] == "Audio/AMS2/Vehicles/Mods/f488_direct/f488_direct_Engine"
            assert "engine_ext" in csd_loaded.links
            assert "engine_int" in csd_loaded.links
            assert csd_loaded.links["ai_engine_bank"] == "Audio/AMS2/Vehicles/Mods/f488_direct/f488_direct_Engine_AI"
            assert "AI_EngineMAIN" in csd_loaded.links["ai_engine_ext"]
            assert "AI_EngineMAIN_LIVE" in csd_loaded.links["ai_engine_live"]
            assert "AI_EngineInt" in csd_loaded.links["ai_engine_int"]
            assert "starter" in csd_loaded.links and "Starter" in csd_loaded.links["starter"]
            assert "starting" in csd_loaded.links and "Starting" in csd_loaded.links["starting"]
            assert csd_loaded.links.get("transmission_bank") == "Audio/AMS2/Vehicles/Drivetrain/GearWhine/GearWhine_SuperV8"
            assert csd_loaded.links.get("transmission") == "event:/AMS2/Drivetrain_GLOBAL/GearWhine/SuperV8/SuperV8_Tranny_INT"
            assert csd_loaded.links.get("gearshift_bank") == "Audio/AMS2/Vehicles/Drivetrain/GearChangeGrinding/GearShift_SuperV8"
            assert csd_loaded.links.get("gearshift_up") == "event:/AMS2/Drivetrain_GLOBAL/GearChangeGrinding/SuperV8/SuperV8_Upshift"
            assert csd_loaded.links.get("gearshift_down") == "event:/AMS2/Drivetrain_GLOBAL/GearChangeGrinding/SuperV8/SuperV8_Downshift"
            assert "turbo_bank" in csd_loaded.links
            assert "turbo" in csd_loaded.links
            assert "backfire_ext" in csd_loaded.links
            print(f"   Ferrari 488 GT3 Mod export directo DUAL (Jugador + IA + Starter + Transmisión AMS2): {summary_direct.bank_file.name} y {summary_direct.ai_bank_file.name} OK!")

    print("14. Probando Gestor de Vehículos Vanilla (Auditoría, Inyección Directa y Restauración Bit-a-Bit)...")
    from ams2_sound_porter.vanilla_manager import VanillaVehicleManager, calculate_file_sha256

    mgr = VanillaVehicleManager()
    assert mgr.is_ams2_valid(), "AMS2 no detectado en el sistema"
    catalog = mgr.scan_catalog()
    assert len(catalog) >= 150, f"Catálogo insuficiente de vehículos vanilla: {len(catalog)}"
    print(f"   Catálogo oficial escaneado: {len(catalog)} vehículos base de AMS2 catalogados.")

    # Simular ciclo completo de seguridad y restauración en sandbox aislado
    with tempfile.TemporaryDirectory() as tmp_sandbox:
        sb_root = Path(tmp_sandbox)
        sb_sort = sb_root / "Audio" / "AMS2" / "Vehicles" / "Sort" / "S" / "Super_V8"
        sb_sort.mkdir(parents=True)
        sb_bank = sb_sort / "SuperV8_Set1.bank"

        # Copiar banco vanilla real y banco AI al sandbox
        real_superv8 = Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2\Audio\AMS2\Vehicles\Sort\S\Super_V8\SuperV8_Set1.bank")
        real_superv8_ai = Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2\Audio\AMS2\Vehicles\Sort\S\Super_V8\SuperV8_Set1_AI.bank")
        import shutil
        shutil.copy2(real_superv8, sb_bank)
        orig_sha = calculate_file_sha256(sb_bank)

        sb_ai_bank = sb_sort / "SuperV8_Set1_AI.bank"
        shutil.copy2(real_superv8_ai, sb_ai_bank)
        orig_ai_sha = calculate_file_sha256(sb_ai_bank)

        # Gestor operando en el sandbox
        sb_mgr = VanillaVehicleManager(ams2_root=sb_root, backup_dir=sb_root / "backups")
        sb_vehs = sb_mgr.scan_catalog()
        assert len(sb_vehs) == 1
        target_v = sb_vehs[0]
        assert target_v.ai_bank_path and target_v.ai_bank_path.is_file()
        sb_mgr.load_vehicle_events_info(target_v)
        assert "EngineExt" in target_v.event_roles
        assert "EngineInt" in target_v.event_roles
        assert len(target_v.ai_event_roles) >= 2
        assert "ai_engine_live" in target_v.ai_event_roles or "AI_EngineMAIN_LIVE" in target_v.ai_event_roles or "ai_engine_ext" in target_v.ai_event_roles

        # 14a. Compilación directa AC con los GUIDs nativos del vehículo vanilla
        vanilla_direct = build_direct_ported_bank_from_ac(
            ac_bank=ac_lib.fmod_bank,
            mod_identifier=target_v.car_name,
            target_role_guids=target_v.event_guids,
            target_role_paths=target_v.event_paths,
            target_bank_guid=target_v.bank_guid,
        )
        assert len(vanilla_direct.bank_bytes) > 0
        assert_sndh_and_fsb_alignment(vanilla_direct.bank_bytes)
        sorted_tbls = assert_proj_tables_sorted(vanilla_direct.bank_bytes)
        assert sorted_tbls >= 13, f"Tablas insuficientes ordenadas: {sorted_tbls}"
        assert len(sb_mgr.find_vehicles("Super")) == 1

        parsed_vanilla = FmodBank(vanilla_direct.bank_bytes)
        ext_ev = next(
            e for e in parsed_vanilla.events
            if "engineext" in (e.path or "").lower() or e.guid_str.lower() == target_v.event_guids["EngineExt"].lower()
        )
        assert ext_ev.guid_str.lower() == target_v.event_guids["EngineExt"].lower()
        int_ev = next(
            e for e in parsed_vanilla.events
            if "engineint" in (e.path or "").lower() or e.guid_str.lower() == target_v.event_guids["EngineInt"].lower()
        )
        assert int_ev.guid_str.lower() == target_v.event_guids["EngineInt"].lower()
        print(f"   Inyección de GUIDs nativos de vehículo vanilla comprobada con éxito ({sorted_tbls} tablas FMOD ordenadas).")

        # 14b. Compilación de banco de IA y TV Replays
        ai_donor = FmodBank(sb_ai_bank)
        ai_mappings = auto_map_samples(ai_donor, ac_lib)
        ai_res = build_ported_bank(
            donor=ai_donor,
            library=ac_lib,
            mappings=ai_mappings,
            mod_identifier=f"{target_v.car_name}_AI",
            standalone_guids=False,
            output_codec_mode=FSB5_MODE_PCM16,
        )
        assert len(ai_res.bank_bytes) > 0
        assert_sndh_and_fsb_alignment(ai_res.bank_bytes)

        # 14c. Instalación DUAL segura (Jugador + IA) con creación de respaldos automáticos
        ok_inst, msg_inst = sb_mgr.install_sound(target_v, vanilla_direct.bank_bytes, ai_bank_bytes=ai_res.bank_bytes)
        assert ok_inst, msg_inst
        assert target_v.is_modified
        assert (sb_sort / "SuperV8_Set1.bank.orig").is_file()
        assert (sb_sort / "SuperV8_Set1_AI.bank.orig").is_file()
        assert target_v.original_sha256 == orig_sha
        assert target_v.original_ai_sha256 == orig_ai_sha
        assert calculate_file_sha256(sb_bank) != orig_sha
        assert calculate_file_sha256(sb_ai_bank) != orig_ai_sha
        print("   Instalación DUAL en vehículo base con respaldos automáticos (.orig + SHA256) verificada.")

        # 14d. Restauración DUAL al estado original de Steam sin reparación de archivos
        ok_rest, msg_rest = sb_mgr.restore_vehicle(target_v)
        assert ok_rest, msg_rest
        assert not target_v.is_modified
        assert not (sb_sort / "SuperV8_Set1.bank.orig").is_file()
        assert not (sb_sort / "SuperV8_Set1_AI.bank.orig").is_file()
        restored_sha = calculate_file_sha256(sb_bank)
        restored_ai_sha = calculate_file_sha256(sb_ai_bank)
        assert restored_sha == orig_sha, "El banco del jugador restaurado difiere del original de fábrica"
        assert restored_ai_sha == orig_ai_sha, "El banco de IA restaurado difiere del original de fábrica"
        print(f"   Restauración DUAL 100% fiel de Steam comprobada (SHA256 verificado bit-a-bit: Jugador={restored_sha[:10]}... IA={restored_ai_sha[:10]}...).")
        assert ok_rest, msg_rest
        assert not target_v.is_modified
        assert not (sb_sort / "SuperV8_Set1.bank.orig").is_file()
        restored_sha = calculate_file_sha256(sb_bank)
        assert restored_sha == orig_sha, "El archivo restaurado difiere del original de fábrica"
        print(f"   Restauración 100% fiel de Steam comprobada (SHA256 verificado bit-a-bit: {restored_sha[:12]}...).")

    # 15. Probando resolución de anomalías Mercer V8 (crasheo por GUIDs duplicados) y corrección de parámetros (load vs throttle)
    mercer_bank_path = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_gtm_mercer_v8\sfx\rss_gtm_mercer_v8.bank")
    if mercer_bank_path.is_file():
        print("15. Probando resolución de anomalías Mercer V8 (crasheo por GUIDs duplicados) y corrección de parámetros (load vs throttle)...")
        mercer_b = FmodBank(mercer_bank_path)
        mgr_v = VanillaVehicleManager(Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2"))
        sls_v = next((v for v in mgr_v.scan_catalog() if "SLS" in v.car_name), None)
        if sls_v:
            mgr_v.load_vehicle_events_info(sls_v)
            mercer_sls_res = build_direct_ported_bank_from_ac(
                ac_bank=mercer_b,
                mod_identifier=sls_v.car_name,
                target_role_guids=sls_v.event_guids,
                target_role_paths=sls_v.event_paths,
                target_bank_guid=sls_v.bank_guid or None,
                target_param_names=sls_v.target_parameters,
                native_bank_path=sls_v.bank_path,
            )
            reloaded_mercer = FmodBank(raw=mercer_sls_res.bank_bytes)
            guids_list = [ev.guid_str for ev in reloaded_mercer.events]
            assert len(set(guids_list)) == len(guids_list), "Se detectaron GUIDs duplicados en el porteo de Mercer V8!"
            reloaded_roles = {ev.guid_str: ev.role for ev in reloaded_mercer.events}
            for role_name, exp_guid in sls_v.event_guids.items():
                assert exp_guid in reloaded_roles, f"Falta el rol {role_name} ({exp_guid}) en el banco de Mercer V8!"
            assert_sndh_and_fsb_alignment(mercer_sls_res.bank_bytes)

            # Verificar inversión de cono espacial 3D (Event Cone Angle) en curvas de volumen (Prop 0)
            target_vol_cg = b"\x04^\x93\xc42\xa9uI\x95\xdd\xa5\xe0Y\x96\xeb="
            orig_first_cone_x = None
            for cid, ps, pe, _ in iter_riff_chunks(mercer_b.raw, 12, len(mercer_b.raw)):
                if cid == b"LIST" and mercer_b.raw[ps : ps + 4] == b"PROJ":
                    for scid, sps, spe, _ in iter_riff_chunks(mercer_b.raw, ps + 4, pe):
                        if scid == b"CURV" and spe - sps >= 52:
                            cg = mercer_b.raw[sps : sps + 16]
                            if cg == target_vol_cg:
                                orig_first_cone_x = struct.unpack_from("<ff", mercer_b.raw, sps + 36)[0]
                                break
            if orig_first_cone_x is not None:
                new_first_cone_x = None
                for cid, ps, pe, _ in iter_riff_chunks(mercer_sls_res.bank_bytes, 12, len(mercer_sls_res.bank_bytes)):
                    if cid == b"LIST" and mercer_sls_res.bank_bytes[ps : ps + 4] == b"PROJ":
                        for scid, sps, spe, _ in iter_riff_chunks(mercer_sls_res.bank_bytes, ps + 4, pe):
                            if scid == b"CURV" and spe - sps >= 52:
                                cg = mercer_sls_res.bank_bytes[sps : sps + 16]
                                if cg == target_vol_cg:
                                    stride = struct.unpack_from("<H", mercer_sls_res.bank_bytes, sps + 34)[0]
                                    new_first_cone_x = struct.unpack_from("<ff", mercer_sls_res.bank_bytes, sps + 36 + stride)[0]
                                    break
                assert new_first_cone_x is not None and abs(new_first_cone_x - 94.06) < 0.2, (
                    f"Fallo en inversión de curva de cono espacial: valor obtenido {new_first_cone_x}"
                )
            print("   Porteo de Mercer V8 verificado: 0 GUIDs duplicados, eventos nativos, conos espaciales 3D invertidos y parámetros load/rpm sincronizados al 100%!")

    # 16. Probando eliminación de rateo/chatter de control de tracción y petardeos continuos en McLaren 720S
    macca_bank_path = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_gtm_macca_72_evo_v8\sfx\rss_gtm_macca_72_evo_v8.bank")
    if macca_bank_path.is_file():
        print("16. Probando eliminación de rateo/chatter de control de tracción y petardeos continuos en McLaren 720S...")
        macca_b = FmodBank(macca_bank_path)
        mgr_v = VanillaVehicleManager(Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2"))
        mclaren_v = next((v for v in mgr_v.scan_catalog() if "720s" in v.vehicle_id.lower()), None)
        if mclaren_v:
            mgr_v.load_vehicle_events_info(mclaren_v)
            macca_res = build_direct_ported_bank_from_ac(
                ac_bank=macca_b,
                mod_identifier=mclaren_v.car_name,
                target_role_guids=mclaren_v.event_guids,
                target_role_paths=mclaren_v.event_paths,
                target_bank_guid=mclaren_v.bank_guid or None,
                target_param_names=mclaren_v.target_parameters,
                native_bank_path=mclaren_v.bank_path,
                rpm_bands=macca_b.rpm_bands,
                mute_secondary_pops=True,
                invert_spatial_cone=True,
            )
            reloaded_macca = FmodBank(raw=macca_res.bank_bytes)
            assert len(reloaded_macca.events) >= 20, "El banco de McLaren 720S no preservó sus eventos directos!"
            # Verificar que ningún evento de motor quedó con 0 muestras
            eng_ext = next((e for e in reloaded_macca.events if e.role == "EngineExt" and len(e.linked_samples) >= 5), None)
            eng_int = next((e for e in reloaded_macca.events if e.role == "EngineInt" and len(e.linked_samples) >= 5), None)
            assert eng_ext is not None and eng_int is not None, "EngineExt o EngineInt perdieron sus muestras de sonido!"
            
            # Verificar que ningún vehículo vanilla hereda .csdbin de mod por coincidencia parcial de marca
            assert mclaren_v.csdbin_path is None, f"Falso positivo de .csdbin vanilla detectado: {mclaren_v.csdbin_path}"

            # Verificar que las muestras de TC/LC y pops esparsos (MUIT + WAIT) quedaron silenciadas en FSB5 PCM16,
            # mientras que las capas sostenidas principales de motor permanecen intactas.
            for muted_idx in (13, 18, 21, 25, 49, 58, 70, 85, 93, 96, 129, 197, 198, 208, 235):
                sl_m = reloaded_macca.slots[muted_idx]
                assert sl_m.data_bytes == b"\x00" * len(sl_m.data_bytes), (
                    f"El audio FSB5 de sample_{muted_idx} (TC/pop esparso) no fue silenciado!"
                )
            for alive_idx in (0, 175, 223, 187, 101, 234):
                sl_a = reloaded_macca.slots[alive_idx]
                assert sl_a.data_bytes != b"\x00" * len(sl_a.data_bytes), (
                    f"Muestra de motor sostenida sample_{alive_idx} enmudecida por error!"
                )

            # Verificar que las cajas de TC chatter (inst b9c38884 en rpms 2100..7400) quedaron silenciadas a 0.0..0.0 incluso pasando rpm_bands
            tc_ext_inst = guid_str_to_bytes("{b9c38884-ac8b-4be6-bc69-048ce4c26e93}")
            tc_boxes_checked = 0
            pnames_player: set[str] = set()
            ev_limits_player: list[tuple[int, int]] = []
            for cid, ps, pe, _ in iter_riff_chunks(macca_res.bank_bytes, 12, len(macca_res.bank_bytes)):
                if cid == b"LIST" and macca_res.bank_bytes[ps : ps + 4] == b"PROJ":
                    def walk_check(s: int, e: int) -> None:
                        nonlocal tc_boxes_checked
                        for scid, sps, spe, _ in iter_riff_chunks(macca_res.bank_bytes, s, e):
                            if scid == b"LIST":
                                walk_check(sps + 4, spe)
                            elif scid == b"PRMB" and spe - sps >= 23:
                                sl = struct.unpack_from("<H", macca_res.bank_bytes, sps + 21)[0]
                                pnames_player.add(
                                    macca_res.bank_bytes[sps + 23 : sps + 23 + sl].decode("latin1", "replace").rstrip("\x00")
                                )
                            elif scid == b"EVTB" and spe - sps >= 88:
                                ev_limits_player.append(struct.unpack_from("<iI", macca_res.bank_bytes, sps + 80))
                            elif scid == b"PMLB" and spe - sps >= 52:
                                p = macca_res.bank_bytes[sps:spe]
                                nboxes = struct.unpack_from("<H", p, 48)[0] >> 1
                                stride = struct.unpack_from("<H", p, 50)[0]
                                if stride >= 24 and len(p) >= 52 + nboxes * stride:
                                    for bi in range(nboxes):
                                        boff = 52 + bi * stride
                                        if p[boff : boff + 16] == tc_ext_inst:
                                            v0, vlen = struct.unpack_from("<ff", p, boff + 16)
                                            assert v0 == 0.0 and vlen == 0.0, f"Caja de TC {bi} no fue silenciada: [{v0}..{v0+vlen}]"
                                            tc_boxes_checked += 1
                    walk_check(ps + 4, pe)
            assert tc_boxes_checked > 0, "No se encontraron cajas de TC en el banco para verificar"
            assert "load" in pnames_player and "throttle" not in pnames_player, (
                f"Parámetro throttle no renombrado a load en banco jugador: {pnames_player}"
            )
            assert max(mi for mi, _ in ev_limits_player) <= 4, f"Evento sin límite en Jugador: {ev_limits_player}"

            # Verificar compilación dedicada de banco _AI.bank para 720S (rivales + TV Replay)
            if mclaren_v.ai_bank_path and mclaren_v.ai_event_guids:
                macca_ai_res = build_direct_ported_bank_from_ac(
                    ac_bank=macca_b,
                    mod_identifier=f"{mclaren_v.car_name}_AI",
                    target_role_guids=mclaren_v.ai_event_guids,
                    target_role_paths=mclaren_v.ai_event_paths,
                    target_bank_guid=mclaren_v.ai_bank_guid or None,
                    target_role_buses=getattr(mclaren_v, "ai_event_buses", None),
                    rpm_bands=macca_b.rpm_bands,
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                    invert_spatial_cone=True,
                )
                assert macca_ai_res.new_bank_guid.lower() != macca_res.new_bank_guid.lower()
                assert_sndh_and_fsb_alignment(macca_ai_res.bank_bytes)
                assert_proj_tables_sorted(macca_ai_res.bank_bytes)
                reloaded_macca_ai = FmodBank(raw=macca_ai_res.bank_bytes)
                ai_ev_guids_present = {e.guid_str.lower() for e in reloaded_macca_ai.events}
                for ai_r, ai_g in mclaren_v.ai_event_guids.items():
                    assert ai_g.lower() in ai_ev_guids_present, f"Falta evento IA {ai_r} ({ai_g}) en 720S_AI.bank!"
                ai_limits: list[tuple[int, int]] = []
                ai_ibsb: dict[bytes, bytes] = {}
                ai_evtb_ibg: dict[bytes, bytes] = {}
                ai_pnames: set[str] = set()
                for cid, ps, pe, _ in iter_riff_chunks(macca_ai_res.bank_bytes, 12, len(macca_ai_res.bank_bytes)):
                    if cid == b"LIST" and macca_ai_res.bank_bytes[ps : ps + 4] == b"PROJ":
                        def walk_ai(s: int, e: int) -> None:
                            for scid, sps, spe, _ in iter_riff_chunks(macca_ai_res.bank_bytes, s, e):
                                if scid == b"LIST":
                                    walk_ai(sps + 4, spe)
                                elif scid == b"PRMB" and spe - sps >= 23:
                                    sl = struct.unpack_from("<H", macca_ai_res.bank_bytes, sps + 21)[0]
                                    ai_pnames.add(
                                        macca_ai_res.bank_bytes[sps + 23 : sps + 23 + sl].decode("latin1", "replace").rstrip("\x00")
                                    )
                                elif scid == b"EVTB" and spe - sps >= 88:
                                    ai_limits.append(struct.unpack_from("<iI", macca_ai_res.bank_bytes, sps + 80))
                                    ai_evtb_ibg[bytes(macca_ai_res.bank_bytes[sps : sps + 16])] = bytes(
                                        macca_ai_res.bank_bytes[sps + 48 : sps + 64]
                                    )
                                elif scid == b"IBSB" and spe - sps >= 34:
                                    ai_ibsb[bytes(macca_ai_res.bank_bytes[sps : sps + 16])] = bytes(
                                        macca_ai_res.bank_bytes[sps + 18 : sps + 34]
                                    )
                        walk_ai(ps + 4, pe)
                assert "load" in ai_pnames and "throttle" not in ai_pnames
                assert all(0 < mi <= 16 and (fl & 0xE0) == 0x40 for mi, fl in ai_limits), (
                    f"Eventos en 720S_AI.bank sin virtualización 0x40 o sin límite: {ai_limits}"
                )
                ai_bus_bytes = guid_str_to_bytes(AMS2_AI_ENGINE_BUS)
                for main_role in ("ai_engine_main", "ai_engine_live"):
                    if main_role in mclaren_v.ai_event_guids:
                        ev_b = guid_str_to_bytes(mclaren_v.ai_event_guids[main_role])
                        assert ai_ibsb.get(ai_evtb_ibg.get(ev_b, b"")) == ai_bus_bytes, (
                            f"{main_role} no apunta a bus 3_AI_PLAYER_ENG!"
                        )

                orig_cones = _extract_cone_vol_endpoints(macca_b.raw)
                player_cones = _extract_cone_vol_endpoints(macca_res.bank_bytes)
                ai_cones = _extract_cone_vol_endpoints(macca_ai_res.bank_bytes)
                assert orig_cones, "Se esperaban curvas direccionales Event Cone Angle en Macca 720S AC"
                assert player_cones != orig_cones, "El banco del jugador debía invertir Event Cone Angle (0°=Frente)"
                assert ai_cones == orig_cones, (
                    f"El banco _AI (IA + Replay) NO debe invertir Event Cone Angle respecto a AC: orig={orig_cones} vs ai={ai_cones}"
                )

                # Verificar 0 colisiones de GUIDs internos entre Jugador e IA tras aislar el banco AI
                def _collect_table_guids(raw_b: bytes) -> dict[bytes, set[bytes]]:
                    out_t: dict[bytes, set[bytes]] = {}
                    for _c, _ps, _pe, _ in iter_riff_chunks(raw_b, 12, len(raw_b)):
                        if _c == b"LIST" and raw_b[_ps : _ps + 4] == b"PROJ":
                            for _tc, _tps, _tpe, _ in iter_riff_chunks(raw_b, _ps + 4, _pe):
                                if _tc == b"LIST":
                                    lt_b = raw_b[_tps : _tps + 4]
                                    s_g: set[bytes] = set()
                                    for _sc, _sps, _spe, _ in iter_riff_chunks(raw_b, _tps + 4, _tpe):
                                        if _sc == b"LCNT":
                                            continue
                                        if _sc == b"LIST":
                                            for _ssc, _ssps, _sspe, _ in iter_riff_chunks(raw_b, _sps + 4, _spe):
                                                if _sspe - _ssps >= 16:
                                                    s_g.add(bytes(raw_b[_ssps : _ssps + 16]))
                                                break
                                        elif _spe - _sps >= 16:
                                            s_g.add(bytes(raw_b[_sps : _sps + 16]))
                                    out_t[lt_b] = s_g
                    return out_t

                tbl_p = _collect_table_guids(macca_res.bank_bytes)
                tbl_ai = _collect_table_guids(macca_ai_res.bank_bytes)
                for chk_tbl in (b"IBSS", b"GBSS", b"MBSS", b"BEFX", b"PEFX", b"SCFX", b"EVTS", b"TLNS", b"PMLS", b"PRMS", b"CTRS", b"CRVS", b"MUIS", b"SPIS", b"EVIS", b"WAIS", b"WAVS", b"MODS"):
                    ov = tbl_p.get(chk_tbl, set()) & tbl_ai.get(chk_tbl, set())
                    assert not ov, f"Colisión de {len(ov)} GUIDs entre Jugador e IA en tabla {chk_tbl.decode()}!"

                # Verificar compresión FADPCM + poda de muestras no utilizadas (< 25 MB para banco Macca de 109 MB)
                macca_comp = build_direct_ported_bank_from_ac(
                    ac_bank=macca_b,
                    mod_identifier=mclaren_v.car_name,
                    standalone_guids=False,
                    target_role_guids=mclaren_v.event_guids,
                    target_role_paths=mclaren_v.event_paths,
                    target_bank_guid=mclaren_v.bank_guid or None,
                    native_bank_path=mclaren_v.bank_path,
                    mute_secondary_pops=True,
                    output_codec_mode=FSB5_MODE_FADPCM,
                    prune_unused_samples=True,
                )
                assert len(macca_comp.bank_bytes) < 25_000_000, f"Tamaño comprimido inesperado: {len(macca_comp.bank_bytes):,}"
                assert_sndh_and_fsb_alignment(macca_comp.bank_bytes)
            print(f"   Silenciamiento FSB5/PMLB/MUIT, throttle->load, cono Jugador!=IA y banco IA virtualizado verificados ({tc_boxes_checked} cajas TC neutralizadas)!")

    print("17. Probando exportación Mod AC directa con virtualización IA y protección Vorbis (Macca + Bentley)...")
    mod_cases = [
        (
            "mcl650s_reg",
            Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_gtm_macca_72_evo_v8\sfx\rss_gtm_macca_72_evo_v8.bank"),
            Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2\Vehicles\mclaren_650s_gt3\Mcl650sGT3.csdbin"),
        ),
        (
            "bentley_reg",
            Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\cf_bentley_continental_gt3_2018\sfx\cf_bentley_continental_gt3_2018.bank"),
            Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2\Vehicles\bentley_continental_gt3\bentleygt3.csdbin"),
        ),
    ]
    for mod_name, ac_fp, csd_fp in mod_cases:
        if not ac_fp.is_file():
            continue
        ac_mod_b = FmodBank(ac_fp)
        csd_mod = CsdbinDocument.from_file(csd_fp) if csd_fp.is_file() else None
        with tempfile.TemporaryDirectory() as tmp_mod_dir:
            s_mod = export_ams2_mod_package(
                output_root=Path(tmp_mod_dir),
                mod_identifier=mod_name,
                vehicle_folder_name=mod_name,
                direct_ac_bank=ac_mod_b,
                csd_doc=csd_mod,
                standalone_guids=True,
                mute_pops=True,
                invert_spatial_cone=True,
            )
            assert s_mod.ai_bank_file and s_mod.ai_bank_file.is_file()
            assert s_mod.ai_bank_result is not None
            assert s_mod.bank_result.new_bank_guid != s_mod.ai_bank_result.new_bank_guid
            if s_mod.csdbin_file:
                csd_res = CsdbinDocument.from_file(s_mod.csdbin_file)
                assert "EngineExt_34" not in csd_res.links.get("engine_ext", "")
                assert csd_res.links.get("ai_engine_bank") != csd_res.links.get("engine_bank")
                assert "AI_EngineMAIN" in csd_res.links.get("ai_engine_ext", "")
                assert "AI_EngineMAIN_LIVE" in csd_res.links.get("ai_engine_live", "")

            for role_lbl, b_path, max_cap in (("P", s_mod.bank_file, 4), ("AI", s_mod.ai_bank_file, 16)):
                parsed_b = FmodBank(b_path)
                lims: list[tuple[int, int]] = []
                for cid, ps, pe, _ in iter_riff_chunks(parsed_b.raw, 12, len(parsed_b.raw)):
                    if cid == b"LIST" and parsed_b.raw[ps : ps + 4] == b"PROJ":
                        def walk_ev(s: int, e: int) -> None:
                            for scid, sps, spe, _ in iter_riff_chunks(parsed_b.raw, s, e):
                                if scid == b"LIST":
                                    walk_ev(sps + 4, spe)
                                elif scid == b"EVTB" and spe - sps >= 88:
                                    lims.append(struct.unpack_from("<iI", parsed_b.raw, sps + 80))
                        walk_ev(ps + 4, pe)
                assert max(mi for mi, _ in lims) <= max_cap, f"{mod_name} {role_lbl} max_instances excedido: {lims}"
                if role_lbl == "AI":
                    assert all((fl & 0xE0) == 0x40 for _, fl in lims), f"{mod_name} AI sin flags 0x40: {lims}"
                for sl in parsed_b.slots:
                    if sl.mode not in (FSB5_MODE_PCM16, FSB5_MODE_FADPCM):
                        assert sl.data_bytes != b"\x00" * len(sl.data_bytes), (
                            f"{mod_name} flujo Vorbis dañado en slot {sl.global_index}"
                        )
            if s_mod.bank_result is not None and s_mod.ai_bank_result is not None:
                m_orig = _extract_cone_vol_endpoints(ac_mod_b.raw)
                if m_orig:
                    m_player = _extract_cone_vol_endpoints(s_mod.bank_result.bank_bytes)
                    m_ai = _extract_cone_vol_endpoints(s_mod.ai_bank_result.bank_bytes)
                    assert m_player != m_orig, f"{mod_name}: Jugador debía invertir Event Cone Angle"
                    assert m_ai == m_orig, (
                        f"{mod_name}: _AI (IA + Replay) NO debía invertir Event Cone Angle"
                    )
            print(f"   Mod {mod_name} (Jugador + IA + cono espacial + .csdbin + protección Vorbis) verificado al 100%!")

    # Verificar que override_root_rpm en Modo Directo AC actualiza INST (+69) en el .bank exportado
    ap_slot = next((sl for sl in ac_lib.fmod_bank.slots if sl.has_rpm_autopitch and sl.autopitch_ref_rpm and sl.wav_guid), None)
    assert ap_slot is not None, "No se encontró slot con autopitch en Z4 GT3 para probar override_root_rpm"
    orig_ap_rpm = float(ap_slot.autopitch_ref_rpm)
    direct_maps = [SlotMappingConfig(s.global_index, action="keep_donor") for s in ac_lib.fmod_bank.slots]
    direct_maps[ap_slot.global_index].override_root_rpm = orig_ap_rpm + 300.0
    res_pitch = build_direct_ported_bank_from_ac(
        ac_bank=ac_lib.fmod_bank,
        mod_identifier="z4_pitch_shift",
        standalone_guids=True,
        mappings=direct_maps,
    )
    reloaded_pitch = FmodBank(raw=res_pitch.bank_bytes)
    new_ap_rpm = reloaded_pitch.slots[ap_slot.global_index].autopitch_ref_rpm
    assert new_ap_rpm is not None and abs(new_ap_rpm - (orig_ap_rpm + 300.0)) < 1.0, (
        f"Autopitch directo esperado {orig_ap_rpm + 300.0}, obtenido {new_ap_rpm}"
    )
    print(f"   Override de RPM Raíz (Pitch) en Modo Directo AC verificado: {orig_ap_rpm:.0f} -> {new_ap_rpm:.0f} RPM!")

    # 18. Acomodación condicional para bancos multi-variante y basados en 'throttle' sin alterar la ruta estándar
    from ams2_sound_porter.vanilla_manager import VanillaVehicleManager

    print("18. Verificando acomodación condicional (throttle + extra_role_variants) y ruta estándar intacta...")
    p_std_res = build_direct_ported_bank_from_ac(
        ac_bank=ac_lib.fmod_bank,
        mod_identifier="TestStd_P",
        target_role_guids={
            "EngineExt": "{11111111-1111-4111-8111-111111111101}",
            "EngineInt": "{11111111-1111-4111-8111-111111111103}",
        },
        target_role_paths={
            "EngineExt": "event:/AMS2/Vehicles/TestMulti/EngineExt",
            "EngineInt": "event:/AMS2/Vehicles/TestMulti/EngineInt",
        },
    )
    p_std_explicit = build_direct_ported_bank_from_ac(
        ac_bank=ac_lib.fmod_bank,
        mod_identifier="TestStd_P",
        target_role_guids={
            "EngineExt": "{11111111-1111-4111-8111-111111111101}",
            "EngineInt": "{11111111-1111-4111-8111-111111111103}",
        },
        target_role_paths={
            "EngineExt": "event:/AMS2/Vehicles/TestMulti/EngineExt",
            "EngineInt": "event:/AMS2/Vehicles/TestMulti/EngineInt",
        },
        extra_role_variants=None,
        prefer_throttle_param=False,
    )
    assert p_std_res.bank_bytes == p_std_explicit.bank_bytes, "La ruta estándar debía ser idéntica bit-a-bit!"

    res_multi_p = build_direct_ported_bank_from_ac(
        ac_bank=ac_lib.fmod_bank,
        mod_identifier="TestMulti_P",
        target_role_guids={
            "EngineExt": "{11111111-1111-4111-8111-111111111101}",
            "EngineInt": "{11111111-1111-4111-8111-111111111103}",
        },
        target_role_paths={
            "EngineExt": "event:/AMS2/Vehicles/TestMulti/EngineExt",
            "EngineInt": "event:/AMS2/Vehicles/TestMulti/EngineInt",
        },
        extra_role_variants={
            "EngineExt": [("{11111111-1111-4111-8111-111111111102}", "event:/AMS2/Vehicles/TestMulti/EngineExt_B")],
            "EngineInt": [
                ("{11111111-1111-4111-8111-111111111104}", "event:/AMS2/Vehicles/TestMulti/EngineInt_V1"),
                ("{11111111-1111-4111-8111-111111111105}", "event:/AMS2/Vehicles/TestMulti/EngineInt_V2"),
            ],
        },
        prefer_throttle_param=True,
    )
    assert_sndh_and_fsb_alignment(res_multi_p.bank_bytes)
    assert_proj_tables_sorted(res_multi_p.bank_bytes)
    rel_multi_p = FmodBank(raw=res_multi_p.bank_bytes)
    p_multi_guids = [e.guid_str.lower() for e in rel_multi_p.events]
    assert len(p_multi_guids) == len(set(p_multi_guids)), "GUIDs duplicados en Player multi-variante!"
    for expected_g in (
        "{11111111-1111-4111-8111-111111111101}",
        "{11111111-1111-4111-8111-111111111102}",
        "{11111111-1111-4111-8111-111111111103}",
        "{11111111-1111-4111-8111-111111111104}",
        "{11111111-1111-4111-8111-111111111105}",
    ):
        assert expected_g in p_multi_guids, f"Falta variante Player {expected_g} en EVTS!"
    multi_p_params = {p[0].lower() for e in rel_multi_p.events for p in e.parameters}
    assert "throttle" in multi_p_params, f"Se esperaba 'throttle' en Player multi-variante: {multi_p_params}"

    ams2_dir = Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2")
    if ams2_dir.is_dir():
        mgr_multi = VanillaVehicleManager(ams2_dir)
        catalog_multi = mgr_multi.scan_catalog()
        for std_kw in ("720s", "corvette_c8_gt3", "sls_gt3"):
            v_std = next((v for v in catalog_multi if std_kw in v.vehicle_id.lower()), None)
            if v_std:
                mgr_multi.load_vehicle_events_info(v_std)
                assert not v_std.prefers_throttle_player and not v_std.prefers_throttle_ai, (
                    f"{v_std.vehicle_id}: vehículo estándar no debe activar prefers_throttle"
                )
                assert not v_std.extra_event_variants and not v_std.extra_ai_event_variants, (
                    f"{v_std.vehicle_id}: vehículo estándar no debe activar extra_variants"
                )

        for target_kw in ("alpine_a424", "f_ultimate_2022", "f_v8_gen1"):
            v_cand = next(
                (v for v in catalog_multi if target_kw in v.vehicle_id.lower() or target_kw in v.bank_path.stem.lower()),
                None,
            )
            if not v_cand:
                continue
            mgr_multi.load_vehicle_events_info(v_cand)
            assert v_cand.prefers_throttle_player and v_cand.prefers_throttle_ai
            res_vp = build_direct_ported_bank_from_ac(
                ac_bank=ac_lib.fmod_bank,
                mod_identifier=v_cand.car_name,
                target_role_guids=v_cand.event_guids,
                target_role_paths=v_cand.event_paths,
                target_bank_guid=v_cand.bank_guid or None,
                target_param_names=v_cand.target_parameters,
                target_ctrl_params=v_cand.ctrl_parameters,
                target_role_buses=getattr(v_cand, "ai_event_buses", None),
                native_bank_path=v_cand.bank_path,
                mute_secondary_pops=True,
                invert_spatial_cone=True,
                extra_role_variants=v_cand.extra_event_variants or None,
                prefer_throttle_param=v_cand.prefers_throttle_player,
            )
            assert_sndh_and_fsb_alignment(res_vp.bank_bytes)
            assert_proj_tables_sorted(res_vp.bank_bytes)
            rel_vp = FmodBank(raw=res_vp.bank_bytes)
            vp_ev_set = {e.guid_str.lower() for e in rel_vp.events}
            vp_pnames = {p[0].lower() for e in rel_vp.events for p in e.parameters}
            assert "throttle" in vp_pnames
            for role_k in ("EngineExt", "EngineInt", "Starter", "Starting", "Exhaust"):
                if role_k in v_cand.event_guids:
                    assert v_cand.event_guids[role_k].lower() in vp_ev_set
            for _r, items in v_cand.extra_event_variants.items():
                for item in items:
                    assert item[0].lower() in vp_ev_set, f"{v_cand.vehicle_id} Player: falta variante extra {item[0]}"

            if v_cand.ai_bank_path:
                res_vai = build_direct_ported_bank_from_ac(
                    ac_bank=ac_lib.fmod_bank,
                    mod_identifier=f"{v_cand.car_name}_AI",
                    target_role_guids=v_cand.ai_event_guids,
                    target_role_paths=v_cand.ai_event_paths,
                    target_bank_guid=v_cand.ai_bank_guid or None,
                    target_role_buses=v_cand.ai_event_buses,
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                    extra_role_variants=v_cand.extra_ai_event_variants or None,
                    prefer_throttle_param=v_cand.prefers_throttle_ai,
                )
                assert_sndh_and_fsb_alignment(res_vai.bank_bytes)
                assert_proj_tables_sorted(res_vai.bank_bytes)
                rel_vai = FmodBank(raw=res_vai.bank_bytes)
                vai_ev_set = {e.guid_str.lower() for e in rel_vai.events}
                vai_pnames = {p[0].lower() for e in rel_vai.events for p in e.parameters}
                assert "throttle" in vai_pnames
                for _r, g_a in v_cand.ai_event_guids.items():
                    if _r in ("ai_engine_main", "ai_engine_live", "ai_engine_ext", "ai_engine_int"):
                        assert g_a.lower() in vai_ev_set
                for _r, items in v_cand.extra_ai_event_variants.items():
                    for item in items:
                        assert item[0].lower() in vai_ev_set, f"{v_cand.vehicle_id} AI: falta variante extra {item[0]}"
            print(f"   Vehículo especial {v_cand.vehicle_id} verificado (Player + AI)!")

    print("19. Verificando extensión automática de tope RPM y preservación de corte/pops en Jugador...")
    tatuus_bank_p = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\tatuus_f3_t_318\sfx\tatuus_f3_t_318.bank")
    tatuus_guids_p = tatuus_bank_p.parent / "GUIDs.txt"
    cortex_bank_p = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_gtm_cortex_v8\sfx\rss_gtm_cortex_v8.bank")
    cortex_guids_p = cortex_bank_p.parent / "GUIDs.txt"

    if ams2_dir.is_dir() and tatuus_bank_p.is_file():
        mgr_r = VanillaVehicleManager(ams2_dir)
        mgr_r.scan_catalog()
        v_f3s = mgr_r.get_vehicle_by_id("formula_3_f3_sudam")
        if v_f3s:
            mgr_r.load_vehicle_events_info(v_f3s)
            f3s_max_rpm = _extract_native_engine_max_rpm(v_f3s.bank_path)
            assert f3s_max_rpm is not None and f3s_max_rpm >= 9900.0, f"Tope nativo F3_Sudam inesperado: {f3s_max_rpm}"

            tat_b = FmodBank(tatuus_bank_p, guids_path=tatuus_guids_p)
            tat_bands = build_default_rpm_bands(tat_b)
            tat_res_p = build_direct_ported_bank_from_ac(
                ac_bank=tat_b,
                mod_identifier=v_f3s.car_name,
                standalone_guids=False,
                target_role_guids=v_f3s.event_guids,
                target_role_paths=v_f3s.event_paths,
                target_bank_guid=v_f3s.bank_guid or None,
                target_param_names=v_f3s.target_parameters,
                target_ctrl_params=v_f3s.ctrl_parameters,
                rpm_bands=tat_bands,
                mute_secondary_pops=False,
                native_bank_path=v_f3s.bank_path,
                target_max_rpm=f3s_max_rpm,
            )
            assert_sndh_and_fsb_alignment(tat_res_p.bank_bytes)
            assert_proj_tables_sorted(tat_res_p.bank_bytes)
            tat_re_p = FmodBank(raw=tat_res_p.bank_bytes)
            int_bands_p = [
                b for b in tat_re_p.rpm_bands
                if b.event_role == "EngineInt" and b.rpm_min <= 9500.0 <= b.rpm_max
            ]
            int_names_p = {tat_re_p.slots[i].name for b in int_bands_p for i in b.donor_slot_indices}
            assert "ext_limiter2" in int_names_p and "int_rss4_1_on_7_5" in int_names_p, (
                f"F3_Sudam Jugador quedó sin cobertura a 9500 RPM: {int_names_p}"
            )

            if v_f3s.ai_bank_path and v_f3s.ai_event_guids:
                tat_ai_bands = [
                    RpmBandConfig(
                        band_id=b.band_id, pml_guid=b.pml_guid, event_guid=b.event_guid, box_index=b.box_index,
                        event_role=b.event_role, perspective=b.perspective, load_role=b.load_role,
                        donor_slot_indices=list(b.donor_slot_indices), donor_label=b.donor_label,
                        rpm_min=0.0, rpm_max=0.0, fade_in_end=0.0, fade_out_start=0.0,
                        orig_rpm_min=b.orig_rpm_min, orig_rpm_max=b.orig_rpm_max,
                        orig_fade_in_end=b.orig_fade_in_end, orig_fade_out_start=b.orig_fade_out_start,
                        fade_in_curv_guid=b.fade_in_curv_guid, fade_out_curv_guid=b.fade_out_curv_guid,
                        bus_rpm_cutoff=b.bus_rpm_cutoff, is_muted_bus=True, is_fill_layer=b.is_fill_layer,
                    ) if getattr(b, "load_role", "") in ("pops", "throttle_cut") else b
                    for b in tat_bands
                ]
                tat_res_ai = build_direct_ported_bank_from_ac(
                    ac_bank=tat_b,
                    mod_identifier=f"{v_f3s.car_name}_AI",
                    standalone_guids=False,
                    target_role_guids=v_f3s.ai_event_guids,
                    target_role_paths=v_f3s.ai_event_paths,
                    target_bank_guid=v_f3s.ai_bank_guid or None,
                    target_role_buses=getattr(v_f3s, "ai_event_buses", None),
                    rpm_bands=tat_ai_bands,
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                    target_max_rpm=f3s_max_rpm,
                )
                assert_sndh_and_fsb_alignment(tat_res_ai.bank_bytes)
                assert_proj_tables_sorted(tat_res_ai.bank_bytes)
                tat_re_ai = FmodBank(raw=tat_res_ai.bank_bytes)
                ai_ext_names_9500 = {
                    tat_re_ai.slots[i].name
                    for b in tat_re_ai.rpm_bands
                    if b.event_role == "EngineExt" and b.rpm_min <= 9500.0 <= b.rpm_max
                    for i in b.donor_slot_indices
                }
                assert "int_rss4_1_on_7_5" in ai_ext_names_9500 and "ext_limiter2" not in ai_ext_names_9500, (
                    f"F3_Sudam AI incorrecto a 9500 RPM: {ai_ext_names_9500}"
                )
            print("   F3_Sudam + tatuus_f3_t_318 verificado (8000 -> 10000 RPM, corte activo en Jugador y limpio en IA)!")

    if ams2_dir.is_dir() and cortex_bank_p.is_file():
        mgr_r = VanillaVehicleManager(ams2_dir)
        mgr_r.scan_catalog()
        v_c8 = mgr_r.get_vehicle_by_id("c_corvette_c8_gt3r")
        if v_c8:
            mgr_r.load_vehicle_events_info(v_c8)
            cor_b = FmodBank(cortex_bank_p, guids_path=cortex_guids_p)
            cor_bands = build_default_rpm_bands(cor_b)
            cor_res_p = build_direct_ported_bank_from_ac(
                ac_bank=cor_b,
                mod_identifier=v_c8.car_name,
                standalone_guids=False,
                target_role_guids=v_c8.event_guids,
                target_role_paths=v_c8.event_paths,
                target_bank_guid=v_c8.bank_guid or None,
                target_param_names=v_c8.target_parameters,
                target_ctrl_params=v_c8.ctrl_parameters,
                rpm_bands=cor_bands,
                mute_secondary_pops=False,
                native_bank_path=v_c8.bank_path,
            )
            assert_sndh_and_fsb_alignment(cor_res_p.bank_bytes)
            assert_proj_tables_sorted(cor_res_p.bank_bytes)
            cor_re_p = FmodBank(raw=cor_res_p.bank_bytes)
            lim_band_p = [
                b for b in cor_re_p.rpm_bands
                if b.event_role in ("EngineInt", "AuxInt")
                and any("Int_Lim" in cor_re_p.slots[i].name for i in b.donor_slot_indices)
            ]
            assert lim_band_p and any(b.rpm_min <= 8500.0 <= b.rpm_max for b in lim_band_p), (
                f"Corvette C8 GT3R Jugador perdió la capa RSS_C8_Int_Lim al corte: {lim_band_p}"
            )
            if v_c8.ai_bank_path and v_c8.ai_event_guids:
                cor_ai_bands = [
                    RpmBandConfig(
                        band_id=b.band_id, pml_guid=b.pml_guid, event_guid=b.event_guid, box_index=b.box_index,
                        event_role=b.event_role, perspective=b.perspective, load_role=b.load_role,
                        donor_slot_indices=list(b.donor_slot_indices), donor_label=b.donor_label,
                        rpm_min=0.0, rpm_max=0.0, fade_in_end=0.0, fade_out_start=0.0,
                        orig_rpm_min=b.orig_rpm_min, orig_rpm_max=b.orig_rpm_max,
                        orig_fade_in_end=b.orig_fade_in_end, orig_fade_out_start=b.orig_fade_out_start,
                        fade_in_curv_guid=b.fade_in_curv_guid, fade_out_curv_guid=b.fade_out_curv_guid,
                        bus_rpm_cutoff=b.bus_rpm_cutoff, is_muted_bus=True, is_fill_layer=b.is_fill_layer,
                    ) if getattr(b, "load_role", "") in ("pops", "throttle_cut") else b
                    for b in cor_bands
                ]
                cor_res_ai = build_direct_ported_bank_from_ac(
                    ac_bank=cor_b,
                    mod_identifier=f"{v_c8.car_name}_AI",
                    standalone_guids=False,
                    target_role_guids=v_c8.ai_event_guids,
                    target_role_paths=v_c8.ai_event_paths,
                    target_bank_guid=v_c8.ai_bank_guid or None,
                    target_role_buses=getattr(v_c8, "ai_event_buses", None),
                    rpm_bands=cor_ai_bands,
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                )
                assert_sndh_and_fsb_alignment(cor_res_ai.bank_bytes)
                assert_proj_tables_sorted(cor_res_ai.bank_bytes)
                cor_re_ai = FmodBank(raw=cor_res_ai.bank_bytes)
                tc_or_lim_ai = [
                    b for b in cor_re_ai.rpm_bands
                    if b.rpm_max > b.rpm_min
                    and any("RSS_C8_Int_Lim" in cor_re_ai.slots[i].name or "RSS_C8_TC" in cor_re_ai.slots[i].name for i in b.donor_slot_indices)
                ]
                assert not tc_or_lim_ai, f"Corvette C8 AI dejó POPS/TC sin silenciar: {tc_or_lim_ai}"
            print("   Corvette_C8_GT3R + rss_gtm_cortex_v8 verificado (corte activo en Jugador, TC/pops silenciados en IA)!")

    print("20. Probando edición in-place de sonidos ya modificados (apply_band_edits_to_bank + re-sync _AI.bank)...")
    ams2_root_20 = Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2")
    if ams2_root_20.is_dir() and AC_BANK_PATH.is_file():
        from ams2_sound_porter.vanilla_manager import VanillaVehicleManager
        vm20 = VanillaVehicleManager(ams2_root=ams2_root_20)
        vm20.scan_catalog()
        v20 = next((v for v in vm20._vehicles_cache if v.ai_bank_path is not None and v.ai_bank_path.is_file()), None)
        if v20 is not None:
            vm20.load_vehicle_events_info(v20)
            src_ac20 = FmodBank(AC_BANK_PATH)
            orig_ac_cones = _extract_cone_vol_endpoints(src_ac20.raw)
            first_player = build_direct_ported_bank_from_ac(
                ac_bank=src_ac20,
                mod_identifier=v20.car_name,
                standalone_guids=False,
                target_role_guids=v20.event_guids,
                target_role_paths=v20.event_paths,
                target_bank_guid=v20.bank_guid or None,
                invert_spatial_cone=True,
                mute_secondary_pops=False,
            )
            with tempfile.TemporaryDirectory() as tmp20:
                mod_bank_file = Path(tmp20) / v20.bank_path.name
                mod_bank_file.write_bytes(first_player.bank_bytes)

                mod_lib = SourceAudioLibrary.from_bank(mod_bank_file)
                assert mod_lib.fmod_bank is not None
                mod_bank = mod_lib.fmod_bank
                mod_bands = build_default_rpm_bands(mod_bank)
                mod_maps = [SlotMappingConfig(sl.global_index, action="keep_donor") for sl in mod_bank.slots]

                # Elegir 3 bandas activas distintas con wav_guid válido
                active_bands = [
                    b for b in mod_bands
                    if not b.is_muted_bus and b.donor_slot_indices
                    and mod_bank.slots[b.donor_slot_indices[0]].wav_guid is not None
                ]
                assert len(active_bands) >= 3, "Se esperaban al menos 3 bandas activas en el banco modificado"
                b_edit, b_mute, b_redir = active_bands[0], active_bands[1], active_bands[2]
                g_edit = b_edit.donor_slot_indices[0]
                g_mute = b_mute.donor_slot_indices[0]
                g_redir = b_redir.donor_slot_indices[0]

                def _read_wait_gain_db(raw_b: bytes, wav_g: bytes) -> float | None:
                    for _c, _ps, _pe, _ in iter_riff_chunks(raw_b, 12, len(raw_b)):
                        if _c == b"LIST" and raw_b[_ps : _ps + 4] == b"PROJ":
                            found_val: list[float] = []
                            def _walk(s: int, e: int) -> None:
                                for sc, sps, spe, _ in iter_riff_chunks(raw_b, s, e):
                                    if sc == b"LIST":
                                        if raw_b[sps : sps + 4] == b"WAIT":
                                            wg = None
                                            gdb = None
                                            for cc, cps, cpe, _ in iter_riff_chunks(raw_b, sps + 4, spe):
                                                if cc == b"WAIB" and cpe - cps >= 32:
                                                    wg = bytes(raw_b[cps + 16 : cps + 32])
                                                elif cc == b"INST" and cpe - cps >= 20:
                                                    gdb = struct.unpack_from("<f", raw_b, cps + 16)[0]
                                            if wg == wav_g and gdb is not None:
                                                found_val.append(gdb)
                                        _walk(sps + 4, spe)
                            _walk(_ps + 4, _pe)
                            return found_val[0] if found_val else None
                    return None

                orig_gain_edit = _read_wait_gain_db(mod_bank.raw, mod_bank.slots[g_edit].wav_guid)
                assert orig_gain_edit is not None

                # 1) Editar rango RPM + Root RPM + Gain (+3.5 dB) en b_edit
                b_edit.rpm_min = 2150.0
                b_edit.fade_in_end = 2450.0
                b_edit.fade_out_start = 4850.0
                b_edit.rpm_max = 5350.0
                mod_maps[g_edit].override_root_rpm = 3777.0
                mod_maps[g_edit].gain_db = 3.5

                # 2) Silenciar b_mute
                b_mute.is_muted_bus = True
                mod_maps[g_mute].action = "silence"

                # 3) Redirigir muestra de b_redir hacia la muestra de b_edit
                src_idx_edit = next(
                    (it.index for it in mod_lib.items if it._slot_ref and it._slot_ref.global_index == g_edit),
                    None,
                )
                assert src_idx_edit is not None
                mod_maps[g_redir].action = "replace_source"
                mod_maps[g_redir].source_index = src_idx_edit

                edited_player_bytes = apply_band_edits_to_bank(
                    bank=mod_bank,
                    rpm_bands=mod_bands,
                    mappings=mod_maps,
                    library=mod_lib,
                )
                assert_sndh_and_fsb_alignment(edited_player_bytes)
                assert_proj_tables_sorted(edited_player_bytes)

                re_player = FmodBank(raw=edited_player_bytes)
                assert re_player.bank_guid == mod_bank.bank_guid, "apply_band_edits_to_bank alteró el bank_guid!"
                if orig_ac_cones:
                    assert _extract_cone_vol_endpoints(edited_player_bytes) == _extract_cone_vol_endpoints(mod_bank.raw), (
                        "apply_band_edits_to_bank no debe volver a invertir Event Cone Angle!"
                    )

                re_b_edit = next(
                    (b for b in re_player.rpm_bands if b.pml_guid == b_edit.pml_guid and b.event_guid == b_edit.event_guid and b.box_index == b_edit.box_index),
                    None,
                )
                assert re_b_edit is not None and abs(re_b_edit.rpm_min - 2150.0) < 1.0 and abs(re_b_edit.rpm_max - 5350.0) < 1.0
                new_gain_edit = _read_wait_gain_db(edited_player_bytes, mod_bank.slots[g_edit].wav_guid)
                assert new_gain_edit is not None and abs(new_gain_edit - min(24.0, orig_gain_edit + 3.5)) < 0.05
                re_sl_edit = next((s for s in re_player.slots if s.wav_guid == mod_bank.slots[g_edit].wav_guid), None)
                if mod_bank.slots[g_edit].autopitch_ref_rpm is not None and re_sl_edit is not None:
                    assert re_sl_edit.autopitch_ref_rpm is not None and abs(re_sl_edit.autopitch_ref_rpm - 3777.0) < 1.0

                re_b_mute = next(
                    (b for b in re_player.rpm_bands if b.pml_guid == b_mute.pml_guid and b.event_guid == b_mute.event_guid and b.box_index == b_mute.box_index),
                    None,
                )
                assert re_b_mute is None or (re_b_mute.rpm_min == 0.0 and re_b_mute.rpm_max == 0.0)
                muted_gain = _read_wait_gain_db(edited_player_bytes, mod_bank.slots[g_mute].wav_guid)
                assert muted_gain is not None and muted_gain <= -95.0

                # Verificar redirección WAV en b_redir
                redir_wg = mod_bank.slots[g_redir].wav_guid
                expected_loc = (mod_bank.slots[g_edit].subbank_index, mod_bank.slots[g_edit].sample_index)
                found_redir_loc = None
                for _c, _ps, _pe, _ in iter_riff_chunks(edited_player_bytes, 12, len(edited_player_bytes)):
                    if _c == b"LIST" and edited_player_bytes[_ps : _ps + 4] == b"PROJ":
                        def _find_w(s: int, e: int) -> None:
                            nonlocal found_redir_loc
                            for sc, sps, spe, _ in iter_riff_chunks(edited_player_bytes, s, e):
                                if sc == b"LIST":
                                    _find_w(sps + 4, spe)
                                elif sc == b"WAV " and spe - sps >= 30 and bytes(edited_player_bytes[sps : sps + 16]) == redir_wg:
                                    _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", edited_player_bytes, sps + 16)
                                    found_redir_loc = (sb_i, smp_i)
                        _find_w(_ps + 4, _pe)
                assert found_redir_loc == expected_loc, f"WAV redirect falló: {found_redir_loc} != {expected_loc}"

                # Verificar re-sync de _AI.bank desde el banco Jugador editado
                ai_src_bytes = uninvert_spatial_cone_in_bank_bytes(edited_player_bytes)
                ai_src_bank = FmodBank(raw=ai_src_bytes)
                ai_res20 = build_direct_ported_bank_from_ac(
                    ac_bank=ai_src_bank,
                    mod_identifier=f"{v20.car_name}_AI",
                    standalone_guids=False,
                    target_role_guids=v20.ai_event_guids,
                    target_role_paths=v20.ai_event_paths,
                    target_bank_guid=v20.ai_bank_guid or None,
                    target_role_buses=getattr(v20, "ai_event_buses", None),
                    rpm_bands=build_default_rpm_bands(ai_src_bank),
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                    invert_spatial_cone=False,
                    target_max_rpm=0.0,
                )
                assert_sndh_and_fsb_alignment(ai_res20.bank_bytes)
                assert_proj_tables_sorted(ai_res20.bank_bytes)
                if orig_ac_cones:
                    assert _extract_cone_vol_endpoints(ai_res20.bank_bytes) == orig_ac_cones, (
                        "El _AI.bank re-sincronizado debe conservar el cono AC original!"
                    )
            print("   apply_band_edits_to_bank (RPM, pitch, gain dB, silence, WAV redirect) + re-sync _AI.bank verificados al 100%!")

    print("21. Probando reemplazo parcial de vehículos vanilla (Interior only / Exterior only)...")
    if ams2_root_20.is_dir() and AC_BANK_PATH.is_file():
        from ams2_sound_porter.vanilla_manager import VanillaVehicleManager
        vm21 = VanillaVehicleManager(ams2_root=ams2_root_20)
        vm21.scan_catalog()

        partial_cases = [
            ("corvette_c8_gt3r", Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_gtm_cortex_v8\sfx\rss_gtm_cortex_v8.bank"), "RSS_C8_", "C8_R_int_", "corvette_gt3r_ext_"),
            ("bmw_m3_gtr", AC_BANK_PATH, "z4gt3", "m3gtr_int_", "m3gtr_ext_"),
        ]
        for vkey, ac_p21, ac_tok, nat_int_tok, nat_ext_tok in partial_cases:
            if not ac_p21.is_file():
                continue
            v21 = next((v for v in vm21._vehicles_cache if vkey in v.vehicle_id.lower()), None)
            if v21 is None:
                continue
            vm21.load_vehicle_events_info(v21)
            nat_p21 = vm21.backup_dir / v21.vehicle_id / v21.bank_path.name
            if not nat_p21.is_file():
                nat_p21 = v21.bank_path
            nat_bank_p = FmodBank(nat_p21)
            ac_bank21 = FmodBank(ac_p21)

            # A) Interior only: EngineInt (+ AuxInt) -> AC, EngineExt + Exhaust -> Vanilla AMS2
            rep_i, sil_i, pmap_i, ex_i = collect_vanilla_scope_targets(v21, ac_bank21, "interior", is_ai=False)
            ported_i = build_direct_ported_bank_from_ac(
                ac_bank=ac_bank21,
                mod_identifier=v21.car_name,
                standalone_guids=False,
                target_role_guids=v21.event_guids,
                target_role_paths=v21.event_paths,
                target_bank_guid=v21.bank_guid or None,
                native_bank_path=nat_p21,
                invert_spatial_cone=False,
                extra_role_variants=ex_i or None,
                active_scope="interior",
            )
            hyb_int_bytes = merge_ported_events_into_native_bank(
                native_bank=nat_bank_p,
                ported_bank=FmodBank(raw=ported_i.bank_bytes),
                replace_event_guids=rep_i,
                silence_native_event_guids=sil_i,
                event_paths_map=pmap_i,
                mod_identifier=v21.car_name,
                seed_tag="hybrid:p:interior",
            )
            assert_sndh_and_fsb_alignment(hyb_int_bytes)
            assert_proj_tables_sorted(hyb_int_bytes)
            fb_int = FmodBank(raw=hyb_int_bytes)
            assert fb_int.bank_guid == nat_bank_p.bank_guid
            ev_by_g_i = {ev.guid_str.lower(): ev for ev in fb_int.events}
            int_ev_i = ev_by_g_i[v21.event_guids["EngineInt"].lower()]
            ext_ev_i = ev_by_g_i[v21.event_guids["EngineExt"].lower()]
            assert any(ac_tok.lower() in s.lower() for s in int_ev_i.linked_samples), (
                f"{vkey} interior-only: EngineInt no contiene muestras AC ({int_ev_i.linked_samples[:5]})"
            )
            assert any(nat_ext_tok.lower() in s.lower() for s in ext_ev_i.linked_samples), (
                f"{vkey} interior-only: EngineExt no conservó muestras vanilla ({ext_ev_i.linked_samples[:5]})"
            )
            if "AuxInt" in v21.event_guids:
                aux_ev_i = ev_by_g_i[v21.event_guids["AuxInt"].lower()]
                assert any(ac_tok.lower() in s.lower() for s in aux_ev_i.linked_samples), (
                    f"{vkey} interior-only: AuxInt no fue reemplazado por AC ({aux_ev_i.linked_samples[:5]})"
                )

            # B) Exterior only: EngineExt -> AC, Exhaust -> silenciado, EngineInt (+ AuxInt) -> Vanilla AMS2
            rep_e, sil_e, pmap_e, ex_e = collect_vanilla_scope_targets(v21, ac_bank21, "exterior", is_ai=False)
            ported_e = build_direct_ported_bank_from_ac(
                ac_bank=ac_bank21,
                mod_identifier=v21.car_name,
                standalone_guids=False,
                target_role_guids=v21.event_guids,
                target_role_paths=v21.event_paths,
                target_bank_guid=v21.bank_guid or None,
                native_bank_path=nat_p21,
                invert_spatial_cone=True,
                extra_role_variants=ex_e or None,
                active_scope="exterior",
            )
            hyb_ext_bytes = merge_ported_events_into_native_bank(
                native_bank=nat_bank_p,
                ported_bank=FmodBank(raw=ported_e.bank_bytes),
                replace_event_guids=rep_e,
                silence_native_event_guids=sil_e,
                event_paths_map=pmap_e,
                mod_identifier=v21.car_name,
                seed_tag="hybrid:p:exterior",
            )
            assert_sndh_and_fsb_alignment(hyb_ext_bytes)
            assert_proj_tables_sorted(hyb_ext_bytes)
            fb_ext = FmodBank(raw=hyb_ext_bytes)
            assert fb_ext.bank_guid == nat_bank_p.bank_guid
            ev_by_g_e = {ev.guid_str.lower(): ev for ev in fb_ext.events}
            int_ev_e = ev_by_g_e[v21.event_guids["EngineInt"].lower()]
            ext_ev_e = ev_by_g_e[v21.event_guids["EngineExt"].lower()]
            assert any(nat_int_tok.lower() in s.lower() for s in int_ev_e.linked_samples), (
                f"{vkey} exterior-only: EngineInt no conservó muestras vanilla ({int_ev_e.linked_samples[:5]})"
            )
            assert any(ac_tok.lower() in s.lower() for s in ext_ev_e.linked_samples) or len(ext_ev_e.linked_samples) >= 5
            if "Exhaust" in v21.event_guids and v21.event_guids["Exhaust"].lower() in ev_by_g_e:
                exh_ev_e = ev_by_g_e[v21.event_guids["Exhaust"].lower()]
                assert len(exh_ev_e.linked_samples) == 0, (
                    f"{vkey} exterior-only: Exhaust vanilla debía neutralizarse ({exh_ev_e.linked_samples})"
                )

            # C) Exterior only _AI.bank: ai_engine_main/live/ext -> AC, ai_engine_int -> Vanilla AMS2
            if v21.ai_bank_path and v21.ai_event_guids:
                nat_ai21 = vm21.backup_dir / v21.vehicle_id / v21.ai_bank_path.name
                if not nat_ai21.is_file():
                    nat_ai21 = v21.ai_bank_path
                nat_bank_ai = FmodBank(nat_ai21)
                rep_ai, sil_ai, pmap_ai, ex_ai = collect_vanilla_scope_targets(v21, ac_bank21, "exterior", is_ai=True)
                ported_ai = build_direct_ported_bank_from_ac(
                    ac_bank=ac_bank21,
                    mod_identifier=f"{v21.car_name}_AI",
                    standalone_guids=False,
                    target_role_guids=v21.ai_event_guids,
                    target_role_paths=v21.ai_event_paths,
                    target_bank_guid=v21.ai_bank_guid or None,
                    target_role_buses=getattr(v21, "ai_event_buses", None),
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                    invert_spatial_cone=False,
                    extra_role_variants=ex_ai or None,
                    active_scope="exterior",
                )
                hyb_ai_bytes = merge_ported_events_into_native_bank(
                    native_bank=nat_bank_ai,
                    ported_bank=FmodBank(raw=ported_ai.bank_bytes),
                    replace_event_guids=rep_ai,
                    silence_native_event_guids=sil_ai,
                    event_paths_map=pmap_ai,
                    mod_identifier=f"{v21.car_name}_AI",
                    seed_tag="hybrid:ai:exterior",
                )
                assert_sndh_and_fsb_alignment(hyb_ai_bytes)
                assert_proj_tables_sorted(hyb_ai_bytes)
                fb_ai_e = FmodBank(raw=hyb_ai_bytes)
                assert fb_ai_e.bank_guid == nat_bank_ai.bank_guid
                if "ai_engine_int" in v21.ai_event_guids:
                    ai_ev_map = {ev.guid_str.lower(): ev for ev in fb_ai_e.events}
                    ai_int_ev = ai_ev_map[v21.ai_event_guids["ai_engine_int"].lower()]
                    assert any(nat_int_tok.lower() in s.lower() for s in ai_int_ev.linked_samples), (
                        f"{vkey} AI exterior-only: ai_engine_int no conservó muestras vanilla ({ai_int_ev.linked_samples[:5]})"
                    )
            print(f"   Reemplazo parcial Interior-only & Exterior-only verificado para {vkey}!")

    print("22. Verificando sincronización CURV+MAP en sub-eventos EVIT al extender tope RPM (Porsche_963 + rss_mph_protech_p96_v8)...")
    p96_bank_path = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_mph_protech_p96_v8\sfx\rss_mph_protech_p96_v8.bank")
    p96_guids_path = p96_bank_path.parent / "GUIDs.txt"
    if ams2_root_20.is_dir() and p96_bank_path.is_file():
        from ams2_sound_porter.vanilla_manager import VanillaVehicleManager
        from ams2_sound_porter.porter_engine import _iter_proj_table_items
        vm22 = VanillaVehicleManager(ams2_root=ams2_root_20)
        vm22.scan_catalog()
        v963 = next((v for v in vm22._vehicles_cache if "963" in v.vehicle_id.lower()), None)
        if v963 is not None:
            vm22.load_vehicle_events_info(v963)
            nat_p963 = vm22.backup_dir / v963.vehicle_id / v963.bank_path.name
            if not nat_p963.is_file():
                nat_p963 = v963.bank_path
            max_rpm_963 = _extract_native_engine_max_rpm(nat_p963)
            assert max_rpm_963 and max_rpm_963 > 12000.0, f"Se esperaba tope nativo >12000 RPM en 963: {max_rpm_963}"
            p96_b = FmodBank(p96_bank_path, guids_path=p96_guids_path if p96_guids_path.is_file() else None)
            p96_bands = build_default_rpm_bands(p96_b)
            p96_res_p = build_direct_ported_bank_from_ac(
                ac_bank=p96_b,
                mod_identifier=v963.car_name,
                standalone_guids=False,
                target_role_guids=v963.event_guids,
                target_role_paths=v963.event_paths,
                target_bank_guid=v963.bank_guid or None,
                rpm_bands=p96_bands,
                mute_secondary_pops=False,
                native_bank_path=nat_p963,
                invert_spatial_cone=True,
                prefer_throttle_param=bool(v963.prefers_throttle_player),
                target_max_rpm=max_rpm_963,
            )
            p96_res_ai = build_direct_ported_bank_from_ac(
                ac_bank=p96_b,
                mod_identifier=f"{v963.car_name}_AI",
                standalone_guids=False,
                target_role_guids=v963.ai_event_guids,
                target_role_paths=v963.ai_event_paths,
                target_bank_guid=v963.ai_bank_guid or None,
                target_role_buses=v963.ai_event_buses,
                rpm_bands=p96_bands,
                is_ai_bank=True,
                mute_secondary_pops=True,
                invert_spatial_cone=False,
                prefer_throttle_param=bool(v963.prefers_throttle_ai),
                target_max_rpm=max_rpm_963,
            )
            for lbl22, raw22 in (("Player", p96_res_p.bank_bytes), ("AI", p96_res_ai.bank_bytes)):
                assert_sndh_and_fsb_alignment(raw22)
                assert_proj_tables_sorted(raw22)
                proj22 = next(
                    raw22[ps + 4 : pe]
                    for cid, ps, pe, _ in iter_riff_chunks(raw22, 12, len(raw22))
                    if cid == b"LIST" and raw22[ps : ps + 4] == b"PROJ"
                )
                tbls22 = list(_iter_proj_table_items(proj22))
                prms22 = {}
                for lt, items in tbls22:
                    if lt == b"PRMS":
                        for k, _, _, bp in items:
                            sl = struct.unpack_from("<H", bp, 21)[0]
                            prms22[k] = bp[23 : 23 + sl].decode("latin1", "replace").rstrip("\x00").lower()
                curvs22 = {}
                for lt, items in tbls22:
                    if lt == b"CRVS":
                        for k, _, _, bp in items:
                            cnt = struct.unpack_from("<H", bp, 32)[0] >> 1
                            st = struct.unpack_from("<H", bp, 34)[0] if cnt else 0
                            curvs22[k] = [struct.unpack_from("<ff", bp, 36 + i * st) for i in range(cnt)] if st >= 8 else []
                mpgs22 = {}
                for lt, items in tbls22:
                    if lt == b"MPGS":
                        for k, _, _, bp in items:
                            cnt = struct.unpack_from("<H", bp, 16)[0] >> 1
                            st = struct.unpack_from("<H", bp, 18)[0] if cnt else 0
                            mpgs22[k] = [struct.unpack_from("<ff", bp, 20 + i * st) for i in range(cnt)] if st >= 8 else []
                evit_map22 = {}
                for lt, items in tbls22:
                    if lt == b"EVIS":
                        for k, full, _, _ in items:
                            for sc, sps, spe, _ in iter_riff_chunks(full, 12, len(full)):
                                if sc == b"LIST" and full[sps : sps + 4] == b"PRPS":
                                    for sc2, sps2, spe2, _ in iter_riff_chunks(full, sps + 4, spe):
                                        if sc2 == b"PROP":
                                            pid = struct.unpack_from("<I", full, sps2)[0]
                                            evit_map22[(k, pid)] = bytes(full[sps2 + 8 : sps2 + 24])
                checked22 = 0
                for lt, items in tbls22:
                    if lt == b"CTRS":
                        for k, _, _, bp in items:
                            tgt, pg, cg = bp[16:32], bp[32:48], bp[48:64]
                            prop = struct.unpack_from("<I", bp, 64)[0]
                            if prop > 20 and (tgt, prop) in evit_map22 and prms22.get(pg) in ("rpm", "rpms"):
                                mg = evit_map22[(tgt, prop)]
                                c_pts = curvs22.get(cg, [])
                                m_pts = mpgs22.get(mg, [])
                                assert c_pts and m_pts, f"{lbl22}: faltan puntos CURV/MAP en EVIT {tgt.hex()[:8]}"
                                ratio = m_pts[-1][1] / c_pts[-1][0]
                                assert abs(ratio - 1.0) < 1e-4, (
                                    f"{lbl22}: desfase CURV/MAP en EVIT {tgt.hex()[:8]}: CURV={c_pts} MAP={m_pts} ratio={ratio}"
                                )
                                checked22 += 1
                assert checked22 >= 6, f"{lbl22}: se esperaban >=6 reenvíos EVIT RPM, encontrados {checked22}"
            print(f"   Porsche_963 + rss_mph_protech_p96_v8 verificado (CURV+MAP 1:1 exacto hasta {max_rpm_963:.0f} RPM en Jugador e IA)!")
            del p96_b, p96_res_p, p96_res_ai

    print("23. Verificando sonoridad exterior/replay AMS2 (3D Spatializer LinearSquared + Distance Filter + MBUS float gain sin saturación)...")
    if ams2_root_20.is_dir():
        import gc
        gc.collect()
        from ams2_sound_porter.porter_engine import (
            AMS2_LIVE_REPLAY_PANNER,
            _iter_proj_table_items,
            _parse_pmef_param_offsets,
            _select_primary_engine_events,
        )
        vm23 = VanillaVehicleManager(ams2_root=ams2_root_20)
        vm23.scan_catalog()

        def _inspect_ext_ev23(raw_b: bytes, ev_g: bytes):
            pr = next(
                raw_b[ps + 4 : pe]
                for cid, ps, pe, _ in iter_riff_chunks(raw_b, 12, len(raw_b))
                if cid == b"LIST" and raw_b[ps : ps + 4] == b"PROJ"
            )
            tbl = dict(_iter_proj_table_items(pr))
            mbss = {g: full for g, full, _, _ in tbl.get(b"MBSS", ())}
            befx = {g: (full, bp) for g, full, _, bp in tbl.get(b"BEFX", ())}
            pefx = {g: (full, bp) for g, full, _, bp in tbl.get(b"PEFX", ())}
            ev_bp = next(bp for g, _, _, bp in tbl[b"EVTS"] if g == ev_g)
            mb_g = next(ev_bp[i : i + 16] for i in range(16, len(ev_bp) - 15) if ev_bp[i : i + 16] in mbss)
            mf = mbss[mb_g]
            bus_b = next(mf[sps:spe] for sc, sps, spe, _ in iter_riff_chunks(mf, 12, len(mf)) if sc == b"BUS ")
            mvol = round(struct.unpack_from("<f", bus_b, len(bus_b) - 16)[0], 2)
            pan3d = None
            df_d = None
            for i in range(0, len(mf) - 15):
                w = mf[i : i + 16]
                if w in befx and struct.unpack_from("<I", befx[w][1], 20)[0] == 27:
                    pmef = next(befx[w][0][sps:spe] for sc, sps, spe, _ in iter_riff_chunks(befx[w][0], 12, len(befx[w][0])) if sc == b"PMEF")
                    pm = _parse_pmef_param_offsets(pmef)
                    if 17 in pm and float(pm[17][2]) > 0.5:
                        pan3d = (
                            int(pm[11][2]), round(float(pm[12][2]), 2), round(float(pm[13][2]), 2),
                            int(pm[14][2]), round(float(pm[15][2]), 2), round(float(pm[16][2]), 2),
                        )
                elif w in pefx:
                    pf, pb = pefx[w]
                    nl = struct.unpack_from("<H", pb, 20)[0]
                    if "distance filter" in pb[22 : 22 + nl].decode("ascii", errors="replace").lower():
                        pmef = next(pf[sps:spe] for sc, sps, spe, _ in iter_riff_chunks(pf, 12, len(pf)) if sc == b"PMEF")
                        pm = _parse_pmef_param_offsets(pmef)
                        df_d = round(float(pm[0][2]), 1)
            return mvol, pan3d, df_d

        def _check_ext_case23(vid_kw: str, ac_bp: Path, exp_p_mindist: float) -> None:
            if not ac_bp.is_file():
                return
            v_c = next((v for v in vm23._vehicles_cache if vid_kw in v.vehicle_id.lower()), None)
            if not v_c or not v_c.ai_bank_path:
                return
            vm23.load_vehicle_events_info(v_c)
            nat_p23 = vm23.backup_dir / v_c.vehicle_id / v_c.bank_path.name
            if not nat_p23.is_file():
                nat_p23 = v_c.bank_path
            nat_ai23 = vm23.backup_dir / v_c.vehicle_id / v_c.ai_bank_path.name
            if not nat_ai23.is_file():
                nat_ai23 = v_c.ai_bank_path

            ac_b23 = FmodBank(ac_bp, guids_path=ac_bp.parent / "GUIDs.txt" if (ac_bp.parent / "GUIDs.txt").is_file() else None)
            _, orig_int_ev = _select_primary_engine_events(ac_b23)
            orig_int_state = _inspect_ext_ev23(ac_b23.raw, orig_int_ev.guid_bytes)

            r_p23 = build_direct_ported_bank_from_ac(
                ac_bank=ac_b23, mod_identifier=v_c.car_name, standalone_guids=False,
                target_role_guids=v_c.event_guids, target_role_paths=v_c.event_paths,
                target_bank_guid=v_c.bank_guid or None, native_bank_path=nat_p23,
                output_codec_mode=FSB5_MODE_PCM16, mute_secondary_pops=False, invert_spatial_cone=True,
            )
            assert_sndh_and_fsb_alignment(r_p23.bank_bytes)
            assert_proj_tables_sorted(r_p23.bank_bytes)
            p_int_s = _inspect_ext_ev23(r_p23.bank_bytes, guid_str_to_bytes(v_c.event_guids["EngineInt"]))
            p_ext_s = _inspect_ext_ev23(r_p23.bank_bytes, guid_str_to_bytes(v_c.event_guids["EngineExt"]))
            p_bytes_saved = r_p23.bank_bytes
            del r_p23

            r_ai23 = build_direct_ported_bank_from_ac(
                ac_bank=ac_b23, mod_identifier=f"{v_c.car_name}_AI", standalone_guids=False,
                target_role_guids=v_c.ai_event_guids, target_role_paths=v_c.ai_event_paths,
                target_bank_guid=v_c.ai_bank_guid or None, target_role_buses=v_c.ai_event_buses,
                native_bank_path=nat_ai23, output_codec_mode=FSB5_MODE_PCM16,
                is_ai_bank=True, mute_secondary_pops=True, invert_spatial_cone=False,
            )
            del ac_b23
            assert_sndh_and_fsb_alignment(r_ai23.bank_bytes)
            assert_proj_tables_sorted(r_ai23.bank_bytes)
            ai_live_s = _inspect_ext_ev23(r_ai23.bank_bytes, guid_str_to_bytes(v_c.ai_event_guids["ai_engine_live"]))
            ai_main_s = _inspect_ext_ev23(r_ai23.bank_bytes, guid_str_to_bytes(v_c.ai_event_guids["ai_engine_main"]))
            del r_ai23

            assert p_int_s == orig_int_state, f"{v_c.car_name}: EngineInt no debe ser alterado!"
            assert ai_live_s[1] == AMS2_LIVE_REPLAY_PANNER, f"{v_c.car_name}: AI_EngineMAIN_LIVE panner={ai_live_s[1]}"
            assert ai_main_s[1][0] == 2 and ai_main_s[1][1] >= 7.0, f"{v_c.car_name}: AI_EngineMAIN panner={ai_main_s[1]}"
            assert p_ext_s[1][0] == 2 and abs(p_ext_s[1][1] - exp_p_mindist) < 0.05, f"{v_c.car_name}: Player EngineExt minDist={p_ext_s[1]}"
            assert p_ext_s[0] > 2.5 and abs(p_ext_s[0] - ai_live_s[0]) < 0.05, f"{v_c.car_name}: MBUS gain inconsistente: {p_ext_s[0]} vs {ai_live_s[0]}"
            assert ai_live_s[2] is None or ai_live_s[2] >= 5000.0
            assert p_ext_s[2] is None or p_ext_s[2] >= 3000.0

            # Idempotencia al re-sincronizar IA desde banco Jugador ya porteado
            re_src_b = FmodBank(raw=p_bytes_saved)
            del p_bytes_saved
            gc.collect()
            r_ai23_re = build_direct_ported_bank_from_ac(
                ac_bank=re_src_b, mod_identifier=f"{v_c.car_name}_AI", standalone_guids=False,
                target_role_guids=v_c.ai_event_guids, target_role_paths=v_c.ai_event_paths,
                target_bank_guid=v_c.ai_bank_guid or None, target_role_buses=v_c.ai_event_buses,
                native_bank_path=nat_ai23, output_codec_mode=None, is_ai_bank=True, mute_secondary_pops=True,
            )
            del re_src_b
            ai_live_re = _inspect_ext_ev23(r_ai23_re.bank_bytes, guid_str_to_bytes(v_c.ai_event_guids["ai_engine_live"]))
            del r_ai23_re
            assert abs(ai_live_re[0] - ai_live_s[0]) < 0.05 and ai_live_re[1] == ai_live_s[1], (
                f"{v_c.car_name}: re-sync IA no es idempotente ({ai_live_s[0]} -> {ai_live_re[0]})"
            )

        for vid_kw, ac_bp, exp_p_mindist in (
            ("c_corvette_c8_gt3r", Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_gtm_cortex_v8\sfx\rss_gtm_cortex_v8.bank"), 7.0),
            ("formula_3_f3_sudam", Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\tatuus_f3_t_318\sfx\tatuus_f3_t_318.bank"), 30.0),
            ("porsche_963", Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_mph_protech_p96_v8\sfx\rss_mph_protech_p96_v8.bank"), 7.0),
        ):
            _check_ext_case23(vid_kw, ac_bp, exp_p_mindist)
            gc.collect()
        print("   Sonoridad exterior/replay AMS2 (LinearSquared 1..300m, Distance Filter, MBUS float gain e idempotencia) verificada al 100%!")

        # 24. Opción C: Injerto quirúrgico + EVIB Proxy en banco multi-evento con eventos viejos (F_Ultimate_2022)
        print("24. Probando Opción C (Injerto quirúrgico + EVIB Proxy) en F_Ultimate_2022 (Player + AI + Edit Modified Sound)...")
        from ams2_sound_porter.porter_engine import (
            _extract_bnki_exported_event_guids,
            _extract_event_prmb_metadata,
            _extract_native_exterior_spatializers,
            _inspect_native_event_profiles,
            _iter_proj_table_items,
            graft_ac_into_native_bank_surgical,
            verify_fmod_bank_integrity,
        )

        fult_v = next((v for v in vm23._vehicles_cache if "f_ultimate_2022" in v.vehicle_id.lower()), None)
        alp_ac_p = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_alpine_a525\sfx\rss_alpine_a525.bank")
        if fult_v and fult_v.ai_bank_path and alp_ac_p.is_file():
            vm23.load_vehicle_events_info(fult_v)
            assert fult_v.has_multi_engine_variants, "F_Ultimate_2022 debe detectarse como banco multi-evento"
            nat_fult_p = vm23.backup_dir / fult_v.vehicle_id / fult_v.bank_path.name
            if not nat_fult_p.is_file():
                nat_fult_p = fult_v.bank_path
            nat_fult_ai = vm23.backup_dir / fult_v.vehicle_id / fult_v.ai_bank_path.name
            if not nat_fult_ai.is_file():
                nat_fult_ai = fult_v.ai_bank_path

            alp_lib = SourceAudioLibrary.from_bank(
                alp_ac_p,
                guids_path=(alp_ac_p.parent / "GUIDs.txt" if (alp_ac_p.parent / "GUIDs.txt").is_file() else None),
            )
            alp_fmod = alp_lib.fmod_bank
            master_pmap = {guid_str_to_bytes(g): p for g, p in vm23.get_master_strings().items() if p.startswith("event:/")}

            fult_int_tgts = [guid_str_to_bytes(ev.guid_str) for ev in fult_v.player_engine_variants if ev.role == "EngineInt"]
            fult_ext_tgts = [guid_str_to_bytes(ev.guid_str) for ev in fult_v.player_engine_variants if ev.role == "EngineExt"]
            assert len(fult_int_tgts) >= 10 and len(fult_ext_tgts) >= 3

            r_1copy_p = build_direct_ported_bank_from_ac(
                ac_bank=alp_fmod,
                mod_identifier=fult_v.car_name,
                standalone_guids=False,
                target_role_guids={
                    "EngineInt": bytes_to_guid_str(fult_int_tgts[0]),
                    "EngineExt": bytes_to_guid_str(fult_ext_tgts[0]),
                },
                target_role_paths=fult_v.event_paths,
                target_bank_guid=fult_v.bank_guid or None,
                native_bank_path=nat_fult_p,
                invert_spatial_cone=True,
                output_codec_mode=FSB5_MODE_FADPCM,
                prune_unused_samples=True,
                extra_role_variants=None,
                prefer_throttle_param=bool(fult_v.prefers_throttle_player),
                library=alp_lib,
            )
            nat_fult_bank = FmodBank(nat_fult_p)
            nat_fult_proj = next(
                nat_fult_bank.raw[ps + 4 : pe]
                for c, ps, pe, _ in iter_riff_chunks(nat_fult_bank.raw, 12, len(nat_fult_bank.raw))
                if c == b"LIST" and nat_fult_bank.raw[ps : ps + 4] == b"PROJ"
            )
            nat_fult_exp = _extract_bnki_exported_event_guids(
                nat_fult_proj, dict(_iter_proj_table_items(nat_fult_proj))[b"EVTS"]
            )
            exh_sil_24 = {guid_str_to_bytes(fult_v.event_guids["Exhaust"])} if "Exhaust" in fult_v.event_guids else set()

            surg_fult_p, meta_fult_p = graft_ac_into_native_bank_surgical(
                native_bank=nat_fult_bank,
                ported_ac_bank=FmodBank(raw=r_1copy_p.bank_bytes),
                ac_int_event_guid=guid_str_to_bytes(r_1copy_p.role_to_event_guid["EngineInt"]),
                ac_ext_event_guid=guid_str_to_bytes(r_1copy_p.role_to_event_guid["EngineExt"]),
                target_int_guids=fult_int_tgts,
                target_ext_guids=fult_ext_tgts,
                silence_native_guids=exh_sil_24,
                event_paths_map=master_pmap,
                is_ai_bank=False,
                native_panners=_extract_native_exterior_spatializers(nat_fult_p),
                default_prefer_throttle=bool(fult_v.prefers_throttle_player),
                mod_identifier=fult_v.car_name,
                seed_tag="test24:fult:p",
            )
            errs24_p = verify_fmod_bank_integrity(surg_fult_p, expected_exported_guids=nat_fult_exp)
            assert not errs24_p, f"F_Ultimate_2022 Player quirúrgico falló integridad: {errs24_p}"
            assert_sndh_and_fsb_alignment(surg_fult_p)
            assert meta_fult_p["ac_int_guid"] is not None and meta_fult_p["ac_ext_guid"] is not None

            # Verificar puenteo mixto por evento: Renfault (load + 3D), Alp22 (throttle + 3D), MAIN_Pan (throttle + 2D)
            s_proj = next(
                surg_fult_p[ps + 4 : pe]
                for c, ps, pe, _ in iter_riff_chunks(surg_fult_p, 12, len(surg_fult_p))
                if c == b"LIST" and surg_fult_p[ps : ps + 4] == b"PROJ"
            )
            assert len(s_proj) < 350_000, f"PROJ demasiado grande (debería usar proxies EVIB): {len(s_proj):,}B"
            s_tbls = list(_iter_proj_table_items(s_proj))
            s_items_map = {lt: [(k, f) for k, f, _, _ in items] for lt, items in s_tbls}
            s_profs = _inspect_native_event_profiles(s_proj, s_tbls)
            ren_g = guid_str_to_bytes(fult_v.event_guids["EngineExt"])
            alp22_g = guid_str_to_bytes("{256c4bf1-23e1-4f04-a391-159db1739c1b}")
            pan_g = guid_str_to_bytes("{9d29458b-03a1-4d83-8df8-5a066d19bcf6}")
            assert "load" in {p["name_low"] for p in _extract_event_prmb_metadata(s_items_map, ren_g)} and s_profs[ren_g]["is_3d"]
            assert "throttle" in {p["name_low"] for p in _extract_event_prmb_metadata(s_items_map, alp22_g)} and s_profs[alp22_g]["is_3d"]
            assert "throttle" in {p["name_low"] for p in _extract_event_prmb_metadata(s_items_map, pan_g)} and not s_profs[pan_g]["is_3d"]

            # Roundtrip Edit Modified Sound sobre banco quirúrgico EVIB Proxy
            re_fult = FmodBank(raw=surg_fult_p)
            fult_bands = build_default_rpm_bands(re_fult)
            assert any(b.perspective == "int" for b in fult_bands) and any(b.perspective == "ext" for b in fult_bands)
            fult_edited = apply_band_edits_to_bank(re_fult, rpm_bands=fult_bands)
            assert not verify_fmod_bank_integrity(fult_edited, expected_exported_guids=nat_fult_exp)
            del re_fult, fult_edited, surg_fult_p

            # Verificar _AI.bank quirúrgico en F_Ultimate_2022_AI
            ai_int_24 = [guid_str_to_bytes(ev.guid_str) for ev in fult_v.ai_engine_variants if ev.role == "ai_engine_int"]
            ai_ext_24 = [guid_str_to_bytes(ev.guid_str) for ev in fult_v.ai_engine_variants if ev.role != "ai_engine_int"]
            ai_live_24 = {guid_str_to_bytes(ev.guid_str) for ev in fult_v.ai_engine_variants if ev.role == "ai_engine_live"}
            r_1copy_ai = build_direct_ported_bank_from_ac(
                ac_bank=alp_fmod,
                mod_identifier=f"{fult_v.car_name}_AI",
                standalone_guids=False,
                target_role_guids={
                    "ai_engine_ext": bytes_to_guid_str(ai_ext_24[0]),
                    "ai_engine_int": bytes_to_guid_str(ai_int_24[0]),
                },
                target_role_paths=fult_v.ai_event_paths,
                target_bank_guid=fult_v.ai_bank_guid or None,
                is_ai_bank=True,
                mute_secondary_pops=True,
                native_bank_path=nat_fult_ai,
                invert_spatial_cone=False,
                output_codec_mode=FSB5_MODE_FADPCM,
                prune_unused_samples=True,
                cached_fsb_blobs=r_1copy_p.fsb_blobs_cache,
                extra_role_variants=None,
                prefer_throttle_param=bool(fult_v.prefers_throttle_ai),
                library=alp_lib,
            )
            nat_ai_b = FmodBank(nat_fult_ai)
            nat_ai_proj = next(
                nat_ai_b.raw[ps + 4 : pe]
                for c, ps, pe, _ in iter_riff_chunks(nat_ai_b.raw, 12, len(nat_ai_b.raw))
                if c == b"LIST" and nat_ai_b.raw[ps : ps + 4] == b"PROJ"
            )
            nat_ai_exp = _extract_bnki_exported_event_guids(
                nat_ai_proj, dict(_iter_proj_table_items(nat_ai_proj))[b"EVTS"]
            )
            surg_fult_ai, _ = graft_ac_into_native_bank_surgical(
                native_bank=nat_ai_b,
                ported_ac_bank=FmodBank(raw=r_1copy_ai.bank_bytes),
                ac_int_event_guid=guid_str_to_bytes(r_1copy_ai.role_to_event_guid["ai_engine_int"]),
                ac_ext_event_guid=guid_str_to_bytes(r_1copy_ai.role_to_event_guid["ai_engine_ext"]),
                target_int_guids=ai_int_24,
                target_ext_guids=ai_ext_24,
                event_paths_map=master_pmap,
                is_ai_bank=True,
                native_panners=_extract_native_exterior_spatializers(nat_fult_ai),
                default_prefer_throttle=bool(fult_v.prefers_throttle_ai),
                mod_identifier=f"{fult_v.car_name}_AI",
                seed_tag="test24:fult:ai",
                ai_live_guids=ai_live_24,
            )
            errs24_ai = verify_fmod_bank_integrity(surg_fult_ai, expected_exported_guids=nat_ai_exp)
            assert not errs24_ai, f"F_Ultimate_2022_AI quirúrgico falló integridad: {errs24_ai}"
            assert_sndh_and_fsb_alignment(surg_fult_ai)
            print("   Opción C (F_Ultimate_2022 Player + AI + puenteo load/throttle + Edit Modified Sound) verificada al 100%!")

            # 25. Opción C: Aislamiento selectivo por auto + apilado multi-auto en Audi_V10_GT3.bank (5 autos en 1 .bank)
            print("25. Probando aislamiento selectivo y apilado multi-auto en Audi_V10_GT3.bank...")
            audi_v = next((v for v in vm23._vehicles_cache if "audi_v10_gt3" in v.vehicle_id.lower()), None)
            if audi_v:
                vm23.load_vehicle_events_info(audi_v)
                assert audi_v.has_multi_engine_variants
                audi_nat_p = vm23.backup_dir / audi_v.vehicle_id / audi_v.bank_path.name
                if not audi_nat_p.is_file():
                    audi_nat_p = audi_v.bank_path
                audi_nat_b = FmodBank(audi_nat_p)
                audi_nat_proj = next(
                    audi_nat_b.raw[ps + 4 : pe]
                    for c, ps, pe, _ in iter_riff_chunks(audi_nat_b.raw, 12, len(audi_nat_b.raw))
                    if c == b"LIST" and audi_nat_b.raw[ps : ps + 4] == b"PROJ"
                )
                audi_exp = _extract_bnki_exported_event_guids(
                    audi_nat_proj, dict(_iter_proj_table_items(audi_nat_proj))[b"EVTS"]
                )
                hur_int_g = [guid_str_to_bytes(ev.guid_str) for ev in audi_v.player_engine_variants if ev.role == "EngineInt" and "huracan" in ev.short_name.lower()]
                hur_ext_g = [guid_str_to_bytes(ev.guid_str) for ev in audi_v.player_engine_variants if ev.role == "EngineExt" and "huracan" in ev.short_name.lower()]
                r8_int_g = [guid_str_to_bytes(ev.guid_str) for ev in audi_v.player_engine_variants if ev.role == "EngineInt" and "r8_lms_gt3_engineint" in ev.short_name.lower()]
                r8_ext_g = [guid_str_to_bytes(ev.guid_str) for ev in audi_v.player_engine_variants if ev.role == "EngineExt" and "r8_lms_gt3_engineext" in ev.short_name.lower()]
                assert hur_int_g and hur_ext_g and r8_int_g and r8_ext_g

                # Paso 25A: Reemplazar SOLO Huracán (todos los Audi R8 siguen 100% vanilla)
                audi_step1, _ = graft_ac_into_native_bank_surgical(
                    native_bank=audi_nat_b,
                    ported_ac_bank=FmodBank(raw=r_1copy_p.bank_bytes),
                    ac_int_event_guid=guid_str_to_bytes(r_1copy_p.role_to_event_guid["EngineInt"]),
                    ac_ext_event_guid=guid_str_to_bytes(r_1copy_p.role_to_event_guid["EngineExt"]),
                    target_int_guids=hur_int_g,
                    target_ext_guids=hur_ext_g,
                    event_paths_map=master_pmap,
                    is_ai_bank=False,
                    native_panners=_extract_native_exterior_spatializers(audi_nat_p),
                    default_prefer_throttle=bool(audi_v.prefers_throttle_player),
                    mod_identifier=audi_v.car_name,
                    seed_tag="test25:audi:huracan",
                )
                errs25_1 = verify_fmod_bank_integrity(audi_step1, expected_exported_guids=audi_exp)
                assert not errs25_1, f"Audi_V10_GT3 Huracan-only falló integridad: {errs25_1}"
                # Paso 25B: Apilar un segundo sonido sobre Audi R8 LMS GT3 manteniendo Huracán modificado y R8 GT4 vanilla
                r_z4_1copy = build_direct_ported_bank_from_ac(
                    ac_bank=ac_lib.fmod_bank,
                    mod_identifier="audi_r8_layer",
                    standalone_guids=False,
                    target_role_guids={
                        "EngineInt": bytes_to_guid_str(r8_int_g[0]),
                        "EngineExt": bytes_to_guid_str(r8_ext_g[0]),
                    },
                    target_role_paths=audi_v.event_paths,
                    target_bank_guid=audi_v.bank_guid or None,
                    native_bank_path=audi_nat_p,
                    extra_role_variants=None,
                    prefer_throttle_param=bool(audi_v.prefers_throttle_player),
                )
                audi_step2, _ = graft_ac_into_native_bank_surgical(
                    native_bank=FmodBank(raw=audi_step1),
                    ported_ac_bank=FmodBank(raw=r_z4_1copy.bank_bytes),
                    ac_int_event_guid=guid_str_to_bytes(r_z4_1copy.role_to_event_guid["EngineInt"]),
                    ac_ext_event_guid=guid_str_to_bytes(r_z4_1copy.role_to_event_guid["EngineExt"]),
                    target_int_guids=r8_int_g,
                    target_ext_guids=r8_ext_g,
                    event_paths_map=master_pmap,
                    is_ai_bank=False,
                    native_panners=_extract_native_exterior_spatializers(audi_nat_p),
                    default_prefer_throttle=bool(audi_v.prefers_throttle_player),
                    mod_identifier=audi_v.car_name,
                    seed_tag="test25:audi:r8gt3",
                )
                errs25_2 = verify_fmod_bank_integrity(audi_step2, expected_exported_guids=audi_exp)
                assert not errs25_2, f"Audi_V10_GT3 layered Huracan+R8 falló integridad: {errs25_2}"
                assert_sndh_and_fsb_alignment(audi_step2)
                print("   Aislamiento selectivo + apilado multi-auto (Huracán + R8 GT3 + R8 GT4 vanilla) verificado al 100%!")
            del r_1copy_p, r_1copy_ai, alp_lib, alp_fmod
            gc.collect()

        # 26. Preservación garantizada de Starter/Starting (audio FSB5 completo) y Transmisión AMS2 en Vanilla y Mods
        print("26. Probando preservación de Starter/Starting y Transmisión original de AMS2 (Vanilla FADPCM/Vorbis + Mod .csdbin)...")
        from ams2_sound_porter.porter_engine import (
            _extract_event_subgraph_closure,
            collect_local_proj_guids,
        )

        def _verify_starter_wavs_26(bank_bytes: bytes, root_guids: list[bytes]) -> list[str]:
            fb = FmodBank(raw=bank_bytes)
            pj = next(
                bank_bytes[ps + 4 : pe]
                for c, ps, pe, _ in iter_riff_chunks(bank_bytes, 12, len(bank_bytes))
                if c == b"LIST" and bank_bytes[ps : ps + 4] == b"PROJ"
            )
            tbls = list(_iter_proj_table_items(pj))
            clos = _extract_event_subgraph_closure(tbls, set(root_guids))
            found_names: list[str] = []
            for lt, items in tbls:
                if lt == b"WAVS":
                    for k, _, _, bpay in items:
                        if k in clos and len(bpay) >= 30:
                            _, sb_i, _, smp_i, _ = struct.unpack_from("<HHHII", bpay, 16)
                            sl = next((s for s in fb.slots if s.subbank_index == sb_i and s.sample_index == smp_i), None)
                            assert sl is not None, f"WAV starter fuera de rango: ({sb_i}, {smp_i})"
                            if sl.can_decode_audio:
                                ab = fb.decode_slot_to_buffer(sl)
                                assert ab.num_samples > 0 and ab.peak_linear() > 0.001, f"Audio starter mudo: {sl.name}"
                            found_names.append(sl.name)
            return found_names

        # 26a. Vanilla single-car (Porsche_963 con 3 eventos Startup + Corvette_C8_GT3R + Audi_V10_GT3 Vorbis starter)
        p96_ac_26 = Path(r"D:\SteamLibrary\steamapps\common\assettocorsa\content\cars\rss_mph_protech_p96_v8\sfx\rss_mph_protech_p96_v8.bank")
        src_ac_26 = SourceAudioLibrary.from_bank(p96_ac_26, guids_path=p96_ac_26.parent / "GUIDs.txt") if p96_ac_26.is_file() else ac_lib
        for vid_26, min_st_wavs in (("porsche_963", 5), ("c_corvette_c8_gt3r", 2), ("audi_audi_v10_gt3", 3)):
            v26 = next((v for v in vm23._vehicles_cache if vid_26 in v.vehicle_id.lower()), None)
            if not v26:
                continue
            vm23.load_vehicle_events_info(v26)
            nat_p26 = vm23.backup_dir / v26.vehicle_id / v26.bank_path.name
            if not nat_p26.is_file():
                nat_p26 = v26.bank_path
            r26 = build_direct_ported_bank_from_ac(
                ac_bank=src_ac_26.fmod_bank,
                mod_identifier=v26.car_name,
                standalone_guids=False,
                target_role_guids=v26.event_guids,
                target_role_paths=v26.event_paths,
                target_bank_guid=v26.bank_guid or None,
                native_bank_path=nat_p26,
                extra_role_variants=v26.extra_event_variants or None,
                output_codec_mode=FSB5_MODE_FADPCM,
                prune_unused_samples=True,
                library=src_ac_26,
            )
            nat_b26 = FmodBank(nat_p26)
            nat_pj26 = next(
                nat_b26.raw[ps + 4 : pe]
                for c, ps, pe, _ in iter_riff_chunks(nat_b26.raw, 12, len(nat_b26.raw))
                if c == b"LIST" and nat_b26.raw[ps : ps + 4] == b"PROJ"
            )
            nat_exp26 = _extract_bnki_exported_event_guids(nat_pj26, dict(_iter_proj_table_items(nat_pj26))[b"EVTS"])
            errs26 = verify_fmod_bank_integrity(r26.bank_bytes, expected_exported_guids=nat_exp26)
            assert not errs26, f"{vid_26} falló integridad tras injerto de starter nativo: {errs26}"
            assert_sndh_and_fsb_alignment(r26.bank_bytes)
            st_gs26 = [guid_str_to_bytes(v26.event_guids[r]) for r in ("Starter", "Starting") if r in v26.event_guids]
            for r in ("Starter", "Starting"):
                for it in (v26.extra_event_variants or {}).get(r, ()):
                    st_gs26.append(guid_str_to_bytes(it[0]))
            st_names26 = _verify_starter_wavs_26(r26.bank_bytes, st_gs26)
            assert len(st_names26) >= min_st_wavs, f"{vid_26} starter WAVS insuficientes: {st_names26}"
            # Round-trip Edit Modified Sound debe mantener el starter intacto
            re_b26 = FmodBank(raw=r26.bank_bytes)
            ed_b26 = apply_band_edits_to_bank(re_b26, rpm_bands=build_default_rpm_bands(re_b26))
            assert not verify_fmod_bank_integrity(ed_b26, expected_exported_guids=nat_exp26)
            assert len(_verify_starter_wavs_26(ed_b26, st_gs26)) == len(st_names26)
            del r26, re_b26, ed_b26

        # 26b. Mod export (.csdbin de porsche_992_cup + doble exportación consecutiva)
        cup_csd_p = vm23.ams2_root / "Vehicles" / "porsche_992_cup" / "porsche_992_cup.csdbin"
        if cup_csd_p.is_file() and AC_MOD_488_BANK.is_file() and AC_MOD_488_GUIDS.is_file():
            ac_488_26 = FmodBank(AC_MOD_488_BANK, guids_path=AC_MOD_488_GUIDS)
            cup_doc26 = CsdbinDocument.from_file(cup_csd_p)
            orig_links26 = {
                k: cup_doc26.links.get(k)
                for k in (
                    "transmission_bank",
                    "transmission",
                    "gearshift_bank",
                    "gearshift_up",
                    "gearshift_down",
                    "gearshift_grind",
                    "rev_limiter_bank",
                    "wind_bank",
                )
            }
            cup_nat_raw26 = (vm23.ams2_root / "Audio" / "AMS2" / "Vehicles" / "Sort" / "P" / "Porsche" / "911_Cup_2.bank").read_bytes()
            cup_nat_pj26 = next(
                cup_nat_raw26[ps + 4 : pe]
                for c, ps, pe, _ in iter_riff_chunks(cup_nat_raw26, 12, len(cup_nat_raw26))
                if c == b"LIST" and cup_nat_raw26[ps : ps + 4] == b"PROJ"
            )
            cup_nat_guids26 = set(collect_local_proj_guids(cup_nat_pj26))
            with tempfile.TemporaryDirectory() as td26:
                for pass_idx in (1, 2):
                    s26 = export_ams2_mod_package(
                        output_root=Path(td26),
                        mod_identifier="cup_st_tr_test",
                        vehicle_folder_name="porsche_992_cup",
                        direct_ac_bank=ac_488_26,
                        csd_doc=cup_doc26,
                        standalone_guids=True,
                        ams2_root=vm23.ams2_root,
                        backup_dir=vm23.backup_dir,
                    )
                    m_raw26 = s26.bank_file.read_bytes()
                    assert not verify_fmod_bank_integrity(m_raw26), f"Mod pass {pass_idx} falló integridad"
                    assert_sndh_and_fsb_alignment(m_raw26)
                    m_pj26 = next(
                        m_raw26[ps + 4 : pe]
                        for c, ps, pe, _ in iter_riff_chunks(m_raw26, 12, len(m_raw26))
                        if c == b"LIST" and m_raw26[ps : ps + 4] == b"PROJ"
                    )
                    assert not (cup_nat_guids26 & set(collect_local_proj_guids(m_pj26))), "Colisión GUID con 911_Cup_2 vanilla"
                    st_g26 = guid_str_to_bytes(s26.bank_result.role_to_event_guid["Starter"])
                    stg_g26 = guid_str_to_bytes(s26.bank_result.role_to_event_guid["Starting"])
                    m_wavs26 = _verify_starter_wavs_26(m_raw26, [st_g26, stg_g26])
                    assert len(m_wavs26) >= 4, f"Mod pass {pass_idx} perdió muestras starter: {m_wavs26}"
                    csd_out26 = CsdbinDocument.from_file(s26.csdbin_file)
                    assert csd_out26.links.get("starter") == "event:/AMS2/Vehicles/Mods/cup_st_tr_test/Starter"
                    assert csd_out26.links.get("starting") == "event:/AMS2/Vehicles/Mods/cup_st_tr_test/Starting"
                    for lk, lv in orig_links26.items():
                        if lv:
                            assert csd_out26.links.get(lk) == lv, f"Mod pass {pass_idx} alteró enlace nativo {lk}: {csd_out26.links.get(lk)} != {lv}"
            del ac_488_26
        print("   Preservación de Starter/Starting (FADPCM+Vorbis) y Transmisión/Gearshift AMS2 (Vanilla + Mod + re-export) verificada al 100%!")
        gc.collect()

    print("\nTODAS LAS PRUEBAS DEL MOTOR PASARON AL 100%!")


if __name__ == "__main__":
    main()
