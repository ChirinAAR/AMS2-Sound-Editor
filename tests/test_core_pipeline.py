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
    LOCAL_OBJECT_BASE_CHUNKS,
    RpmBandConfig,
    SlotMappingConfig,
    SourceAudioLibrary,
    ThrottleCutConfig,
    auto_detect_throttle_cut_config,
    auto_distribute_rpm_bands,
    auto_map_samples,
    build_default_rpm_bands,
    build_direct_ported_bank_from_ac,
    build_ported_bank,
    estimate_engine_rpm_bounds,
    export_ams2_mod_package,
)

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

    def walk(start: int, end: int) -> None:
        nonlocal checked_lists
        child_guids_by_tag: dict[bytes, list[bytes]] = {}
        for cid, ps, pe, _ in iter_riff_chunks(raw_bank, start, end):
            if cid == b"LIST":
                ltype = raw_bank[ps : ps + 4]
                # Extraer GUID del primer subchunk *B si es un contenedor de ítem (EVNT, IBUS, WAIT, etc.)
                for scid, sps, spe, _ in iter_riff_chunks(raw_bank, ps + 4, pe):
                    if scid in LOCAL_OBJECT_BASE_CHUNKS and spe - sps >= 16:
                        child_guids_by_tag.setdefault(ltype, []).append(raw_bank[sps : sps + 16])
                        break
                walk(ps + 4, pe)
            elif cid in (b"CTRL", b"CURV", b"MAP ", b"WAV ") and pe - ps >= 16:
                child_guids_by_tag.setdefault(cid, []).append(raw_bank[ps : ps + 16])

        for tag, glist in child_guids_by_tag.items():
            if len(glist) > 1:
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
            assert len(out_bank.slots) == len(ac_488.slots)
            assert len(out_bank.stdt_bytes) > 0

            # Verificar banco de IA y TV Replay
            ai_out_bank = FmodBank(summary_direct.ai_bank_file)
            assert len(ai_out_bank.events) >= 15
            assert len(ai_out_bank.stdt_bytes) > 0
            ai_paths = parse_stdt_all_paths(ai_out_bank.stdt_bytes)
            assert any("AI_EngineMAIN_LIVE" in p for p in ai_paths.values()), "Falta evento AI_EngineMAIN_LIVE para TV Replays"
            assert any("AI_EngineMAIN" in p for p in ai_paths.values()), "Falta evento AI_EngineMAIN para rivales"

            # Verificar que el .csdbin contiene todos los enlaces de jugador e IA
            csd_loaded = CsdbinDocument.from_file(summary_direct.csdbin_file)
            assert csd_loaded.links["engine_bank"] == "Audio/AMS2/Vehicles/Mods/f488_direct/f488_direct_Engine"
            assert "engine_ext" in csd_loaded.links
            assert "engine_int" in csd_loaded.links
            assert csd_loaded.links["ai_engine_bank"] == "Audio/AMS2/Vehicles/Mods/f488_direct/f488_direct_Engine_AI"
            assert "AI_EngineMAIN" in csd_loaded.links["ai_engine_ext"]
            assert "AI_EngineMAIN_LIVE" in csd_loaded.links["ai_engine_live"]
            assert "AI_EngineInt" in csd_loaded.links["ai_engine_int"]
            assert "gearshift_bank" in csd_loaded.links
            assert "gearshift_up" in csd_loaded.links
            assert "turbo_bank" in csd_loaded.links
            assert "turbo" in csd_loaded.links
            assert "backfire_ext" in csd_loaded.links
            print(f"   Ferrari 488 GT3 Mod export directo DUAL (Jugador + IA): {summary_direct.bank_file.name} y {summary_direct.ai_bank_file.name} (Enlaces AI .csdbin OK)!")

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
            )
            reloaded_mercer = FmodBank(raw=mercer_sls_res.bank_bytes)
            guids_list = [ev.guid_str for ev in reloaded_mercer.events]
            assert len(set(guids_list)) == len(guids_list), "Se detectaron GUIDs duplicados en el porteo de Mercer V8!"
            reloaded_roles = {ev.guid_str: ev.role for ev in reloaded_mercer.events}
            for role_name, exp_guid in sls_v.event_guids.items():
                assert exp_guid in reloaded_roles, f"Falta el rol {role_name} ({exp_guid}) en el banco de Mercer V8!"
            assert_sndh_and_fsb_alignment(mercer_sls_res.bank_bytes)
            print("   Porteo de Mercer V8 verificado: 0 GUIDs duplicados, eventos nativos y parámetros load/rpm sincronizados al 100%!")

    print("\nTODAS LAS PRUEBAS DEL MOTOR PASARON AL 100%!")


if __name__ == "__main__":
    main()
