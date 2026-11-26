"""Prueba de humo automatizada de la interfaz gráfica SoundPorterApp."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ams2_sound_porter.gui_app import SoundPorterApp

REF_DIR = Path(
    r"C:\Users\Usuario\.gemini\antigravity\brain\f847d4e0-8cea-412c-a1b6-15d13cc65709\scratch\sonidos"
)
MOD_488_DIR = Path(
    r"C:\Users\Usuario\.gemini\antigravity\brain\f847d4e0-8cea-412c-a1b6-15d13cc65709\scratch\488gt3_sfx"
)


def main() -> None:
    app = SoundPorterApp()
    app.withdraw()
    try:
        app._load_source_library_from_bank(REF_DIR / "Sonido Asseto Corsa.bank")
        assert len(app.src_tree.get_children()) == 46

        app._load_donor_bank_file(REF_DIR / "Sonido Automobilista 2.bank")
        assert len(app.donor_tree.get_children()) == 27
        # Verificar que las variaciones de pops rotaron entre distintas muestras
        pop_srcs = [
            app.mappings[i].source_index
            for i in (7, 14, 18)
        ]
        print("Variaciones de petardeos asignadas (slots 7, 14, 18):", pop_srcs)
        assert len(set(pop_srcs)) >= 2

        app._load_csdbin_path(REF_DIR / "Archivo Sonido Vehiculo Automobilista 2.csdbin")
        app.mod_id_var.set("bmw_z4_gt3_ac")
        app.cmd_autofill_csdbin_standalone()
        assert app.csd_link_vars["engine_bank"].get() == "Audio/AMS2/Vehicles/Mods/bmw_z4_gt3_ac/bmw_z4_gt3_ac_Engine"
        assert app.csd_link_vars["engine_ext"].get() == "event:/AMS2/Vehicles/Mods/bmw_z4_gt3_ac/EngineExt"

        # Probar reproducción asíncrona real (winsound) en fila AC, donante AMS2 y reemplazo mapeado!
        errors_caught = []
        import tkinter.messagebox as mb
        orig_err = mb.showerror
        mb.showerror = lambda title, msg, **kw: errors_caught.append((title, msg))
        try:
            app.src_tree.selection_set("1")
            app.cmd_play_selected_source()
            assert not errors_caught, f"Error en preview AC: {errors_caught}"

            # Probar preview de slot donante decodificable (ej. starter FADPCM en slot 0)
            app.donor_tree.selection_set("0")
            app._on_donor_slot_selected()
            app.cmd_preview_original_donor_slot()
            assert not errors_caught, f"Error en preview donante AMS2: {errors_caught}"

            # Probar preview de comparación / reemplazo mapeado (ej. slot 19 o slot 1)
            app.donor_tree.selection_set("1")
            app._on_donor_slot_selected()
            app.cmd_preview_mapped_slot()
            assert not errors_caught, f"Error en preview reemplazo mapeado: {errors_caught}"

            app.stop_audio_preview()
        finally:
            mb.showerror = orig_err

        # Probar flujo de sonido MOD AC con GUIDs.txt separado y auto-actualización del mapeo
        mod_bank = MOD_488_DIR / "ks_ferrari_488_gt3.bank"
        mod_guids = MOD_488_DIR / "GUIDs.txt"
        if mod_bank.is_file() and mod_guids.is_file():
            with tempfile.TemporaryDirectory() as tmp_d:
                isolated_bank = Path(tmp_d) / "ks_ferrari_488_gt3.bank"
                isolated_bank.write_bytes(mod_bank.read_bytes())
                app._load_source_library_from_bank(isolated_bank)
                assert "sin GUIDs.txt" in app.src_info_var.get()
                assert len(app.src_tree.get_children()) == 174

                # Ahora vincular manualmente GUIDs.txt (equivalente al botón '📄 Cargar GUIDs.txt...')
                app._load_source_library_from_bank(isolated_bank, guids_path=mod_guids)
                assert "GUIDs.txt" in app.src_info_var.get()
                assert app.source_lib is not None and app.source_lib.matched_guids_count >= 20
                # Verificar que el slot interior 25 (m3gtr_int_idle) se auto-mapeó a '488_Idle'
                idle_src_idx = app.mappings[25].source_index
                assert idle_src_idx is not None
                assert app.source_lib.items[idle_src_idx].name == "488_Idle"

                # Verificar Pestaña 3: Bandas RPM, Throttle Cut y Simulador
                assert len(app.rpm_bands) > 0
                assert app.throttle_cut_cfg.enabled is True
                assert app.tcut_int_combo_var.get() != "[Ninguno]"
                assert app.tcut_ext_combo_var.get() != "[Ninguno]"
                print("   Throttle Cut GUI auto-configurado:", app.tcut_int_combo_var.get(), app.tcut_ext_combo_var.get())

                # Probar auto-distribución de escalera RPM sin solapamientos
                app.cmd_auto_distribute_rpm()
                assert "0 solapamientos" in app.rpm_overlap_status_var.get()

                # Probar edición de una banda individual desde la GUI
                vis_items = app.rpm_tree.get_children()
                assert len(vis_items) > 0
                first_band_id = vis_items[0]
                app.rpm_tree.selection_set(first_band_id)
                app._on_rpm_band_selected()
                orig_rmin = app.band_rmin_var.get()
                app.band_rmin_var.set("750")
                app.band_fi_var.set("850")
                app.band_fo_var.set("1800")
                app.band_rmax_var.set("2200")
                app.cmd_apply_band_edit()
                assert app.rpm_bands[int(first_band_id)].rpm_min == 750.0

                # Probar preview de Throttle Cut INT y EXT sin errores
                orig_mb_err = mb.showerror
                tcut_errs = []
                mb.showerror = lambda title, msg, **kw: tcut_errs.append((title, msg))
                try:
                    app.cmd_preview_throttle_cut("int")
                    app.cmd_preview_throttle_cut("ext")
                    assert not tcut_errs, f"Error en preview Throttle Cut: {tcut_errs}"
                    app.stop_audio_preview()
                finally:
                    mb.showerror = orig_mb_err

                # --- Novedades v1.4.0 en Pestaña 3 ---
                # 1. Probar que el diálogo para añadir una nueva banda abre sin errores
                app.cmd_add_rpm_band()
                print("   GUI Dialog Añadir Banda probado y verificado sin errores.")

                # 2. Probar amplificación masiva de bandas (Batch Gain)
                app.batch_gain_var.set("+2.0")
                prev_gain = app.mappings[0].gain_db
                app.cmd_batch_gain_all_bands()
                assert "Ganancia ajustada" in app.status_var.get()
                print("   GUI Batch Gain probado con éxito:", app.status_var.get())

                # 2b. Probar "Modificar todas las RPM" (+450 y -200 RPM en min, fade-in, fade-out y max)
                vis_batch = [int(iid) for iid in app.rpm_tree.get_children()]
                app.rpm_tree.selection_set(str(vis_batch[0]))
                app._on_rpm_band_selected()
                before_ranges = {
                    i: (app.rpm_bands[i].rpm_min, app.rpm_bands[i].fade_in_end, app.rpm_bands[i].fade_out_start, app.rpm_bands[i].rpm_max)
                    for i in vis_batch
                }
                app.batch_rpm_var.set("+450")
                app.cmd_batch_shift_all_rpm()
                assert "Rangos RPM ajustados (+450 RPM)" in app.status_var.get()
                for i, (r0, fi0, fo0, rm0) in before_ranges.items():
                    b_cur = app.rpm_bands[i]
                    assert (b_cur.rpm_min, b_cur.fade_in_end, b_cur.fade_out_start, b_cur.rpm_max) == (
                        r0 + 450.0, fi0 + 450.0, fo0 + 450.0, rm0 + 450.0
                    )
                assert app.band_rmin_var.get() == f"{app.rpm_bands[vis_batch[0]].rpm_min:.0f}"
                app.batch_rpm_var.set("-200")
                app.cmd_batch_shift_all_rpm()
                for i, (r0, _fi0, _fo0, rm0) in before_ranges.items():
                    b_cur = app.rpm_bands[i]
                    assert b_cur.rpm_min >= 0.0 and b_cur.rpm_min <= b_cur.fade_in_end <= b_cur.fade_out_start <= b_cur.rpm_max
                    assert b_cur.rpm_max == rm0 + 250.0
                print("   GUI Modificar todas las RPM (+450 / -200) probado con éxito:", app.status_var.get())

                # 2c. Probar "Modificar todos los RPM Raíz" (+300 y -150 RPM sobre pitch raíz)
                before_roots = {}
                for i in vis_batch:
                    g0 = app.rpm_bands[i].donor_slot_indices[0]
                    before_roots[i] = (
                        g0,
                        (app.mappings[g0].override_root_rpm or None) or app.active_engine_bank.slots[g0].autopitch_ref_rpm,
                    )
                app.batch_root_rpm_var.set("+300")
                app.cmd_batch_shift_all_root_rpm()
                assert "RPM Raíz (Pitch) ajustado (+300 RPM)" in app.status_var.get()
                seen_batch_slots: set[int] = set()
                for i, (g0, rr0) in before_roots.items():
                    if g0 in seen_batch_slots:
                        continue
                    seen_batch_slots.add(g0)
                    if rr0 is not None and rr0 > 0:
                        assert app.mappings[g0].override_root_rpm == round(rr0 + 300.0, 1)
                    else:
                        assert app.mappings[g0].override_root_rpm is None
                app.batch_root_rpm_var.set("-150")
                app.cmd_batch_shift_all_root_rpm()
                seen_batch_slots.clear()
                for i, (g0, rr0) in before_roots.items():
                    if g0 in seen_batch_slots:
                        continue
                    seen_batch_slots.add(g0)
                    if rr0 is not None and rr0 > 0:
                        assert app.mappings[g0].override_root_rpm == max(200.0, round(rr0 + 150.0, 1))
                print("   GUI Modificar todos los RPM Raíz (+300 / -150) probado con éxito:", app.status_var.get())

                # 3. Probar Auto-Ajustar Escalera RPM sobre la banda seleccionada (selectivo)
                app.rpm_tree.selection_set(first_band_id)
                app._on_rpm_band_selected()
                app.cmd_auto_distribute_rpm()
                assert f"Banda #{int(first_band_id):02d}" in app.status_var.get()
                print("   GUI Auto-Ajuste Selectivo probado:", app.status_var.get())

                # 4. Probar Restaurar Originales en la sección elegida (Interior o Exterior)
                # Debe modificar y acomodar TODAS las bandas de la sección y añadir las bandas necesarias
                app.rpm_view_persp_var.set("ext")
                app.cmd_reset_rpm_bands()
                assert "recreada fielmente desde AC" in app.status_var.get()
                assert "sincronizadas" in app.status_var.get()
                print("   GUI Recreación Fiel Sección Exterior probada:", app.status_var.get())

                app.rpm_view_persp_var.set("int")
                app.cmd_reset_rpm_bands()
                assert "recreada fielmente desde AC" in app.status_var.get()
                assert "sincronizadas" in app.status_var.get()
                print("   GUI Recreación Fiel Sección Interior probada:", app.status_var.get())

                # 5. Probar Eliminar Banda seleccionada y Reutilización Segura
                vis_after = app.rpm_tree.get_children()
                if vis_after:
                    target_del_id = vis_after[0]
                    del_idx = int(target_del_id)
                    app.rpm_tree.selection_set(target_del_id)
                    app._on_rpm_band_selected()
                    orig_askyesno = mb.askyesno
                    mb.askyesno = lambda *args, **kw: True
                    try:
                        app.cmd_delete_rpm_band()
                        assert "silenciada" in app.status_var.get()
                        print("   GUI Eliminar Banda probada:", app.status_var.get())

                        # Probar Reactivación de Banda
                        app._selected_band_idx = del_idx
                        app.cmd_reactivate_selected_band()
                        assert "reactivada y desmuteada" in app.status_var.get()
                        print("   GUI Reactivar Banda probada:", app.status_var.get())

                        # Probar Sobreescribir Muestra de la Banda Seleccionada
                        app._selected_band_idx = del_idx
                        avail_vals = app.band_source_combo["values"]
                        if len(avail_vals) > 1:
                            app.band_source_var.set(avail_vals[1])
                            app.cmd_overwrite_selected_band_sample()
                            assert "sobreescrita exitosamente" in app.status_var.get()
                            print("   GUI Sobreescribir Muestra probada:", app.status_var.get())
                    finally:
                        mb.askyesno = orig_askyesno

        # 6. Probar Sistema GUI de Vehículos Vanilla de AMS2 (Pestaña 4 y Pestaña 5)
        print("6. Probando Sistema GUI de Vehículos Vanilla de AMS2...")
        assert app.target_mode_var.get() == "vanilla"
        assert len(app.vanilla_mgr.scan_catalog()) >= 150
        assert len(app.vanilla_combo["values"]) >= 150

        # Probar filtro de búsqueda en tiempo real
        app.vanilla_search_var.set("Super V8")
        app._filter_vanilla_vehicles()
        assert any("Super V8" in val for val in app.vanilla_combo["values"])
        assert app.vanilla_selected_vehicle is not None
        assert "super_v8" in app.vanilla_selected_vehicle.vehicle_id

        # Verificar actualización de detalles técnicos
        assert "Super V8" in app.vanilla_det_name_var.get()
        assert "SuperV8_Set1.bank" in app.vanilla_det_path_var.get()
        assert "EngineExt" in app.vanilla_det_events_var.get()

        # Probar alternancia de modos (Vanilla <-> Mod .csdbin)
        app.target_mode_var.set("mod")
        app._on_target_mode_changed()
        assert "COMPILAR Y EXPORTAR" in app.btn_export.cget("text")

        app.target_mode_var.set("vanilla")
        app._on_target_mode_changed()
        assert "INYECTAR" in app.btn_export.cget("text") and "SUPER V8" in app.btn_export.cget("text")
        print("   GUI Selector Vanilla, Búsqueda, Ficha Técnica y Alternancia de Modo verificados con éxito.")

        # Probar modo directo 1:1 en vehículo vanilla (cmd_set_vanilla_direct_mode y build_direct_ported_bank_from_ac)
        app.cmd_set_vanilla_direct_mode()
        assert app.port_mode_var.get() == "direct"
        assert app.target_mode_var.get() == "vanilla"
        assert "DIRECTO 1:1" in app.btn_export["text"]
        print("   GUI Porteo Directo 1:1 Vanilla (cmd_set_vanilla_direct_mode) verificado con éxito.")

        orig_install = app.vanilla_mgr.install_sound
        direct_installed = []
        app.vanilla_mgr.install_sound = lambda vehicle, bank_bytes, ai_bank_bytes=None, **kw: (direct_installed.append((len(bank_bytes), len(ai_bank_bytes) if ai_bank_bytes else 0)) or True, "OK mock direct")
        orig_info = mb.showinfo
        orig_yesno = mb.askyesno
        from ams2_sound_porter import gui_app as ga
        orig_ga_info = getattr(ga, "messagebox", None) and ga.messagebox.showinfo
        orig_ga_yesno = getattr(ga, "messagebox", None) and ga.messagebox.askyesno
        mb.showinfo = lambda *args, **kw: None
        mb.askyesno = lambda *args, **kw: True
        if hasattr(ga, "messagebox"):
            ga.messagebox.showinfo = lambda *args, **kw: None
            ga.messagebox.askyesno = lambda *args, **kw: True
        try:
            import time
            app._run_inject_vanilla()
            timeout = time.time() + 60
            while not direct_installed and time.time() < timeout:
                app.update()
                time.sleep(0.05)
            assert len(direct_installed) > 0, f"No se llamó a install_sound en inyección directa vanilla. Error: {app.status_var.get()}"
            assert direct_installed[0][0] > 10000000, "El banco directo generado tiene tamaño inválido"
            print("   Inyección vanilla en Modo Directo 1:1 DUAL (build_direct_ported_bank_from_ac) verificada sin excepciones:", direct_installed[0], "bytes")
        finally:
            app.vanilla_mgr.install_sound = orig_install
            mb.showinfo = orig_info
            mb.askyesno = orig_yesno
            if hasattr(ga, "messagebox"):
                if orig_ga_info: ga.messagebox.showinfo = orig_ga_info
                if orig_ga_yesno: ga.messagebox.askyesno = orig_ga_yesno

        # Probar inyección en vehículo vanilla en modo donante (asegurando que build_ported_bank esté definido)
        installed_called = []
        app.vanilla_mgr.install_sound = lambda vehicle, bank_bytes, ai_bank_bytes=None, **kw: (installed_called.append((len(bank_bytes), len(ai_bank_bytes) if ai_bank_bytes else 0)) or True, "OK mock")
        orig_info = mb.showinfo
        orig_yesno = mb.askyesno
        mb.showinfo = lambda *args, **kw: None
        mb.askyesno = lambda *args, **kw: True
        if hasattr(ga, "messagebox"):
            ga.messagebox.showinfo = lambda *args, **kw: None
            ga.messagebox.askyesno = lambda *args, **kw: True
        try:
            app.port_mode_var.set("donor")
            import time
            app._run_inject_vanilla()
            timeout = time.time() + 60
            while not installed_called and time.time() < timeout:
                app.update()
                time.sleep(0.05)
            assert len(installed_called) > 0, f"No se llamó a install_sound en inyección vanilla. Error: {app.status_var.get()}"
            assert installed_called[0][0] > 100000, "El banco generado tiene tamaño inválido"
            print("   Inyección vanilla en modo donante (build_ported_bank) verificada sin excepciones:", installed_called[0], "bytes")
        finally:
            app.vanilla_mgr.install_sound = orig_install
            mb.showinfo = orig_info
            mb.askyesno = orig_yesno
            if hasattr(ga, "messagebox"):
                if orig_ga_info: ga.messagebox.showinfo = orig_ga_info
                if orig_ga_yesno: ga.messagebox.askyesno = orig_ga_yesno

        # 7. Verificar las 4 vistas nuevas (Menú, Custom Car sound, Vanilla car sounds, Test/edit Sounds) y opciones bloqueadas
        assert app.title() == "AMS 2 Sound Importer"
        assert app.mute_all_pops_var.get() is True
        assert app.invert_cone_var.get() is True
        assert app.standalone_var.get() is True
        assert app.sync_pitch_var.get() is True
        assert app.auto_link_csd_var.get() is True
        assert app.open_folder_after_var.get() is True
        assert app.vanilla_auto_backup_var.get() is True
        assert app.vanilla_sync_ai_var.get() is True
        assert app.vanilla_mute_pops_var.get() is True
        assert app.codec_choice_var.get() == "PCM16"
        assert not app.notebook.winfo_ismapped()

        app.show_view("menu")
        assert app._current_view == "menu"
        app.show_view("custom")
        assert app._current_view == "custom"
        assert app.target_mode_var.get() == "mod"
        assert app.port_mode_var.get() == "direct"
        assert app.ui_bank_check_var.get() == "✔"
        assert app.ui_guids_check_var.get() == "✔"
        assert app.ui_csdbin_check_var.get() == "✔"

        app.cmd_open_test_edit("custom")
        assert app._current_view == "test_edit"
        app.cmd_finish_test_edit()
        assert app._current_view == "custom"

        app.cmd_open_csdbin_editor()
        assert app._current_view == "csdbin"
        orig_vol_ext = app.csd_vol_ext_var.get()
        app.csd_vol_ext_var.set("1.25")
        app.cmd_finish_csdbin_edit()
        assert app._current_view == "custom"
        assert app.csd_vol_ext_var.get() == "1.25"
        app.csd_vol_ext_var.set(orig_vol_ext)

        app.show_view("vanilla")
        assert app._current_view == "vanilla"
        assert app.target_mode_var.get() == "vanilla"
        assert app.port_mode_var.get() == "direct"
        app.cmd_open_test_edit("vanilla")
        assert app._current_view == "test_edit"
        assert app.btn_test_edit_finish.winfo_manager() == "pack"
        assert app.btn_test_edit_confirm.winfo_manager() == ""
        app.cmd_finish_test_edit()
        assert app._current_view == "vanilla"
        print("   Navegación de las vistas nuevas y opciones de exportación bloqueadas verificadas!")

        # 8. Verificar vista 'Edit modified sounds' + edición de bandas + Confirm (en sandbox temporal sin tocar Steam)
        print("8. Probando flujo 'Edit modified sounds' -> selector -> Test/Edit Audio -> Confirm (sandbox)...")
        from ams2_sound_porter.fmod_bank import FmodBank
        from ams2_sound_porter.porter_engine import build_direct_ported_bank_from_ac
        from ams2_sound_porter.vanilla_manager import VanillaVehicleManager

        real_mgr = app.vanilla_mgr
        sample_veh = next(
            (v for v in real_mgr.scan_catalog() if v.bank_path.is_file() and v.ai_bank_path and v.ai_bank_path.is_file()),
            None,
        )
        ac_ref_bank = REF_DIR / "Sonido Asseto Corsa.bank"
        if sample_veh is not None and ac_ref_bank.is_file():
            orig_p_bytes = sample_veh.bank_path.read_bytes()
            orig_ai_bytes = sample_veh.ai_bank_path.read_bytes()
            with tempfile.TemporaryDirectory() as sbx_dir:
                sbx_root = Path(sbx_dir) / "AMS2"
                sbx_backup = Path(sbx_dir) / "backups"
                rel_p = sample_veh.bank_path.relative_to(real_mgr.ams2_root)
                rel_ai = sample_veh.ai_bank_path.relative_to(real_mgr.ams2_root)
                (sbx_root / rel_p).parent.mkdir(parents=True, exist_ok=True)
                (sbx_root / rel_p).write_bytes(orig_p_bytes)
                (sbx_root / rel_ai).write_bytes(orig_ai_bytes)

                sbx_mgr = VanillaVehicleManager(ams2_root=sbx_root, backup_dir=sbx_backup)
                sbx_vehs = sbx_mgr.scan_catalog(force=True)
                assert len(sbx_vehs) == 1
                sv = sbx_vehs[0]
                sbx_mgr.load_vehicle_events_info(sv)

                ac_b_sbx = FmodBank(ac_ref_bank)
                init_p = build_direct_ported_bank_from_ac(
                    ac_bank=ac_b_sbx,
                    mod_identifier=sv.car_name,
                    standalone_guids=False,
                    target_role_guids=sv.event_guids,
                    target_role_paths=sv.event_paths,
                    target_bank_guid=sv.bank_guid or None,
                )
                init_ai = build_direct_ported_bank_from_ac(
                    ac_bank=ac_b_sbx,
                    mod_identifier=f"{sv.car_name}_AI",
                    standalone_guids=False,
                    target_role_guids=sv.ai_event_guids,
                    target_role_paths=sv.ai_event_paths,
                    target_bank_guid=sv.ai_bank_guid or None,
                    is_ai_bank=True,
                    mute_secondary_pops=True,
                    invert_spatial_cone=False,
                )
                ok_inst, msg_inst = sbx_mgr.install_sound(sv, init_p.bank_bytes, ai_bank_bytes=init_ai.bank_bytes)
                assert ok_inst, msg_inst
                backup_file = sbx_backup / sv.vehicle_id / sv.bank_path.name
                assert backup_file.is_file() and backup_file.read_bytes() == orig_p_bytes

                orig_ga_info2 = ga.messagebox.showinfo if hasattr(ga, "messagebox") else None
                orig_ga_err2 = ga.messagebox.showerror if hasattr(ga, "messagebox") else None
                orig_mb_info2 = mb.showinfo
                orig_mb_err2 = mb.showerror
                info_msgs: list[str] = []
                err_msgs: list[str] = []
                mb.showinfo = lambda *a, **kw: info_msgs.append(str(a))
                mb.showerror = lambda *a, **kw: err_msgs.append(str(a))
                if hasattr(ga, "messagebox"):
                    ga.messagebox.showinfo = lambda *a, **kw: info_msgs.append(str(a))
                    ga.messagebox.showerror = lambda *a, **kw: err_msgs.append(str(a))
                try:
                    app.vanilla_mgr = sbx_mgr
                    app.show_view("modified")
                    app.update()
                    assert app._current_view == "modified"
                    tree_ids = app.modified_tree.get_children()
                    assert sv.vehicle_id in tree_ids, f"Vehículo modificado no listado en modified_tree: {tree_ids}"
                    app.modified_tree.selection_set(sv.vehicle_id)
                    app.cmd_open_modified_vehicle_editor()
                    app.update()

                    assert app._current_view == "test_edit"
                    assert app._editing_modified_vanilla is True
                    assert app.btn_test_edit_confirm.winfo_manager() == "pack"
                    assert app.btn_test_edit_finish.winfo_manager() == ""
                    assert len(app.rpm_bands) > 0

                    # Editar primera banda visible y pulsar Confirm
                    vis_idx = next(i for i, b in enumerate(app.rpm_bands) if not b.is_muted_bus and b.donor_slot_indices)
                    app._selected_band_idx = vis_idx
                    edited_b = app.rpm_bands[vis_idx]
                    app.band_rmin_var.set("2220")
                    app.band_fi_var.set("2520")
                    app.band_fo_var.set("5120")
                    app.band_rmax_var.set("5620")
                    app.band_gain_var.set("2.0")
                    app.cmd_apply_band_edit()

                    info_msgs.clear()
                    err_msgs.clear()
                    app.cmd_confirm_modified_edit()
                    timeout2 = time.time() + 60
                    while not any("Modified Sound Updated!" in m for m in info_msgs) and not err_msgs and time.time() < timeout2:
                        app.update()
                        time.sleep(0.05)

                    assert not err_msgs, f"Confirm reportó error: {err_msgs}"
                    assert any("Modified Sound Updated!" in m for m in info_msgs), (
                        f"Confirm no completó la re-inyección. info={info_msgs} status={app.status_var.get()}"
                    )
                    assert app._current_view == "modified", f"Vista esperada 'modified', actual: {app._current_view}"
                    assert backup_file.read_bytes() == orig_p_bytes, "El respaldo original Steam fue alterado!"

                    re_sbx_p = FmodBank(sv.bank_path)
                    chk_b = next(
                        (
                            b
                            for b in re_sbx_p.rpm_bands
                            if b.pml_guid == edited_b.pml_guid
                            and b.event_guid == edited_b.event_guid
                            and b.box_index == edited_b.box_index
                        ),
                        None,
                    )
                    assert chk_b is not None and abs(chk_b.rpm_min - 2220.0) < 1.0 and abs(chk_b.rpm_max - 5620.0) < 1.0, (
                        f"La banda editada no se reflejó en el .bank re-inyectado: {chk_b}"
                    )
                    assert sv.ai_bank_path.is_file() and sv.ai_bank_path.stat().st_size > 100000
                    print("   Flujo 'Edit modified sounds' -> edición de bandas -> 'Confirm' verificado al 100% en sandbox!")
                finally:
                    app.vanilla_mgr = real_mgr
                    mb.showinfo = orig_mb_info2
                    mb.showerror = orig_mb_err2
                    if hasattr(ga, "messagebox") and orig_ga_info2:
                        ga.messagebox.showinfo = orig_ga_info2
                    if hasattr(ga, "messagebox") and orig_ga_err2:
                        ga.messagebox.showerror = orig_ga_err2

        # 9. Verificar selector de alcance en Vanilla ('Full' por defecto, 'Interior only', 'Exterior only') en sandbox
        print("9. Probando selector de alcance Vanilla ('Full' / 'Interior only' / 'Exterior only') en sandbox...")
        assert app.vanilla_audio_scope_var.get() == "full", (
            f"El valor por defecto debe ser 'full', actual: {app.vanilla_audio_scope_var.get()}"
        )
        m3_veh = next(
            (v for v in real_mgr.scan_catalog() if "bmw_m3_gtr" in v.vehicle_id.lower() and v.ai_bank_path and v.ai_bank_path.is_file()),
            None,
        )
        if m3_veh is not None and ac_ref_bank.is_file():
            orig_m3_p_path = real_mgr.backup_dir / m3_veh.vehicle_id / m3_veh.bank_path.name
            if not orig_m3_p_path.is_file():
                orig_m3_p_path = m3_veh.bank_path
            orig_m3_ai_path = real_mgr.backup_dir / m3_veh.vehicle_id / m3_veh.ai_bank_path.name
            if not orig_m3_ai_path.is_file():
                orig_m3_ai_path = m3_veh.ai_bank_path
            orig_m3_p = orig_m3_p_path.read_bytes()
            orig_m3_ai = orig_m3_ai_path.read_bytes()

            with tempfile.TemporaryDirectory() as sbx9_dir:
                sbx9_root = Path(sbx9_dir) / "AMS2"
                sbx9_backup = Path(sbx9_dir) / "backups"
                rel_p9 = m3_veh.bank_path.relative_to(real_mgr.ams2_root)
                rel_ai9 = m3_veh.ai_bank_path.relative_to(real_mgr.ams2_root)
                (sbx9_root / rel_p9).parent.mkdir(parents=True, exist_ok=True)
                (sbx9_root / rel_p9).write_bytes(orig_m3_p)
                (sbx9_root / rel_ai9).write_bytes(orig_m3_ai)

                sbx9_mgr = VanillaVehicleManager(ams2_root=sbx9_root, backup_dir=sbx9_backup)
                sv9 = sbx9_mgr.scan_catalog(force=True)[0]
                sbx9_mgr.load_vehicle_events_info(sv9)

                orig_mb_i9, orig_mb_y9, orig_mb_e9, orig_mb_w9 = mb.showinfo, mb.askyesno, mb.showerror, mb.showwarning
                orig_ga_i9 = ga.messagebox.showinfo if hasattr(ga, "messagebox") else None
                orig_ga_y9 = ga.messagebox.askyesno if hasattr(ga, "messagebox") else None
                orig_ga_e9 = ga.messagebox.showerror if hasattr(ga, "messagebox") else None
                orig_ga_w9 = ga.messagebox.showwarning if hasattr(ga, "messagebox") else None
                msgs9: list[str] = []
                errs9: list[str] = []
                mb.showinfo = lambda *a, **kw: msgs9.append(str(a))
                mb.askyesno = lambda *a, **kw: True
                mb.showerror = lambda *a, **kw: errs9.append(str(a))
                mb.showwarning = lambda *a, **kw: errs9.append("WARN:" + str(a))
                if hasattr(ga, "messagebox"):
                    ga.messagebox.showinfo = lambda *a, **kw: msgs9.append(str(a))
                    ga.messagebox.askyesno = lambda *a, **kw: True
                    ga.messagebox.showerror = lambda *a, **kw: errs9.append(str(a))
                    ga.messagebox.showwarning = lambda *a, **kw: errs9.append("WARN:" + str(a))
                try:
                    app.vanilla_mgr = sbx9_mgr
                    app.vanilla_search_var.set("")
                    app._populate_vanilla_vehicles_list(select_id=sv9.vehicle_id)
                    app._load_source_library_from_bank(ac_ref_bank)
                    app.show_view("vanilla")
                    app.vanilla_selected_vehicle = sv9
                    app.port_mode_var.set("direct")

                    # 9a. Inyectar Exterior only
                    app.vanilla_audio_scope_var.set("exterior")
                    app._on_vanilla_scope_changed()
                    assert "Interior" in app.vanilla_scope_hint_var.get() or "interior" in app.vanilla_scope_hint_var.get()
                    msgs9.clear()
                    errs9.clear()
                    app._run_inject_vanilla()
                    t9 = time.time() + 60
                    while not any("Sound Successfully Imported" in m for m in msgs9) and not errs9 and time.time() < t9:
                        app.update()
                        time.sleep(0.05)
                    assert not errs9, f"Error en inyección Exterior-only: {errs9}"
                    assert sv9.modified_scope == "exterior"
                    fb_ext9 = FmodBank(sv9.bank_path)
                    ev_map_ext9 = {ev.guid_str.lower(): ev for ev in fb_ext9.events}
                    assert any("m3gtr_int_" in s.lower() for s in ev_map_ext9[sv9.event_guids["EngineInt"].lower()].linked_samples)
                    assert sv9.ai_bank_path.read_bytes() != orig_m3_ai, "_AI.bank debió actualizarse en modo Exterior-only"

                    # 9b. Inyectar Interior only sobre el mismo auto -> debe mantener Exterior vanilla Y restaurar _AI.bank a vanilla!
                    app.vanilla_audio_scope_var.set("interior")
                    app._on_vanilla_scope_changed()
                    assert "vanilla" in app.vanilla_scope_hint_var.get().lower()
                    msgs9.clear()
                    errs9.clear()
                    app._run_inject_vanilla()
                    t9 = time.time() + 60
                    while not any("Sound Successfully Imported" in m for m in msgs9) and not errs9 and time.time() < t9:
                        app.update()
                        time.sleep(0.05)
                    assert not errs9, f"Error en inyección Interior-only: {errs9}"
                    assert sv9.modified_scope == "interior"
                    fb_int9 = FmodBank(sv9.bank_path)
                    ev_map_int9 = {ev.guid_str.lower(): ev for ev in fb_int9.events}
                    assert any("z4gt3" in s.lower() for s in ev_map_int9[sv9.event_guids["EngineInt"].lower()].linked_samples)
                    assert any("m3gtr_ext_" in s.lower() for s in ev_map_int9[sv9.event_guids["EngineExt"].lower()].linked_samples)
                    assert sv9.ai_bank_path.read_bytes() == orig_m3_ai, "_AI.bank debió restaurarse a vanilla en modo Interior-only!"

                    app.show_view("modified")
                    app.update()
                    row_vals = app.modified_tree.item(sv9.vehicle_id, "values")
                    assert row_vals[2] == "Interior only" and row_vals[3] == "Kept vanilla", f"Fila inesperada en modified_tree: {row_vals}"
                    assert (sbx9_backup / sv9.vehicle_id / sv9.bank_path.name).read_bytes() == orig_m3_p
                    print("   Selector Vanilla ('Full' / 'Interior only' / 'Exterior only') y respaldo intacto verificados al 100%!")

                    # 10. Verificar UI y flujo quirúrgico multi-evento (Opción C) en F_Ultimate_2022 y Porsche_963
                    print("10. Probando UI y selector quirúrgico multi-evento (Opción C: single-car vs multi-event)...")
                    app.vanilla_mgr = real_mgr
                    app.vanilla_search_var.set("F Ultimate 2022")
                    app._filter_vanilla_vehicles()
                    assert app.vanilla_selected_vehicle is not None
                    assert app.vanilla_selected_vehicle.has_multi_engine_variants
                    assert app.vanilla_ev_box.winfo_manager() == "grid", "El panel multi-evento debía mostrarse para F_Ultimate_2022"
                    assert "Surgical EVIB Proxy" in app.vanilla_event_target_badge_var.get()
                    assert app.vanilla_event_target_mode_var.get() == "all"

                    app.vanilla_event_target_mode_var.set("specific")
                    app._on_vanilla_event_target_mode_changed()
                    assert str(app.btn_choose_vanilla_events["state"]) == "normal"

                    app.vanilla_search_var.set("Corvette C8 GT3R")
                    app._filter_vanilla_vehicles()
                    assert app.vanilla_selected_vehicle is not None
                    assert not app.vanilla_selected_vehicle.has_multi_engine_variants
                    assert app.vanilla_ev_box.winfo_manager() == "", "El panel multi-evento debía ocultarse en bancos de 1 solo auto"
                    app.vanilla_search_var.set("")
                    app._filter_vanilla_vehicles()
                    print("   UI contextual multi-evento (Opción C) verificada al 100%!")
                finally:
                    app.vanilla_audio_scope_var.set("full")
                    app._on_vanilla_scope_changed()
                    app.vanilla_mgr = real_mgr
                    mb.showinfo, mb.askyesno, mb.showerror, mb.showwarning = orig_mb_i9, orig_mb_y9, orig_mb_e9, orig_mb_w9
                    if hasattr(ga, "messagebox"):
                        if orig_ga_i9: ga.messagebox.showinfo = orig_ga_i9
                        if orig_ga_y9: ga.messagebox.askyesno = orig_ga_y9
                        if orig_ga_e9: ga.messagebox.showerror = orig_ga_e9
                        if orig_ga_w9: ga.messagebox.showwarning = orig_ga_w9

        app.update_idletasks()
        print("GUI Smoke Test + Audio Preview Async + Soporte GUIDs.txt + Pestaña 3 (v1.4.0) + Vanilla Manager (v1.6.0) superado al 100% sin errores!")
    finally:
        app._on_close()
        try:
            app.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
