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
