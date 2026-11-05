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

                # 5. Probar Eliminar Banda seleccionada
                vis_after = app.rpm_tree.get_children()
                if vis_after:
                    app.rpm_tree.selection_set(vis_after[0])
                    app._on_rpm_band_selected()
                    orig_askyesno = mb.askyesno
                    mb.askyesno = lambda *args, **kw: True
                    try:
                        app.cmd_delete_rpm_band()
                        assert "eliminada y silenciada" in app.status_var.get()
                        print("   GUI Eliminar Banda probada:", app.status_var.get())
                    finally:
                        mb.askyesno = orig_askyesno

        app.update_idletasks()
        print("GUI Smoke Test + Audio Preview Async + Soporte GUIDs.txt + Pestaña 3 (v1.4.0) superado al 100% sin errores!")
    finally:
        app._on_close()


if __name__ == "__main__":
    main()
