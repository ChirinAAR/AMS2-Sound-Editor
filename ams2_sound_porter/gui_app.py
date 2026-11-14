"""Interfaz Gráfica de Escritorio (GUI) para AMS2 Sound Porter Studio."""
from __future__ import annotations

import ctypes
import os
import queue
import re
import struct
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore[assignment]

from .audio_dsp import conform_audio, read_wav_file, write_wav_bytes
from .engine_sim import EnginePassTrajectory, render_engine_pass_wav_bytes
from .fmod_bank import FSB5_MODE_FADPCM, FSB5_MODE_PCM16, FmodBank
from .omvt_csdbin import SOUND_SLOT_GROUPS, CsdbinDocument
from .vanilla_manager import VanillaVehicle, VanillaVehicleManager
from .porter_engine import (
    ModExportSummary,
    RpmBandConfig,
    SlotMappingConfig,
    SourceAudioLibrary,
    ThrottleCutConfig,
    apply_ported_bank_to_csdbin,
    auto_detect_throttle_cut_config,
    auto_distribute_rpm_bands,
    auto_distribute_single_rpm_band,
    auto_map_samples,
    build_default_rpm_bands,
    build_direct_ported_bank_from_ac,
    build_mod_event_paths,
    build_ported_bank,
    estimate_engine_rpm_bounds,
    export_ams2_mod_package,
    sync_faithful_assetto_corsa_setup,
    sync_faithful_section_from_source,
)


# Paleta oscura estilo SimRacing Studio
BG_DARK = "#0f141c"
BG_PANEL = "#171f2c"
BG_CARD = "#1e293b"
BG_INPUT = "#0b1017"
FG_TEXT = "#f1f5f9"
FG_MUTED = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_BLUE_HOVER = "#2563eb"
ACCENT_GREEN = "#10b981"
ACCENT_GREEN_HOVER = "#059669"
ACCENT_AMBER = "#f59e0b"
BORDER_COLOR = "#334155"


def enable_windows_dpi_awareness() -> None:
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


class SoundPorterApp(tk.Tk):
    """Ventana principal de AMS2 Sound Porter Studio."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AMS 2 Sound Importer")
        self.geometry("1240x780")
        self.minsize(1060, 680)
        self.configure(bg=BG_DARK)

        # Estado de datos en memoria
        self.port_mode_var = tk.StringVar(value="direct")
        self.source_lib: SourceAudioLibrary | None = None
        self.donor_bank: FmodBank | None = None
        self.mappings: list[SlotMappingConfig] = []
        self.rpm_bands: list[RpmBandConfig] = []
        self.throttle_cut_cfg: ThrottleCutConfig = ThrottleCutConfig(enabled=False)
        self.csd_doc: CsdbinDocument | None = None
        self._preview_slot_toggle: int = 0
        self._temp_preview_files: list[Path] = []
        self._selected_donor_gidx: int | None = None
        self._selected_band_idx: int | None = None
        self._canvas_band_rects: list[tuple[float, float, float, float, int]] = []
        self._sim_anim_after_id: str | None = None
        self._sim_start_perf: float = 0.0
        self._sim_trajectory: EnginePassTrajectory | None = None
        self._sim_current_rpm: float | None = None

        # Variables de control Tk
        self.status_var = tk.StringVar(
            value="Ready. Select 'Edit custom car sounds' or 'Edit vanilla car sounds'."
        )
        self.src_filter_var = tk.StringVar()
        self.src_info_var = tk.StringVar(value="Ningún origen cargado")
        self.donor_info_var = tk.StringVar(value="Ningún banco donante AMS2 cargado")
        self.csd_info_var = tk.StringVar(value="Ningún archivo .csdbin / .csd cargado")

        # Controles de edición de slot (Pestaña 2)
        self.slot_action_var = tk.StringVar(value="keep_donor")
        self.slot_src_combo_var = tk.StringVar(value="")
        self.slot_wav_path_var = tk.StringVar(value="")
        self.slot_rpm_var = tk.StringVar(value="")
        self.slot_gain_var = tk.StringVar(value="0.0")

        # Controles de Bandas RPM, Throttle Cut y Simulador (Pestaña 3)
        self.sim_idle_rpm_var = tk.StringVar(value="950")
        self.sim_max_rpm_var = tk.StringVar(value="7800")
        self.rpm_view_persp_var = tk.StringVar(value="int")
        self.show_muted_bands_var = tk.BooleanVar(value=False)
        self.rpm_overlap_status_var = tk.StringVar(
            value="Carga un banco donante de AMS2 en el Paso 2 para visualizar y editar las bandas de RPM."
        )
        self.tcut_enabled_var = tk.BooleanVar(value=False)
        self.tcut_int_combo_var = tk.StringVar(value="")
        self.tcut_ext_combo_var = tk.StringVar(value="")
        self.tcut_int_wav_var = tk.StringVar(value="")
        self.tcut_ext_wav_var = tk.StringVar(value="")
        self.tcut_gain_var = tk.StringVar(value="1.5")

        self.band_rmin_var = tk.StringVar(value="")
        self.band_fi_var = tk.StringVar(value="")
        self.band_fo_var = tk.StringVar(value="")
        self.band_rmax_var = tk.StringVar(value="")
        self.band_root_rpm_var = tk.StringVar(value="")
        self.band_gain_var = tk.StringVar(value="0.0")
        self.band_source_var = tk.StringVar(value="")
        self.band_custom_wav_var = tk.StringVar(value="")
        self.batch_root_rpm_var = tk.StringVar(value="+300")
        self.batch_rpm_var = tk.StringVar(value="+450")
        self.batch_gain_var = tk.StringVar(value="+1.5")

        self.sim_telemetry_var = tk.StringVar(
            value="Simulador listo — Prueba la pasada en 5 etapas (Idle -> Corte 1ª -> Cambio + Subida 2ª -> 3s Corte + 3s Desacel -> Rebaje Auto-Blip -> Idle)."
        )
        self.sim_stage_var = tk.StringVar(value="Stage: Idle")

        # Controles de metadatos .csdbin (Pestaña 3)
        self.csd_id_var = tk.StringVar(value="0000000000000000")
        self.csd_class_var = tk.StringVar(value="AMS2_closed_cockpit")
        self.csd_mix_var = tk.StringVar(value="RaceCarClosedCockpit_V4_set2")
        self.csd_gears_var = tk.StringVar(value="6")
        self.csd_vol_ext_var = tk.StringVar(value="1.0")
        self.csd_vol_int_var = tk.StringVar(value="1.0")
        self.csd_pos_x_var = tk.StringVar(value="0.0")
        self.csd_pos_y_var = tk.StringVar(value="0.0")
        self.csd_pos_z_var = tk.StringVar(value="0.0")
        self.csd_link_vars: dict[str, tk.StringVar] = {}

        # Controles de exportación (Pestaña 4)
        self.mod_id_var = tk.StringVar(value="mi_sonido_ac")
        self.veh_folder_var = tk.StringVar(value="mi_vehiculo_mod")
        self.out_dir_var = tk.StringVar(value=str(Path.home() / "Desktop" / "Salida_Mod_Sonido_AMS2"))
        self.standalone_var = tk.BooleanVar(value=True)
        self.sync_pitch_var = tk.BooleanVar(value=True)
        self.auto_link_csd_var = tk.BooleanVar(value=True)
        self.open_folder_after_var = tk.BooleanVar(value=True)
        self.codec_choice_var = tk.StringVar(value="PCM16")

        # Controles y estado de Vehículos Vanilla / Base de AMS2
        self.vanilla_mgr = VanillaVehicleManager()
        self.target_mode_var = tk.StringVar(value="vanilla")
        self.vanilla_ams2_path_var = tk.StringVar(value=str(self.vanilla_mgr.ams2_root))
        self.vanilla_search_var = tk.StringVar(value="")
        self.vanilla_selected_id_var = tk.StringVar(value="")
        self.vanilla_selected_vehicle: VanillaVehicle | None = None
        self.vanilla_status_card_var = tk.StringVar(
            value="Selecciona un vehículo vanilla de la lista para ver su información de audio."
        )
        self.vanilla_auto_backup_var = tk.BooleanVar(value=True)
        self.vanilla_sync_ai_var = tk.BooleanVar(value=True)
        self.vanilla_mute_pops_var = tk.BooleanVar(value=True)
        self.mute_all_pops_var = tk.BooleanVar(value=True)
        self.invert_cone_var = tk.BooleanVar(value=True)
        self.vanilla_filtered_vehicles: list[VanillaVehicle] = []
        self.vanilla_export_dest_var = tk.StringVar(value="Ningún vehículo vanilla seleccionado.")

        # Estado de navegación y vista simplificada (4 vistas)
        self._current_view: str = "menu"
        self._test_return_view: str = "custom"
        self._pending_guids_path: Path | None = None
        self.ui_bank_check_var = tk.StringVar(value="○")
        self.ui_bank_status_var = tk.StringVar(value=".bank : Not selected")
        self.ui_guids_check_var = tk.StringVar(value="○")
        self.ui_guids_status_var = tk.StringVar(value="GUIDs : Not selected")
        self.ui_csdbin_check_var = tk.StringVar(value="○")
        self.ui_csdbin_status_var = tk.StringVar(value=".csdbin : Not selected")

        self._configure_styles()
        self._build_legacy_notebook_hidden()
        self._build_new_views()
        self._build_statusbar()
        self.show_view("menu")

        # Inicializar catálogo de vehículos vanilla
        self._populate_vanilla_vehicles_list()

        # Cola de despacho para comunicación segura entre hilos de fondo y la UI
        self._ui_queue: queue.Queue[tuple[Any, tuple, dict]] = queue.Queue()
        self._poll_ui_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _poll_ui_queue(self) -> None:
        while not self._ui_queue.empty():
            try:
                fn, args, kwargs = self._ui_queue.get_nowait()
                fn(*args, **kwargs)
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error en despacho de cola UI: {e}")
        self.after(40, self._poll_ui_queue)

    def dispatch_to_ui(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Encola una función para ejecutarse de forma segura en el hilo principal de Tkinter."""
        self._ui_queue.put((fn, args, kwargs))

    # ========================================================================
    # Estilos visuales TTK
    # ========================================================================
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=FG_TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("Card.TFrame", background=BG_CARD)

        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_TEXT)
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_TEXT)
        style.configure("Muted.TLabel", background=BG_PANEL, foreground=FG_MUTED, font=("Segoe UI", 9))
        style.configure("CardMuted.TLabel", background=BG_CARD, foreground=FG_MUTED, font=("Segoe UI", 9))
        style.configure("HeaderTitle.TLabel", background=BG_PANEL, foreground="#ffffff", font=("Segoe UI Semibold", 15))
        style.configure("Section.TLabel", background=BG_PANEL, foreground="#60a5fa", font=("Segoe UI Semibold", 11))
        style.configure("CardSection.TLabel", background=BG_CARD, foreground="#60a5fa", font=("Segoe UI Semibold", 11))

        style.configure(
            "TNotebook",
            background=BG_DARK,
            borderwidth=0,
            tabmargins=[8, 6, 8, 0],
        )
        style.configure(
            "TNotebook.Tab",
            background=BG_PANEL,
            foreground=FG_MUTED,
            padding=[16, 9],
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", ACCENT_BLUE), ("active", BG_CARD)],
            foreground=[("selected", "#ffffff"), ("active", FG_TEXT)],
        )

        style.configure(
            "TButton",
            background=BG_CARD,
            foreground=FG_TEXT,
            bordercolor=BORDER_COLOR,
            focusthickness=0,
            padding=[12, 6],
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "TButton",
            background=[("active", "#334155"), ("disabled", "#1e2430")],
            foreground=[("disabled", "#64748b")],
        )

        style.configure(
            "Primary.TButton",
            background=ACCENT_BLUE,
            foreground="#ffffff",
            bordercolor=ACCENT_BLUE,
            padding=[14, 7],
            font=("Segoe UI Semibold", 10),
        )
        style.map("Primary.TButton", background=[("active", ACCENT_BLUE_HOVER)])

        style.configure(
            "Success.TButton",
            background=ACCENT_GREEN,
            foreground="#ffffff",
            bordercolor=ACCENT_GREEN,
            padding=[18, 10],
            font=("Segoe UI Bold", 11),
        )
        style.map("Success.TButton", background=[("active", ACCENT_GREEN_HOVER)])

        style.configure(
            "TEntry",
            fieldbackground=BG_INPUT,
            foreground=FG_TEXT,
            insertcolor=FG_TEXT,
            bordercolor=BORDER_COLOR,
            padding=5,
        )
        style.configure(
            "TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_CARD,
            foreground=FG_TEXT,
            arrowcolor=FG_TEXT,
            bordercolor=BORDER_COLOR,
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_INPUT)],
            foreground=[("readonly", FG_TEXT)],
            selectbackground=[("readonly", ACCENT_BLUE)],
            selectforeground=[("readonly", "#ffffff")],
        )

        style.configure(
            "TCheckbutton",
            background=BG_CARD,
            foreground=FG_TEXT,
            font=("Segoe UI", 10),
        )
        style.map("TCheckbutton", background=[("active", BG_CARD)])

        style.configure(
            "TRadiobutton",
            background=BG_CARD,
            foreground=FG_TEXT,
            font=("Segoe UI", 10),
        )
        style.map("TRadiobutton", background=[("active", BG_CARD)])

        style.configure(
            "Treeview",
            background=BG_INPUT,
            fieldbackground=BG_INPUT,
            foreground=FG_TEXT,
            rowheight=27,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_CARD,
            foreground="#cbd5e1",
            font=("Segoe UI Semibold", 9),
            padding=[6, 6],
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", "#1d4ed8")],
            foreground=[("selected", "#ffffff")],
        )
        style.map("Treeview.Heading", background=[("active", "#334155")])

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=BG_INPUT,
            background=ACCENT_GREEN,
            bordercolor=BG_INPUT,
            lightcolor=ACCENT_GREEN,
            darkcolor=ACCENT_GREEN,
        )

    # ========================================================================
    # Construcción de Cabecera, Pestañas y Barra de Estado
    # ========================================================================
    def _build_header(self) -> None:
        hdr = ttk.Frame(self, style="Panel.TFrame", padding=(18, 12))
        hdr.pack(fill="x", side="top")

        left = ttk.Frame(hdr, style="Panel.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(
            left,
            text="🏎️  AMS2 Sound Porter Studio",
            style="HeaderTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            left,
            text="Porteo directo de bancos FMOD de Assetto Corsa / WAV hacia Automobilista 2 + Editor visual .csdbin (OMVT)",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(hdr, style="Panel.TFrame")
        right.pack(side="right")
        ttk.Button(
            right,
            text="🛡️ Restaurar Sonidos Vanilla...",
            command=self.cmd_open_vanilla_restore_dialog,
        ).pack(side="right", padx=(6, 0))
        ttk.Button(
            right,
            text="⏹ Detener Audio",
            command=self.stop_audio_preview,
        ).pack(side="right", padx=(6, 0))

    @property
    def is_direct_ac_mode(self) -> bool:
        return (
            self.port_mode_var.get() == "direct"
            and self.source_lib is not None
            and self.source_lib.fmod_bank is not None
        )

    @property
    def active_engine_bank(self) -> FmodBank | None:
        if self.is_direct_ac_mode:
            return self.source_lib.fmod_bank  # type: ignore[union-attr]
        return self.donor_bank

    def _build_new_views(self) -> None:
        """Construye las 4 vistas principales del nuevo flujo AMS 2 Sound Importer."""
        self.view_container = ttk.Frame(self, style="TFrame")
        self.view_container.pack(fill="both", expand=True)

        self.view_menu = ttk.Frame(self.view_container, style="TFrame")
        self.view_custom = ttk.Frame(self.view_container, style="TFrame")
        self.view_vanilla = ttk.Frame(self.view_container, style="TFrame")
        self.view_test_edit = ttk.Frame(self.view_container, style="TFrame")

        self._build_view_menu()
        self._build_view_custom()
        self._build_view_vanilla()
        self._build_view_test_edit()

    def _build_legacy_notebook_hidden(self) -> None:
        """Mantiene inicializados los componentes internos legados (incluido modo donante) sin mostrarlos."""
        self._legacy_host = ttk.Frame(self)
        self.notebook = ttk.Notebook(self._legacy_host)

        self.tab_source = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)
        self.tab_donor = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)
        self.tab_rpm_sim = ttk.Frame(self.notebook, style="Panel.TFrame", padding=10)
        self.tab_csdbin = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)
        self.tab_export = ttk.Frame(self.notebook, style="Panel.TFrame", padding=12)

        self.notebook.add(self.tab_source, text="  1. Audio Origen (AC .bank / WAV)  ")
        self.notebook.add(self.tab_donor, text="  2. Banco Donante AMS2 y Mapeo  ")
        self.notebook.add(self.tab_rpm_sim, text="  3. Bandas RPM, Throttle Cut y Simulador  ")
        self.notebook.add(self.tab_csdbin, text="  4. Vehículo .csdbin (OMVT)  ")
        self.notebook.add(self.tab_export, text="  5. Compilar y Exportar Mod  ")

        self._build_tab_source()
        self._build_tab_donor()
        self._build_tab_csdbin()
        self._build_tab_export()

    def show_view(self, view_name: str) -> None:
        """Cambia la vista activa entre 'menu', 'custom', 'vanilla' y 'test_edit'."""
        self._current_view = view_name
        for vf in (self.view_menu, self.view_custom, self.view_vanilla, self.view_test_edit):
            vf.pack_forget()

        if view_name == "menu":
            self.view_menu.pack(fill="both", expand=True)
        elif view_name == "custom":
            self.target_mode_var.set("mod")
            self.port_mode_var.set("direct")
            self._on_target_mode_changed()
            self.view_custom.pack(fill="both", expand=True)
        elif view_name == "vanilla":
            self.target_mode_var.set("vanilla")
            self.port_mode_var.set("direct")
            self._on_target_mode_changed()
            self.view_vanilla.pack(fill="both", expand=True)
        elif view_name == "test_edit":
            self.view_test_edit.pack(fill="both", expand=True)
            self._refresh_rpm_tab()
            self.after(20, self._draw_rpm_bands_canvas)

    def _build_view_menu(self) -> None:
        outer = ttk.Frame(self.view_menu, style="TFrame")
        outer.place(relx=0.5, rely=0.47, anchor="center")

        tk.Label(
            outer,
            text="AMS 2 Sound Importer",
            bg=BG_DARK,
            fg="#ffffff",
            font=("Segoe UI", 32, "bold"),
        ).pack(pady=(0, 38))

        btn_opts = {
            "bg": BG_CARD,
            "fg": "#f1f5f9",
            "activebackground": "#25344d",
            "activeforeground": "#ffffff",
            "font": ("Segoe UI", 13),
            "bd": 1,
            "relief": "solid",
            "highlightthickness": 1,
            "highlightbackground": "#94a3b8",
            "cursor": "hand2",
            "pady": 8,
        }

        tk.Button(
            outer,
            text="Edit custom car sounds",
            width=30,
            command=lambda: self.show_view("custom"),
            **btn_opts,
        ).pack(pady=(0, 26))

        tk.Button(
            outer,
            text="Edit vanilla car sounds",
            width=30,
            command=lambda: self.show_view("vanilla"),
            **btn_opts,
        ).pack(pady=(0, 34))

        tk.Button(
            outer,
            text="Close",
            width=18,
            command=self._on_close,
            **btn_opts,
        ).pack()

    def _build_source_step_rows(self, container: ttk.Frame) -> None:
        """Construye las filas 1) .bank y 2) GUIDs compartidas entre Custom y Vanilla."""
        step_font = ("Segoe UI", 15)
        status_font = ("Segoe UI", 12)
        check_font = ("Segoe UI", 14, "bold")
        btn_cfg = {
            "bg": BG_CARD,
            "fg": "#f1f5f9",
            "activebackground": "#25344d",
            "activeforeground": "#ffffff",
            "font": ("Segoe UI", 11),
            "bd": 1,
            "relief": "solid",
            "cursor": "hand2",
            "width": 24,
            "pady": 6,
        }

        # Fila 1: Select sounds .bank
        tk.Label(container, text="1)", bg=BG_DARK, fg="#f1f5f9", font=step_font).grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=12
        )
        tk.Button(container, text="Select sounds .bank", command=self.cmd_load_ac_bank, **btn_cfg).grid(
            row=0, column=1, sticky="w", padx=(0, 26), pady=12
        )
        st1 = ttk.Frame(container, style="TFrame")
        st1.grid(row=0, column=2, sticky="w", pady=12)
        tk.Label(st1, textvariable=self.ui_bank_check_var, bg=BG_DARK, fg="#ffffff", font=check_font).pack(
            side="left", padx=(0, 8)
        )
        tk.Label(st1, textvariable=self.ui_bank_status_var, bg=BG_DARK, fg="#e2e8f0", font=status_font).pack(
            side="left"
        )

        # Fila 2: Select GUIDs
        tk.Label(container, text="2)", bg=BG_DARK, fg="#f1f5f9", font=step_font).grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=12
        )
        tk.Button(container, text="Select GUIDs", command=self.cmd_load_ac_guids_txt, **btn_cfg).grid(
            row=1, column=1, sticky="w", padx=(0, 26), pady=12
        )
        st2 = ttk.Frame(container, style="TFrame")
        st2.grid(row=1, column=2, sticky="w", pady=12)
        tk.Label(st2, textvariable=self.ui_guids_check_var, bg=BG_DARK, fg="#ffffff", font=check_font).pack(
            side="left", padx=(0, 8)
        )
        tk.Label(st2, textvariable=self.ui_guids_status_var, bg=BG_DARK, fg="#e2e8f0", font=status_font).pack(
            side="left"
        )

    def _build_view_custom(self) -> None:
        hdr = tk.Frame(self.view_custom, bg="#080c13", padx=20, pady=14)
        hdr.pack(fill="x", side="top")
        tk.Label(
            hdr,
            text="Custom Car sound",
            bg="#080c13",
            fg="#ffffff",
            font=("Segoe UI", 22, "bold"),
        ).pack(side="left")
        ttk.Button(
            hdr,
            text="← Menu",
            command=lambda: self.show_view("menu"),
        ).pack(side="right", pady=2)

        body = ttk.Frame(self.view_custom, style="TFrame", padding=(56, 36))
        body.pack(fill="both", expand=True)

        rows = ttk.Frame(body, style="TFrame")
        rows.pack(anchor="w")
        self._build_source_step_rows(rows)

        # Fila 3: Select car .csdbin
        tk.Label(rows, text="3)", bg=BG_DARK, fg="#f1f5f9", font=("Segoe UI", 15)).grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=12
        )
        tk.Button(
            rows,
            text="Select car .csdbin",
            command=self.cmd_open_csdbin,
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 11),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=24,
            pady=6,
        ).grid(row=2, column=1, sticky="w", padx=(0, 26), pady=12)
        st3 = ttk.Frame(rows, style="TFrame")
        st3.grid(row=2, column=2, sticky="w", pady=12)
        tk.Label(st3, textvariable=self.ui_csdbin_check_var, bg=BG_DARK, fg="#ffffff", font=("Segoe UI", 14, "bold")).pack(
            side="left", padx=(0, 8)
        )
        tk.Label(st3, textvariable=self.ui_csdbin_status_var, bg=BG_DARK, fg="#e2e8f0", font=("Segoe UI", 12)).pack(
            side="left"
        )

        # Acciones inferiores
        actions = ttk.Frame(body, style="TFrame")
        actions.pack(anchor="w", pady=(34, 0))
        self.btn_custom_import = tk.Button(
            actions,
            text="Import sound",
            command=lambda: self.cmd_import_sound("mod"),
            bg=ACCENT_BLUE,
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            font=("Segoe UI", 14),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=24,
            pady=6,
        )
        self.btn_custom_import.pack(side="left", padx=(0, 32))

        tk.Button(
            actions,
            text="Test/edit sounds",
            command=lambda: self.cmd_open_test_edit("custom"),
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 13),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=22,
            pady=6,
        ).pack(side="left")

    def _build_view_vanilla(self) -> None:
        hdr = tk.Frame(self.view_vanilla, bg="#080c13", padx=20, pady=14)
        hdr.pack(fill="x", side="top")
        tk.Label(
            hdr,
            text="Vanilla car sounds",
            bg="#080c13",
            fg="#ffffff",
            font=("Segoe UI", 22, "bold"),
        ).pack(side="left")

        right_hdr = tk.Frame(hdr, bg="#080c13")
        right_hdr.pack(side="right")
        tk.Label(
            right_hdr,
            text="Game directory:",
            bg="#080c13",
            fg="#f1f5f9",
            font=("Segoe UI", 11),
        ).pack(side="left", padx=(0, 8))
        ttk.Entry(right_hdr, textvariable=self.vanilla_ams2_path_var, width=42).pack(side="left", padx=(0, 8))
        tk.Button(
            right_hdr,
            text="Browse...",
            command=self.cmd_browse_ams2_path,
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
            bd=1,
            relief="solid",
            cursor="hand2",
            padx=12,
            pady=3,
        ).pack(side="left", padx=(0, 12))
        ttk.Button(
            right_hdr,
            text="← Menu",
            command=lambda: self.show_view("menu"),
        ).pack(side="left")

        body = ttk.Frame(self.view_vanilla, style="TFrame", padding=(56, 36))
        body.pack(fill="both", expand=True)

        rows = ttk.Frame(body, style="TFrame")
        rows.pack(anchor="w")
        self._build_source_step_rows(rows)

        # Fila 3: Select car + combobox vehículo vanilla
        tk.Label(rows, text="3)", bg=BG_DARK, fg="#f1f5f9", font=("Segoe UI", 15)).grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=12
        )
        tk.Button(
            rows,
            text="Select car",
            command=self.cmd_open_vanilla_car_picker,
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 11),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=24,
            pady=6,
        ).grid(row=2, column=1, sticky="w", padx=(0, 26), pady=12)

        self.vanilla_combo = ttk.Combobox(
            rows,
            textvariable=self.vanilla_selected_id_var,
            state="readonly",
            width=36,
        )
        self.vanilla_combo.grid(row=2, column=2, sticky="w", pady=12)
        self.vanilla_combo.bind("<<ComboboxSelected>>", self._on_vanilla_vehicle_selected)

        # Acciones inferiores
        actions = ttk.Frame(body, style="TFrame")
        actions.pack(anchor="w", pady=(34, 0))
        self.btn_vanilla_import = tk.Button(
            actions,
            text="Import sound",
            command=lambda: self.cmd_import_sound("vanilla"),
            bg=ACCENT_BLUE,
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            font=("Segoe UI", 14),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=24,
            pady=6,
        )
        self.btn_vanilla_import.pack(side="left", padx=(0, 24))

        tk.Button(
            actions,
            text="Test/edit sounds",
            command=lambda: self.cmd_open_test_edit("vanilla"),
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 13),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=20,
            pady=6,
        ).pack(side="left", padx=(0, 24))

        tk.Button(
            actions,
            text="Restore original sounds",
            command=self.cmd_open_vanilla_restore_dialog,
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 13),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=22,
            pady=6,
        ).pack(side="left")

    def _build_view_test_edit(self) -> None:
        hdr = tk.Frame(self.view_test_edit, bg="#080c13", padx=16, pady=10)
        hdr.pack(fill="x", side="top")
        tk.Label(
            hdr,
            text="Test/edit Sounds",
            bg="#080c13",
            fg="#ffffff",
            font=("Segoe UI", 22, "bold"),
        ).pack(side="left")
        tk.Button(
            hdr,
            text="Finish",
            command=self.cmd_finish_test_edit,
            bg=ACCENT_BLUE,
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            font=("Segoe UI", 14),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=18,
            pady=4,
        ).pack(side="right")

        content = ttk.Frame(self.view_test_edit, style="Panel.TFrame", padding=(10, 6))
        content.pack(fill="both", expand=True)

        # Barra de vistas: Views: (o) Interior  ( ) Exterior  [+] [-]
        view_bar = ttk.Frame(content, style="Panel.TFrame", padding=(4, 4))
        view_bar.pack(fill="x", side="top", pady=(0, 4))
        ttk.Label(view_bar, text="Views:", style="Section.TLabel").pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            view_bar,
            text="🎧 Interior (EngineInt + Cockpit)",
            variable=self.rpm_view_persp_var,
            value="int",
            command=self._refresh_rpm_tab,
        ).pack(side="left", padx=(0, 14))
        ttk.Radiobutton(
            view_bar,
            text="🔊 Exterior (EngineExt)",
            variable=self.rpm_view_persp_var,
            value="ext",
            command=self._refresh_rpm_tab,
        ).pack(side="left", padx=(0, 14))
        ttk.Button(view_bar, text="+", width=3, command=self.cmd_add_rpm_band).pack(side="left", padx=(0, 4))
        ttk.Button(view_bar, text="-", width=3, command=self.cmd_delete_rpm_band).pack(side="left")

        # Canvas RPM
        self.rpm_canvas = tk.Canvas(
            content,
            height=136,
            bg=BG_INPUT,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            cursor="hand2",
        )
        self.rpm_canvas.pack(fill="x", side="top", pady=(0, 6))
        self.rpm_canvas.bind("<Configure>", lambda _e: self._draw_rpm_bands_canvas())
        self.rpm_canvas.bind("<Button-1>", self._on_rpm_canvas_click)

        # Panel inferior: Simulador + Throttle Cut
        sim_tcut_f = ttk.Frame(content, style="Panel.TFrame", padding=(4, 6))
        sim_tcut_f.pack(fill="x", side="bottom")

        sim_btns = ttk.Frame(sim_tcut_f, style="Panel.TFrame")
        sim_btns.pack(fill="x", pady=(0, 8))
        self.btn_sim_int = tk.Button(
            sim_btns,
            text="Test interior sound",
            command=self.cmd_simulate_pass_int,
            bg=ACCENT_BLUE,
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            font=("Segoe UI", 12),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=20,
            pady=4,
        )
        self.btn_sim_int.pack(side="left", padx=(0, 8))

        self.btn_sim_ext = tk.Button(
            sim_btns,
            text="Test exterior sound",
            command=self.cmd_simulate_pass_ext,
            bg=ACCENT_BLUE,
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            font=("Segoe UI", 12),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=20,
            pady=4,
        )
        self.btn_sim_ext.pack(side="left", padx=(0, 8))

        tk.Button(
            sim_btns,
            text="Stop",
            command=self.stop_audio_preview,
            bg=BG_CARD,
            fg="#f1f5f9",
            activebackground="#25344d",
            activeforeground="#ffffff",
            font=("Segoe UI", 12),
            bd=1,
            relief="solid",
            cursor="hand2",
            width=12,
            pady=4,
        ).pack(side="left", padx=(0, 14))

        ttk.Label(sim_btns, textvariable=self.sim_stage_var, style="Muted.TLabel").pack(side="left")

        # Bloque Throttle Cut debajo de botones del simulador
        tc_box = ttk.Frame(sim_tcut_f, style="Card.TFrame", padding=(8, 6))
        tc_box.pack(anchor="w")

        tc_r1 = ttk.Frame(tc_box, style="Card.TFrame")
        tc_r1.pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(
            tc_r1,
            text="Inject Throttle Cut into engine",
            variable=self.tcut_enabled_var,
        ).pack(side="left", padx=(0, 8))
        ttk.Entry(tc_r1, textvariable=self.tcut_gain_var, width=5).pack(side="left", padx=(0, 8))
        ttk.Label(tc_r1, text="Gain (dB):", style="Card.TLabel").pack(side="left")

        tc_r2 = ttk.Frame(tc_box, style="Card.TFrame")
        tc_r2.pack(anchor="w")
        ttk.Label(tc_r2, text="Cut INT:", style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.tcut_int_combo = ttk.Combobox(tc_r2, textvariable=self.tcut_int_combo_var, state="readonly", width=26)
        self.tcut_int_combo.pack(side="left", padx=(0, 4))
        self.tcut_int_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_tcut_combo_selected("int"))
        ttk.Button(tc_r2, text="▶", width=3, command=lambda: self.cmd_preview_throttle_cut("int")).pack(
            side="left", padx=(0, 3)
        )
        ttk.Button(tc_r2, text="📂", width=3, command=lambda: self.cmd_pick_tcut_wav("int")).pack(
            side="left", padx=(0, 14)
        )

        ttk.Label(tc_r2, text="Cut EXT:", style="Card.TLabel").pack(side="left", padx=(0, 4))
        self.tcut_ext_combo = ttk.Combobox(tc_r2, textvariable=self.tcut_ext_combo_var, state="readonly", width=26)
        self.tcut_ext_combo.pack(side="left", padx=(0, 4))
        self.tcut_ext_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_tcut_combo_selected("ext"))
        ttk.Button(tc_r2, text="▶", width=3, command=lambda: self.cmd_preview_throttle_cut("ext")).pack(
            side="left", padx=(0, 3)
        )
        ttk.Button(tc_r2, text="📂", width=3, command=lambda: self.cmd_pick_tcut_wav("ext")).pack(side="left")

        # Panel de edición numérica de bandas (encima del simulador)
        band_ed = ttk.Frame(content, style="Card.TFrame", padding=(10, 7))
        band_ed.pack(fill="x", side="bottom", pady=(0, 6))

        bed_r1 = ttk.Frame(band_ed, style="Card.TFrame")
        bed_r1.pack(fill="x", pady=(0, 5))
        ttk.Button(
            bed_r1,
            text="Shift all Root RPM",
            command=self.cmd_batch_shift_all_root_rpm,
        ).pack(side="left", padx=(0, 4))
        ttk.Entry(bed_r1, textvariable=self.batch_root_rpm_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Button(
            bed_r1,
            text="Shift all RPM",
            command=self.cmd_batch_shift_all_rpm,
        ).pack(side="left", padx=(0, 4))
        ttk.Entry(bed_r1, textvariable=self.batch_rpm_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Button(
            bed_r1,
            text="Amplify all bands",
            command=self.cmd_batch_gain_all_bands,
        ).pack(side="left", padx=(0, 4))
        ttk.Entry(bed_r1, textvariable=self.batch_gain_var, width=5).pack(side="left", padx=(0, 8))
        ttk.Button(
            bed_r1,
            text="▶ Preview Band Sample",
            command=self.cmd_preview_selected_band_sample,
        ).pack(side="left")

        self.selected_band_lbl = ttk.Label(bed_r1, text="", style="CardMuted.TLabel")
        self.selected_band_lbl.pack(side="right")

        bed_r_sample = ttk.Frame(band_ed, style="Card.TFrame")
        bed_r_sample.pack(fill="x", pady=(0, 5))
        ttk.Label(bed_r_sample, text="Assigned Sample / Replacement:", style="Card.TLabel").pack(side="left", padx=(0, 6))
        self.band_source_combo = ttk.Combobox(
            bed_r_sample,
            textvariable=self.band_source_var,
            state="readonly",
            width=48,
        )
        self.band_source_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.band_source_combo.bind("<<ComboboxSelected>>", self._on_band_source_combo_selected)
        ttk.Button(bed_r_sample, text="📂 WAV...", width=8, command=self.cmd_pick_band_custom_wav).pack(
            side="left", padx=(0, 6)
        )
        self.btn_overwrite_sample = ttk.Button(
            bed_r_sample,
            text="🔁 Overwrite Sample",
            command=self.cmd_overwrite_selected_band_sample,
        )
        self.btn_overwrite_sample.pack(side="left", padx=(0, 6))
        self.btn_reactivate_band = ttk.Button(
            bed_r_sample,
            text="♻ Reactivate Band",
            command=self.cmd_reactivate_selected_band,
        )
        self.btn_reactivate_band.pack(side="left")

        bed_r2 = ttk.Frame(band_ed, style="Card.TFrame")
        bed_r2.pack(fill="x")
        ed_fields = [
            ("Min RPM:", self.band_rmin_var, 7),
            ("Fade-In End:", self.band_fi_var, 7),
            ("Fade-Out Start:", self.band_fo_var, 7),
            ("Max RPM:", self.band_rmax_var, 7),
            ("Root RPM (Pitch):", self.band_root_rpm_var, 8),
            ("Gain (dB):", self.band_gain_var, 6),
        ]
        for col_idx, (lbl_t, tk_v, w_ch) in enumerate(ed_fields):
            ttk.Label(bed_r2, text=lbl_t, style="Card.TLabel").grid(
                row=0, column=col_idx * 2, sticky="w", padx=(0, 4)
            )
            ttk.Entry(bed_r2, textvariable=tk_v, width=w_ch).grid(
                row=0, column=col_idx * 2 + 1, sticky="w", padx=(0, 10)
            )
        ttk.Button(
            bed_r2,
            text="✔ Apply Band Changes",
            style="Primary.TButton",
            command=self.cmd_apply_band_edit,
        ).grid(row=0, column=len(ed_fields) * 2, sticky="e", padx=(6, 0))

        # Tabla central de Bandas RPM
        b_tree_frame = ttk.Frame(content, style="Panel.TFrame")
        b_tree_frame.pack(fill="both", expand=True, pady=(0, 6))
        bcols = ("idx", "role", "sample", "rmin", "fi", "fo", "rmax", "root_rpm", "gain", "status")
        self.rpm_tree = ttk.Treeview(b_tree_frame, columns=bcols, show="headings", selectmode="browse", height=6)
        bheadings = {
            "idx": ("Band", 62),
            "role": ("Layer / Role", 96),
            "sample": ("Currently Assigned Sample", 265),
            "rmin": ("Min RPM", 84),
            "fi": ("Fade-In End", 92),
            "fo": ("Fade-Out Start", 104),
            "rmax": ("Max RPM", 84),
            "root_rpm": ("Root RPM (Pitch)", 106),
            "gain": ("Gain", 78),
            "status": ("Status / Overlap", 195),
        }
        for c, (txt, w) in bheadings.items():
            self.rpm_tree.heading(c, text=txt)
            self.rpm_tree.column(c, width=w, anchor="w" if c in ("sample", "status") else "center")
        self.rpm_tree.tag_configure("band_ok", foreground="#f1f5f9")
        self.rpm_tree.tag_configure("band_replaced", foreground="#86efac")
        self.rpm_tree.tag_configure("band_overlap", foreground="#fbbf24")
        self.rpm_tree.tag_configure("band_muted", foreground="#64748b")
        b_vsb = ttk.Scrollbar(b_tree_frame, orient="vertical", command=self.rpm_tree.yview)
        self.rpm_tree.configure(yscrollcommand=b_vsb.set)
        self.rpm_tree.pack(side="left", fill="both", expand=True)
        b_vsb.pack(side="right", fill="y")
        self.rpm_tree.bind("<<TreeviewSelect>>", self._on_rpm_band_selected)
        self.rpm_tree.bind("<Double-1>", lambda _e: self.cmd_preview_selected_band_sample())

    def _refresh_ui_file_statuses(self) -> None:
        if self.source_lib is not None and self.source_lib.source_path:
            self.ui_bank_check_var.set("✔")
            self.ui_bank_status_var.set(f".bank :  {self.source_lib.source_path.name}")
            if self.source_lib.guids_path is not None and self.source_lib.matched_guids_count > 0:
                self.ui_guids_check_var.set("✔")
                self.ui_guids_status_var.set("GUIDs")
            else:
                self.ui_guids_check_var.set("○")
                self.ui_guids_status_var.set("GUIDs : Not selected (optional)")
        else:
            self.ui_bank_check_var.set("○")
            self.ui_bank_status_var.set(".bank : Not selected")
            if self._pending_guids_path is not None:
                self.ui_guids_check_var.set("✔")
                self.ui_guids_status_var.set(f"GUIDs :  {self._pending_guids_path.name}")
            else:
                self.ui_guids_check_var.set("○")
                self.ui_guids_status_var.set("GUIDs : Not selected")

        if self.csd_doc is not None and self.csd_doc.source_path:
            self.ui_csdbin_check_var.set("✔")
            self.ui_csdbin_status_var.set(f".csdbin :  {self.csd_doc.source_path.name}")
        else:
            self.ui_csdbin_check_var.set("○")
            self.ui_csdbin_status_var.set(".csdbin : Not selected")

    def _set_import_buttons_state(self, state: str) -> None:
        for attr in ("btn_export", "btn_custom_import", "btn_vanilla_import"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.config(state=state)

    def _enforce_locked_export_options(self) -> None:
        """Aplica y bloquea todas las opciones de exportación por defecto en modo directo."""
        self.port_mode_var.set("direct")
        self.standalone_var.set(True)
        self.sync_pitch_var.set(True)
        self.auto_link_csd_var.set(True)
        self.open_folder_after_var.set(True)
        self.codec_choice_var.set("PCM16")
        self.vanilla_auto_backup_var.set(True)
        self.vanilla_sync_ai_var.set(True)
        self.vanilla_mute_pops_var.set(True)
        self.mute_all_pops_var.set(True)
        self.invert_cone_var.set(True)

    def cmd_import_sound(self, mode: str) -> None:
        self._enforce_locked_export_options()
        self.target_mode_var.set(mode)
        self.cmd_run_export()

    def cmd_open_test_edit(self, return_view: str) -> None:
        if self.source_lib is None or self.source_lib.fmod_bank is None:
            messagebox.showwarning(
                "No sound bank loaded",
                "Please select an Assetto Corsa .bank file first (Step 1).",
            )
            return
        self._test_return_view = return_view
        self.show_view("test_edit")

    def cmd_finish_test_edit(self) -> None:
        self.stop_audio_preview()
        self.show_view(self._test_return_view or "menu")

    def cmd_open_vanilla_car_picker(self) -> None:
        """Abre un selector rápido con búsqueda para elegir el vehículo vanilla de AMS2."""
        dlg = tk.Toplevel(self)
        dlg.title("Select vanilla car")
        dlg.geometry("520x460")
        dlg.configure(bg=BG_DARK)
        dlg.transient(self)
        dlg.grab_set()

        top = ttk.Frame(dlg, style="Panel.TFrame", padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Search:", style="Panel.TLabel").pack(side="left", padx=(0, 8))
        ent = ttk.Entry(top, textvariable=self.vanilla_search_var)
        ent.pack(side="left", fill="x", expand=True)

        lb = tk.Listbox(
            dlg,
            bg=BG_INPUT,
            fg="#f1f5f9",
            selectbackground=ACCENT_BLUE,
            selectforeground="#ffffff",
            font=("Segoe UI", 10),
            relief="flat",
        )
        lb.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        def _populate_lb(*_args: Any) -> None:
            self._filter_vanilla_vehicles()
            lb.delete(0, "end")
            for val in self.vanilla_combo["values"]:
                lb.insert("end", val)

        def _confirm_sel(_ev: Any = None) -> None:
            sel = lb.curselection()
            if sel:
                idx = sel[0]
                if 0 <= idx < len(self.vanilla_filtered_vehicles):
                    v = self.vanilla_filtered_vehicles[idx]
                    self.vanilla_selected_id_var.set(self.vanilla_combo["values"][idx])
                    self._on_vanilla_vehicle_selected()
            dlg.destroy()

        ent.bind("<KeyRelease>", _populate_lb)
        lb.bind("<Double-1>", _confirm_sel)
        _populate_lb()
        ent.focus_set()

        bot = ttk.Frame(dlg, style="Panel.TFrame", padding=(10, 6))
        bot.pack(fill="x")
        ttk.Button(bot, text="Select", style="Primary.TButton", command=_confirm_sel).pack(side="right")
        ttk.Button(bot, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 8))

    def _build_notebook(self) -> None:
        self._build_legacy_notebook_hidden()

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, style="Panel.TFrame", padding=(14, 7))
        bar.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=(10, 0))
        ttk.Label(bar, textvariable=self.status_var, style="Muted.TLabel").pack(side="left", fill="x")

    # ========================================================================
    # PESTAÑA 1: Audio de Origen (AC .bank o Carpeta WAV)
    # ========================================================================
    def _build_tab_source(self) -> None:
        top = ttk.Frame(self.tab_source, style="Panel.TFrame")
        top.pack(fill="x", pady=(0, 10))

        ttk.Button(
            top,
            text="📂 Cargar .bank de Assetto Corsa...",
            style="Primary.TButton",
            command=self.cmd_load_ac_bank,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="📄 Cargar GUIDs.txt...",
            command=self.cmd_load_ac_guids_txt,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="📁 Cargar Carpeta de WAVs...",
            command=self.cmd_load_wav_folder,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="▶ Escuchar Muestra Seleccionada",
            command=self.cmd_play_selected_source,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="💾 Extraer WAVs...",
            command=self.cmd_extract_all_source_wavs,
        ).pack(side="left", padx=(0, 12))

        ttk.Label(top, textvariable=self.src_info_var, style="Section.TLabel").pack(side="right")

        # Barra de filtro
        flt = ttk.Frame(self.tab_source, style="Panel.TFrame")
        flt.pack(fill="x", pady=(0, 8))
        ttk.Label(flt, text="🔍 Filtrar muestras:", style="Panel.TLabel").pack(side="left", padx=(0, 6))
        ent_flt = ttk.Entry(flt, textvariable=self.src_filter_var, width=32)
        ent_flt.pack(side="left")
        self.src_filter_var.trace_add("write", lambda *_: self._refresh_source_tree())
        ttk.Label(
            flt,
            text="   💡 GUIDs.txt se detecta automáticamente si está junto al .bank (o usa '📄 Cargar GUIDs.txt...'). Doble clic para escuchar.",
            style="Muted.TLabel",
        ).pack(side="left", padx=10)

        # Tabla Treeview
        tree_frame = ttk.Frame(self.tab_source, style="Panel.TFrame")
        tree_frame.pack(fill="both", expand=True)

        cols = ("idx", "name", "event", "persp", "role", "rpm", "sr", "ch", "loop", "dur", "codec")
        self.src_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        headings = {
            "idx": ("#", 45),
            "name": ("Nombre de Muestra", 265),
            "event": ("Evento FMOD", 160),
            "persp": ("Vista", 72),
            "role": ("Rol Detectado", 105),
            "rpm": ("RPM FMOD / Ref.", 125),
            "sr": ("Frecuencia", 88),
            "ch": ("Canales", 68),
            "loop": ("Loop", 118),
            "dur": ("Duración", 78),
            "codec": ("Códec", 78),
        }
        for c, (txt, w) in headings.items():
            self.src_tree.heading(c, text=txt)
            self.src_tree.column(c, width=w, anchor="w" if c in ("name", "event") else "center")

        self.src_tree.tag_configure("engine_primary", foreground="#f8fafc")
        self.src_tree.tag_configure("engine_aux", foreground="#94a3b8")
        self.src_tree.tag_configure("fx_row", foreground="#64748b")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.src_tree.yview)
        self.src_tree.configure(yscrollcommand=vsb.set)
        self.src_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.src_tree.bind("<Double-1>", lambda _e: self.cmd_play_selected_source())

    # ========================================================================
    # PESTAÑA 2: Eventos / Banco Donante AMS2 y Mapeo
    # ========================================================================
    def _build_tab_donor(self) -> None:
        # Selector principal de Modo de Porteo (Directo vs Plantilla Donante)
        mode_box = ttk.Frame(self.tab_donor, style="Card.TFrame", padding=(12, 8))
        mode_box.pack(fill="x", pady=(0, 8))

        ttk.Label(mode_box, text="Modo de Operación:", style="CardSection.TLabel").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(
            mode_box,
            text="⚡ Transmutación Directa FMOD (Recomendado - 100% Fiel AC, Sin Donante)",
            value="direct",
            variable=self.port_mode_var,
            command=self._on_port_mode_changed,
        ).pack(side="left", padx=(0, 16))
        ttk.Radiobutton(
            mode_box,
            text="🏛 Mapeo sobre Banco Donante de AMS2 (Plantilla)",
            value="donor",
            variable=self.port_mode_var,
            command=self._on_port_mode_changed,
        ).pack(side="left")

        top = ttk.Frame(self.tab_donor, style="Panel.TFrame")
        top.pack(fill="x", pady=(0, 8))

        self.btn_load_donor = ttk.Button(
            top,
            text="📂 Cargar .bank Donante de AMS2...",
            style="Primary.TButton",
            command=self.cmd_load_donor_bank,
        )
        self.btn_load_donor.pack(side="left", padx=(0, 8))

        self.btn_load_vanilla_donor = ttk.Button(
            top,
            text="🚗 Cargar Banco Vanilla de AMS2...",
            command=self.cmd_select_and_load_vanilla_donor,
        )
        self.btn_load_vanilla_donor.pack(side="left", padx=(0, 8))

        self.btn_auto_map = ttk.Button(
            top,
            text="⚡ Auto-Mapear desde Origen (AC -> AMS2)",
            style="Primary.TButton",
            command=self.cmd_auto_map,
        )
        self.btn_auto_map.pack(side="left", padx=(0, 8))

        self.btn_reset_mappings = ttk.Button(
            top,
            text="🔄 Restaurar Todos al Original",
            command=self.cmd_reset_mappings,
        )
        self.btn_reset_mappings.pack(side="left", padx=(0, 12))

        ttk.Label(top, textvariable=self.donor_info_var, style="Section.TLabel").pack(side="right")

        # Tarjeta resumen de eventos detectados
        self.events_banner_var = tk.StringVar(
            value="Eventos del banco donante: Carga un archivo .bank de motor de Automobilista 2 para inspeccionar sus eventos y slots."
        )
        ev_card = ttk.Frame(self.tab_donor, style="Card.TFrame", padding=(12, 8))
        ev_card.pack(fill="x", pady=(0, 8))
        ttk.Label(ev_card, textvariable=self.events_banner_var, style="CardMuted.TLabel").pack(anchor="w")

        # Tabla principal de slots del donante AMS2
        mid = ttk.Frame(self.tab_donor, style="Panel.TFrame")
        mid.pack(fill="both", expand=True, pady=(0, 8))

        cols = ("idx", "sb", "donor_name", "evts", "rpm_box", "mapped_to", "root_rpm", "gain", "status")
        self.donor_tree = ttk.Treeview(mid, columns=cols, show="headings", selectmode="browse")
        headings = {
            "idx": ("#", 42),
            "sb": ("SubB.", 60),
            "donor_name": ("Slot Original en Banco AMS2", 250),
            "evts": ("Evento(s)", 135),
            "rpm_box": ("Rango RPM AMS2", 115),
            "mapped_to": ("Muestra Asignada (Reemplazo)", 250),
            "root_rpm": ("RPM Raíz (Pitch)", 110),
            "gain": ("Ganancia", 80),
            "status": ("Estado / Coincidencia", 210),
        }
        for c, (txt, w) in headings.items():
            self.donor_tree.heading(c, text=txt)
            self.donor_tree.column(
                c,
                width=w,
                anchor="w" if c in ("donor_name", "mapped_to", "status") else "center",
            )

        self.donor_tree.tag_configure("replaced", foreground="#86efac")
        self.donor_tree.tag_configure("kept", foreground="#cbd5e1")
        self.donor_tree.tag_configure("silence", foreground="#94a3b8")

        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.donor_tree.yview)
        self.donor_tree.configure(yscrollcommand=vsb.set)
        self.donor_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.donor_tree.bind("<<TreeviewSelect>>", self._on_donor_slot_selected)
        self.donor_tree.bind("<Double-1>", lambda _e: self.cmd_preview_mapped_slot())

        # Panel inferior de edición detallada del slot seleccionado
        ed = ttk.Frame(self.tab_donor, style="Card.TFrame", padding=12)
        ed.pack(fill="x")

        row1 = ttk.Frame(ed, style="Card.TFrame")
        row1.pack(fill="x", pady=(0, 8))
        self.selected_slot_lbl = ttk.Label(
            row1,
            text="Selecciona un slot de la tabla superior para personalizar su reemplazo o pre-escuchar:",
            style="CardSection.TLabel",
        )
        self.selected_slot_lbl.pack(side="left")

        ttk.Button(
            row1,
            text="▶ Escuchar Reemplazo Mapeado",
            command=self.cmd_preview_mapped_slot,
        ).pack(side="right", padx=(6, 0))
        ttk.Button(
            row1,
            text="▶ Escuchar Original AMS2",
            command=self.cmd_preview_original_donor_slot,
        ).pack(side="right")

        row2 = ttk.Frame(ed, style="Card.TFrame")
        row2.pack(fill="x")

        ttk.Label(row2, text="Acción:", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.action_combo = ttk.Combobox(
            row2,
            textvariable=self.slot_action_var,
            state="readonly",
            width=26,
            values=("replace_source", "replace_wav", "keep_donor", "silence"),
        )
        self.action_combo.grid(row=0, column=1, sticky="w", padx=(0, 14))

        ttk.Label(row2, text="Muestra Origen:", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.src_pick_combo = ttk.Combobox(
            row2,
            textvariable=self.slot_src_combo_var,
            state="readonly",
            width=34,
        )
        self.src_pick_combo.grid(row=0, column=3, sticky="w", padx=(0, 10))
        self.src_pick_combo.bind("<<ComboboxSelected>>", self._on_src_combo_picked)

        ttk.Button(
            row2,
            text="📄 Elegir .wav suelto...",
            command=self.cmd_pick_custom_wav_for_slot,
        ).grid(row=0, column=4, sticky="w", padx=(0, 14))

        ttk.Label(row2, text="RPM Raíz (Auto-Pitch):", style="Card.TLabel").grid(row=0, column=5, sticky="w", padx=(0, 6))
        ttk.Entry(row2, textvariable=self.slot_rpm_var, width=10).grid(row=0, column=6, sticky="w", padx=(0, 12))

        ttk.Label(row2, text="Ganancia (dB):", style="Card.TLabel").grid(row=0, column=7, sticky="w", padx=(0, 6))
        ttk.Entry(row2, textvariable=self.slot_gain_var, width=7).grid(row=0, column=8, sticky="w", padx=(0, 12))

        ttk.Button(
            row2,
            text="✔ Aplicar Cambios al Slot",
            style="Primary.TButton",
            command=self.cmd_apply_slot_edit,
        ).grid(row=0, column=9, sticky="e")

    # ========================================================================
    # PESTAÑA 3: Bandas RPM, Throttle Cut Automático y Simulador en 5 Etapas
    # ========================================================================
    def _build_tab_rpm_sim(self) -> None:
        # --- Bloque superior: Rango del motor + Throttle Cut ---
        top_grid = ttk.Frame(self.tab_rpm_sim, style="Panel.TFrame")
        top_grid.pack(fill="x", side="top", pady=(0, 6))
        top_grid.columnconfigure(0, weight=5)
        top_grid.columnconfigure(1, weight=7)

        # Izquierda: Límites RPM y distribución anti-solapamiento
        eng_card = ttk.Frame(top_grid, style="Card.TFrame", padding=(10, 8))
        eng_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        ttk.Label(
            eng_card,
            text="1) Límites del Motor y Escalera RPM (Sin Solapamientos)",
            style="CardSection.TLabel",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        ttk.Label(eng_card, text="RPM Ralentí (Idle):", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Entry(eng_card, textvariable=self.sim_idle_rpm_var, width=8).grid(
            row=1, column=1, sticky="w", padx=(0, 14), pady=2
        )
        ttk.Label(eng_card, text="RPM Máx (Corte):", style="Card.TLabel").grid(
            row=1, column=2, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Entry(eng_card, textvariable=self.sim_max_rpm_var, width=8).grid(
            row=1, column=3, sticky="w", pady=2
        )

        btn_row_eng = ttk.Frame(eng_card, style="Card.TFrame")
        btn_row_eng.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Button(
            btn_row_eng,
            text="✨ Auto-Ajustar Escalera RPM (Sin Solapamientos)",
            style="Primary.TButton",
            command=self.cmd_auto_distribute_rpm,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            btn_row_eng,
            text="↺ Restaurar Originales",
            command=self.cmd_reset_rpm_bands,
        ).pack(side="left")

        # Derecha: Throttle Cut (Corte de inyección al subir de marcha)
        tc_card = ttk.Frame(top_grid, style="Card.TFrame", padding=(10, 8))
        tc_card.grid(row=0, column=1, sticky="nsew")
        tc_card.columnconfigure(1, weight=1)
        tc_card.columnconfigure(5, weight=1)

        tc_hdr = ttk.Frame(tc_card, style="Card.TFrame")
        tc_hdr.grid(row=0, column=0, columnspan=7, sticky="ew", pady=(0, 5))
        ttk.Label(
            tc_hdr,
            text="2) Throttle Cut / Efecto Cambio de Marcha (AC -> AMS2)",
            style="CardSection.TLabel",
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            tc_hdr,
            text="Inyectar Throttle Cut en el motor",
            variable=self.tcut_enabled_var,
        ).pack(side="left")
        ttk.Label(tc_hdr, text="Ganancia (dB):", style="Card.TLabel").pack(side="right", padx=(8, 4))
        ent_tc_gain = ttk.Entry(tc_hdr, textvariable=self.tcut_gain_var, width=6)
        ent_tc_gain.pack(side="right")

        ttk.Label(tc_card, text="Cut INT:", style="Card.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        self.tcut_int_combo = ttk.Combobox(
            tc_card,
            textvariable=self.tcut_int_combo_var,
            state="readonly",
            width=26,
        )
        self.tcut_int_combo.grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=2)
        self.tcut_int_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_tcut_combo_selected("int"))
        ttk.Button(tc_card, text="▶", width=3, command=lambda: self.cmd_preview_throttle_cut("int")).grid(
            row=1, column=2, padx=(0, 3), pady=2
        )
        ttk.Button(tc_card, text="📂", width=3, command=lambda: self.cmd_pick_tcut_wav("int")).grid(
            row=1, column=3, padx=(0, 10), pady=2
        )

        ttk.Label(tc_card, text="Cut EXT:", style="Card.TLabel").grid(row=1, column=4, sticky="w", padx=(0, 4), pady=2)
        self.tcut_ext_combo = ttk.Combobox(
            tc_card,
            textvariable=self.tcut_ext_combo_var,
            state="readonly",
            width=26,
        )
        self.tcut_ext_combo.grid(row=1, column=5, sticky="ew", padx=(0, 4), pady=2)
        self.tcut_ext_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_tcut_combo_selected("ext"))
        ttk.Button(tc_card, text="▶", width=3, command=lambda: self.cmd_preview_throttle_cut("ext")).grid(
            row=1, column=6, padx=(0, 3), pady=2
        )
        ttk.Button(tc_card, text="📂", width=3, command=lambda: self.cmd_pick_tcut_wav("ext")).grid(
            row=1, column=7, pady=2
        )

        # --- Barra de vista (Interior / Exterior) + Mapa Gráfico de Crossfades ---
        view_bar = ttk.Frame(self.tab_rpm_sim, style="Card.TFrame", padding=(10, 5))
        view_bar.pack(fill="x", side="top", pady=(0, 4))

        ttk.Label(view_bar, text="Vista de Bandas:", style="CardSection.TLabel").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(
            view_bar,
            text="🎧 Interior (EngineInt + Cabina)",
            variable=self.rpm_view_persp_var,
            value="int",
            command=self._refresh_rpm_tab,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            view_bar,
            text="🔊 Exterior (EngineExt)",
            variable=self.rpm_view_persp_var,
            value="ext",
            command=self._refresh_rpm_tab,
        ).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(
            view_bar,
            text="Mostrar pistas mudas / fill secundarias",
            variable=self.show_muted_bands_var,
            command=self._refresh_rpm_tab,
        ).pack(side="left")
        ttk.Button(
            view_bar,
            text="➕",
            width=3,
            command=self.cmd_add_rpm_band,
        ).pack(side="left", padx=(12, 2))
        ttk.Button(
            view_bar,
            text="➖",
            width=3,
            command=self.cmd_delete_rpm_band,
        ).pack(side="left", padx=(2, 6))
        self.btn_toggle_pops = ttk.Button(
            view_bar,
            text="🔇 Silenciar POPS / Petardeos",
            command=self.cmd_toggle_all_pops,
        )
        self.btn_toggle_pops.pack(side="left", padx=(6, 10))
        ttk.Label(view_bar, textvariable=self.rpm_overlap_status_var, style="CardMuted.TLabel").pack(side="right")

        self.rpm_canvas = tk.Canvas(
            self.tab_rpm_sim,
            height=136,
            bg=BG_INPUT,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            cursor="hand2",
        )
        self.rpm_canvas.pack(fill="x", side="top", pady=(0, 6))
        self.rpm_canvas.bind("<Configure>", lambda _e: self._draw_rpm_bands_canvas())
        self.rpm_canvas.bind("<Button-1>", self._on_rpm_canvas_click)

        # --- Empaquetar paneles inferiores (side="bottom") para garantizar visibilidad ---
        sim_card = ttk.Frame(self.tab_rpm_sim, style="Card.TFrame", padding=(10, 8))
        sim_card.pack(fill="x", side="bottom")

        sim_top = ttk.Frame(sim_card, style="Card.TFrame")
        sim_top.pack(fill="x", pady=(0, 4))

        ttk.Label(
            sim_top,
            text="4) Simulador de Pasada Completa en 5 Etapas:",
            style="CardSection.TLabel",
        ).pack(side="left", padx=(0, 10))

        self.btn_sim_int = ttk.Button(
            sim_top,
            text="▶ Probar Pasada INTERIOR (5 Etapas)",
            style="Primary.TButton",
            command=self.cmd_simulate_pass_int,
        )
        self.btn_sim_int.pack(side="left", padx=(0, 8))

        self.btn_sim_ext = ttk.Button(
            sim_top,
            text="▶ Probar Pasada EXTERIOR (5 Etapas)",
            style="Primary.TButton",
            command=self.cmd_simulate_pass_ext,
        )
        self.btn_sim_ext.pack(side="left", padx=(0, 8))

        ttk.Button(
            sim_top,
            text="⏹ Detener",
            command=self.stop_audio_preview,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            sim_top,
            text="💾 Guardar Pasada WAV...",
            command=self.cmd_export_simulated_wav,
        ).pack(side="left")

        ttk.Label(sim_top, textvariable=self.sim_stage_var, style="CardMuted.TLabel").pack(side="right")
        ttk.Label(sim_card, textvariable=self.sim_telemetry_var, style="Card.TLabel").pack(anchor="w")

        # Panel de edición numérica de la banda seleccionada (justo encima del simulador)
        band_ed = ttk.Frame(self.tab_rpm_sim, style="Card.TFrame", padding=(10, 7))
        band_ed.pack(fill="x", side="bottom", pady=(0, 6))

        bed_r1 = ttk.Frame(band_ed, style="Card.TFrame")
        bed_r1.pack(fill="x", pady=(0, 5))
        ttk.Button(
            bed_r1,
            text="▶ Escuchar Muestra de esta Banda",
            command=self.cmd_preview_selected_band_sample,
        ).pack(side="right")
        batch_f = ttk.Frame(bed_r1, style="Card.TFrame")
        batch_f.pack(side="right", padx=(8, 12))
        ttk.Button(
            batch_f,
            text="Modificar todos los RPM Raíz",
            command=self.cmd_batch_shift_all_root_rpm,
        ).pack(side="left", padx=(0, 4))
        ttk.Entry(batch_f, textvariable=self.batch_root_rpm_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Button(
            batch_f,
            text="Modificar todas las RPM",
            command=self.cmd_batch_shift_all_rpm,
        ).pack(side="left", padx=(0, 4))
        ttk.Entry(batch_f, textvariable=self.batch_rpm_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Button(
            batch_f,
            text="Amplificar todas las bandas",
            command=self.cmd_batch_gain_all_bands,
        ).pack(side="left", padx=(0, 4))
        ttk.Entry(batch_f, textvariable=self.batch_gain_var, width=5).pack(side="left")
        self.selected_band_lbl = ttk.Label(
            bed_r1,
            text="3) Selecciona una banda para ajustar sus RPM:",
            style="CardSection.TLabel",
        )
        self.selected_band_lbl.pack(side="left", fill="x", expand=True)

        # Fila de selección y sobreescritura de muestra asignada a la banda
        bed_r_sample = ttk.Frame(band_ed, style="Card.TFrame")
        bed_r_sample.pack(fill="x", pady=(0, 5))

        ttk.Label(bed_r_sample, text="Muestra asignada / Reemplazo:", style="Card.TLabel").pack(side="left", padx=(0, 6))
        self.band_source_combo = ttk.Combobox(
            bed_r_sample,
            textvariable=self.band_source_var,
            state="readonly",
            width=50,
        )
        self.band_source_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.band_source_combo.bind("<<ComboboxSelected>>", self._on_band_source_combo_selected)

        ttk.Button(
            bed_r_sample,
            text="📂 WAV...",
            width=8,
            command=self.cmd_pick_band_custom_wav,
        ).pack(side="left", padx=(0, 6))

        self.btn_overwrite_sample = ttk.Button(
            bed_r_sample,
            text="🔁 Sobreescribir Muestra",
            command=self.cmd_overwrite_selected_band_sample,
        )
        self.btn_overwrite_sample.pack(side="left", padx=(0, 6))

        self.btn_reactivate_band = ttk.Button(
            bed_r_sample,
            text="♻ Reactivar Banda",
            command=self.cmd_reactivate_selected_band,
        )
        self.btn_reactivate_band.pack(side="left")

        bed_r2 = ttk.Frame(band_ed, style="Card.TFrame")
        bed_r2.pack(fill="x")

        ed_fields = [
            ("RPM Mín:", self.band_rmin_var, 7),
            ("Fin Fade-In:", self.band_fi_var, 7),
            ("Inicio Fade-Out:", self.band_fo_var, 7),
            ("RPM Máx:", self.band_rmax_var, 7),
            ("RPM Raíz (Pitch):", self.band_root_rpm_var, 8),
            ("Ganancia (dB):", self.band_gain_var, 6),
        ]
        for col_idx, (lbl_t, tk_v, w_ch) in enumerate(ed_fields):
            ttk.Label(bed_r2, text=lbl_t, style="Card.TLabel").grid(
                row=0, column=col_idx * 2, sticky="w", padx=(0, 4)
            )
            ttk.Entry(bed_r2, textvariable=tk_v, width=w_ch).grid(
                row=0, column=col_idx * 2 + 1, sticky="w", padx=(0, 12)
            )

        ttk.Button(
            bed_r2,
            text="✔ Aplicar Cambios a la Banda",
            style="Primary.TButton",
            command=self.cmd_apply_band_edit,
        ).grid(row=0, column=len(ed_fields) * 2, sticky="e", padx=(6, 0))

        # --- Tabla central de Bandas RPM (expande en el espacio restante) ---
        b_tree_frame = ttk.Frame(self.tab_rpm_sim, style="Panel.TFrame")
        b_tree_frame.pack(fill="both", expand=True, pady=(0, 6))

        bcols = ("idx", "role", "sample", "rmin", "fi", "fo", "rmax", "root_rpm", "gain", "status")
        self.rpm_tree = ttk.Treeview(b_tree_frame, columns=bcols, show="headings", selectmode="browse", height=6)
        bheadings = {
            "idx": ("Banda", 62),
            "role": ("Capa / Rol", 96),
            "sample": ("Muestra Asignada Actualmente", 265),
            "rmin": ("RPM Mín", 84),
            "fi": ("Fin Fade-In", 92),
            "fo": ("Inicio Fade-Out", 104),
            "rmax": ("RPM Máx", 84),
            "root_rpm": ("RPM Raíz (Pitch)", 106),
            "gain": ("Ganancia", 78),
            "status": ("Estado / Solapamiento", 195),
        }
        for c, (txt, w) in bheadings.items():
            self.rpm_tree.heading(c, text=txt)
            self.rpm_tree.column(c, width=w, anchor="w" if c in ("sample", "status") else "center")

        self.rpm_tree.tag_configure("band_ok", foreground="#f1f5f9")
        self.rpm_tree.tag_configure("band_replaced", foreground="#86efac")
        self.rpm_tree.tag_configure("band_overlap", foreground="#fbbf24")
        self.rpm_tree.tag_configure("band_muted", foreground="#64748b")

        b_vsb = ttk.Scrollbar(b_tree_frame, orient="vertical", command=self.rpm_tree.yview)
        self.rpm_tree.configure(yscrollcommand=b_vsb.set)
        self.rpm_tree.pack(side="left", fill="both", expand=True)
        b_vsb.pack(side="right", fill="y")
        self.rpm_tree.bind("<<TreeviewSelect>>", self._on_rpm_band_selected)
        self.rpm_tree.bind("<Double-1>", lambda _e: self.cmd_preview_selected_band_sample())

    # ========================================================================
    # PESTAÑA 4: Vehículo Oficial Vanilla / Editor .csdbin integrado con OMVT
    # ========================================================================
    def _build_tab_csdbin(self) -> None:
        # Selector de Tipo de Vehículo: Vanilla vs Mod
        target_mode_card = ttk.Frame(self.tab_csdbin, style="Card.TFrame", padding=(12, 8))
        target_mode_card.pack(fill="x", pady=(0, 8))
        ttk.Label(target_mode_card, text="Tipo de Vehículo Destino en AMS2:", style="CardSection.TLabel").pack(
            side="left", padx=(0, 14)
        )
        ttk.Radiobutton(
            target_mode_card,
            text="🚗  Vehículo Oficial / Vanilla de AMS2 (174+ autos — Sin requerir archivo .csdbin)",
            value="vanilla",
            variable=self.target_mode_var,
            command=self._on_target_mode_changed,
        ).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(
            target_mode_card,
            text="📦  Vehículo Modificado de la Comunidad (con archivo .csdbin suelto)",
            value="mod",
            variable=self.target_mode_var,
            command=self._on_target_mode_changed,
        ).pack(side="left")

        # Contenedores para cada modo
        self.vanilla_view_frame = ttk.Frame(self.tab_csdbin, style="Panel.TFrame")
        self.mod_view_frame = ttk.Frame(self.tab_csdbin, style="Panel.TFrame")

        self._build_vanilla_view(self.vanilla_view_frame)
        self._build_mod_csdbin_view(self.mod_view_frame)

        self._on_target_mode_changed()

    def _build_vanilla_view(self, parent: ttk.Frame) -> None:
        # Card 1: Directorio AMS2
        path_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        path_card.pack(fill="x", pady=(0, 8))
        path_card.columnconfigure(1, weight=1)

        ttk.Label(path_card, text="Directorio Automobilista 2:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 10), pady=2
        )
        ttk.Entry(path_card, textvariable=self.vanilla_ams2_path_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 10), pady=2
        )
        ttk.Button(path_card, text="📁 Examinar...", command=self.cmd_browse_ams2_path).grid(
            row=0, column=2, padx=(0, 6), pady=2
        )
        ttk.Button(path_card, text="🔄 Re-escanear", command=self.cmd_rescan_vanilla_catalog).grid(
            row=0, column=3, pady=2
        )

        # Card 2: Selector y búsqueda de vehículos vanilla
        sel_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        sel_card.pack(fill="x", pady=(0, 8))
        sel_card.columnconfigure(3, weight=1)

        ttk.Label(sel_card, text="🔍 Filtrar Auto:", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        search_entry = ttk.Entry(sel_card, textvariable=self.vanilla_search_var, width=22)
        search_entry.grid(row=0, column=1, sticky="w", padx=(0, 16), pady=4)
        search_entry.bind("<KeyRelease>", lambda _e: self._filter_vanilla_vehicles())

        ttk.Label(sel_card, text="Vehículo Oficial / Vanilla:", style="Card.TLabel").grid(
            row=0, column=2, sticky="w", padx=(0, 8), pady=4
        )
        if not hasattr(self, "vanilla_combo"):
            self.vanilla_combo = ttk.Combobox(
                sel_card,
                textvariable=self.vanilla_selected_id_var,
                state="readonly",
            )
            self.vanilla_combo.grid(row=0, column=3, sticky="ew", padx=(0, 10), pady=4)
            self.vanilla_combo.bind("<<ComboboxSelected>>", self._on_vanilla_vehicle_selected)

        # Botones de acción directa
        btn_bar = ttk.Frame(sel_card, style="Card.TFrame")
        btn_bar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        ttk.Button(
            btn_bar,
            text="⚡ Porteo Directo 1:1 desde AC (Sin Donante — 100% Fiel)",
            style="Primary.TButton",
            command=self.cmd_set_vanilla_direct_mode,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_bar,
            text="🎯 Mapear sobre Slots (Modo Donante)",
            command=self.cmd_load_vanilla_as_donor,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_bar,
            text="🔄 Restaurar este Vehículo a Original de Steam",
            command=self.cmd_restore_selected_vanilla,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            btn_bar,
            text="🛡️ Restaurar TODOS los Vanilla",
            command=self.cmd_restore_all_vanilla,
        ).pack(side="left")

        # Card 3: Ficha técnica y eventos FMOD del vehículo
        det_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        det_card.pack(fill="both", expand=True, pady=(0, 8))
        det_card.columnconfigure(1, weight=1)

        ttk.Label(
            det_card,
            text="Ficha Técnica y Enlaces de Sonido FMOD del Vehículo",
            style="CardSection.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.vanilla_det_name_var = tk.StringVar(value="—")
        self.vanilla_det_brand_var = tk.StringVar(value="—")
        self.vanilla_det_path_var = tk.StringVar(value="—")
        self.vanilla_det_ai_var = tk.StringVar(value="—")
        self.vanilla_det_events_var = tk.StringVar(value="—")
        self.vanilla_det_integrity_var = tk.StringVar(value="—")
        self.vanilla_det_sha_var = tk.StringVar(value="—")

        det_fields = [
            ("Vehículo Oficial:", self.vanilla_det_name_var),
            ("Marca / Modelo:", self.vanilla_det_brand_var),
            ("Banco de Audio (.bank):", self.vanilla_det_path_var),
            ("Banco AI (_AI.bank):", self.vanilla_det_ai_var),
            ("Eventos y GUIDs Nativos:", self.vanilla_det_events_var),
            ("Estado de Integridad:", self.vanilla_det_integrity_var),
            ("Respaldo SHA256:", self.vanilla_det_sha_var),
        ]
        for idx_d, (lbl_t, var_d) in enumerate(det_fields, start=1):
            ttk.Label(det_card, text=lbl_t, style="Card.TLabel").grid(
                row=idx_d, column=0, sticky="nw", padx=(0, 10), pady=3
            )
            lbl_val = ttk.Label(det_card, textvariable=var_d, style="Card.TLabel", wraplength=760)
            lbl_val.grid(row=idx_d, column=1, sticky="w", pady=3)

        # Card 4: Seguridad y garantía de reversibilidad sin Steam
        sec_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        sec_card.pack(fill="x")
        ttk.Label(
            sec_card,
            text="🛡️ Seguridad y Reversibilidad 100% Garantizada Sin Verificación de Steam",
            style="CardSection.TLabel",
        ).pack(anchor="w", pady=(0, 4))
        sec_desc = (
            "Esta herramienta crea automáticamente una copia de seguridad binaria idéntica (archivo .orig y almacén "
            "seguro con hash SHA256) antes de modificar cualquier archivo oficial del juego.\n"
            "Puedes restaurar cualquier vehículo modificado a su estado original de Steam en cualquier momento con un clic, "
            "sin descargas, sin alterar archivos de sistema y sin requerir la comprobación o reparación de archivos en Steam."
        )
        ttk.Label(sec_card, text=sec_desc, style="CardMuted.TLabel", wraplength=1000).pack(anchor="w")

    def _build_mod_csdbin_view(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="Panel.TFrame")
        top.pack(fill="x", pady=(0, 10))

        ttk.Button(
            top,
            text="📂 Abrir .csdbin o .csd...",
            style="Primary.TButton",
            command=self.cmd_open_csdbin,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="🔗 Autocompletar Rutas event:/ Standalone",
            style="Primary.TButton",
            command=self.cmd_autofill_csdbin_standalone,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="💾 Guardar como .csdbin...",
            command=self.cmd_save_csdbin_as,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            top,
            text="📄 Exportar Texto .csd (OMVT)...",
            command=self.cmd_export_csd_text,
        ).pack(side="left", padx=(0, 12))

        ttk.Label(top, textvariable=self.csd_info_var, style="Section.TLabel").pack(side="right")

        sub_nb = ttk.Notebook(parent)
        sub_nb.pack(fill="both", expand=True)

        visual_tab = ttk.Frame(sub_nb, style="Panel.TFrame", padding=8)
        raw_tab = ttk.Frame(sub_nb, style="Panel.TFrame", padding=8)
        sub_nb.add(visual_tab, text="  🎛️ Editor Visual de Enlaces y Parámetros  ")
        sub_nb.add(raw_tab, text="  📝 Código Fuente .csd Completo (Formato OMVT)  ")

        # --- Sub-pestaña 3A: Formulario Visual con Scroll ---
        canvas = tk.Canvas(visual_tab, bg=BG_PANEL, highlightthickness=0)
        vsb = ttk.Scrollbar(visual_tab, orient="vertical", command=canvas.yview)
        scroll_body = ttk.Frame(canvas, style="Panel.TFrame")

        scroll_body.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        win_id = canvas.create_window((0, 0), window=scroll_body, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(win_id, width=e.width),
        )
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Cabecera de parámetros generales del vehículo
        meta_card = ttk.Frame(scroll_body, style="Card.TFrame", padding=12)
        meta_card.pack(fill="x", pady=(0, 10))
        ttk.Label(meta_card, text="Parámetros Generales y Posición 3D del Emisor", style="CardSection.TLabel").grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 8)
        )

        fields_hdr = [
            ("Clase Sonido (sound_class):", self.csd_class_var, 0, 0),
            ("Set de Mezcla (mix_set):", self.csd_mix_var, 0, 2),
            ("ID Hex (8 bytes):", self.csd_id_var, 0, 4),
            ("Volumen Motor Ext (off_0000):", self.csd_vol_ext_var, 1, 0),
            ("Volumen Motor Int (off_0004):", self.csd_vol_int_var, 1, 2),
            ("N° Marchas (off_00a0):", self.csd_gears_var, 1, 4),
            ("Posición Motor X (m):", self.csd_pos_x_var, 2, 0),
            ("Posición Motor Y / Altura (m):", self.csd_pos_y_var, 2, 2),
            ("Posición Motor Z / Long. (m):", self.csd_pos_z_var, 2, 4),
        ]
        for lbl_txt, var, r, c in fields_hdr:
            ttk.Label(meta_card, text=lbl_txt, style="Card.TLabel").grid(
                row=r + 1, column=c, sticky="w", padx=(0, 6), pady=4
            )
            ttk.Entry(meta_card, textvariable=var, width=26).grid(
                row=r + 1, column=c + 1, sticky="ew", padx=(0, 16), pady=4
            )
        for col_i in (1, 3, 5):
            meta_card.columnconfigure(col_i, weight=1)

        # Secciones de enlaces de bancos y eventos FMOD
        for group_title, slot_defs in SOUND_SLOT_GROUPS:
            gcard = ttk.Frame(scroll_body, style="Card.TFrame", padding=12)
            gcard.pack(fill="x", pady=(0, 10))
            ttk.Label(gcard, text=group_title, style="CardSection.TLabel").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
            )
            gcard.columnconfigure(1, weight=1)
            for idx_s, (slot_key, slot_label) in enumerate(slot_defs, start=1):
                svar = tk.StringVar(value="")
                self.csd_link_vars[slot_key] = svar
                ttk.Label(gcard, text=f"{slot_label} ({slot_key}):", style="Card.TLabel", width=38).grid(
                    row=idx_s, column=0, sticky="w", padx=(0, 10), pady=3
                )
                ttk.Entry(gcard, textvariable=svar).grid(row=idx_s, column=1, sticky="ew", pady=3)

        # --- Sub-pestaña 3B: Editor de texto crudo .csd (OMVT) ---
        raw_bar = ttk.Frame(raw_tab, style="Panel.TFrame")
        raw_bar.pack(fill="x", pady=(0, 6))
        ttk.Button(
            raw_bar,
            text="🔄 Actualizar Texto desde Formulario Visual",
            command=self._sync_visual_to_raw_csd_editor,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            raw_bar,
            text="✔ Aplicar Cambios del Texto .csd al Documento",
            style="Primary.TButton",
            command=self._apply_raw_csd_editor_to_doc,
        ).pack(side="left")

        self.raw_csd_text = tk.Text(
            raw_tab,
            bg=BG_INPUT,
            fg=FG_TEXT,
            insertbackground=FG_TEXT,
            font=("Consolas", 10),
            relief="flat",
            padx=10,
            pady=10,
            undo=True,
        )
        raw_vsb = ttk.Scrollbar(raw_tab, orient="vertical", command=self.raw_csd_text.yview)
        self.raw_csd_text.configure(yscrollcommand=raw_vsb.set)
        self.raw_csd_text.pack(side="left", fill="both", expand=True)
        raw_vsb.pack(side="right", fill="y")

    # ========================================================================
    # PESTAÑA 5: Compilación y Exportación / Inyección de Sonido para AMS2
    # ========================================================================
    def _build_tab_export(self) -> None:
        # Tarjeta 1A: Configuración Modo Vehículo Mod (.csdbin)
        self.mod_cfg_card = ttk.Frame(self.tab_export, style="Card.TFrame", padding=16)
        self.mod_cfg_card.columnconfigure(1, weight=1)

        ttk.Label(
            self.mod_cfg_card,
            text="Configuración del Paquete Final para Automobilista 2",
            style="CardSection.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(self.mod_cfg_card, text="Identificador Único del Banco Mod:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=5
        )
        ttk.Entry(self.mod_cfg_card, textvariable=self.mod_id_var).grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Label(
            self.mod_cfg_card,
            text="Ej: bmw_z4_gt3_ac (define la ruta Audio/AMS2/Vehicles/Mods/<id>/<id>_Engine.bank)",
            style="CardMuted.TLabel",
        ).grid(row=1, column=2, sticky="w", padx=(12, 0))

        ttk.Label(self.mod_cfg_card, text="Carpeta del Vehículo Mod (Vehicles/<carpeta>):", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=5
        )
        ttk.Entry(self.mod_cfg_card, textvariable=self.veh_folder_var).grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Label(
            self.mod_cfg_card,
            text="Ej: bmw_z4_gt3 (conserva el nombre original del archivo .csdbin cargado)",
            style="CardMuted.TLabel",
        ).grid(row=2, column=2, sticky="w", padx=(12, 0))

        ttk.Label(self.mod_cfg_card, text="Carpeta de Salida:", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=5
        )
        ttk.Entry(self.mod_cfg_card, textvariable=self.out_dir_var).grid(row=3, column=1, sticky="ew", pady=5)
        ttk.Button(self.mod_cfg_card, text="📁 Examinar...", command=self.cmd_pick_output_dir).grid(
            row=3, column=2, sticky="w", padx=(12, 0), pady=5
        )

        # Tarjeta 1B: Configuración Modo Vehículo Vanilla
        self.vanilla_cfg_card = ttk.Frame(self.tab_export, style="Card.TFrame", padding=16)
        self.vanilla_cfg_card.columnconfigure(1, weight=1)

        ttk.Label(
            self.vanilla_cfg_card,
            text="Inyección Directa en Vehículo Oficial / Base de Automobilista 2",
            style="CardSection.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(self.vanilla_cfg_card, text="Vehículo Oficial Destino:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=5
        )
        ttk.Label(self.vanilla_cfg_card, textvariable=self.vanilla_export_dest_var, style="Section.TLabel").grid(
            row=1, column=1, sticky="w", pady=5
        )

        # Metodología de Porteo para el Vehículo Oficial: Directo 1:1 vs Mapeo Donante
        vanilla_mode_f = ttk.LabelFrame(
            self.vanilla_cfg_card,
            text="Metodología de Sonido para este Vehículo:",
            padding=(10, 8),
        )
        vanilla_mode_f.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 8))

        ttk.Radiobutton(
            vanilla_mode_f,
            text="⚡ Transmutación Directa 1:1 (Recomendado — Sonido 100% idéntico a Assetto Corsa sin donante)",
            variable=self.port_mode_var,
            value="direct",
            command=self._on_port_mode_changed,
        ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            vanilla_mode_f,
            text="🎯 Mapeo sobre Banco Donante de AMS2 (Plantilla clásica de slots fijos de Automobilista 2)",
            variable=self.port_mode_var,
            value="donor",
            command=self._on_port_mode_changed,
        ).pack(anchor="w", pady=2)

        ttk.Checkbutton(
            self.vanilla_cfg_card,
            text="🛡️ Respaldo Preventivo Garantizado: Crear copia segura del archivo original (.orig y almacén con hash SHA256) antes de modificar nada",
            variable=self.vanilla_auto_backup_var,
            state="disabled",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Checkbutton(
            self.vanilla_cfg_card,
            text="👥 Sincronizar e inyectar también banco de audio de IA (_AI.bank) en un solo paso (Cubre rivales en pista y vistas de Repetición / TV Replay)",
            variable=self.vanilla_sync_ai_var,
            state="disabled",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Checkbutton(
            self.vanilla_cfg_card,
            text="🔇 Silenciar rateo continuo, petardeos y control de tracción (POPS / TC / Limiter) en Replay y vistas externas (Recomendado)",
            variable=self.vanilla_mute_pops_var,
            state="disabled",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=4)

        # Tarjeta 2: Opciones avanzadas de ingeniería FMOD (bloqueadas por defecto)
        self.opts_card = ttk.Frame(self.tab_export, style="Card.TFrame", padding=16)
        self.opts_card.pack(fill="x", pady=(0, 12))

        ttk.Label(self.opts_card, text="Opciones de Ingeniería FMOD", style="CardSection.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        ttk.Checkbutton(
            self.opts_card,
            text="🔇 Silenciar completamente la banda de POPS (Petardeos) en la exportación",
            variable=self.mute_all_pops_var,
            state="disabled",
        ).pack(anchor="w", pady=3)

        ttk.Checkbutton(
            self.opts_card,
            text="🔄 Invertir Cono Espacial 3D (Event Cone Angle): Corrige la orientación 180° entre Assetto Corsa y Automobilista 2",
            variable=self.invert_cone_var,
            state="disabled",
        ).pack(anchor="w", pady=3)

        ttk.Checkbutton(
            self.opts_card,
            text="Modo Standalone con Tabla STDT (Recomendado): Regenera GUIDs únicos e incrusta rutas 'event:/...' dentro del .bank sin tocar MasterBank.strings.bank",
            variable=self.standalone_var,
            state="disabled",
        ).pack(anchor="w", pady=3)

        ttk.Checkbutton(
            self.opts_card,
            text="Sincronizar Auto-Pitch RPM automáticamente en FMOD (INST): Ajusta la referencia de RPM de cada capa para que la afinación del motor sea exacta",
            variable=self.sync_pitch_var,
            state="disabled",
        ).pack(anchor="w", pady=3)

        ttk.Checkbutton(
            self.opts_card,
            text="Vincular y compilar automáticamente el archivo .csdbin cargado junto con el banco .bank",
            variable=self.auto_link_csd_var,
            state="disabled",
        ).pack(anchor="w", pady=3)

        ttk.Checkbutton(
            self.opts_card,
            text="Abrir la carpeta de salida en el Explorador de Windows al finalizar",
            variable=self.open_folder_after_var,
            state="disabled",
        ).pack(anchor="w", pady=3)

        codec_row = ttk.Frame(self.opts_card, style="Card.TFrame")
        codec_row.pack(fill="x", pady=(8, 0))
        ttk.Label(codec_row, text="Códec de Compresión FSB5:", style="Card.TLabel").pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            codec_row,
            text="PCM16 Sin Pérdida (Recomendado — Máxima calidad de sonido y compilación ultrarrápida)",
            variable=self.codec_choice_var,
            value="PCM16",
            state="disabled",
        ).pack(side="left", padx=(0, 18))
        ttk.Radiobutton(
            codec_row,
            text="FADPCM Comprimido Nativo FMOD (~3.6x menor tamaño de archivo .bank)",
            variable=self.codec_choice_var,
            value="FADPCM",
            state="disabled",
        ).pack(side="left")

        # Botón principal de compilación
        action_row = ttk.Frame(self.tab_export, style="Panel.TFrame")
        action_row.pack(fill="x", pady=(0, 10))
        self.btn_export = ttk.Button(
            action_row,
            text="🚀  COMPILAR Y EXPORTAR PAQUETE DE SONIDO PARA AUTOMOBILISTA 2",
            style="Success.TButton",
            command=self.cmd_run_export,
        )
        self.btn_export.pack(fill="x")

        # Registro de salida detallado
        log_frame = ttk.Frame(self.tab_export, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.export_log = tk.Text(
            log_frame,
            bg=BG_INPUT,
            fg="#e2e8f0",
            font=("Consolas", 10),
            relief="flat",
            padx=12,
            pady=10,
        )
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.export_log.yview)
        self.export_log.configure(yscrollcommand=log_vsb.set)
        self.export_log.pack(side="left", fill="both", expand=True)
        log_vsb.pack(side="right", fill="y")
        self._append_log(
            "Bienvenido a AMS2 Sound Porter Studio v1.3.\n"
            "Flujo recomendado:\n"
            "  1) Carga un .bank de Assetto Corsa (+ GUIDs.txt si es mod) en la Pestaña 1.\n"
            "  2) Carga un .bank donante de Automobilista 2 en la Pestaña 2 (Auto-Mapeo + Throttle Cut automático).\n"
            "  3) En la Pestaña 3 ajusta las Bandas RPM sin solapamientos y prueba la Pasada en 5 Etapas (INT y EXT).\n"
            "  4) Abre el archivo .csdbin de tu vehículo mod en la Pestaña 4.\n"
            "  5) Pulsa 'COMPILAR Y EXPORTAR PAQUETE' aquí para generar los archivos listos para AMS2.\n"
        )

    # ========================================================================
    # Utilidades de Audio Preview (winsound asíncrono vía archivo temporal A/B)
    # ========================================================================
    def play_wav_bytes_async(self, wav_bytes: bytes, label: str, keep_sim_anim: bool = False) -> None:
        if not keep_sim_anim:
            self._stop_sim_animation()
        if winsound is None:
            messagebox.showinfo("Audio no disponible", "El módulo winsound no está disponible en este sistema.")
            return
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
            self._preview_slot_toggle = 1 - self._preview_slot_toggle
            tmp_path = (
                Path(tempfile.gettempdir())
                / f"ams2_sound_porter_preview_{os.getpid()}_{self._preview_slot_toggle}.wav"
            )
            tmp_path.write_bytes(wav_bytes)
            if tmp_path not in self._temp_preview_files:
                self._temp_preview_files.append(tmp_path)
            winsound.PlaySound(
                str(tmp_path),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
            self.status_var.set(f"🔊 Reproduciendo: {label}")
        except Exception as exc:
            messagebox.showerror("Error al reproducir audio", str(exc))

    def _stop_sim_animation(self) -> None:
        if self._sim_anim_after_id is not None:
            try:
                self.after_cancel(self._sim_anim_after_id)
            except Exception:
                pass
            self._sim_anim_after_id = None
        if self._sim_current_rpm is not None:
            self._sim_current_rpm = None
            self.sim_stage_var.set("Etapa actual: En espera")
            self._draw_rpm_bands_canvas()

    def stop_audio_preview(self) -> None:
        self._stop_sim_animation()
        if winsound is not None:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        self.status_var.set("⏹ Reproducción detenida.")

    def _on_close(self) -> None:
        self.stop_audio_preview()
        for tp in self._temp_preview_files:
            try:
                tp.unlink(missing_ok=True)
            except Exception:
                pass
        self.destroy()

    def _append_log(self, msg: str) -> None:
        self.export_log.insert("end", msg + "\n")
        self.export_log.see("end")

    # ========================================================================
    # Acciones de Pestaña 1 (Origen AC / WAV)
    # ========================================================================
    def cmd_load_ac_bank(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Assetto Corsa FMOD sound bank (.bank)",
            filetypes=[("FMOD Studio Banks", "*.bank"), ("All files", "*.*")],
        )
        if not p:
            return
        self._load_source_library_from_bank(Path(p))

    def cmd_load_ac_guids_txt(self) -> None:
        has_bank = self.source_lib is not None and str(self.source_lib.source_path).lower().endswith(".bank")
        init_dir = str(self.source_lib.source_path.parent) if has_bank and self.source_lib else ""
        p = filedialog.askopenfilename(
            title="Select Assetto Corsa GUIDs.txt file",
            initialdir=init_dir or None,
            filetypes=[("FMOD GUIDs files", "*.txt"), ("All files", "*.*")],
        )
        if not p:
            return
        gp = Path(p)
        if has_bank and self.source_lib is not None:
            self._pending_guids_path = None
            self._load_source_library_from_bank(self.source_lib.source_path, guids_path=gp)
        else:
            self._pending_guids_path = gp
            self._refresh_ui_file_statuses()

    def _load_source_library_from_bank(self, path: Path, guids_path: Path | None = None) -> None:
        try:
            if guids_path is None and self._pending_guids_path is not None:
                guids_path = self._pending_guids_path
                self._pending_guids_path = None
            lib = SourceAudioLibrary.from_bank(path, guids_path=guids_path)
            if guids_path is not None and lib.matched_guids_count == 0:
                messagebox.showwarning(
                    "GUIDs.txt sin coincidencias",
                    f"El archivo '{guids_path.name}' no contiene ningún GUID que coincida con los eventos o buses de '{path.name}'.\n\n"
                    "Verifica que el GUIDs.txt corresponda exactamente a este banco de sonido.",
                )
            self.source_lib = lib
            self._selected_band_idx = None
            self._selected_donor_gidx = None
            self._refresh_ui_file_statuses()
            if lib.guids_path is not None and lib.matched_guids_count > 0:
                self.src_info_var.set(
                    f"✔ {lib.source_label} ({len(lib.items)} muestras | {lib.total_events_count} eventos)"
                )
                guids_msg = f" + '{lib.guids_path.name}' ({lib.matched_guids_count} GUIDs vinculados)"
            else:
                self.src_info_var.set(
                    f"✔ {path.name} — sin GUIDs.txt, inferencia FMOD ({len(lib.items)} muestras)"
                )
                guids_msg = " (sin GUIDs.txt externo; usando inferencia binaria FMOD)"

            self._refresh_source_tree()
            self._refresh_source_combobox_items()
            # Sugerir identificador limpio a partir del archivo
            stem_clean = "".join(c.lower() if c.isalnum() else "_" for c in path.stem).strip("_")
            if stem_clean:
                self.mod_id_var.set(stem_clean[:32])

            if self.donor_bank is None and lib.fmod_bank is not None:
                self.port_mode_var.set("direct")

            if self.is_direct_ac_mode and lib.fmod_bank is not None:
                self.rpm_bands = build_default_rpm_bands(lib.fmod_bank)
                self.mappings = [
                    SlotMappingConfig(sl.global_index, action="keep_donor")
                    for sl in lib.fmod_bank.slots
                ]
                tc = auto_detect_throttle_cut_config(lib.fmod_bank, lib)
                self.throttle_cut_cfg = tc
                self._sync_tcut_ui_from_cfg()
                est_idle, est_max = estimate_engine_rpm_bounds(lib.fmod_bank, lib)
                self.sim_idle_rpm_var.set(f"{est_idle:.0f}")
                self.sim_max_rpm_var.set(f"{est_max:.0f}")
                self._update_donor_tree_headings()
                self._refresh_donor_tree()
                self._refresh_rpm_tab()
                self.status_var.set(
                    f"Cargado banco AC '{path.name}'{guids_msg} con {len(lib.items)} muestras. "
                    "⚡ Modo Transmutación Directa activado (100% fiel sin donante)."
                )
            else:
                self._refresh_donor_tree()
                self.status_var.set(
                    f"Cargado origen '{path.name}'{guids_msg} con {len(lib.items)} muestras. "
                    "Ve al Paso 2 para auto-mapear al banco donante AMS2."
                )
                if self.donor_bank is not None:
                    self.cmd_auto_map()
        except Exception as exc:
            messagebox.showerror("Error al cargar .bank de origen", str(exc))

    def cmd_load_wav_folder(self) -> None:
        d = filedialog.askdirectory(title="Seleccionar carpeta con archivos .wav")
        if not d:
            return
        try:
            lib = SourceAudioLibrary.from_wav_folder(Path(d))
            self.source_lib = lib
            self._selected_band_idx = None
            self._selected_donor_gidx = None
            self.src_info_var.set(f"✔ {lib.source_label} ({len(lib.items)} WAVs)")
            self._refresh_source_tree()
            self._refresh_source_combobox_items()
            self.status_var.set(f"Cargada carpeta '{Path(d).name}' con {len(lib.items)} archivos .wav.")
            if self.donor_bank is not None:
                self.cmd_auto_map()
        except Exception as exc:
            messagebox.showerror("Error al cargar carpeta WAV", str(exc))

    def _refresh_source_tree(self) -> None:
        self.src_tree.delete(*self.src_tree.get_children())
        if not self.source_lib:
            return
        q = self.src_filter_var.get().strip().lower()
        for it in self.source_lib.items:
            ev_txt = f"{it.event_summary} [Aux/Far]" if it.is_aux_layer else it.event_summary
            if (
                q
                and q not in it.name.lower()
                and q not in ev_txt.lower()
                and q not in it.load_role.lower()
                and q not in it.perspective.lower()
            ):
                continue
            lp_txt = f"{it.loop[0]}..{it.loop[1]}" if it.loop else "One-shot"
            if it.load_role == "fx":
                row_tag = "fx_row"
            elif it.is_aux_layer:
                row_tag = "engine_aux"
            else:
                row_tag = "engine_primary"
            self.src_tree.insert(
                "",
                "end",
                iid=str(it.index),
                tags=(row_tag,),
                values=(
                    f"{it.index:02d}",
                    it.name,
                    ev_txt,
                    it.perspective.upper(),
                    it.load_role.upper(),
                    it.rpm_summary,
                    f"{it.sample_rate} Hz",
                    f"{it.channels} ch",
                    lp_txt,
                    f"{it.duration_sec:.2f} s",
                    it.codec_name,
                ),
            )

    def _refresh_source_combobox_items(self) -> None:
        if not self.source_lib:
            self.src_pick_combo["values"] = ()
            self.tcut_int_combo["values"] = ("[Ninguno]",)
            self.tcut_ext_combo["values"] = ("[Ninguno]",)
            return
        vals = [
            f"[{it.index:02d}] {it.name} ({it.event_summary} | {it.perspective.upper()}/{it.load_role.upper()}, {it.rpm_summary})"
            for it in self.source_lib.items
            if it.decodable
        ]
        self.src_pick_combo["values"] = vals
        tcut_vals = ["[Ninguno]"] + vals
        self.tcut_int_combo["values"] = tcut_vals
        self.tcut_ext_combo["values"] = tcut_vals
        self.throttle_cut_cfg = auto_detect_throttle_cut_config(self.source_lib)
        self._sync_tcut_ui_from_cfg()

    def cmd_play_selected_source(self) -> None:
        if not self.source_lib:
            return
        sel = self.src_tree.selection()
        if not sel:
            messagebox.showinfo("Seleccionar muestra", "Selecciona una muestra de la lista para escucharla.")
            return
        idx = int(sel[0])
        item = self.source_lib.items[idx]
        if not item.decodable:
            messagebox.showwarning("Códec solo lectura", f"La muestra '{item.name}' usa {item.codec_name}.")
            return
        try:
            wav_b = write_wav_bytes(item.load_audio())
            self.play_wav_bytes_async(wav_b, f"Origen [{item.index:02d}] {item.name}")
        except Exception as exc:
            messagebox.showerror("Error al decodificar muestra", str(exc))

    def cmd_extract_all_source_wavs(self) -> None:
        if not self.source_lib:
            messagebox.showinfo("Sin origen", "Carga primero un archivo .bank en el Paso 1.")
            return
        out_d = filedialog.askdirectory(title="Seleccionar carpeta donde extraer los archivos .wav")
        if not out_d:
            return
        written = 0
        for it in self.source_lib.items:
            if not it.decodable:
                continue
            buf = it.load_audio()
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", it.name).strip() or f"sample_{it.index:02d}"
            dst = Path(out_d) / f"{it.index:02d}_{safe_name}.wav"
            dst.write_bytes(write_wav_bytes(buf))
            written += 1
        messagebox.showinfo("Extracción completada", f"Se extrajeron {written} archivos .wav en:\n{out_d}")

    # ========================================================================
    # Acciones de Pestaña 2 (Banco Donante AMS2 y Mapeo)
    # ========================================================================
    def _on_port_mode_changed(self) -> None:
        mode = self.port_mode_var.get()
        if mode == "direct":
            if self.source_lib and self.source_lib.fmod_bank:
                ac_bank = self.source_lib.fmod_bank
                self.rpm_bands = build_default_rpm_bands(ac_bank)
                self.mappings = [
                    SlotMappingConfig(sl.global_index, action="keep_donor")
                    for sl in ac_bank.slots
                ]
                est_idle, est_max = estimate_engine_rpm_bounds(ac_bank, self.source_lib)
                self.sim_idle_rpm_var.set(f"{est_idle:.0f}")
                self.sim_max_rpm_var.set(f"{est_max:.0f}")
                tc = auto_detect_throttle_cut_config(ac_bank, self.source_lib)
                self.throttle_cut_cfg = tc
                self._sync_tcut_ui_from_cfg()
                self._update_donor_tree_headings()
                self._refresh_donor_tree()
                self._refresh_rpm_tab()
                self.status_var.set("Modo Transmutación Directa activo (100% fiel al .bank de Assetto Corsa sin donante).")
            else:
                messagebox.showinfo(
                    "Cargar .bank de AC",
                    "Para usar el Modo Directo, carga primero un archivo .bank de Assetto Corsa en el Paso 1.",
                )
                self.port_mode_var.set("donor")
        else:
            self._update_donor_tree_headings()
            if self.donor_bank:
                self.rpm_bands = build_default_rpm_bands(self.donor_bank)
                self.mappings = (
                    auto_map_samples(self.donor_bank, self.source_lib)
                    if self.source_lib
                    else [SlotMappingConfig(sl.global_index) for sl in self.donor_bank.slots]
                )
                est_idle, est_max = estimate_engine_rpm_bounds(self.donor_bank, self.source_lib)
                self.sim_idle_rpm_var.set(f"{est_idle:.0f}")
                self.sim_max_rpm_var.set(f"{est_max:.0f}")
                self._refresh_donor_tree()
                self._refresh_rpm_tab()
                self.status_var.set("Modo Donante seleccionado. Utilizando la estructura fija de Automobilista 2.")
            else:
                self._refresh_donor_tree()
                self._refresh_rpm_tab()
                self.status_var.set("Modo Donante seleccionado. Carga un archivo .bank donante de AMS2 en el Paso 2.")

        if hasattr(self, "btn_export") and hasattr(self, "target_mode_var") and self.target_mode_var.get() == "vanilla":
            veh_name = (
                self.vanilla_selected_vehicle.display_name
                if self.vanilla_selected_vehicle
                else "Vehículo Vanilla"
            )
            if self.is_direct_ac_mode:
                btn_text = f"⚡ INYECTAR DIRECTO 1:1 EN '{veh_name.upper()}' (SIN DONANTE - 100% FIEL)"
            else:
                btn_text = f"🎯 INYECTAR MAPEO EN '{veh_name.upper()}' (MODO DONANTE)"
            self.btn_export.config(text=btn_text)

    def _update_donor_tree_headings(self) -> None:
        if self.is_direct_ac_mode:
            headings = {
                "idx": ("#", 42),
                "sb": ("Rol FMOD", 110),
                "donor_name": ("Evento Nativo AC", 230),
                "evts": ("Bus Maestro AMS2", 160),
                "rpm_box": ("Ruta STDT Registrada", 240),
                "mapped_to": ("Parámetros (rpms -> rpm)", 180),
                "root_rpm": ("Muestras", 90),
                "gain": ("Ganancia", 75),
                "status": ("Fidelidad", 150),
            }
        else:
            headings = {
                "idx": ("#", 42),
                "sb": ("SubB.", 60),
                "donor_name": ("Slot Original en Banco AMS2", 250),
                "evts": ("Evento(s)", 135),
                "rpm_box": ("Rango RPM AMS2", 115),
                "mapped_to": ("Muestra Asignada (Reemplazo)", 250),
                "root_rpm": ("RPM Raíz (Pitch)", 110),
                "gain": ("Ganancia", 80),
                "status": ("Estado / Coincidencia", 210),
            }
        for c, (txt, w) in headings.items():
            self.donor_tree.heading(c, text=txt)
            self.donor_tree.column(
                c,
                width=w,
                anchor="w" if c in ("donor_name", "mapped_to", "status", "rpm_box", "evts") else "center",
            )

    def cmd_load_donor_bank(self) -> None:
        p = filedialog.askopenfilename(
            title="Seleccionar banco .bank donante de Automobilista 2",
            filetypes=[("Bancos FMOD Studio", "*.bank"), ("Todos los archivos", "*.*")],
        )
        if not p:
            return
        self._load_donor_bank_file(Path(p))

    def _load_donor_bank_file(self, path: Path) -> None:
        try:
            bank = FmodBank(path)
            self.donor_bank = bank
            self._selected_band_idx = None
            self._selected_donor_gidx = None
            self.port_mode_var.set("donor")
            self._update_donor_tree_headings()
            self.mappings = [SlotMappingConfig(sl.global_index) for sl in bank.slots]
            self.rpm_bands = build_default_rpm_bands(bank)
            est_idle, est_max = estimate_engine_rpm_bounds(bank, self.source_lib)
            self.sim_idle_rpm_var.set(f"{est_idle:.0f}")
            self.sim_max_rpm_var.set(f"{est_max:.0f}")
            self.donor_info_var.set(
                f"✔ {path.name} ({len(bank.events)} eventos, {len(bank.slots)} slots en {len(bank.subbanks)} sub-bancos)"
            )
            ev_summary = "  |  ".join(f"{ev.role}: {ev.guid_str[:13]}...}}" for ev in bank.events)
            self.events_banner_var.set(f"Eventos FMOD detectados:  {ev_summary}")
            self._refresh_donor_tree()
            self._refresh_rpm_tab()
            if self.source_lib is not None:
                self.cmd_auto_map()
            else:
                self.status_var.set(
                    f"Banco donante AMS2 '{path.name}' cargado ({len(bank.slots)} slots). Pulsa 'Auto-Mapear' tras cargar un origen."
                )
        except Exception as exc:
            messagebox.showerror("Error al cargar banco donante AMS2", str(exc))

    def cmd_auto_map(self) -> None:
        if self.is_direct_ac_mode:
            messagebox.showinfo(
                "Modo Directo Activo",
                "En Modo Transmutación Directa no es necesario mapear slots: todos los eventos nativos de Assetto Corsa se transfieren al 100% hacia Automobilista 2.",
            )
            return
        if not self.donor_bank:
            messagebox.showinfo("Falta banco donante", "Carga primero un banco .bank donante de AMS2.")
            return
        if not self.source_lib:
            messagebox.showinfo("Falta audio de origen", "Carga primero un .bank de Assetto Corsa o carpeta WAV en el Paso 1.")
            return
        self.mappings = auto_map_samples(self.donor_bank, self.source_lib)
        est_idle, est_max = estimate_engine_rpm_bounds(self.donor_bank, self.source_lib)
        self.sim_idle_rpm_var.set(f"{est_idle:.0f}")
        self.sim_max_rpm_var.set(f"{est_max:.0f}")
        self.throttle_cut_cfg = auto_detect_throttle_cut_config(self.donor_bank, self.source_lib)
        self._sync_tcut_ui_from_cfg()
        self.rpm_bands = auto_distribute_rpm_bands(
            build_default_rpm_bands(self.donor_bank),
            self.donor_bank,
            self.mappings,
            idle_rpm=est_idle,
            max_rpm=est_max,
        )
        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        rep_n = sum(1 for m in self.mappings if m.action != "keep_donor")
        tcut_msg = " + Throttle Cut detectado automáticamente" if self.throttle_cut_cfg.enabled else ""
        self.status_var.set(
            f"⚡ Auto-Mapeo completado: {rep_n}/{len(self.mappings)} slots asignados, escalera RPM ajustada ({est_idle:.0f}–{est_max:.0f} RPM){tcut_msg}."
        )

    def cmd_reset_mappings(self) -> None:
        if self.is_direct_ac_mode and self.source_lib and self.source_lib.fmod_bank:
            ac_bank = self.source_lib.fmod_bank
            self.mappings = [SlotMappingConfig(sl.global_index, action="keep_donor") for sl in ac_bank.slots]
            self.rpm_bands = build_default_rpm_bands(ac_bank)
            self._refresh_donor_tree()
            self._refresh_rpm_tab()
            self.status_var.set("Restaurados todos los eventos y muestras nativas de Assetto Corsa.")
            return
        if not self.donor_bank:
            return
        self.mappings = [SlotMappingConfig(sl.global_index) for sl in self.donor_bank.slots]
        self.rpm_bands = build_default_rpm_bands(self.donor_bank)
        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self.status_var.set("Mapeo y bandas RPM restaurados a los valores originales del donante.")

    def _refresh_donor_tree(self) -> None:
        prev_sel = self.donor_tree.selection()
        self.donor_tree.delete(*self.donor_tree.get_children())

        if self.is_direct_ac_mode:
            ac_bank = self.source_lib.fmod_bank  # type: ignore[union-attr]
            safe_mod = "".join(
                c if (c.isalnum() or c in "_-") else "_" for c in self.mod_id_var.get().strip()
            ) or "custom_mod"
            self.events_banner_var.set(
                f"⚡ MODO TRANSMUTACIÓN DIRECTA ACTIVO: {len(ac_bank.events)} eventos FMOD nativos y {len(ac_bank.rpm_bands)} bandas RPM de Assetto Corsa. "
                "Se transfieren al 100% sin donante: 'rpms' -> 'rpm' in-place y enrutamiento a buses Master de Automobilista 2."
            )
            self.donor_info_var.set(
                f"✔ Modo Directo ({len(ac_bank.events)} eventos AC listos para AMS2)"
            )
            for idx, ev in enumerate(ac_bank.events):
                p_low = ev.path.lower()
                is_ext = (
                    ev.role in ("EngineExt", "GearshiftExt", "BackfireExt")
                    or "_ext" in p_low
                    or "ext_" in p_low
                    or p_low.endswith("ext")
                    or any(k in p_low for k in ("exterior", "rear", "external", "chase", "skid_ext", "tractioncontrol_ext"))
                )
                tgt_bus = "Master Exterior (3D)" if is_ext else "Master Cockpit (2D/3D)"
                last_part = ev.path.split("/")[-1] if ev.path else ev.role
                tgt_path = f"event:/AMS2/Vehicles/Mods/{safe_mod}/{last_part}"
                params_str = ", ".join(p[0] for p in ev.parameters) if ev.parameters else "—"
                params_str = params_str.replace("rpms", "rpm (parcheado)")

                self.donor_tree.insert(
                    "",
                    "end",
                    iid=str(idx),
                    tags=("replaced",),
                    values=(
                        f"{idx:02d}",
                        f"{ev.role}",
                        ev.path or ev.role,
                        tgt_bus,
                        tgt_path,
                        params_str,
                        f"{len(ev.linked_samples)} samples",
                        "0.0 dB",
                        "✔ 100% Fiel AC",
                    ),
                )
            if prev_sel and self.donor_tree.exists(prev_sel[0]):
                self.donor_tree.selection_set(prev_sel[0])
            elif ac_bank.events:
                self.donor_tree.selection_set("0")
                self._on_donor_slot_selected()
            return

        if not self.donor_bank:
            return

        for sl, cfg in zip(self.donor_bank.slots, self.mappings):
            if cfg.action == "replace_source" and cfg.source_index is not None and self.source_lib:
                mapped_lbl = f"AC [{cfg.source_index:02d}] {self.source_lib.items[cfg.source_index].name}"
                tag = "replaced"
            elif cfg.action == "replace_wav" and cfg.custom_wav_path is not None:
                mapped_lbl = f"WAV: {cfg.custom_wav_path.name}"
                tag = "replaced"
            elif cfg.action == "silence":
                mapped_lbl = "[Silencio / Mudo]"
                tag = "silence"
            else:
                mapped_lbl = "[Original del Donante]"
                tag = "kept"

            rpm_val = (
                f"{cfg.override_root_rpm:.0f} RPM"
                if cfg.override_root_rpm
                else (f"{sl.autopitch_ref_rpm:.0f} RPM" if sl.autopitch_ref_rpm else "—")
            )
            ev_txt = ", ".join(sl.event_roles) if sl.event_roles else "—"

            self.donor_tree.insert(
                "",
                "end",
                iid=str(sl.global_index),
                tags=(tag,),
                values=(
                    f"{sl.global_index:02d}",
                    f"SND[{sl.subbank_index}] {sl.codec_name}",
                    sl.name,
                    ev_txt,
                    sl.rpm_summary,
                    mapped_lbl,
                    rpm_val,
                    f"{cfg.gain_db:+.1f} dB",
                    cfg.match_reason,
                ),
            )

        if prev_sel and self.donor_tree.exists(prev_sel[0]):
            self.donor_tree.selection_set(prev_sel[0])

    def _on_donor_slot_selected(self, _event=None) -> None:
        sel = self.donor_tree.selection()
        if not sel:
            return
        gidx = int(sel[0])

        if self.is_direct_ac_mode:
            ac_bank = self.source_lib.fmod_bank  # type: ignore[union-attr]
            if not (0 <= gidx < len(ac_bank.events)):
                return
            ev = ac_bank.events[gidx]
            self._selected_donor_gidx = gidx
            p_low = ev.path.lower()
            is_ext = (
                ev.role in ("EngineExt", "GearshiftExt", "BackfireExt")
                or "_ext" in p_low
                or "ext_" in p_low
                or p_low.endswith("ext")
                or any(k in p_low for k in ("exterior", "rear", "external", "chase", "skid_ext", "tractioncontrol_ext"))
            )
            tgt_bus = "Master Exterior (3D)" if is_ext else "Master Cockpit (2D/3D)"
            self.selected_slot_lbl.config(
                text=f"Evento #{gidx:02d} [{ev.role}]: {ev.path or ev.role} ({len(ev.linked_samples)} muestras | Bus: {tgt_bus})"
            )
            return

        if not self.donor_bank:
            return
        if not (0 <= gidx < len(self.donor_bank.slots)):
            return
        self._selected_donor_gidx = gidx
        sl = self.donor_bank.slots[gidx]
        cfg = self.mappings[gidx]

        loop_info = f"Loop {sl.loop[0]}..{sl.loop[1]}" if sl.loop else ("Loop INST" if sl.inst_infinite_loop else "One-shot")
        self.selected_slot_lbl.config(
            text=f"Slot [{gidx:02d}] {sl.name}  ({sl.sample_rate} Hz, {sl.channels} ch, {loop_info}, Donante Pitch={sl.autopitch_ref_rpm or 'Fijo'})"
        )
        self.slot_action_var.set(cfg.action)
        if cfg.source_index is not None and self.source_lib:
            for val in self.src_pick_combo["values"]:
                if val.startswith(f"[{cfg.source_index:02d}]"):
                    self.slot_src_combo_var.set(val)
                    break
        else:
            self.slot_src_combo_var.set("")

        self.slot_wav_path_var.set(str(cfg.custom_wav_path) if cfg.custom_wav_path else "")
        self.slot_rpm_var.set(f"{cfg.override_root_rpm:.1f}" if cfg.override_root_rpm else "")
        self.slot_gain_var.set(f"{cfg.gain_db:.1f}")

    def _on_src_combo_picked(self, _event=None) -> None:
        val = self.slot_src_combo_var.get()
        if not val.startswith("[") or not self.source_lib:
            return
        idx = int(val[1:3])
        self.slot_action_var.set("replace_source")
        it = self.source_lib.items[idx]
        if it.effective_rpm:
            self.slot_rpm_var.set(f"{it.effective_rpm:.1f}")

    def cmd_pick_custom_wav_for_slot(self) -> None:
        if self._selected_donor_gidx is None:
            messagebox.showinfo("Seleccionar slot", "Selecciona primero un slot en la tabla superior.")
            return
        p = filedialog.askopenfilename(
            title="Elegir archivo .wav de reemplazo para este slot",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not p:
            return
        self.slot_wav_path_var.set(p)
        self.slot_action_var.set("replace_wav")
        self.cmd_apply_slot_edit()

    def cmd_apply_slot_edit(self) -> None:
        if self._selected_donor_gidx is None or not self.active_engine_bank:
            return
        bank = self.active_engine_bank
        gidx = self._selected_donor_gidx
        cfg = self.mappings[gidx]
        act = self.slot_action_var.get()
        cfg.action = act

        if act == "replace_source":
            cval = self.slot_src_combo_var.get()
            if not cval.startswith("["):
                messagebox.showwarning("Elegir muestra", "Selecciona una muestra del desplegable 'Muestra Origen'.")
                return
            sidx = int(cval[1:3])
            cfg.source_index = sidx
            cfg.match_reason = "Asignado manualmente (Origen)"
        elif act == "replace_wav":
            wp = self.slot_wav_path_var.get().strip()
            if not wp or not Path(wp).is_file():
                messagebox.showwarning("Archivo WAV", "Elige un archivo .wav válido.")
                return
            cfg.custom_wav_path = Path(wp)
            cfg.match_reason = f"WAV externo ({Path(wp).name})"
        elif act == "silence":
            cfg.match_reason = "Silenciado manualmente"
        else:
            cfg.match_reason = "Original del donante"

        rpm_txt = self.slot_rpm_var.get().strip()
        if rpm_txt:
            try:
                cfg.override_root_rpm = float(rpm_txt)
            except ValueError:
                messagebox.showwarning("RPM inválido", "Ingresa un número válido para RPM Raíz (ej. 7500).")
                return
        else:
            cfg.override_root_rpm = None

        try:
            cfg.gain_db = float(self.slot_gain_var.get().strip() or "0.0")
        except ValueError:
            cfg.gain_db = 0.0

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        slot_label = bank.slots[gidx].name if 0 <= gidx < len(bank.slots) else f"Slot #{gidx:02d}"
        self.status_var.set(f"✔ Actualizado slot [{gidx:02d}] {slot_label}.")

    def cmd_preview_original_donor_slot(self) -> None:
        if self._selected_donor_gidx is None or not self.active_engine_bank:
            return
        bank = self.active_engine_bank
        gidx = self._selected_donor_gidx
        if self.is_direct_ac_mode:
            if not (0 <= gidx < len(bank.events)):
                return
            ev = bank.events[gidx]
            if not ev.linked_samples:
                messagebox.showinfo("Sin muestras", f"El evento '{ev.role}' no tiene muestras directas asociadas.")
                return
            sname = ev.linked_samples[0]
            matched = [s for s in bank.slots if s.name == sname]
            if not matched:
                return
            try:
                wav_b = bank.extract_slot_wav_bytes(matched[0])
                self.play_wav_bytes_async(wav_b, f"Evento AC [{ev.role}]: {sname}")
            except Exception as exc:
                messagebox.showerror("Error al reproducir muestra del evento", str(exc))
            return

        if not (0 <= gidx < len(bank.slots)):
            return
        sl = bank.slots[gidx]
        if not sl.can_decode_audio:
            messagebox.showinfo(
                "Muestra Vorbis del donante",
                f"El slot original '{sl.name}' está comprimido en {sl.codec_name} dentro del donante.\n\n"
                "No requiere decodificación para reemplazarlo: cualquier muestra AC o WAV que asignes "
                "se escuchará y compilará perfectamente en PCM16 o FADPCM.",
            )
            return
        try:
            wav_b = bank.extract_slot_wav_bytes(sl)
            self.play_wav_bytes_async(wav_b, f"Original AMS2 [{sl.global_index:02d}] {sl.name}")
        except Exception as exc:
            messagebox.showerror("Error al reproducir slot donante", str(exc))

    def cmd_preview_mapped_slot(self) -> None:
        if self._selected_donor_gidx is None or not self.active_engine_bank:
            return
        if self.is_direct_ac_mode:
            self.cmd_preview_original_donor_slot()
            return
        gidx = self._selected_donor_gidx
        if not self.donor_bank or not (0 <= gidx < len(self.donor_bank.slots)):
            return
        sl = self.donor_bank.slots[gidx]
        cfg = self.mappings[gidx]
        try:
            if cfg.action == "replace_source" and cfg.source_index is not None and self.source_lib:
                raw_b = self.source_lib.items[cfg.source_index].load_audio()
                conformed = conform_audio(raw_b, 48000, sl.channels, cfg.gain_db, False)
                self.play_wav_bytes_async(write_wav_bytes(conformed), f"Mapeado -> {sl.name}")
            elif cfg.action == "replace_wav" and cfg.custom_wav_path:
                raw_b = read_wav_file(cfg.custom_wav_path)
                conformed = conform_audio(raw_b, 48000, sl.channels, cfg.gain_db, False)
                self.play_wav_bytes_async(write_wav_bytes(conformed), f"WAV -> {sl.name}")
            elif cfg.action == "keep_donor":
                self.cmd_preview_original_donor_slot()
            else:
                self.status_var.set("El slot está configurado como Silencio (Null).")
        except Exception as exc:
            messagebox.showerror("Error en vista previa", str(exc))

    # ========================================================================
    # Acciones de Pestaña 3 (Bandas RPM, Throttle Cut y Simulador en 5 Etapas)
    # ========================================================================
    def _sync_tcut_ui_from_cfg(self) -> None:
        cfg = self.throttle_cut_cfg
        self.tcut_enabled_var.set(bool(cfg.enabled))
        self.tcut_gain_var.set(f"{cfg.gain_db:.1f}")

        def _fmt_choice(sidx: int | None, wav_p: Path | None, wav_var: tk.StringVar, combo_var: tk.StringVar, combo: ttk.Combobox) -> None:
            if wav_p is not None and Path(wav_p).is_file():
                wav_var.set(str(wav_p))
                combo_var.set(f"WAV: {Path(wav_p).name}")
                return
            wav_var.set("")
            if sidx is not None and self.source_lib:
                prefix = f"[{sidx:02d}]"
                for v in combo["values"]:
                    if v.startswith(prefix):
                        combo_var.set(v)
                        return
            combo_var.set("[Ninguno]")

        _fmt_choice(cfg.int_source_index, cfg.int_custom_wav, self.tcut_int_wav_var, self.tcut_int_combo_var, self.tcut_int_combo)
        _fmt_choice(cfg.ext_source_index, cfg.ext_custom_wav, self.tcut_ext_wav_var, self.tcut_ext_combo_var, self.tcut_ext_combo)

    def _collect_throttle_cut_config(self) -> ThrottleCutConfig:
        enabled = bool(self.tcut_enabled_var.get())
        try:
            gain_db = float(self.tcut_gain_var.get().strip() or "1.5")
        except ValueError:
            gain_db = 1.5

        def _parse_side(combo_txt: str, wav_txt: str) -> tuple[int | None, Path | None]:
            wt = wav_txt.strip()
            if combo_txt.startswith("WAV:") and wt and Path(wt).is_file():
                return None, Path(wt)
            if combo_txt.startswith("[") and "]" in combo_txt and combo_txt != "[Ninguno]":
                try:
                    return int(combo_txt[1 : combo_txt.index("]")]), None
                except ValueError:
                    pass
            return None, None

        int_sidx, int_wav = _parse_side(self.tcut_int_combo_var.get(), self.tcut_int_wav_var.get())
        ext_sidx, ext_wav = _parse_side(self.tcut_ext_combo_var.get(), self.tcut_ext_wav_var.get())

        cfg = ThrottleCutConfig(
            enabled=enabled,
            int_source_index=int_sidx,
            ext_source_index=ext_sidx,
            int_custom_wav=int_wav,
            ext_custom_wav=ext_wav,
            gain_db=gain_db,
        )
        self.throttle_cut_cfg = cfg
        return cfg

    def _on_tcut_combo_selected(self, persp: str) -> None:
        if persp == "int":
            val = self.tcut_int_combo_var.get()
            self.tcut_int_wav_var.set("")
        else:
            val = self.tcut_ext_combo_var.get()
            self.tcut_ext_wav_var.set("")
        if val != "[Ninguno]":
            self.tcut_enabled_var.set(True)
        self._collect_throttle_cut_config()

    def cmd_pick_tcut_wav(self, persp: str) -> None:
        p = filedialog.askopenfilename(
            title=f"Seleccionar archivo WAV para Throttle Cut ({persp.upper()})",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not p:
            return
        wp = Path(p)
        if persp == "int":
            self.tcut_int_wav_var.set(str(wp))
            self.tcut_int_combo_var.set(f"WAV: {wp.name}")
        else:
            self.tcut_ext_wav_var.set(str(wp))
            self.tcut_ext_combo_var.set(f"WAV: {wp.name}")
        self.tcut_enabled_var.set(True)
        self._collect_throttle_cut_config()

    def cmd_preview_throttle_cut(self, persp: str) -> None:
        cfg = self._collect_throttle_cut_config()
        c_wav = cfg.int_custom_wav if persp == "int" else cfg.ext_custom_wav
        s_idx = cfg.int_source_index if persp == "int" else cfg.ext_source_index
        try:
            if c_wav is not None and c_wav.is_file():
                buf = conform_audio(read_wav_file(c_wav), 48000, 2, cfg.gain_db, False)
                self.play_wav_bytes_async(write_wav_bytes(buf), f"Throttle Cut {persp.upper()} ({c_wav.name})")
                return
            if s_idx is not None and self.source_lib and 0 <= s_idx < len(self.source_lib.items):
                it = self.source_lib.items[s_idx]
                buf = conform_audio(it.load_audio(), 48000, 2, cfg.gain_db, False)
                self.play_wav_bytes_async(write_wav_bytes(buf), f"Throttle Cut {persp.upper()} [{s_idx:02d}] {it.name}")
                return
            messagebox.showinfo(
                "Sin muestra de Throttle Cut",
                f"No hay una muestra asignada en Cut {persp.upper()}. Selecciona una del desplegable o carga un archivo .wav.",
            )
        except Exception as exc:
            messagebox.showerror("Error al reproducir Throttle Cut", str(exc))

    def _parse_sim_rpm_bounds(self) -> tuple[float, float]:
        try:
            idle_v = float(self.sim_idle_rpm_var.get().strip() or "950")
        except ValueError:
            idle_v = 950.0
        try:
            max_v = float(self.sim_max_rpm_var.get().strip() or "7800")
        except ValueError:
            max_v = 7800.0
        idle_v = max(550.0, min(3200.0, idle_v))
        max_v = max(idle_v + 2000.0, min(22000.0, max_v))
        return idle_v, max_v

    def cmd_auto_distribute_rpm(self) -> None:
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            messagebox.showinfo("Cargar banco", "Carga primero un banco .bank de AC o donante de AMS2.")
            return
        idle_v, max_v = self._parse_sim_rpm_bounds()
        self.sim_idle_rpm_var.set(f"{idle_v:.0f}")
        self.sim_max_rpm_var.set(f"{max_v:.0f}")

        # Si hay una banda seleccionada, aplicar exclusivamente a ella
        if self._selected_band_idx is not None and 0 <= self._selected_band_idx < len(self.rpm_bands):
            b = self.rpm_bands[self._selected_band_idx]
            auto_distribute_single_rpm_band(
                b,
                self.rpm_bands,
                bank,
                self.mappings,
                library=self.source_lib,
                idle_rpm=idle_v,
                max_rpm=max_v,
            )
            self._on_rpm_band_selected()
            self._refresh_rpm_tab()
            self.status_var.set(
                f"✨ Banda #{self._selected_band_idx:02d} [{b.event_role} / {b.load_role.upper()}] auto-ajustada individualmente sin solapamientos."
            )
            return

        self.rpm_bands = auto_distribute_rpm_bands(
            self.rpm_bands,
            bank,
            self.mappings,
            library=self.source_lib,
            idle_rpm=idle_v,
            max_rpm=max_v,
        )
        self._refresh_rpm_tab()
        self.status_var.set(
            f"✨ Escalera RPM redistribuida automáticamente sin solapamientos ({idle_v:.0f} – {max_v:.0f} RPM)."
        )

    def cmd_reset_rpm_bands(self) -> None:
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            return

        persp = self.rpm_view_persp_var.get()
        idle_v, max_v = self._parse_sim_rpm_bounds()

        if self.is_direct_ac_mode and self.source_lib and self.source_lib.fmod_bank:
            # En Modo Transmutación Directa: restaurar fielmente las curvas nativas de AC para esta perspectiva
            native_bands = build_default_rpm_bands(self.source_lib.fmod_bank)
            kept_bands = [b for b in self.rpm_bands if b.perspective != persp]
            persp_native = [b for b in native_bands if b.perspective == persp]
            self.rpm_bands = sorted(
                kept_bands + persp_native,
                key=lambda b: (0 if b.perspective == "int" else 1, b.box_index),
            )
            self._selected_band_idx = None
            self._refresh_donor_tree()
            self._refresh_rpm_tab()
            self.status_var.set(
                f"↺ Sección {'INTERIOR' if persp == 'int' else 'EXTERIOR'} restaurada a las curvas nativas originales de Assetto Corsa ({len(persp_native)} bandas)."
            )
            return

        # Si tenemos librería de Assetto Corsa y banco donante, recrear fielmente TODAS las bandas de esta sección
        if self.donor_bank is not None and self.source_lib is not None and self.source_lib.items:
            new_bands, new_maps, added_cnt, muted_cnt = sync_faithful_section_from_source(
                self.donor_bank,
                self.source_lib,
                self.mappings,
                self.rpm_bands,
                perspective=persp,
                idle_rpm=idle_v,
                max_rpm=max_v,
            )
            self.rpm_bands = new_bands
            self.mappings = new_maps

            # Sincronizar Throttle Cut de AC si aplica
            eff_tcut = auto_detect_throttle_cut_config(self.donor_bank, self.source_lib)
            if eff_tcut.enabled:
                self.throttle_cut_cfg = eff_tcut
                self._sync_tcut_ui_from_cfg()

            self._selected_band_idx = None
            self._refresh_donor_tree()
            self._refresh_rpm_tab()
            self.status_var.set(
                f"↺ Sección {'INTERIOR' if persp == 'int' else 'EXTERIOR'} recreada fielmente desde AC: {added_cnt} bandas sincronizadas de forma nativa ({muted_cnt} secundarias silenciadas)."
            )
            return

        # Si no hay librería AC cargada, restaurar todas las bandas de esta sección a los rangos del donante
        for b in self.rpm_bands:
            if b.perspective == persp:
                b.rpm_min = b.orig_rpm_min
                b.fade_in_end = b.orig_fade_in_end
                b.fade_out_start = b.orig_fade_out_start
                b.rpm_max = b.orig_rpm_max
                b.is_muted_bus = False
                for gidx in b.donor_slot_indices:
                    if 0 <= gidx < len(self.mappings):
                        self.mappings[gidx].gain_db = 0.0
                        self.mappings[gidx].override_root_rpm = None

        self._selected_band_idx = None
        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self.status_var.set(
            f"↺ Todas las bandas de la sección {'INTERIOR' if persp == 'int' else 'EXTERIOR'} restauradas a los rangos originales del donante."
        )

    def _collect_batch_target_bands(self) -> tuple[str, list[int], list[int]] | None:
        """Devuelve `(perspectiva, indices_visibles_activos, indices_con_gemelas_aux)` para edición por lotes."""
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            messagebox.showinfo("Cargar banco", "Carga primero un banco .bank de AC o donante de AMS2.")
            return None

        persp = self.rpm_view_persp_var.get()
        vis_indices = self._visible_band_indices()
        active_vis = [
            idx
            for idx in vis_indices
            if not (self._band_is_silenced(self.rpm_bands[idx]) or self.rpm_bands[idx].is_muted_bus)
        ]
        if not active_vis:
            messagebox.showinfo("Sin bandas activas", f"No hay bandas activas en la vista {persp.upper()}.")
            return None

        target_indices = list(active_vis)
        target_set = set(target_indices)
        if persp == "int":
            eng_sigs = {
                (
                    self.rpm_bands[i].load_role,
                    int(round(self.rpm_bands[i].orig_rpm_min)),
                    int(round(self.rpm_bands[i].orig_rpm_max)),
                )
                for i in active_vis
                if self.rpm_bands[i].event_role == "EngineInt"
            }
            for oidx, other in enumerate(self.rpm_bands):
                if oidx not in target_set and other.event_role == "AuxInt" and not (
                    self._band_is_silenced(other) or other.is_muted_bus
                ):
                    sig = (other.load_role, int(round(other.orig_rpm_min)), int(round(other.orig_rpm_max)))
                    if sig in eng_sigs:
                        target_indices.append(oidx)
                        target_set.add(oidx)
        return persp, active_vis, target_indices

    def cmd_batch_shift_all_rpm(self) -> None:
        """Suma o resta un desplazamiento en RPM a Mín, Fade-In, Fade-Out y Máx de todas las bandas activas."""
        collected = self._collect_batch_target_bands()
        if not collected:
            return
        persp, active_vis, target_indices = collected

        txt = self.batch_rpm_var.get().strip()
        try:
            delta_rpm = float(txt)
        except ValueError:
            messagebox.showwarning("Valor RPM inválido", "Ingresa un número válido en RPM (ej: +450 o -300).")
            return

        for idx in target_indices:
            b = self.rpm_bands[idx]
            orig_span = b.rpm_max - b.rpm_min
            new_rmin = max(0.0, round(b.rpm_min + delta_rpm, 1))
            new_fi = max(new_rmin, round(b.fade_in_end + delta_rpm, 1))
            new_fo = max(new_fi, round(b.fade_out_start + delta_rpm, 1))
            new_rmax = max(new_fo, round(b.rpm_max + delta_rpm, 1))
            if orig_span > 40.0 and new_rmax < new_rmin + 40.0:
                new_rmax = new_rmin + 40.0
            b.rpm_min = new_rmin
            b.fade_in_end = new_fi
            b.fade_out_start = new_fo
            b.rpm_max = new_rmax

        if self._selected_band_idx is not None and self._selected_band_idx in target_indices:
            b_sel = self.rpm_bands[self._selected_band_idx]
            self.band_rmin_var.set(f"{b_sel.rpm_min:.0f}")
            self.band_fi_var.set(f"{b_sel.fade_in_end:.0f}")
            self.band_fo_var.set(f"{b_sel.fade_out_start:.0f}")
            self.band_rmax_var.set(f"{b_sel.rpm_max:.0f}")

        self._refresh_rpm_tab()
        sign = "+" if delta_rpm >= 0 else ""
        self.status_var.set(
            f"✔ Rangos RPM ajustados ({sign}{delta_rpm:.0f} RPM) para las {len(active_vis)} bandas activas de la vista {persp.upper()}."
        )

    def cmd_batch_shift_all_root_rpm(self) -> None:
        """Suma o resta un desplazamiento al RPM Raíz (Pitch) de todas las bandas activas de la vista."""
        collected = self._collect_batch_target_bands()
        if not collected:
            return
        persp, active_vis, target_indices = collected
        bank = self.active_engine_bank
        assert bank is not None

        txt = self.batch_root_rpm_var.get().strip()
        try:
            delta_root = float(txt)
        except ValueError:
            messagebox.showwarning("RPM Raíz inválido", "Ingresa un número válido en RPM (ej: +300 o -250).")
            return

        modified_slots: set[int] = set()
        for idx in target_indices:
            b = self.rpm_bands[idx]
            for gidx in b.donor_slot_indices:
                if gidx in modified_slots or not (0 <= gidx < len(self.mappings)) or not (0 <= gidx < len(bank.slots)):
                    continue
                cfg = self.mappings[gidx]
                sl = bank.slots[gidx]
                cur_root = cfg.override_root_rpm if (cfg.override_root_rpm and cfg.override_root_rpm > 0.0) else sl.autopitch_ref_rpm
                if cur_root is None or cur_root <= 0.0:
                    continue
                cfg.override_root_rpm = max(200.0, round(cur_root + delta_root, 1))
                modified_slots.add(gidx)

        if self._selected_band_idx is not None and self._selected_band_idx in target_indices:
            b_sel = self.rpm_bands[self._selected_band_idx]
            if b_sel.donor_slot_indices:
                g0 = b_sel.donor_slot_indices[0]
                if 0 <= g0 < len(self.mappings) and 0 <= g0 < len(bank.slots):
                    rr = (self.mappings[g0].override_root_rpm or None) or bank.slots[g0].autopitch_ref_rpm
                    self.band_root_rpm_var.set(f"{rr:.0f}" if rr else "")

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        sign = "+" if delta_root >= 0 else ""
        self.status_var.set(
            f"✔ RPM Raíz (Pitch) ajustado ({sign}{delta_root:.0f} RPM) en {len(modified_slots)} pistas de las {len(active_vis)} bandas de la vista {persp.upper()}."
        )

    def cmd_batch_gain_all_bands(self) -> None:
        """Aumenta o disminuye la ganancia (dB) de todas las bandas activas de la vista seleccionada."""
        collected = self._collect_batch_target_bands()
        if not collected:
            return
        persp, active_vis, target_indices = collected

        txt = self.batch_gain_var.get().strip()
        try:
            delta_db = float(txt)
        except ValueError:
            messagebox.showwarning("Ganancia inválida", "Ingresa un número válido en dB (ej: +2.0 o -1.5).")
            return

        modified_slots: set[int] = set()
        for idx in target_indices:
            b = self.rpm_bands[idx]
            for gidx in b.donor_slot_indices:
                if 0 <= gidx < len(self.mappings) and gidx not in modified_slots:
                    self.mappings[gidx].gain_db = round(self.mappings[gidx].gain_db + delta_db, 1)
                    modified_slots.add(gidx)

        if self._selected_band_idx is not None and self._selected_band_idx in target_indices:
            b_sel = self.rpm_bands[self._selected_band_idx]
            if b_sel.donor_slot_indices:
                g0 = b_sel.donor_slot_indices[0]
                if 0 <= g0 < len(self.mappings):
                    self.band_gain_var.set(f"{self.mappings[g0].gain_db:.1f}")

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        sign = "+" if delta_db >= 0 else ""
        self.status_var.set(
            f"✔ Ganancia ajustada ({sign}{delta_db:.1f} dB) para las {len(active_vis)} bandas activas de la vista {persp.upper()}."
        )

    def cmd_delete_rpm_band(self) -> None:
        """Elimina/silencia la banda de RPM seleccionada actualmente."""
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            return
        if self._selected_band_idx is None or not (0 <= self._selected_band_idx < len(self.rpm_bands)):
            messagebox.showinfo("Seleccionar banda", "Selecciona primero una banda en la tabla o en el gráfico para eliminarla.")
            return

        idx = self._selected_band_idx
        b = self.rpm_bands[idx]
        resp = messagebox.askyesno(
            "Eliminar Banda",
            f"¿Deseas eliminar la Banda #{idx:02d} [{b.event_role} / {b.load_role.upper()}]?\n\n"
            "La banda será silenciada y su pista física quedará libre para ser reasignada o reactivada.",
        )
        if not resp:
            return

        b.is_muted_bus = True
        b.rpm_min = 0.0
        b.fade_in_end = 0.0
        b.fade_out_start = 0.0
        b.rpm_max = 0.0
        for gidx in b.donor_slot_indices:
            if 0 <= gidx < len(self.mappings):
                self.mappings[gidx].action = "silence"
                self.mappings[gidx].match_reason = "Banda eliminada por el usuario"

        # Sincronizar capa gemela AuxInt si aplica
        if b.event_role == "EngineInt":
            orig_sig = (b.load_role, int(round(b.orig_rpm_min)), int(round(b.orig_rpm_max)))
            for other in self.rpm_bands:
                if other.event_role == "AuxInt" and (other.load_role, int(round(other.orig_rpm_min)), int(round(other.orig_rpm_max))) == orig_sig:
                    other.is_muted_bus = True
                    other.rpm_min = 0.0
                    other.fade_in_end = 0.0
                    other.fade_out_start = 0.0
                    other.rpm_max = 0.0
                    for gidx in other.donor_slot_indices:
                        if 0 <= gidx < len(self.mappings):
                            self.mappings[gidx].action = "silence"
                            self.mappings[gidx].match_reason = "Banda eliminada por el usuario"

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self._on_rpm_band_selected()
        self.status_var.set(f"✔ Banda #{idx:02d} silenciada. Su pista física queda disponible para reasignar.")

    def cmd_toggle_all_pops(self) -> None:
        """Silencia o reactiva todas las bandas de POPS / petardeos con un solo clic."""
        if not self.rpm_bands:
            return
        pops_bands = [b for b in self.rpm_bands if b.load_role == "pops"]
        if not pops_bands:
            messagebox.showinfo("POPS", "No hay bandas clasificadas con rol POPS en el banco activo.")
            return

        any_active = any(not b.is_muted_bus and b.rpm_max > b.rpm_min for b in pops_bands)
        if any_active:
            # Silenciar todas las bandas POPS
            for b in pops_bands:
                b.is_muted_bus = True
                b.rpm_min = 0.0
                b.fade_in_end = 0.0
                b.fade_out_start = 0.0
                b.rpm_max = 0.0
                for gidx in b.donor_slot_indices:
                    if 0 <= gidx < len(self.mappings):
                        self.mappings[gidx].action = "silence"
                        self.mappings[gidx].gain_db = -96.0
                        self.mappings[gidx].match_reason = "POPS silenciado por el usuario"
            if hasattr(self, "btn_toggle_pops"):
                self.btn_toggle_pops.config(text="🔊 Reactivar POPS")
            self.status_var.set(f"🔇 Todas las bandas de POPS ({len(pops_bands)}) han sido silenciadas exitosamente.")
        else:
            # Reactivar todas las bandas POPS
            for b in pops_bands:
                b.is_muted_bus = False
                if b.orig_rpm_max > b.orig_rpm_min:
                    b.rpm_min = b.orig_rpm_min
                    b.fade_in_end = b.orig_fade_in_end
                    b.fade_out_start = b.orig_fade_out_start
                    b.rpm_max = b.orig_rpm_max
                else:
                    b.rpm_min = 5500.0
                    b.fade_in_end = 6000.0
                    b.fade_out_start = 8500.0
                    b.rpm_max = 8500.0
                for gidx in b.donor_slot_indices:
                    if 0 <= gidx < len(self.mappings):
                        self.mappings[gidx].action = "keep_donor"
                        self.mappings[gidx].gain_db = 0.0
            if hasattr(self, "btn_toggle_pops"):
                self.btn_toggle_pops.config(text="🔇 Silenciar POPS / Petardeos")
            self.status_var.set(f"🔊 Bandas de POPS ({len(pops_bands)}) reactivadas.")

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self._draw_rpm_bands_canvas()

    def cmd_pick_band_custom_wav(self) -> None:
        """Abre un explorador de archivos para asignar un WAV local directamente a la banda."""
        p = filedialog.askopenfilename(
            title="Elegir archivo WAV de muestra para esta banda",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if p:
            self.band_custom_wav_var.set(p)
            self.band_source_var.set(f"WAV: {Path(p).name}")

    def _on_band_source_combo_selected(self, _event=None) -> None:
        """Al cambiar la selección en el desplegable de muestras de la banda, sugiere RPM Raíz si aplica."""
        val = self.band_source_var.get().strip()
        if val.startswith("[") and self.source_lib:
            try:
                s_idx = int(val[1:3])
                it = self.source_lib.items[s_idx]
                if it.effective_rpm and it.effective_rpm > 100.0:
                    self.band_root_rpm_var.set(f"{it.effective_rpm:.0f}")
            except Exception:
                pass

    def cmd_reactivate_selected_band(self) -> None:
        """Reactiva y desmutea la banda seleccionada actualmente."""
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands or self._selected_band_idx is None:
            messagebox.showinfo("Seleccionar banda", "Selecciona primero una banda silenciada para reactivarla.")
            return
        idx = self._selected_band_idx
        if not (0 <= idx < len(self.rpm_bands)):
            return
        b = self.rpm_bands[idx]
        b.is_muted_bus = False
        if b.rpm_max <= b.rpm_min + 40.0:
            b.rpm_min = b.orig_rpm_min if b.orig_rpm_min > 0 else 2500.0
            b.fade_in_end = b.orig_fade_in_end if b.orig_fade_in_end > 0 else 3000.0
            b.fade_out_start = b.orig_fade_out_start if b.orig_fade_out_start > 0 else 5500.0
            b.rpm_max = b.orig_rpm_max if b.orig_rpm_max > 0 else 6000.0
            self.band_rmin_var.set(f"{b.rpm_min:.0f}")
            self.band_fi_var.set(f"{b.fade_in_end:.0f}")
            self.band_fo_var.set(f"{b.fade_out_start:.0f}")
            self.band_rmax_var.set(f"{b.rpm_max:.0f}")

        for gidx in b.donor_slot_indices:
            if 0 <= gidx < len(self.mappings):
                m = self.mappings[gidx]
                if m.action == "silence":
                    m.action = "replace_source" if (m.source_index is not None and self.source_lib) else "passthrough"
                    m.match_reason = "Reactivada por el usuario"

        if b.event_role == "EngineInt":
            orig_sig = (b.load_role, int(round(b.orig_rpm_min)), int(round(b.orig_rpm_max)))
            for other in self.rpm_bands:
                if other.event_role == "AuxInt" and (other.load_role, int(round(other.orig_rpm_min)), int(round(other.orig_rpm_max))) == orig_sig:
                    other.is_muted_bus = False
                    other.rpm_min = b.rpm_min
                    other.fade_in_end = b.fade_in_end
                    other.fade_out_start = b.fade_out_start
                    other.rpm_max = b.rpm_max
                    for gidx in other.donor_slot_indices:
                        if 0 <= gidx < len(self.mappings):
                            self.mappings[gidx].action = "replace_source" if (self.mappings[gidx].source_index is not None and self.source_lib) else "passthrough"
                            self.mappings[gidx].match_reason = "Reactivada por el usuario"

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self._on_rpm_band_selected()
        self.status_var.set(f"✔ Banda #{idx:02d} reactivada y desmuteada con éxito.")

    def cmd_overwrite_selected_band_sample(self) -> None:
        """Sobreescribe directamente la muestra asignada a la banda seleccionada con la muestra elegida de AC o WAV."""
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands or self._selected_band_idx is None:
            messagebox.showinfo("Seleccionar banda", "Selecciona primero una banda en la tabla o en el gráfico para sobreescribir su muestra.")
            return

        idx = self._selected_band_idx
        if not (0 <= idx < len(self.rpm_bands)):
            return
        b = self.rpm_bands[idx]

        val = self.band_source_var.get().strip()
        c_wav = self.band_custom_wav_var.get().strip()
        c_wav_p = Path(c_wav) if (c_wav and Path(c_wav).is_file()) else None

        src_idx: int | None = None
        sample_name = ""

        if c_wav_p:
            sample_name = c_wav_p.name
        elif val.startswith("[") and self.source_lib:
            try:
                src_idx = int(val[1:3])
                sample_name = self.source_lib.items[src_idx].name
            except Exception:
                pass

        if not sample_name:
            messagebox.showwarning(
                "Seleccionar muestra",
                "Elige una muestra de Assetto Corsa en la lista desplegable o selecciona un archivo WAV externo.",
            )
            return

        b.is_muted_bus = False
        src_item = self.source_lib.items[src_idx] if (src_idx is not None and self.source_lib and 0 <= src_idx < len(self.source_lib.items)) else None

        if b.rpm_max <= b.rpm_min + 40.0:
            if src_item and src_item.effective_rpm and src_item.effective_rpm > 100.0:
                er = float(src_item.effective_rpm)
                b.rpm_min = max(0.0, er - 800.0)
                b.fade_in_end = max(0.0, er - 300.0)
                b.fade_out_start = er + 300.0
                b.rpm_max = er + 800.0
            else:
                b.rpm_min = b.orig_rpm_min if b.orig_rpm_min > 0 else 2500.0
                b.fade_in_end = b.orig_fade_in_end if b.orig_fade_in_end > 0 else 3000.0
                b.fade_out_start = b.orig_fade_out_start if b.orig_fade_out_start > 0 else 5500.0
                b.rpm_max = b.orig_rpm_max if b.orig_rpm_max > 0 else 6000.0
            self.band_rmin_var.set(f"{b.rpm_min:.0f}")
            self.band_fi_var.set(f"{b.fade_in_end:.0f}")
            self.band_fo_var.set(f"{b.fade_out_start:.0f}")
            self.band_rmax_var.set(f"{b.rpm_max:.0f}")

        if src_item and src_item.load_role in ("on", "off", "idle", "pops"):
            b.load_role = src_item.load_role

        try:
            curr_gain = float(self.band_gain_var.get().strip() or "0.0")
        except ValueError:
            curr_gain = 0.0

        root_val: float | None = None
        if src_item and src_item.effective_rpm and src_item.effective_rpm > 100.0:
            root_val = float(src_item.effective_rpm)
            self.band_root_rpm_var.set(f"{root_val:.0f}")
        else:
            root_txt = self.band_root_rpm_var.get().strip()
            if root_txt:
                try:
                    root_val = float(root_txt)
                except ValueError:
                    pass

        b.donor_label = f"AC: {sample_name}"

        for gidx in b.donor_slot_indices:
            while len(self.mappings) <= gidx:
                self.mappings.append(SlotMappingConfig(len(self.mappings)))
            m = self.mappings[gidx]
            if c_wav_p:
                m.action = "replace_wav"
                m.custom_wav_path = c_wav_p
                m.match_reason = f"Sobreescrito por WAV: {sample_name}"
            elif src_idx is not None:
                m.action = "replace_source"
                m.source_index = src_idx
                m.match_reason = f"Sobreescrito por usuario: {sample_name}"
            m.gain_db = curr_gain
            m.override_root_rpm = root_val

        # Sincronizar capa gemela AuxInt si aplica
        if b.event_role == "EngineInt":
            orig_sig = (b.load_role, int(round(b.orig_rpm_min)), int(round(b.orig_rpm_max)))
            for other in self.rpm_bands:
                if other.event_role == "AuxInt" and (other.load_role, int(round(other.orig_rpm_min)), int(round(other.orig_rpm_max))) == orig_sig:
                    other.is_muted_bus = False
                    other.rpm_min = b.rpm_min
                    other.fade_in_end = b.fade_in_end
                    other.fade_out_start = b.fade_out_start
                    other.rpm_max = b.rpm_max
                    other.load_role = b.load_role
                    for gidx in other.donor_slot_indices:
                        if 0 <= gidx < len(self.mappings):
                            self.mappings[gidx].action = "replace_wav" if c_wav_p else "replace_source"
                            self.mappings[gidx].source_index = src_idx
                            self.mappings[gidx].custom_wav_path = c_wav_p
                            self.mappings[gidx].gain_db = curr_gain
                            self.mappings[gidx].override_root_rpm = root_val

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self._on_rpm_band_selected()
        self.status_var.set(f"✔ Banda #{idx:02d} sobreescrita exitosamente con '{sample_name}'.")

    def cmd_add_rpm_band(self) -> None:
        """Abre un diálogo interactivo para añadir/reasignar una banda al sonido (INT o EXT) desde Assetto Corsa."""
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            messagebox.showinfo("Cargar banco de audio", "Carga primero un banco .bank de AC o donante de AMS2.")
            return

        persp = self.rpm_view_persp_var.get()
        target_ev_role = "EngineInt" if persp == "int" else "EngineExt"
        idle_v, max_v = self._parse_sim_rpm_bounds()

        # Crear diálogo modal
        dlg = tk.Toplevel(self)
        dlg.title(f"Add Sound Band ({'Interior' if persp == 'int' else 'Exterior'}) from AC")
        dlg.geometry("590x510")
        dlg.configure(bg=BG_DARK)
        dlg.transient(self)
        dlg.grab_set()

        card = ttk.Frame(dlg, style="Card.TFrame", padding=(16, 14))
        card.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(
            card,
            text=f"➕ Add RPM Band to {'INTERIOR (Cockpit)' if persp == 'int' else 'EXTERIOR'} view:",
            style="CardSection.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        # Detectar si hay pistas físicas libres (silenciadas) o si se reutilizará una existente
        free_b = next((b for b in self.rpm_bands if b.perspective == persp and (b.is_muted_bus or self._band_is_silenced(b))), None)
        if free_b:
            free_idx = self.rpm_bands.index(free_b)
            info_txt = f"ℹ Free physical track #{free_idx:02d} [{free_b.load_role.upper()}] will be reused so AMS2 plays it without silent gaps."
        else:
            info_txt = "ℹ All tracks are active. The closest or selected band will be reassigned while keeping the FMOD structure intact."
        ttk.Label(card, text=info_txt, style="CardMuted.TLabel", wraplength=530).pack(anchor="w", pady=(0, 10))

        # 1. Selector de muestra de Assetto Corsa
        ttk.Label(card, text="Source Sample (Assetto Corsa / WAV):", style="Card.TLabel").pack(anchor="w")
        src_row = ttk.Frame(card, style="Card.TFrame")
        src_row.pack(fill="x", pady=(2, 10))

        src_combo_vals: list[str] = []
        if self.source_lib and self.source_lib.items:
            for it in self.source_lib.items:
                if it.decodable:
                    er_txt = f"{it.effective_rpm:.0f} RPM" if it.effective_rpm else "s/RPM"
                    src_combo_vals.append(f"[{it.index:02d}] {it.name} ({it.load_role.upper()}, {er_txt})")

        add_src_var = tk.StringVar(value=src_combo_vals[0] if src_combo_vals else "")
        add_src_combo = ttk.Combobox(src_row, textvariable=add_src_var, values=src_combo_vals, state="readonly")
        add_src_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))

        add_custom_wav_var = tk.StringVar(value="")

        def _pick_add_wav() -> None:
            p = filedialog.askopenfilename(
                title="Elegir archivo WAV de muestra",
                filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")],
            )
            if p:
                add_custom_wav_var.set(p)
                add_src_var.set(f"WAV: {Path(p).name}")

        ttk.Button(src_row, text="📂 WAV...", width=9, command=_pick_add_wav).pack(side="left")

        # 2. Rol y rangos numéricos
        f_grid = ttk.Frame(card, style="Card.TFrame")
        f_grid.pack(fill="x", pady=(0, 12))
        f_grid.columnconfigure(1, weight=1)
        f_grid.columnconfigure(3, weight=1)

        add_role_var = tk.StringVar(value="on")
        ttk.Label(f_grid, text="Layer / Role:", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        role_cb = ttk.Combobox(f_grid, textvariable=add_role_var, values=["on", "off", "idle", "pops"], state="readonly", width=12)
        role_cb.grid(row=0, column=1, sticky="w", pady=4, padx=(0, 12))

        add_gain_var = tk.StringVar(value="0.0")
        ttk.Label(f_grid, text="Gain (dB):", style="Card.TLabel").grid(row=0, column=2, sticky="w", pady=4)
        ttk.Entry(f_grid, textvariable=add_gain_var, width=8).grid(row=0, column=3, sticky="w", pady=4)

        add_rmin_var = tk.StringVar(value="3000")
        add_fi_var = tk.StringVar(value="3500")
        add_fo_var = tk.StringVar(value="4800")
        add_rmax_var = tk.StringVar(value="5400")
        add_root_var = tk.StringVar(value="4000")

        def _on_source_change(_event=None) -> None:
            val = add_src_var.get()
            if val.startswith("[") and self.source_lib:
                try:
                    s_idx = int(val[1:3])
                    it = self.source_lib.items[s_idx]
                    if it.load_role in ("on", "off", "idle", "pops"):
                        add_role_var.set(it.load_role)
                    if it.effective_rpm and it.effective_rpm > 100.0:
                        er = float(it.effective_rpm)
                        add_root_var.set(f"{er:.0f}")
                        if it.load_role == "idle":
                            add_rmin_var.set("0")
                            add_fi_var.set("0")
                            add_fo_var.set(f"{idle_v * 1.3:.0f}")
                            add_rmax_var.set(f"{idle_v * 2.0:.0f}")
                        elif it.load_role == "pops":
                            add_rmin_var.set(f"{max_v * 0.4:.0f}")
                            add_fi_var.set(f"{max_v * 0.5:.0f}")
                            add_fo_var.set(f"{max_v * 1.2:.0f}")
                            add_rmax_var.set(f"{max_v * 1.2:.0f}")
                        else:
                            add_rmin_var.set(f"{max(0.0, er - 900.0):.0f}")
                            add_fi_var.set(f"{max(0.0, er - 400.0):.0f}")
                            add_fo_var.set(f"{er + 400.0:.0f}")
                            add_rmax_var.set(f"{er + 900.0:.0f}")
                except Exception:
                    pass

        add_src_combo.bind("<<ComboboxSelected>>", _on_source_change)
        _on_source_change()

        row_defs = [
            ("Min RPM:", add_rmin_var, "Fade-In End:", add_fi_var),
            ("Fade-Out Start:", add_fo_var, "Max RPM:", add_rmax_var),
        ]
        for r_idx, (l1, v1, l2, v2) in enumerate(row_defs, start=1):
            ttk.Label(f_grid, text=l1, style="Card.TLabel").grid(row=r_idx, column=0, sticky="w", pady=4)
            ttk.Entry(f_grid, textvariable=v1, width=12).grid(row=r_idx, column=1, sticky="w", pady=4, padx=(0, 12))
            ttk.Label(f_grid, text=l2, style="Card.TLabel").grid(row=r_idx, column=2, sticky="w", pady=4)
            ttk.Entry(f_grid, textvariable=v2, width=12).grid(row=r_idx, column=3, sticky="w", pady=4)

        ttk.Label(f_grid, text="Root RPM (Pitch):", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(f_grid, textvariable=add_root_var, width=12).grid(row=3, column=1, sticky="w", pady=4, padx=(0, 12))

        # Preescucha
        def _preview_add_sample() -> None:
            c_wav = add_custom_wav_var.get().strip()
            if c_wav and Path(c_wav).is_file():
                try:
                    ab = read_wav_file(Path(c_wav))
                    self.play_wav_bytes_async(write_wav_bytes(ab), f"Preescucha {Path(c_wav).name}")
                    return
                except Exception as ex:
                    messagebox.showerror("Error de audio", str(ex))
                    return
            val = add_src_var.get()
            if val.startswith("[") and self.source_lib:
                try:
                    s_idx = int(val[1:3])
                    ab = self.source_lib.items[s_idx].load_audio()
                    self.play_wav_bytes_async(write_wav_bytes(ab), f"Preescucha {self.source_lib.items[s_idx].name}")
                except Exception as ex:
                    messagebox.showerror("Error de audio", str(ex))

        act_bar = ttk.Frame(card, style="Card.TFrame")
        act_bar.pack(fill="x", pady=(10, 0))
        ttk.Button(act_bar, text="▶ Preview Sample", command=_preview_add_sample).pack(side="left")

        def _do_confirm_add() -> None:
            try:
                rmin = float(add_rmin_var.get().strip())
                fi = float(add_fi_var.get().strip())
                fo = float(add_fo_var.get().strip())
                rmax = float(add_rmax_var.get().strip())
                gain_val = float(add_gain_var.get().strip() or "0.0")
            except ValueError:
                messagebox.showwarning("Valores inválidos", "Ingresa números válidos en los campos de RPM y ganancia.")
                return

            if rmax <= rmin + 40.0:
                messagebox.showwarning("Rango inválido", "RPM Máx debe ser al menos 40 RPM mayor que RPM Mín.")
                return
            fi = max(rmin, min(rmax, fi))
            fo = max(fi, min(rmax, fo))

            root_txt = add_root_var.get().strip()
            root_val: float | None = float(root_txt) if root_txt else None

            # Determinar muestra
            c_wav_p: Path | None = Path(add_custom_wav_var.get().strip()) if add_custom_wav_var.get().strip() else None
            src_idx: int | None = None
            sample_name = "Nueva Banda"
            if c_wav_p and c_wav_p.is_file():
                sample_name = c_wav_p.name
            else:
                val = add_src_var.get()
                if val.startswith("[") and self.source_lib:
                    src_idx = int(val[1:3])
                    sample_name = self.source_lib.items[src_idx].name

            # Asignación segura a una pista física del donante existente para evitar bandas mudas
            target_band: RpmBandConfig | None = None

            # 1. Buscar si hay una banda silenciada/eliminada en esta perspectiva
            for b_cand in self.rpm_bands:
                if b_cand.perspective == persp and (b_cand.is_muted_bus or self._band_is_silenced(b_cand)):
                    target_band = b_cand
                    break

            # 2. Si no hay silenciadas, buscar una capa de fill secundario
            if target_band is None:
                for b_cand in self.rpm_bands:
                    if b_cand.perspective == persp and b_cand.is_fill_layer:
                        target_band = b_cand
                        break

            # 3. Si no hay libre ni fill, usar la banda seleccionada o la más cercana
            if target_band is None:
                if self._selected_band_idx is not None and 0 <= self._selected_band_idx < len(self.rpm_bands):
                    sel_b = self.rpm_bands[self._selected_band_idx]
                    if sel_b.perspective == persp:
                        target_band = sel_b

            if target_band is None:
                candidates = [b_cand for b_cand in self.rpm_bands if b_cand.perspective == persp]
                if candidates:
                    candidates.sort(key=lambda b_cand: abs(0.5 * (b_cand.rpm_min + b_cand.rpm_max) - 0.5 * (rmin + rmax)))
                    target_band = candidates[0]

            if target_band is None:
                messagebox.showerror("Error", "No se encontró ninguna pista física de audio disponible en esta perspectiva.")
                return

            target_band.is_muted_bus = False
            target_band.rpm_min = rmin
            target_band.fade_in_end = fi
            target_band.fade_out_start = fo
            target_band.rpm_max = rmax
            target_band.load_role = add_role_var.get()
            target_band.donor_label = f"AC: {sample_name}"

            for gidx in target_band.donor_slot_indices:
                while len(self.mappings) <= gidx:
                    self.mappings.append(SlotMappingConfig(len(self.mappings)))
                m = self.mappings[gidx]
                if c_wav_p:
                    m.action = "replace_wav"
                    m.custom_wav_path = c_wav_p
                    m.match_reason = f"Banda reasignada: {c_wav_p.name}"
                elif src_idx is not None:
                    m.action = "replace_source"
                    m.source_index = src_idx
                    m.match_reason = f"Banda reasignada: AC [{src_idx:02d}]"
                m.gain_db = gain_val
                m.override_root_rpm = root_val

            # Sincronizar capa gemela AuxInt si aplica
            if target_band.event_role == "EngineInt":
                orig_sig = (target_band.load_role, int(round(target_band.orig_rpm_min)), int(round(target_band.orig_rpm_max)))
                for other in self.rpm_bands:
                    if other.event_role == "AuxInt" and (other.load_role, int(round(other.orig_rpm_min)), int(round(other.orig_rpm_max))) == orig_sig:
                        other.is_muted_bus = False
                        other.rpm_min = rmin
                        other.fade_in_end = fi
                        other.fade_out_start = fo
                        other.rpm_max = rmax
                        other.load_role = target_band.load_role
                        for gidx in other.donor_slot_indices:
                            if 0 <= gidx < len(self.mappings):
                                self.mappings[gidx].action = "replace_wav" if c_wav_p else "replace_source"
                                self.mappings[gidx].source_index = src_idx
                                self.mappings[gidx].custom_wav_path = c_wav_p
                                self.mappings[gidx].gain_db = gain_val
                                self.mappings[gidx].override_root_rpm = root_val

            dlg.destroy()
            target_idx = self.rpm_bands.index(target_band)
            self._selected_band_idx = target_idx
            self._refresh_donor_tree()
            self._refresh_rpm_tab()
            self._on_rpm_band_selected()
            self.status_var.set(
                f"✔ Banda #{target_idx:02d} [{target_band.load_role.upper()}] asignada exitosamente con '{sample_name}' ({rmin:.0f} - {rmax:.0f} RPM)."
            )

        ttk.Button(act_bar, text="✔ Add Band to Engine", style="Primary.TButton", command=_do_confirm_add).pack(side="right")
        ttk.Button(act_bar, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 8))


    def _band_is_silenced(self, b: RpmBandConfig) -> bool:
        if b.is_muted_bus or not b.donor_slot_indices:
            return True
        if self.is_direct_ac_mode:
            return False
        return all(
            0 <= gidx < len(self.mappings) and self.mappings[gidx].action == "silence"
            for gidx in b.donor_slot_indices
        )

    def _band_sample_summary(self, b: RpmBandConfig) -> tuple[str, bool]:
        """Devuelve (etiqueta_muestra, es_reemplazada)."""
        bank = self.active_engine_bank
        if not bank or not b.donor_slot_indices:
            return b.donor_label, False
        gidx = b.donor_slot_indices[0]
        if self.is_direct_ac_mode:
            if 0 <= gidx < len(bank.slots):
                sl = bank.slots[gidx]
                return f"AC [{gidx:02d}] {sl.name}", True
            return b.donor_label, True
        if 0 <= gidx < len(self.mappings):
            cfg = self.mappings[gidx]
            if cfg.action == "replace_source" and cfg.source_index is not None and self.source_lib:
                return f"AC [{cfg.source_index:02d}] {self.source_lib.items[cfg.source_index].name}", True
            if cfg.action == "replace_wav" and cfg.custom_wav_path is not None:
                return f"WAV: {cfg.custom_wav_path.name}", True
            if cfg.action == "silence":
                return f"[Silenciada] ({bank.slots[gidx].name})", False
        if 0 <= gidx < len(bank.slots):
            return f"Donante: {bank.slots[gidx].name}", False
        return b.donor_label, False

    def _visible_band_indices(self) -> list[int]:
        persp = self.rpm_view_persp_var.get()
        show_extra = bool(self.show_muted_bands_var.get())
        primary_int_keys: set[tuple[str, int, int]] = set()
        if persp == "int" and not show_extra:
            for b in self.rpm_bands:
                if b.event_role == "EngineInt" and not self._band_is_silenced(b):
                    primary_int_keys.add((b.load_role, int(round(b.orig_rpm_min)), int(round(b.orig_rpm_max))))

        vis: list[int] = []
        for i, b in enumerate(self.rpm_bands):
            if b.perspective != persp:
                continue
            silenced = self._band_is_silenced(b)
            if not show_extra:
                if silenced or b.is_muted_bus:
                    continue
                if b.event_role == "AuxInt":
                    key = (b.load_role, int(round(b.orig_rpm_min)), int(round(b.orig_rpm_max)))
                    if key in primary_int_keys:
                        continue
            vis.append(i)

        role_order = {"idle": 0, "on": 1, "off": 2, "pops": 3}
        vis.sort(key=lambda idx: (role_order.get(self.rpm_bands[idx].load_role, 9), self.rpm_bands[idx].rpm_min, idx))
        return vis

    def _detect_band_overlaps(self, persp: str) -> set[int]:
        """Detecta índices de bandas activas que sufren solapamiento excesivo en la misma capa."""
        overlapped: set[int] = set()
        by_role: dict[str, list[int]] = {}
        for idx, b in enumerate(self.rpm_bands):
            if b.perspective != persp or self._band_is_silenced(b):
                continue
            if persp == "int" and b.event_role == "AuxInt":
                continue
            by_role.setdefault(b.load_role, []).append(idx)

        for _role, idxs in by_role.items():
            idxs.sort(key=lambda k: (self.rpm_bands[k].rpm_min, self.rpm_bands[k].rpm_max))
            for pos in range(len(idxs) - 1):
                ia = idxs[pos]
                ib = idxs[pos + 1]
                ba = self.rpm_bands[ia]
                bb = self.rpm_bands[ib]
                # Solapamiento de meseta al 100% o caja A invadiendo más allá del centro de B
                mid_b = 0.5 * (bb.rpm_min + bb.rpm_max)
                if ba.fade_out_start > bb.fade_in_end + 40.0 or ba.rpm_max > mid_b + 140.0:
                    overlapped.add(ia)
                    overlapped.add(ib)
                if pos + 2 < len(idxs):
                    ic = idxs[pos + 2]
                    bc = self.rpm_bands[ic]
                    if ba.rpm_max > bc.rpm_min + 35.0:
                        overlapped.add(ia)
                        overlapped.add(ib)
                        overlapped.add(ic)
        return overlapped

    def _refresh_rpm_tab(self) -> None:
        prev_sel = self.rpm_tree.selection()
        self.rpm_tree.delete(*self.rpm_tree.get_children())
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            self._draw_rpm_bands_canvas()
            return

        persp = self.rpm_view_persp_var.get()
        overlaps = self._detect_band_overlaps(persp)
        vis_indices = self._visible_band_indices()

        if overlaps:
            self.rpm_overlap_status_var.set(
                f"⚠ {len(overlaps)} bandas con solapamiento en vista {persp.upper()} — Pulsa 'Auto-Ajustar Escalera RPM' o edita abajo"
            )
        else:
            self.rpm_overlap_status_var.set(
                f"✔ Vista {persp.upper()}: 0 solapamientos ({len(vis_indices)} bandas activas con crossfade limpio)"
            )

        for idx in vis_indices:
            b = self.rpm_bands[idx]
            sample_lbl, is_rep = self._band_sample_summary(b)
            silenced = self._band_is_silenced(b)

            root_rpm_txt = "—"
            gain_txt = "0.0 dB"
            if b.donor_slot_indices and bank:
                g0 = b.donor_slot_indices[0]
                if 0 <= g0 < len(bank.slots):
                    sl0 = bank.slots[g0]
                    cfg0 = self.mappings[g0] if 0 <= g0 < len(self.mappings) else None
                    rr = (cfg0.override_root_rpm if cfg0 else None) or sl0.autopitch_ref_rpm
                    if rr:
                        root_rpm_txt = f"{rr:.0f} RPM"
                    if cfg0:
                        gain_txt = f"{cfg0.gain_db:+.1f} dB"

            if silenced:
                tag = "band_muted"
                st_txt = "Silenciada / Muda"
            elif idx in overlaps:
                tag = "band_overlap"
                st_txt = "⚠ Solapamiento detectado"
            elif is_rep:
                tag = "band_replaced"
                st_txt = "✔ Activa (Muestra AC/WAV)"
            else:
                tag = "band_ok"
                st_txt = "Activa (Original Donante)"

            role_badge = b.load_role.upper()
            if b.event_role == "AuxInt":
                role_badge += " [Aux]"

            self.rpm_tree.insert(
                "",
                "end",
                iid=str(idx),
                tags=(tag,),
                values=(
                    f"#{idx:02d}",
                    role_badge,
                    sample_lbl,
                    f"{b.rpm_min:.0f}",
                    f"{b.fade_in_end:.0f}",
                    f"{b.fade_out_start:.0f}",
                    f"{b.rpm_max:.0f}",
                    root_rpm_txt,
                    gain_txt,
                    st_txt,
                ),
            )

        combo_vals: list[str] = []
        if self.source_lib and self.source_lib.items:
            for it in self.source_lib.items:
                if it.decodable:
                    er_txt = f"{it.effective_rpm:.0f} RPM" if it.effective_rpm else "s/RPM"
                    combo_vals.append(f"[{it.index:02d}] {it.name} ({it.perspective.upper()}/{it.load_role.upper()}, {er_txt})")
        self.band_source_combo["values"] = combo_vals

        if self._selected_band_idx is not None and self.rpm_tree.exists(str(self._selected_band_idx)):
            self.rpm_tree.selection_set(str(self._selected_band_idx))
        else:
            self._selected_band_idx = None
            self.selected_band_lbl.config(
                text="Ninguna banda seleccionada — Haz clic en una banda de la tabla o del gráfico para editarla"
            )
            self.band_rmin_var.set("")
            self.band_fi_var.set("")
            self.band_fo_var.set("")
            self.band_rmax_var.set("")
            self.band_root_rpm_var.set("")
            self.band_gain_var.set("0.0")
            self.band_source_var.set("")
            self.band_custom_wav_var.set("")
            self.btn_reactivate_band.config(state="disabled")

        self._draw_rpm_bands_canvas()

    def _draw_rpm_bands_canvas(self) -> None:
        c = self.rpm_canvas
        c.delete("all")
        self._canvas_band_rects.clear()
        w = max(c.winfo_width(), 680)
        h = max(c.winfo_height(), 132)

        pad_l = 86.0
        pad_r = 18.0
        pad_t = 18.0
        pad_b = 18.0
        plot_w = max(100.0, w - pad_l - pad_r)
        plot_h = max(60.0, h - pad_t - pad_b)

        idle_v, max_v = self._parse_sim_rpm_bounds()
        domain_max = max(max_v * 1.08, 4000.0)

        def rpm_to_x(r: float) -> float:
            rc = max(0.0, min(domain_max, r))
            return pad_l + (rc / domain_max) * plot_w

        # Líneas verticales de rejilla cada 1000 RPM
        step_rpm = 1000 if domain_max <= 12000 else 2000
        for g_rpm in range(0, int(domain_max) + 1, step_rpm):
            gx = rpm_to_x(float(g_rpm))
            c.create_line(gx, pad_t, gx, h - pad_b, fill="#1e293b", dash=(2, 3))
            c.create_text(gx, h - 8, text=f"{g_rpm}", fill="#64748b", font=("Segoe UI", 8))

        # Marca de corte (Max RPM)
        rx_max = rpm_to_x(max_v)
        c.create_line(rx_max, pad_t - 4, rx_max, h - pad_b, fill="#ef4444", width=1, dash=(4, 2))
        c.create_text(rx_max - 4, 9, text=f"Corte {max_v:.0f}", fill="#f87171", anchor="e", font=("Segoe UI Semibold", 8))

        lanes = [
            ("idle", "IDLE", "#38bdf8", "#0c4a6e"),
            ("on", "ON (Carga)", "#34d399", "#064e3b"),
            ("off", "OFF (Desac.)", "#60a5fa", "#1e3a8a"),
            ("pops", "POPS", "#c084fc", "#4c1d95"),
        ]
        lane_h = plot_h / len(lanes)
        lane_map: dict[str, tuple[float, float, str, str]] = {}
        for li, (rk, rlbl, stroke_col, fill_col) in enumerate(lanes):
            y0 = pad_t + li * lane_h + 2.0
            y1 = pad_t + (li + 1) * lane_h - 2.0
            lane_map[rk] = (y0, y1, stroke_col, fill_col)
            c.create_line(pad_l, y1 + 2.0, w - pad_r, y1 + 2.0, fill="#172030")
            c.create_text(pad_l - 8, 0.5 * (y0 + y1), text=rlbl, fill="#94a3b8", anchor="e", font=("Segoe UI Semibold", 8))

        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            c.create_text(
                w * 0.5,
                h * 0.5,
                text="Carga un banco de audio para visualizar las curvas de crossfade por RPM",
                fill="#64748b",
                font=("Segoe UI", 9),
            )
            return

        persp = self.rpm_view_persp_var.get()
        overlaps = self._detect_band_overlaps(persp)
        vis_indices = self._visible_band_indices()

        for idx in vis_indices:
            b = self.rpm_bands[idx]
            if self._band_is_silenced(b) or b.event_role == "AuxInt":
                continue
            if b.load_role not in lane_map:
                continue
            y_top, y_bot, stroke_col, fill_col = lane_map[b.load_role]
            x0 = rpm_to_x(b.rpm_min)
            x1 = rpm_to_x(max(b.rpm_min, min(b.fade_in_end, b.rpm_max)))
            x2 = rpm_to_x(max(b.rpm_min, min(b.fade_out_start, b.rpm_max)))
            x3 = rpm_to_x(b.rpm_max)

            is_sel = idx == self._selected_band_idx
            is_ov = idx in overlaps
            out_col = "#ffffff" if is_sel else ("#f59e0b" if is_ov else stroke_col)
            in_col = "#1d4ed8" if is_sel else ("#78350f" if is_ov else fill_col)
            lw = 2 if (is_sel or is_ov) else 1

            c.create_polygon(
                x0, y_bot, x1, y_top, x2, y_top, x3, y_bot,
                fill=in_col,
                outline=out_col,
                width=lw,
            )
            if x3 - x0 >= 36:
                lbl_short = f"#{idx:02d}"
                c.create_text(
                    0.5 * (x1 + x2),
                    0.5 * (y_top + y_bot),
                    text=lbl_short,
                    fill="#f8fafc",
                    font=("Segoe UI Semibold", 8),
                )
            self._canvas_band_rects.append((x0, y_top, x3, y_bot, idx))

        # Aguja animada en tiempo real durante la simulación de las 5 etapas
        if self._sim_current_rpm is not None:
            nx = rpm_to_x(self._sim_current_rpm)
            c.create_line(nx, 4, nx, h - pad_b + 2, fill="#facc15", width=2)
            c.create_rectangle(nx - 34, 2, nx + 34, 16, fill="#facc15", outline="#ca8a04")
            c.create_text(
                nx,
                9,
                text=f"{self._sim_current_rpm:.0f} RPM",
                fill="#0f172a",
                font=("Segoe UI Bold", 8),
            )

    def _on_rpm_canvas_click(self, event) -> None:
        cx, cy = float(event.x), float(event.y)
        for x0, y0, x1, y1, b_idx in reversed(self._canvas_band_rects):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                iid = str(b_idx)
                if self.rpm_tree.exists(iid):
                    self.rpm_tree.selection_set(iid)
                    self.rpm_tree.see(iid)
                    self._on_rpm_band_selected()
                break

    def _on_rpm_band_selected(self, _event=None) -> None:
        bank = self.active_engine_bank
        if not bank or not self.rpm_bands:
            return
        sel = self.rpm_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if not (0 <= idx < len(self.rpm_bands)):
            return
        self._selected_band_idx = idx
        b = self.rpm_bands[idx]
        sample_lbl, _ = self._band_sample_summary(b)
        silenced = self._band_is_silenced(b)

        # Actualizar valores del combo de muestras de AC si hay librería disponible
        combo_vals: list[str] = []
        if self.source_lib and self.source_lib.items:
            for it in self.source_lib.items:
                if it.decodable:
                    er_txt = f"{it.effective_rpm:.0f} RPM" if it.effective_rpm else "s/RPM"
                    combo_vals.append(f"[{it.index:02d}] {it.name} ({it.perspective.upper()}/{it.load_role.upper()}, {er_txt})")
        self.band_source_combo["values"] = combo_vals

        # Detectar qué muestra tiene actualmente asignada
        current_sample_val = ""
        if b.donor_slot_indices and bank:
            g0 = b.donor_slot_indices[0]
            if 0 <= g0 < len(self.mappings):
                cfg0 = self.mappings[g0]
                if cfg0.action == "replace_source" and cfg0.source_index is not None:
                    s_idx = cfg0.source_index
                    for cv in combo_vals:
                        if cv.startswith(f"[{s_idx:02d}]"):
                            current_sample_val = cv
                            break
                    if not current_sample_val and self.source_lib and 0 <= s_idx < len(self.source_lib.items):
                        current_sample_val = f"[{s_idx:02d}] {self.source_lib.items[s_idx].name}"
                elif cfg0.action == "replace_wav" and cfg0.custom_wav_path:
                    current_sample_val = f"WAV: {cfg0.custom_wav_path.name}"
                elif cfg0.action == "silence":
                    current_sample_val = f"[Silenciada / Sin Muestra] ({bank.slots[g0].name})"
                else:
                    current_sample_val = f"Donante: {bank.slots[g0].name}"
        if current_sample_val:
            self.band_source_var.set(current_sample_val)
        elif combo_vals:
            self.band_source_var.set(combo_vals[0])

        if silenced or b.is_muted_bus:
            self.btn_reactivate_band.config(state="normal")
            self.selected_band_lbl.config(
                text=f"Banda #{idx:02d} [{b.event_role} / {b.load_role.upper()}] — [SILENCIADA / LIBRE] — Pulsa 'Sobreescribir Muestra' o 'Reactivar'"
            )
        else:
            self.btn_reactivate_band.config(state="disabled")
            self.selected_band_lbl.config(
                text=f"Banda #{idx:02d} [{b.event_role} / {b.load_role.upper()}] — {sample_lbl}"
            )

        self.band_rmin_var.set(f"{b.rpm_min:.0f}")
        self.band_fi_var.set(f"{b.fade_in_end:.0f}")
        self.band_fo_var.set(f"{b.fade_out_start:.0f}")
        self.band_rmax_var.set(f"{b.rpm_max:.0f}")

        if b.donor_slot_indices and bank:
            g0 = b.donor_slot_indices[0]
            if 0 <= g0 < len(bank.slots):
                sl0 = bank.slots[g0]
                cfg0 = self.mappings[g0] if 0 <= g0 < len(self.mappings) else None
                rr = (cfg0.override_root_rpm if cfg0 else None) or sl0.autopitch_ref_rpm
                self.band_root_rpm_var.set(f"{rr:.0f}" if rr else "")
                self.band_gain_var.set(f"{cfg0.gain_db:.1f}" if cfg0 else "0.0")
            else:
                self.band_root_rpm_var.set("")
                self.band_gain_var.set("0.0")
        else:
            self.band_root_rpm_var.set("")
            self.band_gain_var.set("0.0")

        self._draw_rpm_bands_canvas()

    def cmd_apply_band_edit(self) -> None:
        bank = self.active_engine_bank
        if self._selected_band_idx is None or not bank or not self.rpm_bands:
            return
        idx = self._selected_band_idx
        b = self.rpm_bands[idx]
        try:
            rmin = float(self.band_rmin_var.get().strip())
            fi = float(self.band_fi_var.get().strip())
            fo = float(self.band_fo_var.get().strip())
            rmax = float(self.band_rmax_var.get().strip())
        except ValueError:
            messagebox.showwarning("Valores RPM inválidos", "Ingresa valores numéricos válidos para los rangos de RPM.")
            return

        if rmax <= rmin + 40.0:
            messagebox.showwarning("Rango inválido", "RPM Máx debe ser al menos 40 RPM mayor que RPM Mín.")
            return
        fi = max(rmin, min(rmax, fi))
        fo = max(fi, min(rmax, fo))

        orig_signature = (b.load_role, int(round(b.orig_rpm_min)), int(round(b.orig_rpm_max)))
        b.rpm_min = rmin
        b.fade_in_end = fi
        b.fade_out_start = fo
        b.rpm_max = rmax

        # Sincronizar capa gemela AuxInt si se edita una banda principal EngineInt
        if b.event_role == "EngineInt":
            for other in self.rpm_bands:
                if other.event_role == "AuxInt":
                    other_sig = (other.load_role, int(round(other.orig_rpm_min)), int(round(other.orig_rpm_max)))
                    if other_sig == orig_signature:
                        other.rpm_min = rmin
                        other.fade_in_end = fi
                        other.fade_out_start = fo
                        other.rpm_max = rmax

        root_txt = self.band_root_rpm_var.get().strip()
        new_root: float | None = None
        if root_txt:
            try:
                new_root = float(root_txt)
            except ValueError:
                messagebox.showwarning("RPM Raíz inválido", "Ingresa un número válido para RPM Raíz (Auto-Pitch).")
                return

        try:
            new_gain = float(self.band_gain_var.get().strip() or "0.0")
        except ValueError:
            new_gain = 0.0

        for gidx in b.donor_slot_indices:
            if 0 <= gidx < len(self.mappings):
                self.mappings[gidx].override_root_rpm = new_root
                self.mappings[gidx].gain_db = new_gain

        # Si se seleccionó una muestra en el combo de la banda y difiere de la actual, aplicarla también
        val_src = self.band_source_var.get().strip()
        c_wav = self.band_custom_wav_var.get().strip()
        c_wav_p = Path(c_wav) if (c_wav and Path(c_wav).is_file()) else None
        if c_wav_p:
            b.is_muted_bus = False
            b.donor_label = f"WAV: {c_wav_p.name}"
            for gidx in b.donor_slot_indices:
                if 0 <= gidx < len(self.mappings):
                    self.mappings[gidx].action = "replace_wav"
                    self.mappings[gidx].custom_wav_path = c_wav_p
                    self.mappings[gidx].match_reason = f"Sobreescrito por WAV: {c_wav_p.name}"
        elif val_src.startswith("[") and self.source_lib:
            try:
                s_idx = int(val_src[1:3])
                if 0 <= s_idx < len(self.source_lib.items):
                    src_it = self.source_lib.items[s_idx]
                    b.is_muted_bus = False
                    b.donor_label = f"AC: {src_it.name}"
                    for gidx in b.donor_slot_indices:
                        if 0 <= gidx < len(self.mappings):
                            self.mappings[gidx].action = "replace_source"
                            self.mappings[gidx].source_index = s_idx
                            self.mappings[gidx].match_reason = f"Sobreescrito por usuario: {src_it.name}"
            except Exception:
                pass

        self._refresh_donor_tree()
        self._refresh_rpm_tab()
        self.status_var.set(
            f"✔ Banda #{idx:02d} actualizada: [{rmin:.0f} -> {fi:.0f} .. {fo:.0f} -> {rmax:.0f} RPM]."
        )

    def cmd_preview_selected_band_sample(self) -> None:
        bank = self.active_engine_bank
        if self._selected_band_idx is None or not bank or not self.rpm_bands:
            return
        b = self.rpm_bands[self._selected_band_idx]
        if not b.donor_slot_indices:
            return
        self._selected_donor_gidx = b.donor_slot_indices[0]
        self.cmd_preview_mapped_slot()

    def cmd_simulate_pass_int(self) -> None:
        self._run_engine_pass_simulation("int")

    def cmd_simulate_pass_ext(self) -> None:
        self._run_engine_pass_simulation("ext")

    def _run_engine_pass_simulation(self, persp: str) -> None:
        bank = self.active_engine_bank
        if not bank:
            messagebox.showinfo(
                "Cargar Banco de Audio",
                "Carga primero el banco .bank de origen (Paso 1) o el banco donante de AMS2 (Paso 2) antes de simular.",
            )
            return
        self.stop_audio_preview()
        self.rpm_view_persp_var.set(persp)
        self._refresh_rpm_tab()

        idle_v, max_v = self._parse_sim_rpm_bounds()
        tcut_cfg = self._collect_throttle_cut_config()

        self.btn_sim_int.config(state="disabled")
        self.btn_sim_ext.config(state="disabled")
        self.progress.start(12)
        persp_lbl = "INTERIOR (Cabina)" if persp == "int" else "EXTERIOR (Escape/Pista)"
        self.status_var.set(f"⏳ Sintetizando pasada de motor en 5 etapas — {persp_lbl}...")

        def worker() -> None:
            try:
                wav_bytes, traj = render_engine_pass_wav_bytes(
                    perspective=persp,
                    donor=bank,
                    mappings=self.mappings,
                    rpm_bands=self.rpm_bands,
                    throttle_cut=tcut_cfg,
                    library=self.source_lib,
                    idle_rpm=idle_v,
                    max_rpm=max_v,
                    sample_rate=44100,
                )
                self.dispatch_to_ui(self._on_simulation_ready, persp_lbl, wav_bytes, traj)
            except Exception as exc:
                err_s = str(exc)
                self.dispatch_to_ui(self._on_simulation_error, err_s)

        threading.Thread(target=worker, daemon=True).start()

    def _on_simulation_ready(self, persp_lbl: str, wav_bytes: bytes, traj: EnginePassTrajectory) -> None:
        self.progress.stop()
        self.btn_sim_int.config(state="normal")
        self.btn_sim_ext.config(state="normal")
        self._sim_trajectory = traj
        self._sim_start_perf = time.perf_counter()
        self.play_wav_bytes_async(wav_bytes, f"Simulador 5 Etapas — {persp_lbl}", keep_sim_anim=True)
        self._tick_sim_animation()

    def _on_simulation_error(self, err_msg: str) -> None:
        self.progress.stop()
        self.btn_sim_int.config(state="normal")
        self.btn_sim_ext.config(state="normal")
        self.status_var.set("❌ Error al sintetizar la simulación.")
        messagebox.showerror("Error en Simulador de Motor", err_msg)

    def _tick_sim_animation(self) -> None:
        traj = self._sim_trajectory
        if traj is None:
            return
        elapsed = time.perf_counter() - self._sim_start_perf
        if elapsed > traj.duration_sec + 0.15:
            self._stop_sim_animation()
            self.sim_telemetry_var.set("✔ Pasada completada (Etapas i -> v finalizadas en ralentí).")
            return

        rpm, thr, gear, st_num, st_desc = traj.sample_telemetry_at(elapsed)
        self._sim_current_rpm = rpm
        self.sim_stage_var.set(f"Tiempo: {min(elapsed, traj.duration_sec):4.1f} / {traj.duration_sec:.1f} s")
        self.sim_telemetry_var.set(
            f"🏎️  {rpm:5.0f} RPM   |   Marcha: {gear}ª   |   Acelerador: {thr * 100:3.0f}%   —   {st_desc}"
        )
        self._draw_rpm_bands_canvas()
        self._sim_anim_after_id = self.after(35, self._tick_sim_animation)

    def cmd_export_simulated_wav(self) -> None:
        bank = self.active_engine_bank
        if not bank:
            messagebox.showinfo("Sin banco cargado", "Carga primero un banco .bank de AC o donante de AMS2.")
            return
        persp = self.rpm_view_persp_var.get()
        p = filedialog.asksaveasfilename(
            title=f"Guardar simulación de 5 etapas ({persp.upper()}) como WAV",
            initialfile=f"simulacion_motor_{persp}_5_etapas.wav",
            defaultextension=".wav",
            filetypes=[("Archivo de Audio WAV", "*.wav")],
        )
        if not p:
            return
        try:
            idle_v, max_v = self._parse_sim_rpm_bounds()
            tcut_cfg = self._collect_throttle_cut_config()
            wav_bytes, _ = render_engine_pass_wav_bytes(
                perspective=persp,
                donor=bank,
                mappings=self.mappings,
                rpm_bands=self.rpm_bands,
                throttle_cut=tcut_cfg,
                library=self.source_lib,
                idle_rpm=idle_v,
                max_rpm=max_v,
                sample_rate=48000,
            )
            Path(p).write_bytes(wav_bytes)
            self.status_var.set(f"💾 Simulación {persp.upper()} guardada en: {p}")
        except Exception as exc:
            messagebox.showerror("Error al exportar simulación WAV", str(exc))
        persp = self.rpm_view_persp_var.get()
        p = filedialog.asksaveasfilename(
            title=f"Guardar simulación de 5 etapas ({persp.upper()}) como WAV",
            initialfile=f"simulacion_motor_{persp}_5_etapas.wav",
            defaultextension=".wav",
            filetypes=[("Archivo de Audio WAV", "*.wav")],
        )
        if not p:
            return
        try:
            idle_v, max_v = self._parse_sim_rpm_bounds()
            tcut_cfg = self._collect_throttle_cut_config()
            wav_bytes, _ = render_engine_pass_wav_bytes(
                perspective=persp,
                donor=self.donor_bank,
                mappings=self.mappings,
                rpm_bands=self.rpm_bands,
                throttle_cut=tcut_cfg,
                library=self.source_lib,
                idle_rpm=idle_v,
                max_rpm=max_v,
                sample_rate=48000,
            )
            Path(p).write_bytes(wav_bytes)
            self.status_var.set(f"💾 Simulación {persp.upper()} guardada en: {p}")
        except Exception as exc:
            messagebox.showerror("Error al exportar simulación WAV", str(exc))

    # ========================================================================
    # Acciones de Pestaña 4 (.csdbin integrado con OMVT)
    # ========================================================================
    def cmd_open_csdbin(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Automobilista 2 .csdbin or .csd file",
            filetypes=[
                ("AMS2 Vehicle Sound Files", "*.csdbin *.csd"),
                ("CSDBIN Binary", "*.csdbin"),
                ("OMVT CSD Text", "*.csd"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self._load_csdbin_path(Path(p))

    def _load_csdbin_path(self, path: Path) -> None:
        try:
            doc = CsdbinDocument.from_file(path)
            self.csd_doc = doc
            self.csd_info_var.set(f"✔ {path.name} ({len(doc.links)} enlaces, {len(doc.fields)} parámetros)")
            self._refresh_ui_file_statuses()
            self._populate_visual_csd_from_doc()
            self.raw_csd_text.delete("1.0", "end")
            self.raw_csd_text.insert("1.0", doc.to_csd_text())
            # Si el archivo proviene de Vehicles/<carpeta_vehiculo>/<archivo>.csdbin, usar <carpeta_vehiculo>
            if path.parent.parent.name.lower() == "vehicles" and path.parent.name:
                self.veh_folder_var.set(path.parent.name)
            elif path.stem and not path.stem.lower().startswith("archivo"):
                self.veh_folder_var.set(path.stem)
            self.status_var.set(f"Archivo '{path.name}' desempaquetado con éxito mediante el codec OMVT.")
        except Exception as exc:
            messagebox.showerror("Error al leer .csdbin / .csd", str(exc))

    def _populate_visual_csd_from_doc(self) -> None:
        if not self.csd_doc:
            return
        d = self.csd_doc
        self.csd_id_var.set(d.doc_id)
        self.csd_class_var.set(d.sound_class)
        self.csd_mix_var.set(d.mix_set)
        self.csd_vol_ext_var.set(d.fields.get("off_0000", "1.0"))
        self.csd_vol_int_var.set(d.fields.get("off_0004", "1.0"))
        self.csd_gears_var.set(d.fields.get("off_00a0", "6"))
        x, y, z = d.get_emitter_xyz()
        self.csd_pos_x_var.set(f"{x:g}")
        self.csd_pos_y_var.set(f"{y:g}")
        self.csd_pos_z_var.set(f"{z:g}")

        for k, svar in self.csd_link_vars.items():
            svar.set(d.links.get(k, ""))

    def _collect_visual_into_doc(self) -> CsdbinDocument:
        if self.csd_doc is None:
            self.csd_doc = CsdbinDocument()
        d = self.csd_doc
        d.doc_id = self.csd_id_var.get().strip() or "0000000000000000"
        d.sound_class = self.csd_class_var.get().strip()
        d.mix_set = self.csd_mix_var.get().strip()
        d.fields["off_0000"] = self.csd_vol_ext_var.get().strip() or "1.0"
        d.fields["off_0004"] = self.csd_vol_int_var.get().strip() or "1.0"
        d.fields["off_00a0"] = self.csd_gears_var.get().strip() or "6"
        try:
            d.set_emitter_xyz(
                float(self.csd_pos_x_var.get()),
                float(self.csd_pos_y_var.get()),
                float(self.csd_pos_z_var.get()),
            )
        except ValueError:
            pass

        for k, svar in self.csd_link_vars.items():
            val = svar.get().strip()
            if val:
                d.links[k] = val
            else:
                d.links.pop(k, None)
        return d

    def _sync_visual_to_raw_csd_editor(self) -> None:
        doc = self._collect_visual_into_doc()
        self.raw_csd_text.delete("1.0", "end")
        self.raw_csd_text.insert("1.0", doc.to_csd_text())
        self.status_var.set("Texto .csd sincronizado desde el formulario visual.")

    def _apply_raw_csd_editor_to_doc(self) -> None:
        try:
            txt = self.raw_csd_text.get("1.0", "end")
            self.csd_doc = CsdbinDocument.from_csd_text(txt)
            self._populate_visual_csd_from_doc()
            self.status_var.set("✔ Cambios del código fuente .csd aplicados correctamente.")
        except Exception as exc:
            messagebox.showerror("Error de sintaxis en .csd", str(exc))

    def cmd_autofill_csdbin_standalone(self) -> None:
        bank = self.active_engine_bank
        if not bank:
            messagebox.showinfo(
                "Banco de audio requerido",
                "Carga primero un banco .bank de Assetto Corsa (Paso 1) o donante de AMS2 (Paso 2) para generar las rutas 'event:/...' de sus eventos.",
            )
            return
        doc = self._collect_visual_into_doc()
        safe_mod = "".join(
            c if (c.isalnum() or c in "_-") else "_" for c in self.mod_id_var.get().strip()
        ) or "custom_sound"

        role_paths = build_mod_event_paths(safe_mod, bank.events)
        bank_rel = f"Audio/AMS2/Vehicles/Mods/{safe_mod}/{safe_mod}_Engine"
        doc.links["engine_bank"] = bank_rel
        if "EngineExt" in role_paths:
            doc.links["engine_ext"] = role_paths["EngineExt"]
        if "EngineInt" in role_paths:
            doc.links["engine_int"] = role_paths["EngineInt"]
        if "Exhaust" in role_paths:
            doc.links["exhaust_int"] = role_paths["Exhaust"]
        else:
            doc.links.pop("exhaust_int", None)
            doc.links.pop("exhaust_ext", None)
        if "Starter" in role_paths:
            doc.links["starter"] = role_paths["Starter"]
        else:
            doc.links.pop("starter", None)
        if "Starting" in role_paths:
            doc.links["starting"] = role_paths["Starting"]
        else:
            doc.links.pop("starting", None)

        gear_path = (
            role_paths.get("GearshiftExt")
            or role_paths.get("Gearshift")
            or role_paths.get("GearshiftInt")
            or role_paths.get("gear_ext")
            or role_paths.get("gear_int")
        )
        if gear_path:
            doc.links["gearshift_bank"] = bank_rel
            doc.links["gearshift_up"] = gear_path
            doc.links["gearshift_down"] = gear_path
        grind_path = role_paths.get("gear_grind") or role_paths.get("GearshiftGrind")
        if grind_path:
            doc.links["gearshift_grind"] = grind_path

        turbo_path = role_paths.get("Turbo") or role_paths.get("turbo")
        if turbo_path:
            doc.links["turbo_bank"] = bank_rel
            doc.links["turbo"] = turbo_path

        trans_path = (
            role_paths.get("Transmission")
            or role_paths.get("transmission")
            or role_paths.get("transmission_ext")
        )
        if trans_path:
            doc.links["transmission_bank"] = bank_rel
            doc.links["transmission"] = trans_path

        limiter_path = role_paths.get("Limiter") or role_paths.get("limiter")
        if limiter_path:
            doc.links["rev_limiter_bank"] = bank_rel
            doc.links["rev_limiter"] = limiter_path

        if "BackfireExt" in role_paths:
            doc.links["backfire_ext"] = role_paths["BackfireExt"]
        elif "backfire_ext" in role_paths:
            doc.links["backfire_ext"] = role_paths["backfire_ext"]

        if "BackfireInt" in role_paths:
            doc.links["backfire_int"] = role_paths["BackfireInt"]
        elif "backfire_int" in role_paths:
            doc.links["backfire_int"] = role_paths["backfire_int"]

        wind_path = role_paths.get("Wind") or role_paths.get("wind")
        if wind_path:
            doc.links["wind_bank"] = bank_rel
            doc.links["wind"] = wind_path

        self._populate_visual_csd_from_doc()
        self._sync_visual_to_raw_csd_editor()
        self.status_var.set(
            f"🔗 Enlaces de motor actualizados a '{bank_rel}' con rutas FMOD 'event:/AMS2/Vehicles/Mods/{safe_mod}/...'."
        )

    def cmd_save_csdbin_as(self) -> None:
        doc = self._collect_visual_into_doc()
        p = filedialog.asksaveasfilename(
            title="Guardar archivo binario .csdbin",
            defaultextension=".csdbin",
            filetypes=[("Binario Sonido AMS2", "*.csdbin")],
        )
        if not p:
            return
        try:
            doc.save_csdbin(Path(p))
            self.status_var.set(f"✔ Archivo guardado: {p}")
        except Exception as exc:
            messagebox.showerror("Error al empaquetar .csdbin", str(exc))

    def cmd_export_csd_text(self) -> None:
        doc = self._collect_visual_into_doc()
        p = filedialog.asksaveasfilename(
            title="Exportar texto editable .csd (OMVT)",
            defaultextension=".csd",
            filetypes=[("Texto OMVT CSD", "*.csd")],
        )
        if not p:
            return
        try:
            doc.save_csd(Path(p))
            self.status_var.set(f"✔ Texto .csd exportado: {p}")
        except Exception as exc:
            messagebox.showerror("Error al exportar .csd", str(exc))

    # ========================================================================
    # Acciones y Lógica de Vehículos Vanilla / Base de Automobilista 2
    # ========================================================================
    def _on_target_mode_changed(self) -> None:
        mode = self.target_mode_var.get()
        if mode == "vanilla":
            if hasattr(self, "mod_view_frame"):
                self.mod_view_frame.pack_forget()
            if hasattr(self, "vanilla_view_frame"):
                self.vanilla_view_frame.pack(fill="both", expand=True)
            if hasattr(self, "mod_cfg_card") and hasattr(self, "vanilla_cfg_card"):
                self.mod_cfg_card.pack_forget()
                self.vanilla_cfg_card.pack(fill="x", pady=(0, 12), before=self.opts_card)
            if hasattr(self, "btn_export"):
                veh_name = (
                    self.vanilla_selected_vehicle.display_name
                    if self.vanilla_selected_vehicle
                    else "Vehículo Vanilla"
                )
                if self.is_direct_ac_mode:
                    btn_text = f"⚡ INYECTAR DIRECTO 1:1 EN '{veh_name.upper()}' (SIN DONANTE - 100% FIEL)"
                else:
                    btn_text = f"🎯 INYECTAR MAPEO EN '{veh_name.upper()}' (MODO DONANTE)"
                self.btn_export.config(text=btn_text)
        else:
            if hasattr(self, "vanilla_view_frame"):
                self.vanilla_view_frame.pack_forget()
            if hasattr(self, "mod_view_frame"):
                self.mod_view_frame.pack(fill="both", expand=True)
            if hasattr(self, "mod_cfg_card") and hasattr(self, "vanilla_cfg_card"):
                self.vanilla_cfg_card.pack_forget()
                self.mod_cfg_card.pack(fill="x", pady=(0, 12), before=self.opts_card)
            if hasattr(self, "btn_export"):
                self.btn_export.config(text="🚀  COMPILAR Y EXPORTAR PAQUETE DE SONIDO PARA AUTOMOBILISTA 2")

    def _populate_vanilla_vehicles_list(self, select_id: str | None = None) -> None:
        if not hasattr(self, "vanilla_mgr"):
            return
        vehicles = self.vanilla_mgr.scan_catalog()
        self.vanilla_filtered_vehicles = list(vehicles)
        self._filter_vanilla_vehicles(select_id=select_id)

    def _filter_vanilla_vehicles(self, select_id: str | None = None) -> None:
        if not hasattr(self, "vanilla_mgr"):
            return
        query = self.vanilla_search_var.get().strip().lower()
        all_vehs = self.vanilla_mgr.scan_catalog()
        if query:
            filtered = [
                v
                for v in all_vehs
                if query in v.display_name.lower()
                or query in v.brand.lower()
                or query in v.car_name.lower()
            ]
        else:
            filtered = list(all_vehs)

        self.vanilla_filtered_vehicles = filtered
        combo_vals: list[str] = []
        for v in filtered:
            status_tag = "🟠 [Modificado]" if v.is_modified else "🟢 [Original]"
            combo_vals.append(f"{status_tag} {v.display_name}")

        if hasattr(self, "vanilla_combo"):
            self.vanilla_combo["values"] = combo_vals
            if select_id:
                for idx, v in enumerate(filtered):
                    if v.vehicle_id == select_id:
                        self.vanilla_combo.current(idx)
                        self._on_vanilla_vehicle_selected()
                        return
            if combo_vals:
                current_selected = self.vanilla_selected_vehicle
                if current_selected and any(v.vehicle_id == current_selected.vehicle_id for v in filtered):
                    idx = next(i for i, v in enumerate(filtered) if v.vehicle_id == current_selected.vehicle_id)
                    self.vanilla_combo.current(idx)
                else:
                    self.vanilla_combo.current(0)
                self._on_vanilla_vehicle_selected()
            else:
                self.vanilla_selected_vehicle = None
                self._clear_vanilla_details()

    def _clear_vanilla_details(self) -> None:
        if hasattr(self, "vanilla_det_name_var"):
            self.vanilla_det_name_var.set("—")
            self.vanilla_det_brand_var.set("—")
            self.vanilla_det_path_var.set("—")
            self.vanilla_det_ai_var.set("—")
            self.vanilla_det_events_var.set("—")
            self.vanilla_det_integrity_var.set("—")
            self.vanilla_det_sha_var.set("—")
            self.vanilla_export_dest_var.set("Ningún vehículo vanilla coincide con la búsqueda.")

    def _on_vanilla_vehicle_selected(self, _event=None) -> None:
        if not hasattr(self, "vanilla_combo"):
            return
        idx = self.vanilla_combo.current()
        if idx < 0 or idx >= len(self.vanilla_filtered_vehicles):
            return
        v = self.vanilla_filtered_vehicles[idx]
        self.vanilla_selected_vehicle = v
        self.vanilla_mgr.load_vehicle_events_info(v)

        if hasattr(self, "vanilla_det_name_var"):
            self.vanilla_det_name_var.set(v.display_name)
            self.vanilla_det_brand_var.set(f"{v.brand} ({v.car_name})")
            size_mb = v.bank_path.stat().st_size / (1024 * 1024) if v.bank_path.is_file() else 0.0
            self.vanilla_det_path_var.set(f"{v.bank_path}  ({size_mb:.2f} MB, {v.slots_count} slots)")
            self.vanilla_det_ai_var.set(str(v.ai_bank_path) if v.ai_bank_path else "Sin banco AI separado")

            ev_list = []
            for r in v.event_roles:
                g_short = v.event_guids.get(r, "")[:8]
                ev_list.append(f"{r} [{g_short}...]")
            self.vanilla_det_events_var.set(", ".join(ev_list) if ev_list else "Eventos nativos estándar")

            if v.is_modified:
                self.vanilla_det_integrity_var.set(
                    "🟠 MODIFICADO: Utilizando sonido porteado (Copia de fábrica protegida)"
                )
            else:
                self.vanilla_det_integrity_var.set("🟢 ORIGINAL DE STEAM: Archivo de fábrica intacto")

            if v.original_sha256:
                self.vanilla_det_sha_var.set(
                    f"Original: {v.original_sha256} (Registrado: {v.backup_timestamp or 'N/A'})"
                )
            else:
                self.vanilla_det_sha_var.set(
                    "Archivo original de fábrica de Steam (se respaldará automáticamente al inyectar)"
                )

            self.vanilla_export_dest_var.set(f"{v.display_name} -> {v.relative_audio_path}")
            if hasattr(self, "btn_export") and self.target_mode_var.get() == "vanilla":
                if self.is_direct_ac_mode:
                    btn_text = f"⚡ INYECTAR DIRECTO 1:1 EN '{v.display_name.upper()}' (SIN DONANTE - 100% FIEL)"
                else:
                    btn_text = f"🎯 INYECTAR MAPEO EN '{v.display_name.upper()}' (MODO DONANTE)"
                self.btn_export.config(text=btn_text)

    def cmd_browse_ams2_path(self) -> None:
        d = filedialog.askdirectory(title="Seleccionar carpeta de instalación de Automobilista 2")
        if d:
            p = Path(d)
            if (p / "Audio" / "AMS2" / "Vehicles" / "Sort").is_dir():
                self.vanilla_mgr.ams2_root = p
                self.vanilla_ams2_path_var.set(str(p))
                self.cmd_rescan_vanilla_catalog()
            else:
                messagebox.showerror(
                    "Directorio Inválido",
                    "La carpeta seleccionada no parece contener una instalación válida de AMS2.\n\n"
                    "No se encontró la subcarpeta obligatoria 'Audio/AMS2/Vehicles/Sort'.",
                )

    def cmd_rescan_vanilla_catalog(self) -> None:
        self.vanilla_mgr.scan_catalog(force=True)
        self._populate_vanilla_vehicles_list()
        self.status_var.set(
            f"✔ Catálogo re-escaneado: {len(self.vanilla_mgr._vehicles_cache)} vehículos vanilla disponibles."
        )

    def cmd_set_vanilla_direct_mode(self) -> None:
        v = self.vanilla_selected_vehicle
        if not v or not v.bank_path.is_file():
            messagebox.showwarning("Sin Selección", "Por favor selecciona primero un vehículo oficial de la lista.")
            return

        self.port_mode_var.set("direct")
        self.target_mode_var.set("vanilla")
        self.vanilla_mgr.load_vehicle_events_info(v)
        self._on_port_mode_changed()

        if self.source_lib and self.source_lib.fmod_bank:
            self.status_var.set(
                f"⚡ Modo Porteo Directo 1:1 activo para '{v.display_name}'. "
                f"Sonido de Assetto Corsa 100% puro sin donante listo para inyectar en Paso 5."
            )
            self.notebook.select(self.tab_export)
        else:
            self.status_var.set(
                f"⚡ '{v.display_name}' configurado para Porteo Directo 1:1. "
                f"Carga el archivo .bank de Assetto Corsa en el Paso 1 para transmutarlo."
            )
            self.notebook.select(self.tab_source)

    def cmd_load_vanilla_as_donor(self) -> None:
        v = self.vanilla_selected_vehicle
        if not v or not v.bank_path.is_file():
            messagebox.showwarning("Sin Selección", "Por favor selecciona un vehículo vanilla de la lista.")
            return
        self._load_donor_bank_file(v.bank_path)
        self.notebook.select(self.tab_donor)
        self.status_var.set(f"✔ Banco vanilla de '{v.display_name}' cargado como donante en Paso 2.")

    def cmd_select_and_load_vanilla_donor(self) -> None:
        if self.vanilla_selected_vehicle:
            self.cmd_load_vanilla_as_donor()
        else:
            self.notebook.select(self.tab_csdbin)
            self.target_mode_var.set("vanilla")
            self._on_target_mode_changed()

    def cmd_restore_selected_vanilla(self) -> None:
        v = self.vanilla_selected_vehicle
        if not v:
            messagebox.showwarning("Sin Selección", "Selecciona un vehículo vanilla de la lista.")
            return
        if not v.is_modified and not v.has_backup and not (v.bank_path.parent / f"{v.bank_path.name}.orig").is_file():
            messagebox.showinfo("Vehículo Ya Original", f"'{v.display_name}' ya está en su estado 100% original de Steam.")
            return

        ans = messagebox.askyesno(
            "Restaurar Vehículo a Original de Steam",
            f"¿Deseas restaurar '{v.display_name}' a sus sonidos originales de fábrica de Steam?\n\n"
            f"• Archivo: {v.bank_path}\n"
            f"• Comprobación: Hash SHA256 idéntico al original\n\n"
            f"La restauración es instantánea y no requiere comprobación en Steam.",
        )
        if not ans:
            return

        ok, msg = self.vanilla_mgr.restore_vehicle(v)
        if ok:
            self._populate_vanilla_vehicles_list(select_id=v.vehicle_id)
            self.status_var.set(f"✔ '{v.display_name}' restaurado con éxito a original.")
            messagebox.showinfo("Restauración Exitosa", msg)
        else:
            messagebox.showerror("Error al Restaurar", msg)

    def cmd_restore_all_vanilla(self) -> None:
        modded = self.vanilla_mgr.get_modified_vehicles()
        if not modded:
            messagebox.showinfo(
                "Todos los Vehículos Originales",
                "Todos los vehículos oficiales de Automobilista 2 ya están en su estado 100% original de Steam.",
            )
            return

        ans = messagebox.askyesno(
            "Restaurar TODOS los Vehículos Vanilla",
            f"Se han detectado {len(modded)} vehículo(s) vanilla con sonidos modificados:\n\n"
            + "\n".join(f"• {v.display_name}" for v in modded[:10])
            + (f"\n... y {len(modded)-10} más." if len(modded) > 10 else "")
            + "\n\n¿Deseas restaurar TODOS los vehículos al estado original de Steam ahora?",
        )
        if not ans:
            return

        s_cnt, f_cnt, msgs = self.vanilla_mgr.restore_all()
        self._populate_vanilla_vehicles_list()
        res_summary = f"Restauración masiva completada:\n• {s_cnt} vehículo(s) restaurados con éxito al 100% de Steam.\n"
        if f_cnt > 0:
            res_summary += f"• {f_cnt} error(es).\n"
        res_summary += "\n" + "\n".join(msgs[:15])
        messagebox.showinfo("Restauración Masiva Finalizada", res_summary)

    def cmd_open_vanilla_restore_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Vanilla Sound Restore Manager — Automobilista 2")
        dlg.geometry("860x520")
        dlg.minsize(700, 420)
        dlg.configure(bg=BG_DARK)
        dlg.transient(self)
        dlg.grab_set()

        top_f = ttk.Frame(dlg, style="Panel.TFrame", padding=14)
        top_f.pack(fill="x")
        ttk.Label(
            top_f,
            text="🛡️  Restore Original Steam Vanilla Car Sounds",
            style="HeaderTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            top_f,
            text="Restore any modified official car back to its original Steam files in one click.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        mid_f = ttk.Frame(dlg, style="Panel.TFrame", padding=14)
        mid_f.pack(fill="both", expand=True)

        cols = ("veh", "path", "date", "sha")
        tree = ttk.Treeview(mid_f, columns=cols, show="headings", selectmode="browse")
        tree.heading("veh", text="Official Vehicle")
        tree.heading("path", text="Game File")
        tree.heading("date", text="Backup Date")
        tree.heading("sha", text="Original SHA256")
        tree.column("veh", width=240)
        tree.column("path", width=280)
        tree.column("date", width=150)
        tree.column("sha", width=140)

        vsb = ttk.Scrollbar(mid_f, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def refresh_tree() -> None:
            for item in tree.get_children():
                tree.delete(item)
            mod_list = self.vanilla_mgr.get_modified_vehicles()
            if not mod_list:
                tree.insert("", "end", values=("(No modified vehicles — All factory original)", "—", "—", "—"))
                return
            for v in mod_list:
                sha_short = (v.original_sha256[:12] + "...") if v.original_sha256 else ".orig file"
                date_str = (v.backup_timestamp[:19].replace("T", " ")) if v.backup_timestamp else "Available"
                tree.insert(
                    "", "end", iid=v.vehicle_id, values=(v.display_name, v.bank_path.name, date_str, sha_short)
                )

        refresh_tree()

        bot_f = ttk.Frame(dlg, style="Panel.TFrame", padding=14)
        bot_f.pack(fill="x")

        def do_restore_selected() -> None:
            sel = tree.selection()
            if not sel or sel[0].startswith("("):
                messagebox.showwarning(
                    "No Selection", "Please select a vehicle from the list to restore.", parent=dlg
                )
                return
            veh_id = sel[0]
            veh = self.vanilla_mgr.get_vehicle_by_id(veh_id)
            if veh:
                ok, msg = self.vanilla_mgr.restore_vehicle(veh)
                if ok:
                    messagebox.showinfo("Restored", msg, parent=dlg)
                    refresh_tree()
                    self._populate_vanilla_vehicles_list()
                else:
                    messagebox.showerror("Error", msg, parent=dlg)

        def do_restore_all() -> None:
            mod_list = self.vanilla_mgr.get_modified_vehicles()
            if not mod_list:
                messagebox.showinfo("No Modified Vehicles", "There are no modified vanilla vehicles.", parent=dlg)
                return
            if not messagebox.askyesno(
                "Confirm Full Restore",
                "Do you want to restore ALL modified vanilla vehicles back to their original Steam state?",
                parent=dlg,
            ):
                return
            s_cnt, f_cnt, msgs = self.vanilla_mgr.restore_all()
            refresh_tree()
            self._populate_vanilla_vehicles_list()
            messagebox.showinfo("Done", f"Successfully restored {s_cnt} vehicle(s).", parent=dlg)

        ttk.Button(
            bot_f,
            text="🔄 Restore Selected",
            style="Primary.TButton",
            command=do_restore_selected,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(bot_f, text="🛡️ Restore ALL", command=do_restore_all).pack(side="left", padx=(0, 8))
        ttk.Button(bot_f, text="Close", command=dlg.destroy).pack(side="right")

    # ========================================================================
    # Acciones de Pestaña 5 (Compilación en segundo plano y Exportación / Inyección)
    # ========================================================================
    def cmd_pick_output_dir(self) -> None:
        d = filedialog.askdirectory(title="Seleccionar carpeta de salida para el Mod")
        if d:
            self.out_dir_var.set(d)

    def cmd_run_export(self) -> None:
        if self.target_mode_var.get() == "vanilla":
            self._run_inject_vanilla()
            return

        if self.is_direct_ac_mode:
            if not self.source_lib or not self.source_lib.fmod_bank:
                messagebox.showwarning(
                    "Falta Banco de Assetto Corsa",
                    "Por favor carga un archivo .bank de Assetto Corsa en la Pestaña 1 antes de compilar.",
                )
                self.notebook.select(self.tab_source)
                return
        else:
            if not self.donor_bank:
                messagebox.showwarning(
                    "Falta Banco Donante AMS2",
                    "Por favor carga un archivo .bank donante de Automobilista 2 en la Pestaña 2 antes de compilar.",
                )
                self.notebook.select(self.tab_donor)
                return

        csd_to_export: CsdbinDocument | None = None
        if self.auto_link_csd_var.get():
            if self.csd_doc is not None:
                csd_to_export = self._collect_visual_into_doc()
            else:
                ans = messagebox.askyesno(
                    "Sin archivo .csdbin cargado",
                    "No has abierto ningún archivo .csdbin en la Pestaña 4.\n\n"
                    "¿Deseas compilar únicamente el archivo .bank y el reporte JSON de GUIDs?",
                )
                if not ans:
                    self.notebook.select(self.tab_csdbin)
                    return

        out_root = Path(self.out_dir_var.get().strip())
        mod_id = self.mod_id_var.get().strip() or "custom_sound"
        veh_name = self.veh_folder_var.get().strip() or mod_id
        standalone = self.standalone_var.get()
        sync_pitch = self.sync_pitch_var.get()
        codec_mode = FSB5_MODE_PCM16 if self.codec_choice_var.get() == "PCM16" else FSB5_MODE_FADPCM
        tcut_cfg = self._collect_throttle_cut_config()
        bands_to_export = list(self.rpm_bands) if self.rpm_bands else None
        if self.mute_all_pops_var.get() and bands_to_export:
            clean_bands: list[RpmBandConfig] = []
            for b in bands_to_export:
                is_tc_band = getattr(b, "load_role", "") in ("pops", "throttle_cut")
                if is_tc_band:
                    clean_bands.append(
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
                            rpm_min=0.0,
                            rpm_max=0.0,
                            fade_in_end=0.0,
                            fade_out_start=0.0,
                            orig_rpm_min=b.orig_rpm_min,
                            orig_rpm_max=b.orig_rpm_max,
                            orig_fade_in_end=b.orig_fade_in_end,
                            orig_fade_out_start=b.orig_fade_out_start,
                            fade_in_curv_guid=b.fade_in_curv_guid,
                            fade_out_curv_guid=b.fade_out_curv_guid,
                            bus_rpm_cutoff=b.bus_rpm_cutoff,
                            is_muted_bus=True,
                            is_fill_layer=b.is_fill_layer,
                        )
                    )
                else:
                    clean_bands.append(b)
            bands_to_export = clean_bands
        is_direct = self.is_direct_ac_mode

        self._set_import_buttons_state("disabled")
        self.progress.start(12)
        mode_label = (
            "Transmutación Directa (Sin Donante - 100% Fiel AC)"
            if is_direct
            else f"Mapeo Donante ({self.codec_choice_var.get()})"
        )
        self.status_var.set(f"⏳ Compilando banco FMOD ({mode_label}) y empaquetando .csdbin...")
        self._append_log("\n----------------------------------------------------------------------")
        self._append_log(f"Iniciando compilación del paquete '{mod_id}' [{mode_label}]...")

        def worker() -> None:
            try:
                direct_bank = self.source_lib.fmod_bank if is_direct else None
                donor_bank = None if is_direct else self.donor_bank
                summary = export_ams2_mod_package(
                    output_root=out_root,
                    mod_identifier=mod_id,
                    vehicle_folder_name=veh_name,
                    donor=donor_bank,
                    direct_ac_bank=direct_bank,
                    mappings=self.mappings,
                    library=self.source_lib,
                    csd_doc=csd_to_export,
                    standalone_guids=standalone,
                    sync_autopitch=sync_pitch,
                    output_codec_mode=codec_mode,
                    force_sample_rate=48000,
                    auto_link_csdbin=True,
                    rpm_bands=bands_to_export,
                    throttle_cut=tcut_cfg,
                    mute_pops=self.mute_all_pops_var.get(),
                    invert_spatial_cone=self.invert_cone_var.get(),
                )
                self.dispatch_to_ui(self._on_export_success, summary)
            except Exception as exc:
                err_msg = str(exc)
                self.dispatch_to_ui(self._on_export_error, err_msg)

        threading.Thread(target=worker, daemon=True).start()

    def _on_export_success(self, summary: ModExportSummary) -> None:
        self.progress.stop()
        self._set_import_buttons_state("normal")
        if self.csd_doc is not None:
            self._populate_visual_csd_from_doc()
            self._sync_visual_to_raw_csd_editor()

        b_res = summary.bank_result
        size_mb = summary.bank_file.stat().st_size / (1024 * 1024)
        self._append_log(f"✔ Banco Jugador FMOD generado: {summary.bank_file} ({size_mb:.2f} MB)")
        if summary.ai_bank_file and summary.ai_bank_file.is_file():
            ai_size_mb = summary.ai_bank_file.stat().st_size / (1024 * 1024)
            self._append_log(f"✔ Banco IA + TV Replays generado: {summary.ai_bank_file} ({ai_size_mb:.2f} MB)")
        if self.is_direct_ac_mode:
            self._append_log(
                f"  ⚡ Modo: {b_res.codec_used} | {b_res.replaced_slots_count} muestras de AC preservadas al 100%"
            )
        else:
            self._append_log(
                f"  Slots reemplazados: {b_res.replaced_slots_count} | Conservados: {b_res.kept_slots_count} | Códec: {b_res.codec_used}"
            )
        if b_res.throttle_cut_injected_events > 0:
            self._append_log(
                f"  ⚡ Throttle Cut: Inyectado activamente en {b_res.throttle_cut_injected_events} evento(s) de motor"
            )
        self._append_log(f"  Bank GUID: {b_res.new_bank_guid} (Tabla de rutas STDT incrustada)")
        for role, gstr in b_res.role_to_event_guid.items():
            ev_path = b_res.role_to_event_path.get(role, "")
            self._append_log(f"  Evento {role:10s} -> {ev_path}  ({gstr})")
        if summary.csdbin_file:
            self._append_log(f"✔ Archivo .csdbin compilado: {summary.csdbin_file}")
            self._append_log(f"✔ Respaldo .csd (OMVT):      {summary.csd_text_file}")
        self._append_log(f"✔ Guía de instalación:       {summary.readme_file}")

        self.status_var.set(f"✅ Export succeeded in {summary.output_root}!")
        if self.open_folder_after_var.get() and os.name == "nt":
            try:
                subprocess.Popen(["explorer", str(summary.output_root)])
            except Exception:
                pass
        messagebox.showinfo(
            "Mod Package Generated Successfully!",
            f"The FMOD bank and vehicle configuration were compiled successfully in:\n\n{summary.output_root}",
        )

    def _on_export_error(self, err_msg: str) -> None:
        self.progress.stop()
        self._set_import_buttons_state("normal")
        self._append_log(f"❌ BUILD ERROR: {err_msg}")
        self.status_var.set("❌ Error during compilation.")
        messagebox.showerror("Export Error", err_msg)

    def _run_inject_vanilla(self) -> None:
        veh = self.vanilla_selected_vehicle
        if not veh:
            messagebox.showwarning(
                "No Vanilla Vehicle Selected",
                "Please select an official / vanilla vehicle from the list before importing.",
            )
            return

        if self.is_direct_ac_mode:
            if not self.source_lib or not self.source_lib.fmod_bank:
                messagebox.showwarning(
                    "Missing Assetto Corsa Bank",
                    "Please select an Assetto Corsa .bank file in Step 1 before importing.",
                )
                return
        else:
            if not self.donor_bank:
                ans = messagebox.askyesno(
                    "Load Donor Bank Automatically",
                    f"No donor bank is loaded.\n\n"
                    f"Do you want to load '{veh.display_name}' original bank as donor?",
                )
                if ans:
                    self.cmd_load_vanilla_as_donor()
                return

        ans_confirm = messagebox.askyesno(
            "Confirm Vanilla Sound Import",
            f"Do you want to import the new sound into the official vehicle:\n\n"
            f"'{veh.display_name}'?\n\n"
            f"• File: {veh.bank_path}\n"
            f"• Mode: {'Direct AC Transmutation (100% Faithful)' if self.is_direct_ac_mode else 'Donor Mapping'}\n\n"
            f"🛡️ A safe backup (SHA256) will be created automatically before modifying anything.\n"
            f"You can restore the original Steam sound at any time with one click.",
        )
        if not ans_confirm:
            return

        self._set_import_buttons_state("disabled")
        self.progress.start(12)
        mode_label = (
            "Transmutación Directa (Sin Donante - 100% Fiel AC)"
            if self.is_direct_ac_mode
            else f"Mapeo Donante ({self.codec_choice_var.get()})"
        )
        self.status_var.set(f"⏳ Inyectando sonido porteado en '{veh.display_name}' con respaldo seguro...")
        self._append_log("\n----------------------------------------------------------------------")
        self._append_log(f"Iniciando inyección en vehículo vanilla '{veh.display_name}' [{mode_label}]...")

        sync_ai = self.vanilla_sync_ai_var.get()
        mute_pops = self.vanilla_mute_pops_var.get()
        invert_cone = self.invert_cone_var.get()
        tcut_cfg = self._collect_throttle_cut_config()
        sync_pitch = self.sync_pitch_var.get()
        codec_mode = FSB5_MODE_PCM16 if self.codec_choice_var.get() == "PCM16" else FSB5_MODE_FADPCM
        is_direct = self.is_direct_ac_mode

        bands_to_export = list(self.rpm_bands) if self.rpm_bands else None
        if mute_pops and bands_to_export:
            clean_bands: list[RpmBandConfig] = []
            for b in bands_to_export:
                is_tc_band = getattr(b, "load_role", "") in ("pops", "throttle_cut")
                if is_tc_band:
                    clean_bands.append(
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
                            rpm_min=0.0,
                            rpm_max=0.0,
                            fade_in_end=0.0,
                            fade_out_start=0.0,
                            orig_rpm_min=b.orig_rpm_min,
                            orig_rpm_max=b.orig_rpm_max,
                            orig_fade_in_end=b.orig_fade_in_end,
                            orig_fade_out_start=b.orig_fade_out_start,
                            fade_in_curv_guid=b.fade_in_curv_guid,
                            fade_out_curv_guid=b.fade_out_curv_guid,
                            bus_rpm_cutoff=b.bus_rpm_cutoff,
                            is_muted_bus=True,
                            is_fill_layer=b.is_fill_layer,
                        )
                    )
                else:
                    clean_bands.append(b)
            bands_to_export = clean_bands

        def worker() -> None:
            try:
                self.vanilla_mgr.load_vehicle_events_info(veh)

                bank_bytes: bytes
                if is_direct:
                    assert self.source_lib and self.source_lib.fmod_bank
                    veh_backup_folder = self.vanilla_mgr.backup_dir / veh.vehicle_id
                    native_p = (
                        veh_backup_folder / veh.bank_path.name
                        if (veh_backup_folder / veh.bank_path.name).is_file()
                        else veh.bank_path
                    )
                    direct_res = build_direct_ported_bank_from_ac(
                        ac_bank=self.source_lib.fmod_bank,
                        mod_identifier=veh.car_name,
                        standalone_guids=False,
                        target_role_guids=veh.event_guids,
                        target_role_paths=veh.event_paths,
                        target_bank_guid=veh.bank_guid or None,
                        target_param_names=veh.target_parameters,
                        target_ctrl_params=veh.ctrl_parameters,
                        target_role_buses=getattr(veh, "ai_event_buses", None),
                        rpm_bands=bands_to_export,
                        mute_secondary_pops=mute_pops,
                        native_bank_path=native_p,
                        invert_spatial_cone=invert_cone,
                        mappings=self.mappings,
                    )
                    bank_bytes = direct_res.bank_bytes
                else:
                    assert self.donor_bank
                    eff_mappings = self.mappings
                    if mute_pops and self.mappings:
                        eff_mappings = []
                        for i, m in enumerate(self.mappings):
                            sl = self.donor_bank.slots[m.donor_global_index] if 0 <= m.donor_global_index < len(self.donor_bank.slots) else None
                            if sl and (sl.load_role in ("pops", "throttle_cut") or "limiter" in sl.name.lower() or "backfire" in sl.name.lower()):
                                eff_mappings.append(SlotMappingConfig(
                                    donor_global_index=m.donor_global_index,
                                    action="silence",
                                    source_index=None,
                                    gain_db=-96.0,
                                    match_reason="Silenciado para Replay / IA limpia",
                                ))
                            else:
                                eff_mappings.append(m)
                    donor_res = build_ported_bank(
                        donor=self.donor_bank,
                        library=self.source_lib,
                        mappings=eff_mappings,
                        mod_identifier=veh.car_name,
                        standalone_guids=False,
                        sync_autopitch=sync_pitch,
                        output_codec_mode=codec_mode,
                        rpm_bands=bands_to_export,
                        throttle_cut=tcut_cfg,
                    )
                    bank_bytes = donor_res.bank_bytes

                # 2. Configurar banco de la IA si está marcado y existe
                # El _AI.bank debe conservar la GUID de banco y las GUIDs de eventos nativas de la IA
                # (MAIN, MAIN_LIVE, ext_AI, int_AI). Copiar el banco del jugador duplicaba la GUID de banco
                # (FMOD rechaza el segundo banco) y dejaba a los rivales y repeticiones sin eventos -> mudos.
                ai_bytes: bytes | None = None
                if sync_ai and veh.ai_bank_path and veh.ai_event_guids:
                    if is_direct:
                        ai_direct_res = build_direct_ported_bank_from_ac(
                            ac_bank=self.source_lib.fmod_bank,
                            mod_identifier=f"{veh.car_name}_AI",
                            standalone_guids=False,
                            target_role_guids=veh.ai_event_guids,
                            target_role_paths=veh.ai_event_paths,
                            target_bank_guid=veh.ai_bank_guid or None,
                            target_role_buses=getattr(veh, "ai_event_buses", None),
                            rpm_bands=bands_to_export,
                            is_ai_bank=True,
                            mute_secondary_pops=mute_pops,
                            invert_spatial_cone=False,
                            mappings=self.mappings,
                        )
                        ai_bytes = ai_direct_res.bank_bytes
                    else:
                        veh_backup_folder = self.vanilla_mgr.backup_dir / veh.vehicle_id
                        native_ai_p = (
                            veh_backup_folder / veh.ai_bank_path.name
                            if (veh_backup_folder / veh.ai_bank_path.name).is_file()
                            else veh.ai_bank_path
                        )
                        if native_ai_p.is_file() and self.source_lib:
                            ai_donor = FmodBank(native_ai_p)
                            ai_donor_res = build_ported_bank(
                                donor=ai_donor,
                                mappings=auto_map_samples(ai_donor, self.source_lib),
                                library=self.source_lib,
                                mod_identifier=f"{veh.car_name}_AI",
                                standalone_guids=False,
                                sync_autopitch=sync_pitch,
                                output_codec_mode=codec_mode,
                            )
                            ai_bytes = ai_donor_res.bank_bytes

                # 3. Optimizar .csdbin suelto para Chase Cam (3ra persona) si existe
                csd_bytes: bytes | None = None
                if veh.csdbin_path and veh.csdbin_path.is_file():
                    try:
                        csd_doc = CsdbinDocument.from_file(veh.csdbin_path)
                        if "engine_placement" in csd_doc.links:
                            raw_blob = csd_doc.links["engine_placement"].replace("blob:", "")
                            bdata = bytearray(bytes.fromhex(raw_blob))
                            num_slots = len(bdata) // 32
                            for s in (3, 4):
                                if s < num_slots:
                                    offset = s * 32
                                    vol = struct.unpack_from("<f", bdata, offset)[0]
                                    struct.pack_into("<f", bdata, offset, min(1.0, max(vol * 1.35, 0.40)))
                                    struct.pack_into("<f", bdata, offset + 20, 8.0)
                            csd_doc.links["engine_placement"] = f"blob:{bdata.hex()}"
                            csd_bytes = csd_doc.save_csdbin()
                    except Exception:
                        pass

                ok, msg = self.vanilla_mgr.install_sound(
                    vehicle=veh,
                    bank_bytes=bank_bytes,
                    ai_bank_bytes=ai_bytes,
                    csdbin_bytes=csd_bytes,
                )
                if not ok:
                    raise RuntimeError(msg)

                self.dispatch_to_ui(self._on_inject_vanilla_success, veh, len(bank_bytes), msg)
            except Exception as exc:
                err_str = str(exc)
                self.dispatch_to_ui(self._on_export_error, f"Fallo en inyección vanilla: {err_str}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_inject_vanilla_success(self, veh: VanillaVehicle, byte_count: int, msg: str) -> None:
        self.progress.stop()
        self._set_import_buttons_state("normal")
        self._append_log(f"✔ ¡Inyección exitosa en vehículo vanilla '{veh.display_name}'!")
        self._append_log(f"  Ruta en juego: {veh.bank_path} ({byte_count / (1024*1024):.2f} MB)")
        self._append_log(f"  Respaldo seguro original: {veh.original_sha256}")
        self._append_log(f"  Estado de restauración: Listo para restaurar con 1 clic sin Steam.")
        self.status_var.set(f"✅ Sound successfully imported into '{veh.display_name}'!")

        self._populate_vanilla_vehicles_list(select_id=veh.vehicle_id)
        messagebox.showinfo(
            "Sound Successfully Imported into Official Vehicle!",
            f"The new sound has been installed and activated for '{veh.display_name}'.\n\n"
            f"• File: {veh.bank_path.name}\n"
            f"• Size: {byte_count:,} bytes\n"
            f"• Original backup: Protected (SHA256)\n\n"
            f"This vehicle will now use the imported sound in Automobilista 2.\n"
            f"You can restore the original Steam sound at any time from this tool "
            f"without needing to verify game files in Steam.",
        )


def launch_gui() -> None:
    enable_windows_dpi_awareness()
    app = SoundPorterApp()
    app.mainloop()
