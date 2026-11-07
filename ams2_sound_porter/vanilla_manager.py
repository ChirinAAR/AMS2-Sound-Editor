"""Gestor de Vehículos Vanilla de Automobilista 2.

Permite auditar, catalogar, modificar sonidos y restaurar al estado original de Steam
los 174+ vehículos base/oficiales de Automobilista 2 sin requerir comprobación ni
reparación de archivos en la plataforma Steam.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import winreg
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .fmod_bank import FmodBank, bytes_to_guid_str, guid_str_to_bytes, iter_riff_chunks
from .fmod_stdt import parse_stdt_all_paths


DEFAULT_AMS2_SEARCH_PATHS = [
    Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2"),
    Path(r"C:\Program Files (x86)\Steam\steamapps\common\Automobilista 2"),
    Path(r"C:\SteamLibrary\steamapps\common\Automobilista 2"),
    Path(r"E:\SteamLibrary\steamapps\common\Automobilista 2"),
]


def detect_ams2_root() -> Path | None:
    """Detecta la ruta de instalación de Automobilista 2 en el sistema."""
    # 1. Registro de Windows: Desinstalador Steam App 1066890
    for hkey in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for subkey in (
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 1066890",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 1066890",
        ):
            try:
                with winreg.OpenKey(hkey, subkey) as k:
                    val, _ = winreg.QueryValueEx(k, "InstallLocation")
                    if val and Path(val).is_dir():
                        p = Path(val)
                        if (p / "Audio" / "AMS2" / "Vehicles" / "Sort").is_dir():
                            return p
            except Exception:
                pass

    # 2. Rutas conocidas comunes
    for p in DEFAULT_AMS2_SEARCH_PATHS:
        if p.is_dir() and (p / "Audio" / "AMS2" / "Vehicles" / "Sort").is_dir():
            return p

    return None


def calculate_file_sha256(path: Path | str) -> str:
    """Calcula el hash SHA256 de un archivo en disco."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class VanillaVehicle:
    """Información completa de un vehículo vanilla / base de Automobilista 2."""
    vehicle_id: str                      # Clave normalizada única (e.g. 'super_v8_superv8_set1')
    display_name: str                    # Nombre visual para la UI (e.g. 'Super V8 — Set 1')
    brand: str                           # Marca o categoría (e.g. 'Super_V8', 'BMW', 'Porsche')
    car_name: str                        # Nombre del auto / banco (e.g. 'SuperV8_Set1')
    bank_path: Path                      # Ruta absoluta del .bank del jugador
    ai_bank_path: Path | None            # Ruta absoluta del _AI.bank o None
    relative_audio_path: str             # Ruta interna (e.g. 'Audio/AMS2/Vehicles/Sort/S/Super_V8/SuperV8_Set1')
    slots_count: int = 0
    event_roles: list[str] = field(default_factory=list)
    event_guids: dict[str, str] = field(default_factory=dict)       # role -> guid_str
    event_paths: dict[str, str] = field(default_factory=dict)       # role -> path
    bank_guid: str = ""
    ai_bank_guid: str = ""
    ai_event_roles: list[str] = field(default_factory=list)
    ai_event_guids: dict[str, str] = field(default_factory=dict)    # role -> guid_str
    ai_event_paths: dict[str, str] = field(default_factory=dict)    # role -> path
    is_modified: bool = False            # True si está usando sonido porteado
    has_backup: bool = False             # True si existe respaldo seguro original
    original_sha256: str | None = None
    original_ai_sha256: str | None = None
    current_sha256: str | None = None
    backup_timestamp: str | None = None
    csdbin_path: Path | None = None      # Archivo .csdbin suelto en Vehicles/<carpeta> si existe
    target_parameters: list[str] = field(default_factory=list) # Parámetros detectados en el banco nativo (load, rpm, etc.)
    ctrl_parameters: set[str] = field(default_factory=set)     # Parámetros activos en CTRS/CTRL (load, throttle, rpm)



class VanillaVehicleManager:
    """Gestor integral de vehículos base de Automobilista 2 con respaldo y restauración garantizada."""

    def __init__(
        self,
        ams2_root: Path | str | None = None,
        backup_dir: Path | str | None = None,
    ) -> None:
        if ams2_root is None:
            detected = detect_ams2_root()
            self.ams2_root = detected if detected else Path(r"D:\SteamLibrary\steamapps\common\Automobilista 2")
        else:
            self.ams2_root = Path(ams2_root)

        if backup_dir is None:
            # Directorio de respaldo local seguro dentro del paquete
            app_dir = Path(__file__).resolve().parent.parent
            self.backup_dir = app_dir / "backups" / "vanilla"
        else:
            self.backup_dir = Path(backup_dir)

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.backup_dir / "vanilla_backups.json"
        self._vehicles_cache: list[VanillaVehicle] = []
        self._manifest_cache: dict[str, Any] = {}
        self._master_strings_cache: dict[str, str] | None = None
        self._load_manifest()

    def get_master_strings(self) -> dict[str, str]:
        """Carga y cachea el diccionario de GUID -> path desde MasterBank.strings.bank de AMS2."""
        if self._master_strings_cache is not None:
            return self._master_strings_cache
        master_path = self.ams2_root / "Audio" / "MasterBank.strings.bank"
        if not master_path.is_file():
            sys_root = detect_ams2_root()
            if sys_root and (sys_root / "Audio" / "MasterBank.strings.bank").is_file():
                master_path = sys_root / "Audio" / "MasterBank.strings.bank"

        out: dict[str, str] = {}
        if master_path.is_file():
            try:
                b_master = FmodBank(master_path)
                if b_master.stdt_bytes:
                    raw_paths = parse_stdt_all_paths(b_master.stdt_bytes)
                    for k_bytes, path_str in raw_paths.items():
                        out[bytes_to_guid_str(k_bytes).lower()] = path_str
            except Exception:
                pass
        self._master_strings_cache = out
        return out

    def _load_manifest(self) -> None:
        """Carga el manifiesto de respaldos."""
        if self.manifest_path.is_file():
            try:
                self._manifest_cache = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                self._manifest_cache = {}
        else:
            self._manifest_cache = {}

    def _save_manifest(self) -> None:
        """Guarda el manifiesto de respaldos en disco."""
        try:
            self.manifest_path.write_text(
                json.dumps(self._manifest_cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    @property
    def sort_root(self) -> Path:
        return self.ams2_root / "Audio" / "AMS2" / "Vehicles" / "Sort"

    def is_ams2_valid(self) -> bool:
        """Verifica que la carpeta configurada contenga la estructura de audio de AMS2."""
        return self.sort_root.is_dir()

    def scan_catalog(self, force: bool = False) -> list[VanillaVehicle]:
        """Escanea todos los bancos de vehículos vanilla en Audio/AMS2/Vehicles/Sort."""
        if self._vehicles_cache and not force:
            return self._vehicles_cache

        if not self.is_ams2_valid():
            return []

        vehicles: list[VanillaVehicle] = []
        player_banks = sorted([p for p in self.sort_root.rglob("*.bank") if not p.name.endswith("_AI.bank")])

        for bank_path in player_banks:
            try:
                rel = bank_path.relative_to(self.sort_root)
                parts = rel.parts
                if len(parts) >= 3:
                    brand = parts[1]
                    car_name = parts[-1].replace(".bank", "")
                elif len(parts) == 2:
                    brand = parts[0]
                    car_name = parts[1].replace(".bank", "")
                else:
                    brand = "General"
                    car_name = rel.stem

                # Formatear display_name legible
                brand_clean = brand.replace("_", " ").strip()
                car_clean = car_name.replace("_", " ").strip()
                if brand_clean.lower() in car_clean.lower():
                    display_name = car_clean
                else:
                    display_name = f"{brand_clean} — {car_clean}"

                veh_id = f"{brand.lower()}_{car_name.lower()}"

                ai_candidate = bank_path.parent / f"{bank_path.stem}_AI.bank"
                ai_bank_path = ai_candidate if ai_candidate.is_file() else None

                # Comprobar estado de respaldo en manifiesto y archivo .orig
                orig_file = bank_path.parent / f"{bank_path.name}.orig"
                manifest_entry = self._manifest_cache.get(veh_id)

                has_backup = False
                original_sha = None
                backup_time = None
                is_modified = False

                if manifest_entry:
                    has_backup = True
                    original_sha = manifest_entry.get("original_sha256")
                    backup_time = manifest_entry.get("backup_timestamp")
                    is_modified = manifest_entry.get("is_modified", False)
                elif orig_file.is_file():
                    has_backup = True
                    is_modified = True

                rel_audio_path = f"Audio/AMS2/Vehicles/Sort/{rel.as_posix()[:-5]}"

                v = VanillaVehicle(
                    vehicle_id=veh_id,
                    display_name=display_name,
                    brand=brand,
                    car_name=car_name,
                    bank_path=bank_path,
                    ai_bank_path=ai_bank_path,
                    relative_audio_path=rel_audio_path,
                    is_modified=is_modified,
                    has_backup=has_backup,
                    original_sha256=original_sha,
                    backup_timestamp=backup_time,
                )
                vehicles.append(v)
            except Exception:
                continue

        # Ordenar por display_name
        vehicles.sort(key=lambda x: x.display_name.lower())
        self._vehicles_cache = vehicles
        return vehicles

    def load_vehicle_events_info(self, vehicle: VanillaVehicle) -> None:
        """Carga y analiza los slots, eventos y GUIDs del banco del vehículo si aún no están cargados."""
        if vehicle.csdbin_path is None:
            veh_dir = self.ams2_root / "Vehicles"
            if veh_dir.is_dir():
                c_clean = vehicle.car_name.lower().replace("_", "").replace("-", "")
                b_clean = vehicle.brand.lower().replace("_", "").replace("-", "")
                for csd in veh_dir.rglob("*.csdbin"):
                    rel = str(csd.relative_to(veh_dir)).lower().replace("\\", "/").replace("_", "").replace("-", "")
                    if c_clean in rel or (b_clean in rel and len(b_clean) > 3):
                        vehicle.csdbin_path = csd
                        break

        if vehicle.event_roles and vehicle.target_parameters and (vehicle.ai_event_roles or not vehicle.ai_bank_path):
            return
        if not vehicle.bank_path.is_file():
            return
        try:
            master_strings = self.get_master_strings()
            b = FmodBank(vehicle.bank_path)
            vehicle.slots_count = len(b.slots)
            vehicle.bank_guid = bytes_to_guid_str(b.bank_guid) if b.bank_guid else ""
            roles: set[str] = set()
            guids: dict[str, str] = {}
            paths: dict[str, str] = {}
            int_count = 0
            for ev in b.events:
                ev_p = master_strings.get(ev.guid_str.lower(), ev.path or "")
                p_low = ev_p.lower()
                role = ev.role or "Unknown"

                # Clasificación precisa por ruta en master_strings para evitar solapamientos y eventos perdidos
                if "starter" in p_low:
                    role = "Starter"
                elif "starting" in p_low:
                    role = "Starting"
                elif "exhaust" in p_low:
                    role = "Exhaust"
                elif any(k in p_low for k in ("engineext", "engine_ext", "ext_engine")):
                    role = "EngineExt"
                elif any(k in p_low for k in ("engineint", "engine_int", "int_engine")):
                    int_count += 1
                    role = "EngineInt" if int_count == 1 else "AuxInt"
                elif role == "Starting" and "Starter" not in guids:
                    role = "Starter"

                if role:
                    roles.add(role)
                    if ev.guid_str and role not in guids:
                        guids[role] = ev.guid_str
                    if ev_p and role not in paths:
                        paths[role] = ev_p
                    elif ev.path and role not in paths:
                        paths[role] = ev.path
            vehicle.event_roles = sorted(list(roles))
            vehicle.event_guids = guids
            vehicle.event_paths = paths

            # Extraer nombres de parámetros nativos y controladores activos del banco de jugador (load, rpm, throttle, etc.)
            p_names: set[str] = set()
            prm_guid_to_name: dict[bytes, str] = {}
            ctrl_prms: set[str] = set()
            for cid, ps, pe, _ in iter_riff_chunks(b.raw, 12, len(b.raw)):
                if cid == b"LIST" and b.raw[ps : ps + 4] == b"PROJ":
                    proj_pay = b.raw[ps + 4 : pe]
                    def scan_prms(s: int, e: int) -> None:
                        for scid, sps, spe, _ in iter_riff_chunks(proj_pay, s, e):
                            if scid == b"PRMB" and spe - sps >= 25:
                                g = proj_pay[sps : sps + 16]
                                slen = struct.unpack_from("<H", proj_pay, sps + 21)[0]
                                name = proj_pay[sps + 23 : sps + 23 + slen].decode("latin1", "replace").rstrip("\x00")
                                p_names.add(name)
                                prm_guid_to_name[g] = name
                            elif scid == b"LIST":
                                scan_prms(sps + 4, spe)
                    scan_prms(0, len(proj_pay))

                    # Escaneo de controladores activos CTRS/CTRL
                    for scid, sps, spe, _ in iter_riff_chunks(proj_pay, 0, len(proj_pay)):
                        if scid == b"LIST" and proj_pay[sps : sps + 4] == b"CTRS":
                            ctrs = proj_pay[sps + 4 : spe]
                            for ccid, cps, cpe, _ in iter_riff_chunks(ctrs, 0, len(ctrs)):
                                if ccid == b"CTRL" and cpe - cps >= 48:
                                    cpay = ctrs[cps:cpe]
                                    for pg, pname in prm_guid_to_name.items():
                                        if pg in cpay and pname in ("load", "throttle", "rpm"):
                                            ctrl_prms.add(pname)
                    break
            vehicle.target_parameters = sorted(list(p_names))
            vehicle.ctrl_parameters = ctrl_prms

            # Analizar también banco de IA si existe
            if vehicle.ai_bank_path and vehicle.ai_bank_path.is_file():
                ai_b = FmodBank(vehicle.ai_bank_path)
                vehicle.ai_bank_guid = bytes_to_guid_str(ai_b.bank_guid) if ai_b.bank_guid else ""
                ai_roles: set[str] = set()
                ai_guids: dict[str, str] = {}
                ai_paths: dict[str, str] = {}
                for ev in ai_b.events:
                    p = master_strings.get(ev.guid_str.lower(), ev.path or "")
                    p_low = p.lower()
                    if "main_live" in p_low or "_live" in p_low:
                        role = "ai_engine_live"
                    elif "main" in p_low:
                        role = "ai_engine_ext"
                    elif "engineext" in p_low:
                        role = "ai_engine_ext"
                    elif "engineint" in p_low:
                        role = "ai_engine_int"
                    elif ev.role == "EngineExt":
                        role = "ai_engine_ext"
                    elif ev.role == "EngineInt":
                        role = "ai_engine_int"
                    else:
                        role = ev.role or f"AI_Event_{ev.index}"
                    ai_roles.add(role)
                    if ev.guid_str:
                        ai_guids[role] = ev.guid_str
                    if p:
                        ai_paths[role] = p
                vehicle.ai_event_roles = sorted(list(ai_roles))
                vehicle.ai_event_guids = ai_guids
                vehicle.ai_event_paths = ai_paths
        except Exception:
            pass

    def get_vehicle_by_id(self, vehicle_id: str) -> VanillaVehicle | None:
        """Busca un vehículo por su identificador normalizado."""
        for v in self.scan_catalog():
            if v.vehicle_id == vehicle_id:
                return v
        return None

    def find_vehicles(self, query: str) -> list[VanillaVehicle]:
        """Busca vehículos por texto en su ID, nombre, marca o modelo."""
        q = query.lower().strip()
        if not q:
            return list(self.scan_catalog())
        return [
            v
            for v in self.scan_catalog()
            if q in v.vehicle_id
            or q in v.display_name.lower()
            or q in v.brand.lower()
            or q in v.car_name.lower()
        ]

    def create_backup(self, vehicle: VanillaVehicle) -> tuple[bool, str]:
        """Crea una copia de seguridad garantizada y verificada del banco original del vehículo.

        NUNCA sobreescribe un respaldo existente para evitar guardar un banco ya modificado.
        """
        if not vehicle.bank_path.is_file():
            return False, f"El archivo del vehículo no existe: {vehicle.bank_path}"

        veh_backup_folder = self.backup_dir / vehicle.vehicle_id
        veh_backup_folder.mkdir(parents=True, exist_ok=True)
        backup_file = veh_backup_folder / vehicle.bank_path.name
        orig_file = vehicle.bank_path.parent / f"{vehicle.bank_path.name}.orig"

        manifest_entry = self._manifest_cache.get(vehicle.vehicle_id)

        # Si ya existe un respaldo seguro en disco y en el manifiesto, protegerlo
        if backup_file.is_file() and manifest_entry:
            vehicle.has_backup = True
            vehicle.original_sha256 = manifest_entry.get("original_sha256")
            return True, f"Respaldo ya existente y protegido (SHA256: {vehicle.original_sha256[:12]}...)"

        try:
            curr_sha = calculate_file_sha256(vehicle.bank_path)
            file_size = vehicle.bank_path.stat().st_size

            # 1. Copiar al almacén seguro
            shutil.copy2(vehicle.bank_path, backup_file)

            # 2. Copiar también como .orig junto al archivo (estándar visible para simracers)
            if not orig_file.is_file():
                shutil.copy2(vehicle.bank_path, orig_file)

            # 3. Si existe _AI.bank, respaldarlo también
            backup_ai_name = None
            curr_ai_sha = None
            if vehicle.ai_bank_path and vehicle.ai_bank_path.is_file():
                curr_ai_sha = calculate_file_sha256(vehicle.ai_bank_path)
                ai_backup_file = veh_backup_folder / vehicle.ai_bank_path.name
                shutil.copy2(vehicle.ai_bank_path, ai_backup_file)
                ai_orig_file = vehicle.ai_bank_path.parent / f"{vehicle.ai_bank_path.name}.orig"
                if not ai_orig_file.is_file():
                    shutil.copy2(vehicle.ai_bank_path, ai_orig_file)
                backup_ai_name = vehicle.ai_bank_path.name

            # 3b. Si existe archivo .csdbin suelto en Vehicles/, respaldarlo también
            backup_csd_name = None
            if vehicle.csdbin_path and vehicle.csdbin_path.is_file():
                csd_backup_file = veh_backup_folder / vehicle.csdbin_path.name
                shutil.copy2(vehicle.csdbin_path, csd_backup_file)
                csd_orig_file = vehicle.csdbin_path.parent / f"{vehicle.csdbin_path.name}.orig"
                if not csd_orig_file.is_file():
                    shutil.copy2(vehicle.csdbin_path, csd_orig_file)
                backup_csd_name = vehicle.csdbin_path.name

            # 4. Registrar en el manifiesto
            now_str = datetime.now().isoformat()
            self._manifest_cache[vehicle.vehicle_id] = {
                "vehicle_id": vehicle.vehicle_id,
                "display_name": vehicle.display_name,
                "bank_path": str(vehicle.bank_path),
                "ai_bank_path": str(vehicle.ai_bank_path) if vehicle.ai_bank_path else None,
                "csdbin_path": str(vehicle.csdbin_path) if vehicle.csdbin_path else None,
                "backup_file": str(backup_file),
                "backup_ai_file": str(veh_backup_folder / backup_ai_name) if backup_ai_name else None,
                "backup_csd_file": str(veh_backup_folder / backup_csd_name) if backup_csd_name else None,
                "original_sha256": curr_sha,
                "original_ai_sha256": curr_ai_sha,
                "original_size": file_size,
                "backup_timestamp": now_str,
                "is_modified": False,
            }
            self._save_manifest()

            vehicle.has_backup = True
            vehicle.original_sha256 = curr_sha
            vehicle.original_ai_sha256 = curr_ai_sha
            vehicle.backup_timestamp = now_str
            return True, f"Copia de seguridad creada con éxito (SHA256: {curr_sha[:12]}...)"
        except Exception as e:
            return False, f"Error al crear copia de seguridad: {e}"

    def install_sound(
        self,
        vehicle: VanillaVehicle,
        bank_bytes: bytes,
        ai_bank_bytes: bytes | None = None,
        csdbin_bytes: bytes | None = None,
    ) -> tuple[bool, str]:
        """Inyecta el nuevo sonido compilado en el vehículo vanilla, creando primero el respaldo seguro."""
        if not bank_bytes:
            return False, "Los datos binarios del banco de sonido están vacíos."

        # 1. Asegurar copia de seguridad antes de modificar nada
        ok_backup, msg_backup = self.create_backup(vehicle)
        if not ok_backup and not vehicle.has_backup:
            return False, f"Fallo al crear respaldo preventivo: {msg_backup}"

        # 2. Escribir el nuevo archivo .bank de forma segura (con archivo temporal atómico)
        target_path = vehicle.bank_path
        tmp_target = target_path.with_suffix(".bank.tmp")
        try:
            tmp_target.write_bytes(bank_bytes)
            if target_path.is_file():
                target_path.unlink()
            tmp_target.rename(target_path)

            # 3. Si se proporciona banco AI, escribirlo
            ai_msg_part = ""
            if ai_bank_bytes and vehicle.ai_bank_path:
                ai_tmp = vehicle.ai_bank_path.with_suffix(".bank.tmp")
                ai_tmp.write_bytes(ai_bank_bytes)
                if vehicle.ai_bank_path.is_file():
                    vehicle.ai_bank_path.unlink()
                ai_tmp.rename(vehicle.ai_bank_path)
                ai_msg_part = f"\n• Banco IA / TV Replay: {vehicle.ai_bank_path.name} ({len(ai_bank_bytes):,} bytes)"

            # 3b. Si se proporciona .csdbin suelto, escribirlo
            if csdbin_bytes and vehicle.csdbin_path:
                csd_tmp = vehicle.csdbin_path.with_suffix(".csdbin.tmp")
                csd_tmp.write_bytes(csdbin_bytes)
                if vehicle.csdbin_path.is_file():
                    vehicle.csdbin_path.unlink()
                csd_tmp.rename(vehicle.csdbin_path)

            # 4. Actualizar estado en el manifiesto
            if vehicle.vehicle_id in self._manifest_cache:
                self._manifest_cache[vehicle.vehicle_id]["is_modified"] = True
                self._manifest_cache[vehicle.vehicle_id]["modified_timestamp"] = datetime.now().isoformat()
                self._save_manifest()

            vehicle.is_modified = True
            return True, (
                f"¡Sonido inyectado con éxito en '{vehicle.display_name}'!\n"
                f"• Banco Jugador: {target_path.name} ({len(bank_bytes):,} bytes){ai_msg_part}\n"
                f"El vehículo ya usará este sonido en AMS2 (Jugador + IA + TV Replays). Respaldo original seguro y activo."
            )
        except Exception as e:
            if tmp_target.is_file():
                try:
                    tmp_target.unlink()
                except Exception:
                    pass
            return False, f"Error al escribir banco en el juego: {e}"

    def restore_vehicle(self, vehicle: VanillaVehicle) -> tuple[bool, str]:
        """Restaura un vehículo vanilla modificado a su estado 100% original de Steam sin verificar archivos."""
        veh_backup_folder = self.backup_dir / vehicle.vehicle_id
        backup_file = veh_backup_folder / vehicle.bank_path.name
        orig_file = vehicle.bank_path.parent / f"{vehicle.bank_path.name}.orig"

        manifest_entry = self._manifest_cache.get(vehicle.vehicle_id)

        source_to_restore: Path | None = None
        if backup_file.is_file():
            source_to_restore = backup_file
        elif orig_file.is_file():
            source_to_restore = orig_file

        if not source_to_restore:
            return False, f"No se encontró copia de seguridad original para '{vehicle.display_name}'."

        try:
            # 1. Verificar hash del respaldo si está registrado
            expected_sha = manifest_entry.get("original_sha256") if manifest_entry else None
            if expected_sha:
                backup_sha = calculate_file_sha256(source_to_restore)
                if backup_sha != expected_sha:
                    return False, "La copia de seguridad ha sido alterada externamente. Hash SHA256 no coincide."

            # 2. Restaurar banco del jugador
            shutil.copy2(source_to_restore, vehicle.bank_path)

            # 3. Restaurar AI si existía respaldo
            ai_backup = veh_backup_folder / (vehicle.ai_bank_path.name if vehicle.ai_bank_path else "")
            ai_orig = vehicle.ai_bank_path.parent / f"{vehicle.ai_bank_path.name}.orig" if vehicle.ai_bank_path else None
            if vehicle.ai_bank_path:
                if ai_backup.is_file():
                    shutil.copy2(ai_backup, vehicle.ai_bank_path)
                elif ai_orig and ai_orig.is_file():
                    shutil.copy2(ai_orig, vehicle.ai_bank_path)

            # 3b. Restaurar .csdbin si existía respaldo
            if vehicle.csdbin_path:
                csd_backup = veh_backup_folder / vehicle.csdbin_path.name
                csd_orig = vehicle.csdbin_path.parent / f"{vehicle.csdbin_path.name}.orig"
                if csd_backup.is_file():
                    shutil.copy2(csd_backup, vehicle.csdbin_path)
                elif csd_orig.is_file():
                    shutil.copy2(csd_orig, vehicle.csdbin_path)
                if csd_orig.is_file():
                    try:
                        csd_orig.unlink()
                    except Exception:
                        pass

            # 4. Limpiar archivos .orig opcionales si la restauración fue 100% exitosa
            if orig_file.is_file():
                try:
                    orig_file.unlink()
                except Exception:
                    pass
            if ai_orig and ai_orig.is_file():
                try:
                    ai_orig.unlink()
                except Exception:
                    pass

            # 5. Comprobar que el archivo restaurado coincide con el hash original
            restored_sha = calculate_file_sha256(vehicle.bank_path)
            if expected_sha and restored_sha != expected_sha:
                return False, "Error de verificación: El archivo restaurado no coincide con el hash de fábrica."

            # 6. Actualizar manifiesto
            if vehicle.vehicle_id in self._manifest_cache:
                self._manifest_cache[vehicle.vehicle_id]["is_modified"] = False
                self._save_manifest()

            vehicle.is_modified = False
            return True, f"Vehículo '{vehicle.display_name}' restaurado con éxito al 100% original de Steam (SHA256 verificado)."
        except Exception as e:
            return False, f"Error durante la restauración: {e}"

    def restore_all(self) -> tuple[int, int, list[str]]:
        """Restaura TODOS los vehículos vanilla modificados a su estado original de Steam en una sola operación."""
        vehicles = self.scan_catalog()
        success_count = 0
        fail_count = 0
        messages: list[str] = []

        for v in vehicles:
            if v.is_modified or (v.bank_path.parent / f"{v.bank_path.name}.orig").is_file():
                ok, msg = self.restore_vehicle(v)
                if ok:
                    success_count += 1
                    messages.append(f"✓ {v.display_name}: Restaurado")
                else:
                    fail_count += 1
                    messages.append(f"✗ {v.display_name}: {msg}")

        return success_count, fail_count, messages

    def get_modified_vehicles(self) -> list[VanillaVehicle]:
        """Devuelve la lista de vehículos vanilla actualmente modificados."""
        res: list[VanillaVehicle] = []
        for v in self.scan_catalog():
            if v.is_modified or (v.bank_path.parent / f"{v.bank_path.name}.orig").is_file():
                res.append(v)
        return res
