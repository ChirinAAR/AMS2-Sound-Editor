# AMS2 Sound Porter Studio (v1.5.0)

Herramienta gráfica de escritorio (sin uso de consola) para **portear sonidos de vehículos desde Assetto Corsa (`.bank` + `GUIDs.txt` o `.wav`) hacia Automobilista 2**, implementando tanto la **Transmutación Directa Sin Donante (100% Fiel)** como el **Mapeo sobre Banco Donante de AMS2**, con inyección `FSB5` + Tabla `STDT` RadixTree embebida para rutas `event:/AMS2/Vehicles/Mods/...` independientes de `MasterBank.strings.bank` e integrando nativamente el codec `.csdbin` de **Open Madness Vehicle Tools (OMVT)**.

## Inicio Rápido (Sin Consola)

Haz doble clic sobre **`Iniciar_AMS2_Sound_Porter.bat`** (o ejecuta `python main.py`).

## Modos de Porteo (Pestaña 2)

### ⚡ Modo 1: Transmutación Directa Sin Donante (100% Fiel a Assetto Corsa) — ¡NUEVO en v1.5.0!
- **¿Por qué usarlo?** Tomar un banco donante de AMS2 a menudo degrada el sonido: si el donante tiene pocas bandas (ej. 6 bandas en un M3 GTR), comprime un sonido de AC de 68 bandas en 6 slots; o si el donante carece de efectos de turbo, levas o engranajes, esos efectos se pierden en el donante.
- **¿Cómo funciona?** Los motores FMOD Studio de Assetto Corsa y Automobilista 2 comparten la misma arquitectura binaria (FMOD Studio 1.08, versión 0x50, compat 0x46). La herramienta realiza una transmutación binaria directa:
  1. **Parcheo binario in-place `rpms` -> `rpm`:** Reconfigura el parámetro de revoluciones de AC al nombre nativo que lee Madness Engine sin alterar offsets ni tablas FMOD.
  2. **Reenrutamiento de buses a AMS2:** Remapea los buses de salida de AC (`IBSB`) directamente a los buses maestros de AMS2: `{c1986fb5-b329-4565-8342-b223857006d5}` (Exterior) y `{a8959bf6-0b54-43ea-9d9a-06a8831c40d4}` (Interior/Cockpit).
  3. **Incrustación de RadixTree STDT:** Construye la tabla de rutas interna requerida por Automobilista 2 para registrar todos los eventos (`event:/AMS2/Vehicles/Mods/...`) sin depender del strings bank global.
  4. **Preservación total:** Conserva intactos todos los eventos nativos (motor, escape, petardeos, cambio de marchas, sobrealimentación, transmisión, limitador de corte y viento), todas las curvas de volumen, pitch automations, moduladores y timelines originales.
  5. **Auto-enlace `.csdbin` completo:** Conecta automáticamente los 20 enlaces del vehículo en el archivo `.csdbin` (`engine_ext`, `engine_int`, `gearshift_up/down`, `gearshift_grind`, `turbo`, `transmission`, `rev_limiter`, `backfire_ext/int`, `wind`).

### 🏛 Modo 2: Mapeo sobre Banco Donante de AMS2
- Útil cuando se desea forzar la arquitectura y cajas de un vehículo vanilla específico de AMS2 como plantilla, adaptando las muestras de AC o WAVs a sus slots.

## Flujo de Trabajo en 5 Pasos

1. **Pestaña 1 — Audio Origen (`AC .bank` + `GUIDs.txt` / `WAV`)**
   - Carga cualquier archivo `.bank` de Assetto Corsa (vanilla o mod) o una carpeta con muestras `.wav`.
   - Si el sonido mod de AC incluye un archivo `GUIDs.txt` en la misma carpeta `sfx` (o carpeta superior), la herramienta **lo detecta y vincula automáticamente**; si lo tienes guardado en otra ruta, puedes cargarlo con el botón **📄 Cargar GUIDs.txt...**.
   - El analizador binario FMOD reconstruye eventos (`engine_int`, `engine_ext`, `backfire_int/ext`, `gearup/down`, `throttle_cut`, `turbo`, `transmission`, etc.), `MUIT` Multi-Instruments con herencia de `Auto-Pitch RPM`, timelines `TLNB` y curvas de automatización `throttle` / `Distance` para clasificar semánticamente cada capa.
   - Haz doble clic sobre cualquier muestra para escucharla al instante o extráelas todas a `.wav`.
2. **Pestaña 2 — Banco Donante AMS2 y Mapeo Inteligente**
   - Carga un archivo `.bank` oficial de motor de Automobilista 2 que sirva como plantilla de curvas RPM/carga.
   - Pulsa **⚡ Auto-Mapear desde Origen**: el motor analiza el grafo interno FMOD (`EVTB`, `PMLB`, `WAIB`, `INST`, buses atenuados) tanto de AC como de AMS2 para emparejar cada capa (`Idle`, `On-Throttle`, `Off-Throttle`, `Interior`, `Exterior`, `Pops`) según su perspectiva, carga y rango real de RPM, eliminando capas duplicadas y buses silenciados.
   - Personaliza cualquier slot manualmente, ajusta ganancia en dB o RPM raíz, y compara **▶ Escuchar Original AMS2** vs **▶ Escuchar Reemplazo Mapeado**.
3. **Pestaña 3 — Bandas RPM, Throttle Cut y Simulador Acústico**
   - **Añadir y Eliminar Bandas (`➕` / `➖`):** Añade libremente nuevas bandas de RPM a la vista activa (**Interior** o **Exterior**) directamente desde las muestras del `.bank` de Assetto Corsa o archivos `.wav`, y elimina/silencia bandas secundarias que no desees utilizar.
   - **Auto-Ajuste de Escalera RPM (Global o Individual):** Adapta los crossfades de fundido FMOD (`rpm_min -> fade_in_end -> fade_out_start -> rpm_max`) al rango de revoluciones real del motor. Si tienes una banda seleccionada, el ajuste se aplica **quirúrgicamente a esa banda** alineándola con sus adyacentes; sin selección, redistribuye toda la escalera.
   - **Restaurar Originales / Recreación Fiel de Assetto Corsa:**
     - **Alineación de Pitch y Cajas Nativas FMOD:** Mapea directamente sobre las cajas de automatización nativas (`PMLB`) y slots físicos del banco donante de AMS2, sincronizando los valores de `Autopitch Reference RPM` en los chunks binarios `WAIT` / `MUIT` para que coincidan exactamente con las RPMs de grabación del sonido de origen. Esto elimina por completo el desfase de pitch, deformaciones de afinación o cortes antinaturales en el juego.
     - **Aislamiento Estricto de Chase Cam en Exterior:** En la vista exterior, descarta automáticamente todos los micrófonos secundarios y de repetición (`front`, `far`, `ext_far`, `flyby`, `passby`, `trackside`, rebotes de chasis, etc.) para emplear **únicamente las muestras de escape y carga directa de la cámara de persecución (Chase Cam)**.
     - **Clusterización Limpia Anti-Duplicados:** Agrupa tomas grabadas a las mismas revoluciones (tomas `_a`, `_b`, etc.) seleccionando la muestra canónica más limpia de cada escalón y redistribuyendo la escalera de RPM con cruces cruzados simétricos.
     - **Silenciado Inteligente de Capas Sobrantes:** Silencia las capas de vibración, transmisiones o cajas secundarias del donante que no existan en el audio de AC para preservar la pureza acústica original.
   - **Amplificación Masiva de Bandas (`Amplificar todas las bandas`):** Permite subir o bajar la ganancia (dB) de todas las bandas activas de la vista activa (INT o EXT) en un solo clic, evitando la edición manual banda por banda.
   - **Editor Visual Interactivo:** Diagrama gráfico de bandas por capas (`IDLE`, `ON`, `OFF`, `POPS`) tanto para **Interior (`INT`)** como **Exterior (`EXT`)** con detección de solapamientos excesivos y ajuste fino de cada banda (frecuencia, fundidos cruzados, RPM raíz y ganancia dB).
   - **Throttle Cut Automático (Corte de Inyección al Subir Marcha):** Detección automática del efecto de cambio de marcha/corte en el `.bank` de Assetto Corsa o asignación personalizada con vista previa individual.
   - **Simulador de Motor en 5 Etapas (Interior y Exterior):** Sintetiza y reproduce en tiempo real la pasada completa con aguja de RPM y telemetría en vivo:
     - **i)** Arranca en ralentí (Idle) y sube progresivamente hasta el corte.
     - **ii)** Al llegar al corte, sube de marcha disparando el `ThrottleCut` con caída de RPM y segunda aceleración a fondo.
     - **iii)** Mantiene el corte durante 3 segundos y desacelera progresivamente durante 3 segundos con petardeos overrun.
     - **iv)** Rebaja una marcha disparando el "Auto-Blip" (punta-tacón) y continúa desacelerando desde el golpe de gas.
     - **v)** Desciende suavemente hasta estabilizar en ralentí y detenerse.
4. **Pestaña 4 — Vehículo `.csdbin` (Motor OMVT Integrado)**
   - Abre el archivo `.csdbin` (o texto `.csd` de OMVT) de tu vehículo mod.
   - Edita visualmente bancos, eventos, volúmenes, marchas y posición 3D del motor, o edita el código `.csd` completo sin pasar por la terminal.
5. **Pestaña 5 — Compilar y Exportar Mod para AMS2**
   - Elige un identificador único para tu mod, selecciona **PCM16 Sin Pérdida** (recomendado) o **FADPCM Comprimido**, deja activo el **Modo Standalone (`STDT` embebido + GUIDs únicos)** y la **Sincronización automática de Auto-Pitch RPM**, y pulsa **🚀 COMPILAR Y EXPORTAR PAQUETE**.
   - Se generará la estructura lista para copiar a tu mod de Automobilista 2 (`Audio/AMS2/Vehicles/Mods/.../*.bank` + `Vehicles/<carpeta>/<archivo>.csdbin`).

## Requisitos

- Windows 10 / 11
- Python 3.10+ con `numpy` (`tkinter` y `winsound` vienen incluidos en la instalación estándar de Python para Windows).

## Créditos y Licencia

- Incluye e integra el codec `Q02 CSDBIN` de **Open Madness Vehicle Tools (OMVT)** (`Copyright (C) 2026 ohyeah2389`), distribuido bajo licencia **GPL-3.0-or-later** con la excepción **OMVT Output Exception v1.0** (los archivos `.csdbin`, `.csd` y `.bank` generados por la herramienta no quedan sujetos a GPL por el hecho de ser generados con ella).
