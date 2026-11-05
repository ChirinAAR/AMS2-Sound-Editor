"""Simulador acústico FMOD de pasada completa de motor en 5 etapas (Interior y Exterior)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_dsp import AudioBuffer, conform_audio, read_wav_file, write_wav_bytes
from .fmod_bank import FmodBank
from .porter_engine import (
    RpmBandConfig,
    SlotMappingConfig,
    SourceAudioLibrary,
    ThrottleCutConfig,
    auto_detect_throttle_cut_config,
    build_default_rpm_bands,
)

STAGE_DESCRIPTIONS: dict[int, str] = {
    1: "i) Ralentí (Idle) -> Subida progresiva en 1ª hasta el corte",
    2: "ii) Cambio de marcha (Throttle Cut) -> 2ª subida progresiva al máximo",
    3: "iii) 3s manteniendo RPM máximas -> 3s desacelerando suavemente",
    4: "iv) Rebaje con Auto-Blip (punta-tacón) -> Continúa desaceleración",
    5: "v) Caída continua de RPM hasta estabilizar en Idle y detenerse",
}


@dataclass
class EnginePassTrajectory:
    """Trayectoria temporal completa de las 5 etapas de prueba del motor."""

    sample_rate: int
    idle_rpm: float
    max_rpm: float
    duration_sec: float
    t_ctrl: np.ndarray
    rpm_ctrl: np.ndarray
    throttle_ctrl: np.ndarray
    throttle_cut_ctrl: np.ndarray
    gear_ctrl: np.ndarray
    stage_ctrl: np.ndarray
    upshift_time_sec: float
    hold_start_sec: float
    liftoff_time_sec: float
    blip_start_sec: float
    blip_peak_sec: float
    idle_return_sec: float

    def sample_telemetry_at(self, elapsed_sec: float) -> tuple[float, float, int, int, str]:
        """Devuelve (rpm, throttle_0_1, gear, stage_num, stage_label) en el instante dado."""
        t = float(np.clip(elapsed_sec, 0.0, self.duration_sec))
        idx = int(np.searchsorted(self.t_ctrl, t, side="right")) - 1
        idx = max(0, min(len(self.t_ctrl) - 1, idx))
        rpm = float(self.rpm_ctrl[idx])
        thr = float(self.throttle_ctrl[idx])
        gear = int(self.gear_ctrl[idx])
        st = int(self.stage_ctrl[idx])
        return rpm, thr, gear, st, STAGE_DESCRIPTIONS.get(st, "")


def build_engine_pass_trajectory(
    idle_rpm: float = 950.0,
    max_rpm: float = 7800.0,
    sample_rate: int = 44100,
    control_hz: int = 150,
) -> EnginePassTrajectory:
    """Construye la curva exacta de RPM, acelerador, marcha y ThrottleCut para las 5 etapas."""
    idle = max(550.0, min(3200.0, float(idle_rpm)))
    redline = max(idle + 2000.0, min(22000.0, float(max_rpm)))
    span = redline - idle

    # Hitos temporales exactos (total ~24.8 s)
    t_idle_end = 1.10
    t_pull1_end = 6.90          # Etapa (i): subida lenta desde idle hasta max_rpm (5.80 s)
    t_upshift_end = 7.16        # Etapa (ii): corte de inyección / subida de marcha (0.26 s)
    t_pull2_end = 11.80         # Etapa (ii): subida progresiva en siguiente marcha hasta max_rpm (4.64 s)
    t_hold_end = 14.80          # Etapa (iii): 3.00 s exactos al máximo de RPM
    t_decel1_end = 17.80        # Etapa (iii): 3.00 s exactos desacelerando lentamente
    t_blip_peak = 18.02         # Etapa (iv): subida rápida de RPM por auto-blip / punta-tacón (0.22 s)
    t_blip_settle = 18.20       # Fin del golpe de gas del blip
    t_stage4_mark = 19.60       # Transición visual de etapa (iv) -> (v) durante la caída continua
    t_idle_reach = 23.60        # Etapa (v): llegada continua de vuelta a idle
    t_total = 24.80             # Estabilización en idle y parada

    n_ctrl = int(round(t_total * control_hz)) + 1
    t = np.linspace(0.0, t_total, n_ctrl, dtype=np.float64)
    rpm = np.full_like(t, idle)
    thr = np.zeros_like(t)
    tcut = np.zeros_like(t)
    gear = np.ones(n_ctrl, dtype=np.int32)
    stage = np.ones(n_ctrl, dtype=np.int32)

    shift_drop_rpm = max(idle * 1.55, idle + 0.64 * span)
    pre_blip_rpm = idle + 0.48 * span
    post_blip_rpm = min(redline * 0.89, pre_blip_rpm + 0.25 * span)

    for i, ti in enumerate(t):
        if ti < t_idle_end:
            # (i-a) Ralentí estable inicial
            wobble = 14.0 * np.sin(2.0 * np.pi * 4.2 * ti)
            rpm[i] = idle + wobble
            thr[i] = 0.04
            gear[i] = 1
            stage[i] = 1
        elif ti < t_pull1_end:
            # (i-b) Subida lenta y progresiva desde idle hasta max_rpm
            u = (ti - t_idle_end) / (t_pull1_end - t_idle_end)
            rpm[i] = idle + span * (u ** 0.94)
            thr[i] = min(1.0, 0.38 + 0.62 * (u ** 0.55))
            gear[i] = 1
            stage[i] = 1
        elif ti < t_upshift_end:
            # (ii-a) Subida de marcha en el corte -> dispara ThrottleCut y cae RPM a la siguiente relación
            u = (ti - t_pull1_end) / (t_upshift_end - t_pull1_end)
            smooth_u = 0.5 - 0.5 * np.cos(np.pi * u)
            rpm[i] = redline - (redline - shift_drop_rpm) * smooth_u
            thr[i] = 1.0
            tcut[i] = 1.0 if u < 0.85 else max(0.0, 1.0 - (u - 0.85) / 0.15)
            gear[i] = 2
            stage[i] = 2
        elif ti < t_pull2_end:
            # (ii-b) Segunda subida progresiva en 2ª marcha hasta max_rpm
            u = (ti - t_upshift_end) / (t_pull2_end - t_upshift_end)
            rpm[i] = shift_drop_rpm + (redline - shift_drop_rpm) * (u ** 0.92)
            thr[i] = 1.0
            gear[i] = 2
            stage[i] = 2
        elif ti < t_hold_end:
            # (iii-a) Se mantiene al máximo de RPM durante 3.0 segundos (con micro-oscilación de limitador)
            dt_h = ti - t_pull2_end
            lim_osc = 38.0 * (0.5 - 0.5 * np.cos(2.0 * np.pi * 15.5 * dt_h))
            rpm[i] = redline - lim_osc
            thr[i] = 1.0
            gear[i] = 2
            stage[i] = 3
        elif ti < t_decel1_end:
            # (iii-b) Suelta acelerador y desacelera lentamente durante 3.0 segundos
            u = (ti - t_hold_end) / (t_decel1_end - t_hold_end)
            rpm[i] = redline - (redline - pre_blip_rpm) * (u ** 0.90)
            thr[i] = max(0.0, 1.0 - (ti - t_hold_end) / 0.08)
            gear[i] = 2
            stage[i] = 3
        elif ti < t_blip_peak:
            # (iv-a) Rebaje de marcha con golpe de gas (Auto-Blip / Punta-Tacón): subida rápida de RPM
            u = (ti - t_decel1_end) / (t_blip_peak - t_decel1_end)
            smooth_u = 0.5 - 0.5 * np.cos(np.pi * u)
            rpm[i] = pre_blip_rpm + (post_blip_rpm - pre_blip_rpm) * smooth_u
            thr[i] = 0.95 * np.sin(0.5 * np.pi * u)
            gear[i] = 1
            stage[i] = 4
        elif ti < t_idle_reach:
            # (iv-b) y (v): Continúa desacelerando continuamente desde las RPM post-blip hasta llegar al idle
            u = (ti - t_blip_peak) / (t_idle_reach - t_blip_peak)
            # Curva natural de freno motor (ligeramente más rápida arriba, suave al llegar a ralentí)
            curve = 1.0 - ((1.0 - u) ** 1.28)
            rpm[i] = post_blip_rpm - (post_blip_rpm - idle) * curve
            if ti < t_blip_settle:
                thr[i] = max(0.0, 0.95 * (1.0 - (ti - t_blip_peak) / (t_blip_settle - t_blip_peak)))
            else:
                thr[i] = 0.0
            gear[i] = 1
            stage[i] = 4 if ti < t_stage4_mark else 5
        else:
            # (v-b) Estabilizado en ralentí antes de detenerse
            wobble = 12.0 * np.sin(2.0 * np.pi * 4.1 * ti) * max(0.0, 1.0 - (ti - t_idle_reach) / 1.1)
            rpm[i] = idle + wobble
            thr[i] = 0.02
            gear[i] = 1
            stage[i] = 5

    return EnginePassTrajectory(
        sample_rate=sample_rate,
        idle_rpm=idle,
        max_rpm=redline,
        duration_sec=t_total,
        t_ctrl=t,
        rpm_ctrl=rpm,
        throttle_ctrl=thr,
        throttle_cut_ctrl=tcut,
        gear_ctrl=gear,
        stage_ctrl=stage,
        upshift_time_sec=t_pull1_end,
        hold_start_sec=t_pull2_end,
        liftoff_time_sec=t_hold_end,
        blip_start_sec=t_decel1_end,
        blip_peak_sec=t_blip_peak,
        idle_return_sec=t_idle_reach,
    )


def _prepare_seamless_loop_stereo(buf: AudioBuffer) -> tuple[np.ndarray, int]:
    """Convierte AudioBuffer a float32 estéreo [-1..1] y construye un bucle continuo libre de clics."""
    x = buf.pcm.astype(np.float32) / 32768.0
    if x.ndim == 1:
        x = np.repeat(x[:, None], 2, axis=1)
    elif x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]

    n = x.shape[0]
    if n < 64:
        return np.zeros((1024, 2), dtype=np.float32), buf.sample_rate

    if buf.loop is not None and buf.loop[1] > buf.loop[0] + 64:
        ls = max(0, min(n - 65, int(buf.loop[0])))
        le = max(ls + 64, min(n, int(buf.loop[1]) + 1))
    else:
        margin = min(n // 12, int(buf.sample_rate * 0.05))
        ls = max(0, margin)
        le = max(ls + 64, n - margin)

    seg = x[ls:le].copy()
    seg_len = seg.shape[0]
    xf = min(seg_len // 6, max(64, int(buf.sample_rate * 0.035)))
    if xf >= 16 and seg_len > 2 * xf:
        # Fundido cruzado cabeza-cola para que el salto de fin->inicio sea 100% imperceptible
        head = seg[:xf].copy()
        tail = seg[seg_len - xf :].copy()
        w = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        blended = tail * (1.0 - w) + head * w
        loop_pcm = np.vstack([blended, seg[xf : seg_len - xf]])
    else:
        loop_pcm = seg

    return np.ascontiguousarray(loop_pcm, dtype=np.float32), buf.sample_rate


def _to_stereo_float(buf: AudioBuffer, target_sr: int, gain_db: float = 0.0) -> np.ndarray:
    """Adapta un AudioBuffer one-shot a float32 estéreo [-1..1] a target_sr."""
    conformed = conform_audio(buf, target_sr, 2, gain_db, need_loop=False)
    return conformed.pcm.astype(np.float32) / 32768.0


def _compute_rpm_band_gain(rpm_arr: np.ndarray, b: RpmBandConfig) -> np.ndarray:
    """Evalúa la envolvente exacta PMLB + CURV (prop=4) de FMOD para una banda de RPM."""
    rmin = float(b.rpm_min)
    rmax = max(rmin + 1.0, float(b.rpm_max))
    fi_end = max(rmin, min(rmax, float(b.fade_in_end)))
    fo_start = max(rmin, min(rmax, float(b.fade_out_start)))

    g = np.zeros_like(rpm_arr, dtype=np.float64)
    inside = (rpm_arr >= rmin) & (rpm_arr <= rmax)
    g[inside] = 1.0

    if fi_end > rmin + 1.0:
        m_in = inside & (rpm_arr < fi_end)
        alpha_in = (rpm_arr[m_in] - rmin) / (fi_end - rmin)
        # Curva de potencia constante suave equivalente al crossfade FMOD
        g[m_in] *= np.sin(0.5 * np.pi * np.clip(alpha_in, 0.0, 1.0))

    if fo_start < rmax - 1.0:
        m_out = inside & (rpm_arr > fo_start)
        alpha_out = (rmax - rpm_arr[m_out]) / (rmax - fo_start)
        g[m_out] *= np.sin(0.5 * np.pi * np.clip(alpha_out, 0.0, 1.0))

    return g


def render_engine_pass_audio(
    perspective: str,
    donor: FmodBank,
    mappings: list[SlotMappingConfig],
    rpm_bands: list[RpmBandConfig] | None = None,
    throttle_cut: ThrottleCutConfig | None = None,
    library: SourceAudioLibrary | None = None,
    idle_rpm: float = 950.0,
    max_rpm: float = 7800.0,
    sample_rate: int = 44100,
) -> tuple[AudioBuffer, EnginePassTrajectory]:
    """Sintetiza el audio estéreo completo de la pasada de 5 etapas (Interior 'int' o Exterior 'ext')."""
    persp = "int" if perspective.strip().lower().startswith("int") else "ext"
    target_ev_role = "EngineInt" if persp == "int" else "EngineExt"
    if not any(ev.role == target_ev_role for ev in donor.events):
        # Fallback por si el banco donante tuviera un único evento de motor
        target_ev_role = "EngineExt" if any(ev.role == "EngineExt" for ev in donor.events) else donor.events[0].role

    bands = rpm_bands if rpm_bands is not None else build_default_rpm_bands(donor)
    eff_tcut = throttle_cut if throttle_cut is not None else auto_detect_throttle_cut_config(donor, library)
    traj = build_engine_pass_trajectory(idle_rpm=idle_rpm, max_rpm=max_rpm, sample_rate=sample_rate)

    total_frames = int(round(traj.duration_sec * sample_rate))
    mix = np.zeros((total_frames, 2), dtype=np.float32)

    map_by_gidx = {m.donor_global_index: m for m in mappings}
    lib_by_idx = {it.index: it for it in library.items} if library else {}

    # Caché de buffers decodificados por slot donante para no decodificar dos veces
    slot_audio_cache: dict[int, tuple[AudioBuffer, float | None, float] | None] = {}

    def _resolve_slot_audio(gidx: int) -> tuple[AudioBuffer, float | None, float] | None:
        if gidx in slot_audio_cache:
            return slot_audio_cache[gidx]
        if not (0 <= gidx < len(donor.slots)):
            slot_audio_cache[gidx] = None
            return None
        sl = donor.slots[gidx]
        cfg = map_by_gidx.get(gidx, SlotMappingConfig(gidx))
        if cfg.action == "silence":
            slot_audio_cache[gidx] = None
            return None
        try:
            if cfg.action == "replace_source" and cfg.source_index is not None and cfg.source_index in lib_by_idx:
                it = lib_by_idx[cfg.source_index]
                ab = it.load_audio()
                root_r = cfg.override_root_rpm if cfg.override_root_rpm else (it.effective_rpm or sl.autopitch_ref_rpm)
                res = (ab, root_r, cfg.gain_db)
            elif cfg.action == "replace_wav" and cfg.custom_wav_path is not None and Path(cfg.custom_wav_path).is_file():
                ab = read_wav_file(Path(cfg.custom_wav_path))
                root_r = cfg.override_root_rpm or sl.autopitch_ref_rpm
                res = (ab, root_r, cfg.gain_db)
            elif cfg.action == "keep_donor" and sl.can_decode_audio:
                ab = donor.decode_slot_to_buffer(sl)
                root_r = cfg.override_root_rpm or sl.autopitch_ref_rpm
                res = (ab, root_r, cfg.gain_db)
            else:
                res = None
        except Exception:
            res = None
        slot_audio_cache[gidx] = res
        return res

    # Vector de tiempos a frecuencia de audio
    t_audio = np.arange(total_frames, dtype=np.float64) / float(sample_rate)
    rpm_audio = np.interp(t_audio, traj.t_ctrl, traj.rpm_ctrl)

    ev_bands = [b for b in bands if b.event_role == target_ev_role and not b.is_muted_bus]
    pops_oneshots: list[tuple[RpmBandConfig, np.ndarray]] = []

    for b_idx, b in enumerate(ev_bands):
        if not b.donor_slot_indices:
            continue

        # Si es una banda de petardeos (pops), separar one-shots de posibles loops de limitador
        if b.load_role == "pops":
            first_gidx = b.donor_slot_indices[0]
            first_sl = donor.slots[first_gidx]
            is_limiter_loop = (first_sl.loop is not None or first_sl.inst_infinite_loop) and first_sl.has_rpm_autopitch
            if not is_limiter_loop:
                for gidx in b.donor_slot_indices[:6]:
                    resolved = _resolve_slot_audio(gidx)
                    if resolved is not None:
                        ab, _, g_db = resolved
                        pops_oneshots.append((b, _to_stereo_float(ab, sample_rate, g_db - 4.0)))
                continue

        resolved_primary = _resolve_slot_audio(b.donor_slot_indices[0])
        if resolved_primary is None:
            continue
        abuf, root_rpm, slot_gain_db = resolved_primary
        sl_ref = donor.slots[b.donor_slot_indices[0]]

        # Calcular envolvente combinada (RPM * Carga Acelerador * Ducking ThrottleCut) a tasa de control
        g_rpm = _compute_rpm_band_gain(traj.rpm_ctrl, b)
        if np.max(g_rpm) < 1e-4:
            continue

        if b.load_role == "idle":
            g_thr = 1.0 - 0.18 * traj.throttle_ctrl
        elif b.load_role == "off":
            # Curva FMOD Off-Throttle: 0 dB hasta thr=0.08, cae a -42 dB en thr=0.63
            db_off = np.interp(traj.throttle_ctrl, [0.0, 0.08, 0.63, 1.0], [0.0, 0.0, -42.0, -42.0])
            g_thr = np.where(db_off <= -41.5, 0.0, 10.0 ** (db_off / 20.0))
        elif b.load_role == "pops":
            # Limitador en bucle: activo a pleno gas cerca del corte
            db_on = np.interp(traj.throttle_ctrl, [0.0, 0.75, 1.0], [-42.0, -12.0, 0.0])
            g_thr = np.where(db_on <= -41.5, 0.0, 10.0 ** (db_on / 20.0))
        else:
            # Curva FMOD On-Throttle: -42 dB en thr=0.0, sube a 0 dB en thr>=0.70 + Ducking durante ThrottleCut
            db_on = np.interp(traj.throttle_ctrl, [0.0, 0.18, 0.70, 1.0], [-38.0, -9.5, 0.0, 0.0])
            if eff_tcut.enabled and eff_tcut.duck_on_throttle_db < -0.5:
                db_duck = traj.throttle_cut_ctrl * float(eff_tcut.duck_on_throttle_db)
                db_on = db_on + db_duck
            g_thr = np.where(db_on <= -41.5, 0.0, 10.0 ** (db_on / 20.0))

        slot_lin_gain = 10.0 ** (float(slot_gain_db) / 20.0)
        g_ctrl = g_rpm * g_thr * slot_lin_gain
        active_ctrl_indices = np.flatnonzero(g_ctrl > 5e-4)
        if active_ctrl_indices.size == 0:
            continue

        t_start = max(0.0, float(traj.t_ctrl[max(0, int(active_ctrl_indices[0]) - 1)]))
        t_end = min(traj.duration_sec, float(traj.t_ctrl[min(len(g_ctrl) - 1, int(active_ctrl_indices[-1]) + 1)]))
        i0 = max(0, int(np.floor(t_start * sample_rate)))
        i1 = min(total_frames, int(np.ceil(t_end * sample_rate)))
        if i1 <= i0 + 16:
            continue

        loop_pcm, loop_sr = _prepare_seamless_loop_stereo(abuf)
        loop_len = loop_pcm.shape[0]

        g_slice = np.interp(t_audio[i0:i1], traj.t_ctrl, g_ctrl).astype(np.float32)
        if root_rpm and root_rpm > 100.0 and (sl_ref.has_rpm_autopitch or root_rpm > 400.0):
            pitch = np.clip(rpm_audio[i0:i1] / float(root_rpm), 0.22, 3.6)
        else:
            pitch = np.ones(i1 - i0, dtype=np.float64)

        step = pitch * (float(loop_sr) / float(sample_rate))
        # Desfase inicial determinista distinto por banda para evitar cancelaciones artificiales de fase
        phase0 = float((b_idx * 0.37) % 1.0) * loop_len
        phase = (phase0 + np.cumsum(step)) % float(loop_len)
        idx0 = phase.astype(np.int32)
        frac = (phase - idx0).astype(np.float32)[:, None]
        idx1 = (idx0 + 1) % loop_len

        rendered = loop_pcm[idx0] * (1.0 - frac) + loop_pcm[idx1] * frac
        mix[i0:i1] += rendered * g_slice[:, None]

    # Mezclar muestra One-Shot de Throttle Cut en la subida de marcha — Etapa (ii)
    def _mix_oneshot_at(oneshot_pcm: np.ndarray, time_sec: float, scale: float = 1.0) -> None:
        if oneshot_pcm.size == 0:
            return
        start_f = int(round(time_sec * sample_rate))
        if start_f >= total_frames:
            return
        start_f = max(0, start_f)
        avail = min(oneshot_pcm.shape[0], total_frames - start_f)
        if avail > 0:
            mix[start_f : start_f + avail] += oneshot_pcm[:avail] * float(scale)

    if eff_tcut.enabled:
        tcut_pcm: np.ndarray | None = None
        c_wav = eff_tcut.int_custom_wav if persp == "int" else eff_tcut.ext_custom_wav
        s_idx = eff_tcut.int_source_index if persp == "int" else eff_tcut.ext_source_index
        if c_wav is None and s_idx is None:
            c_wav = eff_tcut.ext_custom_wav if persp == "int" else eff_tcut.int_custom_wav
            s_idx = eff_tcut.ext_source_index if persp == "int" else eff_tcut.int_source_index

        if c_wav is not None and Path(c_wav).is_file():
            tcut_pcm = _to_stereo_float(read_wav_file(Path(c_wav)), sample_rate, eff_tcut.gain_db)
        elif s_idx is not None and s_idx in lib_by_idx:
            tcut_pcm = _to_stereo_float(lib_by_idx[s_idx].load_audio(), sample_rate, eff_tcut.gain_db)
        else:
            # Si el propio donante tenía un slot nativo de ThrottleCut
            for sl in donor.slots:
                if sl.load_role == "throttle_cut" and sl.perspective == persp:
                    res_tc = _resolve_slot_audio(sl.global_index)
                    if res_tc is not None:
                        tcut_pcm = _to_stereo_float(res_tc[0], sample_rate, res_tc[2])
                        break

        if tcut_pcm is not None:
            _mix_oneshot_at(tcut_pcm, traj.upshift_time_sec, scale=0.95)

    # Mezclar petardeos (pops / overrun) sutiles al soltar gas en Etapa (iii) y tras el Blip en Etapa (iv)
    if pops_oneshots:
        pop_triggers = [
            (traj.liftoff_time_sec + 0.06, 0.72),
            (traj.liftoff_time_sec + 0.34, 0.55),
            (traj.blip_peak_sec + 0.16, 0.68),
            (traj.blip_peak_sec + 0.44, 0.48),
        ]
        for p_idx, (trig_t, p_scale) in enumerate(pop_triggers):
            rpm_at_trig, _, _, _, _ = traj.sample_telemetry_at(trig_t)
            b_pop, pcm_pop = pops_oneshots[p_idx % len(pops_oneshots)]
            if b_pop.rpm_min <= rpm_at_trig <= b_pop.rpm_max:
                _mix_oneshot_at(pcm_pop, trig_t, scale=p_scale)

    # Fundido suave de entrada/salida y limitador de techo suave para evitar saturación digital
    fade_in_n = min(total_frames // 20, int(0.05 * sample_rate))
    fade_out_n = min(total_frames // 10, int(0.32 * sample_rate))
    if fade_in_n > 1:
        mix[:fade_in_n] *= np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)[:, None]
    if fade_out_n > 1:
        mix[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)[:, None]

    mix *= 0.72
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.92:
        # Compresión suave tipo soft-clip por encima de -1.5 dBFS
        mix = np.tanh(mix * (0.95 / max(0.95, peak * 0.88))) * 0.96

    pcm16 = np.clip(np.round(mix * 32767.0), -32768, 32767).astype(np.int16)
    return AudioBuffer(pcm=pcm16, sample_rate=sample_rate, loop=None), traj


def render_engine_pass_wav_bytes(
    perspective: str,
    donor: FmodBank,
    mappings: list[SlotMappingConfig],
    rpm_bands: list[RpmBandConfig] | None = None,
    throttle_cut: ThrottleCutConfig | None = None,
    library: SourceAudioLibrary | None = None,
    idle_rpm: float = 950.0,
    max_rpm: float = 7800.0,
    sample_rate: int = 44100,
) -> tuple[bytes, EnginePassTrajectory]:
    """Genera los bytes RIFF WAVE listos para reproducir de la pasada completa en 5 etapas."""
    buf, traj = render_engine_pass_audio(
        perspective=perspective,
        donor=donor,
        mappings=mappings,
        rpm_bands=rpm_bands,
        throttle_cut=throttle_cut,
        library=library,
        idle_rpm=idle_rpm,
        max_rpm=max_rpm,
        sample_rate=sample_rate,
    )
    return write_wav_bytes(buf), traj
