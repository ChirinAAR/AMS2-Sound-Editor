"""Procesamiento digital de audio (WAV, Resampling, Loops, PCM16 y FMOD FADPCM)."""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Tabla exacta de predictores IIR de FMOD FADPCM (escala /64)
FADPCM_COEFFS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (60, 0),
    (122, -60),
    (115, -52),
    (98, -55),
    (0, 0),
    (0, 0),
    (0, 0),
)


@dataclass
class AudioBuffer:
    """Buffer de audio PCM de 16 bits con signo (shape: [num_samples, channels])."""

    pcm: np.ndarray  # dtype=np.int16, shape=(N, ch)
    sample_rate: int
    loop: tuple[int, int] | None = None  # (start_sample, end_sample_inclusive)

    @property
    def num_samples(self) -> int:
        return int(self.pcm.shape[0])

    @property
    def channels(self) -> int:
        return int(self.pcm.shape[1])

    @property
    def duration_sec(self) -> float:
        return self.num_samples / max(1, self.sample_rate)

    def peak_linear(self) -> float:
        if self.pcm.size == 0:
            return 0.0
        pk = float(np.max(np.abs(self.pcm.astype(np.int32)))) / 32768.0
        return min(1.0, pk)


def read_wav_bytes(raw: bytes) -> AudioBuffer:
    """Lee un archivo RIFF WAVE estándar (PCM 8/16/24/32-bit o IEEE Float 32/64-bit) y puntos de loop 'smpl'."""
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("El archivo no es un contenedor RIFF WAVE válido.")

    pos = 12
    end = len(raw)
    audio_fmt = 1
    channels = 1
    sample_rate = 44100
    bits_per_sample = 16
    data_bytes = b""
    loop: tuple[int, int] | None = None

    while pos + 8 <= end:
        cid = raw[pos : pos + 4]
        csz = struct.unpack_from("<I", raw, pos + 4)[0]
        ps = pos + 8
        pe = min(end, ps + csz)
        if cid == b"fmt " and csz >= 16:
            audio_fmt, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
                "<HHIIHH", raw, ps
            )
            if audio_fmt == 0xFFFE and csz >= 40:
                audio_fmt = struct.unpack_from("<H", raw, ps + 24)[0]
        elif cid == b"data":
            data_bytes = raw[ps:pe]
        elif cid == b"smpl" and csz >= 60:
            num_loops = struct.unpack_from("<I", raw, ps + 28)[0]
            if num_loops >= 1:
                l_start, l_end = struct.unpack_from("<II", raw, ps + 36 + 8)
                if l_end > l_start:
                    loop = (int(l_start), int(l_end))
        pos = pe + (csz & 1)

    if not data_bytes:
        return AudioBuffer(np.zeros((0, max(1, channels)), dtype=np.int16), sample_rate, None)

    if audio_fmt == 1:  # PCM entero
        if bits_per_sample == 8:
            u8 = np.frombuffer(data_bytes, dtype=np.uint8)
            s16 = ((u8.astype(np.int16) - 128) << 8).astype(np.int16)
        elif bits_per_sample == 16:
            n = (len(data_bytes) // 2) * 2
            s16 = np.frombuffer(data_bytes[:n], dtype="<i2").copy()
        elif bits_per_sample == 24:
            n3 = (len(data_bytes) // 3) * 3
            b = np.frombuffer(data_bytes[:n3], dtype=np.uint8).reshape(-1, 3)
            # Tomamos los 2 bytes más significativos para pasar limpiamente a 16-bit LE
            hi16 = b[:, 1].astype(np.uint16) | (b[:, 2].astype(np.uint16) << 8)
            s16 = hi16.view(np.int16).copy()
        elif bits_per_sample == 32:
            n = (len(data_bytes) // 4) * 4
            s32 = np.frombuffer(data_bytes[:n], dtype="<i4")
            s16 = (s32 >> 16).astype(np.int16)
        else:
            raise ValueError(f"Profundidad de bits PCM no soportada: {bits_per_sample}")
    elif audio_fmt == 3:  # IEEE Float
        if bits_per_sample == 32:
            n = (len(data_bytes) // 4) * 4
            f = np.frombuffer(data_bytes[:n], dtype="<f4")
        elif bits_per_sample == 64:
            n = (len(data_bytes) // 8) * 8
            f = np.frombuffer(data_bytes[:n], dtype="<f8")
        else:
            raise ValueError(f"Profundidad IEEE Float no soportada: {bits_per_sample}")
        s16 = np.clip(np.round(f * 32767.0), -32768, 32767).astype(np.int16)
    else:
        raise ValueError(f"Formato de compresión WAV no soportado (código {audio_fmt}). Usa WAV PCM/Float.")

    frames = len(s16) // channels
    pcm = s16[: frames * channels].reshape(frames, channels)
    if loop is not None:
        ls = max(0, min(frames - 1, loop[0]))
        le = max(ls, min(frames - 1, loop[1]))
        loop = (ls, le) if le > ls else None
    return AudioBuffer(pcm=pcm, sample_rate=sample_rate, loop=loop)


def read_wav_file(path: Path) -> AudioBuffer:
    return read_wav_bytes(Path(path).read_bytes())


def write_wav_bytes(buf: AudioBuffer) -> bytes:
    """Genera un archivo RIFF WAVE PCM 16-bit con chunk 'smpl' opcional si tiene loop."""
    pcm_le = np.ascontiguousarray(buf.pcm, dtype="<i2")
    data_payload = pcm_le.tobytes()
    ch = buf.channels
    sr = buf.sample_rate
    byte_rate = sr * ch * 2
    block_align = ch * 2

    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        ch,
        sr,
        byte_rate,
        block_align,
        16,
    )
    data_hdr = struct.pack("<4sI", b"data", len(data_payload))
    pad = b"\x00" if (len(data_payload) & 1) else b""

    smpl_chunk = b""
    if buf.loop is not None and buf.loop[1] > buf.loop[0]:
        ls, le = int(buf.loop[0]), int(buf.loop[1])
        sample_period = int(1_000_000_000 / max(1, sr))
        smpl_body = struct.pack(
            "<9I6I",
            0,
            0,
            sample_period,
            60,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            ls,
            le,
            0,
            0,
        )
        smpl_chunk = struct.pack("<4sI", b"smpl", len(smpl_body)) + smpl_body

    riff_size = 4 + len(fmt_chunk) + len(data_hdr) + len(data_payload) + len(pad) + len(smpl_chunk)
    out = io.BytesIO()
    out.write(struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE"))
    out.write(fmt_chunk)
    out.write(data_hdr)
    out.write(data_payload)
    out.write(pad)
    if smpl_chunk:
        out.write(smpl_chunk)
    return out.getvalue()


def write_wav_file(path: Path, buf: AudioBuffer) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_wav_bytes(buf))
    return path


def conform_audio(
    buf: AudioBuffer,
    target_sr: int,
    target_channels: int,
    gain_db: float = 0.0,
    need_loop: bool = False,
) -> AudioBuffer:
    """Adapta frecuencia de muestreo, canales, ganancia y puntos de loop para un slot FMOD."""
    if buf.num_samples == 0:
        silent = np.zeros((max(256, target_sr // 10), target_channels), dtype=np.int16)
        lp = (0, silent.shape[0] - 1) if need_loop else None
        return AudioBuffer(silent, target_sr, lp)

    x = buf.pcm.astype(np.float32)

    # 1. Conversión de canales (Mono <-> Estéreo)
    cur_ch = x.shape[1]
    if cur_ch != target_channels:
        if cur_ch == 1 and target_channels == 2:
            x = np.repeat(x, 2, axis=1)
        elif cur_ch >= 2 and target_channels == 1:
            x = np.mean(x[:, :2], axis=1, keepdims=True)
        elif target_channels > cur_ch:
            reps = [x[:, i % cur_ch : (i % cur_ch) + 1] for i in range(target_channels)]
            x = np.concatenate(reps, axis=1)
        else:
            x = x[:, :target_channels]

    # 2. Resampling band-limited por FFT (rfft/irfft exacto y rápido en NumPy)
    loop = buf.loop
    if buf.sample_rate != target_sr and x.shape[0] > 1:
        ratio = target_sr / float(buf.sample_rate)
        new_len = max(1, int(round(x.shape[0] * ratio)))
        spec = np.fft.rfft(x, axis=0)
        new_bins = new_len // 2 + 1
        out_spec = np.zeros((new_bins, target_channels), dtype=np.complex64)
        copy_bins = min(spec.shape[0], new_bins)
        out_spec[:copy_bins] = spec[:copy_bins]
        x = np.fft.irfft(out_spec, n=new_len, axis=0) * (new_len / float(x.shape[0]))
        if loop is not None:
            ls = max(0, min(new_len - 2, int(round(loop[0] * ratio))))
            le = max(ls + 1, min(new_len - 1, int(round(loop[1] * ratio))))
            loop = (ls, le)

    # 3. Ajuste de ganancia en dB
    if abs(gain_db) > 1e-4:
        scale = 10.0 ** (gain_db / 20.0)
        x = x * scale

    # 4. Síntesis / ajuste de loop limpio si el slot del donante lo exige
    n = x.shape[0]
    if need_loop and loop is None and n >= 1024:
        margin = min(n // 10, int(target_sr * 0.08))
        ls = max(0, margin)
        le = max(ls + 256, n - 1 - margin)
        # Crossfade corto en el borde final para evitar clics al ciclar
        xfade = min(512, (le - ls) // 8)
        if xfade > 8 and ls >= xfade:
            w = np.linspace(0.0, 1.0, xfade, dtype=np.float32)[:, None]
            x[le - xfade + 1 : le + 1] = (
                x[le - xfade + 1 : le + 1] * (1.0 - w) + x[ls - xfade : ls] * w
            )
        loop = (ls, le)

    pcm_out = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    return AudioBuffer(pcm=pcm_out, sample_rate=target_sr, loop=loop)


def decode_fadpcm(data: bytes, channels: int, total_samples: int) -> np.ndarray:
    """Decodifica audio FMOD FADPCM (FSB5 Mode 16) a PCM int16 (shape: [total_samples, channels])."""
    if total_samples <= 0 or channels <= 0:
        return np.zeros((0, max(1, channels)), dtype=np.int16)

    num_blocks = (total_samples + 255) // 256
    avail_blocks = len(data) // (140 * channels)
    blocks = min(num_blocks, avail_blocks)
    if blocks == 0:
        return np.zeros((total_samples, channels), dtype=np.int16)

    # Pre-extraemos los bloques en NumPy para máxima velocidad
    raw_arr = np.frombuffer(data[: blocks * channels * 140], dtype=np.uint8).reshape(
        blocks, channels, 140
    )
    out = np.zeros((num_blocks * 256, channels), dtype=np.int16)

    for ch in range(channels):
        ch_blocks = raw_arr[:, ch, :]
        hdr_bytes = np.ascontiguousarray(ch_blocks[:, :12]).tobytes()
        # Extraer nibbles con signo (-8..+7) vectorizadamente para todos los bloques del canal
        pay = ch_blocks[:, 12:]  # (blocks, 128)
        lo = (pay & 0x0F).astype(np.int16)
        hi = (pay >> 4).astype(np.int16)
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = np.where(hi >= 8, hi - 16, hi)
        # Cada byte produce [lo, hi] en orden lineal
        nibs = np.empty((blocks, 256), dtype=np.int32)
        nibs[:, 0::2] = lo
        nibs[:, 1::2] = hi

        for b in range(blocks):
            coefs_u32, shifts_u32, h1, h2 = struct.unpack_from("<IIhh", hdr_bytes, b * 12)
            s1, s2 = int(h1), int(h2)
            bnibs = nibs[b]
            dst_base = b * 256
            b_out = out[dst_base : dst_base + 256, ch]
            for g in range(8):
                ci = (coefs_u32 >> (g * 4)) & 0x7
                sh = (shifts_u32 >> (g * 4)) & 0xF
                c1, c2 = FADPCM_COEFFS[ci]
                rsh = 22 - sh
                g_off = g * 32
                for i in range(32):
                    sn = int(bnibs[g_off + i])
                    step64 = (sn << 28) >> rsh if rsh >= 0 else (sn << (28 - rsh))
                    smp = (step64 + s1 * c1 + s2 * c2) >> 6
                    if smp > 32767:
                        smp = 32767
                    elif smp < -32768:
                        smp = -32768
                    b_out[g_off + i] = smp
                    s2 = s1
                    s1 = smp

    return out[:total_samples]


def encode_fadpcm(pcm: np.ndarray) -> tuple[bytes, int]:
    """Codifica PCM int16 (shape: [N, ch]) al formato nativo FMOD FADPCM (Mode 16).

    Retorna (bytes_alineados_a_32, num_pcm_samples_multiplo_256).
    """
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_orig, channels = pcm.shape
    num_blocks = max(1, (n_orig + 255) // 256)
    padded_n = num_blocks * 256
    if padded_n > n_orig:
        pad_arr = np.zeros((padded_n - n_orig, channels), dtype=np.int16)
        pcm_pad = np.vstack([pcm, pad_arr])
    else:
        pcm_pad = pcm

    out_frames = bytearray(num_blocks * channels * 140)
    active_coeffs = FADPCM_COEFFS[:5]

    for ch in range(channels):
        sig = pcm_pad[:, ch].astype(np.int32)
        s1, s2 = int(sig[0]), int(sig[0])
        for b in range(num_blocks):
            blk = sig[b * 256 : (b + 1) * 256]
            h1_init, h2_init = s1, s2
            coefs_u32 = 0
            shifts_u32 = 0
            pay = bytearray(128)

            for g in range(8):
                sub = blk[g * 32 : (g + 1) * 32]
                best_err = 1 << 60
                best_ci = 0
                best_sh = 4
                best_nibs: list[int] = [0] * 32
                best_s1, best_s2 = s1, s2

                # Elegimos el filtro con menor residuo máximo primero para acotar la búsqueda
                best_filter_ci = 0
                min_peak_res = 1 << 30
                ts1, ts2 = s1, s2
                for ci, (c1, c2) in enumerate(active_coeffs):
                    max_r = 0
                    ps1, ps2 = ts1, ts2
                    for smp_v in sub:
                        pred = (ps1 * c1 + ps2 * c2) >> 6
                        r = abs(int(smp_v) - pred)
                        if r > max_r:
                            max_r = r
                        ps2, ps1 = ps1, int(smp_v)
                    if max_r < min_peak_res:
                        min_peak_res = max_r
                        best_filter_ci = ci

                # Calculamos el shift óptimo alrededor del pico residual
                c1, c2 = active_coeffs[best_filter_ci]
                est_sh = max(0, min(12, int(math.ceil(math.log2(max(1.0, min_peak_res / 7.0))))))
                cand_shifts = sorted({max(0, est_sh - 1), est_sh, min(12, est_sh + 1)})

                for sh in cand_shifts:
                    rsh = 22 - sh
                    step_unit = 1 << sh  # equivalente en escala PCM
                    cs1, cs2 = s1, s2
                    err_sum = 0
                    cur_nibs = [0] * 32
                    for i in range(32):
                        target = int(sub[i])
                        pred64 = cs1 * c1 + cs2 * c2
                        diff = target - (pred64 >> 6)
                        q = int(round(diff / float(step_unit)))
                        if q > 7:
                            q = 7
                        elif q < -8:
                            q = -8
                        step64 = (q << 28) >> rsh
                        rec = (step64 + pred64) >> 6
                        if rec > 32767:
                            rec = 32767
                        elif rec < -32768:
                            rec = -32768
                        d = target - rec
                        err_sum += d * d
                        cur_nibs[i] = q & 0xF
                        cs2, cs1 = cs1, rec
                    if err_sum < best_err:
                        best_err = err_sum
                        best_ci = best_filter_ci
                        best_sh = sh
                        best_nibs = cur_nibs
                        best_s1, best_s2 = cs1, cs2

                coefs_u32 |= (best_ci & 0x7) << (g * 4)
                shifts_u32 |= (best_sh & 0xF) << (g * 4)
                p_off = g * 16
                for bi in range(16):
                    pay[p_off + bi] = best_nibs[bi * 2] | (best_nibs[bi * 2 + 1] << 4)
                s1, s2 = best_s1, best_s2

            dst_off = (b * channels + ch) * 140
            struct.pack_into("<IIhh", out_frames, dst_off, coefs_u32, shifts_u32, h1_init, h2_init)
            out_frames[dst_off + 12 : dst_off + 140] = pay

    pad32 = (32 - (len(out_frames) % 32)) % 32
    if pad32:
        out_frames += b"\x00" * pad32
    return bytes(out_frames), padded_n
