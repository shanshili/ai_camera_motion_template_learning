"""
WP2 · 音乐分析
==============
从零实现（不依赖 librosa）：起音包络、速度估计、Ellis 动态规划拍点跟踪、
强拍、能量曲线、Foote 段落分段；按 scipy STFT 的真实帧中心时间戳换算，
并量化/重采样到视频帧。
"""

import subprocess
from fractions import Fraction

import numpy as np
from scipy.signal import stft as _stft, find_peaks

from .log import log

SR = 22050
HOP = 256
WIN = 1024


def load_audio(path, sr=SR):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码音频失败: {out.stderr.decode()[:200]}")
    y = np.frombuffer(out.stdout, np.float32).astype(np.float64)
    if y.size == 0:
        raise RuntimeError("音频为空")
    return y


def _spectrogram(y):
    f, t, Z = _stft(y, fs=SR, nperseg=WIN, noverlap=WIN - HOP,
                    boundary=None, padded=False)
    return np.abs(Z), t


def onset_envelope(y):
    mag, t = _spectrogram(y)
    logmag = np.log1p(mag)
    flux = np.maximum(0.0, np.diff(logmag, axis=1)).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    k = 8
    local = np.convolve(flux, np.ones(k) / k, mode="same")
    env = np.maximum(0.0, flux - local)
    if env.max() > 0:
        env = env / env.max()
    return env, t


def estimate_tempo(env, env_fps, bpm_min=50, bpm_max=200, prior_bpm=120.0):
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lag_min = int(round(60.0 / bpm_max * env_fps))
    lag_max = min(int(round(60.0 / bpm_min * env_fps)), len(ac) - 2)
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * env_fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / prior_bpm) / 0.7) ** 2)
    score = ac[lags] * prior
    bi = int(np.argmax(score))
    if 0 < bi < len(score) - 1:
        y0, y1, y2 = score[bi - 1], score[bi], score[bi + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        frac = 0.0
    period = lags[bi] + frac
    return 60.0 * env_fps / period, period


def track_beats(env, period):
    n = len(env)
    if n == 0:
        return np.array([], dtype=int)
    tightness = 100.0
    C = np.copy(env)
    back = -np.ones(n, dtype=int)
    for t in range(n):
        v0 = max(0, int(t - 2 * period))
        v1 = int(t - period / 2)
        if v1 < v0:
            continue
        v = np.arange(v0, v1 + 1)
        score = C[v] - tightness * (np.log((t - v) / period + 1e-9)) ** 2
        j = int(np.argmax(score))
        C[t] += score[j]
        back[t] = v[j]
    tail = max(0, n - int(period))
    b = tail + int(np.argmax(C[tail:]))
    beats = []
    while b >= 0:
        beats.append(b)
        b = back[b]
    return np.array(sorted(beats), dtype=int)


def refine_beats(env, beat_frames, period):
    if len(beat_frames) == 0:
        return np.array([], dtype=float)
    w = max(1, int(round(period * 0.12)))
    out = []
    for b in beat_frames:
        lo, hi = max(0, b - w), min(len(env), b + w + 1)
        if hi - lo < 3:
            out.append(float(b)); continue
        gp = lo + int(np.argmax(env[lo:hi]))
        if 0 < gp < len(env) - 1:
            y0, y1, y2 = env[gp - 1], env[gp], env[gp + 1]
            denom = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            frac = float(np.clip(frac, -0.5, 0.5))
        else:
            frac = 0.0
        out.append(gp + frac)
    return np.array(out, dtype=float)


def detect_downbeats(env, beat_frames, meter=4):
    if len(beat_frames) == 0:
        return np.zeros(0, dtype=bool)
    s = env[np.clip(beat_frames, 0, len(env) - 1)]
    best = int(np.argmax([s[p::meter].sum() for p in range(meter)]))
    is_down = np.zeros(len(beat_frames), dtype=bool)
    is_down[best::meter] = True
    return is_down


def energy_curve_env(y):
    n = 1 + (len(y) - WIN) // HOP if len(y) >= WIN else 1
    rms = np.empty(n)
    for i in range(n):
        seg = y[i * HOP:i * HOP + WIN]
        rms[i] = np.sqrt(np.mean(seg ** 2)) if seg.size else 0.0
    abs_loudness = float(np.median(rms))    # 未归一化的绝对响度（跨歌可比）
    if rms.max() > 0:
        rms = rms / rms.max()
    t = (np.arange(n) * HOP + WIN / 2) / SR
    # ★绝对响度参考：归一化会让慢歌/燥歌各自铺满[0,1]，跨歌"慢vs快"的依据丢失。
    #   这里在归一化后仍以未归一 RMS 中位数为准另存，供 shotplan 判"安静歌"。
    return rms, t, abs_loudness


def _chroma(y):
    mag, _ = _spectrogram(y)
    freqs = np.linspace(0, SR / 2, mag.shape[0])
    chroma = np.zeros((12, mag.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(freqs > 0, freqs, np.nan) / 440.0)
    pc = np.mod(np.round(midi), 12)
    for b in range(mag.shape[0]):
        if not np.isnan(pc[b]):
            chroma[int(pc[b])] += mag[b]
    return chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)


def segment_sections(y, n_frames, kernel=32):
    chroma = _chroma(y).T
    ds = max(1, chroma.shape[0] // 512)
    c = chroma[::ds]
    S = c @ c.T
    L = kernel
    g = np.outer(np.hanning(2 * L), np.hanning(2 * L))
    ck = np.ones((2 * L, 2 * L)); ck[:L, L:] = -1; ck[L:, :L] = -1; ck *= g
    nov = np.zeros(c.shape[0])
    for i in range(L, c.shape[0] - L):
        nov[i] = (S[i - L:i + L, i - L:i + L] * ck).sum()
    nov = np.maximum(0, nov)
    if nov.max() > 0:
        nov = nov / nov.max()
    peaks, _ = find_peaks(nov, height=0.25, distance=max(1, c.shape[0] // 8))
    bounds = np.unique(np.concatenate([[0], peaks * ds, [n_frames - 1]]))
    return bounds, nov, ds


def _resample(signal, t_base, fps, n_video):
    t_vid = np.arange(n_video) / float(fps)
    return np.interp(t_vid, t_base, signal, left=signal[0], right=signal[-1])


def analyze_music(path, fps, n_video_frames=None):
    fps = float(Fraction(fps)) if not isinstance(fps, (int, float)) else float(fps)
    log('  音乐 · 解码音频…')
    y = load_audio(path)
    duration = len(y) / SR
    if n_video_frames is None:
        n_video_frames = int(round(duration * fps))

    log('  音乐 · 起音包络（STFT）…')
    env, t_env = onset_envelope(y)
    env_fps = 1.0 / np.median(np.diff(t_env))
    log('  音乐 · 速度/拍点跟踪…')
    bpm, period = estimate_tempo(env, env_fps)
    beat_env = track_beats(env, period)
    beat_ref = refine_beats(env, beat_env, period)
    is_down = detect_downbeats(env, beat_env)

    idx = np.arange(len(t_env))
    beat_t = np.interp(beat_ref, idx, t_env)
    beat_grid = [{
        "t": round(float(beat_t[k]), 4),
        "frame": int(round(beat_t[k] * fps)),
        "is_downbeat": bool(is_down[k]),
        "strength": round(float(env[min(int(beat_env[k]), len(env) - 1)]), 4),
    } for k in range(len(beat_env))]

    log('  音乐 · 能量曲线…')
    energy_env, t_e, abs_loudness = energy_curve_env(y)
    energy_curve = _resample(energy_env, t_e, fps, n_video_frames)

    log('  音乐 · Foote 段落分段…')
    bounds, nov, ds = segment_sections(y, len(env))
    bounds_t = np.interp(bounds, idx, t_env)
    bounds_f = np.unique(np.clip(np.round(bounds_t * fps).astype(int),
                                 0, n_video_frames))
    sections = []
    for i in range(len(bounds_f) - 1):
        s, e = int(bounds_f[i]), int(bounds_f[i + 1])
        if e <= s:
            continue
        em = float(energy_curve[s:e].mean())
        lab = "low" if em < 0.33 else ("mid" if em < 0.66 else "high")
        sections.append({"start_f": s, "end_f": e, "label": lab,
                         "energy": round(em, 4)})

    return {
        "fps": fps, "bpm": round(float(bpm), 2),
        "duration": round(float(duration), 3), "n_frames": int(n_video_frames),
        "beat_grid": beat_grid,
        "energy_curve": [round(float(x), 4) for x in energy_curve],
        "sections": sections,
        "abs_loudness": round(float(abs_loudness), 6),
        "_debug": {"env": env, "t_env": t_env, "novelty": nov, "ds": ds,
                   "energy_env": energy_env, "t_e": t_e, "y": y},
    }
