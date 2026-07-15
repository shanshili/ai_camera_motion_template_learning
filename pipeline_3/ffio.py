"""
ffmpeg / ffprobe 输入输出工具
=============================
用 ffmpeg 做解码与编码；帧以 numpy(BGR) 在 Python 里流动。
解码/编码用同一套精确帧率/帧数，保证不丢帧、不漂移。
"""

import json
import subprocess
from fractions import Fraction

import numpy as np


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def ffprobe_meta(path: str) -> dict:
    """探测视频元信息：宽高、精确帧率、帧数、时长、是否有音轨。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = _run(cmd)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {out.stderr}")
    data = json.loads(out.stdout)
    st = data["streams"][0]

    rate_str = st.get("r_frame_rate", "0/0")
    if rate_str in ("0/0", "0/1"):
        rate_str = st.get("avg_frame_rate", "0/0")
    fps = Fraction(rate_str) if rate_str not in ("0/0", "") else Fraction(30, 1)

    duration = None
    for src in (st.get("duration"), data.get("format", {}).get("duration")):
        if src not in (None, "N/A"):
            duration = float(src)
            break

    nb = st.get("nb_frames")
    if nb not in (None, "N/A"):
        frame_count = int(nb)
    elif duration is not None:
        frame_count = int(round(duration * float(fps)))
    else:
        frame_count = _count_frames(path)

    a = _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
              "-show_entries", "stream=index", "-of", "csv=p=0", path])
    has_audio = bool(a.stdout.strip())

    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration if duration is not None else float(frame_count / fps),
        "has_audio": has_audio,
    }


def _count_frames(path: str) -> int:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", path])
    return int(out.stdout.strip())


def frame_reader(path: str, width: int, height: int):
    """逐帧解码为 BGR numpy 数组（生成器）。"""
    frame_bytes = width * height * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=frame_bytes,
    )
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, np.uint8).reshape((height, width, 3))
    finally:
        proc.stdout.close()
        proc.wait()


class FrameWriter:
    """把 BGR numpy 帧编码成视频，并可从原文件复用音轨（保持 A/V 同步）。"""

    def __init__(self, out_path, width, height, fps: Fraction,
                 audio_from=None, render_cfg=None):
        rc = render_cfg or {}
        if not isinstance(fps, Fraction):
            fps = Fraction(fps).limit_denominator(100000)
        fps_str = f"{fps.numerator}/{fps.denominator}"
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", fps_str, "-i", "pipe:0",
        ]
        if audio_from:
            cmd += ["-i", audio_from]
        cmd += [
            "-map", "0:v:0",
            "-c:v", rc.get("video_codec", "libx264"),
            "-preset", rc.get("preset", "medium"),
            "-crf", str(rc.get("crf", 18)),
            "-pix_fmt", rc.get("pix_fmt", "yuv420p"),
            "-fps_mode", "cfr", "-r", fps_str,
        ]
        if audio_from:
            cmd += [
                "-map", "1:a:0?",
                "-c:a", rc.get("audio_codec", "aac"),
                "-b:a", rc.get("audio_bitrate", "320k"),
                "-ar", str(rc.get("audio_rate", 48000)),
                "-shortest",
            ]
        cmd += [out_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr, np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError("ffmpeg 编码失败")
