"""
统一进度日志
============
- 行缓冲 + flush：即时可见，管道/重定向下也不憋到最后。
- log(msg)：带总耗时前缀。
- Progress：长循环心跳，按百分比或最小步长打印 i/n + 速率 + 预估剩余。
- stage_banner / stage_done：阶段横幅与耗时。
"""

import sys
import time

_T0 = time.time()

# 让 print 即时可见（重定向/子进程下尤其重要，避免"看着像卡死"）
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def elapsed():
    return time.time() - _T0


def log(msg):
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def stage_banner(name, idx=None, total=None):
    tag = f"{idx}/{total} " if idx is not None else ""
    log(f"──────── ▶ 阶段 {tag}[{name}] 开始 ────────")


def stage_done(name, t0):
    log(f"──────── ✔ 阶段 [{name}] 完成，用时 {time.time() - t0:.1f}s ────────")


class Progress:
    """长循环心跳。用法：p = Progress(n, '渲染'); for i in ...: p.update(i)。"""

    def __init__(self, total, label="", every_frac=0.05, min_step=30):
        self.total = max(1, int(total))
        self.label = label
        self.step = max(int(min_step), int(self.total * every_frac))
        self.last = -1
        self.t0 = time.time()

    def update(self, i):
        if i - self.last < self.step and i + 1 < self.total:
            return
        self.last = i
        done = i + 1
        pct = 100.0 * done / self.total
        dt = time.time() - self.t0
        rate = done / dt if dt > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        print(f"[{elapsed():7.1f}s]     {self.label} {done}/{self.total} "
              f"({pct:4.1f}%) · {rate:5.0f} it/s · 剩约 {eta:4.0f}s", flush=True)

    def done(self):
        dt = time.time() - self.t0
        print(f"[{elapsed():7.1f}s]     {self.label} 完成 {self.total} 项，用时 {dt:.1f}s",
              flush=True)
