from __future__ import annotations

from pathlib import Path

# 指向 multi_person_tracking/configs（本包内）
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

TRACKER_ALIASES = {
    "bytetrack": CONFIG_DIR / "bytetrack_default.yaml",
    "bytetrack_default": CONFIG_DIR / "bytetrack_default.yaml",
    "bytetrack_loose": CONFIG_DIR / "bytetrack_dance_loose.yaml",
    "bytetrack_dance_loose": CONFIG_DIR / "bytetrack_dance_loose.yaml",
}


def resolve_tracker_config(value: str) -> str:
    """把 tracker 别名或 YAML 路径解析成 Ultralytics 可用的路径。"""
    alias = TRACKER_ALIASES.get(value)
    path = alias if alias is not None else Path(value)
    if not path.exists():
        known = ", ".join(sorted(TRACKER_ALIASES))
        raise FileNotFoundError(f"找不到 tracker 配置: {value}. 已知别名: {known}")
    return str(path)
