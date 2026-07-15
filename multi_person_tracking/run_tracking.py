from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from multi_person_tracking.trackers.registry import TRACKER_ALIASES, resolve_tracker_config


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def tensor_to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def result_to_tracked_record(result: Any, frame_index: int) -> dict[str, Any]:
    keypoints_xy = tensor_to_list(result.keypoints.xy) if result.keypoints is not None else []
    keypoints_xyn = tensor_to_list(result.keypoints.xyn) if result.keypoints is not None else []
    keypoints_data = tensor_to_list(result.keypoints.data) if result.keypoints is not None else []
    boxes = tensor_to_list(result.boxes.xyxy) if result.boxes is not None else []
    scores = tensor_to_list(result.boxes.conf) if result.boxes is not None else []
    tracker_ids = tensor_to_list(result.boxes.id) if result.boxes is not None and result.boxes.id is not None else []

    people = []
    for person_index, xy in enumerate(keypoints_xy):
        normalized = keypoints_xyn[person_index] if person_index < len(keypoints_xyn) else []
        raw = keypoints_data[person_index] if person_index < len(keypoints_data) else []
        tracker_id = tracker_ids[person_index] if person_index < len(tracker_ids) else None
        people.append(
            {
                "person_index": person_index,
                "tracker_id": int(tracker_id) if tracker_id is not None else None,
                "box_xyxy": boxes[person_index] if person_index < len(boxes) else None,
                "score": scores[person_index] if person_index < len(scores) else None,
                "keypoints": [
                    {
                        "name": KEYPOINT_NAMES[kpt_index] if kpt_index < len(KEYPOINT_NAMES) else f"kpt_{kpt_index}",
                        "xy": point,
                        "xyn": normalized[kpt_index] if kpt_index < len(normalized) else None,
                        "confidence": raw[kpt_index][2] if kpt_index < len(raw) and len(raw[kpt_index]) > 2 else None,
                    }
                    for kpt_index, point in enumerate(xy)
                ],
            }
        )

    return {
        "frame_index": frame_index,
        "source_path": str(result.path),
        "original_shape": list(result.orig_shape),
        "people": people,
    }


def run_tracking(
    *,
    source: str,
    model_path: str,
    output_dir: Path,
    tracker: str,
    device: str,
    imgsz: int,
    conf: float,
    save: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_config = resolve_tracker_config(tracker)
    print(f"[track] source={source}", flush=True)
    print(f"[track] tracker={tracker} -> {tracker_config}", flush=True)
    print(f"[track] 加载模型 {model_path} …（首次会下载/初始化，稍候）", flush=True)

    # 预检：带路径分隔符 = 当作本地文件，不会自动下载；不存在就早点给人话提示
    import os as _os
    if (("/" in model_path) or ("\\" in model_path)) and not _os.path.exists(model_path):
        raise FileNotFoundError(
            f"权重文件不存在: {model_path}\n"
            "  · 若想自动下载官方权重，请用不带路径的名字，例如 --model yolo11m-pose.pt "
            "（可选 n/s/m/l/x，越大越准越慢；名字里不能带 / 或 \\，否则会被当成本地文件）。\n"
            "  · 若已有本地权重，请把 --model 指向真实存在的 .pt 路径。")

    model = YOLO(model_path)
    track_kwargs: dict[str, Any] = {
        "source": source,
        "imgsz": imgsz,
        "conf": conf,
        "tracker": tracker_config,
        "save": save,
        "project": str(output_dir),
        "name": "rendered",
        "exist_ok": True,
        "stream": True,
        "persist": True,
        "verbose": False,   # 关掉 ultralytics 的逐帧 "video 1/1 (frame x/n) ..." 刷屏
    }
    if device:
        track_kwargs["device"] = device

    print(f"[track] 开始逐帧推理 + 跟踪（device={device or 'auto'}, imgsz={imgsz}, conf={conf}）…",
          flush=True)
    records = []
    tracked_people = 0
    for frame_index, result in enumerate(model.track(**track_kwargs)):
        record = result_to_tracked_record(result, frame_index)
        tracked_people += sum(1 for person in record["people"] if person.get("tracker_id") is not None)
        records.append(record)
        if frame_index % 30 == 0:
            npeople = len(record["people"])
            print(f"[track]   帧 {frame_index} · 当前 {npeople} 人", flush=True)

    print(f"[track] 推理完成：{len(records)} 帧，累计跟踪人次 {tracked_people}", flush=True)

    keypoints_path = output_dir / "tracked_keypoints.json"
    keypoints_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[track] 已写出 {keypoints_path}", flush=True)

    summary = {
        "source": source,
        "model": model_path,
        "tracker": tracker,
        "tracker_config": tracker_config,
        "output": str(keypoints_path),
        "frames": len(records),
        "tracked_people": tracked_people,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-pose with a bbox-based multi-object tracker.")
    parser.add_argument("--source", required=True, help="Video source accepted by Ultralytics.")
    parser.add_argument("--model", required=True, help="Path to YOLO pose weights.")
    parser.add_argument("--output-dir", required=True, help="Directory for tracked_keypoints.json and rendered media.")
    parser.add_argument(
        "--tracker",
        default="bytetrack_loose",
        help=f"Tracker alias or YAML path. Aliases: {', '.join(sorted(TRACKER_ALIASES))}",
    )
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or empty for Ultralytics default.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold for tracking.")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated tracking media.")
    return parser.parse_args()


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    print(f"[track] 启动：source={args.source} model={args.model} "
          f"out={args.output_dir} tracker={args.tracker}", flush=True)
    summary = run_tracking(
        source=args.source,
        model_path=args.model,
        output_dir=Path(args.output_dir).resolve(),
        tracker=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        save=not args.no_save,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

