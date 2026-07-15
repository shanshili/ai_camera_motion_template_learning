from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_frames(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_track_stats(frames: list[dict[str, Any]], *, long_thresholds: list[int] | None = None) -> dict[str, Any]:
    thresholds = long_thresholds or [25, 50, 100, 200, 300, 500, 800, 1000, 1200, 1500]
    person_count_hist: Counter[int] = Counter()
    tracks: dict[int, list[int]] = defaultdict(list)

    for fallback_frame_index, frame in enumerate(frames):
        people = frame.get("people", [])
        frame_index = int(frame.get("frame_index", frame.get("frame", fallback_frame_index)))
        person_count_hist[len(people)] += 1
        for person in people:
            tracker_id = person.get("tracker_id")
            if tracker_id is not None:
                tracks[int(tracker_id)].append(frame_index)

    rows = []
    for tracker_id, track_frames in tracks.items():
        sorted_frames = sorted(track_frames)
        gaps = [b - a for a, b in zip(sorted_frames, sorted_frames[1:])]
        rows.append(
            {
                "tracker_id": tracker_id,
                "length": len(sorted_frames),
                "first_frame": sorted_frames[0],
                "last_frame": sorted_frames[-1],
                "max_gap": max(gaps) if gaps else 0,
                "gap_count": sum(1 for gap in gaps if gap > 1),
            }
        )

    rows.sort(key=lambda item: (item["length"], item["tracker_id"]), reverse=True)
    return {
        "frames": len(frames),
        "person_count_hist": dict(sorted(person_count_hist.items())),
        "tracked_people": sum(count * frames_with_count for count, frames_with_count in person_count_hist.items()),
        "unique_track_ids": len(tracks),
        "long_counts": {threshold: sum(1 for track_frames in tracks.values() if len(track_frames) >= threshold) for threshold in thresholds},
        "tracks": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tracked_keypoints.json track continuity.")
    parser.add_argument("--keypoints", required=True, help="Path to tracked_keypoints.json.")
    parser.add_argument("--top", type=int, default=30, help="Number of longest tracks to print.")
    parser.add_argument("--json", action="store_true", help="Print full statistics as JSON.")
    return parser.parse_args()


def print_text(stats: dict[str, Any], *, top: int) -> None:
    print(f"frames {stats['frames']}")
    print(f"person_count_hist {stats['person_count_hist']}")
    print(f"tracked_people {stats['tracked_people']}")
    print(f"unique_track_ids {stats['unique_track_ids']}")
    print(f"long_counts {stats['long_counts']}")
    print("top_tracks")
    for track in stats["tracks"][:top]:
        print(
            f"track_id={track['tracker_id']:>4} "
            f"length={track['length']:>4} "
            f"first={track['first_frame']:>4} "
            f"last={track['last_frame']:>4} "
            f"max_gap={track['max_gap']:>3} "
            f"gap_count={track['gap_count']}"
        )


def main() -> None:
    args = parse_args()
    stats = compute_track_stats(load_frames(Path(args.keypoints)))
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print_text(stats, top=args.top)
