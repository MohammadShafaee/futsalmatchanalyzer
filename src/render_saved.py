"""Rebuild an annotated video from detections.csv without re-running YOLO."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

DEFAULT_VIDEO = Path(r"D:\Football Match Analyzer\Manzano-Jesolo.App.mp4")


def parse_args():
    p = argparse.ArgumentParser(description="Render saved detections over a source match video.")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--detections", type=Path, required=True, help="Saved detections.csv from an analysis run")
    p.add_argument("--output", type=Path, required=True, help="New MP4 path on a drive with adequate free space")
    p.add_argument("--stride", type=int, default=15, help="Must match the stride used during analysis")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    by_frame = defaultdict(list)
    with args.detections.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row["frame_index"] = int(row["frame_index"])
            row["track_id"] = int(row["track_id"]) if row["track_id"] else None
            for key in ("x1", "y1", "x2", "y2"):
                row[key] = int(float(row[key]))
            by_frame[row["frame_index"]].append(row)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Cannot create output video: {args.output}")

    palette = {"Team 1": (70, 210, 255), "Team 2": (255, 130, 70),
               "Manzano": (70, 210, 255), "Jesolo": (255, 130, 70)}
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.stride == 0:
                for row in by_frame.get(frame_index, []):
                    x1, y1, x2, y2 = (row[k] for k in ("x1", "y1", "x2", "y2"))
                    ball = row["object_type"] == "ball"
                    team = row.get("team", "")
                    color = (40, 240, 255) if ball else palette.get(team, (150, 150, 150))
                    tid = row.get("track_id")
                    label = f"Ball {float(row['confidence']):.2f}" if ball else (
                        f"{team} #{tid}" if team and tid is not None else
                        f"Player #{tid}" if tid is not None else "Player")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if ball else 3)
                    y = max(22, y1 - 7)
                    cv2.rectangle(frame, (x1, y - 22), (min(width - 1, x1 + 245), y + 3), color, -1)
                    cv2.putText(frame, label, (x1 + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                .58, (10, 10, 10), 2, cv2.LINE_AA)
                writer.write(frame)
            frame_index += 1
            if frame_index % 3000 == 0:
                print(f"Rendered {frame_index:,} source frames")
    finally:
        cap.release()
        writer.release()
    print(f"Saved annotated video: {args.output}")


if __name__ == "__main__":
    main()
