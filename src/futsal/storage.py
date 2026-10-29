"""CSV/JSON files written to a run directory.

raw_detections.csv  YOLO + ByteTrack output; lets `analyze` re-run without the slow detector
detections.csv      on-court players with stitched IDs and kits
frames.csv          one row per analyzed frame: ball position and possession
summary.json        configuration, video metadata and statistics
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .detect import Detection
from .metrics import FrameState
from .tracks import Track

RAW_FIELDS = ["frame", "kind", "conf", "track_id", "x1", "y1", "x2", "y2"]


def write_raw(path: Path, detections: list[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RAW_FIELDS)
        for d in detections:
            w.writerow([d.frame, d.kind, d.conf, "" if d.track_id is None else d.track_id,
                        *(round(float(v), 1) for v in (d.x1, d.y1, d.x2, d.y2))])


def read_raw(path: Path) -> list[Detection]:
    with path.open(newline="", encoding="utf-8") as f:
        return [Detection(int(r["frame"]), r["kind"], float(r["conf"]),
                          float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                          int(r["track_id"]) if r["track_id"] else None)
                for r in csv.DictReader(f)]


def write_detections(path: Path, tracks: dict[int, Track], fps: float) -> None:
    rows = sorted(((d, t) for t in tracks.values() for d in t.detections), key=lambda x: (x[0].frame, x[1].id))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "frame", "track_id", "kit", "team", "role", "conf",
                    "x1", "y1", "x2", "y2", "foot_x", "foot_y", "source"])
        for d, t in rows:
            k = t.kit
            w.writerow([round(d.frame / fps, 3), d.frame, t.id, k.name if k else "", (k.team or "") if k else "",
                        k.role if k else "", d.conf, *(round(float(v), 1) for v in (d.x1, d.y1, d.x2, d.y2)),
                        round(d.foot[0], 1), round(d.foot[1], 1), "interpolated" if d.interpolated else "detected"])


def write_frames(path: Path, states: list[FrameState], fps: float) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "frame", "ball_x", "ball_y", "ball_source", "ball_conf",
                    "controller_track", "controller_team", "possession_team"])
        for s in states:
            b = s.ball
            w.writerow([round(s.frame / fps, 3), s.frame,
                        round(b.x, 1) if b else "", round(b.y, 1) if b else "", b.source if b else "",
                        b.conf if b and b.conf is not None else "",
                        "" if s.controller is None else s.controller, s.controller_team or "", s.possession or ""])


def write_summary(path: Path, data: dict) -> None:
    def default(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        return str(o)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)
