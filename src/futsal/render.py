"""Draw a run's saved results over the source video (no detector needed)."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

from . import video
from .court import Court

LOG = logging.getLogger(__name__)
GREY = (150, 150, 150)
BALL = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _label(row: dict) -> str:
    if row["role"] == "referee":
        return "REF"
    prefix = "GK " if row["role"] == "goalkeeper" else ""
    return f"{prefix}{row['team'] or '?'} {row['track_id']}"


def render(run_dir: Path, dest: Path | None = None) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    src = Path(summary["video"]["path"])
    stride = summary["config"]["detector"]["stride"]
    top, bottom = summary["video"]["crop_rows"]
    colors = {k["name"]: tuple(reversed(k["rgb"])) for k in summary["config"]["kits"]}
    team_colors = {k["team"]: tuple(reversed(k["rgb"])) for k in summary["config"]["kits"]
                   if k["team"] and k["role"] == "player"}
    court = Court(summary["config"]["court"])

    people = defaultdict(list)
    for r in _read(run_dir / "detections.csv"):
        people[int(r["frame"])].append(r)
    states = {int(r["frame"]): r for r in _read(run_dir / "frames.csv")}

    info = video.probe(src)
    dest = dest or run_dir / "annotated.mp4"
    # write to a temporary file so an interrupted render never leaves a broken annotated.mp4
    tmp = dest.with_name(dest.stem + ".partial.mp4")
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), info.fps / stride, (info.width, bottom - top))
    if not out.isOpened():
        raise RuntimeError(f"Cannot create {tmp}")
    held = defaultdict(int)
    try:
        for index, frame in video.frames(src, wanted=set(states)):
            court.draw(frame)
            for r in people.get(index, []):
                x1, y1, x2, y2 = (int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
                color = colors.get(r["kit"], GREY)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, _label(r), (x1, y1 - 4), FONT, .45, color, 1, cv2.LINE_AA)
            s = states[index]
            if s["ball_x"]:
                center = (int(float(s["ball_x"])), int(float(s["ball_y"])))
                cv2.circle(frame, center, 9, BALL, 2 if s["ball_source"] == "detected" else 1, cv2.LINE_AA)
            if s["possession_team"]:
                held[s["possession_team"]] += 1
            crop = frame[top:bottom]
            _hud(crop, s, held, team_colors)
            out.write(crop)
    finally:
        out.release()
    _finish(tmp, dest)
    LOG.info("Annotated video: %s", dest)
    return dest


def _finish(tmp: Path, dest: Path) -> None:
    """Re-encode to H.264 when ffmpeg is available; mp4v does not play in most players."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        LOG.warning("ffmpeg not found; %s is mp4v and may not play everywhere", dest.name)
        tmp.replace(dest)
        return
    h264 = dest.with_name(dest.stem + ".h264.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264)], check=True)
    h264.replace(dest)
    tmp.unlink()


def _hud(frame, state: dict, held: dict, team_colors: dict) -> None:
    total = sum(held.values())
    text = f"t={float(state['timestamp_s']):6.1f}s  possession: {state['possession_team'] or '-'}"
    cv2.rectangle(frame, (0, 0), (720, 26), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 18), FONT, .55, (255, 255, 255), 1, cv2.LINE_AA)
    x = 400
    for team, color in team_colors.items():
        pct = 100 * held[team] / total if total else 0
        label = f"{team} {pct:.0f}%"
        cv2.putText(frame, label, (x, 18), FONT, .55, color, 2, cv2.LINE_AA)
        x += 140
