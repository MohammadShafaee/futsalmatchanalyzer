"""Ball trajectory: pick one plausible ball per frame and fill short gaps."""
from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass

from .config import AnalysisConfig
from .court import Court
from .detect import Detection

REACQUIRE_CONF = 0.4  # a confident detection may "teleport" the ball (missed detections, long pass)
HOTSPOT_CELL_PX = 24


@dataclass
class BallPoint:
    frame: int
    x: float
    y: float
    conf: float | None      # None when interpolated
    source: str             # "detected" or "interpolated"


def hotspots(balls: list[Detection], n_frames: int, max_share: float, cell: int = HOTSPOT_CELL_PX) -> set:
    """Grid cells holding a 'ball' in more than max_share of all frames: a lamp or a spare ball, not the
    match ball, which never stays in one spot for minutes of a match."""
    counts = Counter((int(d.center[0] // cell), int(d.center[1] // cell)) for d in balls)
    return {c for c, n in counts.items() if n > max_share * n_frames}


def trajectory(detections: list[Detection], frames: list[int], court: Court,
               cfg: AnalysisConfig, fps: float) -> dict[int, BallPoint]:
    balls = [d for d in detections if d.kind == "ball" and court.contains(d.center)]
    if cfg.ball_hotspot_max_share:
        hot = hotspots(balls, len(frames), cfg.ball_hotspot_max_share)
        balls = [d for d in balls if (int(d.center[0] // HOTSPOT_CELL_PX), int(d.center[1] // HOTSPOT_CELL_PX)) not in hot]
    by_frame: dict[int, list[Detection]] = {}
    for d in balls:
        by_frame.setdefault(d.frame, []).append(d)

    max_gap = cfg.ball_max_gap_s * fps
    picked: dict[int, BallPoint] = {}
    last: BallPoint | None = None
    for f in frames:
        candidates = sorted(by_frame.get(f, []), key=lambda d: d.conf, reverse=True)
        if not candidates:
            continue
        choice = None
        if last is None or f - last.frame > max_gap:
            choice = candidates[0]
        else:
            reach = 30 + cfg.ball_max_speed_px_s * (f - last.frame) / fps
            near = [d for d in candidates if math.hypot(d.center[0] - last.x, d.center[1] - last.y) <= reach]
            choice = near[0] if near else candidates[0] if candidates[0].conf >= REACQUIRE_CONF else None
        if choice is not None:
            last = picked[f] = BallPoint(f, *choice.center, choice.conf, "detected")

    detected = sorted(picked.values(), key=lambda p: p.frame)
    for a, b in zip(detected, detected[1:]):
        if b.frame - a.frame > max_gap:
            continue
        for f in frames[bisect.bisect_right(frames, a.frame):bisect.bisect_left(frames, b.frame)]:
            t = (f - a.frame) / (b.frame - a.frame)
            picked[f] = BallPoint(f, a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), None, "interpolated")
    return picked
