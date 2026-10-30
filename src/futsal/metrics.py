"""Possession, per-player statistics and data-quality indicators."""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from .ball import BallPoint
from .config import AnalysisConfig, Config
from .detect import Detection
from .tracks import Track


@dataclass
class FrameState:
    frame: int
    ball: BallPoint | None
    controller: int | None       # track id of the player closest to the ball, if close enough
    controller_team: str | None
    possession: str | None       # smoothed team in possession


def ball_distance(ball: BallPoint, d: Detection) -> float:
    """Distance from the ball to the lower half of a player's box (0 if inside)."""
    top = (d.y1 + d.y2) / 2
    dx = max(d.x1 - ball.x, 0, ball.x - d.x2)
    dy = max(top - ball.y, 0, ball.y - d.y2)
    return math.hypot(dx, dy)


def possession(frames: list[int], ball: dict[int, BallPoint], tracks: dict[int, Track],
               cfg: AnalysisConfig, fps: float) -> list[FrameState]:
    people: dict[int, list[tuple[Detection, Track]]] = defaultdict(list)
    for t in tracks.values():
        if t.kit is not None and t.kit.team is not None:
            for d in t.detections:
                people[d.frame].append((d, t))

    states, current, pending, pending_since, last_ball = [], None, None, None, None
    for f in frames:
        b = ball.get(f)
        controller = team = None
        if b is not None:
            last_ball = f
            near = [(ball_distance(b, d), t) for d, t in people.get(f, [])]
            if near:
                dist, t = min(near, key=lambda x: x[0])
                if dist <= cfg.possession_radius_px:
                    controller, team = t.id, t.kit.team
        elif last_ball is None or f - last_ball > cfg.ball_max_gap_s * fps:
            current = pending = None   # ball lost for too long: possession unknown

        if team is not None and team != current:
            if pending != team:
                pending, pending_since = team, f
            if f - pending_since >= cfg.possession_min_s * fps:
                current, pending = team, None
        elif team is not None:
            pending = None
        states.append(FrameState(f, b, controller, team, current))
    return states


def summarize(cfg: Config, fps: float, stride: int, frames: list[int], tracks: dict[int, Track],
              raw_track_count: int, states: list[FrameState]) -> dict:
    step = stride / fps
    held = Counter(s.possession for s in states if s.possession)
    total_held = sum(held.values())
    teams = [{"team": name,
              "possession_seconds": round(held[name] * step, 1),
              "possession_percent": round(100 * held[name] / total_held, 1) if total_held else None,
              "tracks": sum(1 for t in tracks.values() if t.kit and t.kit.team == name)}
             for name in cfg.teams]

    touches = Counter()
    previous = None
    for s in states:
        if s.controller is not None and s.controller != previous:
            touches[s.controller] += 1
        previous = s.controller

    players = []
    for t in tracks.values():
        dist = 0.0
        for a, b in zip(t.detections, t.detections[1:]):
            if b.frame - a.frame <= 2 * stride:   # skip jumps across gaps
                dist += math.hypot(b.foot[0] - a.foot[0], b.foot[1] - a.foot[1])
        players.append({"track_id": t.id, "kit": t.kit.name if t.kit else None,
                        "team": t.kit.team if t.kit else None, "role": t.kit.role if t.kit else None,
                        "first_seen_s": round(t.start / fps, 2), "last_seen_s": round(t.end / fps, 2),
                        "observed_seconds": round(len(t.detections) * step, 2),
                        "distance_px": round(dist, 1), "ball_touches": touches[t.id],
                        "merged_from": t.merged_ids})
    players.sort(key=lambda p: (p["team"] or "~", -p["observed_seconds"]))

    per_frame = defaultdict(Counter)
    for t in tracks.values():
        for d in t.detections:
            per_frame[d.frame][t.kit.name if t.kit else "unassigned"] += 1
    labels = [k.name for k in cfg.kits] + ["unassigned"]
    detected = sum(1 for s in states if s.ball and s.ball.source == "detected")
    interpolated = sum(1 for s in states if s.ball and s.ball.source == "interpolated")
    n = len(frames) or 1
    quality = {
        "analyzed_frames": len(frames),
        "ball_detected_percent": round(100 * detected / n, 1),
        "ball_interpolated_percent": round(100 * interpolated / n, 1),
        "possession_known_percent": round(100 * total_held / n, 1),
        "raw_track_ids": raw_track_count,
        "tracks_after_stitching": len(tracks),
        "median_people_per_frame": {k: statistics.median(per_frame[f][k] for f in frames) if frames else 0
                                    for k in labels},
    }
    if cfg.expected_on_court:
        # share of frames where each kit is seen exactly the expected number of times
        expected = cfg.expected_on_court
        scored = frames
        if cfg.in_play:
            scored = [f for f in frames if any(a <= f / fps < b for a, b in cfg.in_play)]
            quality["in_play_frames"] = len(scored)
        m = len(scored) or 1
        exact = {k: round(100 * sum(per_frame[f][k] == want for f in scored) / m, 1) for k, want in expected.items()}
        all_ok = sum(all(per_frame[f][k] == want for k, want in expected.items()) for f in scored)
        quality["expected_count_match_percent"] = {**exact, "all": round(100 * all_ok / m, 1)}
    return {"teams": teams, "quality": quality, "players": players, "limitations": [
        "Possession uses proximity to the ball, not verified contact.",
        "Distances are in image pixels; court calibration is needed for metres.",
        "Passes, shots, goals and fouls are not classified yet."]}
