"""Coach-report export: turns an analysed run into the small JSON files behind the dashboard.

Everything is aggregated so a full match fits the dashboard's file limits:
- heat.json      time spent per 16x16 px cell of the cropped picture, per team / ball / long player track and half
- pitch_heat.json  (with a pitch in the config) the same per 1 m cell of the court seen from above
- running.json   (with a pitch) metres per player-minute of each team's outfield players, per five minutes of play
- shape.json     (with a pitch) each team's outfield block per minute of play: centre, width, depth, line height, area
- replay.csv     (with a pitch) everyone on court and the ball in metres, every third analysed frame during play;
                 kind H/A = home/away outfield player, h/a = their goalkeeper, R = referee, B = ball
- pitch_heat rows carry a phase: with / without the ball (team in possession), none (nobody has it), all (ball layer)
- timeline.json  possession and team positions in short windows
- zones.json     time the ball spent in each third, per half
- players.json   tracks seen on court long enough to matter
- match.json     clip length, half-time, tracking quality, arbiter activity
- court.txt      the empty court (median of frames) as a JPEG data URI

Positions between the goals are expressed from Manzano's point of view (the first team in the config):
0 = the goal it defends, 1 = the goal it attacks, so both halves read the same way after the teams change ends.
"""
from __future__ import annotations

import base64
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import arbiter, video
from .config import Config
from .court import Pitch
from .metrics import MAX_RUN_SPEED_M_S

CELL = 16


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _line_x(points, y: float) -> float:
    (x1, y1), (x2, y2) = points
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


def along(cfg: Config, x: float, y: float) -> float:
    """0 at the left goal line, 0.5 at the halfway line, 1 at the right goal line (piecewise linear)."""
    g = cfg.goals
    left, right, half = _line_x(g.left, y), _line_x(g.right, y), g.halfway_x
    u = 0.5 * (x - left) / (half - left) if x < half else 0.5 + 0.5 * (x - half) / (right - half)
    return float(min(1.0, max(0.0, u)))


def _halftime(cfg: Config, dets: list[dict], team: str, u_of) -> tuple[float | None, bool]:
    """Second at which the team's goalkeeper changes ends, and whether it starts in the left goal."""
    gk = next((k.name for k in cfg.kits if k.team == team and k.role == "goalkeeper"), None)
    by_min = defaultdict(list)
    for d in dets:
        if d["kit"] == gk:
            by_min[int(float(d["timestamp_s"]) // 60)].append(u_of(float(d["foot_x"]), float(d["foot_y"])))
    sides = {m: statistics.median(v) < 0.5 for m, v in sorted(by_min.items()) if len(v) >= 10}
    if not sides:
        return None, True
    minutes = sorted(sides)
    starts_left = Counter(sides[m] for m in minutes[:5]).most_common(1)[0][0]
    for i, m in enumerate(minutes):
        later = [sides[n] for n in minutes[i:i + 4]]
        if len(later) >= 3 and all(s != starts_left for s in later):
            return m * 60.0, starts_left
    return None, starts_left


def export(cfg: Config, run_dir: Path, out: Path) -> dict:
    if cfg.goals is None and cfg.pitch is None:
        raise ValueError("The config needs a 'pitch' or 'goals' section for the coach report")
    pitch = Pitch(cfg.pitch.length_m, cfg.pitch.width_m, cfg.pitch.points) if cfg.pitch else None
    u_of = pitch.along if pitch else (lambda x, y: along(cfg, x, y))
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    meta = summary["video"]
    top, bottom = meta["crop_rows"]
    step = meta["stride"] / meta["fps"]
    home, away = cfg.teams[0], cfg.teams[1]
    dets, frames = _rows(run_dir / "detections.csv"), _rows(run_dir / "frames.csv")
    duration = meta["duration_s"]

    halftime, home_left = _halftime(cfg, dets, home, u_of)
    if cfg.in_play and len(cfg.in_play) >= 2:
        halftime = cfg.in_play[1][0]  # the configured periods beat the goalkeeper guess
    half_of = (lambda t: 1) if halftime is None else (lambda t: 1 if t < halftime else 2)
    live = (lambda t: True) if not cfg.in_play else (lambda t: any(a <= t < b for a, b in cfg.in_play))

    def home_view(u: float, t: float) -> float:
        """Position between the goals as seen by the home team (0 = its own goal)."""
        left_is_home = home_left if half_of(t) == 1 else not home_left
        return u if left_is_home else 1 - u

    # ---- heatmaps: seconds per cell ----
    on_court = defaultdict(float)
    for d in dets:
        if d["team"] and d["role"] in ("player", "goalkeeper"):
            on_court[int(d["track_id"])] += step
    min_track = 5.0 if duration < 120 else 60.0
    ranked = defaultdict(list)
    for p in summary["players"]:
        if p["team"] and p["role"] == "player" and on_court[p["track_id"]] >= min_track:
            ranked[p["team"]].append(p["track_id"])
    heat_tracks = {tid for team in ranked.values() for tid in sorted(team, key=lambda t: -on_court[t])[:8]}

    people = [d for d in dets if d["team"] and d["role"] in ("player", "goalkeeper")]
    people_m = pitch.to_m([(float(d["foot_x"]), float(d["foot_y"])) for d in people]) if pitch and people else None
    balls = [f for f in frames if f["ball_x"]]
    balls_m = pitch.to_m([(float(f["ball_x"]), float(f["ball_y"])) for f in balls]) if pitch and balls else None
    u_m = lambda xm: float(min(1.0, max(0.0, xm / pitch.length)))

    cells, pitch_cells = defaultdict(float), defaultdict(float)

    def add_pitch_cell(layer: str, t: float, m, phase: str = "all") -> None:
        # 1 m cells on the court seen from above, a metre of slack round the lines for feet on them
        if -1 <= m[0] <= pitch.length + 1 and -1 <= m[1] <= pitch.width + 1:
            gx, gy = min(int(max(m[0], 0)), int(pitch.length) - 1), min(int(max(m[1], 0)), int(pitch.width) - 1)
            pitch_cells[(layer, half_of(t), phase, gx, gy)] += step

    poss_of = {f["frame"]: f["possession_team"] for f in frames}
    per_frame_u = defaultdict(list)
    tracks_m = defaultdict(list)
    shape_pts = defaultdict(list)   # (frame, team) -> outfield positions in metres, during play
    for i, d in enumerate(people):
        t, x, y = float(d["timestamp_s"]), float(d["foot_x"]), min(float(d["foot_y"]), bottom - 1)
        cell = (int(x // CELL), int((y - top) // CELL))
        if d["role"] == "player":
            cells[(d["team"], half_of(t), *cell)] += step
            if pitch:
                holder = poss_of.get(d["frame"])
                phase = "none" if not holder else "with" if holder == d["team"] else "without"
                add_pitch_cell(d["team"], t, people_m[i], phase)
                tracks_m[int(d["track_id"])].append((int(d["frame"]), t, d["team"], people_m[i]))
                if live(t):
                    shape_pts[(d["frame"], d["team"])].append(people_m[i])
            per_frame_u[(d["frame"], d["team"])].append(home_view(u_m(people_m[i][0]) if pitch else u_of(x, y), t))
        tid = int(d["track_id"])
        if tid in heat_tracks:
            cells[(f"#{tid}", half_of(t), *cell)] += step

    ball_u = {}
    for i, f in enumerate(balls):
        t, bx, by = float(f["timestamp_s"]), float(f["ball_x"]), float(f["ball_y"])
        cells[("Ball", half_of(t), int(bx // CELL), int((min(by, bottom - 1) - top) // CELL))] += step
        if pitch:
            add_pitch_cell("Ball", t, balls_m[i])
        ball_u[f["frame"]] = home_view(u_m(balls_m[i][0]) if pitch else u_of(bx, by), t)

    timeline_src, zones = [], defaultdict(float)
    for f in frames:
        t = float(f["timestamp_s"])
        bu = ball_u.get(f["frame"])
        if bu is not None and live(t):
            zones[(half_of(t), "own" if bu < 1 / 3 else "middle" if bu < 2 / 3 else "attack")] += step
        timeline_src.append((t, f["possession_team"], bu, f["frame"]))

    # ---- running: metres per player-minute of each team's outfield players, per five minutes of play ----
    running_rows = []
    if pitch:
        stride = meta["stride"]
        metres, seen = defaultdict(float), defaultdict(float)
        for points in tracks_m.values():
            points.sort(key=lambda p: p[0])
            for (fa, ta, team, ma), (fb, tb, _, mb) in zip(points, points[1:]):
                if not live(ta):
                    continue
                key = (int(ta // 300) * 5, team)
                seen[key] += step
                if fb - fa <= 2 * stride:
                    d = float(np.hypot(*(mb - ma)))
                    if d <= MAX_RUN_SPEED_M_S * (tb - ta):
                        metres[key] += d
        running_rows = [{"block_start_min": b, "half": half_of(b * 60.0 + 1), "team": team,
                         "metres": round(metres[(b, team)], 1), "player_minutes": round(seen[(b, team)] / 60, 2),
                         "m_per_min": round(metres[(b, team)] / (seen[(b, team)] / 60), 1) if seen[(b, team)] >= 30 else None}
                        for b, team in sorted(seen)]

    # ---- team shape per minute of play, seen with Manzano attacking left to right ----
    shape_rows = []
    if pitch:
        L, Wd = pitch.length, pitch.width
        per_min = defaultdict(lambda: defaultdict(list))
        for (frame, team), pts in shape_pts.items():
            if len(pts) < 3:
                continue
            t = int(frame) / meta["fps"]
            p = np.asarray(pts, dtype=np.float32)
            if not (home_left if half_of(t) == 1 else not home_left):
                p = np.c_[L - p[:, 0], Wd - p[:, 1]]   # turn the court round so Manzano's goal is on the left
            # line height: the deepest outfield player's distance from his own goal line
            line = float(p[:, 0].min()) if team == home else float(L - p[:, 0].max())
            area = float(cv2.contourArea(cv2.convexHull(p)))
            m = per_min[(int(t // 60), team)]
            for k, v in (("cx", p[:, 0].mean()), ("cy", p[:, 1].mean()), ("width", np.ptp(p[:, 1])),
                         ("depth", np.ptp(p[:, 0])), ("line", line), ("area", area)):
                m[k].append(float(v))
        shape_rows = [{"min": mi, "half": half_of(mi * 60.0 + 30), "team": team,
                       "centroid_x_m": round(np.mean(v["cx"]), 2), "centroid_y_m": round(np.mean(v["cy"]), 2),
                       "width_m": round(np.mean(v["width"]), 2), "depth_m": round(np.mean(v["depth"]), 2),
                       "line_height_m": round(np.mean(v["line"]), 2), "area_m2": round(np.mean(v["area"]), 1),
                       "frames": len(v["cx"])}
                      for (mi, team), v in sorted(per_min.items()) if len(v["cx"]) >= 20]

    # ---- 2D replay: everyone on court and the ball, ~2.5 times a second during play, in metres ----
    replay_lines = []
    if pitch:
        every = 3 * meta["stride"]
        # kind: H/A = home/away outfield player, h/a = their goalkeeper, R = referee, B = ball
        side = {home: "H", away: "A"}
        code = {k.name: "R" if k.role == "referee" else side.get(k.team, "?") if k.role == "player"
                else side.get(k.team, "?").lower() for k in cfg.kits}
        sample = [d for d in dets if d["kit"] and int(d["frame"]) % every == 0 and live(float(d["timestamp_s"]))]
        if sample:
            sm = pitch.to_m([(float(d["foot_x"]), float(d["foot_y"])) for d in sample])
            replay_lines += [f'{float(d["timestamp_s"]):.1f},{code.get(d["kit"], "?")},{m[0]:.1f},{m[1]:.1f}'
                             for d, m in zip(sample, sm) if -2 <= m[0] <= pitch.length + 2 and -2 <= m[1] <= pitch.width + 2]
        bs = [(i, f) for i, f in enumerate(balls) if int(f["frame"]) % every == 0 and live(float(f["timestamp_s"]))]
        replay_lines += [f'{float(f["timestamp_s"]):.1f},B,{balls_m[i][0]:.1f},{balls_m[i][1]:.1f}' for i, f in bs]
        replay_lines.sort(key=lambda s: float(s.split(",", 1)[0]))

    heat = [{"layer": layer, "half": half, "x": (gx + .5) * CELL, "y": (gy + .5) * CELL, "s": round(s, 2)}
            for (layer, half, gx, gy), s in cells.items()]

    # ---- timeline in windows ----
    window = 1.0 if duration < 120 else 10.0
    buckets = defaultdict(lambda: {"home": 0.0, "away": 0.0, "ball": [], "hu": [], "au": [], "n": 0, "players": []})
    for t, team, bu, frame in timeline_src:
        b = buckets[int(t // window)]
        b["n"] += 1
        b["home"] += step if team == home else 0
        b["away"] += step if team == away else 0
        if bu is not None:
            b["ball"].append(bu)
        hu, au = per_frame_u.get((frame, home)), per_frame_u.get((frame, away))
        if hu:
            b["hu"].append(np.mean(hu))
        if au:
            b["au"].append(np.mean(au))
        b["players"].append(len(hu or []) + len(au or []))
    mean = lambda v: round(float(np.mean(v)), 3) if v else None
    timeline = [{"min": round(w * window / 60, 3), "half": half_of(w * window),
                 "home_poss_s": round(b["home"], 2), "away_poss_s": round(b["away"], 2),
                 "ball_u": mean(b["ball"]), "home_u": mean(b["hu"]), "away_u": mean(b["au"]),
                 # the configured play periods, else: play is on when most outfield players are on court
                 "in_play": live(w * window) if cfg.in_play else bool(statistics.median(b["players"]) >= 5)}
                for w, b in sorted(buckets.items())]

    zone_names = {"own": f"{home} defensive third", "middle": "Middle third", "attack": f"{home} attacking third"}
    zone_rows = [{"half": h, "zone": zone_names[z], "seconds": round(zones[(h, z)], 1)}
                 for h in sorted({h for h, _ in zones} or {1}) for z in ("own", "middle", "attack")]

    players = sorted(({"track": p["track_id"], "team": p["team"], "role": p["role"],
                       "on_court_s": round(on_court[p["track_id"]], 1), "ball_touches": p["ball_touches"],
                       "first_min": round(p["first_seen_s"] / 60, 1), "last_min": round(p["last_seen_s"] / 60, 1),
                       "heatmap": p["track_id"] in heat_tracks}
                      for p in summary["players"] if p["team"] and on_court[p["track_id"]] >= min(20.0, min_track)),
                     key=lambda r: (r["team"], -r["on_court_s"]))

    q = summary["quality"]
    cache = run_dir / arbiter.CACHE
    answers = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    match = [{"duration_min": round(duration / 60, 1), "analysed_frames": q["analyzed_frames"],
              "crop_w": meta["width"], "crop_h": bottom - top,
              "halftime_min": round(halftime / 60, 1) if halftime is not None else None,
              "home_starts_left": home_left,
              "ball_seen_pct": q["ball_detected_percent"], "ball_filled_pct": q["ball_interpolated_percent"],
              "possession_known_pct": q["possession_known_percent"],
              "count_accuracy_pct": q.get("expected_count_match_percent", {}).get("all"),
              "ai_questions": len(answers),
              "ai_not_playing": sum(a.get("label") == arbiter.NOT_PLAYING for a in answers.values()),
              "ai_swaps": sum(a.get("label") == arbiter.MIXED for a in answers.values())}]

    files = {"heat": heat, "timeline": timeline, "zones": zone_rows, "players": players, "match": match}
    if pitch:
        match[0] |= {"pitch_length_m": pitch.length, "pitch_width_m": pitch.width,
                     "pitch_fit_px": round(pitch.residual_px, 1)}
        files["pitch_heat"] = [{"layer": layer, "half": half, "phase": phase, "x_m": gx + .5, "y_m": gy + .5, "s": round(s, 2)}
                               for (layer, half, phase, gx, gy), s in pitch_cells.items()]
        files["running"] = running_rows
        files["shape"] = shape_rows
        (out / "replay.csv").write_text("t,k,x,y\n" + "\n".join(replay_lines) + "\n", encoding="utf-8")
    for name, data in files.items():
        (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    # empty court: the per-pixel median of frames spread over the video removes the moving players
    wanted = set(np.linspace(0, meta["last_analyzed_frame"], 25).astype(int))
    stack = [f[top:bottom] for _, f in video.frames(Path(meta["path"]), wanted=wanted)]
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    bg = cv2.resize(bg, (960, round(960 * bg.shape[0] / bg.shape[1])), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 72])
    (out / "court.txt").write_text("data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(), encoding="utf-8")
    return {name: len(data) for name, data in files.items()} | {"court_px": [bg.shape[1], bg.shape[0]],
                                                              "crop": [meta["width"], bottom - top]}
