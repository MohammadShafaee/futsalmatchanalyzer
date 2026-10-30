"""Jersey numbers: read the number on a player's back with a scene-text model and give it to the whole track.

Numbers are only legible when a player is close enough and turned away from the camera, so most crops
read nothing useful. The approach follows the SoccerNet jersey-number pipelines:
1. candidates: detected (not interpolated) boxes at least `min_height_px` tall, a few spread over each track;
2. read the torso crop with PARSeq (pretrained scene-text recogniser, via torch.hub);
3. keep only one- or two-digit reads above `min_conf`;
4. per track, sum the confidence of each number on the team's roster (if one is given); a number wins with
   enough votes and a clear majority;
5. per team, a number can't be on court twice at once: overlapping tracks keep the better-supported one.

Reads are cached in jersey_reads.csv (keyed by frame and box), so re-analysing a run and voting again
costs nothing; jerseys.json holds the result per stitched track.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import video
from .config import JerseyConfig

LOG = logging.getLogger(__name__)
READS = "jersey_reads.csv"
RESULT = "jerseys.json"


def _key(frame, x1, y1, x2, y2) -> str:
    return f"{int(frame)}:{round(float(x1))}:{round(float(y1))}:{round(float(x2))}:{round(float(y2))}"


def torso(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """The back/chest of the shirt, where the number is: below the shoulders, above the waist."""
    h, w = y2 - y1, x2 - x1
    return frame[int(y1 + .12 * h):int(y1 + .5 * h), int(x1 + .1 * w):int(x2 - .1 * w)]


def is_number(label: str) -> bool:
    return label.isdigit() and 1 <= len(label) <= 2 and label != "00"


def candidates(rows: list[dict], cfg: JerseyConfig) -> list[dict]:
    """Up to cfg.max_reads_per_track tall, detected boxes per track, spread evenly over the track."""
    by_track = defaultdict(list)
    for r in rows:
        if r["role"] in ("player", "goalkeeper") and r["source"] == "detected" \
                and float(r["y2"]) - float(r["y1"]) >= cfg.min_height_px:
            by_track[r["track_id"]].append(r)
    picked = []
    for rs in by_track.values():
        idx = np.unique(np.linspace(0, len(rs) - 1, min(len(rs), cfg.max_reads_per_track)).round().astype(int))
        picked += [rs[i] for i in idx]
    return picked


class Reader:
    """PARSeq from torch.hub (downloads ~90 MB once into the torch hub cache)."""

    def __init__(self, device: str | None = None):
        import torch
        from torchvision import transforms as T
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(self.device)
        self.tf = T.Compose([T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])

    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import cv2
        from PIL import Image
        batch = self.torch.stack([self.tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]).to(self.device)
        with self.torch.no_grad():
            labels, confs = self.model.tokenizer.decode(self.model(batch).softmax(-1))
        return [(lab, float(c.prod())) for lab, c in zip(labels, confs)]


def read_all(rows: list[dict], cfg: JerseyConfig, video_path: Path, run_dir: Path) -> dict[str, tuple[str, float]]:
    """OCR every candidate not read before; returns key -> (label, conf) for all cached reads."""
    path = run_dir / READS
    cache = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            cache = {r["key"]: (r["label"], float(r["conf"])) for r in csv.DictReader(f)}
    todo = [r for r in candidates(rows, cfg) if _key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"]) not in cache]
    LOG.info("Jerseys: %d crops to read (%d cached)", len(todo), len(cache))
    if todo:
        reader = Reader(cfg.device)
        by_frame = defaultdict(list)
        for r in todo:
            by_frame[int(r["frame"])].append(r)
        batch, keys, done = [], [], 0
        fresh = not path.exists()
        new = open(path, "a", newline="", encoding="utf-8")
        w = csv.writer(new)
        if fresh:
            w.writerow(["key", "label", "conf"])

        def flush():
            nonlocal batch, keys
            for k, (lab, c) in zip(keys, reader.read(batch)):
                cache[k] = (lab, c)
                w.writerow([k, lab, round(c, 4)])
            batch, keys = [], []

        try:
            for index, frame in video.frames(video_path, wanted=set(by_frame)):
                for r in by_frame[index]:
                    c = torso(frame, *(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
                    if c.size and min(c.shape[:2]) >= 4:
                        batch.append(c)
                        keys.append(_key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"]))
                if len(batch) >= 256:
                    flush()
                    done += 256
                    if done % 5120 == 0:
                        LOG.info("Jerseys: %d/%d read", done, len(todo))
            if batch:
                flush()
        finally:
            new.close()
    return cache


def vote(rows: list[dict], reads: dict[str, tuple[str, float]], cfg: JerseyConfig) -> dict[str, dict]:
    """Number per track from its confident reads, then drop same-number tracks that overlap in time."""
    votes = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    span, team_of = {}, {}
    for r in rows:
        tid = r["track_id"]
        f = int(r["frame"])
        lo, hi = span.get(tid, (f, f))
        span[tid] = (min(lo, f), max(hi, f))
        team_of[tid] = r["team"]
        got = reads.get(_key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"]))
        allowed = cfg.roster.get(r["team"]) if cfg.roster else None
        if got and is_number(got[0]) and got[1] >= cfg.min_conf and (allowed is None or int(got[0]) in allowed):
            votes[tid][got[0]] += got[1]
            counts[tid][got[0]] += 1
    result = {}
    for tid, v in votes.items():
        number, best = max(v.items(), key=lambda kv: kv[1])
        total = sum(v.values())
        if best >= cfg.min_votes and best / total >= cfg.min_share and team_of[tid]:
            result[tid] = {"team": team_of[tid], "number": int(number), "votes": round(best, 2),
                           "reads": counts[tid][number], "share": round(best / total, 2),
                           "first_frame": span[tid][0], "last_frame": span[tid][1]}
    # one shirt per team on court at a time: keep the better-supported of two overlapping tracks
    kept = {}
    by_shirt = defaultdict(list)
    for tid, j in result.items():
        by_shirt[(j["team"], j["number"])].append(tid)
    dropped = 0
    for tids in by_shirt.values():
        taken = []
        for tid in sorted(tids, key=lambda t: -result[t]["votes"]):
            a, b = result[tid]["first_frame"], result[tid]["last_frame"]
            if any(a < y and x < b for x, y in taken):
                dropped += 1
                continue
            taken.append((a, b))
            kept[tid] = result[tid]
    LOG.info("Jerseys: %d tracks numbered (%d dropped as overlapping a better read of the same shirt)",
             len(kept), dropped)
    return kept


def run(cfg: JerseyConfig, run_dir: Path) -> dict[str, dict]:
    with (run_dir / "detections.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["video"]
    reads = read_all(rows, cfg, Path(meta["path"]), run_dir)
    result = vote(rows, reads, cfg)
    (run_dir / RESULT).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
