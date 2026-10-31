"""Jersey numbers: read the number on a player's back with a scene-text model and give it to the whole track.

Numbers are only legible when a player is close enough and turned away from the camera, so most crops
read nothing useful. The approach follows the SoccerNet jersey-number pipelines:
1. candidates: detected (not interpolated) boxes at least `min_height_px` tall, a few spread over each track;
2. read the torso crop with PARSeq (pretrained scene-text recogniser, via torch.hub);
3. keep only one- or two-digit reads above `min_conf`;
4. per track, sum the confidence of each number on the team's roster (if one is given); a number wins with
   enough votes and a clear majority;
5. per team, a number can't be on court twice at once: overlapping tracks keep the better-supported one.

With `jev` on, tracks PARSeq leaves unnumbered (above all the dark-red numbers it can't read) are shown
to Jev (TypeSafe, see jev.py) a few times each, and its answers join the vote; cached in jersey_jev.csv.

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
JEV_READS = "jersey_jev.csv"
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


def jev_candidates(rows: list[dict], cfg: JerseyConfig, skip: set[str], fps: float) -> list[dict]:
    """The tallest detected boxes of each track not in `skip`, at least cfg.jev_gap_s apart."""
    by_track = defaultdict(list)
    for r in rows:
        if r["track_id"] not in skip and r["role"] in ("player", "goalkeeper") and r["source"] == "detected" \
                and float(r["y2"]) - float(r["y1"]) >= cfg.min_height_px:
            by_track[r["track_id"]].append(r)
    picked = []
    for rs in by_track.values():
        chosen = []
        for r in sorted(rs, key=lambda r: float(r["y1"]) - float(r["y2"])):
            if all(abs(int(r["frame"]) - int(c["frame"])) >= cfg.jev_gap_s * fps for c in chosen):
                chosen.append(r)
            if len(chosen) == cfg.jev_reads_per_track:
                break
        picked += chosen
    return picked


JEV_QUESTION = ("This is a futsal player filmed from the stand. Which number is printed on the back "
                "(or the front) of the shirt?")
JEV_OPTIONS = {**{str(n): f"the shirt number is {n}" for n in range(1, 100)},
               "none": "no number can be read: the player is turned sideways, the number is hidden or too blurry"}


def read_jev(rows: list[dict], cfg: JerseyConfig, video_path: Path, run_dir: Path, skip: set[str],
             fps: float) -> dict[str, tuple[str, float]]:
    """Ask Jev for the number on tracks the first reader left unnumbered; cached in jersey_jev.csv."""
    from concurrent.futures import ThreadPoolExecutor

    from . import jev
    path = run_dir / JEV_READS
    cache = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            cache = {r["key"]: (r["label"], float(r["conf"])) for r in csv.DictReader(f)}
    todo = [r for r in jev_candidates(rows, cfg, skip, fps) if _key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"]) not in cache]
    LOG.info("Jerseys: %d crops to ask Jev about (%d cached)", len(todo), len(cache))
    key = jev.api_key()
    if todo and not key:
        LOG.warning("Jerseys: JEV_API_KEY is not set (environment or .env); skipping Jev")
        return cache
    if not todo:
        return cache
    client = jev.Client(key, model=cfg.jev_model)
    by_frame = defaultdict(list)
    for r in todo:
        by_frame[int(r["frame"])].append(r)
    jobs = []
    for index, frame in video.frames(video_path, wanted=set(by_frame)):
        for r in by_frame[index]:
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            h, w = y2 - y1, x2 - x1
            crop = frame[max(0, int(y1 - .05 * h)):int(y2), max(0, int(x1 - .15 * w)):int(x2 + .15 * w)]
            jobs.append((_key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"]), crop.copy()))

    def ask(job):
        k, crop = job
        try:
            choice, conf, _ = client.choice(JEV_QUESTION, JEV_OPTIONS, image=crop)
            return k, choice, conf
        except jev.JevError as e:
            LOG.warning("Jerseys: Jev failed on %s: %s", k, e)
            return k, None, 0.0

    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(cfg.jev_threads) as pool:
        w = csv.writer(f)
        if fresh:
            w.writerow(["key", "label", "conf"])
        for n, (k, choice, conf) in enumerate(pool.map(ask, jobs), 1):
            if choice is not None:
                cache[k] = (choice, conf)
                w.writerow([k, choice, round(conf, 4)])
            if n % 500 == 0:
                f.flush()
                LOG.info("Jerseys: %d/%d asked (%d input tokens, about $%.2f)", n, len(jobs), client.input_tokens,
                         client.input_tokens * 0.042 / 1e6)
    LOG.info("Jerseys: Jev used %d input tokens (about $%.2f)", client.input_tokens, client.input_tokens * 0.042 / 1e6)
    return cache


def vote(rows: list[dict], reads: dict[str, tuple[str, float]], cfg: JerseyConfig,
         extra: dict[str, tuple[str, float]] | None = None) -> dict[str, dict]:
    """Number per track from its confident reads, then drop same-number tracks that overlap in time.

    `extra` holds a second reader's reads (Jev), counted the same way with its own confidence bar.
    """
    votes = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    span, team_of = {}, {}
    for r in rows:
        tid = r["track_id"]
        f = int(r["frame"])
        lo, hi = span.get(tid, (f, f))
        span[tid] = (min(lo, f), max(hi, f))
        team_of[tid] = r["team"]
        k = _key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"])
        allowed = cfg.roster.get(r["team"]) if cfg.roster else None
        for got, bar, ignored in ((reads.get(k), cfg.min_conf, cfg.parseq_ignore),
                                  ((extra or {}).get(k), cfg.jev_min_conf, ())):
            if got and is_number(got[0]) and got[1] >= bar and int(got[0]) not in ignored \
                    and (allowed is None or int(got[0]) in allowed):
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
    if cfg.jev:
        extra = read_jev(rows, cfg, Path(meta["path"]), run_dir, set(result), meta["fps"])
        result = vote(rows, reads, cfg, extra)
    (run_dir / RESULT).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
