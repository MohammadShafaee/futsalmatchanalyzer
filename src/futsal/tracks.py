"""Per-player tracks: kit assignment, stitching of fragmented ByteTrack IDs, cleanup."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig, Kit
from .detect import Detection
from .kits import KitClassifier


@dataclass
class Track:
    id: int
    detections: list[Detection] = field(default_factory=list)
    shares: np.ndarray | None = None   # summed kit votes
    samples: int = 0
    kit: Kit | None = None
    merged_ids: list[int] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.detections[0].frame

    @property
    def end(self) -> int:
        return self.detections[-1].frame

    def add_votes(self, shares: np.ndarray | None) -> None:
        if shares is None:
            return
        self.shares = shares.copy() if self.shares is None else self.shares + shares
        self.samples += 1


def group(detections: list[Detection]) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    for d in sorted((d for d in detections if d.kind == "person"), key=lambda d: d.frame):
        tracks.setdefault(d.track_id, Track(d.track_id)).detections.append(d)
    return tracks


def _compatible(a: Kit | None, b: Kit | None) -> bool:
    return a is None or b is None or a.name == b.name


def stitch(tracks: dict[int, Track], classifier: KitClassifier, cfg: AnalysisConfig, fps: float) -> dict[int, Track]:
    """Greedily join a track that ends to one that starts shortly after, nearby and in the same kit."""
    max_gap = cfg.stitch_max_gap_s * fps
    pairs = []
    for a in tracks.values():
        ax, ay = a.detections[-1].foot
        for b in tracks.values():
            gap = b.start - a.end
            if b is a or gap <= 0 or gap > max_gap or not _compatible(a.kit, b.kit):
                continue
            bx, by = b.detections[0].foot
            dist = math.hypot(bx - ax, by - ay)
            # allow some slack for box jitter even across short gaps
            if dist <= 40 + cfg.stitch_max_speed_px_s * gap / fps:
                pairs.append((dist + gap, a.id, b.id))

    parent = {tid: tid for tid in tracks}
    has_next, has_prev = set(), set()
    def root(t):
        while parent[t] != t:
            t = parent[t]
        return t
    for _, a, b in sorted(pairs):
        if a in has_next or b in has_prev or root(a) == root(b):
            continue
        # chains must stay in one kit once merged
        if not _compatible(tracks[root(a)].kit, tracks[b].kit):
            continue
        has_next.add(a); has_prev.add(b)
        parent[b] = root(a)
        tracks[root(a)].kit = tracks[root(a)].kit or tracks[b].kit

    merged: dict[int, Track] = {}
    for tid in sorted(tracks, key=lambda t: tracks[t].start):
        src, r = tracks[tid], root(tid)
        dst = merged.setdefault(r, Track(r))
        dst.detections.extend(src.detections)
        dst.merged_ids.append(tid)
        if src.shares is not None:
            dst.shares = src.shares.copy() if dst.shares is None else dst.shares + src.shares
            dst.samples += src.samples
    for t in merged.values():
        t.detections.sort(key=lambda d: d.frame)
        for d in t.detections:
            d.track_id = t.id
        t.kit = classifier.decide(t.shares, t.samples) if t.shares is not None else None
    return merged


def drop_short(tracks: dict[int, Track], min_frames: int) -> dict[int, Track]:
    return {tid: t for tid, t in tracks.items() if len(t.detections) >= min_frames}
