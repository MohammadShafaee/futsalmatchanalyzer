"""Per-player tracks: kit assignment, ID-switch splitting, stitching of fragments, cleanup."""
from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig, Kit
from .detect import Detection
from .kits import KitClassifier

MAX_ON_COURT = {"goalkeeper": 1, "referee": 2}


@dataclass
class Track:
    id: int
    detections: list[Detection] = field(default_factory=list)
    kit: Kit | None = None
    merged_ids: list[int] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.detections[0].frame

    @property
    def end(self) -> int:
        return self.detections[-1].frame

    def decide_kit(self, classifier: KitClassifier) -> None:
        votes = [d.kit_shares for d in self.detections if d.kit_shares is not None]
        self.kit = classifier.decide(np.sum(votes, axis=0), len(votes)) if votes else None


def group(detections: list[Detection]) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    for d in sorted((d for d in detections if d.kind == "person"), key=lambda d: d.frame):
        tracks.setdefault(d.track_id, Track(d.track_id)).detections.append(d)
    return tracks


def _frame_labels(track: Track, classifier: KitClassifier, window: int) -> list[str | None]:
    """Kit name per detection, smoothed by a majority vote over neighbouring detections."""
    raw = [classifier.decide(d.kit_shares, 1) if d.kit_shares is not None else None for d in track.detections]
    raw = [k.name if k else None for k in raw]
    smoothed = []
    for i in range(len(raw)):
        votes = Counter(k for k in raw[max(0, i - window):i + window + 1] if k is not None)
        smoothed.append(votes.most_common(1)[0][0] if votes else None)
    return smoothed


def split_id_switches(tracks: dict[int, Track], classifier: KitClassifier, min_run: int) -> dict[int, Track]:
    """ByteTrack sometimes hands an ID from one person to another when they cross.

    When the shirt colour changes for at least `min_run` detections, the track is cut there.
    """
    next_id = itertools.count(max(tracks, default=0) + 1)
    out: dict[int, Track] = {}
    for t in tracks.values():
        labels = _frame_labels(t, classifier, window=min_run // 2)
        runs: list[list] = []                      # [label, first_index, last_index]
        for i, label in enumerate(labels):
            if runs and (label == runs[-1][0] or label is None):
                runs[-1][2] = i
            else:
                runs.append([label, i, i])
        # absorb short runs into the previous one; they are noise, not a different person
        merged: list[list] = []
        for run in runs:
            if merged and (run[2] - run[1] + 1 < min_run or run[0] in (None, merged[-1][0])):
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        if len(merged) > 1 and (merged[0][0] is None or merged[0][2] - merged[0][1] + 1 < min_run):
            merged[1][1] = merged[0][1]
            merged.pop(0)
        for n, (_, first, last) in enumerate(merged):
            tid = t.id if n == 0 else next(next_id)
            part = Track(tid, t.detections[first:last + 1])
            for d in part.detections:
                d.track_id = tid
            out[tid] = part
    for t in out.values():
        t.decide_kit(classifier)
    return out


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
        has_next.add(a)
        has_prev.add(b)
        parent[b] = root(a)
        tracks[root(a)].kit = tracks[root(a)].kit or tracks[b].kit

    merged: dict[int, Track] = {}
    for tid in sorted(tracks, key=lambda t: tracks[t].start):
        src, r = tracks[tid], root(tid)
        dst = merged.setdefault(r, Track(r))
        dst.detections.extend(src.detections)
        dst.merged_ids.append(tid)
    for t in merged.values():
        t.detections.sort(key=lambda d: d.frame)
        for d in t.detections:
            d.track_id = t.id
        t.decide_kit(classifier)
    return merged


def enforce_role_limits(tracks: dict[int, Track], overlap_tolerance: int) -> None:
    """At most one goalkeeper per kit and two referees at a time; longer tracks win.

    A staff member in a light shirt by the bench must not become a second goalkeeper.
    """
    for role, limit in MAX_ON_COURT.items():
        by_kit: dict[str, list[Track]] = {}
        for t in tracks.values():
            if t.kit is not None and t.kit.role == role:
                by_kit.setdefault(t.kit.name, []).append(t)
        for group_ in by_kit.values():
            accepted: list[Track] = []
            for t in sorted(group_, key=lambda t: len(t.detections), reverse=True):
                overlapping = sum(1 for a in accepted
                                  if min(a.end, t.end) - max(a.start, t.start) > overlap_tolerance)
                if overlapping < limit:
                    accepted.append(t)
                else:
                    t.kit = None


def drop_short(tracks: dict[int, Track], min_frames: int) -> dict[int, Track]:
    return {tid: t for tid, t in tracks.items() if len(t.detections) >= min_frames}
