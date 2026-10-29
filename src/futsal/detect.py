"""YOLO detection of people and the ball, with ByteTrack identities for people."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .config import DetectorConfig

LOG = logging.getLogger(__name__)
PERSON_NAMES = {"person", "player", "goalkeeper", "goalie", "keeper", "referee",
                "football player", "soccer player", "futsal player"}
BALL_NAMES = {"sports ball", "ball", "football", "soccer ball", "futsal ball"}


@dataclass
class Detection:
    """One box in source-frame pixels. `track_id` is None for balls."""
    frame: int
    kind: str           # "person" or "ball"
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None
    kit_shares: np.ndarray | None = field(default=None, repr=False)  # filled during analysis
    interpolated: bool = False  # filled in between two real detections of the same track

    @property
    def foot(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


def class_kinds(names) -> dict[int, str]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    kinds = {}
    for cid, name in items:
        label = " ".join(str(name).lower().replace("_", " ").split())
        if label in BALL_NAMES:
            kinds[int(cid)] = "ball"
        elif label in PERSON_NAMES:
            kinds[int(cid)] = "person"
    if "person" not in kinds.values():
        raise ValueError("Model has no person/player class.")
    if "ball" not in kinds.values():
        LOG.warning("Model has no ball class; possession will be unavailable.")
    return kinds


class Detector:
    """Runs YOLO once per frame; people go through ByteTrack, balls are kept raw.

    Ultralytics' built-in `model.track` drops every box the tracker does not keep,
    which removes the low-confidence ball detections, so the tracker is driven here.
    """

    def __init__(self, cfg: DetectorConfig, analyzed_fps: float):
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.kinds = class_kinds(self.model.names)
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=cfg.track_high_thresh, track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh, match_thresh=cfg.match_thresh,
            track_buffer=max(1, round(cfg.track_buffer_s * analyzed_fps)), fuse_score=True))

    def __call__(self, index: int, frame: np.ndarray, y_offset: int = 0) -> list[Detection]:
        opts = {"imgsz": self.cfg.imgsz, "classes": list(self.kinds), "verbose": False,
                "conf": min(self.cfg.ball_conf, self.cfg.track_low_thresh)}
        if self.cfg.device is not None:
            opts["device"] = self.cfg.device
        boxes = self.model.predict(frame, **opts)[0].boxes.cpu().numpy()
        kinds = np.array([self.kinds[int(c)] for c in boxes.cls], dtype=object)

        people = boxes[kinds == "person"]
        out = []
        for x1, y1, x2, y2, tid, score, *_ in self.tracker.update(people, frame):
            out.append(Detection(index, "person", round(float(score), 4),
                                 x1, y1 + y_offset, x2, y2 + y_offset, int(tid)))
        balls = boxes[(kinds == "ball") & (boxes.conf >= self.cfg.ball_conf)]
        for (x1, y1, x2, y2), conf in zip(balls.xyxy, balls.conf):
            out.append(Detection(index, "ball", round(float(conf), 4), x1, y1 + y_offset, x2, y2 + y_offset))
        return out
