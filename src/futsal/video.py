"""Video reading helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


def probe(path: Path) -> VideoInfo:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid frame rate in {path}")
        return VideoInfo(path, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


def frames(path: Path, stride: int = 1, wanted: set[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, frame) for every `stride`-th frame, or only for indices in `wanted`."""
    cap = cv2.VideoCapture(str(path))
    last = max(wanted) if wanted else None
    index = 0
    try:
        while last is None or index <= last:
            keep = index in wanted if wanted is not None else index % stride == 0
            if keep:
                ok, frame = cap.read()
                if not ok:
                    break
                yield index, frame
            elif not cap.grab():
                break
            index += 1
    finally:
        cap.release()


def letterbox_rows(path: Path, samples: int = 15, threshold: int = 24) -> tuple[int, int]:
    """Return (top, bottom) rows of the picture, excluding black bars above and below."""
    info = probe(path)
    step = max(info.frame_count // samples, 1)
    brightest = np.zeros(info.height, dtype=np.uint8)
    for _, frame in frames(path, wanted=set(range(0, info.frame_count, step))):
        brightest = np.maximum(brightest, frame.max(axis=(1, 2)))
    rows = np.flatnonzero(brightest > threshold)
    if rows.size == 0:
        return 0, info.height
    return int(rows[0]), int(rows[-1]) + 1
