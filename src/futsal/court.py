"""Playing-area filter: people whose feet are outside the court polygon are ignored."""
from __future__ import annotations

import cv2
import numpy as np


class Court:
    def __init__(self, polygon: list[tuple[int, int]] | None):
        self.polygon = None if polygon is None else np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)

    def contains(self, point: tuple[float, float]) -> bool:
        if self.polygon is None:
            return True
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0

    def draw(self, frame: np.ndarray, color=(255, 255, 255)) -> None:
        if self.polygon is not None:
            cv2.polylines(frame, [self.polygon.astype(np.int32)], True, color, 1, cv2.LINE_AA)
