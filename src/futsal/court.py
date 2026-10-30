"""Playing-area filter (people whose feet are outside the court polygon are ignored) and the
image-to-court homography that turns feet positions into metres."""
from __future__ import annotations

import cv2
import numpy as np


class Pitch:
    """Court coordinates in metres: x from the left goal line (0) to the right one (length), y from the
    far touchline (0) to the near one (width), fitted by least squares to pixel/metre point pairs."""

    def __init__(self, length_m: float, width_m: float, points: list[tuple[float, float, float, float]]):
        if len(points) < 4:
            raise ValueError("pitch: at least four [x_px, y_px, x_m, y_m] points are needed")
        self.length, self.width = float(length_m), float(width_m)
        src = np.float32([p[:2] for p in points])
        dst = np.float32([p[2:] for p in points])
        self.H, _ = cv2.findHomography(src, dst, 0)
        back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.linalg.inv(self.H)).reshape(-1, 2)
        self.residual_px = float(np.max(np.linalg.norm(back - src, axis=1)))

    def to_m(self, points) -> np.ndarray:
        """(n, 2) pixel points -> (n, 2) metres."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def along(self, x: float, y: float) -> float:
        """0 at the left goal line, 1 at the right one."""
        return float(min(1.0, max(0.0, self.to_m([(x, y)])[0, 0] / self.length)))


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
