"""Shirt-colour classification against the kits listed in the config.

Every torso pixel votes for the nearest kit colour in Lab space (if it is close enough);
votes are summed per track, so one bad frame does not flip a player's team.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Kit

L_WEIGHT = 0.5  # lighting changes brightness much more than hue, so L counts less


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR (..., 3) -> float Lab with L in 0..100."""
    flat = bgr.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).reshape(bgr.shape)


def torso(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 16:
        return None
    crop = frame[int(y1 + .15 * h):int(y1 + .5 * h), int(x1 + .2 * w):int(x2 - .2 * w)]
    return crop if crop.size else None


class KitClassifier:
    def __init__(self, kits: list[Kit], max_distance: float, min_share: float):
        self.kits = kits
        self.max_distance = max_distance
        self.min_share = min_share
        rgb = np.array([k.rgb for k in kits], dtype=np.uint8).reshape(-1, 3)
        self.refs = to_lab(rgb[:, ::-1].copy())
        self.refs[:, 0] *= L_WEIGHT

    def shares(self, frame: np.ndarray, box) -> np.ndarray | None:
        """Share of torso pixels voting for each kit, or None when the box is too small."""
        crop = torso(frame, box)
        if crop is None or not self.kits:
            return None
        lab = to_lab(crop).reshape(-1, 3)
        lab[:, 0] *= L_WEIGHT
        dist = np.linalg.norm(lab[:, None, :] - self.refs[None, :, :], axis=2)
        nearest = dist.argmin(axis=1)
        close = dist[np.arange(len(lab)), nearest] <= self.max_distance
        return np.bincount(nearest[close], minlength=len(self.kits)) / len(lab)

    def decide(self, total_shares: np.ndarray, samples: int) -> Kit | None:
        if samples == 0:
            return None
        mean = total_shares / samples
        best = int(mean.argmax())
        return self.kits[best] if mean[best] >= self.min_share else None
