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


def _weighted_lab(bgr: np.ndarray) -> np.ndarray:
    lab = to_lab(bgr).reshape(-1, 3)
    lab[:, 0] *= L_WEIGHT
    return lab


def estimate_background(frames: list[np.ndarray], polygon: np.ndarray | None, kits: list[Kit],
                        max_distance: float, k: int = 6, samples: int = 20000) -> np.ndarray:
    """Typical floor colours (weighted Lab) from pixels inside the court.

    Players cover a small part of the court, so k-means centres describe the floor.
    Centres nearly identical to a kit colour (white lines vs. a white goalkeeper shirt, ~11 apart)
    are discarded; dark wood vs. a dark green shirt (~27 apart) must stay masked.
    """
    rng = np.random.default_rng(0)
    pixels = []
    for frame in frames:
        mask = np.ones(frame.shape[:2], np.uint8)
        if polygon is not None:
            mask[:] = 0
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        pts = frame[mask.astype(bool)]
        pixels.append(pts[rng.choice(len(pts), min(len(pts), samples // len(frames)), replace=False)])
    lab = _weighted_lab(np.concatenate(pixels).reshape(-1, 1, 3)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .5)
    _, _, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    kit_refs = KitClassifier(kits, max_distance, 0).refs
    if len(kit_refs):
        near_kit = np.linalg.norm(centers[:, None] - kit_refs[None], axis=2).min(axis=1) < max_distance / 2
        centers = centers[~near_kit]
    return centers


class KitClassifier:
    MIN_PIXELS = 12

    def __init__(self, kits: list[Kit], max_distance: float, min_share: float,
                 background: np.ndarray | None = None):
        self.kits = kits
        self.max_distance = max_distance
        self.min_share = min_share
        rgb = np.array([k.rgb for k in kits], dtype=np.uint8).reshape(-1, 1, 3)
        self.refs = _weighted_lab(rgb[:, :, ::-1].copy())
        self.background = np.zeros((0, 3), np.float32) if background is None else background

    def shares(self, frame: np.ndarray, box) -> np.ndarray | None:
        """Share of non-floor torso pixels voting for each kit; None when too few pixels."""
        crop = torso(frame, box)
        if crop is None or not self.kits:
            return None
        lab = _weighted_lab(crop)
        refs = np.concatenate([self.refs, self.background])
        dist = np.linalg.norm(lab[:, None, :] - refs[None, :, :], axis=2)
        nearest = dist.argmin(axis=1)
        shirt = nearest < len(self.kits)          # pixels closest to a floor colour are ignored
        if shirt.sum() < self.MIN_PIXELS:
            return None
        close = shirt & (dist[np.arange(len(lab)), nearest] <= self.max_distance)
        return np.bincount(nearest[close], minlength=len(self.kits))[:len(self.kits)] / shirt.sum()

    def decide(self, total_shares: np.ndarray, samples: int) -> Kit | None:
        if samples == 0:
            return None
        mean = total_shares / samples
        best = int(mean.argmax())
        return self.kits[best] if mean[best] >= self.min_share else None
