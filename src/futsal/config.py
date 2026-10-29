"""Run configuration, loaded from a YAML file (see configs/)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROLES = ("player", "goalkeeper", "referee")


@dataclass
class DetectorConfig:
    model: str = "models/yolo11m.pt"
    imgsz: int = 2560
    person_conf: float = 0.25
    ball_conf: float = 0.05
    stride: int = 2                 # analyze every Nth source frame
    device: str | None = None       # e.g. "0" for the first GPU, "cpu"
    track_buffer_s: float = 2.0     # how long ByteTrack keeps a lost player alive
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.3
    match_thresh: float = 0.8


@dataclass
class Kit:
    """A shirt colour that identifies a team, goalkeeper or referee."""
    name: str
    rgb: tuple[int, int, int]
    role: str = "player"
    team: str | None = None         # None for referees


@dataclass
class AnalysisConfig:
    kit_max_distance: float = 28.0  # Lab distance for a torso pixel to vote for a kit
    kit_min_share: float = 0.12     # winning kit must cover this share of torso pixels
    stitch_max_gap_s: float = 1.5   # join a track that ends with one that starts within this gap...
    stitch_max_speed_px_s: float = 300.0  # ...if the jump is physically plausible
    min_track_s: float = 0.4        # drop shorter tracks (usually false positives)
    # cut tracks where ByteTrack swapped two differently dressed people; off by default because on
    # the test clip it cut more correct tracks than it fixed (count accuracy 28% -> 22%)
    split_on_kit_change: bool = False
    ball_max_speed_px_s: float = 2500.0
    ball_max_gap_s: float = 0.8     # interpolate missing ball positions up to this gap
    possession_radius_px: float = 45.0
    possession_min_s: float = 0.3   # control must last this long before possession switches


@dataclass
class Config:
    video: Path
    output_root: Path = Path("outputs")
    teams: list[str] = field(default_factory=lambda: ["Team 1", "Team 2"])
    kits: list[Kit] = field(default_factory=list)
    crop: str | list[int] = "auto"  # "auto", "none" or [y_top, y_bottom] in source pixels
    court: list[tuple[int, int]] | None = None  # polygon in source pixels; feet outside are ignored
    # how many people of each kit should be visible on court; used only to score accuracy
    expected_on_court: dict[str, int] = field(default_factory=dict)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def kit(self, name: str | None) -> Kit | None:
        return next((k for k in self.kits if k.name == name), None)


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}")
    return cls(**data)


def load(path: Path) -> Config:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "video" not in data:
        raise ValueError(f"{path}: 'video' is required")
    kits = [_build(Kit, k) for k in data.pop("kits", [])]
    for k in kits:
        k.rgb = tuple(int(v) for v in k.rgb)
        if k.role not in ROLES:
            raise ValueError(f"Kit {k.name}: role must be one of {ROLES}")
    cfg = _build(Config, {**data, "kits": kits,
                          "detector": _build(DetectorConfig, data.get("detector")),
                          "analysis": _build(AnalysisConfig, data.get("analysis"))})
    cfg.video = Path(cfg.video)
    cfg.output_root = Path(cfg.output_root)
    if cfg.court is not None:
        cfg.court = [tuple(int(v) for v in p) for p in cfg.court]
    if cfg.detector.stride < 1:
        raise ValueError("detector.stride must be at least 1")
    for name in cfg.expected_on_court:
        if cfg.kit(name) is None:
            raise ValueError(f"expected_on_court: unknown kit {name!r}")
    for k in kits:
        if k.team is not None and k.team not in cfg.teams:
            raise ValueError(f"Kit {k.name}: team {k.team!r} is not in teams {cfg.teams}")
    return cfg
