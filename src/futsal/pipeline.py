"""The two stages of a run: `detect` (slow, YOLO) and `analyze` (fast, re-runnable)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from . import ball, metrics, storage, tracks, video
from .config import Config
from .court import Court
from .detect import Detector
from .kits import KitClassifier

LOG = logging.getLogger(__name__)
DETECT_META = "detect_meta.json"


def new_run_dir(cfg: Config) -> Path:
    run = cfg.output_root / f"{cfg.video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def detect(cfg: Config, run_dir: Path) -> None:
    info = video.probe(cfg.video)
    if cfg.crop == "auto":
        top, bottom = video.letterbox_rows(cfg.video)
    elif cfg.crop == "none":
        top, bottom = 0, info.height
    else:
        top, bottom = cfg.crop
    stride = cfg.detector.stride
    LOG.info("Video %s: %dx%d, %.1f fps, %.1f s; analyzing rows %d-%d every %d frames",
             info.path.name, info.width, info.height, info.fps, info.duration_s, top, bottom, stride)

    detector = Detector(cfg.detector, info.fps / stride)
    detections, last, started = [], 0, time.monotonic()
    total = info.frame_count // stride + 1
    for n, (index, frame) in enumerate(video.frames(cfg.video, stride), 1):
        detections.extend(detector(index, frame[top:bottom], y_offset=top))
        last = index
        if n % 50 == 0:
            rate = n / (time.monotonic() - started)
            LOG.info("Detected %d/%d frames (%.2f frames/s, ~%.0f s left)", n, total, rate, (total - n) / rate)

    storage.write_raw(run_dir / "raw_detections.csv", detections)
    meta = {"path": str(info.path), "fps": info.fps, "frame_count": info.frame_count, "width": info.width,
            "height": info.height, "duration_s": round(info.duration_s, 2), "crop_rows": [top, bottom],
            "last_analyzed_frame": last, "stride": stride, "model": cfg.detector.model,
            "detect_seconds": round(time.monotonic() - started, 1)}
    (run_dir / DETECT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def analyze(cfg: Config, run_dir: Path) -> dict:
    meta = json.loads((run_dir / DETECT_META).read_text(encoding="utf-8"))
    fps, stride = meta["fps"], meta["stride"]
    if stride != cfg.detector.stride:
        LOG.warning("Run was detected with stride %d; using it instead of %d", stride, cfg.detector.stride)
        cfg.detector.stride = stride
    frames = list(range(0, meta["last_analyzed_frame"] + 1, stride))
    court = Court(cfg.court)
    raw = storage.read_raw(run_dir / "raw_detections.csv")

    people = [d for d in raw if d.kind == "person" and court.contains(d.foot)]
    by_track = tracks.group(people)
    raw_count = len(by_track)

    classifier = KitClassifier(cfg.kits, cfg.analysis.kit_max_distance, cfg.analysis.kit_min_share)
    by_frame: dict[int, list] = {}
    for t in by_track.values():
        for d in t.detections:
            by_frame.setdefault(d.frame, []).append((d, t))
    for index, frame in video.frames(Path(meta["path"]), wanted=set(by_frame)):
        for d, t in by_frame[index]:
            t.add_votes(classifier.shares(frame, (d.x1, d.y1, d.x2, d.y2)))
    for t in by_track.values():
        t.kit = classifier.decide(t.shares, t.samples) if t.shares is not None else None

    stitched = tracks.stitch(by_track, classifier, cfg.analysis, fps)
    min_frames = max(1, round(cfg.analysis.min_track_s * fps / stride))
    kept = tracks.drop_short(stitched, min_frames)

    ball_path = ball.trajectory(raw, frames, court, cfg.analysis, fps)
    states = metrics.possession(frames, ball_path, kept, cfg.analysis, fps)
    stats = metrics.summarize(cfg, fps, stride, frames, kept, raw_count, states)

    storage.write_detections(run_dir / "detections.csv", kept, fps)
    storage.write_frames(run_dir / "frames.csv", states, fps)
    summary = {"schema_version": "2.0", "analysis_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
               "video": meta, "config": asdict(cfg), **stats}
    storage.write_summary(run_dir / "summary.json", summary)
    q = stats["quality"]
    LOG.info("Tracks %d -> %d after stitching; ball detected %.1f%% (+%.1f%% interpolated); possession known %.1f%%",
             q["raw_track_ids"], q["tracks_after_stitching"], q["ball_detected_percent"],
             q["ball_interpolated_percent"], q["possession_known_percent"])
    for team in stats["teams"]:
        LOG.info("%s: possession %s%% (%s s), %d tracks", team["team"], team["possession_percent"],
                 team["possession_seconds"], team["tracks"])
    return summary
