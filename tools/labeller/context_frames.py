"""For each 'who is this' question, grab the track's analyzed frames just before/after the chosen
one (same spacing as detection: stride source frames apart) so a labeller can watch the player move
instead of judging a single, possibly blurred or turned-away, crop.

Writes one JSON file per question into <out>/context/<question id>.json: {"before": [...], "after": [...]},
oldest to newest, each entry {"t": seconds, "image": "data:image/jpeg;base64,..."} with the player's box
already drawn in (so the viewer doesn't need the full frame or box data to show it).

Usage: python tools/labeller/context_frames.py <config> <run_dir> <identity.json> <out_dir> [n_each]
"""
import base64
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from futsal import config  # noqa: E402

N_EACH = int(sys.argv[5]) if len(sys.argv) > 5 else 5
OUT_W, OUT_H = 220, 165   # 4:3, matching the main zoom view
JPEG_Q = 78

cfg_path, run, identity_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
cfg = config.load(cfg_path)
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
fps = summary["video"]["fps"]
top, bottom = summary["video"]["crop_rows"]
questions = json.loads(identity_path.read_text(encoding="utf-8"))["questions"]

by_track = defaultdict(list)
for r in csv.DictReader(open(run / "detections.csv", encoding="utf-8-sig")):
    by_track[int(r["track_id"])].append((int(r["frame"]), {k: float(r[k]) for k in ("x1", "y1", "x2", "y2")}))
for tid in by_track:
    by_track[tid].sort(key=lambda p: p[0])

# for each question: up to N_EACH entries before and after its chosen frame, on the same track
wanted = {}         # question id -> {"before": [(frame, box)...], "after": [(frame, box)...]}
needed_frames = set()
for q in questions:
    track = by_track.get(q["track"], [])
    idx = next((i for i, (f, _) in enumerate(track) if f == q["frame"]), None)
    if idx is None:
        wanted[q["id"]] = {"before": [], "after": []}
        continue
    before = track[max(0, idx - N_EACH):idx]
    after = track[idx + 1:idx + 1 + N_EACH]
    wanted[q["id"]] = {"before": before, "after": after}
    needed_frames.update(f for f, _ in before + after)

print(f"{len(questions)} questions, {len(needed_frames)} distinct frames to read")

crops = {}   # (frame, box tuple) -> data URL
cap = cv2.VideoCapture(str(cfg.video))
if needed_frames:
    lo, hi = min(needed_frames), max(needed_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    needed_by_frame = defaultdict(list)
    for q in questions:
        for side in ("before", "after"):
            for f, box in wanted[q["id"]][side]:
                needed_by_frame[f].append(box)
    f = lo
    while f <= hi:
        ok, img = cap.read()
        if not ok:
            break
        if f in needed_by_frame:
            row = img[top:bottom]
            h, w = row.shape[:2]
            for box in needed_by_frame[f]:
                key = (f, box["x1"], box["y1"], box["x2"], box["y2"])
                if key in crops:
                    continue
                bx1, by1, bx2, by2 = box["x1"], box["y1"] - top, box["x2"], box["y2"] - top
                bw, bh = bx2 - bx1, by2 - by1
                rh = max(bh * 2.2, 120)
                rw = rh * OUT_W / OUT_H
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                sx = min(max(0, cx - rw / 2), max(0, w - rw))
                sy = min(max(0, cy - rh / 2), max(0, h - rh))
                rw, rh = min(rw, w), min(rh, h)
                region = row[int(sy):int(sy + rh), int(sx):int(sx + rw)]
                if region.size == 0:
                    continue
                out_img = cv2.resize(region, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                k = OUT_W / rw
                x1, y1 = (bx1 - sx) * k, (by1 - sy) * k
                x2, y2 = (bx2 - sx) * k, (by2 - sy) * k
                cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 212, 255), 2)
                ok2, buf = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                crops[key] = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
        f += 1

(out / "context").mkdir(parents=True, exist_ok=True)
total = 0
for q in questions:
    doc = {}
    for side in ("before", "after"):
        entries = []
        for f, box in wanted[q["id"]][side]:
            key = (f, box["x1"], box["y1"], box["x2"], box["y2"])
            if key in crops:
                entries.append({"t": round(f / fps, 2), "image": crops[key]})
        doc[side] = entries
    path = out / "context" / f"{q['id']}.json"
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    total += path.stat().st_size
print(f"wrote {len(questions)} context files, {total} bytes total, {total / len(questions):.0f} avg")
