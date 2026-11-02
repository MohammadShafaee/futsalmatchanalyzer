"""Labelling tasks for a ground-truth window: frames every 2 s, a 'who is this' question per track, frame checks every 6 s."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, "src")
from futsal import config  # noqa: E402

cfg_path, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
t0, t1 = float(sys.argv[4]), float(sys.argv[5])
cfg = config.load(cfg_path)
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
fps = summary["video"]["fps"]
top, bottom = summary["video"]["crop_rows"]
EVERY = 60          # frames between sampled frames (2 s at 30 fps; a multiple of the stride 4)
CHECK_EVERY = 3     # every 3rd sampled frame is also a frame check (6 s)

f0, f1 = int(t0 * fps) // EVERY * EVERY, int(t1 * fps)
sampled = list(range(f0, f1, EVERY))
boxes = defaultdict(list)
for r in csv.DictReader(open(run / "detections.csv", encoding="utf-8-sig")):
    f = int(r["frame"])
    if f in sampled and r["source"] == "detected":
        boxes[f].append(r)

(out / "img").mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(cfg.video))
frames, identity = [], []
best = {}   # track -> (height, frame, row)
for i, f in enumerate(sampled):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    name = f"f{f}.jpg"
    cv2.imwrite(str(out / "img" / name), img[top:bottom], [cv2.IMWRITE_JPEG_QUALITY, 82])
    bx = []
    for j, r in enumerate(sorted(boxes[f], key=lambda r: float(r["x1"]))):
        x1, y1, x2, y2 = (round(float(r[k]), 1) for k in ("x1", "y1", "x2", "y2"))
        bx.append({"id": j, "track": int(r["track_id"]), "kit": r["kit"] or "", "x1": x1, "y1": round(y1 - top, 1),
                   "x2": x2, "y2": round(y2 - top, 1)})
        h = y2 - y1
        if r["track_id"] not in best or h > best[r["track_id"]][0]:
            best[r["track_id"]] = (h, f, j)
    frames.append({"id": f"fr{f}", "frame": f, "t": round(f / fps, 2), "image": name, "check": i % CHECK_EVERY == 0,
                   "boxes": bx})

by_frame = {fr["frame"]: fr for fr in frames}
# skip tracks `futsal jerseys` already read with enough confidence to call (see jersey.vote()): asking a
# human to redo those would give no ground truth we don't already have a model answer for.
jerseys_path = run / "jerseys.json"
already_read = set(json.loads(jerseys_path.read_text(encoding="utf-8"))) if jerseys_path.exists() else set()
skipped = 0
for tid, (h, f, j) in sorted(best.items(), key=lambda kv: (kv[1][1], kv[1][2])):
    if str(tid) in already_read:
        skipped += 1
        continue
    b = by_frame[f]["boxes"][j]
    identity.append({"id": f"id{tid}", "track": int(tid), "frame": f, "t": by_frame[f]["t"], "image": by_frame[f]["image"],
                     "box": {k: b[k] for k in ("x1", "y1", "x2", "y2")}, "kit": b["kit"]})

roster = {team: [{"number": n, "name": cfg.player_name(team, n)} for n in sorted(cfg.jersey.roster.get(team, {}))]
          for team in cfg.teams}
meta = {"window": [t0, t1], "every_s": EVERY / fps, "crop_rows": [top, bottom], "width": summary["video"]["width"],
        "run": run.name, "teams": cfg.teams}
(out / "frames.json").write_text(json.dumps({"frames": frames}, separators=(",", ":")), encoding="utf-8")
(out / "identity.json").write_text(json.dumps({"questions": identity}, separators=(",", ":")), encoding="utf-8")
(out / "roster.json").write_text(json.dumps({"roster": roster, "meta": meta}, separators=(",", ":")), encoding="utf-8")
print(len(frames), "frames,", sum(fr["check"] for fr in frames), "frame checks,", len(identity), "who-is-this questions",
      f"({skipped} skipped, already read by jerseys.json)" if skipped else "")
print("frames.json", (out / "frames.json").stat().st_size, "identity.json", (out / "identity.json").stat().st_size)
