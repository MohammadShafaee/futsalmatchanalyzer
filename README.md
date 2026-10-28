# Futsal Match Analyzer

Analyzes fixed-camera futsal video: YOLO finds people and the ball, ByteTrack follows players,
shirt colours identify teams / goalkeepers / referees, and possession is estimated from the ball position.

## Setup (Windows, VS Code)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Model weights go in `models/` (e.g. `models/yolo11m.pt`; Ultralytics downloads known names automatically).
Videos go in `input/`. Neither is committed to git.

## Usage

Every match/camera has a YAML config in `configs/` (video path, team names, kit colours,
court polygon, detector settings). See `configs/manzano_clip.yaml`.

```powershell
python -m futsal run configs\manzano_clip.yaml              # detect + analyze + render
python -m futsal analyze configs\manzano_clip.yaml outputs\<run> --render   # re-analyze, no YOLO
python -m futsal render outputs\<run>                         # redraw the video only
python -m pytest                                               # unit tests
```

`detect` is the slow step (YOLO on every `stride`-th frame). `analyze` reuses `raw_detections.csv`,
so kit colours, the court polygon and possession settings can be tuned in seconds.

Each run directory contains:

| File | Content |
|---|---|
| `raw_detections.csv` | every YOLO box (people with ByteTrack IDs, ball candidates) |
| `detections.csv` | on-court people with stitched track IDs, kit, team, role |
| `frames.csv` | per analyzed frame: chosen ball position (detected/interpolated), player in control, team in possession |
| `summary.json` | video metadata, config used, team possession, per-player stats, quality indicators |
| `annotated.mp4` | boxes coloured by kit, ball circle, possession counter |

## Pipeline

```
video ─► crop black bars ─► YOLO (people + ball) ─► ByteTrack (people) ─► raw_detections.csv
                                                                                 │
   court polygon filter ◄────────────────────────────────────────────────────────┘
   ─► kit votes per torso pixel, summed per track ─► stitch fragmented tracks ─► drop short tracks
   ─► ball trajectory (plausibility filter + gap interpolation) ─► possession (with hysteresis)
   ─► CSV / JSON ─► render
```

Code lives in `src/futsal/`: `config`, `video`, `detect`, `court`, `kits`, `tracks`, `ball`,
`metrics`, `storage`, `render`, `pipeline`, `cli`.

## Known limitations

- The ball is ~10 px wide in a panoramic view; a general COCO model misses it often. A futsal-trained detector would help most.
- Possession is proximity-based, not verified contact. Distances are in pixels (no court calibration yet).
- Passes, shots, goals and fouls are not classified.
