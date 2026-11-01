# Futsal Match Analyzer

Analyzes fixed-camera futsal video: YOLO finds people and the ball, ByteTrack follows players,
shirt colours identify teams / goalkeepers / referees, possession is estimated from the ball
position, jersey numbers (OCR) attach a roster name to each shirt, and — with a court calibration —
positions are mapped onto the pitch in metres for heatmaps, team shape and a 2D replay.

## Setup (Windows, VS Code)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs Ultralytics/OpenCV/NumPy/PyYAML/pytest. `requirements.txt` adds the jersey-number
reader (PARSeq) and is optional — skip it if you only need detection, tracking and possession:

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` does not pin `torch`/`torchvision` themselves — install the build that matches
your GPU first (CPU-only works but is much slower):

```powershell
# example: NVIDIA GTX 10-series (CUDA 12.6 build)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt --no-deps   # keep pip from swapping torch back out
```

Model weights go in `models/` (e.g. `models/yolo11m.pt`; Ultralytics downloads known names
automatically). The jersey reader downloads PARSeq (~90 MB) into the torch hub cache on first use.
Videos go in `input/`. Neither is committed to git.

A second, optional jersey-number reader ("Jev", for shirts PARSeq can't call) needs an API key.
Put it in a git-ignored `.env` file in the project root:

```
JEV_API_KEY=...
```

Set `jersey.jev: true` in a config to enable it (see `configs/manzano_full.yaml`).

## Usage

Every match/camera has a YAML config in `configs/` (video path, team names, kit colours, court
polygon, detector settings). See `configs/manzano_clip.yaml` for a worked example.

```powershell
python -m futsal run configs\manzano_clip.yaml                    # detect + analyze + render
python -m futsal analyze configs\manzano_clip.yaml outputs\<run> --render   # re-analyze, no YOLO
python -m futsal render outputs\<run>                              # redraw the video only
python -m futsal jerseys configs\manzano_clip.yaml outputs\<run>   # OCR jersey numbers onto tracks
python -m futsal export configs\manzano_clip.yaml outputs\<run>    # write coach-report data
python -m pytest                                                   # unit tests
```

`detect` is the slow step (YOLO on every `stride`-th frame). `analyze` reuses `raw_detections.csv`,
so kit colours, the court polygon and possession settings can be tuned in seconds. `jerseys` and
`export` both read an already-analyzed run directory and can be re-run as the config changes.

Each run directory (`outputs/<video>_<timestamp>/`) contains:

| File | Content |
|---|---|
| `raw_detections.csv` | every YOLO box (people with ByteTrack IDs, ball candidates) |
| `detections.csv` | on-court people with stitched track IDs, kit, team, role |
| `frames.csv` | per analyzed frame: chosen ball position (detected/interpolated), player in control, team in possession |
| `summary.json` | video metadata, config used, team possession, per-player stats, quality indicators |
| `annotated.mp4` | boxes coloured by kit, ball circle, possession counter |
| `jersey_reads.csv` | cached OCR reads, written by `jerseys` |

`export` writes its files (heatmaps, team shape, replay, per-number stats, timeline, zones...) into
`<run_dir>/report/` by default — see the header of `src/futsal/report.py` for the full list and what
each file needs (a `pitch:` calibration in the config enables the metre-based ones; running
`jerseys` first enables per-number stats).

## Pipeline

```
video ─► crop black bars ─► YOLO (people + ball) ─► ByteTrack (people) ─► raw_detections.csv
                                                                                 │
   court polygon filter ◄────────────────────────────────────────────────────────┘
   ─► kit votes per torso pixel, summed per track ─► stitch fragmented tracks ─► drop short tracks
   ─► ball trajectory (plausibility filter + gap interpolation) ─► possession (with hysteresis)
   ─► CSV / JSON ─► render
```

Jersey numbers (optional, run separately): torso crops of each track ─► PARSeq OCR ─► confidence
vote per track, checked against the configured roster ─► (optionally) Jev OCR on tracks PARSeq
couldn't call ─► number + roster name attached to the track.

With a `pitch:` homography in the config, player/ball pixel positions are also mapped onto the
court in metres, which `export` uses for heatmaps, team shape and the replay.

Code lives in `src/futsal/`: `config`, `video`, `detect`, `court`, `kits`, `tracks`, `ball`,
`metrics`, `storage`, `render`, `pipeline`, `cli`, `jersey`, `jev`, `report`.

## Known limitations

- The ball is ~10 px wide in a panoramic view; a general COCO model misses it often. A futsal-trained detector would help most.
- Possession is proximity-based, not verified contact.
- Passes, shots, goals and fouls are not classified.
- Jersey-number OCR needs a roughly front/back-on, legible shirt; numbers on players seen only from the side or far away often go unread.
