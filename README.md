# Futsal Video Analyzer

Python + YOLO starter project for the fixed wide-angle indoor Manzano–Jesolo recording. It detects people and the ball, tracks detections, groups shirts into two color clusters, and exports time-coded rows.

## Setup in VS Code (Windows)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

First run downloads `yolo11n.pt`. A GPU works when the PyTorch installation supports it; CPU is used otherwise.

## Run

The project points at the existing match file:

```powershell
python src\match_analyzer.py --stride 5 --imgsz 960 --team1-name Manzano --team2-name Jesolo
```

For a more detailed, slower pass, use `--stride 2 --imgsz 1280 --verbose`. Set a custom detector with `--model path\to\weights.pt`; model class names should include `person`/`player` and `sports ball`/`ball`. `--device 0` selects a supported GPU. `--no-render` skips annotated video.

Each run writes `outputs/<video>_<time>/annotated.mp4`, `detections.csv`, and `summary.json`. Detection rows include timestamps, object types, IDs, confidence, bounds, center position, and inferred team. Summary statistics are rough: possession is nearest-player distance, shirt clustering may need manual remapping, and player movement is in pixels. Court calibration is needed before reporting meters. This starter does not yet classify passes, shots, goals, fouls, or tactics. Review the annotated video before using the estimates.

## Method and references

The workflow follows the stages in the tutorial you linked: YOLO detection, ByteTrack tracking, jersey-color grouping, and movement analysis. This first version uses a general COCO model; a sports-trained detector should perform better on the small ball.

- [Your linked tutorial](https://youtu.be/neBZ6huolkg)
- [Tutorial code repository](https://github.com/abdullahtarek/football_analysis)
- [Ultralytics tracking mode](https://docs.ultralytics.com/modes/track/)
