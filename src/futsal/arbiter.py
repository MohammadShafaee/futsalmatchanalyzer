"""AI arbiter: a vision-language model (Gemini) settles the tracks the colour rules can't.

Only uncertain ByteTrack tracks are sent: the shirt-colour vote is close between two kits or undecided,
or the person stays at the edge of the court polygon (a substitute or staff member by the bench?).
Each question is one image (four close-ups of the person plus the scene around them) and the answer
is cached in the run directory, so re-running `analyze` never pays twice for the same track.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import video
from .config import ArbiterConfig, Kit
from .kits import KitClassifier
from .tracks import Track

LOG = logging.getLogger(__name__)
CACHE = "arbiter.json"
NOT_PLAYING = "not_playing"   # substitute, coach, spectator, ball boy...
MIXED = "mixed"               # the close-ups show different people: the tracker swapped IDs
UNCLEAR = "unclear"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _rgb_name(rgb) -> str:
    return "rgb({}, {}, {})".format(*rgb)


def _prompt(kits: list[Kit]) -> str:
    lines = [f'- "{k.name}": {k.role}' + (f" of {k.team}" if k.team else "") + f", shirt colour about {_rgb_name(k.rgb)}"
             for k in kits]
    return (
        "You are checking the output of a futsal player tracker. The image shows ONE tracked person: the top row "
        "has four close-ups taken at different moments, the bottom shows the scene around the person (red box).\n"
        "Decide who this is. Options:\n" + "\n".join(lines) + "\n"
        f'- "{NOT_PLAYING}": a substitute in a training bib, coach, staff, ball boy or spectator, not taking part in play\n'
        f'- "{MIXED}": the close-ups clearly show different people (different kits)\n'
        f'- "{UNCLEAR}": you cannot tell\n'
        "Goalkeepers often wear a different colour from their team. Referees usually wear one colour for the whole "
        "match. Judge from the shirt and from where the person stands. Also label each close-up on its own "
        "(left to right). Answer with JSON only."
    )


def _schema(kits: list[Kit]) -> dict:
    labels = [k.name for k in kits] + [NOT_PLAYING, MIXED, UNCLEAR]
    return {"type": "OBJECT", "properties": {
        "label": {"type": "STRING", "enum": labels},
        "confidence": {"type": "NUMBER", "description": "0 to 1"},
        "reason": {"type": "STRING", "description": "one short sentence"},
        "closeups": {"type": "ARRAY", "items": {"type": "STRING", "enum": labels},
                     "description": "who each of the four close-ups shows, left to right"},
    }, "required": ["label", "confidence", "reason", "closeups"]}


class QuotaExhausted(RuntimeError):
    """The API key's quota is used up for hours (e.g. the free tier's daily limit)."""


def _retry_delay_s(error_body: bytes) -> float:
    try:
        details = json.loads(error_body).get("error", {}).get("details", [])
    except ValueError:
        return 0.0
    delay = next((d["retryDelay"] for d in details if "retryDelay" in d), "0s")
    return float(str(delay).rstrip("s") or 0)


def _ask(cfg: ArbiterConfig, key: str, prompt: str, schema: dict, jpeg: bytes) -> dict:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema},
    }).encode()
    for attempt in range(6):
        req = urllib.request.Request(API.format(model=cfg.model), data=body, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        wait = 5 * 2 ** attempt
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                reply = json.load(r)
            return json.loads(reply["candidates"][0]["content"]["parts"][0]["text"])
        except urllib.error.HTTPError as e:
            if e.code == 429:
                delay = _retry_delay_s(e.read())
                if delay > 300:  # a daily quota, not a per-minute one: waiting won't help
                    raise QuotaExhausted(f"{cfg.model}: quota used up, retry in {delay / 3600:.1f} h") from e
                wait = max(wait, delay)
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(min(wait, 120))
    raise RuntimeError("unreachable")


def _uncertain(track: Track, classifier: KitClassifier, polygon: np.ndarray | None, cfg: ArbiterConfig) -> str | None:
    """Why this track needs a second opinion, or None when the colour vote is clear."""
    votes = [d.kit_shares for d in track.detections if d.kit_shares is not None]
    if not votes:
        return "no shirt colour"
    mean = np.mean(votes, axis=0)
    top = np.sort(mean)[::-1]
    if classifier.decide(mean, 1) is None:
        return "no kit matched"
    if len(top) > 1 and top[0] > 0 and (top[0] - top[1]) / top[0] < cfg.vote_margin:
        return "two kits close"
    if polygon is not None:
        feet = np.median([d.foot for d in track.detections], axis=0)
        if cv2.pointPolygonTest(polygon, (float(feet[0]), float(feet[1])), True) < cfg.edge_px:
            return "at the court edge"
    return None


def _sheet(frames: dict[int, np.ndarray], track: Track, picks: list) -> bytes:
    crops = []
    for d in picks:
        f = frames[d.frame]
        w, h = d.x2 - d.x1, d.y2 - d.y1
        x1, y1 = int(max(0, d.x1 - .25 * w)), int(max(0, d.y1 - .1 * h))
        x2, y2 = int(min(f.shape[1], d.x2 + .25 * w)), int(min(f.shape[0], d.y2 + .1 * h))
        c = f[y1:y2, x1:x2]
        crops.append(cv2.resize(c, (int(200 * c.shape[1] / max(1, c.shape[0])), 200)))
    top = np.hstack([np.pad(c, ((0, 0), (4, 4), (0, 0))) for c in crops])
    mid = picks[len(picks) // 2]
    f = frames[mid.frame].copy()
    cx, cy = int((mid.x1 + mid.x2) / 2), int((mid.y1 + mid.y2) / 2)
    cv2.rectangle(f, (int(mid.x1), int(mid.y1)), (int(mid.x2), int(mid.y2)), (0, 0, 255), 3)
    x1, y1 = max(0, cx - 450), max(0, cy - 160)
    scene = f[y1:y1 + 320, x1:x1 + 900]
    width = max(top.shape[1], scene.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, width - im.shape[1]), (0, 0)))
    ok, jpg = cv2.imencode(".jpg", np.vstack([pad(top), np.zeros((8, width, 3), np.uint8), pad(scene)]),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


def _one_hot(index: int, n: int) -> np.ndarray:
    vote = np.zeros(n)
    vote[index] = 1.0
    return vote


def _anchor_swap(track: Track, closeups: list[str], by_name: dict[str, int]) -> int:
    """A swapped track: fix the kit on each side of the swap and leave the colour vote to find the cut.

    Detections up to the last close-up of the first person and from the first close-up of the second are
    pinned; the stretch in between keeps its colour votes, so split_id_switches places the cut there.
    """
    picks = [round(i) for i in np.linspace(0, len(track.detections) - 1, 4)]
    known = [(p, by_name[c]) for p, c in zip(picks, closeups) if c in by_name]
    if len({k for _, k in known}) != 2:
        return 0
    first = known[0][1]
    switch = next(i for i, (_, k) in enumerate(known) if k != first)
    second = known[switch][1]
    if any(k != second for _, k in known[switch:]):
        return 0  # A, B, A: more than one swap, leave it to the colour vote
    end_a, start_b = known[switch - 1][0], known[switch][0]
    for i, d in enumerate(track.detections):
        if i <= end_a:
            d.kit_shares = _one_hot(first, len(by_name))
        elif i >= start_b:
            d.kit_shares = _one_hot(second, len(by_name))
    return 1


def resolve(tracks: dict[int, Track], classifier: KitClassifier, kits: list[Kit], cfg: ArbiterConfig,
            video_path: Path, run_dir: Path, polygon: np.ndarray | None, fps: float, stride: int) -> dict[int, Track]:
    """Ask the model about uncertain tracks and apply confident answers; returns the kept tracks."""
    cache_path = run_dir / CACHE
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    min_len = max(4, round(cfg.min_track_s * fps / stride))
    candidates = []
    for t in sorted(tracks.values(), key=lambda t: len(t.detections), reverse=True):
        if len(t.detections) < min_len:
            continue
        why = _uncertain(t, classifier, polygon, cfg)
        if why:
            candidates.append((t, why))
    todo = [(t, why) for t, why in candidates if str(t.id) not in cache][:max(0, cfg.max_calls - len(cache))]
    LOG.info("Arbiter: %d uncertain tracks, %d cached, %d to ask %s", len(candidates),
             sum(str(t.id) in cache for t, _ in candidates), len(todo), cfg.model)

    key = os.environ.get("GEMINI_API_KEY")
    if todo and not key:
        LOG.warning("Arbiter: GEMINI_API_KEY is not set; skipping %d questions", len(todo))
        todo = []
    if todo:
        picks = {t.id: [t.detections[round(i)] for i in np.linspace(0, len(t.detections) - 1, 4)] for t, _ in todo}
        wanted = {d.frame for ds in picks.values() for d in ds}
        frames = dict(video.frames(video_path, wanted=wanted))
        prompt, schema = _prompt(kits), _schema(kits)
        last_call = 0.0
        try:
            for n, (t, why) in enumerate(todo, 1):
                time.sleep(max(0.0, cfg.min_interval_s - (time.monotonic() - last_call)))  # stay under the quota
                last_call = time.monotonic()
                try:
                    answer = _ask(cfg, key, prompt, schema, _sheet(frames, t, picks[t.id]))
                except QuotaExhausted as e:
                    LOG.warning("Arbiter: %s; %d questions left for the next run (answers so far are cached)",
                                e, len(todo) - n + 1)
                    break
                except Exception as e:  # a failed call must not stop the analysis
                    LOG.warning("Arbiter: track %d failed: %s", t.id, e)
                    continue
                cache[str(t.id)] = {**answer, "why_asked": why, "model": cfg.model, "detections": len(t.detections)}
                if n % 10 == 0:
                    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                    LOG.info("Arbiter: %d/%d asked", n, len(todo))
        finally:
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    by_name = {k.name: i for i, k in enumerate(kits)}
    kept, changed, dropped = {}, 0, 0
    for tid, t in tracks.items():
        answer = cache.get(str(tid))
        if answer and answer.get("confidence", 0) >= cfg.min_confidence:
            if answer["label"] == NOT_PLAYING:
                dropped += 1
                continue
            if answer["label"] in by_name:
                for d in t.detections:
                    d.kit_shares = _one_hot(by_name[answer["label"]], len(kits))
                changed += 1
            elif answer["label"] == MIXED:
                changed += _anchor_swap(t, answer.get("closeups") or [], by_name)
        kept[tid] = t
    LOG.info("Arbiter: set the kit of %d tracks, removed %d people not playing", changed, dropped)
    return kept
