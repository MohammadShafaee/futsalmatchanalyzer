"""Minimal client for TypeSafe's Jev (System One) decision API, with image input.

The official `jevclient` package needs Python 3.12 and aiohttp; this project runs on 3.11, so this
speaks the same JSON over urllib. The request shape follows jevclient 1.3.1:
POST {base}/v1/systemone  {"state", "model", "questions": {name: {"type", "instructions", "criteria"}},
"assets": {ref: {"type": "image", "format", "data_base64"}}}, with the header
TypeSafe-Danger-Zone-Images: true when an image is attached (images are experimental at TypeSafe).

The key is read from the JEV_API_KEY environment variable or from a .env file in the working directory.
"""
from __future__ import annotations

import base64
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

URL = "https://api.typesafe.ai/v1/systemone"
MIN_SIDE, MAX_SIDE = 280, 1568   # the API rejects images outside these sides instead of resizing them


class JevError(RuntimeError):
    pass


def api_key(env_file: Path = Path(".env")) -> str | None:
    if os.environ.get("JEV_API_KEY"):
        return os.environ["JEV_API_KEY"].strip()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "JEV_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def _ssl_context() -> ssl.SSLContext:
    # On some Windows machines Python's own certificate store can't verify api.typesafe.ai;
    # pip's bundled certifi list can, so fall back to it when it is there.
    try:
        from pip._vendor import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fit_image(img: np.ndarray) -> np.ndarray:
    """Pad and scale a BGR image so both sides are within the API's limits."""
    h, w = img.shape[:2]
    scale = min(MAX_SIDE / max(h, w), max(1.0, MIN_SIDE / min(h, w)))
    if scale != 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    pad_h, pad_w = max(0, MIN_SIDE - h), max(0, MIN_SIDE - w)
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img


class Client:
    def __init__(self, key: str, model: str = "jev-latest", timeout_s: float = 60.0):
        self.key, self.model, self.timeout_s = key, model, timeout_s
        self.ctx = _ssl_context()
        self.input_tokens = 0

    def choice(self, instructions: str, options: dict[str, str], image: np.ndarray | None = None,
               state: str = "") -> tuple[str, float, dict[str, float]]:
        """Ask one multiple-choice question, optionally about an image; returns (choice, confidence, probs)."""
        assets = {}
        if image is not None:
            ref = str(uuid.uuid4())
            ok, jpg = cv2.imencode(".jpg", fit_image(image), [cv2.IMWRITE_JPEG_QUALITY, 90])
            assets[ref] = {"type": "image", "format": "jpg", "data_base64": base64.b64encode(jpg.tobytes()).decode()}
            state = f"{state}\nImage: {ref}".strip()
        payload = {"state": state, "model": self.model,
                   "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": options}}}
        if assets:
            payload["assets"] = assets
        answer = self._post(payload, images=bool(assets))["answers"]["q"]
        return str(answer["choice"]), float(answer["confidence"]), {k: float(v) for k, v in answer["probabilities"].items()}

    def _post(self, payload: dict, images: bool) -> dict:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        if images:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        body = json.dumps(payload).encode()
        for attempt in range(5):
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=self.ctx) as r:
                    data = json.load(r)
                self.input_tokens += int((data.get("usage") or {}).get("input_tokens", 0))
                return data
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                    time.sleep(float(e.headers.get("Retry-After") or 2 ** attempt))
                    continue
                raise JevError(f"HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise JevError(f"no answer from Jev: {e}") from e
        raise JevError("unreachable")
