from __future__ import annotations

import json
from typing import Any, Protocol
from urllib import error, request


class SAM3Provider(Protocol):
    mode: str

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float, video_source: str | None = None): ...

    def set_object_detection_prompts(self, prompts: list[str]) -> list[str]: ...

    def get_object_detection_prompts(self) -> list[str]: ...

    def detector_status(self) -> dict: ...

    def get_first_frame_path(self) -> str | None: ...


class SAM3DProvider(Protocol):
    mode: str

    def reconstruct_object_meshes(self, detections, objects) -> dict[str, dict]: ...


class VLMProvider(Protocol):
    mode: str

    def infer_object_physics_priors(self, detections, objects) -> dict[str, dict]: ...

    def discover_movable_object_categories(self, image_b64: str) -> list[str]: ...


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON response from {url} must be object")
    return data


def get_json(base_url: str, path: str, timeout_sec: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON response from {url} must be object")
    return data
