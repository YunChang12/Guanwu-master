from __future__ import annotations

import json
import os
import shutil
from typing import Any
from urllib import error, request
from pathlib import Path

from fastapi.testclient import TestClient
from guanwu.video.model_backend.api.server import create_app

from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import FrameDetections
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# Maximum characters for JSON payloads in log messages
_LOG_TRUNCATE_LEN = 800


def _truncate_json(obj: Any, max_len: int = _LOG_TRUNCATE_LEN) -> str:
    """Serialize *obj* to a JSON string, truncated to *max_len* characters."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "…(truncated)"
    return s


class ModelBackendAPI:
    def __init__(
        self,
        base_url: str = "inproc://model-backend",
        timeout_sec: float = 15.0,
        retry: int = 1,
        config_path: str | None = None,
    ) -> None:
        self.base_url = (base_url or "inproc://model-backend").strip()
        self.timeout_sec = float(timeout_sec)
        self.retry = max(0, int(retry))
        self._test_client: TestClient | None = None

        if self.base_url.startswith("inproc://"):
            cfg = config_path or os.getenv("SPWM_MODEL_BACKEND_CONFIG")
            self._test_client = TestClient(create_app(config_path=cfg))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        logger.debug(f"[ModelBackendAPI] POST {url} - Payload: {_truncate_json(payload)}")

        if self._test_client is not None:
            resp = self._test_client.post(path, json=payload)
            if resp.status_code >= 400:
                logger.error(f"[ModelBackendAPI] POST {url} failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(f"Model-backend request failed: {path} status={resp.status_code} body={resp.text}")
            data = resp.json()
            if not isinstance(data, dict):
                logger.error(f"[ModelBackendAPI] POST {url} returned non-object")
                raise RuntimeError(f"Model-backend response must be object: {path}")
            logger.debug(f"[ModelBackendAPI] POST {url} response: {_truncate_json(data)}")
            return data

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        data = self._urlopen_json(req, url, "POST")
        logger.debug(f"[ModelBackendAPI] POST {url} response: {_truncate_json(data)}")
        return data

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        logger.debug(f"[ModelBackendAPI] GET {url}")

        if self._test_client is not None:
            resp = self._test_client.get(path)
            if resp.status_code >= 400:
                logger.error(f"[ModelBackendAPI] GET {url} failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(f"Model-backend request failed: {path} status={resp.status_code} body={resp.text}")
            data = resp.json()
            if not isinstance(data, dict):
                logger.error(f"[ModelBackendAPI] GET {url} returned non-object")
                raise RuntimeError(f"Model-backend response must be object: {path}")
            logger.debug(f"[ModelBackendAPI] GET {url} response: {_truncate_json(data)}")
            return data

        req = request.Request(url=url, method="GET")
        data = self._urlopen_json(req, url, "GET")
        logger.debug(f"[ModelBackendAPI] GET {url} response: {_truncate_json(data)}")
        return data

    def _urlopen_json(self, req: request.Request, url: str, method: str = "GET") -> dict[str, Any]:
        last_exc: Exception | None = None
        for _ in range(self.retry + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    logger.error(f"[ModelBackendAPI] {method} {url} returned non-object JSON")
                    raise RuntimeError(f"JSON response from {url} must be object")
                return data
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                logger.error(f"[ModelBackendAPI] {method} {url} HTTP {exc.code} - {detail}")
                last_exc = RuntimeError(f"HTTP {exc.code} from {url}: {detail}")
            except Exception as exc:
                logger.error(f"[ModelBackendAPI] {method} {url} Request failed: {exc}")
                last_exc = RuntimeError(f"HTTP request failed for {url}: {exc}")
        assert last_exc is not None
        raise last_exc

    def detect_objects_in_frame(
        self,
        frame_idx: int,
        timestamp: float,
        prompts: list[str] | None = None,
        image_b64: str | None = None,
    ) -> FrameDetections:
        logger.info(
            f"[ModelBackendAPI] Calling detector backend ({self.base_url}/v1/tasks/detect-objects-in-frame) for frame {frame_idx}..."
        )
        payload: dict[str, Any] = {"frame_idx": frame_idx, "timestamp": timestamp}
        if prompts is not None:
            payload["prompts"] = prompts
        if image_b64:
            payload["image_b64"] = image_b64
        data = self._post("/v1/tasks/detect-objects-in-frame", payload)
        detections = FrameDetections.model_validate(data.get("detections", {}))
        logger.info(
            f"[ModelBackendAPI] Detector backend ({self.base_url}) returned {len(detections.instances)} instances."
        )
        return detections

    def reconstruct_object_meshes(self, detections: FrameDetections, objects: list[ObjectNode]) -> dict[str, dict]:
        logger.info(
            f"[ModelBackendAPI] Calling mesh reconstruction backend ({self.base_url}/v1/tasks/reconstruct-object-meshes) for {len(objects)} objects..."
        )
        data = self._post(
            "/v1/tasks/reconstruct-object-meshes",
            {
                "detections": detections.model_dump(),
                "objects": [obj.model_dump() for obj in objects],
            },
        )
        out = data.get("meshes")
        meshes = out if isinstance(out, dict) else {}
        logger.info(f"[ModelBackendAPI] Reconstruction backend ({self.base_url}) finished. {len(meshes)} meshes returned.")
        return meshes

    def infer_object_physics_priors(
        self,
        detections: FrameDetections,
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        logger.info(
            f"[ModelBackendAPI] Calling physics priors backend ({self.base_url}/v1/tasks/infer-object-physics-priors) for {len(objects)} objects..."
        )
        data = self._post(
            "/v1/tasks/infer-object-physics-priors",
            {
                "detections": detections.model_dump(),
                "objects": [obj.model_dump() for obj in objects],
            },
        )
        out = data.get("priors")
        priors = out if isinstance(out, dict) else {}
        logger.info(f"[ModelBackendAPI] Physics priors backend ({self.base_url}) finished. {len(priors)} priors returned.")
        return priors

    def discover_movable_object_categories(self, image_b64: str) -> list[str]:
        logger.info(
            f"[ModelBackendAPI] Calling movable object discovery backend ({self.base_url}/v1/tasks/discover-movable-object-categories)..."
        )
        data = self._post("/v1/tasks/discover-movable-object-categories", {"image_b64": image_b64})
        categories = data.get("categories")
        result = [str(x) for x in categories] if isinstance(categories, list) else []
        logger.info(f"[ModelBackendAPI] Discovery backend ({self.base_url}) found {len(result)} categories: {result}")
        return result

    def set_object_detection_prompts(self, prompts: list[str]) -> list[str]:
        data = self._post("/v1/object-detection/prompts", {"prompts": prompts})
        got = data.get("prompts")
        return [str(x) for x in got] if isinstance(got, list) else []

    def get_object_detection_prompts(self) -> list[str]:
        data = self._get("/v1/object-detection/prompts")
        got = data.get("prompts")
        return [str(x) for x in got] if isinstance(got, list) else []

    def get_detector_status(self) -> dict:
        data = self._get("/v1/detector/status")
        status = data.get("status")
        return status if isinstance(status, dict) else {}




class VideoBackedObjectDetector:
    def __init__(self, client: ModelBackendAPI, video_source: str | None = None) -> None:
        self._client = client
        self._video_source = (video_source or "").strip() or None
        self._prompts = []
        self._cap = None
        self._first_frame_b64: str | None = None

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        self._prompts = self._client.set_object_detection_prompts(prompts)

    def get_object_detection_prompts(self) -> list[str]:
        return list(self._prompts)

    def detector_status(self) -> dict:
        return self._client.get_detector_status()

    def _read_b64_frame(self) -> str | None:
        if not self._video_source:
            return None
            
        try:
            import cv2
            import base64
        except ImportError:
            return None
            
        if self._cap is None:
            target = int(self._video_source) if self._video_source.isdigit() else self._video_source
            self._cap = cv2.VideoCapture(target)
            
        ok, frame = self._cap.read()
        if not ok:
            # try to re-init once
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return None
                
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
            
        b64_str = base64.b64encode(buffer).decode("ascii")
        if self._first_frame_b64 is None:
            self._first_frame_b64 = b64_str
        return b64_str

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float) -> FrameDetections:
        image_b64 = self._read_b64_frame()
        return self._client.detect_objects_in_frame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            prompts=self._prompts,
            image_b64=image_b64,
        )

    def get_first_frame_b64(self) -> str | None:
        if self._first_frame_b64 is None:
            # Force read first frame
            self._read_b64_frame()
        return self._first_frame_b64


class BackendSAM3DAdapter:
    def __init__(
        self,
        client: ModelBackendAPI,
        materialization_root: str | None = None,
        materialization_mode: str = "copy",
    ) -> None:
        self._client = client
        self._materialization_root = Path(materialization_root).resolve() if materialization_root else None
        self._materialization_mode = materialization_mode

    def reconstruct_object_meshes(
        self,
        best_frames: dict[str, tuple["FrameDetections", "DetectedInstance"]],
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        # Use any available frame for the backend API call (no per-object best-frame API yet)
        any_frame = next((fd for fd, _inst in best_frames.values()), None)
        if any_frame is None:
            return {}
        meshes = self._client.reconstruct_object_meshes(any_frame, objects)
        if not self._materialization_root:
            return meshes
        frame_root = self._materialization_root / "intermediate" / f"frame_{int(any_frame.frame_idx):06d}" / "objects"
        out: dict[str, dict] = {}
        for object_id, entry in meshes.items():
            if not isinstance(entry, dict):
                out[object_id] = entry
                continue
            src = str(entry.get("mesh_path", "")).strip()
            if not src:
                out[object_id] = entry
                continue
            object_root = frame_root / _safe_name(object_id) / "assets"
            object_root.mkdir(parents=True, exist_ok=True)
            src_path = Path(src).expanduser()
            if not src_path.exists():
                out[object_id] = entry
                continue
            ext = src_path.suffix or ".bin"
            dst = _unique_path(object_root / f"object{ext}")
            _materialize_file(src_path, dst, self._materialization_mode)
            updated = dict(entry)
            updated["mesh_path"] = str(dst)
            updated.setdefault("files", []).append({"format": ext.lstrip("."), "path": str(dst)})
            out[object_id] = updated
        return out


def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    mode = (mode or "copy").strip().lower()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return safe or "unknown"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


class BackendVLMAdapter:
    def __init__(self, client: ModelBackendAPI) -> None:
        self._client = client

    def infer_object_physics_priors(
        self,
        detections: FrameDetections,
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        return self._client.infer_object_physics_priors(detections, objects)

    def discover_movable_object_categories(self, image_b64: str) -> list[str]:
        return self._client.discover_movable_object_categories(image_b64)
