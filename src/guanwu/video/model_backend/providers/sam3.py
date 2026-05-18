from __future__ import annotations

import logging
import tempfile
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

from guanwu.video.model_backend.config import SAM3ProviderConfig
from guanwu.video.model_backend.schemas import DetectedInstanceModel, FrameDetectionsModel
from guanwu.video.model_backend.providers.base import get_json, post_json


class DisabledSAM3Provider:
    mode = "disabled"

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float, image_b64: str | None = None) -> FrameDetectionsModel:
        _ = image_b64
        return FrameDetectionsModel(frame_idx=frame_idx, timestamp=timestamp, instances=[])

    def set_object_detection_prompts(self, prompts: list[str]) -> list[str]:
        _ = prompts
        return []

    def get_object_detection_prompts(self) -> list[str]:
        return []

    def detector_status(self) -> dict:
        return {"backend": "disabled", "ready": False}

    def get_first_frame_path(self) -> str | None:
        return None


class EmbeddedSAM3Provider:
    mode = "embedded"

    def __init__(self, cfg: SAM3ProviderConfig) -> None:
        self._cfg = cfg
        self._prompts = [p.strip().lower() for p in cfg.prompts if str(p).strip()]
        self._backend = cfg.backend.strip().lower()
        self._ultra = None
        if self._backend == "ultralytics":
            self._ultra = _UltralyticsRunner(cfg)

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float, image_b64: str | None = None) -> FrameDetectionsModel:
        if self._backend == "ultralytics" and self._ultra is not None:
            return self._ultra.detect_objects_in_frame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                prompts=self._prompts,
                image_b64=image_b64,
            )
        raise NotImplementedError("SAM3 embedded backend only supports 'ultralytics' backend mode. 'mock' mode has been removed.")

    def set_object_detection_prompts(self, prompts: list[str]) -> list[str]:
        self._prompts = [p.strip().lower() for p in prompts if str(p).strip()]
        return self._prompts

    def get_object_detection_prompts(self) -> list[str]:
        return list(self._prompts)

    def detector_status(self) -> dict:
        device = "cpu"
        if self._ultra is not None:
            device = self._ultra.device
        return {
            "backend": f"embedded:{self._backend}",
            "device": device,
            "weights_path": None,
            "video_source": None,
            "frame_dump_dir": None,
            "ready": True,
        }

    def get_first_frame_path(self) -> str | None:
        return None

    @staticmethod
    def _segment_kind_for_label(label: str) -> str:
        body_terms = {
            "person",
            "human",
            "body",
            "hand",
            "arm",
            "leg",
            "face",
            "head",
            "torso",
            "man",
            "woman",
            "boy",
            "girl",
        }
        if any(term in label for term in body_terms):
            return "body"
        return "object"


class HttpSAM3Provider:
    def __init__(
        self,
        base_url: str,
        timeout_sec: float,
        mode: str,
    ) -> None:
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.mode = mode

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float, image_b64: str | None = None) -> FrameDetectionsModel:
        payload: dict[str, Any] = {"frame_idx": frame_idx, "timestamp": timestamp}
        if image_b64:
            payload["image_b64"] = image_b64
        data = post_json(
            self.base_url,
            "/v1/tasks/detect-objects-in-frame",
            payload,
            self.timeout_sec,
        )
        return FrameDetectionsModel.model_validate(data.get("detections", {}))

    def set_object_detection_prompts(self, prompts: list[str]) -> list[str]:
        data = post_json(self.base_url, "/v1/object-detection/prompts", {"prompts": prompts}, self.timeout_sec)
        got = data.get("prompts")
        return [str(x) for x in got] if isinstance(got, list) else []

    def get_object_detection_prompts(self) -> list[str]:
        data = get_json(self.base_url, "/v1/object-detection/prompts", self.timeout_sec)
        got = data.get("prompts")
        return [str(x) for x in got] if isinstance(got, list) else []

    def detector_status(self) -> dict:
        return get_json(self.base_url, "/v1/detector/status", self.timeout_sec)

    def get_first_frame_path(self) -> str | None:
        data = get_json(self.base_url, "/v1/perception/first-frame", self.timeout_sec)
        val = data.get("frame_path")
        return str(val) if isinstance(val, str) and val else None


def build_sam3_provider(cfg: SAM3ProviderConfig):
    mode = cfg.mode.strip().lower()
    if mode == "http":
        return HttpSAM3Provider(cfg.service.base_url, cfg.service.timeout_sec, mode="http")
    if mode == "embedded":
        return EmbeddedSAM3Provider(cfg)
    return DisabledSAM3Provider()


class _UltralyticsRunner:
    def __init__(self, cfg: SAM3ProviderConfig) -> None:
        self._weights_path = (cfg.yolo_weights or "").strip()
        self._conf = float(cfg.confidence)
        self._frame_dump_dir = cfg.frame_dump_dir
        self._device = "cpu"
        self._model = self._load_model()
        self._caps: dict[str, Any] = {}

    @property
    def device(self) -> str:
        return self._device

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "SAM3 embedded ultralytics backend requires ultralytics+timm+lapx. "
                "Install extras: uv pip install -e '.[vision]'"
            ) from exc

        weights = Path(self._weights_path)
        if not self._weights_path or not weights.exists():
            raise RuntimeError("SAM3 embedded ultralytics backend requires existing yolo_weights path.")

        self._device = self._select_device()
        fname = weights.name.lower()
        if "sam" in fname:
            from ultralytics import SAM
            return SAM(str(weights))
        return YOLO(str(weights))

    def detect_objects_in_frame(
        self,
        frame_idx: int,
        timestamp: float,
        prompts: list[str],
        image_b64: str | None = None,
    ) -> FrameDetectionsModel:
        image_data = (image_b64 or "").strip()
        if not image_data:
            return FrameDetectionsModel(frame_idx=frame_idx, timestamp=timestamp, instances=[])

        frame = self._read_b64_frame(image_data)
        if frame is None:
            return FrameDetectionsModel(frame_idx=frame_idx, timestamp=timestamp, instances=[])
        frame_path = self._dump_frame(frame_idx, frame)

        # Fix: ensure imgsz is multiple of stride to avoid warning logs every frame
        stride = 32
        # Try different ways to get the model's max stride (usually 32 for YOLOv8/v11, 14 for some FastSAM)
        if hasattr(self._model, "stride"):
            stride = int(self._model.stride)
        elif hasattr(self._model, "model") and hasattr(self._model.model, "stride"):
            try:
                stride = int(self._model.model.stride.max())
            except Exception:
                pass
        
        # We usually want a base of 1024 or 640.
        base_imgsz = 1024
        imgsz = base_imgsz
        if imgsz % stride != 0:
            imgsz = int(((imgsz // stride) + 1) * stride)

        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self._conf,
            tracker="bytetrack.yaml",
            device=self._device,
            imgsz=imgsz,
        )
        if not results:
            return FrameDetectionsModel(frame_idx=frame_idx, timestamp=timestamp, instances=[], image_b64=image_b64)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return FrameDetectionsModel(frame_idx=frame_idx, timestamp=timestamp, instances=[], image_b64=image_b64)

        names = getattr(result, "names", {})
        xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
        conf_list = boxes.conf.tolist() if boxes.conf is not None else []
        cls_list = boxes.cls.tolist() if boxes.cls is not None else []
        id_list = boxes.id.tolist() if boxes.id is not None else []
        instances: list[DetectedInstanceModel] = []

        normalized_prompts = [p.strip().lower() for p in prompts if p.strip()]
        for idx, bbox in enumerate(xyxy_list):
            cls_idx = int(cls_list[idx]) if idx < len(cls_list) else -1
            label = str(names.get(cls_idx, f"cls_{cls_idx}")).lower()
            if normalized_prompts and not self._prompt_match(label, normalized_prompts):
                continue
            conf = float(conf_list[idx]) if idx < len(conf_list) else 0.0
            track_id = f"trk_{int(id_list[idx])}" if idx < len(id_list) else f"trk_f{frame_idx}_{idx}"
            instances.append(
                DetectedInstanceModel(
                    mask_ref=f"mask://frame_{frame_idx:05d}/{track_id}",
                    bbox=[float(v) for v in bbox],
                    track_id=track_id,
                    concept_label=label,
                    segment_kind=self._segment_kind_for_label(label),
                    score=conf,
                )
            )

        return FrameDetectionsModel(frame_idx=frame_idx, timestamp=timestamp, instances=instances, image_b64=image_b64)

    def _read_b64_frame(self, b64_str: str):
        try:
            import cv2
            import numpy as np
            import base64
        except Exception as exc:
            raise RuntimeError(
                "SAM3 embedded ultralytics backend requires OpenCV to decode base64 images. Install extras: uv pip install -e '.[vision]'"
            ) from exc
            
        try:
            # Handle typical data URI prefix (data:image/jpeg;base64,... )
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            jpg_original = base64.b64decode(b64_str)
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            return frame
        except Exception:
            return None

    def _dump_frame(self, frame_idx: int, frame) -> str | None:
        try:
            import cv2
        except Exception:
            return None
        if self._frame_dump_dir:
            out_dir = Path(self._frame_dump_dir)
        else:
            out_dir = Path(tempfile.gettempdir()) / "spwm_guanwu.video.model_backend_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{frame_idx:06d}.jpg"
        ok = cv2.imwrite(str(out_file), frame)
        return str(out_file) if ok else None

    def _select_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("SAM3 using CUDA device: %s (%s)", device, gpu_name)
                return device
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("SAM3 using MPS (Apple Silicon GPU)")
                return "mps"
            logger.warning("SAM3 falling back to CPU – no CUDA or MPS device found")
            return "cpu"
        except Exception:
            logger.warning("SAM3 falling back to CPU – torch not available or device detection failed")
            return "cpu"

    @staticmethod
    def _prompt_match(label: str, prompts: list[str]) -> bool:
        for prompt in prompts:
            if prompt in label or label in prompt:
                return True
        return False

    @staticmethod
    def _segment_kind_for_label(label: str) -> str:
        body_terms = {
            "person",
            "human",
            "body",
            "hand",
            "arm",
            "leg",
            "face",
            "head",
            "torso",
            "man",
            "woman",
            "boy",
            "girl",
        }
        if any(term in label for term in body_terms):
            return "body"
        return "object"
