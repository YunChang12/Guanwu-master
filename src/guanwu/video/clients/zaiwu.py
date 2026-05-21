from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from guanwu.video.core.config import SPWMSettings
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)


def normalize_provider_mode(value: str | None) -> str:
    mode = str(value or "mock").strip().lower()
    if mode in {"mcp", "model_backend"}:
        return "zaiwu"
    return mode or "mock"


def normalize_service_id(value: str | None) -> str:
    service_id = str(value or "").strip()
    if not service_id:
        return ""
    if service_id.startswith("mcps."):
        return f"services.{service_id.split('.', 1)[1]}"
    return service_id


def _worker_status_is_ready(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return True
    return normalized in {"running", "ready", "active", "serving", "started"}


def _is_retryable_job_poll_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "unexpected eof while reading",
            "eof occurred in violation of protocol",
            "server disconnected without sending a response",
            "connection reset",
            "timed out",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "unknown job",
            "gateway error 502",
            "gateway error 503",
            "gateway error 504",
            "gateway error 404",
        )
    )


@dataclass(frozen=True)
class ZaiwuServiceEndpoint:
    service_id: str
    host: str
    ready_port: int
    run_group: str
    status: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.ready_port}"

    @property
    def sse_url(self) -> str:
        return f"{self.base_url}/sse"


class ZaiwuGatewayClient:
    def __init__(
        self,
        *,
        gateway_url: str,
        request_timeout_sec: float = 30.0,
        job_timeout_sec: float = 1800.0,
        job_poll_interval_sec: float = 1.0,
        auto_start_workers: bool = True,
        worker_run_group: str = "services",
    ) -> None:
        self.gateway_url = str(gateway_url).rstrip("/")
        self.request_timeout_sec = float(request_timeout_sec)
        self.job_timeout_sec = float(job_timeout_sec)
        self.job_poll_interval_sec = max(0.1, float(job_poll_interval_sec))
        self.auto_start_workers = bool(auto_start_workers)
        self.worker_run_group = str(worker_run_group or "services")
        self._service_cache: dict[str, ZaiwuServiceEndpoint] = {}
        self._gateway_api: dict[str, str] = {}

        parsed = urlparse(self.gateway_url)
        self._gateway_scheme = parsed.scheme or "http"
        self._gateway_host = parsed.hostname or "127.0.0.1"
        if self._gateway_host in {"0.0.0.0", "::"}:
            self._gateway_host = "127.0.0.1"

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.gateway_url}{path}"
        timeout = float(timeout_sec or self.request_timeout_sec)
        with httpx.Client(verify=False, timeout=timeout) as client:
            response = client.request(method.upper(), url, json=json_payload)
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text}
        if response.is_error:
            raise RuntimeError(f"Zaiwu gateway error {response.status_code} for {method.upper()} {path}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Zaiwu gateway payload for {method.upper()} {path}: {payload!r}")
        return payload

    def _upload_via_gateway(
        self,
        path: str,
        *,
        content: bytes,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.gateway_url}{path}"
        timeout = max(300.0, float(timeout_sec or self.request_timeout_sec))
        with httpx.Client(verify=False, timeout=timeout) as client:
            response = client.post(url, content=content)
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text}
        if response.is_error:
            raise RuntimeError(f"Zaiwu gateway upload error {response.status_code} for POST {path}: {payload}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Zaiwu gateway upload payload for POST {path}: {payload!r}")
        return payload

    def _download_via_gateway(
        self,
        path: str,
        *,
        timeout_sec: float | None = None,
    ) -> bytes:
        url = f"{self.gateway_url}{path}"
        timeout = max(120.0, float(timeout_sec or self.request_timeout_sec))
        with httpx.Client(verify=False, timeout=timeout) as client:
            response = client.get(url)
        response.raise_for_status()
        return response.content

    def _remember_gateway_api(self, payload: dict[str, Any]) -> None:
        api = payload.get("api")
        if not isinstance(api, dict):
            return
        for key in (
            "workers_endpoint",
            "jobs_endpoint",
            "job_status_template",
            "upload_endpoint",
            "download_template",
            "files_endpoint",
            "artifact_metadata_template",
        ):
            value = api.get(key)
            if isinstance(value, str) and value.strip():
                self._gateway_api[key] = value.strip()

    def _gateway_api_path(self, key: str, default: str, **params: str) -> str:
        template = self._gateway_api.get(key)
        if not template:
            try:
                self.list_workers()
            except Exception:
                pass
            template = self._gateway_api.get(key)
        path = template or default
        for name, value in params.items():
            path = path.replace(f"{{{name}}}", quote(str(value), safe=""))
        return path

    def list_workers(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/api/v1/workers")
        self._remember_gateway_api(payload)
        return payload

    def start_service(self, service_id: str) -> dict[str, Any]:
        service_id = normalize_service_id(service_id)
        logger.info("[Zaiwu] Starting worker for %s via gateway ...", service_id)
        return self._request_json(
            "POST",
            "/api/v1/workers/actions/start",
            json_payload={"run_group": self.worker_run_group, "service_id": service_id},
        )

    def _extract_endpoint(self, service_id: str, payload: dict[str, Any]) -> ZaiwuServiceEndpoint | None:
        items = payload.get("items", [])
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("service_id", "")).strip() != service_id:
                continue
            ready_port = item.get("ready_port")
            if ready_port in {None, "", 0}:
                continue
            try:
                port = int(ready_port)
            except (TypeError, ValueError):
                continue
            status = str(item.get("status") or "")
            if not _worker_status_is_ready(status):
                continue
            return ZaiwuServiceEndpoint(
                service_id=service_id,
                host=self._gateway_host,
                ready_port=port,
                run_group=str(item.get("run_group") or self.worker_run_group),
                status=status,
            )
        return None

    def get_ready_service(self, service_id: str) -> ZaiwuServiceEndpoint | None:
        service_id = normalize_service_id(service_id)
        cached = self._service_cache.get(service_id)
        if cached is not None:
            return cached
        payload = self.list_workers()
        endpoint = self._extract_endpoint(service_id, payload)
        if endpoint is not None:
            self._service_cache[service_id] = endpoint
        return endpoint

    def ensure_service(self, service_id: str, *, timeout_sec: float = 60.0) -> ZaiwuServiceEndpoint:
        service_id = normalize_service_id(service_id)
        cached = self._service_cache.get(service_id)
        if cached is not None:
            return cached

        deadline = time.time() + max(1.0, float(timeout_sec))
        started = False
        last_payload: dict[str, Any] | None = None

        while time.time() < deadline:
            payload = self.list_workers()
            last_payload = payload
            endpoint = self._extract_endpoint(service_id, payload)
            if endpoint is not None:
                self._service_cache[service_id] = endpoint
                return endpoint
            if self.auto_start_workers and not started:
                self.start_service(service_id)
                started = True
            time.sleep(self.job_poll_interval_sec)

        raise RuntimeError(
            f"Timed out waiting for Zaiwu worker {service_id} to become ready via {self.gateway_url}. "
            f"Last worker payload: {last_payload}"
        )

    def service_base_url(self, service_id: str) -> str:
        return self.ensure_service(service_id).base_url

    def service_sse_url(self, service_id: str) -> str:
        return self.ensure_service(service_id).sse_url

    def run_service_job(
        self,
        service_id: str,
        operation: str,
        payload: dict[str, Any],
        *,
        requested_by: str = "guanwu",
        execution_labels: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        service_id = normalize_service_id(service_id)
        self.ensure_service(service_id)
        labels = {"service_id": service_id}
        if execution_labels:
            labels.update({str(key): str(value) for key, value in execution_labels.items()})
        return self.run_job(
            handler=f"{service_id}.{operation}",
            payload=payload,
            requested_by=requested_by,
            execution_labels=labels,
            timeout_sec=timeout_sec,
        )

    def submit_job(
        self,
        *,
        handler: str,
        payload: dict[str, Any],
        requested_by: str = "guanwu",
        execution_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/api/v1/jobs",
            json_payload={
                "handler": handler,
                "payload": payload,
                "requested_by": requested_by,
                "execution_labels": execution_labels or {},
            },
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            self._gateway_api_path("job_status_template", "/api/v1/jobs/{job_id}", job_id=job_id),
        )

    def wait_for_job(self, job_id: str, *, timeout_sec: float | None = None) -> dict[str, Any]:
        deadline = time.time() + float(timeout_sec or self.job_timeout_sec)
        last_record: dict[str, Any] | None = None
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                record = self.get_job(job_id)
                last_error = None
            except Exception as exc:  # noqa: BLE001
                if not _is_retryable_job_poll_error(exc):
                    raise
                last_error = exc
                logger.warning("[Zaiwu] Transient polling error for job %s: %s", job_id, exc)
                time.sleep(self.job_poll_interval_sec)
                continue
            last_record = record
            status = str(record.get("status") or "")
            if status == "succeeded":
                return record
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Zaiwu job {job_id} ended with status={status}: {record.get('error')}")
            time.sleep(self.job_poll_interval_sec)
        raise RuntimeError(f"Timed out waiting for Zaiwu job {job_id}: last_record={last_record}, last_error={last_error}")

    def run_job(
        self,
        *,
        handler: str,
        payload: dict[str, Any],
        requested_by: str = "guanwu",
        execution_labels: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        submitted = self.submit_job(
            handler=handler,
            payload=payload,
            requested_by=requested_by,
            execution_labels=execution_labels,
        )
        spec = submitted.get("spec", {})
        job_id = str(spec.get("job_id") or submitted.get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"Zaiwu gateway did not return a job_id for handler {handler}: {submitted}")
        record = self.wait_for_job(job_id, timeout_sec=timeout_sec)
        result = record.get("result")
        if isinstance(result, str) and result.strip():
            return {"artifact_id": result.strip()}
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected Zaiwu job result for {job_id}: {record}")
        return result

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            self._gateway_api_path(
                "artifact_metadata_template",
                "/api/v1/artifacts/{artifact_id}",
                artifact_id=artifact_id,
            ),
        )

    def upload_file(self, service_id: str, path: str | Path) -> str:
        service_id = normalize_service_id(service_id)
        file_path = Path(path).expanduser().resolve()
        payload = self._upload_via_gateway(
            f"{self._gateway_api_path('upload_endpoint', '/upload')}?filename={quote(file_path.name)}",
            content=file_path.read_bytes(),
        )
        file_id = str(payload.get("file_id") or "")
        if not file_id:
            raise RuntimeError(f"Zaiwu gateway did not return file_id for upload {file_path} (service={service_id})")
        return file_id

    def download_bytes(self, service_id: str, file_id: str) -> bytes:
        service_id = normalize_service_id(service_id)
        _ = service_id
        return self._download_via_gateway(
            self._gateway_api_path("download_template", "/download/{file_id}", file_id=file_id)
        )


def build_zaiwu_gateway_client(settings: SPWMSettings) -> ZaiwuGatewayClient:
    return ZaiwuGatewayClient(
        gateway_url=settings.zaiwu.gateway_url,
        request_timeout_sec=settings.zaiwu.request_timeout_sec,
        job_timeout_sec=settings.zaiwu.job_timeout_sec,
        job_poll_interval_sec=settings.zaiwu.job_poll_interval_sec,
        auto_start_workers=settings.zaiwu.auto_start_workers,
        worker_run_group=settings.zaiwu.worker_run_group,
    )


class _LocalVideoMixin:
    def __init__(self, video_source: str | None = None) -> None:
        self._video_source = (video_source or "").strip() or None
        self._video_capture = None
        self._first_frame_b64: str | None = None

    def _read_b64_frame(self) -> str | None:
        if not self._video_source:
            return None
        try:
            import cv2
        except ImportError:
            return None

        if self._video_capture is None:
            target = int(self._video_source) if self._video_source.isdigit() else self._video_source
            self._video_capture = cv2.VideoCapture(target)

        ok, frame = self._video_capture.read()
        if not ok:
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._video_capture.read()
            if not ok:
                return None

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None

        image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        if self._first_frame_b64 is None:
            self._first_frame_b64 = image_b64
        return image_b64

    def _read_b64_frame_at(self, frame_idx: int) -> str | None:
        if not self._video_source or self._video_source.isdigit():
            return None
        try:
            import cv2
        except ImportError:
            return None

        cap = cv2.VideoCapture(self._video_source)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx) - 1))
            ok, frame = cap.read()
            if not ok:
                return None
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                return None
            image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
            if int(frame_idx) == 1 and self._first_frame_b64 is None:
                self._first_frame_b64 = image_b64
            return image_b64
        finally:
            cap.release()

    def get_first_frame_b64(self) -> str | None:
        if self._first_frame_b64 is None:
            self._read_b64_frame()
        return self._first_frame_b64


class _ZaiwuVideoDetectorBase(_LocalVideoMixin):
    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        video_source: str | None = None,
        max_frames: int | None = None,
    ) -> None:
        super().__init__(video_source=video_source)
        self.gateway = gateway
        self.service_id = service_id
        self._max_frames = max_frames
        self._video_batches: dict[int, Any] = {}
        self._video_prefetch_attempted = False
        self._uploaded_video_cache: dict[tuple[str, int, int], str] = {}
        self._prompts: list[str] = []

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        self._prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]

    def get_object_detection_prompts(self) -> list[str]:
        return list(self._prompts)

    def detector_status(self) -> dict[str, Any]:
        return {
            "backend": "zaiwu_jobs",
            "service_id": self.service_id,
        }

    def prefetch_video(self) -> None:
        if self._video_prefetch_attempted:
            return
        self._video_prefetch_attempted = True
        if not self._video_source or self._video_source.isdigit():
            return

        payload = self._video_job_payload()
        result = self.gateway.run_service_job(
            self.service_id,
            self._video_operation(),
            payload,
        )
        frames = self._video_frames_from_result(result)
        loaded = 0
        for item in frames:
            detections = _normalize_frame_payload(item)
            self._video_batches[detections.frame_idx] = detections
            loaded += 1
        logger.info("[Zaiwu] %s loaded %d frame batches via jobs.", self.service_id, loaded)

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float):  # noqa: ANN201
        cached = self._video_batches.get(frame_idx)
        if cached is not None:
            return self._attach_image_if_missing(cached)

        frame_payload = self._frame_job_payload(frame_idx, timestamp)
        if frame_payload is None:
            self.prefetch_video()
            cached = self._video_batches.get(frame_idx)
            if cached is not None:
                return self._attach_image_if_missing(cached)
            from guanwu.video.core.types import FrameDetections

            return FrameDetections(frame_idx=frame_idx, timestamp=timestamp, instances=[])

        result = self.gateway.run_service_job(
            self.service_id,
            self._frame_operation(),
            frame_payload,
        )
        detections = _normalize_frame_payload(result)
        return self._attach_image_if_missing(detections)

    def _attach_image_if_missing(self, detections):  # noqa: ANN001, ANN201
        if detections.image_b64:
            return detections
        image_b64 = self._read_b64_frame_at(detections.frame_idx)
        if image_b64:
            detections = detections.model_copy(update={"image_b64": image_b64})
            self._video_batches[detections.frame_idx] = detections
        return detections

    def _video_frames_from_result(self, result: Any) -> list[dict[str, Any]]:
        if not isinstance(result, dict):
            return []
        frames = result.get("frames")
        if isinstance(frames, list):
            return [item for item in frames if isinstance(item, dict)]
        artifact_payload = self._load_json_artifact_payload(result)
        if not isinstance(artifact_payload, dict):
            return []
        artifact_frames = artifact_payload.get("frames")
        if not isinstance(artifact_frames, list):
            return []
        return [item for item in artifact_frames if isinstance(item, dict)]

    def _load_json_artifact_payload(self, result: dict[str, Any]) -> dict[str, Any] | None:
        artifact_id = ""
        for key in ("output_file_id", "result_file_id", "frames_file_id", "artifact_id", "file_id"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                artifact_id = value.strip()
                break
        if not artifact_id:
            return None
        data = self.gateway.download_bytes(self.service_id, artifact_id)
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Zaiwu {self.service_id} returned artifact payload {artifact_id} that is not valid UTF-8 JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Zaiwu {self.service_id} returned artifact payload {artifact_id} with unsupported type: {type(payload)!r}"
            )
        return payload

    def _uploaded_video_id(self) -> str:
        if not self._video_source:
            raise ValueError(f"{self.service_id} requires a local video path.")
        video_path = Path(self._video_source).expanduser().resolve()
        stat = video_path.stat()
        cache_key = (str(video_path), int(stat.st_mtime_ns), int(stat.st_size))
        file_id = self._uploaded_video_cache.get(cache_key)
        if file_id is None:
            file_id = self.gateway.upload_file(self.service_id, video_path)
            self._uploaded_video_cache = {cache_key: file_id}
        return file_id

    def _video_operation(self) -> str:
        raise NotImplementedError

    def _frame_operation(self) -> str:
        raise NotImplementedError

    def _video_job_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def _frame_job_payload(self, frame_idx: int, timestamp: float) -> dict[str, Any] | None:
        raise NotImplementedError


class ZaiwuSAM3Detector(_ZaiwuVideoDetectorBase):
    def _video_operation(self) -> str:
        return "parse_video"

    def _frame_operation(self) -> str:
        return "parse_frame"

    def _video_job_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video_file_id": self._uploaded_video_id(),
            "prompts": list(self._prompts),
            "start_frame": 1,
        }
        if self._max_frames is not None and int(self._max_frames) > 0:
            payload["max_frames"] = int(self._max_frames)
        return payload

    def _frame_job_payload(self, frame_idx: int, timestamp: float) -> dict[str, Any] | None:
        image_b64 = self._read_b64_frame()
        if not image_b64:
            return None
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "image_base64": image_b64,
            "prompts": list(self._prompts),
        }


class ZaiwuGroundedSAM2Detector(_ZaiwuVideoDetectorBase):
    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        video_source: str | None = None,
        max_frames: int | None = None,
        step: int = 20,
        iou_threshold: float = 0.8,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        super().__init__(gateway, service_id=service_id, video_source=video_source, max_frames=max_frames)
        self._step = int(step)
        self._iou_threshold = float(iou_threshold)
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)
        self._text_prompt = ""

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        super().set_object_detection_prompts(prompts)
        self._text_prompt = ". ".join(prompt for prompt in self._prompts)
        if self._text_prompt and not self._text_prompt.endswith("."):
            self._text_prompt += "."

    def _video_operation(self) -> str:
        return "gsam2_parse_video"

    def _frame_operation(self) -> str:
        return "gsam2_parse_frame"

    def _video_job_payload(self) -> dict[str, Any]:
        return {
            "video_file_id": self._uploaded_video_id(),
            "text_prompt": self._text_prompt or "object.",
            "step": self._step,
            "iou_threshold": self._iou_threshold,
            "box_threshold": self._box_threshold,
            "text_threshold": self._text_threshold,
        }

    def _frame_job_payload(self, frame_idx: int, timestamp: float) -> dict[str, Any] | None:
        image_b64 = self._read_b64_frame()
        if not image_b64:
            return None
        return {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "image_base64": image_b64,
            "text_prompt": self._text_prompt or "object.",
        }


class ZaiwuSeg2TrackDetector(_ZaiwuVideoDetectorBase):
    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        video_source: str | None = None,
        detect_interval: int = 5,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        super().__init__(gateway, service_id=service_id, video_source=video_source)
        self._detect_interval = int(detect_interval)
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)
        self._text_prompt = ""

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        super().set_object_detection_prompts(prompts)
        self._text_prompt = ". ".join(prompt for prompt in self._prompts)
        if self._text_prompt and not self._text_prompt.endswith("."):
            self._text_prompt += "."

    def _video_operation(self) -> str:
        return "seg2track_parse_video"

    def _frame_operation(self) -> str:
        return ""

    def _video_job_payload(self) -> dict[str, Any]:
        return {
            "video_file_id": self._uploaded_video_id(),
            "text_prompt": self._text_prompt or "object.",
            "detect_interval": self._detect_interval,
            "box_threshold": self._box_threshold,
            "text_threshold": self._text_threshold,
        }

    def _frame_job_payload(self, frame_idx: int, timestamp: float) -> dict[str, Any] | None:
        return None


class ZaiwuSAM3DAdapter:
    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        materialization_root: str | None = None,
        materialization_mode: str = "copy",
        per_object_timeout_sec: float = 180.0,
    ) -> None:
        self.gateway = gateway
        self.service_id = service_id
        self._materialization_root = Path(materialization_root).resolve() if materialization_root else None
        self._materialization_mode = materialization_mode
        self._per_object_timeout_sec = float(per_object_timeout_sec)

    def reconstruct_object_meshes(
        self,
        best_frames: dict[str, tuple["FrameDetections", "DetectedInstance"]],
        objects: list["ObjectNode"],
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for obj in objects:
            frame_data = best_frames.get(obj.object_id)
            if frame_data is None:
                continue
            detections, instance = frame_data
            if not detections.image_b64:
                continue
            try:
                if obj.segment_kind == "body":
                    result = self.gateway.run_service_job(
                        self.service_id,
                        "reconstruct_body",
                        {"image_base64": detections.image_b64},
                        timeout_sec=self._per_object_timeout_sec,
                    )
                else:
                    payload: dict[str, Any] = {"image_base64": detections.image_b64}
                    if instance.mask_rle:
                        payload["mask_rle"] = instance.mask_rle
                    else:
                        payload["bbox_normalized"] = _bbox_normalized_from_instance(instance, detections.image_b64)
                    result = self.gateway.run_service_job(
                        self.service_id,
                        "reconstruct_objects",
                        payload,
                        timeout_sec=self._per_object_timeout_sec,
                    )
                normalized = self._normalize_result(
                    object_id=obj.object_id,
                    segment_kind=obj.segment_kind,
                    frame_idx=detections.frame_idx,
                    raw=result if isinstance(result, dict) else {},
                )
                if obj.segment_kind == "body" and isinstance(result, dict):
                    _extract_body_camera_and_pose(normalized, result)
                out[obj.object_id] = normalized
            except Exception as exc:
                logger.error("[Zaiwu] SAM3D reconstruction failed for %s: %s", obj.object_id, exc)
        return out

    def _normalize_result(self, *, object_id: str, segment_kind: str, frame_idx: int, raw: dict[str, Any]) -> dict[str, Any]:
        files = raw.get("files") if isinstance(raw.get("files"), list) else []
        source_files: list[tuple[str, Path]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            file_id = str(item.get("file_id") or "").strip()
            if not file_id:
                continue
            local_path = self._download_file(file_id, item)
            if local_path is None:
                continue
            file_format = str(item.get("format") or local_path.suffix.lstrip("."))
            source_files.append((file_format, local_path))

        materialized_files: list[dict[str, str]] = []
        if self._materialization_root:
            object_root = (
                self._materialization_root
                / "intermediate"
                / f"frame_{int(frame_idx):06d}"
                / "objects"
                / _safe_name(object_id)
                / "assets"
            )
            object_root.mkdir(parents=True, exist_ok=True)
            for file_format, src in source_files:
                ext = src.suffix or (f".{file_format}" if file_format else ".bin")
                dst = _unique_path(object_root / f"object{ext}")
                _materialize_file(src, dst, self._materialization_mode)
                materialized_files.append({"format": file_format or ext.lstrip("."), "path": str(dst)})
        else:
            for file_format, src in source_files:
                materialized_files.append({"format": file_format or src.suffix.lstrip("."), "path": str(src)})

        chosen = ""
        if materialized_files:
            ply = next((item for item in materialized_files if item.get("format") == "ply"), None)
            chosen = str((ply or materialized_files[0]).get("path", ""))
        return {
            "instance_id": object_id,
            "segment_kind": segment_kind,
            "source": "zaiwu_sam3d",
            "request_id": str(raw.get("request_id", "")),
            "quality": float(raw.get("quality", 0.6)),
            "mesh_path": chosen,
            "files": materialized_files,
        }

    def _download_file(self, file_id: str, item: dict[str, Any]) -> Path | None:
        file_format = str(item.get("format") or "bin")
        ext = f".{file_format}" if file_format else ".bin"
        try:
            data = self.gateway.download_bytes(self.service_id, file_id)
        except Exception as exc:
            logger.warning("[Zaiwu] Failed to download %s from %s: %s", file_id, self.service_id, exc)
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()
        path = Path(tmp.name)
        path.write_bytes(data)
        return path


class ZaiwuWildGSAdapter:
    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        output_root: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.service_id = service_id
        self._output_root = Path(output_root).resolve() if output_root else None

    def run_slam(
        self,
        video_path: str | None = None,
        frames_dir: str | None = None,
        intrinsics: dict | None = None,
        fps_override: float | None = None,
        run_id: str | None = None,
        export_depth_every_frame: bool | None = None,
        depth_export_stride: int | None = None,
        pose_export_stride: int | None = None,
        extract_every_input_frame: bool | None = None,
        frame_stride: int | None = None,
    ) -> dict[str, Any]:
        if not video_path:
            raise ValueError("Zaiwu WildGS jobs currently require video_path input.")
        if frames_dir:
            raise ValueError("frames_dir is not supported by the current Zaiwu WildGS job interface.")

        video_file_id = self.gateway.upload_file(self.service_id, video_path)
        payload: dict[str, Any] = {"video_file_id": video_file_id}
        if intrinsics:
            payload["intrinsics_json"] = json.dumps(intrinsics)
        if fps_override is not None:
            payload["fps_override"] = float(fps_override)
        if run_id:
            payload["run_id"] = str(run_id)
        if export_depth_every_frame is not None:
            payload["export_depth_every_frame"] = bool(export_depth_every_frame)
        if depth_export_stride is not None:
            payload["depth_export_stride"] = int(depth_export_stride)
        if pose_export_stride is not None:
            payload["pose_export_stride"] = int(pose_export_stride)
        if extract_every_input_frame is not None:
            payload["extract_every_input_frame"] = bool(extract_every_input_frame)
        if frame_stride is not None:
            payload["frame_stride"] = int(frame_stride)

        result = self.gateway.run_service_job(
            self.service_id,
            "wildgs_run_slam",
            payload,
            timeout_sec=max(7200.0, self.gateway.job_timeout_sec),
        )
        self._write_run_result_debug(result)

        camera_poses_file_id = str(result.get("camera_poses_file_id") or "").strip()
        if not camera_poses_file_id:
            raise RuntimeError(
                "WildGS-SLAM result missing required camera_poses_file_id: "
                f"{self._result_debug_summary(result)}"
            )
        depth_maps_file_id = str(result.get("depth_maps_file_id") or "").strip()
        if not depth_maps_file_id:
            raise RuntimeError(
                "WildGS-SLAM result missing required depth_maps_file_id: "
                f"{self._result_debug_summary(result)}"
            )

        camera_poses_path = self._download_required_file(camera_poses_file_id, "camera_poses.jsonl")
        static_map_dir = self._download_static_map(result.get("static_map_file_id", ""))
        dynamic_prior_dir = self._download_and_unpack(result.get("dynamic_prior_file_id", ""), "dynamic_prior")
        depth_maps_dir = self._download_required_and_unpack(depth_maps_file_id, "depth_maps")
        return {
            "camera_poses_path": str(camera_poses_path) if camera_poses_path else None,
            "static_map_dir": str(static_map_dir) if static_map_dir else None,
            "dynamic_prior_dir": str(dynamic_prior_dir) if dynamic_prior_dir else None,
            "depth_maps_dir": str(depth_maps_dir) if depth_maps_dir else None,
            "num_frames": result.get("num_frames", 0),
            "slam_quality": result.get("slam_quality", 0.0),
            "camera_poses_file_id": result.get("camera_poses_file_id", ""),
            "static_map_file_id": result.get("static_map_file_id", ""),
            "dynamic_prior_file_id": result.get("dynamic_prior_file_id", ""),
            "depth_maps_file_id": result.get("depth_maps_file_id", ""),
        }

    def _write_run_result_debug(self, result: dict[str, Any]) -> None:
        if not self._output_root:
            return
        dst = self._output_root / "exports" / "wildgs_slam_result.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    def _result_debug_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "camera_poses_file_id",
            "depth_maps_file_id",
            "static_map_file_id",
            "dynamic_prior_file_id",
            "output_root",
            "run_dir",
            "job_dir",
            "num_frames",
            "slam_quality",
        )
        summary = {field: result.get(field, "") for field in fields if field in result}
        summary["keys"] = sorted(result.keys())
        return summary

    def reconstruct_background_mesh(
        self,
        static_map_file_id: str,
        poisson_depth: int = 7,
        opacity_threshold: float = 0.3,
    ) -> dict[str, Any]:
        _ = (static_map_file_id, poisson_depth, opacity_threshold)
        return {}

    def _download_file(self, file_id: str, filename: str) -> Path | None:
        if not file_id:
            return None
        try:
            data = self.gateway.download_bytes(self.service_id, file_id)
        except Exception as exc:
            logger.warning("[Zaiwu] Failed to download %s from %s: %s", file_id, self.service_id, exc)
            return None

        if self._output_root:
            dst = self._output_root / "exports" / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = _unique_path(dst)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
            tmp.close()
            dst = Path(tmp.name)
        dst.write_bytes(data)
        return dst

    def _download_required_file(self, file_id: str, filename: str) -> Path:
        try:
            data = self.gateway.download_bytes(self.service_id, file_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download required WildGS artifact {file_id} as {filename}"
            ) from exc

        if self._output_root:
            dst = self._output_root / "exports" / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = _unique_path(dst)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
            tmp.close()
            dst = Path(tmp.name)
        dst.write_bytes(data)
        return dst

    def _download_and_unpack(self, file_id: str, dir_name: str) -> Path | None:
        tar_path = self._download_file(file_id, f"{dir_name}.tar.gz")
        if not tar_path:
            return None

        if self._output_root:
            unpack_dir = self._output_root / "exports" / dir_name
        else:
            unpack_dir = Path(tempfile.mkdtemp(suffix=f"_{dir_name}"))
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as handle:
            handle.extractall(str(unpack_dir))
        entries = list(unpack_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return unpack_dir

    def _download_required_and_unpack(self, file_id: str, dir_name: str) -> Path:
        tar_path = self._download_required_file(file_id, f"{dir_name}.tar.gz")

        if self._output_root:
            unpack_dir = self._output_root / "exports" / dir_name
        else:
            unpack_dir = Path(tempfile.mkdtemp(suffix=f"_{dir_name}"))
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as handle:
            handle.extractall(str(unpack_dir))
        entries = list(unpack_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return unpack_dir

    def _download_static_map(self, file_id: str) -> Path | None:
        ply_path = self._download_file(file_id, "final_gs.ply")
        if not ply_path:
            return None
        if self._output_root:
            map_dir = self._output_root / "exports" / "static_map"
        else:
            map_dir = Path(tempfile.mkdtemp(suffix="_static_map"))
        map_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ply_path, map_dir / "final_gs.ply")
        return map_dir


class ZaiwuVisualPoseTracker:
    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        timeout_sec: float = 30.0,
    ) -> None:
        self.gateway = gateway
        self.service_id = service_id
        self.timeout_sec = float(timeout_sec)
        self._mesh_upload_cache: dict[tuple[str, int, int], str] = {}

    def refine_pose(self, payload: dict[str, Any]):  # noqa: ANN201
        from guanwu.video.features.spatial.visual_pose_tracking import VisualPoseResult

        prepared = self._prepare_payload(dict(payload))
        try:
            data = self.gateway.run_service_job(
                self.service_id,
                "gotrack_refine_pose",
                prepared,
                timeout_sec=self.timeout_sec,
            )
        except Exception as exc:
            logger.warning(
                "[VisualPose] Zaiwu job failed for %s@%s via %s: %s",
                payload.get("object_id"),
                payload.get("frame_idx"),
                self.service_id,
                exc,
            )
            return None
        if not isinstance(data, dict):
            return None
        translation = data.get("translation_world", data.get("centroid_world"))
        orientation = data.get("orientation_quat")
        if not (isinstance(translation, list) and len(translation) >= 3 and isinstance(orientation, list) and len(orientation) >= 4):
            return None
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"translation_world", "centroid_world", "orientation_quat", "rotation_matrix", "score", "accepted"}
        }
        return VisualPoseResult(
            translation_world=[float(v) for v in translation[:3]],
            orientation_quat=[float(v) for v in orientation[:4]],
            score=float(data.get("score", 1.0) or 0.0),
            accepted=bool(data.get("accepted", True)),
            metadata=metadata,
        )

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        mesh_path = payload.get("mesh_path")
        if not isinstance(mesh_path, str) or not mesh_path.strip():
            return payload

        mesh_file = Path(mesh_path).expanduser()
        if not mesh_file.is_file():
            return payload

        mesh_file = mesh_file.resolve()
        stat = mesh_file.stat()
        cache_key = (str(mesh_file), int(stat.st_mtime_ns), int(stat.st_size))
        mesh_file_id = self._mesh_upload_cache.get(cache_key)
        if mesh_file_id is None:
            mesh_file_id = self.gateway.upload_file(self.service_id, mesh_file)
            self._mesh_upload_cache[cache_key] = mesh_file_id
        payload["mesh_file_id"] = mesh_file_id
        payload.pop("mesh_path", None)
        return payload


def build_zaiwu_object_detector(settings: SPWMSettings, *, video_source: str | None) -> Any:
    gateway = build_zaiwu_gateway_client(settings)
    backend = str(settings.zaiwu.object_detection_backend or "seg2track_sam2").strip().lower()

    if backend == "seg2track_sam2":
        cfg = settings.zaiwu.seg2track_sam2
        return ZaiwuSeg2TrackDetector(
            gateway,
            service_id=settings.zaiwu.seg2track_sam2_service,
            video_source=video_source,
            detect_interval=cfg.detect_interval,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
        )

    if backend == "grounding_dino_sam2":
        cfg = settings.zaiwu.grounded_sam2
        return ZaiwuGroundedSAM2Detector(
            gateway,
            service_id=settings.zaiwu.grounded_sam2_service,
            video_source=video_source,
            step=cfg.step,
            iou_threshold=cfg.iou_threshold,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
        )

    return ZaiwuSAM3Detector(
        gateway,
        service_id=settings.zaiwu.sam3_service,
        video_source=video_source,
    )


def build_zaiwu_sam3d_adapter(
    settings: SPWMSettings,
    *,
    materialization_root: str | None = None,
    materialization_mode: str = "copy",
    per_object_timeout_sec: float = 180.0,
) -> Any:
    gateway = build_zaiwu_gateway_client(settings)
    return ZaiwuSAM3DAdapter(
        gateway,
        service_id=settings.zaiwu.sam3d_service,
        materialization_root=materialization_root,
        materialization_mode=materialization_mode,
        per_object_timeout_sec=per_object_timeout_sec,
    )


def build_zaiwu_wildgs_adapter(settings: SPWMSettings, *, output_root: str | None = None) -> Any:
    gateway = build_zaiwu_gateway_client(settings)
    return ZaiwuWildGSAdapter(
        gateway,
        service_id=settings.zaiwu.wildgs_slam_service,
        output_root=output_root,
    )


def build_zaiwu_visual_pose_tracker(settings: SPWMSettings, *, timeout_sec: float | None = None) -> ZaiwuVisualPoseTracker | None:
    if not settings.zaiwu.enabled:
        return None
    try:
        gateway = build_zaiwu_gateway_client(settings)
        return ZaiwuVisualPoseTracker(
            gateway,
            service_id=settings.zaiwu.gotrack_service,
            timeout_sec=float(timeout_sec or settings.pit.visual_pose_timeout_sec or 30.0),
        )
    except Exception as exc:
        logger.warning("[Zaiwu] Failed to build GoTrack visual pose tracker: %s", exc)
        return None


def resolve_zaiwu_visual_pose_url(settings: SPWMSettings) -> str | None:
    if not settings.zaiwu.enabled:
        return None
    try:
        gateway = build_zaiwu_gateway_client(settings)
        return gateway.service_sse_url(settings.zaiwu.gotrack_service)
    except Exception as exc:
        logger.warning("[Zaiwu] Failed to resolve GoTrack service URL: %s", exc)
        return None


class ZaiwuDepthProvider:
    """Depth provider backed by Zaiwu gateway jobs and artifact downloads."""

    def __init__(
        self,
        gateway: ZaiwuGatewayClient,
        *,
        service_id: str,
        video_path: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.service_id = service_id
        self.video_path = video_path
        self._depth_cache: Any = None
        self._is_metric: bool | None = None

    @property
    def is_metric(self) -> bool:
        if self._is_metric is None:
            self._query_depth_info()
        return bool(self._is_metric)

    def prefetch(self) -> None:
        if self._depth_cache is None and self.video_path:
            self._load_from_video(self.video_path)

    def _query_depth_info(self) -> None:
        from guanwu.video.clients.mcp_backend import sync_call_mcp

        try:
            url = self.gateway.service_sse_url(self.service_id)
            info = sync_call_mcp(url, "depth_info", {})
            model_name = str(info.get("model_name", "")).upper()
            self._is_metric = "METRIC" in model_name
        except Exception as exc:
            logger.warning("[ZaiwuDepthProvider] depth_info query failed, assuming relative depth: %s", exc)
            self._is_metric = False

    def _load_from_video(self, video_path: str) -> None:
        import io
        import numpy as np

        if self._is_metric is None:
            self._query_depth_info()

        video_file_id = self.gateway.upload_file(self.service_id, video_path)
        result = self.gateway.run_service_job(
            self.service_id,
            "estimate_from_video",
            payload={"video_file_id": video_file_id, "sample_every_n": 1},
        )
        output_file_id = str(result.get("output_file_id") or "")
        if not output_file_id:
            raise RuntimeError(f"Depth job for {video_path} returned no output_file_id: {result}")
        data = self.gateway.download_bytes(self.service_id, output_file_id)
        self._depth_cache = np.load(io.BytesIO(data))
        logger.info(
            "[ZaiwuDepthProvider] Depth cache ready from %s: shape=%s metric=%s",
            self.service_id,
            getattr(self._depth_cache, "shape", None),
            self._is_metric,
        )

    def depth_values(
        self,
        image_b64: str,
        samples_uv: list[tuple[float, float]],
        frame_idx: int = 0,
    ) -> list[float | None]:
        if self._depth_cache is None and self.video_path:
            self._load_from_video(self.video_path)

        if self._depth_cache is None:
            return [None] * len(samples_uv)

        depth = self._depth_cache[min(frame_idx, len(self._depth_cache) - 1)]
        h, w = depth.shape
        values: list[float] = []
        if not self.is_metric:
            return [None] * len(samples_uv)
        for u, v in samples_uv:
            x = int(max(0, min(w - 1, (u / 640.0) * w)))
            y = int(max(0, min(h - 1, (v / 480.0) * h)))
            values.append(max(0.01, float(depth[y, x])))
        return values


def _normalize_frame_payload(payload: Any):
    from guanwu.video.core.types import DetectedInstance, FrameDetections

    raw = payload if isinstance(payload, dict) else {}
    frame_idx = int(raw.get("frame_idx", 0) or 0)
    timestamp = float(raw.get("timestamp", 0.0) or 0.0)
    image_b64 = raw.get("image_b64")
    instances: list[DetectedInstance] = []
    raw_instances = raw.get("instances", [])
    if isinstance(raw_instances, list):
        for index, item in enumerate(raw_instances, start=1):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id") or item.get("track_id") or f"obj_{index:06d}")
            label = str(item.get("concept_label") or item.get("label") or "object")
            segment_kind = str(item.get("segment_kind") or ("body" if any(term in label.lower() for term in ("person", "human", "body")) else "object"))
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                bbox = [0.0, 0.0, 0.0, 0.0]
            instances.append(
                DetectedInstance(
                    mask_ref=str(item.get("mask_ref") or f"mask://frame_{frame_idx:05d}/{object_id}"),
                    bbox=[float(v) for v in bbox[:4]],
                    object_id=object_id,
                    concept_label=label,
                    segment_kind=segment_kind,
                    score=float(item.get("score", 0.0) or 0.0),
                    mask_rle=item.get("mask_rle"),
                )
            )
    return FrameDetections(
        frame_idx=frame_idx,
        timestamp=timestamp,
        instances=instances,
        image_b64=image_b64 if isinstance(image_b64, str) and image_b64 else None,
    )


def _bbox_normalized_from_instance(instance: "DetectedInstance", image_b64: str | None) -> list[float]:
    bbox = list(instance.bbox) if instance.bbox else []
    if len(bbox) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    if not image_b64:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        width, height = image.size
    except Exception:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    return [
        max(0.0, min(1.0, float(bbox[0]) / width)),
        max(0.0, min(1.0, float(bbox[1]) / height)),
        max(0.0, min(1.0, float(bbox[2]) / width)),
        max(0.0, min(1.0, float(bbox[3]) / height)),
    ]


def _extract_body_camera_and_pose(result: dict[str, Any], raw: dict[str, Any]) -> None:
    persons = raw if isinstance(raw, list) else [raw]
    if not persons:
        return
    person = persons[0] if isinstance(persons[0], dict) else {}
    focal = person.get("focal_length")
    cam_t = person.get("pred_cam_t")
    global_rot = person.get("global_rot")
    if focal is not None:
        focal_value = focal if isinstance(focal, (int, float)) else (focal[0] if isinstance(focal, list) and focal else None)
        if focal_value is not None:
            result["camera_focal_length"] = float(focal_value)
    if isinstance(cam_t, list) and len(cam_t) >= 3:
        result["camera_translation"] = [float(cam_t[0]), float(cam_t[1]), float(cam_t[2])]
    if global_rot is not None:
        result["body_global_rotation"] = global_rot


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
