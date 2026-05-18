from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client

from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import FrameDetections
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)


def _is_connection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "connecterror" in msg
        or "all connection attempts failed" in msg
        or "connection refused" in msg
        or "name or service not known" in msg
    )


async def async_call_mcp(
    url: str,
    tool_name: str,
    args: dict[str, Any],
    sse_read_timeout: int = 600,
    call_timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Execute a single MCP tool via SSE asynchronously."""
    import json as _json

    async with sse_client(url, sse_read_timeout=sse_read_timeout) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            if call_timeout_sec is not None:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=args),
                    timeout=float(call_timeout_sec),
                )
            else:
                result = await session.call_tool(tool_name, arguments=args)
            if result.isError:
                raise RuntimeError(f"MCP tool error: {result.content}")

            text = result.content[0].text if result.content else "{}"
            try:
                return _json.loads(text)
            except _json.JSONDecodeError:
                return {"result": text}


def sync_call_mcp(
    url: str,
    tool_name: str,
    args: dict[str, Any],
    max_retries: int = 3,
    sse_read_timeout: int = 600,
    call_timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Execute a single MCP tool via SSE synchronously.

    Works both from a plain thread and from within a running asyncio
    event loop (e.g. when called from a FastAPI async endpoint).
    Retries on transient connection errors.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        async_call_mcp(
                            url,
                            tool_name,
                            args,
                            sse_read_timeout=sse_read_timeout,
                            call_timeout_sec=call_timeout_sec,
                        ),
                    )
                    return future.result()
            else:
                return asyncio.run(
                    async_call_mcp(
                        url,
                        tool_name,
                        args,
                        sse_read_timeout=sse_read_timeout,
                        call_timeout_sec=call_timeout_sec,
                    )
                )
        except Exception as exc:
            import traceback
            import sys

            last_exc = exc
            is_conn_err = _is_connection_error(exc)
            if sys.version_info >= (3, 11) and isinstance(exc, BaseExceptionGroup):
                is_conn_err = any(_is_connection_error(sub) for sub in exc.exceptions)

            if is_conn_err and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"MCP connection error (attempt {attempt + 1}/{max_retries}, tool={tool_name}): "
                    f"{exc}. Retrying in {wait}s..."
                )
                import time
                time.sleep(wait)
                continue

            # Non-connection error or final attempt — raise
            if sys.version_info >= (3, 11) and isinstance(exc, BaseExceptionGroup):
                logger.error("MCP tool call failed with TaskGroup errors:")
                for sub_exc in exc.exceptions:
                    logger.error(f"  - {type(sub_exc).__name__}: {sub_exc}")
                    traceback.print_exception(type(sub_exc), sub_exc, sub_exc.__traceback__)
                raise RuntimeError(f"MCP Connection/TaskGroup Error: {exc.exceptions}") from exc

            if is_conn_err:
                logger.error(f"MCP connection failed after {max_retries} attempts (url={url}, tool={tool_name}): {exc}")
            else:
                logger.error(f"MCP tool call failed (url={url}, tool={tool_name}): {exc}")
                traceback.print_exc()
            raise

    raise last_exc  # type: ignore[misc]


class MCPVideoObjectDetector:
    def __init__(
        self,
        url: str,
        video_source: str | None = None,
        max_frames: int | None = None,
        *,
        frame_tool: str = "video_parse_frame",
        video_tool: str = "video_parse_video",
    ) -> None:
        self.url = url
        self._video_source = (video_source or "").strip() or None
        self._max_frames = max_frames
        self._frame_tool = frame_tool
        self._video_tool = video_tool
        self._prompts = []
        self._cap = None
        self._first_frame_b64: str | None = None
        self._video_batches: dict[int, FrameDetections] = {}
        self._video_prefetch_attempted = False

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        self._prompts = prompts

    def get_object_detection_prompts(self) -> list[str]:
        return list(self._prompts)

    def detector_status(self) -> dict:
        return {"backend": "mcp", "url": self.url}

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

    def _read_b64_frame_at(self, frame_idx: int) -> str | None:
        if not self._video_source or self._video_source.isdigit():
            return None
        try:
            import cv2
            import base64
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
            b64_str = base64.b64encode(buffer).decode("ascii")
            if int(frame_idx) == 1 and self._first_frame_b64 is None:
                self._first_frame_b64 = b64_str
            return b64_str
        finally:
            cap.release()

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float) -> FrameDetections:
        cached = self._video_batches.get(frame_idx)
        if cached is not None:
            if not cached.image_b64:
                frame_b64 = self._read_b64_frame_at(frame_idx)
                if frame_b64:
                    cached = cached.model_copy(update={"image_b64": frame_b64})
                    self._video_batches[frame_idx] = cached
            return cached

        image_b64 = self._read_b64_frame()
        logger.info(f"[MCP] Calling video-parsing MCP for frame {frame_idx}...")

        args = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "image_base64": image_b64 or "",
            "prompts": self._prompts,
        }
        try:
            res = sync_call_mcp(self.url, self._frame_tool, args)
            return FrameDetections.model_validate(res)
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Cannot connect to unified MCP at {self.url}. "
                    "Please start/restart the embody-unified SSE server."
                ) from exc
            raise

    def prefetch_video(self) -> None:
        """Parse the full video once and cache frame batches if video path is configured."""
        self._ensure_video_prefetch()

    def _ensure_video_prefetch(self) -> None:
        if self._video_prefetch_attempted:
            return
        self._video_prefetch_attempted = True

        if not self._video_source or self._video_source.isdigit():
            return

        logger.info("[MCP] Calling video-parsing MCP parse_video for full sequence...")

        # Upload the local video to the remote file service first, then pass the file_id.
        file_base_url = self.url.rstrip("/").removesuffix("/sse")
        video_path = Path(self._video_source).expanduser().resolve()
        video_file_id: str | None = None
        try:
            import httpx
            upload_url = f"{file_base_url}/upload?filename={video_path.name}"
            with httpx.Client(verify=False, timeout=300.0) as client:
                resp = client.post(upload_url, content=video_path.read_bytes())
                resp.raise_for_status()
                video_file_id = resp.json()["file_id"]
            logger.info("[MCP] Uploaded video to file service, file_id=%s", video_file_id)
        except Exception as exc:
            logger.warning("[MCP] Failed to upload video for parse_video, will fall back to frame-by-frame: %s", exc)
            return

        args: dict = {
            "video_file_id": video_file_id,
            "prompts": self._prompts,
        }
        if self._max_frames is not None and int(self._max_frames) > 0:
            args["max_frames"] = int(self._max_frames)
        try:
            res = sync_call_mcp(self.url, self._video_tool, args)
            frames = res.get("frames", []) if isinstance(res, dict) else []
            loaded = 0
            for item in frames:
                detections = FrameDetections.model_validate(item)
                self._video_batches[detections.frame_idx] = detections
                loaded += 1
            logger.info(f"[MCP] video_parse_video loaded {loaded} frame batches.")
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Cannot connect to unified MCP at {self.url}. "
                    "Please start/restart the embody-unified SSE server."
                ) from exc
            # Backward-compatible fallback when server does not expose parse_video.
            logger.warning(f"[MCP] video_parse_video unavailable/failed, falling back to video_parse_frame: {exc}")

    def get_first_frame_b64(self) -> str | None:
        if self._first_frame_b64 is None:
            self._read_b64_frame()
        return self._first_frame_b64


class MCPSAM3DAdapter:
    def __init__(
        self,
        url: str,
        materialization_root: str | None = None,
        materialization_mode: str = "copy",
        per_object_timeout_sec: float = 180.0,
        *,
        body_tool: str = "sam3d_reconstruct_body",
        objects_tool: str = "sam3d_reconstruct_objects",
    ) -> None:
        self.url = url
        self._materialization_root = Path(materialization_root).resolve() if materialization_root else None
        self._materialization_mode = materialization_mode
        self._per_object_timeout_sec = float(per_object_timeout_sec)
        self._body_tool = body_tool
        self._objects_tool = objects_tool
        # File HTTP is served on the same host as the MCP SSE server (/download/<file_id>)
        self._file_base_url = url.rstrip("/").removesuffix("/sse")

    def reconstruct_object_meshes(
        self,
        best_frames: dict[str, tuple["FrameDetections", "DetectedInstance"]],
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        """Reconstruct meshes using the best frame per object.

        Args:
            best_frames: mapping from object_id → (FrameDetections, DetectedInstance)
                         for the frame with highest visibility score.
            objects: ObjectNode list (for segment_kind and geometry fallback).
        """
        out = {}
        for obj in objects:
            object_id = obj.object_id
            segment_kind = obj.segment_kind

            frame_data = best_frames.get(object_id)
            if frame_data is None:
                logger.warning(f"[MCP] No best frame found for {object_id}, skipping.")
                continue
            detections, instance = frame_data

            if not detections.image_b64:
                logger.warning(f"[MCP] No image in best frame for {object_id}, skipping.")
                continue

            logger.info(
                f"[MCP] Calling SAM3D reconstruction for {object_id} ({segment_kind}) "
                f"using frame {detections.frame_idx}..."
            )
            try:
                if segment_kind == "body":
                    tool = self._body_tool
                    args = {"image_base64": detections.image_b64}
                else:
                    tool = self._objects_tool
                    args: dict = {"image_base64": detections.image_b64}
                    if instance.mask_rle:
                        # Decode COCO RLE → 2-D float mask expected by SAM3D
                        try:
                            import json as _json
                            import numpy as _np
                            from pycocotools import mask as _coco_mask
                            rle_obj = _json.loads(instance.mask_rle) if isinstance(instance.mask_rle, str) else instance.mask_rle
                            decoded = _coco_mask.decode(rle_obj)  # uint8 H×W
                            args["mask"] = decoded.astype(float).tolist()
                        except Exception as _rle_err:
                            logger.warning(f"[MCP] RLE decode failed for {object_id}, falling back to bbox: {_rle_err}")
                            bbox_norm = self._bbox_normalized_from_instance(instance, detections)
                            args["bbox_normalized"] = bbox_norm
                    else:
                        bbox_norm = self._bbox_normalized_from_instance(instance, detections)
                        bbox_w = bbox_norm[2] - bbox_norm[0]
                        bbox_h = bbox_norm[3] - bbox_norm[1]
                        if bbox_w * bbox_h < 1e-4:
                            logger.warning(
                                f"[MCP] Skipping {object_id}: bbox too small "
                                f"({bbox_w:.4f}x{bbox_h:.4f})"
                            )
                            continue
                        args["bbox_normalized"] = bbox_norm

                res = sync_call_mcp(
                    self.url,
                    tool,
                    args,
                    sse_read_timeout=max(600, int(self._per_object_timeout_sec) + 30),
                    call_timeout_sec=self._per_object_timeout_sec,
                )
                normalized = self._normalize_result(
                    object_id=object_id,
                    segment_kind=segment_kind,
                    frame_idx=detections.frame_idx,
                    raw=res if isinstance(res, dict) else {},
                )
                # For body reconstructions, extract camera and pose info
                if segment_kind == "body" and isinstance(res, dict):
                    self._extract_body_camera_and_pose(normalized, res)
                out[object_id] = normalized
            except Exception as e:
                logger.error(f"[MCP] Reconstruction failed for {object_id}: {e}")
        return out

    @staticmethod
    def _extract_body_camera_and_pose(result: dict, raw: dict) -> None:
        """Extract camera intrinsics and body rotation from SAM3D body output."""
        # SAM3D body returns a list of per-person results; take the first.
        persons = raw if isinstance(raw, list) else [raw]
        if not persons:
            return
        person = persons[0] if isinstance(persons[0], dict) else {}

        focal = person.get("focal_length")
        cam_t = person.get("pred_cam_t")
        global_rot = person.get("global_rot")

        if focal is not None:
            # focal_length is a scalar or [fx] or [fx, fy]
            fl = focal if isinstance(focal, (int, float)) else (focal[0] if isinstance(focal, list) and focal else None)
            if fl is not None:
                result["camera_focal_length"] = float(fl)

        if cam_t is not None and isinstance(cam_t, list) and len(cam_t) >= 3:
            result["camera_translation"] = [float(cam_t[0]), float(cam_t[1]), float(cam_t[2])]

        if global_rot is not None:
            # global_rot can be a 3x3 rotation matrix or axis-angle; store as-is
            result["body_global_rotation"] = global_rot

    @staticmethod
    def _bbox_normalized_from_instance(instance: "DetectedInstance", detections: "FrameDetections") -> list[float]:
        """Return [x1, y1, x2, y2] normalised to [0, 1] from a DetectedInstance bbox."""
        import base64
        from io import BytesIO
        from PIL import Image

        bbox = list(instance.bbox) if instance.bbox else []
        if len(bbox) < 4:
            return [0.0, 0.0, 1.0, 1.0]

        img_data = detections.image_b64 or ""
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        try:
            img = Image.open(BytesIO(base64.b64decode(img_data)))
            w, h = img.size
        except Exception:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

        return [
            max(0.0, min(1.0, float(bbox[0]) / w)),
            max(0.0, min(1.0, float(bbox[1]) / h)),
            max(0.0, min(1.0, float(bbox[2]) / w)),
            max(0.0, min(1.0, float(bbox[3]) / h)),
        ]

    def _normalize_result(self, object_id: str, segment_kind: str, frame_idx: int, raw: dict[str, Any]) -> dict[str, Any]:
        files = raw.get("files") if isinstance(raw.get("files"), list) else []
        source_files: list[tuple[str, Path]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            # Unified MCP returns file_id; fallback to path for direct-server mode.
            file_id = str(item.get("file_id", "")).strip()
            if file_id:
                local_path = self._download_file(file_id, item)
                if local_path is not None:
                    fmt = str(item.get("format", local_path.suffix.lstrip(".")))
                    source_files.append((fmt, local_path))
                continue
            p = str(item.get("path", "")).strip()
            if not p:
                continue
            path = Path(p).expanduser()
            if not path.exists():
                continue
            fmt = str(item.get("format", path.suffix.lstrip(".")))
            source_files.append((fmt, path))
        mesh_candidate = str(raw.get("mesh_path", "")).strip()
        if mesh_candidate:
            p = Path(mesh_candidate).expanduser()
            if p.exists():
                source_files.append((p.suffix.lstrip("."), p))

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
            for fmt, src in source_files:
                ext = src.suffix or (f".{fmt}" if fmt else ".bin")
                dst = _unique_path(object_root / f"object{ext}")
                _materialize_file(src, dst, self._materialization_mode)
                materialized_files.append({"format": fmt or ext.lstrip("."), "path": str(dst)})
        else:
            for fmt, src in source_files:
                materialized_files.append({"format": fmt or src.suffix.lstrip("."), "path": str(src)})

        chosen = ""
        if materialized_files:
            ply = next((f for f in materialized_files if f.get("format") == "ply"), None)
            chosen = str((ply or materialized_files[0]).get("path", ""))
        if not chosen:
            logger.warning(
                "[MCP] SAM3D normalize: no mesh materialized for %s. "
                "raw keys=%s files_in_raw=%s source_files=%d",
                object_id, list(raw.keys()), raw.get("files"), len(source_files),
            )
        return {
            "instance_id": object_id,
            "segment_kind": segment_kind,
            "source": "mcp_sam3d",
            "request_id": str(raw.get("request_id", "")),
            "quality": float(raw.get("quality", 0.6)),
            "mesh_path": chosen,
            "files": materialized_files,
        }

    def _download_file(self, file_id: str, item: dict) -> Path | None:
        """Download a file by file_id from the MCP file HTTP service to a temp location."""
        import tempfile
        fmt = str(item.get("format", "bin"))
        ext = f".{fmt}" if fmt else ".bin"
        url = f"{self._file_base_url}/download/{file_id}"
        try:
            import httpx
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            with httpx.Client(verify=False, timeout=60.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                Path(tmp.name).write_bytes(resp.content)
            return Path(tmp.name)
        except Exception as exc:
            logger.warning(f"[MCP] Failed to download file_id={file_id} from {url}: {exc}")
            return None


class MCPDepthProvider:
    """Depth provider that calls depth_estimate_from_video via the unified MCP.

    On first use, queries depth_info to detect whether the remote model outputs
    metric (absolute) or relative depth.  Then uploads the video file to the
    file service, runs depth estimation for all frames at once, downloads and
    caches the resulting [N, H, W] array.

    When the remote model is metric (e.g. DA3METRIC), raw depth values are
    returned directly in metres.  Relative models return unknown depth values
    instead of fabricating pseudo-metric scene units.
    """

    def __init__(self, mcp_unified_url: str, video_path: str | None = None) -> None:
        self.url = mcp_unified_url
        self._file_base_url = mcp_unified_url.rstrip("/").removesuffix("/sse")
        self._video_path = video_path
        self._depth_cache: Any = None  # np.ndarray [N, H, W] after first load
        self._is_metric: bool | None = None  # None = not yet queried

    @property
    def is_metric(self) -> bool:
        if self._is_metric is None:
            self._query_depth_info()
        return bool(self._is_metric)

    def prefetch(self) -> None:
        """Upload video and cache all depth maps upfront. No-op if already done."""
        if self._depth_cache is None and self._video_path:
            self._load_from_video(self._video_path)

    def _query_depth_info(self) -> None:
        try:
            info = sync_call_mcp(self.url, "depth_info", {})
            model_name = str(info.get("model_name", "")).upper()
            self._is_metric = "METRIC" in model_name
            logger.info(
                "[MCPDepthProvider] Remote model: %s (metric=%s)",
                info.get("model_name"), self._is_metric,
            )
        except Exception as exc:
            logger.warning("[MCPDepthProvider] depth_info query failed, assuming relative depth: %s", exc)
            self._is_metric = False

    def _load_from_video(self, video_path: str) -> None:
        import numpy as np
        import tempfile
        import httpx

        # Ensure we know whether the model is metric before caching
        if self._is_metric is None:
            self._query_depth_info()

        vp = Path(video_path).expanduser().resolve()
        logger.info("[MCPDepthProvider] Uploading video %s …", vp.name)
        upload_url = f"{self._file_base_url}/upload?filename={vp.name}"
        with httpx.Client(verify=False, timeout=300.0) as client:
            resp = client.post(upload_url, content=vp.read_bytes())
            resp.raise_for_status()
            video_file_id = resp.json()["file_id"]

        res = sync_call_mcp(self.url, "depth_estimate_from_video", {"video_file_id": video_file_id})
        output_file_id = res.get("output_file_id", "")
        if not output_file_id:
            logger.warning("[MCPDepthProvider] depth_estimate_from_video returned no output_file_id")
            return

        download_url = f"{self._file_base_url}/download/{output_file_id}"
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        tmp.close()
        try:
            with httpx.Client(verify=False, timeout=60.0) as client:
                resp = client.get(download_url)
                resp.raise_for_status()
                Path(tmp.name).write_bytes(resp.content)
            self._depth_cache = np.load(tmp.name)  # [N, H, W]
            logger.info("[MCPDepthProvider] Depth cache ready: shape=%s, metric=%s", self._depth_cache.shape, self._is_metric)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def depth_values(
        self,
        image_b64: str,
        samples_uv: list[tuple[float, float]],
        frame_idx: int = 0,
    ) -> list[float | None]:
        if self._depth_cache is None and self._video_path:
            self._load_from_video(self._video_path)

        if self._depth_cache is None:
            return [None] * len(samples_uv)

        n = len(self._depth_cache)
        depth = self._depth_cache[min(frame_idx, n - 1)]  # [H, W]
        h, w = depth.shape

        values: list[float] = []
        if not self._is_metric:
            return [None] * len(samples_uv)
        # Metric model: depth values are already in metres — use directly.
        for u, v in samples_uv:
            x = int(max(0, min(w - 1, (u / 640.0) * w)))
            y = int(max(0, min(h - 1, (v / 480.0) * h)))
            values.append(max(0.01, float(depth[y, x])))
        return values


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


class MCPGroundedSAM2Adapter:
    """Perception adapter using Grounded SAM 2 continuous-ID MCP service.

    Same interface as ``MCPVideoObjectDetector`` — provides ``detect_objects_in_frame``,
    ``prefetch_video``, ``set_object_detection_prompts``, and ``get_first_frame_b64``.
    """

    def __init__(
        self,
        url: str,
        video_source: str | None = None,
        max_frames: int | None = None,
        step: int = 20,
        iou_threshold: float = 0.8,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        self.url = url
        self._video_source = (video_source or "").strip() or None
        self._max_frames = max_frames
        self._step = step
        self._iou_threshold = iou_threshold
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._prompts: list[str] = []
        self._text_prompt: str = ""
        self._cap = None
        self._first_frame_b64: str | None = None
        self._video_batches: dict[int, FrameDetections] = {}
        self._video_prefetch_attempted = False

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        self._prompts = prompts
        # Grounding DINO expects dot-separated prompt: "cup. bottle. person."
        self._text_prompt = ". ".join(p.strip() for p in prompts if p.strip())
        if self._text_prompt and not self._text_prompt.endswith("."):
            self._text_prompt += "."

    def detector_status(self) -> dict:
        return {"backend": "mcp_grounded_sam2", "url": self.url}

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

    def _read_b64_frame_at(self, frame_idx: int) -> str | None:
        if not self._video_source or self._video_source.isdigit():
            return None
        try:
            import cv2
            import base64
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
            b64_str = base64.b64encode(buffer).decode("ascii")
            if int(frame_idx) == 1 and self._first_frame_b64 is None:
                self._first_frame_b64 = b64_str
            return b64_str
        finally:
            cap.release()

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float) -> FrameDetections:
        cached = self._video_batches.get(frame_idx)
        if cached is not None:
            if not cached.image_b64:
                frame_b64 = self._read_b64_frame_at(frame_idx)
                if frame_b64:
                    cached = cached.model_copy(update={"image_b64": frame_b64})
                    self._video_batches[frame_idx] = cached
            return cached

        # Fallback to single-frame mode
        image_b64 = self._read_b64_frame()
        if not image_b64:
            return FrameDetections(frame_idx=frame_idx, timestamp=timestamp, instances=[])

        logger.info("[MCP] Calling gsam2_parse_frame for frame %d...", frame_idx)
        args = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "image_base64": image_b64,
            "text_prompt": self._text_prompt or "object.",
        }
        try:
            res = sync_call_mcp(self.url, "gsam2_parse_frame", args)
            return FrameDetections.model_validate(res)
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Cannot connect to unified MCP at {self.url}. "
                    "Please start/restart the embody-unified SSE server."
                ) from exc
            raise

    def prefetch_video(self) -> None:
        """Parse the full video once using gsam2_parse_video and cache frame batches."""
        self._ensure_video_prefetch()

    def _ensure_video_prefetch(self) -> None:
        if self._video_prefetch_attempted:
            return
        self._video_prefetch_attempted = True

        if not self._video_source or self._video_source.isdigit():
            return

        logger.info("[MCP] Calling gsam2_parse_video for full sequence...")

        file_base_url = self.url.rstrip("/").removesuffix("/sse")
        video_path = Path(self._video_source).expanduser().resolve()
        video_file_id: str | None = None
        try:
            import httpx
            upload_url = f"{file_base_url}/upload?filename={video_path.name}"
            with httpx.Client(verify=False, timeout=300.0) as client:
                resp = client.post(upload_url, content=video_path.read_bytes())
                resp.raise_for_status()
                video_file_id = resp.json()["file_id"]
            logger.info("[MCP] Uploaded video to file service, file_id=%s", video_file_id)
        except Exception as exc:
            logger.warning("[MCP] Failed to upload video for gsam2_parse_video, will fall back to frame-by-frame: %s", exc)
            return

        args: dict = {
            "video_file_id": video_file_id,
            "text_prompt": self._text_prompt or "object.",
            "step": self._step,
            "iou_threshold": self._iou_threshold,
            "box_threshold": self._box_threshold,
            "text_threshold": self._text_threshold,
        }
        try:
            res = sync_call_mcp(self.url, "gsam2_parse_video", args)
            frames = res.get("frames", []) if isinstance(res, dict) else []
            loaded = 0
            for item in frames:
                detections = FrameDetections.model_validate(item)
                self._video_batches[detections.frame_idx] = detections
                loaded += 1
            logger.info("[MCP] gsam2_parse_video loaded %d frame batches.", loaded)
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Cannot connect to unified MCP at {self.url}. "
                    "Please start/restart the embody-unified SSE server."
                ) from exc
            logger.warning("[MCP] gsam2_parse_video unavailable/failed, falling back to gsam2_parse_frame: %s", exc)

    def get_object_detection_prompts(self) -> list[str]:
        return list(self._prompts)

    def get_first_frame_b64(self) -> str | None:
        if self._first_frame_b64 is None:
            self._read_b64_frame()
        return self._first_frame_b64


class MCPSeg2TrackAdapter:
    """Perception adapter using Seg2Track-SAM2 session-based streaming MCP service.

    Same interface as ``MCPGroundedSAM2Adapter`` — provides ``detect_objects_in_frame``,
    ``set_object_detection_prompts``, and ``get_first_frame_b64``.

    Unlike the batch-mode adapters, this adapter does NOT implement
    ``prefetch_video``.  Every frame goes through ``seg2track_add_frame``
    which carries the current ``text_prompt``, so VLM-discovered categories
    are picked up by Grounding DINO on the very next detection cycle.
    """

    def __init__(
        self,
        url: str,
        video_source: str | None = None,
        detect_interval: int = 5,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        self.url = url
        self._video_source = (video_source or "").strip() or None
        self._detect_interval = detect_interval
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._prompts: list[str] = []
        self._text_prompt: str = ""
        self._session_id: str | None = None
        self._cap = None
        self._first_frame_b64: str | None = None

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def set_object_detection_prompts(self, prompts: list[str]) -> None:
        self._prompts = prompts
        # Grounding DINO expects dot-separated prompt: "cup. bottle. person."
        self._text_prompt = ". ".join(p.strip() for p in prompts if p.strip())
        if self._text_prompt and not self._text_prompt.endswith("."):
            self._text_prompt += "."

    def detector_status(self) -> dict:
        return {"backend": "mcp_seg2track_sam2", "url": self.url, "session_id": self._session_id}

    def _ensure_session(self) -> str:
        """Lazily initialise a Seg2Track session on first use."""
        if self._session_id is not None:
            return self._session_id

        logger.info("[MCP] Creating Seg2Track session...")
        try:
            res = sync_call_mcp(self.url, "seg2track_init", {
                "text_prompt": self._text_prompt or "object.",
                "detect_interval": self._detect_interval,
            })
            self._session_id = res["session_id"]
            logger.info("[MCP] Seg2Track session created: %s", self._session_id)
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Cannot connect to unified MCP at {self.url}. "
                    "Please start/restart the embody-unified SSE server."
                ) from exc
            raise
        return self._session_id

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

    def _read_b64_frame_at(self, frame_idx: int) -> str | None:
        if not self._video_source or self._video_source.isdigit():
            return None
        try:
            import cv2
            import base64
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
            b64_str = base64.b64encode(buffer).decode("ascii")
            if int(frame_idx) == 1 and self._first_frame_b64 is None:
                self._first_frame_b64 = b64_str
            return b64_str
        finally:
            cap.release()

    def detect_objects_in_frame(self, frame_idx: int, timestamp: float) -> FrameDetections:
        session_id = self._ensure_session()
        image_b64 = self._read_b64_frame()
        if not image_b64:
            return FrameDetections(frame_idx=frame_idx, timestamp=timestamp, instances=[])

        logger.info("[MCP] Calling seg2track_add_frame for frame %d...", frame_idx)
        args = {
            "session_id": session_id,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "image_base64": image_b64,
            "text_prompt": self._text_prompt or "object.",
        }
        try:
            res = sync_call_mcp(self.url, "seg2track_add_frame", args)
            batch = FrameDetections.model_validate(res)
            # Attach the frame image for downstream use
            if not batch.image_b64 and image_b64:
                batch = batch.model_copy(update={"image_b64": image_b64})
            return batch
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"Cannot connect to unified MCP at {self.url}. "
                    "Please start/restart the embody-unified SSE server."
                ) from exc
            raise

    # NOTE: No prefetch_video() — Seg2Track uses frame-by-frame streaming via
    # seg2track_add_frame so that VLM-discovered prompt updates take effect
    # immediately on the next Grounding DINO detection cycle.

    def get_object_detection_prompts(self) -> list[str]:
        return list(self._prompts)

    def get_first_frame_b64(self) -> str | None:
        if self._first_frame_b64 is None:
            self._read_b64_frame()
        return self._first_frame_b64

    def __del__(self) -> None:
        if self._session_id is not None:
            try:
                sync_call_mcp(self.url, "seg2track_close", {"session_id": self._session_id})
            except Exception:
                pass


class MCPStreamSplatAdapter:
    """Adapter for the dynamic_reconstruction (StreamSplat) MCP service.

    Uploads a video (and optionally a depth .npy), calls splat_reconstruct,
    and downloads the rendered output video.
    """

    def __init__(self, url: str, output_root: str | None = None) -> None:
        self.url = url
        self._file_base_url = url.rstrip("/").removesuffix("/sse")
        self._output_root = Path(output_root).resolve() if output_root else None

    def reconstruct_scene(
        self,
        video_path: str,
        depths_file_id: str | None = None,
        fps: int = 24,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Upload video, run StreamSplat scene reconstruction, download result.

        Returns dict with keys: render_path, input_path, num_input_frames,
        num_output_frames, frame_gap, resolution.
        """
        import httpx

        vp = Path(video_path).expanduser().resolve()
        logger.info("[MCPStreamSplat] Uploading video %s ...", vp.name)
        upload_url = f"{self._file_base_url}/upload?filename={vp.name}"
        with httpx.Client(verify=False, timeout=300.0) as client:
            resp = client.post(upload_url, content=vp.read_bytes())
            resp.raise_for_status()
            video_file_id = resp.json()["file_id"]

        logger.info("[MCPStreamSplat] Calling splat_reconstruct ...")
        args: dict[str, Any] = {
            "video_file_id": video_file_id,
            "fps": fps,
            "batch_size": batch_size,
        }
        if depths_file_id:
            args["depths_file_id"] = depths_file_id

        res = sync_call_mcp(self.url, "splat_reconstruct", args)

        render_file_id = res.get("render_file_id", "")
        input_file_id = res.get("input_file_id", "")
        points_file_id = res.get("points_file_id", "")
        gaussians_file_id = res.get("gaussians_file_id", "")

        render_path = self._download_to_output(render_file_id, "scene_render.mp4")
        input_path = self._download_to_output(input_file_id, "scene_input.mp4")
        points_path = self._download_to_output(points_file_id, "scene_points.ply") if points_file_id else None
        gaussians_path = self._download_to_output(gaussians_file_id, "scene_gaussians.ply") if gaussians_file_id else None

        return {
            "render_path": str(render_path) if render_path else "",
            "input_path": str(input_path) if input_path else "",
            "points_path": str(points_path) if points_path else "",
            "gaussians_path": str(gaussians_path) if gaussians_path else "",
            "render_file_id": render_file_id,
            "input_file_id": input_file_id,
            "points_file_id": points_file_id,
            "gaussians_file_id": gaussians_file_id,
            "num_input_frames": res.get("num_input_frames", 0),
            "num_output_frames": res.get("num_output_frames", 0),
            "frame_gap": res.get("frame_gap", 0),
            "resolution": res.get("resolution", []),
        }

    def _download_to_output(self, file_id: str, filename: str) -> Path | None:
        if not file_id:
            return None
        import httpx
        import tempfile

        url = f"{self._file_base_url}/download/{file_id}"
        try:
            with httpx.Client(verify=False, timeout=120.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.content
        except Exception as exc:
            logger.warning("[MCPStreamSplat] Failed to download %s: %s", file_id, exc)
            return None

        if self._output_root:
            dst = self._output_root / "exports" / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = _unique_path(dst)
            dst.write_bytes(data)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
            tmp.close()
            dst = Path(tmp.name)
            dst.write_bytes(data)
        logger.info("[MCPStreamSplat] Downloaded %s -> %s", file_id, dst)
        return dst


class MCPWildGSAdapter:
    """Adapter for the wildgs_slam MCP service.

    Uploads a video or frames archive, calls wildgs_run_slam, and downloads
    camera poses, static map, and dynamic prior to local paths.

    wildgs_direct_url: optional URL to a standalone wildgs_slam server for
    mesh reconstruction (bypasses the unified proxy which may not register
    wildgs tools if the sub-service takes too long to start).

    Usage::

        adapter = MCPWildGSAdapter(url="http://qingyan-mcps:8080/sse",
                                   output_root="/tmp/wildgs_run")
        result = adapter.run_slam(video_path="/data/scene.mp4")
        # result.camera_poses_path  → camera_poses.jsonl (local)
        # result.static_map_dir     → directory containing static_gaussians.ply
        # result.dynamic_prior_dir  → directory containing per-frame .npy uncertainty
    """

    def __init__(self, url: str, output_root: str | None = None, wildgs_direct_url: str | None = None) -> None:
        self.url = url
        self._wildgs_direct_url = wildgs_direct_url or url
        self._file_base_url = url.rstrip("/").removesuffix("/sse")
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
        """Run WildGS-SLAM and return local paths to the outputs.

        Exactly one of ``video_path`` or ``frames_dir`` must be provided.

        Args:
            video_path:    Local path to the input video file.
            frames_dir:    Local directory of extracted frames (JPEG/PNG sorted).
                           Will be packed as a .tar.gz before upload.
            intrinsics:    Optional dict {fx, fy, cx, cy} for camera intrinsics.
            fps_override:  Override the server's default slam_fps.
            run_id:        Optional string tag passed to the server for naming.

        Returns:
            {
              "camera_poses_path":  str | None,   # local path to camera_poses.jsonl
              "static_map_dir":     str | None,   # local directory unpacked from tar
              "dynamic_prior_dir":  str | None,   # local directory unpacked from tar
              "num_frames":         int,
              "slam_quality":       float,
              "camera_poses_file_id":  str,
              "static_map_file_id":    str,
              "dynamic_prior_file_id": str,
            }
        """
        import httpx
        import json as _json
        import tarfile as _tarfile
        import tempfile

        if not video_path and not frames_dir:
            raise ValueError("Provide either video_path or frames_dir.")

        args: dict[str, Any] = {}

        if video_path:
            vp = Path(video_path).expanduser().resolve()
            logger.info("[MCPWildGS] Uploading video %s ...", vp.name)
            upload_url = f"{self._file_base_url}/upload?filename={vp.name}"
            with httpx.Client(verify=False, timeout=300.0) as client:
                resp = client.post(upload_url, content=vp.read_bytes())
                resp.raise_for_status()
                args["video_file_id"] = resp.json()["file_id"]
        else:
            # Pack frames directory as tar.gz
            fp = Path(frames_dir).expanduser().resolve()
            logger.info("[MCPWildGS] Packing frames dir %s ...", fp)
            tmp_tar = Path(tempfile.mktemp(suffix=".tar.gz"))
            with _tarfile.open(tmp_tar, "w:gz") as tar:
                tar.add(str(fp), arcname=fp.name)
            upload_url = f"{self._file_base_url}/upload?filename={fp.name}.tar.gz"
            with httpx.Client(verify=False, timeout=300.0) as client:
                resp = client.post(upload_url, content=tmp_tar.read_bytes())
                resp.raise_for_status()
                args["frames_dir_file_id"] = resp.json()["file_id"]
            tmp_tar.unlink(missing_ok=True)

        if intrinsics:
            args["intrinsics"] = _json.dumps(intrinsics)
        if fps_override is not None:
            args["fps_override"] = fps_override
        if run_id:
            args["run_id"] = run_id
        if export_depth_every_frame is not None:
            args["export_depth_every_frame"] = bool(export_depth_every_frame)
        if depth_export_stride is not None:
            args["depth_export_stride"] = int(depth_export_stride)
        if pose_export_stride is not None:
            args["pose_export_stride"] = int(pose_export_stride)
        if extract_every_input_frame is not None:
            args["extract_every_input_frame"] = bool(extract_every_input_frame)
        if frame_stride is not None:
            args["frame_stride"] = int(frame_stride)

        logger.info("[MCPWildGS] Calling wildgs_run_slam ...")
        res = sync_call_mcp(self.url, "wildgs_run_slam", args, sse_read_timeout=7200)

        # Download outputs
        camera_poses_path = self._download_file(
            res.get("camera_poses_file_id", ""), "camera_poses.jsonl"
        )
        static_map_dir = self._download_static_map(
            res.get("static_map_file_id", "")
        )
        dynamic_prior_dir = self._download_and_unpack(
            res.get("dynamic_prior_file_id", ""), "dynamic_prior"
        )
        depth_maps_dir = self._download_and_unpack(
            res.get("depth_maps_file_id", ""), "depth_maps"
        )
        plots_dir = self._download_and_unpack(
            res.get("plots_file_id", ""), "plots_after_refine"
        )

        return {
            "camera_poses_path": str(camera_poses_path) if camera_poses_path else None,
            "static_map_dir": str(static_map_dir) if static_map_dir else None,
            "dynamic_prior_dir": str(dynamic_prior_dir) if dynamic_prior_dir else None,
            "depth_maps_dir": str(depth_maps_dir) if depth_maps_dir else None,
            "plots_dir": str(plots_dir) if plots_dir else None,
            "num_frames": res.get("num_frames", 0),
            "slam_quality": res.get("slam_quality", 0.0),
            "camera_poses_file_id": res.get("camera_poses_file_id", ""),
            "static_map_file_id": res.get("static_map_file_id", ""),
            "dynamic_prior_file_id": res.get("dynamic_prior_file_id", ""),
            "depth_maps_file_id": res.get("depth_maps_file_id", ""),
            "plots_file_id": res.get("plots_file_id", ""),
        }

    def _download_file(self, file_id: str, filename: str) -> Path | None:
        """Download a single file by file_id to local output root."""
        if not file_id:
            return None
        import httpx
        import tempfile

        url = f"{self._file_base_url}/download/{file_id}"
        try:
            with httpx.Client(verify=False, timeout=120.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.content
        except Exception as exc:
            logger.warning("[MCPWildGS] Failed to download %s: %s", file_id, exc)
            return None

        if self._output_root:
            dst = self._output_root / "exports" / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = _unique_path(dst)
            dst.write_bytes(data)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)
            tmp.close()
            dst = Path(tmp.name)
            dst.write_bytes(data)

        logger.info("[MCPWildGS] Downloaded %s -> %s", file_id, dst)
        return dst

    def _download_and_unpack(self, file_id: str, dir_name: str) -> Path | None:
        """Download a .tar.gz by file_id and unpack it to a local directory."""
        if not file_id:
            return None
        import tarfile as _tarfile

        tar_path = self._download_file(file_id, f"{dir_name}.tar.gz")
        if not tar_path:
            return None

        if self._output_root:
            unpack_dir = self._output_root / "exports" / dir_name
        else:
            import tempfile
            unpack_dir = Path(tempfile.mkdtemp(suffix=f"_{dir_name}"))

        unpack_dir.mkdir(parents=True, exist_ok=True)
        with _tarfile.open(tar_path) as tar:
            tar.extractall(str(unpack_dir))

        # If unpacked into a single subdirectory, return that
        entries = list(unpack_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0]
        return unpack_dir

    def _download_static_map(self, file_id: str) -> Path | None:
        """Download the static map (either a .ply file or a .tar.gz archive).

        Returns a directory containing final_gs.ply (or equivalent) if successful.
        """
        if not file_id:
            return None

        # PLY files are downloaded directly, then placed in a directory
        if file_id.endswith(".ply"):
            ply_path = self._download_file(file_id, "final_gs.ply")
            if not ply_path:
                return None
            if self._output_root:
                map_dir = self._output_root / "exports" / "static_map"
            else:
                import tempfile
                map_dir = Path(tempfile.mkdtemp(suffix="_static_map"))
            map_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(ply_path, map_dir / "final_gs.ply")
            return map_dir

        # Fall back to tar.gz unpacking
        return self._download_and_unpack(file_id, "static_map")

    def reconstruct_background_mesh(
        self,
        static_map_file_id: str,
        poisson_depth: int = 7,
        opacity_threshold: float = 0.3,
    ) -> dict:
        """Call the wildgs_mesh_from_gs tool to convert final_gs.ply → OBJ mesh.

        Returns a dict with:
            mesh_dir:     local Path (str) to directory containing background_mesh.obj
            num_vertices: int
            num_faces:    int
        """
        if not static_map_file_id:
            return {}
        res = sync_call_mcp(
            self._wildgs_direct_url,
            "wildgs_mesh_from_gs",
            {
                "static_map_file_id": static_map_file_id,
                "poisson_depth": poisson_depth,
                "opacity_threshold": opacity_threshold,
            },
            sse_read_timeout=600,
        )
        mesh_dir = self._download_and_unpack(res.get("mesh_file_id", ""), "background_mesh")
        return {
            "mesh_dir": str(mesh_dir) if mesh_dir else None,
            "num_vertices": res.get("num_vertices", 0),
            "num_faces": res.get("num_faces", 0),
        }
