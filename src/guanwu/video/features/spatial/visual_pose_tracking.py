from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VisualPoseResult:
    translation_world: list[float]
    orientation_quat: list[float]
    score: float = 1.0
    accepted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualPoseTracker(Protocol):
    def refine_pose(self, payload: dict[str, Any]) -> VisualPoseResult | None:
        """Refine a world-space object pose for a single frame."""


class CommandVisualPoseTracker:
    """Shell adapter for external visual pose backends such as GoTrack.

    The command receives a JSON payload on stdin and must emit a JSON object on
    stdout. The response should contain:
      - `translation_world` or `centroid_world`: [x, y, z]
      - `orientation_quat` (xyzw) or `rotation_matrix` (3x3)
    Optional fields:
      - `score`: float
      - `accepted`: bool
      - any extra metadata
    """

    def __init__(self, command: str, *, timeout_sec: float = 30.0) -> None:
        self.command = command.strip()
        self.timeout_sec = float(timeout_sec)

    def refine_pose(self, payload: dict[str, Any]) -> VisualPoseResult | None:
        if not self.command:
            return None
        proc = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout_sec,
        )
        if proc.returncode != 0:
            logger.warning(
                "[VisualPose] command failed for %s@%s: %s",
                payload.get("object_id"),
                payload.get("frame_idx"),
                proc.stderr.strip(),
            )
            return None
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            logger.warning(
                "[VisualPose] command returned non-JSON output for %s@%s",
                payload.get("object_id"),
                payload.get("frame_idx"),
            )
            return None
        if not isinstance(data, dict):
            logger.warning(
                "[VisualPose] command returned non-object JSON for %s@%s",
                payload.get("object_id"),
                payload.get("frame_idx"),
            )
            return None
        return _result_from_payload_dict(data, payload)


class MCPVisualPoseTracker:
    """MCP adapter for visual pose refinement backends such as GoTrack."""

    def __init__(
        self,
        url: str,
        *,
        tool_name: str = "gotrack_refine_pose",
        timeout_sec: float = 30.0,
    ) -> None:
        self.url = url.strip()
        self.tool_name = tool_name.strip() or "gotrack_refine_pose"
        self.timeout_sec = float(timeout_sec)
        self._file_base_url = self.url.rstrip("/").removesuffix("/sse")
        self._mesh_upload_cache: dict[tuple[str, int, int], str] = {}

    def refine_pose(self, payload: dict[str, Any]) -> VisualPoseResult | None:
        from guanwu.video.clients.mcp_backend import sync_call_mcp

        prepared_payload = self._prepare_payload(dict(payload))
        try:
            data = sync_call_mcp(
                self.url,
                self.tool_name,
                prepared_payload,
                sse_read_timeout=max(600, int(self.timeout_sec) + 30),
                call_timeout_sec=self.timeout_sec,
            )
        except Exception as exc:
            logger.warning(
                "[VisualPose] MCP tool failed for %s@%s via %s/%s: %s",
                payload.get("object_id"),
                payload.get("frame_idx"),
                self.url,
                self.tool_name,
                exc,
            )
            return None
        if not isinstance(data, dict):
            logger.warning(
                "[VisualPose] MCP tool returned non-object JSON for %s@%s",
                prepared_payload.get("object_id"),
                prepared_payload.get("frame_idx"),
            )
            return None
        return _result_from_payload_dict(data, prepared_payload)

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        mesh_path = payload.get("mesh_path")
        if not isinstance(mesh_path, str) or not mesh_path.strip():
            return payload

        mesh_file = Path(mesh_path).expanduser()
        if not mesh_file.is_file():
            return payload

        try:
            mesh_file = mesh_file.resolve()
            stat = mesh_file.stat()
            cache_key = (str(mesh_file), int(stat.st_mtime_ns), int(stat.st_size))
            mesh_file_id = self._mesh_upload_cache.get(cache_key)
            if mesh_file_id is None:
                mesh_file_id = self._upload_file(mesh_file)
                self._mesh_upload_cache[cache_key] = mesh_file_id
            payload["mesh_file_id"] = mesh_file_id
            payload.pop("mesh_path", None)
        except Exception as exc:
            logger.warning(
                "[VisualPose] failed to upload mesh for %s@%s: %s",
                payload.get("object_id"),
                payload.get("frame_idx"),
                exc,
            )
        return payload

    def _upload_file(self, path: Path) -> str:
        import httpx

        upload_url = f"{self._file_base_url}/upload?filename={path.name}"
        with httpx.Client(verify=False, timeout=max(300.0, self.timeout_sec)) as client:
            resp = client.post(upload_url, content=path.read_bytes())
            resp.raise_for_status()
            file_id = resp.json()["file_id"]
        logger.info("[VisualPose] uploaded %s to MCP file service as %s", path.name, file_id)
        return str(file_id)


def build_visual_pose_tracker(
    *,
    backend: str,
    prefer_mcp: bool = False,
    mcp_url: str | None = None,
    mcp_tool: str = "gotrack_refine_pose",
    command: str | None,
    timeout_sec: float = 30.0,
) -> VisualPoseTracker | None:
    if str(backend).strip().lower() != "gotrack_visual":
        return None
    if prefer_mcp and mcp_url and mcp_url.strip():
        return MCPVisualPoseTracker(
            mcp_url.strip(),
            tool_name=mcp_tool,
            timeout_sec=timeout_sec,
        )
    if not command or not command.strip():
        logger.info("[VisualPose] no MCP or command backend configured for gotrack_visual; falling back to geometry alignment")
        return None
    return CommandVisualPoseTracker(command.strip(), timeout_sec=timeout_sec)


def _is_vec(values: Any, size: int) -> bool:
    return isinstance(values, list) and len(values) >= size


def _result_from_payload_dict(data: dict[str, Any], payload: dict[str, Any]) -> VisualPoseResult | None:
    translation = data.get("translation_world", data.get("centroid_world"))
    orientation = data.get("orientation_quat")
    if orientation is None and data.get("rotation_matrix") is not None:
        orientation = _rotation_matrix_to_quat_xyzw(data["rotation_matrix"])
    if not _is_vec(translation, 3) or not _is_vec(orientation, 4):
        logger.warning(
            "[VisualPose] backend response missing pose fields for %s@%s",
            payload.get("object_id"),
            payload.get("frame_idx"),
        )
        return None

    metadata = {
        key: value
        for key, value in data.items()
        if key not in {"translation_world", "centroid_world", "orientation_quat", "rotation_matrix", "score", "accepted"}
    }
    return VisualPoseResult(
        translation_world=[float(v) for v in translation],
        orientation_quat=[float(v) for v in orientation],
        score=float(data.get("score", 1.0) or 0.0),
        accepted=bool(data.get("accepted", True)),
        metadata=metadata,
    )


def _rotation_matrix_to_quat_xyzw(rotation: Any) -> list[float]:
    import math
    import numpy as np

    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        return [0.0, 0.0, 0.0, 1.0]

    m00, m01, m02 = float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2])
    m10, m11, m12 = float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2])
    m20, m21, m22 = float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return quat.tolist()
