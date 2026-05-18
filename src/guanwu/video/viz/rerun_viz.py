"""Rerun-based real-time 3D scene visualizer for SPWM Agent.

Streams per-frame data to a Rerun viewer:
  - Video frames with detection bbox overlays
  - 3D object positions (boxes / meshes when available)
  - Object trajectories over time
  - Relation edges as 3D line segments
  - Events as text log entries

Usage:
    The visualizer is automatically enabled when ``rerun-sdk`` is installed
    and ``runtime.rerun_enabled = true`` in config.  The Rerun viewer can be
    launched separately with ``rerun`` or ``rerun --web``.

    Alternatively pass ``--connect`` to connect to a remote viewer:
        rerun --serve          # on the viewer host
        # then the SDK connects via RERUN_ADDR or spawn=False
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from guanwu.video.core.schema import Event, ObjectNode, RelationEdge

logger = logging.getLogger(__name__)

try:
    import rerun as rr

    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False
    rr = None  # type: ignore[assignment]


# Category → RGBA colour
_CATEGORY_COLORS: dict[str, tuple[int, int, int, int]] = {
    "container": (31, 119, 180, 220),
    "furniture": (44, 160, 44, 220),
    "manipulator": (214, 39, 40, 220),
    "unknown": (127, 127, 127, 220),
}

_BODY_COLOR = (230, 159, 0, 220)
_RELATION_COLOR = (160, 160, 160, 140)


def _valid_vec3(value: object) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return [float(v) for v in arr]


def _valid_quat_xyzw(value: object) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(4)
    except Exception:
        return None
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return [float(v) for v in (arr / norm)]


class RerunVisualizer:
    """Streams SPWM scene state to a Rerun viewer each frame."""

    def __init__(
        self,
        application_id: str = "spwm_scene",
        spawn: bool = True,
        connect: str | None = None,
    ) -> None:
        if not HAS_RERUN:
            raise RuntimeError(
                "rerun-sdk is not installed.  Install with:  pip install rerun-sdk"
            )
        self._spawn = spawn
        self._connect = connect
        self._initialized = False
        self._application_id = application_id
        # Cache loaded meshes so we don't re-read PLY every frame
        self._mesh_cache: dict[str, Any] = {}

    def init(self) -> None:
        """Initialize Rerun recording. Call once before the first log."""
        if self._initialized:
            return
        rr.init(self._application_id, spawn=self._spawn)
        if self._connect:
            rr.connect(self._connect)

        # Set up a global 3D view with sensible defaults
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._initialized = True
        logger.info("[RerunViz] Initialized (spawn=%s)", self._spawn)

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def log_frame(
        self,
        frame_idx: int,
        timestamp: float,
        image_b64: str | None,
        objects: list[ObjectNode],
        relations: list[RelationEdge],
        events: list[Event],
        sam3d_meshes: dict[str, dict],
    ) -> None:
        """Log one frame's worth of data to Rerun."""
        if not self._initialized:
            self.init()

        rr.set_time_sequence("frame", frame_idx)
        rr.set_time_seconds("time", timestamp)

        self._log_image(image_b64, objects)
        self._log_objects_3d(objects, sam3d_meshes)
        self._log_relations(objects, relations)
        self._log_events(events)

    # ------------------------------------------------------------------
    # Image + 2D detections
    # ------------------------------------------------------------------

    def _log_image(self, image_b64: str | None, objects: list[ObjectNode]) -> None:
        if not image_b64:
            return

        img = _decode_b64_to_rgb(image_b64)
        if img is None:
            return

        rr.log("camera/image", rr.Image(img))

        # Detection bboxes as 2D boxes
        if objects:
            rects, labels, colors = [], [], []
            for obj in objects:
                bbox = obj.geometry.bbox_2d
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                # Rerun uses XYWH format
                rects.append([x1, y1, x2 - x1, y2 - y1])
                labels.append(f"{obj.label} {obj.confidence:.2f}")
                cat = obj.semantic.category if obj.semantic else "unknown"
                colors.append(_BODY_COLOR if obj.segment_kind == "body" else _CATEGORY_COLORS.get(cat, _CATEGORY_COLORS["unknown"]))

            if rects:
                rr.log(
                    "camera/detections",
                    rr.Boxes2D(
                        array=rects,
                        array_format=rr.Box2DFormat.XYWH,
                        labels=labels,
                        colors=colors,
                    ),
                )

    # ------------------------------------------------------------------
    # 3D objects (boxes + mesh when available)
    # ------------------------------------------------------------------

    def _log_objects_3d(self, objects: list[ObjectNode], sam3d_meshes: dict[str, dict]) -> None:
        for obj in objects:
            entity = f"world/objects/{obj.object_id}"
            pos = _valid_vec3(obj.geometry.pose_3d.position)
            if pos is None:
                continue
            scale = _valid_vec3(obj.geometry.scale_3d)
            quat_xyzw = _valid_quat_xyzw(obj.geometry.pose_3d.orientation_quat)  # [qx, qy, qz, qw]
            cat = obj.semantic.category if obj.semantic else "unknown"
            color = _BODY_COLOR if obj.segment_kind == "body" else _CATEGORY_COLORS.get(cat, _CATEGORY_COLORS["unknown"])

            # Set transform
            transform_kwargs: dict[str, Any] = {"translation": pos}
            if quat_xyzw is not None:
                transform_kwargs["rotation"] = rr.Quaternion(xyzw=quat_xyzw)
            rr.log(entity, rr.Transform3D(**transform_kwargs))

            # Try to log mesh if available
            mesh_info = sam3d_meshes.get(obj.object_id, {})
            mesh_path = mesh_info.get("mesh_path", "")
            mesh_logged = False
            if mesh_path and Path(mesh_path).exists():
                mesh_logged = self._log_mesh(f"{entity}/mesh", mesh_path, color)

            if not mesh_logged and scale is not None:
                half = [s / 2.0 for s in scale]
                rr.log(
                    f"{entity}/box",
                    rr.Boxes3D(
                        half_sizes=[half],
                        colors=[color],
                        labels=[obj.label],
                    ),
                )

            # Log trajectory point (accumulated over time by Rerun timeline)
            rr.log(
                f"world/trajectories/{obj.object_id}",
                rr.Points3D(
                    positions=[pos],
                    colors=[color],
                    radii=[0.008],
                ),
            )

            # Velocity arrow
            vel = _valid_vec3(obj.physics.velocity_linear)
            speed = sum(v * v for v in vel) ** 0.5 if vel is not None else 0.0
            if speed > 0.003:
                arrow_scale = 3.0  # exaggerate for visibility
                rr.log(
                    f"{entity}/velocity",
                    rr.Arrows3D(
                        origins=[pos],
                        vectors=[[v * arrow_scale for v in vel]],
                        colors=[(255, 100, 0, 200)],
                    ),
                )

    def _log_mesh(self, entity: str, mesh_path: str, color: tuple) -> bool:
        """Load and log a PLY/GLB mesh. Returns True if successful."""
        if mesh_path in self._mesh_cache:
            cached = self._mesh_cache[mesh_path]
            if cached is None:
                return False
            rr.log(entity, cached)
            return True

        try:
            path = Path(mesh_path)
            if path.suffix.lower() == ".ply":
                mesh = self._load_ply(path, color)
            elif path.suffix.lower() in {".glb", ".gltf"}:
                mesh = rr.Asset3D(path=str(path))
            else:
                self._mesh_cache[mesh_path] = None
                return False

            self._mesh_cache[mesh_path] = mesh
            rr.log(entity, mesh)
            return True
        except Exception as exc:
            logger.debug("[RerunViz] Failed to load mesh %s: %s", mesh_path, exc)
            self._mesh_cache[mesh_path] = None
            return False

    @staticmethod
    def _load_ply(path: Path, color: tuple) -> Any:
        """Minimal PLY loader → rr.Mesh3D."""
        import struct

        text = path.read_bytes()
        # Find header end
        header_end = text.index(b"end_header\n") + len(b"end_header\n")
        header = text[:header_end].decode("ascii", errors="replace")

        vertex_count = 0
        face_count = 0
        is_binary = "format binary" in header
        for line in header.splitlines():
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])

        if vertex_count == 0:
            raise ValueError("No vertices in PLY")

        if is_binary:
            # Assume float32 x,y,z per vertex (12 bytes)
            data = text[header_end:]
            verts = np.frombuffer(data, dtype=np.float32, count=vertex_count * 3, offset=0)
            vertices = verts.reshape(vertex_count, 3)
        else:
            lines = text[header_end:].decode("ascii", errors="replace").strip().split("\n")
            vertices = np.array(
                [[float(v) for v in line.split()[:3]] for line in lines[:vertex_count]],
                dtype=np.float32,
            )

        colors_arr = np.tile(np.array(color[:3], dtype=np.uint8), (vertex_count, 1))
        return rr.Points3D(positions=vertices, colors=colors_arr, radii=[0.003])

    # ------------------------------------------------------------------
    # Relations as 3D line segments
    # ------------------------------------------------------------------

    def _log_relations(self, objects: list[ObjectNode], relations: list[RelationEdge]) -> None:
        if not relations:
            return

        obj_map = {o.object_id: o for o in objects}
        strips, colors, labels = [], [], []
        for rel in relations:
            s = obj_map.get(rel.subject_id)
            o = obj_map.get(rel.object_id)
            if s is None or o is None:
                continue
            sp = _valid_vec3(s.geometry.pose_3d.position)
            op = _valid_vec3(o.geometry.pose_3d.position)
            if sp is None or op is None:
                continue
            strips.append([sp, op])
            colors.append(_RELATION_COLOR)
            labels.append(rel.predicate)

        if strips:
            rr.log(
                "world/relations",
                rr.LineStrips3D(
                    strips,
                    colors=colors,
                    labels=labels,
                ),
            )

    # ------------------------------------------------------------------
    # Events as text log
    # ------------------------------------------------------------------

    def _log_events(self, events: list[Event]) -> None:
        for evt in events:
            actors = ", ".join(evt.actors) if evt.actors else "?"
            targets = ", ".join(evt.targets) if evt.targets else ""
            text = f"[{evt.type}] {actors}"
            if targets:
                text += f" → {targets}"
            rr.log("events", rr.TextLog(text, level=rr.TextLogLevel.INFO))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_b64_to_rgb(image_b64: str | None) -> np.ndarray | None:
    if not image_b64:
        return None
    payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    from PIL import Image

    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
        return np.asarray(img)
    except Exception:
        return None
