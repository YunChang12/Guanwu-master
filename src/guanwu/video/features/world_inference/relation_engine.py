from __future__ import annotations

from itertools import combinations
import math
from typing import Any

from guanwu.video.core.logger import get_logger
from guanwu.video.core.schema import ObjectNode, RelationEdge, RelationTemporal

logger = get_logger(__name__)

# Sentinel IDs for static background elements
_FLOOR_ID = "__floor__"
_WALL_ID = "__wall__"

# Thresholds
_ON_FLOOR_MARGIN_M = 0.15   # object bottom within 15 cm of floor plane → on_floor
_AGAINST_WALL_MARGIN_M = 0.25  # object centroid within 25 cm of wall plane → against_wall
_BACKGROUND_LABEL_TOKENS = (
    "grass",
    "road",
    "ground",
    "floor",
    "wall",
    "ceiling",
    "track",
    "railing",
    "fence",
    "barrier",
)


def _load_ply_points(ply_path: str) -> Any | None:
    """Load PLY vertex positions as numpy array [N, 3]. Returns None on failure."""
    try:
        import numpy as np
        try:
            import trimesh
            cloud = trimesh.load(ply_path, force="pointcloud")
            pts = np.asarray(cloud.vertices, dtype=np.float64)
        except Exception:
            # Minimal fallback: read ASCII/binary PLY manually via open3d
            import open3d as o3d  # type: ignore[import]
            pcd = o3d.io.read_point_cloud(ply_path)
            pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return None
        return pts
    except Exception as exc:
        logger.warning("[RelationEngine] Failed to load PLY %s: %s", ply_path, exc)
        return None


def _estimate_background_planes(pts: Any) -> dict:
    """Extract floor level and wall extents from a WildGS static-map point cloud.

    WildGS world coordinates follow camera convention: Y axis points down.
    The floor is therefore the dominant plane at the high-Y end of the cloud.

    Returns a dict with:
        floor_y   – Y coordinate of the floor plane (None if not found)
        wall_x_min, wall_x_max – approximate wall extents along X
        wall_z_min, wall_z_max – approximate wall extents along Z
    """
    import numpy as np

    result: dict = {
        "floor_y": None,
        "wall_x_min": None, "wall_x_max": None,
        "wall_z_min": None, "wall_z_max": None,
    }

    if pts is None or len(pts) == 0:
        return result

    # Subsample to keep computation fast (max 100k points for histogram)
    if len(pts) > 100_000:
        step = len(pts) // 100_000
        pts = pts[::step]

    y = pts[:, 1]

    # Floor: dominant horizontal plane in the top-20% by Y value (Y is down)
    y_thresh = float(np.percentile(y, 80))
    floor_pts = y[y >= y_thresh]
    if len(floor_pts) > 10:
        counts, edges = np.histogram(floor_pts, bins=40)
        peak = int(np.argmax(counts))
        result["floor_y"] = float((edges[peak] + edges[peak + 1]) / 2.0)

    # Walls: use 5th / 95th percentile of X and Z as boundary planes
    result["wall_x_min"] = float(np.percentile(pts[:, 0], 5))
    result["wall_x_max"] = float(np.percentile(pts[:, 0], 95))
    result["wall_z_min"] = float(np.percentile(pts[:, 2], 5))
    result["wall_z_max"] = float(np.percentile(pts[:, 2], 95))

    logger.debug(
        "[RelationEngine] Background planes: floor_y=%.3f, X=[%.2f,%.2f], Z=[%.2f,%.2f]",
        result["floor_y"] or 0,
        result["wall_x_min"], result["wall_x_max"],
        result["wall_z_min"], result["wall_z_max"],
    )
    return result


class RelationEngine:
    """Rule-based relation inference for v0.1."""

    def infer(
        self,
        objects: list[ObjectNode],
        frame_idx: int,
        timestamp: float,
        background_geometry: dict | None = None,
    ) -> list[RelationEdge]:
        """Infer spatial relations between objects and optional static background.

        Args:
            objects: Active objects in the frame.
            frame_idx: Current frame index.
            timestamp: Frame timestamp (seconds).
            background_geometry: Optional dict with key ``"ply_path"`` pointing to
                the WildGS static-map PLY file.  When provided, emits ``on_floor``
                and ``against_wall`` relations for qualifying objects.
        """
        relations: list[RelationEdge] = []

        # --- Object–object relations ---
        by_label = {o.label: o for o in objects}
        for label in ("cup", "bottle"):
            if label in by_label and "table" in by_label:
                a = by_label[label]
                b = by_label["table"]
                if self._is_on(a, b):
                    relations.append(self._make_relation(a.object_id, "on", b.object_id, 0.85, frame_idx, timestamp))

        for a, b in combinations(objects, 2):
            if self._suppress_object_pair(a, b):
                continue
            if self._next_to(a, b):
                relations.append(self._make_relation(a.object_id, "next_to", b.object_id, 0.72, frame_idx, timestamp))
                relations.append(self._make_relation(b.object_id, "next_to", a.object_id, 0.72, frame_idx, timestamp))
            if self._contact(a, b):
                relations.append(self._make_relation(a.object_id, "contact_with", b.object_id, 0.78, frame_idx, timestamp))
                relations.append(self._make_relation(b.object_id, "contact_with", a.object_id, 0.78, frame_idx, timestamp))

        # --- Object–background relations ---
        if background_geometry:
            ply_path = background_geometry.get("ply_path")
            planes = background_geometry.get("_planes_cache")
            if planes is None and ply_path:
                pts = _load_ply_points(ply_path)
                planes = _estimate_background_planes(pts)
                background_geometry["_planes_cache"] = planes  # cache for subsequent frames

            if planes:
                for obj in objects:
                    if self._on_floor(obj, planes):
                        relations.append(self._make_relation(obj.object_id, "on_floor", _FLOOR_ID, 0.80, frame_idx, timestamp))
                    if self._against_wall(obj, planes):
                        relations.append(self._make_relation(obj.object_id, "against_wall", _WALL_ID, 0.70, frame_idx, timestamp))

        return relations

    # ------------------------------------------------------------------
    # Object–object predicates
    # ------------------------------------------------------------------

    def _make_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float,
        frame_idx: int,
        timestamp: float,
    ) -> RelationEdge:
        edge_id = f"rel_{subject_id}_{predicate}_{object_id}"
        return RelationEdge(
            edge_id=edge_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            temporal=RelationTemporal(start_ts=timestamp, status="active"),
            evidence={"source": "rule_engine", "frame_range": [frame_idx, frame_idx]},
        )

    def _is_on(self, a: ObjectNode, b: ObjectNode) -> bool:
        if not (_has_metric_pose(a) and _has_metric_pose(b)):
            return False
        ax, ay, az = a.geometry.pose_3d.position
        _, by, bz = b.geometry.pose_3d.position
        z_close = abs((az - a.geometry.scale_3d[2] / 2.0) - (bz + b.geometry.scale_3d[2] / 2.0)) < 0.2
        x_overlap = abs(ax - b.geometry.pose_3d.position[0]) < b.geometry.scale_3d[0] / 2.0
        y_top = ay >= by
        return z_close and x_overlap and y_top

    def _suppress_object_pair(self, a: ObjectNode, b: ObjectNode) -> bool:
        return self._is_background_like(a) or self._is_background_like(b)

    def _is_background_like(self, obj: ObjectNode) -> bool:
        label = str(obj.label or "").strip().lower()
        return any(token in label for token in _BACKGROUND_LABEL_TOKENS)

    def _next_to(self, a: ObjectNode, b: ObjectNode) -> bool:
        if not (_has_metric_pose(a) and _has_metric_pose(b)):
            return False
        ax, ay, _ = a.geometry.pose_3d.position
        bx, by, _ = b.geometry.pose_3d.position
        d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        return 0.15 < d < 0.9

    def _contact(self, a: ObjectNode, b: ObjectNode) -> bool:
        if not (_has_metric_pose(a) and _has_metric_pose(b)):
            return False
        ax, ay, az = a.geometry.pose_3d.position
        bx, by, bz = b.geometry.pose_3d.position
        d = ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5
        return d < 0.18

    # ------------------------------------------------------------------
    # Object–background predicates
    # ------------------------------------------------------------------

    def _on_floor(self, obj: ObjectNode, planes: dict) -> bool:
        """True when the object's bottom surface is within margin of the floor plane."""
        if not _has_metric_pose(obj):
            return False
        floor_y = planes.get("floor_y")
        if floor_y is None:
            return False
        # Y is down: object bottom = centroid_y + half_height
        cy = obj.geometry.pose_3d.position[1]
        half_h = obj.geometry.scale_3d[1] / 2.0
        obj_bottom_y = cy + half_h
        return abs(obj_bottom_y - floor_y) < _ON_FLOOR_MARGIN_M

    def _against_wall(self, obj: ObjectNode, planes: dict) -> bool:
        """True when the object's centroid is within margin of any wall boundary plane."""
        if not _valid_vec3(obj.geometry.pose_3d.position):
            return False
        cx = obj.geometry.pose_3d.position[0]
        cz = obj.geometry.pose_3d.position[2]
        for key, val in (
            ("wall_x_min", cx), ("wall_x_max", cx),
            ("wall_z_min", cz), ("wall_z_max", cz),
        ):
            plane = planes.get(key)
            if plane is not None and abs(val - plane) < _AGAINST_WALL_MARGIN_M:
                return True
        return False


def _valid_vec3(values: Any) -> bool:
    if not isinstance(values, list) or len(values) < 3:
        return False
    try:
        return all(math.isfinite(float(v)) for v in values[:3])
    except Exception:
        return False


def _has_metric_pose(obj: ObjectNode) -> bool:
    return _valid_vec3(obj.geometry.pose_3d.position) and _valid_vec3(obj.geometry.scale_3d)
