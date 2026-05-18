from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from guanwu.video.core.logger import get_logger
from guanwu.video.core.schema import ObjectNode
from guanwu.video.features.spatial.alignment_utils import (
    build_depth_point_cloud,
    compute_axis_scale,
    trim_point_cloud_outliers,
)
from guanwu.video.features.spatial.visual_pose_tracking import VisualPoseTracker

logger = get_logger(__name__)

try:
    import trimesh as _trimesh

    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

# ICP: subsample mesh to at most this many points for speed
_ICP_MAX_MESH_PTS = 2_000
# ICP: require at least this many object points to attempt ICP
_ICP_MIN_OBJ_PTS = 32
# ICP: max translation allowed from initial pose (metres) — clamps runaway solutions
_ICP_MAX_TRANSLATION = 2.0
_SURFACE_MESH_EXTS = (".glb", ".gltf", ".obj", ".stl", ".off")
_FALLBACK_MESH_EXTS = (".ply",)


def _preferred_sam3d_mesh_path(sam3d_entry: dict[str, Any] | None) -> str:
    if not isinstance(sam3d_entry, dict):
        return ""

    candidates: list[str] = []
    for file_entry in sam3d_entry.get("files") or []:
        if not isinstance(file_entry, dict):
            continue
        path = str(file_entry.get("path", "")).strip()
        if path:
            candidates.append(path)

    mesh_path = str(sam3d_entry.get("mesh_path", "")).strip()
    if mesh_path:
        candidates.append(mesh_path)

    seen: set[str] = set()
    ordered: list[str] = []
    for suffixes in (_SURFACE_MESH_EXTS, _FALLBACK_MESH_EXTS):
        for path in candidates:
            lowered = path.lower()
            if path in seen or not lowered.endswith(suffixes):
                continue
            seen.add(path)
            ordered.append(path)

    for path in candidates:
        if path not in seen:
            ordered.append(path)

    for path in ordered:
        if Path(path).exists():
            return path
    return ""


class ObjectSceneAlignmentRefiner:
    def __init__(
        self,
        *,
        alignment_backend: str = "depth_icp",
        visual_pose_tracker: VisualPoseTracker | None = None,
        visual_pose_min_score: float = 0.0,
        visual_pose_max_translation_step_m: float = 1.5,
        visual_pose_max_rotation_step_deg: float = 60.0,
    ) -> None:
        self.alignment_backend = str(alignment_backend or "depth_icp").strip().lower()
        self.visual_pose_tracker = visual_pose_tracker
        self.visual_pose_min_score = float(visual_pose_min_score)
        self.visual_pose_max_translation_step_m = float(visual_pose_max_translation_step_m)
        self.visual_pose_max_rotation_step_deg = float(visual_pose_max_rotation_step_deg)

    def refine(self, objects: list[ObjectNode], pit_snapshot: dict) -> tuple[list[ObjectNode], dict]:
        objects_copy = [obj.model_copy(deep=True) for obj in objects]
        snapshot_copy = copy.deepcopy(pit_snapshot)
        object_tracks = snapshot_copy.get("object_trajectories", {})
        if not isinstance(object_tracks, dict):
            return objects_copy, snapshot_copy

        objects_by_id = {obj.object_id: obj for obj in objects_copy}
        for object_id, track in object_tracks.items():
            obj = objects_by_id.get(object_id)
            if obj is None or not track:
                continue
            if not self._is_scene_alignable(obj, snapshot_copy):
                continue

            seed = track[-1].get("centroid_world", obj.geometry.pose_3d.position)
            if not _is_vec3(seed) or not _is_vec3(obj.geometry.scale_3d):
                continue
            mesh_basis = self._mesh_alignment_basis(snapshot_copy, object_id)

            icp_result = self._icp_align(snapshot_copy, object_id, mesh_basis)
            if icp_result is not None:
                refined_center, refined_scale, refined_yaw = icp_result
                logger.info(
                    "[Alignment] %s: ICP succeeded — center=%s scale=%s yaw=%.3f",
                    object_id,
                    refined_center,
                    refined_scale,
                    refined_yaw,
                )
            else:
                continue

            offset = [refined_center[i] - float(seed[i]) for i in range(3)]

            for rec in track:
                centroid = rec.get("centroid_world")
                if not _is_vec3(centroid):
                    continue
                aligned_centroid = [float(centroid[i]) + offset[i] for i in range(3)]
                rec["centroid_world"] = aligned_centroid
                rec["bbox_3d_aabb"] = self._aabb_for_center_scale(aligned_centroid, refined_scale)

            obj.geometry.pose_3d.position = list(track[-1]["centroid_world"])
            obj.geometry.scale_3d = refined_scale

        return objects_copy, snapshot_copy

    # ------------------------------------------------------------------
    # Scene-point loading (WildGS static/background helpers)
    # ------------------------------------------------------------------

    def _is_scene_alignable(self, obj: ObjectNode, pit_snapshot: dict) -> bool:
        attrs = (pit_snapshot.get("object_attrs") or {}).get(obj.object_id, {})
        vlm = (pit_snapshot.get("vlm_priors") or {}).get(obj.object_id, {})
        return bool(attrs.get("is_movable", vlm.get("is_movable", False)))

    # ------------------------------------------------------------------
    # Mesh → world-space transform
    # ------------------------------------------------------------------

    def _mesh_alignment_basis(self, pit_snapshot: dict, object_id: str) -> dict | None:
        """Load SAM3D mesh and transform vertices to world space."""
        sam3d_entry = (pit_snapshot.get("sam3d_meshes") or {}).get(object_id)
        mesh_path = _preferred_sam3d_mesh_path(sam3d_entry)
        if not mesh_path or not HAS_TRIMESH:
            return None
        try:
            import numpy as np

            mesh = _trimesh.load(mesh_path, force="mesh")
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if len(verts) < 8:
                return None

            verts = verts - verts.mean(axis=0, keepdims=True)

            R = self._camera_R_for_object(pit_snapshot, object_id)
            if R is not None:
                verts = (R @ verts.T).T

            extents = np.maximum(
                verts.max(axis=0) - verts.min(axis=0),
                np.array([0.05, 0.05, 0.05]),
            )
            yaw = _principal_yaw_xz(verts)

            if len(verts) > _ICP_MAX_MESH_PTS:
                idx = np.linspace(0, len(verts) - 1, _ICP_MAX_MESH_PTS, dtype=int)
                verts_sub = verts[idx]
            else:
                verts_sub = verts

            return {
                "extents": extents.tolist(),
                "yaw": float(yaw),
                "world_vertices": verts_sub,
            }
        except Exception as exc:
            logger.warning("[Alignment] _mesh_alignment_basis failed for %s: %s", object_id, exc)
            return None

    def _camera_R_for_object(self, pit_snapshot: dict, object_id: str) -> Any | None:
        import numpy as np

        sam3d_entry = (pit_snapshot.get("sam3d_meshes") or {}).get(object_id, {})
        frame_idx = sam3d_entry.get("reconstruction_frame_idx")
        cam_traj = pit_snapshot.get("camera_trajectory") or []
        if not cam_traj:
            return None
        if frame_idx is not None:
            pose = next((p for p in cam_traj if p.get("frame_id") == frame_idx), cam_traj[-1])
        else:
            pose = cam_traj[-1]
        R = pose.get("R")
        if R is None:
            return None
        return np.asarray(R, dtype=np.float64)

    def _centroid_world_for_object(self, pit_snapshot: dict, object_id: str) -> list[float] | None:
        obj_traj = (pit_snapshot.get("object_trajectories") or {}).get(object_id, [])
        if not obj_traj:
            return None
        return obj_traj[-1].get("centroid_world")

    # ------------------------------------------------------------------
    # Object point cloud from depth map + mask
    # ------------------------------------------------------------------

    def _build_object_point_cloud(self, pit_snapshot: dict, object_id: str) -> Any | None:
        """Unproject WildGS depth map pixels within the object mask to world space."""
        depth_maps_dir = pit_snapshot.get("wildgs_depth_maps_dir")
        if not depth_maps_dir:
            return None

        sam3d_entry = (pit_snapshot.get("sam3d_meshes") or {}).get(object_id, {})
        frame_idx = int(sam3d_entry.get("reconstruction_frame_idx", 0) or 0)
        mask_rle = sam3d_entry.get("mask_rle")
        if not mask_rle:
            return None
        cam_traj = pit_snapshot.get("camera_trajectory") or []
        pts_world = build_depth_point_cloud(
            depth_maps_dir,
            frame_idx,
            mask_rle,
            cam_traj=cam_traj,
        )
        if pts_world is None:
            logger.debug("[Alignment] Failed to build depth point cloud for %s", object_id)
        return trim_point_cloud_outliers(pts_world, min_keep=_ICP_MIN_OBJ_PTS) if pts_world is not None else None

    # ------------------------------------------------------------------
    # ICP alignment
    # ------------------------------------------------------------------

    def _icp_align(
        self,
        pit_snapshot: dict,
        object_id: str,
        mesh_basis: dict | None,
    ) -> tuple[list[float], list[float], float] | None:
        """Run ICP between world-space mesh and object point cloud from depth+mask."""
        if not HAS_TRIMESH or mesh_basis is None:
            return None

        import numpy as np

        source_pts = mesh_basis.get("world_vertices")
        if source_pts is None or len(source_pts) < 8:
            return None

        target_pts = self._build_object_point_cloud(pit_snapshot, object_id)
        if target_pts is None or len(target_pts) < _ICP_MIN_OBJ_PTS:
            return None

        try:
            source_center = source_pts.mean(axis=0)
            target_center = target_pts.mean(axis=0)
            source_centered = source_pts - source_center
            target_centered = target_pts - target_center
            axis_scale = compute_axis_scale(source_centered, target_centered)
            source_scaled = source_centered * axis_scale[None, :]
            matrix, transformed, cost = _trimesh.registration.icp(
                source_scaled,
                target_centered,
                max_iterations=30,
                threshold=0.005,
            )
            _ = cost
            translation = matrix[:3, 3]
            if np.linalg.norm(translation) > _ICP_MAX_TRANSLATION:
                logger.debug(
                    "[Alignment] ICP translation too large for %s (%.3fm), discarding",
                    object_id,
                    np.linalg.norm(translation),
                )
                return None

            transformed_world = transformed + target_center[None, :]
            center = transformed_world.mean(axis=0).tolist()
            lo = np.percentile(target_pts, 10, axis=0)
            hi = np.percentile(target_pts, 90, axis=0)
            scale = np.maximum(hi - lo, np.array([0.05, 0.05, 0.05])).tolist()
            yaw = _principal_yaw_xz(transformed_world)
            return center, scale, float(yaw)
        except Exception as exc:
            logger.debug("[Alignment] ICP failed for %s: %s", object_id, exc)
            return None

    def _aabb_for_center_scale(self, center: list[float], scale: list[float]) -> dict[str, list[float]]:
        return {
            "min": [float(center[i]) - float(scale[i]) * 0.5 for i in range(3)],
            "max": [float(center[i]) + float(scale[i]) * 0.5 for i in range(3)],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_vec3(values: Any) -> bool:
    if not isinstance(values, list) or len(values) < 3:
        return False
    try:
        return all(math.isfinite(float(v)) for v in values[:3])
    except Exception:
        return False


def _principal_yaw_xz(points: Any) -> float:
    import numpy as np

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return 0.0
    xz = arr[:, [0, 2]]
    xz = xz - xz.mean(axis=0, keepdims=True)
    cov = xz.T @ xz
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return float(math.atan2(axis[1], axis[0]))
