from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


CV_CAMERA_FROM_USD_CAMERA = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return _normalize(np.asarray(fallback, dtype=np.float64).reshape(3))
    return arr / norm


def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    n = _normalize(normal)
    return arr - float(arr @ n) * n


def _camera_pose_matrix(entry: Any) -> np.ndarray | None:
    if entry is None:
        return None
    if isinstance(entry, dict) and entry.get("T_world_from_cam") is not None:
        T = np.asarray(entry["T_world_from_cam"], dtype=np.float64)
        return T if T.shape == (4, 4) else None
    if isinstance(entry, dict) and entry.get("R") is not None:
        R = np.asarray(entry["R"], dtype=np.float64)
        if R.shape != (3, 3):
            return None
        t = np.asarray(entry.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    return None


def _camera_forward_world(entry: Any, ground_normal: np.ndarray) -> np.ndarray | None:
    T = _camera_pose_matrix(entry)
    if T is None:
        return None
    forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = _project_to_plane(forward, ground_normal)
    if float(np.linalg.norm(forward)) < 1e-6:
        return None
    return _normalize(forward)


def _fallback_forward(ground_normal: np.ndarray) -> np.ndarray:
    candidates = [
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    best = None
    best_norm = -1.0
    for candidate in candidates:
        projected = _project_to_plane(candidate, ground_normal)
        norm = float(np.linalg.norm(projected))
        if norm > best_norm:
            best = projected
            best_norm = norm
    if best is None or best_norm < 1e-6:
        best = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _normalize(best)


def estimate_ground_plane(
    points: np.ndarray | None,
    reference_up: np.ndarray | None = None,
    offset_quantile: float = 1.0,
) -> tuple[np.ndarray, float | None]:
    up_ref = _normalize(
        np.asarray(reference_up, dtype=np.float64).reshape(3)
        if reference_up is not None
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    if points is None:
        return up_ref, None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 3:
        return up_ref, None
    centered = arr - arr.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return up_ref, None
    normal = np.asarray(vh[-1], dtype=np.float64)
    normal = _normalize(normal, fallback=up_ref)
    if float(normal @ up_ref) < 0.0:
        normal = -normal
    offset = float(np.percentile(arr @ normal, offset_quantile))
    return normal, offset


@dataclass
class USDCoordinateConvention:
    R_usd_from_world: np.ndarray
    scene_up_world: np.ndarray
    scene_forward_world: np.ndarray
    ground_plane_offset_world: float | None = None
    stage_up_axis: str = "Z"
    camera_basis_cv_from_usd: np.ndarray = field(default_factory=lambda: CV_CAMERA_FROM_USD_CAMERA.copy())

    def __post_init__(self) -> None:
        self.R_usd_from_world = np.asarray(self.R_usd_from_world, dtype=np.float64).reshape(3, 3)
        self.scene_up_world = _normalize(self.scene_up_world)
        self.scene_forward_world = _normalize(self.scene_forward_world)
        self.camera_basis_cv_from_usd = np.asarray(self.camera_basis_cv_from_usd, dtype=np.float64).reshape(3, 3)


def build_world_to_usd_basis(
    scene_up: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    camera_track: list[dict] | None = None,
    bg_points: np.ndarray | None = None,
) -> USDCoordinateConvention:
    up_ref = _normalize(
        np.asarray(scene_up, dtype=np.float64).reshape(3)
        if scene_up is not None
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    ground_normal, plane_offset = estimate_ground_plane(bg_points, reference_up=up_ref)

    forward_samples: list[np.ndarray] = []
    for entry in camera_track or []:
        forward = _camera_forward_world(entry, ground_normal)
        if forward is not None:
            if forward_samples and float(forward @ forward_samples[0]) < 0.0:
                forward = -forward
            forward_samples.append(forward)
    if forward_samples:
        scene_forward_world = _normalize(np.mean(np.stack(forward_samples, axis=0), axis=0), fallback=forward_samples[0])
    else:
        scene_forward_world = _fallback_forward(ground_normal)

    x_world = np.cross(scene_forward_world, ground_normal)
    if float(np.linalg.norm(x_world)) < 1e-6:
        scene_forward_world = _fallback_forward(ground_normal)
        x_world = np.cross(scene_forward_world, ground_normal)
    x_world = _normalize(x_world, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_world = _normalize(np.cross(ground_normal, x_world), fallback=scene_forward_world)
    z_world = ground_normal

    basis_world_from_usd = np.stack([x_world, y_world, z_world], axis=1)
    if float(np.linalg.det(basis_world_from_usd)) < 0.0:
        basis_world_from_usd[:, 0] *= -1.0
    return USDCoordinateConvention(
        R_usd_from_world=basis_world_from_usd.T,
        scene_up_world=z_world,
        scene_forward_world=y_world,
        ground_plane_offset_world=plane_offset,
    )


def convert_world_points_to_usd(points: np.ndarray, convention: USDCoordinateConvention) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return arr.copy()
    return arr @ convention.R_usd_from_world.T


def convert_world_normals_to_usd(normals: np.ndarray, convention: USDCoordinateConvention) -> np.ndarray:
    arr = np.asarray(normals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return arr.copy()
    return arr @ convention.R_usd_from_world.T


def convert_world_pose_to_usd(
    rotation: np.ndarray | list[list[float]] | None,
    translation: np.ndarray | list[float] | tuple[float, float, float] | None,
    convention: USDCoordinateConvention,
) -> tuple[np.ndarray, np.ndarray]:
    rot = np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trans = (
        np.zeros(3, dtype=np.float64)
        if translation is None
        else np.asarray(translation, dtype=np.float64).reshape(3)
    )
    rot_usd = convention.R_usd_from_world @ rot
    trans_usd = convention.R_usd_from_world @ trans
    return rot_usd, trans_usd


def convert_cv_camera_pose_to_usd(T_world_from_cam: np.ndarray | list[list[float]], convention: USDCoordinateConvention) -> np.ndarray:
    T = np.asarray(T_world_from_cam, dtype=np.float64).reshape(4, 4)
    rot_world = T[:3, :3]
    trans_world = T[:3, 3]
    rot_usd = convention.R_usd_from_world @ rot_world @ convention.camera_basis_cv_from_usd
    trans_usd = convention.R_usd_from_world @ trans_world
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot_usd
    out[:3, 3] = trans_usd
    return out


def build_coordinate_report(
    convention: USDCoordinateConvention,
    camera_track: list[dict] | None = None,
) -> dict[str, Any]:
    camera_heights: list[float] = []
    camera_forward_ground_dots: list[float] = []
    for entry in camera_track or []:
        T = _camera_pose_matrix(entry)
        if T is None:
            continue
        t_world = T[:3, 3]
        if convention.ground_plane_offset_world is not None:
            height = float(t_world @ convention.scene_up_world) - float(convention.ground_plane_offset_world)
            camera_heights.append(height)
        forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        camera_forward_ground_dots.append(float(_normalize(forward_world) @ convention.scene_up_world))

    def _stats(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

    up_usd = convention.R_usd_from_world @ convention.scene_up_world
    forward_usd = convention.R_usd_from_world @ convention.scene_forward_world
    proper_rotation = bool(np.allclose(convention.R_usd_from_world.T @ convention.R_usd_from_world, np.eye(3), atol=1e-6))
    proper_rotation = proper_rotation and bool(abs(float(np.linalg.det(convention.R_usd_from_world)) - 1.0) < 1e-6)
    up_axis_aligned = bool(np.allclose(up_usd, np.array([0.0, 0.0, 1.0], dtype=np.float64), atol=1e-5))
    camera_height_stats = _stats(camera_heights)
    camera_forward_stats = _stats(camera_forward_ground_dots)
    return {
        "stage_up_axis": convention.stage_up_axis,
        "R_usd_from_world": convention.R_usd_from_world.tolist(),
        "camera_basis_cv_from_usd": convention.camera_basis_cv_from_usd.tolist(),
        "ground_normal_world": convention.scene_up_world.tolist(),
        "ground_normal_usd": up_usd.tolist(),
        "ground_plane_offset_world": convention.ground_plane_offset_world,
        "scene_forward_world": convention.scene_forward_world.tolist(),
        "scene_forward_usd": forward_usd.tolist(),
        "camera_height_stats": camera_height_stats,
        "camera_forward_vs_ground": camera_forward_stats,
        "sanity_checks": {
            "proper_rotation": proper_rotation,
            "up_axis_aligned": up_axis_aligned,
            "camera_height_positive": bool(camera_height_stats is None or camera_height_stats["min"] > 0.0),
            "camera_not_parallel_to_ground": bool(
                camera_forward_stats is None or max(abs(camera_forward_stats["min"]), abs(camera_forward_stats["max"])) < 0.98
            ),
        },
    }
