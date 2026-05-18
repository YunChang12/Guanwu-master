from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import open3d as o3d  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    o3d = None

try:
    import small_gicp  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    small_gicp = None

_ENABLE_SMALL_GICP = os.getenv("SPWM_ENABLE_SMALL_GICP", "").strip().lower() in {"1", "true", "yes", "on"}

HAS_OPEN3D = o3d is not None
HAS_SMALL_GICP = small_gicp is not None and _ENABLE_SMALL_GICP

_MIN_GLOBAL_POINTS = 128
_MIN_GLOBAL_PLANAR_RATIO = 0.08
_MIN_GLOBAL_THICKNESS_RATIO = 0.02
_MIN_GLOBAL_VOXELS = 24
_MIN_SMALL_GICP_POINTS = 192
_MIN_REFINE_PLANAR_RATIO = 0.05
_MIN_REFINE_THICKNESS_RATIO = 0.01
_MIN_SMALL_GICP_VOXELS = 48


@dataclass
class RigidRegistrationResult:
    rotation: np.ndarray
    residual: float
    method: str


def _subsample(points: Any, limit: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] <= limit:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, limit, dtype=int)
    return arr[idx]


def _normalize(points: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if scale < 1e-6:
        scale = 1.0
    return points / scale, scale


def _cloud_shape(points: np.ndarray) -> tuple[float, float, float]:
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 3:
        return 0.0, 0.0, 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    diag = float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)))
    try:
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return diag, 0.0, 0.0
    if singular_values.size == 0 or float(singular_values[0]) < 1e-8:
        return diag, 0.0, 0.0
    planar_ratio = float(singular_values[1] / singular_values[0]) if singular_values.size > 1 else 0.0
    thickness_ratio = float(singular_values[2] / singular_values[0]) if singular_values.size > 2 else 0.0
    return diag, planar_ratio, thickness_ratio


def _supports_registration(
    points: np.ndarray,
    *,
    min_points: int,
    min_planar_ratio: float,
    min_thickness_ratio: float,
) -> bool:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < int(min_points):
        return False
    diag, planar_ratio, thickness_ratio = _cloud_shape(points)
    if diag < 1e-3:
        return False
    if planar_ratio < float(min_planar_ratio):
        return False
    if thickness_ratio < float(min_thickness_ratio):
        return False
    return True


def _approx_voxel_count(points: np.ndarray, resolution: float) -> int:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 3:
        return 0
    step = max(float(resolution), 1e-4)
    coords = np.floor(arr / step).astype(np.int32)
    try:
        return int(np.unique(coords, axis=0).shape[0])
    except TypeError:
        return len({tuple(row.tolist()) for row in coords})


def _orthonormalize(rotation: Any) -> np.ndarray:
    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        return np.eye(3, dtype=np.float64)
    try:
        u, _, vt = np.linalg.svd(rot)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _nearest_neighbor_residual(source: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial import KDTree

    if source.ndim != 2 or target.ndim != 2 or len(source) == 0 or len(target) == 0:
        return float("inf")
    tree = KDTree(target)
    dists, _ = tree.query(source)
    return float(np.mean(dists))


def _to_o3d_pcd(points: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def _estimate_voxel_size(source: np.ndarray, target: np.ndarray) -> float:
    diag = max(
        float(np.linalg.norm(source.max(axis=0) - source.min(axis=0))),
        float(np.linalg.norm(target.max(axis=0) - target.min(axis=0))),
    )
    return max(diag / 24.0, 0.03)


def _preprocess_o3d(points: np.ndarray, voxel_size: float):
    cloud = _to_o3d_pcd(points)
    down = cloud.voxel_down_sample(voxel_size)
    if len(down.points) < 16:
        down = cloud
    radius_normal = max(voxel_size * 2.0, 0.05)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = max(voxel_size * 5.0, 0.1)
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return down, feat


def _open3d_global_candidates(source: np.ndarray, target: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if not HAS_OPEN3D:
        return []
    if not _supports_registration(
        source,
        min_points=_MIN_GLOBAL_POINTS,
        min_planar_ratio=_MIN_GLOBAL_PLANAR_RATIO,
        min_thickness_ratio=_MIN_GLOBAL_THICKNESS_RATIO,
    ):
        return []
    if not _supports_registration(
        target,
        min_points=_MIN_GLOBAL_POINTS,
        min_planar_ratio=_MIN_GLOBAL_PLANAR_RATIO,
        min_thickness_ratio=_MIN_GLOBAL_THICKNESS_RATIO,
    ):
        return []

    voxel = _estimate_voxel_size(source, target)
    if _approx_voxel_count(source, voxel) < _MIN_GLOBAL_VOXELS:
        return []
    if _approx_voxel_count(target, voxel) < _MIN_GLOBAL_VOXELS:
        return []
    src_down, src_feat = _preprocess_o3d(source, voxel)
    tgt_down, tgt_feat = _preprocess_o3d(target, voxel)
    if min(len(src_down.points), len(tgt_down.points)) < 24:
        return []

    candidates: list[tuple[str, np.ndarray]] = []
    try:
        fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down,
            tgt_down,
            src_feat,
            tgt_feat,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=max(voxel * 1.5, 0.08)
            ),
        )
        candidates.append(("open3d_fgr", np.asarray(fgr.transformation, dtype=np.float64)[:3, :3]))
    except Exception:
        pass

    try:
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_feat,
            tgt_feat,
            True,
            max(voxel * 1.5, 0.08),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max(voxel * 1.5, 0.08)),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
        )
        candidates.append(("open3d_ransac", np.asarray(ransac.transformation, dtype=np.float64)[:3, :3]))
    except Exception:
        pass

    return candidates


def _small_gicp_refine(source: np.ndarray, target: np.ndarray, init_rotation: np.ndarray) -> np.ndarray | None:
    if not HAS_SMALL_GICP:
        return None
    if not _supports_registration(
        source,
        min_points=_MIN_SMALL_GICP_POINTS,
        min_planar_ratio=_MIN_REFINE_PLANAR_RATIO,
        min_thickness_ratio=_MIN_REFINE_THICKNESS_RATIO,
    ):
        return None
    if not _supports_registration(
        target,
        min_points=_MIN_SMALL_GICP_POINTS,
        min_planar_ratio=_MIN_REFINE_PLANAR_RATIO,
        min_thickness_ratio=_MIN_REFINE_THICKNESS_RATIO,
    ):
        return None
    if _approx_voxel_count(source, 0.05) < _MIN_SMALL_GICP_VOXELS:
        return None
    if _approx_voxel_count(target, 0.05) < _MIN_SMALL_GICP_VOXELS:
        return None
    init = np.eye(4, dtype=np.float64)
    init[:3, :3] = init_rotation
    try:
        result = small_gicp.align(
            target,
            source,
            init_T_target_source=init,
            registration_type="VGICP",
            downsampling_resolution=0.05,
            max_correspondence_distance=0.2,
            num_threads=max(1, min(os.cpu_count() or 1, 8)),
        )
        rot = np.asarray(result.T_target_source, dtype=np.float64)[:3, :3]
        return _orthonormalize(rot)
    except Exception:
        return None


def _open3d_icp_refine(source: np.ndarray, target: np.ndarray, init_rotation: np.ndarray) -> np.ndarray | None:
    if not HAS_OPEN3D:
        return None
    src = _to_o3d_pcd(source)
    tgt = _to_o3d_pcd(target)
    init = np.eye(4, dtype=np.float64)
    init[:3, :3] = init_rotation
    try:
        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            0.2,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        return _orthonormalize(np.asarray(result.transformation, dtype=np.float64)[:3, :3])
    except Exception:
        return None


def estimate_rigid_registration(source_points: Any, target_points: Any, *, init_rotation: Any | None = None) -> RigidRegistrationResult | None:
    src = _subsample(source_points, 3000)
    tgt = _subsample(target_points, 4000)
    if src.ndim != 2 or tgt.ndim != 2 or min(len(src), len(tgt)) < 16:
        return None

    src_centered = src - src.mean(axis=0, keepdims=True)
    tgt_centered = tgt - tgt.mean(axis=0, keepdims=True)
    src_norm, _ = _normalize(src_centered)
    tgt_norm, _ = _normalize(tgt_centered)

    candidates: list[tuple[str, np.ndarray]] = []
    if init_rotation is not None:
        candidates.append(("seed", _orthonormalize(init_rotation)))
    candidates.extend(_open3d_global_candidates(src_norm, tgt_norm))
    if not candidates:
        return None

    best: RigidRegistrationResult | None = None
    for name, rotation in candidates:
        refined = _small_gicp_refine(src_norm, tgt_norm, rotation)
        if refined is None:
            refined = _open3d_icp_refine(src_norm, tgt_norm, rotation)
        if refined is None:
            refined = _orthonormalize(rotation)
        aligned = (refined @ src_norm.T).T
        residual = _nearest_neighbor_residual(aligned, tgt_norm)
        result = RigidRegistrationResult(rotation=refined, residual=residual, method=name)
        if best is None or result.residual < best.residual:
            best = result
    return best
