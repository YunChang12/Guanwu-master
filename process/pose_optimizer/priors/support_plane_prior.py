"""Generic local support-plane prior.

This module intentionally treats floors, roads, and tabletops as the same
optional geometric cue. The generic optimizer can use it only when confidence is
high enough; no hard snap/contact gate is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupportPlaneConfig:
    min_points: int = 120
    ransac_iters: int = 96
    ransac_threshold_m: float = 0.05
    min_confidence: float = 0.70
    residual_scale_m: float = 0.08
    random_seed: int = 17


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def fit_support_plane_ransac(
    points: np.ndarray,
    *,
    config: SupportPlaneConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SupportPlaneConfig()
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.ndim == 2 and pts.shape[1] == 3 else np.empty((0, 3))
    if len(pts) < int(cfg.min_points):
        return {
            "available": False,
            "support_plane_confidence": 0.0,
            "reason": "insufficient_points",
            "num_points": int(len(pts)),
        }

    rng = np.random.default_rng(int(cfg.random_seed))
    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_offset: float | None = None
    indices = np.arange(len(pts))
    for _ in range(max(1, int(cfg.ransac_iters))):
        sample_idx = rng.choice(indices, size=3, replace=False)
        a, b, c = pts[sample_idx]
        normal = np.cross(b - a, c - a)
        normal = _normalize(normal)
        if float(np.linalg.norm(normal)) <= 0.0:
            continue
        offset = -float(np.dot(normal, a))
        distances = np.abs(pts @ normal + offset)
        inliers = distances <= float(cfg.ransac_threshold_m)
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_normal = normal
            best_offset = offset

    if best_inliers is None or best_normal is None or best_offset is None or int(best_inliers.sum()) < 3:
        return {
            "available": False,
            "support_plane_confidence": 0.0,
            "reason": "ransac_failed",
            "num_points": int(len(pts)),
        }

    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = _normalize(vh[-1])
    offset = -float(np.dot(normal, centroid))
    distances = np.abs(pts @ normal + offset)
    residual = float(np.median(distances[best_inliers]))
    inlier_ratio = float(best_inliers.sum() / max(1, len(pts)))
    residual_score = float(np.exp(-residual / max(1e-6, float(cfg.residual_scale_m))))
    confidence = float(np.clip(inlier_ratio * residual_score, 0.0, 1.0))
    return {
        "available": confidence >= float(cfg.min_confidence),
        "normal": normal,
        "offset": offset,
        "point": centroid,
        "inlier_ratio": inlier_ratio,
        "plane_residual_m": residual,
        "normal_stability": residual_score,
        "support_plane_confidence": confidence,
        "num_points": int(len(pts)),
        "num_inliers": int(best_inliers.sum()),
    }


def support_contact_score(
    support_points: np.ndarray,
    plane: dict[str, Any],
    *,
    sigma_m: float = 0.08,
) -> dict[str, Any]:
    if not plane or float(plane.get("support_plane_confidence", 0.0)) <= 0.0:
        return {
            "support_contact_score": 0.0,
            "support_contact_mean_abs_m": None,
            "support_contact_max_abs_m": None,
        }
    pts = np.asarray(support_points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.ndim == 2 and pts.shape[1] == 3 else np.empty((0, 3))
    if len(pts) == 0:
        return {
            "support_contact_score": 0.0,
            "support_contact_mean_abs_m": None,
            "support_contact_max_abs_m": None,
        }
    normal = np.asarray(plane["normal"], dtype=np.float64)
    offset = float(plane["offset"])
    distances = np.abs(pts @ normal + offset)
    mean_abs = float(np.mean(distances))
    max_abs = float(np.max(distances))
    score = float(np.exp(-mean_abs / max(1e-6, float(sigma_m))))
    return {
        "support_contact_score": score,
        "support_contact_mean_abs_m": mean_abs,
        "support_contact_max_abs_m": max_abs,
    }

