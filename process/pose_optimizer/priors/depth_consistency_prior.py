"""Observed/rendered depth consistency prior for generic pose scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthConsistencyConfig:
    depth_sigma: float = 0.50
    min_valid_ratio: float = 0.25
    robust_stat: str = "median"
    eps: float = 1e-6


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def load_depth_map(path: str | Path) -> np.ndarray:
    """Load a metric depth map from common numpy/image formats."""
    depth_path = Path(path)
    if not depth_path.exists():
        raise FileNotFoundError(f"Observed depth map does not exist: {depth_path}")
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path)
    elif suffix == ".npz":
        data = np.load(depth_path)
        if "depth" in data:
            depth = data["depth"]
        else:
            depth = data[data.files[0]]
    else:
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read observed depth map: {depth_path}")
        depth = raw.astype(np.float32)
        if np.issubdtype(raw.dtype, np.integer):
            # 16-bit depth maps are conventionally millimetres. 8-bit maps are
            # ambiguous, so keep their numeric values unless they exceed metric
            # depth ranges.
            if raw.dtype == np.uint16 or float(np.nanmax(depth)) > 255.0:
                depth = depth / 1000.0
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return np.asarray(depth, dtype=np.float32)


class DepthConsistencyPrior:
    """Scores candidates using robust absolute render/observed depth error."""

    def __init__(
        self,
        observed_depth: np.ndarray,
        detection_mask: np.ndarray,
        *,
        config: DepthConsistencyConfig | None = None,
    ) -> None:
        self.config = config or DepthConsistencyConfig()
        self.observed_depth = np.asarray(observed_depth, dtype=np.float32)
        self.detection_mask = _as_binary_mask(detection_mask)
        if self.observed_depth.shape != self.detection_mask.shape:
            raise ValueError(
                "observed_depth and detection_mask must have the same HxW shape: "
                f"{self.observed_depth.shape!r} vs {self.detection_mask.shape!r}"
            )

    def score(
        self,
        render_depth: np.ndarray | None,
        render_mask: np.ndarray,
        visible_region: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if render_depth is None:
            return self._disabled("render_depth_unavailable")

        rendered_depth = np.asarray(render_depth, dtype=np.float32)
        rendered_mask = _as_binary_mask(render_mask).astype(bool)
        if rendered_depth.shape != self.observed_depth.shape:
            return self._disabled("shape_mismatch")

        valid_observed = np.isfinite(self.observed_depth) & (self.observed_depth > 0.0)
        valid_rendered = np.isfinite(rendered_depth) & (rendered_depth > 0.0)
        valid = rendered_mask & self.detection_mask.astype(bool) & valid_observed & valid_rendered
        if visible_region is not None:
            valid &= np.asarray(visible_region).astype(bool)

        denom_mask = self.detection_mask.astype(bool)
        if visible_region is not None:
            denom_mask &= np.asarray(visible_region).astype(bool)
        denom = max(1, int(denom_mask.sum()))
        valid_count = int(valid.sum())
        valid_ratio = float(valid_count / denom)
        if valid_count <= 0:
            return {
                "depth_score": 0.0,
                "depth_confidence": 0.0,
                "depth_error": None,
                "valid_depth_ratio": valid_ratio,
                "debug": {"reason": "no_valid_depth", "valid_count": valid_count, "denominator": denom},
            }

        errors = np.abs(rendered_depth[valid] - self.observed_depth[valid]).astype(np.float32)
        if self.config.robust_stat == "mean":
            depth_error = float(np.mean(errors))
        else:
            depth_error = float(np.median(errors))
        if valid_ratio < float(self.config.min_valid_ratio):
            confidence = float(np.clip(valid_ratio / max(self.config.eps, self.config.min_valid_ratio), 0.0, 1.0))
            return {
                "depth_score": 0.0,
                "depth_confidence": confidence,
                "depth_error": depth_error,
                "valid_depth_ratio": valid_ratio,
                "debug": {
                    "reason": "valid_ratio_below_threshold",
                    "valid_count": valid_count,
                    "denominator": denom,
                    "min_valid_ratio": float(self.config.min_valid_ratio),
                },
            }

        depth_score = float(np.exp(-depth_error / max(self.config.eps, float(self.config.depth_sigma))))
        return {
            "depth_score": depth_score,
            "depth_confidence": 1.0,
            "depth_error": depth_error,
            "valid_depth_ratio": valid_ratio,
            "debug": {"valid_count": valid_count, "denominator": denom},
        }

    @staticmethod
    def _disabled(reason: str) -> dict[str, Any]:
        return {
            "depth_score": 0.0,
            "depth_confidence": 0.0,
            "depth_error": None,
            "valid_depth_ratio": 0.0,
            "debug": {"reason": reason},
        }


def render_depth_by_triangle_zbuffer(
    projected_uv: np.ndarray,
    points_cam_z: np.ndarray,
    faces: np.ndarray,
    image_size: tuple[int, int],
    *,
    batch_size: int = 20000,
) -> np.ndarray:
    """Approximate render depth using a per-triangle z-buffer fill.

    The implementation uses the median vertex depth for each triangle. It is a
    lightweight consistency prior, not a photorealistic renderer.
    """
    width, height = int(image_size[0]), int(image_size[1])
    depth = np.full((height, width), np.inf, dtype=np.float32)
    valid_z = np.isfinite(points_cam_z) & (points_cam_z > 0.0)
    valid_faces = valid_z[faces].all(axis=1)
    if not np.any(valid_faces):
        return np.zeros((height, width), dtype=np.float32)

    triangles = np.asarray(projected_uv[faces[valid_faces]], dtype=np.float32)
    triangle_depths = np.median(np.asarray(points_cam_z[faces[valid_faces]], dtype=np.float32), axis=1)
    finite = np.isfinite(triangles).all(axis=(1, 2)) & np.isfinite(triangle_depths) & (triangle_depths > 0.0)
    triangles = triangles[finite]
    triangle_depths = triangle_depths[finite]
    if len(triangles) == 0:
        return np.zeros((height, width), dtype=np.float32)

    tri_min = triangles.min(axis=1)
    tri_max = triangles.max(axis=1)
    intersects = (
        (tri_max[:, 0] >= 0)
        & (tri_max[:, 1] >= 0)
        & (tri_min[:, 0] < width)
        & (tri_min[:, 1] < height)
    )
    triangles = triangles[intersects]
    triangle_depths = triangle_depths[intersects]
    if len(triangles) == 0:
        return np.zeros((height, width), dtype=np.float32)

    order = np.argsort(triangle_depths)[::-1]
    triangles_i = np.rint(triangles[order]).astype(np.int32)
    triangle_depths = triangle_depths[order]
    triangles_i[:, :, 0] = np.clip(triangles_i[:, :, 0], -width * 2, width * 3)
    triangles_i[:, :, 1] = np.clip(triangles_i[:, :, 1], -height * 2, height * 3)

    scratch = np.zeros((height, width), dtype=np.uint8)
    for start in range(0, len(triangles_i), batch_size):
        end = min(len(triangles_i), start + batch_size)
        for tri, tri_depth in zip(triangles_i[start:end], triangle_depths[start:end]):
            scratch.fill(0)
            cv2.fillPoly(scratch, [tri], color=1)
            update = scratch.astype(bool) & (tri_depth < depth)
            depth[update] = float(tri_depth)
    depth[~np.isfinite(depth)] = 0.0
    return depth

