#!/usr/bin/env python3
"""Generic image-appearance/depth/temporal pose optimizer.

This strategy keeps the road/vehicle-heavy temporal optimizer intact and adds a
separate generic mode. It uses silhouette, bbox, contour, real-image
foreground/background appearance, optional depth consistency, and SE(3)
temporal smoothness. Mesh colors/textures are never used as image supervision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..priors.depth_consistency_prior import (
    DepthConsistencyConfig,
    DepthConsistencyPrior,
    load_depth_map,
    render_depth_by_triangle_zbuffer,
)
from ..priors.image_appearance_prior import AppearancePriorConfig, ImageAppearancePrior, build_image_appearance_prior
from ..priors.support_plane_prior import SupportPlaneConfig, fit_support_plane_ransac, support_contact_score
from . import fast
from . import temporal_fast


GENERIC_CANDIDATE_METRIC_KEYS = [
    "score",
    "mask_iou",
    "soft_mask_iou",
    "mask_blend_score",
    "bbox_iou",
    "bbox_center_error_px",
    "full_bbox_iou",
    "full_bbox_center_error_px",
    "full_projected_bbox",
    "bbox_score_source",
    "visible_bbox_iou",
    "visible_bbox_center_error_px",
    "visible_projected_bbox",
    "visible_target_bbox",
    "visible_bbox_source",
    "contour_score",
    "edge_score",
    "edge_confidence",
    "depth_score",
    "depth_confidence",
    "depth_error",
    "valid_depth_ratio",
    "appearance_score",
    "appearance_confidence",
    "color_soft_iou",
    "color_precision",
    "color_recall",
    "background_leakage",
    "fg_bg_distance",
    "temporal_score",
    "generic_temporal_loss",
    "scale_prior_score",
    "optional_prior_score",
    "support_plane_confidence",
    "support_plane_enabled",
    "support_plane_disable_reason",
    "support_plane_inlier_ratio",
    "support_plane_residual_m",
    "support_contact_score",
    "support_contact_distance_score",
    "support_contact_coverage",
    "support_contact_mean_abs_m",
    "support_contact_max_abs_m",
    "support_bottom_selection_mode",
    "support_axis_index",
    "support_axis_sign",
    "support_normal_alignment",
    "support_normal_angle_deg",
    "support_orientation_score",
    "support_orientation_penalty",
    "support_orientation_penalty_eff",
    "support_bottom_point_count",
    "support_bottom_mean_abs_m",
    "support_bottom_max_abs_m",
    "support_bottom_signed_m",
    "support_floating_distance_m",
    "support_penetration_distance_m",
    "support_floating_penalty",
    "support_penetration_penalty",
    "support_penalty",
    "support_contact_penalty_eff",
    "upright_confidence",
    "heading_confidence",
    "observation_score",
    "observation_quality",
    "optional_prior_gate",
    "invalid_projection_penalty",
    "depth_outlier_penalty",
    "temporal_jump_penalty",
    "projection_valid_ratio",
    "visible_ratio",
    "truncation_ratio",
    "acceptance_status",
    "reject_reasons",
    "projected_bbox",
]


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def compute_generic_temporal_score(
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    prior: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """SE(3) temporal score without vehicle yaw-specific terms."""

    prior_translation = np.asarray(prior["translation_cam"], dtype=np.float64)
    prior_rotation = np.asarray(prior["rotation_cam"], dtype=np.float64)
    prior_scale = fast.scale_to_uniform_scalar(np.asarray(prior["scale"], dtype=np.float64))

    translation = np.asarray(translation_cam, dtype=np.float64)
    rotation = np.asarray(rotation_cam, dtype=np.float64)
    scale_value = fast.scale_to_uniform_scalar(np.asarray(scale, dtype=np.float64))

    delta_translation_vec = translation - prior_translation
    delta_translation = float(np.linalg.norm(delta_translation_vec))
    delta_depth = float(abs(delta_translation_vec[2]))
    delta_rotation = rotation @ prior_rotation.T
    rotation_geodesic_deg = temporal_fast.rotation_angle_deg(delta_rotation)
    delta_scale_log = float(math.log(max(1e-8, scale_value) / max(1e-8, prior_scale)))

    sigma_translation = max(1e-6, float(args.generic_temporal_translation_sigma))
    sigma_depth = max(1e-6, float(args.generic_temporal_depth_sigma))
    sigma_rotation = max(1e-6, float(args.generic_temporal_rotation_sigma_deg))
    sigma_scale = max(1e-6, float(args.generic_temporal_scale_sigma_log))

    loss = (
        (delta_translation / sigma_translation) ** 2
        + (delta_depth / sigma_depth) ** 2
        + (rotation_geodesic_deg / sigma_rotation) ** 2
        + (delta_scale_log / sigma_scale) ** 2
    )
    score = float(math.exp(-min(60.0, loss)))
    scale_ratio = max(scale_value, prior_scale) / max(1e-8, min(scale_value, prior_scale))
    return {
        "delta_translation": delta_translation_vec,
        "delta_translation_norm": delta_translation,
        "delta_depth": delta_depth,
        "rotation_geodesic_deg": rotation_geodesic_deg,
        "delta_rotation_deg": rotation_geodesic_deg,
        "delta_scale_log": delta_scale_log,
        "scale_ratio_from_anchor": float(scale_ratio),
        "generic_temporal_loss": float(loss),
        "generic_temporal_score": score,
        "temporal_loss": float(loss),
        "temporal_score": score,
    }


def contour_score(rendered_mask: np.ndarray, target_mask: np.ndarray, sigma_px: float) -> dict[str, Any]:
    rendered_edge = temporal_fast._mask_contour(rendered_mask)
    target_edge = temporal_fast._mask_contour(target_mask)
    if not np.any(rendered_edge) or not np.any(target_edge):
        return {"contour_score": 0.0, "contour_mean_distance_px": None}
    target_distance = cv2.distanceTransform(np.where(target_edge, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
    rendered_distance = cv2.distanceTransform(np.where(rendered_edge, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
    rendered_to_target = target_distance[rendered_edge].astype(np.float32)
    target_to_rendered = rendered_distance[target_edge].astype(np.float32)
    mean_distance = float(
        0.5
        * (
            np.clip(rendered_to_target, 0.0, 80.0).mean()
            + np.clip(target_to_rendered, 0.0, 80.0).mean()
        )
    )
    return {
        "contour_score": float(math.exp(-mean_distance / max(1e-6, float(sigma_px)))),
        "contour_mean_distance_px": mean_distance,
    }


def projection_valid_ratio(points_cam: np.ndarray, projected_uv: np.ndarray, valid_z: np.ndarray, image_size: tuple[int, int]) -> float:
    if len(points_cam) <= 0:
        return 0.0
    width, height = image_size
    inside = (
        valid_z
        & np.isfinite(projected_uv).all(axis=1)
        & (projected_uv[:, 0] >= 0)
        & (projected_uv[:, 0] < float(width))
        & (projected_uv[:, 1] >= 0)
        & (projected_uv[:, 1] < float(height))
    )
    return float(np.count_nonzero(inside) / max(1, len(points_cam)))


def promote_truncated_visible_bbox_score(result: dict[str, Any], visible_bbox: dict[str, Any]) -> None:
    """Use the in-frame rendered silhouette bbox as the generic truncated bbox score."""

    visible_iou = visible_bbox.get("visible_bbox_iou")
    visible_center = visible_bbox.get("visible_bbox_center_error_px")
    visible_projected = visible_bbox.get("visible_projected_bbox")
    if visible_iou is None or visible_center is None or visible_projected is None:
        return
    result["full_bbox_iou"] = result.get("bbox_iou")
    result["full_bbox_center_error_px"] = result.get("bbox_center_error_px")
    result["full_projected_bbox"] = result.get("projected_bbox")
    result["bbox_iou"] = float(visible_iou)
    result["bbox_center_error_px"] = float(visible_center)
    result["projected_bbox"] = list(visible_projected)
    result["bbox_score_source"] = str(visible_bbox.get("visible_bbox_source") or "visible_projected_bbox")


def _visible_ratio(rendered_mask: np.ndarray, visible_region: np.ndarray | None) -> float:
    rendered = np.asarray(rendered_mask) > 0
    area = int(rendered.sum())
    if area <= 0:
        return 0.0
    if visible_region is None:
        return 1.0
    return float(np.logical_and(rendered, np.asarray(visible_region).astype(bool)).sum() / area)


def _visible_region_from_truncation(
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray | None:
    if not truncation_info.get("is_truncated"):
        return None
    try:
        return temporal_fast._visible_region_mask(image_size, truncation_info, args)
    except Exception:
        return None


def support_bottom_points_for_pose(
    vertices: np.ndarray,
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
    scale: np.ndarray,
    plane: dict[str, Any],
    *,
    bottom_percentile: float = 3.0,
    mode: str = "local_axis",
    max_points: int = 512,
) -> dict[str, Any]:
    """Select support-contact vertices from the object's local bottom band.

    Using signed distance alone can pick only the already-lowest edge of a
    tilted object. The local-axis mode keeps the whole canonical bottom band, so
    tilt turns into lower contact coverage instead of being hidden by sampling.
    """

    local = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(rotation_cam, dtype=np.float64)
    translation = np.asarray(translation_cam, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64).reshape(1, 3)
    normal = _normalize(np.asarray(plane.get("normal"), dtype=np.float64))
    offset = float(plane.get("offset", 0.0))
    scaled_local = local * scale_arr
    all_cam = scaled_local @ rotation.T + translation.reshape(1, 3)
    signed_dist = all_cam @ normal + offset
    mode_value = str(mode or "local_axis").lower()
    bottom_percentile = float(bottom_percentile)

    if mode_value in {"signed_distance", "plane_distance"}:
        support_cutoff = float(np.percentile(signed_dist, bottom_percentile))
        selector = signed_dist <= support_cutoff
        support_axis_index = None
        support_axis_sign = None
        alignment = 0.0
        angle_deg = None
    else:
        axis_vectors = [rotation[:, idx] for idx in range(3)]
        signed_alignments = [
            float(np.dot(axis, normal) / max(1e-12, float(np.linalg.norm(axis)) * float(np.linalg.norm(normal))))
            for axis in axis_vectors
        ]
        support_axis_index = int(np.argmax(np.abs(signed_alignments)))
        support_axis_sign = 1.0 if signed_alignments[support_axis_index] >= 0.0 else -1.0
        alignment = abs(float(signed_alignments[support_axis_index]))
        angle_deg = float(math.degrees(math.acos(np.clip(alignment, -1.0, 1.0))))
        local_support_coord = scaled_local[:, support_axis_index] * support_axis_sign
        local_cutoff = float(np.percentile(local_support_coord, bottom_percentile))
        selector = local_support_coord <= local_cutoff
        support_cutoff = float(np.percentile(signed_dist[selector], 50.0)) if np.any(selector) else float(np.percentile(signed_dist, bottom_percentile))

    support_cam = all_cam[selector]
    if len(support_cam) <= 0:
        support_cam = all_cam[signed_dist <= float(np.percentile(signed_dist, bottom_percentile))]
    if len(support_cam) > int(max_points):
        support_cam = support_cam[np.linspace(0, len(support_cam) - 1, int(max_points), dtype=np.int64)]
    return {
        "support_points_cam": support_cam,
        "support_bottom_signed_m": float(support_cutoff),
        "support_bottom_selection_mode": mode_value,
        "support_axis_index": support_axis_index,
        "support_axis_sign": support_axis_sign,
        "support_normal_alignment": float(alignment),
        "support_normal_angle_deg": angle_deg,
        "support_bottom_point_count": int(len(support_cam)),
    }


def support_orientation_score(
    rotation_cam: np.ndarray,
    plane: dict[str, Any],
    *,
    sigma_deg: float = 20.0,
    tolerance_deg: float = 4.0,
) -> dict[str, Any]:
    normal = _normalize(np.asarray(plane.get("normal"), dtype=np.float64))
    if normal.shape != (3,) or float(np.linalg.norm(normal)) <= 1e-12:
        return {
            "support_axis_index": None,
            "support_axis_sign": None,
            "support_normal_alignment": 0.0,
            "support_normal_angle_deg": None,
            "support_orientation_score": 0.0,
            "support_orientation_penalty": 0.0,
        }
    rotation = np.asarray(rotation_cam, dtype=np.float64)
    alignments = []
    for idx in range(3):
        axis = rotation[:, idx]
        alignments.append(float(np.dot(axis, normal) / max(1e-12, float(np.linalg.norm(axis)) * float(np.linalg.norm(normal)))))
    axis_index = int(np.argmax(np.abs(alignments)))
    axis_sign = 1.0 if alignments[axis_index] >= 0.0 else -1.0
    alignment = abs(float(alignments[axis_index]))
    angle_deg = float(math.degrees(math.acos(np.clip(alignment, -1.0, 1.0))))
    excess = max(0.0, angle_deg - float(tolerance_deg))
    sigma = max(1e-6, float(sigma_deg))
    score = float(math.exp(-((excess / sigma) ** 2)))
    return {
        "support_axis_index": axis_index,
        "support_axis_sign": axis_sign,
        "support_normal_alignment": alignment,
        "support_normal_angle_deg": angle_deg,
        "support_orientation_score": score,
        "support_orientation_penalty": float(1.0 - score),
    }


def render_depth_for_pose(
    vertices: np.ndarray,
    faces: np.ndarray,
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
    max_faces: int = 8000,
) -> np.ndarray:
    depth_faces = select_depth_render_faces(np.asarray(faces, dtype=np.int32), max_faces)
    scaled_vertices = np.asarray(vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64).reshape(1, 3)
    points_cam = scaled_vertices @ np.asarray(rotation_cam, dtype=np.float64).T + np.asarray(translation_cam, dtype=np.float64).reshape(1, 3)
    projected_uv, valid_z = fast.project_points(points_cam, **intrinsics)
    return render_depth_by_triangle_zbuffer(projected_uv, points_cam[:, 2], depth_faces, image_size)


def select_depth_render_faces(faces: np.ndarray, max_faces: int | None) -> np.ndarray:
    faces_arr = np.asarray(faces, dtype=np.int32)
    if max_faces is None:
        return faces_arr
    limit = int(max_faces)
    if limit <= 0 or len(faces_arr) <= limit:
        return faces_arr
    indices = np.linspace(0, len(faces_arr) - 1, limit, dtype=np.int64)
    return faces_arr[indices]


def load_observed_depth_for_task(sample_dir: Path, task: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray | None, dict[str, Any]]:
    path_text = str(getattr(args, "observed_depth_map_path", "") or "").strip()
    source = "argument"
    depth_path: Path | None = Path(path_text) if path_text else None
    if depth_path is None:
        context = task.get("generic_pose_context") if isinstance(task.get("generic_pose_context"), dict) else {}
        if not context:
            context = task.get("vehicle_pose_context") if isinstance(task.get("vehicle_pose_context"), dict) else {}
        path_text = str(context.get("depth_map_path") or context.get("observed_depth_map_path") or "").strip()
        if path_text:
            depth_path = Path(path_text)
            source = "task_context"
    if depth_path is None:
        try:
            _, frame_idx = temporal_fast.parse_task_id_from_sample_dir(sample_dir)
            depth_path = fast.find_depth_map_for_task(sample_dir, frame_idx)
            source = "auto"
        except Exception:
            depth_path = None
    if depth_path is None or not depth_path.exists():
        return None, {"available": False, "reason": "depth_map_not_found"}
    try:
        return load_depth_map(depth_path), {"available": True, "path": str(depth_path), "source": source}
    except Exception as exc:
        return None, {"available": False, "path": str(depth_path), "reason": str(exc), "source": source}


def depth_points_cam_from_region(
    depth: np.ndarray | None,
    region: np.ndarray,
    intrinsics: dict[str, float],
    *,
    max_points: int = 6000,
) -> np.ndarray:
    if depth is None:
        return np.empty((0, 3), dtype=np.float64)
    depth_arr = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(region).astype(bool) & np.isfinite(depth_arr) & (depth_arr > 0.0)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(xs) > int(max_points):
        indices = np.linspace(0, len(xs) - 1, int(max_points), dtype=np.int64)
        xs = xs[indices]
        ys = ys[indices]
    z = depth_arr[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (ys.astype(np.float64) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    return np.stack([x, y, z], axis=1)


def _expanded_bbox_region(
    shape: tuple[int, int],
    bbox_xyxy: list[float],
    expand_ratio: float,
) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    margin_x = bw * max(0.0, float(expand_ratio))
    margin_y = bh * max(0.0, float(expand_ratio))
    ix1 = max(0, int(math.floor(x1 - margin_x)))
    iy1 = max(0, int(math.floor(y1 - margin_y)))
    ix2 = min(width, int(math.ceil(x2 + margin_x)))
    iy2 = min(height, int(math.ceil(y2 + margin_y)))
    region = np.zeros((height, width), dtype=bool)
    if ix2 > ix1 and iy2 > iy1:
        region[iy1:iy2, ix1:ix2] = True
    return region


def _odd_kernel(value: int) -> int:
    size = max(1, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _support_sample_region(
    detection_mask: np.ndarray,
    bbox_xyxy: list[float],
    args: argparse.Namespace,
    other_instance_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = (np.asarray(detection_mask) > 0).astype(np.uint8)
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    lower_band = np.zeros((height, width), dtype=bool)
    if bool(getattr(args, "support_use_lower_band", True)):
        x_expand = float(getattr(args, "support_lower_band_x_expand_ratio", 0.20))
        y_extend = float(getattr(args, "support_lower_band_y_extend_ratio", 0.60))
        sx1 = max(0, int(math.floor(x1 - x_expand * bw)))
        sx2 = min(width, int(math.ceil(x2 + x_expand * bw)))
        sy1 = max(0, int(math.floor(y2)))
        sy2 = min(height, int(math.ceil(y2 + y_extend * bh)))
        if sx2 > sx1 and sy2 > sy1:
            lower_band[sy1:sy2, sx1:sx2] = True

    near_ring = np.zeros((height, width), dtype=bool)
    if bool(getattr(args, "support_use_near_mask_ring", True)):
        inner = _odd_kernel(int(getattr(args, "support_near_mask_inner_kernel", 9)))
        outer = max(_odd_kernel(int(getattr(args, "support_near_mask_outer_kernel", 31))), inner + 2)
        if outer % 2 == 0:
            outer += 1
        outer_mask = cv2.dilate(mask, np.ones((outer, outer), dtype=np.uint8), iterations=1).astype(bool)
        inner_mask = cv2.dilate(mask, np.ones((inner, inner), dtype=np.uint8), iterations=1).astype(bool)
        expanded_bbox = _expanded_bbox_region(mask.shape, bbox_xyxy, float(getattr(args, "support_bbox_expand_ratio", 0.20)))
        near_ring = outer_mask & ~inner_mask & expanded_bbox

    sample = lower_band | near_ring
    exclude_kernel = _odd_kernel(int(getattr(args, "support_exclude_target_mask_dilate_kernel", 9)))
    excluded = cv2.dilate(mask, np.ones((exclude_kernel, exclude_kernel), dtype=np.uint8), iterations=1).astype(bool)
    other_pixels = 0
    if bool(getattr(args, "support_exclude_other_instance_masks", True)) and other_instance_masks:
        for other in other_instance_masks:
            other_arr = np.asarray(other)
            if other_arr.shape == mask.shape:
                other_bool = other_arr.astype(bool)
                other_pixels += int(other_bool.sum())
                excluded |= other_bool
    sample &= ~excluded
    debug = {
        "lower_band_pixels": int(lower_band.sum()),
        "near_mask_ring_pixels": int(near_ring.sum()),
        "excluded_target_pixels": int(excluded.sum()),
        "excluded_other_instance_pixels": int(other_pixels),
        "final_pixels": int(sample.sum()),
    }
    debug_regions = {
        "lower_band_region": lower_band,
        "near_mask_ring_region": near_ring,
    }
    return sample, {**debug, **debug_regions}


def estimate_support_plane_from_observed_depth(
    observed_depth: np.ndarray | None,
    detection_mask: np.ndarray,
    bbox_xyxy: list[float],
    intrinsics: dict[str, float],
    args: argparse.Namespace,
    other_instance_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict[str, Any]:
    if observed_depth is None:
        return {"available": False, "support_plane_confidence": 0.0, "reason": "depth_unavailable"}
    if str(getattr(args, "support_plane_enabled", "auto")).lower() in {"0", "false", "off", "disabled", "none"}:
        return {"available": False, "support_plane_confidence": 0.0, "reason": "disabled"}
    support_region, sample_debug = _support_sample_region(detection_mask, bbox_xyxy, args, other_instance_masks)
    if int(support_region.sum()) <= 0:
        return {
            "available": False,
            "support_plane_confidence": 0.0,
            "reason": "empty_support_region",
            "support_sample_debug": sample_debug,
        }
    points = depth_points_cam_from_region(observed_depth, support_region, intrinsics)
    object_points = depth_points_cam_from_region(observed_depth, np.asarray(detection_mask) > 0, intrinsics)
    plane = fit_support_plane_ransac(
        points,
        config=SupportPlaneConfig(
            min_points=int(getattr(args, "support_plane_min_points", 120)),
            ransac_iters=int(getattr(args, "support_plane_ransac_iters", 96)),
            ransac_threshold_m=float(getattr(args, "support_plane_ransac_threshold_m", 0.05)),
            min_confidence=float(getattr(args, "support_plane_min_confidence", 0.70)),
            residual_scale_m=float(getattr(args, "support_plane_residual_scale_m", 0.08)),
        ),
        object_points_cam=object_points,
    )
    if plane.get("normal") is not None and plane.get("offset") is not None:
        depth_arr = np.asarray(observed_depth, dtype=np.float32)
        valid = support_region & np.isfinite(depth_arr) & (depth_arr > 0.0)
        ys, xs = np.nonzero(valid)
        inlier_region = np.zeros_like(support_region, dtype=bool)
        if len(xs) > 0:
            z = depth_arr[ys, xs].astype(np.float64)
            x = (xs.astype(np.float64) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
            y = (ys.astype(np.float64) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
            pts = np.stack([x, y, z], axis=1)
            distances = np.abs(pts @ np.asarray(plane["normal"], dtype=np.float64) + float(plane["offset"]))
            inlier_region[ys, xs] = distances <= float(getattr(args, "support_plane_ransac_threshold_m", 0.05))
        plane["support_inlier_region"] = inlier_region
    plane["support_sample_debug"] = sample_debug
    plane["support_sample_region"] = support_region
    return plane


def support_plane_report_payload(support_plane: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in support_plane.items()
        if key
        not in {
            "support_sample_region",
            "support_inlier_region",
        }
        and not key.endswith("_region")
    }
    sample_debug = payload.get("support_sample_debug")
    if isinstance(sample_debug, dict):
        payload["support_sample_debug"] = {
            key: value
            for key, value in sample_debug.items()
            if not str(key).endswith("_region")
        }
    return payload


def save_support_plane_debug(
    output_dir: Path,
    support_plane: dict[str, Any],
    detection_mask: np.ndarray,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_region = support_plane.get("support_sample_region")
    if sample_region is not None:
        mask = (np.asarray(detection_mask) > 0)
        sample = np.asarray(sample_region).astype(bool)
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        sample_debug = support_plane.get("support_sample_debug") if isinstance(support_plane.get("support_sample_debug"), dict) else {}
        lower = np.asarray(sample_debug.get("lower_band_region")).astype(bool) if sample_debug.get("lower_band_region") is not None else np.zeros_like(sample)
        ring = np.asarray(sample_debug.get("near_mask_ring_region")).astype(bool) if sample_debug.get("near_mask_ring_region") is not None else np.zeros_like(sample)
        vis[lower] = (255, 0, 0)
        vis[ring] = (0, 180, 255)
        vis[sample] = (255, 160, 0)
        vis[mask] = (0, 255, 0)
        path = output_dir / "support_sample_region.png"
        cv2.imwrite(str(path), vis)
        outputs["support_sample_region"] = str(path)
        inlier_path = output_dir / "support_plane_inliers.png"
        inlier_region = support_plane.get("support_inlier_region")
        inlier = np.asarray(inlier_region).astype(bool) if inlier_region is not None else sample
        cv2.imwrite(str(inlier_path), np.where(inlier, 255, 0).astype(np.uint8))
        outputs["support_plane_inliers"] = str(inlier_path)

    contact_debug = {
        key: fast.to_builtin(value)
        for key, value in support_plane_report_payload(support_plane).items()
    }
    if isinstance(contact_debug.get("support_sample_debug"), dict):
        contact_debug["support_sample_debug"] = {
            key: value
            for key, value in contact_debug["support_sample_debug"].items()
            if not str(key).endswith("_region")
        }
    json_path = output_dir / "support_contact_debug.json"
    json_path.write_text(json.dumps(contact_debug, indent=2), encoding="utf-8")
    outputs["support_contact_debug"] = str(json_path)
    return outputs


class GenericPoseEvaluator(fast.CameraPoseEvaluator):
    """CameraPoseEvaluator with generic observation, appearance, depth, temporal terms."""

    def __init__(
        self,
        *args: Any,
        generic_args: argparse.Namespace,
        temporal_prior: dict[str, Any] | None,
        truncation_info: dict[str, Any],
        edge_context: dict[str, Any] | None,
        appearance_prior: ImageAppearancePrior | None,
        depth_prior: DepthConsistencyPrior | None,
        support_plane: dict[str, Any] | None = None,
        t_world_from_cam: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.generic_args = generic_args
        self.temporal_prior = temporal_prior
        self.truncation_info = truncation_info
        self.edge_context = edge_context
        self.enable_edge_score = bool(generic_args.edge_score_enabled and edge_context)
        self.appearance_prior = appearance_prior
        self.depth_prior = depth_prior
        self.support_plane = support_plane or {"available": False, "support_plane_confidence": 0.0}
        self.t_world_from_cam = None if t_world_from_cam is None else np.asarray(t_world_from_cam, dtype=np.float64)
        self.current_initializer_metadata: dict[str, Any] = {}
        self._target_contour_edge = temporal_fast._mask_contour(self.full_mask)
        self._target_contour_distance = cv2.distanceTransform(
            np.where(self._target_contour_edge, 0, 1).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )

    def set_initializer_metadata(self, metadata: dict[str, Any] | None) -> None:
        self.current_initializer_metadata = dict(metadata or {})

    def _needs_rendered_mask_for_scoring(self) -> bool:
        return True

    def _contour_score_cached(self, rendered_mask: np.ndarray) -> dict[str, Any]:
        rendered_edge = temporal_fast._mask_contour(rendered_mask)
        if not np.any(rendered_edge) or not np.any(self._target_contour_edge):
            return {"contour_score": 0.0, "contour_mean_distance_px": None}

        rendered_to_target = self._target_contour_distance[rendered_edge].astype(np.float32)
        mean_distance = float(np.clip(rendered_to_target, 0.0, 80.0).mean())
        if bool(getattr(self.generic_args, "generic_contour_bidirectional", False)):
            rendered_distance = cv2.distanceTransform(
                np.where(rendered_edge, 0, 1).astype(np.uint8),
                cv2.DIST_L2,
                3,
            )
            target_to_rendered = rendered_distance[self._target_contour_edge].astype(np.float32)
            mean_distance = float(
                0.5
                * (
                    mean_distance
                    + np.clip(target_to_rendered, 0.0, 80.0).mean()
                )
            )
        return {
            "contour_score": float(math.exp(-mean_distance / max(1e-6, float(getattr(self.generic_args, "generic_contour_sigma_px", 4.0))))),
            "contour_mean_distance_px": mean_distance,
        }

    def _scale_prior_score(self, scale: np.ndarray) -> dict[str, Any]:
        track_scale_prior = getattr(self.generic_args, "generic_track_scale_prior", None)
        if isinstance(track_scale_prior, dict):
            value = track_scale_prior.get("scale")
        else:
            value = None
        if value is None:
            return {"scale_prior_score": 0.0, "scale_prior_delta_log": None}
        try:
            scale_value = fast.scale_to_uniform_scalar(np.asarray(scale, dtype=np.float64))
            prior_value = max(1e-8, float(value))
            delta_log = float(math.log(max(1e-8, scale_value) / prior_value))
            sigma = max(1e-6, float(getattr(self.generic_args, "generic_scale_prior_sigma_log", 0.20)))
            return {"scale_prior_score": float(math.exp(-((delta_log / sigma) ** 2))), "scale_prior_delta_log": delta_log}
        except Exception:
            return {"scale_prior_score": 0.0, "scale_prior_delta_log": None}

    def _render_depth_if_needed(self, result: dict[str, Any]) -> np.ndarray | None:
        if self.depth_prior is None or not bool(getattr(self.generic_args, "depth_enabled", True)):
            return None
        try:
            return render_depth_for_pose(
                vertices=np.asarray(self.vertices, dtype=np.float64),
                faces=np.asarray(self.faces, dtype=np.int32),
                translation_cam=np.asarray(result["translation_cam"], dtype=np.float64),
                rotation_cam=np.asarray(result["rotation_cam"], dtype=np.float64),
                scale=np.asarray(result["scale"], dtype=np.float64),
                intrinsics=self.intrinsics,
                image_size=self.image_size,
                max_faces=int(getattr(self.generic_args, "depth_render_face_limit", 8000)),
            )
        except Exception as exc:
            print(f"[warn] render depth failed; disabling depth score for candidate: {exc}")
            return None

    def _support_contact(self, result: dict[str, Any]) -> dict[str, Any]:
        confidence = float((self.support_plane or {}).get("support_plane_confidence") or 0.0)
        if confidence < float(getattr(self.generic_args, "support_plane_min_confidence", 0.70)):
            return {
                "support_plane_enabled": False,
                "support_plane_disable_reason": (self.support_plane or {}).get("reason", "low_confidence"),
                "support_plane_confidence": confidence,
                "support_plane_inlier_ratio": float((self.support_plane or {}).get("inlier_ratio") or 0.0),
                "support_plane_residual_m": (self.support_plane or {}).get("plane_residual_m"),
                "support_contact_score": 0.0,
                "support_contact_distance_score": 0.0,
                "support_contact_coverage": 0.0,
                "support_contact_mean_abs_m": None,
                "support_contact_max_abs_m": None,
                "support_bottom_selection_mode": None,
                "support_axis_index": None,
                "support_axis_sign": None,
                "support_normal_alignment": 0.0,
                "support_normal_angle_deg": None,
                "support_orientation_score": 0.0,
                "support_orientation_penalty": 0.0,
                "support_bottom_mean_abs_m": None,
                "support_bottom_max_abs_m": None,
                "support_bottom_signed_m": None,
                "support_floating_distance_m": 0.0,
                "support_penetration_distance_m": 0.0,
                "support_floating_penalty": 0.0,
                "support_penetration_penalty": 0.0,
                "support_penalty": 0.0,
            }
        try:
            rotation = np.asarray(result["rotation_cam"], dtype=np.float64)
            translation = np.asarray(result["translation_cam"], dtype=np.float64)
            scale = np.asarray(result["scale"], dtype=np.float64)
            local = np.asarray(self.vertices, dtype=np.float64)
            bottom_percentile = float(getattr(self.generic_args, "support_bottom_percentile", 3.0))
            bottom = support_bottom_points_for_pose(
                local,
                rotation,
                translation,
                scale,
                self.support_plane,
                bottom_percentile=bottom_percentile,
                mode=str(getattr(self.generic_args, "support_bottom_selection_mode", "local_axis")),
                max_points=512,
            )
            support_cam = np.asarray(bottom["support_points_cam"], dtype=np.float64)
            contact = support_contact_score(
                support_cam,
                self.support_plane,
                sigma_m=float(getattr(self.generic_args, "support_contact_sigma_m", 0.10)),
                tolerance_m=float(getattr(self.generic_args, "support_contact_tolerance_m", 0.08)),
                floating_tolerance_m=float(getattr(self.generic_args, "support_floating_tolerance_m", 0.20)),
                penetration_tolerance_m=float(getattr(self.generic_args, "support_penetration_tolerance_m", 0.10)),
                distance_score_weight=float(getattr(self.generic_args, "support_contact_distance_score_weight", 0.70)),
                coverage_score_weight=float(getattr(self.generic_args, "support_contact_coverage_score_weight", 0.30)),
                floating_penalty_weight=float(getattr(self.generic_args, "support_floating_penalty_weight", 0.60)),
                penetration_penalty_weight=float(getattr(self.generic_args, "support_penetration_penalty_weight", 1.00)),
            )
            orientation = support_orientation_score(
                rotation,
                self.support_plane,
                sigma_deg=float(getattr(self.generic_args, "support_orientation_sigma_deg", 20.0)),
                tolerance_deg=float(getattr(self.generic_args, "support_orientation_tolerance_deg", 4.0)),
            )
            support_cutoff = float(bottom["support_bottom_signed_m"])
            floating_distance = max(support_cutoff, 0.0)
            penetration_distance = max(-support_cutoff, 0.0)
            floating_penalty = float(
                np.clip(
                    floating_distance / max(1e-6, float(getattr(self.generic_args, "support_floating_tolerance_m", 0.20))),
                    0.0,
                    1.0,
                )
            )
            penetration_penalty = float(
                np.clip(
                    penetration_distance / max(1e-6, float(getattr(self.generic_args, "support_penetration_tolerance_m", 0.10))),
                    0.0,
                    1.0,
                )
            )
            contact.update(
                {
                    **{key: value for key, value in bottom.items() if key != "support_points_cam"},
                    **orientation,
                    "support_bottom_signed_m": support_cutoff,
                    "support_floating_distance_m": float(floating_distance),
                    "support_penetration_distance_m": float(penetration_distance),
                    "support_floating_penalty": floating_penalty,
                    "support_penetration_penalty": penetration_penalty,
                    "support_penalty": float(
                        float(getattr(self.generic_args, "support_floating_penalty_weight", 0.60)) * floating_penalty
                        + float(getattr(self.generic_args, "support_penetration_penalty_weight", 1.00)) * penetration_penalty
                    ),
                }
            )
            contact["support_plane_enabled"] = True
            contact["support_plane_disable_reason"] = None
            contact["support_plane_confidence"] = confidence
            contact["support_plane_inlier_ratio"] = float((self.support_plane or {}).get("inlier_ratio") or 0.0)
            contact["support_plane_residual_m"] = (self.support_plane or {}).get("plane_residual_m")
            return contact
        except Exception as exc:
            return {
                "support_plane_enabled": False,
                "support_plane_disable_reason": str(exc),
                "support_plane_confidence": confidence,
                "support_plane_inlier_ratio": float((self.support_plane or {}).get("inlier_ratio") or 0.0),
                "support_plane_residual_m": (self.support_plane or {}).get("plane_residual_m"),
                "support_contact_score": 0.0,
                "support_contact_distance_score": 0.0,
                "support_contact_coverage": 0.0,
                "support_contact_mean_abs_m": None,
                "support_contact_max_abs_m": None,
                "support_bottom_selection_mode": None,
                "support_axis_index": None,
                "support_axis_sign": None,
                "support_normal_alignment": 0.0,
                "support_normal_angle_deg": None,
                "support_orientation_score": 0.0,
                "support_orientation_penalty": 0.0,
                "support_bottom_mean_abs_m": None,
                "support_bottom_max_abs_m": None,
                "support_bottom_signed_m": None,
                "support_floating_distance_m": 0.0,
                "support_penetration_distance_m": 0.0,
                "support_floating_penalty": 0.0,
                "support_penetration_penalty": 0.0,
                "support_penalty": 0.0,
                "support_contact_debug": {"reason": str(exc)},
            }

    def _acceptance(self, result: dict[str, Any]) -> dict[str, Any]:
        reject_reasons: list[str] = []
        visible_iou = float(result.get("visible_mask_iou") or result.get("soft_mask_iou") or result.get("mask_iou") or 0.0)
        bbox_iou = float(result.get("bbox_iou") or 0.0)
        center_error_value = result.get("bbox_center_error_px")
        center_error = float(center_error_value) if center_error_value is not None else 1e9
        bbox_diag = self.target_bbox_diagonal
        center_threshold = max(120.0, float(getattr(self.generic_args, "generic_acceptance_max_center_error_ratio", 0.35)) * bbox_diag)
        projection_ratio = float(result.get("projection_valid_ratio") or 0.0)
        if visible_iou < float(getattr(self.generic_args, "generic_acceptance_min_visible_mask_iou", 0.12)):
            reject_reasons.append("visible_mask_or_soft_iou_below_threshold")
        if bbox_iou < float(getattr(self.generic_args, "generic_acceptance_min_bbox_iou", 0.10)):
            reject_reasons.append("bbox_iou_below_threshold")
        if center_error > center_threshold:
            reject_reasons.append("bbox_center_error_above_threshold")
        if projection_ratio < float(getattr(self.generic_args, "generic_acceptance_min_projection_valid_ratio", 0.50)):
            reject_reasons.append("projection_valid_ratio_below_threshold")
        if (
            float(result.get("depth_confidence") or 0.0) >= float(getattr(self.generic_args, "generic_acceptance_depth_confidence_high", 0.70))
            and float(result.get("depth_score") or 0.0) < float(getattr(self.generic_args, "generic_acceptance_depth_min_threshold", 0.25))
        ):
            reject_reasons.append("depth_score_below_threshold")
        return {
            "acceptance_status": "accepted" if not reject_reasons else "rejected",
            "reject_reasons": reject_reasons,
        }

    def _augment_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("_generic_augmented"):
            return result

        coarse_scoring = bool(getattr(self.generic_args, "generic_coarse_scoring", False))
        result.setdefault("rendered_mask", self._empty_mask())
        rendered_mask = result.get("rendered_mask")
        visible_region = _visible_region_from_truncation(self.image_size, self.truncation_info, self.generic_args)
        if result.get("projected_bbox") is None:
            result.update(
                {
                    "mask_blend_score": 0.0,
                    "contour_score": 0.0,
                    "edge_score": 0.0,
                    "edge_confidence": 0.0,
                    "depth_score": 0.0,
                    "depth_confidence": 0.0,
                    "appearance_score": 0.0,
                    "appearance_confidence": 0.0,
                    "temporal_score": 0.0,
                    "scale_prior_score": 0.0,
                    "optional_prior_score": 0.0,
                    "support_plane_confidence": 0.0,
                    "observation_score": 0.0,
                    "invalid_projection_penalty": 1.0,
                    "projection_valid_ratio": 0.0,
                }
            )
            result["score"] = -1.0
            result["final_score"] = -1.0
            result.update(self._acceptance(result))
            result["_generic_augmented"] = True
            return result

        hard_weight = float(getattr(self.generic_args, "hard_mask_weight", 0.30))
        if self.truncation_info.get("is_truncated") and rendered_mask is not None:
            partial = temporal_fast.compute_partial_mask_score(
                rendered_mask=rendered_mask,
                target_mask=self.full_mask,
                soft_target_mask=self.soft_full_mask,
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                args=self.generic_args,
            )
            result.update(partial)
            visible_bbox = temporal_fast.compute_visible_bbox_score(
                result["projected_bbox"],
                self.json_bbox,
                self.image_size,
                self.truncation_info,
                self.generic_args,
                rendered_mask=rendered_mask,
                target_mask=self.full_mask,
            )
            result.update(visible_bbox)
            promote_truncated_visible_bbox_score(result, visible_bbox)
            contour = temporal_fast.compute_visible_contour_score(
                rendered_mask,
                self.full_mask,
                self.image_size,
                self.truncation_info,
                self.generic_args,
            )
            result.update(contour)
            mask_blend_score = (
                (1.0 - hard_weight) * float(result.get("visible_soft_mask_iou") or 0.0)
                + hard_weight * float(result.get("visible_mask_iou") or 0.0)
            )
            bbox_score = float(result.get("visible_bbox_iou") or result.get("bbox_iou") or 0.0)
            contour_value = float(result.get("visible_contour_score") or 0.0)
            result["contour_score"] = contour_value
        else:
            mask_blend_score = (
                (1.0 - hard_weight) * float(result.get("soft_mask_iou") or 0.0)
                + hard_weight * float(result.get("mask_iou") or 0.0)
            )
            bbox_score = float(result.get("bbox_iou") or 0.0)
            contour = self._contour_score_cached(rendered_mask)
            result.update(contour)
            contour_value = float(contour["contour_score"])
            result["visible_mask_iou"] = result.get("mask_iou")
            result["visible_soft_mask_iou"] = result.get("soft_mask_iou")
            result["visible_bbox_iou"] = result.get("bbox_iou")
            result["visible_bbox_center_error_px"] = result.get("bbox_center_error_px")

        result["mask_blend_score"] = float(mask_blend_score)

        edge = temporal_fast.compute_edge_score(rendered_mask, self.edge_context, self.generic_args) if self.enable_edge_score else {
            "edge_score": 0.0,
            "edge_mean_distance_px": None,
            "edge_rendered_points": 0,
            "edge_roi": self.edge_context.get("roi") if self.edge_context else None,
        }
        result.update(edge)
        edge_confidence = 1.0 if self.enable_edge_score and int(edge.get("edge_rendered_points") or 0) > 0 else 0.0
        result["edge_confidence"] = edge_confidence

        appearance = (
            self.appearance_prior.score_render_mask(rendered_mask, visible_region=visible_region)
            if (
                not coarse_scoring
                and self.appearance_prior is not None
                and bool(getattr(self.generic_args, "appearance_enabled", True))
            )
            else {
                "appearance_score": 0.0,
                "appearance_confidence": 0.0,
                "color_soft_iou": 0.0,
                "color_precision": 0.0,
                "color_recall": 0.0,
                "background_leakage": 0.0,
                "fg_bg_distance": 0.0,
                "debug": {"reason": "disabled"},
            }
        )
        result.update({key: value for key, value in appearance.items() if key != "debug"})
        result["appearance_debug"] = appearance.get("debug")

        render_depth = None if coarse_scoring else self._render_depth_if_needed(result)
        depth = (
            self.depth_prior.score(render_depth, rendered_mask, visible_region=visible_region)
            if (
                not coarse_scoring
                and self.depth_prior is not None
                and bool(getattr(self.generic_args, "depth_enabled", True))
            )
            else {
                "depth_score": 0.0,
                "depth_confidence": 0.0,
                "depth_error": None,
                "valid_depth_ratio": 0.0,
                "debug": {"reason": "disabled"},
            }
        )
        result.update({key: value for key, value in depth.items() if key != "debug"})
        result["depth_debug"] = depth.get("debug")

        temporal_score = 0.0
        if (
            not coarse_scoring
            and bool(getattr(self.generic_args, "temporal_enabled", True))
            and self.temporal_prior is not None
        ):
            temporal = compute_generic_temporal_score(
                np.asarray(result["translation_cam"], dtype=np.float64),
                np.asarray(result["rotation_cam"], dtype=np.float64),
                np.asarray(result["scale"], dtype=np.float64),
                self.temporal_prior,
                self.generic_args,
            )
            result.update(temporal)
            temporal_score = float(temporal["temporal_score"])
        else:
            result.update({"temporal_score": 0.0, "generic_temporal_loss": None, "temporal_loss": None})

        scale_prior = self._scale_prior_score(np.asarray(result["scale"], dtype=np.float64))
        result.update(scale_prior)

        support = (
            {
                "support_plane_enabled": False,
                "support_plane_disable_reason": "coarse_scoring",
                "support_plane_confidence": 0.0,
                "support_plane_inlier_ratio": 0.0,
                "support_plane_residual_m": None,
                "support_contact_score": 0.0,
                "support_contact_distance_score": 0.0,
                "support_contact_coverage": 0.0,
                "support_contact_mean_abs_m": None,
                "support_contact_max_abs_m": None,
                "support_bottom_selection_mode": None,
                "support_axis_index": None,
                "support_axis_sign": None,
                "support_normal_alignment": 0.0,
                "support_normal_angle_deg": None,
                "support_orientation_score": 0.0,
                "support_orientation_penalty": 0.0,
                "support_bottom_mean_abs_m": None,
                "support_bottom_max_abs_m": None,
                "support_bottom_signed_m": None,
                "support_floating_distance_m": 0.0,
                "support_penetration_distance_m": 0.0,
                "support_floating_penalty": 0.0,
                "support_penetration_penalty": 0.0,
                "support_penalty": 0.0,
            }
            if coarse_scoring
            else self._support_contact(result)
        )
        support_plane_confidence = float(support.get("support_plane_confidence") or 0.0)
        support_contact_value = float(support.get("support_contact_score") or 0.0)
        support_weight = float(getattr(self.generic_args, "support_plane_weight", 0.20))
        optional_prior_score = support_plane_confidence * support_weight * support_contact_value
        result.update(
            {
                **support,
                "upright_confidence": 0.0,
                "upright_score": 0.0,
                "heading_confidence": 0.0,
                "heading_score": 0.0,
                "optional_prior_score": optional_prior_score,
            }
        )

        observation_score = (
            float(getattr(self.generic_args, "generic_mask_weight", 1.0)) * mask_blend_score
            + float(getattr(self.generic_args, "generic_bbox_weight", 0.15)) * bbox_score
            + float(getattr(self.generic_args, "generic_contour_weight", 0.35)) * contour_value
            + float(getattr(self.generic_args, "generic_edge_weight", getattr(self.generic_args, "edge_weight", 0.20)))
            * edge_confidence
            * float(result.get("edge_score") or 0.0)
            + float(getattr(self.generic_args, "generic_depth_weight", 0.35))
            * float(result.get("depth_confidence") or 0.0)
            * float(result.get("depth_score") or 0.0)
            + float(getattr(self.generic_args, "generic_appearance_weight", 0.25))
            * float(result.get("appearance_confidence") or 0.0)
            * float(result.get("appearance_score") or 0.0)
        )
        observation_score_max = (
            float(getattr(self.generic_args, "generic_mask_weight", 1.0))
            + float(getattr(self.generic_args, "generic_bbox_weight", 0.15))
            + float(getattr(self.generic_args, "generic_contour_weight", 0.35))
            + (float(getattr(self.generic_args, "generic_edge_weight", 0.20)) if self.enable_edge_score else 0.0)
            + (float(getattr(self.generic_args, "generic_depth_weight", 0.35)) if self.depth_prior is not None and bool(getattr(self.generic_args, "depth_enabled", True)) else 0.0)
            + (float(getattr(self.generic_args, "generic_appearance_weight", 0.25)) if self.appearance_prior is not None and bool(getattr(self.generic_args, "appearance_enabled", True)) else 0.0)
            + float(getattr(self.generic_args, "generic_extent_weight", 0.0) or 0.0)
        )
        observation_quality = clamp01(observation_score / max(1e-6, observation_score_max))
        gate_start = float(getattr(self.generic_args, "optional_prior_gate_start", 0.35))
        gate_range = max(1e-6, float(getattr(self.generic_args, "optional_prior_gate_range", 0.45)))
        optional_prior_gate = clamp01((observation_quality - gate_start) / gate_range)
        invalid_projection_penalty = 0.0
        depth_outlier_penalty = 0.0
        if (
            float(result.get("depth_confidence") or 0.0) >= 0.7
            and float(result.get("depth_score") or 0.0) < float(getattr(self.generic_args, "depth_outlier_score_threshold", 0.10))
        ):
            depth_outlier_penalty = float(getattr(self.generic_args, "depth_outlier_penalty", 0.0))
        temporal_jump_penalty = 0.0
        support_contact_penalty_eff = (
            support_plane_confidence
            * optional_prior_gate
            * float(getattr(self.generic_args, "support_penalty_weight", 0.15))
            * float(support.get("support_penalty") or 0.0)
        )
        support_orientation_penalty_eff = (
            support_plane_confidence
            * optional_prior_gate
            * float(getattr(self.generic_args, "support_orientation_penalty_weight", 0.0))
            * float(support.get("support_orientation_penalty") or 0.0)
        )

        scaled_vertices = self.vertices * np.asarray(result["scale"], dtype=self.dtype).reshape(1, 3)
        points_cam = scaled_vertices @ np.asarray(result["rotation_cam"], dtype=self.dtype).T + np.asarray(result["translation_cam"], dtype=self.dtype).reshape(1, 3)
        projected_uv, valid_z = fast.project_points(points_cam, **self.intrinsics)
        proj_ratio = projection_valid_ratio(points_cam, projected_uv, valid_z, self.image_size)

        generic_score = (
            observation_score
            + float(getattr(self.generic_args, "generic_temporal_weight", 0.55)) * temporal_score
            + float(getattr(self.generic_args, "generic_scale_prior_weight", 0.30)) * float(scale_prior.get("scale_prior_score") or 0.0)
            + optional_prior_gate * optional_prior_score
            - invalid_projection_penalty
            - depth_outlier_penalty
            - temporal_jump_penalty
            - support_contact_penalty_eff
            - support_orientation_penalty_eff
        )
        result.update(
            {
                "observation_score": float(observation_score),
                "observation_quality": float(observation_quality),
                "optional_prior_gate": optional_prior_gate,
                "invalid_projection_penalty": invalid_projection_penalty,
                "depth_outlier_penalty": depth_outlier_penalty,
                "temporal_jump_penalty": temporal_jump_penalty,
                "support_contact_penalty_eff": float(support_contact_penalty_eff),
                "support_orientation_penalty_eff": float(support_orientation_penalty_eff),
                "projection_valid_ratio": proj_ratio,
                "visible_ratio": _visible_ratio(rendered_mask, visible_region),
                "truncation_ratio": 1.0 - _visible_ratio(rendered_mask, visible_region),
                "score": float(generic_score),
                "final_score": float(generic_score),
            }
        )
        result.update(self._acceptance(result))
        if self.current_initializer_metadata:
            result["initializer_metadata"] = dict(self.current_initializer_metadata)
        result["_generic_augmented"] = True
        return result

    def evaluate_absolute(
        self,
        translation_cam: np.ndarray,
        rotation_cam: np.ndarray,
        scale: np.ndarray,
        keep_mask: bool = False,
    ) -> dict[str, Any]:
        needs_mask = keep_mask or self._needs_rendered_mask_for_scoring()
        result = super().evaluate_absolute(translation_cam, rotation_cam, scale, keep_mask=needs_mask)
        self._augment_result(result)
        if not keep_mask and needs_mask:
            result.pop("rendered_mask", None)
        return result

    def evaluate_absolute_batch(
        self,
        translations_cam: np.ndarray,
        rotations_cam: np.ndarray,
        scales: np.ndarray,
        batch_size: int = 32,
        keep_masks: bool = False,
    ) -> list[dict[str, Any]]:
        needs_masks = keep_masks or self._needs_rendered_mask_for_scoring()
        results = super().evaluate_absolute_batch(
            translations_cam,
            rotations_cam,
            scales,
            batch_size=batch_size,
            keep_masks=needs_masks,
        )
        for result in results:
            self._augment_result(result)
            if not keep_masks and needs_masks:
                result.pop("rendered_mask", None)
        return results


def generic_optimization_history_row(
    phase: str,
    iteration: int,
    parameter: str,
    direction: int,
    result: dict[str, Any],
    step_value: float,
) -> dict[str, Any]:
    row = fast.optimization_history_row(phase, iteration, parameter, direction, result, step_value)
    for key in GENERIC_CANDIDATE_METRIC_KEYS:
        row[key] = result.get(key)
    return row


def generic_local_search_stage(
    evaluator: GenericPoseEvaluator,
    base_translation_cam: np.ndarray,
    base_rotation_cam: np.ndarray,
    base_scale: np.ndarray,
    stage_name: str,
    max_iters: int,
    step_decay: float,
    max_translation_delta: float,
    max_rotation_delta_deg: float,
    scale_min_factor: float,
    scale_max_factor: float,
    save_full_history: bool,
    initializer_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluator.set_initializer_metadata(initializer_metadata)
    groups = fast.stage_groups(stage_name)
    params = np.zeros(fast.PARAM_DIM, dtype=np.float64)
    group_steps = np.array([group["step"] for group in groups], dtype=np.float64)
    group_min_steps = np.array([group["min_step"] for group in groups], dtype=np.float64)

    best = evaluator.evaluate_delta(base_translation_cam, base_rotation_cam, base_scale, params)
    history = [generic_optimization_history_row(stage_name, 0, "initial", -1, best, 0.0)]
    print(
        f"  [{stage_name}] start score={best['score']:.6f} "
        f"mask_iou={best['mask_iou']:.6f} bbox_iou={best['bbox_iou']:.6f}"
    )
    for iteration in range(1, max_iters + 1):
        improved = False
        for group_index, group in enumerate(groups):
            current_best = best
            current_params = params
            current_direction = 0
            step_value = float(group_steps[group_index])
            candidates = [
                fast.clamp_delta_params(
                    params + direction * step_value * group["vector"],
                    max_translation_delta,
                    max_rotation_delta_deg,
                    scale_min_factor,
                    scale_max_factor,
                )
                for direction in (1.0, -1.0)
            ]
            results = evaluator.evaluate_delta_batch(
                base_translation_cam,
                base_rotation_cam,
                base_scale,
                np.stack(candidates, axis=0),
            )
            for direction, candidate, result in zip((1.0, -1.0), candidates, results):
                if save_full_history:
                    history.append(generic_optimization_history_row(stage_name, iteration, group["name"], int(direction), result, step_value))
                if result["score"] > current_best["score"] + 1e-8:
                    current_best = result
                    current_params = candidate
                    current_direction = int(direction)
            if current_best["score"] > best["score"] + 1e-8:
                params = current_params
                best = current_best
                improved = True
                if not save_full_history:
                    history.append(generic_optimization_history_row(stage_name, iteration, group["name"], current_direction, best, step_value))
                print(
                    f"  [{stage_name} iter {iteration:02d}] improve {group['name']} "
                    f"score={best['score']:.6f} mask_iou={best['mask_iou']:.6f} bbox_iou={best['bbox_iou']:.6f}"
                )
        if not improved:
            group_steps *= step_decay
            print(f"  [{stage_name} iter {iteration:02d}] no improvement, shrink steps")
            if not save_full_history:
                history.append(generic_optimization_history_row(stage_name, iteration, "step_shrink", 0, best, float(group_steps.max())))
        if np.all(group_steps <= group_min_steps):
            print(f"  [{stage_name}] converged")
            break
    final = evaluator.evaluate_delta(base_translation_cam, base_rotation_cam, base_scale, params, keep_mask=True)
    final["params"] = params.copy()
    if not save_full_history:
        history.append(generic_optimization_history_row(stage_name, max_iters + 1, "final", 0, final, 0.0))
    evaluator.set_initializer_metadata(None)
    return final, history


def refine_candidate_stages(
    coarse_result: dict[str, Any],
    proxy_evaluator: GenericPoseEvaluator,
    full_evaluator: GenericPoseEvaluator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    translation_cam = np.asarray(coarse_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(coarse_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(coarse_result["scale"], dtype=np.float64)
    initializer_metadata = dict(coarse_result.get("initializer_metadata") or {})
    history: list[dict[str, Any]] = []
    for stage_name, evaluator, max_iters in (
        ("coarse", proxy_evaluator, args.stage1_iters),
        ("rotation", proxy_evaluator, args.stage2_iters),
        ("fine", full_evaluator, args.stage3_iters),
    ):
        result, stage_history = generic_local_search_stage(
            evaluator=evaluator,
            base_translation_cam=translation_cam,
            base_rotation_cam=rotation_cam,
            base_scale=scale,
            stage_name=stage_name,
            max_iters=max_iters,
            step_decay=args.step_decay,
            max_translation_delta=args.max_translation_delta,
            max_rotation_delta_deg=args.max_rotation_delta_deg,
            scale_min_factor=args.scale_min_factor,
            scale_max_factor=args.scale_max_factor,
            save_full_history=args.save_full_history,
            initializer_metadata=initializer_metadata,
        )
        history.extend(stage_history)
        translation_cam = np.asarray(result["translation_cam"], dtype=np.float64)
        rotation_cam = np.asarray(result["rotation_cam"], dtype=np.float64)
        scale = np.asarray(result["scale"], dtype=np.float64)
    return result, history


def make_generic_temporal_seed(prior: dict[str, Any] | None, evaluator: GenericPoseEvaluator) -> dict[str, Any] | None:
    if prior is None:
        return None
    result = evaluator.evaluate_absolute(
        np.asarray(prior["translation_cam"], dtype=np.float64),
        np.asarray(prior["rotation_cam"], dtype=np.float64),
        np.asarray(prior["scale"], dtype=np.float64),
    )
    if result.get("projected_bbox") is None:
        return None
    result["initializer_metadata"] = {
        "source": "temporal_prior",
        "prior_frame_id": prior.get("frame_idx"),
        "prior_output_dir": prior.get("output_dir"),
        "prior_pose_source": prior.get("pose_source"),
    }
    return result


def parse_angle_list(value: str | list[Any] | tuple[Any, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def augment_generic_rotation_candidates(
    candidates: list[dict[str, Any]],
    evaluator: GenericPoseEvaluator,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not bool(getattr(args, "generic_rotation_grid_enabled", True)) or not candidates:
        return candidates
    yaw_values = parse_angle_list(getattr(args, "generic_yaw_degrees", "0,45,90,135,180,225,270,315"))
    pitch_values = parse_angle_list(getattr(args, "generic_pitch_degrees", "-10,0,10"))
    roll_values = parse_angle_list(getattr(args, "generic_roll_degrees", "-10,0,10"))
    source_candidates = candidates[: max(1, int(getattr(args, "generic_rotation_grid_source_top_k", 2)))]
    augmented = list(candidates)
    seen = {fast.pose_signature(item) for item in augmented}
    for base in source_candidates:
        base_rotation = np.asarray(base["rotation_cam"], dtype=np.float64)
        for yaw in yaw_values:
            for pitch in pitch_values:
                for roll in roll_values:
                    if abs(float(yaw)) < 1e-8 and abs(float(pitch)) < 1e-8 and abs(float(roll)) < 1e-8:
                        continue
                    delta = fast.euler_xyz_to_matrix(math.radians(pitch), math.radians(yaw), math.radians(roll))
                    rotation = delta @ base_rotation
                    result = evaluator.evaluate_absolute(
                        np.asarray(base["translation_cam"], dtype=np.float64),
                        rotation,
                        np.asarray(base["scale"], dtype=np.float64),
                    )
                    if result.get("projected_bbox") is None:
                        continue
                    sig = fast.pose_signature(result)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    meta = dict(base.get("initializer_metadata") or {})
                    meta.update({"source": "generic_rotation_grid", "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll})
                    result["initializer_metadata"] = meta
                    augmented.append(result)
    return sorted(augmented, key=lambda item: float(item.get("score", -1e9)), reverse=True)[: max(len(candidates), int(args.top_k_candidates))]


def generate_generic_grid_candidates(
    evaluator: GenericPoseEvaluator,
    obs: dict[str, Any],
    mesh_meta: dict[str, Any],
    base_scale: np.ndarray,
    corrected_seed: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Generate camera-frame rotation/scale/depth candidates without vehicle up/front assumptions."""

    bounds = np.asarray(mesh_meta["bounds"], dtype=np.float64)
    bbox_corners_local = np.asarray(
        [
            [bounds[0, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
        ],
        dtype=np.float64,
    )
    anchors = [
        ("bbox_center", np.asarray(mesh_meta["center"], dtype=np.float64), np.asarray(obs["bbox_center"], dtype=np.float64)),
        ("mask_centroid", np.asarray(mesh_meta["center"], dtype=np.float64), np.asarray(obs["centroid"], dtype=np.float64)),
    ]
    yaw_values = parse_angle_list(getattr(args, "generic_yaw_degrees", "0,45,90,135,180,225,270,315"))
    pitch_values = parse_angle_list(getattr(args, "generic_pitch_degrees", "-10,0,10"))
    roll_values = parse_angle_list(getattr(args, "generic_roll_degrees", "-10,0,10"))
    scale_factors = fast.parse_comma_floats(str(getattr(args, "init_scale_factors", "0.75,0.90,1.00,1.10,1.25")))
    depth_factors = fast.parse_comma_floats(str(getattr(args, "init_depth_factors", "0.8,1.0,1.2")))

    heap: list[tuple[float, int, dict[str, Any]]] = []
    seen: set[tuple[float, ...]] = set()
    batch_candidate_specs: list[dict[str, Any]] = []
    counter = 0
    if corrected_seed is not None:
        counter = fast.keep_top_k_results(heap, seen, corrected_seed, int(args.top_k_candidates), counter)

    for yaw in yaw_values:
        for pitch in pitch_values:
            for roll in roll_values:
                rotation = fast.euler_xyz_to_matrix(math.radians(pitch), math.radians(yaw), math.radians(roll))
                for scale_factor in scale_factors:
                    scale = np.asarray(base_scale, dtype=np.float64) * float(scale_factor)
                    depth_guess = fast.estimate_depth_guess(obs["mask_bbox"], evaluator.intrinsics, rotation, scale, bbox_corners_local)
                    if depth_guess is None:
                        continue
                    for depth_factor in depth_factors:
                        tz = float(depth_guess) * float(depth_factor)
                        if tz <= 0.05:
                            continue
                        for anchor_name, anchor_local, target_uv in anchors:
                            metadata = {
                                "source": "generic_grid",
                                "yaw_deg": float(yaw),
                                "pitch_deg": float(pitch),
                                "roll_deg": float(roll),
                                "global_scale_factor": float(scale_factor),
                                "depth_factor": float(depth_factor),
                                "anchor_name": anchor_name,
                            }
                            spec = fast.build_coarse_candidate_spec(
                                rotation_cam=rotation,
                                scale=scale,
                                tz=tz,
                                anchor_local=anchor_local,
                                target_uv=target_uv,
                                metadata=metadata,
                                intrinsics=evaluator.intrinsics,
                            )
                            if spec is None:
                                continue
                            if bool(getattr(args, "enable_batch_gpu_eval", False)):
                                batch_candidate_specs.append(spec)
                                continue
                            candidate = fast.evaluate_coarse_candidate_spec(evaluator, spec)
                            if candidate is not None:
                                counter = fast.keep_top_k_results(heap, seen, candidate, int(args.top_k_candidates), counter)
    if bool(getattr(args, "enable_batch_gpu_eval", False)):
        for candidate in fast.batch_prefilter_initial_candidates(evaluator, batch_candidate_specs, args):
            counter = fast.keep_top_k_results(heap, seen, candidate, int(args.top_k_candidates), counter)
    return [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]


def select_generic_refine_candidates(
    candidates: list[dict[str, Any]],
    refine_top_k: int,
    *,
    corrected_seed: dict[str, Any] | None = None,
    temporal_seed: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select high-scoring generic candidates while preserving trusted seeds."""

    limit = max(1, int(refine_top_k))
    combined = [item for item in candidates if item is not None]
    for seed in (corrected_seed, temporal_seed):
        if seed is not None:
            combined.append(seed)
    if not combined:
        return []

    if limit == 1 and temporal_seed is not None:
        return [temporal_seed]

    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for item in sorted(combined, key=lambda row: float(row.get("score", -1e9)), reverse=True):
        signature = fast.pose_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)

    required: list[dict[str, Any]] = []
    required_signatures: set[tuple[float, ...]] = set()

    def add_required(source: str, preferred: dict[str, Any] | None = None) -> None:
        source_item = None
        if preferred is not None and preferred.get("initializer_metadata", {}).get("source") == source:
            source_item = preferred
        if source_item is None:
            source_item = next(
                (item for item in unique if item.get("initializer_metadata", {}).get("source") == source),
                None,
            )
        if source_item is None:
            return
        signature = fast.pose_signature(source_item)
        if signature in required_signatures:
            return
        required_signatures.add(signature)
        required.append(source_item)

    add_required("task_json_corrected_pose", corrected_seed)
    add_required("temporal_prior", temporal_seed)
    if len(required) >= limit:
        return required[:limit]

    selected: list[dict[str, Any]] = []
    fill_limit = limit - len(required)
    for item in unique:
        if fast.pose_signature(item) in required_signatures:
            continue
        selected.append(item)
        if len(selected) >= fill_limit:
            break
    selected.extend(required)
    return selected[:limit]


def candidate_summary(result: dict[str, Any], *, t_world_from_cam: np.ndarray) -> dict[str, Any]:
    uniform_scale = fast.make_uniform_scale(fast.scale_to_uniform_scalar(np.asarray(result["scale"], dtype=np.float64)))
    translation_world, rotation_world = fast.camera_pose_to_world_pose(
        np.asarray(t_world_from_cam, dtype=np.float64),
        np.asarray(result["translation_cam"], dtype=np.float64),
        np.asarray(result["rotation_cam"], dtype=np.float64),
    )
    metrics = {key: result.get(key) for key in GENERIC_CANDIDATE_METRIC_KEYS if key in result}
    metrics["scale"] = uniform_scale
    return {
        "candidate_rank": result.get("candidate_rank"),
        "initializer_metadata": result.get("initializer_metadata", {}),
        "optimized_camera_pose": {
            "translation_cam": result["translation_cam"],
            "rotation_cam": result["rotation_cam"],
            "scale": uniform_scale,
        },
        "optimized_corrected_pose_world": {
            "translation_world": translation_world,
            "rotation_matrix": rotation_world,
            "scale": uniform_scale,
        },
        "metrics": metrics,
        "pose": {
            "translation_cam": result["translation_cam"],
            "rotation_cam": result["rotation_cam"],
            "scale": uniform_scale,
        },
    }


def save_generic_breakdown(output_dir: Path, candidates: list[dict[str, Any]]) -> Path | None:
    if not candidates:
        return None
    width = 1280
    row_h = 34
    height = max(220, row_h * (min(12, len(candidates)) + 2))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    headers = ["rank", "score", "mask", "bbox", "contour", "edge", "depth", "app", "temp", "support", "status"]
    xs = [12, 80, 180, 280, 380, 500, 610, 730, 850, 955, 1110]
    for x, text in zip(xs, headers):
        cv2.putText(canvas, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    for row, cand in enumerate(candidates[:12], start=1):
        y = 28 + row_h * row
        row_values = generic_breakdown_row_values(cand, row=row)
        values = [row_values[header] for header in headers]
        color = (20, 120, 20) if cand.get("acceptance_status") == "accepted" else (30, 30, 180)
        for x, text in zip(xs, values):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    path = output_dir / "candidate_score_breakdown.png"
    cv2.imwrite(str(path), canvas)
    return path


def generic_breakdown_row_values(cand: dict[str, Any], *, row: int) -> dict[str, str]:
    support_score = float(cand.get("support_contact_score") or 0.0)
    support_penalty = float(cand.get("support_contact_penalty_eff") or 0.0) + float(cand.get("support_orientation_penalty_eff") or 0.0)
    return {
        "rank": str(cand.get("candidate_rank", row)),
        "score": f"{float(cand.get('score') or 0.0):.3f}",
        "mask": f"{float(cand.get('mask_blend_score') or cand.get('mask_iou') or 0.0):.3f}",
        "bbox": f"{float(cand.get('bbox_iou') or 0.0):.3f}",
        "contour": f"{float(cand.get('contour_score') or 0.0):.3f}",
        "edge": f"{float(cand.get('edge_score') or 0.0):.3f}",
        "depth": f"{float(cand.get('depth_score') or 0.0):.3f}/{float(cand.get('depth_confidence') or 0.0):.2f}",
        "app": f"{float(cand.get('appearance_score') or 0.0):.3f}/{float(cand.get('appearance_confidence') or 0.0):.2f}",
        "temp": f"{float(cand.get('temporal_score') or 0.0):.3f}",
        "support": f"{support_score:.3f}/-{support_penalty:.3f}",
        "status": str(cand.get("acceptance_status", "")),
    }


def _load_temporal_prior(sample_dir: Path, output_dir: Path, task: dict[str, Any], t_world_from_cam: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        object_id, frame_idx = temporal_fast.parse_task_id_from_sample_dir(sample_dir)
    except Exception:
        object_id, frame_idx = str(task.get("object_id", sample_dir.name)), 1
    temporal_prior = None
    if bool(args.temporal_enabled) and frame_idx > 1:
        temporal_prior = temporal_fast.load_prior_pose_payload(task.get("temporal_prior_pose"), t_world_from_cam)
        if temporal_prior is None:
            temporal_prior = temporal_fast.find_temporal_prior(
                output_dir=output_dir,
                object_id=object_id,
                frame_idx=frame_idx,
                lookback=args.temporal_lookback,
                suffixes=temporal_fast.parse_suffixes(args.temporal_search_output_suffixes),
                t_world_from_cam=t_world_from_cam,
            )
    return temporal_prior, {
        "enabled": bool(args.temporal_enabled),
        "prior_found": temporal_prior is not None,
        "prior_frame_id": temporal_prior.get("frame_idx") if temporal_prior else None,
        "prior_output_dir": temporal_prior.get("output_dir") if temporal_prior else None,
        "prior_pose_source": temporal_prior.get("pose_source") if temporal_prior else None,
    }


def optimize_sample(args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = fast.resolve_sample_dir(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{sample_dir.name}_generic_appearance_temporal"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_backend == "pytorch3d" or args.validate_pytorch3d_alignment:
        fast.import_torch_and_pytorch3d()
        fast.resolve_torch_device(args.device, allow_auto_fallback=False)

    task = fast.read_json(sample_dir / "task.json")
    image = fast.read_image(sample_dir / "image.jpg", mode="color")
    crop_mask = fast.read_image(sample_dir / "mask.png", mode="gray")
    crop_image_path = sample_dir / "crop.jpg"
    crop_image = fast.read_image(crop_image_path, mode="color") if crop_image_path.exists() else None
    image_size = fast.image_size_from_task(task, image)
    json_bbox = [float(v) for v in task["bbox_xyxy"]]
    full_mask, mask_placement = fast.paste_crop_mask_to_full_image(crop_mask, json_bbox, image_size, full_image=image, crop_image=crop_image)
    soft_full_mask = fast.make_soft_mask(full_mask)
    if args.fast_float32:
        soft_full_mask = soft_full_mask.astype(np.float32, copy=False)
    obs = fast.extract_mask_observations(full_mask)

    mesh_path = fast.find_mesh_path(sample_dir, task)
    mesh = fast.load_glb_as_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh_meta = fast.mesh_axis_metadata(vertices)
    proxy_vertices, proxy_faces = fast.build_proxy_mesh(vertices, faces, target_faces=args.proxy_face_count)

    t_world_from_cam = np.asarray(task["camera"]["T_world_from_cam"], dtype=np.float64)
    intrinsics = {
        "fx": float(task["camera"]["fx"]),
        "fy": float(task["camera"]["fy"]),
        "cx": float(task["camera"]["cx"]),
        "cy": float(task["camera"]["cy"]),
    }

    appearance_prior = None
    appearance_debug_outputs: dict[str, str] = {}
    appearance_report: dict[str, Any] = {"enabled": bool(args.appearance_enabled), "available": False}
    if bool(args.appearance_enabled):
        appearance_prior = build_image_appearance_prior(
            image,
            full_mask,
            json_bbox,
            config=AppearancePriorConfig(
                foreground_erode_kernel=args.appearance_foreground_erode_kernel,
                foreground_min_pixels=args.appearance_foreground_min_pixels,
                background_inner_dilate_kernel=args.appearance_background_inner_dilate_kernel,
                background_outer_dilate_kernel=args.appearance_background_outer_dilate_kernel,
                background_bbox_expand_ratio=args.appearance_background_bbox_expand_ratio,
                background_min_pixels=args.appearance_background_min_pixels,
                confidence_low_threshold=args.appearance_confidence_low_threshold,
                confidence_high_threshold=args.appearance_confidence_high_threshold,
                min_confidence=args.appearance_min_confidence,
                soft_iou_weight=args.appearance_soft_iou_weight,
                precision_weight=args.appearance_precision_weight,
                recall_weight=args.appearance_recall_weight,
                leakage_weight=args.appearance_leakage_weight,
            ),
        )
        appearance_report = {
            "enabled": True,
            "available": True,
            "appearance_confidence": appearance_prior.appearance_confidence,
            "fg_bg_distance": appearance_prior.fg_bg_distance,
            "debug": appearance_prior.debug_info,
        }

    observed_depth, depth_info = load_observed_depth_for_task(sample_dir, task, args)
    depth_prior = None
    if bool(args.depth_enabled) and observed_depth is not None:
        try:
            if observed_depth.shape != full_mask.shape:
                observed_depth = cv2.resize(observed_depth, (full_mask.shape[1], full_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth_prior = DepthConsistencyPrior(
                observed_depth,
                full_mask,
                config=DepthConsistencyConfig(
                    depth_sigma=args.depth_sigma,
                    min_valid_ratio=args.depth_min_valid_ratio,
                    robust_stat=args.depth_robust_stat,
                ),
            )
            depth_info["available"] = True
        except Exception as exc:
            depth_info = {"available": False, "reason": str(exc)}

    support_plane = estimate_support_plane_from_observed_depth(
        observed_depth,
        full_mask,
        json_bbox,
        intrinsics,
        args,
    )
    support_debug_outputs = save_support_plane_debug(output_dir, support_plane, full_mask)

    temporal_prior, temporal_report = _load_temporal_prior(sample_dir, output_dir, task, t_world_from_cam, args)
    if temporal_prior is None:
        print("[generic-temporal] no prior found; using frame-local scoring")
    else:
        print(f"[generic-temporal] prior found: frame={temporal_prior.get('frame_idx')} source={temporal_prior.get('pose_source')}")

    truncation_info = temporal_fast.detect_truncation(
        mask=full_mask,
        bbox=json_bbox,
        image_size=image_size,
        args=args,
        prior_mask_area_px=temporal_prior.get("prior_mask_area_px") if temporal_prior else None,
    )
    print(
        f"[generic-partial] enabled={bool(args.partial_visibility_enabled)} "
        f"is_truncated={truncation_info['is_truncated']} sides={truncation_info['truncation_sides']}"
    )

    edge_context = temporal_fast.prepare_image_edge_map(image, full_mask, json_bbox, image_size, args)
    print(f"[generic-edge] enabled={bool(args.edge_score_enabled)} available={edge_context is not None}")

    pose = task.get("corrected_pose", {})
    base_scale = fast.make_uniform_scale(
        fast.scale_to_uniform_scalar(np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
    )

    proxy_args = argparse.Namespace(**vars(args))
    proxy_args.generic_coarse_scoring = True

    evaluator_kwargs = dict(
        full_mask=full_mask,
        soft_full_mask=soft_full_mask,
        json_bbox=json_bbox,
        intrinsics=intrinsics,
        image_size=image_size,
        bbox_weight=args.bbox_weight,
        hard_mask_weight=args.hard_mask_weight,
        enable_bbox_prefilter=args.enable_bbox_prefilter,
        prefilter_bbox_iou_min=args.prefilter_bbox_iou_min,
        prefilter_center_factor=args.prefilter_center_factor,
        prefilter_size_ratio_min=args.prefilter_size_ratio_min,
        prefilter_size_ratio_max=args.prefilter_size_ratio_max,
        roi_iou_margin=args.roi_iou_margin,
        disable_roi_iou=args.disable_roi_iou,
        fast_float32=args.fast_float32,
        profile_timings=args.profile_timings,
        device=args.device,
        pytorch3d_faces_per_pixel=args.pytorch3d_faces_per_pixel,
        pytorch3d_cull_backfaces=args.pytorch3d_cull_backfaces,
        pytorch3d_bin_size=args.pytorch3d_bin_size,
        pytorch3d_max_faces_per_bin=args.pytorch3d_max_faces_per_bin,
        temporal_prior=temporal_prior,
        truncation_info=truncation_info,
        appearance_prior=appearance_prior,
        depth_prior=depth_prior,
        support_plane=support_plane,
        t_world_from_cam=t_world_from_cam,
    )
    proxy_evaluator = GenericPoseEvaluator(
        vertices=proxy_vertices,
        faces=proxy_faces,
        mesh=None,
        backend="triangle_fill",
        edge_context=None,
        generic_args=proxy_args,
        **evaluator_kwargs,
    )
    full_evaluator = GenericPoseEvaluator(
        vertices=vertices,
        faces=faces,
        mesh=mesh,
        backend=args.render_backend,
        edge_context=edge_context,
        generic_args=args,
        **evaluator_kwargs,
    )

    corrected_seed = fast.corrected_pose_seed(task, t_world_from_cam, proxy_evaluator) if args.include_corrected_seed else None
    generic_grid_candidates = generate_generic_grid_candidates(
        evaluator=proxy_evaluator,
        obs=obs,
        mesh_meta=mesh_meta,
        base_scale=base_scale,
        corrected_seed=corrected_seed,
        args=args,
    )
    initial_candidates = fast.generate_initial_candidates(
        evaluator=proxy_evaluator,
        obs=obs,
        mesh_meta=mesh_meta,
        base_scale=base_scale,
        corrected_seed=corrected_seed,
        t_world_from_cam=t_world_from_cam,
        args=args,
    )
    merged_candidates = list(generic_grid_candidates) + list(initial_candidates)
    seen_candidates: set[tuple[float, ...]] = set()
    deduped_candidates: list[dict[str, Any]] = []
    for candidate in sorted(merged_candidates, key=lambda item: float(item.get("score", -1e9)), reverse=True):
        signature = fast.pose_signature(candidate)
        if signature in seen_candidates:
            continue
        seen_candidates.add(signature)
        deduped_candidates.append(candidate)
    initial_candidates = augment_generic_rotation_candidates(
        deduped_candidates[: int(args.top_k_candidates)],
        proxy_evaluator,
        args,
    )

    temporal_seed = None
    if bool(args.temporal_enabled) and bool(args.temporal_seed_enabled):
        temporal_seed = make_generic_temporal_seed(temporal_prior, proxy_evaluator)
    initial_candidates = temporal_fast.merge_temporal_seed(
        initial_candidates,
        temporal_seed,
        top_k=args.top_k_candidates,
        refine_top_k=args.refine_top_k,
    )
    print(f"[generic-search] generated {len(initial_candidates)} candidates")

    preview_rows = []
    for index, candidate in enumerate(initial_candidates[: min(8, len(initial_candidates))]):
        preview_rows.append({key: fast.to_builtin(candidate.get(key)) for key in GENERIC_CANDIDATE_METRIC_KEYS if key in candidate})
        preview_rows[-1]["rank"] = index + 1
        preview_rows[-1]["initializer_metadata"] = candidate.get("initializer_metadata", {})

    candidates_to_refine = select_generic_refine_candidates(
        initial_candidates,
        args.refine_top_k,
        corrected_seed=corrected_seed,
        temporal_seed=temporal_seed,
    )
    refined_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    best_result: dict[str, Any] | None = None
    best_history: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates_to_refine, start=1):
        meta = candidate.get("initializer_metadata", {})
        print(
            f"[generic-candidate {rank:02d}] init score={candidate['score']:.6f} "
            f"mask={candidate.get('mask_iou', 0.0):.6f} bbox={candidate.get('bbox_iou', 0.0):.6f} "
            f"app={candidate.get('appearance_score', 0.0):.6f} source={meta.get('source', 'unknown')}"
        )
        refined_result, history = refine_candidate_stages(candidate, proxy_evaluator, full_evaluator, args)
        refined_result["initializer_metadata"] = candidate.get("initializer_metadata", {})
        refined_result["candidate_rank"] = rank
        refined_results.append((refined_result, history))
        print(
            f"[generic-candidate {rank:02d}] refined score={refined_result['score']:.6f} "
            f"mask={refined_result.get('mask_iou', 0.0):.6f} bbox={refined_result.get('bbox_iou', 0.0):.6f} "
            f"app={refined_result.get('appearance_score', 0.0):.6f} status={refined_result.get('acceptance_status')}"
        )
        if best_result is None or float(refined_result["score"]) > float(best_result["score"]):
            best_result = refined_result
            best_history = history
        if best_result["mask_iou"] >= args.early_stop_mask_iou and best_result["bbox_iou"] >= args.early_stop_bbox_iou:
            break

    accepted = [item for item in refined_results if item[0].get("acceptance_status") == "accepted"]
    if accepted:
        best_result, best_history = max(accepted, key=lambda item: float(item[0].get("score", -1e9)))
    if best_result is None:
        raise RuntimeError("No valid pose candidate survived refinement.")

    best_uniform_scale = fast.scale_to_uniform_scalar(np.asarray(best_result["scale"], dtype=np.float64))
    best_result["scale"] = fast.make_uniform_scale(best_uniform_scale)
    translation_world, rotation_world = fast.camera_pose_to_world_pose(
        t_world_from_cam,
        np.asarray(best_result["translation_cam"], dtype=np.float64),
        np.asarray(best_result["rotation_cam"], dtype=np.float64),
    )

    fast.save_mask_comparison(
        output_dir,
        image,
        full_mask,
        best_result["rendered_mask"],
        json_bbox,
        best_result["projected_bbox"],
        "01_best",
    )
    fast.save_glb_native_shape_views(output_dir, vertices, faces, mesh_meta)
    fast.save_optimized_glb_projection_views(
        output_dir=output_dir,
        image=image,
        vertices=vertices,
        faces=faces,
        best_result=best_result,
        intrinsics=intrinsics,
        image_size=image_size,
        json_bbox=json_bbox,
    )
    fast.save_optimized_glb_pose_render(
        output_dir=output_dir,
        mesh=mesh,
        vertices=vertices,
        faces=faces,
        best_result=best_result,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    collage_paths = fast.save_result_collages(
        output_dir=output_dir,
        json_bbox=json_bbox,
        projected_bbox=best_result["projected_bbox"],
    )
    temporal_debug_path = temporal_fast.save_temporal_edge_debug(
        output_dir=output_dir,
        image=image,
        vertices=vertices,
        faces=faces,
        intrinsics=intrinsics,
        image_size=image_size,
        best_result=best_result,
        temporal_prior=temporal_prior,
        edge_context=edge_context,
    )
    fast.write_history_csv(output_dir / "optimization_history.csv", best_history)
    if appearance_prior is not None:
        if bool(args.save_color_soft_mask) or bool(args.save_fg_bg_samples) or bool(args.save_candidate_appearance_overlay):
            appearance_debug_outputs = appearance_prior.save_debug_images(
                output_dir,
                render_mask=best_result.get("rendered_mask") if bool(args.save_candidate_appearance_overlay) else None,
            )
            if not bool(args.save_color_soft_mask):
                try:
                    Path(appearance_debug_outputs.get("color_soft_mask", "")).unlink(missing_ok=True)
                except Exception:
                    pass
                appearance_debug_outputs.pop("color_soft_mask", None)
            if not bool(args.save_fg_bg_samples):
                try:
                    Path(appearance_debug_outputs.get("fg_bg_samples", "")).unlink(missing_ok=True)
                except Exception:
                    pass
                appearance_debug_outputs.pop("fg_bg_samples", None)
    breakdown_path = save_generic_breakdown(output_dir, [item[0] for item in refined_results]) if bool(args.save_score_breakdown) else None

    optimized_task = json.loads(json.dumps(task))
    optimized_task.setdefault("corrected_pose", {})
    optimized_task["corrected_pose"]["translation_world"] = fast.to_builtin(translation_world)
    optimized_task["corrected_pose"]["rotation_matrix"] = fast.to_builtin(rotation_world)
    optimized_task["corrected_pose"]["scale"] = fast.to_builtin(best_result["scale"])
    with (output_dir / "task_with_optimized_corrected_pose.json").open("w", encoding="utf-8") as f:
        json.dump(fast.to_builtin(optimized_task), f, indent=2)

    temporal_report.update(
        {
            "temporal_score": best_result.get("temporal_score"),
            "generic_temporal_loss": best_result.get("generic_temporal_loss"),
            "used_temporal_seed": temporal_seed is not None,
            "best_started_from_temporal_seed": best_result.get("initializer_metadata", {}).get("source") == "temporal_prior",
        }
    )
    depth_report = dict(depth_info)
    depth_report.update(
        {
            "enabled": bool(args.depth_enabled),
            "depth_score": best_result.get("depth_score"),
            "depth_confidence": best_result.get("depth_confidence"),
            "depth_error": best_result.get("depth_error"),
            "valid_depth_ratio": best_result.get("valid_depth_ratio"),
        }
    )
    appearance_report.update(
        {
            "appearance_score": best_result.get("appearance_score"),
            "appearance_confidence": best_result.get("appearance_confidence"),
            "color_soft_iou": best_result.get("color_soft_iou"),
            "color_precision": best_result.get("color_precision"),
            "color_recall": best_result.get("color_recall"),
            "background_leakage": best_result.get("background_leakage"),
        }
    )

    report = {
        "task_id": task["task_id"],
        "object_id": task["object_id"],
        "label": task["label"],
        "sample_dir": str(sample_dir),
        "mesh_path": str(mesh_path),
        "image_size": list(image_size),
        "json_bbox": json_bbox,
        "mask_placement": mask_placement,
        "mask_observations": obs,
        "mesh_axis_metadata": mesh_meta,
        "camera_intrinsics": intrinsics,
        "render_backend": full_evaluator.active_backend or full_evaluator.backend_preference,
        "bbox_weight": args.bbox_weight,
        "mode": "generic_appearance_temporal",
        "scale_constraint": "uniform_xyz",
        "optimized_uniform_scale": best_uniform_scale,
        "initializer_top_candidates": preview_rows,
        "best_initializer_metadata": best_result.get("initializer_metadata", {}),
        "best_candidate_rank": best_result.get("candidate_rank"),
        "appearance": appearance_report,
        "depth": depth_report,
        "temporal": temporal_report,
        "optional_priors": {
            "road_plane_enabled": bool(getattr(args, "road_constraint_enabled", False)),
            "heading_enabled": bool(getattr(args, "heading_prior_enabled", False)),
            "support_plane_enabled": str(getattr(args, "support_plane_enabled", "auto")),
            "upright_enabled": str(getattr(args, "generic_upright_enabled", "auto")),
            "optional_prior_score": best_result.get("optional_prior_score"),
            "support_plane_confidence": best_result.get("support_plane_confidence"),
            "support_plane": support_plane_report_payload(support_plane),
            "support_contact_score": best_result.get("support_contact_score"),
            "support_contact_distance_score": best_result.get("support_contact_distance_score"),
            "support_contact_coverage": best_result.get("support_contact_coverage"),
            "support_contact_mean_abs_m": best_result.get("support_contact_mean_abs_m"),
            "support_contact_max_abs_m": best_result.get("support_contact_max_abs_m"),
            "support_bottom_signed_m": best_result.get("support_bottom_signed_m"),
            "support_floating_distance_m": best_result.get("support_floating_distance_m"),
            "support_penetration_distance_m": best_result.get("support_penetration_distance_m"),
            "support_penalty": best_result.get("support_penalty"),
            "support_contact_penalty_eff": best_result.get("support_contact_penalty_eff"),
            "support_bottom_selection_mode": best_result.get("support_bottom_selection_mode"),
            "support_axis_index": best_result.get("support_axis_index"),
            "support_axis_sign": best_result.get("support_axis_sign"),
            "support_normal_alignment": best_result.get("support_normal_alignment"),
            "support_normal_angle_deg": best_result.get("support_normal_angle_deg"),
            "support_orientation_score": best_result.get("support_orientation_score"),
            "support_orientation_penalty": best_result.get("support_orientation_penalty"),
            "support_orientation_penalty_eff": best_result.get("support_orientation_penalty_eff"),
        },
        "edge_assist": {
            "enabled": bool(args.edge_score_enabled),
            "available": edge_context is not None,
            "edge_score": best_result.get("edge_score"),
            "edge_confidence": best_result.get("edge_confidence"),
            "effective_edge_weight": args.generic_edge_weight * float(best_result.get("edge_confidence") or 0.0),
        },
        "optimized_camera_pose": {
            "translation_cam": best_result["translation_cam"],
            "rotation_cam": best_result["rotation_cam"],
            "scale": best_result["scale"],
        },
        "optimized_corrected_pose_world": {
            "translation_world": translation_world,
            "rotation_matrix": rotation_world,
            "scale": best_result["scale"],
        },
        "metrics": {key: best_result.get(key) for key in GENERIC_CANDIDATE_METRIC_KEYS if key in best_result},
        "outputs": {
            "alignment_collage": str(collage_paths["alignment_collage"]),
            "pose_closeup_collage": str(collage_paths["pose_closeup_collage"]),
            "model_reference_collage": str(collage_paths["model_reference_collage"]),
            "temporal_edge_debug": str(temporal_debug_path) if temporal_debug_path else None,
            "optimization_history": str(output_dir / "optimization_history.csv"),
            "optimization_report": str(output_dir / "optimization_report.json"),
            "optimized_task": str(output_dir / "task_with_optimized_corrected_pose.json"),
            "score_breakdown": str(breakdown_path) if breakdown_path else None,
            **appearance_debug_outputs,
            **support_debug_outputs,
        },
        "refined_pose_candidates": [
            candidate_summary(item[0], t_world_from_cam=t_world_from_cam)
            for item in refined_results
        ],
    }
    if args.profile_timings:
        report["profiling"] = fast.combine_profile_stats([proxy_evaluator, full_evaluator])
    with (output_dir / "optimization_report.json").open("w", encoding="utf-8") as f:
        json.dump(fast.to_builtin(report), f, indent=2)
    fast.cleanup_result_images(output_dir)
    proxy_evaluator.close()
    full_evaluator.close()
    return report


def add_generic_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--appearance_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--appearance_foreground_erode_kernel", type=int, default=5)
    parser.add_argument("--appearance_foreground_min_pixels", type=int, default=80)
    parser.add_argument("--appearance_background_inner_dilate_kernel", type=int, default=9)
    parser.add_argument("--appearance_background_outer_dilate_kernel", type=int, default=31)
    parser.add_argument("--appearance_background_bbox_expand_ratio", type=float, default=0.15)
    parser.add_argument("--appearance_background_min_pixels", type=int, default=120)
    parser.add_argument("--appearance_confidence_low_threshold", type=float, default=0.40)
    parser.add_argument("--appearance_confidence_high_threshold", type=float, default=1.20)
    parser.add_argument("--appearance_min_confidence", type=float, default=0.30)
    parser.add_argument("--appearance_soft_iou_weight", type=float, default=0.45)
    parser.add_argument("--appearance_precision_weight", type=float, default=0.35)
    parser.add_argument("--appearance_recall_weight", type=float, default=0.20)
    parser.add_argument("--appearance_leakage_weight", type=float, default=0.25)

    parser.add_argument("--depth_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observed_depth_map_path", default="")
    parser.add_argument("--depth_min_valid_ratio", type=float, default=0.25)
    parser.add_argument("--depth_sigma", type=float, default=0.50)
    parser.add_argument("--depth_robust_stat", choices=["median", "mean"], default="median")
    parser.add_argument("--depth_render_face_limit", type=int, default=8000)
    parser.add_argument("--depth_outlier_score_threshold", type=float, default=0.10)
    parser.add_argument("--depth_outlier_penalty", type=float, default=0.0)

    parser.add_argument("--generic_mask_weight", type=float, default=1.00)
    parser.add_argument("--generic_bbox_weight", type=float, default=0.15)
    parser.add_argument("--generic_contour_weight", type=float, default=0.35)
    parser.add_argument("--generic_edge_weight", type=float, default=0.20)
    parser.add_argument("--generic_depth_weight", type=float, default=0.35)
    parser.add_argument("--generic_appearance_weight", type=float, default=0.25)
    parser.add_argument("--generic_temporal_weight", type=float, default=0.55)
    parser.add_argument("--generic_scale_prior_weight", type=float, default=0.30)
    parser.add_argument("--generic_scale_prior_sigma_log", type=float, default=0.20)
    parser.add_argument("--generic_contour_sigma_px", type=float, default=4.0)

    parser.add_argument("--generic_temporal_translation_sigma", type=float, default=1.00)
    parser.add_argument("--generic_temporal_depth_sigma", type=float, default=0.80)
    parser.add_argument("--generic_temporal_rotation_sigma_deg", type=float, default=25.0)
    parser.add_argument("--generic_temporal_scale_sigma_log", type=float, default=0.20)
    parser.add_argument("--generic_temporal_use_yaw_specific_term", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--generic_rotation_grid_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generic_rotation_grid_source_top_k", type=int, default=2)
    parser.add_argument("--generic_yaw_degrees", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--generic_pitch_degrees", default="-10,0,10")
    parser.add_argument("--generic_roll_degrees", default="-10,0,10")
    parser.add_argument("--generic_coarse_scoring", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--support_plane_enabled", default="auto")
    parser.add_argument("--support_plane_min_confidence", type=float, default=0.70)
    parser.add_argument("--support_plane_min_points", type=int, default=120)
    parser.add_argument("--support_plane_ransac_iters", type=int, default=96)
    parser.add_argument("--support_plane_ransac_threshold_m", type=float, default=0.05)
    parser.add_argument("--support_plane_residual_scale_m", type=float, default=0.08)
    parser.add_argument("--support_use_lower_band", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support_lower_band_x_expand_ratio", type=float, default=0.20)
    parser.add_argument("--support_lower_band_y_extend_ratio", type=float, default=0.60)
    parser.add_argument("--support_use_near_mask_ring", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support_near_mask_inner_kernel", type=int, default=9)
    parser.add_argument("--support_near_mask_outer_kernel", type=int, default=31)
    parser.add_argument("--support_bbox_expand_ratio", type=float, default=0.20)
    parser.add_argument("--support_exclude_target_mask_dilate_kernel", type=int, default=9)
    parser.add_argument("--support_plane_weight", type=float, default=0.20)
    parser.add_argument("--support_penalty_weight", type=float, default=0.15)
    parser.add_argument("--support_bottom_percentile", type=float, default=3.0)
    parser.add_argument("--support_bottom_selection_mode", choices=["local_axis", "signed_distance", "plane_distance"], default="local_axis")
    parser.add_argument("--support_contact_sigma_m", type=float, default=0.10)
    parser.add_argument("--support_contact_tolerance_m", type=float, default=0.08)
    parser.add_argument("--support_floating_tolerance_m", type=float, default=0.20)
    parser.add_argument("--support_penetration_tolerance_m", type=float, default=0.10)
    parser.add_argument("--support_contact_distance_score_weight", type=float, default=0.70)
    parser.add_argument("--support_contact_coverage_score_weight", type=float, default=0.30)
    parser.add_argument("--support_floating_penalty_weight", type=float, default=0.60)
    parser.add_argument("--support_penetration_penalty_weight", type=float, default=1.00)
    parser.add_argument("--support_orientation_penalty_weight", type=float, default=0.0)
    parser.add_argument("--support_orientation_sigma_deg", type=float, default=20.0)
    parser.add_argument("--support_orientation_tolerance_deg", type=float, default=4.0)
    parser.add_argument("--optional_prior_gate_start", type=float, default=0.35)
    parser.add_argument("--optional_prior_gate_range", type=float, default=0.45)
    parser.add_argument("--generic_upright_enabled", default="auto")
    parser.add_argument("--generic_upright_min_confidence", type=float, default=0.70)
    parser.add_argument("--generic_upright_weight", type=float, default=0.15)
    parser.add_argument("--generic_heading_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generic_heading_weight", type=float, default=0.0)

    parser.add_argument("--generic_acceptance_min_visible_mask_iou", type=float, default=0.12)
    parser.add_argument("--generic_acceptance_min_bbox_iou", type=float, default=0.10)
    parser.add_argument("--generic_acceptance_max_center_error_ratio", type=float, default=0.35)
    parser.add_argument("--generic_acceptance_min_projection_valid_ratio", type=float, default=0.50)
    parser.add_argument("--generic_acceptance_depth_confidence_high", type=float, default=0.70)
    parser.add_argument("--generic_acceptance_depth_min_threshold", type=float, default=0.25)

    parser.add_argument("--save_color_soft_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_fg_bg_samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_candidate_appearance_overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_score_breakdown", action=argparse.BooleanOptionalAction, default=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic pose optimizer with image appearance, depth consistency, and SE(3) temporal smoothness."
    )
    temporal_fast.add_fast_arguments(parser)
    temporal_fast.add_temporal_arguments(parser)
    add_generic_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = optimize_sample(args)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1) from None
    metrics = report["metrics"]
    pose_world = report["optimized_corrected_pose_world"]
    print(f"task_id: {report['task_id']}")
    print(f"best_score: {float(metrics.get('score') or 0.0):.6f}")
    print(f"best_mask_iou: {float(metrics.get('mask_iou') or 0.0):.6f}")
    print(f"best_bbox_iou: {float(metrics.get('bbox_iou') or 0.0):.6f}")
    print(f"appearance_score: {float(metrics.get('appearance_score') or 0.0):.6f}")
    print(f"depth_score: {float(metrics.get('depth_score') or 0.0):.6f}")
    print(f"optimized_translation_world: {fast.to_builtin(pose_world['translation_world'])}")
    print(f"optimized_scale: {fast.to_builtin(pose_world['scale'])}")
    print(f"report_path: {report['outputs']['optimization_report']}")


if __name__ == "__main__":
    main()
