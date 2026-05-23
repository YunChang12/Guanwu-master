#!/usr/bin/env python3
"""Temporal fast pose optimizer.

This strategy layers a previous-frame prior, partial-visibility scoring, and
edge-assisted scoring on top of the existing fast optimizer utilities.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import fast


TEMPORAL_EXTRA_HISTORY_KEYS = [
    "base_geometry_score",
    "geometry_score",
    "temporal_score",
    "temporal_loss",
    "edge_score",
    "partial_mask_score",
    "adjusted_mask_score",
    "partial_score_boost",
    "visible_mask_iou",
    "visible_soft_mask_iou",
    "visible_target_fraction",
    "visible_target_area_px",
    "visible_bbox_iou",
    "visible_bbox_center_error_px",
    "visible_contour_score",
    "visible_contour_chamfer_score",
    "visible_profile_score",
    "visible_contour_mean_distance_px",
    "visible_profile_mean_distance_px",
    "visible_profile_coverage",
    "effective_visible_contour_weight",
    "truncation_severity",
    "low_observability",
    "truncation_observability_score",
    "truncation_observability_reasons",
    "truncated_visual_quality_gate",
    "truncated_visual_quality_reason",
    "truncated_visual_bbox_factor",
    "truncated_visual_center_factor",
    "truncated_visual_mask_factor",
    "truncated_visual_contour_factor",
    "truncated_visual_profile_factor",
    "truncated_visual_overflow_factor",
    "truncated_visual_overflow_loss",
    "truncated_visual_quality_penalty",
    "truncated_bbox_score",
    "truncated_bbox_loss",
    "truncated_bbox_penalty",
    "ground_contact_score",
    "ground_contact_mean_abs_m",
    "ground_contact_max_abs_m",
    "ground_gate_passed",
    "ground_gate_rejected",
    "ground_contact_penalty",
    "bbox_bottom_score",
    "upright_score",
    "upright_angle_error_deg",
    "upright_gate_passed",
    "upright_gate_rejected",
    "upright_gate_penalty",
    "heading_prior_score",
    "heading_prior_angle_error_deg",
    "heading_front_sign_enabled",
    "heading_front_sign_confidence",
    "heading_depth_trend_score",
    "heading_depth_trend_direction",
    "heading_depth_trend_confidence",
    "heading_front_depth_cam",
    "heading_front_sign_penalty",
    "effective_front_sign_penalty_weight",
    "visual_gate_factor",
    "visual_gate_reason",
    "visual_gate_mask_iou_min",
    "visual_gate_bbox_iou_min",
    "visual_gate_center_error_px_max",
    "visible_mask_bonus",
    "visible_mask_bonus_weight",
    "effective_temporal_weight",
    "effective_edge_weight",
    "effective_ground_contact_weight",
    "effective_bbox_bottom_weight",
    "effective_upright_weight",
    "effective_heading_prior_weight",
    "final_score",
    "is_truncated",
    "prior_frame_id",
]


def parse_task_id_from_sample_dir(sample_dir: str | Path) -> tuple[str, int]:
    """Parse obj_000001@000003 into (obj_000001, 3)."""
    name = Path(sample_dir).name
    match = re.fullmatch(r"(.+)@(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse object/frame id from sample_dir name: {name!r}")
    return match.group(1), int(match.group(2))


def parse_suffixes(value: str | list[Any] | tuple[Any, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def world_pose_to_camera_pose(
    t_world_from_cam: np.ndarray,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_world_from_object = fast.make_transform(rotation_world, translation_world)
    t_cam_from_world = np.linalg.inv(t_world_from_cam)
    t_cam_from_object = t_cam_from_world @ t_world_from_object
    return t_cam_from_object[:3, 3].copy(), t_cam_from_object[:3, :3].copy()


def load_prior_pose(
    pose_path: Path,
    t_world_from_cam: np.ndarray,
) -> dict[str, Any] | None:
    data = fast.read_json(pose_path)
    pose_source = pose_path.name
    prior_mask_area_px: int | None = None

    report_path = pose_path.parent / "optimization_report.json"
    if report_path.exists():
        try:
            report = fast.read_json(report_path)
            prior_mask_area_px = int(report.get("mask_observations", {}).get("area_px", 0)) or None
        except Exception:
            prior_mask_area_px = None

    if pose_path.name == "task_with_optimized_corrected_pose.json":
        pose = data.get("corrected_pose", {})
        if not pose:
            return None
        translation_world = np.asarray(pose["translation_world"], dtype=np.float64)
        rotation_world = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation_cam, rotation_cam = world_pose_to_camera_pose(
            t_world_from_cam,
            translation_world,
            rotation_world,
        )
        scale = fast.make_uniform_scale(
            fast.scale_to_uniform_scalar(np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
        )
        return {
            "pose_source": pose_source,
            "translation_world": translation_world,
            "rotation_world": rotation_world,
            "translation_cam": translation_cam,
            "rotation_cam": rotation_cam,
            "scale": scale,
            "prior_mask_area_px": prior_mask_area_px,
        }

    world_pose = data.get("optimized_corrected_pose_world")
    if world_pose:
        translation_world = np.asarray(world_pose["translation_world"], dtype=np.float64)
        rotation_world = np.asarray(world_pose["rotation_matrix"], dtype=np.float64)
        translation_cam, rotation_cam = world_pose_to_camera_pose(
            t_world_from_cam,
            translation_world,
            rotation_world,
        )
        scale = fast.make_uniform_scale(
            fast.scale_to_uniform_scalar(np.asarray(world_pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
        )
        return {
            "pose_source": pose_source,
            "translation_world": translation_world,
            "rotation_world": rotation_world,
            "translation_cam": translation_cam,
            "rotation_cam": rotation_cam,
            "scale": scale,
            "prior_mask_area_px": prior_mask_area_px,
        }

    camera_pose = data.get("optimized_camera_pose")
    if camera_pose:
        return {
            "pose_source": pose_source,
            "translation_world": None,
            "rotation_world": None,
            "translation_cam": np.asarray(camera_pose["translation_cam"], dtype=np.float64),
            "rotation_cam": np.asarray(camera_pose["rotation_cam"], dtype=np.float64),
            "scale": fast.make_uniform_scale(
                fast.scale_to_uniform_scalar(np.asarray(camera_pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
            ),
            "prior_mask_area_px": prior_mask_area_px,
        }

    return None


def load_prior_pose_payload(
    payload: dict[str, Any] | None,
    t_world_from_cam: np.ndarray,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    pose = payload.get("pose") if isinstance(payload.get("pose"), dict) else payload
    if not isinstance(pose, dict):
        return None
    try:
        translation_world = np.asarray(pose["translation_world"], dtype=np.float64)
        rotation_world = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        scale = fast.make_uniform_scale(
            fast.scale_to_uniform_scalar(np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
        )
    except Exception:
        return None
    if translation_world.shape != (3,) or rotation_world.shape != (3, 3):
        return None
    translation_cam, rotation_cam = world_pose_to_camera_pose(
        t_world_from_cam,
        translation_world,
        rotation_world,
    )
    return {
        "pose_source": str(payload.get("source") or "task_temporal_prior_pose"),
        "translation_world": translation_world,
        "rotation_world": rotation_world,
        "translation_cam": translation_cam,
        "rotation_cam": rotation_cam,
        "scale": scale,
        "prior_mask_area_px": payload.get("prior_mask_area_px"),
        "frame_idx": int(payload.get("frame_id") or payload.get("frame_idx") or 0),
        "output_dir": payload.get("output_dir"),
        "path": payload.get("path"),
    }


def find_temporal_prior(
    output_dir: Path,
    object_id: str,
    frame_idx: int,
    lookback: int,
    suffixes: list[str],
    t_world_from_cam: np.ndarray,
) -> dict[str, Any] | None:
    output_root = output_dir.parent
    start = frame_idx - 1
    stop = max(1, frame_idx - int(lookback))
    for prior_frame_idx in range(start, stop - 1, -1):
        task_id = f"{object_id}@{prior_frame_idx:06d}"
        for suffix in ["", *suffixes]:
            candidate_dir = output_root / f"{task_id}{suffix}"
            if not candidate_dir.exists():
                continue
            for filename in ("task_with_optimized_corrected_pose.json", "optimization_report.json"):
                candidate_path = candidate_dir / filename
                if not candidate_path.exists():
                    continue
                try:
                    prior = load_prior_pose(candidate_path, t_world_from_cam)
                except Exception as exc:
                    print(f"[warn] temporal prior load failed: {candidate_path} ({exc})")
                    continue
                if prior is None:
                    continue
                prior.update(
                    {
                        "object_id": object_id,
                        "frame_idx": int(prior_frame_idx),
                        "output_dir": str(candidate_dir),
                        "path": str(candidate_path),
                    }
                )
                return prior
    return None


def should_skip_disk_temporal_prior(vehicle_pose_context: dict[str, Any] | None) -> bool:
    """Return True when temporal candidates must not read stale result dirs.

    The all-frames pipeline first generates independent per-frame candidates and
    only later chooses a smooth trajectory.  Looking up the previous frame from
    disk during that candidate pass can make an early wrong heading become a
    strong temporal seed for every later frame.
    """

    if not isinstance(vehicle_pose_context, dict):
        return False
    if bool(vehicle_pose_context.get("disable_disk_temporal_prior")):
        return True
    window = vehicle_pose_context.get("temporal_window")
    if not isinstance(window, dict):
        return False
    return str(window.get("mode") or "").lower() == "all_frames"


def rotation_angle_deg(rotation_delta: np.ndarray) -> float:
    value = (float(np.trace(rotation_delta)) - 1.0) * 0.5
    value = max(-1.0, min(1.0, value))
    return float(math.degrees(math.acos(value)))


def matrix_to_euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    """Inverse of fast.euler_xyz_to_matrix for Rz @ Ry @ Rx."""
    r = np.asarray(rotation, dtype=np.float64)
    sy = max(-1.0, min(1.0, -float(r[2, 0])))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rx = 0.0
        rz = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return rx, ry, rz


def compute_temporal_score(
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    prior: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
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
    delta_rotation_deg = rotation_angle_deg(delta_rotation)
    rx, ry, rz = matrix_to_euler_xyz(delta_rotation)
    delta_pitch_deg = abs(math.degrees(rx))
    delta_yaw_deg = abs(math.degrees(ry))
    delta_roll_deg = abs(math.degrees(rz))
    delta_scale_log = float(math.log(max(1e-8, scale_value) / max(1e-8, prior_scale)))

    translation_sigma = max(1e-6, float(args.temporal_translation_sigma))
    depth_sigma = max(1e-6, float(args.temporal_depth_sigma))
    rotation_sigma = max(1e-6, float(args.temporal_rotation_sigma_deg))
    yaw_sigma = max(1e-6, float(args.temporal_yaw_sigma_deg))
    scale_sigma = max(1e-6, float(args.temporal_scale_sigma))

    loss = (
        (delta_translation / translation_sigma) ** 2
        + (delta_depth / depth_sigma) ** 2
        + (delta_rotation_deg / rotation_sigma) ** 2
        + (delta_yaw_deg / yaw_sigma) ** 2
        + (delta_scale_log / scale_sigma) ** 2
    )

    max_jump = float(args.temporal_max_allowed_jump_deg)
    if delta_rotation_deg > max_jump:
        loss += ((delta_rotation_deg - max_jump) / rotation_sigma) ** 2

    max_scale_ratio = max(1.0 + 1e-6, float(args.temporal_max_allowed_scale_ratio))
    scale_ratio = max(scale_value, prior_scale) / max(1e-8, min(scale_value, prior_scale))
    if scale_ratio > max_scale_ratio:
        loss += (math.log(scale_ratio / max_scale_ratio) / scale_sigma) ** 2

    temporal_score = float(math.exp(-min(60.0, loss)))
    return {
        "delta_translation": delta_translation_vec,
        "delta_translation_norm": delta_translation,
        "delta_depth": delta_depth,
        "delta_rotation_deg": delta_rotation_deg,
        "delta_pitch_deg": delta_pitch_deg,
        "delta_yaw_deg": delta_yaw_deg,
        "delta_roll_deg": delta_roll_deg,
        "delta_scale_log": delta_scale_log,
        "temporal_loss": float(loss),
        "temporal_score": temporal_score,
    }


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    av = fast.normalize(np.asarray(a, dtype=np.float64))
    bv = fast.normalize(np.asarray(b, dtype=np.float64))
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom < 1e-8:
        return 180.0
    dot = float(np.clip(np.dot(av, bv), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def compute_road_and_heading_score(
    *,
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    t_world_from_cam: np.ndarray,
    mesh_meta: dict[str, Any],
    vehicle_pose_context: dict[str, Any],
    projected_bbox: list[float] | None,
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    initializer_metadata: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "road_constraint_available": False,
        "ground_contact_score": 0.0,
        "ground_contact_mean_abs_m": None,
        "ground_contact_max_abs_m": None,
        "ground_gate_passed": True,
        "ground_gate_rejected": False,
        "ground_contact_penalty": 0.0,
        "bbox_bottom_score": 0.0,
        "bbox_bottom_distance_m": None,
        "upright_score": 0.0,
        "upright_angle_error_deg": None,
        "upright_gate_passed": True,
        "upright_gate_rejected": False,
        "upright_gate_penalty": 0.0,
        "heading_prior_score": 0.0,
        "heading_prior_angle_error_deg": None,
        "heading_front_sign_enabled": False,
        "heading_front_sign_confidence": 0.0,
        "heading_front_sign_source": None,
        "heading_candidate_forward_sign": None,
        "heading_semantic_front_sign": None,
        "heading_tail_light_front_sign": None,
        "heading_tail_light_flipped": False,
        "heading_front_sign_hard_rejected": False,
        "heading_front_angle_penalty": 0.0,
        "heading_depth_trend_score": None,
        "heading_depth_trend_direction": None,
        "heading_depth_trend_confidence": None,
        "heading_front_depth_cam": None,
        "heading_front_sign_penalty": 0.0,
        "effective_front_sign_penalty_weight": 0.0,
        "effective_ground_contact_weight": 0.0,
        "effective_bbox_bottom_weight": 0.0,
        "effective_upright_weight": 0.0,
        "effective_heading_prior_weight": 0.0,
    }

    axis_prior = mesh_meta.get("axis_prior") if isinstance(mesh_meta.get("axis_prior"), dict) else {}
    up_axis = int(axis_prior.get("up_axis_idx", mesh_meta.get("shortest_axis", 1)))
    up_sign = 1.0 if float(axis_prior.get("up_sign", 1.0)) >= 0 else -1.0
    forward_axis = int(axis_prior.get("forward_axis_idx", mesh_meta.get("longest_axis", 2)))
    default_forward_sign = 1.0 if float(axis_prior.get("forward_sign", 1.0)) >= 0 else -1.0
    forward_sign = candidate_forward_sign({"initializer_metadata": initializer_metadata or {}}, default_forward_sign)

    translation_cam = np.asarray(translation_cam, dtype=np.float64)
    rotation_cam = np.asarray(rotation_cam, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    t_world_from_cam = np.asarray(t_world_from_cam, dtype=np.float64)
    rotation_world = t_world_from_cam[:3, :3] @ rotation_cam
    translation_world = t_world_from_cam[:3, :3] @ translation_cam + t_world_from_cam[:3, 3]

    road = vehicle_pose_context.get("road_constraint", {})
    if bool(getattr(args, "road_constraint_enabled", True)) and isinstance(road, dict) and bool(road.get("available")):
        plane = road.get("road_plane", {})
        try:
            normal_world, offset = fast.oriented_plane(
                np.asarray(plane["normal_world"], dtype=np.float64),
                float(plane["offset"]),
                str(getattr(args, "world_up_axis", "y")),
            )
            result["road_constraint_available"] = True
            bottom_local = fast.bottom_contact_points_local(np.asarray(mesh_meta["bounds"], dtype=np.float64), up_axis, up_sign)
            bottom_world = (rotation_world @ (bottom_local * scale.reshape(1, 3)).T).T + translation_world.reshape(1, 3)
            signed = bottom_world @ normal_world + offset
            abs_dist = np.abs(signed)
            mean_abs = float(np.mean(abs_dist))
            max_abs = float(np.max(abs_dist))
            ground_sigma = max(1e-6, float(getattr(args, "ground_contact_sigma_m", 0.18)))
            ground_score = float(math.exp(-min(60.0, (mean_abs / ground_sigma) ** 2)))
            ground_mean_max = float(getattr(args, "ground_contact_hard_gate_mean_m", 0.30))
            ground_point_max = float(getattr(args, "ground_contact_hard_gate_max_m", 0.60))
            ground_gate_passed = bool(mean_abs <= ground_mean_max and max_abs <= ground_point_max)
            sides = set(truncation_info.get("truncation_sides", []))
            truncated_ground_gate = bool(getattr(args, "truncated_ground_contact_hard_gate_enabled", False))
            ground_gate_applicable = "bottom" not in sides or truncated_ground_gate
            ground_gate_rejected = (
                bool(getattr(args, "ground_contact_hard_gate_enabled", True))
                and ground_gate_applicable
                and not ground_gate_passed
            )

            local_up = fast.axis_vector(up_axis, up_sign)
            up_world = fast.normalize(rotation_world @ local_up)
            upright_angle = angle_between_deg(up_world, normal_world)
            upright_sigma = max(1e-6, float(getattr(args, "upright_angle_sigma_deg", 10.0)))
            upright_score = float(math.exp(-min(60.0, (upright_angle / upright_sigma) ** 2)))
            upright_soft_max = float(getattr(args, "upright_strong_penalty_angle_deg", 15.0))
            upright_max = float(getattr(args, "upright_hard_gate_max_angle_deg", 60.0))
            upright_gate_passed = bool(upright_angle <= upright_max)
            upright_gate_rejected = (
                bool(getattr(args, "upright_hard_gate_enabled", True))
                and not bool(truncation_info.get("is_truncated"))
                and not upright_gate_passed
            )
            upright_gate_penalty = 0.0
            if bool(getattr(args, "upright_hard_gate_enabled", True)) and upright_angle > upright_soft_max:
                excess = max(0.0, upright_angle - upright_soft_max)
                sigma = max(1e-6, float(getattr(args, "upright_hard_gate_sigma_deg", 15.0)))
                penalty_weight = float(getattr(args, "upright_hard_gate_penalty", 2.0))
                upright_gate_penalty = penalty_weight * (excess / sigma) ** 2

            bbox_bottom_score = 0.0
            bbox_bottom_distance = None
            bottom_ref = road.get("bbox_bottom_ground")
            if isinstance(bottom_ref, dict) and "point_world" in bottom_ref:
                bbox_bottom_world = np.asarray(bottom_ref["point_world"], dtype=np.float64)
                center_bottom = bottom_world[-1]
                bbox_bottom_distance = float(abs(np.dot(center_bottom - bbox_bottom_world, normal_world)))
                bbox_sigma = max(1e-6, float(getattr(args, "bbox_bottom_ground_sigma_m", 0.45)))
                bbox_bottom_score = float(math.exp(-min(60.0, (bbox_bottom_distance / bbox_sigma) ** 2)))

            ground_weight = float(getattr(args, "road_constraint_weight", 0.25))
            bbox_weight = float(getattr(args, "bbox_bottom_ground_weight", 0.15))
            ground_penalty = 0.0
            if "bottom" in sides:
                ground_weight *= float(getattr(args, "bottom_truncated_ground_contact_weight_factor", 1.60))
                bbox_weight *= float(getattr(args, "bottom_truncated_ground_weight_factor", 0.25))
                severity = str(truncation_info.get("truncation_severity", "light"))
                if severity == "moderate":
                    bbox_weight *= float(getattr(args, "moderate_bottom_truncated_bbox_bottom_weight_factor", 0.40))
                elif severity == "severe":
                    bbox_weight *= float(getattr(args, "severe_bottom_truncated_bbox_bottom_weight_factor", 0.0))
                tolerance = float(getattr(args, "bottom_truncated_ground_soft_tolerance_m", 0.12))
                penalty_weight = float(getattr(args, "bottom_truncated_ground_penalty_weight", 0.35))
                if mean_abs > tolerance:
                    sigma = max(1e-6, float(getattr(args, "bottom_truncated_ground_penalty_sigma_m", 0.18)))
                    ground_penalty = penalty_weight * ((mean_abs - tolerance) / sigma) ** 2
            upright_weight = float(getattr(args, "upright_weight", 0.10))

            result.update(
                {
                    "ground_contact_score": ground_score,
                    "ground_contact_mean_abs_m": mean_abs,
                    "ground_contact_max_abs_m": max_abs,
                    "ground_gate_passed": ground_gate_passed,
                    "ground_gate_rejected": ground_gate_rejected,
                    "ground_contact_penalty": ground_penalty,
                    "bbox_bottom_score": bbox_bottom_score,
                    "bbox_bottom_distance_m": bbox_bottom_distance,
                    "upright_score": upright_score,
                    "upright_angle_error_deg": upright_angle,
                    "upright_gate_passed": upright_gate_passed,
                    "upright_gate_rejected": upright_gate_rejected,
                    "upright_gate_penalty": upright_gate_penalty,
                    "effective_ground_contact_weight": ground_weight,
                    "effective_bbox_bottom_weight": bbox_weight,
                    "effective_upright_weight": upright_weight,
                }
            )
        except Exception as exc:
            result["road_constraint_error"] = str(exc)

    heading = vehicle_pose_context.get("heading_prior", {})
    if not isinstance(heading, dict):
        heading = {}
    heading_motion_enabled = (
        bool(getattr(args, "heading_prior_enabled", True))
        and bool(heading.get("enabled", True))
    )
    vector_image = heading.get("vector_image") if heading_motion_enabled else None
    motion_target = None
    if isinstance(vector_image, list) and len(vector_image) >= 2:
        candidate_target = np.asarray([float(vector_image[0]), float(vector_image[1])], dtype=np.float64)
        if float(np.linalg.norm(candidate_target)) > 1e-6:
            motion_target = fast.normalize(candidate_target)

    area_trend = heading.get("bbox_area_trend")
    if not isinstance(area_trend, dict):
        area_trend = None

    tail_prior = mesh_meta.get("tail_light_prior")
    if not isinstance(tail_prior, dict):
        tail_prior = vehicle_pose_context.get("mesh_tail_light_prior", {})

    local_candidate_forward = fast.axis_vector(forward_axis, forward_sign)
    candidate_forward_cam = fast.normalize(rotation_cam @ local_candidate_forward)
    semantic_forward_cam = candidate_forward_cam
    front_sign_enabled = False
    front_confidence = 0.0
    semantic_front_sign = forward_sign
    tail_front_sign = None
    tail_flipped = False
    front_sign_source = None
    sign_mismatch = False

    if (
        bool(getattr(args, "mesh_tail_light_front_sign_enabled", True))
        and isinstance(tail_prior, dict)
        and bool(tail_prior.get("available"))
        and int(tail_prior.get("axis_idx", forward_axis)) == forward_axis
    ):
        front_confidence = float(np.clip(float(tail_prior.get("confidence", 0.0) or 0.0), 0.0, 1.0))
        min_conf = float(getattr(args, "mesh_tail_light_front_sign_min_confidence", 0.35))
        standalone_min_conf = float(getattr(args, "mesh_tail_light_front_sign_standalone_min_confidence", 0.75))
        standalone_min_ratio = float(getattr(args, "mesh_tail_light_front_sign_standalone_min_density_ratio", 5.0))
        density_ratio = float(tail_prior.get("density_ratio", 0.0) or 0.0)
        strong_tail_prior = (
            bool(tail_prior.get("strong_available"))
            and front_confidence >= standalone_min_conf
            and density_ratio >= standalone_min_ratio
        )
        if front_confidence >= min_conf and (motion_target is not None or strong_tail_prior):
            front_sign_enabled = True
            front_sign_source = "mesh_tail_light"
            tail_front_sign = -1.0 if float(tail_prior.get("front_sign", forward_sign)) < 0.0 else 1.0
            semantic_front_sign = tail_front_sign
            semantic_forward_cam = fast.normalize(rotation_cam @ fast.axis_vector(forward_axis, semantic_front_sign))
            sign_mismatch = float(forward_sign) != float(semantic_front_sign)

            # A strong red-tail-light prior is a mesh semantic fact. Only weak priors
            # may be flipped to avoid forcing noisy color cues onto the optimizer.
            if (
                not strong_tail_prior
                and bool(getattr(args, "tail_light_motion_consistency_flip_enabled", True))
                and isinstance(area_trend, dict)
                and area_trend.get("direction") in ("approaching", "receding")
                and float(area_trend.get("confidence", 0.0) or 0.0) >= float(getattr(args, "tail_light_motion_consistency_min_confidence", 0.60))
            ):
                expected = -1.0 if area_trend.get("direction") == "approaching" else 1.0
                tail_score = float(np.clip(0.5 * (1.0 + expected * float(semantic_forward_cam[2])), 0.0, 1.0))
                opposite_forward_cam = fast.normalize(rotation_cam @ fast.axis_vector(forward_axis, -semantic_front_sign))
                opposite_score = float(np.clip(0.5 * (1.0 + expected * float(opposite_forward_cam[2])), 0.0, 1.0))
                if opposite_score > tail_score + float(getattr(args, "tail_light_motion_consistency_flip_margin", 0.20)):
                    tail_flipped = True
                    tail_front_sign = -tail_front_sign
                    semantic_front_sign = tail_front_sign
                    semantic_forward_cam = opposite_forward_cam
                    sign_mismatch = float(forward_sign) != float(semantic_front_sign)

    if (
        not front_sign_enabled
        and bool(getattr(args, "bbox_area_trend_front_sign_enabled", True))
        and isinstance(area_trend, dict)
        and area_trend.get("direction") in ("approaching", "receding")
        and not bool(area_trend.get("truncated_tail"))
        and not bool(truncation_info.get("is_truncated"))
    ):
        trend_conf = float(np.clip(float(area_trend.get("confidence", 0.0) or 0.0), 0.0, 1.0))
        trend_monotonicity = float(np.clip(float(area_trend.get("monotonicity", 1.0) or 0.0), 0.0, 1.0))
        axis_confidence = float(np.clip(float(axis_prior.get("confidence", 1.0) or 0.0), 0.0, 1.0))
        if (
            trend_conf >= float(getattr(args, "bbox_area_trend_front_sign_min_confidence", 0.75))
            and trend_monotonicity >= float(getattr(args, "bbox_area_trend_front_sign_min_monotonicity", 0.75))
            and axis_confidence >= float(getattr(args, "bbox_area_trend_front_sign_min_axis_confidence", 0.50))
        ):
            front_sign_enabled = True
            front_sign_source = "bbox_area_trend"
            front_confidence = float(
                np.clip(
                    trend_conf
                    * trend_monotonicity
                    * axis_confidence
                    * float(getattr(args, "bbox_area_trend_front_sign_confidence_scale", 0.90)),
                    0.0,
                    1.0,
                )
            )
            semantic_front_sign = default_forward_sign
            semantic_forward_cam = fast.normalize(rotation_cam @ fast.axis_vector(forward_axis, semantic_front_sign))
            sign_mismatch = float(forward_sign) != float(semantic_front_sign)

    bbox_motion_front_sign_only = front_sign_source == "bbox_area_trend"
    hard_reject = False
    heading_confidence = float(heading.get("confidence", 1.0) or 0.0) if heading_motion_enabled else 0.0
    heading_confidence = float(np.clip(heading_confidence, 0.0, 1.0))
    if front_sign_enabled:
        result.update(
            {
                "heading_front_sign_enabled": True,
                "heading_front_sign_confidence": front_confidence,
                "heading_front_sign_source": front_sign_source,
                "heading_candidate_forward_sign": forward_sign,
                "heading_semantic_front_sign": semantic_front_sign,
                "heading_tail_light_front_sign": tail_front_sign,
                "heading_tail_light_flipped": tail_flipped,
                "heading_prior_confidence": heading_confidence if motion_target is not None else front_confidence,
            }
        )

    if motion_target is not None:
        heading_forward_cam = semantic_forward_cam if front_sign_enabled else candidate_forward_cam
        projected = np.asarray([heading_forward_cam[0], heading_forward_cam[1]], dtype=np.float64)
        if float(np.linalg.norm(projected)) > 1e-6:
            projected = fast.normalize(projected)
            angle = angle_between_deg(projected, motion_target)
            if not front_sign_enabled and not bool(getattr(args, "heading_prior_lock_front_sign", False)):
                angle = min(angle, 180.0 - angle)
            sigma = max(1e-6, float(getattr(args, "heading_prior_sigma_deg", 25.0)))
            planar_score = float(math.exp(-min(60.0, (angle / sigma) ** 2)))
            score = planar_score
            confidence = heading_confidence
            weight = float(getattr(args, "heading_prior_weight", 0.06)) * confidence
            front_angle_penalty = 0.0
            if front_sign_enabled:
                confidence *= front_confidence
                weight = float(getattr(args, "front_sign_heading_prior_weight", 0.18)) * confidence
                if sign_mismatch:
                    result["heading_front_sign_penalty"] = max(
                        float(result.get("heading_front_sign_penalty") or 0.0),
                        float(getattr(args, "front_sign_mismatch_penalty", 0.80)) * confidence,
                    )
                if bbox_motion_front_sign_only and angle < 90.0:
                    weight = 0.0
                else:
                    front_angle_penalty = (
                        float(getattr(args, "front_sign_angle_penalty_weight", 1.20))
                        * confidence
                        * (1.0 - planar_score)
                    )
                hard_gate_blocked_for_truncation = (
                    bool(truncation_info.get("is_truncated"))
                    and angle < float(getattr(args, "front_sign_hard_gate_angle_deg", 120.0))
                )
                if (
                    bool(getattr(args, "front_sign_hard_gate_enabled", True))
                    and not hard_gate_blocked_for_truncation
                    and confidence >= float(getattr(args, "front_sign_hard_gate_min_confidence", 0.25))
                    and angle >= float(getattr(args, "front_sign_hard_gate_angle_deg", 120.0))
                ):
                    hard_reject = True
            if bool(truncation_info.get("is_truncated")):
                weight = max(weight, float(getattr(args, "truncated_heading_prior_weight", 0.08)) * confidence)
                if str(truncation_info.get("truncation_severity", "")) == "severe":
                    weight = max(
                        weight,
                        float(getattr(args, "severe_truncation_heading_prior_weight", 0.18)) * max(confidence, heading_confidence),
                    )
            result.update(
                {
                    "heading_prior_score": score,
                    "heading_prior_angle_error_deg": angle,
                    "heading_planar_score": planar_score,
                    "effective_heading_prior_weight": weight,
                    "heading_front_sign_enabled": front_sign_enabled,
                    "heading_front_sign_confidence": front_confidence,
                    "heading_front_sign_source": front_sign_source,
                    "heading_candidate_forward_sign": forward_sign,
                    "heading_semantic_front_sign": semantic_front_sign,
                    "heading_tail_light_front_sign": tail_front_sign,
                    "heading_tail_light_flipped": tail_flipped,
                    "heading_front_sign_hard_rejected": hard_reject,
                    "heading_front_angle_penalty": front_angle_penalty,
                    "heading_prior_confidence": confidence,
                    "heading_prior_projected_vector_image": projected.tolist(),
                    "heading_prior_target_vector_image": motion_target.tolist(),
                }
            )

    if (
        front_sign_enabled
        and bool(getattr(args, "front_sign_depth_trend_enabled", True))
        and isinstance(area_trend, dict)
        and area_trend.get("direction") in ("approaching", "receding")
    ):
        front_depth = float(semantic_forward_cam[2])
        expected = -1.0 if area_trend.get("direction") == "approaching" else 1.0
        depth_score = float(np.clip(0.5 * (1.0 + expected * front_depth), 0.0, 1.0))
        trend_conf = float(np.clip(float(area_trend.get("confidence", 0.0) or 0.0), 0.0, 1.0))
        trend_monotonicity = float(np.clip(float(area_trend.get("monotonicity", 1.0) or 0.0), 0.0, 1.0))
        trend_reliable = (
            trend_monotonicity >= float(getattr(args, "front_sign_depth_trend_min_monotonicity", 0.75))
            and not bool(area_trend.get("truncated_tail"))
        )
        penalty_weight = float(getattr(args, "front_sign_depth_trend_penalty", 0.45))
        front_sign_penalty = max(
            float(result.get("heading_front_sign_penalty") or 0.0),
            penalty_weight * front_confidence * trend_conf * (1.0 - depth_score),
        )
        depth_gate_score_min = float(getattr(args, "front_sign_depth_trend_hard_gate_score_min", 0.35))
        if (
            bool(getattr(args, "front_sign_hard_gate_enabled", True))
            and not bool(truncation_info.get("is_truncated"))
            and front_confidence >= float(getattr(args, "front_sign_hard_gate_min_confidence", 0.25))
            and trend_conf >= float(getattr(args, "tail_light_motion_consistency_min_confidence", 0.60))
            and trend_reliable
            and float(result.get("heading_prior_angle_error_deg") or 180.0) >= float(getattr(args, "front_sign_depth_trend_hard_gate_min_heading_angle_deg", 45.0))
            and depth_score < depth_gate_score_min
        ):
            hard_reject = True
        result.update(
            {
                "heading_depth_trend_score": depth_score,
                "heading_depth_trend_direction": area_trend.get("direction"),
                "heading_depth_trend_confidence": trend_conf,
                "heading_depth_trend_monotonicity": trend_monotonicity,
                "heading_depth_trend_reliable": trend_reliable,
                "heading_depth_trend_truncated_tail": bool(area_trend.get("truncated_tail")),
                "heading_front_depth_cam": front_depth,
                "heading_front_sign_penalty": front_sign_penalty,
                "heading_front_sign_hard_rejected": hard_reject,
            }
        )

    return result


def compute_visual_gate_for_pose_priors(
    *,
    result: dict[str, Any],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Gate non-truncated geometry priors behind reliable visual evidence."""

    if bool(truncation_info.get("is_truncated")):
        return {
            "visual_gate_factor": 1.0,
            "visual_gate_reason": "truncated_partial_scoring",
            "visual_gate_mask_iou_min": None,
            "visual_gate_bbox_iou_min": None,
            "visual_gate_center_error_px_max": None,
        }

    if not bool(getattr(args, "visual_gate_enabled", True)):
        return {
            "visual_gate_factor": 1.0,
            "visual_gate_reason": "disabled",
            "visual_gate_mask_iou_min": None,
            "visual_gate_bbox_iou_min": None,
            "visual_gate_center_error_px_max": None,
        }

    mask_min = float(getattr(args, "visual_gate_mask_iou_min", 0.65))
    bbox_min = float(getattr(args, "visual_gate_bbox_iou_min", 0.75))
    center_max = float(getattr(args, "visual_gate_center_error_px_max", 20.0))
    mask_iou = float(result.get("mask_iou", 0.0) or 0.0)
    bbox_iou = float(result.get("bbox_iou", 0.0) or 0.0)
    center_raw = result.get("bbox_center_error_px")
    center_error = float(center_raw) if center_raw is not None else 1e9

    if mask_iou >= mask_min and bbox_iou >= bbox_min and center_error <= center_max:
        factor = 1.0
        reason = "passed"
    else:
        factor = 0.0
        reason = "failed"

    return {
        "visual_gate_factor": factor,
        "visual_gate_reason": reason,
        "visual_gate_mask_iou_min": mask_min,
        "visual_gate_bbox_iou_min": bbox_min,
        "visual_gate_center_error_px_max": center_max,
    }


def compute_track_scale_prior_score(
    scale: np.ndarray,
    track_prior: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not isinstance(track_prior, dict) or not bool(track_prior.get("available", True)):
        return {
            "track_scale_prior_available": False,
            "track_scale_prior_score": None,
            "track_scale_prior_loss": None,
            "track_scale_prior_value": None,
            "track_scale_prior_delta_log": None,
            "effective_track_scale_prior_weight": 0.0,
        }
    if not bool(getattr(args, "track_scale_prior_enabled", True)):
        return {
            "track_scale_prior_available": False,
            "track_scale_prior_score": None,
            "track_scale_prior_loss": None,
            "track_scale_prior_value": None,
            "track_scale_prior_delta_log": None,
            "effective_track_scale_prior_weight": 0.0,
        }
    try:
        prior_scale = fast.scale_to_uniform_scalar(np.asarray(track_prior.get("scale"), dtype=np.float64))
        scale_value = fast.scale_to_uniform_scalar(np.asarray(scale, dtype=np.float64))
    except Exception:
        return {
            "track_scale_prior_available": False,
            "track_scale_prior_score": None,
            "track_scale_prior_loss": None,
            "track_scale_prior_value": None,
            "track_scale_prior_delta_log": None,
            "effective_track_scale_prior_weight": 0.0,
        }
    if prior_scale <= 1e-8 or scale_value <= 1e-8:
        score = 0.0
        delta_log = None
        loss = None
    else:
        sigma = max(1e-6, float(getattr(args, "track_scale_prior_sigma", 0.10)))
        delta_log = float(math.log(scale_value / prior_scale))
        loss = float((delta_log / sigma) ** 2)
        score = float(math.exp(-min(60.0, loss)))
    return {
        "track_scale_prior_available": True,
        "track_scale_prior_score": score,
        "track_scale_prior_loss": loss,
        "track_scale_prior_value": float(prior_scale),
        "track_scale_prior_delta_log": delta_log,
        "effective_track_scale_prior_weight": float(getattr(args, "track_scale_prior_weight", 0.18)),
    }


def compute_visible_mask_bonus(
    *,
    result: dict[str, Any],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prefer tighter visible-mask overlap after geometry gates pass.

    Truncated objects get their own bonus weight so tuning partial visibility
    does not change the scoring balance for fully visible targets.
    """

    mask_iou = float(result.get("mask_iou", 0.0) or 0.0)
    soft_iou = float(result.get("soft_mask_iou", mask_iou) or mask_iou)
    if bool(truncation_info.get("is_truncated")):
        weight = max(0.0, float(getattr(args, "truncated_visible_mask_bonus_weight", 0.0)))
        if weight <= 0.0:
            return {"visible_mask_bonus": 0.0, "visible_mask_bonus_weight": 0.0}
        visible_iou = float(result.get("adjusted_mask_score", mask_iou) or mask_iou)
        gate = float(result.get("truncated_visual_quality_gate", 1.0) or 1.0)
        bonus = 0.80 * visible_iou + 0.20 * soft_iou
        return {
            "visible_mask_bonus": float(np.clip(bonus, 0.0, 1.0) * np.clip(gate, 0.0, 1.0)),
            "visible_mask_bonus_weight": weight,
        }

    weight = max(0.0, float(getattr(args, "visible_mask_bonus_weight", 0.0)))
    if weight <= 0.0:
        return {"visible_mask_bonus": 0.0, "visible_mask_bonus_weight": 0.0}

    bonus = 0.75 * mask_iou + 0.25 * soft_iou
    return {
        "visible_mask_bonus": float(np.clip(bonus, 0.0, 1.0)),
        "visible_mask_bonus_weight": weight,
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _linear_factor(value: float | None, *, good: float, bad: float, higher_is_better: bool) -> float:
    if value is None:
        return 1.0
    if higher_is_better:
        if value >= good:
            return 1.0
        if value <= bad:
            return 0.0
        return float(np.clip((value - bad) / max(1e-6, good - bad), 0.0, 1.0))
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(np.clip(1.0 - (value - good) / max(1e-6, bad - good), 0.0, 1.0))


def _truncation_severity_rank(severity: Any) -> int:
    order = {"none": 0, "light": 1, "moderate": 2, "severe": 3}
    return order.get(str(severity or "none").lower(), 0)


def classify_truncation_observability(
    truncation_info: dict[str, Any],
    metrics: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Classify truncated targets by how much pose evidence remains visible."""

    if not bool(truncation_info.get("is_truncated")):
        result = {
            "severity": "none",
            "low_observability": False,
            "score": 1.0,
            "reasons": [],
        }
        truncation_info.update(
            {
                "truncation_severity": result["severity"],
                "low_observability": result["low_observability"],
                "truncation_observability_score": result["score"],
                "truncation_observability_reasons": result["reasons"],
            }
        )
        return result

    metrics = metrics or {}
    sides = set(truncation_info.get("truncation_sides") or [])
    visible_mask = _finite_float(metrics.get("visible_mask_iou"))
    visible_target_fraction = _finite_float(metrics.get("visible_target_fraction"))
    visible_contour_mean = _finite_float(metrics.get("visible_contour_mean_distance_px"))
    visible_profile_mean = _finite_float(metrics.get("visible_profile_mean_distance_px"))
    area_drop_ratio = _finite_float(truncation_info.get("area_drop_ratio"))

    moderate_mask = float(getattr(args, "truncation_moderate_visible_mask_iou", 0.78))
    severe_mask = float(getattr(args, "truncation_severe_visible_mask_iou", 0.70))
    moderate_contour = float(getattr(args, "truncation_moderate_contour_mean_px", 5.0))
    severe_contour = float(getattr(args, "truncation_severe_contour_mean_px", 7.0))
    moderate_profile = float(getattr(args, "truncation_moderate_profile_mean_px", 8.0))
    severe_profile = float(getattr(args, "truncation_severe_profile_mean_px", 10.0))
    severe_area_drop = float(getattr(args, "truncation_area_drop_ratio", 0.72))
    moderate_visible_fraction = float(getattr(args, "truncation_moderate_visible_target_fraction", 0.35))
    severe_visible_fraction = float(getattr(args, "truncation_severe_visible_target_fraction", 0.12))

    score = 1.0
    reasons: list[str] = []
    severity_rank = 1

    if "bottom" in sides:
        reasons.append("bottom")
    if "bottom" in sides and ({"left", "right"} & sides):
        severity_rank = max(severity_rank, 3)
        reasons.append("multi_side")

    if area_drop_ratio is not None and area_drop_ratio < severe_area_drop:
        severity_rank = max(severity_rank, 3)
        score = min(score, max(0.0, area_drop_ratio / max(1e-6, severe_area_drop)))
        reasons.append("area_drop")

    if visible_mask is not None:
        if visible_mask < severe_mask:
            severity_rank = max(severity_rank, 3)
            reasons.append("visible_mask")
        elif visible_mask < moderate_mask:
            severity_rank = max(severity_rank, 2)
            reasons.append("visible_mask")
        score = min(
            score,
            _linear_factor(
                visible_mask,
                good=moderate_mask,
                bad=severe_mask,
                higher_is_better=True,
            ),
        )

    if visible_target_fraction is not None:
        if visible_target_fraction < severe_visible_fraction:
            severity_rank = max(severity_rank, 3)
            reasons.append("visible_fraction")
        elif visible_target_fraction < moderate_visible_fraction:
            severity_rank = max(severity_rank, 2)
            reasons.append("visible_fraction")
        score = min(
            score,
            _linear_factor(
                visible_target_fraction,
                good=moderate_visible_fraction,
                bad=severe_visible_fraction,
                higher_is_better=True,
            ),
        )

    if visible_contour_mean is not None:
        if visible_contour_mean > severe_contour:
            severity_rank = max(severity_rank, 3)
            reasons.append("visible_contour")
        elif visible_contour_mean > moderate_contour:
            severity_rank = max(severity_rank, 2)
            reasons.append("visible_contour")
        score = min(
            score,
            _linear_factor(
                visible_contour_mean,
                good=moderate_contour,
                bad=severe_contour,
                higher_is_better=False,
            ),
        )

    if visible_profile_mean is not None:
        if visible_profile_mean > severe_profile:
            severity_rank = max(severity_rank, 3)
            reasons.append("visible_profile")
        elif visible_profile_mean > moderate_profile:
            severity_rank = max(severity_rank, 2)
            reasons.append("visible_profile")
        score = min(
            score,
            _linear_factor(
                visible_profile_mean,
                good=moderate_profile,
                bad=severe_profile,
                higher_is_better=False,
            ),
        )

    severity = ("none", "light", "moderate", "severe")[int(np.clip(severity_rank, 0, 3))]
    low_observability = severity_rank >= 2
    result = {
        "severity": severity,
        "low_observability": bool(low_observability),
        "score": float(np.clip(score, 0.0, 1.0)),
        "reasons": sorted(set(reasons)),
    }
    truncation_info.update(
        {
            "truncation_severity": result["severity"],
            "low_observability": result["low_observability"],
            "truncation_observability_score": result["score"],
            "truncation_observability_reasons": result["reasons"],
        }
    )
    return result


def compute_truncated_visual_quality_gate(
    *,
    result: dict[str, Any],
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Downweight truncated visual rewards when the visible geometry drifts.

    Visible-only mask scoring is necessary for border-truncated objects, but it
    can be fooled by scaling the projection outside the frame.  This gate keeps
    the visible mask/contour reward high only when the clipped bbox and center
    still agree with the observation and the projection does not overflow far
    beyond the truncated image border.
    """

    if not bool(truncation_info.get("is_truncated")):
        return {
            "truncated_visual_quality_gate": 1.0,
            "truncated_visual_quality_reason": "not_truncated",
            "truncated_visual_bbox_factor": 1.0,
            "truncated_visual_center_factor": 1.0,
            "truncated_visual_mask_factor": 1.0,
            "truncated_visual_contour_factor": 1.0,
            "truncated_visual_profile_factor": 1.0,
            "truncated_visual_overflow_factor": 1.0,
            "truncated_visual_overflow_loss": 0.0,
            "truncated_visual_quality_penalty": 0.0,
        }

    if not bool(getattr(args, "truncated_visual_quality_gate_enabled", True)):
        return {
            "truncated_visual_quality_gate": 1.0,
            "truncated_visual_quality_reason": "disabled",
            "truncated_visual_bbox_factor": 1.0,
            "truncated_visual_center_factor": 1.0,
            "truncated_visual_mask_factor": 1.0,
            "truncated_visual_contour_factor": 1.0,
            "truncated_visual_profile_factor": 1.0,
            "truncated_visual_overflow_factor": 1.0,
            "truncated_visual_overflow_loss": 0.0,
            "truncated_visual_quality_penalty": 0.0,
        }

    visible_bbox_iou = float(result.get("visible_bbox_iou", result.get("bbox_iou", 0.0)) or 0.0)
    center_raw = result.get("visible_bbox_center_error_px", result.get("bbox_center_error_px"))
    center_error = float(center_raw) if center_raw is not None else 1e9

    bbox_min = float(getattr(args, "truncated_visual_gate_bbox_iou_min", 0.88))
    bbox_soft = max(1e-6, float(getattr(args, "truncated_visual_gate_bbox_iou_softness", 0.08)))
    bbox_factor = float(np.clip((visible_bbox_iou - bbox_min) / bbox_soft, 0.0, 1.0))

    center_max = float(getattr(args, "truncated_visual_gate_center_error_px", 6.0))
    center_soft = max(1e-6, float(getattr(args, "truncated_visual_gate_center_softness_px", 8.0)))
    center_factor = float(np.clip(1.0 - max(0.0, center_error - center_max) / center_soft, 0.0, 1.0))

    width, height = [float(v) for v in image_size]
    projected = result.get("projected_bbox")
    sides = set(truncation_info.get("truncation_sides", []))
    overflow_loss = 0.0
    if projected is not None:
        px1, py1, px2, py2 = [float(v) for v in projected]
        overflow_sigma = max(1e-6, float(getattr(args, "truncated_visual_gate_overflow_sigma_px", 32.0)))
        if "left" in sides:
            overflow_loss += (max(0.0, -px1) / overflow_sigma) ** 2
        if "right" in sides:
            overflow_loss += (max(0.0, px2 - width) / overflow_sigma) ** 2
        if "top" in sides:
            overflow_loss += (max(0.0, -py1) / overflow_sigma) ** 2
        if "bottom" in sides:
            # Bottom truncation means the true extent is outside the image.
            # Keep large overflow visible in diagnostics, but do not use it as
            # a hard visual-quality gate for otherwise good visible masks.
            overflow_loss += 0.15 * (max(0.0, py2 - height) / overflow_sigma) ** 2
    overflow_factor = float(math.exp(-min(60.0, overflow_loss)))

    visible_mask = _finite_float(result.get("visible_mask_iou"))
    mask_factor = _linear_factor(
        visible_mask,
        good=float(getattr(args, "truncated_visual_gate_visible_mask_iou_good", 0.78)),
        bad=float(getattr(args, "truncated_visual_gate_visible_mask_iou_bad", 0.68)),
        higher_is_better=True,
    )
    contour_factor = _linear_factor(
        _finite_float(result.get("visible_contour_mean_distance_px")),
        good=float(getattr(args, "truncated_visual_gate_contour_mean_px_good", 4.5)),
        bad=float(getattr(args, "truncated_visual_gate_contour_mean_px_bad", 7.0)),
        higher_is_better=False,
    )
    profile_factor = _linear_factor(
        _finite_float(result.get("visible_profile_mean_distance_px")),
        good=float(getattr(args, "truncated_visual_gate_profile_mean_px_good", 7.5)),
        bad=float(getattr(args, "truncated_visual_gate_profile_mean_px_bad", 10.0)),
        higher_is_better=False,
    )

    floor = float(np.clip(getattr(args, "truncated_visual_quality_gate_floor", 0.25), 0.0, 1.0))
    raw_gate = min(bbox_factor, center_factor, mask_factor, contour_factor, profile_factor)
    raw_gate = max(0.0, raw_gate - 0.15 * (1.0 - overflow_factor))
    gate = float(np.clip(floor + (1.0 - floor) * raw_gate, floor, 1.0))
    penalty_weight = float(getattr(args, "truncated_visual_quality_penalty_weight", 0.08))
    penalty = penalty_weight * (1.0 - raw_gate)

    failed = []
    if bbox_factor < 0.999:
        failed.append("visible_bbox")
    if center_factor < 0.999:
        failed.append("visible_center")
    if overflow_factor < 0.999:
        failed.append("border_overflow")
    if mask_factor < 0.999:
        failed.append("visible_mask")
    if contour_factor < 0.999:
        failed.append("visible_contour")
    if profile_factor < 0.999:
        failed.append("visible_profile")
    reason = "passed" if not failed else ",".join(failed)

    return {
        "truncated_visual_quality_gate": gate,
        "truncated_visual_quality_reason": reason,
        "truncated_visual_bbox_factor": bbox_factor,
        "truncated_visual_center_factor": center_factor,
        "truncated_visual_mask_factor": mask_factor,
        "truncated_visual_contour_factor": contour_factor,
        "truncated_visual_profile_factor": profile_factor,
        "truncated_visual_overflow_factor": overflow_factor,
        "truncated_visual_overflow_loss": float(overflow_loss),
        "truncated_visual_quality_penalty": float(penalty),
    }


def detect_truncation(
    mask: np.ndarray,
    bbox: list[float],
    image_size: tuple[int, int],
    args: argparse.Namespace,
    prior_mask_area_px: int | None = None,
) -> dict[str, Any]:
    width, height = [int(v) for v in image_size]
    border_margin = int(args.truncation_border_margin)
    bbox_margin = int(args.truncation_bbox_margin)
    mask_bool = mask.astype(bool)
    ys, xs = np.nonzero(mask_bool)
    sides: list[str] = []
    border_touch: dict[str, bool] = {"left": False, "right": False, "top": False, "bottom": False}

    if len(xs) > 0 and len(ys) > 0:
        border_touch["left"] = int(xs.min()) <= border_margin
        border_touch["right"] = int(xs.max()) >= width - 1 - border_margin
        border_touch["top"] = int(ys.min()) <= border_margin
        border_touch["bottom"] = int(ys.max()) >= height - 1 - border_margin
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_touch = {
        "left": x1 <= bbox_margin,
        "right": x2 >= width - bbox_margin,
        "top": y1 <= bbox_margin,
        "bottom": y2 >= height - bbox_margin,
    }

    for side in ("left", "right", "top", "bottom"):
        if border_touch[side] or bbox_touch[side]:
            sides.append(side)

    area_px = int(mask_bool.sum())
    area_drop_ratio: float | None = None
    area_drop = False
    if prior_mask_area_px and prior_mask_area_px > 0:
        area_drop_ratio = float(area_px / prior_mask_area_px)
        area_drop = area_drop_ratio < 0.72

    bottom_close = y2 >= height - border_margin or bool(border_touch["bottom"])
    is_truncated = bool(sides) or area_drop or bottom_close
    if bottom_close and "bottom" not in sides:
        sides.append("bottom")

    severity = "light" if is_truncated else "none"
    low_observability = False
    return {
        "is_truncated": bool(is_truncated),
        "truncation_sides": sides,
        "border_touch": border_touch,
        "bbox_touch": bbox_touch,
        "mask_area_px": area_px,
        "prior_mask_area_px": prior_mask_area_px,
        "area_drop_ratio": area_drop_ratio,
        "area_drop": bool(area_drop),
        "truncation_severity": severity,
        "low_observability": low_observability,
        "truncation_observability_score": 1.0,
        "truncation_observability_reasons": ["area_drop"] if area_drop else [],
    }


def _visible_region_mask(
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    width, height = [int(v) for v in image_size]
    visible = np.ones((height, width), dtype=bool)
    band = max(0, int(getattr(args, "ignore_truncated_border_band_px", 0)))
    sides = set(truncation_info.get("truncation_sides", []))
    if band <= 0:
        return visible
    if "bottom" in sides:
        visible[max(0, height - band) : height, :] = False
    if "top" in sides:
        visible[: min(height, band), :] = False
    if "left" in sides:
        visible[:, : min(width, band)] = False
    if "right" in sides:
        visible[:, max(0, width - band) : width] = False
    return visible


def _clip_bbox_to_image(bbox: list[float], image_size: tuple[int, int]) -> list[float]:
    width, height = [float(v) for v in image_size]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [
        float(np.clip(x1, 0.0, width)),
        float(np.clip(y1, 0.0, height)),
        float(np.clip(x2, 0.0, width)),
        float(np.clip(y2, 0.0, height)),
    ]


def _bbox_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    return fast.bbox_iou(box_a, box_b)


def _bbox_from_binary_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def compute_visible_bbox_score(
    projected_bbox: list[float],
    target_bbox: list[float],
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
    rendered_mask: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if not truncation_info.get("is_truncated"):
        bbox_iou = fast.bbox_iou(projected_bbox, target_bbox)
        return {
            "visible_bbox_iou": bbox_iou,
            "visible_bbox_center_error_px": fast.bbox_center_error(projected_bbox, target_bbox),
            "visible_projected_bbox": list(projected_bbox),
            "visible_target_bbox": list(target_bbox),
        }

    width, height = [float(v) for v in image_size]
    sides = set(truncation_info.get("truncation_sides", []))
    projected = _clip_bbox_to_image(projected_bbox, image_size)
    target = _clip_bbox_to_image(target_bbox, image_size)
    band = max(0.0, float(getattr(args, "ignore_truncated_border_band_px", 0)))
    visible = _visible_region_mask(image_size, truncation_info, args)

    if rendered_mask is not None and target_mask is not None:
        visible_rendered = np.logical_and(np.asarray(rendered_mask).astype(bool), visible)
        visible_target = np.logical_and(np.asarray(target_mask).astype(bool), visible)
        rendered_bbox = _bbox_from_binary_mask(visible_rendered)
        target_mask_bbox = _bbox_from_binary_mask(visible_target)
        if rendered_bbox is not None and target_mask_bbox is not None:
            return {
                "visible_bbox_iou": _bbox_iou_xyxy(rendered_bbox, target_mask_bbox),
                "visible_bbox_center_error_px": fast.bbox_center_error(rendered_bbox, target_mask_bbox),
                "visible_projected_bbox": rendered_bbox,
                "visible_target_bbox": target_mask_bbox,
                "visible_bbox_source": "visible_mask_bbox",
            }

    if "left" in sides:
        projected[0] = max(projected[0], min(width, band))
        target[0] = max(target[0], min(width, band))
    if "right" in sides:
        projected[2] = min(projected[2], max(0.0, width - band))
        target[2] = min(target[2], max(0.0, width - band))
    if "top" in sides:
        projected[1] = max(projected[1], min(height, band))
        target[1] = max(target[1], min(height, band))
    if "bottom" in sides:
        projected[3] = min(projected[3], max(0.0, height - band))
        target[3] = min(target[3], max(0.0, height - band))

    if projected[2] <= projected[0] or projected[3] <= projected[1] or target[2] <= target[0] or target[3] <= target[1]:
        projected = _clip_bbox_to_image(projected_bbox, image_size)
        target = _clip_bbox_to_image(target_bbox, image_size)

    return {
        "visible_bbox_iou": _bbox_iou_xyxy(projected, target),
        "visible_bbox_center_error_px": fast.bbox_center_error(projected, target),
        "visible_projected_bbox": projected,
        "visible_target_bbox": target,
        "visible_bbox_source": "clipped_projected_bbox",
    }


def compute_partial_mask_score(
    rendered_mask: np.ndarray,
    target_mask: np.ndarray,
    soft_target_mask: np.ndarray | None,
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rendered = rendered_mask.astype(bool)
    target = target_mask.astype(bool)
    original_iou = fast.mask_iou(rendered_mask, target_mask)
    if not truncation_info.get("is_truncated"):
        target_area = float(target.sum())
        return {
            "partial_mask_score": original_iou,
            "adjusted_mask_score": original_iou,
            "original_mask_iou": original_iou,
            "visible_mask_iou": original_iou,
            "visible_soft_mask_iou": fast.soft_mask_iou(rendered_mask, soft_target_mask) if soft_target_mask is not None else original_iou,
            "visible_target_fraction": 1.0,
            "visible_target_area_px": target_area,
        }

    visible = _visible_region_mask(image_size, truncation_info, args)
    visible_rendered = np.logical_and(rendered, visible)
    visible_target = np.logical_and(target, visible)
    visible_mask_iou = fast.mask_iou(visible_rendered.astype(np.uint8), visible_target.astype(np.uint8))
    if soft_target_mask is not None:
        visible_soft = np.where(visible, soft_target_mask, 0.0)
        visible_soft_iou = fast.soft_mask_iou(visible_rendered.astype(np.uint8), visible_soft)
    else:
        visible_soft_iou = visible_mask_iou

    width, height = [int(v) for v in image_size]
    false_positive_weight = visible.astype(np.float32)
    band = max(0, int(args.ignore_truncated_border_band_px))
    sides = set(truncation_info.get("truncation_sides", []))

    if band > 0:
        if "bottom" in sides:
            false_positive_weight[max(0, height - band) : height, :] = 0.0
        if "top" in sides:
            false_positive_weight[: min(height, band), :] = 0.0
        if "left" in sides:
            false_positive_weight[:, : min(width, band)] = 0.0
        if "right" in sides:
            false_positive_weight[:, max(0, width - band) : width] = 0.0

    kernel = np.ones((3, 3), dtype=np.uint8)
    target_for_misses = visible_target
    if visible_target.sum() > 0:
        eroded = cv2.erode(visible_target.astype(np.uint8), kernel, iterations=1).astype(bool)
        if eroded.sum() > max(8, 0.25 * visible_target.sum()):
            target_for_misses = eroded

    true_positive = float(np.logical_and(visible_rendered, visible_target).sum())
    target_area = float(target.sum())
    visible_target_area = float(visible_target.sum())
    visible_target_fraction = 1.0 if target_area <= 0.0 else visible_target_area / target_area
    false_negative = float(np.logical_and(~visible_rendered, target_for_misses).sum())
    false_positive = float((np.logical_and(rendered, ~target).astype(np.float32) * false_positive_weight).sum())
    denom = true_positive + false_negative + false_positive
    partial_score = 0.0 if denom <= 0.0 else true_positive / denom
    weight = float(args.partial_visibility_weight)
    visible_blend = 0.65 * partial_score + 0.35 * visible_mask_iou
    blended = (1.0 - weight) * original_iou + weight * visible_blend
    boost_cap = max(0.0, float(args.partial_score_boost_cap))
    adjusted = min(blended, original_iou + boost_cap)
    return {
        "partial_mask_score": float(partial_score),
        "adjusted_mask_score": float(adjusted),
        "original_mask_iou": float(original_iou),
        "visible_mask_iou": float(visible_mask_iou),
        "visible_soft_mask_iou": float(visible_soft_iou),
        "visible_target_fraction": float(visible_target_fraction),
        "visible_target_area_px": visible_target_area,
        "partial_score_boost": float(adjusted - original_iou),
        "partial_false_positive_weighted": false_positive,
        "partial_false_negative": false_negative,
    }


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) <= 0:
        return np.zeros_like(binary, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return (cv2.dilate(binary, kernel, iterations=1) > cv2.erode(binary, kernel, iterations=1)).astype(bool)


def _profile_score_1d(
    rendered_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    axis: int,
    side: str,
    sigma_px: float,
) -> dict[str, Any] | None:
    rendered = rendered_mask.astype(bool)
    target = target_mask.astype(bool)
    if axis == 0:
        # Per-column top/bottom profile.
        rendered_has = rendered.any(axis=0)
        target_has = target.any(axis=0)
        common = np.where(rendered_has & target_has)[0]
        union = np.where(rendered_has | target_has)[0]
        if len(common) == 0 or len(union) == 0:
            return None
        if side == "top":
            rendered_pos = np.argmax(rendered[:, common], axis=0)
            target_pos = np.argmax(target[:, common], axis=0)
        else:
            rendered_pos = rendered.shape[0] - 1 - np.argmax(rendered[::-1, common], axis=0)
            target_pos = target.shape[0] - 1 - np.argmax(target[::-1, common], axis=0)
    else:
        # Per-row left/right profile.
        rendered_has = rendered.any(axis=1)
        target_has = target.any(axis=1)
        common = np.where(rendered_has & target_has)[0]
        union = np.where(rendered_has | target_has)[0]
        if len(common) == 0 or len(union) == 0:
            return None
        if side == "left":
            rendered_pos = np.argmax(rendered[common, :], axis=1)
            target_pos = np.argmax(target[common, :], axis=1)
        else:
            rendered_pos = rendered.shape[1] - 1 - np.argmax(rendered[common, ::-1], axis=1)
            target_pos = target.shape[1] - 1 - np.argmax(target[common, ::-1], axis=1)

    distances = np.abs(rendered_pos.astype(np.float32) - target_pos.astype(np.float32))
    mean_distance = float(np.clip(distances, 0.0, 50.0).mean())
    coverage = float(len(common) / max(1, len(union)))
    score = float(math.exp(-mean_distance / max(1e-6, sigma_px)) * coverage)
    return {"score": score, "mean_distance_px": mean_distance, "coverage": coverage}


def compute_visible_contour_score(
    rendered_mask: np.ndarray,
    target_mask: np.ndarray,
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not truncation_info.get("is_truncated"):
        return {
            "visible_contour_score": None,
            "visible_contour_chamfer_score": None,
            "visible_profile_score": None,
            "visible_contour_mean_distance_px": None,
            "visible_profile_mean_distance_px": None,
            "effective_visible_contour_weight": 0.0,
        }

    visible = _visible_region_mask(image_size, truncation_info, args)
    rendered_visible = np.logical_and(rendered_mask.astype(bool), visible)
    target_visible = np.logical_and(target_mask.astype(bool), visible)
    rendered_edge = np.logical_and(_mask_contour(rendered_mask), visible)
    target_edge = np.logical_and(_mask_contour(target_mask), visible)
    if not np.any(rendered_edge) or not np.any(target_edge):
        return {
            "visible_contour_score": 0.0,
            "visible_contour_chamfer_score": 0.0,
            "visible_profile_score": 0.0,
            "visible_contour_mean_distance_px": None,
            "visible_profile_mean_distance_px": None,
            "effective_visible_contour_weight": float(getattr(args, "truncated_visible_contour_weight", 0.0)),
        }

    sigma = max(1e-6, float(getattr(args, "truncated_visible_contour_sigma_px", 4.0)))
    target_distance = cv2.distanceTransform(np.where(target_edge, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
    rendered_distance = cv2.distanceTransform(np.where(rendered_edge, 0, 1).astype(np.uint8), cv2.DIST_L2, 3)
    rendered_to_target = target_distance[rendered_edge].astype(np.float32)
    target_to_rendered = rendered_distance[target_edge].astype(np.float32)
    rendered_to_target_mean = float(np.clip(rendered_to_target, 0.0, 50.0).mean())
    target_to_rendered_mean = float(np.clip(target_to_rendered, 0.0, 50.0).mean())
    chamfer_mean = 0.5 * (rendered_to_target_mean + target_to_rendered_mean)
    chamfer_score = float(math.exp(-chamfer_mean / sigma))

    sides = set(truncation_info.get("truncation_sides", []))
    profile_sigma = max(1e-6, float(getattr(args, "truncated_visible_profile_sigma_px", 5.0)))
    profiles: list[dict[str, Any]] = []
    if "top" not in sides:
        score = _profile_score_1d(rendered_visible, target_visible, axis=0, side="top", sigma_px=profile_sigma)
        if score is not None:
            profiles.append(score)
    if "bottom" not in sides:
        score = _profile_score_1d(rendered_visible, target_visible, axis=0, side="bottom", sigma_px=profile_sigma)
        if score is not None:
            profiles.append(score)
    if "left" not in sides:
        score = _profile_score_1d(rendered_visible, target_visible, axis=1, side="left", sigma_px=profile_sigma)
        if score is not None:
            profiles.append(score)
    if "right" not in sides:
        score = _profile_score_1d(rendered_visible, target_visible, axis=1, side="right", sigma_px=profile_sigma)
        if score is not None:
            profiles.append(score)

    if profiles:
        profile_score = float(np.mean([item["score"] for item in profiles]))
        profile_mean = float(np.mean([item["mean_distance_px"] for item in profiles]))
        profile_coverage = float(np.mean([item["coverage"] for item in profiles]))
    else:
        profile_score = chamfer_score
        profile_mean = chamfer_mean
        profile_coverage = 0.0

    profile_weight = float(np.clip(getattr(args, "truncated_visible_profile_weight", 0.35), 0.0, 1.0))
    contour_score = (1.0 - profile_weight) * chamfer_score + profile_weight * profile_score
    return {
        "visible_contour_score": float(np.clip(contour_score, 0.0, 1.0)),
        "visible_contour_chamfer_score": float(chamfer_score),
        "visible_profile_score": float(profile_score),
        "visible_contour_mean_distance_px": float(chamfer_mean),
        "visible_profile_mean_distance_px": float(profile_mean),
        "visible_profile_coverage": float(profile_coverage),
        "effective_visible_contour_weight": float(getattr(args, "truncated_visible_contour_weight", 0.0)),
    }


def compute_truncated_bbox_constraint(
    projected_bbox: list[float],
    target_bbox: list[float],
    image_size: tuple[int, int],
    truncation_info: dict[str, Any],
    args: argparse.Namespace,
    visible_projected_bbox: list[float] | None = None,
    visible_target_bbox: list[float] | None = None,
) -> dict[str, Any]:
    if not bool(args.truncated_bbox_constraint_enabled) or not truncation_info.get("is_truncated"):
        return {
            "truncated_bbox_score": None,
            "truncated_bbox_loss": None,
            "truncated_bbox_penalty": 0.0,
            "truncated_bbox_components": {},
        }

    width, height = [float(v) for v in image_size]
    visible_bbox_available = visible_projected_bbox is not None and visible_target_bbox is not None
    if visible_bbox_available:
        px1, py1, px2, py2 = [float(v) for v in visible_projected_bbox]
        tx1, ty1, tx2, ty2 = [float(v) for v in visible_target_bbox]
    else:
        px1, py1, px2, py2 = [float(v) for v in projected_bbox]
        tx1, ty1, tx2, ty2 = [float(v) for v in target_bbox]
    projected_cx = 0.5 * (px1 + px2)
    target_cx = 0.5 * (tx1 + tx2)

    side_sigma = max(1e-6, float(args.truncated_bbox_side_sigma_px))
    top_sigma = max(1e-6, float(args.truncated_bbox_top_sigma_px))
    center_sigma = max(1e-6, float(args.truncated_bbox_center_x_sigma_px))
    overflow_sigma = max(1e-6, float(args.truncated_bbox_bottom_overflow_sigma_px))

    components = {
        "left": ((px1 - tx1) / side_sigma) ** 2,
        "right": ((px2 - tx2) / side_sigma) ** 2,
        "top": ((py1 - ty1) / top_sigma) ** 2,
        "center_x": ((projected_cx - target_cx) / center_sigma) ** 2,
        "bottom_overflow": (max(0.0, float(projected_bbox[3]) - height) / overflow_sigma) ** 2,
    }

    # For bottom-truncated objects, the bottom edge itself is unreliable, but
    # a projection that extends far outside the image usually indicates scale
    # or depth drift. Keep this as an overflow penalty only.
    if "bottom" not in set(truncation_info.get("truncation_sides", [])):
        components["bottom"] = ((py2 - ty2) / top_sigma) ** 2

    loss = float(sum(components.values()))
    score = float(math.exp(-min(60.0, loss)))
    raw_penalty = float(args.truncated_bbox_weight) * loss
    penalty = min(max(0.0, raw_penalty), max(0.0, float(args.truncated_bbox_penalty_cap)))
    return {
        "truncated_bbox_score": score,
        "truncated_bbox_loss": loss,
        "truncated_bbox_penalty": penalty,
        "truncated_bbox_components": {key: float(value) for key, value in components.items()},
        "truncated_bbox_source": "visible_mask_bbox" if visible_bbox_available else "projected_bbox",
    }


def prepare_image_edge_map(
    image: np.ndarray,
    target_mask: np.ndarray,
    bbox: list[float],
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not bool(args.edge_score_enabled):
        return None
    try:
        width, height = [int(v) for v in image_size]
        margin = int(args.edge_roi_margin)
        x1 = max(0, int(math.floor(float(bbox[0]) - margin)))
        y1 = max(0, int(math.floor(float(bbox[1]) - margin)))
        x2 = min(width, int(math.ceil(float(bbox[2]) + margin)))
        y2 = min(height, int(math.ceil(float(bbox[3]) + margin)))
        if x2 <= x1 or y2 <= y1:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, int(args.edge_canny_low), int(args.edge_canny_high))
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 1
        if bool(args.edge_use_mask_erode):
            object_roi = cv2.dilate((target_mask > 0).astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1)
            roi_mask = np.logical_and(roi_mask > 0, object_roi > 0).astype(np.uint8)
        edges = (edges > 0).astype(np.uint8) * roi_mask
        inverse_edges = np.where(edges > 0, 0, 1).astype(np.uint8)
        distance = cv2.distanceTransform(inverse_edges, cv2.DIST_L2, 3)
        return {
            "edges": edges,
            "distance": distance,
            "roi": [int(x1), int(y1), int(x2), int(y2)],
        }
    except Exception as exc:
        print(f"[warn] edge map preparation failed; edge assist disabled: {exc}")
        return None


def compute_edge_score(
    rendered_mask: np.ndarray,
    edge_context: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if edge_context is None:
        return {
            "edge_score": 0.0,
            "edge_mean_distance_px": None,
            "edge_rendered_points": 0,
            "edge_roi": None,
        }
    try:
        rendered = (rendered_mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        rendered_edge = cv2.dilate(rendered, kernel, iterations=1) - cv2.erode(rendered, kernel, iterations=1)
        x1, y1, x2, y2 = edge_context["roi"]
        roi_edge = rendered_edge[y1:y2, x1:x2] > 0
        if not np.any(roi_edge):
            return {
                "edge_score": 0.0,
                "edge_mean_distance_px": None,
                "edge_rendered_points": 0,
                "edge_roi": edge_context["roi"],
            }
        distances = edge_context["distance"][y1:y2, x1:x2][roi_edge].astype(np.float32)
        sigma = max(1e-6, float(args.edge_distance_sigma_px))
        edge_score = float(np.exp(-np.clip(distances, 0.0, 50.0) / sigma).mean())
        return {
            "edge_score": edge_score,
            "edge_mean_distance_px": float(distances.mean()),
            "edge_rendered_points": int(distances.size),
            "edge_roi": edge_context["roi"],
        }
    except Exception as exc:
        print(f"[warn] edge score failed; using edge_score=0: {exc}")
        return {
            "edge_score": 0.0,
            "edge_mean_distance_px": None,
            "edge_rendered_points": 0,
            "edge_roi": edge_context.get("roi") if edge_context else None,
        }


class TemporalPoseEvaluator(fast.CameraPoseEvaluator):
    """CameraPoseEvaluator with temporal, partial-visibility, and edge terms."""

    def __init__(
        self,
        *args: Any,
        temporal_args: argparse.Namespace,
        temporal_prior: dict[str, Any] | None,
        truncation_info: dict[str, Any],
        edge_context: dict[str, Any] | None,
        enable_edge_score: bool,
        t_world_from_cam: np.ndarray | None = None,
        mesh_meta: dict[str, Any] | None = None,
        vehicle_pose_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_args = temporal_args
        self.temporal_prior = temporal_prior
        self.truncation_info = truncation_info
        self.edge_context = edge_context
        self.enable_edge_score = bool(enable_edge_score and temporal_args.edge_score_enabled and edge_context)
        self.t_world_from_cam = None if t_world_from_cam is None else np.asarray(t_world_from_cam, dtype=np.float64)
        self.mesh_meta = mesh_meta or {}
        self.vehicle_pose_context = vehicle_pose_context or {}
        self.current_initializer_metadata: dict[str, Any] = {}

    def set_initializer_metadata(self, metadata: dict[str, Any] | None) -> None:
        self.current_initializer_metadata = dict(metadata or {})

    def _needs_rendered_mask_for_scoring(self) -> bool:
        partial_needed = (
            bool(self.temporal_args.partial_visibility_enabled)
            and bool(self.truncation_info.get("is_truncated"))
        )
        return partial_needed or self.enable_edge_score

    def _augment_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("_temporal_augmented"):
            return result

        base_geometry_score = float(result.get("score", -1.0))
        geometry_score = base_geometry_score
        final_score = base_geometry_score

        result["base_geometry_score"] = base_geometry_score
        result["geometry_score"] = geometry_score
        result["original_mask_iou"] = float(result.get("mask_iou", 0.0))
        result["partial_mask_score"] = None
        result["adjusted_mask_score"] = None
        result["partial_score_boost"] = None
        result["visible_mask_iou"] = None
        result["visible_soft_mask_iou"] = None
        result["visible_target_fraction"] = None
        result["visible_target_area_px"] = None
        result["visible_bbox_iou"] = None
        result["visible_bbox_center_error_px"] = None
        result["visible_contour_score"] = None
        result["visible_contour_chamfer_score"] = None
        result["visible_profile_score"] = None
        result["visible_contour_mean_distance_px"] = None
        result["visible_profile_mean_distance_px"] = None
        result["visible_profile_coverage"] = None
        result["effective_visible_contour_weight"] = 0.0
        result["truncated_visual_quality_gate"] = 1.0
        result["truncated_visual_quality_reason"] = "not_evaluated"
        result["truncated_visual_bbox_factor"] = 1.0
        result["truncated_visual_center_factor"] = 1.0
        result["truncated_visual_overflow_factor"] = 1.0
        result["truncated_visual_overflow_loss"] = 0.0
        result["truncated_visual_quality_penalty"] = 0.0
        result["temporal_score"] = None
        result["temporal_loss"] = None
        result["track_scale_prior_score"] = None
        result["track_scale_prior_loss"] = None
        result["track_scale_prior_value"] = None
        result["track_scale_prior_delta_log"] = None
        result["edge_score"] = 0.0
        result["edge_mean_distance_px"] = None
        result["edge_rendered_points"] = 0
        result["edge_roi"] = self.edge_context.get("roi") if self.edge_context else None
        result["truncated_bbox_score"] = None
        result["truncated_bbox_loss"] = None
        result["truncated_bbox_penalty"] = 0.0
        result["truncated_bbox_components"] = {}
        result["ground_contact_score"] = 0.0
        result["ground_contact_mean_abs_m"] = None
        result["ground_contact_max_abs_m"] = None
        result["ground_gate_passed"] = True
        result["ground_gate_rejected"] = False
        result["ground_contact_penalty"] = 0.0
        result["bbox_bottom_score"] = 0.0
        result["bbox_bottom_distance_m"] = None
        result["upright_score"] = 0.0
        result["upright_angle_error_deg"] = None
        result["upright_gate_passed"] = True
        result["upright_gate_rejected"] = False
        result["upright_gate_penalty"] = 0.0
        result["heading_prior_score"] = 0.0
        result["heading_prior_angle_error_deg"] = None
        result["heading_front_sign_enabled"] = False
        result["heading_front_sign_confidence"] = 0.0
        result["heading_candidate_forward_sign"] = None
        result["heading_semantic_front_sign"] = None
        result["heading_tail_light_front_sign"] = None
        result["heading_tail_light_flipped"] = False
        result["heading_front_sign_hard_rejected"] = False
        result["heading_depth_trend_score"] = None
        result["heading_depth_trend_direction"] = None
        result["heading_depth_trend_confidence"] = None
        result["heading_front_depth_cam"] = None
        result["heading_front_sign_penalty"] = 0.0
        result["effective_front_sign_penalty_weight"] = 0.0
        result["visual_gate_factor"] = 0.0
        result["visual_gate_reason"] = "not_evaluated"
        result["visual_gate_mask_iou_min"] = None
        result["visual_gate_bbox_iou_min"] = None
        result["visual_gate_center_error_px_max"] = None
        result["visible_mask_bonus"] = 0.0
        result["visible_mask_bonus_weight"] = 0.0
        result["effective_temporal_weight"] = float(self.temporal_args.temporal_weight)
        result["effective_track_scale_prior_weight"] = 0.0
        result["effective_edge_weight"] = float(self.temporal_args.edge_weight)
        result["effective_ground_contact_weight"] = 0.0
        result["effective_bbox_bottom_weight"] = 0.0
        result["effective_upright_weight"] = 0.0
        result["effective_heading_prior_weight"] = 0.0
        result["is_truncated"] = bool(self.truncation_info.get("is_truncated", False))
        result["truncation_severity"] = self.truncation_info.get("truncation_severity", "none")
        result["low_observability"] = bool(self.truncation_info.get("low_observability", False))
        result["truncation_observability_score"] = self.truncation_info.get("truncation_observability_score", 1.0)
        result["truncation_observability_reasons"] = self.truncation_info.get("truncation_observability_reasons", [])
        result["prior_frame_id"] = self.temporal_prior.get("frame_idx") if self.temporal_prior else None
        if self.current_initializer_metadata:
            result["initializer_metadata"] = dict(self.current_initializer_metadata)

        if result.get("projected_bbox") is None:
            result["final_score"] = final_score
            result["_temporal_augmented"] = True
            return result

        rendered_mask = result.get("rendered_mask")
        if (
            bool(self.temporal_args.partial_visibility_enabled)
            and bool(self.truncation_info.get("is_truncated"))
            and rendered_mask is not None
        ):
            partial = compute_partial_mask_score(
                rendered_mask=rendered_mask,
                target_mask=self.full_mask,
                soft_target_mask=self.soft_full_mask,
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                args=self.temporal_args,
            )
            result.update(partial)
            visible_bbox = compute_visible_bbox_score(
                projected_bbox=result["projected_bbox"],
                target_bbox=self.json_bbox,
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                args=self.temporal_args,
                rendered_mask=rendered_mask,
                target_mask=self.full_mask,
            )
            result.update(visible_bbox)
            contour = compute_visible_contour_score(
                rendered_mask=rendered_mask,
                target_mask=self.full_mask,
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                args=self.temporal_args,
            )
            result.update(contour)
            visual_quality = compute_truncated_visual_quality_gate(
                result=result,
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                args=self.temporal_args,
            )
            result.update(visual_quality)
            observability = classify_truncation_observability(self.truncation_info, result, self.temporal_args)
            result.update(
                {
                    "truncation_severity": observability["severity"],
                    "low_observability": observability["low_observability"],
                    "truncation_observability_score": observability["score"],
                    "truncation_observability_reasons": observability["reasons"],
                }
            )
            visual_quality_gate = float(visual_quality["truncated_visual_quality_gate"])
            adjusted_mask_score = float(partial["adjusted_mask_score"])
            visible_soft_score = float(partial.get("visible_soft_mask_iou", adjusted_mask_score) or adjusted_mask_score)
            hard_weight = float(
                getattr(
                    self.temporal_args,
                    "truncated_hard_mask_weight",
                    getattr(self.temporal_args, "hard_mask_weight", 0.0),
                )
            )
            bbox_weight = float(
                getattr(self.temporal_args, "truncated_visual_bbox_weight", self.bbox_weight)
            )
            visible_bbox_iou = float(visible_bbox.get("visible_bbox_iou", result.get("bbox_iou", 0.0)) or 0.0)
            visual_score = (1.0 - hard_weight) * visible_soft_score + hard_weight * adjusted_mask_score
            geometry_score = visual_score - bbox_weight * (1.0 - visible_bbox_iou)
            contour_weight = float(getattr(self.temporal_args, "truncated_visible_contour_weight", 0.0))
            geometry_score += (
                visual_quality_gate
                * contour_weight
                * float(result.get("visible_contour_score") or 0.0)
            )
            geometry_score -= float(visual_quality.get("truncated_visual_quality_penalty") or 0.0)
            final_score = geometry_score
            result["geometry_score"] = geometry_score
            result["effective_visual_bbox_weight"] = bbox_weight
            result["effective_hard_mask_weight"] = hard_weight

        if bool(self.truncation_info.get("is_truncated")):
            truncated_bbox = compute_truncated_bbox_constraint(
                projected_bbox=result["projected_bbox"],
                target_bbox=self.json_bbox,
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                args=self.temporal_args,
                visible_projected_bbox=result.get("visible_projected_bbox"),
                visible_target_bbox=result.get("visible_target_bbox"),
            )
            result.update(truncated_bbox)
            final_score -= float(truncated_bbox.get("truncated_bbox_penalty") or 0.0)

        if bool(self.temporal_args.temporal_enabled) and self.temporal_prior is not None:
            temporal = compute_temporal_score(
                translation_cam=np.asarray(result["translation_cam"], dtype=np.float64),
                rotation_cam=np.asarray(result["rotation_cam"], dtype=np.float64),
                scale=np.asarray(result["scale"], dtype=np.float64),
                prior=self.temporal_prior,
                args=self.temporal_args,
            )
            result.update(temporal)
            temporal_weight = float(self.temporal_args.temporal_weight)
            if bool(self.truncation_info.get("is_truncated")):
                temporal_weight = float(self.temporal_args.truncated_temporal_weight)
            result["effective_temporal_weight"] = temporal_weight
            final_score += temporal_weight * float(temporal["temporal_score"])

        track_scale_prior = self.vehicle_pose_context.get("track_scale_prior")
        track_scale = compute_track_scale_prior_score(
            scale=np.asarray(result["scale"], dtype=np.float64),
            track_prior=track_scale_prior,
            args=self.temporal_args,
        )
        result.update(track_scale)
        if track_scale.get("track_scale_prior_score") is not None:
            final_score += float(track_scale["effective_track_scale_prior_weight"]) * float(track_scale["track_scale_prior_score"])

        if self.enable_edge_score and rendered_mask is not None:
            edge = compute_edge_score(rendered_mask, self.edge_context, self.temporal_args)
            result.update(edge)
            edge_weight = float(self.temporal_args.edge_weight)
            if bool(self.truncation_info.get("is_truncated")):
                edge_weight = max(edge_weight, float(self.temporal_args.truncated_edge_weight))
            result["effective_edge_weight"] = edge_weight
            final_score += edge_weight * float(edge["edge_score"])

        if self.t_world_from_cam is not None and self.mesh_meta and self.vehicle_pose_context:
            pose_context = compute_road_and_heading_score(
                translation_cam=np.asarray(result["translation_cam"], dtype=np.float64),
                rotation_cam=np.asarray(result["rotation_cam"], dtype=np.float64),
                scale=np.asarray(result["scale"], dtype=np.float64),
                t_world_from_cam=self.t_world_from_cam,
                mesh_meta=self.mesh_meta,
                vehicle_pose_context=self.vehicle_pose_context,
                projected_bbox=result.get("projected_bbox"),
                image_size=self.image_size,
                truncation_info=self.truncation_info,
                initializer_metadata=self.current_initializer_metadata,
                args=self.temporal_args,
            )
            result.update(pose_context)
            if (
                bool(pose_context.get("ground_gate_rejected"))
                or bool(pose_context.get("upright_gate_rejected"))
                or bool(pose_context.get("heading_front_sign_hard_rejected"))
            ):
                final_score = -1_000_000.0
                result["score"] = float(final_score)
                result["final_score"] = float(final_score)
                result["_temporal_augmented"] = True
                return result
            final_score -= float(pose_context.get("upright_gate_penalty") or 0.0)
            visual_gate = compute_visual_gate_for_pose_priors(
                result=result,
                truncation_info=self.truncation_info,
                args=self.temporal_args,
            )
            result.update(visual_gate)
            gate = float(visual_gate["visual_gate_factor"])
            for key in (
                "effective_ground_contact_weight",
                "effective_bbox_bottom_weight",
                "effective_upright_weight",
                "effective_heading_prior_weight",
            ):
                result[key] = float(result.get(key) or 0.0) * gate
            if bool(self.truncation_info.get("is_truncated")):
                result["heading_front_sign_penalty"] = 0.0
                result["heading_front_angle_penalty"] = 0.0
            front_penalty_gate = 0.0 if bool(self.truncation_info.get("is_truncated")) else 1.0
            result["effective_front_sign_penalty_weight"] = front_penalty_gate
            final_score -= float(result.get("heading_front_sign_penalty") or 0.0) * front_penalty_gate
            final_score -= float(result.get("heading_front_angle_penalty") or 0.0) * front_penalty_gate
            final_score -= float(result.get("ground_contact_penalty") or 0.0)
            final_score += float(result.get("effective_ground_contact_weight") or 0.0) * float(
                pose_context.get("ground_contact_score") or 0.0
            )
            final_score += float(result.get("effective_bbox_bottom_weight") or 0.0) * float(
                pose_context.get("bbox_bottom_score") or 0.0
            )
            final_score += float(result.get("effective_upright_weight") or 0.0) * float(
                pose_context.get("upright_score") or 0.0
            )
            final_score += float(result.get("effective_heading_prior_weight") or 0.0) * float(
                pose_context.get("heading_prior_score") or 0.0
            )

        mask_bonus = compute_visible_mask_bonus(
            result=result,
            truncation_info=self.truncation_info,
            args=self.temporal_args,
        )
        result.update(mask_bonus)
        final_score += float(mask_bonus["visible_mask_bonus_weight"]) * float(mask_bonus["visible_mask_bonus"])

        result["score"] = float(final_score)
        result["final_score"] = float(final_score)
        result["_temporal_augmented"] = True
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


def temporal_optimization_history_row(
    phase: str,
    iteration: int,
    parameter: str,
    direction: int,
    result: dict[str, Any],
    step_value: float,
) -> dict[str, Any]:
    row = fast.optimization_history_row(phase, iteration, parameter, direction, result, step_value)
    for key in TEMPORAL_EXTRA_HISTORY_KEYS:
        row[key] = result.get(key)
    return row


def local_search_stage(
    evaluator: TemporalPoseEvaluator,
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
    history = [temporal_optimization_history_row(stage_name, 0, "initial", -1, best, 0.0)]
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

            directions = (1.0, -1.0)
            candidates = [
                fast.clamp_delta_params(
                    params + direction * step_value * group["vector"],
                    max_translation_delta,
                    max_rotation_delta_deg,
                    scale_min_factor,
                    scale_max_factor,
                )
                for direction in directions
            ]
            results = evaluator.evaluate_delta_batch(
                base_translation_cam,
                base_rotation_cam,
                base_scale,
                np.stack(candidates, axis=0),
            )

            for direction, candidate, result in zip(directions, candidates, results):
                if save_full_history:
                    history.append(
                        temporal_optimization_history_row(
                            stage_name,
                            iteration,
                            group["name"],
                            int(direction),
                            result,
                            step_value,
                        )
                    )
                if result["score"] > current_best["score"] + 1e-8:
                    current_best = result
                    current_params = candidate
                    current_direction = int(direction)

            if current_best["score"] > best["score"] + 1e-8:
                params = current_params
                best = current_best
                improved = True
                if not save_full_history:
                    history.append(
                        temporal_optimization_history_row(
                            stage_name,
                            iteration,
                            group["name"],
                            current_direction,
                            best,
                            step_value,
                        )
                    )
                print(
                    f"  [{stage_name} iter {iteration:02d}] improve {group['name']} "
                    f"score={best['score']:.6f} mask_iou={best['mask_iou']:.6f} bbox_iou={best['bbox_iou']:.6f}"
                )

        if not improved:
            group_steps *= step_decay
            print(f"  [{stage_name} iter {iteration:02d}] no improvement, shrink steps")
            if not save_full_history:
                history.append(
                    temporal_optimization_history_row(
                        stage_name,
                        iteration,
                        "step_shrink",
                        0,
                        best,
                        float(group_steps.max()),
                    )
                )
        if np.all(group_steps <= group_min_steps):
            break

    final = evaluator.evaluate_delta(base_translation_cam, base_rotation_cam, base_scale, params, keep_mask=True)
    final["params"] = params.copy()
    if not save_full_history:
        history.append(temporal_optimization_history_row(stage_name, max_iters + 1, "final", 0, final, 0.0))
    return final, history


def refine_candidate_stages(
    coarse_result: dict[str, Any],
    proxy_evaluator: TemporalPoseEvaluator,
    full_evaluator: TemporalPoseEvaluator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    translation_cam = np.asarray(coarse_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(coarse_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(coarse_result["scale"], dtype=np.float64)
    initializer_metadata = dict(coarse_result.get("initializer_metadata") or {})

    history: list[dict[str, Any]] = []
    stage1_result, stage1_history = local_search_stage(
        evaluator=proxy_evaluator,
        base_translation_cam=translation_cam,
        base_rotation_cam=rotation_cam,
        base_scale=scale,
        stage_name="coarse",
        max_iters=args.stage1_iters,
        step_decay=args.step_decay,
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor,
        save_full_history=args.save_full_history,
        initializer_metadata=initializer_metadata,
    )
    history.extend(stage1_history)
    translation_cam = np.asarray(stage1_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(stage1_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(stage1_result["scale"], dtype=np.float64)

    stage2_result, stage2_history = local_search_stage(
        evaluator=proxy_evaluator,
        base_translation_cam=translation_cam,
        base_rotation_cam=rotation_cam,
        base_scale=scale,
        stage_name="rotation",
        max_iters=args.stage2_iters,
        step_decay=args.step_decay,
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor,
        save_full_history=args.save_full_history,
        initializer_metadata=initializer_metadata,
    )
    history.extend(stage2_history)
    translation_cam = np.asarray(stage2_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(stage2_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(stage2_result["scale"], dtype=np.float64)

    stage3_result, stage3_history = local_search_stage(
        evaluator=full_evaluator,
        base_translation_cam=translation_cam,
        base_rotation_cam=rotation_cam,
        base_scale=scale,
        stage_name="fine",
        max_iters=args.stage3_iters,
        step_decay=args.step_decay,
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor,
        save_full_history=args.save_full_history,
        initializer_metadata=initializer_metadata,
    )
    history.extend(stage3_history)
    return stage3_result, history


def make_temporal_seed(
    prior: dict[str, Any] | None,
    evaluator: TemporalPoseEvaluator,
) -> dict[str, Any] | None:
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


def road_snap_candidate(
    candidate: dict[str, Any],
    evaluator: TemporalPoseEvaluator,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Project a candidate onto the road plane by moving it along road normal."""

    if not bool(getattr(args, "road_snap_candidate_enabled", True)):
        return None
    if not bool(evaluator.truncation_info.get("is_truncated")):
        return None
    if evaluator.t_world_from_cam is None or not evaluator.mesh_meta or not evaluator.vehicle_pose_context:
        return None
    road = evaluator.vehicle_pose_context.get("road_constraint", {})
    if not isinstance(road, dict) or not bool(road.get("available")):
        return None
    try:
        plane = road.get("road_plane", {})
        normal_world, offset = fast.oriented_plane(
            np.asarray(plane["normal_world"], dtype=np.float64),
            float(plane["offset"]),
            str(getattr(args, "world_up_axis", "y")),
        )
        translation_cam = np.asarray(candidate["translation_cam"], dtype=np.float64)
        rotation_cam = np.asarray(candidate["rotation_cam"], dtype=np.float64)
        scale = np.asarray(candidate["scale"], dtype=np.float64)
        t_world_from_cam = np.asarray(evaluator.t_world_from_cam, dtype=np.float64)
        r_world_from_cam = t_world_from_cam[:3, :3]
        rotation_world = r_world_from_cam @ rotation_cam
        translation_world = r_world_from_cam @ translation_cam + t_world_from_cam[:3, 3]

        up_axis, up_sign = fast.mesh_up_axis_and_sign(
            evaluator.mesh_meta,
            bool(getattr(args, "vehicle_mesh_axis_override_enabled", True)),
            int(getattr(args, "vehicle_mesh_up_axis_idx", 1)),
            float(getattr(args, "vehicle_mesh_up_sign", -1.0)),
        )
        bottom_local = fast.bottom_contact_points_local(
            np.asarray(evaluator.mesh_meta["bounds"], dtype=np.float64),
            up_axis,
            up_sign,
        )
        bottom_world = (rotation_world @ (bottom_local * scale.reshape(1, 3)).T).T + translation_world.reshape(1, 3)
        signed = bottom_world @ normal_world + offset
        if not np.all(np.isfinite(signed)):
            return None
        # Align the average bottom contact point to the plane.  A capped snap
        # avoids turning bad faraway hypotheses into plausible-looking ones.
        snap_distance = float(np.mean(signed))
        max_snap = float(getattr(args, "road_snap_candidate_max_distance_m", 0.30))
        if abs(snap_distance) <= 1e-6 or abs(snap_distance) > max_snap:
            return None
        snapped_world = translation_world - snap_distance * normal_world
        snapped_cam = r_world_from_cam.T @ (snapped_world - t_world_from_cam[:3, 3])
        if not np.all(np.isfinite(snapped_cam)) or float(snapped_cam[2]) <= 0.05:
            return None
        meta = dict(candidate.get("initializer_metadata") or {})
        meta["source"] = str(meta.get("source", "candidate")) + "_road_snap"
        meta["road_snap_from_score"] = float(candidate.get("score", 0.0))
        meta["road_snap_distance_m"] = snap_distance
        result = evaluator.evaluate_absolute(snapped_cam, rotation_cam, scale)
        if result.get("projected_bbox") is None:
            return None
        result["initializer_metadata"] = meta
        return result
    except Exception:
        return None


def augment_with_road_snap_candidates(
    candidates: list[dict[str, Any]],
    evaluator: TemporalPoseEvaluator,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not bool(getattr(args, "road_snap_candidate_enabled", True)):
        return candidates
    if not candidates:
        return candidates
    limit = max(1, int(getattr(args, "road_snap_candidate_source_top_k", 16)))
    augmented = list(candidates)
    seen = {fast.pose_signature(item) for item in augmented}
    for candidate in candidates[:limit]:
        snapped = road_snap_candidate(candidate, evaluator, args)
        if snapped is None:
            continue
        signature = fast.pose_signature(snapped)
        if signature in seen:
            continue
        seen.add(signature)
        augmented.append(snapped)
    return sorted(augmented, key=lambda item: float(item.get("score", -1e9)), reverse=True)


def candidate_forward_sign(result: dict[str, Any], default_sign: float) -> float:
    meta = result.get("initializer_metadata")
    if isinstance(meta, dict) and meta.get("forward_sign") is not None:
        try:
            return -1.0 if float(meta.get("forward_sign")) < 0.0 else 1.0
        except Exception:
            return default_sign
    return default_sign


def merge_temporal_seed(
    candidates: list[dict[str, Any]],
    temporal_seed: dict[str, Any] | None,
    top_k: int,
    refine_top_k: int,
) -> list[dict[str, Any]]:
    if temporal_seed is None:
        return candidates

    combined = [temporal_seed, *candidates]
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in sorted(combined, key=lambda item: float(item["score"]), reverse=True):
        signature = fast.pose_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)

    limit = max(int(top_k), int(refine_top_k), 1)
    selected = unique[:limit]
    if not any(item.get("initializer_metadata", {}).get("source") == "temporal_prior" for item in selected):
        if len(selected) >= limit:
            selected[-1] = temporal_seed
        else:
            selected.append(temporal_seed)
        selected = sorted(selected, key=lambda item: float(item["score"]), reverse=True)
    return selected


def _force_source_candidate(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return selected
    if any(item.get("initializer_metadata", {}).get("source") == source for item in selected):
        return selected[:limit]
    source_item = next(
        (item for item in candidates if item.get("initializer_metadata", {}).get("source") == source),
        None,
    )
    if source_item is None:
        return selected[:limit]
    if len(selected) >= limit:
        selected[-1] = source_item
    else:
        selected.append(source_item)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for item in selected:
        signature = fast.pose_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)
    return unique[:limit]


def select_pareto_refine_candidates(
    candidates: list[dict[str, Any]],
    refine_top_k: int,
    args: argparse.Namespace,
    truncation_info: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if not bool(getattr(args, "pareto_refine_selection_enabled", True)) or not bool(truncation_info.get("is_truncated")):
        limit = max(1, int(refine_top_k))
        return _force_source_candidate(
            candidates,
            candidates[:limit],
            "task_json_corrected_pose",
            limit,
        )

    source_candidates = list(candidates)
    if bool(getattr(args, "truncated_candidate_ground_gate_enabled", True)):
        mean_limit = float(getattr(args, "truncated_candidate_ground_gate_mean_m", 0.085))
        max_limit = float(getattr(args, "truncated_candidate_ground_gate_max_m", 0.18))
        gated = [
            item
            for item in source_candidates
            if item.get("ground_contact_mean_abs_m") is not None
            and item.get("ground_contact_max_abs_m") is not None
            and float(item.get("ground_contact_mean_abs_m")) <= mean_limit
            and float(item.get("ground_contact_max_abs_m")) <= max_limit
        ]
        fallback = max(0, int(getattr(args, "truncated_candidate_ground_gate_fallback_top_k", 2)))
        if gated:
            fallback_items = sorted(
                source_candidates,
                key=lambda item: float(item.get("visible_mask_iou", item.get("mask_iou", 0.0)) or 0.0),
                reverse=True,
            )[:fallback]
            source_candidates = [*gated, *fallback_items]
        elif fallback > 0:
            source_candidates = sorted(
                source_candidates,
                key=lambda item: float(item.get("visible_mask_iou", item.get("mask_iou", 0.0)) or 0.0),
                reverse=True,
            )[:fallback]
        print(
            f"[ground-gate] candidate ground gate kept {len(gated)}/{len(candidates)} "
            f"primary, source_pool={len(source_candidates)} mean<={mean_limit:.3f} max<={max_limit:.3f}"
        )

    limit = max(1, int(refine_top_k))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()

    def add(items: list[dict[str, Any]]) -> None:
        for item in items:
            if len(selected) >= limit:
                return
            signature = fast.pose_signature(item)
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(item)

    add(sorted(source_candidates, key=lambda item: float(item.get("score", -1e9)), reverse=True)[: max(1, limit // 2)])

    quota = max(1, int(getattr(args, "pareto_refine_branch_quota", 2)))
    branches = [
        lambda item: float(item.get("visible_mask_iou", item.get("mask_iou", 0.0)) or 0.0),
        lambda item: float(item.get("visible_contour_score", 0.0) or 0.0),
        lambda item: float(item.get("ground_contact_score", 0.0) or 0.0),
        lambda item: -float(item.get("ground_contact_max_abs_m", 1e9) or 1e9),
        lambda item: float(item.get("visible_bbox_iou", item.get("bbox_iou", 0.0)) or 0.0),
    ]
    for key_fn in branches:
        if len(selected) >= limit:
            break
        add(sorted(source_candidates, key=key_fn, reverse=True)[:quota])

    add(sorted(source_candidates, key=lambda item: float(item.get("score", -1e9)), reverse=True))
    selected = _force_source_candidate(candidates, selected, "task_json_corrected_pose", limit)
    return selected[:limit]


def candidate_satisfies_ground_constraint(result: dict[str, Any], args: argparse.Namespace) -> bool:
    mean_raw = result.get("ground_contact_mean_abs_m")
    max_raw = result.get("ground_contact_max_abs_m")
    if mean_raw is None or max_raw is None:
        return False
    mean_limit = float(getattr(args, "final_ground_select_mean_max_m", 0.055))
    max_limit = float(getattr(args, "final_ground_select_max_max_m", 0.12))
    return float(mean_raw) <= mean_limit and float(max_raw) <= max_limit


def candidate_satisfies_severe_truncation_gate(result: dict[str, Any], args: argparse.Namespace) -> bool:
    reasons: list[str] = []

    def check_min(key: str, limit: float, reason: str) -> None:
        value = _finite_float(result.get(key))
        if value is None or value < limit:
            reasons.append(reason)

    def check_max(key: str, limit: float, reason: str) -> None:
        value = _finite_float(result.get(key))
        if value is None or value > limit:
            reasons.append(reason)

    check_max(
        "ground_contact_mean_abs_m",
        float(getattr(args, "severe_truncation_ground_mean_m_max", 0.085)),
        "ground_mean",
    )
    check_max(
        "ground_contact_max_abs_m",
        float(getattr(args, "severe_truncation_ground_max_m_max", 0.18)),
        "ground_max",
    )
    check_max(
        "upright_angle_error_deg",
        float(getattr(args, "severe_truncation_upright_deg_max", 35.0)),
        "upright",
    )
    check_min(
        "visible_mask_iou",
        float(getattr(args, "severe_truncation_visible_mask_iou_min", 0.68)),
        "visible_mask",
    )
    visible_fraction = _finite_float(result.get("visible_target_fraction"))
    visible_fraction_min = float(getattr(args, "severe_truncation_visible_target_fraction_min", 0.12))
    if visible_fraction is not None and visible_fraction < visible_fraction_min:
        reasons.append("visible_fraction")
    check_max(
        "visible_contour_mean_distance_px",
        float(getattr(args, "severe_truncation_visible_contour_mean_px_max", 7.0)),
        "visible_contour",
    )
    profile = _finite_float(result.get("visible_profile_mean_distance_px"))
    if profile is not None and profile > float(getattr(args, "severe_truncation_visible_profile_mean_px_max", 10.0)):
        reasons.append("visible_profile")
    yaw_jump = _finite_float(result.get("yaw_jump_from_anchor_deg"))
    if yaw_jump is not None and yaw_jump > float(getattr(args, "severe_truncation_yaw_jump_deg_max", 25.0)):
        reasons.append("yaw_jump")
    if bool(result.get("heading_front_sign_hard_rejected")):
        reasons.append("front_sign")

    passed = not reasons
    result["severe_truncation_gate_passed"] = passed
    result["severe_truncation_gate_reasons"] = reasons
    return passed


def candidate_satisfies_severe_truncation_fallback(result: dict[str, Any], args: argparse.Namespace) -> bool:
    reasons: list[str] = []

    def check_min(key: str, limit: float, reason: str) -> None:
        value = _finite_float(result.get(key))
        if value is None or value < limit:
            reasons.append(reason)

    def check_max(key: str, limit: float, reason: str) -> None:
        value = _finite_float(result.get(key))
        if value is None or value > limit:
            reasons.append(reason)

    check_min(
        "visible_mask_iou",
        float(getattr(args, "severe_truncation_fallback_visible_mask_iou_min", 0.76)),
        "visible_mask",
    )
    visible_fraction = _finite_float(result.get("visible_target_fraction"))
    visible_fraction_min = float(getattr(args, "severe_truncation_fallback_visible_target_fraction_min", 0.12))
    if visible_fraction is not None and visible_fraction < visible_fraction_min:
        reasons.append("visible_fraction")
    check_max(
        "visible_contour_mean_distance_px",
        float(getattr(args, "severe_truncation_fallback_visible_contour_mean_px_max", 5.5)),
        "visible_contour",
    )
    check_max(
        "visible_profile_mean_distance_px",
        float(getattr(args, "severe_truncation_fallback_visible_profile_mean_px_max", 8.5)),
        "visible_profile",
    )
    if reasons:
        result["severe_truncation_fallback_rejected"] = True
        result["severe_truncation_fallback_reasons"] = reasons
        return False
    result["severe_truncation_fallback_rejected"] = False
    result["severe_truncation_fallback_reasons"] = []
    return True


def front_sign_selection_penalty(result: dict[str, Any]) -> float:
    if not bool(result.get("heading_front_sign_enabled")):
        return 0.0
    try:
        angle = float(result.get("heading_prior_angle_error_deg"))
    except Exception:
        return 0.0
    if not math.isfinite(angle):
        return 0.0
    try:
        confidence = float(result.get("heading_front_sign_confidence") or 0.0)
    except Exception:
        confidence = 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if confidence <= 0.0:
        return 0.0
    if bool(result.get("heading_front_sign_hard_rejected")):
        return 10.0
    wrong_sign = max(0.0, angle - 90.0) / 90.0
    soft_angle = min(angle, 90.0) / 180.0
    return confidence * (0.5 * wrong_sign + 0.05 * soft_angle)


def front_sign_rank_score(result: dict[str, Any]) -> float:
    return float(result.get("score", -1e9)) - front_sign_selection_penalty(result)


def choose_best_refined_result(
    refined_results: list[dict[str, Any]],
    args: argparse.Namespace,
    truncation_info: dict[str, Any],
) -> dict[str, Any] | None:
    if not refined_results:
        return None
    if not (
        bool(getattr(args, "final_ground_constrained_selection_enabled", True))
        and bool(truncation_info.get("is_truncated"))
    ):
        selected = max(refined_results, key=front_sign_rank_score)
        selected["final_front_sign_rank_score"] = front_sign_rank_score(selected)
        selected["final_front_sign_selection_penalty"] = front_sign_selection_penalty(selected)
        selected["final_selection_mode"] = "front_sign_consistent_rank"
        return selected
    feasible = [item for item in refined_results if candidate_satisfies_ground_constraint(item, args)]
    if not feasible:
        selected = max(refined_results, key=lambda item: float(item.get("score", -1e9)))
        selected["final_selection_mode"] = "score_no_ground_feasible"
        selected["final_ground_constrained_selected"] = False
        return selected

    severe = str(truncation_info.get("truncation_severity", "")) == "severe"
    if severe and bool(getattr(args, "severe_truncation_final_gate_enabled", True)):
        gated = [item for item in feasible if candidate_satisfies_severe_truncation_gate(item, args)]
        if gated:
            feasible = gated
        else:
            for item in feasible:
                item.setdefault("severe_truncation_gate_passed", False)
                item.setdefault("severe_truncation_gate_reasons", ["fallback_no_gate_pass"])
            fallback_pool = [
                item for item in feasible if candidate_satisfies_severe_truncation_fallback(item, args)
            ]
            if not fallback_pool:
                return None
            fallback = max(
                fallback_pool,
                key=lambda item: (
                    float(item.get("visible_mask_iou", item.get("mask_iou", 0.0)) or 0.0),
                    -float(item.get("visible_contour_mean_distance_px", 1e9) or 1e9),
                    float(item.get("visible_contour_score", 0.0) or 0.0),
                    -float(item.get("yaw_jump_from_anchor_deg", 180.0) or 180.0),
                ),
            )
            fallback["final_selection_mode"] = "severe_truncation_fallback"
            fallback["final_ground_constrained_selected"] = True
            fallback["low_confidence"] = True
            return fallback

    if not bool(getattr(args, "truncated_final_visual_selection_enabled", True)):
        selected = max(feasible, key=lambda item: float(item.get("score", -1e9)))
        selected["final_selection_mode"] = "ground_feasible_score"
        selected["final_ground_constrained_selected"] = True
        return selected

    min_bbox = float(getattr(args, "truncated_final_visual_min_bbox_iou", 0.0))
    min_quality_gate = float(getattr(args, "truncated_final_visual_min_quality_gate", 0.0))
    visual_pool = [
        item
        for item in feasible
        if float(item.get("visible_bbox_iou", item.get("bbox_iou", 0.0)) or 0.0) >= min_bbox
        and float(item.get("truncated_visual_quality_gate", 1.0) or 0.0) >= min_quality_gate
    ]
    if not visual_pool:
        visual_pool = feasible

    mask_weight = float(getattr(args, "truncated_final_visual_mask_weight", 1.0))
    contour_weight = float(getattr(args, "truncated_final_visual_contour_weight", 0.35))
    bbox_weight = float(getattr(args, "truncated_final_visual_bbox_weight", 0.08))
    if severe:
        bbox_weight = float(getattr(args, "severe_truncated_final_visual_bbox_weight", 0.02))
    quality_weight = float(getattr(args, "truncated_final_visual_quality_weight", 0.05))
    ground_mean_weight = float(getattr(args, "truncated_final_visual_ground_mean_weight", 0.25))
    ground_max_weight = float(getattr(args, "truncated_final_visual_ground_max_weight", 0.10))
    score_weight = float(getattr(args, "truncated_final_visual_score_weight", 0.03))

    for item in feasible:
        visible_mask = float(item.get("visible_mask_iou", item.get("mask_iou", 0.0)) or 0.0)
        contour = float(item.get("visible_contour_score", 0.0) or 0.0)
        bbox = float(item.get("visible_bbox_iou", item.get("bbox_iou", 0.0)) or 0.0)
        quality_gate = float(item.get("truncated_visual_quality_gate", 1.0) or 0.0)
        ground_mean = float(item.get("ground_contact_mean_abs_m", 1.0) or 1.0)
        ground_max = float(item.get("ground_contact_max_abs_m", 1.0) or 1.0)
        score = float(item.get("score", 0.0) or 0.0)
        item["final_ground_constrained_selected"] = True
        item["final_ground_constrained_rank_score"] = (
            mask_weight * visible_mask
            + contour_weight * contour
            + bbox_weight * bbox
            + quality_weight * quality_gate
            + score_weight * score
            - ground_mean_weight * ground_mean
            - ground_max_weight * ground_max
        )
        item["final_selection_mode"] = "ground_feasible_visual_rank"
        item["final_visual_selection_weights"] = {
            "mask": mask_weight,
            "contour": contour_weight,
            "bbox": bbox_weight,
            "quality_gate": quality_weight,
            "ground_mean": ground_mean_weight,
            "ground_max": ground_max_weight,
            "score": score_weight,
            "min_bbox_iou": min_bbox,
            "min_quality_gate": min_quality_gate,
        }
    return max(visual_pool, key=lambda item: float(item.get("final_ground_constrained_rank_score", -1e9)))


def refined_pose_candidate_summary(
    result: dict[str, Any],
    *,
    t_world_from_cam: np.ndarray,
) -> dict[str, Any]:
    """Return a JSON-friendly pose candidate summary for window-level selection."""

    uniform_scale = fast.make_uniform_scale(
        fast.scale_to_uniform_scalar(np.asarray(result["scale"], dtype=np.float64))
    )
    translation_world, rotation_world = fast.camera_pose_to_world_pose(
        np.asarray(t_world_from_cam, dtype=np.float64),
        np.asarray(result["translation_cam"], dtype=np.float64),
        np.asarray(result["rotation_cam"], dtype=np.float64),
    )
    metric_keys = [
        "score",
        "base_geometry_score",
        "geometry_score",
        "final_score",
        "mask_iou",
        "soft_mask_iou",
        "bbox_iou",
        "bbox_center_error_px",
        "projected_bbox",
        "temporal_score",
        "temporal_loss",
        "track_scale_prior_score",
        "track_scale_prior_loss",
        "track_scale_prior_value",
        "edge_score",
        "visible_mask_iou",
        "visible_soft_mask_iou",
        "visible_target_fraction",
        "visible_target_area_px",
        "visible_bbox_iou",
        "visible_bbox_center_error_px",
        "visible_projected_bbox",
        "visible_target_bbox",
        "visible_bbox_source",
        "visible_contour_score",
        "visible_profile_score",
        "visible_contour_mean_distance_px",
        "visible_profile_mean_distance_px",
        "truncation_severity",
        "low_observability",
        "truncation_observability_score",
        "truncation_observability_reasons",
        "truncated_visual_quality_gate",
        "truncated_visual_quality_reason",
        "truncated_visual_bbox_factor",
        "truncated_visual_center_factor",
        "truncated_visual_mask_factor",
        "truncated_visual_contour_factor",
        "truncated_visual_profile_factor",
        "truncated_visual_overflow_factor",
        "truncated_visual_overflow_loss",
        "severe_truncation_gate_passed",
        "severe_truncation_gate_reasons",
        "truncated_bbox_score",
        "truncated_bbox_loss",
        "truncated_bbox_penalty",
        "truncated_bbox_components",
        "truncated_bbox_source",
        "ground_contact_score",
        "ground_contact_mean_abs_m",
        "ground_contact_max_abs_m",
        "ground_gate_passed",
        "ground_gate_rejected",
        "bbox_bottom_distance_m",
        "upright_score",
        "upright_angle_error_deg",
        "upright_gate_passed",
        "upright_gate_rejected",
        "heading_prior_score",
        "heading_prior_angle_error_deg",
        "heading_front_sign_enabled",
        "heading_front_sign_confidence",
        "heading_candidate_forward_sign",
        "heading_semantic_front_sign",
        "heading_tail_light_front_sign",
        "heading_front_sign_hard_rejected",
        "heading_front_sign_penalty",
        "heading_front_angle_penalty",
        "final_selection_mode",
        "final_ground_constrained_selected",
        "final_ground_constrained_rank_score",
    ]
    metrics = {key: result.get(key) for key in metric_keys if key in result}
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
    }


def save_temporal_edge_debug(
    output_dir: Path,
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
    best_result: dict[str, Any],
    temporal_prior: dict[str, Any] | None,
    edge_context: dict[str, Any] | None,
) -> Path | None:
    if edge_context is None or best_result.get("rendered_mask") is None:
        return None
    try:
        rendered_mask = best_result["rendered_mask"].astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        rendered_edge = cv2.dilate(rendered_mask, kernel, iterations=1) - cv2.erode(rendered_mask, kernel, iterations=1)
        image_edges = edge_context["edges"].astype(np.uint8)
        distance = edge_context["distance"]
        distance_vis = cv2.applyColorMap(
            np.asarray(np.clip(distance / max(1e-6, distance.max()), 0.0, 1.0) * 255, dtype=np.uint8),
            cv2.COLORMAP_TURBO,
        )

        overlay = image.copy()
        overlay[image_edges > 0] = (0, 255, 0)
        overlay[rendered_edge > 0] = (0, 0, 255)

        rendered_edge_vis = cv2.cvtColor(rendered_edge * 255, cv2.COLOR_GRAY2BGR)
        image_edge_vis = cv2.cvtColor(image_edges * 255, cv2.COLOR_GRAY2BGR)
        panels: list[tuple[str, np.ndarray]] = [
            ("edge overlay", overlay),
            ("rendered edge", rendered_edge_vis),
            ("image edge", image_edge_vis),
            ("edge distance", distance_vis),
        ]

        if temporal_prior is not None:
            prior_mask, _, _, _ = fast.render_triangle_mask_for_pose(
                vertices=vertices,
                faces=faces,
                translation_cam=np.asarray(temporal_prior["translation_cam"], dtype=np.float64),
                rotation_cam=np.asarray(temporal_prior["rotation_cam"], dtype=np.float64),
                scale=np.asarray(temporal_prior["scale"], dtype=np.float64),
                intrinsics=intrinsics,
                image_size=image_size,
            )
            prior_overlay = image.copy()
            prior_overlay[prior_mask.astype(bool)] = cv2.addWeighted(
                prior_overlay[prior_mask.astype(bool)],
                0.45,
                np.full_like(prior_overlay[prior_mask.astype(bool)], (255, 120, 0)),
                0.55,
                0.0,
            )
            panels.append(("temporal prior", prior_overlay))

        return fast.save_image_collage(
            output_dir / "04_temporal_edge_debug.png",
            panels,
            columns=3,
            content_size=(360, 220),
        )
    except Exception as exc:
        print(f"[warn] temporal edge debug image failed: {exc}")
        return None


def optimize_sample(args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = fast.resolve_sample_dir(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{sample_dir.name}_temporal_fast"
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
    mesh_tail_light_prior = fast.infer_tail_light_axis_prior(mesh, mesh_meta)

    t_world_from_cam = np.asarray(task["camera"]["T_world_from_cam"], dtype=np.float64)
    intrinsics = {
        "fx": float(task["camera"]["fx"]),
        "fy": float(task["camera"]["fy"]),
        "cx": float(task["camera"]["cx"]),
        "cy": float(task["camera"]["cy"]),
    }
    vehicle_pose_context = fast.build_vehicle_pose_context(
        task=task,
        sample_dir=sample_dir,
        full_mask=full_mask,
        json_bbox=json_bbox,
        image_size=image_size,
        intrinsics=intrinsics,
        t_world_from_cam=t_world_from_cam,
        args=args,
    )
    vehicle_pose_context["mesh_tail_light_prior"] = mesh_tail_light_prior
    mesh_meta = fast.apply_mesh_axis_prior(mesh_meta, vehicle_pose_context.get("mesh_axis_prior"))
    mesh_meta["tail_light_prior"] = mesh_tail_light_prior
    proxy_vertices, proxy_faces = fast.build_proxy_mesh(vertices, faces, target_faces=args.proxy_face_count)

    object_id, frame_idx = parse_task_id_from_sample_dir(sample_dir)
    suffixes = parse_suffixes(args.temporal_search_output_suffixes)
    temporal_prior = None
    skip_disk_temporal_prior = should_skip_disk_temporal_prior(vehicle_pose_context)
    if bool(args.temporal_enabled) and frame_idx > 1 and not skip_disk_temporal_prior:
        temporal_prior = find_temporal_prior(
            output_dir=output_dir,
            object_id=object_id,
            frame_idx=frame_idx,
            lookback=args.temporal_lookback,
            suffixes=suffixes,
            t_world_from_cam=t_world_from_cam,
        )

    if temporal_prior is None and skip_disk_temporal_prior:
        print(f"[temporal] disk prior disabled for all-frames candidate pass: {object_id}@{frame_idx:06d}")
    elif temporal_prior is None:
        print(f"[temporal] no prior found for {object_id}@{frame_idx:06d}; falling back to fast-style scoring")
    else:
        print(
            f"[temporal] prior found: frame={temporal_prior['frame_idx']} "
            f"dir={temporal_prior['output_dir']} source={temporal_prior['pose_source']}"
        )

    truncation_info = detect_truncation(
        mask=full_mask,
        bbox=json_bbox,
        image_size=image_size,
        args=args,
        prior_mask_area_px=temporal_prior.get("prior_mask_area_px") if temporal_prior else None,
    )
    print(
        f"[partial] enabled={bool(args.partial_visibility_enabled)} "
        f"is_truncated={truncation_info['is_truncated']} sides={truncation_info['truncation_sides']}"
    )

    edge_context = prepare_image_edge_map(image, full_mask, json_bbox, image_size, args)
    print(f"[edge] enabled={bool(args.edge_score_enabled)} available={edge_context is not None}")

    pose = task.get("corrected_pose", {})
    base_scale = fast.make_uniform_scale(
        fast.scale_to_uniform_scalar(np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
    )

    proxy_evaluator = TemporalPoseEvaluator(
        vertices=proxy_vertices,
        faces=proxy_faces,
        mesh=None,
        full_mask=full_mask,
        soft_full_mask=soft_full_mask,
        json_bbox=json_bbox,
        intrinsics=intrinsics,
        image_size=image_size,
        bbox_weight=args.bbox_weight,
        hard_mask_weight=args.hard_mask_weight,
        backend="triangle_fill",
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
        temporal_args=args,
        temporal_prior=temporal_prior,
        truncation_info=truncation_info,
        edge_context=None,
        enable_edge_score=False,
        t_world_from_cam=t_world_from_cam,
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
    )
    full_evaluator = TemporalPoseEvaluator(
        vertices=vertices,
        faces=faces,
        mesh=mesh,
        full_mask=full_mask,
        soft_full_mask=soft_full_mask,
        json_bbox=json_bbox,
        intrinsics=intrinsics,
        image_size=image_size,
        bbox_weight=args.bbox_weight,
        hard_mask_weight=args.hard_mask_weight,
        backend=args.render_backend,
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
        temporal_args=args,
        temporal_prior=temporal_prior,
        truncation_info=truncation_info,
        edge_context=edge_context,
        enable_edge_score=True,
        t_world_from_cam=t_world_from_cam,
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
    )

    corrected_seed = fast.corrected_pose_seed(task, t_world_from_cam, proxy_evaluator) if args.include_corrected_seed else None
    initial_candidates = fast.generate_initial_candidates(
        evaluator=proxy_evaluator,
        obs=obs,
        mesh_meta=mesh_meta,
        base_scale=base_scale,
        corrected_seed=corrected_seed,
        t_world_from_cam=t_world_from_cam,
        args=args,
    )
    initial_candidates = augment_with_road_snap_candidates(initial_candidates, proxy_evaluator, args)

    temporal_seed = None
    if bool(args.temporal_enabled) and bool(args.temporal_seed_enabled):
        temporal_seed = make_temporal_seed(temporal_prior, proxy_evaluator)
        if temporal_seed is not None:
            print(
                f"[temporal] seed candidate score={temporal_seed['score']:.6f} "
                f"mask_iou={temporal_seed['mask_iou']:.6f} bbox_iou={temporal_seed['bbox_iou']:.6f}"
            )
    initial_candidates = merge_temporal_seed(
        initial_candidates,
        temporal_seed,
        top_k=args.top_k_candidates,
        refine_top_k=args.refine_top_k,
    )
    initial_candidates = augment_with_road_snap_candidates(initial_candidates, proxy_evaluator, args)

    best_result: dict[str, Any] | None = None
    best_history: list[dict[str, Any]] = []
    refined_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    coarse_preview = initial_candidates[: min(5, len(initial_candidates))]
    preview_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(coarse_preview):
        preview_rows.append(
            {
                "rank": index + 1,
                "score": candidate["score"],
                "base_geometry_score": candidate.get("base_geometry_score"),
                "temporal_score": candidate.get("temporal_score"),
                "partial_mask_score": candidate.get("partial_mask_score"),
                "edge_score": candidate.get("edge_score"),
                "mask_iou": candidate["mask_iou"],
        "bbox_iou": candidate["bbox_iou"],
        "bbox_center_error_px": candidate["bbox_center_error_px"],
        "visual_gate_factor": candidate.get("visual_gate_factor"),
        "visual_gate_reason": candidate.get("visual_gate_reason"),
        "initializer_metadata": candidate.get("initializer_metadata", {}),
    }
        )

    candidates_to_refine = select_pareto_refine_candidates(
        initial_candidates,
        args.refine_top_k,
        args,
        truncation_info,
    )
    if temporal_seed is not None and not any(
        candidate.get("initializer_metadata", {}).get("source") == "temporal_prior"
        for candidate in candidates_to_refine
    ):
        if candidates_to_refine:
            candidates_to_refine[-1] = temporal_seed
        else:
            candidates_to_refine.append(temporal_seed)
        print("[temporal] temporal seed forced into refinement set")

    print(f"[search] generated {len(initial_candidates)} candidates, refining top {len(candidates_to_refine)}")

    for rank, candidate in enumerate(candidates_to_refine, start=1):
        meta = candidate.get("initializer_metadata", {})
        source = meta.get("source", "unknown")
        yaw = meta.get("yaw_deg", "n/a")
        scale_factor = meta.get("global_scale_factor", "n/a")
        depth_factor = meta.get("depth_factor", "n/a")
        anchor = meta.get("anchor_name", "n/a")
        print(
            f"[candidate {rank:02d}] init score={candidate['score']:.6f} "
            f"mask_iou={candidate['mask_iou']:.6f} bbox_iou={candidate['bbox_iou']:.6f} "
            f"source={source} yaw={yaw} scale_factor={scale_factor} depth_factor={depth_factor} anchor={anchor}"
        )
        refined_result, history = refine_candidate_stages(candidate, proxy_evaluator, full_evaluator, args)
        print(
            f"[candidate {rank:02d}] refined score={refined_result['score']:.6f} "
            f"mask_iou={refined_result['mask_iou']:.6f} bbox_iou={refined_result['bbox_iou']:.6f}"
        )
        refined_result["initializer_metadata"] = candidate.get("initializer_metadata", {})
        refined_result["candidate_rank"] = rank
        refined_results.append((refined_result, history))
        if best_result is None or refined_result["score"] > best_result["score"]:
            best_result = refined_result
            best_history = history

        if (
            best_result["mask_iou"] >= args.early_stop_mask_iou
            and best_result["bbox_iou"] >= args.early_stop_bbox_iou
        ):
            print(
                f"[early-stop] mask_iou={best_result['mask_iou']:.6f} "
                f"bbox_iou={best_result['bbox_iou']:.6f}"
            )
            break

    selected_result = choose_best_refined_result(
        [item[0] for item in refined_results],
        args,
        truncation_info,
    )
    if selected_result is None:
        best_result = None
        best_history = []
    elif selected_result is not best_result:
        best_result = selected_result
        for refined_result, history in refined_results:
            if refined_result is best_result:
                best_history = history
                break

    if best_result is None:
        raise RuntimeError("No valid pose candidate survived refinement.")

    if bool(truncation_info.get("is_truncated")):
        truncation_info["truncation_severity"] = best_result.get("truncation_severity", truncation_info.get("truncation_severity"))
        truncation_info["low_observability"] = bool(best_result.get("low_observability", truncation_info.get("low_observability", False)))
        truncation_info["truncation_observability_score"] = best_result.get(
            "truncation_observability_score",
            truncation_info.get("truncation_observability_score"),
        )
        truncation_info["truncation_observability_reasons"] = best_result.get(
            "truncation_observability_reasons",
            truncation_info.get("truncation_observability_reasons", []),
        )

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
    temporal_debug_path = save_temporal_edge_debug(
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

    render_validation_outputs: dict[str, Any] = {}
    if args.validate_render_backends:
        render_validation_outputs["render_backends"] = fast.save_render_backend_validation(
            output_dir=output_dir,
            evaluator=full_evaluator,
            mesh=mesh,
            vertices=vertices,
            faces=faces,
            best_result=best_result,
            intrinsics=intrinsics,
            image_size=image_size,
        )
    if args.validate_pytorch3d_alignment:
        render_validation_outputs["pytorch3d_alignment"] = fast.save_pytorch3d_alignment_validation(
            output_dir=output_dir,
            evaluator=full_evaluator,
            vertices=vertices,
            faces=faces,
            best_result=best_result,
            intrinsics=intrinsics,
            image_size=image_size,
        )

    optimized_task = json.loads(json.dumps(task))
    optimized_task["corrected_pose"]["translation_world"] = fast.to_builtin(translation_world)
    optimized_task["corrected_pose"]["rotation_matrix"] = fast.to_builtin(rotation_world)
    optimized_task["corrected_pose"]["scale"] = fast.to_builtin(best_result["scale"])
    with (output_dir / "task_with_optimized_corrected_pose.json").open("w", encoding="utf-8") as f:
        json.dump(fast.to_builtin(optimized_task), f, indent=2)

    final_temporal = None
    if temporal_prior is not None:
        final_temporal = compute_temporal_score(
            np.asarray(best_result["translation_cam"], dtype=np.float64),
            np.asarray(best_result["rotation_cam"], dtype=np.float64),
            np.asarray(best_result["scale"], dtype=np.float64),
            temporal_prior,
            args,
        )

    temporal_report = {
        "enabled": bool(args.temporal_enabled),
        "prior_found": temporal_prior is not None,
        "prior_frame_id": temporal_prior.get("frame_idx") if temporal_prior else None,
        "prior_output_dir": temporal_prior.get("output_dir") if temporal_prior else None,
        "prior_pose_source": temporal_prior.get("pose_source") if temporal_prior else None,
        "delta_translation": final_temporal.get("delta_translation") if final_temporal else None,
        "delta_translation_norm": final_temporal.get("delta_translation_norm") if final_temporal else None,
        "delta_depth": final_temporal.get("delta_depth") if final_temporal else None,
        "delta_rotation_deg": final_temporal.get("delta_rotation_deg") if final_temporal else None,
        "delta_yaw_deg": final_temporal.get("delta_yaw_deg") if final_temporal else None,
        "delta_pitch_deg": final_temporal.get("delta_pitch_deg") if final_temporal else None,
        "delta_roll_deg": final_temporal.get("delta_roll_deg") if final_temporal else None,
        "delta_scale_log": final_temporal.get("delta_scale_log") if final_temporal else None,
        "temporal_score": final_temporal.get("temporal_score") if final_temporal else None,
        "temporal_loss": final_temporal.get("temporal_loss") if final_temporal else None,
        "used_temporal_seed": temporal_seed is not None,
        "best_started_from_temporal_seed": best_result.get("initializer_metadata", {}).get("source") == "temporal_prior",
    }
    partial_report = {
        "enabled": bool(args.partial_visibility_enabled),
        "is_truncated": bool(truncation_info.get("is_truncated")),
        "truncation_sides": truncation_info.get("truncation_sides", []),
        "truncation_severity": truncation_info.get("truncation_severity", "none"),
        "low_observability": bool(truncation_info.get("low_observability", False)),
        "truncation_observability_score": truncation_info.get("truncation_observability_score"),
        "truncation_observability_reasons": truncation_info.get("truncation_observability_reasons", []),
        "border_touch": truncation_info.get("border_touch", {}),
        "bbox_touch": truncation_info.get("bbox_touch", {}),
        "mask_area_px": truncation_info.get("mask_area_px"),
        "prior_mask_area_px": truncation_info.get("prior_mask_area_px"),
        "area_drop_ratio": truncation_info.get("area_drop_ratio"),
        "original_mask_iou": best_result.get("original_mask_iou", best_result.get("mask_iou")),
        "partial_mask_score": best_result.get("partial_mask_score"),
        "adjusted_mask_score": best_result.get("adjusted_mask_score"),
        "partial_score_boost": best_result.get("partial_score_boost"),
        "partial_score_boost_cap": args.partial_score_boost_cap,
        "visible_mask_iou": best_result.get("visible_mask_iou"),
        "visible_soft_mask_iou": best_result.get("visible_soft_mask_iou"),
        "visible_target_fraction": best_result.get("visible_target_fraction"),
        "visible_target_area_px": best_result.get("visible_target_area_px"),
        "visible_bbox_iou": best_result.get("visible_bbox_iou"),
        "visible_bbox_center_error_px": best_result.get("visible_bbox_center_error_px"),
        "visible_contour_score": best_result.get("visible_contour_score"),
        "visible_contour_chamfer_score": best_result.get("visible_contour_chamfer_score"),
        "visible_profile_score": best_result.get("visible_profile_score"),
        "visible_contour_mean_distance_px": best_result.get("visible_contour_mean_distance_px"),
        "visible_profile_mean_distance_px": best_result.get("visible_profile_mean_distance_px"),
        "visible_profile_coverage": best_result.get("visible_profile_coverage"),
        "effective_visible_contour_weight": best_result.get("effective_visible_contour_weight"),
        "truncated_visual_quality_gate": best_result.get("truncated_visual_quality_gate"),
        "truncated_visual_quality_reason": best_result.get("truncated_visual_quality_reason"),
        "truncated_visual_bbox_factor": best_result.get("truncated_visual_bbox_factor"),
        "truncated_visual_center_factor": best_result.get("truncated_visual_center_factor"),
        "truncated_visual_mask_factor": best_result.get("truncated_visual_mask_factor"),
        "truncated_visual_contour_factor": best_result.get("truncated_visual_contour_factor"),
        "truncated_visual_profile_factor": best_result.get("truncated_visual_profile_factor"),
        "truncated_visual_overflow_factor": best_result.get("truncated_visual_overflow_factor"),
        "truncated_visual_overflow_loss": best_result.get("truncated_visual_overflow_loss"),
        "truncated_visual_quality_penalty": best_result.get("truncated_visual_quality_penalty"),
        "severe_truncation_gate_passed": best_result.get("severe_truncation_gate_passed"),
        "severe_truncation_gate_reasons": best_result.get("severe_truncation_gate_reasons"),
        "visible_projected_bbox": best_result.get("visible_projected_bbox"),
        "visible_target_bbox": best_result.get("visible_target_bbox"),
        "visible_bbox_source": best_result.get("visible_bbox_source"),
        "truncated_bbox_score": best_result.get("truncated_bbox_score"),
        "truncated_bbox_loss": best_result.get("truncated_bbox_loss"),
        "truncated_bbox_penalty": best_result.get("truncated_bbox_penalty"),
        "truncated_bbox_components": best_result.get("truncated_bbox_components"),
        "truncated_bbox_source": best_result.get("truncated_bbox_source"),
    }
    edge_report = {
        "enabled": bool(args.edge_score_enabled),
        "available": edge_context is not None,
        "edge_score": best_result.get("edge_score"),
        "edge_mean_distance_px": best_result.get("edge_mean_distance_px"),
        "edge_rendered_points": best_result.get("edge_rendered_points"),
        "edge_roi": best_result.get("edge_roi"),
        "effective_edge_weight": best_result.get("effective_edge_weight"),
    }
    heading_report = {
        "enabled": bool(args.heading_prior_enabled),
        "available": bool((vehicle_pose_context.get("heading_prior") or {}).get("vector_image")),
        "source": (vehicle_pose_context.get("heading_prior") or {}).get("source"),
        "confidence": best_result.get("heading_prior_confidence"),
        "score": best_result.get("heading_prior_score"),
        "angle_error_deg": best_result.get("heading_prior_angle_error_deg"),
        "front_sign_enabled": best_result.get("heading_front_sign_enabled"),
        "front_sign_confidence": best_result.get("heading_front_sign_confidence"),
        "candidate_forward_sign": best_result.get("heading_candidate_forward_sign"),
        "semantic_front_sign": best_result.get("heading_semantic_front_sign"),
        "tail_light_front_sign": best_result.get("heading_tail_light_front_sign"),
        "tail_light_flipped": best_result.get("heading_tail_light_flipped"),
        "front_sign_hard_rejected": best_result.get("heading_front_sign_hard_rejected"),
        "front_angle_penalty": best_result.get("heading_front_angle_penalty"),
        "depth_trend_score": best_result.get("heading_depth_trend_score"),
        "depth_trend_direction": best_result.get("heading_depth_trend_direction"),
        "depth_trend_confidence": best_result.get("heading_depth_trend_confidence"),
        "front_depth_cam": best_result.get("heading_front_depth_cam"),
        "front_sign_penalty": best_result.get("heading_front_sign_penalty"),
        "effective_front_sign_penalty_weight": best_result.get("effective_front_sign_penalty_weight"),
        "bbox_area_trend": (vehicle_pose_context.get("heading_prior") or {}).get("bbox_area_trend"),
        "projected_vector_image": best_result.get("heading_prior_projected_vector_image"),
        "target_vector_image": best_result.get("heading_prior_target_vector_image"),
        "effective_weight": best_result.get("effective_heading_prior_weight"),
    }

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
        "scale_constraint": "uniform_xyz",
        "optimized_uniform_scale": best_uniform_scale,
        "initializer_top_candidates": preview_rows,
        "best_initializer_metadata": best_result.get("initializer_metadata", {}),
        "best_candidate_rank": best_result.get("candidate_rank"),
        "temporal": temporal_report,
        "partial_visibility": partial_report,
        "edge_assist": edge_report,
        "heading_prior": heading_report,
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
        "metrics": {
            "score": best_result["score"],
            "base_geometry_score": best_result.get("base_geometry_score"),
            "geometry_score": best_result.get("geometry_score"),
            "truncated_bbox_penalty": best_result.get("truncated_bbox_penalty"),
            "effective_temporal_weight": best_result.get("effective_temporal_weight"),
            "effective_edge_weight": best_result.get("effective_edge_weight"),
            "ground_contact_penalty": best_result.get("ground_contact_penalty"),
            "heading_prior_score": best_result.get("heading_prior_score"),
            "heading_prior_angle_error_deg": best_result.get("heading_prior_angle_error_deg"),
            "heading_front_sign_enabled": best_result.get("heading_front_sign_enabled"),
            "heading_front_sign_confidence": best_result.get("heading_front_sign_confidence"),
            "heading_candidate_forward_sign": best_result.get("heading_candidate_forward_sign"),
            "heading_semantic_front_sign": best_result.get("heading_semantic_front_sign"),
            "heading_tail_light_front_sign": best_result.get("heading_tail_light_front_sign"),
            "heading_tail_light_flipped": best_result.get("heading_tail_light_flipped"),
            "heading_front_sign_hard_rejected": best_result.get("heading_front_sign_hard_rejected"),
            "heading_front_angle_penalty": best_result.get("heading_front_angle_penalty"),
            "heading_depth_trend_score": best_result.get("heading_depth_trend_score"),
            "heading_depth_trend_direction": best_result.get("heading_depth_trend_direction"),
            "heading_depth_trend_confidence": best_result.get("heading_depth_trend_confidence"),
            "heading_front_depth_cam": best_result.get("heading_front_depth_cam"),
            "heading_front_sign_penalty": best_result.get("heading_front_sign_penalty"),
            "effective_heading_prior_weight": best_result.get("effective_heading_prior_weight"),
            "effective_front_sign_penalty_weight": best_result.get("effective_front_sign_penalty_weight"),
            "visible_mask_bonus": best_result.get("visible_mask_bonus"),
            "visible_mask_bonus_weight": best_result.get("visible_mask_bonus_weight"),
            "visible_mask_iou": best_result.get("visible_mask_iou"),
            "visible_soft_mask_iou": best_result.get("visible_soft_mask_iou"),
            "visible_target_fraction": best_result.get("visible_target_fraction"),
            "visible_target_area_px": best_result.get("visible_target_area_px"),
            "visible_bbox_iou": best_result.get("visible_bbox_iou"),
            "visible_bbox_center_error_px": best_result.get("visible_bbox_center_error_px"),
            "visible_contour_score": best_result.get("visible_contour_score"),
            "visible_contour_chamfer_score": best_result.get("visible_contour_chamfer_score"),
            "visible_profile_score": best_result.get("visible_profile_score"),
            "visible_contour_mean_distance_px": best_result.get("visible_contour_mean_distance_px"),
            "visible_profile_mean_distance_px": best_result.get("visible_profile_mean_distance_px"),
            "effective_visible_contour_weight": best_result.get("effective_visible_contour_weight"),
            "truncation_severity": best_result.get("truncation_severity", truncation_info.get("truncation_severity")),
            "low_observability": best_result.get("low_observability", truncation_info.get("low_observability")),
            "truncation_observability_score": best_result.get("truncation_observability_score", truncation_info.get("truncation_observability_score")),
            "truncation_observability_reasons": best_result.get("truncation_observability_reasons", truncation_info.get("truncation_observability_reasons")),
            "truncated_visual_quality_gate": best_result.get("truncated_visual_quality_gate"),
            "truncated_visual_quality_reason": best_result.get("truncated_visual_quality_reason"),
            "truncated_visual_bbox_factor": best_result.get("truncated_visual_bbox_factor"),
            "truncated_visual_center_factor": best_result.get("truncated_visual_center_factor"),
            "truncated_visual_mask_factor": best_result.get("truncated_visual_mask_factor"),
            "truncated_visual_contour_factor": best_result.get("truncated_visual_contour_factor"),
            "truncated_visual_profile_factor": best_result.get("truncated_visual_profile_factor"),
            "truncated_visual_overflow_factor": best_result.get("truncated_visual_overflow_factor"),
            "truncated_visual_overflow_loss": best_result.get("truncated_visual_overflow_loss"),
            "truncated_visual_quality_penalty": best_result.get("truncated_visual_quality_penalty"),
            "severe_truncation_gate_passed": best_result.get("severe_truncation_gate_passed"),
            "severe_truncation_gate_reasons": best_result.get("severe_truncation_gate_reasons"),
            "visible_projected_bbox": best_result.get("visible_projected_bbox"),
            "visible_target_bbox": best_result.get("visible_target_bbox"),
            "visible_bbox_source": best_result.get("visible_bbox_source"),
            "truncated_bbox_source": best_result.get("truncated_bbox_source"),
            "final_ground_constrained_selected": best_result.get("final_ground_constrained_selected"),
            "final_ground_constrained_rank_score": best_result.get("final_ground_constrained_rank_score"),
            "final_selection_mode": best_result.get("final_selection_mode"),
            "final_visual_selection_weights": best_result.get("final_visual_selection_weights"),
            "effective_visual_bbox_weight": best_result.get("effective_visual_bbox_weight"),
            "effective_hard_mask_weight": best_result.get("effective_hard_mask_weight"),
            "mask_iou": best_result["mask_iou"],
            "soft_mask_iou": best_result.get("soft_mask_iou"),
            "bbox_iou": best_result["bbox_iou"],
            "bbox_center_error_px": best_result["bbox_center_error_px"],
            "projected_bbox": best_result["projected_bbox"],
        },
        "outputs": {
            "alignment_collage": str(collage_paths["alignment_collage"]),
            "pose_closeup_collage": str(collage_paths["pose_closeup_collage"]),
            "model_reference_collage": str(collage_paths["model_reference_collage"]),
            "temporal_edge_debug": str(temporal_debug_path) if temporal_debug_path else None,
            "optimization_history": str(output_dir / "optimization_history.csv"),
            "optimization_report": str(output_dir / "optimization_report.json"),
            "optimized_task": str(output_dir / "task_with_optimized_corrected_pose.json"),
        },
    }
    if render_validation_outputs:
        report["render_validation"] = render_validation_outputs
    if args.profile_timings:
        report["profiling"] = fast.combine_profile_stats([proxy_evaluator, full_evaluator])
    report["refined_pose_candidates"] = [
        refined_pose_candidate_summary(item[0], t_world_from_cam=t_world_from_cam)
        for item in refined_results
    ]
    with (output_dir / "optimization_report.json").open("w", encoding="utf-8") as f:
        json.dump(fast.to_builtin(report), f, indent=2)
    fast.cleanup_result_images(output_dir)
    proxy_evaluator.close()
    full_evaluator.close()
    return report


def add_fast_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample_dir", default=r"E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--render_backend", choices=["auto", "pyrender", "triangle_fill", "pytorch3d"], default="triangle_fill")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pytorch3d_faces_per_pixel", type=int, default=1)
    parser.add_argument("--pytorch3d_cull_backfaces", nargs="?", const=True, default=False, type=fast.parse_bool_arg)
    parser.add_argument("--pytorch3d_bin_size", type=int, default=None)
    parser.add_argument("--pytorch3d_max_faces_per_bin", type=int, default=None)
    parser.add_argument("--validate_render_backends", action="store_true")
    parser.add_argument("--validate_pytorch3d_alignment", action="store_true")
    parser.add_argument("--enable_batch_gpu_eval", action="store_true")
    parser.add_argument("--batch_gpu_size", type=int, default=32)
    parser.add_argument("--bbox_weight", type=float, default=0.1)
    parser.add_argument("--hard_mask_weight", type=float, default=0.0)
    parser.add_argument("--truncated_hard_mask_weight", type=float, default=None)
    parser.add_argument("--truncated_visual_bbox_weight", type=float, default=None)
    parser.add_argument("--include_corrected_seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_bbox_prefilter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefilter_bbox_iou_min", type=float, default=0.05)
    parser.add_argument("--prefilter_center_factor", type=float, default=1.8)
    parser.add_argument("--prefilter_size_ratio_min", type=float, default=0.35)
    parser.add_argument("--prefilter_size_ratio_max", type=float, default=3.0)
    parser.add_argument("--roi_iou_margin", type=int, default=30)
    parser.add_argument("--disable_roi_iou", action="store_true")
    parser.add_argument("--fast_float32", action="store_true")
    parser.add_argument("--save_full_history", action="store_true")
    parser.add_argument("--profile_timings", action="store_true")
    parser.add_argument("--world_up_axis", choices=["x", "y", "z", "+x", "+y", "+z", "-x", "-y", "-z"], default="-y")
    parser.add_argument("--proxy_face_count", type=int, default=1800)
    parser.add_argument("--top_k_candidates", type=int, default=8)
    parser.add_argument("--refine_top_k", type=int, default=3)
    parser.add_argument("--early_stop_mask_iou", type=float, default=0.90)
    parser.add_argument("--early_stop_bbox_iou", type=float, default=0.85)
    parser.add_argument("--init_yaw_step_deg", type=float, default=15.0)
    parser.add_argument("--init_scale_factors", default="0.5,0.7,1.0,1.3,1.6")
    parser.add_argument("--init_depth_factors", default="0.8,1.0,1.2")
    parser.add_argument("--stage1_iters", type=int, default=10)
    parser.add_argument("--stage2_iters", type=int, default=8)
    parser.add_argument("--stage3_iters", type=int, default=14)
    parser.add_argument("--step_decay", type=float, default=0.5)
    parser.add_argument("--max_translation_delta", type=float, default=0.8)
    parser.add_argument("--max_rotation_delta_deg", type=float, default=45.0)
    parser.add_argument("--scale_min_factor", type=float, default=0.5)
    parser.add_argument("--scale_max_factor", type=float, default=2.2)


def add_temporal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temporal_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal_lookback", type=int, default=5)
    parser.add_argument("--temporal_seed_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal_weight", type=float, default=0.15)
    parser.add_argument("--truncated_temporal_weight", type=float, default=0.08)
    parser.add_argument("--temporal_translation_sigma", type=float, default=0.35)
    parser.add_argument("--temporal_depth_sigma", type=float, default=0.45)
    parser.add_argument("--temporal_rotation_sigma_deg", type=float, default=20.0)
    parser.add_argument("--temporal_yaw_sigma_deg", type=float, default=15.0)
    parser.add_argument("--temporal_scale_sigma", type=float, default=0.12)
    parser.add_argument("--temporal_max_allowed_jump_deg", type=float, default=60.0)
    parser.add_argument("--temporal_max_allowed_scale_ratio", type=float, default=1.6)
    parser.add_argument("--temporal_search_output_suffixes", default="_temporal_fast_quick,_fast_quick,_fast,_baseline")
    parser.add_argument("--track_scale_prior_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--track_scale_prior_weight", type=float, default=0.18)
    parser.add_argument("--track_scale_prior_sigma", type=float, default=0.10)

    parser.add_argument("--partial_visibility_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncation_border_margin", type=int, default=3)
    parser.add_argument("--truncation_bbox_margin", type=int, default=5)
    parser.add_argument("--partial_visibility_weight", type=float, default=0.35)
    parser.add_argument("--partial_score_boost_cap", type=float, default=0.03)
    parser.add_argument("--ignore_truncated_border_band_px", type=int, default=8)
    parser.add_argument("--partial_iou_mode", default="visible_region")
    parser.add_argument("--partial_use_one_sided_distance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncation_moderate_visible_mask_iou", type=float, default=0.78)
    parser.add_argument("--truncation_severe_visible_mask_iou", type=float, default=0.70)
    parser.add_argument("--truncation_moderate_contour_mean_px", type=float, default=5.0)
    parser.add_argument("--truncation_severe_contour_mean_px", type=float, default=7.0)
    parser.add_argument("--truncation_moderate_profile_mean_px", type=float, default=8.0)
    parser.add_argument("--truncation_severe_profile_mean_px", type=float, default=10.0)
    parser.add_argument("--truncation_area_drop_ratio", type=float, default=0.72)
    parser.add_argument("--truncation_moderate_visible_target_fraction", type=float, default=0.35)
    parser.add_argument("--truncation_severe_visible_target_fraction", type=float, default=0.12)
    parser.add_argument("--truncated_bbox_constraint_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncated_bbox_weight", type=float, default=0.08)
    parser.add_argument("--truncated_bbox_top_sigma_px", type=float, default=10.0)
    parser.add_argument("--truncated_bbox_side_sigma_px", type=float, default=8.0)
    parser.add_argument("--truncated_bbox_center_x_sigma_px", type=float, default=8.0)
    parser.add_argument("--truncated_bbox_bottom_overflow_sigma_px", type=float, default=5.0)
    parser.add_argument("--truncated_bbox_penalty_cap", type=float, default=0.30)

    parser.add_argument("--edge_score_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge_weight", type=float, default=0.08)
    parser.add_argument("--truncated_edge_weight", type=float, default=0.18)
    parser.add_argument("--edge_canny_low", type=int, default=50)
    parser.add_argument("--edge_canny_high", type=int, default=150)
    parser.add_argument("--edge_distance_sigma_px", type=float, default=4.0)
    parser.add_argument("--edge_roi_margin", type=int, default=20)
    parser.add_argument("--edge_use_mask_erode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge_topk_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--road_constraint_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vehicle_mesh_axis_override_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vehicle_mesh_up_axis_idx", type=int, default=1)
    parser.add_argument("--vehicle_mesh_up_sign", type=float, default=-1.0)
    parser.add_argument("--road_depth_fallback_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--road_depth_map_path", default="")
    parser.add_argument("--road_depth_ransac_max_points", type=int, default=20000)
    parser.add_argument("--road_depth_ransac_iters", type=int, default=96)
    parser.add_argument("--road_depth_ransac_threshold_m", type=float, default=0.12)
    parser.add_argument("--road_constraint_weight", type=float, default=0.25)
    parser.add_argument("--bbox_bottom_ground_weight", type=float, default=0.15)
    parser.add_argument("--bottom_truncated_ground_weight_factor", type=float, default=0.25)
    parser.add_argument("--moderate_bottom_truncated_bbox_bottom_weight_factor", type=float, default=0.40)
    parser.add_argument("--severe_bottom_truncated_bbox_bottom_weight_factor", type=float, default=0.0)
    parser.add_argument("--bottom_truncated_ground_contact_weight_factor", type=float, default=1.60)
    parser.add_argument("--bottom_truncated_ground_soft_tolerance_m", type=float, default=0.12)
    parser.add_argument("--bottom_truncated_ground_penalty_weight", type=float, default=0.35)
    parser.add_argument("--bottom_truncated_ground_penalty_sigma_m", type=float, default=0.18)
    parser.add_argument("--truncated_ground_contact_hard_gate_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upright_weight", type=float, default=0.10)
    parser.add_argument("--ground_contact_sigma_m", type=float, default=0.18)
    parser.add_argument("--ground_contact_hard_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ground_contact_hard_gate_mean_m", type=float, default=0.30)
    parser.add_argument("--ground_contact_hard_gate_max_m", type=float, default=0.60)
    parser.add_argument("--bbox_bottom_ground_sigma_m", type=float, default=0.45)
    parser.add_argument("--upright_angle_sigma_deg", type=float, default=10.0)
    parser.add_argument("--upright_hard_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upright_strong_penalty_angle_deg", type=float, default=15.0)
    parser.add_argument("--upright_hard_gate_max_angle_deg", type=float, default=60.0)
    parser.add_argument("--upright_hard_gate_sigma_deg", type=float, default=15.0)
    parser.add_argument("--upright_hard_gate_penalty", type=float, default=2.0)
    parser.add_argument("--road_aligned_initialization_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--road_aligned_initialization_for_truncated_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--road_snap_candidate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--road_snap_candidate_source_top_k", type=int, default=16)
    parser.add_argument("--road_snap_candidate_max_distance_m", type=float, default=0.30)
    parser.add_argument("--pareto_refine_selection_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pareto_refine_branch_quota", type=int, default=2)
    parser.add_argument("--truncated_candidate_ground_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncated_candidate_ground_gate_mean_m", type=float, default=0.085)
    parser.add_argument("--truncated_candidate_ground_gate_max_m", type=float, default=0.18)
    parser.add_argument("--truncated_candidate_ground_gate_fallback_top_k", type=int, default=2)
    parser.add_argument("--final_ground_constrained_selection_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final_ground_select_mean_max_m", type=float, default=0.055)
    parser.add_argument("--final_ground_select_max_max_m", type=float, default=0.12)
    parser.add_argument("--truncated_final_visual_selection_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncated_final_visual_min_bbox_iou", type=float, default=0.88)
    parser.add_argument("--truncated_final_visual_min_quality_gate", type=float, default=0.80)
    parser.add_argument("--truncated_final_visual_mask_weight", type=float, default=1.0)
    parser.add_argument("--truncated_final_visual_contour_weight", type=float, default=0.35)
    parser.add_argument("--truncated_final_visual_bbox_weight", type=float, default=0.08)
    parser.add_argument("--severe_truncated_final_visual_bbox_weight", type=float, default=0.02)
    parser.add_argument("--truncated_final_visual_quality_weight", type=float, default=0.05)
    parser.add_argument("--truncated_final_visual_ground_mean_weight", type=float, default=0.25)
    parser.add_argument("--truncated_final_visual_ground_max_weight", type=float, default=0.10)
    parser.add_argument("--truncated_final_visual_score_weight", type=float, default=0.03)
    parser.add_argument("--visual_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_gate_mask_iou_min", type=float, default=0.65)
    parser.add_argument("--visual_gate_bbox_iou_min", type=float, default=0.75)
    parser.add_argument("--visual_gate_center_error_px_max", type=float, default=20.0)
    parser.add_argument("--visible_mask_bonus_weight", type=float, default=0.0)
    parser.add_argument("--truncated_visible_mask_bonus_weight", type=float, default=0.0)
    parser.add_argument("--truncated_visible_contour_weight", type=float, default=0.0)
    parser.add_argument("--truncated_visible_contour_sigma_px", type=float, default=4.0)
    parser.add_argument("--truncated_visible_profile_weight", type=float, default=0.35)
    parser.add_argument("--truncated_visible_profile_sigma_px", type=float, default=5.0)
    parser.add_argument("--truncated_visual_quality_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncated_visual_quality_gate_floor", type=float, default=0.25)
    parser.add_argument("--truncated_visual_gate_bbox_iou_min", type=float, default=0.88)
    parser.add_argument("--truncated_visual_gate_bbox_iou_softness", type=float, default=0.08)
    parser.add_argument("--truncated_visual_gate_center_error_px", type=float, default=6.0)
    parser.add_argument("--truncated_visual_gate_center_softness_px", type=float, default=8.0)
    parser.add_argument("--truncated_visual_gate_overflow_sigma_px", type=float, default=32.0)
    parser.add_argument("--truncated_visual_gate_visible_mask_iou_good", type=float, default=0.78)
    parser.add_argument("--truncated_visual_gate_visible_mask_iou_bad", type=float, default=0.68)
    parser.add_argument("--truncated_visual_gate_contour_mean_px_good", type=float, default=4.5)
    parser.add_argument("--truncated_visual_gate_contour_mean_px_bad", type=float, default=7.0)
    parser.add_argument("--truncated_visual_gate_profile_mean_px_good", type=float, default=7.5)
    parser.add_argument("--truncated_visual_gate_profile_mean_px_bad", type=float, default=10.0)
    parser.add_argument("--truncated_visual_quality_penalty_weight", type=float, default=0.08)
    parser.add_argument("--severe_truncation_final_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--severe_truncation_visible_mask_iou_min", type=float, default=0.68)
    parser.add_argument("--severe_truncation_visible_contour_mean_px_max", type=float, default=7.0)
    parser.add_argument("--severe_truncation_visible_profile_mean_px_max", type=float, default=10.0)
    parser.add_argument("--severe_truncation_visible_target_fraction_min", type=float, default=0.12)
    parser.add_argument("--severe_truncation_ground_mean_m_max", type=float, default=0.085)
    parser.add_argument("--severe_truncation_ground_max_m_max", type=float, default=0.18)
    parser.add_argument("--severe_truncation_upright_deg_max", type=float, default=35.0)
    parser.add_argument("--severe_truncation_yaw_jump_deg_max", type=float, default=25.0)
    parser.add_argument("--severe_truncation_fallback_visible_mask_iou_min", type=float, default=0.76)
    parser.add_argument("--severe_truncation_fallback_visible_contour_mean_px_max", type=float, default=5.5)
    parser.add_argument("--severe_truncation_fallback_visible_profile_mean_px_max", type=float, default=8.5)
    parser.add_argument("--severe_truncation_fallback_visible_target_fraction_min", type=float, default=0.12)
    parser.add_argument("--heading_prior_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heading_prior_weight", type=float, default=0.06)
    parser.add_argument("--front_sign_heading_prior_weight", type=float, default=0.18)
    parser.add_argument("--mesh_tail_light_front_sign_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mesh_tail_light_front_sign_min_confidence", type=float, default=0.20)
    parser.add_argument("--mesh_tail_light_front_sign_standalone_min_confidence", type=float, default=0.75)
    parser.add_argument("--mesh_tail_light_front_sign_standalone_min_density_ratio", type=float, default=5.0)
    parser.add_argument("--front_sign_depth_trend_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_sign_depth_trend_weight", type=float, default=0.35)
    parser.add_argument("--front_sign_depth_trend_penalty", type=float, default=0.45)
    parser.add_argument("--front_sign_depth_trend_hard_gate_score_min", type=float, default=0.35)
    parser.add_argument("--front_sign_depth_trend_min_monotonicity", type=float, default=0.75)
    parser.add_argument("--front_sign_depth_trend_hard_gate_min_heading_angle_deg", type=float, default=45.0)
    parser.add_argument("--bbox_area_trend_front_sign_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bbox_area_trend_front_sign_min_confidence", type=float, default=0.75)
    parser.add_argument("--bbox_area_trend_front_sign_min_monotonicity", type=float, default=0.75)
    parser.add_argument("--bbox_area_trend_front_sign_min_axis_confidence", type=float, default=0.50)
    parser.add_argument("--bbox_area_trend_front_sign_confidence_scale", type=float, default=0.90)
    parser.add_argument("--front_sign_mismatch_penalty", type=float, default=0.80)
    parser.add_argument("--front_sign_angle_penalty_weight", type=float, default=1.20)
    parser.add_argument("--front_sign_hard_gate_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--front_sign_hard_gate_angle_deg", type=float, default=120.0)
    parser.add_argument("--front_sign_hard_gate_min_confidence", type=float, default=0.25)
    parser.add_argument("--tail_light_motion_consistency_flip_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tail_light_motion_consistency_min_confidence", type=float, default=0.60)
    parser.add_argument("--tail_light_motion_consistency_flip_margin", type=float, default=0.20)
    parser.add_argument("--truncated_heading_prior_weight", type=float, default=0.08)
    parser.add_argument("--severe_truncation_heading_prior_weight", type=float, default=0.18)
    parser.add_argument("--heading_prior_sigma_deg", type=float, default=25.0)
    parser.add_argument("--heading_prior_lock_front_sign", action=argparse.BooleanOptionalAction, default=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal fast pose optimizer with prior, partial visibility, and edge assist."
    )
    add_fast_arguments(parser)
    add_temporal_arguments(parser)
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
    print(f"best_mask_iou: {metrics['mask_iou']:.6f}")
    print(f"best_bbox_iou: {metrics['bbox_iou']:.6f}")
    print(f"best_bbox_center_error_px: {metrics['bbox_center_error_px']:.6f}")
    print(f"best_projected_bbox: {metrics['projected_bbox']}")
    print(f"optimized_translation_world: {fast.to_builtin(pose_world['translation_world'])}")
    print(f"optimized_scale: {fast.to_builtin(pose_world['scale'])}")
    print(f"render_backend: {report['render_backend']}")
    print(f"alignment_collage_path: {report['outputs']['alignment_collage']}")
    print(f"pose_closeup_collage_path: {report['outputs']['pose_closeup_collage']}")
    print(f"model_reference_collage_path: {report['outputs']['model_reference_collage']}")
    print(f"report_path: {report['outputs']['optimization_report']}")
    print(f"optimized_task_path: {report['outputs']['optimized_task']}")


if __name__ == "__main__":
    main()
