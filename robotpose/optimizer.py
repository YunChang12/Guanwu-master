from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .losses import LossConfig, LossResult, compute_partial_observation_loss
from .render import (
    RobotMesh,
    bbox_from_mask,
    bbox_from_projected_points,
    project_points,
    render_robot_mask,
    scale_camera_K,
    transform_points,
)


@dataclass(frozen=True)
class OptimizerOptions:
    loss_config: LossConfig = field(default_factory=LossConfig)
    coarse_depths: tuple[float, ...] = (0.8, 1.2, 1.8, 2.5, 3.5, 5.0)
    coarse_yaws_deg: tuple[float, ...] = tuple(float(v) for v in range(0, 360, 30))
    coarse_pitch_deg: tuple[float, ...] = (0.0,)
    coarse_roll_deg: tuple[float, ...] = (0.0,)
    coarse_lateral_offsets_px: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    top_k: int = 8
    stage_scales: tuple[float, ...] = (0.35, 0.65, 1.0)
    max_iterations: int = 90
    optimizer_method: str = "Powell"
    max_translation_delta: float | None = None
    max_rotation_delta_deg: float | None = None
    min_depth: float | None = None
    max_depth: float | None = None
    invalid_pose_loss: float = 1_000_000.0
    coarse_search: bool = True
    pattern_translation_steps: tuple[float, ...] = (0.25, 0.12, 0.06, 0.03, 0.015)
    pattern_rotation_steps_deg: tuple[float, ...] = (8.0, 4.0, 2.0)
    moment_refine_iterations: int = 4
    enable_initial_bbox_prefilter: bool = True
    initial_prefilter_render_limit: int = 512
    initial_prefilter_vertex_limit: int = 512
    prefilter_bbox_iou_min: float = 0.01
    prefilter_center_factor: float = 1.25
    prefilter_size_ratio_min: float = 0.05
    prefilter_size_ratio_max: float = 8.0
    prefilter_min_valid_ratio: float = 0.05


@dataclass(frozen=True)
class RobotPoseResult:
    T_C_B: np.ndarray
    loss: LossResult
    initial_loss: LossResult
    rendered_mask: np.ndarray
    selected_init_index: int
    history: list[dict[str, Any]]
    projected_bbox: list[float] | None
    initial_candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    initial_candidate_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def translation(self) -> np.ndarray:
        return self.T_C_B[:3, 3]

    @property
    def rotation_vector(self) -> np.ndarray:
        return Rotation.from_matrix(self.T_C_B[:3, :3]).as_rotvec()

    @property
    def quaternion_xyzw(self) -> np.ndarray:
        return Rotation.from_matrix(self.T_C_B[:3, :3]).as_quat()


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def compose_delta_pose(T0_C_B: np.ndarray, delta: np.ndarray) -> np.ndarray:
    params = np.asarray(delta, dtype=np.float64).reshape(6)
    dT = make_transform(Rotation.from_rotvec(params[:3]).as_matrix(), params[3:6])
    return dT @ np.asarray(T0_C_B, dtype=np.float64)


def resize_mask_nearest(mask: np.ndarray, scale: float) -> np.ndarray:
    from PIL import Image

    binary = np.asarray(mask).astype(bool)
    if abs(float(scale) - 1.0) < 1e-9:
        return binary
    height, width = binary.shape
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(binary.astype(np.uint8) * 255).resize(new_size, Image.Resampling.NEAREST)) > 0


def generate_pose_candidates(
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    options: OptimizerOptions,
) -> list[np.ndarray]:
    bbox = bbox_from_mask(observed_mask)
    K = np.asarray(camera_K, dtype=np.float64)
    if bbox is None:
        center_u = float(K[0, 2])
        center_v = float(K[1, 2])
    else:
        center_u = 0.5 * (bbox[0] + bbox[2])
        center_v = 0.5 * (bbox[1] + bbox[3])
    fx = max(1e-9, float(K[0, 0]))
    fy = max(1e-9, float(K[1, 1]))
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    candidates: list[np.ndarray] = []
    for depth in options.coarse_depths:
        z = float(depth)
        for du, dv in options.coarse_lateral_offsets_px:
            x = ((center_u + du) - cx) * z / fx
            y = ((center_v + dv) - cy) * z / fy
            for yaw in options.coarse_yaws_deg:
                for pitch in options.coarse_pitch_deg:
                    for roll in options.coarse_roll_deg:
                        R = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
                        candidates.append(make_transform(R, np.array([x, y, z], dtype=np.float64)))
    return candidates


def optimize_robot_pose(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    init_poses: list[np.ndarray] | None = None,
    options: OptimizerOptions | None = None,
) -> RobotPoseResult:
    opts = options or OptimizerOptions()
    K = np.asarray(camera_K, dtype=np.float64)
    obs = np.asarray(observed_mask).astype(bool)
    if obs.shape != tuple(image_shape):
        raise ValueError(f"observed_mask shape {obs.shape} does not match image_shape {image_shape}")
    if init_poses:
        candidates = [np.asarray(T, dtype=np.float64).reshape(4, 4) for T in init_poses]
    elif opts.coarse_search:
        candidates = generate_pose_candidates(K, obs, opts)
    else:
        raise ValueError("init_poses are required when coarse_search is disabled.")
    if not candidates:
        raise ValueError("No initial pose candidates available.")

    scored_candidates, initial_rows, initial_summary = _score_initial_candidates_with_details(
        mesh,
        K,
        obs,
        image_shape,
        candidates,
        opts,
    )
    top = scored_candidates[: max(1, int(opts.top_k))]
    best: RobotPoseResult | None = None
    history: list[dict[str, Any]] = []
    for rank, (init_index, _, T0) in enumerate(top):
        result = _optimize_from_init(mesh, K, obs, image_shape, T0, init_index=init_index, options=opts)
        history.extend({"candidate_rank": rank, **row} for row in result.history)
        if best is None or result.loss.total < best.loss.total:
            best = result
    if best is None:
        raise RuntimeError("Pose optimization produced no result.")
    return RobotPoseResult(
        T_C_B=best.T_C_B,
        loss=best.loss,
        initial_loss=best.initial_loss,
        rendered_mask=best.rendered_mask,
        selected_init_index=best.selected_init_index,
        history=history,
        projected_bbox=best.projected_bbox,
        initial_candidate_scores=initial_rows,
        initial_candidate_summary=initial_summary,
    )


def _score_initial_candidates(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    candidates: list[np.ndarray],
    options: OptimizerOptions,
) -> list[tuple[int, float, np.ndarray]]:
    scored, _, _ = _score_initial_candidates_with_details(
        mesh,
        camera_K,
        observed_mask,
        image_shape,
        candidates,
        options,
    )
    return scored


def _score_initial_candidates_with_details(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    candidates: list[np.ndarray],
    options: OptimizerOptions,
) -> tuple[list[tuple[int, float, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    scale = min(1.0, float(options.stage_scales[0]) if options.stage_scales else 1.0)
    K_s = scale_camera_K(camera_K, scale)
    obs_s = resize_mask_nearest(observed_mask, scale)
    shape_s = obs_s.shape
    candidate_rows = _prefilter_initial_candidates(mesh, K_s, obs_s, shape_s, candidates, options)
    if options.enable_initial_bbox_prefilter:
        render_rows = [row for row in candidate_rows if row["prefilter_keep"]]
        if not render_rows:
            render_rows = [
                row
                for row in candidate_rows
                if row["projected_bbox"] is not None and np.isfinite(float(row["prefilter_loss"]))
            ]
        render_rows.sort(key=lambda row: float(row["prefilter_loss"]))
        limit = int(options.initial_prefilter_render_limit)
        if limit > 0:
            render_rows = render_rows[:limit]
    else:
        render_rows = candidate_rows

    scored: list[tuple[int, float, np.ndarray]] = []
    scored_rows: list[dict[str, Any]] = []
    for row in render_rows:
        index = int(row["index"])
        T0 = candidates[index]
        loss = _loss_for_pose(mesh, K_s, obs_s, shape_s, T0, T0, options)
        scored.append((index, loss.total, T0))
        scored_rows.append(
            {
                **row,
                "rendered_initial_score": True,
                "loss_total": float(loss.total),
                **{f"term_{key}": float(value) for key, value in loss.terms.items()},
                **{f"metric_{key}": float(value) for key, value in loss.metrics.items()},
            }
        )
    scored.sort(key=lambda item: item[1])
    scored_rows.sort(key=lambda item: float(item.get("loss_total", float("inf"))))
    summary = {
        "total_candidates": int(len(candidates)),
        "prefilter_enabled": bool(options.enable_initial_bbox_prefilter),
        "prefilter_scale": float(scale),
        "prefilter_kept": int(sum(1 for row in candidate_rows if row["prefilter_keep"])),
        "initial_render_limit": int(options.initial_prefilter_render_limit),
        "rendered_initial_candidates": int(len(scored_rows)),
    }
    return scored, scored_rows, summary


def _prefilter_initial_candidates(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    candidates: list[np.ndarray],
    options: OptimizerOptions,
) -> list[dict[str, Any]]:
    vertices = _prefilter_vertices(mesh, int(options.initial_prefilter_vertex_limit))
    obs_bbox = bbox_from_mask(observed_mask)
    if obs_bbox is None:
        return [
            {
                "index": int(index),
                "prefilter_keep": True,
                "prefilter_reject_reason": "",
                "prefilter_loss": 0.0,
                "projected_bbox": None,
                "bbox_iou": 0.0,
                "bbox_center_error_px": 0.0,
                "bbox_width_ratio": 1.0,
                "bbox_height_ratio": 1.0,
                "projection_valid_ratio": 1.0,
            }
            for index in range(len(candidates))
        ]

    obs_width = max(1e-6, float(obs_bbox[2] - obs_bbox[0]))
    obs_height = max(1e-6, float(obs_bbox[3] - obs_bbox[1]))
    obs_diag = float(np.hypot(obs_width, obs_height))
    rows: list[dict[str, Any]] = []
    for index, pose in enumerate(candidates):
        projected_bbox = None
        bbox_iou_value = 0.0
        center_error = float("inf")
        width_ratio = float("inf")
        height_ratio = float("inf")
        valid_ratio = 0.0
        reject_reason = ""
        keep = True
        bounds_penalty = _pose_bounds_penalty(pose, options)
        if bounds_penalty > 0.0:
            keep = False
            reject_reason = "pose_bounds"
            prefilter_loss = float("inf")
        else:
            points_C = transform_points(vertices, pose)
            uv, valid = project_points(points_C, camera_K)
            valid_ratio = float(np.count_nonzero(valid) / max(1, valid.size))
            projected_bbox = bbox_from_projected_points(uv, valid)
            if projected_bbox is None:
                keep = False
                reject_reason = "no_projected_bbox"
                prefilter_loss = float("inf")
            else:
                bbox_iou_value = _bbox_iou(projected_bbox, obs_bbox)
                center_error = _bbox_center_error(projected_bbox, obs_bbox)
                width = max(1e-6, float(projected_bbox[2] - projected_bbox[0]))
                height = max(1e-6, float(projected_bbox[3] - projected_bbox[1]))
                width_ratio = width / obs_width
                height_ratio = height / obs_height
                size_ok = (
                    width_ratio >= float(options.prefilter_size_ratio_min)
                    and width_ratio <= float(options.prefilter_size_ratio_max)
                    and height_ratio >= float(options.prefilter_size_ratio_min)
                    and height_ratio <= float(options.prefilter_size_ratio_max)
                )
                valid_ok = valid_ratio >= float(options.prefilter_min_valid_ratio)
                center_ok = center_error <= float(options.prefilter_center_factor) * obs_diag
                bbox_ok = bbox_iou_value >= float(options.prefilter_bbox_iou_min)
                keep = bool(
                    (not options.enable_initial_bbox_prefilter)
                    or (valid_ok and size_ok and (bbox_ok or center_ok))
                )
                if not keep:
                    if not valid_ok:
                        reject_reason = "low_projection_valid_ratio"
                    elif not size_ok:
                        reject_reason = "bbox_size_ratio"
                    else:
                        reject_reason = "bbox_iou_and_center"
                prefilter_loss = _bbox_prefilter_loss(
                    bbox_iou_value,
                    center_error,
                    obs_diag,
                    width_ratio,
                    height_ratio,
                )
        rows.append(
            {
                "index": int(index),
                "prefilter_keep": bool(keep),
                "prefilter_reject_reason": reject_reason,
                "prefilter_loss": float(prefilter_loss),
                "projected_bbox": projected_bbox,
                "bbox_iou": float(bbox_iou_value),
                "bbox_center_error_px": float(center_error),
                "bbox_width_ratio": float(width_ratio),
                "bbox_height_ratio": float(height_ratio),
                "projection_valid_ratio": float(valid_ratio),
            }
        )
    rows.sort(key=lambda row: (not bool(row["prefilter_keep"]), float(row["prefilter_loss"])))
    return rows


def _prefilter_vertices(mesh: RobotMesh, limit: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        return vertices.reshape(0, 3)
    if limit <= 0 or len(vertices) <= limit:
        return vertices
    sample_count = max(1, int(limit))
    sample_indices = np.unique(np.linspace(0, len(vertices) - 1, num=sample_count, dtype=np.int64))
    sampled = vertices[sample_indices]
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    corners = np.asarray(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
        ],
        dtype=np.float64,
    )
    return np.vstack([sampled, corners])


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _bbox_center_error(box_a: list[float], box_b: list[float]) -> float:
    ax = 0.5 * (float(box_a[0]) + float(box_a[2]))
    ay = 0.5 * (float(box_a[1]) + float(box_a[3]))
    bx = 0.5 * (float(box_b[0]) + float(box_b[2]))
    by = 0.5 * (float(box_b[1]) + float(box_b[3]))
    return float(np.hypot(ax - bx, ay - by))


def _bbox_prefilter_loss(
    bbox_iou_value: float,
    center_error_px: float,
    target_diag: float,
    width_ratio: float,
    height_ratio: float,
) -> float:
    diag = max(1e-6, float(target_diag))
    center_term = float(center_error_px) / diag if np.isfinite(center_error_px) else 1e6
    width_term = abs(float(np.log(max(1e-6, min(1e6, width_ratio)))))
    height_term = abs(float(np.log(max(1e-6, min(1e6, height_ratio)))))
    return float(center_term + 0.5 * (width_term + height_term) + (1.0 - float(bbox_iou_value)))


def _optimize_from_init(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    T0: np.ndarray,
    *,
    init_index: int,
    options: OptimizerOptions,
) -> RobotPoseResult:
    T_current = np.asarray(T0, dtype=np.float64)
    initial_render = render_robot_mask(mesh, camera_K, T_current, image_shape)
    initial_loss = _loss_for_pose(mesh, camera_K, observed_mask, image_shape, T_current, T0, options)
    history: list[dict[str, Any]] = [
        {
            "stage": -1,
            "init_index": init_index,
            "iteration": 0,
            "loss": initial_loss.total,
            **{f"term_{k}": v for k, v in initial_loss.terms.items()},
        }
    ]
    total_evals = 0
    for stage_index, scale in enumerate(options.stage_scales):
        scale_value = float(scale)
        K_s = scale_camera_K(camera_K, scale_value)
        obs_s = resize_mask_nearest(observed_mask, scale_value)
        shape_s = obs_s.shape

        def objective(delta: np.ndarray) -> float:
            nonlocal total_evals
            delta = _clamp_delta(delta, options)
            T = compose_delta_pose(T_current, delta)
            loss = _loss_for_pose(mesh, K_s, obs_s, shape_s, T, T0, options)
            total_evals += 1
            history.append(
                {
                    "stage": stage_index,
                    "init_index": init_index,
                    "iteration": total_evals,
                    "loss": loss.total,
                    **{f"term_{k}": v for k, v in loss.terms.items()},
                }
            )
            return loss.total

        if _uses_scipy_optimizer(options.optimizer_method):
            result = minimize(
                objective,
                np.zeros(6, dtype=np.float64),
                method=options.optimizer_method,
                options={"maxiter": int(options.max_iterations), "disp": False},
            )
            best_delta = _clamp_delta(np.asarray(result.x, dtype=np.float64), options)
            T_current = compose_delta_pose(T_current, best_delta)
        else:
            loss = _loss_for_pose(mesh, K_s, obs_s, shape_s, T_current, T0, options)
            history.append(
                {
                    "stage": stage_index,
                    "init_index": init_index,
                    "iteration": "pattern_start",
                    "loss": loss.total,
                    **{f"term_{k}": v for k, v in loss.terms.items()},
                }
            )
        T_current = _pattern_search_refine(
            mesh,
            K_s,
            obs_s,
            shape_s,
            T_current,
            reference_pose=T0,
            stage_index=stage_index,
            init_index=init_index,
            options=options,
            history=history,
        )
        T_current = _moment_refine(
            mesh,
            K_s,
            obs_s,
            shape_s,
            T_current,
            reference_pose=T0,
            stage_index=stage_index,
            init_index=init_index,
            options=options,
            history=history,
        )

    final_render = render_robot_mask(mesh, camera_K, T_current, image_shape)
    final_loss = _loss_for_pose(mesh, camera_K, observed_mask, image_shape, T_current, T0, options)
    history.append(
        {
            "stage": len(options.stage_scales),
            "init_index": init_index,
            "iteration": total_evals,
            "loss": final_loss.total,
            **{f"term_{k}": v for k, v in final_loss.terms.items()},
        }
    )
    return RobotPoseResult(
        T_C_B=T_current,
        loss=final_loss,
        initial_loss=initial_loss,
        rendered_mask=final_render.mask,
        selected_init_index=init_index,
        history=history,
        projected_bbox=final_render.projected_bbox,
    )


def _uses_scipy_optimizer(method: str) -> bool:
    return str(method).strip().lower() not in {"pattern", "local", "pattern_search", "none"}


def _pattern_search_refine(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    start_pose: np.ndarray,
    *,
    reference_pose: np.ndarray,
    stage_index: int,
    init_index: int,
    options: OptimizerOptions,
    history: list[dict[str, Any]],
) -> np.ndarray:
    current = np.asarray(start_pose, dtype=np.float64)
    current_loss = _loss_for_pose(mesh, camera_K, observed_mask, image_shape, current, reference_pose, options).total
    eval_index = len(history)
    for translation_step in options.pattern_translation_steps:
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    delta = np.zeros(6, dtype=np.float64)
                    delta[3 + axis] = sign * float(translation_step)
                    candidate = compose_delta_pose(current, delta)
                    loss = _loss_for_pose(mesh, camera_K, observed_mask, image_shape, candidate, reference_pose, options)
                    eval_index += 1
                    history.append(
                        {
                            "stage": stage_index,
                            "init_index": init_index,
                            "iteration": f"pattern_{eval_index}",
                            "loss": loss.total,
                            **{f"term_{k}": v for k, v in loss.terms.items()},
                        }
                    )
                    if loss.total + 1e-12 < current_loss:
                        current = candidate
                        current_loss = loss.total
                        improved = True
    for rotation_step_deg in options.pattern_rotation_steps_deg:
        step = np.deg2rad(float(rotation_step_deg))
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    delta = np.zeros(6, dtype=np.float64)
                    delta[axis] = sign * step
                    candidate = compose_delta_pose(current, delta)
                    loss = _loss_for_pose(mesh, camera_K, observed_mask, image_shape, candidate, reference_pose, options)
                    eval_index += 1
                    history.append(
                        {
                            "stage": stage_index,
                            "init_index": init_index,
                            "iteration": f"pattern_{eval_index}",
                            "loss": loss.total,
                            **{f"term_{k}": v for k, v in loss.terms.items()},
                        }
                    )
                    if loss.total + 1e-12 < current_loss:
                        current = candidate
                        current_loss = loss.total
                        improved = True
    return current


def _moment_refine(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    start_pose: np.ndarray,
    *,
    reference_pose: np.ndarray,
    stage_index: int,
    init_index: int,
    options: OptimizerOptions,
    history: list[dict[str, Any]],
) -> np.ndarray:
    current = np.asarray(start_pose, dtype=np.float64)
    current_loss = _loss_for_pose(mesh, camera_K, observed_mask, image_shape, current, reference_pose, options)
    obs_stats = _mask_moments(observed_mask)
    if obs_stats is None:
        return current
    eval_index = len(history)
    K = np.asarray(camera_K, dtype=np.float64)
    fx = max(1e-9, float(K[0, 0]))
    fy = max(1e-9, float(K[1, 1]))
    for _ in range(max(0, int(options.moment_refine_iterations))):
        rendered = render_robot_mask(mesh, K, current, image_shape)
        ren_stats = _mask_moments(rendered.mask)
        if ren_stats is None:
            break
        obs_cx, obs_cy, obs_area = obs_stats
        ren_cx, ren_cy, ren_area = ren_stats
        candidate = current.copy()
        z = max(1e-6, float(candidate[2, 3]))
        candidate[0, 3] += (obs_cx - ren_cx) * z / fx
        candidate[1, 3] += (obs_cy - ren_cy) * z / fy
        if obs_area > 0.0 and ren_area > 0.0:
            z_scale = float(np.sqrt(ren_area / obs_area))
            z_scale = float(np.clip(z_scale, 0.75, 1.35))
            candidate[2, 3] = max(1e-6, z * z_scale)
        loss = _loss_for_pose(mesh, K, observed_mask, image_shape, candidate, reference_pose, options)
        eval_index += 1
        history.append(
            {
                "stage": stage_index,
                "init_index": init_index,
                "iteration": f"moment_{eval_index}",
                "loss": loss.total,
                **{f"term_{k}": v for k, v in loss.terms.items()},
            }
        )
        if loss.total + 1e-12 < current_loss.total:
            current = candidate
            current_loss = loss
        else:
            break
    return current


def _mask_moments(mask: np.ndarray) -> tuple[float, float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(xs.size)


def _loss_for_pose(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    pose: np.ndarray,
    reference_pose: np.ndarray,
    options: OptimizerOptions,
) -> LossResult:
    bounds_penalty = _pose_bounds_penalty(pose, options)
    if bounds_penalty > 0.0:
        total = float(options.invalid_pose_loss) + bounds_penalty
        return LossResult(
            total=total,
            terms={"pose_bounds": bounds_penalty},
            weighted_terms={"pose_bounds": total},
            metrics={},
        )
    rendered = render_robot_mask(mesh, camera_K, pose, image_shape)
    return compute_partial_observation_loss(
        observed_mask,
        rendered.mask,
        pose,
        reference_pose,
        config=options.loss_config,
    )


def _pose_bounds_penalty(pose: np.ndarray, options: OptimizerOptions) -> float:
    z = float(np.asarray(pose, dtype=np.float64)[2, 3])
    penalty = 0.0
    if options.min_depth is not None:
        min_depth = float(options.min_depth)
        if z < min_depth:
            penalty += (min_depth - z) ** 2
    if options.max_depth is not None:
        max_depth = float(options.max_depth)
        if z > max_depth:
            penalty += (z - max_depth) ** 2
    return float(penalty)


def _clamp_delta(delta: np.ndarray, options: OptimizerOptions) -> np.ndarray:
    out = np.asarray(delta, dtype=np.float64).copy()
    if options.max_rotation_delta_deg is not None:
        max_angle = np.deg2rad(float(options.max_rotation_delta_deg))
        angle = float(np.linalg.norm(out[:3]))
        if angle > max_angle > 0.0:
            out[:3] *= max_angle / angle
    if options.max_translation_delta is not None:
        max_t = float(options.max_translation_delta)
        norm = float(np.linalg.norm(out[3:6]))
        if norm > max_t > 0.0:
            out[3:6] *= max_t / norm
    return out


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        out.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in history:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for row in history:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.12g}")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
