from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .losses import LossConfig, LossResult, compute_partial_observation_loss
from .render import RobotMesh, bbox_from_mask, render_robot_mask, scale_camera_K


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
    coarse_search: bool = True
    pattern_translation_steps: tuple[float, ...] = (0.25, 0.12, 0.06, 0.03, 0.015)
    pattern_rotation_steps_deg: tuple[float, ...] = (8.0, 4.0, 2.0)
    moment_refine_iterations: int = 4


@dataclass(frozen=True)
class RobotPoseResult:
    T_C_B: np.ndarray
    loss: LossResult
    initial_loss: LossResult
    rendered_mask: np.ndarray
    selected_init_index: int
    history: list[dict[str, Any]]
    projected_bbox: list[float] | None

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

    scored_candidates = _score_initial_candidates(mesh, K, obs, image_shape, candidates, opts)
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
    )


def _score_initial_candidates(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    observed_mask: np.ndarray,
    image_shape: tuple[int, int],
    candidates: list[np.ndarray],
    options: OptimizerOptions,
) -> list[tuple[int, float, np.ndarray]]:
    scale = min(1.0, float(options.stage_scales[0]) if options.stage_scales else 1.0)
    K_s = scale_camera_K(camera_K, scale)
    obs_s = resize_mask_nearest(observed_mask, scale)
    shape_s = obs_s.shape
    scored: list[tuple[int, float, np.ndarray]] = []
    for index, T0 in enumerate(candidates):
        rendered = render_robot_mask(mesh, K_s, T0, shape_s)
        loss = compute_partial_observation_loss(obs_s, rendered.mask, T0, T0, config=options.loss_config)
        scored.append((index, loss.total, T0))
    scored.sort(key=lambda item: item[1])
    return scored


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
    initial_loss = compute_partial_observation_loss(
        observed_mask,
        initial_render.mask,
        T_current,
        T0,
        config=options.loss_config,
    )
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
            rendered = render_robot_mask(mesh, K_s, T, shape_s)
            loss = compute_partial_observation_loss(obs_s, rendered.mask, T, T0, config=options.loss_config)
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

        result = minimize(
            objective,
            np.zeros(6, dtype=np.float64),
            method=options.optimizer_method,
            options={"maxiter": int(options.max_iterations), "disp": False},
        )
        best_delta = _clamp_delta(np.asarray(result.x, dtype=np.float64), options)
        T_current = compose_delta_pose(T_current, best_delta)
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
    final_loss = compute_partial_observation_loss(
        observed_mask,
        final_render.mask,
        T_current,
        T0,
        config=options.loss_config,
    )
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
    rendered = render_robot_mask(mesh, camera_K, pose, image_shape)
    return compute_partial_observation_loss(
        observed_mask,
        rendered.mask,
        pose,
        reference_pose,
        config=options.loss_config,
    )


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
