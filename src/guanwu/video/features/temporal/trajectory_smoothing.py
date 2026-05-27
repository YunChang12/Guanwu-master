from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


def smooth_object_trajectories(
    trajectories: dict[str, Any],
    *,
    max_frame_gap: int = 2,
    max_high_quality_adjust_m: float = 0.05,
    max_low_quality_adjust_m: float = 0.60,
    translation_spike_ratio: float = 4.0,
    min_translation_spike_m: float = 0.30,
    rotation_spike_deg: float = 45.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Smooth per-object vehicle trajectories without crossing track gaps.

    The function preserves the original schema: object values that contain a
    ``frames`` list remain dictionaries, while list-valued tracks remain lists.
    It only performs aggressive edits on isolated low-quality outliers and
    applies very small weighted smoothing to otherwise stable high-quality
    frames.
    """

    smoothed: dict[str, Any] = copy.deepcopy(trajectories or {})
    report: dict[str, Any] = {
        "schema": "guanwu.trajectory_smoothing.v1",
        "enabled": True,
        "objects": {},
        "params": {
            "max_frame_gap": int(max_frame_gap),
            "max_high_quality_adjust_m": float(max_high_quality_adjust_m),
            "max_low_quality_adjust_m": float(max_low_quality_adjust_m),
            "translation_spike_ratio": float(translation_spike_ratio),
            "min_translation_spike_m": float(min_translation_spike_m),
            "rotation_spike_deg": float(rotation_spike_deg),
        },
    }

    for obj_id, value in list(smoothed.items()):
        frames = _track_frames(value)
        obj_report = _empty_object_report()
        if frames is None:
            obj_report["reason"] = "missing_frames"
            report["objects"][obj_id] = obj_report
            continue

        valid_indices = [idx for idx, frame in enumerate(frames) if _valid_vec3(frame.get("centroid_world")) is not None]
        obj_report["input_frame_count"] = len(frames)
        obj_report["valid_frame_count"] = len(valid_indices)
        if len(valid_indices) < 3:
            obj_report["reason"] = "too_few_valid_frames"
            report["objects"][obj_id] = obj_report
            continue

        segments = _segments_from_indices(frames, valid_indices, max_frame_gap=max_frame_gap)
        obj_report["segment_count"] = len(segments)
        for segment in segments:
            if len(segment) < 3:
                continue
            _smooth_translation_segment(
                frames,
                segment,
                obj_report,
                max_high_quality_adjust_m=max_high_quality_adjust_m,
                max_low_quality_adjust_m=max_low_quality_adjust_m,
                translation_spike_ratio=translation_spike_ratio,
                min_translation_spike_m=min_translation_spike_m,
            )
            _smooth_rotation_segment(
                frames,
                segment,
                obj_report,
                max_low_quality_adjust_m=max_low_quality_adjust_m,
                rotation_spike_deg=rotation_spike_deg,
            )

        report["objects"][obj_id] = obj_report

    report["object_count"] = len(report["objects"])
    report["corrected_translation_outliers"] = int(
        sum(obj.get("corrected_translation_outliers", 0) for obj in report["objects"].values())
    )
    report["corrected_rotation_outliers"] = int(
        sum(obj.get("corrected_rotation_outliers", 0) for obj in report["objects"].values())
    )
    report["max_translation_adjust_m"] = float(
        max((obj.get("max_translation_adjust_m", 0.0) for obj in report["objects"].values()), default=0.0)
    )
    report["max_rotation_adjust_deg"] = float(
        max((obj.get("max_rotation_adjust_deg", 0.0) for obj in report["objects"].values()), default=0.0)
    )
    return smoothed, report


def _empty_object_report() -> dict[str, Any]:
    return {
        "input_frame_count": 0,
        "valid_frame_count": 0,
        "segment_count": 0,
        "corrected_translation_outliers": 0,
        "corrected_rotation_outliers": 0,
        "max_translation_adjust_m": 0.0,
        "max_rotation_adjust_deg": 0.0,
        "mean_translation_adjust_m": 0.0,
        "adjusted_frame_ids": [],
    }


def _track_frames(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        frames = value.get("frames")
    else:
        frames = value
    if not isinstance(frames, list):
        return None
    return [frame for frame in frames if isinstance(frame, dict)]


def _segments_from_indices(
    frames: list[dict[str, Any]],
    indices: list[int],
    *,
    max_frame_gap: int,
) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(indices, key=lambda idx: int(float(frames[idx].get("frame_id", 0) or 0)))
    segments: list[list[int]] = [[ordered[0]]]
    prev_fid = int(float(frames[ordered[0]].get("frame_id", 0) or 0))
    for idx in ordered[1:]:
        fid = int(float(frames[idx].get("frame_id", 0) or 0))
        if fid - prev_fid > int(max_frame_gap):
            segments.append([idx])
        else:
            segments[-1].append(idx)
        prev_fid = fid
    return segments


def _smooth_translation_segment(
    frames: list[dict[str, Any]],
    segment: list[int],
    report: dict[str, Any],
    *,
    max_high_quality_adjust_m: float,
    max_low_quality_adjust_m: float,
    translation_spike_ratio: float,
    min_translation_spike_m: float,
) -> None:
    pts = np.asarray([_valid_vec3(frames[idx].get("centroid_world")) for idx in segment], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return

    candidate = pts.copy()
    outlier_mask = _translation_outlier_mask(
        pts,
        [frames[idx] for idx in segment],
        translation_spike_ratio=translation_spike_ratio,
        min_translation_spike_m=min_translation_spike_m,
    )
    for local_idx in np.flatnonzero(outlier_mask):
        if local_idx <= 0 or local_idx >= len(segment) - 1:
            continue
        candidate[local_idx] = 0.5 * (pts[local_idx - 1] + pts[local_idx + 1])
        report["corrected_translation_outliers"] += 1

    smoothed = candidate.copy()
    if len(segment) >= 5:
        for local_idx in range(1, len(segment) - 1):
            if outlier_mask[local_idx]:
                continue
            local_mean = 0.25 * candidate[local_idx - 1] + 0.5 * candidate[local_idx] + 0.25 * candidate[local_idx + 1]
            smoothed[local_idx] = _limited_point_update(
                candidate[local_idx],
                local_mean,
                max_delta=max_high_quality_adjust_m,
            )

    adjustments: list[float] = []
    for local_idx, frame_idx in enumerate(segment):
        original = pts[local_idx]
        target = smoothed[local_idx]
        max_delta = _translation_adjust_limit(
            frames[frame_idx],
            outlier=bool(outlier_mask[local_idx]),
            max_high_quality_adjust_m=max_high_quality_adjust_m,
            max_low_quality_adjust_m=max_low_quality_adjust_m,
        )
        updated = _limited_point_update(original, target, max_delta=max_delta)
        adjust = float(np.linalg.norm(updated - original))
        if adjust <= 1e-9:
            continue
        frames[frame_idx]["centroid_world"] = [float(v) for v in updated.tolist()]
        _mark_frame_smoothing(frames[frame_idx], translation_adjust_m=adjust)
        _update_adjustment_report(report, frames[frame_idx], translation_adjust_m=adjust)
        adjustments.append(adjust)

    if adjustments:
        report["mean_translation_adjust_m"] = float(np.mean(adjustments))


def _translation_outlier_mask(
    pts: np.ndarray,
    frames: list[dict[str, Any]],
    *,
    translation_spike_ratio: float,
    min_translation_spike_m: float,
) -> np.ndarray:
    mask = np.zeros(len(pts), dtype=bool)
    if len(pts) < 3:
        return mask
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    positive = steps[steps > 1e-6]
    baseline_step = float(np.percentile(positive, 25)) if len(positive) else 0.0
    threshold = max(float(min_translation_spike_m), float(translation_spike_ratio) * max(baseline_step, 1e-6))
    for idx in range(1, len(pts) - 1):
        predicted = 0.5 * (pts[idx - 1] + pts[idx + 1])
        residual = float(np.linalg.norm(pts[idx] - predicted))
        step_before = float(np.linalg.norm(pts[idx] - pts[idx - 1]))
        step_after = float(np.linalg.norm(pts[idx + 1] - pts[idx]))
        neighbor_step = float(np.linalg.norm(pts[idx + 1] - pts[idx - 1]))
        is_spike = (
            residual >= threshold
            and step_before >= threshold
            and step_after >= threshold
            and neighbor_step <= max(threshold, baseline_step * 2.5 + 1e-6)
        )
        mask[idx] = bool(is_spike and _frame_quality_weight(frames[idx]) < 0.65)
    return mask


def _smooth_rotation_segment(
    frames: list[dict[str, Any]],
    segment: list[int],
    report: dict[str, Any],
    *,
    max_low_quality_adjust_m: float,
    rotation_spike_deg: float,
) -> None:
    quats: list[np.ndarray | None] = []
    for idx in segment:
        quat = _valid_quat_xyzw(frames[idx].get("orientation_quat"))
        quats.append(quat)
    if sum(q is not None for q in quats) < 3:
        return

    continuous = _continuous_quaternions(quats)
    _repair_terminal_rotation_outliers(
        frames,
        segment,
        continuous,
        report,
        rotation_spike_deg=rotation_spike_deg,
    )
    continuous = _continuous_quaternions(
        [_valid_quat_xyzw(frames[idx].get("orientation_quat")) for idx in segment]
    )
    for local_idx in range(1, len(segment) - 1):
        quat = continuous[local_idx]
        prev_quat = continuous[local_idx - 1]
        next_quat = continuous[local_idx + 1]
        if quat is None or prev_quat is None or next_quat is None:
            continue
        prev_angle = _quat_angle_deg(prev_quat, quat)
        next_angle = _quat_angle_deg(quat, next_quat)
        neighbor_angle = _quat_angle_deg(prev_quat, next_quat)
        if (
            prev_angle < rotation_spike_deg
            or next_angle < rotation_spike_deg
            or neighbor_angle > rotation_spike_deg
            or _frame_quality_weight(frames[segment[local_idx]]) >= 0.65
        ):
            continue
        replacement = _quat_slerp(prev_quat, next_quat, 0.5)
        adjust = _quat_angle_deg(quat, replacement)
        if adjust <= rotation_spike_deg:
            continue
        frame = frames[segment[local_idx]]
        frame["orientation_quat"] = [float(v) for v in replacement.tolist()]
        rot = _rotation_matrix_from_quat_xyzw(replacement)
        frame["rotation_matrix"] = [[float(v) for v in row] for row in rot.tolist()]
        _mark_frame_smoothing(frame, rotation_adjust_deg=adjust)
        _update_adjustment_report(report, frame, rotation_adjust_deg=adjust)
        report["corrected_rotation_outliers"] += 1


def _repair_terminal_rotation_outliers(
    frames: list[dict[str, Any]],
    segment: list[int],
    quats: list[np.ndarray | None],
    report: dict[str, Any],
    *,
    rotation_spike_deg: float,
) -> None:
    if len(segment) < 3:
        return
    endpoints = [
        (0, 1, 2),
        (len(segment) - 1, len(segment) - 2, len(segment) - 3),
    ]
    for endpoint_idx, neighbor_idx, stable_idx in endpoints:
        quat = quats[endpoint_idx]
        neighbor = quats[neighbor_idx]
        stable = quats[stable_idx]
        if quat is None or neighbor is None or stable is None:
            continue
        if _quat_angle_deg(neighbor, stable) > rotation_spike_deg * 0.5:
            continue
        if _quat_angle_deg(quat, neighbor) <= rotation_spike_deg:
            continue
        frame = frames[segment[endpoint_idx]]
        replacement = neighbor.copy()
        adjust = _quat_angle_deg(quat, replacement)
        frame["orientation_quat"] = [float(v) for v in replacement.tolist()]
        rot = _rotation_matrix_from_quat_xyzw(replacement)
        frame["rotation_matrix"] = [[float(v) for v in row] for row in rot.tolist()]
        _sync_transform_matrix(frame, rot)
        _mark_frame_smoothing(frame, rotation_adjust_deg=adjust)
        _update_adjustment_report(report, frame, rotation_adjust_deg=adjust)
        report["corrected_rotation_outliers"] += 1


def _translation_adjust_limit(
    frame: dict[str, Any],
    *,
    outlier: bool,
    max_high_quality_adjust_m: float,
    max_low_quality_adjust_m: float,
) -> float:
    if outlier:
        return float(max_low_quality_adjust_m)
    quality = _frame_quality_weight(frame)
    if quality >= 0.80:
        return float(max_high_quality_adjust_m)
    return float(max_high_quality_adjust_m + (1.0 - quality) * (max_low_quality_adjust_m - max_high_quality_adjust_m))


def _limited_point_update(original: np.ndarray, target: np.ndarray, *, max_delta: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(original, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-12 or dist <= float(max_delta):
        return np.asarray(target, dtype=np.float64)
    return np.asarray(original, dtype=np.float64) + delta * (float(max_delta) / dist)


def _mark_frame_smoothing(
    frame: dict[str, Any],
    *,
    translation_adjust_m: float | None = None,
    rotation_adjust_deg: float | None = None,
) -> None:
    data = frame.setdefault("trajectory_smoothing", {})
    data["applied"] = True
    if translation_adjust_m is not None:
        data["translation_adjust_m"] = float(max(float(data.get("translation_adjust_m", 0.0)), translation_adjust_m))
    if rotation_adjust_deg is not None:
        data["rotation_adjust_deg"] = float(max(float(data.get("rotation_adjust_deg", 0.0)), rotation_adjust_deg))


def _update_adjustment_report(
    report: dict[str, Any],
    frame: dict[str, Any],
    *,
    translation_adjust_m: float | None = None,
    rotation_adjust_deg: float | None = None,
) -> None:
    try:
        fid = int(float(frame.get("frame_id", 0) or 0))
    except Exception:
        fid = 0
    if fid and fid not in report["adjusted_frame_ids"]:
        report["adjusted_frame_ids"].append(fid)
        report["adjusted_frame_ids"].sort()
    if translation_adjust_m is not None:
        report["max_translation_adjust_m"] = float(
            max(float(report.get("max_translation_adjust_m", 0.0)), translation_adjust_m)
        )
    if rotation_adjust_deg is not None:
        report["max_rotation_adjust_deg"] = float(
            max(float(report.get("max_rotation_adjust_deg", 0.0)), rotation_adjust_deg)
        )


def _frame_quality_weight(frame: dict[str, Any]) -> float:
    metrics = _frame_metrics(frame)
    scores: list[float] = []
    mask = _bounded_metric(metrics.get("mask_iou"))
    visible_mask = _bounded_metric(metrics.get("visible_mask_iou"))
    bbox = _bounded_metric(metrics.get("bbox_iou"))
    visible_bbox = _bounded_metric(metrics.get("visible_bbox_iou"))
    if visible_mask is not None:
        scores.append(visible_mask)
    elif mask is not None:
        scores.append(mask)
    if visible_bbox is not None:
        scores.append(visible_bbox)
    elif bbox is not None:
        scores.append(bbox)
    center_error = _safe_float(metrics.get("bbox_center_error_px"))
    if center_error is not None:
        scores.append(max(0.0, min(1.0, 1.0 - center_error / 40.0)))
    confidence = _safe_float(frame.get("confidence"))
    if confidence is not None:
        scores.append(max(0.0, min(1.0, confidence)))
    if not scores:
        return 0.75
    return float(max(0.0, min(1.0, np.mean(scores))))


def _frame_metrics(frame: dict[str, Any]) -> dict[str, Any]:
    quality = frame.get("quality")
    if isinstance(quality, dict):
        metrics = quality.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    metrics = frame.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _bounded_metric(value: Any) -> float | None:
    v = _safe_float(value)
    if v is None:
        return None
    return max(0.0, min(1.0, v))


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _valid_vec3(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _valid_quat_xyzw(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(4)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return None
    return arr / norm


def _continuous_quaternions(quats: list[np.ndarray | None]) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    prev: np.ndarray | None = None
    for quat in quats:
        if quat is None:
            out.append(None)
            continue
        q = quat.copy()
        if prev is not None and float(np.dot(prev, q)) < 0.0:
            q = -q
        out.append(q)
        prev = q
    return out


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = abs(float(np.dot(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(2.0 * math.acos(dot)))


def _quat_slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    q0 = a / np.linalg.norm(a)
    q1 = b / np.linalg.norm(b)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = q0 + float(alpha) * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(dot)
    theta = theta_0 * float(alpha)
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = s0 * q0 + s1 * q1
    return out / np.linalg.norm(out)


def _rotation_matrix_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = [float(v) for v in quat]
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _sync_transform_matrix(frame: dict[str, Any], rotation: np.ndarray) -> None:
    transform = frame.get("T_world_from_object")
    if not isinstance(transform, list) or len(transform) < 4:
        return
    try:
        mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    except Exception:
        return
    mat[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    center = _valid_vec3(frame.get("centroid_world"))
    if center is not None:
        mat[:3, 3] = center
    frame["T_world_from_object"] = [[float(v) for v in row] for row in mat.tolist()]
