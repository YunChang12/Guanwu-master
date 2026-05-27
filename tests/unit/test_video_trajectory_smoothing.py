from __future__ import annotations

import math

import numpy as np

from guanwu.video.features.temporal.trajectory_smoothing import (
    smooth_object_trajectories,
)


def _yaw_quat(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) * 0.5
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _frame(
    frame_id: int,
    x: float,
    *,
    yaw_deg: float = 0.0,
    mask_iou: float = 0.92,
    bbox_iou: float = 0.90,
    center_error: float = 3.0,
) -> dict:
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    return {
        "frame_id": frame_id,
        "timestamp_sec": (frame_id - 1) / 30.0,
        "centroid_world": [x, 0.0, 8.0],
        "orientation_quat": _yaw_quat(yaw_deg),
        "rotation_matrix": [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        "scale": [2.0, 2.0, 2.0],
        "quality": {
            "metrics": {
                "mask_iou": mask_iou,
                "bbox_iou": bbox_iou,
                "bbox_center_error_px": center_error,
            }
        },
    }


def test_smoothing_repairs_isolated_low_quality_translation_spike() -> None:
    frames = [
        _frame(1, 0.00),
        _frame(2, 0.10),
        _frame(3, 0.75, mask_iou=0.42, bbox_iou=0.35, center_error=48.0),
        _frame(4, 0.30),
        _frame(5, 0.40),
    ]

    smoothed, report = smooth_object_trajectories({"obj_000001": {"frames": frames}})

    out = smoothed["obj_000001"]["frames"]
    assert np.isclose(out[2]["centroid_world"][0], np.float64(0.2))
    assert out[2]["trajectory_smoothing"]["translation_adjust_m"] > 0.5
    assert out[0]["centroid_world"] == frames[0]["centroid_world"]
    assert report["objects"]["obj_000001"]["corrected_translation_outliers"] == 1
    assert report["objects"]["obj_000001"]["max_translation_adjust_m"] > 0.5


def test_smoothing_limits_high_quality_frame_adjustments() -> None:
    frames = [
        _frame(1, 0.00),
        _frame(2, 0.18),
        _frame(3, 0.31),
        _frame(4, 0.45),
        _frame(5, 0.58),
        _frame(6, 0.74),
    ]

    smoothed, report = smooth_object_trajectories({"obj_000001": {"frames": frames}})

    out = smoothed["obj_000001"]["frames"]
    max_adjust = max(
        float(frame.get("trajectory_smoothing", {}).get("translation_adjust_m", 0.0))
        for frame in out
    )
    assert max_adjust <= 0.05
    assert report["objects"]["obj_000001"]["corrected_translation_outliers"] == 0


def test_smoothing_repairs_isolated_rotation_flip() -> None:
    frames = [
        _frame(1, 0.00, yaw_deg=0.0),
        _frame(2, 0.10, yaw_deg=2.0),
        _frame(3, 0.20, yaw_deg=150.0, mask_iou=0.45, bbox_iou=0.40, center_error=35.0),
        _frame(4, 0.30, yaw_deg=4.0),
        _frame(5, 0.40, yaw_deg=5.0),
    ]

    smoothed, report = smooth_object_trajectories({"obj_000001": {"frames": frames}})

    out = smoothed["obj_000001"]["frames"]
    quat = np.asarray(out[2]["orientation_quat"], dtype=np.float64)
    expected = np.asarray(_yaw_quat(3.0), dtype=np.float64)
    assert abs(float(np.dot(quat, expected))) > 0.999
    assert out[2]["trajectory_smoothing"]["rotation_adjust_deg"] > 100.0
    assert report["objects"]["obj_000001"]["corrected_rotation_outliers"] == 1


def test_smoothing_repairs_terminal_rotation_flip() -> None:
    frames = [
        _frame(73, 0.00, yaw_deg=5.0),
        _frame(74, 0.10, yaw_deg=4.0),
        _frame(75, 0.20, yaw_deg=3.0),
        _frame(76, 0.30, yaw_deg=4.0),
        _frame(77, 0.40, yaw_deg=5.0),
        _frame(78, 0.50, yaw_deg=4.5),
        _frame(79, 0.60, yaw_deg=-176.0),
    ]

    smoothed, report = smooth_object_trajectories({"obj_000013": {"frames": frames}})

    out = smoothed["obj_000013"]["frames"]
    quat = np.asarray(out[-1]["orientation_quat"], dtype=np.float64)
    expected = np.asarray(_yaw_quat(4.5), dtype=np.float64)
    assert abs(float(np.dot(quat, expected))) > 0.999
    assert out[-1]["trajectory_smoothing"]["rotation_adjust_deg"] > 170.0
    assert report["objects"]["obj_000013"]["corrected_rotation_outliers"] == 1


def test_smoothing_does_not_cross_track_gaps() -> None:
    frames = [
        _frame(1, 0.00),
        _frame(2, 0.10),
        _frame(10, 3.00, mask_iou=0.40, bbox_iou=0.40, center_error=50.0),
        _frame(11, 3.10),
    ]

    smoothed, report = smooth_object_trajectories({"obj_000001": {"frames": frames}}, max_frame_gap=2)

    out = smoothed["obj_000001"]["frames"]
    assert out[2]["centroid_world"] == frames[2]["centroid_world"]
    assert report["objects"]["obj_000001"]["segment_count"] == 2
    assert report["objects"]["obj_000001"]["corrected_translation_outliers"] == 0
