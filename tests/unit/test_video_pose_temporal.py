from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

import guanwu.video.project.executor as project_executor
from guanwu.video.project.executor import ProjectExecutor
from process.pose_optimizer.strategies.temporal_fast import (
    classify_truncation_observability,
    compute_visible_bbox_score,
    compute_truncated_visual_quality_gate,
    compute_road_and_heading_score,
    compute_temporal_anchor_visual_gate,
    compute_trusted_anchor_gate,
    choose_best_refined_result,
    detect_truncation,
    load_prior_pose_payload,
    passes_non_truncated_visual_ground_rescue_replacement_gate,
    should_skip_disk_temporal_prior,
    select_pareto_refine_candidates,
    should_run_non_truncated_visual_ground_rescue,
)
from process.pose_optimizer.strategies.fast import build_vehicle_pose_context
from process.pose_optimizer.strategies.fast import find_depth_map_for_task
from process.pose_optimizer.strategies.fast import world_up_vector_from_arg


def _pose_record(
    *,
    frame_id: int,
    x: float = 0.0,
    yaw_deg: float = 0.0,
    scale: float = 1.0,
    score: float = 0.8,
    mask_iou: float = 0.8,
    bbox_iou: float = 0.8,
    status: str = "accepted",
) -> dict:
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    rotation = [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ]
    return {
        "frame_id": frame_id,
        "status": status,
        "pose": {
            "translation_world": [x, 0.0, 8.0],
            "rotation_matrix": rotation,
            "scale": [scale, scale, scale],
        },
        "metrics": {
            "score": score,
            "mask_iou": mask_iou,
            "bbox_iou": bbox_iou,
            "bbox_center_error_px": 8.0,
            "ground_contact_max_abs_m": 0.04,
            "upright_angle_error_deg": 2.0,
        },
    }


def test_pose_track_scale_prior_ignores_seed_track_and_outliers() -> None:
    records = [
        {"frame_id": 1, "status": "seed", "scale": [0.1, 0.1, 0.1]},
        {"frame_id": 2, "status": "seed", "scale": [9.0, 9.0, 9.0]},
        _pose_record(frame_id=3, scale=1.0, mask_iou=0.82, bbox_iou=0.85),
        _pose_record(frame_id=4, scale=1.1, mask_iou=0.78, bbox_iou=0.83),
    ]

    prior = ProjectExecutor._pose_track_scale_prior(
        records,
        source="accepted_track_median_scale",
        require_high_quality=True,
        max_frame_id=4,
    )

    assert prior is not None
    assert prior["sample_count"] == 2
    assert np.allclose(prior["scale"], [1.05, 1.05, 1.05])
    assert prior["frame_ids"] == [3, 4]


def test_pose_track_scale_prior_excludes_low_observability_severe_truncation() -> None:
    anchor_a = _pose_record(frame_id=1, scale=1.0, mask_iou=0.88, bbox_iou=0.88)
    anchor_b = _pose_record(frame_id=2, scale=1.1, mask_iou=0.86, bbox_iou=0.86)
    severe = _pose_record(frame_id=3, scale=2.5, mask_iou=0.82, bbox_iou=0.96)
    severe["metrics"].update(
        {
            "truncation_severity": "severe",
            "low_observability": True,
            "visible_mask_iou": 0.82,
            "visible_bbox_iou": 0.96,
            "visible_contour_mean_distance_px": 8.5,
        }
    )

    prior = ProjectExecutor._pose_track_scale_prior(
        [anchor_a, anchor_b, severe],
        source="accepted_track_median_scale",
        require_high_quality=True,
        max_frame_id=3,
    )

    assert prior is not None
    assert prior["sample_count"] == 2
    assert np.allclose(prior["scale"], [1.05, 1.05, 1.05])
    assert prior["frame_ids"] == [1, 2]


def test_pose_temporal_anchor_excludes_low_observability_severe_truncation() -> None:
    severe = _pose_record(frame_id=3, scale=2.5, mask_iou=0.82, bbox_iou=0.96)
    severe["metrics"].update(
        {
            "truncation_severity": "severe",
            "low_observability": True,
            "visible_mask_iou": 0.82,
            "visible_bbox_iou": 0.96,
            "visible_contour_mean_distance_px": 8.5,
        }
    )
    light = _pose_record(frame_id=4, scale=1.05, mask_iou=0.90, bbox_iou=0.89)
    light["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.90,
            "visible_bbox_iou": 0.89,
            "visible_contour_mean_distance_px": 4.0,
        }
    )

    assert ProjectExecutor._pose_record_updates_temporal_anchor(severe) is False
    assert ProjectExecutor._pose_record_updates_temporal_anchor(light) is True


def test_anchor_temporal_gate_rejects_visual_degradation_yaw_jump() -> None:
    anchor = _pose_record(frame_id=10, yaw_deg=-2.4, mask_iou=0.926, bbox_iou=0.923)
    degraded = _pose_record(frame_id=11, yaw_deg=8.4, mask_iou=0.809, bbox_iou=0.776)
    degraded["metrics"].update(
        {
            "heading_prior_angle_error_deg": 13.8,
            "heading_front_sign_enabled": True,
            "heading_front_sign_confidence": 1.0,
        }
    )

    decision = ProjectExecutor._pose_anchor_temporal_gate(degraded, anchor)

    assert decision["accepted"] is False
    assert decision["reason"].startswith("anchor_visual_degradation_yaw_jump")
    assert decision["yaw_jump_deg"] > 10.0
    assert decision["mask_drop"] > 0.10
    assert decision["bbox_drop"] > 0.14


def test_anchor_temporal_gate_rejects_low_observability_heading_and_scale_jump() -> None:
    anchor = _pose_record(frame_id=15, yaw_deg=4.0, scale=1.0, mask_iou=0.90, bbox_iou=0.88)
    bad = _pose_record(frame_id=16, yaw_deg=88.0, scale=1.2, mask_iou=0.84, bbox_iou=0.95)
    bad["metrics"].update(
        {
            "truncation_severity": "moderate",
            "low_observability": True,
            "visible_target_fraction": 0.22,
            "visible_mask_iou": 0.82,
            "visible_bbox_iou": 0.88,
            "heading_prior_angle_error_deg": 108.0,
        }
    )

    decision = ProjectExecutor._pose_anchor_temporal_gate(bad, anchor)

    assert decision["accepted"] is False
    assert decision["reason"].startswith("anchor_low_observability")


def test_anchor_temporal_gate_accepts_stable_visual_pose() -> None:
    anchor = _pose_record(frame_id=10, yaw_deg=-2.4, scale=1.0, mask_iou=0.926, bbox_iou=0.923)
    stable = _pose_record(frame_id=11, yaw_deg=-1.0, scale=1.01, mask_iou=0.90, bbox_iou=0.90)
    stable["metrics"].update({"heading_prior_angle_error_deg": 28.0})

    decision = ProjectExecutor._pose_anchor_temporal_gate(stable, anchor)

    assert decision["accepted"] is True
    assert decision["reason"] == "accepted"


def test_anchor_temporal_gate_filter_keeps_prior_track_and_skips_remaining_frames() -> None:
    frame_records: dict[str, dict] = {}
    selected = [
        _pose_record(frame_id=10, yaw_deg=-2.4, mask_iou=0.926, bbox_iou=0.923),
        _pose_record(frame_id=11, yaw_deg=8.4, mask_iou=0.809, bbox_iou=0.776),
        _pose_record(frame_id=12, yaw_deg=-3.3, mask_iou=0.906, bbox_iou=0.916),
    ]

    kept, summary = ProjectExecutor._apply_anchor_temporal_gate_to_selected_records(
        selected,
        frame_ids=[10, 11, 12],
        frame_records=frame_records,
    )

    assert [record["frame_id"] for record in kept] == [10]
    assert summary["failed_frame_id"] == 11
    assert summary["accepted_frame_count_before_failure"] == 1
    assert frame_records["frame_000011"]["status"] == "rejected"
    assert frame_records["frame_000011"]["reason"].startswith("anchor_visual_degradation_yaw_jump")
    assert frame_records["frame_000012"]["status"] == "skipped"
    assert frame_records["frame_000012"]["reason"] == "skipped_after_anchor_temporal_gate_failure"


def test_stable_temporal_anchor_allows_consecutive_handoffs() -> None:
    first_anchor = _pose_record(frame_id=12, yaw_deg=-177.0, scale=2.67, mask_iou=0.923, bbox_iou=0.911)
    first_anchor["metrics"].update({"ground_contact_max_abs_m": 0.11})

    stable_followup = _pose_record(frame_id=13, yaw_deg=-173.8, scale=2.69, mask_iou=0.832, bbox_iou=0.792)
    stable_followup["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.832,
            "visible_bbox_iou": 0.792,
            "visible_contour_mean_distance_px": 5.2,
            "visible_profile_mean_distance_px": 6.1,
            "ground_contact_max_abs_m": 0.17,
            "heading_prior_angle_error_deg": 16.0,
        }
    )

    next_followup = _pose_record(frame_id=14, yaw_deg=-171.9, scale=2.70, mask_iou=0.803, bbox_iou=0.751)
    next_followup["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.803,
            "visible_bbox_iou": 0.751,
            "visible_contour_mean_distance_px": 5.7,
            "visible_profile_mean_distance_px": 6.6,
            "ground_contact_max_abs_m": 0.18,
            "heading_prior_angle_error_deg": 11.0,
        }
    )

    handoff_13 = ProjectExecutor._pose_anchor_temporal_gate(stable_followup, first_anchor)
    stable_followup["anchor_temporal_gate"] = handoff_13
    assert handoff_13["accepted"] is True
    assert ProjectExecutor._pose_record_is_stable_temporal_anchor(stable_followup) is True
    assert ProjectExecutor._pose_record_updates_temporal_anchor(stable_followup) is False
    assert ProjectExecutor._pose_record_promotes_temporal_candidate(
        stable_followup,
        stable_streak_count=1,
    ) is False

    stable_followup["temporal_anchor_kind"] = ProjectExecutor._pose_temporal_anchor_kind(stable_followup)
    handoff_14 = ProjectExecutor._pose_anchor_temporal_gate(next_followup, stable_followup)

    assert handoff_14["accepted"] is True
    assert handoff_14["anchor_kind"] == "stable"
    assert handoff_14["yaw_jump_deg"] < 6.0
    assert handoff_14["scale_ratio"] < 1.02
    next_followup["anchor_temporal_gate"] = handoff_14
    assert ProjectExecutor._pose_record_promotes_temporal_candidate(
        next_followup,
        stable_streak_count=2,
    ) is True


def test_anchor_temporal_gate_rejects_obj18_frame14_quality_drift() -> None:
    anchor = _pose_record(frame_id=13, yaw_deg=-1.29, scale=2.63, mask_iou=0.916, bbox_iou=0.878)
    drifted = _pose_record(frame_id=14, yaw_deg=6.52, scale=2.61, mask_iou=0.821, bbox_iou=0.773)
    drifted["metrics"].update(
        {
            "heading_front_sign_enabled": True,
            "heading_front_sign_confidence": 1.0,
            "heading_prior_angle_error_deg": 15.7,
            "ground_contact_max_abs_m": 0.085,
        }
    )

    decision = ProjectExecutor._pose_anchor_temporal_gate(drifted, anchor)

    assert decision["accepted"] is False
    assert decision["reason"].startswith("anchor_visual_degradation")
    assert decision["mask_drop"] > 0.09
    assert decision["bbox_drop"] > 0.10


def test_pose_record_does_not_promote_obj18_frame14_as_stable_anchor() -> None:
    drifted = _pose_record(frame_id=14, yaw_deg=6.52, scale=2.61, mask_iou=0.821, bbox_iou=0.773)
    drifted["metrics"].update(
        {
            "heading_front_sign_enabled": True,
            "heading_front_sign_confidence": 1.0,
            "heading_prior_angle_error_deg": 15.7,
            "ground_contact_max_abs_m": 0.085,
            "mask_drop": 0.095,
            "bbox_drop": 0.105,
        }
    )
    drifted["anchor_temporal_gate"] = {
        "accepted": True,
        "yaw_jump_deg": 7.8,
        "scale_ratio": 1.006,
        "mask_drop": 0.095,
        "bbox_drop": 0.105,
    }

    assert ProjectExecutor._pose_record_is_stable_temporal_anchor(drifted) is False


def test_pose_record_does_not_make_obj18_frame12_quality_drop_stable_anchor() -> None:
    degraded = _pose_record(frame_id=12, yaw_deg=4.58, scale=2.63, mask_iou=0.8498, bbox_iou=0.8085)
    degraded["metrics"].update(
        {
            "heading_front_sign_enabled": True,
            "heading_front_sign_confidence": 1.0,
            "heading_prior_angle_error_deg": 21.4,
            "ground_contact_max_abs_m": 0.080,
        }
    )
    degraded["anchor_temporal_gate"] = {
        "accepted": True,
        "yaw_jump_deg": 7.79,
        "scale_ratio": 1.010,
        "mask_drop": 0.0917,
        "bbox_drop": 0.1228,
        "combined_visual_drop": 0.2145,
        "anchor_kind": "high_quality",
        "anchor_frame_id": 11,
        "anchor_mask_iou": 0.9416,
        "anchor_bbox_iou": 0.9313,
    }

    assert ProjectExecutor._pose_record_is_stable_temporal_anchor(degraded) is False
    assert ProjectExecutor._pose_record_promotes_temporal_candidate(
        degraded,
        stable_streak_count=2,
    ) is False


def test_anchor_temporal_gate_rejects_obj18_frame12_quality_drift() -> None:
    anchor = _pose_record(frame_id=11, yaw_deg=-3.17, scale=2.6566, mask_iou=0.9416, bbox_iou=0.9313)
    drifted = _pose_record(frame_id=12, yaw_deg=4.58, scale=2.6302, mask_iou=0.8498, bbox_iou=0.8085)
    drifted["metrics"].update(
        {
            "heading_front_sign_enabled": True,
            "heading_front_sign_confidence": 1.0,
            "heading_prior_angle_error_deg": 21.39,
            "ground_contact_max_abs_m": 0.0802,
        }
    )

    decision = ProjectExecutor._pose_anchor_temporal_gate(drifted, anchor)

    assert decision["accepted"] is False
    assert decision["reason"].startswith("anchor_visual_degradation")
    assert decision["mask_drop"] > 0.08
    assert decision["bbox_drop"] > 0.12


def test_trusted_anchor_gate_rejects_obj18_frame12_quality_drift() -> None:
    args = type(
        "Args",
        (),
        {
            "trusted_anchor_gate_enabled": True,
            "trusted_anchor_prior_mask_iou_min": 0.88,
            "trusted_anchor_prior_bbox_iou_min": 0.85,
            "trusted_anchor_visual_drop_max": 0.18,
            "trusted_anchor_mask_drop_max": 0.08,
            "trusted_anchor_bbox_drop_max": 0.10,
            "trusted_anchor_yaw_jump_deg": 6.0,
            "trusted_anchor_scale_ratio": 1.04,
            "trusted_anchor_min_mask_iou": 0.86,
            "trusted_anchor_min_bbox_iou": 0.82,
            "trusted_anchor_penalty": 1_000_000.0,
        },
    )()

    gate = compute_trusted_anchor_gate(
        result={
            "mask_iou": 0.8498,
            "bbox_iou": 0.8085,
            "yaw_jump_from_trusted_anchor_deg": 7.79,
            "scale_ratio_from_trusted_anchor": 1.01,
        },
        trusted_anchor={
            "quality": {
                "mask_iou": 0.9416,
                "bbox_iou": 0.9313,
            }
        },
        args=args,
    )

    assert gate["trusted_anchor_gate_passed"] is False
    assert gate["trusted_anchor_gate_reason"].startswith("visual_degradation")
    assert gate["trusted_anchor_mask_drop"] > 0.08
    assert gate["trusted_anchor_bbox_drop"] > 0.12


def test_trusted_anchor_payload_converts_world_pose_to_camera_prior() -> None:
    prior = load_prior_pose_payload(
        {
            "source": "trusted_temporal_anchor_pose",
            "frame_id": 11,
            "pose": {
                "translation_world": [1.0, 2.0, 3.0],
                "rotation_matrix": np.eye(3).tolist(),
                "scale": [2.0, 2.0, 2.0],
            },
            "quality": {"mask_iou": 0.94, "bbox_iou": 0.93},
        },
        np.eye(4),
    )

    assert prior is not None
    assert prior["pose_source"] == "trusted_temporal_anchor_pose"
    assert prior["frame_idx"] == 11
    assert np.allclose(prior["translation_cam"], [1.0, 2.0, 3.0])
    assert np.allclose(prior["scale"], [2.0, 2.0, 2.0])
    assert prior["quality"]["mask_iou"] == 0.94


def test_truncated_fail_fast_triggers_for_low_observability_failure() -> None:
    record = {
        "status": "failed",
        "reason": "optimizer_failed",
        "metrics": {
            "low_observability": True,
            "truncation_severity": "severe",
        },
    }

    decision = ProjectExecutor._pose_truncated_object_fail_fast_decision(
        record,
        inst={"bbox_xyxy": [100.0, 240.0, 220.0, 360.0], "image_width": 640, "image_height": 360},
        frame_id=9,
    )

    assert decision["skip_object"] is True
    assert decision["frame_id"] == 9
    assert decision["reason"] == "optimizer_failed"
    assert decision["truncation_severity"] == "severe"


def test_truncated_fail_fast_ignores_non_truncated_failure() -> None:
    record = {
        "status": "failed",
        "reason": "optimizer_failed",
        "metrics": {
            "low_observability": False,
            "truncation_severity": "normal",
        },
    }

    decision = ProjectExecutor._pose_truncated_object_fail_fast_decision(
        record,
        inst={"bbox_xyxy": [100.0, 80.0, 220.0, 180.0], "image_width": 640, "image_height": 360},
        frame_id=3,
    )

    assert decision["skip_object"] is False


def test_truncated_fail_fast_keeps_object_when_prior_frames_were_accepted() -> None:
    fail_fast = {
        "skip_object": True,
        "reason": "optimizer_failed",
        "frame_id": 10,
        "truncation_severity": "severe",
        "low_observability": True,
    }
    accepted_records = [_pose_record(frame_id=1), _pose_record(frame_id=2)]
    frame_ids = [1, 2, 10, 11, 12]
    frame_records = {
        "frame_000001": accepted_records[0],
        "frame_000002": accepted_records[1],
        "frame_000010": {"status": "failed", "reason": "optimizer_failed"},
    }

    summary = ProjectExecutor._apply_truncated_object_fail_fast(
        fail_fast,
        frame_ids=frame_ids,
        frame_records=frame_records,
        accepted_records=accepted_records,
    )

    assert summary["skip_entire_object"] is False
    assert summary["remaining_frame_count"] == 2
    assert summary["accepted_frame_count_before_failure"] == 2
    assert frame_records["frame_000011"]["status"] == "skipped"
    assert frame_records["frame_000012"]["failed_frame_id"] == 10


def test_truncated_fail_fast_keeps_all_frames_object_with_accepted_frame_records() -> None:
    fail_fast = {
        "skip_object": True,
        "reason": "optimizer_failed",
        "frame_id": 10,
        "truncation_severity": "severe",
        "low_observability": True,
    }
    frame_records = {
        "frame_000001": {"status": "accepted", "reason": "accepted"},
        "frame_000002": {"status": "accepted", "reason": "accepted"},
        "frame_000010": {"status": "failed", "reason": "optimizer_failed"},
    }

    summary = ProjectExecutor._apply_truncated_object_fail_fast(
        fail_fast,
        frame_ids=[1, 2, 10, 11],
        frame_records=frame_records,
        accepted_records=[],
    )

    assert summary["skip_entire_object"] is False
    assert summary["accepted_frame_count_before_failure"] == 2


def test_usd_visibility_segments_split_contiguous_trajectory_frames() -> None:
    assert ProjectExecutor._trajectory_visibility_segments([6, 7, 8, 10, 11, 15]) == [
        (6, 8),
        (10, 11),
        (15, 15),
    ]


def test_apply_usd_visibility_samples_shows_track_from_first_frame() -> None:
    pytest.importorskip("pxr", reason="usd-core required for USD visibility checks")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Objects/obj_000001").GetPrim()

    ProjectExecutor._apply_trajectory_visibility_samples(
        UsdGeom.Imageable(prim),
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        stage_start_frame=1,
        stage_end_frame=12,
    )

    imageable = UsdGeom.Imageable(prim)
    attr = imageable.GetVisibilityAttr()
    assert attr.GetTimeSamples() == [1.0, 10.0]
    assert imageable.ComputeVisibility(Usd.TimeCode(1.0)) == UsdGeom.Tokens.inherited
    assert imageable.ComputeVisibility(Usd.TimeCode(9.0)) == UsdGeom.Tokens.inherited
    assert imageable.ComputeVisibility(Usd.TimeCode(10.0)) == UsdGeom.Tokens.invisible


def test_apply_usd_visibility_samples_hides_before_between_and_after_segments() -> None:
    pytest.importorskip("pxr", reason="usd-core required for USD visibility checks")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Objects/obj_000009").GetPrim()

    ProjectExecutor._apply_trajectory_visibility_samples(
        UsdGeom.Imageable(prim),
        [6, 7, 8, 61, 62, 63],
        stage_start_frame=1,
        stage_end_frame=79,
    )

    imageable = UsdGeom.Imageable(prim)
    attr = imageable.GetVisibilityAttr()
    assert attr.GetTimeSamples() == [1.0, 6.0, 9.0, 61.0, 64.0]
    assert imageable.ComputeVisibility(Usd.TimeCode(1.0)) == UsdGeom.Tokens.invisible
    assert imageable.ComputeVisibility(Usd.TimeCode(6.0)) == UsdGeom.Tokens.inherited
    assert imageable.ComputeVisibility(Usd.TimeCode(8.0)) == UsdGeom.Tokens.inherited
    assert imageable.ComputeVisibility(Usd.TimeCode(9.0)) == UsdGeom.Tokens.invisible
    assert imageable.ComputeVisibility(Usd.TimeCode(60.0)) == UsdGeom.Tokens.invisible
    assert imageable.ComputeVisibility(Usd.TimeCode(61.0)) == UsdGeom.Tokens.inherited
    assert imageable.ComputeVisibility(Usd.TimeCode(64.0)) == UsdGeom.Tokens.invisible


def test_temporal_candidate_trajectory_prefers_smooth_pose_over_visual_outlier() -> None:
    frame_candidates = {
        1: [_pose_record(frame_id=1, x=0.0, yaw_deg=0.0, score=0.80)],
        2: [
            _pose_record(frame_id=2, x=0.1, yaw_deg=180.0, score=0.95),
            _pose_record(frame_id=2, x=0.15, yaw_deg=4.0, score=0.86),
        ],
    }

    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        frame_candidates,
        target_frame_id=2,
    )

    assert summary["selected_frame_count"] == 2
    assert selected[1]["frame_id"] == 2
    assert selected[1]["metrics"]["score"] == 0.86
    assert selected[1]["trajectory_selection"]["candidate_index"] == 1


def test_temporal_candidate_trajectory_prefers_front_sign_consistent_candidate() -> None:
    visually_better_reversed = _pose_record(
        frame_id=5,
        score=2.30,
        mask_iou=0.89,
        bbox_iou=0.88,
    )
    visually_better_reversed["metrics"].update(
        {
            "heading_front_sign_enabled": True,
            "heading_prior_angle_error_deg": 172.0,
            "heading_front_sign_confidence": 1.0,
            "heading_prior_score": 0.0,
        }
    )
    front_sign_consistent = _pose_record(
        frame_id=5,
        score=2.28,
        mask_iou=0.83,
        bbox_iou=0.75,
    )
    front_sign_consistent["metrics"].update(
        {
            "heading_front_sign_enabled": True,
            "heading_prior_angle_error_deg": 6.0,
            "heading_front_sign_confidence": 1.0,
            "heading_prior_score": 0.94,
        }
    )

    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        {5: [visually_better_reversed, front_sign_consistent]},
        target_frame_id=5,
    )

    assert summary["selected_frame_count"] == 1
    assert selected[0]["metrics"]["heading_prior_angle_error_deg"] == 6.0
    assert selected[0]["trajectory_selection"]["candidate_index"] == 1


def test_temporal_candidate_trajectory_downweights_bbox_for_severe_truncation() -> None:
    anchor = _pose_record(frame_id=6, x=0.0, yaw_deg=0.0, score=0.90, mask_iou=0.88, bbox_iou=0.88)
    bbox_only = _pose_record(frame_id=7, x=0.1, yaw_deg=0.0, score=1.65, mask_iou=0.72, bbox_iou=0.98)
    bbox_only["metrics"].update(
        {
            "truncation_severity": "severe",
            "low_observability": True,
            "visible_mask_iou": 0.72,
            "visible_bbox_iou": 0.98,
            "visible_contour_score": 0.11,
            "visible_contour_mean_distance_px": 8.5,
            "visible_profile_mean_distance_px": 11.0,
        }
    )
    contour_consistent = _pose_record(frame_id=7, x=0.12, yaw_deg=3.0, score=1.30, mask_iou=0.78, bbox_iou=0.86)
    contour_consistent["metrics"].update(
        {
            "truncation_severity": "severe",
            "low_observability": True,
            "visible_mask_iou": 0.78,
            "visible_bbox_iou": 0.86,
            "visible_contour_score": 0.42,
            "visible_contour_mean_distance_px": 4.5,
            "visible_profile_mean_distance_px": 7.5,
        }
    )

    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        {6: [anchor], 7: [bbox_only, contour_consistent]},
        target_frame_id=7,
    )

    assert summary["selected_frame_count"] == 2
    assert selected[1]["metrics"]["visible_contour_mean_distance_px"] == 4.5
    assert selected[1]["trajectory_selection"]["candidate_index"] == 1


def test_temporal_candidate_trajectory_prefers_visible_contour_for_bottom_truncation() -> None:
    anchor = _pose_record(frame_id=6, x=0.0, yaw_deg=0.0, score=0.90, mask_iou=0.88, bbox_iou=0.88)
    bbox_center_fit = _pose_record(frame_id=7, x=0.1, yaw_deg=0.0, score=1.70, mask_iou=0.84, bbox_iou=0.94)
    bbox_center_fit["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.86,
            "visible_bbox_iou": 0.96,
            "visible_contour_score": 0.32,
            "visible_contour_mean_distance_px": 4.5,
            "visible_profile_mean_distance_px": 6.3,
            "bbox_center_error_px": 0.7,
        }
    )
    contour_fit = _pose_record(frame_id=7, x=0.14, yaw_deg=4.0, score=0.70, mask_iou=0.96, bbox_iou=0.71)
    contour_fit["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.96,
            "visible_bbox_iou": 0.95,
            "visible_contour_score": 0.72,
            "visible_contour_mean_distance_px": 1.0,
            "visible_profile_mean_distance_px": 2.2,
            "bbox_center_error_px": 24.0,
        }
    )

    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        {6: [anchor], 7: [bbox_center_fit, contour_fit]},
        target_frame_id=7,
    )

    assert summary["selected_frame_count"] == 2
    assert selected[1]["metrics"]["visible_contour_mean_distance_px"] == 1.0
    assert selected[1]["trajectory_selection"]["candidate_index"] == 1


def test_temporal_candidate_trajectory_allows_small_rotation_jump_for_strong_bottom_truncation_contour() -> None:
    anchor = _pose_record(frame_id=5, x=0.0, yaw_deg=0.0, score=0.90, mask_iou=0.88, bbox_iou=0.88)
    bbox_center_fit = _pose_record(frame_id=6, x=0.1, yaw_deg=0.0, score=1.69, mask_iou=0.84, bbox_iou=0.94)
    bbox_center_fit["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.86,
            "visible_bbox_iou": 0.96,
            "visible_contour_score": 0.32,
            "visible_contour_mean_distance_px": 4.5,
            "visible_profile_mean_distance_px": 6.3,
            "bbox_center_error_px": 0.7,
        }
    )
    contour_fit = _pose_record(frame_id=6, x=0.25, yaw_deg=5.0, scale=1.10, score=0.69, mask_iou=0.96, bbox_iou=0.71)
    contour_fit["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.96,
            "visible_bbox_iou": 0.95,
            "visible_contour_score": 0.72,
            "visible_contour_mean_distance_px": 1.0,
            "visible_profile_mean_distance_px": 2.2,
            "bbox_center_error_px": 24.0,
        }
    )

    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        {5: [anchor], 6: [bbox_center_fit, contour_fit]},
        target_frame_id=6,
    )

    assert summary["selected_frame_count"] == 2
    assert selected[1]["metrics"]["visible_contour_mean_distance_px"] == 1.0
    assert selected[1]["trajectory_selection"]["candidate_index"] == 1


def test_temporal_candidate_trajectory_does_not_force_isolated_truncated_contour_jump() -> None:
    anchor = _pose_record(frame_id=5, x=0.0, yaw_deg=0.0, score=0.90, mask_iou=0.88, bbox_iou=0.88)
    bbox_center_fit = _pose_record(frame_id=6, x=0.1, yaw_deg=0.0, score=1.69, mask_iou=0.84, bbox_iou=0.94)
    bbox_center_fit["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.86,
            "visible_bbox_iou": 0.96,
            "visible_contour_score": 0.32,
            "visible_contour_mean_distance_px": 4.5,
            "visible_profile_mean_distance_px": 6.3,
            "bbox_center_error_px": 0.7,
        }
    )
    contour_anchor = _pose_record(frame_id=6, x=0.45, yaw_deg=5.0, scale=1.15, score=0.69, mask_iou=0.96, bbox_iou=0.71)
    contour_anchor["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.96,
            "visible_bbox_iou": 0.95,
            "visible_contour_score": 0.72,
            "visible_contour_mean_distance_px": 1.0,
            "visible_profile_mean_distance_px": 2.2,
            "bbox_center_error_px": 24.0,
        }
    )
    next_bbox_path = _pose_record(frame_id=7, x=0.11, yaw_deg=1.0, scale=1.0, score=1.68, mask_iou=0.84, bbox_iou=0.94)
    next_bbox_path["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.86,
            "visible_bbox_iou": 0.96,
            "visible_contour_score": 0.34,
            "visible_contour_mean_distance_px": 4.4,
            "visible_profile_mean_distance_px": 6.1,
            "bbox_center_error_px": 0.9,
        }
    )
    next_contour_path = _pose_record(frame_id=7, x=0.2, yaw_deg=8.0, scale=1.0, score=1.34, mask_iou=0.86, bbox_iou=0.88)
    next_contour_path["metrics"].update(
        {
            "truncation_severity": "light",
            "low_observability": False,
            "visible_mask_iou": 0.88,
            "visible_bbox_iou": 0.96,
            "visible_contour_score": 0.39,
            "visible_contour_mean_distance_px": 3.6,
            "visible_profile_mean_distance_px": 5.2,
            "bbox_center_error_px": 5.6,
        }
    )

    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        {5: [anchor], 6: [bbox_center_fit, contour_anchor], 7: [next_bbox_path, next_contour_path]},
        target_frame_id=7,
    )

    assert summary["selected_frame_count"] == 3
    assert selected[1]["metrics"]["visible_contour_mean_distance_px"] == 4.5
    assert selected[1]["trajectory_selection"]["candidate_index"] == 0


def test_optimizer_final_selection_prefers_front_sign_consistent_candidate() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
        },
    )()
    selected = choose_best_refined_result(
        [
            {
                "score": 2.30,
                "mask_iou": 0.89,
                "bbox_iou": 0.88,
                "heading_front_sign_enabled": True,
                "heading_front_sign_confidence": 1.0,
                "heading_prior_angle_error_deg": 172.0,
            },
            {
                "score": 2.28,
                "mask_iou": 0.83,
                "bbox_iou": 0.75,
                "heading_front_sign_enabled": True,
                "heading_front_sign_confidence": 1.0,
                "heading_prior_angle_error_deg": 6.0,
            },
        ],
        args,
        {"is_truncated": False},
    )

    assert selected is not None
    assert selected["heading_prior_angle_error_deg"] == 6.0
    assert selected["final_selection_mode"] == "front_sign_consistent_rank"


def test_non_truncated_visual_ground_rescue_uses_wide_trigger() -> None:
    args = type(
        "Args",
        (),
        {
            "non_truncated_visual_ground_rescue_enabled": True,
            "rescue_ground_mean_trigger_m": 0.20,
            "rescue_ground_max_trigger_m": 0.30,
            "rescue_bbox_iou_trigger": 0.82,
            "rescue_center_error_trigger_px": 8.0,
            "rescue_mask_iou_trigger": 0.72,
            "rescue_mask_drop_trigger": 0.05,
            "visual_gate_enabled": True,
        },
    )()
    selected = {
        "mask_iou": 0.704,
        "bbox_iou": 0.708,
        "bbox_center_error_px": 13.24,
        "ground_contact_mean_abs_m": 0.297,
        "ground_contact_max_abs_m": 0.385,
        "visual_gate_factor": 0.0,
    }
    temporal_prior = {
        "quality": {
            "mask_iou": 0.892,
            "bbox_iou": 0.938,
        }
    }
    vehicle_pose_context = {
        "road_constraint": {"available": True},
        "track_scale_prior": {"available": True, "scale": [2.63, 2.63, 2.63]},
    }

    decision = should_run_non_truncated_visual_ground_rescue(
        selected,
        args,
        {"is_truncated": False},
        temporal_prior=temporal_prior,
        vehicle_pose_context=vehicle_pose_context,
    )

    assert decision["run"] is True
    assert decision["trigger_count"] >= 2
    assert "visual_gate_failed" in decision["reasons"]
    assert "ground_max" in decision["reasons"]


def test_non_truncated_visual_ground_rescue_requires_strict_replacement_gain() -> None:
    args = type(
        "Args",
        (),
        {
            "rescue_scale_delta_log_max": 0.08,
            "rescue_scale_ratio_from_anchor_max": 1.10,
            "rescue_replace_mask_gain_min": 0.04,
            "rescue_replace_bbox_gain_min": 0.06,
            "rescue_replace_center_gain_min_px": 3.0,
            "rescue_replace_temporal_loss_margin": 0.25,
            "rescue_replace_temporal_loss_hard_max": 80.0,
            "rescue_replace_strong_visual_mask_gain": 0.08,
            "rescue_replace_strong_visual_bbox_gain": 0.08,
            "rescue_replace_strong_visual_center_gain_px": 6.0,
            "rescue_replace_ground_mean_max_m": 0.30,
            "rescue_replace_ground_max_max_m": 0.50,
            "rescue_replace_mask_drop_max": 0.01,
            "rescue_replace_bbox_drop_max": 0.02,
            "rescue_replace_yaw_jump_max_deg": 15.0,
        },
    )()
    current = {
        "mask_iou": 0.704,
        "bbox_iou": 0.708,
        "bbox_center_error_px": 13.2,
        "ground_contact_mean_abs_m": 0.297,
        "ground_contact_max_abs_m": 0.385,
        "temporal_loss": 0.30,
    }
    rescued = {
        "mask_iou": 0.884,
        "bbox_iou": 0.856,
        "bbox_center_error_px": 1.3,
        "ground_gate_rejected": False,
        "upright_gate_rejected": False,
        "heading_front_sign_hard_rejected": False,
        "ground_contact_mean_abs_m": 0.19,
        "ground_contact_max_abs_m": 0.42,
        "scale_ratio_from_anchor": 1.04,
        "track_scale_prior_delta_log": 0.02,
        "temporal_loss": 46.0,
        "yaw_jump_from_anchor_deg": 5.0,
    }
    bad_scale = dict(rescued, scale_ratio_from_anchor=1.22)

    accepted = passes_non_truncated_visual_ground_rescue_replacement_gate(rescued, current, args)
    rejected = passes_non_truncated_visual_ground_rescue_replacement_gate(bad_scale, current, args)

    assert accepted["accepted"] is True
    assert accepted["visual_gain"] is True
    assert rejected["accepted"] is False
    assert "scale_ratio" in rejected["reasons"]


def test_optimizer_temporal_anchor_visual_gate_rejects_obj18_frame11_regression() -> None:
    args = type(
        "Args",
        (),
        {
            "temporal_anchor_visual_gate_enabled": True,
            "temporal_anchor_visual_gate_prior_mask_iou_min": 0.88,
            "temporal_anchor_visual_gate_prior_bbox_iou_min": 0.85,
            "temporal_anchor_visual_gate_mask_drop_max": 0.08,
            "temporal_anchor_visual_gate_bbox_drop_max": 0.10,
            "temporal_anchor_visual_gate_yaw_jump_deg": 8.0,
            "temporal_anchor_visual_gate_min_mask_iou": 0.86,
            "temporal_anchor_visual_gate_min_bbox_iou": 0.84,
            "temporal_anchor_visual_gate_temporal_loss_max": 0.50,
            "temporal_anchor_visual_gate_penalty": 1_000_000.0,
        },
    )()

    gate = compute_temporal_anchor_visual_gate(
        result={
            "mask_iou": 0.8090440755580997,
            "bbox_iou": 0.7761221199906715,
            "delta_yaw_deg": 9.999997913562225,
            "temporal_loss": 0.7659476237599382,
            "heading_prior_angle_error_deg": 13.778497797813854,
            "heading_prior_confidence": 0.35,
        },
        temporal_prior={
            "quality": {
                "mask_iou": 0.9260330578512397,
                "bbox_iou": 0.9234311966194116,
            }
        },
        truncation_info={"is_truncated": False},
        args=args,
    )

    assert gate["temporal_anchor_visual_gate_passed"] is False
    assert gate["temporal_anchor_visual_gate_reason"].startswith("visual_degradation_yaw_jump")
    assert gate["temporal_anchor_visual_mask_drop"] > 0.11
    assert gate["temporal_anchor_visual_bbox_drop"] > 0.14


def test_optimizer_temporal_anchor_visual_gate_accepts_stable_visual_followup() -> None:
    args = type(
        "Args",
        (),
        {
            "temporal_anchor_visual_gate_enabled": True,
            "temporal_anchor_visual_gate_prior_mask_iou_min": 0.88,
            "temporal_anchor_visual_gate_prior_bbox_iou_min": 0.85,
            "temporal_anchor_visual_gate_mask_drop_max": 0.08,
            "temporal_anchor_visual_gate_bbox_drop_max": 0.10,
            "temporal_anchor_visual_gate_yaw_jump_deg": 8.0,
            "temporal_anchor_visual_gate_min_mask_iou": 0.86,
            "temporal_anchor_visual_gate_min_bbox_iou": 0.84,
            "temporal_anchor_visual_gate_temporal_loss_max": 0.50,
            "temporal_anchor_visual_gate_penalty": 1_000_000.0,
        },
    )()

    gate = compute_temporal_anchor_visual_gate(
        result={
            "mask_iou": 0.906,
            "bbox_iou": 0.916,
            "delta_yaw_deg": 2.0,
            "temporal_loss": 0.08,
        },
        temporal_prior={
            "quality": {
                "mask_iou": 0.926,
                "bbox_iou": 0.923,
            }
        },
        truncation_info={"is_truncated": False},
        args=args,
    )

    assert gate["temporal_anchor_visual_gate_passed"] is True
    assert gate["temporal_anchor_visual_gate_reason"] == "passed"


def test_optimizer_temporal_anchor_visual_gate_falls_back_when_visible_iou_is_none() -> None:
    args = type(
        "Args",
        (),
        {
            "temporal_anchor_visual_gate_enabled": True,
            "temporal_anchor_visual_gate_prior_mask_iou_min": 0.88,
            "temporal_anchor_visual_gate_prior_bbox_iou_min": 0.85,
            "temporal_anchor_visual_gate_mask_drop_max": 0.08,
            "temporal_anchor_visual_gate_bbox_drop_max": 0.10,
            "temporal_anchor_visual_gate_yaw_jump_deg": 8.0,
            "temporal_anchor_visual_gate_min_mask_iou": 0.86,
            "temporal_anchor_visual_gate_min_bbox_iou": 0.84,
            "temporal_anchor_visual_gate_temporal_loss_max": 0.50,
            "temporal_anchor_visual_gate_penalty": 1_000_000.0,
        },
    )()

    gate = compute_temporal_anchor_visual_gate(
        result={
            "visible_mask_iou": None,
            "visible_bbox_iou": None,
            "mask_iou": 0.90,
            "bbox_iou": 0.89,
            "delta_yaw_deg": 2.0,
            "temporal_loss": 0.08,
        },
        temporal_prior={
            "quality": {
                "mask_iou": 0.926,
                "bbox_iou": 0.923,
            }
        },
        truncation_info={"is_truncated": False},
        args=args,
    )

    assert gate["temporal_anchor_visual_gate_passed"] is True
    assert gate["temporal_anchor_visual_mask_drop"] == pytest.approx(0.026)
    assert gate["temporal_anchor_visual_bbox_drop"] == pytest.approx(0.033)


def test_pose_record_rejects_temporal_prior_after_anchor_visual_degradation() -> None:
    anchor = _pose_record(frame_id=11, yaw_deg=-177.0, scale=2.67, mask_iou=0.923, bbox_iou=0.911)
    degraded = _pose_record(frame_id=12, yaw_deg=173.9, scale=2.65, mask_iou=0.839, bbox_iou=0.801)

    decision = ProjectExecutor._pose_anchor_temporal_gate(degraded, anchor)
    degraded["anchor_temporal_gate"] = decision

    assert decision["accepted"] is False
    assert decision["reason"].startswith("anchor_visual_degradation")
    assert decision["mask_drop"] > 0.08
    assert decision["bbox_drop"] > 0.10
    assert ProjectExecutor._pose_record_updates_temporal_anchor(degraded) is False
    assert ProjectExecutor._pose_record_is_high_quality_anchor(degraded) is False


def test_truncated_final_selection_prefers_anchor_consistent_temporal_candidate() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
            "final_ground_select_mean_max_m": 0.09,
            "final_ground_select_max_max_m": 0.16,
            "truncated_final_visual_selection_enabled": True,
            "truncated_final_visual_min_bbox_iou": 0.88,
            "truncated_final_visual_min_quality_gate": 0.80,
            "truncated_final_visual_mask_weight": 1.2,
            "truncated_final_visual_contour_weight": 0.5,
            "truncated_final_visual_bbox_weight": 0.04,
            "severe_truncated_final_visual_bbox_weight": 0.02,
            "truncated_final_visual_quality_weight": 0.05,
            "truncated_final_visual_ground_mean_weight": 0.18,
            "truncated_final_visual_ground_max_weight": 0.08,
            "truncated_final_visual_score_weight": 0.01,
            "truncated_final_visual_temporal_weight": 0.35,
            "truncated_final_visual_temporal_loss_weight": 0.04,
            "truncated_final_visual_heading_weight": 0.04,
            "truncated_final_visual_prefer_temporal_seed_bonus": 0.08,
            "truncated_final_visual_scale_jump_weight": 0.35,
            "truncated_final_visual_anchor_yaw_weight": 0.02,
            "truncated_anchor_gate_enabled": True,
            "truncated_anchor_gate_yaw_jump_deg": 12.0,
            "truncated_anchor_gate_scale_ratio": 1.08,
            "truncated_anchor_gate_temporal_loss_max": 2.0,
            "low_observability_anchor_gate_yaw_jump_deg": 8.0,
            "low_observability_anchor_gate_scale_ratio": 1.04,
            "low_observability_anchor_gate_temporal_loss_max": 1.0,
        },
    )()
    temporal_prior = {
        "score": 2.09,
        "visible_mask_iou": 0.9625,
        "visible_bbox_iou": 0.9761,
        "visible_contour_score": 0.755,
        "visible_contour_mean_distance_px": 0.79,
        "visible_profile_score": 0.674,
        "truncated_visual_quality_gate": 0.92,
        "ground_contact_mean_abs_m": 0.074,
        "ground_contact_max_abs_m": 0.149,
        "upright_angle_error_deg": 1.0,
        "temporal_score": 0.961,
        "temporal_loss": 0.04,
        "heading_prior_angle_error_deg": 45.8,
        "scale_ratio_from_anchor": 1.006,
        "yaw_jump_from_anchor_deg": 0.8,
        "initializer_metadata": {"source": "temporal_prior"},
    }
    flipped_coarse = {
        "score": 1.84,
        "visible_mask_iou": 0.9070,
        "visible_bbox_iou": 0.9618,
        "visible_contour_score": 0.474,
        "visible_contour_mean_distance_px": 2.48,
        "visible_profile_score": 0.397,
        "truncated_visual_quality_gate": 0.896,
        "ground_contact_mean_abs_m": 0.021,
        "ground_contact_max_abs_m": 0.036,
        "upright_angle_error_deg": 0.6,
        "temporal_score": 0.0,
        "temporal_loss": 114.3,
        "heading_prior_angle_error_deg": 37.3,
        "scale_ratio_from_anchor": 1.01,
        "yaw_jump_from_anchor_deg": 178.0,
        "initializer_metadata": {"source": "coarse_search"},
    }

    selected = choose_best_refined_result(
        [temporal_prior, flipped_coarse],
        args,
        {"is_truncated": True, "truncation_sides": ["right"], "truncation_severity": "light"},
    )

    assert selected is temporal_prior
    assert flipped_coarse["truncated_anchor_gate_passed"] is False
    assert "yaw_jump" in flipped_coarse["truncated_anchor_gate_reasons"]


def test_truncated_final_selection_does_not_discard_strong_temporal_for_slight_ground_excess() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
            "final_ground_select_mean_max_m": 0.055,
            "final_ground_select_max_max_m": 0.12,
            "final_ground_select_relaxed_mean_max_m": 0.075,
            "final_ground_select_relaxed_max_max_m": 0.16,
            "final_ground_relaxed_temporal_score_min": 0.70,
            "final_ground_relaxed_visible_mask_iou_min": 0.92,
            "final_ground_relaxed_visible_contour_mean_px_max": 2.5,
            "truncated_final_visual_selection_enabled": True,
            "truncated_final_visual_min_bbox_iou": 0.88,
            "truncated_final_visual_min_quality_gate": 0.80,
            "truncated_final_visual_mask_weight": 1.0,
            "truncated_final_visual_contour_weight": 0.35,
            "truncated_final_visual_bbox_weight": 0.08,
            "severe_truncated_final_visual_bbox_weight": 0.02,
            "truncated_final_visual_quality_weight": 0.05,
            "truncated_final_visual_ground_mean_weight": 0.25,
            "truncated_final_visual_ground_max_weight": 0.10,
            "truncated_final_visual_score_weight": 0.03,
            "truncated_final_visual_temporal_weight": 0.18,
            "truncated_final_visual_temporal_loss_weight": 0.02,
            "truncated_final_visual_heading_weight": 0.02,
            "truncated_final_visual_prefer_temporal_seed_bonus": 0.04,
            "truncated_final_visual_scale_jump_weight": 0.20,
            "truncated_final_visual_anchor_yaw_weight": 0.01,
            "truncated_final_visual_front_flip_penalty": 1.0,
            "truncated_anchor_gate_enabled": True,
            "truncated_anchor_gate_yaw_jump_deg": 12.0,
            "truncated_anchor_gate_scale_ratio": 1.08,
            "truncated_anchor_gate_temporal_loss_max": 2.0,
            "low_observability_anchor_gate_yaw_jump_deg": 8.0,
            "low_observability_anchor_gate_scale_ratio": 1.04,
            "low_observability_anchor_gate_temporal_loss_max": 1.0,
        },
    )()
    temporal_prior = {
        "score": 2.13,
        "visible_mask_iou": 0.959,
        "visible_bbox_iou": 0.984,
        "visible_contour_score": 0.753,
        "visible_contour_mean_distance_px": 0.80,
        "visible_profile_mean_distance_px": 1.99,
        "truncated_visual_quality_gate": 1.0,
        "ground_contact_mean_abs_m": 0.062,
        "ground_contact_max_abs_m": 0.115,
        "temporal_score": 0.883,
        "temporal_loss": 0.125,
        "heading_prior_angle_error_deg": 44.2,
        "heading_candidate_forward_sign": 1.0,
        "heading_semantic_front_sign": 1.0,
        "scale_ratio_from_anchor": 1.003,
        "yaw_jump_from_anchor_deg": 3.3,
        "initializer_metadata": {"source": "temporal_prior"},
    }
    flipped = {
        "score": 1.93,
        "visible_mask_iou": 0.925,
        "visible_bbox_iou": 1.0,
        "visible_contour_score": 0.573,
        "visible_contour_mean_distance_px": 1.84,
        "visible_profile_mean_distance_px": 3.46,
        "truncated_visual_quality_gate": 1.0,
        "ground_contact_mean_abs_m": 0.048,
        "ground_contact_max_abs_m": 0.087,
        "temporal_score": 0.0,
        "temporal_loss": 114.7,
        "heading_prior_angle_error_deg": 46.7,
        "heading_candidate_forward_sign": 1.0,
        "heading_semantic_front_sign": -1.0,
        "scale_ratio_from_anchor": 1.024,
        "yaw_jump_from_anchor_deg": 178.4,
        "initializer_metadata": {"source": "coarse_search"},
    }

    selected = choose_best_refined_result(
        [temporal_prior, flipped],
        args,
        {"is_truncated": True, "truncation_sides": ["right"], "truncation_severity": "light"},
    )

    assert selected is temporal_prior
    assert temporal_prior["final_ground_constrained_selected"] is True
    assert temporal_prior["final_ground_relaxed_selected"] is True
    assert "yaw_jump" in flipped["truncated_anchor_gate_reasons"]


def test_severe_truncation_fallback_rejects_anchor_inconsistent_candidate() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
            "final_ground_select_mean_max_m": 0.09,
            "final_ground_select_max_max_m": 0.16,
            "truncated_final_visual_selection_enabled": True,
            "severe_truncation_final_gate_enabled": True,
            "severe_truncation_visible_mask_iou_min": 0.95,
            "severe_truncation_visible_contour_mean_px_max": 1.0,
            "severe_truncation_visible_profile_mean_px_max": 2.0,
            "severe_truncation_ground_mean_m_max": 0.085,
            "severe_truncation_ground_max_m_max": 0.18,
            "severe_truncation_upright_deg_max": 35.0,
            "severe_truncation_yaw_jump_deg_max": 25.0,
            "severe_truncation_fallback_visible_mask_iou_min": 0.76,
            "severe_truncation_fallback_visible_contour_mean_px_max": 5.5,
            "severe_truncation_fallback_visible_profile_mean_px_max": 8.5,
            "truncated_anchor_gate_enabled": True,
            "truncated_anchor_gate_yaw_jump_deg": 12.0,
            "truncated_anchor_gate_scale_ratio": 1.08,
            "truncated_anchor_gate_temporal_loss_max": 2.0,
            "low_observability_anchor_gate_yaw_jump_deg": 8.0,
            "low_observability_anchor_gate_scale_ratio": 1.04,
            "low_observability_anchor_gate_temporal_loss_max": 1.0,
        },
    )()
    flipped = {
        "score": 1.93,
        "visible_mask_iou": 0.925,
        "visible_bbox_iou": 1.0,
        "visible_contour_score": 0.57,
        "visible_contour_mean_distance_px": 1.84,
        "visible_profile_mean_distance_px": 3.46,
        "truncated_visual_quality_gate": 0.95,
        "ground_contact_mean_abs_m": 0.03,
        "ground_contact_max_abs_m": 0.06,
        "upright_angle_error_deg": 1.0,
        "temporal_score": 0.0,
        "temporal_loss": 114.0,
        "scale_ratio_from_anchor": 1.004,
        "yaw_jump_from_anchor_deg": 173.0,
        "initializer_metadata": {"source": "coarse_search"},
    }
    anchor_consistent = {
        "score": 2.03,
        "visible_mask_iou": 0.953,
        "visible_bbox_iou": 0.975,
        "visible_contour_score": 0.71,
        "visible_contour_mean_distance_px": 0.99,
        "visible_profile_mean_distance_px": 2.44,
        "truncated_visual_quality_gate": 0.95,
        "ground_contact_mean_abs_m": 0.07,
        "ground_contact_max_abs_m": 0.14,
        "upright_angle_error_deg": 1.0,
        "temporal_score": 0.07,
        "temporal_loss": 2.6,
        "scale_ratio_from_anchor": 1.0,
        "yaw_jump_from_anchor_deg": 5.9,
        "initializer_metadata": {"source": "temporal_prior"},
    }

    selected = choose_best_refined_result(
        [flipped, anchor_consistent],
        args,
        {"is_truncated": True, "truncation_sides": ["right"], "truncation_severity": "severe"},
    )

    assert selected is anchor_consistent
    assert flipped["truncated_anchor_gate_passed"] is False
    assert "yaw_jump" in flipped["truncated_anchor_gate_reasons"]


def test_truncation_observability_marks_bottom_contour_drift_as_severe() -> None:
    args = type(
        "Args",
        (),
        {
            "truncation_border_margin": 3,
            "truncation_bbox_margin": 5,
            "truncation_moderate_visible_mask_iou": 0.78,
            "truncation_severe_visible_mask_iou": 0.70,
            "truncation_moderate_contour_mean_px": 5.0,
            "truncation_severe_contour_mean_px": 7.0,
            "truncation_moderate_profile_mean_px": 8.0,
            "truncation_severe_profile_mean_px": 10.0,
            "truncation_area_drop_ratio": 0.72,
        },
    )()
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[34:100, 40:120] = 1

    info = detect_truncation(
        mask,
        [40.0, 34.0, 120.0, 100.0],
        (160, 100),
        args,
        prior_mask_area_px=None,
    )
    result = classify_truncation_observability(
        info,
        {
            "visible_mask_iou": 0.75,
            "visible_bbox_iou": 0.96,
            "visible_contour_mean_distance_px": 8.5,
            "visible_profile_mean_distance_px": 11.0,
        },
        args,
    )

    assert result["severity"] == "severe"
    assert result["low_observability"] is True
    assert "visible_contour" in result["reasons"]
    assert info["truncation_severity"] == "severe"


def test_truncated_visual_quality_gate_uses_visible_contour_and_mask() -> None:
    args = type(
        "Args",
        (),
        {
            "truncated_visual_quality_gate_enabled": True,
            "truncated_visual_quality_gate_floor": 0.25,
            "truncated_visual_gate_bbox_iou_min": 0.88,
            "truncated_visual_gate_bbox_iou_softness": 0.08,
            "truncated_visual_gate_center_error_px": 6.0,
            "truncated_visual_gate_center_softness_px": 8.0,
            "truncated_visual_gate_overflow_sigma_px": 32.0,
            "truncated_visual_quality_penalty_weight": 0.08,
            "truncated_visual_gate_visible_mask_iou_good": 0.78,
            "truncated_visual_gate_visible_mask_iou_bad": 0.68,
            "truncated_visual_gate_contour_mean_px_good": 4.5,
            "truncated_visual_gate_contour_mean_px_bad": 7.0,
            "truncated_visual_gate_profile_mean_px_good": 7.5,
            "truncated_visual_gate_profile_mean_px_bad": 10.0,
        },
    )()

    quality = compute_truncated_visual_quality_gate(
        result={
            "visible_bbox_iou": 0.97,
            "visible_bbox_center_error_px": 2.0,
            "projected_bbox": [20.0, 20.0, 130.0, 100.0],
            "visible_mask_iou": 0.72,
            "visible_contour_mean_distance_px": 8.5,
            "visible_profile_mean_distance_px": 11.0,
        },
        image_size=(160, 100),
        truncation_info={"is_truncated": True, "truncation_sides": ["bottom"], "truncation_severity": "severe"},
        args=args,
    )

    assert quality["truncated_visual_quality_gate"] <= 0.35
    assert "visible_contour" in quality["truncated_visual_quality_reason"]
    assert "visible_profile" in quality["truncated_visual_quality_reason"]


def test_bottom_truncation_quality_gate_does_not_floor_good_visible_mask_for_overflow() -> None:
    args = type(
        "Args",
        (),
        {
            "truncated_visual_quality_gate_enabled": True,
            "truncated_visual_quality_gate_floor": 0.25,
            "truncated_visual_gate_bbox_iou_min": 0.88,
            "truncated_visual_gate_bbox_iou_softness": 0.08,
            "truncated_visual_gate_center_error_px": 6.0,
            "truncated_visual_gate_center_softness_px": 8.0,
            "truncated_visual_gate_overflow_sigma_px": 32.0,
            "truncated_visual_quality_penalty_weight": 0.08,
            "truncated_visual_gate_visible_mask_iou_good": 0.78,
            "truncated_visual_gate_visible_mask_iou_bad": 0.68,
            "truncated_visual_gate_contour_mean_px_good": 4.5,
            "truncated_visual_gate_contour_mean_px_bad": 7.0,
            "truncated_visual_gate_profile_mean_px_good": 7.5,
            "truncated_visual_gate_profile_mean_px_bad": 10.0,
        },
    )()

    quality = compute_truncated_visual_quality_gate(
        result={
            "visible_bbox_iou": 0.95,
            "visible_bbox_center_error_px": 2.0,
            "projected_bbox": [20.0, 20.0, 130.0, 170.0],
            "visible_mask_iou": 0.96,
            "visible_contour_mean_distance_px": 1.0,
            "visible_profile_mean_distance_px": 2.2,
        },
        image_size=(160, 100),
        truncation_info={"is_truncated": True, "truncation_sides": ["bottom"], "truncation_severity": "light"},
        args=args,
    )

    assert quality["truncated_visual_quality_gate"] > 0.80
    assert quality["truncated_visual_overflow_factor"] < 1.0


def test_truncated_visible_bbox_uses_visible_rendered_mask_bbox() -> None:
    args = type("Args", (), {"ignore_truncated_border_band_px": 16})()
    rendered_mask = np.zeros((100, 160), dtype=np.uint8)
    rendered_mask[25:84, 45:115] = 1
    rendered_mask[84:100, 20:150] = 1
    target_mask = np.zeros((100, 160), dtype=np.uint8)
    target_mask[25:84, 45:115] = 1

    score = compute_visible_bbox_score(
        projected_bbox=[20.0, 25.0, 150.0, 140.0],
        target_bbox=[45.0, 25.0, 115.0, 100.0],
        image_size=(160, 100),
        truncation_info={"is_truncated": True, "truncation_sides": ["bottom"]},
        args=args,
        rendered_mask=rendered_mask,
        target_mask=target_mask,
    )

    assert score["visible_projected_bbox"] == [45.0, 25.0, 115.0, 84.0]
    assert score["visible_target_bbox"] == [45.0, 25.0, 115.0, 84.0]
    assert score["visible_bbox_iou"] == 1.0


def test_truncated_visible_bbox_falls_back_without_visible_mask() -> None:
    args = type("Args", (), {"ignore_truncated_border_band_px": 16})()

    score = compute_visible_bbox_score(
        projected_bbox=[20.0, 25.0, 150.0, 140.0],
        target_bbox=[45.0, 25.0, 115.0, 100.0],
        image_size=(160, 100),
        truncation_info={"is_truncated": True, "truncation_sides": ["bottom"]},
        args=args,
    )

    assert score["visible_projected_bbox"] == [20.0, 25.0, 150.0, 84.0]
    assert score["visible_target_bbox"] == [45.0, 25.0, 115.0, 84.0]


def test_severe_bottom_truncation_disables_bbox_bottom_but_keeps_heading_prior() -> None:
    args = type(
        "Args",
        (),
        {
            "road_constraint_enabled": True,
            "ground_contact_sample_count": 16,
            "ground_contact_sample_percentile": 8.0,
            "ground_contact_sigma_m": 0.18,
            "ground_contact_hard_gate_enabled": True,
            "ground_contact_hard_gate_mean_m": 0.30,
            "ground_contact_hard_gate_max_m": 0.60,
            "bbox_bottom_ground_sigma_m": 0.45,
            "road_constraint_weight": 0.25,
            "bbox_bottom_ground_weight": 0.15,
            "bottom_truncated_ground_contact_weight_factor": 1.60,
            "bottom_truncated_ground_weight_factor": 0.25,
            "severe_bottom_truncated_bbox_bottom_weight_factor": 0.0,
            "bottom_truncated_ground_soft_tolerance_m": 0.12,
            "bottom_truncated_ground_penalty_weight": 0.35,
            "bottom_truncated_ground_penalty_sigma_m": 0.18,
            "upright_angle_sigma_deg": 10.0,
            "upright_strong_penalty_angle_deg": 15.0,
            "upright_hard_gate_max_angle_deg": 60.0,
            "upright_hard_gate_enabled": True,
            "upright_hard_gate_sigma_deg": 15.0,
            "upright_hard_gate_penalty": 2.0,
            "upright_weight": 0.10,
            "heading_prior_enabled": True,
            "heading_prior_sigma_deg": 25.0,
            "heading_prior_weight": 0.06,
            "front_sign_heading_prior_weight": 0.18,
            "truncated_heading_prior_weight": 0.08,
            "severe_truncation_heading_prior_weight": 0.18,
            "heading_prior_lock_front_sign": False,
            "mesh_tail_light_front_sign_enabled": False,
            "bbox_area_trend_front_sign_enabled": False,
            "front_sign_hard_gate_enabled": True,
            "front_sign_hard_gate_min_confidence": 0.25,
            "front_sign_hard_gate_angle_deg": 120.0,
            "front_sign_depth_trend_enabled": False,
            "world_up_axis": "y",
        },
    )()
    vehicle_pose_context = {
        "road_constraint": {
            "available": True,
            "road_plane": {"normal_world": [0.0, 1.0, 0.0], "offset": 0.0},
            "bbox_bottom_ground": {"point_world": [0.0, 0.0, 8.0]},
        },
        "heading_prior": {
            "enabled": True,
            "confidence": 1.0,
            "vector_image": [1.0, 0.0],
            "source": "temporal_anchor",
        },
    }
    mesh_meta = {
        "axis_prior": {
            "up_axis_idx": 1,
            "up_sign": 1.0,
            "forward_axis_idx": 2,
            "forward_sign": 1.0,
        },
        "bounds": [[-1.0, 0.0, -2.0], [1.0, 1.0, 2.0]],
    }

    result = compute_road_and_heading_score(
        translation_cam=np.array([0.0, 0.0, 8.0], dtype=np.float64),
        rotation_cam=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        scale=np.ones(3, dtype=np.float64),
        t_world_from_cam=np.eye(4, dtype=np.float64),
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
        projected_bbox=[20.0, 20.0, 120.0, 100.0],
        image_size=(160, 100),
        truncation_info={"is_truncated": True, "truncation_sides": ["bottom"], "truncation_severity": "severe"},
        initializer_metadata={},
        args=args,
    )

    assert result["effective_bbox_bottom_weight"] == 0.0
    assert result["effective_ground_contact_weight"] > 0.0
    assert result["effective_heading_prior_weight"] >= 0.18


def test_severe_truncated_final_selection_rejects_bbox_only_candidate() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
            "final_ground_select_mean_max_m": 0.055,
            "final_ground_select_max_max_m": 0.12,
            "truncated_final_visual_selection_enabled": True,
            "truncated_final_visual_min_bbox_iou": 0.0,
            "truncated_final_visual_min_quality_gate": 0.0,
            "severe_truncation_final_gate_enabled": True,
            "severe_truncation_visible_mask_iou_min": 0.68,
            "severe_truncation_visible_contour_mean_px_max": 7.0,
            "severe_truncation_visible_profile_mean_px_max": 10.0,
            "severe_truncation_ground_mean_m_max": 0.085,
            "severe_truncation_ground_max_m_max": 0.18,
            "severe_truncation_upright_deg_max": 35.0,
            "severe_truncation_yaw_jump_deg_max": 25.0,
            "truncated_final_visual_mask_weight": 1.0,
            "truncated_final_visual_contour_weight": 0.50,
            "truncated_final_visual_bbox_weight": 0.02,
            "truncated_final_visual_quality_weight": 0.05,
            "truncated_final_visual_ground_mean_weight": 0.25,
            "truncated_final_visual_ground_max_weight": 0.10,
            "truncated_final_visual_score_weight": 0.03,
        },
    )()
    bbox_only = {
        "score": 1.9,
        "visible_mask_iou": 0.72,
        "visible_bbox_iou": 0.98,
        "visible_contour_score": 0.11,
        "visible_contour_mean_distance_px": 8.5,
        "visible_profile_mean_distance_px": 11.0,
        "truncated_visual_quality_gate": 0.95,
        "ground_contact_mean_abs_m": 0.02,
        "ground_contact_max_abs_m": 0.04,
        "upright_angle_error_deg": 3.0,
        "yaw_jump_from_anchor_deg": 6.0,
    }
    contour_consistent = {
        "score": 1.5,
        "visible_mask_iou": 0.78,
        "visible_bbox_iou": 0.90,
        "visible_contour_score": 0.42,
        "visible_contour_mean_distance_px": 4.5,
        "visible_profile_mean_distance_px": 7.5,
        "truncated_visual_quality_gate": 0.92,
        "ground_contact_mean_abs_m": 0.02,
        "ground_contact_max_abs_m": 0.04,
        "upright_angle_error_deg": 3.0,
        "yaw_jump_from_anchor_deg": 6.0,
    }

    selected = choose_best_refined_result(
        [bbox_only, contour_consistent],
        args,
        {"is_truncated": True, "truncation_sides": ["bottom"], "truncation_severity": "severe"},
    )

    assert selected is contour_consistent
    assert bbox_only["severe_truncation_gate_passed"] is False
    assert "visible_contour" in bbox_only["severe_truncation_gate_reasons"]


def test_severe_truncated_final_selection_rejects_when_only_visible_bbox_matches() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
            "final_ground_select_mean_max_m": 0.055,
            "final_ground_select_max_max_m": 0.12,
            "truncated_final_visual_selection_enabled": True,
            "truncated_final_visual_min_bbox_iou": 0.0,
            "truncated_final_visual_min_quality_gate": 0.0,
            "severe_truncation_final_gate_enabled": True,
            "severe_truncation_visible_mask_iou_min": 0.68,
            "severe_truncation_visible_contour_mean_px_max": 7.0,
            "severe_truncation_visible_profile_mean_px_max": 10.0,
            "severe_truncation_ground_mean_m_max": 0.085,
            "severe_truncation_ground_max_m_max": 0.18,
            "severe_truncation_upright_deg_max": 35.0,
            "severe_truncation_yaw_jump_deg_max": 25.0,
            "severe_truncation_fallback_visible_mask_iou_min": 0.78,
            "severe_truncation_fallback_visible_contour_mean_px_max": 5.0,
            "severe_truncation_fallback_visible_profile_mean_px_max": 8.0,
        },
    )()
    bbox_only = {
        "score": 1.9,
        "visible_mask_iou": 0.74,
        "visible_bbox_iou": 1.0,
        "visible_contour_score": 0.08,
        "visible_contour_mean_distance_px": 8.3,
        "visible_profile_mean_distance_px": 12.0,
        "truncated_visual_quality_gate": 0.25,
        "ground_contact_mean_abs_m": 0.01,
        "ground_contact_max_abs_m": 0.03,
        "upright_angle_error_deg": 3.0,
    }

    selected = choose_best_refined_result(
        [bbox_only],
        args,
        {"is_truncated": True, "truncation_sides": ["bottom"], "truncation_severity": "severe"},
    )

    assert selected is None
    assert bbox_only["severe_truncation_gate_passed"] is False
    assert bbox_only["severe_truncation_fallback_rejected"] is True


def test_severe_truncated_final_selection_rejects_tiny_visible_target_fraction() -> None:
    args = type(
        "Args",
        (),
        {
            "final_ground_constrained_selection_enabled": True,
            "final_ground_select_mean_max_m": 0.055,
            "final_ground_select_max_max_m": 0.12,
            "truncated_final_visual_selection_enabled": True,
            "truncated_final_visual_min_bbox_iou": 0.0,
            "truncated_final_visual_min_quality_gate": 0.0,
            "severe_truncation_final_gate_enabled": True,
            "severe_truncation_visible_mask_iou_min": 0.68,
            "severe_truncation_visible_contour_mean_px_max": 7.0,
            "severe_truncation_visible_profile_mean_px_max": 10.0,
            "severe_truncation_visible_target_fraction_min": 0.18,
            "severe_truncation_ground_mean_m_max": 0.085,
            "severe_truncation_ground_max_m_max": 0.18,
            "severe_truncation_upright_deg_max": 35.0,
            "severe_truncation_yaw_jump_deg_max": 25.0,
            "severe_truncation_fallback_visible_mask_iou_min": 0.76,
            "severe_truncation_fallback_visible_contour_mean_px_max": 5.5,
            "severe_truncation_fallback_visible_profile_mean_px_max": 8.5,
            "severe_truncation_fallback_visible_target_fraction_min": 0.18,
        },
    )()
    tiny_visible = {
        "score": 1.7,
        "visible_mask_iou": 0.95,
        "visible_bbox_iou": 0.99,
        "visible_contour_score": 0.80,
        "visible_contour_mean_distance_px": 1.0,
        "visible_profile_mean_distance_px": 2.0,
        "visible_target_fraction": 0.07,
        "truncated_visual_quality_gate": 0.97,
        "ground_contact_mean_abs_m": 0.01,
        "ground_contact_max_abs_m": 0.03,
        "upright_angle_error_deg": 3.0,
        "yaw_jump_from_anchor_deg": 3.0,
    }

    selected = choose_best_refined_result(
        [tiny_visible],
        args,
        {"is_truncated": True, "truncation_sides": ["bottom"], "truncation_severity": "severe"},
    )

    assert selected is None
    assert tiny_visible["severe_truncation_gate_passed"] is False
    assert "visible_fraction" in tiny_visible["severe_truncation_gate_reasons"]
    assert tiny_visible["severe_truncation_fallback_rejected"] is True


def test_temporal_jump_rejection_can_fallback_to_high_quality_visual_pose() -> None:
    previous = _pose_record(frame_id=1, x=0.0, yaw_deg=0.0)
    report = {
        "frame_idx": 2,
        "optimized_corrected_pose_world": _pose_record(frame_id=2, x=0.2, yaw_deg=180.0)["pose"],
        "metrics": {
            "score": 0.92,
            "mask_iou": 0.76,
            "bbox_iou": 0.81,
            "bbox_center_error_px": 9.0,
        },
    }

    jump = ProjectExecutor._pose_optimizer_temporal_jump_acceptance(report, previous)
    fallback = ProjectExecutor._pose_optimizer_temporal_fallback_acceptance(report, jump)

    assert jump["accepted"] is False
    assert fallback["accepted"] is True
    assert fallback["low_confidence"] is True


def test_motion_heading_prior_respects_target_window_radius() -> None:
    detections = []
    for frame_id in range(1, 12):
        detections.append(
            {
                "frame_id": frame_id,
                "instances": [
                    {
                        "object_id": "obj_000001",
                        "bbox_xyxy": [
                            200.0 - frame_id * 5.0,
                            170.0 + frame_id * 9.0,
                            300.0 - frame_id * 2.0,
                            300.0 + frame_id * 8.0,
                        ],
                    }
                ],
            }
        )

    class DummyExecutor:
        _motion_heading_prior_for_track = ProjectExecutor._motion_heading_prior_for_track

        def _get_instance_for_frame(self, obj_id, frame_id, detection_frames):
            for entry in detection_frames:
                if entry.get("frame_id") != frame_id:
                    continue
                for inst in entry.get("instances", []):
                    if inst.get("object_id") == obj_id:
                        return inst
            return None

    prior = DummyExecutor()._motion_heading_prior_for_track(
        obj_id="obj_000001",
        frame_id=3,
        detection_frames=detections,
        window=2,
    )

    assert prior is not None
    assert prior["frame_window"] == [1, 5]
    assert prior["from_frame"] == 1
    assert prior["to_frame"] == 5


def test_all_frames_seed_uses_local_window_frame_ids() -> None:
    local = ProjectExecutor._pose_local_seed_frame_ids(
        frame_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        current_frame_id=2,
        window_radius=2,
    )

    assert local == [1, 2, 3, 4, 5]


def test_all_frames_candidate_pass_can_use_in_memory_temporal_prior() -> None:
    previous = _pose_record(frame_id=1, x=0.0, yaw_deg=-178.0, scale=2.5)

    prior = ProjectExecutor._edge_pose_candidate_temporal_prior_payload(
        previous,
        all_frames_mode=True,
    )

    assert prior is not None
    assert prior["source"] == "previous_accepted_pose_in_memory"
    assert prior["frame_id"] == 1
    assert prior["pose"]["scale"] == [2.5, 2.5, 2.5]


def test_all_frames_scale_prior_uses_previous_accepted_records_only() -> None:
    previous_records = [
        _pose_record(frame_id=1, scale=2.50, mask_iou=0.91, bbox_iou=0.93),
        _pose_record(frame_id=2, scale=2.60, mask_iou=0.92, bbox_iou=0.94),
    ]

    prior = ProjectExecutor._pose_track_scale_prior(
        previous_records,
        source="accepted_track_median_scale",
        require_high_quality=True,
        max_frame_id=3,
    )

    assert prior is not None
    assert prior["frame_ids"] == [1, 2]
    assert prior["sample_count"] == 2
    assert prior["scale"] == [2.55, 2.55, 2.55]


def test_all_frame_candidate_frame_ids_respect_optional_env_range(monkeypatch) -> None:
    detections = []
    for frame_id in range(1, 8):
        detections.append(
            {
                "frame_id": frame_id,
                "instances": [
                    {
                        "object_id": "obj_000003",
                        "concept_label": "car",
                        "bbox_xyxy": [10.0, 20.0, 90.0, 100.0],
                    }
                ],
            }
        )
    monkeypatch.setenv("GUANWU_POSE_FRAME_ID_RANGE", "1-5")

    frame_ids = ProjectExecutor._pose_all_frame_candidate_frame_ids(
        obj_id="obj_000003",
        detection_frames=detections,
        min_bbox_area_px=1000.0,
    )

    assert frame_ids == [1, 2, 3, 4, 5]


def test_vehicle_pose_context_heading_uses_long_track_window_despite_dynamic_radius() -> None:
    detections = []
    for frame_id in range(1, 12):
        detections.append(
            {
                "frame_id": frame_id,
                "instances": [
                    {
                        "object_id": "obj_000001",
                        "bbox_xyxy": [
                            200.0 - frame_id * 5.0,
                            170.0 + frame_id * 9.0,
                            300.0 - frame_id * 2.0,
                            300.0 + frame_id * 8.0,
                        ],
                    }
                ],
            }
        )

    class DummyExecutor:
        _vehicle_pose_context_for_task = ProjectExecutor._vehicle_pose_context_for_task
        _motion_heading_prior_for_track = ProjectExecutor._motion_heading_prior_for_track

        def _get_instance_for_frame(self, obj_id, frame_id, detection_frames):
            for entry in detection_frames:
                if entry.get("frame_id") != frame_id:
                    continue
                for inst in entry.get("instances", []):
                    if inst.get("object_id") == obj_id:
                        return inst
            return None

    context = DummyExecutor()._vehicle_pose_context_for_task(
        obj_id="obj_000001",
        frame_id=1,
        bbox_xyxy=[195.0, 179.0, 298.0, 308.0],
        camera={},
        detection_frames=detections,
        road_geometry=None,
        target_window_radius=2,
    )

    prior = context["heading_prior"]
    assert prior["target_window_radius"] >= 8
    assert prior["to_frame"] >= 9
    assert prior["displacement_px"] > 35.0


def test_refine_candidate_selection_always_keeps_task_json_corrected_pose() -> None:
    args = type("Args", (), {"pareto_refine_selection_enabled": True})()
    candidates = [
        {
            "score": 2.0 - idx * 0.01,
            "translation_cam": [float(idx), 0.0, 8.0],
            "rotation_cam": np.eye(3).tolist(),
            "scale": [1.0, 1.0, 1.0],
            "initializer_metadata": {"source": "coarse_search"},
        }
        for idx in range(4)
    ]
    corrected = {
        "score": 1.2,
        "translation_cam": [99.0, 0.0, 8.0],
        "rotation_cam": np.eye(3).tolist(),
        "scale": [1.0, 1.0, 1.0],
        "initializer_metadata": {"source": "task_json_corrected_pose"},
    }

    selected = select_pareto_refine_candidates(
        [*candidates, corrected],
        refine_top_k=4,
        args=args,
        truncation_info={"is_truncated": False},
    )

    assert len(selected) == 4
    assert any(item["initializer_metadata"]["source"] == "task_json_corrected_pose" for item in selected)


def test_depth_trend_does_not_hard_reject_good_heading_alignment() -> None:
    args = type(
        "Args",
        (),
        {
            "heading_prior_enabled": True,
            "mesh_tail_light_front_sign_enabled": True,
            "mesh_tail_light_front_sign_min_confidence": 0.2,
            "mesh_tail_light_front_sign_standalone_min_confidence": 0.75,
            "mesh_tail_light_front_sign_standalone_min_density_ratio": 5.0,
            "tail_light_motion_consistency_flip_enabled": True,
            "tail_light_motion_consistency_min_confidence": 0.6,
            "tail_light_motion_consistency_flip_margin": 0.2,
            "heading_prior_sigma_deg": 25.0,
            "heading_prior_weight": 0.03,
            "front_sign_heading_prior_weight": 0.10,
            "front_sign_mismatch_penalty": 0.8,
            "front_sign_angle_penalty_weight": 1.2,
            "front_sign_hard_gate_enabled": True,
            "front_sign_hard_gate_angle_deg": 120.0,
            "front_sign_hard_gate_min_confidence": 0.25,
            "front_sign_depth_trend_enabled": True,
            "front_sign_depth_trend_penalty": 0.45,
            "front_sign_depth_trend_hard_gate_score_min": 0.35,
            "truncated_heading_prior_weight": 0.08,
            "road_constraint_enabled": False,
        },
    )()
    vehicle_pose_context = {
        "heading_prior": {
            "enabled": True,
            "confidence": 0.35,
            "vector_image": [0.0, -1.0],
            "bbox_area_trend": {
                "direction": "receding",
                "confidence": 0.82,
                "monotonicity": 0.4,
                "truncated_tail": True,
            },
        },
        "mesh_tail_light_prior": {
            "available": True,
            "axis_idx": 2,
            "front_sign": 1.0,
            "confidence": 1.0,
            "strong_available": True,
            "density_ratio": 10.0,
        },
    }
    mesh_meta = {
        "axis_prior": {
            "forward_axis_idx": 2,
            "forward_sign": 1.0,
        }
    }
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    rotation_cam = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, -inv_sqrt2, -inv_sqrt2],
            [0.0, inv_sqrt2, -inv_sqrt2],
        ],
        dtype=np.float64,
    )

    result = compute_road_and_heading_score(
        translation_cam=np.zeros(3, dtype=np.float64),
        rotation_cam=rotation_cam,
        scale=np.ones(3, dtype=np.float64),
        t_world_from_cam=np.eye(4, dtype=np.float64),
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
        projected_bbox=[0.0, 0.0, 10.0, 10.0],
        image_size=(640, 360),
        truncation_info={"is_truncated": False},
        initializer_metadata={},
        args=args,
    )

    assert result["heading_prior_angle_error_deg"] < 1e-6
    assert result["heading_front_sign_hard_rejected"] is False


def test_all_frames_context_skips_disk_temporal_prior_lookup() -> None:
    assert should_skip_disk_temporal_prior(
        {
            "temporal_window": {
                "mode": "all_frames",
                "dynamic": True,
                "radius": 2,
            }
        }
    )
    assert not should_skip_disk_temporal_prior(
        {
            "temporal_window": {
                "mode": "target_window",
                "dynamic": True,
                "radius": 2,
            }
        }
    )


def test_vehicle_pose_context_preserves_temporal_window_for_optimizer() -> None:
    task = {
        "object_id": "obj_000003",
        "frame_idx": 2,
        "vehicle_pose_context": {
            "temporal_window": {
                "mode": "all_frames",
                "base_radius": 2,
                "radius": 2,
                "dynamic": True,
            },
            "heading_prior": {"enabled": False},
        },
    }
    args = type(
        "Args",
        (),
        {
            "vehicle_mesh_axis_override_enabled": False,
            "road_depth_fallback_enabled": False,
        },
    )()

    context = build_vehicle_pose_context(
        task=task,
        sample_dir=Path("E:/QingYan/Guanwu-master2/workspace/projects/video/codex_allframes_20260521_1600/outputs/08_pose_optimize/tasks/obj_000003@000002"),
        full_mask=np.zeros((10, 10), dtype=np.uint8),
        json_bbox=[1.0, 1.0, 8.0, 8.0],
        image_size=(10, 10),
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 5.0, "cy": 5.0},
        t_world_from_cam=np.eye(4, dtype=np.float64),
        args=args,
    )

    assert context["temporal_window"]["mode"] == "all_frames"
    assert should_skip_disk_temporal_prior(context)


def test_vehicle_pose_context_preserves_locked_mesh_up_sign() -> None:
    task = {
        "object_id": "obj_000003",
        "frame_idx": 1,
        "vehicle_pose_context": {
            "mesh_axis_prior": {
                "available": True,
                "up_axis_idx": 1,
                "up_sign": 1.0,
                "up_sign_candidates": [1.0],
                "lock_up_sign": True,
                "up_sign_source": "sam3d_vehicle_local_positive_y_roof_prior",
                "forward_axis_idx": 2,
                "forward_sign": 1.0,
                "forward_sign_candidates": [1.0, -1.0],
                "right_axis_idx": 0,
            },
            "heading_prior": {"enabled": False},
        },
    }
    args = type(
        "Args",
        (),
        {
            "vehicle_mesh_axis_override_enabled": True,
            "vehicle_mesh_up_axis_idx": 1,
            "vehicle_mesh_up_sign": -1.0,
            "road_depth_fallback_enabled": False,
        },
    )()

    context = build_vehicle_pose_context(
        task=task,
        sample_dir=Path("E:/QingYan/Guanwu-master2/workspace/projects/video/codex_allframes_20260521_1600/outputs/08_pose_optimize/tasks/obj_000003@000001"),
        full_mask=np.zeros((10, 10), dtype=np.uint8),
        json_bbox=[1.0, 1.0, 8.0, 8.0],
        image_size=(10, 10),
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 5.0, "cy": 5.0},
        t_world_from_cam=np.eye(4, dtype=np.float64),
        args=args,
    )

    prior = context["mesh_axis_prior"]
    assert prior["up_sign"] == 1.0
    assert prior["up_sign_candidates"] == [1.0]
    assert prior["lock_up_sign"] is True


def test_bbox_motion_front_sign_does_not_penalize_same_half_plane_alignment() -> None:
    args = type(
        "Args",
        (),
        {
            "heading_prior_enabled": True,
            "mesh_tail_light_front_sign_enabled": False,
            "bbox_area_trend_front_sign_enabled": True,
            "bbox_area_trend_front_sign_min_confidence": 0.75,
            "bbox_area_trend_front_sign_min_monotonicity": 0.75,
            "bbox_area_trend_front_sign_min_axis_confidence": 0.50,
            "bbox_area_trend_front_sign_confidence_scale": 0.90,
            "heading_prior_sigma_deg": 25.0,
            "heading_prior_weight": 0.03,
            "front_sign_heading_prior_weight": 0.10,
            "front_sign_mismatch_penalty": 0.8,
            "front_sign_angle_penalty_weight": 1.2,
            "front_sign_hard_gate_enabled": True,
            "front_sign_hard_gate_angle_deg": 120.0,
            "front_sign_hard_gate_min_confidence": 0.25,
            "front_sign_depth_trend_enabled": False,
            "truncated_heading_prior_weight": 0.08,
            "road_constraint_enabled": False,
        },
    )()
    vehicle_pose_context = {
        "heading_prior": {
            "enabled": True,
            "confidence": 0.35,
            "vector_image": [0.7660444431, 0.6427876097],
            "bbox_area_trend": {
                "direction": "approaching",
                "confidence": 1.0,
                "monotonicity": 1.0,
                "truncated_tail": False,
            },
        }
    }
    mesh_meta = {
        "axis_prior": {
            "forward_axis_idx": 2,
            "forward_sign": 1.0,
            "confidence": 0.75,
        }
    }
    rotation_cam = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = compute_road_and_heading_score(
        translation_cam=np.zeros(3, dtype=np.float64),
        rotation_cam=rotation_cam,
        scale=np.ones(3, dtype=np.float64),
        t_world_from_cam=np.eye(4, dtype=np.float64),
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
        projected_bbox=[0.0, 0.0, 10.0, 10.0],
        image_size=(640, 360),
        truncation_info={"is_truncated": False},
        initializer_metadata={},
        args=args,
    )

    assert result["heading_front_sign_enabled"] is True
    assert 40.0 < result["heading_prior_angle_error_deg"] < 60.0
    assert result["heading_front_sign_hard_rejected"] is False
    assert result["heading_front_angle_penalty"] == 0.0
    assert result["heading_front_sign_penalty"] == 0.0


def test_bbox_area_trend_enables_front_sign_without_tail_light_prior() -> None:
    args = type(
        "Args",
        (),
        {
            "heading_prior_enabled": True,
            "mesh_tail_light_front_sign_enabled": True,
            "mesh_tail_light_front_sign_min_confidence": 0.2,
            "mesh_tail_light_front_sign_standalone_min_confidence": 0.75,
            "mesh_tail_light_front_sign_standalone_min_density_ratio": 5.0,
            "tail_light_motion_consistency_flip_enabled": True,
            "tail_light_motion_consistency_min_confidence": 0.6,
            "tail_light_motion_consistency_flip_margin": 0.2,
            "heading_prior_sigma_deg": 25.0,
            "heading_prior_weight": 0.03,
            "front_sign_heading_prior_weight": 0.10,
            "front_sign_mismatch_penalty": 0.8,
            "front_sign_angle_penalty_weight": 1.2,
            "front_sign_hard_gate_enabled": True,
            "front_sign_hard_gate_angle_deg": 120.0,
            "front_sign_hard_gate_min_confidence": 0.25,
            "front_sign_depth_trend_enabled": True,
            "front_sign_depth_trend_penalty": 0.45,
            "front_sign_depth_trend_hard_gate_score_min": 0.35,
            "front_sign_depth_trend_min_monotonicity": 0.75,
            "front_sign_depth_trend_hard_gate_min_heading_angle_deg": 0.0,
            "bbox_area_trend_front_sign_enabled": True,
            "bbox_area_trend_front_sign_min_confidence": 0.75,
            "bbox_area_trend_front_sign_min_monotonicity": 0.75,
            "bbox_area_trend_front_sign_confidence_scale": 0.7,
            "truncated_heading_prior_weight": 0.08,
            "road_constraint_enabled": False,
        },
    )()
    vehicle_pose_context = {
        "heading_prior": {
            "enabled": True,
            "confidence": 0.35,
            "vector_image": [1.0, 0.0],
            "bbox_area_trend": {
                "direction": "approaching",
                "confidence": 1.0,
                "monotonicity": 1.0,
                "truncated_tail": False,
            },
        },
        "mesh_tail_light_prior": {
            "available": False,
            "axis_idx": 2,
            "front_sign": 1.0,
            "confidence": 0.1,
            "strong_available": False,
            "density_ratio": 1.0,
        },
    }
    mesh_meta = {
        "axis_prior": {
            "forward_axis_idx": 2,
            "forward_sign": 1.0,
        }
    }

    reversed_result = compute_road_and_heading_score(
        translation_cam=np.zeros(3, dtype=np.float64),
        rotation_cam=np.eye(3, dtype=np.float64),
        scale=np.ones(3, dtype=np.float64),
        t_world_from_cam=np.eye(4, dtype=np.float64),
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
        projected_bbox=[0.0, 0.0, 10.0, 10.0],
        image_size=(640, 360),
        truncation_info={"is_truncated": False},
        initializer_metadata={},
        args=args,
    )
    correct_result = compute_road_and_heading_score(
        translation_cam=np.zeros(3, dtype=np.float64),
        rotation_cam=np.diag([1.0, 1.0, -1.0]),
        scale=np.ones(3, dtype=np.float64),
        t_world_from_cam=np.eye(4, dtype=np.float64),
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
        projected_bbox=[0.0, 0.0, 10.0, 10.0],
        image_size=(640, 360),
        truncation_info={"is_truncated": False},
        initializer_metadata={},
        args=args,
    )

    assert reversed_result["heading_front_sign_enabled"] is True
    assert reversed_result["heading_depth_trend_score"] < 0.35
    assert reversed_result["heading_front_sign_hard_rejected"] is True
    assert correct_result["heading_front_sign_enabled"] is True
    assert correct_result["heading_depth_trend_score"] > 0.65
    assert correct_result["heading_front_sign_hard_rejected"] is False


def test_pose_tracks_build_refined_object_trajectories() -> None:
    refined = ProjectExecutor._refined_trajectories_from_pose_tracks(
        {
            "obj_000001": {
                "pose_source": "edge_contour_fast_temporal",
                "frames": [
                    {
                        "frame_id": 1,
                        "timestamp_sec": 0.1,
                        "centroid_world": [1.0, 2.0, 3.0],
                        "rotation_matrix": np.eye(3).tolist(),
                        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.2, 1.2, 1.2],
                        "confidence": 0.9,
                        "source": "edge_contour_fast_temporal",
                        "quality": {"metrics": {"mask_iou": 0.8}},
                    }
                ],
            }
        }
    )

    assert list(refined) == ["obj_000001"]
    assert refined["obj_000001"][0]["frame_id"] == 1
    assert refined["obj_000001"][0]["centroid_world"] == [1.0, 2.0, 3.0]
    assert refined["obj_000001"][0]["position_xyz"] == [1.0, 2.0, 3.0]
    assert refined["obj_000001"][0]["scale"] == [1.2, 1.2, 1.2]
    assert refined["obj_000001"][0]["trajectory_source"] == "pose_optimize"
    assert refined["obj_000001"][0]["pose_source"] == "edge_contour_fast_temporal"


def test_vehicle_mesh_axis_prior_uses_local_positive_y_as_roof_up() -> None:
    verts = np.asarray(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-0.5, 0.5)
            for z in (-2.0, 2.0)
        ],
        dtype=np.float64,
    )
    observations = [
        {"frame_id": 1, "points": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 1.0]])},
        {"frame_id": 2, "points": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [1.0, 0.0, 2.0]])},
    ]

    rotation, mesh_basis, _world_basis, axis_roles, _heading = ProjectExecutor._pose_track_object_rotation(
        verts,
        observations,
        scene_up=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )
    prior = ProjectExecutor._mesh_axis_prior_for_pose_optimizer(verts, axis_roles=axis_roles)

    assert axis_roles["up_axis_idx"] == 1
    assert axis_roles["up_axis_sign"] == 1.0
    assert np.allclose(mesh_basis[:, 1], [0.0, 1.0, 0.0])
    assert float(rotation[:, 1] @ np.asarray([0.0, 1.0, 0.0])) > 0.99
    assert prior["up_sign"] == 1.0
    assert prior["up_sign_candidates"] == [1.0]
    assert prior["lock_up_sign"] is True


def test_signed_world_up_axis_supports_wildgs_negative_y_up() -> None:
    np.testing.assert_allclose(world_up_vector_from_arg("-y"), [0.0, -1.0, 0.0])
    np.testing.assert_allclose(world_up_vector_from_arg("+z"), [0.0, 0.0, 1.0])


def test_refined_object_trajectories_override_geometry_by_frame() -> None:
    coarse = {
        "obj_000001": [
            {"frame_id": 1, "timestamp_sec": 0.1, "centroid_world": [0.0, 0.0, 0.0]},
            {"frame_id": 2, "timestamp_sec": 0.2, "centroid_world": [2.0, 0.0, 0.0]},
        ],
        "obj_000099": [
            {"frame_id": 1, "timestamp_sec": 0.1, "centroid_world": [9.0, 0.0, 0.0]},
        ],
    }
    refined = {
        "obj_000001": [
            {
                "frame_id": 1,
                "timestamp_sec": 0.1,
                "centroid_world": [1.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "frame_id": 3,
                "timestamp_sec": 0.3,
                "centroid_world": [3.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        ]
    }

    merged = ProjectExecutor._merge_refined_object_trajectories(coarse, refined)

    assert [rec["frame_id"] for rec in merged["obj_000001"]] == [1, 2, 3]
    assert merged["obj_000001"][0]["centroid_world"] == [1.0, 0.0, 0.0]
    assert merged["obj_000001"][0]["trajectory_source"] == "pose_optimize"
    assert merged["obj_000001"][1]["centroid_world"] == [2.0, 0.0, 0.0]
    assert merged["obj_000099"][0]["centroid_world"] == [9.0, 0.0, 0.0]


def test_usd_export_preserves_pose_optimizer_mesh_local_forward_axis() -> None:
    verts = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    prepared = ProjectExecutor._prepare_usd_object_mesh_vertices(verts)

    assert prepared[1, 2] > prepared[0, 2]


def test_usd_export_preserves_pose_optimizer_mesh_local_origin() -> None:
    verts = np.asarray(
        [
            [10.0, 0.0, -1.0],
            [12.0, 1.0, 3.0],
            [11.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    prepared = ProjectExecutor._prepare_usd_object_mesh_vertices(verts)

    np.testing.assert_allclose(prepared, verts)


def test_pose_match_bbox_area_threshold_is_800_px() -> None:
    assert project_executor._POSE_MATCH_MIN_BBOX_AREA_PX == 800.0


def test_pose_target_frame_mode_reads_all_frames_env(monkeypatch) -> None:
    monkeypatch.setenv("GUANWU_POSE_TARGET_FRAME_MODE", "all_frames")

    assert ProjectExecutor._pose_target_frame_mode() == "all_frames"


def test_find_depth_map_for_task_skips_inaccessible_candidates(tmp_path: Path, monkeypatch) -> None:
    outputs_dir = tmp_path / "outputs"
    sample_dir = outputs_dir / "08_pose_optimize" / "tasks" / "obj_000002@000003"
    sample_dir.mkdir(parents=True)
    inaccessible = outputs_dir / "06_geometry_lift" / "wildgs" / "exports" / "depth_maps" / "depth_maps" / "00003.npy"

    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path == inaccessible:
            raise PermissionError("access denied")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert find_depth_map_for_task(sample_dir, 3) is None


def test_pose_all_frame_candidate_frame_ids_keep_vehicle_frames_above_area() -> None:
    detection_frames = [
        {
            "frame_idx": 1,
            "instances": [
                {
                    "object_id": "obj_000001",
                    "label": "car",
                    "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
                }
            ],
        },
        {
            "frame_idx": 2,
            "instances": [
                {
                    "object_id": "obj_000001",
                    "label": "car",
                    "bbox_xyxy": [10.0, 10.0, 20.0, 20.0],
                }
            ],
        },
        {
            "frame_idx": 3,
            "instances": [
                {
                    "object_id": "obj_000001",
                    "label": "fence",
                    "bbox_xyxy": [0.0, 0.0, 200.0, 40.0],
                }
            ],
        },
        {
            "frame_idx": 4,
            "instances": [
                {
                    "object_id": "obj_000001",
                    "label": "truck",
                    "bbox_xyxy": [0.0, 0.0, 120.0, 80.0],
                }
            ],
        },
    ]

    frame_ids = ProjectExecutor._pose_all_frame_candidate_frame_ids(
        obj_id="obj_000001",
        detection_frames=detection_frames,
        min_bbox_area_px=800.0,
    )

    assert frame_ids == [1, 4]


def test_pose_all_frame_candidate_frame_ids_read_detection_file(tmp_path: Path) -> None:
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "object_id": "obj_000001",
                        "label": "car",
                        "bbox_xyxy": [10.0, 10.0, 90.0, 70.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frame_ids = ProjectExecutor._pose_all_frame_candidate_frame_ids(
        obj_id="obj_000001",
        detection_frames=[{"frame_idx": 7, "detections": str(detections_path)}],
        min_bbox_area_px=800.0,
    )

    assert frame_ids == [7]


def test_pose_dynamic_window_radius_expands_for_truncated_or_small_targets() -> None:
    large_clear = {
        "bbox_xyxy": [120.0, 80.0, 260.0, 180.0],
        "image_width": 640,
        "image_height": 360,
    }
    right_truncated = {
        "bbox_xyxy": [540.0, 70.0, 640.0, 170.0],
        "image_width": 640,
        "image_height": 360,
    }
    small = {
        "bbox_xyxy": [100.0, 80.0, 130.0, 120.0],
        "image_width": 640,
        "image_height": 360,
    }

    assert ProjectExecutor._pose_dynamic_window_radius(
        large_clear,
        {"bbox_area_px": 14000.0},
        base_radius=2,
        min_radius=1,
        max_radius=4,
    ) == 1
    assert ProjectExecutor._pose_dynamic_window_radius(
        right_truncated,
        {"bbox_area_px": 10000.0},
        base_radius=2,
        min_radius=1,
        max_radius=4,
    ) == 3
    assert ProjectExecutor._pose_dynamic_window_radius(
        small,
        {"bbox_area_px": 1200.0},
        base_radius=2,
        min_radius=1,
        max_radius=4,
    ) == 3


def test_candidate_trajectory_selection_all_frames_does_not_require_target_frame() -> None:
    selected, summary = ProjectExecutor._select_edge_pose_candidate_trajectory(
        {
            1: [_pose_record(frame_id=1, x=0.0, yaw_deg=0.0)],
            4: [_pose_record(frame_id=4, x=0.5, yaw_deg=8.0)],
        },
        target_frame_id=None,
    )

    assert [record["frame_id"] for record in selected] == [1, 4]
    assert summary["target_frame_id"] is None
