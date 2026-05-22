from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

import guanwu.video.project.executor as project_executor
from guanwu.video.project.executor import ProjectExecutor
from process.pose_optimizer.strategies.temporal_fast import (
    compute_road_and_heading_score,
    choose_best_refined_result,
    should_skip_disk_temporal_prior,
    select_pareto_refine_candidates,
)
from process.pose_optimizer.strategies.fast import build_vehicle_pose_context
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
