from __future__ import annotations

import argparse

import numpy as np


def test_appearance_prior_scores_background_leakage_lower_than_target() -> None:
    from process.pose_optimizer.priors.image_appearance_prior import (
        AppearancePriorConfig,
        build_image_appearance_prior,
    )

    image = np.full((80, 100, 3), (40, 130, 40), dtype=np.uint8)
    image[25:55, 35:65] = (30, 30, 210)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[25:55, 35:65] = 1

    prior = build_image_appearance_prior(
        image,
        mask,
        [35.0, 25.0, 65.0, 55.0],
        config=AppearancePriorConfig(
            foreground_erode_kernel=3,
            background_inner_dilate_kernel=5,
            background_outer_dilate_kernel=17,
        ),
    )

    good_render = mask.copy()
    leaking_render = np.zeros_like(mask)
    leaking_render[20:60, 30:80] = 1
    good = prior.score_render_mask(good_render)
    leaking = prior.score_render_mask(leaking_render)

    assert good["appearance_confidence"] > 0.9
    assert good["appearance_score"] > leaking["appearance_score"]
    assert leaking["background_leakage"] > good["background_leakage"]
    assert leaking["appearance_score"] < 0.75


def test_appearance_prior_lowers_confidence_when_colors_are_similar() -> None:
    from process.pose_optimizer.priors.image_appearance_prior import (
        AppearancePriorConfig,
        build_image_appearance_prior,
    )

    image = np.full((80, 100, 3), (100, 104, 108), dtype=np.uint8)
    image[25:55, 35:65] = (102, 106, 110)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[25:55, 35:65] = 1

    prior = build_image_appearance_prior(
        image,
        mask,
        [35.0, 25.0, 65.0, 55.0],
        config=AppearancePriorConfig(
            foreground_erode_kernel=3,
            background_inner_dilate_kernel=5,
            background_outer_dilate_kernel=17,
        ),
    )
    score = prior.score_render_mask(mask)

    assert score["appearance_confidence"] < 0.30
    assert score["appearance_score"] >= 0.0


def test_depth_prior_scores_consistent_depth_and_disables_low_overlap() -> None:
    from process.pose_optimizer.priors.depth_consistency_prior import (
        DepthConsistencyConfig,
        DepthConsistencyPrior,
    )

    observed = np.full((32, 32), 4.0, dtype=np.float32)
    detection = np.zeros((32, 32), dtype=np.uint8)
    detection[8:24, 8:24] = 1
    prior = DepthConsistencyPrior(
        observed,
        detection,
        config=DepthConsistencyConfig(depth_sigma=0.5, min_valid_ratio=0.25),
    )

    render_mask = detection.copy()
    render_depth = np.zeros_like(observed)
    render_depth[render_mask > 0] = 4.1
    good = prior.score(render_depth, render_mask)

    sparse_mask = np.zeros_like(detection)
    sparse_mask[8:10, 8:10] = 1
    sparse_depth = np.zeros_like(observed)
    sparse_depth[sparse_mask > 0] = 4.0
    sparse = prior.score(sparse_depth, sparse_mask)

    assert good["depth_confidence"] == 1.0
    assert good["depth_score"] > 0.80
    assert good["depth_error"] < 0.11
    assert sparse["depth_confidence"] < 0.25
    assert sparse["depth_score"] == 0.0


def test_depth_render_face_limit_samples_deterministically() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_depth_render_faces

    faces = np.arange(300, dtype=np.int32).reshape(100, 3)

    limited = select_depth_render_faces(faces, 10)

    expected = faces[np.linspace(0, len(faces) - 1, 10, dtype=np.int64)]
    assert np.array_equal(limited, expected)
    assert np.array_equal(select_depth_render_faces(faces, 0), faces)
    assert np.array_equal(select_depth_render_faces(faces, 200), faces)


def test_support_plane_normal_is_oriented_toward_object_points() -> None:
    from process.pose_optimizer.priors.support_plane_prior import (
        SupportPlaneConfig,
        fit_support_plane_ransac,
    )

    xs, zs = np.meshgrid(np.linspace(-1.0, 1.0, 16), np.linspace(2.0, 4.0, 16))
    plane_points = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    object_points = np.array([[0.0, 0.6, 3.0], [0.2, 0.8, 3.1], [-0.2, 0.7, 2.9]], dtype=np.float64)

    plane = fit_support_plane_ransac(
        plane_points,
        config=SupportPlaneConfig(min_points=20, ransac_iters=16, ransac_threshold_m=0.02, min_confidence=0.5),
        object_points_cam=object_points,
    )

    signed_centroid = float(np.dot(plane["normal"], object_points.mean(axis=0)) + plane["offset"])
    assert plane["available"]
    assert signed_centroid > 0.0


def test_support_contact_score_reports_floating_and_penetration_penalties() -> None:
    from process.pose_optimizer.priors.support_plane_prior import support_contact_score

    plane = {
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "offset": 0.0,
        "support_plane_confidence": 1.0,
    }
    touching_points = np.array([[-0.2, 0.01, 2.0], [0.2, 0.02, 2.0], [0.0, 0.00, 2.1]], dtype=np.float64)
    floating_points = touching_points + np.array([0.0, 0.25, 0.0])
    penetrating_points = touching_points - np.array([0.0, 0.16, 0.0])

    touching = support_contact_score(
        touching_points,
        plane,
        sigma_m=0.10,
        tolerance_m=0.08,
        floating_tolerance_m=0.20,
        penetration_tolerance_m=0.10,
    )
    floating = support_contact_score(
        floating_points,
        plane,
        sigma_m=0.10,
        tolerance_m=0.08,
        floating_tolerance_m=0.20,
        penetration_tolerance_m=0.10,
    )
    penetrating = support_contact_score(
        penetrating_points,
        plane,
        sigma_m=0.10,
        tolerance_m=0.08,
        floating_tolerance_m=0.20,
        penetration_tolerance_m=0.10,
    )

    assert touching["support_contact_score"] > floating["support_contact_score"]
    assert touching["support_contact_coverage"] == 1.0
    assert floating["support_floating_distance_m"] > 0.20
    assert floating["support_penetration_distance_m"] == 0.0
    assert penetrating["support_penetration_distance_m"] > 0.10
    assert penetrating["support_floating_distance_m"] == 0.0
    assert penetrating["support_penalty"] > floating["support_penalty"] * 0.9


def test_support_bottom_points_use_local_axis_to_penalize_tilted_contact() -> None:
    from process.pose_optimizer.priors.support_plane_prior import support_contact_score
    from process.pose_optimizer.strategies.generic_appearance_temporal import support_bottom_points_for_pose

    xs, zs = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(1.8, 2.2, 9))
    bottom = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
    top = bottom + np.array([0.0, 0.4, 0.0], dtype=np.float64)
    vertices = np.concatenate([bottom, top], axis=0)
    plane = {
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "offset": 0.0,
        "support_plane_confidence": 1.0,
    }

    aligned = support_bottom_points_for_pose(
        vertices,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.ones(3, dtype=np.float64),
        plane,
        bottom_percentile=8.0,
        mode="local_axis",
    )
    theta = np.deg2rad(15.0)
    tilted_rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Put the lowest tilted edge on the plane; the opposite bottom edge floats.
    tilted_translation = np.array([0.0, 0.5 * np.sin(theta), 0.0], dtype=np.float64)
    tilted = support_bottom_points_for_pose(
        vertices,
        tilted_rotation,
        tilted_translation,
        np.ones(3, dtype=np.float64),
        plane,
        bottom_percentile=8.0,
        mode="local_axis",
    )

    aligned_contact = support_contact_score(
        aligned["support_points_cam"],
        plane,
        sigma_m=0.08,
        tolerance_m=0.06,
        floating_tolerance_m=0.15,
        penetration_tolerance_m=0.07,
    )
    tilted_contact = support_contact_score(
        tilted["support_points_cam"],
        plane,
        sigma_m=0.08,
        tolerance_m=0.06,
        floating_tolerance_m=0.15,
        penetration_tolerance_m=0.07,
    )

    assert aligned["support_axis_index"] == 1
    assert tilted["support_axis_index"] == 1
    assert aligned["support_normal_angle_deg"] < 1.0
    assert tilted["support_normal_angle_deg"] > 10.0
    assert aligned_contact["support_contact_coverage"] > 0.95
    assert tilted_contact["support_contact_coverage"] < 0.60
    assert aligned_contact["support_contact_score"] > tilted_contact["support_contact_score"]


def test_support_orientation_score_penalizes_axis_tilt_softly() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import support_orientation_score

    plane = {"normal": np.array([0.0, 1.0, 0.0], dtype=np.float64)}
    theta = np.deg2rad(20.0)
    tilted_rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    aligned = support_orientation_score(np.eye(3, dtype=np.float64), plane, sigma_deg=20.0, tolerance_deg=4.0)
    tilted = support_orientation_score(tilted_rotation, plane, sigma_deg=20.0, tolerance_deg=4.0)

    assert aligned["support_orientation_score"] == 1.0
    assert aligned["support_orientation_penalty"] == 0.0
    assert tilted["support_normal_angle_deg"] > 19.0
    assert tilted["support_orientation_score"] < aligned["support_orientation_score"]
    assert tilted["support_orientation_penalty"] > 0.0


def test_support_sample_region_uses_near_mask_ring_when_lower_band_is_sparse() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        estimate_support_plane_from_observed_depth,
    )

    depth = np.full((80, 80), 3.0, dtype=np.float32)
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[62:76, 32:48] = 1
    bbox = [32.0, 62.0, 48.0, 76.0]
    intrinsics = {"fx": 80.0, "fy": 80.0, "cx": 40.0, "cy": 40.0}
    args = argparse.Namespace(
        support_plane_enabled="auto",
        support_plane_min_points=120,
        support_plane_ransac_iters=24,
        support_plane_ransac_threshold_m=0.05,
        support_plane_residual_scale_m=0.08,
        support_plane_min_confidence=0.70,
        support_use_lower_band=True,
        support_lower_band_x_expand_ratio=0.20,
        support_lower_band_y_extend_ratio=0.60,
        support_use_near_mask_ring=True,
        support_near_mask_inner_kernel=9,
        support_near_mask_outer_kernel=31,
        support_bbox_expand_ratio=0.40,
        support_exclude_target_mask_dilate_kernel=9,
    )

    plane = estimate_support_plane_from_observed_depth(depth, mask, bbox, intrinsics, args)

    assert plane["num_points"] >= 120
    assert plane["support_sample_debug"]["near_mask_ring_pixels"] > 0
    assert plane["support_sample_debug"]["lower_band_pixels"] < plane["support_sample_debug"]["final_pixels"]


def test_support_plane_report_payload_omits_debug_region_arrays() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import support_plane_report_payload

    payload = support_plane_report_payload(
        {
            "available": True,
            "support_plane_confidence": 0.8,
            "support_sample_region": np.ones((4, 4), dtype=bool),
            "support_inlier_region": np.ones((4, 4), dtype=bool),
            "support_sample_debug": {
                "final_pixels": 10,
                "lower_band_region": np.ones((4, 4), dtype=bool),
                "near_mask_ring_region": np.ones((4, 4), dtype=bool),
            },
        }
    )

    assert "support_sample_region" not in payload
    assert "support_inlier_region" not in payload
    assert payload["support_sample_debug"] == {"final_pixels": 10}


def test_candidate_breakdown_row_includes_support_contact_score() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import generic_breakdown_row_values

    row = generic_breakdown_row_values(
        {
            "candidate_rank": 1,
            "score": 2.4,
            "mask_blend_score": 0.8,
            "bbox_iou": 0.9,
            "contour_score": 0.7,
            "edge_score": 0.6,
            "depth_score": 0.5,
            "depth_confidence": 1.0,
            "appearance_score": 0.4,
            "appearance_confidence": 0.8,
            "temporal_score": 0.3,
            "support_contact_score": 0.3942,
            "support_contact_penalty_eff": 0.0443,
            "acceptance_status": "accepted",
        },
        row=1,
    )

    assert "support" in row
    assert row["support"] == "0.394/-0.044"


def test_generic_history_row_keeps_stable_support_fields_when_missing() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import generic_optimization_history_row

    row = generic_optimization_history_row(
        "coarse",
        0,
        "initial",
        -1,
        {
            "score": 1.0,
            "mask_iou": 0.5,
            "bbox_iou": 0.5,
            "bbox_center_error_px": 0.0,
        },
        0.0,
    )

    assert "support_bottom_point_count" in row
    assert "support_orientation_penalty_eff" in row
    assert row["support_bottom_point_count"] is None
    assert row["support_orientation_penalty_eff"] is None


def test_generic_variant_is_registered_and_config_parses() -> None:
    from process.pose_optimizer.config import config_to_argv, load_config
    from process.pose_optimizer.variants import VARIANTS

    variant = VARIANTS["generic_appearance_temporal"]
    cfg = load_config(variant.config_path)
    argv = config_to_argv(cfg)

    assert variant.module_name.endswith(".generic_appearance_temporal")
    assert "--appearance_enabled" in argv
    assert "--mode" not in argv
    assert "--no-road_constraint_enabled" in argv
    assert "--no-heading_prior_enabled" in argv


def test_generic_grid_candidates_use_batch_gpu_prefilter_when_enabled(monkeypatch) -> None:
    from process.pose_optimizer.strategies import fast
    from process.pose_optimizer.strategies.generic_appearance_temporal import generate_generic_grid_candidates

    class DummyEvaluator:
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}

    calls = {"count": 0, "specs": 0}

    def fake_build_coarse_candidate_spec(**kwargs):
        return {
            "translation_cam": np.array([float(kwargs["target_uv"][0]) * 0.001, 0.0, float(kwargs["tz"])], dtype=np.float64),
            "rotation_cam": kwargs["rotation_cam"],
            "scale": kwargs["scale"],
            "initializer_metadata": kwargs["metadata"],
        }

    def fake_estimate_depth_guess(*_args, **_kwargs):
        return 4.0

    def fake_batch_prefilter(_evaluator, specs, _args):
        calls["count"] += 1
        calls["specs"] += len(specs)
        return [
            {
                "translation_cam": spec["translation_cam"],
                "rotation_cam": spec["rotation_cam"],
                "scale": spec["scale"],
                "initializer_metadata": spec["initializer_metadata"],
                "projected_bbox": [1.0, 1.0, 5.0, 5.0],
                "score": float(index + 1),
            }
            for index, spec in enumerate(specs)
        ]

    monkeypatch.setattr(fast, "build_coarse_candidate_spec", fake_build_coarse_candidate_spec)
    monkeypatch.setattr(fast, "estimate_depth_guess", fake_estimate_depth_guess)
    monkeypatch.setattr(fast, "batch_prefilter_initial_candidates", fake_batch_prefilter)

    args = argparse.Namespace(
        generic_yaw_degrees="0",
        generic_pitch_degrees="0",
        generic_roll_degrees="0",
        init_scale_factors="1.0",
        init_depth_factors="1.0",
        top_k_candidates=4,
        enable_batch_gpu_eval=True,
    )
    obs = {
        "mask_bbox": [10.0, 10.0, 30.0, 30.0],
        "bbox_center": np.array([20.0, 20.0], dtype=np.float64),
        "centroid": np.array([21.0, 20.0], dtype=np.float64),
    }
    mesh_meta = {
        "bounds": np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64),
        "center": np.zeros(3, dtype=np.float64),
    }

    candidates = generate_generic_grid_candidates(
        evaluator=DummyEvaluator(),
        obs=obs,
        mesh_meta=mesh_meta,
        base_scale=np.ones(3, dtype=np.float64),
        corrected_seed=None,
        args=args,
    )

    assert calls == {"count": 1, "specs": 2}
    assert len(candidates) == 2


def test_generic_coarse_scoring_skips_depth_and_appearance_heavy_terms() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    class ExplodingAppearance:
        appearance_confidence = 1.0
        fg_bg_distance = 1.0

        def score_render_mask(self, *_args, **_kwargs):
            raise AssertionError("appearance should not run during coarse generic scoring")

    class ExplodingDepth:
        def score(self, *_args, **_kwargs):
            raise AssertionError("depth should not run during coarse generic scoring")

    vertices = np.array(
        [
            [-0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 0.4, 0.0],
            [-0.2, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 1
    args = argparse.Namespace(
        edge_score_enabled=False,
        generic_coarse_scoring=True,
        depth_enabled=True,
        appearance_enabled=True,
        hard_mask_weight=0.3,
        generic_contour_sigma_px=4.0,
        generic_mask_weight=1.0,
        generic_bbox_weight=0.15,
        generic_contour_weight=0.35,
        generic_edge_weight=0.20,
        generic_depth_weight=0.35,
        generic_appearance_weight=0.25,
        temporal_enabled=False,
        generic_temporal_weight=0.55,
        generic_scale_prior_weight=0.30,
        support_plane_min_confidence=0.70,
        support_plane_weight=0.20,
        support_contact_sigma_m=0.08,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.0,
        generic_acceptance_min_bbox_iou=0.0,
        generic_acceptance_min_projection_valid_ratio=0.0,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
        depth_outlier_score_threshold=0.1,
        depth_outlier_penalty=0.0,
    )
    evaluator = GenericPoseEvaluator(
        vertices=vertices,
        faces=faces,
        mesh=None,
        full_mask=mask,
        soft_full_mask=mask.astype(np.float32),
        json_bbox=[24.0, 24.0, 40.0, 40.0],
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 32.0, "cy": 32.0},
        image_size=(64, 64),
        bbox_weight=0.0,
        hard_mask_weight=0.3,
        backend="triangle_fill",
        enable_bbox_prefilter=False,
        prefilter_bbox_iou_min=0.0,
        prefilter_center_factor=2.0,
        prefilter_size_ratio_min=0.1,
        prefilter_size_ratio_max=10.0,
        roi_iou_margin=4,
        disable_roi_iou=False,
        fast_float32=True,
        profile_timings=False,
        generic_args=args,
        temporal_prior=None,
        truncation_info={"is_truncated": False, "truncation_sides": []},
        edge_context=None,
        appearance_prior=ExplodingAppearance(),
        depth_prior=ExplodingDepth(),
        support_plane={"available": False, "support_plane_confidence": 0.0},
    )

    result = evaluator.evaluate_absolute(
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )

    assert result["appearance_confidence"] == 0.0
    assert result["depth_confidence"] == 0.0
    assert result["score"] >= 0.0


def test_generic_support_penalty_uses_normalized_observation_gate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    vertices = np.array(
        [
            [-0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 0.4, 0.0],
            [-0.2, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 1
    args = argparse.Namespace(
        edge_score_enabled=False,
        generic_coarse_scoring=False,
        depth_enabled=False,
        appearance_enabled=False,
        hard_mask_weight=0.3,
        generic_contour_sigma_px=4.0,
        generic_mask_weight=1.0,
        generic_bbox_weight=0.15,
        generic_contour_weight=0.35,
        generic_edge_weight=0.20,
        generic_depth_weight=0.35,
        generic_appearance_weight=0.25,
        temporal_enabled=False,
        generic_temporal_weight=0.55,
        generic_scale_prior_weight=0.30,
        support_plane_min_confidence=0.70,
        support_plane_weight=0.20,
        support_penalty_weight=0.15,
        support_bottom_percentile=3.0,
        support_contact_sigma_m=0.10,
        support_contact_tolerance_m=0.08,
        support_floating_tolerance_m=0.20,
        support_penetration_tolerance_m=0.10,
        support_contact_distance_score_weight=0.70,
        support_contact_coverage_score_weight=0.30,
        support_floating_penalty_weight=0.60,
        support_penetration_penalty_weight=1.00,
        optional_prior_gate_start=0.35,
        optional_prior_gate_range=0.45,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.0,
        generic_acceptance_min_bbox_iou=0.0,
        generic_acceptance_min_projection_valid_ratio=0.0,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
        depth_outlier_score_threshold=0.1,
        depth_outlier_penalty=0.0,
    )
    evaluator = GenericPoseEvaluator(
        vertices=vertices,
        faces=faces,
        mesh=None,
        full_mask=mask,
        soft_full_mask=mask.astype(np.float32),
        json_bbox=[24.0, 24.0, 40.0, 40.0],
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 32.0, "cy": 32.0},
        image_size=(64, 64),
        bbox_weight=0.0,
        hard_mask_weight=0.3,
        backend="triangle_fill",
        enable_bbox_prefilter=False,
        prefilter_bbox_iou_min=0.0,
        prefilter_center_factor=2.0,
        prefilter_size_ratio_min=0.1,
        prefilter_size_ratio_max=10.0,
        roi_iou_margin=4,
        disable_roi_iou=False,
        fast_float32=True,
        profile_timings=False,
        generic_args=args,
        temporal_prior=None,
        truncation_info={"is_truncated": False, "truncation_sides": []},
        edge_context=None,
        appearance_prior=None,
        depth_prior=None,
        support_plane={
            "available": True,
            "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "offset": 0.0,
            "support_plane_confidence": 1.0,
            "inlier_ratio": 1.0,
            "plane_residual_m": 0.0,
        },
    )

    touching = evaluator.evaluate_absolute(
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )
    floating = evaluator.evaluate_absolute(
        np.array([0.0, 0.25, 2.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )

    assert 0.0 < touching["observation_quality"] <= 1.0
    expected_gate = np.clip((touching["observation_quality"] - 0.35) / 0.45, 0.0, 1.0)
    assert np.isclose(touching["optional_prior_gate"], expected_gate)
    assert floating["support_floating_distance_m"] > touching["support_floating_distance_m"]
    assert floating["support_penalty"] > touching["support_penalty"]
    assert np.isclose(
        floating["support_contact_penalty_eff"],
        floating["support_plane_confidence"]
        * floating["optional_prior_gate"]
        * args.support_penalty_weight
        * floating["support_penalty"],
    )
    assert floating["score"] < touching["score"]


def test_generic_config_keeps_full_scoring_after_proxy_coarse_mode() -> None:
    from process.pose_optimizer.config import load_config
    from process.pose_optimizer.variants import VARIANTS

    cfg = load_config(VARIANTS["generic_appearance_temporal"].config_path)

    assert cfg["generic_coarse_scoring"] is False


def test_generic_config_uses_moderate_support_contact_defaults() -> None:
    from process.pose_optimizer.config import load_config
    from process.pose_optimizer.variants import VARIANTS

    cfg = load_config(VARIANTS["generic_appearance_temporal"].config_path)

    assert cfg["support_plane_weight"] == 0.30
    assert cfg["support_penalty_weight"] == 0.25
    assert cfg["support_contact_sigma_m"] == 0.08
    assert cfg["support_contact_tolerance_m"] == 0.06
    assert cfg["support_floating_tolerance_m"] == 0.15
    assert cfg["support_penetration_tolerance_m"] == 0.07


def test_generic_proxy_scoring_forces_light_mode_while_full_keeps_config(monkeypatch, tmp_path) -> None:
    from process.pose_optimizer.strategies import generic_appearance_temporal as generic

    captured = {}

    class DummyEvaluator:
        def __init__(self, *args, **kwargs):
            self.generic_args = kwargs["generic_args"]
            self.active_backend = None
            self.backend_preference = kwargs.get("backend", "")

        def close(self):
            pass

    def fake_generate_generic_grid_candidates(**kwargs):
        captured["proxy_coarse"] = bool(kwargs["evaluator"].generic_args.generic_coarse_scoring)
        return [
            {
                "score": 1.0,
                "mask_iou": 1.0,
                "soft_mask_iou": 1.0,
                "bbox_iou": 1.0,
                "translation_cam": np.array([0.0, 0.0, 2.0]),
                "rotation_cam": np.eye(3),
                "scale": np.ones(3),
                "projected_bbox": [1.0, 1.0, 8.0, 8.0],
                "initializer_metadata": {},
            }
        ]

    def fake_report(*_args, **_kwargs):
        return tmp_path / "debug.png"

    monkeypatch.setattr(generic.fast, "resolve_sample_dir", lambda value: tmp_path)
    monkeypatch.setattr(generic.fast, "read_json", lambda _path: {
        "task_id": "obj_000001@000001",
        "object_id": "obj_000001",
        "label": "object",
        "bbox_xyxy": [1.0, 1.0, 8.0, 8.0],
        "camera": {"fx": 10.0, "fy": 10.0, "cx": 5.0, "cy": 5.0, "T_world_from_cam": np.eye(4).tolist()},
        "corrected_pose": {"translation_world": [0.0, 0.0, 2.0], "rotation_matrix": np.eye(3).tolist(), "scale": [1.0, 1.0, 1.0]},
    })
    monkeypatch.setattr(generic.fast, "read_image", lambda *_args, **_kwargs: np.ones((10, 10, 3), dtype=np.uint8) * 127)
    monkeypatch.setattr(generic.fast, "image_size_from_task", lambda *_args, **_kwargs: (10, 10))
    monkeypatch.setattr(generic.fast, "paste_crop_mask_to_full_image", lambda *_args, **_kwargs: (np.ones((10, 10), dtype=np.uint8), {}))
    monkeypatch.setattr(generic.fast, "make_soft_mask", lambda mask: mask.astype(np.float32))
    monkeypatch.setattr(generic.fast, "extract_mask_observations", lambda _mask: {"mask_bbox": [1.0, 1.0, 8.0, 8.0], "bbox_center": np.array([4.5, 4.5]), "centroid": np.array([4.5, 4.5])})
    monkeypatch.setattr(generic.fast, "find_mesh_path", lambda *_args, **_kwargs: tmp_path / "object.glb")
    monkeypatch.setattr(generic.fast, "load_glb_as_mesh", lambda _path: type("Mesh", (), {"vertices": np.zeros((8, 3)), "faces": np.array([[0, 1, 2]])})())
    monkeypatch.setattr(generic.fast, "mesh_axis_metadata", lambda _vertices: {"bounds": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), "center": np.zeros(3)})
    monkeypatch.setattr(generic.fast, "build_proxy_mesh", lambda vertices, faces, target_faces: (vertices, faces))
    monkeypatch.setattr(generic, "build_image_appearance_prior", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generic, "load_observed_depth_for_task", lambda *_args, **_kwargs: (None, {"available": False}))
    monkeypatch.setattr(generic, "estimate_support_plane_from_observed_depth", lambda *_args, **_kwargs: {"available": False, "support_plane_confidence": 0.0})
    monkeypatch.setattr(generic, "_load_temporal_prior", lambda *_args, **_kwargs: (None, {"enabled": False}))
    monkeypatch.setattr(generic.temporal_fast, "detect_truncation", lambda **_kwargs: {"is_truncated": False, "truncation_sides": []})
    monkeypatch.setattr(generic.temporal_fast, "prepare_image_edge_map", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generic, "GenericPoseEvaluator", DummyEvaluator)
    monkeypatch.setattr(generic.fast, "corrected_pose_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generic, "generate_generic_grid_candidates", fake_generate_generic_grid_candidates)
    monkeypatch.setattr(generic.fast, "generate_initial_candidates", lambda **_kwargs: [])
    monkeypatch.setattr(generic, "augment_generic_rotation_candidates", lambda candidates, *_args, **_kwargs: candidates)
    monkeypatch.setattr(generic.temporal_fast, "merge_temporal_seed", lambda candidates, *_args, **_kwargs: candidates)
    monkeypatch.setattr(generic, "make_generic_temporal_seed", lambda *_args, **_kwargs: None)
    def fake_refine_candidate_stages(candidate, proxy_evaluator, full_evaluator, _args):
        captured["refine_proxy_coarse"] = bool(proxy_evaluator.generic_args.generic_coarse_scoring)
        captured["refine_full_coarse"] = bool(full_evaluator.generic_args.generic_coarse_scoring)
        return (
            candidate
            | {
                "candidate_rank": 1,
                "acceptance_status": "accepted",
                "rendered_mask": np.ones((10, 10), dtype=np.uint8),
            },
            [],
        )

    monkeypatch.setattr(generic, "refine_candidate_stages", fake_refine_candidate_stages)
    monkeypatch.setattr(generic.fast, "save_mask_comparison", lambda *_args, **_kwargs: tmp_path / "mask.png")
    monkeypatch.setattr(generic.fast, "save_result_collages", lambda **_kwargs: {"alignment_collage": tmp_path / "a.png", "pose_closeup_collage": tmp_path / "b.png", "model_reference_collage": tmp_path / "c.png"})
    monkeypatch.setattr(generic.temporal_fast, "save_temporal_edge_debug", lambda **_kwargs: tmp_path / "d.png")
    monkeypatch.setattr(generic.fast, "write_history_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generic, "save_generic_breakdown", fake_report)
    monkeypatch.setattr(generic.fast, "cleanup_result_images", lambda *_args, **_kwargs: None)

    parser = argparse.ArgumentParser()
    generic.temporal_fast.add_fast_arguments(parser)
    generic.temporal_fast.add_temporal_arguments(parser)
    generic.add_generic_arguments(parser)
    args = parser.parse_args([
        "--sample_dir", str(tmp_path),
        "--output_dir", str(tmp_path / "out"),
        "--no-generic_coarse_scoring",
        "--no-appearance_enabled",
        "--no-depth_enabled",
        "--no-edge_score_enabled",
        "--no-temporal_enabled",
        "--refine_top_k", "1",
        "--top_k_candidates", "1",
    ])

    generic.optimize_sample(args)

    assert captured["proxy_coarse"] is True
    assert captured["refine_proxy_coarse"] is True
    assert captured["refine_full_coarse"] is False


def test_select_generic_refine_candidates_keeps_corrected_and_temporal_seeds() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    def candidate(score: float, tx: float, source: str) -> dict[str, object]:
        return {
            "score": score,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": {"source": source},
        }

    candidates = [
        candidate(10.0, 0.0, "generic_grid"),
        candidate(9.0, 1.0, "generic_grid"),
        candidate(8.0, 2.0, "generic_grid"),
        candidate(7.0, 3.0, "generic_grid"),
        candidate(0.2, 4.0, "task_json_corrected_pose"),
    ]
    temporal_seed = candidate(0.1, 5.0, "temporal_prior")

    selected = select_generic_refine_candidates(candidates, refine_top_k=4, temporal_seed=temporal_seed)
    sources = [item.get("initializer_metadata", {}).get("source") for item in selected]

    assert len(selected) == 4
    assert "task_json_corrected_pose" in sources
    assert "temporal_prior" in sources


def test_select_generic_refine_candidates_uses_temporal_seed_when_single_refine() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    def candidate(score: float, tx: float, source: str) -> dict[str, object]:
        return {
            "score": score,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": {"source": source},
        }

    temporal_seed = candidate(0.1, 5.0, "temporal_prior")
    selected = select_generic_refine_candidates(
        [candidate(10.0, 0.0, "generic_grid"), candidate(0.2, 4.0, "task_json_corrected_pose")],
        refine_top_k=1,
        temporal_seed=temporal_seed,
    )

    assert len(selected) == 1
    assert selected[0]["initializer_metadata"]["source"] == "temporal_prior"


def test_generic_refine_uses_full_evaluator_for_fine_stage(monkeypatch) -> None:
    from process.pose_optimizer.strategies import generic_appearance_temporal as generic

    calls = []

    class DummyEvaluator:
        def __init__(self, name: str, coarse: bool):
            self.name = name
            self.generic_args = argparse.Namespace(generic_coarse_scoring=coarse)

        def evaluate_absolute(self, translation, rotation, scale, keep_mask=False):
            calls.append((self.name, bool(self.generic_args.generic_coarse_scoring), bool(keep_mask)))
            return {
                "score": 2.0 if self.name == "full" and not self.generic_args.generic_coarse_scoring else 1.0,
                "mask_iou": 0.9,
                "soft_mask_iou": 0.9,
                "bbox_iou": 0.9,
                "bbox_center_error_px": 0.0,
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "projected_bbox": [1.0, 1.0, 8.0, 8.0],
                "rendered_mask": np.ones((10, 10), dtype=np.uint8),
                "depth_score": 0.6 if self.name == "full" and not self.generic_args.generic_coarse_scoring else 0.0,
                "appearance_score": 0.7 if self.name == "full" and not self.generic_args.generic_coarse_scoring else 0.0,
            }

    def fake_stage(*, evaluator, base_translation_cam, base_rotation_cam, base_scale, stage_name, **_kwargs):
        calls.append((f"stage:{stage_name}", bool(evaluator.generic_args.generic_coarse_scoring), False))
        return (
            {
                "score": 1.0,
                "mask_iou": 0.9,
                "bbox_iou": 0.9,
                "translation_cam": np.asarray(base_translation_cam, dtype=np.float64),
                "rotation_cam": np.asarray(base_rotation_cam, dtype=np.float64),
                "scale": np.asarray(base_scale, dtype=np.float64),
            },
            [],
        )

    monkeypatch.setattr(generic, "generic_local_search_stage", fake_stage)
    args = argparse.Namespace(
        stage1_iters=1,
        stage2_iters=1,
        stage3_iters=1,
        step_decay=0.5,
        max_translation_delta=0.8,
        max_rotation_delta_deg=45.0,
        scale_min_factor=0.5,
        scale_max_factor=2.2,
        save_full_history=False,
    )
    coarse_result = {
        "translation_cam": np.array([0.0, 0.0, 2.0]),
        "rotation_cam": np.eye(3),
        "scale": np.ones(3),
        "initializer_metadata": {},
    }

    result, _history = generic.refine_candidate_stages(
        coarse_result,
        DummyEvaluator("proxy", True),
        DummyEvaluator("full", False),
        args,
    )

    assert ("stage:coarse", True, False) in calls
    assert ("stage:rotation", True, False) in calls
    assert ("stage:fine", False, False) in calls
    assert ("full", False, True) not in calls
    assert result["score"] == 1.0


def test_generic_temporal_score_omits_yaw_specific_term() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import compute_generic_temporal_score

    prior = {
        "translation_cam": np.array([0.0, 0.0, 5.0]),
        "rotation_cam": np.eye(3),
        "scale": np.array([1.0, 1.0, 1.0]),
    }
    theta = np.deg2rad(20.0)
    rot = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=np.float64,
    )
    args = argparse.Namespace(
        generic_temporal_translation_sigma=1.0,
        generic_temporal_depth_sigma=0.8,
        generic_temporal_rotation_sigma_deg=25.0,
        generic_temporal_scale_sigma_log=0.20,
    )

    score = compute_generic_temporal_score(
        np.array([0.1, 0.0, 5.2]),
        rot,
        np.array([1.05, 1.05, 1.05]),
        prior,
        args,
    )

    expected_loss = (np.linalg.norm([0.1, 0.0, 0.2]) / 1.0) ** 2
    expected_loss += (0.2 / 0.8) ** 2
    expected_loss += (20.0 / 25.0) ** 2
    expected_loss += (np.log(1.05) / 0.20) ** 2
    assert np.isclose(score["generic_temporal_loss"], expected_loss, atol=1e-6)
    assert np.isclose(score["generic_temporal_score"], np.exp(-expected_loss), atol=1e-6)


def test_generic_acceptance_treats_zero_bbox_center_error_as_valid() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = object.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 180.0
    evaluator.generic_args = argparse.Namespace(
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
    )

    decision = evaluator._acceptance(
        {
            "visible_mask_iou": 0.96,
            "bbox_iou": 1.0,
            "bbox_center_error_px": 0.0,
            "projection_valid_ratio": 0.80,
            "depth_confidence": 1.0,
            "depth_score": 0.56,
        }
    )

    assert decision == {"acceptance_status": "accepted", "reject_reasons": []}


def test_executor_generic_mode_uses_generic_acceptance_without_road_gates(monkeypatch) -> None:
    from guanwu.video.project.executor import ProjectExecutor

    monkeypatch.setenv("GUANWU_POSE_OPTIMIZER_MODE", "generic_appearance_temporal")
    assert ProjectExecutor._pose_optimizer_mode() == "generic_appearance_temporal"

    inst = {"object_id": "obj_1", "label": "book", "bbox_xyxy": [10.0, 12.0, 210.0, 52.0]}
    obs = {"bbox": inst["bbox_xyxy"], "bbox_area_px": 2352.0}
    assert ProjectExecutor._is_target_pose_candidate(inst, obs, generic_mode=True)
    assert not ProjectExecutor._is_target_frame_vehicle_candidate(inst, obs)

    report = {
        "json_bbox": inst["bbox_xyxy"],
        "metrics": {
            "mask_iou": 0.13,
            "soft_mask_iou": 0.16,
            "bbox_iou": 0.12,
            "bbox_center_error_px": 40.0,
            "projection_valid_ratio": 0.65,
            "depth_confidence": 0.0,
            "upright_angle_error_deg": 170.0,
            "ground_contact_max_abs_m": 9.0,
        },
        "optimized_corrected_pose_world": {
            "translation_world": [0.0, 0.0, 3.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }
    assert ProjectExecutor._generic_pose_optimizer_acceptance(report)["accepted"]


def test_generic_truncated_visible_bbox_becomes_primary_bbox_score() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        promote_truncated_visible_bbox_score,
    )

    result = {
        "bbox_iou": 0.25,
        "bbox_center_error_px": 80.0,
        "projected_bbox": [10.0, 20.0, 110.0, 240.0],
    }
    visible_bbox = {
        "visible_bbox_iou": 0.82,
        "visible_bbox_center_error_px": 6.0,
        "visible_projected_bbox": [12.0, 22.0, 108.0, 119.0],
        "visible_bbox_source": "visible_mask_bbox",
    }

    promote_truncated_visible_bbox_score(result, visible_bbox)

    assert result["bbox_iou"] == 0.82
    assert result["bbox_center_error_px"] == 6.0
    assert result["projected_bbox"] == [12.0, 22.0, 108.0, 119.0]
    assert result["bbox_score_source"] == "visible_mask_bbox"
    assert result["full_bbox_iou"] == 0.25
    assert result["full_bbox_center_error_px"] == 80.0
    assert result["full_projected_bbox"] == [10.0, 20.0, 110.0, 240.0]


def test_truncated_visible_bbox_uses_visible_region_silhouette() -> None:
    from process.pose_optimizer.strategies import temporal_fast

    rendered = np.zeros((100, 120), dtype=np.uint8)
    rendered[20:100, 30:80] = 1
    target = np.zeros_like(rendered)
    target[18:100, 28:82] = 1
    args = argparse.Namespace(ignore_truncated_border_band_px=16)
    truncation = {"is_truncated": True, "truncation_sides": ["bottom"]}

    score = temporal_fast.compute_visible_bbox_score(
        [30.0, 20.0, 80.0, 140.0],
        [28.0, 18.0, 82.0, 100.0],
        (120, 100),
        truncation,
        args,
        rendered_mask=rendered,
        target_mask=target,
    )

    assert score["visible_projected_bbox"] == [30.0, 20.0, 80.0, 84.0]
    assert score["visible_target_bbox"] == [28.0, 18.0, 82.0, 84.0]
