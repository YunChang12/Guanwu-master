from __future__ import annotations

import argparse
import json

import numpy as np
import pytest


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


def test_depth_prior_uses_eroded_mask_and_median_z_error_without_penalty() -> None:
    from process.pose_optimizer.priors.depth_consistency_prior import (
        DepthConsistencyConfig,
        DepthConsistencyPrior,
    )

    observed = np.full((40, 40), 4.0, dtype=np.float32)
    detection = np.zeros((40, 40), dtype=np.uint8)
    detection[8:32, 8:32] = 1
    observed[8:32, 8:32] = 4.2
    # Boundary pixels are intentionally noisy. Erosion should remove them from
    # the median z calculation.
    observed[8:11, 8:32] = 9.0
    observed[29:32, 8:32] = 9.0
    observed[8:32, 8:11] = 9.0
    observed[8:32, 29:32] = 9.0
    prior = DepthConsistencyPrior(
        observed,
        detection,
        config=DepthConsistencyConfig(
            depth_sigma=0.75,
            min_valid_ratio=0.10,
            mask_erode_px=5,
            error_mode="median_z",
        ),
    )

    render_mask = detection.copy()
    render_depth = np.zeros_like(observed)
    render_depth[render_mask > 0] = 4.0
    score = prior.score(render_depth, render_mask)

    assert score["depth_enabled"] is True
    assert score["depth_error"] == pytest.approx(0.2, abs=1e-4)
    assert score["median_rendered_depth"] == pytest.approx(4.0)
    assert score["median_observed_depth"] == pytest.approx(4.2)
    assert score["depth_score"] == pytest.approx(np.exp(-0.2 / 0.75), abs=1e-4)


def test_depth_prior_distribution_error_uses_inner_mask_and_trimmed_quantiles() -> None:
    from process.pose_optimizer.priors.depth_consistency_prior import (
        DepthConsistencyConfig,
        DepthConsistencyPrior,
    )

    observed = np.full((48, 48), 8.0, dtype=np.float32)
    detection = np.zeros((48, 48), dtype=np.uint8)
    detection[8:40, 8:40] = 1
    observed[8:40, 8:40] = 4.0
    observed[8:40, 24:40] = 4.4
    observed[8:12, 8:40] = 12.0
    observed[36:40, 8:40] = 12.0
    observed[8:40, 8:12] = 12.0
    observed[8:40, 36:40] = 12.0
    prior = DepthConsistencyPrior(
        observed,
        detection,
        config=DepthConsistencyConfig(
            depth_sigma=0.08,
            min_valid_ratio=0.10,
            mask_erode_px=5,
            error_mode="distribution",
        ),
    )

    render_mask = detection.copy()
    render_depth = np.zeros_like(observed)
    render_depth[render_mask > 0] = 4.0
    score = prior.score(render_depth, render_mask)

    assert score["depth_enabled"] is True
    assert score["median_observed_depth"] == pytest.approx(4.2, abs=1e-4)
    assert score["p25_observed_depth"] == pytest.approx(4.0, abs=1e-4)
    assert score["p75_observed_depth"] == pytest.approx(4.4, abs=1e-4)
    assert score["depth_error"] == pytest.approx(0.20, abs=1e-4)
    assert score["debug"]["trim_percentiles"] == [5.0, 95.0]


def test_depth_prior_disables_low_valid_ratio_without_penalizing_candidate() -> None:
    from process.pose_optimizer.priors.depth_consistency_prior import (
        DepthConsistencyConfig,
        DepthConsistencyPrior,
    )

    observed = np.zeros((40, 40), dtype=np.float32)
    detection = np.zeros((40, 40), dtype=np.uint8)
    detection[8:32, 8:32] = 1
    observed[18:20, 18:20] = 4.0
    prior = DepthConsistencyPrior(
        observed,
        detection,
        config=DepthConsistencyConfig(
            depth_sigma=0.75,
            min_valid_ratio=0.10,
            mask_erode_px=3,
            error_mode="median_z",
        ),
    )

    render_depth = np.full_like(observed, 4.0)
    score = prior.score(render_depth, detection)

    assert score["depth_enabled"] is False
    assert score["depth_confidence"] == 0.0
    assert score["depth_score"] == 0.0
    assert score["debug"]["reason"] == "low_valid_depth_ratio"


def test_depth_prior_scores_roi_cropped_depth_with_visible_region() -> None:
    from process.pose_optimizer.priors.depth_consistency_prior import (
        DepthConsistencyConfig,
        DepthConsistencyPrior,
    )

    observed = np.full((40, 40), 9.0, dtype=np.float32)
    detection = np.zeros((40, 40), dtype=np.uint8)
    detection[12:28, 10:30] = 1
    observed[detection > 0] = 1.5
    rendered = np.zeros_like(observed)
    rendered[detection > 0] = 1.55
    visible = np.zeros_like(detection)
    visible[14:26, 12:28] = 1

    full_prior = DepthConsistencyPrior(
        observed,
        detection,
        config=DepthConsistencyConfig(depth_sigma=0.25, min_valid_ratio=0.10, error_mode="median_z"),
    )
    full_score = full_prior.score(rendered, detection, visible_region=visible)

    roi = (8, 10, 32, 30)
    x1, y1, x2, y2 = roi
    roi_prior = DepthConsistencyPrior(
        observed[y1:y2, x1:x2],
        detection[y1:y2, x1:x2],
        config=DepthConsistencyConfig(depth_sigma=0.25, min_valid_ratio=0.10, error_mode="median_z"),
    )
    roi_score = roi_prior.score(
        rendered[y1:y2, x1:x2],
        detection[y1:y2, x1:x2],
        visible_region=visible[y1:y2, x1:x2],
    )

    assert roi_score["depth_enabled"] is True
    assert roi_score["depth_error"] == pytest.approx(full_score["depth_error"])
    assert roi_score["valid_depth_ratio"] == pytest.approx(full_score["valid_depth_ratio"])


def test_render_depth_for_pose_can_render_only_roi_matching_full_depth() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import render_depth_for_pose

    vertices = np.array(
        [
            [-0.2, -0.2, 0.0],
            [0.2, -0.2, 0.0],
            [0.2, 0.2, 0.0],
            [-0.2, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0}
    translation = np.array([0.0, 0.0, 2.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    scale = np.ones(3, dtype=np.float64)

    full = render_depth_for_pose(vertices, faces, translation, rotation, scale, intrinsics, (64, 64))
    roi = (20, 20, 44, 44)
    cropped = render_depth_for_pose(
        vertices,
        faces,
        translation,
        rotation,
        scale,
        intrinsics,
        (64, 64),
        roi_xyxy=roi,
    )
    x1, y1, x2, y2 = roi

    assert cropped.shape == (y2 - y1, x2 - x1)
    assert np.array_equal(cropped, full[y1:y2, x1:x2])


def test_da3_only_depth_loader_rejects_wildgs_task_depth_without_fallback(tmp_path) -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import load_observed_depth_for_task

    wildgs_depth = tmp_path / "06_geometry_lift" / "wildgs" / "exports" / "depth_maps" / "depth_maps" / "00000.npy"
    wildgs_depth.parent.mkdir(parents=True)
    np.save(wildgs_depth, np.full((8, 8), 2.0, dtype=np.float32))
    task = {
        "task_id": "obj_000001@000001",
        "frame_idx": 1,
        "depth_path": str(wildgs_depth),
    }
    args = argparse.Namespace(
        observed_depth_map_path="",
        depth_source="depth_anything3",
        depth_fallback_to_wildgs=False,
        depth_type="metric",
        depth_unit="meter",
    )

    depth, info = load_observed_depth_for_task(tmp_path / "outputs" / "08_pose_optimize" / "tasks" / "obj_000001@000001", task, args)

    assert depth is None
    assert info["available"] is False
    assert info["reason"] == "wildgs_depth_disallowed"


def test_da3_only_depth_loader_uses_wildgs_only_when_fallback_enabled(tmp_path) -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import load_observed_depth_for_task

    wildgs_depth = tmp_path / "06_geometry_lift" / "wildgs" / "exports" / "depth_maps" / "depth_maps" / "00000.npy"
    wildgs_depth.parent.mkdir(parents=True)
    np.save(wildgs_depth, np.full((8, 8), 2.0, dtype=np.float32))
    task = {
        "task_id": "obj_000001@000001",
        "frame_idx": 1,
        "depth_sources": {
            "primary_depth_path": str(tmp_path / "06_geometry_lift" / "depth_anything3" / "depth_maps" / "00000.npy"),
            "primary_depth_source": "depth_anything3",
            "wildgs_depth_path": str(wildgs_depth),
            "use_wildgs_depth": False,
        },
    }
    args = argparse.Namespace(
        observed_depth_map_path="",
        depth_source="depth_anything3",
        depth_fallback_to_wildgs=True,
        depth_type="metric",
        depth_unit="meter",
    )

    depth, info = load_observed_depth_for_task(tmp_path / "outputs" / "08_pose_optimize" / "tasks" / "obj_000001@000001", task, args)

    assert depth is not None
    assert float(depth[0, 0]) == 2.0
    assert info["available"] is True
    assert info["depth_source"] == "wildgs"
    assert info["fallback_to_wildgs"] is True


def test_support_depth_gate_rejects_wildgs_when_support_fallback_disabled() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_support_observed_depth

    args = argparse.Namespace(
        depth_source="depth_anything3",
        support_depth_source="depth_anything3",
        support_fallback_to_wildgs=False,
    )
    wildgs_depth = np.ones((8, 8), dtype=np.float32)
    depth_info = {
        "available": True,
        "depth_source": "wildgs",
        "fallback_to_wildgs": True,
        "path": "/tmp/06_geometry_lift/wildgs/exports/depth_maps/depth_maps/00000.npy",
    }

    support_depth, support_info = select_support_observed_depth(wildgs_depth, depth_info, args)

    assert support_depth is None
    assert support_info["support_enabled"] is False
    assert support_info["support_source"] == "depth_anything3"
    assert support_info["reason"] == "wildgs_depth_disallowed"


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


def test_support_plane_ransac_rejects_low_quality_plane() -> None:
    from process.pose_optimizer.priors.support_plane_prior import (
        SupportPlaneConfig,
        fit_support_plane_ransac,
    )

    rng = np.random.default_rng(42)
    points = rng.normal(size=(240, 3)).astype(np.float64)

    plane = fit_support_plane_ransac(
        points,
        config=SupportPlaneConfig(
            min_points=150,
            ransac_iters=24,
            ransac_threshold_m=0.02,
            min_confidence=0.0,
            min_inlier_ratio=0.35,
            max_plane_fit_rmse=0.10,
        ),
    )

    assert plane["available"] is False
    assert plane["reason"] == "low_plane_quality"
    assert plane["support_plane_inlier_ratio"] < 0.35 or plane["support_plane_rmse_m"] > 0.10


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


def test_support_bottom_points_respect_locked_mesh_up_axis() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        support_bottom_points_for_pose,
        support_orientation_score,
    )

    vertices = np.array(
        [
            [-0.5, -1.0, -0.2],
            [0.5, -1.0, -0.2],
            [-0.5, 1.0, 0.2],
            [0.5, 1.0, 0.2],
        ],
        dtype=np.float64,
    )
    plane = {
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "offset": 0.0,
        "support_plane_confidence": 1.0,
    }
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }

    bottom = support_bottom_points_for_pose(
        vertices,
        rotation,
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        plane,
        bottom_percentile=10.0,
        mode="local_axis",
        mesh_axis_prior=mesh_axis_prior,
    )
    orientation = support_orientation_score(rotation, plane, mesh_axis_prior=mesh_axis_prior)

    assert bottom["support_axis_index"] == 1
    assert bottom["support_axis_sign"] == 1.0
    assert orientation["support_axis_index"] == 1
    assert orientation["support_axis_sign"] == 1.0
    assert orientation["support_normal_angle_deg"] == pytest.approx(90.0)


def test_semantic_up_orientation_score_uses_locked_axis_without_support_plane() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import semantic_up_orientation_score

    mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }
    aligned_rotation = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    flipped_rotation = np.eye(3, dtype=np.float64)

    aligned = semantic_up_orientation_score(
        aligned_rotation,
        mesh_axis_prior,
        world_up_axis="-y",
        sigma_deg=25.0,
        tolerance_deg=10.0,
    )
    flipped = semantic_up_orientation_score(
        flipped_rotation,
        mesh_axis_prior,
        world_up_axis="-y",
        sigma_deg=25.0,
        tolerance_deg=10.0,
    )

    assert aligned["semantic_up_confidence"] == 1.0
    assert aligned["semantic_up_axis_index"] == 1
    assert aligned["semantic_up_axis_sign"] == 1.0
    assert aligned["semantic_up_angle_deg"] == pytest.approx(0.0)
    assert aligned["semantic_up_score"] == pytest.approx(1.0)
    assert aligned["semantic_up_penalty"] == pytest.approx(0.0)
    assert flipped["semantic_up_angle_deg"] == pytest.approx(180.0)
    assert flipped["semantic_up_score"] < 1e-10
    assert flipped["semantic_up_penalty"] == pytest.approx(1.0)


def test_generic_parser_defaults_keep_semantic_up_as_gate_not_score_penalty() -> None:
    from process.pose_optimizer.strategies import generic_appearance_temporal as generic

    parser = argparse.ArgumentParser()
    generic.add_generic_arguments(parser)
    args, _unknown = parser.parse_known_args([])

    assert args.semantic_up_penalty_weight == pytest.approx(0.0)
    assert args.generic_visual_rescue_current_mask_max == pytest.approx(0.86)
    assert args.generic_visual_rescue_current_bbox_max == pytest.approx(0.84)


def _semantic_up_flip_args(**overrides: object) -> argparse.Namespace:
    values = {
        "world_up_axis": "-y",
        "semantic_up_constraint_enabled": "auto",
        "semantic_up_sigma_deg": 25.0,
        "semantic_up_tolerance_deg": 10.0,
        "semantic_up_hard_gate_enabled": True,
        "semantic_up_hard_gate_max_angle_deg": 90.0,
        "semantic_up_flip_rescue_enabled": True,
        "semantic_up_flip_rescue_source_top_k": 8,
        "semantic_up_flip_rescue_min_mask_iou": 0.55,
        "semantic_up_flip_rescue_min_bbox_iou": 0.70,
        "semantic_up_flip_rescue_keep_top_k": 3,
        "semantic_up_flip_rescue_variants": "cam_z_180,cam_x_180,min_align",
        "semantic_up_flip_rescue_prior_paths": "",
        "top_k_candidates": 1,
        "refine_top_k": 2,
        "generic_depth_refine_bucket_top_k": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _FakeSemanticUpFlipEvaluator:
    def __init__(self) -> None:
        self.mesh_axis_prior = {
            "available": True,
            "up_axis_idx": 1,
            "up_sign": 1.0,
            "lock_up_sign": True,
        }
        self.t_world_from_cam = None
        self.calls: list[np.ndarray] = []

    def evaluate_absolute(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
        scale: np.ndarray,
    ) -> dict[str, object]:
        self.calls.append(np.asarray(rotation, dtype=np.float64))
        return {
            "translation_cam": np.asarray(translation, dtype=np.float64),
            "rotation_cam": np.asarray(rotation, dtype=np.float64),
            "scale": np.asarray(scale, dtype=np.float64),
            "projected_bbox": [0.0, 0.0, 1.0, 1.0],
            "score": 1.0,
            "mask_iou": 0.82,
            "mask_blend_score": 0.82,
            "bbox_iou": 0.84,
            "depth_score": 0.0,
            "depth_confidence": 0.0,
        }


def _reversed_semantic_up_candidate() -> dict[str, object]:
    return {
        "translation_cam": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "projected_bbox": [0.0, 0.0, 1.0, 1.0],
        "score": 1.5,
        "mask_iou": 0.86,
        "mask_blend_score": 0.86,
        "bbox_iou": 0.88,
        "depth_score": 0.0,
        "depth_confidence": 0.0,
        "initializer_metadata": {"source": "generic_grid"},
    }


def test_semantic_up_flip_rescue_builds_camera_z_seed_from_high_visual_reversed_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        annotate_candidate_semantic_up_for_ranking,
        make_semantic_up_flip_seeds,
    )

    args = _semantic_up_flip_args(semantic_up_flip_rescue_variants="cam_z_180")
    evaluator = _FakeSemanticUpFlipEvaluator()
    base = _reversed_semantic_up_candidate()
    annotate_candidate_semantic_up_for_ranking(base, evaluator.mesh_axis_prior, args)

    assert base["semantic_up_angle_deg"] == pytest.approx(180.0)
    assert base["semantic_up_candidate_reject_reason"] == "semantic_up_angle_above_threshold"

    seeds = make_semantic_up_flip_seeds([base], evaluator, args)

    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["semantic_up_flip_rescue_candidate_used"] is True
    assert seed["semantic_up_flip_variant"] == "cam_z_180"
    assert seed["semantic_up_angle_deg_before_flip"] == pytest.approx(180.0)
    assert seed["semantic_up_angle_deg_after_flip"] == pytest.approx(0.0)
    assert seed["semantic_up_candidate_reject_reason"] is None
    assert seed["initializer_metadata"]["source"] == "semantic_up_flip_rescue_seed"
    assert np.allclose(seed["rotation_cam"], np.diag([-1.0, -1.0, 1.0]))


def test_semantic_up_flip_rescue_uses_prior_report_pose_source(tmp_path) -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        load_semantic_up_flip_prior_candidates,
        make_semantic_up_flip_seeds,
    )

    report_path = tmp_path / "optimization_report.json"
    report_path.write_text(
        json.dumps(
            {
                "optimized_corrected_pose_world": {
                    "translation_world": [0.0, 0.0, 2.0],
                    "rotation_matrix": np.eye(3).tolist(),
                    "scale": [1.0, 1.0, 1.0],
                },
                "metrics": {
                    "mask_iou": 0.86,
                    "bbox_iou": 0.88,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _semantic_up_flip_args(
        semantic_up_flip_rescue_variants="cam_z_180",
        semantic_up_flip_rescue_prior_paths=str(report_path),
    )
    evaluator = _FakeSemanticUpFlipEvaluator()
    evaluator.t_world_from_cam = np.eye(4, dtype=np.float64)

    prior_candidates = load_semantic_up_flip_prior_candidates(evaluator, args, np.eye(4, dtype=np.float64))
    seeds = make_semantic_up_flip_seeds(prior_candidates, evaluator, args)

    assert len(prior_candidates) == 1
    assert prior_candidates[0]["initializer_metadata"]["source"] == "semantic_up_flip_prior_pose"
    assert len(seeds) == 1
    assert seeds[0]["initializer_metadata"]["base_source"] == "semantic_up_flip_prior_pose"
    assert seeds[0]["semantic_up_flip_variant"] == "cam_z_180"


def test_merge_protected_initial_candidates_preserves_semantic_up_flip_seed_with_colliding_forward_signature() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import merge_protected_initial_candidates

    visual = _reversed_semantic_up_candidate()
    visual["initializer_metadata"] = {"source": "visual_0"}
    seed = _reversed_semantic_up_candidate()
    seed["rotation_cam"] = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    seed["score"] = 0.5
    seed["initializer_metadata"] = {"source": "semantic_up_flip_rescue_seed"}

    merged = merge_protected_initial_candidates(
        [visual],
        support_aligned_seeds=[],
        depth_snapped_seeds=[],
        semantic_up_flip_seeds=[seed],
        args=_semantic_up_flip_args(top_k_candidates=1, refine_top_k=1, semantic_up_flip_rescue_keep_top_k=1),
    )
    sources = [item.get("initializer_metadata", {}).get("source") for item in merged]

    assert sources == ["visual_0", "semantic_up_flip_rescue_seed"]


def test_select_generic_refine_candidates_keeps_semantic_up_flip_seed() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    visual = _reversed_semantic_up_candidate()
    visual["initializer_metadata"] = {"source": "visual_0"}
    seed = _reversed_semantic_up_candidate()
    seed["rotation_cam"] = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    seed["score"] = 0.5
    seed["initializer_metadata"] = {"source": "semantic_up_flip_rescue_seed"}

    selected = select_generic_refine_candidates(
        [visual],
        refine_top_k=1,
        semantic_up_flip_seed=seed,
    )

    assert len(selected) == 1
    assert selected[0]["initializer_metadata"]["source"] == "semantic_up_flip_rescue_seed"


def test_generic_parser_defaults_enable_semantic_up_flip_rescue() -> None:
    from process.pose_optimizer.strategies import generic_appearance_temporal as generic

    parser = argparse.ArgumentParser()
    generic.add_generic_arguments(parser)
    args, _unknown = parser.parse_known_args([])

    assert args.semantic_up_flip_rescue_enabled is True
    assert args.semantic_up_flip_rescue_source_top_k == 8
    assert args.semantic_up_flip_rescue_min_mask_iou == pytest.approx(0.55)
    assert args.semantic_up_flip_rescue_min_bbox_iou == pytest.approx(0.70)
    assert args.semantic_up_flip_rescue_keep_top_k == 3
    assert args.semantic_up_flip_rescue_variants == "cam_z_180,cam_x_180,min_align"
    assert args.semantic_up_flip_rescue_prior_paths == ""


def test_locked_mesh_up_axis_rejects_sideways_contact_orientation() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        locked_up_axis_orientation_reject_reason,
    )

    mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }
    args = argparse.Namespace(
        support_acceptance_min_confidence=0.70,
        support_locked_up_max_angle_deg=50.0,
    )

    reason = locked_up_axis_orientation_reject_reason(
        {
            "support_plane_enabled": True,
            "support_plane_confidence": 0.95,
            "support_normal_angle_deg": 89.0,
        },
        mesh_axis_prior,
        args,
        "contact_calibration",
    )

    assert reason == "locked_up_axis_support_angle_above_threshold"
    assert (
        locked_up_axis_orientation_reject_reason(
            {
                "support_plane_enabled": True,
                "support_plane_confidence": 0.95,
                "support_normal_angle_deg": 35.0,
            },
            mesh_axis_prior,
            args,
            "contact_calibration",
        )
        is None
    )
    assert (
        locked_up_axis_orientation_reject_reason(
            {
                "support_plane_enabled": True,
                "support_plane_confidence": 0.95,
                "support_normal_angle_deg": 89.0,
            },
            mesh_axis_prior,
            args,
            "free_motion",
        )
        is None
    )


def test_generic_pose_acceptance_rejects_locked_up_axis_sideways_contact() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = GenericPoseEvaluator.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 100.0
    evaluator.truncation_info = {"is_truncated": False}
    evaluator.mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }
    evaluator.generic_args = argparse.Namespace(
        generic_pose_motion_phase="contact_calibration",
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_min_threshold=0.0,
        generic_contact_depth_min_score=0.70,
        generic_acceptance_depth_confidence_high=999.0,
        generic_contact_support_max_separation_m=0.05,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
        support_locked_up_max_angle_deg=50.0,
    )

    acceptance = evaluator._acceptance(
        {
            "visible_mask_iou": 0.88,
            "mask_iou": 0.88,
            "bbox_iou": 0.90,
            "bbox_center_error_px": 4.0,
            "projection_valid_ratio": 0.95,
            "depth_confidence": 0.0,
            "depth_score": 1.0,
            "support_plane_enabled": True,
            "support_plane_confidence": 0.95,
            "support_floating_distance_m": 0.0,
            "support_penetration_distance_m": 0.0,
            "support_normal_angle_deg": 89.0,
        }
    )

    assert acceptance["acceptance_status"] == "rejected"
    assert "locked_up_axis_support_angle_above_threshold" in acceptance["reject_reasons"]


def test_generic_pose_acceptance_rejects_semantic_up_flip_without_support_plane() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = GenericPoseEvaluator.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 100.0
    evaluator.truncation_info = {"is_truncated": False}
    evaluator.mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }
    evaluator.generic_args = argparse.Namespace(
        generic_pose_motion_phase="free_motion",
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_min_threshold=0.0,
        generic_contact_depth_min_score=0.70,
        generic_acceptance_depth_confidence_high=999.0,
        generic_contact_support_max_separation_m=0.05,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
        support_locked_up_max_angle_deg=50.0,
        semantic_up_hard_gate_enabled=True,
        semantic_up_hard_gate_max_angle_deg=135.0,
    )

    acceptance = evaluator._acceptance(
        {
            "visible_mask_iou": 0.88,
            "mask_iou": 0.88,
            "bbox_iou": 0.90,
            "bbox_center_error_px": 4.0,
            "projection_valid_ratio": 0.95,
            "depth_confidence": 0.0,
            "depth_score": 1.0,
            "support_plane_enabled": False,
            "support_plane_confidence": 0.0,
            "support_floating_distance_m": 0.0,
            "support_penetration_distance_m": 0.0,
            "semantic_up_confidence": 1.0,
            "semantic_up_angle_deg": 180.0,
        }
    )

    assert acceptance["acceptance_status"] == "rejected"
    assert "semantic_up_angle_above_threshold" in acceptance["reject_reasons"]


def test_generic_pose_acceptance_rejects_large_semantic_up_deviation_without_support_plane() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = GenericPoseEvaluator.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 100.0
    evaluator.truncation_info = {"is_truncated": False}
    evaluator.mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }
    evaluator.generic_args = argparse.Namespace(
        generic_pose_motion_phase="free_motion",
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_min_threshold=0.0,
        generic_contact_depth_min_score=0.70,
        generic_acceptance_depth_confidence_high=999.0,
        generic_contact_support_max_separation_m=0.05,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
        support_locked_up_max_angle_deg=50.0,
        semantic_up_hard_gate_enabled=True,
        semantic_up_hard_gate_max_angle_deg=90.0,
    )

    acceptance = evaluator._acceptance(
        {
            "visible_mask_iou": 0.88,
            "mask_iou": 0.88,
            "bbox_iou": 0.90,
            "bbox_center_error_px": 4.0,
            "projection_valid_ratio": 0.95,
            "depth_confidence": 1.0,
            "depth_score": 0.96,
            "support_plane_enabled": False,
            "support_plane_confidence": 0.0,
            "support_floating_distance_m": 0.0,
            "support_penetration_distance_m": 0.0,
            "semantic_up_confidence": 1.0,
            "semantic_up_angle_deg": 146.0,
        }
    )

    assert acceptance["acceptance_status"] == "rejected"
    assert "semantic_up_angle_above_threshold" in acceptance["reject_reasons"]


def test_visual_rescue_keeps_current_when_challenger_violates_semantic_up_gate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_visual_rescue_best_candidate

    current = {
        "score": 2.10,
        "mask_iou": 0.733,
        "bbox_iou": 0.712,
        "edge_score": 0.152,
        "contour_score": 0.203,
        "depth_score": 0.980,
        "acceptance_status": "accepted",
        "reject_reasons": [],
    }
    visually_better = {
        "score": 1.80,
        "mask_iou": 0.883,
        "bbox_iou": 0.879,
        "edge_score": 0.345,
        "contour_score": 0.585,
        "depth_score": 0.958,
        "acceptance_status": "rejected",
        "reject_reasons": ["semantic_up_angle_above_threshold"],
        "candidate_rank": 2,
    }
    args = argparse.Namespace(
        generic_visual_rescue_enabled=True,
        generic_visual_rescue_min_depth_score=0.90,
        generic_visual_rescue_current_edge_max=0.18,
        generic_visual_rescue_current_contour_max=0.25,
        generic_visual_rescue_min_visual_margin=0.20,
        generic_mask_weight=0.90,
        generic_bbox_weight=0.20,
        generic_contour_weight=0.45,
        generic_edge_weight=0.30,
    )

    selected, _history = select_visual_rescue_best_candidate(
        [(current, [{"step": "current"}]), (visually_better, [{"step": "visual"}])],
        (current, [{"step": "current"}]),
        args,
    )

    assert selected is current
    assert "visual_rescue_candidate_used" not in visually_better


def test_visual_rescue_keeps_current_when_challenger_depth_is_poor() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_visual_rescue_best_candidate

    current = {
        "score": 2.10,
        "mask_iou": 0.733,
        "bbox_iou": 0.712,
        "edge_score": 0.152,
        "contour_score": 0.203,
        "depth_score": 0.980,
        "acceptance_status": "accepted",
        "reject_reasons": [],
    }
    poor_depth = {
        "score": 1.80,
        "mask_iou": 0.883,
        "bbox_iou": 0.879,
        "edge_score": 0.345,
        "contour_score": 0.585,
        "depth_score": 0.50,
        "acceptance_status": "rejected",
        "reject_reasons": ["semantic_up_angle_above_threshold"],
    }
    args = argparse.Namespace(
        generic_visual_rescue_enabled=True,
        generic_visual_rescue_min_depth_score=0.90,
        generic_visual_rescue_current_edge_max=0.18,
        generic_visual_rescue_current_contour_max=0.25,
        generic_visual_rescue_min_visual_margin=0.20,
        generic_mask_weight=0.90,
        generic_bbox_weight=0.20,
        generic_contour_weight=0.45,
        generic_edge_weight=0.30,
    )

    selected, _history = select_visual_rescue_best_candidate(
        [(current, []), (poor_depth, [])],
        (current, []),
        args,
    )

    assert selected is current


def test_visual_rescue_can_replace_mask_bbox_weak_current_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_visual_rescue_best_candidate

    current = {
        "score": 2.10,
        "mask_iou": 0.80,
        "bbox_iou": 0.78,
        "edge_score": 0.42,
        "contour_score": 0.40,
        "depth_score": 0.97,
        "acceptance_status": "accepted",
        "reject_reasons": [],
    }
    visually_better = {
        "score": 1.95,
        "mask_iou": 0.87,
        "bbox_iou": 0.88,
        "edge_score": 0.37,
        "contour_score": 0.52,
        "depth_score": 0.96,
        "semantic_up_candidate_reject_reason": None,
        "acceptance_status": "accepted",
        "reject_reasons": [],
        "candidate_rank": 2,
    }
    args = argparse.Namespace(
        generic_visual_rescue_enabled=True,
        generic_visual_rescue_min_depth_score=0.90,
        generic_visual_rescue_current_edge_max=0.18,
        generic_visual_rescue_current_contour_max=0.25,
        generic_visual_rescue_current_mask_max=0.86,
        generic_visual_rescue_current_bbox_max=0.84,
        generic_visual_rescue_min_visual_margin=0.05,
        generic_mask_weight=0.90,
        generic_bbox_weight=0.20,
        generic_contour_weight=0.45,
        generic_edge_weight=0.30,
    )

    selected, _history = select_visual_rescue_best_candidate(
        [(current, [{"step": "current"}]), (visually_better, [{"step": "visual"}])],
        (current, [{"step": "current"}]),
        args,
    )

    assert selected is visually_better
    assert visually_better["visual_rescue_candidate_used"] is True


def test_candidate_semantic_up_annotation_marks_flipped_candidate_for_ranking() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import annotate_candidate_semantic_up_for_ranking

    args = argparse.Namespace(
        semantic_up_constraint_enabled="auto",
        semantic_up_sigma_deg=25.0,
        semantic_up_tolerance_deg=10.0,
        semantic_up_hard_gate_enabled=True,
        semantic_up_hard_gate_max_angle_deg=90.0,
        world_up_axis="-y",
    )
    mesh_axis_prior = {
        "available": True,
        "up_axis_idx": 1,
        "up_sign": 1.0,
        "lock_up_sign": True,
    }
    aligned = {"rotation_cam": np.diag([1.0, -1.0, 1.0]).astype(np.float64)}
    flipped = {"rotation_cam": np.eye(3, dtype=np.float64)}

    annotate_candidate_semantic_up_for_ranking(aligned, mesh_axis_prior, args)
    annotate_candidate_semantic_up_for_ranking(flipped, mesh_axis_prior, args)

    assert aligned["semantic_up_angle_deg"] == pytest.approx(0.0)
    assert aligned["semantic_up_candidate_reject_reason"] is None
    assert flipped["semantic_up_angle_deg"] == pytest.approx(180.0)
    assert flipped["semantic_up_candidate_reject_reason"] == "semantic_up_angle_above_threshold"


def test_select_generic_refine_candidates_prefers_semantic_up_valid_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    visually_strong_flipped = {
        "score": 10.0,
        "depth_score": 0.90,
        "depth_confidence": 1.0,
        "mask_blend_score": 0.95,
        "bbox_iou": 0.95,
        "contour_score": 0.90,
        "appearance_score": 0.90,
        "appearance_confidence": 1.0,
        "semantic_up_confidence": 1.0,
        "semantic_up_angle_deg": 180.0,
        "semantic_up_candidate_reject_reason": "semantic_up_angle_above_threshold",
        "translation_cam": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "initializer_metadata": {"source": "generic_grid"},
    }
    visually_weaker_valid = {
        "score": 8.0,
        "depth_score": 0.85,
        "depth_confidence": 1.0,
        "mask_blend_score": 0.70,
        "bbox_iou": 0.72,
        "contour_score": 0.65,
        "appearance_score": 0.70,
        "appearance_confidence": 1.0,
        "semantic_up_confidence": 1.0,
        "semantic_up_angle_deg": 5.0,
        "semantic_up_candidate_reject_reason": None,
        "translation_cam": np.array([1.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.diag([1.0, -1.0, 1.0]).astype(np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "initializer_metadata": {"source": "generic_grid"},
    }

    selected = select_generic_refine_candidates(
        [visually_strong_flipped, visually_weaker_valid],
        refine_top_k=1,
    )

    assert selected[0] is visually_weaker_valid


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


def test_align_rotation_to_support_normal_reduces_axis_tilt() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        align_rotation_to_support_normal,
        support_orientation_score,
    )

    plane = {"normal": np.array([0.0, 1.0, 0.0], dtype=np.float64)}
    theta = np.deg2rad(18.0)
    tilted_rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    aligned_rotation, metadata = align_rotation_to_support_normal(tilted_rotation, plane)
    before = support_orientation_score(tilted_rotation, plane)
    after = support_orientation_score(aligned_rotation, plane)

    assert before["support_normal_angle_deg"] > 17.0
    assert after["support_normal_angle_deg"] < 1.0
    assert metadata["support_normal_angle_deg_before_alignment"] > 17.0
    assert metadata["support_normal_angle_deg_after_alignment"] < 1.0
    assert metadata["support_alignment_axis_index"] == 1


def test_make_support_aligned_seed_evaluates_corrected_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import make_support_aligned_seed

    plane = {"normal": np.array([0.0, 1.0, 0.0], dtype=np.float64), "support_plane_confidence": 0.9}
    theta = np.deg2rad(16.0)
    tilted_rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    class DummyEvaluator:
        support_plane = plane

        def evaluate_absolute(self, translation, rotation, scale):
            from process.pose_optimizer.strategies.generic_appearance_temporal import support_orientation_score

            result = {
                "score": 0.9,
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "projected_bbox": [0.0, 0.0, 10.0, 10.0],
            }
            result.update(support_orientation_score(rotation, plane))
            result["support_plane_confidence"] = plane["support_plane_confidence"]
            return result

    args = argparse.Namespace(
        support_aligned_seed_enabled=True,
        support_plane_min_confidence=0.70,
        support_alignment_trigger_deg=6.0,
    )
    base = {
        "score": 1.0,
        "translation_cam": np.array([0.0, 0.0, 4.0], dtype=np.float64),
        "rotation_cam": tilted_rotation,
        "scale": np.ones(3, dtype=np.float64),
        "support_normal_angle_deg": 16.0,
        "support_axis_index": 1,
        "support_axis_sign": 1.0,
        "initializer_metadata": {"source": "temporal_prior"},
    }

    seed = make_support_aligned_seed(base, DummyEvaluator(), args)

    assert seed is not None
    assert seed["support_normal_angle_deg"] < 1.0
    assert seed["support_aligned_candidate_used"] is True
    assert seed["support_normal_angle_deg_before_alignment"] > 15.0
    assert seed["support_normal_angle_deg_after_alignment"] < 1.0
    assert seed["initializer_metadata"]["source"] == "support_aligned_seed"
    assert seed["initializer_metadata"]["base_source"] == "temporal_prior"


def test_make_support_aligned_seed_snaps_translation_to_support_plane() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        make_support_aligned_seed,
        support_bottom_points_for_pose,
    )

    plane = {
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "offset": 0.0,
        "support_plane_confidence": 0.9,
    }

    class DummyEvaluator:
        support_plane = plane
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

        def evaluate_absolute(self, translation, rotation, scale):
            from process.pose_optimizer.strategies.generic_appearance_temporal import support_orientation_score

            bottom = support_bottom_points_for_pose(
                self.vertices,
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64),
                np.asarray(scale, dtype=np.float64),
                plane,
            )
            result = {
                "score": 0.9,
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "projected_bbox": [0.0, 0.0, 10.0, 10.0],
                "support_bottom_signed_m": bottom["support_bottom_signed_m"],
            }
            result.update(support_orientation_score(rotation, plane))
            result["support_plane_confidence"] = plane["support_plane_confidence"]
            return result

    args = argparse.Namespace(
        support_aligned_seed_enabled=True,
        support_plane_min_confidence=0.70,
        support_alignment_trigger_deg=6.0,
        support_contact_snap_enabled=True,
    )
    theta = np.deg2rad(16.0)
    tilted_rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    base = {
        "score": 1.0,
        "translation_cam": np.array([0.0, 1.2, 4.0], dtype=np.float64),
        "rotation_cam": tilted_rotation,
        "scale": np.ones(3, dtype=np.float64),
        "support_normal_angle_deg": 16.0,
        "support_axis_index": 1,
        "support_axis_sign": 1.0,
        "initializer_metadata": {"source": "temporal_prior"},
    }

    seed = make_support_aligned_seed(base, DummyEvaluator(), args)

    assert seed is not None
    assert abs(seed["support_bottom_signed_m"]) < 1e-9
    assert np.allclose(seed["translation_cam"], [0.0, 0.5, 4.0])
    assert seed["initializer_metadata"]["support_contact_snap_applied"] is True


def test_contact_snap_rescue_snaps_visual_candidate_to_support_plane() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        make_contact_snap_rescue_candidate,
        support_bottom_points_for_pose,
    )

    plane = {
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "offset": 0.0,
        "support_plane_confidence": 0.95,
    }

    class DummyEvaluator:
        support_plane = plane
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

        def evaluate_absolute(self, translation, rotation, scale, keep_mask=False):
            bottom = support_bottom_points_for_pose(
                self.vertices,
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64),
                np.asarray(scale, dtype=np.float64),
                plane,
            )
            support_separation = abs(float(bottom["support_bottom_signed_m"]))
            result = {
                "score": 2.0 - support_separation,
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "mask_iou": 0.90,
                "soft_mask_iou": 0.90,
                "bbox_iou": 0.94,
                "bbox_center_error_px": 2.0,
                "projected_bbox": [0.0, 0.0, 10.0, 10.0],
                "rendered_mask": np.ones((8, 8), dtype=np.uint8) if keep_mask else None,
                "support_plane_enabled": True,
                "support_plane_confidence": plane["support_plane_confidence"],
                "support_bottom_signed_m": bottom["support_bottom_signed_m"],
                "support_floating_distance_m": max(float(bottom["support_bottom_signed_m"]), 0.0),
                "support_penetration_distance_m": max(-float(bottom["support_bottom_signed_m"]), 0.0),
                "depth_confidence": 1.0,
                "depth_score": 0.8,
                "acceptance_status": "accepted" if support_separation <= 0.05 else "rejected",
                "reject_reasons": [] if support_separation <= 0.05 else ["contact_support_separation_above_threshold"],
            }
            if not keep_mask:
                result.pop("rendered_mask", None)
            return result

    args = argparse.Namespace(
        generic_pose_motion_phase="contact_calibration",
        support_contact_snap_enabled=True,
        generic_contact_snap_rescue_enabled=True,
        support_acceptance_min_confidence=0.70,
        generic_contact_support_max_separation_m=0.05,
    )
    base = {
        "score": 1.2,
        "translation_cam": np.array([0.0, 1.2, 4.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "mask_iou": 0.90,
        "soft_mask_iou": 0.90,
        "bbox_iou": 0.94,
        "bbox_center_error_px": 2.0,
        "projected_bbox": [0.0, 0.0, 10.0, 10.0],
        "support_plane_enabled": True,
        "support_plane_confidence": 0.95,
        "support_bottom_signed_m": 0.7,
        "support_floating_distance_m": 0.7,
        "support_penetration_distance_m": 0.0,
        "initializer_metadata": {"source": "visual_candidate"},
    }

    rescued = make_contact_snap_rescue_candidate(base, DummyEvaluator(), args)

    assert rescued is not None
    assert rescued["acceptance_status"] == "accepted"
    assert abs(rescued["support_bottom_signed_m"]) < 1e-9
    assert np.allclose(rescued["translation_cam"], [0.0, 0.5, 4.0])
    assert rescued["initializer_metadata"]["source"] == "contact_snap_rescue"
    assert rescued["initializer_metadata"]["base_source"] == "visual_candidate"
    assert rescued["contact_snap_rescue_applied"] is True


def test_contact_snap_rescue_preserves_projection_by_scaling_depth_and_mesh() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        make_contact_snap_rescue_candidate,
        support_bottom_points_for_pose,
    )

    plane = {
        "normal": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "offset": -1.0,
        "support_plane_confidence": 0.95,
    }

    class DummyEvaluator:
        support_plane = plane
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

        def evaluate_absolute(self, translation, rotation, scale, keep_mask=False):
            bottom = support_bottom_points_for_pose(
                self.vertices,
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64),
                np.asarray(scale, dtype=np.float64),
                plane,
            )
            support_separation = abs(float(bottom["support_bottom_signed_m"]))
            scale_value = float(np.median(np.asarray(scale, dtype=np.float64)))
            projected_bbox = [10.0, 20.0, 110.0, 120.0]
            return {
                "score": 2.0 - support_separation,
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "mask_iou": 0.91,
                "soft_mask_iou": 0.91,
                "bbox_iou": 0.95,
                "bbox_center_error_px": 1.0,
                "projected_bbox": projected_bbox,
                "support_plane_enabled": True,
                "support_plane_confidence": plane["support_plane_confidence"],
                "support_bottom_signed_m": bottom["support_bottom_signed_m"],
                "support_floating_distance_m": max(float(bottom["support_bottom_signed_m"]), 0.0),
                "support_penetration_distance_m": max(-float(bottom["support_bottom_signed_m"]), 0.0),
                "depth_confidence": 1.0,
                "depth_score": 0.8,
                "acceptance_status": "accepted" if support_separation <= 0.05 else "rejected",
                "reject_reasons": [] if support_separation <= 0.05 else ["contact_support_separation_above_threshold"],
                "uniform_scale_debug": scale_value,
            }

    args = argparse.Namespace(
        generic_pose_motion_phase="contact_calibration",
        support_contact_snap_enabled=True,
        generic_contact_snap_rescue_enabled=True,
        generic_contact_scale_depth_rescue_enabled=True,
        support_acceptance_min_confidence=0.70,
        generic_contact_support_max_separation_m=0.05,
    )
    base = {
        "score": 1.2,
        "translation_cam": np.array([0.0, 2.0, 4.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "mask_iou": 0.91,
        "soft_mask_iou": 0.91,
        "bbox_iou": 0.95,
        "bbox_center_error_px": 1.0,
        "projected_bbox": [10.0, 20.0, 110.0, 120.0],
        "support_plane_enabled": True,
        "support_plane_confidence": 0.95,
        "support_bottom_signed_m": 0.5,
        "support_floating_distance_m": 0.5,
        "support_penetration_distance_m": 0.0,
        "initializer_metadata": {"source": "visual_candidate"},
    }

    rescued = make_contact_snap_rescue_candidate(base, DummyEvaluator(), args)

    assert rescued is not None
    assert rescued["acceptance_status"] == "accepted"
    assert abs(rescued["support_bottom_signed_m"]) < 1e-9
    assert np.allclose(rescued["translation_cam"], [0.0, 4.0 / 3.0, 8.0 / 3.0])
    assert np.allclose(rescued["scale"], [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert rescued["initializer_metadata"]["source"] == "contact_scale_depth_rescue"
    assert rescued["contact_scale_depth_rescue_applied"] is True


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


def test_background_geometry_reference_builds_camera_support_plane() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        support_plane_from_background_geometry_reference,
    )

    task = {
        "vehicle_pose_context": {
            "background_geometry_reference": {
                "schema": "guanwu.background_geometry_reference.v1",
                "reference_type": "support_surface",
                "source": "clean_background_depth",
                "normal_world": [0.0, -2.0, 0.0],
                "offset": 0.5,
                "support_plane_confidence": 0.91,
            }
        }
    }
    t_world_from_cam = np.eye(4, dtype=np.float64)
    t_world_from_cam[:3, 3] = [0.0, 1.0, 0.0]

    plane = support_plane_from_background_geometry_reference(task, t_world_from_cam)

    assert plane["available"] is True
    assert plane["source"] == "background_geometry_reference"
    assert np.allclose(plane["normal"], [0.0, -1.0, 0.0])
    assert np.isclose(plane["offset"], -0.75)
    assert plane["support_plane_confidence"] == 0.91


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


def test_generic_coarse_scoring_skips_cpu_heavy_terms(monkeypatch) -> None:
    from process.pose_optimizer.strategies import generic_appearance_temporal as generic
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    def explode(*_args, **_kwargs):
        raise AssertionError("coarse scoring should not run this CPU-heavy term")

    monkeypatch.setattr(GenericPoseEvaluator, "_contour_score_cached", explode)
    monkeypatch.setattr(GenericPoseEvaluator, "_support_contact", explode)
    monkeypatch.setattr(generic.temporal_fast, "compute_edge_score", explode)
    monkeypatch.setattr(generic, "projection_valid_ratio", explode)

    vertices = np.array(
        [
            [-0.2, -0.2, 0.0],
            [0.2, -0.2, 0.0],
            [0.2, 0.2, 0.0],
            [-0.2, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 1
    args = argparse.Namespace(
        edge_score_enabled=True,
        generic_coarse_scoring=True,
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
        optional_prior_gate_start=0.35,
        optional_prior_gate_range=0.45,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.0,
        generic_acceptance_min_bbox_iou=0.0,
        generic_acceptance_min_projection_valid_ratio=0.50,
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
        edge_context={"roi": [0, 0, 64, 64]},
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

    result = evaluator.evaluate_absolute(
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )

    assert result["contour_score"] == 0.0
    assert result["edge_confidence"] == 0.0
    assert result["support_plane_enabled"] is False
    assert result["projection_valid_ratio"] == 1.0
    assert result["acceptance_status"] == "accepted"


def test_generic_rotation_grid_uses_batch_evaluation_when_enabled() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import augment_generic_rotation_candidates

    class DummyEvaluator:
        def __init__(self):
            self.batch_calls = 0

        def evaluate_absolute(self, *_args, **_kwargs):
            raise AssertionError("rotation grid should use evaluate_absolute_batch")

        def evaluate_absolute_batch(self, translations, rotations, scales, batch_size=32):
            self.batch_calls += 1
            return [
                {
                    "translation_cam": np.asarray(translation, dtype=np.float64),
                    "rotation_cam": np.asarray(rotation, dtype=np.float64),
                    "scale": np.asarray(scale, dtype=np.float64),
                    "projected_bbox": [1.0, 1.0, 5.0, 5.0],
                    "score": float(index + 1),
                    "mask_iou": 0.8,
                    "soft_mask_iou": 0.8,
                    "bbox_iou": 0.8,
                    "bbox_center_error_px": 0.0,
                }
                for index, (translation, rotation, scale) in enumerate(zip(translations, rotations, scales))
            ]

    base = {
        "translation_cam": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "projected_bbox": [1.0, 1.0, 5.0, 5.0],
        "score": 0.5,
        "initializer_metadata": {"source": "base"},
    }
    evaluator = DummyEvaluator()
    args = argparse.Namespace(
        generic_rotation_grid_enabled=True,
        generic_rotation_grid_source_top_k=1,
        generic_yaw_degrees="0,45",
        generic_pitch_degrees="0",
        generic_roll_degrees="0",
        top_k_candidates=4,
        enable_batch_gpu_eval=True,
        batch_gpu_size=8,
    )

    candidates = augment_generic_rotation_candidates([base], evaluator, args)

    assert evaluator.batch_calls == 1
    assert len(candidates) == 2
    assert any(
        item.get("initializer_metadata", {}).get("source") == "generic_rotation_grid"
        for item in candidates
    )


def test_generic_refine_candidate_stages_light_search_then_single_full_score() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import refine_candidate_stages

    class DummyEvaluator:
        def __init__(self, coarse: bool, label: str):
            self.generic_args = argparse.Namespace(generic_coarse_scoring=coarse)
            self.label = label
            self.delta_modes: list[bool] = []
            self.absolute_modes: list[bool] = []

        def set_initializer_metadata(self, _metadata):
            pass

        def evaluate_delta(self, base_translation, base_rotation, base_scale, params, keep_mask=False):
            self.delta_modes.append(bool(self.generic_args.generic_coarse_scoring))
            return self._result(base_translation, base_rotation, base_scale, score=1.0, keep_mask=keep_mask)

        def evaluate_absolute(self, translation, rotation, scale, keep_mask=False):
            self.absolute_modes.append(bool(self.generic_args.generic_coarse_scoring))
            return self._result(translation, rotation, scale, score=2.0, keep_mask=keep_mask)

        def _result(self, translation, rotation, scale, *, score: float, keep_mask: bool):
            result = {
                "score": score,
                "mask_iou": 0.8,
                "soft_mask_iou": 0.8,
                "bbox_iou": 0.8,
                "bbox_center_error_px": 0.0,
                "projected_bbox": [1.0, 1.0, 5.0, 5.0],
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "acceptance_status": "accepted",
            }
            if keep_mask:
                result["rendered_mask"] = np.ones((8, 8), dtype=np.uint8)
            return result

    candidate = {
        "translation_cam": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "initializer_metadata": {"source": "candidate"},
    }
    proxy = DummyEvaluator(coarse=True, label="proxy")
    full = DummyEvaluator(coarse=False, label="full")
    args = argparse.Namespace(
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=0,
        step_decay=0.5,
        max_translation_delta=0.8,
        max_rotation_delta_deg=45.0,
        scale_min_factor=0.5,
        scale_max_factor=2.2,
        save_full_history=False,
        generic_lightweight_search_scoring=True,
    )

    refined, _history = refine_candidate_stages(candidate, proxy, full, args)

    assert full.delta_modes == [True, True]
    assert full.absolute_modes == [False]
    assert full.generic_args.generic_coarse_scoring is False
    assert refined["score"] == 2.0
    assert "rendered_mask" in refined


def test_generic_refine_contact_phase_uses_full_scoring_during_fine_search() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import refine_candidate_stages

    class DummyEvaluator:
        def __init__(self, coarse: bool, label: str):
            self.generic_args = argparse.Namespace(generic_coarse_scoring=coarse)
            self.label = label
            self.delta_modes: list[bool] = []
            self.absolute_modes: list[bool] = []

        def set_initializer_metadata(self, _metadata):
            pass

        def evaluate_delta(self, base_translation, base_rotation, base_scale, params, keep_mask=False):
            self.delta_modes.append(bool(self.generic_args.generic_coarse_scoring))
            return self._result(base_translation, base_rotation, base_scale, score=1.0, keep_mask=keep_mask)

        def evaluate_delta_batch(self, base_translation, base_rotation, base_scale, params_batch, **_kwargs):
            self.delta_modes.extend([bool(self.generic_args.generic_coarse_scoring)] * len(params_batch))
            return [
                self._result(base_translation, base_rotation, base_scale, score=1.0, keep_mask=False)
                for _params in params_batch
            ]

        def evaluate_absolute(self, translation, rotation, scale, keep_mask=False):
            self.absolute_modes.append(bool(self.generic_args.generic_coarse_scoring))
            return self._result(translation, rotation, scale, score=2.0, keep_mask=keep_mask)

        def _result(self, translation, rotation, scale, *, score: float, keep_mask: bool):
            result = {
                "score": score,
                "mask_iou": 0.8,
                "soft_mask_iou": 0.8,
                "bbox_iou": 0.8,
                "bbox_center_error_px": 0.0,
                "projected_bbox": [1.0, 1.0, 5.0, 5.0],
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "support_plane_enabled": True,
                "support_plane_confidence": 0.95,
                "support_bottom_signed_m": 0.0,
                "support_floating_distance_m": 0.0,
                "support_penetration_distance_m": 0.0,
                "acceptance_status": "accepted",
            }
            if keep_mask:
                result["rendered_mask"] = np.ones((8, 8), dtype=np.uint8)
            return result

    candidate = {
        "translation_cam": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "initializer_metadata": {"source": "candidate"},
    }
    proxy = DummyEvaluator(coarse=True, label="proxy")
    full = DummyEvaluator(coarse=False, label="full")
    args = argparse.Namespace(
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=0,
        step_decay=0.5,
        max_translation_delta=0.8,
        max_rotation_delta_deg=45.0,
        scale_min_factor=0.5,
        scale_max_factor=2.2,
        save_full_history=False,
        generic_lightweight_search_scoring=True,
        generic_pose_motion_phase="contact_calibration",
        generic_contact_snap_rescue_enabled=True,
        support_contact_snap_enabled=True,
        support_acceptance_min_confidence=0.70,
        generic_contact_support_max_separation_m=0.05,
    )

    refined, _history = refine_candidate_stages(candidate, proxy, full, args)

    assert full.delta_modes == [False, False]
    assert full.absolute_modes == [False]
    assert refined["score"] == 2.0
    assert "rendered_mask" in refined


def test_generic_refine_depth_enabled_uses_full_scoring_during_fine_search() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import refine_candidate_stages

    class DummyEvaluator:
        def __init__(self, coarse: bool, label: str):
            self.generic_args = argparse.Namespace(generic_coarse_scoring=coarse)
            self.label = label
            self.delta_modes: list[bool] = []
            self.absolute_modes: list[bool] = []

        def set_initializer_metadata(self, _metadata):
            pass

        def evaluate_delta(self, base_translation, base_rotation, base_scale, params, keep_mask=False):
            self.delta_modes.append(bool(self.generic_args.generic_coarse_scoring))
            return self._result(base_translation, base_rotation, base_scale, score=1.0, keep_mask=keep_mask)

        def evaluate_delta_batch(self, base_translation, base_rotation, base_scale, params_batch, **_kwargs):
            self.delta_modes.extend([bool(self.generic_args.generic_coarse_scoring)] * len(params_batch))
            return [
                self._result(base_translation, base_rotation, base_scale, score=1.0, keep_mask=False)
                for _params in params_batch
            ]

        def evaluate_absolute(self, translation, rotation, scale, keep_mask=False):
            self.absolute_modes.append(bool(self.generic_args.generic_coarse_scoring))
            return self._result(translation, rotation, scale, score=2.0, keep_mask=keep_mask)

        def _result(self, translation, rotation, scale, *, score: float, keep_mask: bool):
            result = {
                "score": score,
                "mask_iou": 0.8,
                "soft_mask_iou": 0.8,
                "bbox_iou": 0.8,
                "bbox_center_error_px": 0.0,
                "projected_bbox": [1.0, 1.0, 5.0, 5.0],
                "translation_cam": np.asarray(translation, dtype=np.float64),
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "depth_enabled": True,
                "depth_confidence": 1.0,
                "depth_score": 1.0,
                "depth_error": 0.0,
                "valid_depth_ratio": 0.8,
                "acceptance_status": "accepted",
            }
            if keep_mask:
                result["rendered_mask"] = np.ones((8, 8), dtype=np.uint8)
            return result

    candidate = {
        "translation_cam": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "initializer_metadata": {"source": "candidate"},
    }
    proxy = DummyEvaluator(coarse=True, label="proxy")
    full = DummyEvaluator(coarse=False, label="full")
    args = argparse.Namespace(
        stage1_iters=0,
        stage2_iters=0,
        stage3_iters=0,
        step_decay=0.5,
        max_translation_delta=0.8,
        max_rotation_delta_deg=45.0,
        scale_min_factor=0.5,
        scale_max_factor=2.2,
        save_full_history=False,
        generic_lightweight_search_scoring=True,
        generic_pose_motion_phase="free_motion",
        depth_enabled=True,
        generic_depth_refine_fine_full_scoring=True,
    )

    refined, _history = refine_candidate_stages(candidate, proxy, full, args)

    assert full.delta_modes == [False, False]
    assert full.absolute_modes == [False]
    assert refined["score"] == 2.0
    assert "rendered_mask" in refined


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


def test_generic_config_enables_depth_and_disables_support_defaults() -> None:
    from process.pose_optimizer.config import load_config
    from process.pose_optimizer.variants import VARIANTS

    cfg = load_config(VARIANTS["generic_appearance_temporal"].config_path)

    assert cfg["depth_enabled"] is True
    assert cfg["generic_depth_weight"] > 0.0
    assert cfg["support_plane_enabled"] == "disabled"
    assert cfg["support_plane_weight"] == 0.0
    assert cfg["support_penalty_weight"] == 0.0
    assert cfg["support_contact_sigma_m"] == 0.08
    assert cfg["support_contact_tolerance_m"] == 0.06
    assert cfg["support_floating_tolerance_m"] == 0.15
    assert cfg["support_penetration_tolerance_m"] == 0.07
    assert cfg["support_orientation_penalty_weight"] == 0.0
    assert cfg["support_orientation_sigma_deg"] == 8.0
    assert cfg["support_orientation_tolerance_deg"] == 2.0
    assert cfg["support_aligned_seed_enabled"] is False
    assert cfg["support_alignment_trigger_deg"] == 6.0
    assert cfg["support_aligned_seed_score_margin"] == 0.20
    assert cfg["support_aligned_seed_source_top_k"] == 2
    assert cfg["support_contact_snap_enabled"] is False
    assert cfg["generic_contact_snap_rescue_enabled"] is False
    assert cfg["generic_contact_snap_rescue_iters"] == 0
    assert cfg["generic_contact_snap_rescue_min_mask_iou"] == 0.12
    assert cfg["generic_contact_snap_rescue_min_bbox_iou"] == 0.10
    assert cfg["generic_contact_scale_depth_rescue_enabled"] is True
    assert cfg["generic_contact_scale_depth_rescue_min_factor"] == 0.25
    assert cfg["generic_contact_scale_depth_rescue_max_factor"] == 2.50
    assert cfg["support_acceptance_min_confidence"] == 0.70
    assert cfg["support_acceptance_max_separation_m"] == 0.15
    assert cfg["generic_acceptance_truncated_min_projection_valid_ratio"] == 0.30
    assert cfg["generic_acceptance_projection_temporal_exempt_enabled"] is True
    assert cfg["generic_truncated_temporal_early_stop_enabled"] is True
    assert cfg["generic_prefer_temporal_refine_first"] is True
    assert cfg["pytorch3d_bin_size"] is None
    assert cfg["save_color_soft_mask"] is False
    assert cfg["save_fg_bg_samples"] is False
    assert cfg["save_candidate_appearance_overlay"] is False
    assert cfg["save_score_breakdown"] is False


def test_generic_motion_constraints_keep_target_depth_in_free_motion() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import apply_generic_task_motion_constraints

    args = argparse.Namespace(
        generic_pose_motion_phase="auto",
        depth_enabled=True,
        generic_depth_weight=0.15,
        generic_free_motion_depth_enabled=True,
        generic_free_motion_depth_weight=0.60,
        support_plane_enabled="auto",
        support_plane_weight=0.12,
        support_penalty_weight=0.15,
        support_orientation_penalty_weight=0.35,
        support_aligned_seed_enabled=True,
        generic_locked_scale_init_factors="0.95,1.0,1.05",
        scale_min_factor=0.5,
        scale_max_factor=2.2,
        generic_scale_prior_weight=0.30,
        generic_scale_prior_sigma_log=0.20,
    )
    task = {
        "vehicle_pose_context": {
            "generic_pose_motion_phase": "free_motion",
            "track_scale_prior": {
                "scale": [0.1, 0.1, 0.1],
                "source": "contact_supported_scale",
                "sample_count": 10,
                "frame_ids": [1, 2, 3],
            },
        }
    }

    report = apply_generic_task_motion_constraints(args, task)

    assert report["phase"] == "free_motion"
    assert args.depth_enabled is True
    assert args.generic_depth_weight == 0.60
    assert args.support_plane_enabled == "disabled"
    assert args.support_plane_weight == 0.0
    assert args.support_penalty_weight == 0.0
    assert args.support_orientation_penalty_weight == 0.0
    assert args.generic_scale_lock_applied is True
    assert task["corrected_pose"]["scale"] == [0.1, 0.1, 0.1]


def test_generic_motion_constraints_do_not_reenable_disabled_support_in_contact_calibration() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import apply_generic_task_motion_constraints

    args = argparse.Namespace(
        generic_pose_motion_phase="auto",
        depth_enabled=True,
        generic_acceptance_depth_confidence_high=999.0,
        generic_depth_weight=0.55,
        generic_mask_weight=0.80,
        generic_appearance_weight=0.25,
        support_plane_enabled="disabled",
        support_plane_weight=0.0,
        support_penalty_weight=0.0,
        support_orientation_penalty_weight=0.0,
        support_aligned_seed_enabled=False,
    )
    task = {
        "vehicle_pose_context": {
            "generic_pose_motion_phase": "contact_calibration",
        }
    }

    report = apply_generic_task_motion_constraints(args, task)

    assert report["phase"] == "contact_calibration"
    assert args.depth_enabled is True
    assert args.generic_depth_weight >= 1.10
    assert args.support_plane_enabled == "disabled"
    assert args.support_plane_weight == 0.0
    assert args.support_penalty_weight == 0.0
    assert args.support_orientation_penalty_weight == 0.0
    assert args.support_aligned_seed_enabled is False


def test_depth_snapped_candidate_shifts_camera_z_by_median_depth_error() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        make_depth_snapped_candidate,
    )

    class DummyEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate_absolute(self, translation, rotation, scale):
            translation = np.asarray(translation, dtype=np.float64)
            self.calls.append(translation.copy())
            depth_error = float(translation[2] - 3.0)
            return {
                "score": 0.8,
                "translation_cam": translation,
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "projected_bbox": [1.0, 1.0, 10.0, 10.0],
                "depth_enabled": True,
                "depth_confidence": 1.0,
                "depth_score": float(np.exp(-abs(depth_error) / 0.06)),
                "depth_error": abs(depth_error),
                "median_rendered_depth": float(translation[2]),
                "median_observed_depth": 3.0,
            }

    candidate = {
        "score": 0.9,
        "translation_cam": np.array([0.2, -0.1, 3.25], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "projected_bbox": [1.0, 1.0, 10.0, 10.0],
        "depth_enabled": True,
        "depth_confidence": 1.0,
        "depth_error": 0.25,
        "valid_depth_ratio": 0.80,
        "median_rendered_depth": 3.25,
        "median_observed_depth": 3.0,
        "initializer_metadata": {"source": "visual_candidate"},
    }

    snapped = make_depth_snapped_candidate(
        candidate,
        DummyEvaluator(),
        argparse.Namespace(generic_depth_snap_candidate_max_delta_m=0.60),
    )

    assert snapped is not None
    assert np.allclose(snapped["translation_cam"], [0.2, -0.1, 3.0])
    assert snapped["depth_error"] == pytest.approx(0.0)
    assert snapped["initializer_metadata"]["source"] == "depth_snapped_seed"
    assert snapped["initializer_metadata"]["base_source"] == "visual_candidate"
    assert snapped["initializer_metadata"]["depth_snap_delta_m"] == pytest.approx(-0.25)


def test_depth_snapped_candidate_temporarily_full_scores_coarse_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import (
        make_depth_snapped_candidate,
    )

    class DummyEvaluator:
        def __init__(self):
            self.generic_args = argparse.Namespace(generic_coarse_scoring=True)
            self.modes = []

        def evaluate_absolute(self, translation, rotation, scale):
            self.modes.append(bool(self.generic_args.generic_coarse_scoring))
            translation = np.asarray(translation, dtype=np.float64)
            depth_error = float(translation[2] - 3.0)
            return {
                "score": 0.8,
                "translation_cam": translation,
                "rotation_cam": np.asarray(rotation, dtype=np.float64),
                "scale": np.asarray(scale, dtype=np.float64),
                "projected_bbox": [1.0, 1.0, 10.0, 10.0],
                "depth_enabled": not bool(self.generic_args.generic_coarse_scoring),
                "depth_confidence": 0.0 if self.generic_args.generic_coarse_scoring else 1.0,
                "depth_score": 0.0 if self.generic_args.generic_coarse_scoring else float(np.exp(-abs(depth_error) / 0.06)),
                "depth_error": None if self.generic_args.generic_coarse_scoring else abs(depth_error),
                "valid_depth_ratio": 0.0 if self.generic_args.generic_coarse_scoring else 0.8,
                "median_rendered_depth": None if self.generic_args.generic_coarse_scoring else float(translation[2]),
                "median_observed_depth": None if self.generic_args.generic_coarse_scoring else 3.0,
            }

    evaluator = DummyEvaluator()
    candidate = {
        "score": 0.9,
        "translation_cam": np.array([0.2, -0.1, 3.25], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "projected_bbox": [1.0, 1.0, 10.0, 10.0],
        "initializer_metadata": {"source": "coarse_search"},
    }

    snapped = make_depth_snapped_candidate(
        candidate,
        evaluator,
        argparse.Namespace(
            depth_enabled=True,
            generic_depth_snap_candidate_max_delta_m=0.60,
        ),
    )

    assert snapped is not None
    assert evaluator.modes == [False, False]
    assert evaluator.generic_args.generic_coarse_scoring is True
    assert np.allclose(snapped["translation_cam"], [0.2, -0.1, 3.0])


def test_select_generic_refine_candidates_prioritizes_depth_ranked_buckets() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    def candidate(score: float, depth_score: float, tx: float, source: str) -> dict[str, object]:
        return {
            "score": score,
            "depth_score": depth_score,
            "depth_confidence": 1.0,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": {"source": source},
        }

    candidates = [
        candidate(10.0, 0.10, 0.0, "visual_top"),
        candidate(9.0, 0.20, 1.0, "visual_second"),
        candidate(3.0, 0.99, 2.0, "depth_good"),
        candidate(2.0, 0.98, 3.0, "depth_second"),
    ]

    selected = select_generic_refine_candidates(
        candidates,
        refine_top_k=3,
        depth_bucket_top_k=1,
        visual_bucket_top_k=2,
    )
    sources = [item.get("initializer_metadata", {}).get("source") for item in selected]

    assert sources[:2] == ["depth_good", "depth_second"]
    assert "depth_good" in sources


def test_select_generic_refine_candidates_depth_rank_beats_mask_only_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    def candidate(
        *,
        score: float,
        depth_score: float,
        mask_score: float,
        bbox_score: float,
        contour_score: float,
        tx: float,
        source: str,
    ) -> dict[str, object]:
        return {
            "score": score,
            "depth_score": depth_score,
            "depth_confidence": 1.0,
            "mask_blend_score": mask_score,
            "bbox_iou": bbox_score,
            "contour_score": contour_score,
            "appearance_score": 0.5,
            "appearance_confidence": 1.0,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": {"source": source},
        }

    mask_only = candidate(
        score=10.0,
        depth_score=0.05,
        mask_score=0.98,
        bbox_score=0.95,
        contour_score=0.95,
        tx=0.0,
        source="mask_only",
    )
    depth_correct = candidate(
        score=8.0,
        depth_score=0.95,
        mask_score=0.80,
        bbox_score=0.82,
        contour_score=0.70,
        tx=1.0,
        source="depth_correct",
    )

    selected = select_generic_refine_candidates([mask_only, depth_correct], refine_top_k=1)

    assert selected[0]["initializer_metadata"]["source"] == "depth_correct"
    assert selected[0]["candidate_rank_score"] > mask_only["candidate_rank_score"]


def test_merge_depth_snapped_initial_candidates_preserves_low_visual_depth_seed() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import merge_depth_snapped_initial_candidates

    def candidate(score: float, tx: float, source: str, depth_score: float = 0.0) -> dict[str, object]:
        return {
            "score": score,
            "depth_score": depth_score,
            "depth_confidence": 1.0 if depth_score > 0.0 else 0.0,
            "valid_depth_ratio": 0.8 if depth_score > 0.0 else 0.0,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": {"source": source},
        }

    visual_candidates = [
        candidate(10.0, 0.0, "visual_0"),
        candidate(9.0, 1.0, "visual_1"),
    ]
    depth_seed = candidate(1.0, 2.0, "depth_snapped_seed", depth_score=0.99)

    merged = merge_depth_snapped_initial_candidates(
        visual_candidates,
        [depth_seed],
        argparse.Namespace(top_k_candidates=2, refine_top_k=2),
    )
    sources = [item.get("initializer_metadata", {}).get("source") for item in merged]

    assert sources[:2] == ["visual_0", "visual_1"]
    assert "depth_snapped_seed" in sources


def test_merge_protected_initial_candidates_preserves_support_and_depth_seeds() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import merge_protected_initial_candidates

    def candidate(score: float, tx: float, source: str, depth_score: float = 0.0) -> dict[str, object]:
        return {
            "score": score,
            "depth_score": depth_score,
            "depth_confidence": 1.0 if depth_score > 0.0 else 0.0,
            "valid_depth_ratio": 0.8 if depth_score > 0.0 else 0.0,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": {"source": source},
        }

    visual_candidates = [
        candidate(10.0, 0.0, "visual_0"),
        candidate(9.0, 1.0, "visual_1"),
    ]
    support_seed = candidate(0.5, 2.0, "support_aligned_seed")
    depth_seed = candidate(0.4, 3.0, "depth_snapped_seed", depth_score=0.99)

    merged = merge_protected_initial_candidates(
        visual_candidates,
        support_aligned_seeds=[support_seed],
        depth_snapped_seeds=[depth_seed],
        args=argparse.Namespace(top_k_candidates=2, refine_top_k=2, generic_depth_refine_bucket_top_k=1),
    )
    sources = [item.get("initializer_metadata", {}).get("source") for item in merged]

    assert sources[:2] == ["visual_0", "visual_1"]
    assert "support_aligned_seed" in sources
    assert "depth_snapped_seed" in sources


def test_generic_pose_acceptance_rejects_free_motion_depth_outlier() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = GenericPoseEvaluator.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 100.0
    evaluator.truncation_info = {"is_truncated": False}
    evaluator.mesh_axis_prior = None
    evaluator.generic_args = argparse.Namespace(
        generic_pose_motion_phase="free_motion",
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_min_threshold=0.0,
        generic_contact_depth_min_score=0.70,
        generic_acceptance_depth_confidence_high=999.0,
        generic_depth_hard_gate_enabled=True,
        generic_depth_gate_min_confidence=0.70,
        generic_depth_min_overlap_ratio=0.35,
        generic_depth_hard_gate_max_error_m=0.12,
        generic_contact_support_max_separation_m=0.05,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
    )

    acceptance = evaluator._acceptance(
        {
            "visible_mask_iou": 0.88,
            "mask_iou": 0.88,
            "bbox_iou": 0.90,
            "bbox_center_error_px": 4.0,
            "projection_valid_ratio": 0.95,
            "depth_confidence": 1.0,
            "depth_score": 0.05,
            "depth_error": 0.18,
            "valid_depth_ratio": 0.72,
            "support_plane_enabled": False,
            "support_plane_confidence": 0.0,
        }
    )

    assert acceptance["acceptance_status"] == "rejected"
    assert "target_depth_error_above_threshold" in acceptance["reject_reasons"]


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


def test_select_generic_refine_candidates_prefers_support_aligned_seed_when_temporal_is_tilted() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    def candidate(score: float, tx: float, source: str, angle: float) -> dict[str, object]:
        return {
            "score": score,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "support_normal_angle_deg": angle,
            "initializer_metadata": {"source": source},
        }

    temporal_seed = candidate(1.20, 5.0, "temporal_prior", 14.0)
    support_seed = candidate(1.10, 5.1, "support_aligned_seed", 0.6)
    selected = select_generic_refine_candidates(
        [candidate(1.40, 0.0, "generic_grid", 8.0)],
        refine_top_k=1,
        temporal_seed=temporal_seed,
        support_aligned_seed=support_seed,
        support_alignment_trigger_deg=6.0,
        support_aligned_seed_score_margin=0.20,
    )

    assert len(selected) == 1
    assert selected[0]["initializer_metadata"]["source"] == "support_aligned_seed"


def test_select_generic_refine_candidates_can_prioritize_temporal_like_candidates() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    def candidate(score: float, tx: float, source: str, base_source: str | None = None) -> dict[str, object]:
        meta = {"source": source}
        if base_source is not None:
            meta["base_source"] = base_source
        return {
            "score": score,
            "translation_cam": np.array([tx, 0.0, 2.0], dtype=np.float64),
            "rotation_cam": np.eye(3, dtype=np.float64),
            "scale": np.ones(3, dtype=np.float64),
            "initializer_metadata": meta,
        }

    temporal_seed = candidate(1.2, 5.0, "temporal_prior")
    support_seed = candidate(1.3, 5.1, "support_aligned_seed", "temporal_prior")
    selected = select_generic_refine_candidates(
        [candidate(10.0, 0.0, "generic_grid"), candidate(0.2, 4.0, "task_json_corrected_pose")],
        refine_top_k=4,
        temporal_seed=temporal_seed,
        support_aligned_seed=support_seed,
        prefer_temporal_first=True,
    )

    sources = [item.get("initializer_metadata", {}).get("source") for item in selected]
    assert sources[:2] == ["support_aligned_seed", "temporal_prior"]


def test_select_generic_refine_candidates_uses_alignment_before_angle_when_temporal_angle_missing() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import select_generic_refine_candidates

    temporal_seed = {
        "score": 1.20,
        "translation_cam": np.array([5.0, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "initializer_metadata": {"source": "temporal_prior"},
    }
    support_seed = {
        "score": 1.05,
        "translation_cam": np.array([5.1, 0.0, 2.0], dtype=np.float64),
        "rotation_cam": np.eye(3, dtype=np.float64),
        "scale": np.ones(3, dtype=np.float64),
        "support_normal_angle_deg": 0.8,
        "support_normal_angle_deg_before_alignment": 13.0,
        "support_normal_angle_deg_after_alignment": 0.8,
        "initializer_metadata": {"source": "support_aligned_seed"},
    }

    selected = select_generic_refine_candidates(
        [],
        refine_top_k=1,
        temporal_seed=temporal_seed,
        support_aligned_seed=support_seed,
        support_alignment_trigger_deg=6.0,
        support_aligned_seed_score_margin=0.20,
    )

    assert len(selected) == 1
    assert selected[0]["initializer_metadata"]["source"] == "support_aligned_seed"


def test_generic_early_stop_allows_high_quality_truncated_temporal_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import generic_early_stop_reached

    args = argparse.Namespace(
        early_stop_mask_iou=0.94,
        early_stop_bbox_iou=0.94,
        generic_truncated_temporal_early_stop_enabled=True,
        generic_truncated_temporal_early_stop_min_score=2.0,
        generic_truncated_temporal_early_stop_min_mask_iou=0.90,
        generic_truncated_temporal_early_stop_min_bbox_iou=0.60,
        generic_truncated_temporal_early_stop_min_temporal_score=0.50,
    )
    result = {
        "score": 2.33,
        "mask_iou": 0.94,
        "bbox_iou": 0.71,
        "temporal_score": 0.86,
        "acceptance_status": "accepted",
        "initializer_metadata": {"source": "temporal_prior"},
    }

    assert generic_early_stop_reached(result, args, {"is_truncated": True, "truncation_sides": ["bottom"]})


def test_generic_early_stop_keeps_full_bbox_rule_for_non_truncated_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import generic_early_stop_reached

    args = argparse.Namespace(
        early_stop_mask_iou=0.94,
        early_stop_bbox_iou=0.94,
        generic_truncated_temporal_early_stop_enabled=True,
        generic_truncated_temporal_early_stop_min_score=2.0,
        generic_truncated_temporal_early_stop_min_mask_iou=0.90,
        generic_truncated_temporal_early_stop_min_bbox_iou=0.60,
        generic_truncated_temporal_early_stop_min_temporal_score=0.50,
    )
    result = {
        "score": 2.33,
        "mask_iou": 0.94,
        "bbox_iou": 0.71,
        "temporal_score": 0.86,
        "acceptance_status": "accepted",
        "initializer_metadata": {"source": "temporal_prior"},
    }

    assert not generic_early_stop_reached(result, args, {"is_truncated": False, "truncation_sides": []})


def test_generic_refine_uses_light_fine_search_then_full_rescore(monkeypatch) -> None:
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
        generic_lightweight_search_scoring=True,
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
    assert ("stage:fine", True, False) in calls
    assert ("full", False, True) in calls
    assert result["score"] == 2.0
    assert result["depth_score"] == 0.6
    assert result["appearance_score"] == 0.7


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
    evaluator.truncation_info = {"is_truncated": False, "truncation_sides": []}
    evaluator.generic_args = argparse.Namespace(
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
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

    assert decision["acceptance_status"] == "accepted"
    assert decision["reject_reasons"] == []
    assert decision["projection_acceptance_threshold"] == 0.50
    assert decision["projection_acceptance_exempt"] is False


def test_generic_acceptance_allows_high_quality_truncated_temporal_projection_ratio() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = object.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 180.0
    evaluator.truncation_info = {"is_truncated": True, "truncation_sides": ["bottom"]}
    evaluator.generic_args = argparse.Namespace(
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
    )

    decision = evaluator._acceptance(
        {
            "visible_mask_iou": 0.94,
            "mask_iou": 0.94,
            "bbox_iou": 0.71,
            "bbox_center_error_px": 9.5,
            "projection_valid_ratio": 0.445,
            "temporal_score": 0.86,
            "initializer_metadata": {"source": "temporal_prior"},
            "depth_confidence": 1.0,
            "depth_score": 0.61,
        }
    )

    assert decision["acceptance_status"] == "accepted"
    assert decision["reject_reasons"] == []
    assert decision["projection_acceptance_threshold"] == 0.30
    assert decision["projection_acceptance_exempt"] is True


def test_generic_acceptance_still_rejects_low_projection_non_truncated_candidate() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = object.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 180.0
    evaluator.truncation_info = {"is_truncated": False, "truncation_sides": []}
    evaluator.generic_args = argparse.Namespace(
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
    )

    decision = evaluator._acceptance(
        {
            "visible_mask_iou": 0.94,
            "mask_iou": 0.94,
            "bbox_iou": 0.71,
            "bbox_center_error_px": 9.5,
            "projection_valid_ratio": 0.445,
            "temporal_score": 0.86,
            "initializer_metadata": {"source": "temporal_prior"},
            "depth_confidence": 1.0,
            "depth_score": 0.61,
        }
    )

    assert decision["acceptance_status"] == "rejected"
    assert decision["reject_reasons"] == ["projection_valid_ratio_below_threshold"]


def test_generic_acceptance_rejects_high_confidence_support_separation() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = object.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 180.0
    evaluator.truncation_info = {"is_truncated": False, "truncation_sides": []}
    evaluator.generic_args = argparse.Namespace(
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
    )

    decision = evaluator._acceptance(
        {
            "visible_mask_iou": 0.94,
            "mask_iou": 0.94,
            "bbox_iou": 0.91,
            "bbox_center_error_px": 3.0,
            "projection_valid_ratio": 1.0,
            "depth_confidence": 1.0,
            "depth_score": 0.61,
            "support_plane_enabled": True,
            "support_plane_confidence": 0.91,
            "support_floating_distance_m": 0.0,
            "support_penetration_distance_m": 0.35,
        }
    )

    assert decision["acceptance_status"] == "rejected"
    assert decision["reject_reasons"] == ["support_separation_above_threshold"]


def test_generic_acceptance_contact_phase_requires_tight_support() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = object.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 180.0
    evaluator.truncation_info = {"is_truncated": False, "truncation_sides": []}
    evaluator.generic_args = argparse.Namespace(
        generic_pose_motion_phase="contact_calibration",
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
        generic_contact_depth_min_score=0.70,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
        generic_contact_support_max_separation_m=0.05,
    )

    decision = evaluator._acceptance(
        {
            "visible_mask_iou": 0.88,
            "mask_iou": 0.88,
            "bbox_iou": 0.91,
            "bbox_center_error_px": 3.0,
            "projection_valid_ratio": 1.0,
            "depth_confidence": 1.0,
            "depth_score": 0.82,
            "support_plane_enabled": True,
            "support_plane_confidence": 0.95,
            "support_floating_distance_m": 0.0,
            "support_penetration_distance_m": 0.08,
        }
    )

    assert decision["acceptance_status"] == "rejected"
    assert decision["reject_reasons"] == ["contact_support_separation_above_threshold"]


def test_generic_acceptance_free_motion_ignores_depth_and_support_contact() -> None:
    from process.pose_optimizer.strategies.generic_appearance_temporal import GenericPoseEvaluator

    evaluator = object.__new__(GenericPoseEvaluator)
    evaluator.target_bbox_diagonal = 180.0
    evaluator.truncation_info = {"is_truncated": False, "truncation_sides": []}
    evaluator.generic_args = argparse.Namespace(
        generic_pose_motion_phase="free_motion",
        generic_acceptance_max_center_error_ratio=0.35,
        generic_acceptance_min_visible_mask_iou=0.12,
        generic_acceptance_min_bbox_iou=0.10,
        generic_acceptance_min_projection_valid_ratio=0.50,
        generic_acceptance_truncated_min_projection_valid_ratio=0.30,
        generic_acceptance_projection_temporal_exempt_enabled=True,
        generic_acceptance_projection_exempt_min_mask_iou=0.90,
        generic_acceptance_projection_exempt_min_bbox_iou=0.60,
        generic_acceptance_projection_exempt_min_temporal_score=0.50,
        generic_acceptance_depth_confidence_high=0.70,
        generic_acceptance_depth_min_threshold=0.25,
        generic_contact_depth_min_score=0.70,
        support_acceptance_min_confidence=0.70,
        support_acceptance_max_separation_m=0.15,
        generic_contact_support_max_separation_m=0.05,
    )

    decision = evaluator._acceptance(
        {
            "visible_mask_iou": 0.88,
            "mask_iou": 0.88,
            "bbox_iou": 0.91,
            "bbox_center_error_px": 3.0,
            "projection_valid_ratio": 1.0,
            "depth_confidence": 1.0,
            "depth_score": 0.0,
            "support_plane_enabled": True,
            "support_plane_confidence": 0.95,
            "support_floating_distance_m": 0.22,
            "support_penetration_distance_m": 0.0,
        }
    )

    assert decision["acceptance_status"] == "accepted"
    assert decision["reject_reasons"] == []


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


def test_executor_generic_acceptance_rejects_high_confidence_support_separation() -> None:
    from guanwu.video.project.executor import ProjectExecutor

    report = {
        "json_bbox": [10.0, 12.0, 210.0, 52.0],
        "metrics": {
            "mask_iou": 0.90,
            "soft_mask_iou": 0.80,
            "bbox_iou": 0.95,
            "bbox_center_error_px": 2.0,
            "projection_valid_ratio": 1.0,
            "depth_confidence": 1.0,
            "depth_score": 0.60,
            "support_plane_enabled": True,
            "support_plane_confidence": 0.95,
            "support_floating_distance_m": 0.0,
            "support_penetration_distance_m": 0.37,
        },
        "optimized_corrected_pose_world": {
            "translation_world": [0.0, 0.0, 1.0],
            "scale": [0.05, 0.05, 0.05],
        },
    }

    decision = ProjectExecutor._generic_pose_optimizer_acceptance(report)

    assert decision["accepted"] is False
    assert decision["reason"] == "support_separation_above_threshold:0.370"


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


def test_truncated_visible_bbox_uses_full_in_frame_silhouette() -> None:
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

    assert score["visible_projected_bbox"] == [30.0, 20.0, 80.0, 100.0]
    assert score["visible_target_bbox"] == [28.0, 18.0, 82.0, 84.0]
