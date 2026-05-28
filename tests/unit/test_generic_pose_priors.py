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
    assert score["visible_target_bbox"] == [28.0, 18.0, 82.0, 100.0]
