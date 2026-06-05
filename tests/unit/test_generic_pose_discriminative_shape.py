from __future__ import annotations

import argparse

import numpy as np
import trimesh

from process.pose_optimizer.strategies import generic_appearance_temporal as generic


def test_refinement_mesh_preserves_existing_task_proxy_above_coarse_face_count() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=4)
    assert len(mesh.faces) > 1800
    assert len(mesh.faces) < 8000

    args = argparse.Namespace(
        proxy_face_count=1800,
        generic_refine_face_count=8000,
    )

    refined = generic.build_refinement_mesh(mesh, args)

    assert refined.report["source_face_count"] == len(mesh.faces)
    assert refined.report["refine_face_count"] == len(mesh.faces)
    assert refined.report["refine_face_limit"] == 8000
    assert refined.report["simplified_for_refine"] is False


def test_discriminative_contour_penalizes_missing_small_asymmetric_part() -> None:
    target = np.zeros((96, 96), dtype=np.uint8)
    target[24:72, 32:64] = 1
    target[40:56, 16:32] = 1

    missing_part = np.zeros_like(target)
    missing_part[24:72, 32:64] = 1

    perfect = target.copy()
    weights, debug = generic.build_discriminative_mask_weights(
        target,
        argparse.Namespace(
            generic_discriminative_contour_center_weight=0.75,
            generic_discriminative_contour_thin_weight=0.50,
            generic_discriminative_contour_hole_weight=0.75,
            generic_discriminative_contour_max_weight=4.0,
        ),
    )

    missing_score = generic.discriminative_contour_score(
        missing_part,
        target,
        weights,
        argparse.Namespace(generic_discriminative_contour_sigma_px=4.0),
    )
    perfect_score = generic.discriminative_contour_score(
        perfect,
        target,
        weights,
        argparse.Namespace(generic_discriminative_contour_sigma_px=4.0),
    )

    assert debug["target_edge_pixels"] > 0
    assert debug["weight_max"] > 1.0
    assert perfect_score["discriminative_contour_score"] > 0.98
    assert missing_score["discriminative_contour_score"] < 0.75
    assert missing_score["discriminative_contour_target_to_render_px"] > 1.0
