#!/usr/bin/env python3
"""Edge contour fast pose optimizer.

This strategy adds GrabCut-based mask enhancement on top of the temporal_fast
optimizer.  When the provided mask.png is inaccurate (e.g. does not fully cover
the vehicle), GrabCut re-segments the vehicle inside the bbox from the real
image and the result is merged (union) with the original mask so that the
evaluator works with a more complete target mask.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import fast
from . import temporal_fast


# ---------------------------------------------------------------------------
# GrabCut mask extraction
# ---------------------------------------------------------------------------

def extract_grabcut_mask(
    image: np.ndarray,
    bbox: list[float],
    image_size: tuple[int, int],
    original_mask: np.ndarray,
    grabcut_iters: int = 5,
    grabcut_margin: int = 10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run GrabCut inside *bbox* and return a foreground mask.

    The original mask is used to seed probable-foreground so that GrabCut
    starts from a reasonable initialisation.
    """
    width, height = int(image_size[0]), int(image_size[1])

    x1 = max(0, int(math.floor(float(bbox[0]))) - grabcut_margin)
    y1 = max(0, int(math.floor(float(bbox[1]))) - grabcut_margin)
    x2 = min(width, int(math.ceil(float(bbox[2]))) + grabcut_margin)
    y2 = min(height, int(math.ceil(float(bbox[3]))) + grabcut_margin)

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return original_mask.copy(), {"succeeded": False, "reason": "bbox_too_small"}

    # Build the GrabCut mask seeded from the provided annotation.
    gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    # Everything inside the expanded bbox is probable-background by default.
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
    # Pixels covered by the original mask are probable-foreground.
    gc_mask[original_mask > 0] = cv2.GC_PR_FGD

    rect = (x1, y1, x2 - x1, y2 - y1)

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        image,
        gc_mask,
        rect,
        bgd_model,
        fgd_model,
        grabcut_iters,
        cv2.GC_INIT_WITH_MASK,
    )

    foreground = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.uint8)

    info: dict[str, Any] = {
        "succeeded": True,
        "rect": list(rect),
        "grabcut_iters": grabcut_iters,
        "grabcut_margin": grabcut_margin,
        "grabcut_mask_area_px": int(foreground.sum()),
    }
    return foreground, info


def merge_masks(
    original_mask: np.ndarray,
    grabcut_mask: np.ndarray,
    merge_mode: str = "union",
) -> np.ndarray:
    """Merge *original_mask* and *grabcut_mask* according to *merge_mode*."""
    if merge_mode == "union":
        return ((original_mask > 0) | (grabcut_mask > 0)).astype(np.uint8)
    if merge_mode == "grabcut_only":
        return (grabcut_mask > 0).astype(np.uint8)
    if merge_mode == "original_only":
        return (original_mask > 0).astype(np.uint8)
    if merge_mode == "intersection":
        return ((original_mask > 0) & (grabcut_mask > 0)).astype(np.uint8)
    raise ValueError(f"Unsupported mask merge mode: {merge_mode!r}")


def apply_grabcut_enhancement(
    image: np.ndarray,
    bbox: list[float],
    image_size: tuple[int, int],
    full_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Orchestrate GrabCut extraction, validation, merge, and fallback."""

    original_area = int((full_mask > 0).sum())
    bbox_w = max(1.0, float(bbox[2]) - float(bbox[0]))
    bbox_h = max(1.0, float(bbox[3]) - float(bbox[1]))
    bbox_area = bbox_w * bbox_h

    grabcut_info: dict[str, Any] = {
        "enabled": True,
        "original_mask_area_px": original_area,
        "grabcut_mask_area_px": 0,
        "merged_mask_area_px": original_area,
        "merge_mode": str(args.mask_merge_mode),
        "fallback_used": False,
        "fallback_reason": None,
        "grabcut_succeeded": False,
        "grabcut_iters": int(args.grabcut_iters),
    }

    try:
        grabcut_mask, gc_info = extract_grabcut_mask(
            image=image,
            bbox=bbox,
            image_size=image_size,
            original_mask=full_mask,
            grabcut_iters=int(args.grabcut_iters),
            grabcut_margin=int(args.grabcut_margin),
        )
    except Exception as exc:
        print(f"[grabcut] extraction failed, falling back to original mask: {exc}")
        grabcut_info["fallback_used"] = True
        grabcut_info["fallback_reason"] = f"exception: {exc}"
        return full_mask.copy(), grabcut_info

    grabcut_info.update(gc_info)

    if not gc_info.get("succeeded", False):
        print(f"[grabcut] extraction unsuccessful ({gc_info.get('reason')}), using original mask")
        grabcut_info["fallback_used"] = True
        grabcut_info["fallback_reason"] = gc_info.get("reason", "unknown")
        return full_mask.copy(), grabcut_info

    gc_area = int((grabcut_mask > 0).sum())
    grabcut_info["grabcut_mask_area_px"] = gc_area

    # --- Sanity checks ---
    min_area_ratio = float(args.grabcut_min_area_ratio)
    max_fill_ratio = float(args.grabcut_max_bbox_fill_ratio)

    if original_area > 0 and gc_area < original_area * min_area_ratio:
        reason = (
            f"grabcut area {gc_area} < {min_area_ratio:.0%} of original {original_area}"
        )
        print(f"[grabcut] {reason}, falling back")
        grabcut_info["fallback_used"] = True
        grabcut_info["fallback_reason"] = reason
        return full_mask.copy(), grabcut_info

    if gc_area > bbox_area * max_fill_ratio:
        reason = (
            f"grabcut area {gc_area} > {max_fill_ratio:.0%} of bbox area {int(bbox_area)}"
        )
        print(f"[grabcut] {reason}, falling back")
        grabcut_info["fallback_used"] = True
        grabcut_info["fallback_reason"] = reason
        return full_mask.copy(), grabcut_info

    merged = merge_masks(full_mask, grabcut_mask, str(args.mask_merge_mode))
    merged_area = int((merged > 0).sum())
    grabcut_info["merged_mask_area_px"] = merged_area
    grabcut_info["grabcut_succeeded"] = True
    if original_area > 0:
        grabcut_info["area_change_ratio"] = float(merged_area / original_area)
    else:
        grabcut_info["area_change_ratio"] = None

    print(
        f"[grabcut] succeeded: original={original_area}px grabcut={gc_area}px "
        f"merged={merged_area}px mode={args.mask_merge_mode}"
    )
    return merged, grabcut_info


# ---------------------------------------------------------------------------
# GrabCut debug visualisation
# ---------------------------------------------------------------------------

def save_grabcut_debug(
    output_dir: Path,
    image: np.ndarray,
    original_mask: np.ndarray,
    grabcut_info: dict[str, Any],
    enhanced_mask: np.ndarray,
) -> Path | None:
    """Save a 4-panel debug image showing the GrabCut enhancement."""
    try:
        alpha = 0.45

        # Panel 1 – original mask overlay
        original_overlay = image.copy()
        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[original_mask > 0] = (0, 255, 0)  # green
        original_overlay = cv2.addWeighted(original_overlay, 1.0, color_layer, alpha, 0.0)

        # Panel 2 – grabcut mask overlay (if available)
        grabcut_overlay = image.copy()
        if grabcut_info.get("grabcut_succeeded"):
            # Reconstruct grabcut-only mask from enhanced & original
            gc_only = (enhanced_mask > 0) & ~(original_mask > 0)
            both = (enhanced_mask > 0) & (original_mask > 0)
            color_gc = np.zeros_like(image, dtype=np.uint8)
            color_gc[both.astype(bool)] = (0, 255, 255)  # yellow = overlap
            color_gc[gc_only.astype(bool)] = (255, 165, 0)  # orange = grabcut added
            grabcut_overlay = cv2.addWeighted(grabcut_overlay, 1.0, color_gc, alpha, 0.0)
        else:
            fast.draw_label(grabcut_overlay, "fallback to original", (10, 30), color=(0, 0, 255))

        # Panel 3 – merged mask overlay
        merged_overlay = image.copy()
        color_merged = np.zeros_like(image, dtype=np.uint8)
        color_merged[enhanced_mask > 0] = (0, 200, 255)  # bright orange
        merged_overlay = cv2.addWeighted(merged_overlay, 1.0, color_merged, alpha, 0.0)

        # Panel 4 – diff: green = original only, orange = grabcut added
        diff_overlay = image.copy()
        only_original = (original_mask > 0) & ~(enhanced_mask > 0)
        only_enhanced = (enhanced_mask > 0) & ~(original_mask > 0)
        overlap = (original_mask > 0) & (enhanced_mask > 0)
        diff_color = np.zeros_like(image, dtype=np.uint8)
        diff_color[overlap.astype(bool)] = (0, 255, 255)  # yellow
        diff_color[only_original.astype(bool)] = (0, 0, 255)  # red = removed
        diff_color[only_enhanced.astype(bool)] = (0, 255, 0)  # green = added
        diff_overlay = cv2.addWeighted(diff_overlay, 1.0, diff_color, 0.55, 0.0)

        panels: list[tuple[str, np.ndarray]] = [
            ("original mask", original_overlay),
            ("grabcut result", grabcut_overlay),
            ("merged mask", merged_overlay),
            ("diff (green=added, red=removed)", diff_overlay),
        ]

        return fast.save_image_collage(
            output_dir / "05_grabcut_debug.png",
            panels,
            columns=2,
            content_size=(420, 280),
        )
    except Exception as exc:
        print(f"[warn] grabcut debug image failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# optimize_sample – mirrors temporal_fast.optimize_sample with GrabCut step
# ---------------------------------------------------------------------------

def optimize_sample(args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = fast.resolve_sample_dir(args.sample_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / f"{sample_dir.name}_edge_contour_fast"
    )
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_backend == "pytorch3d" or args.validate_pytorch3d_alignment:
        fast.import_torch_and_pytorch3d()
        fast.resolve_torch_device(args.device, allow_auto_fallback=False)

    # ── read inputs (same as temporal_fast) ──
    task = fast.read_json(sample_dir / "task.json")
    image = fast.read_image(sample_dir / "image.jpg", mode="color")
    crop_mask = fast.read_image(sample_dir / "mask.png", mode="gray")
    crop_image_path = sample_dir / "crop.jpg"
    crop_image = fast.read_image(crop_image_path, mode="color") if crop_image_path.exists() else None
    image_size = fast.image_size_from_task(task, image)
    json_bbox = [float(v) for v in task["bbox_xyxy"]]
    full_mask, mask_placement = fast.paste_crop_mask_to_full_image(crop_mask, json_bbox, image_size, full_image=image, crop_image=crop_image)

    # Use the pasted original mask directly. GrabCut helpers are kept in this
    # module for compatibility, but this strategy path no longer expands the
    # target mask before scoring.
    original_mask = full_mask.copy()
    enhanced_mask = full_mask
    original_area = int((full_mask > 0).sum())
    grabcut_info = {
        "enabled": False,
        "requested_enabled": bool(getattr(args, "enable_grabcut", False)),
        "skipped": True,
        "skip_reason": "using original mask directly",
        "original_mask_area_px": original_area,
        "grabcut_mask_area_px": 0,
        "merged_mask_area_px": original_area,
        "merge_mode": "original_only",
        "fallback_used": False,
        "fallback_reason": None,
        "grabcut_succeeded": False,
        "grabcut_iters": int(getattr(args, "grabcut_iters", 0)),
        "area_change_ratio": 1.0 if original_area > 0 else None,
    }

    # Use enhanced mask for all downstream scoring
    soft_full_mask = fast.make_soft_mask(enhanced_mask)
    if args.fast_float32:
        soft_full_mask = soft_full_mask.astype(np.float32, copy=False)
    obs = fast.extract_mask_observations(enhanced_mask)

    # ── load mesh (same as temporal_fast) ──
    mesh_path = fast.find_mesh_path(sample_dir, task)
    mesh = fast.load_glb_as_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh_meta = fast.mesh_axis_metadata(vertices)
    mesh_tail_light_prior = fast.infer_tail_light_axis_prior(mesh, mesh_meta)

    t_world_from_cam = np.asarray(task["camera"]["T_world_from_cam"], dtype=np.float64)
    intrinsics = {
        "fx": float(task["camera"]["fx"]),
        "fy": float(task["camera"]["fy"]),
        "cx": float(task["camera"]["cx"]),
        "cy": float(task["camera"]["cy"]),
    }
    vehicle_pose_context = fast.build_vehicle_pose_context(
        task=task,
        sample_dir=sample_dir,
        full_mask=enhanced_mask,
        json_bbox=json_bbox,
        image_size=image_size,
        intrinsics=intrinsics,
        t_world_from_cam=t_world_from_cam,
        args=args,
    )
    trusted_anchor_payload = vehicle_pose_context.get("trusted_temporal_anchor_pose")
    trusted_anchor_pose = temporal_fast.load_prior_pose_payload(trusted_anchor_payload, t_world_from_cam)
    if trusted_anchor_pose is not None:
        vehicle_pose_context["trusted_temporal_anchor_pose"] = trusted_anchor_pose
    else:
        vehicle_pose_context.pop("trusted_temporal_anchor_pose", None)
    vehicle_pose_context["mesh_tail_light_prior"] = mesh_tail_light_prior
    mesh_meta = fast.apply_mesh_axis_prior(mesh_meta, vehicle_pose_context.get("mesh_axis_prior"))
    mesh_meta["tail_light_prior"] = mesh_tail_light_prior
    proxy_vertices, proxy_faces = fast.build_proxy_mesh(vertices, faces, target_faces=args.proxy_face_count)

    # ── temporal prior search (same as temporal_fast) ──
    object_id, frame_idx = temporal_fast.parse_task_id_from_sample_dir(sample_dir)
    suffixes = temporal_fast.parse_suffixes(args.temporal_search_output_suffixes)
    temporal_prior = None
    skip_disk_temporal_prior = temporal_fast.should_skip_disk_temporal_prior(vehicle_pose_context)
    if bool(args.temporal_enabled) and frame_idx > 1:
        temporal_prior = temporal_fast.load_prior_pose_payload(
            task.get("temporal_prior_pose"),
            t_world_from_cam=t_world_from_cam,
        )
        if temporal_prior is None and not skip_disk_temporal_prior:
            temporal_prior = temporal_fast.find_temporal_prior(
                output_dir=output_dir,
                object_id=object_id,
                frame_idx=frame_idx,
                lookback=args.temporal_lookback,
                suffixes=suffixes,
                t_world_from_cam=t_world_from_cam,
            )

    if temporal_prior is None and skip_disk_temporal_prior:
        print(f"[temporal] disk prior disabled for all-frames candidate pass: {object_id}@{frame_idx:06d}")
    elif temporal_prior is None:
        print(f"[temporal] no prior found for {object_id}@{frame_idx:06d}; falling back to fast-style scoring")
    else:
        print(
            f"[temporal] prior found: frame={temporal_prior['frame_idx']} "
            f"dir={temporal_prior['output_dir']} source={temporal_prior['pose_source']}"
        )

    # ── truncation detection (using enhanced mask) ──
    truncation_info = temporal_fast.detect_truncation(
        mask=enhanced_mask,
        bbox=json_bbox,
        image_size=image_size,
        args=args,
        prior_mask_area_px=temporal_prior.get("prior_mask_area_px") if temporal_prior else None,
    )
    print(
        f"[partial] enabled={bool(args.partial_visibility_enabled)} "
        f"is_truncated={truncation_info['is_truncated']} sides={truncation_info['truncation_sides']}"
    )

    # ── edge context ──
    edge_context = temporal_fast.prepare_image_edge_map(image, enhanced_mask, json_bbox, image_size, args)
    print(f"[edge] enabled={bool(args.edge_score_enabled)} available={edge_context is not None}")

    # ── scale ──
    pose = task.get("corrected_pose", {})
    base_scale = fast.make_uniform_scale(
        fast.scale_to_uniform_scalar(np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))
    )

    # ── evaluators (using enhanced_mask) ──
    proxy_evaluator = temporal_fast.TemporalPoseEvaluator(
        vertices=proxy_vertices,
        faces=proxy_faces,
        mesh=None,
        full_mask=enhanced_mask,
        soft_full_mask=soft_full_mask,
        json_bbox=json_bbox,
        intrinsics=intrinsics,
        image_size=image_size,
        bbox_weight=args.bbox_weight,
        hard_mask_weight=args.hard_mask_weight,
        backend="triangle_fill",
        enable_bbox_prefilter=args.enable_bbox_prefilter,
        prefilter_bbox_iou_min=args.prefilter_bbox_iou_min,
        prefilter_center_factor=args.prefilter_center_factor,
        prefilter_size_ratio_min=args.prefilter_size_ratio_min,
        prefilter_size_ratio_max=args.prefilter_size_ratio_max,
        roi_iou_margin=args.roi_iou_margin,
        disable_roi_iou=args.disable_roi_iou,
        fast_float32=args.fast_float32,
        profile_timings=args.profile_timings,
        device=args.device,
        pytorch3d_faces_per_pixel=args.pytorch3d_faces_per_pixel,
        pytorch3d_cull_backfaces=args.pytorch3d_cull_backfaces,
        pytorch3d_bin_size=args.pytorch3d_bin_size,
        pytorch3d_max_faces_per_bin=args.pytorch3d_max_faces_per_bin,
        temporal_args=args,
        temporal_prior=temporal_prior,
        truncation_info=truncation_info,
        edge_context=None,
        enable_edge_score=False,
        t_world_from_cam=t_world_from_cam,
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
    )
    full_evaluator = temporal_fast.TemporalPoseEvaluator(
        vertices=vertices,
        faces=faces,
        mesh=mesh,
        full_mask=enhanced_mask,
        soft_full_mask=soft_full_mask,
        json_bbox=json_bbox,
        intrinsics=intrinsics,
        image_size=image_size,
        bbox_weight=args.bbox_weight,
        hard_mask_weight=args.hard_mask_weight,
        backend=args.render_backend,
        enable_bbox_prefilter=args.enable_bbox_prefilter,
        prefilter_bbox_iou_min=args.prefilter_bbox_iou_min,
        prefilter_center_factor=args.prefilter_center_factor,
        prefilter_size_ratio_min=args.prefilter_size_ratio_min,
        prefilter_size_ratio_max=args.prefilter_size_ratio_max,
        roi_iou_margin=args.roi_iou_margin,
        disable_roi_iou=args.disable_roi_iou,
        fast_float32=args.fast_float32,
        profile_timings=args.profile_timings,
        device=args.device,
        pytorch3d_faces_per_pixel=args.pytorch3d_faces_per_pixel,
        pytorch3d_cull_backfaces=args.pytorch3d_cull_backfaces,
        pytorch3d_bin_size=args.pytorch3d_bin_size,
        pytorch3d_max_faces_per_bin=args.pytorch3d_max_faces_per_bin,
        temporal_args=args,
        temporal_prior=temporal_prior,
        truncation_info=truncation_info,
        edge_context=edge_context,
        enable_edge_score=True,
        t_world_from_cam=t_world_from_cam,
        mesh_meta=mesh_meta,
        vehicle_pose_context=vehicle_pose_context,
    )

    # ── candidate search ──
    corrected_seed = fast.corrected_pose_seed(task, t_world_from_cam, proxy_evaluator) if args.include_corrected_seed else None
    initial_candidates = fast.generate_initial_candidates(
        evaluator=proxy_evaluator,
        obs=obs,
        mesh_meta=mesh_meta,
        base_scale=base_scale,
        corrected_seed=corrected_seed,
        t_world_from_cam=t_world_from_cam,
        args=args,
    )
    initial_candidates = temporal_fast.augment_with_road_snap_candidates(initial_candidates, proxy_evaluator, args)

    # ── temporal seed ──
    temporal_seed = None
    if bool(args.temporal_enabled) and bool(args.temporal_seed_enabled):
        temporal_seed = temporal_fast.make_temporal_seed(temporal_prior, proxy_evaluator)
        if temporal_seed is not None:
            print(
                f"[temporal] seed candidate score={temporal_seed['score']:.6f} "
                f"mask_iou={temporal_seed['mask_iou']:.6f} bbox_iou={temporal_seed['bbox_iou']:.6f}"
            )
    initial_candidates = temporal_fast.merge_temporal_seed(
        initial_candidates,
        temporal_seed,
        top_k=args.top_k_candidates,
        refine_top_k=args.refine_top_k,
    )
    initial_candidates = temporal_fast.augment_with_road_snap_candidates(initial_candidates, proxy_evaluator, args)

    # ── refinement ──
    best_result: dict[str, Any] | None = None
    best_history: list[dict[str, Any]] = []
    refined_results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    coarse_preview = initial_candidates[: min(5, len(initial_candidates))]
    preview_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(coarse_preview):
        preview_rows.append(
            {
                "rank": index + 1,
                "score": candidate["score"],
                "base_geometry_score": candidate.get("base_geometry_score"),
                "temporal_score": candidate.get("temporal_score"),
                "partial_mask_score": candidate.get("partial_mask_score"),
                "edge_score": candidate.get("edge_score"),
                "ground_contact_score": candidate.get("ground_contact_score"),
                "upright_score": candidate.get("upright_score"),
                "heading_prior_score": candidate.get("heading_prior_score"),
                "mask_iou": candidate["mask_iou"],
                "bbox_iou": candidate["bbox_iou"],
                "bbox_center_error_px": candidate["bbox_center_error_px"],
                "initializer_metadata": candidate.get("initializer_metadata", {}),
            }
        )

    candidates_to_refine = temporal_fast.select_pareto_refine_candidates(
        initial_candidates,
        args.refine_top_k,
        args,
        truncation_info,
    )
    if temporal_seed is not None and not any(
        candidate.get("initializer_metadata", {}).get("source") == "temporal_prior"
        for candidate in candidates_to_refine
    ):
        if candidates_to_refine:
            candidates_to_refine[-1] = temporal_seed
        else:
            candidates_to_refine.append(temporal_seed)
        print("[temporal] temporal seed forced into refinement set")

    print(f"[search] generated {len(initial_candidates)} candidates, refining top {len(candidates_to_refine)}")

    for rank, candidate in enumerate(candidates_to_refine, start=1):
        meta = candidate.get("initializer_metadata", {})
        source = meta.get("source", "unknown")
        yaw = meta.get("yaw_deg", "n/a")
        scale_factor = meta.get("global_scale_factor", "n/a")
        depth_factor = meta.get("depth_factor", "n/a")
        anchor = meta.get("anchor_name", "n/a")
        print(
            f"[candidate {rank:02d}] init score={candidate['score']:.6f} "
            f"mask_iou={candidate['mask_iou']:.6f} bbox_iou={candidate['bbox_iou']:.6f} "
            f"source={source} yaw={yaw} scale_factor={scale_factor} depth_factor={depth_factor} anchor={anchor}"
        )
        refined_result, history = temporal_fast.refine_candidate_stages(candidate, proxy_evaluator, full_evaluator, args)
        print(
            f"[candidate {rank:02d}] refined score={refined_result['score']:.6f} "
            f"mask_iou={refined_result['mask_iou']:.6f} bbox_iou={refined_result['bbox_iou']:.6f}"
        )
        if bool(truncation_info.get("is_truncated")):
            ground_ok = temporal_fast.candidate_satisfies_ground_constraint(refined_result, args)
            print(
                f"[candidate {rank:02d}] truncated metrics "
                f"visible_mask={float(refined_result.get('visible_mask_iou') or 0.0):.6f} "
                f"visible_contour={float(refined_result.get('visible_contour_score') or 0.0):.6f} "
                f"visible_bbox={float(refined_result.get('visible_bbox_iou', refined_result.get('bbox_iou', 0.0)) or 0.0):.6f} "
                f"quality_gate={float(refined_result.get('truncated_visual_quality_gate') or 0.0):.6f} "
                f"ground_mean={float(refined_result.get('ground_contact_mean_abs_m') or 999.0):.6f} "
                f"ground_max={float(refined_result.get('ground_contact_max_abs_m') or 999.0):.6f} "
                f"ground_feasible={ground_ok}"
            )
        refined_result["initializer_metadata"] = candidate.get("initializer_metadata", {})
        refined_result["candidate_rank"] = rank
        refined_results.append((refined_result, history))
        if best_result is None or refined_result["score"] > best_result["score"]:
            best_result = refined_result
            best_history = history

        if (
            best_result["mask_iou"] >= args.early_stop_mask_iou
            and best_result["bbox_iou"] >= args.early_stop_bbox_iou
        ):
            print(
                f"[early-stop] mask_iou={best_result['mask_iou']:.6f} "
                f"bbox_iou={best_result['bbox_iou']:.6f}"
            )
            break

    selected_result = temporal_fast.choose_best_refined_result(
        [item[0] for item in refined_results],
        args,
        truncation_info,
    )
    rescue_diagnostics: dict[str, Any] | None = None
    if selected_result is not None:
        rescue_result, rescue_history, rescue_diagnostics = temporal_fast.try_non_truncated_visual_ground_rescue(
            selected_result=selected_result,
            initial_candidates=initial_candidates,
            refined_results=[item[0] for item in refined_results],
            proxy_evaluator=proxy_evaluator,
            full_evaluator=full_evaluator,
            args=args,
            truncation_info=truncation_info,
        )
        if rescue_result is not None:
            rescue_result["candidate_rank"] = len(refined_results) + 1
            refined_results.append((rescue_result, rescue_history))
            selected_result = rescue_result
    if selected_result is None:
        best_result = None
        best_history = []
    elif selected_result is not best_result:
        best_result = selected_result
        for refined_result, history in refined_results:
            if refined_result is best_result:
                best_history = history
                break

    if best_result is None:
        raise RuntimeError("No valid pose candidate survived refinement.")

    # ── finalise ──
    if bool(truncation_info.get("is_truncated")):
        truncation_info["truncation_severity"] = best_result.get("truncation_severity", truncation_info.get("truncation_severity"))
        truncation_info["low_observability"] = bool(best_result.get("low_observability", truncation_info.get("low_observability", False)))
        truncation_info["truncation_observability_score"] = best_result.get(
            "truncation_observability_score",
            truncation_info.get("truncation_observability_score"),
        )
        truncation_info["truncation_observability_reasons"] = best_result.get(
            "truncation_observability_reasons",
            truncation_info.get("truncation_observability_reasons", []),
        )

    best_uniform_scale = fast.scale_to_uniform_scalar(np.asarray(best_result["scale"], dtype=np.float64))
    best_result["scale"] = fast.make_uniform_scale(best_uniform_scale)
    translation_world, rotation_world = fast.camera_pose_to_world_pose(
        t_world_from_cam,
        np.asarray(best_result["translation_cam"], dtype=np.float64),
        np.asarray(best_result["rotation_cam"], dtype=np.float64),
    )

    # ── save visualisations ──
    fast.save_mask_comparison(
        output_dir,
        image,
        enhanced_mask,
        best_result["rendered_mask"],
        json_bbox,
        best_result["projected_bbox"],
        "01_best",
    )
    fast.save_glb_native_shape_views(output_dir, vertices, faces, mesh_meta)
    fast.save_optimized_glb_projection_views(
        output_dir=output_dir,
        image=image,
        vertices=vertices,
        faces=faces,
        best_result=best_result,
        intrinsics=intrinsics,
        image_size=image_size,
        json_bbox=json_bbox,
    )
    fast.save_optimized_glb_pose_render(
        output_dir=output_dir,
        mesh=mesh,
        vertices=vertices,
        faces=faces,
        best_result=best_result,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    collage_paths = fast.save_result_collages(
        output_dir=output_dir,
        json_bbox=json_bbox,
        projected_bbox=best_result["projected_bbox"],
    )
    temporal_debug_path = temporal_fast.save_temporal_edge_debug(
        output_dir=output_dir,
        image=image,
        vertices=vertices,
        faces=faces,
        intrinsics=intrinsics,
        image_size=image_size,
        best_result=best_result,
        temporal_prior=temporal_prior,
        edge_context=edge_context,
    )
    grabcut_debug_path = None
    fast.write_history_csv(output_dir / "optimization_history.csv", best_history)

    # ── render validation ──
    render_validation_outputs: dict[str, Any] = {}
    if args.validate_render_backends:
        render_validation_outputs["render_backends"] = fast.save_render_backend_validation(
            output_dir=output_dir,
            evaluator=full_evaluator,
            mesh=mesh,
            vertices=vertices,
            faces=faces,
            best_result=best_result,
            intrinsics=intrinsics,
            image_size=image_size,
        )
    if args.validate_pytorch3d_alignment:
        render_validation_outputs["pytorch3d_alignment"] = fast.save_pytorch3d_alignment_validation(
            output_dir=output_dir,
            evaluator=full_evaluator,
            vertices=vertices,
            faces=faces,
            best_result=best_result,
            intrinsics=intrinsics,
            image_size=image_size,
        )

    # ── write optimised task ──
    optimized_task = json.loads(json.dumps(task))
    optimized_task["corrected_pose"]["translation_world"] = fast.to_builtin(translation_world)
    optimized_task["corrected_pose"]["rotation_matrix"] = fast.to_builtin(rotation_world)
    optimized_task["corrected_pose"]["scale"] = fast.to_builtin(best_result["scale"])
    with (output_dir / "task_with_optimized_corrected_pose.json").open("w", encoding="utf-8") as f:
        json.dump(fast.to_builtin(optimized_task), f, indent=2)

    # ── temporal report ──
    final_temporal = None
    if temporal_prior is not None:
        final_temporal = temporal_fast.compute_temporal_score(
            np.asarray(best_result["translation_cam"], dtype=np.float64),
            np.asarray(best_result["rotation_cam"], dtype=np.float64),
            np.asarray(best_result["scale"], dtype=np.float64),
            temporal_prior,
            args,
        )

    temporal_report = {
        "enabled": bool(args.temporal_enabled),
        "prior_found": temporal_prior is not None,
        "prior_frame_id": temporal_prior.get("frame_idx") if temporal_prior else None,
        "prior_output_dir": temporal_prior.get("output_dir") if temporal_prior else None,
        "prior_pose_source": temporal_prior.get("pose_source") if temporal_prior else None,
        "delta_translation": final_temporal.get("delta_translation") if final_temporal else None,
        "delta_translation_norm": final_temporal.get("delta_translation_norm") if final_temporal else None,
        "delta_depth": final_temporal.get("delta_depth") if final_temporal else None,
        "delta_rotation_deg": final_temporal.get("delta_rotation_deg") if final_temporal else None,
        "delta_yaw_deg": final_temporal.get("delta_yaw_deg") if final_temporal else None,
        "delta_pitch_deg": final_temporal.get("delta_pitch_deg") if final_temporal else None,
        "delta_roll_deg": final_temporal.get("delta_roll_deg") if final_temporal else None,
        "delta_scale_log": final_temporal.get("delta_scale_log") if final_temporal else None,
        "temporal_score": final_temporal.get("temporal_score") if final_temporal else None,
        "temporal_loss": final_temporal.get("temporal_loss") if final_temporal else None,
        "used_temporal_seed": temporal_seed is not None,
        "best_started_from_temporal_seed": best_result.get("initializer_metadata", {}).get("source") == "temporal_prior",
    }
    track_prior = vehicle_pose_context.get("track_scale_prior") if isinstance(vehicle_pose_context, dict) else {}
    track_report = {
        "enabled": bool(getattr(args, "track_scale_prior_enabled", True)),
        "available": bool((track_prior or {}).get("available", False)),
        "source": (track_prior or {}).get("source"),
        "scale": (track_prior or {}).get("scale"),
        "score": best_result.get("track_scale_prior_score"),
        "loss": best_result.get("track_scale_prior_loss"),
        "delta_log": best_result.get("track_scale_prior_delta_log"),
        "effective_weight": best_result.get("effective_track_scale_prior_weight"),
    }
    partial_report = {
        "enabled": bool(args.partial_visibility_enabled),
        "is_truncated": bool(truncation_info.get("is_truncated")),
        "truncation_sides": truncation_info.get("truncation_sides", []),
        "truncation_severity": truncation_info.get("truncation_severity", "none"),
        "low_observability": bool(truncation_info.get("low_observability", False)),
        "truncation_observability_score": truncation_info.get("truncation_observability_score"),
        "truncation_observability_reasons": truncation_info.get("truncation_observability_reasons", []),
        "border_touch": truncation_info.get("border_touch", {}),
        "bbox_touch": truncation_info.get("bbox_touch", {}),
        "mask_area_px": truncation_info.get("mask_area_px"),
        "prior_mask_area_px": truncation_info.get("prior_mask_area_px"),
        "area_drop_ratio": truncation_info.get("area_drop_ratio"),
        "original_mask_iou": best_result.get("original_mask_iou", best_result.get("mask_iou")),
        "partial_mask_score": best_result.get("partial_mask_score"),
        "adjusted_mask_score": best_result.get("adjusted_mask_score"),
        "partial_score_boost": best_result.get("partial_score_boost"),
        "partial_score_boost_cap": args.partial_score_boost_cap,
        "visible_mask_iou": best_result.get("visible_mask_iou"),
        "visible_soft_mask_iou": best_result.get("visible_soft_mask_iou"),
        "visible_target_fraction": best_result.get("visible_target_fraction"),
        "visible_target_area_px": best_result.get("visible_target_area_px"),
        "visible_bbox_iou": best_result.get("visible_bbox_iou"),
        "visible_bbox_center_error_px": best_result.get("visible_bbox_center_error_px"),
        "visible_contour_score": best_result.get("visible_contour_score"),
        "visible_contour_chamfer_score": best_result.get("visible_contour_chamfer_score"),
        "visible_profile_score": best_result.get("visible_profile_score"),
        "visible_contour_mean_distance_px": best_result.get("visible_contour_mean_distance_px"),
        "visible_profile_mean_distance_px": best_result.get("visible_profile_mean_distance_px"),
        "visible_profile_coverage": best_result.get("visible_profile_coverage"),
        "effective_visible_contour_weight": best_result.get("effective_visible_contour_weight"),
        "truncated_visual_quality_gate": best_result.get("truncated_visual_quality_gate"),
        "truncated_visual_quality_reason": best_result.get("truncated_visual_quality_reason"),
        "truncated_visual_bbox_factor": best_result.get("truncated_visual_bbox_factor"),
        "truncated_visual_center_factor": best_result.get("truncated_visual_center_factor"),
        "truncated_visual_mask_factor": best_result.get("truncated_visual_mask_factor"),
        "truncated_visual_contour_factor": best_result.get("truncated_visual_contour_factor"),
        "truncated_visual_profile_factor": best_result.get("truncated_visual_profile_factor"),
        "truncated_visual_overflow_factor": best_result.get("truncated_visual_overflow_factor"),
        "truncated_visual_overflow_loss": best_result.get("truncated_visual_overflow_loss"),
        "truncated_visual_quality_penalty": best_result.get("truncated_visual_quality_penalty"),
        "severe_truncation_gate_passed": best_result.get("severe_truncation_gate_passed"),
        "severe_truncation_gate_reasons": best_result.get("severe_truncation_gate_reasons"),
        "visible_projected_bbox": best_result.get("visible_projected_bbox"),
        "visible_target_bbox": best_result.get("visible_target_bbox"),
        "visible_bbox_source": best_result.get("visible_bbox_source"),
        "truncated_bbox_score": best_result.get("truncated_bbox_score"),
        "truncated_bbox_loss": best_result.get("truncated_bbox_loss"),
        "truncated_bbox_penalty": best_result.get("truncated_bbox_penalty"),
        "truncated_bbox_components": best_result.get("truncated_bbox_components"),
        "truncated_bbox_source": best_result.get("truncated_bbox_source"),
    }
    edge_report = {
        "enabled": bool(args.edge_score_enabled),
        "available": edge_context is not None,
        "edge_score": best_result.get("edge_score"),
        "edge_mean_distance_px": best_result.get("edge_mean_distance_px"),
        "edge_rendered_points": best_result.get("edge_rendered_points"),
        "edge_roi": best_result.get("edge_roi"),
        "effective_edge_weight": best_result.get("effective_edge_weight"),
    }
    road_report = {
        "enabled": bool(args.road_constraint_enabled),
        "available": bool((vehicle_pose_context.get("road_constraint") or {}).get("available")),
        "source": (vehicle_pose_context.get("road_constraint") or {}).get("source"),
        "reason": (vehicle_pose_context.get("road_constraint") or {}).get("reason"),
        "road_plane": (vehicle_pose_context.get("road_constraint") or {}).get("road_plane"),
        "bbox_bottom_ground": (vehicle_pose_context.get("road_constraint") or {}).get("bbox_bottom_ground"),
        "ground_contact_score": best_result.get("ground_contact_score"),
        "ground_contact_mean_abs_m": best_result.get("ground_contact_mean_abs_m"),
        "ground_contact_max_abs_m": best_result.get("ground_contact_max_abs_m"),
        "ground_gate_passed": best_result.get("ground_gate_passed"),
        "ground_gate_rejected": best_result.get("ground_gate_rejected"),
        "bbox_bottom_score": best_result.get("bbox_bottom_score"),
        "bbox_bottom_distance_m": best_result.get("bbox_bottom_distance_m"),
        "upright_score": best_result.get("upright_score"),
        "upright_angle_error_deg": best_result.get("upright_angle_error_deg"),
        "upright_gate_passed": best_result.get("upright_gate_passed"),
        "upright_gate_rejected": best_result.get("upright_gate_rejected"),
        "upright_gate_penalty": best_result.get("upright_gate_penalty"),
        "effective_ground_contact_weight": best_result.get("effective_ground_contact_weight"),
        "effective_bbox_bottom_weight": best_result.get("effective_bbox_bottom_weight"),
        "effective_upright_weight": best_result.get("effective_upright_weight"),
        "visual_gate_factor": best_result.get("visual_gate_factor"),
        "visual_gate_reason": best_result.get("visual_gate_reason"),
        "visual_gate_mask_iou_min": best_result.get("visual_gate_mask_iou_min"),
        "visual_gate_bbox_iou_min": best_result.get("visual_gate_bbox_iou_min"),
        "visual_gate_center_error_px_max": best_result.get("visual_gate_center_error_px_max"),
    }
    heading_report = {
        "enabled": bool(args.heading_prior_enabled),
        "available": bool((vehicle_pose_context.get("heading_prior") or {}).get("vector_image")),
        "source": (vehicle_pose_context.get("heading_prior") or {}).get("source"),
        "confidence": best_result.get("heading_prior_confidence"),
        "score": best_result.get("heading_prior_score"),
        "angle_error_deg": best_result.get("heading_prior_angle_error_deg"),
        "front_sign_enabled": best_result.get("heading_front_sign_enabled"),
        "front_sign_confidence": best_result.get("heading_front_sign_confidence"),
        "candidate_forward_sign": best_result.get("heading_candidate_forward_sign"),
        "semantic_front_sign": best_result.get("heading_semantic_front_sign"),
        "tail_light_front_sign": best_result.get("heading_tail_light_front_sign"),
        "tail_light_flipped": best_result.get("heading_tail_light_flipped"),
        "front_sign_hard_rejected": best_result.get("heading_front_sign_hard_rejected"),
        "front_angle_penalty": best_result.get("heading_front_angle_penalty"),
        "depth_trend_score": best_result.get("heading_depth_trend_score"),
        "depth_trend_direction": best_result.get("heading_depth_trend_direction"),
        "depth_trend_confidence": best_result.get("heading_depth_trend_confidence"),
        "front_depth_cam": best_result.get("heading_front_depth_cam"),
        "front_sign_penalty": best_result.get("heading_front_sign_penalty"),
        "effective_front_sign_penalty_weight": best_result.get("effective_front_sign_penalty_weight"),
        "bbox_area_trend": (vehicle_pose_context.get("heading_prior") or {}).get("bbox_area_trend"),
        "projected_vector_image": best_result.get("heading_prior_projected_vector_image"),
        "target_vector_image": best_result.get("heading_prior_target_vector_image"),
        "effective_weight": best_result.get("effective_heading_prior_weight"),
    }

    report = {
        "task_id": task["task_id"],
        "object_id": task["object_id"],
        "label": task["label"],
        "sample_dir": str(sample_dir),
        "mesh_path": str(mesh_path),
        "image_size": list(image_size),
        "json_bbox": json_bbox,
        "mask_placement": mask_placement,
        "mask_observations": obs,
        "mesh_axis_metadata": mesh_meta,
        "mesh_tail_light_prior": mesh_tail_light_prior,
        "camera_intrinsics": intrinsics,
        "render_backend": full_evaluator.active_backend or full_evaluator.backend_preference,
        "bbox_weight": args.bbox_weight,
        "scale_constraint": "uniform_xyz",
        "optimized_uniform_scale": best_uniform_scale,
        "initializer_top_candidates": preview_rows,
        "best_initializer_metadata": best_result.get("initializer_metadata", {}),
        "best_candidate_rank": best_result.get("candidate_rank"),
        "grabcut": grabcut_info,
        "temporal": temporal_report,
        "track_prior": track_report,
        "non_truncated_visual_ground_rescue": rescue_diagnostics,
        "partial_visibility": partial_report,
        "edge_assist": edge_report,
        "road_constraint": road_report,
        "heading_prior": heading_report,
        "optimized_camera_pose": {
            "translation_cam": best_result["translation_cam"],
            "rotation_cam": best_result["rotation_cam"],
            "scale": best_result["scale"],
        },
        "optimized_corrected_pose_world": {
            "translation_world": translation_world,
            "rotation_matrix": rotation_world,
            "scale": best_result["scale"],
        },
        "metrics": {
            "score": best_result["score"],
            "base_geometry_score": best_result.get("base_geometry_score"),
            "geometry_score": best_result.get("geometry_score"),
            "truncated_bbox_penalty": best_result.get("truncated_bbox_penalty"),
            "effective_temporal_weight": best_result.get("effective_temporal_weight"),
            "track_scale_prior_score": best_result.get("track_scale_prior_score"),
            "track_scale_prior_loss": best_result.get("track_scale_prior_loss"),
            "track_scale_prior_value": best_result.get("track_scale_prior_value"),
            "track_scale_prior_delta_log": best_result.get("track_scale_prior_delta_log"),
            "effective_track_scale_prior_weight": best_result.get("effective_track_scale_prior_weight"),
            "effective_edge_weight": best_result.get("effective_edge_weight"),
            "ground_contact_score": best_result.get("ground_contact_score"),
            "ground_contact_mean_abs_m": best_result.get("ground_contact_mean_abs_m"),
            "ground_contact_max_abs_m": best_result.get("ground_contact_max_abs_m"),
            "ground_gate_passed": best_result.get("ground_gate_passed"),
            "ground_gate_rejected": best_result.get("ground_gate_rejected"),
            "bbox_bottom_score": best_result.get("bbox_bottom_score"),
            "bbox_bottom_distance_m": best_result.get("bbox_bottom_distance_m"),
            "upright_score": best_result.get("upright_score"),
            "upright_angle_error_deg": best_result.get("upright_angle_error_deg"),
            "upright_gate_passed": best_result.get("upright_gate_passed"),
            "upright_gate_rejected": best_result.get("upright_gate_rejected"),
            "upright_gate_penalty": best_result.get("upright_gate_penalty"),
            "heading_prior_score": best_result.get("heading_prior_score"),
            "heading_prior_angle_error_deg": best_result.get("heading_prior_angle_error_deg"),
            "heading_front_sign_enabled": best_result.get("heading_front_sign_enabled"),
            "heading_front_sign_confidence": best_result.get("heading_front_sign_confidence"),
            "heading_candidate_forward_sign": best_result.get("heading_candidate_forward_sign"),
            "heading_semantic_front_sign": best_result.get("heading_semantic_front_sign"),
            "heading_tail_light_front_sign": best_result.get("heading_tail_light_front_sign"),
            "heading_tail_light_flipped": best_result.get("heading_tail_light_flipped"),
            "heading_front_sign_hard_rejected": best_result.get("heading_front_sign_hard_rejected"),
            "heading_front_angle_penalty": best_result.get("heading_front_angle_penalty"),
            "heading_depth_trend_score": best_result.get("heading_depth_trend_score"),
            "heading_depth_trend_direction": best_result.get("heading_depth_trend_direction"),
            "heading_depth_trend_confidence": best_result.get("heading_depth_trend_confidence"),
            "heading_front_depth_cam": best_result.get("heading_front_depth_cam"),
            "heading_front_sign_penalty": best_result.get("heading_front_sign_penalty"),
            "effective_ground_contact_weight": best_result.get("effective_ground_contact_weight"),
            "effective_bbox_bottom_weight": best_result.get("effective_bbox_bottom_weight"),
            "effective_upright_weight": best_result.get("effective_upright_weight"),
            "effective_heading_prior_weight": best_result.get("effective_heading_prior_weight"),
            "effective_front_sign_penalty_weight": best_result.get("effective_front_sign_penalty_weight"),
            "visual_gate_factor": best_result.get("visual_gate_factor"),
            "visual_gate_reason": best_result.get("visual_gate_reason"),
            "temporal_anchor_visual_gate_passed": best_result.get("temporal_anchor_visual_gate_passed"),
            "temporal_anchor_visual_gate_reason": best_result.get("temporal_anchor_visual_gate_reason"),
            "temporal_anchor_visual_mask_drop": best_result.get("temporal_anchor_visual_mask_drop"),
            "temporal_anchor_visual_bbox_drop": best_result.get("temporal_anchor_visual_bbox_drop"),
            "temporal_anchor_visual_yaw_jump_deg": best_result.get("temporal_anchor_visual_yaw_jump_deg"),
            "temporal_anchor_visual_prior_mask_iou": best_result.get("temporal_anchor_visual_prior_mask_iou"),
            "temporal_anchor_visual_prior_bbox_iou": best_result.get("temporal_anchor_visual_prior_bbox_iou"),
            "temporal_anchor_visual_penalty": best_result.get("temporal_anchor_visual_penalty"),
            "visible_mask_bonus": best_result.get("visible_mask_bonus"),
            "visible_mask_bonus_weight": best_result.get("visible_mask_bonus_weight"),
            "visible_mask_iou": best_result.get("visible_mask_iou"),
            "visible_soft_mask_iou": best_result.get("visible_soft_mask_iou"),
            "visible_target_fraction": best_result.get("visible_target_fraction"),
            "visible_target_area_px": best_result.get("visible_target_area_px"),
            "visible_bbox_iou": best_result.get("visible_bbox_iou"),
            "visible_bbox_center_error_px": best_result.get("visible_bbox_center_error_px"),
            "visible_contour_score": best_result.get("visible_contour_score"),
            "visible_contour_chamfer_score": best_result.get("visible_contour_chamfer_score"),
            "visible_profile_score": best_result.get("visible_profile_score"),
            "visible_contour_mean_distance_px": best_result.get("visible_contour_mean_distance_px"),
            "visible_profile_mean_distance_px": best_result.get("visible_profile_mean_distance_px"),
            "effective_visible_contour_weight": best_result.get("effective_visible_contour_weight"),
            "truncation_severity": best_result.get("truncation_severity", truncation_info.get("truncation_severity")),
            "low_observability": best_result.get("low_observability", truncation_info.get("low_observability")),
            "truncation_observability_score": best_result.get("truncation_observability_score", truncation_info.get("truncation_observability_score")),
            "truncation_observability_reasons": best_result.get("truncation_observability_reasons", truncation_info.get("truncation_observability_reasons")),
            "truncated_visual_quality_gate": best_result.get("truncated_visual_quality_gate"),
            "truncated_visual_quality_reason": best_result.get("truncated_visual_quality_reason"),
            "truncated_visual_bbox_factor": best_result.get("truncated_visual_bbox_factor"),
            "truncated_visual_center_factor": best_result.get("truncated_visual_center_factor"),
            "truncated_visual_mask_factor": best_result.get("truncated_visual_mask_factor"),
            "truncated_visual_contour_factor": best_result.get("truncated_visual_contour_factor"),
            "truncated_visual_profile_factor": best_result.get("truncated_visual_profile_factor"),
            "truncated_visual_overflow_factor": best_result.get("truncated_visual_overflow_factor"),
            "truncated_visual_overflow_loss": best_result.get("truncated_visual_overflow_loss"),
            "truncated_visual_quality_penalty": best_result.get("truncated_visual_quality_penalty"),
            "severe_truncation_gate_passed": best_result.get("severe_truncation_gate_passed"),
            "severe_truncation_gate_reasons": best_result.get("severe_truncation_gate_reasons"),
            "visible_projected_bbox": best_result.get("visible_projected_bbox"),
            "visible_target_bbox": best_result.get("visible_target_bbox"),
            "visible_bbox_source": best_result.get("visible_bbox_source"),
            "truncated_bbox_source": best_result.get("truncated_bbox_source"),
            "final_ground_constrained_selected": best_result.get("final_ground_constrained_selected"),
            "final_ground_constrained_rank_score": best_result.get("final_ground_constrained_rank_score"),
            "final_selection_mode": best_result.get("final_selection_mode"),
            "non_truncated_visual_ground_rescue_gate": best_result.get("non_truncated_visual_ground_rescue_gate"),
            "final_visual_selection_weights": best_result.get("final_visual_selection_weights"),
            "effective_visual_bbox_weight": best_result.get("effective_visual_bbox_weight"),
            "effective_hard_mask_weight": best_result.get("effective_hard_mask_weight"),
            "mask_iou": best_result["mask_iou"],
            "soft_mask_iou": best_result.get("soft_mask_iou"),
            "bbox_iou": best_result["bbox_iou"],
            "bbox_center_error_px": best_result["bbox_center_error_px"],
            "projected_bbox": best_result["projected_bbox"],
        },
        "outputs": {
            "alignment_collage": str(collage_paths["alignment_collage"]),
            "pose_closeup_collage": str(collage_paths["pose_closeup_collage"]),
            "model_reference_collage": str(collage_paths["model_reference_collage"]),
            "temporal_edge_debug": str(temporal_debug_path) if temporal_debug_path else None,
            "grabcut_debug": str(grabcut_debug_path) if grabcut_debug_path else None,
            "optimization_history": str(output_dir / "optimization_history.csv"),
            "optimization_report": str(output_dir / "optimization_report.json"),
            "optimized_task": str(output_dir / "task_with_optimized_corrected_pose.json"),
        },
    }
    if render_validation_outputs:
        report["render_validation"] = render_validation_outputs
    if args.profile_timings:
        report["profiling"] = fast.combine_profile_stats([proxy_evaluator, full_evaluator])
    report["refined_pose_candidates"] = [
        temporal_fast.refined_pose_candidate_summary(item[0], t_world_from_cam=t_world_from_cam)
        for item in refined_results
    ]
    with (output_dir / "optimization_report.json").open("w", encoding="utf-8") as f:
        json.dump(fast.to_builtin(report), f, indent=2)
    fast.cleanup_result_images(output_dir)
    proxy_evaluator.close()
    full_evaluator.close()
    return report


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def add_grabcut_arguments(parser: argparse.ArgumentParser) -> None:
    """Register GrabCut-specific arguments."""
    parser.add_argument(
        "--enable_grabcut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GrabCut mask enhancement.",
    )
    parser.add_argument("--grabcut_iters", type=int, default=5, help="GrabCut iteration count.")
    parser.add_argument("--grabcut_margin", type=int, default=10, help="Pixel margin to expand bbox for GrabCut.")
    parser.add_argument(
        "--mask_merge_mode",
        choices=["union", "grabcut_only", "original_only", "intersection"],
        default="union",
        help="How to merge GrabCut mask with the original mask.",
    )
    parser.add_argument(
        "--grabcut_min_area_ratio",
        type=float,
        default=0.20,
        help="Fallback if GrabCut area < this fraction of original mask area.",
    )
    parser.add_argument(
        "--grabcut_max_bbox_fill_ratio",
        type=float,
        default=0.95,
        help="Fallback if GrabCut area > this fraction of bbox area.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge contour fast optimizer with GrabCut mask enhancement, temporal prior, and edge assist."
    )
    temporal_fast.add_fast_arguments(parser)
    temporal_fast.add_temporal_arguments(parser)
    add_grabcut_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = optimize_sample(args)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1) from None
    metrics = report["metrics"]
    pose_world = report["optimized_corrected_pose_world"]
    print(f"task_id: {report['task_id']}")
    print(f"best_mask_iou: {metrics['mask_iou']:.6f}")
    print(f"best_bbox_iou: {metrics['bbox_iou']:.6f}")
    print(f"best_bbox_center_error_px: {metrics['bbox_center_error_px']:.6f}")
    print(f"best_projected_bbox: {metrics['projected_bbox']}")
    print(f"optimized_translation_world: {fast.to_builtin(pose_world['translation_world'])}")
    print(f"optimized_scale: {fast.to_builtin(pose_world['scale'])}")
    print(f"render_backend: {report['render_backend']}")
    grabcut = report.get("grabcut", {})
    print(f"grabcut_enabled: {grabcut.get('enabled')}")
    print(f"grabcut_succeeded: {grabcut.get('grabcut_succeeded')}")
    print(f"grabcut_fallback: {grabcut.get('fallback_used')}")


if __name__ == "__main__":
    main()
