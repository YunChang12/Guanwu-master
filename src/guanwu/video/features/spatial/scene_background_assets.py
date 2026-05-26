from __future__ import annotations

import base64
import json
import math
import zlib
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter

from guanwu.video.features.spatial.road_geometry import resolve_depth_maps_dir, select_road_plane_for_frame


DYNAMIC_LABELS = ("car", "truck", "bus", "van", "motorcycle", "bicycle", "person")
STATIC_GUARD_LABELS = ("fence", "road", "sidewalk", "rail", "wall", "track", "building")
ROAD_LABELS = ("road", "roadway", "asphalt", "lane", "street")
NON_ROAD_STATIC_LABELS = ("sidewalk", "rail", "track", "fence", "wall", "building", "grass", "curb")


def build_dynamic_mask(
    detections: dict[str, Any],
    image_shape: tuple[int, int],
    *,
    foreground_expand_px: int = 8,
    shadow_expand_px: int = 10,
) -> np.ndarray:
    height, width = image_shape
    dynamic = np.zeros((height, width), dtype=bool)
    for inst in detections.get("instances", []) or []:
        label = str(inst.get("concept_label") or inst.get("label") or inst.get("class_name") or "").lower()
        if any(token in label for token in STATIC_GUARD_LABELS):
            continue
        if not any(token in label for token in DYNAMIC_LABELS):
            continue
        mask = _decode_instance_mask(inst, (height, width))
        if mask is None:
            mask = _bbox_mask(inst.get("bbox"), (height, width))
        if not mask.any():
            continue
        bbox = inst.get("bbox") or _mask_bbox(mask)
        area = max(1.0, float(np.count_nonzero(mask)))
        adaptive = int(np.clip(math.sqrt(area) * 0.08, 4, 30))
        expand = min(max(int(foreground_expand_px), 0), adaptive) if foreground_expand_px > 0 else adaptive
        mask = _dilate(mask, expand)
        shadow = _shadow_mask(bbox, (height, width), shadow_expand_px)
        dynamic |= mask | shadow
    return dynamic


def build_static_guard_mask(
    detections: dict[str, Any],
    image_shape: tuple[int, int],
    *,
    expand_px: int = 2,
) -> np.ndarray:
    height, width = image_shape
    guard = np.zeros((height, width), dtype=bool)
    for inst in detections.get("instances", []) or []:
        label = str(inst.get("concept_label") or inst.get("label") or inst.get("class_name") or "").lower()
        if "road" in label:
            continue
        if not any(token in label for token in STATIC_GUARD_LABELS):
            continue
        mask = _decode_instance_mask(inst, (height, width))
        if mask is None:
            mask = _bbox_mask(inst.get("bbox"), (height, width))
        guard |= mask
    if expand_px > 0 and guard.any():
        guard = _dilate(guard, int(expand_px))
    return guard


def build_road_visible_mask(
    detections: dict[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    road = np.zeros((height, width), dtype=bool)
    for inst in detections.get("instances", []) or []:
        label = str(inst.get("concept_label") or inst.get("label") or inst.get("class_name") or "").lower()
        if not _is_road_label(label):
            continue
        mask = _decode_instance_mask(inst, (height, width))
        if mask is None:
            mask = _bbox_mask(inst.get("bbox"), (height, width))
        road |= mask
    if not road.any():
        return road
    static_guard = build_static_guard_mask(detections, (height, width), expand_px=0)
    if static_guard.any():
        road &= ~static_guard
    return _clean_semantic_road_mask(road)


def expand_road_mask_with_side_boundaries(
    road_mask: np.ndarray,
    *,
    static_guard_mask: np.ndarray | None = None,
) -> np.ndarray:
    road = road_mask.astype(bool)
    if not road.any():
        return road
    height, width = road.shape[:2]
    guard = static_guard_mask.astype(bool) if static_guard_mask is not None and static_guard_mask.shape == (height, width) else None
    sample = road.copy()
    if guard is not None:
        sample &= ~guard
    rows: list[int] = []
    lefts: list[float] = []
    rights: list[float] = []
    min_pixels = max(6, int(round(width * 0.015)))
    clip_margin = max(2, int(round(width * 0.01)))
    for y in range(height):
        xs = np.flatnonzero(sample[y])
        if len(xs) < min_pixels:
            continue
        left = float(xs.min())
        right = float(xs.max())
        if right - left + 1.0 < max(8.0, width * 0.04):
            continue
        if left <= clip_margin or right >= width - 1 - clip_margin:
            continue
        rows.append(y)
        lefts.append(left)
        rights.append(right)
    if len(rows) < max(8, int(round(height * 0.08))):
        return road
    row_arr = np.asarray(rows, dtype=np.float64)
    left_arr = np.asarray(lefts, dtype=np.float64)
    right_arr = np.asarray(rights, dtype=np.float64)
    left_fit = _fit_boundary_line(row_arr, left_arr)
    right_fit = _fit_boundary_line(row_arr, right_arr)
    if left_fit is None or right_fit is None:
        return road
    y0 = 0
    y1 = height - 1
    yy = np.arange(height, dtype=np.float64)
    left_line = left_fit[0] * yy + left_fit[1]
    right_line = right_fit[0] * yy + right_fit[1]
    min_width = max(6.0, width * 0.01)
    widths = right_line - left_line + 1.0
    valid_rows = (yy >= y0) & (yy <= y1) & (widths >= min_width) & (left_line < right_line)
    if int(np.count_nonzero(valid_rows)) < max(8, int(round(height * 0.08))):
        return road
    xx = np.arange(width, dtype=np.float64)[None, :]
    left_fill = np.clip(left_line, 0.0, width - 1.0)
    right_fill = np.clip(right_line, 0.0, width - 1.0)
    envelope = (xx >= left_fill[:, None]) & (xx <= right_fill[:, None])
    envelope &= valid_rows[:, None]
    expanded = envelope.copy()
    expanded |= road & valid_rows[:, None]
    if guard is not None:
        expanded &= ~guard
    return _fill_internal_holes(_bottom_connected_mask(expanded))


def build_road_full_mask_from_visible(
    road_visible_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    *,
    static_guard_mask: np.ndarray | None = None,
) -> np.ndarray:
    road = road_visible_mask.astype(bool).copy()
    shape = road.shape[:2]
    if not road.any():
        return road
    guard = static_guard_mask.astype(bool) if static_guard_mask is not None and static_guard_mask.shape == shape else None
    if guard is not None:
        road &= ~guard
    fill_candidates = dynamic_mask.astype(bool) if dynamic_mask is not None and dynamic_mask.shape == shape else np.zeros(shape, dtype=bool)
    if fill_candidates.any():
        near_road = cv2.dilate(road.astype(np.uint8), np.ones((17, 17), dtype=np.uint8), iterations=1) > 0
        fill = fill_candidates & near_road
        if guard is not None:
            fill &= ~guard
        road |= fill
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)) > 0
    road = expand_road_mask_with_side_boundaries(road, static_guard_mask=guard)
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)) > 0
    road = _bottom_connected_mask(road)
    if guard is not None:
        road &= ~guard
    return _fill_internal_holes(road)


def generate_target_frame_background_assets(
    *,
    summary_path: str | Path,
    output_dir: str | Path,
    target_frame_id: int = 3,
    road_geometry_path: str | Path | None = None,
    object_index_path: str | Path | None = None,
    depth_maps_dir: str | Path | None = None,
    camera_trajectory_path: str | Path | None = None,
    clean_depth_estimator: Callable[[Path], Any] | None = None,
    semantic_road_estimator: Callable[..., Any] | None = None,
    grid_stride: int = 4,
    top_k: int = 5,
) -> dict[str, str]:
    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame_entries = list(summary.get("frames", []) or [])
    if not frame_entries:
        raise ValueError(f"No frame entries in {summary_path}")
    target_entry = _select_frame(frame_entries, target_frame_id)
    target_det = _load_json(target_entry["detections"])
    target_rgb = _decode_image_b64(target_det["image_b64"])
    height, width = target_rgb.shape[:2]
    object_index_masks = _load_object_index_masks(object_index_path, (height, width))
    target_mask = build_dynamic_mask(target_det, (height, width))
    target_mask |= object_index_masks.get(int(target_frame_id), np.zeros((height, width), dtype=bool))
    target_road_visible_mask = build_road_visible_mask(target_det, (height, width))
    target_road_visible_mask |= _load_sidecar_road_mask(target_entry, target_det, (height, width), summary_path=summary_path)
    static_guard_mask = build_static_guard_mask(target_det, (height, width))

    source_rgbs: list[np.ndarray] = []
    source_weights: list[np.ndarray] = []
    source_count = np.zeros((height, width), dtype=np.uint16)
    road_visible_votes = np.zeros((height, width), dtype=np.uint16)
    road_full_votes = np.zeros((height, width), dtype=np.uint16)
    road_frame_count = 0
    limit_frames = frame_entries if top_k <= 0 else _rank_frames(frame_entries, target_frame_id)[: max(top_k, 1) * 8]
    for entry in limit_frames:
        det = _load_json(entry["detections"])
        if not det.get("image_b64"):
            continue
        rgb = _decode_image_b64(det["image_b64"])
        if rgb.shape[:2] != (height, width):
            rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))
        frame_id = int(entry.get("frame_idx") or det.get("frame_idx") or target_frame_id)
        mask = build_dynamic_mask(det, (height, width))
        mask |= object_index_masks.get(frame_id, np.zeros((height, width), dtype=bool))
        frame_static_guard = build_static_guard_mask(det, (height, width))
        static_guard_mask |= frame_static_guard
        frame_road_visible = build_road_visible_mask(det, (height, width))
        frame_road_visible |= _load_sidecar_road_mask(entry, det, (height, width), summary_path=summary_path)
        frame_road_full = build_road_full_mask_from_visible(
            frame_road_visible,
            mask,
            static_guard_mask=frame_static_guard,
        )
        if _is_usable_semantic_road_mask(frame_road_full):
            road_frame_count += 1
            road_visible_votes += frame_road_visible.astype(np.uint16)
            road_full_votes += frame_road_full.astype(np.uint16)
        usable = ~mask
        if not usable.any():
            continue
        temporal = 1.0 / (1.0 + abs(frame_id - target_frame_id))
        distance = cv2.distanceTransform(usable.astype(np.uint8), cv2.DIST_L2, 3)
        boundary = np.clip(distance / 12.0, 0.0, 1.0)
        target_penalty = 0.35 if frame_id == target_frame_id else 1.0
        weight = usable.astype(np.float32) * float(temporal) * float(target_penalty) * boundary.astype(np.float32)
        source_rgbs.append(rgb.astype(np.float32))
        source_weights.append(weight)
        source_count += usable.astype(np.uint16)

    if source_rgbs:
        clean_rgb = _robust_median_rgb(source_rgbs, source_weights, fallback=target_rgb)
        confidence = _confidence_from_weights(source_weights)
    else:
        clean_rgb = target_rgb.copy()
        confidence = np.zeros((height, width), dtype=np.float32)

    fallback_mask = (source_count < 2) & (~target_mask)
    clean_rgb[fallback_mask] = target_rgb[fallback_mask]
    clean_rgb = _fill_low_candidate_dynamic_regions(clean_rgb, target_rgb, target_mask, source_count)

    road_visible_mask = target_road_visible_mask
    road_full_mask = build_road_full_mask_from_visible(
        road_visible_mask,
        target_mask,
        static_guard_mask=static_guard_mask,
    )
    global_road_mask = _global_road_mask_from_votes(
        visible_votes=road_visible_votes,
        full_votes=road_full_votes,
        frame_count=road_frame_count,
        target_fallback=road_full_mask,
        static_guard_mask=static_guard_mask,
    )
    road_mask = global_road_mask if global_road_mask.any() else road_full_mask
    road_plane = _load_road_plane(road_geometry_path, target_frame_id)
    clean_rgb_path = output_dir / "clean_target_rgb.png"
    dynamic_mask_path = output_dir / "dynamic_mask_target.png"
    static_guard_mask_path = output_dir / "static_guard_mask_target.png"
    confidence_path = output_dir / "confidence_map.png"
    source_count_path = output_dir / "source_count_map.png"
    road_mask_path = output_dir / "road_mask.png"
    road_visible_mask_path = output_dir / "road_visible_mask.png"
    road_full_mask_path = output_dir / "road_full_mask.png"
    global_road_mask_path = output_dir / "global_road_full_mask.png"
    Image.fromarray(clean_rgb).save(clean_rgb_path)
    semantic_road_mask_path: Path | None = None
    semantic_road_estimator_used = False
    semantic_estimator_mask = _run_semantic_road_estimator(
        semantic_road_estimator,
        clean_rgb_path=clean_rgb_path,
        frame_id=int(target_frame_id),
        image_shape=(height, width),
    )
    if semantic_estimator_mask is not None and semantic_estimator_mask.any():
        semantic_road_estimator_used = True
        semantic_road_mask_path = output_dir / "semantic_road_estimator_mask.png"
        Image.fromarray((semantic_estimator_mask.astype(np.uint8) * 255)).save(semantic_road_mask_path)
        road_visible_mask |= semantic_estimator_mask
        road_full_mask = build_road_full_mask_from_visible(
            road_visible_mask,
            target_mask,
            static_guard_mask=static_guard_mask,
        )
        global_road_mask = _global_road_mask_from_votes(
            visible_votes=road_visible_votes,
            full_votes=road_full_votes,
            frame_count=road_frame_count,
            target_fallback=road_full_mask,
            static_guard_mask=static_guard_mask,
        )
        road_mask = global_road_mask if global_road_mask.any() else road_full_mask
    if road_plane is None:
        road_plane = _estimate_global_road_plane_from_semantic_depth(
            road_mask=road_mask,
            target_frame_id=target_frame_id,
            depth_maps_dir=depth_maps_dir,
            camera_trajectory_path=camera_trajectory_path,
        )
    Image.fromarray((target_mask.astype(np.uint8) * 255)).save(dynamic_mask_path)
    Image.fromarray((static_guard_mask.astype(np.uint8) * 255)).save(static_guard_mask_path)
    Image.fromarray(np.clip(confidence * 255.0, 0, 255).astype(np.uint8)).save(confidence_path)
    Image.fromarray(np.clip(source_count, 0, 255).astype(np.uint8)).save(source_count_path)
    Image.fromarray((road_mask.astype(np.uint8) * 255)).save(road_mask_path)
    Image.fromarray((road_visible_mask.astype(np.uint8) * 255)).save(road_visible_mask_path)
    Image.fromarray((road_full_mask.astype(np.uint8) * 255)).save(road_full_mask_path)
    Image.fromarray((global_road_mask.astype(np.uint8) * 255)).save(global_road_mask_path)

    road_mesh = mesh_dir / "road_mesh.obj"
    structures_mesh = mesh_dir / "structures_mesh.obj"
    far_mesh = mesh_dir / "far_mesh.obj"
    _write_textured_grid_obj(
        road_mesh,
        output_dir,
        clean_rgb_path,
        width,
        height,
        mask=road_mask,
        grid_stride=grid_stride,
        layer="road",
        road_plane=road_plane,
    )
    _write_textured_grid_obj(
        structures_mesh,
        output_dir,
        clean_rgb_path,
        width,
        height,
        mask=(~road_mask) & (~target_mask),
        grid_stride=grid_stride,
        layer="structures",
        road_plane=None,
    )
    _write_far_mesh(far_mesh, output_dir, clean_rgb_path, width, height)

    manifest_path = output_dir / "background_manifest.json"
    manifest = {
        "schema": "guanwu.target_frame_background_assets.v1",
        "target_frame_id": int(target_frame_id),
        "image_size": [int(width), int(height)],
        "assets": {
            "clean_rgb": str(clean_rgb_path),
            "dynamic_mask": str(dynamic_mask_path),
            "static_guard_mask": str(static_guard_mask_path),
            "confidence_map": str(confidence_path),
            "source_count_map": str(source_count_path),
            "road_mask": str(road_mask_path),
            "road_visible_mask": str(road_visible_mask_path),
            "road_full_mask": str(road_full_mask_path),
            "global_road_full_mask": str(global_road_mask_path),
            **({"semantic_road_estimator_mask": str(semantic_road_mask_path)} if semantic_road_mask_path else {}),
            "road_mesh": str(road_mesh),
            "structures_mesh": str(structures_mesh),
            "far_mesh": str(far_mesh),
        },
        "quality": {
            "source_frame_count": len(source_rgbs),
            "target_dynamic_fraction": float(np.mean(target_mask)),
            "road_fraction": float(np.mean(road_mask)),
            "road_visible_fraction": float(np.mean(road_visible_mask)),
            "road_full_fraction": float(np.mean(road_full_mask)),
            "road_mask_source": (
                "semantic_estimator"
                if semantic_road_estimator_used
                else ("semantic_multiframe" if road_frame_count > 0 else "semantic_target_or_empty")
            ),
            "road_semantic_frame_count": int(road_frame_count),
            "mean_confidence": float(np.mean(confidence)),
        },
        "road_plane": road_plane,
    }
    depth_calibration_mask = road_visible_mask & (~target_mask)
    if int(np.count_nonzero(depth_calibration_mask)) < min(128, max(16, int(depth_calibration_mask.size // 64))):
        depth_calibration_mask = (~target_mask) & (~static_guard_mask)
    depth_asset = _try_generate_depth_background_asset_from_estimator(
        clean_rgb_path=clean_rgb_path,
        output_dir=output_dir,
        target_frame_id=target_frame_id,
        camera_trajectory_path=camera_trajectory_path,
        depth_maps_dir=depth_maps_dir,
        calibration_mask=depth_calibration_mask,
        grid_stride=grid_stride,
        clean_depth_estimator=clean_depth_estimator,
    )
    if not depth_asset:
        depth_asset = _try_generate_depth_background_asset(
            clean_rgb_path=clean_rgb_path,
            output_dir=output_dir,
            target_frame_id=target_frame_id,
            depth_maps_dir=depth_maps_dir,
            camera_trajectory_path=camera_trajectory_path,
            grid_stride=grid_stride,
        )
    if depth_asset:
        manifest["schema"] = "guanwu.target_frame_background_assets.v2"
        manifest["assets"].update(depth_asset["assets"])
        manifest["quality"].update(depth_asset["quality"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "mesh_dir": str(mesh_dir)}


def load_background_asset_meshes(
    background_assets_manifest: str | Path | None,
    *,
    road_geometry_path: str | Path | None = None,
    camera_trajectory_path: str | Path | None = None,
) -> list[tuple[str, Path]]:
    if not background_assets_manifest:
        return []
    manifest_path = Path(background_assets_manifest)
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = data.get("assets", {})
    depth_bg = assets.get("depth_background_glb") or assets.get("depth_background_mesh")
    if depth_bg:
        path = Path(depth_bg)
        if path.exists():
            multiframe_assets = _build_multiframe_global_background_assets(
                manifest_path=manifest_path,
                manifest=data,
                depth_background_path=path,
                road_geometry_path=road_geometry_path,
                camera_trajectory_path=camera_trajectory_path,
            )
            if multiframe_assets:
                return multiframe_assets
            return [("depth_background", path)]
    ordered = [
        ("road", assets.get("road_mesh")),
        ("structures", assets.get("structures_mesh")),
        ("far", assets.get("far_mesh")),
    ]
    out: list[tuple[str, Path]] = []
    for name, raw in ordered:
        if not raw:
            continue
        path = Path(raw)
        if path.exists():
            out.append((name, path))
    return out


def _build_multiframe_global_background_assets(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    depth_background_path: Path,
    road_geometry_path: str | Path | None,
    camera_trajectory_path: str | Path | None,
) -> list[tuple[str, Path]] | None:
    if not road_geometry_path or not camera_trajectory_path:
        return None
    assets = manifest.get("assets", {})
    clean_rgb_path = assets.get("clean_rgb")
    if not clean_rgb_path:
        return None
    clean_rgb_file = Path(clean_rgb_path)
    if not clean_rgb_file.exists():
        return None
    target_frame_id = int(manifest.get("target_frame_id", 0) or 0)
    if target_frame_id <= 0:
        return None
    road_geometry = _load_json(road_geometry_path)
    road_plane = select_road_plane_for_frame(road_geometry, target_frame_id, policy="global_for_fixed_camera")
    camera = _camera_for_frame(camera_trajectory_path, target_frame_id)
    if not road_plane or camera is None:
        return None
    depth_maps_dir = resolve_depth_maps_dir(road_geometry.get("depth_maps_dir"))
    if depth_maps_dir is None:
        return None
    depth_files = _depth_map_files(depth_maps_dir)
    if not depth_files:
        return None

    out_dir = depth_background_path.parent
    road_out = out_dir / "road_surface_global_multiframe_v1.glb"
    static_out = out_dir / "static_background_multiframe_no_road_v1.glb"
    mask_out = out_dir / "road_support_global_multiframe_v1.png"
    try:
        source_mtimes = [
            Path(__file__).stat().st_mtime,
            clean_rgb_file.stat().st_mtime,
            depth_background_path.stat().st_mtime,
            Path(road_geometry_path).stat().st_mtime,
            Path(camera_trajectory_path).stat().st_mtime,
            manifest_path.stat().st_mtime,
        ]
        for key in ("global_road_full_mask", "road_full_mask", "road_mask", "dynamic_mask", "static_guard_mask"):
            raw = assets.get(key)
            if raw and Path(raw).exists():
                source_mtimes.append(Path(raw).stat().st_mtime)
        source_mtimes.extend(path.stat().st_mtime for path in depth_files)
        source_mtime = max(source_mtimes)
        if road_out.exists() and static_out.exists() and min(road_out.stat().st_mtime, static_out.stat().st_mtime) >= source_mtime:
            return [("road_surface", road_out), ("static_background", static_out)]
    except OSError:
        pass

    try:
        rgb = np.asarray(Image.open(clean_rgb_file).convert("RGB"))
    except Exception:
        return None
    height, width = rgb.shape[:2]
    road_mask = _read_optional_mask(
        assets.get("global_road_full_mask") or assets.get("road_full_mask") or assets.get("road_mask"),
        (height, width),
    )
    dynamic_mask = _read_optional_mask(assets.get("dynamic_mask"), (height, width))
    static_guard_mask = _read_optional_mask(assets.get("static_guard_mask"), (height, width))

    depth_stack = _load_depth_stack_for_shape(depth_files, (height, width))
    if depth_stack is None:
        return None
    road_support = _semantic_road_support_mask(
        road_mask=road_mask,
        dynamic_mask=dynamic_mask,
        static_guard_mask=static_guard_mask,
        shape=(height, width),
    )
    if int(np.count_nonzero(road_support)) == 0:
        return None

    road_mesh = _build_road_surface_mesh_from_mask(
        rgb=rgb,
        road_mask=road_support,
        camera=camera,
        road_plane=road_plane,
    )
    if road_mesh is None:
        return None

    static_remove_radius = max(1, min(2, int(round(width * 0.003))))
    static_remove_kernel = np.ones((static_remove_radius * 2 + 1, static_remove_radius * 2 + 1), dtype=np.uint8)
    static_remove = cv2.dilate(road_support.astype(np.uint8), static_remove_kernel, iterations=1) > 0
    if dynamic_mask is not None:
        near_road = cv2.dilate(road_support.astype(np.uint8), np.ones((13, 13), dtype=np.uint8), iterations=1) > 0
        static_remove |= dynamic_mask & near_road
    static_depth = _robust_multiframe_depth(depth_stack)
    static_mask = np.isfinite(static_depth) & (static_depth > 1e-6) & (~static_remove)
    static_mesh = _build_masked_depth_mesh(
        rgb=rgb,
        depth=static_depth,
        mask=static_mask,
        camera=camera,
        grid_stride=4,
        max_depth=120.0,
    )
    if static_mesh is None:
        return None

    try:
        road_mesh.export(str(road_out))
        static_mesh.export(str(static_out))
        Image.fromarray((road_support.astype(np.uint8) * 255)).save(mask_out)
    except Exception:
        return None
    return [("road_surface", road_out), ("static_background", static_out)]


def _depth_map_files(depth_maps_dir: Path) -> list[Path]:
    try:
        return sorted(depth_maps_dir.glob("*.npy"), key=lambda path: int(path.stem))
    except Exception:
        return sorted(depth_maps_dir.glob("*.npy"))


def _read_optional_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        return None
    mask = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _run_semantic_road_estimator(
    estimator: Callable[..., Any] | None,
    *,
    clean_rgb_path: Path,
    frame_id: int,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if estimator is None:
        return None
    try:
        result = estimator(clean_rgb_path, frame_id=int(frame_id))
    except TypeError:
        result = estimator(clean_rgb_path)
    except Exception:
        return None
    if result is None:
        return None
    mask_value = result.get("mask") if isinstance(result, dict) else result
    if mask_value is None:
        return None
    if isinstance(mask_value, (str, Path)):
        return _read_optional_mask(mask_value, image_shape)
    try:
        mask = np.asarray(mask_value)
    except Exception:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        return None
    if mask.shape != image_shape:
        mask = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _load_sidecar_road_mask(
    entry: dict[str, Any],
    detections: dict[str, Any],
    shape: tuple[int, int],
    *,
    summary_path: Path | None = None,
) -> np.ndarray:
    candidates: list[Path] = []
    for key in ("road_mask", "road_visible_mask", "road_full_mask"):
        raw = entry.get(key) or detections.get(key)
        if raw:
            candidates.append(Path(raw))
    det_path = entry.get("detections")
    if det_path:
        frame_dir = Path(det_path).parent
        candidates.extend(
            [
                frame_dir / "road" / "road_mask.png",
                frame_dir / "road_mask.png",
                frame_dir / "road_visible_mask.png",
            ]
        )
    frame_id = int(entry.get("frame_idx") or detections.get("frame_idx") or 0)
    if summary_path is not None and frame_id > 0:
        project_outputs = _find_project_outputs_dir(summary_path)
        if project_outputs is not None:
            candidates.extend(
                [
                    project_outputs / "road_gsam2_probe" / f"frame_{frame_id:06d}" / "road_mask.png",
                    project_outputs / "road_gsam2_probe" / f"frame_{frame_id:06d}" / "road_visible_mask.png",
                ]
            )
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        mask = _read_optional_mask(candidate, shape)
        if mask is not None:
            out |= mask
    return _clean_semantic_road_mask(out) if out.any() else out


def _find_project_outputs_dir(path: Path) -> Path | None:
    current = Path(path).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "outputs":
            return parent
    return None


def _is_road_label(label: str) -> bool:
    text = str(label or "").lower()
    if any(token in text for token in NON_ROAD_STATIC_LABELS):
        return False
    return any(token in text for token in ROAD_LABELS)


def _clean_semantic_road_mask(mask: np.ndarray) -> np.ndarray:
    road = mask.astype(bool)
    if not road.any():
        return road
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)) > 0
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)) > 0
    return _bottom_connected_mask(road)


def _is_usable_semantic_road_mask(mask: np.ndarray) -> bool:
    if mask is None or not mask.any():
        return False
    height, width = mask.shape[:2]
    area = float(np.mean(mask))
    if area < 0.005 or area > 0.85:
        return False
    bottom_band = mask[int(round(height * 0.75)) :, :]
    if float(np.mean(bottom_band)) < 0.01:
        return False
    return True


def _global_road_mask_from_votes(
    *,
    visible_votes: np.ndarray,
    full_votes: np.ndarray,
    frame_count: int,
    target_fallback: np.ndarray,
    static_guard_mask: np.ndarray,
) -> np.ndarray:
    shape = target_fallback.shape[:2]
    guard = static_guard_mask.astype(bool) if static_guard_mask.shape == shape else np.zeros(shape, dtype=bool)
    if frame_count <= 0:
        out = target_fallback.astype(bool).copy()
        out &= ~guard
        return _clean_semantic_road_mask(out) if out.any() else out
    visible_votes = visible_votes.astype(np.uint16)
    full_votes = full_votes.astype(np.uint16)
    threshold = max(1, int(math.ceil(float(frame_count) * 0.35)))
    road = full_votes >= threshold
    road |= visible_votes >= 1
    road &= ~guard
    if target_fallback.any():
        road |= target_fallback.astype(bool) & ~guard
    if not road.any():
        return road
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)) > 0
    road = expand_road_mask_with_side_boundaries(road, static_guard_mask=guard)
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)) > 0
    road &= ~guard
    return _bottom_connected_mask(road)


def _load_depth_stack_for_shape(depth_files: list[Path], shape: tuple[int, int], *, max_frames: int = 96) -> np.ndarray | None:
    if not depth_files:
        return None
    if len(depth_files) > max_frames:
        indices = np.linspace(0, len(depth_files) - 1, max_frames, dtype=int)
        selected = [depth_files[int(index)] for index in indices]
    else:
        selected = depth_files
    depth_maps: list[np.ndarray] = []
    height, width = shape
    for path in selected:
        try:
            depth = np.load(str(path)).astype(np.float64)
        except Exception:
            continue
        if depth.ndim == 3:
            depth = depth[0]
        if depth.ndim != 2:
            continue
        if depth.shape != (height, width):
            depth = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        depth_maps.append(depth)
    if not depth_maps:
        return None
    return np.stack(depth_maps, axis=0)


def _semantic_road_support_mask(
    *,
    road_mask: np.ndarray | None,
    dynamic_mask: np.ndarray | None,
    static_guard_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if road_mask is None or road_mask.shape != shape:
        return np.zeros(shape, dtype=bool)
    road = road_mask.astype(bool).copy()
    static_guard = _expand_static_guard_mask(static_guard_mask, shape)
    if static_guard is not None:
        road &= ~static_guard
    if dynamic_mask is not None and dynamic_mask.shape == shape and dynamic_mask.any() and road.any():
        corridor = _road_corridor_from_seed(road, shape)
        near_road = cv2.dilate(road.astype(np.uint8), np.ones((17, 17), dtype=np.uint8), iterations=1) > 0
        fill = dynamic_mask.astype(bool) & near_road
        if corridor is not None:
            fill &= corridor
        if static_guard is not None:
            fill &= ~static_guard
        road |= fill
    if not road.any():
        return road
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)) > 0
    road = expand_road_mask_with_side_boundaries(road, static_guard_mask=static_guard)
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)) > 0
    road = _bottom_connected_mask(road)
    if static_guard is not None:
        road &= ~static_guard
    return road


def _estimate_global_road_plane_from_semantic_depth(
    *,
    road_mask: np.ndarray,
    target_frame_id: int,
    depth_maps_dir: str | Path | None,
    camera_trajectory_path: str | Path | None,
) -> dict[str, Any] | None:
    if road_mask is None or not road_mask.any() or not depth_maps_dir or not camera_trajectory_path:
        return None
    shape = road_mask.shape[:2]
    depth_dir = resolve_depth_maps_dir(depth_maps_dir)
    if depth_dir is None:
        return None
    depth_files = _depth_map_files(depth_dir)
    if not depth_files:
        return None
    camera = _camera_for_frame(camera_trajectory_path, int(target_frame_id))
    if camera is None:
        return None

    if len(depth_files) > 96:
        indices = np.linspace(0, len(depth_files) - 1, 96, dtype=int)
        depth_files = [depth_files[int(index)] for index in indices]
    planes: list[tuple[np.ndarray, float, dict[str, float]]] = []
    sample_mask = road_mask.astype(bool)
    min_samples = min(512, max(80, int(sample_mask.size // 128)))
    for depth_path in depth_files:
        try:
            depth = np.load(str(depth_path)).astype(np.float64)
        except Exception:
            continue
        if depth.ndim == 3:
            depth = depth[0]
        if depth.ndim != 2:
            continue
        if depth.shape != shape:
            depth = cv2.resize(depth.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        valid = sample_mask & np.isfinite(depth) & (depth > 1e-6)
        if int(np.count_nonzero(valid)) < min_samples:
            continue
        points = _camera_depth_points_for_mask(depth, valid, camera)
        fit = _fit_plane_from_world_points(points)
        if fit is None:
            continue
        normal, offset, stats = fit
        planes.append((normal, offset, stats))
    if not planes:
        return None

    normals = np.stack([item[0] for item in planes], axis=0)
    reference = normals[0]
    for idx in range(len(normals)):
        if float(normals[idx] @ reference) < 0.0:
            normals[idx] = -normals[idx]
    weights = np.asarray(
        [max(1e-3, item[2]["inlier_ratio"] / max(item[2]["rmse_m"], 1e-3)) for item in planes],
        dtype=np.float64,
    )
    normal = np.sum(normals * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-8)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    offsets = np.asarray([item[1] for item in planes], dtype=np.float64)
    offset = _weighted_quantile(offsets, weights, 0.5)
    scene_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if float(normal @ scene_up) < 0.0:
        normal = -normal
        offset = -float(offset)
    return {
        "source": "weighted_keyframe_mean+semantic_bg_depth_tar",
        "normal_world": [float(v) for v in normal],
        "offset": float(offset),
        "quality": {
            "keyframe_count": int(len(planes)),
            "mean_inlier_ratio": float(np.mean([item[2]["inlier_ratio"] for item in planes])),
            "mean_rmse_m": float(np.mean([item[2]["rmse_m"] for item in planes])),
        },
        "selection": {
            "mode": "global",
            "policy": "global_for_fixed_camera",
            "target_frame_id": int(target_frame_id),
        },
    }


def _camera_depth_points_for_mask(depth: np.ndarray, mask: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) > 50000:
        indices = np.linspace(0, len(xs) - 1, 50000, dtype=int)
        xs = xs[indices]
        ys = ys[indices]
    z = depth[ys, xs].astype(np.float64)
    fx = float(camera.get("fx", max(depth.shape) * 0.8))
    fy = float(camera.get("fy", max(depth.shape) * 0.8))
    cx = float(camera.get("cx", depth.shape[1] * 0.5))
    cy = float(camera.get("cy", depth.shape[0] * 0.5))
    points_cam = np.column_stack(
        [
            (xs.astype(np.float64) - cx) * z / max(abs(fx), 1e-8),
            (ys.astype(np.float64) - cy) * z / max(abs(fy), 1e-8),
            z,
        ]
    )
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return (rotation @ points_cam.T).T + translation


def _fit_plane_from_world_points(points: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]] | None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 80:
        return None
    centroid = np.median(points, axis=0)
    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    offset = -float(normal @ centroid)
    distances = points @ normal + offset
    abs_dist = np.abs(distances)
    med = float(np.median(abs_dist))
    mad = float(np.median(np.abs(abs_dist - med)))
    threshold = max(0.08, med + 3.0 * mad)
    inliers = abs_dist <= threshold
    if int(np.count_nonzero(inliers)) < 80:
        return None
    centroid = np.mean(points[inliers], axis=0)
    centered = points[inliers] - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    offset = -float(normal @ centroid)
    residual = points @ normal + offset
    residual_in = residual[inliers]
    return normal, offset, {
        "inlier_count": int(np.count_nonzero(inliers)),
        "candidate_count": int(len(points)),
        "inlier_ratio": float(np.mean(inliers)),
        "rmse_m": float(np.sqrt(np.mean(residual_in**2))),
        "p95_abs_m": float(np.percentile(np.abs(residual_in), 95)),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(np.median(values))
    cdf = np.cumsum(weights) / total
    idx = int(np.searchsorted(cdf, float(quantile), side="left"))
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def _expand_static_guard_mask(static_guard_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if static_guard_mask is None or static_guard_mask.shape != shape:
        return None
    guard = static_guard_mask.astype(bool)
    if not guard.any():
        return None
    height, width = shape
    radius = max(1, int(round(min(height, width) * 0.006)))
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(guard.astype(np.uint8), kernel, iterations=1) > 0


def _fit_boundary_line(rows: np.ndarray, xs: np.ndarray) -> tuple[float, float] | None:
    rows = np.asarray(rows, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if len(rows) != len(xs) or len(rows) < 2:
        return None
    keep = np.ones(len(rows), dtype=bool)
    coeff: np.ndarray | None = None
    for _ in range(3):
        if int(np.count_nonzero(keep)) < 2:
            break
        try:
            coeff = np.polyfit(rows[keep], xs[keep], 1)
        except Exception:
            return None
        fitted = np.polyval(coeff, rows)
        residual = xs - fitted
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        keep = np.abs(residual - med) <= max(3.0, mad * 3.0)
    if coeff is None or not np.isfinite(coeff).all():
        return None
    return float(coeff[0]), float(coeff[1])


def _road_corridor_from_seed(seed_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if seed_mask is None or seed_mask.shape != shape:
        return None
    seed = seed_mask.astype(bool)
    height, width = shape
    row_bounds: list[tuple[int, float, float]] = []
    min_width = max(6.0, width * 0.05)
    for y in range(height):
        xs = np.flatnonzero(seed[y])
        if len(xs) < min_width:
            continue
        row_bounds.append((y, float(xs.min()), float(xs.max())))
    if len(row_bounds) < 2:
        return None
    rows = np.array([item[0] for item in row_bounds], dtype=np.float64)
    left = np.array([item[1] for item in row_bounds], dtype=np.float64)
    right = np.array([item[2] for item in row_bounds], dtype=np.float64)
    all_rows = np.arange(height, dtype=np.float64)
    left_interp = np.interp(all_rows, rows, left, left[0], left[-1])
    right_interp = np.interp(all_rows, rows, right, right[0], right[-1])
    width_interp = np.maximum(right_interp - left_interp + 1.0, 1.0)
    margin = np.clip(width_interp * 0.18, 3.0, max(4.0, width * 0.08))
    # Avoid extrapolating a bottom-row road seed into the far sky/structures.
    top_seed = int(rows.min())
    top_margin = max(4, int(round(height * 0.08)))
    valid_rows = all_rows >= max(0, top_seed - top_margin)
    xx = np.arange(width, dtype=np.float64)[None, :]
    corridor = (xx >= (left_interp[:, None] - margin[:, None])) & (xx <= (right_interp[:, None] + margin[:, None]))
    corridor &= valid_rows[:, None]
    return corridor


def _signed_distance_images_to_plane(depth_stack: np.ndarray, camera: dict[str, Any], road_plane: dict[str, Any]) -> np.ndarray:
    depth_stack = np.asarray(depth_stack, dtype=np.float64)
    frame_count, height, width = depth_stack.shape
    fx = float(camera.get("fx", max(width, height) * 0.8))
    fy = float(camera.get("fy", max(width, height) * 0.8))
    cx = float(camera.get("cx", width * 0.5))
    cy = float(camera.get("cy", height * 0.5))
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    normal = np.asarray(road_plane.get("normal_world", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return np.full((frame_count, height, width), np.nan, dtype=np.float64)
    normal = normal / norm
    offset = float(road_plane.get("offset", 0.0))
    yy, xx = np.indices((height, width), dtype=np.float64)
    ray_x = (xx - cx) / max(abs(fx), 1e-8)
    ray_y = (yy - cy) / max(abs(fy), 1e-8)
    ray_cam = np.stack([ray_x, ray_y, np.ones_like(ray_x)], axis=-1)
    ray_world_dot = (ray_cam.reshape(-1, 3) @ rotation.T @ normal).reshape(height, width)
    origin_dot = float(translation @ normal) + offset
    return depth_stack * ray_world_dot[None, :, :] + origin_dot


def _bottom_connected_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels = cv2.connectedComponents(mask_u8, connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    bottom_labels = np.unique(labels[max(0, labels.shape[0] - 3) :, :])
    bottom_labels = bottom_labels[bottom_labels > 0]
    if len(bottom_labels) == 0:
        areas = np.bincount(labels.reshape(-1), minlength=num)
        areas[0] = 0
        keep = int(np.argmax(areas))
        return labels == keep
    return np.isin(labels, bottom_labels)


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(bool).copy()
    if not out.any():
        return out
    inv = (~out).astype(np.uint8)
    num, labels = cv2.connectedComponents(inv, connectivity=8)
    if num <= 1:
        return out
    height, width = out.shape[:2]
    border_labels = set(int(v) for v in np.unique(labels[0, :]))
    border_labels.update(int(v) for v in np.unique(labels[height - 1, :]))
    border_labels.update(int(v) for v in np.unique(labels[:, 0]))
    border_labels.update(int(v) for v in np.unique(labels[:, width - 1]))
    for label in range(1, num):
        if label in border_labels:
            continue
        out[labels == label] = True
    return out


def _robust_multiframe_depth(depth_stack: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_stack, dtype=np.float64)
    depth = np.where(np.isfinite(depth) & (depth > 1e-6), depth, np.nan)
    with np.errstate(all="ignore"):
        median = np.nanmedian(depth, axis=0)
    return median.astype(np.float64)


def _build_masked_depth_mesh(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    camera: dict[str, Any],
    grid_stride: int,
    max_depth: float,
):
    import trimesh
    from trimesh.visual import ColorVisuals

    height, width = depth.shape
    stride = max(1, int(grid_stride))
    xs = list(range(0, width, stride))
    ys = list(range(0, height, stride))
    if xs[-1] != width - 1:
        xs.append(width - 1)
    if ys[-1] != height - 1:
        ys.append(height - 1)

    fx = float(camera.get("fx", max(width, height) * 0.8))
    fy = float(camera.get("fy", max(width, height) * 0.8))
    cx = float(camera.get("cx", width * 0.5))
    cy = float(camera.get("cy", height * 0.5))
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)

    vertices: list[list[float]] = []
    colors: list[list[int]] = []
    valid_index: dict[tuple[int, int], int] = {}
    mask = mask.astype(bool)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            d = float(depth[y, x])
            if not mask[y, x] or not math.isfinite(d) or d <= 0.01 or d > max_depth:
                continue
            point_cam = np.array([(float(x) - cx) * d / fx, (float(y) - cy) * d / fy, d], dtype=np.float64)
            point_world = rotation @ point_cam + translation
            valid_index[(yi, xi)] = len(vertices)
            vertices.append([float(value) for value in point_world])
            colors.append([int(value) for value in rgb[y, x, :3]] + [255])

    faces: list[list[int]] = []
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            keys = [(yi, xi), (yi, xi + 1), (yi + 1, xi), (yi + 1, xi + 1)]
            if any(key not in valid_index for key in keys):
                continue
            y_mid = int(round((ys[yi] + ys[yi + 1]) * 0.5))
            x_mid = int(round((xs[xi] + xs[xi + 1]) * 0.5))
            if not mask[min(max(y_mid, 0), height - 1), min(max(x_mid, 0), width - 1)]:
                continue
            z_values = [float(depth[ys[key[0]], xs[key[1]]]) for key in keys]
            if max(z_values) / max(min(z_values), 1e-6) > 1.8:
                continue
            v00 = valid_index[(yi, xi)]
            v10 = valid_index[(yi, xi + 1)]
            v01 = valid_index[(yi + 1, xi)]
            v11 = valid_index[(yi + 1, xi + 1)]
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if not vertices or not faces:
        return None
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=np.asarray(colors, dtype=np.uint8))
    return mesh


def _project_world_vertices_to_image(vertices: np.ndarray, image_shape: tuple[int, int], camera: dict[str, Any]) -> dict[str, np.ndarray]:
    height, width = image_shape
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    fx = float(camera.get("fx", max(width, height) * 0.8))
    fy = float(camera.get("fy", max(width, height) * 0.8))
    cx = float(camera.get("cx", width * 0.5))
    cy = float(camera.get("cy", height * 0.5))
    points_cam = (rotation.T @ (np.asarray(vertices, dtype=np.float64) - translation).T).T
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    u = np.full(len(vertices), -1, dtype=np.int64)
    v = np.full(len(vertices), -1, dtype=np.int64)
    u[valid] = np.rint(fx * points_cam[valid, 0] / z[valid] + cx).astype(np.int64)
    v[valid] = np.rint(fy * points_cam[valid, 1] / z[valid] + cy).astype(np.int64)
    inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return {"inside": inside, "u": u, "v": v, "points_cam": points_cam}


def _signed_distance_to_plane(points: np.ndarray, road_plane: dict[str, Any]) -> np.ndarray:
    normal = np.asarray(road_plane.get("normal_world", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return np.full(len(points), np.nan, dtype=np.float64)
    normal = normal / norm
    offset = float(road_plane.get("offset", 0.0))
    return np.asarray(points, dtype=np.float64) @ normal + offset


def _build_road_surface_mesh_from_mask(
    *,
    rgb: np.ndarray,
    road_mask: np.ndarray,
    camera: dict[str, Any],
    road_plane: dict[str, Any],
    grid_stride: int = 4,
):
    import trimesh
    from trimesh.visual import ColorVisuals

    height, width = road_mask.shape[:2]
    stride = max(1, int(grid_stride))
    xs = list(range(0, width, stride))
    ys = list(range(0, height, stride))
    if xs[-1] != width - 1:
        xs.append(width - 1)
    if ys[-1] != height - 1:
        ys.append(height - 1)

    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    index: dict[tuple[int, int], int] = {}
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            if not road_mask[y, x]:
                continue
            point = _camera_pixel_ray_plane_intersection(float(x), float(y), camera, road_plane)
            if point is None:
                continue
            index[(yi, xi)] = len(vertices)
            vertices.append(point)
            colors.append(rgb[y, x])

    faces: list[list[int]] = []
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            keys = ((yi, xi), (yi + 1, xi), (yi, xi + 1), (yi + 1, xi + 1))
            if not all(key in index for key in keys):
                continue
            y_mid = int(round((ys[yi] + ys[yi + 1]) * 0.5))
            x_mid = int(round((xs[xi] + xs[xi + 1]) * 0.5))
            if not road_mask[min(max(y_mid, 0), height - 1), min(max(x_mid, 0), width - 1)]:
                continue
            v00 = index[(yi, xi)]
            v10 = index[(yi + 1, xi)]
            v01 = index[(yi, xi + 1)]
            v11 = index[(yi + 1, xi + 1)]
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    if not vertices or not faces:
        return None
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=np.asarray(colors, dtype=np.uint8))
    return mesh


def _camera_pixel_ray_plane_intersection(
    x: float,
    y: float,
    camera: dict[str, Any],
    road_plane: dict[str, Any],
) -> np.ndarray | None:
    fx = float(camera.get("fx", 1.0))
    fy = float(camera.get("fy", 1.0))
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return None
    cx = float(camera.get("cx", 0.0))
    cy = float(camera.get("cy", 0.0))
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    origin = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    ray_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
    ray_world = rotation @ ray_cam
    normal = np.asarray(road_plane.get("normal_world", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    offset = float(road_plane.get("offset", 0.0))
    denom = float(normal @ ray_world)
    if abs(denom) < 1e-8:
        return None
    t = -float(normal @ origin + offset) / denom
    if not math.isfinite(t) or t <= 1e-6:
        return None
    point = origin + ray_world * t
    return point.astype(np.float64)


def generate_depth_background_mesh_assets(
    *,
    clean_rgb_path: str | Path,
    depth_path: str | Path,
    output_dir: str | Path,
    camera: dict[str, Any],
    target_frame_id: int,
    grid_stride: int = 4,
    max_depth: float = 120.0,
) -> dict[str, str]:
    import trimesh

    clean_rgb_path = Path(clean_rgb_path)
    depth_path = Path(depth_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(clean_rgb_path).convert("RGB"))
    depth = np.load(str(depth_path)).astype(np.float64)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got {depth.shape}")
    height, width = rgb.shape[:2]
    if depth.shape != (height, width):
        depth = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float64)

    mesh = _build_depth_textured_mesh(
        rgb=rgb,
        depth=depth,
        camera=camera,
        grid_stride=grid_stride,
        max_depth=max_depth,
    )
    glb_path = output_dir / "depth_background.glb"
    mesh.export(str(glb_path))

    depth_out = output_dir / "clean_target_depth.npy"
    if depth_path.resolve() != depth_out.resolve():
        np.save(depth_out, depth.astype(np.float32))
    else:
        depth_out = depth_path

    manifest_path = output_dir / "background_manifest.json"
    manifest = {
        "schema": "guanwu.target_frame_background_assets.v2",
        "target_frame_id": int(target_frame_id),
        "image_size": [int(width), int(height)],
        "assets": {
            "clean_rgb": str(clean_rgb_path),
            "clean_depth": str(depth_out),
            "depth_background_glb": str(glb_path),
        },
        "quality": {
            "source": "clean_rgb_depth_mesh",
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "mesh_dir": str(output_dir)}


def _try_generate_depth_background_asset(
    *,
    clean_rgb_path: Path,
    output_dir: Path,
    target_frame_id: int,
    depth_maps_dir: str | Path | None,
    camera_trajectory_path: str | Path | None,
    grid_stride: int,
) -> dict[str, Any] | None:
    if not depth_maps_dir or not camera_trajectory_path:
        return None
    depth_path = _resolve_depth_for_frame(depth_maps_dir, target_frame_id)
    camera = _camera_for_frame(camera_trajectory_path, target_frame_id)
    if depth_path is None or camera is None:
        return None
    try:
        result = generate_depth_background_mesh_assets(
            clean_rgb_path=clean_rgb_path,
            depth_path=depth_path,
            output_dir=output_dir / "depth_mesh",
            camera=camera,
            target_frame_id=target_frame_id,
            grid_stride=max(2, int(grid_stride)),
        )
    except Exception:
        return None
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assets = manifest.get("assets", {})
    quality = manifest.get("quality", {})
    quality = dict(quality)
    quality["depth_background_source"] = "wildgs_depth_map_aligned_to_clean_rgb"
    quality["depth_background_manifest"] = result["manifest_path"]
    return {
        "assets": {
            "clean_depth": assets.get("clean_depth"),
            "depth_background_glb": assets.get("depth_background_glb"),
        },
        "quality": quality,
    }


def _try_generate_depth_background_asset_from_estimator(
    *,
    clean_rgb_path: Path,
    output_dir: Path,
    target_frame_id: int,
    camera_trajectory_path: str | Path | None,
    depth_maps_dir: str | Path | None,
    calibration_mask: np.ndarray | None,
    grid_stride: int,
    clean_depth_estimator: Callable[[Path], Any] | None,
) -> dict[str, Any] | None:
    if clean_depth_estimator is None or not camera_trajectory_path:
        return None
    camera = _camera_for_frame(camera_trajectory_path, target_frame_id)
    if camera is None:
        return None
    try:
        estimate = clean_depth_estimator(clean_rgb_path)
        depth_path, source, extra_quality = _normalize_depth_estimate_result(estimate)
        if depth_path is None:
            return None
        depth_path, calibration_quality = _calibrate_depth_to_metric_reference(
            depth_path=depth_path,
            output_dir=output_dir / "depth_mesh",
            target_frame_id=target_frame_id,
            depth_maps_dir=depth_maps_dir,
            calibration_mask=calibration_mask,
        )
        result = generate_depth_background_mesh_assets(
            clean_rgb_path=clean_rgb_path,
            depth_path=depth_path,
            output_dir=output_dir / "depth_mesh",
            camera=camera,
            target_frame_id=target_frame_id,
            grid_stride=max(2, int(grid_stride)),
        )
    except Exception:
        return None
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assets = manifest.get("assets", {})
    quality = dict(manifest.get("quality", {}))
    quality["depth_background_source"] = source
    quality["depth_background_manifest"] = result["manifest_path"]
    quality.update(calibration_quality)
    quality.update(extra_quality)
    return {
        "assets": {
            "clean_depth": assets.get("clean_depth"),
            "depth_background_glb": assets.get("depth_background_glb"),
        },
        "quality": quality,
    }


def _normalize_depth_estimate_result(estimate: Any) -> tuple[Path | None, str, dict[str, Any]]:
    source = "external_clean_rgb_depth_estimator"
    extra_quality: dict[str, Any] = {}
    if estimate is None:
        return None, source, extra_quality
    if isinstance(estimate, (str, Path)):
        return Path(estimate), source, extra_quality
    if isinstance(estimate, dict):
        raw_path = estimate.get("depth_path") or estimate.get("path") or estimate.get("clean_depth")
        if not raw_path:
            return None, source, extra_quality
        source = str(estimate.get("source") or source)
        quality = estimate.get("quality")
        if isinstance(quality, dict):
            extra_quality.update(quality)
        return Path(raw_path), source, extra_quality
    return None, source, extra_quality


def _calibrate_depth_to_metric_reference(
    *,
    depth_path: Path,
    output_dir: Path,
    target_frame_id: int,
    depth_maps_dir: str | Path | None,
    calibration_mask: np.ndarray | None,
) -> tuple[Path, dict[str, Any]]:
    if not depth_maps_dir:
        return depth_path, {"depth_calibration_source": "none"}
    reference_path = _resolve_depth_for_frame(depth_maps_dir, target_frame_id)
    if reference_path is None:
        return depth_path, {"depth_calibration_source": "none"}
    try:
        source = np.load(str(depth_path)).astype(np.float64)
        reference = np.load(str(reference_path)).astype(np.float64)
    except Exception:
        return depth_path, {"depth_calibration_source": "none"}
    if source.ndim == 3:
        source = source[0]
    if reference.ndim == 3:
        reference = reference[0]
    if source.ndim != 2 or reference.ndim != 2:
        return depth_path, {"depth_calibration_source": "none"}
    if reference.shape != source.shape:
        reference = cv2.resize(reference.astype(np.float32), (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = np.isfinite(source) & np.isfinite(reference) & (source > 1e-6) & (reference > 1e-6)
    if calibration_mask is not None and calibration_mask.shape == source.shape:
        mask &= calibration_mask
    min_samples = min(128, max(16, int(source.size // 64)))
    if int(np.count_nonzero(mask)) < min_samples:
        return depth_path, {"depth_calibration_source": "none_insufficient_overlap"}
    x = source[mask].reshape(-1)
    y = reference[mask].reshape(-1)
    if len(x) > 50000:
        idx = np.linspace(0, len(x) - 1, 50000, dtype=int)
        x = x[idx]
        y = y[idx]
    keep = np.ones_like(x, dtype=bool)
    scale = 1.0
    bias = 0.0
    for _ in range(3):
        if int(np.count_nonzero(keep)) < min_samples:
            break
        A = np.stack([x[keep], np.ones(int(np.count_nonzero(keep)))], axis=1)
        scale, bias = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        residual = (x * scale + bias) - y
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        keep = np.abs(residual - med) <= max(0.25, mad * 3.0)
    if not math.isfinite(float(scale)) or not math.isfinite(float(bias)) or abs(float(scale)) < 1e-6:
        return depth_path, {"depth_calibration_source": "none_invalid_fit"}
    calibrated = source * float(scale) + float(bias)
    positive = reference[mask]
    ref_min = max(0.01, float(np.percentile(positive, 0.5)) * 0.5)
    ref_max = float(np.percentile(positive, 99.5)) * 1.8
    calibrated = np.clip(calibrated, ref_min, ref_max)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_path = output_dir / "clean_target_depth_metric_calibrated.npy"
    np.save(calibrated_path, calibrated.astype(np.float32))
    full_residual = calibrated[mask] - reference[mask]
    return calibrated_path, {
        "depth_calibration_source": "wildgs_metric_depth_affine",
        "depth_calibration_reference": str(reference_path),
        "depth_calibration_scale": float(scale),
        "depth_calibration_bias": float(bias),
        "depth_calibration_sample_count": int(np.count_nonzero(mask)),
        "depth_calibration_median_abs_error": float(np.median(np.abs(full_residual))),
        "depth_calibration_p95_abs_error": float(np.percentile(np.abs(full_residual), 95)),
    }


def _resolve_depth_for_frame(depth_maps_dir: str | Path, target_frame_id: int) -> Path | None:
    root = Path(depth_maps_dir)
    candidates = [
        root / f"{int(target_frame_id):05d}.npy",
        root / f"{max(int(target_frame_id) - 1, 0):05d}.npy",
        root / "depth_maps" / f"{int(target_frame_id):05d}.npy",
        root / "depth_maps" / f"{max(int(target_frame_id) - 1, 0):05d}.npy",
    ]
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    try:
        files = sorted(root.glob("*.npy"), key=lambda p: int(p.stem))
    except OSError:
        files = []
    if not files:
        return None
    target = max(int(target_frame_id) - 1, 0)
    return min(files, key=lambda p: abs(int(p.stem) - target))


def _camera_for_frame(camera_trajectory_path: str | Path, target_frame_id: int) -> dict[str, Any] | None:
    path = Path(camera_trajectory_path)
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(records, list) or not records:
        return None
    target = min(records, key=lambda item: abs(int(item.get("frame_id", 0)) - int(target_frame_id)))
    k = target.get("K") or [[512.0, 0.0, 320.0], [0.0, 512.0, 180.0], [0.0, 0.0, 1.0]]
    return {
        "fx": float(k[0][0]),
        "fy": float(k[1][1]),
        "cx": float(k[0][2]),
        "cy": float(k[1][2]),
        "R": target.get("R", np.eye(3).tolist()),
        "t": target.get("t", [0.0, 0.0, 0.0]),
    }


def _build_depth_textured_mesh(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: dict[str, Any],
    grid_stride: int,
    max_depth: float,
):
    import trimesh
    from trimesh.visual import ColorVisuals

    height, width = depth.shape
    stride = max(1, int(grid_stride))
    xs = list(range(0, width, stride))
    ys = list(range(0, height, stride))
    if xs[-1] != width - 1:
        xs.append(width - 1)
    if ys[-1] != height - 1:
        ys.append(height - 1)

    fx = float(camera.get("fx", max(width, height) * 0.8))
    fy = float(camera.get("fy", max(width, height) * 0.8))
    cx = float(camera.get("cx", width * 0.5))
    cy = float(camera.get("cy", height * 0.5))
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)

    vertices: list[list[float]] = []
    colors: list[list[int]] = []
    valid_index: dict[tuple[int, int], int] = {}
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            d = float(depth[y, x])
            if not math.isfinite(d) or d <= 0.01 or d > max_depth:
                continue
            point_cam = np.array([(float(x) - cx) * d / fx, (float(y) - cy) * d / fy, d], dtype=np.float64)
            point_world = rotation @ point_cam + translation
            valid_index[(yi, xi)] = len(vertices)
            vertices.append([float(v) for v in point_world])
            r, g, b = [int(v) for v in rgb[y, x, :3]]
            colors.append([r, g, b, 255])

    faces: list[list[int]] = []
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            keys = [(yi, xi), (yi, xi + 1), (yi + 1, xi), (yi + 1, xi + 1)]
            if any(key not in valid_index for key in keys):
                continue
            v00 = valid_index[(yi, xi)]
            v10 = valid_index[(yi, xi + 1)]
            v01 = valid_index[(yi + 1, xi)]
            v11 = valid_index[(yi + 1, xi + 1)]
            z_values = [float(depth[ys[key[0]], xs[key[1]]]) for key in keys]
            if max(z_values) / max(min(z_values), 1e-6) > 1.8:
                continue
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if not vertices or not faces:
        raise ValueError("Depth background mesh has no valid geometry")
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=np.asarray(colors, dtype=np.uint8))
    return mesh


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_frame(frame_entries: list[dict[str, Any]], frame_id: int) -> dict[str, Any]:
    for entry in frame_entries:
        if int(entry.get("frame_idx", -1)) == int(frame_id):
            return entry
    return min(frame_entries, key=lambda e: abs(int(e.get("frame_idx", 0)) - int(frame_id)))


def _rank_frames(frame_entries: list[dict[str, Any]], frame_id: int) -> list[dict[str, Any]]:
    return sorted(frame_entries, key=lambda e: abs(int(e.get("frame_idx", 0)) - int(frame_id)))


def _decode_image_b64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image_b64")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_instance_mask(inst: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    raw = inst.get("mask_rle") or inst.get("mask")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    size = tuple(int(v) for v in raw.get("size", shape))
    counts = raw.get("counts")
    if raw.get("encoding") == "zlib_packbits" and isinstance(counts, str):
        packed = zlib.decompress(base64.b64decode(counts))
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        mask = bits[: size[0] * size[1]].reshape(size).astype(bool)
        return _resize_mask(mask, shape)
    try:
        from pycocotools import mask as mask_utils

        rle = {"size": list(size), "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
        mask = mask_utils.decode(rle).astype(bool)
        return _resize_mask(mask, shape)
    except Exception:
        return None


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _bbox_mask(bbox: Any, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if not bbox or len(bbox) != 4:
        return mask
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    mask[y1:y2, x1:x2] = True
    return mask


def _mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _shadow_mask(bbox: Any, shape: tuple[int, int], expand: int) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    if not bbox or len(bbox) != 4:
        return out
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    sx1 = max(0, x1 - int(0.10 * bw) - expand)
    sx2 = min(w, x2 + int(0.10 * bw) + expand)
    sy1 = max(0, y2 - int(0.18 * bh))
    sy2 = min(h, y2 + expand + int(0.15 * bh))
    out[sy1:sy2, sx1:sx2] = True
    return out


def _weighted_average_rgb(rgbs: list[np.ndarray], weights: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(rgbs[0], dtype=np.float32)
    total = np.zeros(rgbs[0].shape[:2], dtype=np.float32)
    for rgb, weight in zip(rgbs, weights):
        acc += rgb * weight[..., None]
        total += weight
    out = fallback.astype(np.float32)
    valid = total > 1e-6
    out[valid] = acc[valid] / total[valid, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _robust_median_rgb(rgbs: list[np.ndarray], weights: list[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    stack = np.stack(rgbs, axis=0)
    wstack = np.stack(weights, axis=0)
    valid = wstack > 1e-6
    masked = np.where(valid[..., None], stack, np.nan)
    median = np.nanmedian(masked, axis=0)
    out = fallback.astype(np.float32)
    ok = np.isfinite(median).all(axis=2)
    out[ok] = median[ok]
    return np.clip(out, 0, 255).astype(np.uint8)


def _confidence_from_weights(weights: list[np.ndarray]) -> np.ndarray:
    if not weights:
        return np.zeros((1, 1), dtype=np.float32)
    total = np.zeros_like(weights[0], dtype=np.float32)
    count = np.zeros_like(weights[0], dtype=np.float32)
    for weight in weights:
        total += weight
        count += (weight > 1e-6).astype(np.float32)
    return np.clip((total / max(float(len(weights)), 1.0)) * np.clip(count / 3.0, 0.0, 1.0), 0.0, 1.0)


def _fill_low_candidate_dynamic_regions(
    clean_rgb: np.ndarray,
    target_rgb: np.ndarray,
    dynamic_mask: np.ndarray,
    source_count: np.ndarray,
) -> np.ndarray:
    fill_mask = dynamic_mask & (source_count == 0)
    if not fill_mask.any():
        return clean_rgb
    num, labels = cv2.connectedComponents(fill_mask.astype(np.uint8), connectivity=8)
    out = clean_rgb.copy()
    reliable = (~dynamic_mask) & (source_count >= 3)
    for label in range(1, num):
        region = labels == label
        area = int(np.count_nonzero(region))
        if area <= 0 or area > 4000:
            continue
        ys, xs = np.where(region)
        x1, x2 = max(0, int(xs.min()) - 12), min(out.shape[1], int(xs.max()) + 13)
        y1, y2 = max(0, int(ys.min()) - 12), min(out.shape[0], int(ys.max()) + 13)
        ring = reliable[y1:y2, x1:x2]
        if int(np.count_nonzero(ring)) < 8:
            continue
        region_center_y = float((ys.min() + ys.max()) * 0.5)
        use_row_fill = region_center_y < out.shape[0] * 0.55 or area < 1400
        if use_row_fill:
            _fill_region_from_horizontal_neighbors(out, region, reliable)
            _feather_region_edges(out, region)
            continue
        local_mask = region[y1:y2, x1:x2].astype(np.uint8) * 255
        local_rgb = out[y1:y2, x1:x2].copy()
        try:
            local_bgr = cv2.cvtColor(local_rgb, cv2.COLOR_RGB2BGR)
            repaired = cv2.inpaint(local_bgr, local_mask, 3.0, cv2.INPAINT_TELEA)
            repaired_rgb = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
            out[y1:y2, x1:x2][region[y1:y2, x1:x2]] = repaired_rgb[region[y1:y2, x1:x2]]
        except Exception:
            colors = out[y1:y2, x1:x2][ring]
            color = np.median(colors.astype(np.float32), axis=0)
            out[region] = np.clip(color, 0, 255).astype(np.uint8)
    return out


def _feather_region_edges(out: np.ndarray, region: np.ndarray) -> None:
    if not region.any():
        return
    boundary = region & (cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3) <= 2.5)
    if not boundary.any():
        return
    try:
        blurred = np.asarray(Image.fromarray(np.ascontiguousarray(out)).filter(ImageFilter.GaussianBlur(radius=1.0)))
    except Exception:
        return
    out[boundary] = np.clip(out[boundary].astype(np.float32) * 0.55 + blurred[boundary].astype(np.float32) * 0.45, 0, 255).astype(np.uint8)


def _fill_region_from_horizontal_neighbors(out: np.ndarray, region: np.ndarray, reliable: np.ndarray) -> None:
    height, width = region.shape
    ys, xs = np.where(region)
    if len(xs) == 0:
        return
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    pad = 24
    fallback_mask = reliable[max(0, y_min - pad) : min(height, y_max + pad + 1), max(0, x_min - pad) : min(width, x_max + pad + 1)]
    fallback_rgb = out[max(0, y_min - pad) : min(height, y_max + pad + 1), max(0, x_min - pad) : min(width, x_max + pad + 1)]
    if fallback_mask.any():
        fallback_color = np.median(fallback_rgb[fallback_mask].astype(np.float32), axis=0)
    else:
        fallback_color = np.median(out[reliable].astype(np.float32), axis=0) if reliable.any() else np.array([96.0, 96.0, 96.0])

    for y in range(y_min, y_max + 1):
        row = region[y]
        if not row.any():
            continue
        row_x = np.where(row)[0]
        left_x, left = _sample_side_color(out, reliable, y, int(row_x.min()), -1)
        right_x, right = _sample_side_color(out, reliable, y, int(row_x.max()), 1)

        if left is not None and right is not None and right_x is not None and left_x is not None and right_x > left_x:
            alpha = ((row_x.astype(np.float32) - float(left_x)) / float(right_x - left_x))[:, None]
            colors = left[None, :] * (1.0 - alpha) + right[None, :] * alpha
        elif left is not None:
            colors = np.repeat(left[None, :], len(row_x), axis=0)
        elif right is not None:
            colors = np.repeat(right[None, :], len(row_x), axis=0)
        else:
            colors = np.repeat(fallback_color[None, :], len(row_x), axis=0)
        out[y, row_x] = np.clip(colors, 0, 255).astype(np.uint8)


def _sample_side_color(
    out: np.ndarray,
    reliable: np.ndarray,
    y: int,
    edge_x: int,
    direction: int,
) -> tuple[int | None, np.ndarray | None]:
    height, width = reliable.shape
    step = 1 if direction > 0 else -1
    start = edge_x + step
    if start < 0 or start >= width:
        return None, None
    max_dist = 44
    sample_span = 14
    for dist in range(1, max_dist + 1):
        x = edge_x + step * dist
        if x < 0 or x >= width:
            break
        if not reliable[y, x]:
            continue
        if direction > 0:
            x1, x2 = x, min(width, x + sample_span)
        else:
            x1, x2 = max(0, x - sample_span + 1), x + 1
        y1, y2 = max(0, y - 2), min(height, y + 3)
        mask = reliable[y1:y2, x1:x2]
        if int(np.count_nonzero(mask)) < 3:
            continue
        colors = out[y1:y2, x1:x2][mask].astype(np.float32)
        return x, _robust_local_color(colors)
    return None, None


def _robust_local_color(colors: np.ndarray) -> np.ndarray:
    if len(colors) == 0:
        return np.array([96.0, 96.0, 96.0], dtype=np.float32)
    luma = colors @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    med = float(np.median(luma))
    mad = float(np.median(np.abs(luma - med)))
    keep = np.abs(luma - med) <= max(10.0, mad * 2.5)
    if int(np.count_nonzero(keep)) >= 3:
        colors = colors[keep]
    return np.median(colors, axis=0).astype(np.float32)


def _load_road_plane(path: str | Path | None, frame_id: int) -> dict[str, Any] | None:
    if not path or not Path(path).exists():
        return None
    data = _load_json(path)
    plane = select_road_plane_for_frame(data, int(frame_id), policy="global_for_fixed_camera")
    if not plane:
        return None
    return plane


def _load_object_index_masks(path: str | Path | None, shape: tuple[int, int]) -> dict[int, np.ndarray]:
    if not path or not Path(path).exists():
        return {}
    h, w = shape
    masks: dict[int, np.ndarray] = {}
    try:
        objects = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(objects, list):
        return {}
    for obj in objects:
        label = str(obj.get("label") or obj.get("class_name") or "").lower()
        if any(token in label for token in STATIC_GUARD_LABELS):
            continue
        if not any(token in label for token in DYNAMIC_LABELS):
            continue
        for rec in obj.get("frames", []) or []:
            bbox = rec.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if max(x1, y1, x2, y2) <= 0.0:
                continue
            area = max(1.0, (x2 - x1) * (y2 - y1))
            pad = float(np.clip(math.sqrt(area) * 0.35, 4.0, 24.0))
            frame_id = int(rec.get("frame_idx") or rec.get("frame_id") or -1)
            if frame_id < 0:
                continue
            masks.setdefault(frame_id, np.zeros((h, w), dtype=bool))
            expanded = [x1 - pad, y1 - pad, x2 + pad, y2 + pad * 1.6]
            masks[frame_id] |= _bbox_mask(expanded, (h, w))
    return masks


def _write_textured_grid_obj(
    path: Path,
    output_dir: Path,
    texture_path: Path,
    width: int,
    height: int,
    *,
    mask: np.ndarray,
    grid_stride: int,
    layer: str,
    road_plane: dict[str, Any] | None,
) -> None:
    stride = max(2, int(grid_stride))
    xs = list(range(0, width, stride))
    ys = list(range(0, height, stride))
    if xs[-1] != width - 1:
        xs.append(width - 1)
    if ys[-1] != height - 1:
        ys.append(height - 1)
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    index: dict[tuple[int, int], int] = {}
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            wx = (x / max(width - 1, 1) - 0.5) * 12.0
            wz = (y / max(height - 1, 1) - 0.5) * -8.0
            if layer == "road":
                wy = _road_y(wx, wz, road_plane)
            else:
                wy = 1.5 + (0.5 - y / max(height - 1, 1)) * 3.0
                wz -= 2.0
            index[(yi, xi)] = len(vertices) + 1
            vertices.append((wx, wy, wz))
            uvs.append((x / max(width - 1, 1), 1.0 - y / max(height - 1, 1)))
    faces: list[tuple[int, int, int]] = []
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            cx = min(width - 1, int((xs[xi] + xs[xi + 1]) * 0.5))
            cy = min(height - 1, int((ys[yi] + ys[yi + 1]) * 0.5))
            if not mask[cy, cx]:
                continue
            v00 = index[(yi, xi)]
            v10 = index[(yi, xi + 1)]
            v01 = index[(yi + 1, xi)]
            v11 = index[(yi + 1, xi + 1)]
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    if not faces:
        faces = [(1, 2, min(3, len(vertices)))]
    _write_obj_with_mtl(path, output_dir, texture_path, vertices, uvs, faces)


def _write_far_mesh(path: Path, output_dir: Path, texture_path: Path, width: int, height: int) -> None:
    vertices = [(-8.0, 4.0, -16.0), (8.0, 4.0, -16.0), (8.0, -1.0, -16.0), (-8.0, -1.0, -16.0)]
    uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    faces = [(1, 2, 3), (1, 3, 4)]
    _write_obj_with_mtl(path, output_dir, texture_path, vertices, uvs, faces)


def _road_y(x: float, z: float, road_plane: dict[str, Any] | None) -> float:
    if not road_plane:
        return 0.0
    n = np.asarray(road_plane.get("normal_world", [0.0, 1.0, 0.0]), dtype=np.float64)
    d = float(road_plane.get("offset", 0.0))
    if abs(float(n[1])) < 1e-6:
        return 0.0
    return float(-(n[0] * x + n[2] * z + d) / n[1])


def _write_obj_with_mtl(
    obj_path: Path,
    output_dir: Path,
    texture_path: Path,
    vertices: list[tuple[float, float, float]],
    uvs: list[tuple[float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    mtl_path = obj_path.with_suffix(".mtl")
    tex_rel = Path(texture_path).resolve().relative_to(output_dir.resolve()).as_posix()
    mtl_path.write_text(f"newmtl background\nKd 1 1 1\nmap_Kd {tex_rel}\n", encoding="utf-8")
    lines = [f"mtllib {mtl_path.name}", "usemtl background"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces]
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
