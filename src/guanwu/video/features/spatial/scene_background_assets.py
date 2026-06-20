from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import cv2
import fcntl
import numpy as np
from PIL import Image

DYNAMIC_LABELS = ("car", "truck", "bus", "van", "motorcycle", "bicycle", "person")
STATIC_GUARD_LABELS = ("fence", "road", "sidewalk", "rail", "wall", "track", "building")
TASK_BACKGROUND_LABELS = (
    "table",
    "tabletop",
    "wooden board",
    "board",
    "metal table",
    "metal surface",
    "groove",
    "rail",
    "track",
    "base",
    "floor",
    "wall",
)
DEFAULT_OPENAI_IMAGE_EDIT_PROMPT = (
    "Edit the input image minimally. Keep the scene as close as possible to the original frame.\n\n"
    "Remove only the active manipulation objects on the central wooden board: the target wooden block, "
    "loose small blocks, and robot parts that directly occlude the board. Preserve everything else "
    "unless it clearly covers the central board surface.\n\n"
    "Maintain the exact camera perspective, wood board boundaries, metal grooved table, edge context, "
    "lighting, reflections, shadows, stains, scratches, seams, and natural texture. The result should "
    "still feel like the same robotic workbench scene, only with the active objects gently removed.\n\n"
    "Avoid over-cleaning, avoid replacing the environment, avoid making the tabletop pristine, and avoid "
    "changing geometry or style."
)
ROAD_OPENAI_IMAGE_EDIT_PROMPT = (
    "Edit the input image minimally. Keep the scene as close as possible to the original frame.\n\n"
    "Generate a clean static road background.\n\n"
    "Remove all vehicles, road users, distant cars, vehicle shadows, reflections, motion residues, and "
    "ghosting artifacts. Fill only the regions occluded by vehicles or road users.\n\n"
    "Preserve the exact camera perspective, road geometry, lane width, lane markings, curbs, sidewalks, "
    "buildings, lighting, reflections, shadows, asphalt texture, weathering, and scene style.\n\n"
    "Avoid over-cleaning, avoid changing the environment, avoid changing road geometry, avoid moving lane "
    "markings, and avoid inventing new structures."
)
GENERIC_OPENAI_IMAGE_EDIT_PROMPT = (
    "Edit the input image minimally. Keep the scene as close as possible to the original frame.\n\n"
    "Generate a clean static background by removing only temporary foreground objects and occluders. "
    "Preserve the camera perspective, geometry, lighting, shadows, reflections, material texture, scene "
    "boundaries, and natural imperfections.\n\n"
    "Avoid over-cleaning, avoid replacing the environment, and avoid changing geometry or style."
)


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


def build_task_foreground_mask(
    detections: dict[str, Any],
    image_shape: tuple[int, int],
    object_ids: list[str] | tuple[str, ...] | set[str] | None,
    *,
    foreground_expand_px: int = 2,
) -> np.ndarray:
    height, width = image_shape
    target_ids = {str(object_id).strip() for object_id in (object_ids or []) if str(object_id).strip()}
    if not target_ids:
        return build_dynamic_mask(detections, image_shape, foreground_expand_px=foreground_expand_px)
    foreground = np.zeros((height, width), dtype=bool)
    for inst in detections.get("instances", []) or []:
        inst_ids = {
            str(inst.get(key) or "").strip()
            for key in ("object_id", "track_id", "instance_id", "id")
            if str(inst.get(key) or "").strip()
        }
        if target_ids.isdisjoint(inst_ids):
            continue
        mask = _decode_instance_mask(inst, (height, width))
        if mask is None:
            mask = _bbox_mask(inst.get("bbox"), (height, width))
        if not mask.any():
            continue
        if foreground_expand_px > 0:
            mask = _dilate(mask, int(foreground_expand_px))
        foreground |= mask
    return foreground


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


def _scene_detection_labels(frame_entries: list[dict[str, Any]], target_det: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for inst in target_det.get("instances", []) or []:
        label = str(inst.get("concept_label") or inst.get("label") or inst.get("class_name") or "").strip().lower()
        if label:
            labels.append(label)
    for entry in frame_entries[: min(len(frame_entries), 8)]:
        try:
            det = _load_json(entry["detections"])
        except Exception:
            continue
        for inst in det.get("instances", []) or []:
            label = str(inst.get("concept_label") or inst.get("label") or inst.get("class_name") or "").strip().lower()
            if label:
                labels.append(label)
    return labels


def _is_road_scene_from_labels(labels: list[str]) -> bool:
    road_tokens = (
        "road",
        "roadway",
        "asphalt",
        "lane",
        "street",
        "curb",
        "sidewalk",
        "traffic",
        "car",
        "vehicle",
        "truck",
        "bus",
        "van",
        "motorcycle",
        "bicycle",
        "pedestrian",
    )
    return any(any(token in label for token in road_tokens) for label in labels)


def _select_clean_scene_prompt_profile(
    requested_profile: Any,
    *,
    mode: str,
    tabletop_task_mode: bool,
    frame_entries: list[dict[str, Any]],
    target_det: dict[str, Any],
) -> tuple[str, str]:
    profile = str(requested_profile or "auto").strip().lower()
    aliases = {
        "table": "tabletop",
        "tabletop_task": "tabletop",
        "task": "tabletop",
        "manipulation": "tabletop",
        "robot_task": "tabletop",
        "road_task": "road",
        "vehicle": "road",
        "traffic": "road",
        "scene": "generic",
        "clean_scene": "generic",
    }
    profile = aliases.get(profile, profile)
    if profile in {"tabletop", "road", "generic"}:
        return profile, "config"
    if tabletop_task_mode:
        return "tabletop", "background_mode"
    if mode in {"road", "road_task", "vehicle", "traffic"}:
        return "road", "background_mode"
    if _is_road_scene_from_labels(_scene_detection_labels(frame_entries, target_det)):
        return "road", "detections"
    return "generic", "default"


def _prompt_for_clean_scene_profile(profile: str) -> str:
    if profile == "road":
        return ROAD_OPENAI_IMAGE_EDIT_PROMPT
    if profile == "generic":
        return GENERIC_OPENAI_IMAGE_EDIT_PROMPT
    return DEFAULT_OPENAI_IMAGE_EDIT_PROMPT


def generate_target_frame_background_assets(
    *,
    summary_path: str | Path,
    output_dir: str | Path,
    target_frame_id: int = 1,
    object_index_path: str | Path | None = None,
    depth_maps_dir: str | Path | None = None,
    camera_trajectory_path: str | Path | None = None,
    clean_depth_estimator: Callable[[Path], Any] | None = None,
    grid_stride: int = 4,
    top_k: int = 5,
    background_mode: str = "auto",
    task_foreground_object_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    background_cleaner: str = "openai_image_edit",
    background_cleaner_config: dict[str, Any] | None = None,
    background_cleaner_reference_frame_id: int | None = None,
    background_image_cleaner: Callable[..., dict[str, Any] | None] | None = None,
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
    mode = str(background_mode or "auto").strip().lower()
    task_ids = [str(object_id).strip() for object_id in (task_foreground_object_ids or []) if str(object_id).strip()]
    tabletop_task_mode = mode in {"tabletop_task", "task", "manipulation", "robot_task"}
    cleaner_name = str(background_cleaner or "openai_image_edit").strip().lower()
    cleaner_config = dict(background_cleaner_config or {})
    requested_clean_profile = (
        cleaner_config.get("scene_prompt_profile")
        or cleaner_config.get("prompt_profile")
        or cleaner_config.get("background_prompt_profile")
    )
    selected_clean_profile, _selected_clean_profile_source = _select_clean_scene_prompt_profile(
        requested_clean_profile,
        mode=mode,
        tabletop_task_mode=tabletop_task_mode,
        frame_entries=frame_entries,
        target_det=target_det,
    )
    openai_cleaner_aliases = {"openai_image_edit", "gpt_image_edit", "gpt-image-edit"}
    if cleaner_name not in openai_cleaner_aliases:
        raise ValueError(
            f"Unsupported background_cleaner={background_cleaner!r}. "
            "Legacy temporal/donor background generation has been removed; use openai_image_edit."
        )
    openai_image_cleaner_requested = True
    clean_scene_background_mode = True
    object_index_masks = {} if task_ids else _load_object_index_masks(object_index_path, (height, width))
    target_mask = (
        build_task_foreground_mask(target_det, (height, width), task_ids)
        if tabletop_task_mode
        else build_dynamic_mask(target_det, (height, width))
    )
    if not tabletop_task_mode:
        target_mask |= object_index_masks.get(int(target_frame_id), np.zeros((height, width), dtype=bool))
    source_count = np.zeros((height, width), dtype=np.uint16)
    openai_image_cleaner_enabled = clean_scene_background_mode and openai_image_cleaner_requested
    clean_rgb = target_rgb.copy()
    confidence = np.zeros((height, width), dtype=np.float32)

    if clean_scene_background_mode:
        clean_rgb_path = output_dir / "clean_target_rgb.png"
        dynamic_mask_path = output_dir / "dynamic_mask_target.png"
        confidence_path = output_dir / "confidence_map.png"
        source_count_path = output_dir / "source_count_map.png"
        tabletop_mesh = mesh_dir / "tabletop_background.obj"
        cleaner_assets: dict[str, str] = {}
        cleaner_quality: dict[str, Any] = {"clean_rgb_source": "openai_image_edit"}
        if openai_image_cleaner_enabled:
            profile_was_requested = any(
                key in cleaner_config for key in ("scene_prompt_profile", "prompt_profile", "background_prompt_profile")
            )
            cleaner_config = _load_openai_image_edit_config(cleaner_config)
            if profile_was_requested or any(
                key in cleaner_config for key in ("scene_prompt_profile", "prompt_profile", "background_prompt_profile")
            ):
                selected_profile, selected_profile_source = _select_clean_scene_prompt_profile(
                    cleaner_config.get("scene_prompt_profile")
                    or cleaner_config.get("prompt_profile")
                    or cleaner_config.get("background_prompt_profile"),
                    mode=mode,
                    tabletop_task_mode=tabletop_task_mode,
                    frame_entries=frame_entries,
                    target_det=target_det,
                )
                cleaner_config["scene_prompt_profile"] = selected_profile
                cleaner_config["scene_prompt_profile_source"] = selected_profile_source
                cleaner_config["prompt"] = _prompt_for_clean_scene_profile(selected_profile)
            reference_frame_id = int(background_cleaner_reference_frame_id or target_frame_id)
            reference_entry = _select_frame(frame_entries, reference_frame_id)
            reference_det = _load_json(reference_entry["detections"])
            reference_rgb = _decode_image_b64(reference_det["image_b64"])
            if reference_rgb.shape[:2] != (height, width):
                reference_rgb = np.asarray(Image.fromarray(reference_rgb).resize((width, height), Image.BILINEAR))
            cleaner_mask_mode = str(cleaner_config.get("mask_mode") or cleaner_config.get("edit_mask_mode") or "target").strip().lower()
            mask_frame_mode = str(cleaner_config.get("mask_frame_mode") or "reference_frame").strip().lower()
            mask_frame_entries = frame_entries if mask_frame_mode in {"all", "all_frames", "union"} else [reference_entry]
            prompt_only_mode = cleaner_mask_mode in {"none", "no_mask", "prompt", "prompt_only", "image_prompt", "reference_only"}
            full_image_output = _as_bool(
                cleaner_config.get("use_full_image_output"),
                default=prompt_only_mode or cleaner_mask_mode in {"full", "full_image", "whole_image", "all"},
            )
            if prompt_only_mode:
                cleaner_mask = np.zeros((height, width), dtype=bool)
            elif cleaner_mask_mode in {"full", "full_image", "whole_image", "all"}:
                cleaner_mask = np.ones((height, width), dtype=bool)
            elif cleaner_mask_mode in {"foreground", "foreground_objects", "all_objects", "detected_objects"}:
                cleaner_mask = _build_foreground_instance_mask_for_frames(
                    frame_entries=mask_frame_entries,
                    image_shape=(height, width),
                    expand_px=int(cleaner_config.get("mask_expand_px", 8) or 8),
                )
                if not cleaner_mask.any():
                    cleaner_mask = target_mask
            else:
                cleaner_mask = _build_task_foreground_mask_for_frames(
                    frame_entries=mask_frame_entries,
                    image_shape=(height, width),
                    object_ids=task_ids,
                    expand_px=int(cleaner_config.get("mask_expand_px", 8) or 8),
                )
                if not cleaner_mask.any():
                    cleaner_mask = target_mask
            reference_path = output_dir / "openai_image_edit_reference.png"
            cleaner_mask_path = output_dir / "openai_image_edit_mask.png"
            raw_clean_path = output_dir / "openai_image_edit_raw.png"
            Image.fromarray(reference_rgb).save(reference_path)
            if prompt_only_mode:
                Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), mode="RGBA").save(cleaner_mask_path)
            else:
                _write_openai_edit_mask_png(cleaner_mask_path, cleaner_mask)
            cleaner = background_image_cleaner or run_openai_image_edit_background_cleaner
            cleaner_result = cleaner(
                image_path=reference_path,
                mask_path=cleaner_mask_path,
                output_path=raw_clean_path,
                config=cleaner_config,
                reference_frame_id=reference_frame_id,
            )
            raw_path = Path((cleaner_result or {}).get("raw_output_path") or (cleaner_result or {}).get("clean_rgb_path") or raw_clean_path)
            edited_rgb = np.asarray(Image.open(raw_path).convert("RGB"))
            if edited_rgb.shape[:2] != (height, width):
                edited_rgb = np.asarray(Image.fromarray(edited_rgb).resize((width, height), Image.BILINEAR))
            clean_rgb = edited_rgb if full_image_output else _blend_cleaner_output(reference_rgb, edited_rgb, cleaner_mask)
            cleaner_assets = {
                "openai_image_edit_reference": str(reference_path),
                "openai_image_edit_mask": str(cleaner_mask_path),
                "openai_image_edit_raw": str(raw_path),
            }
            cleaner_quality = {
                "clean_rgb_source": "openai_image_edit",
                "clean_rgb_reference_frame_id": int(reference_frame_id),
                "clean_rgb_model": str((cleaner_result or {}).get("model") or cleaner_config.get("model") or "gpt-image-2"),
                "clean_rgb_prompt": str((cleaner_result or {}).get("prompt") or cleaner_config.get("prompt") or DEFAULT_OPENAI_IMAGE_EDIT_PROMPT),
                "scene_prompt_profile": str(cleaner_config.get("scene_prompt_profile") or "custom"),
                "scene_prompt_profile_source": str(cleaner_config.get("scene_prompt_profile_source") or "custom_prompt"),
                "clean_rgb_mask_fraction": float(np.mean(cleaner_mask)),
                "clean_rgb_mask_mode": cleaner_mask_mode,
                "clean_rgb_mask_frame_mode": mask_frame_mode,
                "clean_rgb_full_image_output": bool(full_image_output),
            }
        Image.fromarray(clean_rgb).save(clean_rgb_path)
        Image.fromarray((target_mask.astype(np.uint8) * 255)).save(dynamic_mask_path)
        Image.fromarray(np.clip(confidence * 255.0, 0, 255).astype(np.uint8)).save(confidence_path)
        Image.fromarray(np.clip(source_count, 0, 255).astype(np.uint8)).save(source_count_path)
        _write_tabletop_background_obj(tabletop_mesh, output_dir, clean_rgb_path, width, height)
        depth_asset = _try_generate_depth_background_asset_from_estimator(
            clean_rgb_path=clean_rgb_path,
            output_dir=output_dir,
            target_frame_id=target_frame_id,
            camera_trajectory_path=camera_trajectory_path,
            depth_maps_dir=depth_maps_dir,
            calibration_mask=(~target_mask),
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
        tabletop_reference_asset = None
        if depth_asset and str(depth_asset.get("quality", {}).get("depth_background_source", "")).strip() != "wildgs_depth_map_aligned_to_clean_rgb":
            support_mask = _dilate(target_mask, 8) if target_mask.any() else target_mask
            tabletop_reference_asset = _try_write_tabletop_reference_asset(
                clean_depth_path=depth_asset.get("assets", {}).get("clean_depth"),
                output_dir=output_dir,
                target_frame_id=target_frame_id,
                camera_trajectory_path=camera_trajectory_path,
                support_mask=support_mask,
                foreground_mask_path=dynamic_mask_path,
            )
        manifest_path = output_dir / "background_manifest.json"
        manifest = {
            "schema": (
                "guanwu.target_frame_background_assets.tabletop.v1"
                if tabletop_task_mode
                else "guanwu.target_frame_background_assets.clean_scene.v1"
            ),
            "target_frame_id": int(target_frame_id),
            "image_size": [int(width), int(height)],
            "assets": {
                "clean_rgb": str(clean_rgb_path),
                "dynamic_mask": str(dynamic_mask_path),
                "confidence_map": str(confidence_path),
                "source_count_map": str(source_count_path),
                "tabletop_mesh": str(tabletop_mesh),
                "task_background_mesh": str(tabletop_mesh),
                **cleaner_assets,
            },
            "quality": {
                "background_mode": "tabletop_task" if tabletop_task_mode else "clean_scene_background",
                "requested_background_mode": mode,
                "source_frame_count": 0,
                "target_dynamic_fraction": float(np.mean(target_mask)),
                "target_foreground_object_ids": task_ids,
                "mean_confidence": float(np.mean(confidence)),
                **cleaner_quality,
            },
            "road_plane": None,
        }
        if depth_asset:
            manifest["schema"] = (
                "guanwu.target_frame_background_assets.tabletop_depth.v2"
                if tabletop_task_mode
                else "guanwu.target_frame_background_assets.clean_scene_depth.v2"
            )
            manifest["assets"].update(
                {key: value for key, value in depth_asset.get("assets", {}).items() if value}
            )
            manifest["quality"].update(depth_asset.get("quality", {}))
        if tabletop_reference_asset:
            manifest["assets"]["tabletop_reference"] = tabletop_reference_asset["path"]
            if tabletop_reference_asset.get("background_geometry_reference_path"):
                manifest["assets"]["background_geometry_reference"] = tabletop_reference_asset["background_geometry_reference_path"]
            manifest["quality"].update(tabletop_reference_asset.get("quality", {}))
            manifest["tabletop_reference"] = {
                "path": tabletop_reference_asset["path"],
                "source": tabletop_reference_asset["source"],
                "target_frame_id": int(target_frame_id),
            }
            if tabletop_reference_asset.get("background_geometry_reference_path"):
                manifest["background_geometry_reference"] = {
                    "path": tabletop_reference_asset["background_geometry_reference_path"],
                    "source": tabletop_reference_asset.get("background_geometry_reference_source", "clean_background_depth"),
                    "target_frame_id": int(target_frame_id),
                    "reference_type": "support_surface",
                }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"manifest_path": str(manifest_path), "mesh_dir": str(mesh_dir)}


def load_background_asset_meshes(
    background_assets_manifest: str | Path | None,
) -> list[tuple[str, Path]]:
    if not background_assets_manifest:
        return []
    manifest_path = Path(background_assets_manifest)
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = data.get("assets", {})
    background_mesh = assets.get("background_mesh")
    if background_mesh:
        path = Path(background_mesh)
        if path.exists():
            return [("background", path)]
    depth_bg = assets.get("depth_background_glb") or assets.get("depth_background_mesh")
    if depth_bg:
        path = Path(depth_bg)
        if path.exists():
            return [("depth_background", path)]
    tabletop_mesh = assets.get("tabletop_mesh") or assets.get("task_background_mesh")
    if tabletop_mesh:
        path = Path(tabletop_mesh)
        if path.exists():
            return [("tabletop", path)]
    return []


def ensure_road_plane_fused_background_asset(
    background_assets_manifest: str | Path | None,
    road_geometry_path: str | Path | None,
) -> dict[str, Any] | None:
    if not background_assets_manifest or not road_geometry_path:
        return None
    manifest_path = Path(background_assets_manifest)
    road_path = Path(road_geometry_path)
    if not manifest_path.exists() or not road_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        road_geometry = json.loads(road_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    quality = manifest.get("quality") if isinstance(manifest.get("quality"), dict) else {}
    requested_mode = str(quality.get("requested_background_mode") or "").strip().lower()
    prompt_profile = str(quality.get("scene_prompt_profile") or "").strip().lower()
    if requested_mode != "road" and prompt_profile != "road":
        return None
    plane = road_geometry.get("global_plane") if isinstance(road_geometry, dict) else None
    if not isinstance(plane, dict):
        return None
    normal = np.asarray(plane.get("normal_world") or [], dtype=np.float64).reshape(-1)
    if normal.shape != (3,):
        return None
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8 or not math.isfinite(norm):
        return None
    normal = normal / norm
    try:
        offset = float(plane.get("offset"))
    except Exception:
        return None

    assets = manifest.get("assets") if isinstance(manifest.get("assets"), dict) else {}
    depth_bg = assets.get("depth_background_glb") or assets.get("depth_background_mesh")
    if not depth_bg:
        return None
    depth_bg_path = Path(depth_bg)
    if not depth_bg_path.exists():
        return None
    fused_path = depth_bg_path.parent / "background_global_fused_v1.glb"
    source_mtime = max(depth_bg_path.stat().st_mtime, road_path.stat().st_mtime)
    try:
        if fused_path.exists() and fused_path.stat().st_mtime >= source_mtime:
            assets_for_update = manifest.setdefault("assets", {})
            assets_for_update["background_mesh"] = str(depth_bg_path)
            assets_for_update["visual_background_mesh"] = str(depth_bg_path)
            assets_for_update["global_fused_background_mesh"] = str(fused_path)
            quality = manifest.setdefault("quality", {})
            quality["background_mode"] = "depth_mesh_with_road_plane_support"
            quality["road_depth_source"] = "global_road_plane_support_only"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"path": str(fused_path), "reused": True}
    except OSError:
        pass

    try:
        import trimesh

        mesh = trimesh.load(str(depth_bg_path), force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return None
        signed = vertices @ normal + offset
        mesh.vertices = vertices - signed[:, None] * normal.reshape(1, 3)
        mesh.export(str(fused_path))
    except Exception:
        return None

    updated_assets = manifest.setdefault("assets", {})
    updated_assets["background_mesh"] = str(depth_bg_path)
    updated_assets["visual_background_mesh"] = str(depth_bg_path)
    updated_assets["global_fused_background_mesh"] = str(fused_path)
    updated_quality = manifest.setdefault("quality", {})
    updated_quality["background_mode"] = "depth_mesh_with_road_plane_support"
    updated_quality["road_support_source"] = "geometric_depth_plane"
    updated_quality["road_depth_source"] = "global_road_plane_support_only"
    updated_quality["static_depth_source"] = str(quality.get("depth_background_source") or "clean_background_depth")
    updated_quality["road_surface_mask_fraction"] = 1.0
    updated_quality["global_fused_background_mesh"] = str(fused_path)
    updated_quality["global_fused_road_plane_source"] = str(plane.get("source") or "global_plane")
    updated_quality["global_fused_road_plane_normal_world"] = [float(v) for v in normal.tolist()]
    updated_quality["global_fused_road_plane_offset"] = float(offset)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(fused_path), "reused": False}


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


def _build_task_foreground_mask_for_frames(
    *,
    frame_entries: list[dict[str, Any]],
    image_shape: tuple[int, int],
    object_ids: list[str] | tuple[str, ...] | set[str],
    expand_px: int,
) -> np.ndarray:
    out = np.zeros(image_shape, dtype=bool)
    for entry in frame_entries:
        try:
            det = _load_json(entry["detections"])
        except Exception:
            continue
        mask = build_task_foreground_mask(det, image_shape, object_ids, foreground_expand_px=0)
        if mask.any():
            out |= mask
    if expand_px > 0 and out.any():
        out = _dilate(out, int(expand_px))
    return out


def _build_foreground_instance_mask_for_frames(
    *,
    frame_entries: list[dict[str, Any]],
    image_shape: tuple[int, int],
    expand_px: int,
) -> np.ndarray:
    out = np.zeros(image_shape, dtype=bool)
    for entry in frame_entries:
        try:
            det = _load_json(entry["detections"])
        except Exception:
            continue
        for inst in det.get("instances", []) or []:
            label = str(inst.get("concept_label") or inst.get("label") or inst.get("class_name") or "").lower()
            if any(token in label for token in TASK_BACKGROUND_LABELS):
                continue
            mask = _decode_instance_mask(inst, image_shape)
            if mask is None:
                mask = _bbox_mask(inst.get("bbox"), image_shape)
            if mask.any():
                out |= mask
    if expand_px > 0 and out.any():
        out = _dilate(out, int(expand_px))
    return out


def _write_openai_edit_mask_png(path: Path, mask: np.ndarray) -> None:
    edit_mask = mask.astype(bool)
    alpha = np.where(edit_mask, 0, 255).astype(np.uint8)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.where(edit_mask[..., None], 255, 0).astype(np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba, mode="RGBA").save(path)


def _blend_cleaner_output(reference_rgb: np.ndarray, edited_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clean = reference_rgb.copy()
    edit_mask = mask.astype(bool)
    if edit_mask.any():
        clean[edit_mask] = edited_rgb[edit_mask]
        try:
            feather = cv2.GaussianBlur(edit_mask.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
            feather = np.clip(feather[..., None], 0.0, 1.0)
            blended = reference_rgb.astype(np.float32) * (1.0 - feather) + clean.astype(np.float32) * feather
            clean = np.clip(blended, 0, 255).astype(np.uint8)
            clean[edit_mask] = edited_rgb[edit_mask]
        except Exception:
            pass
    return clean


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _load_openai_image_edit_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "model": "gpt-image-2",
        "prompt": DEFAULT_OPENAI_IMAGE_EDIT_PROMPT,
        "size": "original",
        "api_size": "1536x1024",
        "mask_mode": "prompt_only",
        "mask_frame_mode": "reference_frame",
        "use_full_image_output": True,
        "quality": "high",
        "response_format": "b64_json",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_sec": 180.0,
    }
    raw = dict(config or {})
    config_path = raw.pop("config_path", None)
    merged.update(raw)
    if config_path:
        file = Path(str(config_path)).expanduser()
        if file.exists():
            try:
                import yaml

                loaded = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            except Exception:
                loaded = {}
            if isinstance(loaded, dict):
                openai_cfg = loaded.get("openai") if isinstance(loaded.get("openai"), dict) else {}
                image_cfg = loaded.get("image_edit") if isinstance(loaded.get("image_edit"), dict) else {}
                merged.update(openai_cfg)
                merged.update(image_cfg)
    return merged


def _resolve_openai_image_edit_size(image_path: str | Path, configured_size: Any) -> str:
    raw = str(configured_size or "original").strip()
    if raw.lower() in {"", "original", "same", "source", "input", "input_image", "reference", "reference_image"}:
        with Image.open(image_path) as image:
            width, height = image.size
        return f"{int(width)}x{int(height)}"
    return raw


def _load_system_vlm_openai_defaults() -> dict[str, Any]:
    try:
        from guanwu.video.core.config import load_settings

        settings, _ = load_settings()
    except Exception:
        return {}
    vlm = getattr(settings, "vlm", None)
    if vlm is None:
        return {}
    defaults: dict[str, Any] = {}
    if getattr(vlm, "api_key", None):
        defaults["api_key"] = getattr(vlm, "api_key")
    if getattr(vlm, "base_url", None):
        defaults["base_url"] = getattr(vlm, "base_url")
    return defaults


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


@contextlib.contextmanager
def _maybe_openai_image_edit_lock():
    if not _env_flag_enabled("GUANWU_OPENAI_IMAGE_EDIT_LOCK"):
        yield
        return
    lock_path = Path(os.environ.get("GUANWU_OPENAI_IMAGE_EDIT_LOCK_PATH") or "/tmp/guanwu_openai_image_edit.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        print(f"[BackgroundCleaner] Waiting for OpenAI image edit lock {lock_path}", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[BackgroundCleaner] Acquired OpenAI image edit lock after {time.time() - start:.2f}s", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print("[BackgroundCleaner] Released OpenAI image edit lock", flush=True)


def run_openai_image_edit_background_cleaner(
    *,
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    reference_frame_id: int | None = None,
) -> dict[str, Any]:
    cfg = _load_openai_image_edit_config(config)
    system_defaults = _load_system_vlm_openai_defaults()
    api_key = cfg.get("api_key") or os.environ.get(str(cfg.get("api_key_env") or "OPENAI_API_KEY"))
    if not api_key:
        api_key = system_defaults.get("api_key")
    if not api_key:
        raise RuntimeError(
            "OpenAI image cleaner requires an API key. Set api_key in the cleaner config "
            "or export the configured api_key_env."
        )
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("OpenAI image cleaner requires the openai Python package.") from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or system_defaults.get("base_url")
    if base_url:
        client_kwargs["base_url"] = str(base_url)
    if cfg.get("timeout_sec"):
        client_kwargs["timeout"] = float(cfg["timeout_sec"])
    client = OpenAI(**client_kwargs)

    output_path = Path(output_path)
    request_size = cfg.get("api_size") or cfg.get("request_size") or cfg.get("size")
    mask_mode = str(cfg.get("mask_mode") or cfg.get("edit_mask_mode") or "").strip().lower()
    prompt_only_mode = mask_mode in {"none", "no_mask", "prompt", "prompt_only", "image_prompt", "reference_only"}
    request_kwargs: dict[str, Any] = {
        "model": str(cfg.get("model") or "gpt-image-2"),
        "prompt": str(cfg.get("prompt") or DEFAULT_OPENAI_IMAGE_EDIT_PROMPT),
        "image": open(image_path, "rb"),
        "size": _resolve_openai_image_edit_size(image_path, request_size),
        "quality": str(cfg.get("quality") or "high"),
        "response_format": "b64_json",
    }
    if not prompt_only_mode:
        request_kwargs["mask"] = open(mask_path, "rb")
    try:
        with _maybe_openai_image_edit_lock():
            response = client.images.edit(**request_kwargs)
    finally:
        try:
            request_kwargs["image"].close()
            mask_file = request_kwargs.get("mask")
            if mask_file is not None:
                mask_file.close()
        except Exception:
            pass

    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError("OpenAI image cleaner returned no image data.")
    first = data[0]
    b64 = getattr(first, "b64_json", None) or (first.get("b64_json") if isinstance(first, dict) else None)
    if not b64:
        url = getattr(first, "url", None) or (first.get("url") if isinstance(first, dict) else None)
        raise RuntimeError(f"OpenAI image cleaner did not return b64_json data; got url={url!r}.")
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    with Image.open(image_path) as reference_image:
        reference_size = reference_image.size
    if image.size != reference_size:
        image = image.resize(reference_size, Image.Resampling.LANCZOS)
    image.save(output_path)
    return {
        "clean_rgb_path": str(output_path),
        "raw_output_path": str(output_path),
        "model": str(cfg.get("model") or "gpt-image-2"),
        "prompt": str(cfg.get("prompt") or DEFAULT_OPENAI_IMAGE_EDIT_PROMPT),
        "reference_frame_id": reference_frame_id,
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

    mesh, mesh_quality = _build_depth_textured_mesh(
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
            **mesh_quality,
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
    quality["depth_background_reference_frame_mapping"] = "pipeline_frame_id_minus_1"
    quality["wildgs_depth_index"] = _wildgs_depth_index_for_pipeline_frame(target_frame_id)
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
        calibration_quality = {"depth_calibration_source": "da3_metric_direct"}
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
        "depth_calibration_reference_frame_mapping": "pipeline_frame_id_minus_1",
        "wildgs_depth_index": _wildgs_depth_index_for_pipeline_frame(target_frame_id),
        "depth_calibration_scale": float(scale),
        "depth_calibration_bias": float(bias),
        "depth_calibration_sample_count": int(np.count_nonzero(mask)),
        "depth_calibration_median_abs_error": float(np.median(np.abs(full_residual))),
        "depth_calibration_p95_abs_error": float(np.percentile(np.abs(full_residual), 95)),
    }


def _try_write_tabletop_reference_asset(
    *,
    clean_depth_path: str | Path | None,
    output_dir: Path,
    target_frame_id: int,
    camera_trajectory_path: str | Path | None,
    support_mask: np.ndarray | None,
    foreground_mask_path: str | Path | None = None,
) -> dict[str, Any] | None:
    if not clean_depth_path or not camera_trajectory_path:
        return None
    camera = _camera_for_frame(camera_trajectory_path, target_frame_id)
    if camera is None:
        return None
    try:
        depth = np.load(str(clean_depth_path)).astype(np.float64)
    except Exception:
        return None
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        return None
    mask = np.isfinite(depth) & (depth > 1e-6)
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != depth.shape:
            support = cv2.resize(support.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        candidate = mask & support
        if int(np.count_nonzero(candidate)) >= min(32, max(8, int(depth.size // 128))):
            mask = candidate
    points = _depth_world_points_from_mask(depth=depth, mask=mask, camera=camera)
    if points is None or len(points) < 8:
        return None
    if len(points) > 30000:
        indices = np.linspace(0, len(points) - 1, 30000, dtype=np.int64)
        points = points[indices]
    plane = _fit_tabletop_plane(points)
    if plane is None:
        return None
    normal, offset, distances = plane
    reference_path = output_dir / "tabletop_reference.json"
    background_geometry_reference_path = output_dir / "background_geometry_reference.json"
    payload = {
        "schema": "guanwu.tabletop_reference.v1",
        "source": "clean_depth_background",
        "target_frame_id": int(target_frame_id),
        "normal_world": [float(v) for v in normal.tolist()],
        "offset": float(offset),
        "quality": {
            "point_count": int(len(points)),
            "rmse_m": float(np.sqrt(np.mean(distances * distances))),
            "median_abs_m": float(np.median(np.abs(distances))),
            "max_abs_m": float(np.max(np.abs(distances))),
        },
    }
    reference_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    support_surface = {
        "id": "support_surface_000001",
        "type": "plane",
        "source": "clean_background_depth",
        "normal_world": [float(v) for v in normal.tolist()],
        "offset": float(offset),
        "confidence": float(
            np.clip(
                1.0 - payload["quality"]["median_abs_m"] / max(1e-6, payload["quality"]["rmse_m"] + 0.05),
                0.0,
                1.0,
            )
        ),
        "quality": dict(payload["quality"]),
    }
    geometry_payload = {
        "schema": "guanwu.background_geometry_reference.v1",
        "reference_type": "support_surface",
        "source": "clean_background_depth",
        "target_frame_id": int(target_frame_id),
        "normal_world": [float(v) for v in normal.tolist()],
        "offset": float(offset),
        "support_surfaces": [support_surface],
        "depth": {
            "clean_depth_path": str(clean_depth_path),
            "coordinate_frame": "world",
        },
        "exclusion": {
            "foreground_mask_path": str(foreground_mask_path) if foreground_mask_path else None,
            "support_mask_source": "foreground_dilated_region" if support_mask is not None else "all_valid_depth",
        },
        "quality": dict(payload["quality"]),
    }
    background_geometry_reference_path.write_text(
        json.dumps(geometry_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "path": str(reference_path),
        "source": "clean_depth_background",
        "background_geometry_reference_path": str(background_geometry_reference_path),
        "background_geometry_reference_source": "clean_background_depth",
        "quality": {
            "tabletop_reference_source": "clean_depth_background",
            "tabletop_reference_point_count": int(len(points)),
            "tabletop_reference_rmse_m": payload["quality"]["rmse_m"],
            "background_geometry_reference_source": "clean_background_depth",
            "background_geometry_reference_type": "support_surface",
        },
    }


def _depth_world_points_from_mask(
    *,
    depth: np.ndarray,
    mask: np.ndarray,
    camera: dict[str, Any],
) -> np.ndarray | None:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None
    d = depth[ys, xs].astype(np.float64)
    fx = float(camera.get("fx", max(depth.shape) * 0.8))
    fy = float(camera.get("fy", max(depth.shape) * 0.8))
    cx = float(camera.get("cx", depth.shape[1] * 0.5))
    cy = float(camera.get("cy", depth.shape[0] * 0.5))
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    points_cam = np.stack(
        [
            (xs.astype(np.float64) - cx) * d / fx,
            (ys.astype(np.float64) - cy) * d / fy,
            d,
        ],
        axis=1,
    )
    points_world = points_cam @ rotation.T + translation.reshape(1, 3)
    valid = np.isfinite(points_world).all(axis=1)
    points_world = points_world[valid]
    return points_world if len(points_world) else None


def _fit_tabletop_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 8:
        return None
    centroid = np.median(pts, axis=0)
    centered = pts - centroid.reshape(1, 3)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    normal = np.asarray(vh[-1], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8 or not math.isfinite(norm):
        return None
    normal = normal / norm
    preferred_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if float(normal @ preferred_up) < 0.0:
        normal = -normal
    offset = -float(normal @ centroid)
    distances = pts @ normal + offset
    finite = np.isfinite(distances)
    if int(np.count_nonzero(finite)) < 8:
        return None
    return normal, offset, distances[finite]


def _wildgs_depth_index_for_pipeline_frame(target_frame_id: int) -> int:
    return max(int(target_frame_id) - 1, 0)


def _resolve_depth_for_frame(depth_maps_dir: str | Path, target_frame_id: int) -> Path | None:
    root = Path(depth_maps_dir)
    wildgs_index = _wildgs_depth_index_for_pipeline_frame(target_frame_id)
    legacy_one_based_index = max(int(target_frame_id), 0)
    roots = [root, root / "depth_maps", root / "depth_maps" / "depth_maps"]
    candidates = [candidate_root / f"{wildgs_index:05d}.npy" for candidate_root in roots]
    if legacy_one_based_index != wildgs_index:
        candidates.extend(candidate_root / f"{legacy_one_based_index:05d}.npy" for candidate_root in roots)
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    files: list[Path] = []
    for candidate_root in roots:
        try:
            files.extend(candidate_root.glob("*.npy"))
        except OSError:
            continue
    files = sorted(set(files), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, str(p)))
    numeric_files = [path for path in files if path.stem.isdigit()]
    if numeric_files:
        return min(numeric_files, key=lambda p: abs(int(p.stem) - wildgs_index))
    try:
        files = sorted(root.glob("*.npy"), key=lambda p: int(p.stem))
    except OSError:
        files = []
    if not files:
        return None
    return min(files, key=lambda p: abs(int(p.stem) - wildgs_index))


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
    discontinuity_faces_removed = 0
    long_edge_faces_removed = 0

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
                discontinuity_faces_removed += 2
                continue
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if not vertices or not faces:
        raise ValueError("Depth background mesh has no valid geometry")
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=np.asarray(colors, dtype=np.uint8))
    return mesh, {
        "discontinuity_faces_removed": int(discontinuity_faces_removed),
        "long_edge_faces_removed": int(long_edge_faces_removed),
    }


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


def _write_tabletop_background_obj(path: Path, output_dir: Path, texture_path: Path, width: int, height: int) -> None:
    aspect = float(width) / max(float(height), 1.0)
    half_w = 4.0 * max(aspect, 1.0)
    half_h = 4.0
    vertices = [
        (-half_w, half_h, -8.0),
        (half_w, half_h, -8.0),
        (half_w, -half_h, -8.0),
        (-half_w, -half_h, -8.0),
    ]
    uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    faces = [(1, 2, 3), (1, 3, 4)]
    _write_obj_with_mtl(path, output_dir, texture_path, vertices, uvs, faces)


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
