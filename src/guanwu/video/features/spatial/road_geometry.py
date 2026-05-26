from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


VEHICLE_LABEL_TOKENS = ("car", "truck", "bus", "van", "vehicle")


def resolve_depth_maps_dir(path: str | Path | None) -> Path | None:
    if not path:
        return None
    root = Path(path)
    try:
        if not root.exists():
            return None
    except OSError:
        return None
    candidates = [
        root,
        root / "depth_maps",
        root / "depth_maps" / "depth_maps",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.exists() and _contains_depth_npy(candidate):
                return candidate
        except OSError:
            continue
    return None


def _contains_depth_npy(path: Path) -> bool:
    try:
        if any(path.glob("*.npy")):
            return True
    except OSError:
        pass
    for index in range(0, 16):
        candidate = path / f"{index:05d}.npy"
        try:
            if candidate.exists():
                return True
        except PermissionError:
            # The depth file may be readable only by the spawned pipeline
            # process. Treat the directory as a candidate and let np.load report
            # a precise read error later if access is still denied.
            return True
        except OSError:
            continue
    return False


def _depth_npy_files(path: Path) -> list[Path]:
    try:
        files = sorted(path.glob("*.npy"), key=lambda p: int(p.stem))
    except OSError:
        files = []
    if files:
        return files
    fallback: list[Path] = []
    for index in range(0, 512):
        candidate = path / f"{index:05d}.npy"
        try:
            if candidate.exists():
                fallback.append(candidate)
        except PermissionError:
            fallback.append(candidate)
        except OSError:
            continue
    return fallback


def estimate_road_geometry(
    *,
    depth_maps_dir: str | Path | None,
    wildgs_poses: list[dict] | None,
    wildgs_K: dict | None,
    detection_frames: list[dict],
    world_up_axis: str = "y",
) -> dict[str, Any]:
    import numpy as np

    depth_dir = resolve_depth_maps_dir(depth_maps_dir)
    if depth_dir is None:
        return {"available": False, "reason": "missing_depth_maps_dir", "keyframe_planes": [], "global_plane": None}
    depth_files = _depth_npy_files(depth_dir)
    if not depth_files:
        return {"available": False, "reason": "no_depth_maps", "keyframe_planes": [], "global_plane": None}

    pose_records = wildgs_poses or []
    if not pose_records and not wildgs_K:
        return {"available": False, "reason": "missing_camera_poses", "keyframe_planes": [], "global_plane": None}

    keyframes: list[dict[str, Any]] = []
    for depth_path in depth_files:
        try:
            depth_frame = int(depth_path.stem)
        except ValueError:
            continue
        pose = _nearest_pose(pose_records, depth_frame)
        intrinsics = _intrinsics_for_pose(pose, wildgs_K)
        t_world_from_cam = pose.get("T_world_from_cam") if pose else None
        if intrinsics is None or t_world_from_cam is None:
            continue
        try:
            depth = np.load(str(depth_path)).astype(np.float64)
        except Exception:
            continue
        if depth.ndim != 2:
            continue
        frame_id = _select_detection_frame_id_for_depth(detection_frames, depth_frame)
        vehicle_boxes = _vehicle_boxes_for_frame(detection_frames, frame_id)
        fit = _fit_road_plane_from_depth(
            depth=depth,
            intrinsics=intrinsics,
            vehicle_boxes=vehicle_boxes,
            t_world_from_cam=np.asarray(t_world_from_cam, dtype=np.float64),
            world_up_axis=world_up_axis,
        )
        if fit is None:
            continue
        fit.update(
            {
                "depth_frame": depth_frame,
                "frame_id": frame_id,
                "frame_id_candidates": _frame_id_candidates_for_depth_frame(depth_frame),
                "depth_path": str(depth_path),
                "vehicle_box_count": len(vehicle_boxes),
            }
        )
        keyframes.append(fit)

    if not keyframes:
        return {"available": False, "reason": "plane_fit_failed", "keyframe_planes": [], "global_plane": None}

    global_plane = _merge_world_planes(keyframes)
    return {
        "available": True,
        "source": "wildgs_depth_maps",
        "depth_maps_dir": str(depth_dir),
        "world_up_axis": world_up_axis,
        "default_plane_policy": "global_for_fixed_camera",
        "keyframe_planes": keyframes,
        "planes": keyframes,
        "global_plane": global_plane,
    }


def select_road_plane_for_frame(
    road_geometry: dict[str, Any] | None,
    frame_id: int,
    *,
    policy: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(road_geometry, dict) or not road_geometry.get("available"):
        return None
    requested_policy = str(policy or road_geometry.get("default_plane_policy") or "nearest_keyframe").strip().lower()
    global_plane = road_geometry.get("global_plane")
    if requested_policy in {"global", "global_for_fixed_camera"} and isinstance(global_plane, dict):
        plane = dict(global_plane)
        plane["selection"] = {
            "mode": "global",
            "policy": requested_policy,
            "target_frame_id": int(frame_id),
        }
        return plane
    keyframes = road_geometry.get("keyframe_planes") or []
    if keyframes:
        target_frame_id = int(frame_id)
        best = min(keyframes, key=lambda item: _road_plane_frame_distance(item, target_frame_id))
        plane = dict(best)
        plane["selection"] = {
            "mode": "nearest_keyframe",
            "target_frame_id": target_frame_id,
            "selected_depth_frame": int(best.get("depth_frame", 0)),
            "selected_frame_id": int(best.get("frame_id", 0)),
            "frame_distance": _road_plane_frame_distance(best, target_frame_id),
        }
        return plane
    if isinstance(global_plane, dict):
        plane = dict(global_plane)
        plane["selection"] = {
            "mode": "global",
            "policy": requested_policy,
            "target_frame_id": int(frame_id),
        }
        return plane
    return None


def _frame_id_candidates_for_depth_frame(depth_frame: int) -> list[int]:
    candidates = {int(depth_frame)}
    if int(depth_frame) <= 0:
        candidates.add(1)
    else:
        candidates.add(int(depth_frame) + 1)
    return sorted(frame_id for frame_id in candidates if frame_id >= 0)


def _select_detection_frame_id_for_depth(detection_frames: list[dict], depth_frame: int) -> int:
    available = {
        int(item.get("frame_idx") or 0)
        for item in detection_frames or []
        if isinstance(item, dict) and int(item.get("frame_idx") or 0) > 0
    }
    candidates = _frame_id_candidates_for_depth_frame(int(depth_frame))
    if int(depth_frame) > 0 and int(depth_frame) in available:
        return int(depth_frame)
    for candidate in candidates:
        if candidate > 0 and candidate in available:
            return int(candidate)
    return int(depth_frame) if int(depth_frame) > 0 else 1


def _road_plane_frame_distance(item: dict[str, Any], target_frame_id: int) -> int:
    candidates: list[int] = []
    for key in ("frame_id", "depth_frame"):
        try:
            candidates.append(int(item.get(key)))
        except Exception:
            pass
    for value in item.get("frame_id_candidates") or []:
        try:
            candidates.append(int(value))
        except Exception:
            pass
    if not candidates:
        return 10**9
    return min(abs(int(value) - int(target_frame_id)) for value in candidates)


def intersect_camera_ray_with_plane(
    *,
    camera: dict[str, Any],
    uv: tuple[float, float],
    plane: dict[str, Any] | None,
) -> dict[str, Any] | None:
    import numpy as np

    if not plane:
        return None
    try:
        normal = np.asarray(plane["normal_world"], dtype=np.float64)
        offset = float(plane["offset"])
        origin = np.asarray(camera["t"], dtype=np.float64)
        rotation = np.asarray(camera["R"], dtype=np.float64)
        fx, fy, cx, cy = [float(camera[key]) for key in ("fx", "fy", "cx", "cy")]
    except Exception:
        return None
    direction_cam = np.array([(float(uv[0]) - cx) / fx, (float(uv[1]) - cy) / fy, 1.0], dtype=np.float64)
    direction_world = rotation @ direction_cam
    denom = float(normal @ direction_world)
    if abs(denom) < 1e-8:
        return None
    depth = -float(normal @ origin + offset) / denom
    if not math.isfinite(depth) or depth <= 0.0:
        return None
    point = origin + depth * direction_world
    return {
        "uv": [float(uv[0]), float(uv[1])],
        "point_world": [float(v) for v in point],
        "ray_depth": float(depth),
        "plane_source": plane.get("selection", {}).get("mode", plane.get("source", "road_plane")),
    }


def _nearest_pose(poses: list[dict], depth_frame: int) -> dict | None:
    if not poses:
        return None
    exact = next((pose for pose in poses if int(pose.get("frame", -10**9)) == int(depth_frame)), None)
    if exact is not None:
        return exact
    return min(poses, key=lambda pose: abs(int(pose.get("frame", -10**9)) - int(depth_frame)))


def _intrinsics_for_pose(pose: dict | None, wildgs_K: dict | None) -> dict[str, float] | None:
    intrinsics = pose.get("intrinsics") if isinstance(pose, dict) else None
    if intrinsics:
        return {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "width": float(intrinsics.get("width", 0.0) or 0.0),
            "height": float(intrinsics.get("height", 0.0) or 0.0),
        }
    if wildgs_K:
        return {
            "fx": float(wildgs_K["fx"]),
            "fy": float(wildgs_K["fy"]),
            "cx": float(wildgs_K["cx"]),
            "cy": float(wildgs_K["cy"]),
            "width": float(wildgs_K.get("width", 0.0) or 0.0),
            "height": float(wildgs_K.get("height", 0.0) or 0.0),
        }
    return None


def _vehicle_boxes_for_frame(detection_frames: list[dict], frame_id: int) -> list[list[float]]:
    entry = next((item for item in detection_frames if int(item.get("frame_idx", -1)) == int(frame_id)), None)
    if not entry:
        return []
    path = entry.get("detections")
    if not path or not Path(path).exists():
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    boxes: list[list[float]] = []
    for inst in payload.get("instances", []):
        label = str(inst.get("concept_label") or inst.get("label") or "").lower()
        if not any(token in label for token in VEHICLE_LABEL_TOKENS):
            continue
        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            continue
        if max(0.0, x2 - x1) * max(0.0, y2 - y1) < 250.0:
            continue
        boxes.append([x1, y1, x2, y2])
    return boxes


def _fit_road_plane_from_depth(
    *,
    depth: Any,
    intrinsics: dict[str, float],
    vehicle_boxes: list[list[float]],
    t_world_from_cam: Any,
    world_up_axis: str,
) -> dict[str, Any] | None:
    import numpy as np

    h, w = depth.shape
    vehicle_mask = np.zeros((h, w), dtype=bool)
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        pad = 6
        xi1 = max(0, int(math.floor(x1 - pad)))
        yi1 = max(0, int(math.floor(y1 - pad)))
        xi2 = min(w, int(math.ceil(x2 + pad)))
        yi2 = min(h, int(math.ceil(y2 + pad)))
        if xi2 > xi1 and yi2 > yi1:
            vehicle_mask[yi1:yi2, xi1:xi2] = True

    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    valid = np.isfinite(depth) & (depth > 0.0)
    band = (yy >= int(h * 0.23)) & (yy <= h - 2) & (xx >= 40) & (xx <= w - 40)
    grid = ((yy % 2) == 0) & ((xx % 2) == 0)
    sample = valid & band & grid & (~vehicle_mask)
    ys, xs = np.nonzero(sample)
    if len(xs) < 100:
        return None
    if len(xs) > 20000:
        rng = np.random.default_rng(2026)
        idx = rng.choice(len(xs), size=20000, replace=False)
        xs, ys = xs[idx], ys[idx]

    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (ys.astype(np.float64) - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points = np.column_stack([x, y, z])

    fit = _ransac_y_plane(points)
    if fit is None:
        return None
    coef, stats = fit
    normal_cam, offset_cam = _camera_plane_from_y_model(coef)
    normal_world, offset_world = _camera_plane_to_world(normal_cam, offset_cam, t_world_from_cam)

    return {
        "source": "wildgs_depth_ransac",
        "normal_world": [float(v) for v in normal_world],
        "offset": float(offset_world),
        "normal_camera": [float(v) for v in normal_cam],
        "offset_camera": float(offset_cam),
        "quality": stats,
    }


def _ransac_y_plane(points: Any) -> tuple[Any, dict[str, float]] | None:
    import numpy as np

    n = len(points)
    if n < 100:
        return None
    rng = np.random.default_rng(17)
    design = np.column_stack([points[:, 0], points[:, 2], np.ones(n)])
    target = points[:, 1]
    best_inliers = None
    best_count = -1
    for _ in range(500):
        idx = rng.choice(n, size=3, replace=False)
        try:
            coef = np.linalg.solve(design[idx], target[idx])
        except np.linalg.LinAlgError:
            continue
        resid = design @ coef - target
        inliers = np.abs(resid) < 0.12
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or int(best_inliers.sum()) < 100:
        return None
    coef, *_ = np.linalg.lstsq(design[best_inliers], target[best_inliers], rcond=None)
    resid = design @ coef - target
    abs_inlier = np.abs(resid[best_inliers])
    stats = {
        "candidate_count": int(n),
        "inlier_count": int(best_inliers.sum()),
        "inlier_ratio": float(best_inliers.mean()),
        "rmse_m": float(np.sqrt(np.mean(resid[best_inliers] ** 2))),
        "p95_abs_m": float(np.percentile(abs_inlier, 95)),
    }
    return coef, stats


def _camera_plane_from_y_model(coef: Any) -> tuple[Any, float]:
    import numpy as np

    a, b, c = [float(v) for v in coef]
    normal = np.asarray([a, -1.0, b], dtype=np.float64)
    norm = max(1e-12, float(np.linalg.norm(normal)))
    return normal / norm, c / norm


def _camera_plane_to_world(normal_cam: Any, offset_cam: float, t_world_from_cam: Any) -> tuple[Any, float]:
    import numpy as np

    transform = np.asarray(t_world_from_cam, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    normal_world = rotation @ np.asarray(normal_cam, dtype=np.float64)
    offset_world = float(offset_cam) - float(normal_world @ translation)
    norm = max(1e-12, float(np.linalg.norm(normal_world)))
    return normal_world / norm, offset_world / norm


def _merge_world_planes(keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    candidates = []
    for item in keyframes:
        quality = item.get("quality", {})
        inlier_ratio = float(quality.get("inlier_ratio", 0.0))
        rmse_m = float(quality.get("rmse_m", 1.0))
        if inlier_ratio < 0.20 or rmse_m > 0.25:
            continue
        normal = np.asarray(item["normal_world"], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-8:
            continue
        candidates.append((normal / normal_norm, float(item["offset"]), max(1e-3, inlier_ratio / max(rmse_m, 1e-3)), item))
    if not candidates:
        for item in keyframes:
            normal = np.asarray(item["normal_world"], dtype=np.float64)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-8:
                continue
            quality = item.get("quality", {})
            candidates.append((normal / normal_norm, float(item["offset"]), max(1e-3, float(quality.get("inlier_ratio", 0.0))), item))
    normals = [normal * weight for normal, _offset, weight, _item in candidates]
    offsets = np.asarray([offset for _normal, offset, _weight, _item in candidates], dtype=np.float64)
    weights = np.asarray([weight for _normal, _offset, weight, _item in candidates], dtype=np.float64)
    total = max(1e-6, float(np.sum(weights)))
    normal = np.sum(np.stack(normals, axis=0), axis=0) / total
    norm = max(1e-12, float(np.linalg.norm(normal)))
    normal = normal / norm
    offset = _weighted_quantile(offsets, weights, 0.5)
    return {
        "source": "weighted_keyframe_robust_global",
        "normal_world": [float(v) for v in normal],
        "offset": offset,
        "quality": {
            "keyframe_count": len(keyframes),
            "support_frame_count": len(candidates),
            "mean_inlier_ratio": float(np.mean([item.get("quality", {}).get("inlier_ratio", 0.0) for item in keyframes])),
            "mean_rmse_m": float(np.mean([item.get("quality", {}).get("rmse_m", 0.0) for item in keyframes])),
            "offset_p05": float(np.percentile(offsets, 5)),
            "offset_p50": float(np.percentile(offsets, 50)),
            "offset_p95": float(np.percentile(offsets, 95)),
        },
    }


def _weighted_quantile(values: Any, weights: Any, quantile: float) -> float:
    import numpy as np

    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if len(values_arr) == 0:
        return 0.0
    order = np.argsort(values_arr)
    sorted_values = values_arr[order]
    sorted_weights = np.maximum(weights_arr[order], 0.0)
    total = float(np.sum(sorted_weights))
    if total <= 1e-12:
        return float(np.median(sorted_values))
    cdf = np.cumsum(sorted_weights) / total
    idx = int(np.searchsorted(cdf, float(quantile), side="left"))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])
