from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MIN_AXIS_SCALE = 0.08
_MAX_AXIS_SCALE = 15.0
_MAX_AXIS_RATIO = 4.0


def resolve_depth_map_path(depth_maps_dir: str | Path, frame_idx: int) -> Path | None:
    depth_dir = Path(depth_maps_dir)
    direct = depth_dir / f"{frame_idx:05d}.npy"
    try:
        if direct.exists():
            return direct
    except OSError:
        pass
    zero_based = depth_dir / f"{max(frame_idx - 1, 0):05d}.npy"
    try:
        if zero_based.exists():
            return zero_based
    except OSError:
        pass
    try:
        avail = sorted(depth_dir.glob("*.npy"))
    except OSError:
        return None
    if not avail:
        return None
    target = max(frame_idx - 1, 0)
    return min(avail, key=lambda p: abs(int(p.stem) - target))


def compute_axis_scale(source_pts: Any, target_pts: Any) -> Any:
    import numpy as np

    src = np.asarray(source_pts, dtype=np.float64)
    tgt = np.asarray(target_pts, dtype=np.float64)
    src_ext = np.maximum(src.max(axis=0) - src.min(axis=0), np.array([1e-6, 1e-6, 1e-6]))
    tgt_ext = np.maximum(tgt.max(axis=0) - tgt.min(axis=0), np.array([1e-6, 1e-6, 1e-6]))
    raw = np.clip(tgt_ext / src_ext, _MIN_AXIS_SCALE, _MAX_AXIS_SCALE)
    median = float(np.median(raw))
    median = min(max(median, _MIN_AXIS_SCALE), _MAX_AXIS_SCALE)
    lower = max(_MIN_AXIS_SCALE, median / _MAX_AXIS_RATIO)
    upper = min(_MAX_AXIS_SCALE, median * _MAX_AXIS_RATIO)
    return np.clip(raw, lower, upper)


def trim_point_cloud_outliers(points: Any, *, sigma: float = 2.5, min_keep: int = 32) -> Any:
    import numpy as np

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < max(int(min_keep), 8) or arr.shape[1] != 3:
        return arr

    current = arr
    for _ in range(2):
        center = np.median(current, axis=0)
        centered = current - center
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return current
        local = centered @ vh.T

        keep = np.ones(len(current), dtype=bool)
        for axis_idx in range(3):
            axis_values = local[:, axis_idx]
            med = float(np.median(axis_values))
            mad = float(np.median(np.abs(axis_values - med)))
            robust_sigma = max(1.4826 * mad, 1e-3)
            keep &= np.abs(axis_values - med) <= float(sigma) * robust_sigma

        kept = int(keep.sum())
        if kept < max(int(min_keep), len(current) // 5):
            break
        if kept == len(current):
            break
        current = current[keep]
    return current


def build_depth_point_cloud(
    depth_maps_dir: str | Path,
    frame_idx: int,
    mask_rle: str | dict,
    *,
    cam_traj: list[dict] | None = None,
    wildgs_poses: list[dict] | None = None,
    wildgs_K: dict | None = None,
) -> Any | None:
    import numpy as np

    if frame_idx is None:
        return None
    depth_path = resolve_depth_map_path(depth_maps_dir, int(frame_idx))
    if depth_path is None or not depth_path.exists():
        return None

    try:
        depth = np.load(str(depth_path)).astype(np.float64)
    except Exception:
        return None

    try:
        from pycocotools import mask as mask_util
        rle = json.loads(mask_rle) if isinstance(mask_rle, str) else mask_rle
        binary_mask = mask_util.decode(rle).astype(bool)
    except Exception:
        return None

    mask_h, mask_w = binary_mask.shape
    depth_h, depth_w = depth.shape
    if binary_mask.shape != (depth_h, depth_w):
        try:
            from PIL import Image as _PILImage
            img = _PILImage.fromarray(binary_mask.astype(np.uint8) * 255)
            binary_mask = np.asarray(img.resize((depth_w, depth_h), _PILImage.NEAREST)) > 127
        except Exception:
            return None

    if wildgs_K:
        fx, fy = float(wildgs_K["fx"]), float(wildgs_K["fy"])
        cx, cy = float(wildgs_K["cx"]), float(wildgs_K["cy"])
    else:
        pose = next((p for p in (cam_traj or []) if p.get("frame_id") == frame_idx), (cam_traj or [None])[-1])
        K = pose.get("K") if pose else None
        if K is not None:
            fx, fy = float(K[0][0]), float(K[1][1])
            cx, cy = float(K[0][2]), float(K[1][2])
        else:
            fx = fy = max(depth_h, depth_w) * 0.8
            cx, cy = depth_w / 2.0, depth_h / 2.0

    scale_x = float(depth_w) / max(float(mask_w), 1.0)
    scale_y = float(depth_h) / max(float(mask_h), 1.0)
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    R, t = np.eye(3), np.zeros(3)
    if wildgs_poses:
        depth_frame = int(Path(depth_path).stem)
        wpose = next((p for p in wildgs_poses if int(p.get("frame", -10**9)) == depth_frame), None)
        if wpose is None:
            wpose = min(wildgs_poses, key=lambda p: abs(int(p.get("frame", -10**9)) - depth_frame))
        T = wpose.get("T_world_from_cam")
        if T is not None:
            T = np.asarray(T, dtype=np.float64)
            R, t = T[:3, :3], T[:3, 3]
    elif cam_traj:
        pose = next((p for p in cam_traj if p.get("frame_id") == frame_idx), cam_traj[-1])
        R = np.asarray(pose.get("R", np.eye(3)), dtype=np.float64)
        t = np.asarray(pose.get("t", [0, 0, 0]), dtype=np.float64)

    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None
    if len(xs) > 2000:
        idx = np.linspace(0, len(xs) - 1, 2000, dtype=int)
        xs, ys = xs[idx], ys[idx]

    d = depth[ys, xs]
    valid = d > 0.01
    xs, ys, d = xs[valid], ys[valid], d[valid]
    if len(xs) < 32:
        return None

    x_c = (xs.astype(np.float64) - cx) * d / fx
    y_c = (ys.astype(np.float64) - cy) * d / fy
    pts_cam = np.stack([x_c, y_c, d], axis=1)
    pts_world = (R @ pts_cam.T).T + t
    return trim_point_cloud_outliers(pts_world, min_keep=32)
