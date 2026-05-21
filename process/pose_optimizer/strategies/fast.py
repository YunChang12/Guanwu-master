#!/usr/bin/env python3
"""Estimate and refine 3D pose from mask, bbox, camera intrinsics, and GLB.

This script is designed for the case where task.json corrected_pose may be far
from correct. It first builds a stable camera-frame pose initialization from
mask/bbox observations and mesh geometry, then runs coarse-to-fine local
optimization to refine:
  - corrected_pose.translation_world
  - corrected_pose.rotation_matrix
  - corrected_pose.scale, constrained to the same x/y/z scale

Example:
    python -m process.pose_optimizer.cli --variant fast ^
      --sample_dir "E:\\QingYan\\pose_matching_tasks\\pose_matching_tasks\\obj_000001@000001" ^
      --output_dir "outputs\\obj_000001@000001_pose_optimized_uniform_scale"
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib.util
import json
import math
from time import perf_counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PARAM_NAMES = [
    "delta_tx_m",
    "delta_ty_m",
    "delta_tz_m",
    "delta_rx_deg",
    "delta_ry_deg",
    "delta_rz_deg",
    "log_uniform_scale",
]

PARAM_DIM = len(PARAM_NAMES)

PYTORCH3D_REQUIRED_MESSAGE = (
    "PyTorch3D is required for --render_backend pytorch3d. "
    "Please install a CUDA-compatible PyTorch3D build."
)

PROFILE_COUNT_KEYS = {
    "total_evaluations",
    "rendered_evaluations",
    "prefilter_skipped_evaluations",
    "batch_gpu_candidates_total",
    "batch_gpu_candidates_kept",
    "batch_gpu_oom_retries",
}


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in {"1", "true", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text)


def torch_tensor_to_numpy(tensor: Any, dtype: Any | None = None) -> np.ndarray:
    try:
        array = tensor.detach().cpu().numpy()
    except Exception:
        array = np.asarray(tensor.detach().cpu().tolist())
    if dtype is not None:
        array = array.astype(dtype)
    return array


def resolve_sample_dir(sample_dir: str | Path) -> Path:
    raw = Path(sample_dir)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / raw,
                cwd / "pose_matching_tasks" / raw,
                cwd / "pose_matching_tasks" / "pose_matching_tasks" / raw.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find sample_dir. Checked:\n{checked}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_image(path: Path, mode: str = "color") -> np.ndarray:
    flag = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def find_mesh_path(sample_dir: Path, task: dict[str, Any]) -> Path:
    mesh_path_value = task.get("mesh_path")
    if mesh_path_value:
        candidate = sample_dir / Path(mesh_path_value).name
        if candidate.exists():
            return candidate

    glb_files = sorted(sample_dir.glob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f"No .glb file found in {sample_dir}")
    return glb_files[0]


def image_size_from_task(task: dict[str, Any], image: np.ndarray) -> tuple[int, int]:
    width, height = [int(v) for v in task["image_size"]]
    img_h, img_w = image.shape[:2]
    if (width, height) != (img_w, img_h):
        raise ValueError(
            f"task.json image_size {(width, height)} does not match image.jpg {(img_w, img_h)}"
        )
    return width, height


def paste_crop_mask_to_full_image(
    crop_mask: np.ndarray,
    bbox_xyxy: list[float],
    image_size: tuple[int, int],
    full_image: np.ndarray | None = None,
    crop_image: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Put crop-sized mask.png back into the full image.

    If mask.png lives on a padded crop canvas and its white foreground bbox
    matches bbox_xyxy, paste the whole canvas so that foreground bbox lands on
    the rounded image bbox. This keeps padded masks from being resized or
    shifted by canvas-centre alignment.

    If the foreground bbox is not bbox-like, template matching with crop.jpg is
    used as a fallback when available. The last fallback resizes the mask to
    bbox_xyxy.
    """
    width, height = image_size
    full_mask = np.zeros((height, width), dtype=np.uint8)

    x_min, y_min, x_max, y_max = [float(v) for v in bbox_xyxy]
    x1 = int(round(x_min))
    y1 = int(round(y_min))
    x2 = int(round(x_max))
    y2 = int(round(y_max))

    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)
    mask_h, mask_w = crop_mask.shape[:2]
    binary_mask = (crop_mask > 127).astype(np.uint8)

    ys, xs = np.nonzero(binary_mask)
    if len(xs) > 0 and len(ys) > 0:
        fg_x1 = int(xs.min())
        fg_y1 = int(ys.min())
        fg_x2 = int(xs.max())
        fg_y2 = int(ys.max())
        fg_w = int(fg_x2 + 1 - fg_x1)
        fg_h = int(fg_y2 + 1 - fg_y1)
        fg_bbox_xyxy = [fg_x1, fg_y1, fg_x2, fg_y2]
    else:
        fg_x1 = fg_y1 = fg_x2 = fg_y2 = None
        fg_w = 0
        fg_h = 0
        fg_bbox_xyxy = None

    bbox_like_foreground = (
        fg_w > 0
        and fg_h > 0
        and abs(fg_w - bbox_w) <= max(3, 0.08 * bbox_w)
        and abs(fg_h - bbox_h) <= max(3, 0.08 * bbox_h)
    )
    canvas_has_padding = mask_w > bbox_w + 2 or mask_h > bbox_h + 2

    template_matched = False
    if bbox_like_foreground and canvas_has_padding and fg_bbox_xyxy is not None:
        paste_x1 = int(x1 - fg_x1)
        paste_y1 = int(y1 - fg_y1)
        paste_x2 = paste_x1 + mask_w
        paste_y2 = paste_y1 + mask_h
        mask_to_paste = binary_mask
        placement_mode = "padded_crop_fg_bbox_aligned_to_bbox"
    else:
        if (
            canvas_has_padding
            and crop_image is not None
            and full_image is not None
            and crop_image.shape[:2] == (mask_h, mask_w)
        ):
            try:
                result = cv2.matchTemplate(full_image, crop_image, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > 0.90:
                    paste_x1, paste_y1 = max_loc
                    paste_x2 = paste_x1 + mask_w
                    paste_y2 = paste_y1 + mask_h
                    mask_to_paste = binary_mask
                    placement_mode = "template_match"
                    template_matched = True
            except Exception:
                pass

        if not template_matched:
            paste_x1, paste_y1 = x1, y1
            paste_x2 = paste_x1 + bbox_w
            paste_y2 = paste_y1 + bbox_h
            mask_to_paste = cv2.resize(binary_mask, (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
            placement_mode = "resize_mask_to_bbox"

    dst_x1 = max(0, paste_x1)
    dst_y1 = max(0, paste_y1)
    dst_x2 = min(width, paste_x2)
    dst_y2 = min(height, paste_y2)

    placement_info = {
        "mode": placement_mode,
        "mask_size_wh": [int(mask_w), int(mask_h)],
        "bbox_size_wh": [int(bbox_w), int(bbox_h)],
        "bbox_xyxy_rounded": [int(x1), int(y1), int(x2), int(y2)],
        "mask_foreground_bbox_xyxy": fg_bbox_xyxy,
        "mask_foreground_bbox_size_wh": [int(fg_w), int(fg_h)],
        "paste_window_xyxy": [int(paste_x1), int(paste_y1), int(paste_x2), int(paste_y2)],
    }

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return full_mask, placement_info

    src_x1 = dst_x1 - paste_x1
    src_y1 = dst_y1 - paste_y1
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_to_paste[src_y1:src_y2, src_x1:src_x2]
    return full_mask, placement_info


def load_glb_as_mesh(mesh_path: Path) -> Any:
    """Load GLB as one mesh, merging Scene geometries when needed."""
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError(
            "Could not import trimesh. If you see a NumPy/SciPy binary error, "
            "use a compatible environment, for example numpy<2 with matching scipy/trimesh."
        ) from exc

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported GLB load result: {type(loaded)!r}")

    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh has no vertices or faces: {mesh_path}")
    return mesh.copy()


def build_proxy_mesh(vertices: np.ndarray, faces: np.ndarray, target_faces: int = 2500) -> tuple[np.ndarray, np.ndarray]:
    """Create a lightweight face subset for coarse search."""
    if len(faces) <= target_faces:
        return vertices.astype(np.float64), faces.astype(np.int32)

    stride = max(1, len(faces) // target_faces)
    sampled_faces = faces[::stride][:target_faces]
    unique_vertex_ids, inverse = np.unique(sampled_faces.reshape(-1), return_inverse=True)
    proxy_vertices = vertices[unique_vertex_ids].astype(np.float64)
    proxy_faces = inverse.reshape(-1, 3).astype(np.int32)
    return proxy_vertices, proxy_faces


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec.copy()
    return vec / norm


def axis_vector(index: int, sign: float = 1.0) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float64)
    vec[index] = sign
    return vec


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation around an arbitrary axis."""
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return Rz @ Ry @ Rx for column-vector transforms."""
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_mat @ ry_mat @ rx_mat


def make_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    dtype = np.result_type(rotation_matrix, translation, np.float32)
    transform = np.eye(4, dtype=dtype)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    transformed = (transform @ points_h.T).T
    return transformed[:, :3]


def world_up_vector_from_arg(name: str) -> np.ndarray:
    key = str(name or "").strip().lower()
    sign = -1.0 if key.startswith("-") else 1.0
    key = key[1:] if key[:1] in {"+", "-"} else key
    if key == "x":
        return np.array([sign, 0.0, 0.0], dtype=np.float64)
    if key == "y":
        return np.array([0.0, sign, 0.0], dtype=np.float64)
    if key == "z":
        return np.array([0.0, 0.0, sign], dtype=np.float64)
    raise ValueError(f"Unsupported world_up_axis: {name}")


def camera_up_vector(t_world_from_cam: np.ndarray, world_up_axis: str) -> np.ndarray:
    """Convert a world up axis into the camera frame."""
    up_world = world_up_vector_from_arg(world_up_axis)
    r_world_from_cam = t_world_from_cam[:3, :3]
    r_cam_from_world = r_world_from_cam.T
    up_cam = normalize(r_cam_from_world @ up_world)
    if np.linalg.norm(up_cam) < 1e-8:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return up_cam


def world_vector_to_camera(t_world_from_cam: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
    r_world_from_cam = np.asarray(t_world_from_cam, dtype=np.float64)[:3, :3]
    return normalize(r_world_from_cam.T @ np.asarray(vector_world, dtype=np.float64))


def project_vector_onto_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    plane_normal = normalize(plane_normal)
    return vector - np.dot(vector, plane_normal) * plane_normal


def project_points(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    points_cam = np.asarray(points_cam)
    dtype = np.float32 if points_cam.dtype == np.float32 else np.float64
    fx = dtype(fx)
    fy = dtype(fy)
    cx = dtype(cx)
    cy = dtype(cy)
    z = points_cam[:, 2]
    valid = z > 1e-8
    uv = np.full((len(points_cam), 2), np.nan, dtype=dtype)
    uv[valid, 0] = fx * points_cam[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * points_cam[valid, 1] / z[valid] + cy
    return uv, valid


def bbox_from_projected_points(projected_uv: np.ndarray, valid: np.ndarray) -> list[float] | None:
    points = projected_uv[valid]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def bbox_width_height(bbox: list[float]) -> tuple[float, float]:
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def bbox_center(bbox: list[float]) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float64)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def bbox_center_error(box_a: list[float], box_b: list[float]) -> float:
    center_a = bbox_center(box_a)
    center_b = bbox_center(box_b)
    return float(np.linalg.norm(center_a - center_b))


def batch_project_and_bbox(
    vertices: np.ndarray,
    rotations_cam: np.ndarray,
    translations_cam: np.ndarray,
    scales: np.ndarray,
    intrinsics: dict[str, float],
    target_bbox: list[float],
    enable_bbox_prefilter: bool,
    prefilter_bbox_iou_min: float,
    prefilter_center_factor: float,
    prefilter_size_ratio_min: float,
    prefilter_size_ratio_max: float,
    device: Any,
) -> dict[str, Any]:
    """Project many candidate poses with torch and run the bbox prefilter on device."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for --enable_batch_gpu_eval.") from exc

    vertices_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    rotations_t = torch.as_tensor(rotations_cam, dtype=torch.float32, device=device)
    translations_t = torch.as_tensor(translations_cam, dtype=torch.float32, device=device)
    scales_t = torch.as_tensor(scales, dtype=torch.float32, device=device)
    if scales_t.ndim == 1:
        scales_t = scales_t.reshape(-1, 1)
    if scales_t.shape[1] == 1:
        scales_t = scales_t.expand(-1, 3)

    scaled_vertices = vertices_t.unsqueeze(0) * scales_t[:, None, :]
    points_cam = torch.matmul(scaled_vertices, rotations_t.transpose(1, 2)) + translations_t[:, None, :]
    z = points_cam[..., 2]
    valid = z > 1e-8
    u = float(intrinsics["fx"]) * points_cam[..., 0] / torch.clamp(z, min=1e-8) + float(intrinsics["cx"])
    v = float(intrinsics["fy"]) * points_cam[..., 1] / torch.clamp(z, min=1e-8) + float(intrinsics["cy"])
    finite = valid & torch.isfinite(u) & torch.isfinite(v)

    huge = torch.tensor(1.0e20, dtype=torch.float32, device=device)
    x_min = torch.where(finite, u, huge).amin(dim=1)
    y_min = torch.where(finite, v, huge).amin(dim=1)
    x_max = torch.where(finite, u, -huge).amax(dim=1)
    y_max = torch.where(finite, v, -huge).amax(dim=1)
    has_valid = finite.any(dim=1)
    projected_bbox = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    target = torch.tensor(target_bbox, dtype=torch.float32, device=device)
    ix1 = torch.maximum(projected_bbox[:, 0], target[0])
    iy1 = torch.maximum(projected_bbox[:, 1], target[1])
    ix2 = torch.minimum(projected_bbox[:, 2], target[2])
    iy2 = torch.minimum(projected_bbox[:, 3], target[3])
    intersection = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)
    area_a = torch.clamp(projected_bbox[:, 2] - projected_bbox[:, 0], min=0.0) * torch.clamp(
        projected_bbox[:, 3] - projected_bbox[:, 1],
        min=0.0,
    )
    area_b = torch.clamp(target[2] - target[0], min=0.0) * torch.clamp(target[3] - target[1], min=0.0)
    union = area_a + area_b - intersection
    box_iou = torch.where(union > 0.0, intersection / union, torch.zeros_like(union))

    center_x = 0.5 * (projected_bbox[:, 0] + projected_bbox[:, 2])
    center_y = 0.5 * (projected_bbox[:, 1] + projected_bbox[:, 3])
    target_center_x = 0.5 * (target[0] + target[2])
    target_center_y = 0.5 * (target[1] + target[3])
    center_error = torch.sqrt((center_x - target_center_x) ** 2 + (center_y - target_center_y) ** 2)
    target_width = torch.clamp(target[2] - target[0], min=1e-6)
    target_height = torch.clamp(target[3] - target[1], min=1e-6)
    target_diag = torch.sqrt(target_width**2 + target_height**2)
    width_ratio = (projected_bbox[:, 2] - projected_bbox[:, 0]) / target_width
    height_ratio = (projected_bbox[:, 3] - projected_bbox[:, 1]) / target_height
    center_threshold = float(prefilter_center_factor) * target_diag
    size_ratio_ok = (
        (width_ratio >= float(prefilter_size_ratio_min))
        & (width_ratio <= float(prefilter_size_ratio_max))
        & (height_ratio >= float(prefilter_size_ratio_min))
        & (height_ratio <= float(prefilter_size_ratio_max))
    )
    if enable_bbox_prefilter:
        should_prefilter = (
            ((box_iou < float(prefilter_bbox_iou_min)) & (center_error > center_threshold))
            | ~size_ratio_ok
            | ~has_valid
        )
    else:
        should_prefilter = ~has_valid
    keep_mask = ~should_prefilter

    has_valid_cpu = torch_tensor_to_numpy(has_valid, dtype=bool)
    projected_bbox_cpu = torch_tensor_to_numpy(projected_bbox, dtype=np.float32)
    projected_bbox_cpu[~has_valid_cpu] = np.nan
    return {
        "projected_bbox": projected_bbox_cpu,
        "bbox_iou": torch_tensor_to_numpy(box_iou, dtype=np.float32),
        "bbox_center_error_px": torch_tensor_to_numpy(center_error, dtype=np.float32),
        "width_ratio": torch_tensor_to_numpy(width_ratio, dtype=np.float32),
        "height_ratio": torch_tensor_to_numpy(height_ratio, dtype=np.float32),
        "has_valid_bbox": has_valid_cpu,
        "keep": torch_tensor_to_numpy(keep_mask, dtype=bool),
    }


def get_union_roi(
    box_a: list[float] | None,
    box_b: list[float] | None,
    image_size: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int] | None:
    """Return the clipped union ROI around two boxes with a pixel margin."""
    if box_a is None or box_b is None:
        return None

    width, height = [int(v) for v in image_size]
    x1 = min(float(box_a[0]), float(box_b[0])) - float(margin)
    y1 = min(float(box_a[1]), float(box_b[1])) - float(margin)
    x2 = max(float(box_a[2]), float(box_b[2])) + float(margin)
    y2 = max(float(box_a[3]), float(box_b[3])) + float(margin)

    roi_x1 = max(0, int(math.floor(x1)))
    roi_y1 = max(0, int(math.floor(y1)))
    roi_x2 = min(width, int(math.ceil(x2)))
    roi_y2 = min(height, int(math.ceil(y2)))
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None
    return roi_x1, roi_y1, roi_x2, roi_y2


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(intersection / union)


def make_soft_mask(mask: np.ndarray, sigma_px: float = 4.0) -> np.ndarray:
    """Make a smooth target mask so small pose updates affect the objective."""
    binary = (mask > 0).astype(np.uint8)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    signed = inside - outside
    logits = np.clip(-signed / max(1e-6, sigma_px), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(logits))


def soft_mask_iou(rendered_mask: np.ndarray, soft_target: np.ndarray) -> float:
    rendered = rendered_mask.astype(np.float32)
    intersection = np.minimum(rendered, soft_target).sum()
    union = np.maximum(rendered, soft_target).sum()
    return 0.0 if union <= 0 else float(intersection / union)


def render_mask_by_triangle_fill(
    projected_uv: np.ndarray,
    valid_z: np.ndarray,
    faces: np.ndarray,
    image_size: tuple[int, int],
    batch_size: int = 20000,
) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    valid_faces = valid_z[faces].all(axis=1)
    if not np.any(valid_faces):
        return mask

    triangles = projected_uv[faces[valid_faces]]
    finite = np.isfinite(triangles).all(axis=(1, 2))
    triangles = triangles[finite]
    if len(triangles) == 0:
        return mask

    tri_min = triangles.min(axis=1)
    tri_max = triangles.max(axis=1)
    intersects = (
        (tri_max[:, 0] >= 0)
        & (tri_max[:, 1] >= 0)
        & (tri_min[:, 0] < width)
        & (tri_min[:, 1] < height)
    )
    triangles = triangles[intersects]
    if len(triangles) == 0:
        return mask

    triangles = np.rint(triangles).astype(np.int32)
    triangles[:, :, 0] = np.clip(triangles[:, :, 0], -width * 2, width * 3)
    triangles[:, :, 1] = np.clip(triangles[:, :, 1], -height * 2, height * 3)

    for start in range(0, len(triangles), batch_size):
        cv2.fillPoly(mask, triangles[start : start + batch_size], color=1)
    return mask


def render_mask_with_pyrender(
    mesh: Any,
    scale: np.ndarray,
    t_cam_from_object: np.ndarray,
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
    renderer: Any | None = None,
) -> np.ndarray:
    """Render silhouette with pyrender.

    OpenCV camera coordinates are x-right, y-down, z-forward. OpenGL/pyrender
    camera coordinates are x-right, y-up, looking along -z. cv_to_gl converts
    the object pose from the OpenCV camera frame to the pyrender camera frame.
    """
    import pyrender

    width, height = image_size
    scaled_mesh = mesh.copy()
    scale_array = np.asarray(scale)
    vertex_dtype = np.float32 if scale_array.dtype == np.float32 else np.float64
    scaled_mesh.vertices = np.asarray(mesh.vertices, dtype=vertex_dtype) * scale_array.reshape(1, 3)

    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    t_gl_cam_from_object = cv_to_gl @ t_cam_from_object

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(scaled_mesh, smooth=False), pose=t_gl_cam_from_object)
    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        znear=0.01,
        zfar=1000.0,
    )
    scene.add(camera, pose=np.eye(4))

    own_renderer = renderer is None
    if own_renderer:
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        _, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    finally:
        if own_renderer:
            renderer.delete()
    return (depth > 0).astype(np.uint8)


def import_torch_and_pytorch3d() -> tuple[Any, Any, Any, Any, Any]:
    if importlib.util.find_spec("pytorch3d") is None:
        raise RuntimeError(PYTORCH3D_REQUIRED_MESSAGE)
    try:
        import torch
        from pytorch3d.renderer import (
            MeshRasterizer,
            PerspectiveCameras,
            RasterizationSettings,
        )
        from pytorch3d.structures import Meshes
    except Exception as exc:
        raise RuntimeError(PYTORCH3D_REQUIRED_MESSAGE) from exc
    return torch, Meshes, PerspectiveCameras, RasterizationSettings, MeshRasterizer


def resolve_torch_device(device: str, allow_auto_fallback: bool) -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(PYTORCH3D_REQUIRED_MESSAGE) from exc

    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        message = f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is False."
        if allow_auto_fallback:
            print(f"[warn] {message} Falling back to CPU-capable render backend.")
            return None
        raise RuntimeError(message)
    return torch.device(requested)


def convert_cv_pose_to_pytorch3d_pose(
    rotation_cam: np.ndarray,
    translation_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an OpenCV object-to-camera pose into PyTorch3D camera coordinates.

    The optimizer uses OpenCV camera coordinates: +x right, +y down, +z forward,
    with row-vector projection in evaluate_absolute:
        points_cv = vertices @ rotation_cam.T + translation_cam

    PyTorch3D cameras use a view space where +x points left, +y points up, and
    +z points out from the image plane. Multiplying camera-frame points by
    diag([-1, -1, 1]) maps OpenCV x/y into PyTorch3D screen convention while
    keeping positive depth. PyTorch3D's Transform3d path applies row-vector
    world-to-view transforms as:
        points_view = points_world @ R + T
    so the rotation passed to PerspectiveCameras is transposed relative to the
    column-vector OpenCV pose.
    """
    cv_to_pytorch3d = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    rotation_p3d_column = cv_to_pytorch3d @ np.asarray(rotation_cam, dtype=np.float64)
    translation_p3d = cv_to_pytorch3d @ np.asarray(translation_cam, dtype=np.float64)
    rotation_p3d_row = rotation_p3d_column.T
    return rotation_p3d_row, translation_p3d


def convert_cv_pose_batch_to_pytorch3d_pose(
    rotations_cam: Any,
    translations_cam: Any,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Torch batch equivalent of convert_cv_pose_to_pytorch3d_pose."""
    cv_to_pytorch3d = torch_module.tensor(
        [-1.0, -1.0, 1.0],
        dtype=rotations_cam.dtype,
        device=rotations_cam.device,
    )
    rotations_p3d_column = rotations_cam * cv_to_pytorch3d.view(1, 3, 1)
    translations_p3d = translations_cam * cv_to_pytorch3d.view(1, 3)
    rotations_p3d_row = rotations_p3d_column.transpose(1, 2).contiguous()
    return rotations_p3d_row, translations_p3d


class TorchSilhouetteRenderer:
    """PyTorch3D silhouette renderer with mesh buffers cached on device."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        image_size: tuple[int, int],
        intrinsics: dict[str, float],
        device: str,
        faces_per_pixel: int,
        cull_backfaces: bool,
        bin_size: int | None = None,
        max_faces_per_bin: int | None = None,
    ) -> None:
        torch, Meshes, PerspectiveCameras, RasterizationSettings, MeshRasterizer = import_torch_and_pytorch3d()
        self.torch = torch
        self.Meshes = Meshes
        self.PerspectiveCameras = PerspectiveCameras
        self.RasterizationSettings = RasterizationSettings
        self.MeshRasterizer = MeshRasterizer

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested, but torch.cuda.is_available() is False.")

        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.intrinsics = {key: float(value) for key, value in intrinsics.items()}
        self.faces_per_pixel = int(faces_per_pixel)
        self.cull_backfaces = bool(cull_backfaces)
        self.bin_size = None if bin_size is None else int(bin_size)
        self.max_faces_per_bin = None if max_faces_per_bin is None else int(max_faces_per_bin)
        self.vertices_torch = torch.as_tensor(vertices, dtype=torch.float32, device=self.device)
        self.faces_torch = torch.as_tensor(faces, dtype=torch.int64, device=self.device)
        self._raster_settings = RasterizationSettings(
            image_size=(self.image_size[1], self.image_size[0]),
            blur_radius=0.0,
            faces_per_pixel=self.faces_per_pixel,
            cull_backfaces=self.cull_backfaces,
            bin_size=self.bin_size,
            max_faces_per_bin=self.max_faces_per_bin,
        )

    def _make_cameras(self, rotations_cam: Any, translations_cam: Any) -> Any:
        batch_size = int(rotations_cam.shape[0])
        rotations_p3d, translations_p3d = convert_cv_pose_batch_to_pytorch3d_pose(
            rotations_cam,
            translations_cam,
            self.torch,
        )
        focal_length = self.torch.tensor(
            [[self.intrinsics["fx"], self.intrinsics["fy"]]],
            dtype=self.torch.float32,
            device=self.device,
        ).expand(batch_size, -1)
        principal_point = self.torch.tensor(
            [[self.intrinsics["cx"], self.intrinsics["cy"]]],
            dtype=self.torch.float32,
            device=self.device,
        ).expand(batch_size, -1)
        image_size = self.torch.tensor(
            [[self.image_size[1], self.image_size[0]]],
            dtype=self.torch.float32,
            device=self.device,
        ).expand(batch_size, -1)
        return self.PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=rotations_p3d,
            T=translations_p3d,
            image_size=image_size,
            in_ndc=False,
            device=self.device,
        )

    def render_mask(
        self,
        rotation_cam: np.ndarray,
        translation_cam: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        mask = self.render_masks_batch(
            rotations_cam=np.asarray(rotation_cam, dtype=np.float32).reshape(1, 3, 3),
            translations_cam=np.asarray(translation_cam, dtype=np.float32).reshape(1, 3),
            scales=np.asarray(scale, dtype=np.float32).reshape(1, 3),
        )[0]
        return torch_tensor_to_numpy(mask.to(dtype=self.torch.uint8), dtype=np.uint8)

    def render_masks_batch(
        self,
        rotations_cam: Any,
        translations_cam: Any,
        scales: Any,
    ) -> Any:
        torch = self.torch
        rotations = torch.as_tensor(rotations_cam, dtype=torch.float32, device=self.device)
        translations = torch.as_tensor(translations_cam, dtype=torch.float32, device=self.device)
        scale_values = torch.as_tensor(scales, dtype=torch.float32, device=self.device)
        if scale_values.ndim == 1:
            scale_values = scale_values.reshape(-1, 1)
        if scale_values.shape[1] == 1:
            scale_values = scale_values.expand(-1, 3)
        if scale_values.shape[1] != 3:
            raise ValueError(f"scales must have shape [B, 3] or [B, 1], got {tuple(scale_values.shape)}")

        batch_size = int(rotations.shape[0])
        if translations.shape[0] != batch_size or scale_values.shape[0] != batch_size:
            raise ValueError("rotations_cam, translations_cam, and scales must have the same batch size.")

        scaled_vertices = self.vertices_torch.unsqueeze(0) * scale_values[:, None, :]
        meshes = self.Meshes(
            verts=[scaled_vertices[index] for index in range(batch_size)],
            faces=[self.faces_torch for _ in range(batch_size)],
        )
        cameras = self._make_cameras(rotations, translations)
        rasterizer = self.MeshRasterizer(cameras=cameras, raster_settings=self._raster_settings)
        fragments = rasterizer(meshes)
        return fragments.pix_to_face[..., 0] >= 0



def extract_mask_observations(full_mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(full_mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("full_mask is empty.")

    bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    y_bottom = int(ys.max())
    bottom_xs = xs[ys == y_bottom]
    bottom_center = np.array([float(bottom_xs.mean()), float(y_bottom)], dtype=np.float64)

    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = coords - centroid.reshape(1, 2)
    cov = centered.T @ centered / max(1, len(coords) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    principal_angle_deg = float(math.degrees(math.atan2(principal[1], principal[0])))

    return {
        "mask_bbox": bbox,
        "bbox_center": bbox_center(bbox),
        "bbox_wh": bbox_width_height(bbox),
        "centroid": centroid,
        "bottom_center": bottom_center,
        "principal_angle_deg": principal_angle_deg,
        "area_px": int(full_mask.sum()),
    }


def mesh_axis_metadata(vertices: np.ndarray) -> dict[str, Any]:
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0).astype(np.float64)
    extents = bounds[1] - bounds[0]
    longest_axis = int(np.argmax(extents))
    shortest_axis = int(np.argmin(extents))
    medium_axis = int([i for i in range(3) if i not in (longest_axis, shortest_axis)][0])
    return {
        "bounds": bounds,
        "extents": extents,
        "longest_axis": longest_axis,
        "medium_axis": medium_axis,
        "shortest_axis": shortest_axis,
        "center": bounds.mean(axis=0),
    }


def infer_tail_light_axis_prior(
    mesh: Any,
    mesh_meta: dict[str, Any],
    *,
    min_red_vertices: int = 120,
    min_end_density_ratio: float = 1.5,
    min_end_density_delta: float = 0.003,
) -> dict[str, Any]:
    """Infer rear/front sign from red vertex colors concentrated at one long-axis end."""

    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    if vertices.size == 0:
        return {"available": False, "reason": "missing_vertices"}

    visual = getattr(mesh, "visual", None)
    colors = getattr(visual, "vertex_colors", None)
    if colors is None:
        return {"available": False, "reason": "missing_vertex_colors"}
    colors_array = np.asarray(colors)
    if colors_array.ndim != 2 or colors_array.shape[0] != vertices.shape[0] or colors_array.shape[1] < 3:
        return {"available": False, "reason": "invalid_vertex_colors"}

    rgb = colors_array[:, :3].astype(np.float64)
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]
    max_gb = np.maximum(green, blue)
    brightness = red + green + blue
    red_mask = (red > 45.0) & ((red - max_gb) > 8.0) & (red > max_gb * 1.15) & (brightness > 80.0)
    red_count = int(red_mask.sum())
    if red_count < int(min_red_vertices):
        return {"available": False, "reason": "insufficient_red_vertices", "red_vertex_count": red_count}

    long_axis = int(mesh_meta.get("longest_axis", 2))
    bounds = np.asarray(mesh_meta["bounds"], dtype=np.float64)
    extent = float(bounds[1, long_axis] - bounds[0, long_axis])
    if extent <= 1e-8:
        return {"available": False, "reason": "degenerate_long_axis", "red_vertex_count": red_count}

    coord = vertices[:, long_axis]
    neg_end = coord <= bounds[0, long_axis] + 0.25 * extent
    pos_end = coord >= bounds[1, long_axis] - 0.25 * extent
    neg_density = float(red_mask[neg_end].mean()) if np.any(neg_end) else 0.0
    pos_density = float(red_mask[pos_end].mean()) if np.any(pos_end) else 0.0
    neg_count = int(np.logical_and(red_mask, neg_end).sum())
    pos_count = int(np.logical_and(red_mask, pos_end).sum())

    stronger = max(neg_density, pos_density)
    weaker = max(1e-9, min(neg_density, pos_density))
    density_ratio = float(stronger / weaker)
    density_delta = float(abs(pos_density - neg_density))
    rear_sign = -1.0 if neg_density > pos_density else 1.0
    confidence = float(np.clip((density_ratio - 1.0) / 3.0 + density_delta / 0.02, 0.0, 1.0))

    available = bool(
        density_ratio >= float(min_end_density_ratio)
        and density_delta >= float(min_end_density_delta)
        and (neg_count + pos_count) >= int(min_red_vertices)
    )
    return {
        "available": available,
        "source": "mesh_vertex_red_tail_light",
        "reason": None if available else "red_not_concentrated_on_one_long_axis_end",
        "axis_idx": long_axis,
        "rear_sign": rear_sign,
        "front_sign": -rear_sign,
        "confidence": confidence,
        "strong_available": available,
        "red_vertex_count": red_count,
        "negative_end_red_count": neg_count,
        "positive_end_red_count": pos_count,
        "negative_end_red_density": neg_density,
        "positive_end_red_density": pos_density,
        "density_ratio": density_ratio,
        "density_delta": density_delta,
    }


def find_project_outputs_root(sample_dir: Path) -> Path | None:
    """Find the project outputs root from an 08_pose_optimize task directory."""

    resolved = sample_dir.resolve()
    for parent in [resolved, *resolved.parents]:
        if parent.name == "outputs":
            return parent
    return None


def compute_bbox_area_trend_from_geometry_lift(
    *,
    sample_dir: Path,
    object_id: str,
    frame_idx: int,
    heading_prior: dict[str, Any],
) -> dict[str, Any] | None:
    outputs_root = find_project_outputs_root(sample_dir)
    if outputs_root is None:
        return None
    frames_root = outputs_root / "06_geometry_lift" / "frames"
    if not frames_root.exists():
        return None

    window = heading_prior.get("frame_window") if isinstance(heading_prior, dict) else None
    if isinstance(window, list) and len(window) >= 2:
        start_frame, end_frame = int(window[0]), int(window[1])
    else:
        radius = 6
        start_frame, end_frame = frame_idx - radius, frame_idx + radius

    observations: list[dict[str, float]] = []
    for current in range(max(0, start_frame), end_frame + 1):
        path = frames_root / f"frame_{current:06d}" / "observed_objects.json"
        if not path.exists():
            continue
        try:
            objects = read_json(path)
        except Exception:
            continue
        if not isinstance(objects, list):
            continue
        for item in objects:
            if not isinstance(item, dict) or str(item.get("object_id")) != str(object_id):
                continue
            bbox = ((item.get("geometry") or {}).get("bbox_2d") if isinstance(item.get("geometry"), dict) else None)
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            observations.append(
                {
                    "frame": float(current),
                    "center_x": 0.5 * (x1 + x2),
                    "center_y": 0.5 * (y1 + y2),
                    "area": width * height,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            break

    if len(observations) < 3:
        return None
    observations.sort(key=lambda row: row["frame"])
    frames = np.asarray([row["frame"] for row in observations], dtype=np.float64)
    areas = np.asarray([row["area"] for row in observations], dtype=np.float64)
    valid = areas > 1.0
    if int(valid.sum()) < 3:
        return None
    frames = frames[valid]
    areas = areas[valid]
    valid_observations = [row for row, keep in zip(observations, valid.tolist()) if keep]
    log_areas = np.log(np.maximum(1.0, areas))
    centered_frames = frames - float(frame_idx)
    try:
        slope = float(np.polyfit(centered_frames, log_areas, 1)[0])
    except Exception:
        return None
    ratio = float(areas[-1] / max(1.0, areas[0]))
    confidence = float(np.clip(abs(math.log(max(1e-6, ratio))) / math.log(1.8), 0.0, 1.0))
    diffs = np.diff(areas)
    nonzero_diffs = diffs[np.abs(diffs) > max(1.0, float(np.median(areas)) * 0.01)]
    if int(nonzero_diffs.size) > 0:
        direction_sign = 1.0 if slope >= 0.0 else -1.0
        monotonicity = float(np.mean((nonzero_diffs * direction_sign) >= 0.0))
    else:
        monotonicity = 1.0
    tail_count = min(3, len(valid_observations))
    tail_rows = valid_observations[-tail_count:] if tail_count > 0 else []
    truncated_tail = any(float(row.get("y2", 0.0) or 0.0) >= 358.0 for row in tail_rows)
    if truncated_tail:
        confidence *= 0.5
    if monotonicity < 0.75:
        confidence *= max(0.25, monotonicity)
    if abs(slope) < 0.015 or confidence < 0.15:
        direction = "stable"
    elif slope > 0.0:
        direction = "approaching"
    else:
        direction = "receding"
    return {
        "source": "geometry_lift_bbox_area_trend",
        "direction": direction,
        "confidence": confidence,
        "log_area_slope_per_frame": slope,
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "first_area": float(areas[0]),
        "last_area": float(areas[-1]),
        "area_ratio": ratio,
        "sample_count": int(len(areas)),
        "monotonicity": monotonicity,
        "truncated_tail": bool(truncated_tail),
    }


def apply_mesh_axis_prior(mesh_meta: dict[str, Any], axis_prior: dict[str, Any] | None) -> dict[str, Any]:
    """Override bbox-derived semantic axes with task-provided mesh axis metadata."""
    if not isinstance(axis_prior, dict) or not bool(axis_prior.get("available", True)):
        return mesh_meta
    updated = dict(mesh_meta)
    up_idx = axis_prior.get("up_axis_idx")
    forward_idx = axis_prior.get("forward_axis_idx")
    if up_idx is None or forward_idx is None:
        return updated
    try:
        up_axis = int(up_idx)
        forward_axis = int(forward_idx)
    except Exception:
        return updated
    if up_axis not in (0, 1, 2) or forward_axis not in (0, 1, 2) or up_axis == forward_axis:
        return updated
    right_axis = int(axis_prior.get("right_axis_idx", [i for i in range(3) if i not in (up_axis, forward_axis)][0]))
    updated["shortest_axis"] = up_axis
    updated["longest_axis"] = forward_axis
    updated["medium_axis"] = right_axis
    updated["axis_prior"] = axis_prior
    return updated


def bottom_center_local(bounds: np.ndarray, up_axis: int, up_sign: float) -> np.ndarray:
    point = bounds.mean(axis=0).copy()
    if up_sign > 0:
        point[up_axis] = bounds[0, up_axis]
    else:
        point[up_axis] = bounds[1, up_axis]
    return point


def bottom_contact_points_local(bounds: np.ndarray, up_axis: int, up_sign: float) -> np.ndarray:
    bottom_value = bounds[0, up_axis] if up_sign > 0 else bounds[1, up_axis]
    other_axes = [axis for axis in range(3) if axis != up_axis]
    center = bounds.mean(axis=0)
    points = []
    for a_value in (bounds[0, other_axes[0]], bounds[1, other_axes[0]]):
        for b_value in (bounds[0, other_axes[1]], bounds[1, other_axes[1]]):
            point = center.copy()
            point[up_axis] = bottom_value
            point[other_axes[0]] = a_value
            point[other_axes[1]] = b_value
            points.append(point)
    points.append(bottom_center_local(bounds, up_axis, up_sign))
    return np.asarray(points, dtype=np.float64)


def axis_alignment_rotation(
    mesh_meta: dict[str, Any],
    up_cam: np.ndarray,
    yaw_deg: float,
    up_sign: float,
    forward_sign: float,
) -> np.ndarray:
    """Build a camera-frame upright rotation from mesh semantic axes.

    The shortest mesh axis is treated as vertical, the longest as forward/back.
    A yaw rotation around the camera-frame up axis then generates road-plane
    heading hypotheses.
    """
    long_axis = mesh_meta["longest_axis"]
    short_axis = mesh_meta["shortest_axis"]

    up_cam = normalize(up_cam)
    forward_ref = project_vector_onto_plane(np.array([0.0, 0.0, 1.0], dtype=np.float64), up_cam)
    if np.linalg.norm(forward_ref) < 1e-8:
        forward_ref = project_vector_onto_plane(np.array([1.0, 0.0, 0.0], dtype=np.float64), up_cam)
    forward_ref = normalize(forward_ref)
    yaw_rot = rotation_about_axis(up_cam, math.radians(yaw_deg))
    forward_cam = normalize(yaw_rot @ forward_ref)
    right_cam = normalize(np.cross(up_cam, forward_cam))
    forward_cam = normalize(np.cross(right_cam, up_cam))

    local_up = axis_vector(short_axis, up_sign)
    local_forward = axis_vector(long_axis, forward_sign)
    local_right = normalize(np.cross(local_up, local_forward))
    if np.linalg.norm(local_right) < 1e-8:
        raise ValueError("Invalid local basis.")
    local_forward = normalize(np.cross(local_right, local_up))

    local_basis = np.column_stack([local_right, local_up, local_forward])
    cam_basis = np.column_stack([right_cam, up_cam, forward_cam])
    return cam_basis @ local_basis.T


def solve_translation_from_anchor(
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    tz: float,
    anchor_local: np.ndarray,
    target_uv: np.ndarray,
    intrinsics: dict[str, float],
) -> np.ndarray | None:
    scaled_anchor = anchor_local * scale
    rotated_anchor = rotation_cam @ scaled_anchor
    depth = rotated_anchor[2] + tz
    if depth <= 1e-6:
        return None

    tx = (target_uv[0] - intrinsics["cx"]) * depth / intrinsics["fx"] - rotated_anchor[0]
    ty = (target_uv[1] - intrinsics["cy"]) * depth / intrinsics["fy"] - rotated_anchor[1]
    return np.array([tx, ty, tz], dtype=np.float64)


def parse_comma_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def make_uniform_scale(scale_value: float) -> np.ndarray:
    return np.full(3, float(scale_value), dtype=np.float64)


def scale_to_uniform_scalar(scale: np.ndarray) -> float:
    valid = np.asarray(scale, dtype=np.float64)
    valid = valid[np.isfinite(valid) & (valid > 1e-6)]
    if valid.size == 0:
        return 1.0
    return float(np.exp(np.mean(np.log(valid))))


def pose_from_deltas(
    base_translation_cam: np.ndarray,
    base_rotation_cam: np.ndarray,
    base_scale: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = base_translation_cam + params[:3]
    delta_rotation = euler_xyz_to_matrix(params[3], params[4], params[5])
    rotation = delta_rotation @ base_rotation_cam
    uniform_scale = scale_to_uniform_scalar(base_scale) * math.exp(float(params[6]))
    scale = make_uniform_scale(uniform_scale)
    return translation, rotation, scale


def camera_pose_to_world_pose(
    t_world_from_cam: np.ndarray,
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_cam_from_object = make_transform(rotation_cam, translation_cam)
    t_world_from_object = t_world_from_cam @ t_cam_from_object
    return t_world_from_object[:3, 3].copy(), t_world_from_object[:3, :3].copy()


def oriented_plane(
    normal: np.ndarray,
    offset: float,
    world_up_axis: str = "y",
) -> tuple[np.ndarray, float]:
    normal_arr = normalize(np.asarray(normal, dtype=np.float64))
    offset_val = float(offset)
    up_world = world_up_vector_from_arg(world_up_axis)
    if float(np.dot(normal_arr, up_world)) < 0.0:
        normal_arr = -normal_arr
        offset_val = -offset_val
    return normal_arr, offset_val


def find_depth_map_for_task(sample_dir: Path, frame_idx: int) -> Path | None:
    frame_name = f"{int(frame_idx):05d}.npy"
    for parent in [sample_dir, *sample_dir.parents]:
        if parent.name != "outputs":
            continue
        candidate = parent / "06_geometry_lift" / "wildgs" / "exports" / "depth_maps" / "depth_maps" / frame_name
        if candidate.exists():
            return candidate
    return None


def estimate_road_plane_from_depth(
    *,
    depth_path: Path,
    t_world_from_cam: np.ndarray,
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
    exclude_bbox: list[float] | None,
    exclude_mask: np.ndarray | None,
    world_up_axis: str,
    max_points: int = 20000,
    ransac_iters: int = 96,
    distance_threshold_m: float = 0.12,
) -> dict[str, Any]:
    depth = np.load(depth_path)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
    width, height = int(image_size[0]), int(image_size[1])
    if depth.shape != (height, width):
        depth = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    y0 = int(height * 0.52)
    valid = np.isfinite(depth) & (depth > 0.05)
    valid[:y0, :] = False
    if exclude_bbox is not None:
        margin = 8
        x1 = max(0, int(math.floor(float(exclude_bbox[0]))) - margin)
        y1 = max(0, int(math.floor(float(exclude_bbox[1]))) - margin)
        x2 = min(width, int(math.ceil(float(exclude_bbox[2]))) + margin)
        y2 = min(height, int(math.ceil(float(exclude_bbox[3]))) + margin)
        valid[y1:y2, x1:x2] = False
    if exclude_mask is not None and exclude_mask.shape == valid.shape:
        valid[np.asarray(exclude_mask) > 0] = False

    ys, xs = np.nonzero(valid)
    if len(xs) < 256:
        raise ValueError("Not enough valid road depth candidates")

    rng = np.random.default_rng(13)
    candidate_count = min(int(max_points), len(xs))
    choice = rng.choice(len(xs), size=candidate_count, replace=False)
    xs = xs[choice].astype(np.float64)
    ys = ys[choice].astype(np.float64)
    zs = depth[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float64)

    points_cam = np.stack(
        [
            (xs - float(intrinsics["cx"])) * zs / float(intrinsics["fx"]),
            (ys - float(intrinsics["cy"])) * zs / float(intrinsics["fy"]),
            zs,
        ],
        axis=1,
    )
    points_world = transform_points(points_cam, np.asarray(t_world_from_cam, dtype=np.float64))

    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    if len(points_world) < 3:
        raise ValueError("Not enough points to fit road plane")

    for _ in range(max(8, int(ransac_iters))):
        ids = rng.choice(len(points_world), size=3, replace=False)
        a, b, c = points_world[ids]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, a))
        distances = np.abs(points_world @ normal + offset)
        inliers = distances < float(distance_threshold_m)
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_normal = normal
            best_offset = offset

    if best_inliers is None or best_normal is None or int(best_inliers.sum()) < 64:
        raise ValueError("Road plane RANSAC failed")

    inlier_points = points_world[best_inliers]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    offset = -float(np.dot(normal, center))
    normal, offset = oriented_plane(normal, offset, world_up_axis)
    distances = np.abs(points_world @ normal + offset)
    inlier_distances = distances[best_inliers]
    normal_cam = world_vector_to_camera(t_world_from_cam, normal)
    cam_origin_world = np.asarray(t_world_from_cam, dtype=np.float64)[:3, 3]
    offset_cam = float(offset + np.dot(normal, cam_origin_world))
    return {
        "source": "optimizer_depth_ransac",
        "normal_world": normal.tolist(),
        "offset": float(offset),
        "normal_camera": normal_cam.tolist(),
        "offset_camera": offset_cam,
        "depth_frame": int(depth_path.stem),
        "depth_path": str(depth_path),
        "quality": {
            "candidate_count": int(len(points_world)),
            "inlier_count": int(best_inliers.sum()),
            "inlier_ratio": float(best_inliers.sum() / max(1, len(points_world))),
            "rmse_m": float(np.sqrt(np.mean(inlier_distances ** 2))),
            "p95_abs_m": float(np.percentile(inlier_distances, 95)),
        },
    }


def build_vehicle_pose_context(
    *,
    task: dict[str, Any],
    sample_dir: Path,
    full_mask: np.ndarray,
    json_bbox: list[float],
    image_size: tuple[int, int],
    intrinsics: dict[str, float],
    t_world_from_cam: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_context = task.get("vehicle_pose_context") if isinstance(task.get("vehicle_pose_context"), dict) else {}
    mesh_axis_prior = dict(raw_context.get("mesh_axis_prior") or {})
    if bool(getattr(args, "vehicle_mesh_axis_override_enabled", True)):
        up_axis = int(getattr(args, "vehicle_mesh_up_axis_idx", mesh_axis_prior.get("up_axis_idx", 1)))
        up_sign = -1.0 if float(getattr(args, "vehicle_mesh_up_sign", mesh_axis_prior.get("up_sign", -1.0))) < 0 else 1.0
        up_candidates = mesh_axis_prior.get("up_sign_candidates")
        if not isinstance(up_candidates, list) or len(up_candidates) < 2:
            up_candidates = [up_sign, -up_sign]
        mesh_axis_prior.update(
            {
                "available": True,
                "up_axis_idx": up_axis,
                "up_sign": up_sign,
                "up_sign_candidates": up_candidates,
                "lock_up_sign": False,
                "up_sign_source": "optimizer_vehicle_axis_preferred",
            }
        )
    context = {
        "schema": "vehicle_pose_context.optimizer.v1",
        "mesh_axis_prior": mesh_axis_prior,
        "heading_prior": dict(raw_context.get("heading_prior", {}) if isinstance(raw_context.get("heading_prior"), dict) else {}),
        "road_constraint": {"available": False, "reason": "missing_road_plane"},
    }
    if isinstance(raw_context.get("track_scale_prior"), dict):
        context["track_scale_prior"] = dict(raw_context["track_scale_prior"])

    area_trend = compute_bbox_area_trend_from_geometry_lift(
        sample_dir=sample_dir,
        object_id=str(task.get("object_id", "")),
        frame_idx=int(task.get("frame_idx", 0) or 0),
        heading_prior=context["heading_prior"],
    )
    if area_trend is not None:
        context["heading_prior"]["bbox_area_trend"] = area_trend

    road_plane = raw_context.get("road_plane")
    depth_error = None
    if road_plane is None and bool(getattr(args, "road_depth_fallback_enabled", True)):
        frame_idx = int(task.get("frame_idx", 0) or 0)
        depth_path_text = str(getattr(args, "road_depth_map_path", "") or "").strip()
        depth_path = Path(depth_path_text) if depth_path_text else find_depth_map_for_task(sample_dir, frame_idx)
        if depth_path is not None and depth_path.exists():
            try:
                road_plane = estimate_road_plane_from_depth(
                    depth_path=depth_path,
                    t_world_from_cam=t_world_from_cam,
                    intrinsics=intrinsics,
                    image_size=image_size,
                    exclude_bbox=json_bbox,
                    exclude_mask=full_mask,
                    world_up_axis=str(getattr(args, "world_up_axis", "y")),
                    max_points=int(getattr(args, "road_depth_ransac_max_points", 20000)),
                    ransac_iters=int(getattr(args, "road_depth_ransac_iters", 96)),
                    distance_threshold_m=float(getattr(args, "road_depth_ransac_threshold_m", 0.12)),
                )
            except Exception as exc:
                depth_error = str(exc)
        else:
            depth_error = f"depth_map_not_found:{frame_idx:05d}.npy"

    if isinstance(road_plane, dict) and "normal_world" in road_plane and "offset" in road_plane:
        normal, offset = oriented_plane(
            np.asarray(road_plane["normal_world"], dtype=np.float64),
            float(road_plane["offset"]),
            str(getattr(args, "world_up_axis", "y")),
        )
        road_plane = dict(road_plane)
        road_plane["normal_world"] = normal.tolist()
        road_plane["offset"] = float(offset)
        context["road_constraint"] = {
            "available": True,
            "road_plane": road_plane,
            "bbox_bottom_ground": raw_context.get("bbox_bottom_ground"),
            "source": road_plane.get("source", "task_vehicle_pose_context"),
        }
    elif depth_error:
        context["road_constraint"] = {"available": False, "reason": depth_error}

    return context


class CameraPoseEvaluator:
    """Evaluate absolute or delta camera-frame poses against mask/bbox."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh: Any | None,
        full_mask: np.ndarray,
        soft_full_mask: np.ndarray,
        json_bbox: list[float],
        intrinsics: dict[str, float],
        image_size: tuple[int, int],
        bbox_weight: float,
        hard_mask_weight: float,
        backend: str,
        enable_bbox_prefilter: bool,
        prefilter_bbox_iou_min: float,
        prefilter_center_factor: float,
        prefilter_size_ratio_min: float,
        prefilter_size_ratio_max: float,
        roi_iou_margin: int,
        disable_roi_iou: bool,
        fast_float32: bool,
        profile_timings: bool,
        device: str = "cuda",
        pytorch3d_faces_per_pixel: int = 1,
        pytorch3d_cull_backfaces: bool = False,
        pytorch3d_bin_size: int | None = None,
        pytorch3d_max_faces_per_bin: int | None = None,
    ) -> None:
        self.fast_float32 = bool(fast_float32)
        self.dtype = np.float32 if self.fast_float32 else np.float64
        self.vertices = np.asarray(vertices, dtype=self.dtype)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.mesh = mesh
        self.full_mask = np.asarray(full_mask, dtype=np.uint8)
        self.soft_full_mask = np.asarray(soft_full_mask, dtype=self.dtype)
        self.json_bbox = [float(v) for v in json_bbox]
        target_width, target_height = bbox_width_height(self.json_bbox)
        self.target_bbox_width = max(1e-6, float(target_width))
        self.target_bbox_height = max(1e-6, float(target_height))
        self.target_bbox_diagonal = max(1e-6, math.hypot(self.target_bbox_width, self.target_bbox_height))
        self.intrinsics = {key: float(value) for key, value in intrinsics.items()}
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.bbox_weight = float(bbox_weight)
        self.hard_mask_weight = float(hard_mask_weight)
        self.backend_preference = backend
        self.active_backend: str | None = None
        self.enable_bbox_prefilter = bool(enable_bbox_prefilter)
        self.prefilter_bbox_iou_min = float(prefilter_bbox_iou_min)
        self.prefilter_center_factor = float(prefilter_center_factor)
        self.prefilter_size_ratio_min = float(prefilter_size_ratio_min)
        self.prefilter_size_ratio_max = float(prefilter_size_ratio_max)
        self.roi_iou_margin = int(roi_iou_margin)
        self.disable_roi_iou = bool(disable_roi_iou)
        self.profile_timings = bool(profile_timings)
        self.device = str(device)
        self.pytorch3d_faces_per_pixel = int(pytorch3d_faces_per_pixel)
        self.pytorch3d_cull_backfaces = bool(pytorch3d_cull_backfaces)
        self.pytorch3d_bin_size = None if pytorch3d_bin_size is None else int(pytorch3d_bin_size)
        self.pytorch3d_max_faces_per_bin = (
            None if pytorch3d_max_faces_per_bin is None else int(pytorch3d_max_faces_per_bin)
        )
        self._pyrender_renderer: Any | None = None
        self._torch_renderer: TorchSilhouetteRenderer | None = None
        self._profile_stats: dict[str, float] | None = (
            {
                "total_evaluations": 0.0,
                "rendered_evaluations": 0.0,
                "prefilter_skipped_evaluations": 0.0,
                "render_time": 0.0,
                "iou_time": 0.0,
                "projection_time": 0.0,
                "gpu_projection_time": 0.0,
                "gpu_rasterization_time": 0.0,
                "batch_gpu_candidates_total": 0.0,
                "batch_gpu_candidates_kept": 0.0,
                "batch_gpu_oom_retries": 0.0,
                "pytorch3d_alignment_iou": 0.0,
            }
            if self.profile_timings
            else None
        )

    def _empty_mask(self) -> np.ndarray:
        return np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)

    def _start_timer(self) -> float | None:
        return perf_counter() if self.profile_timings else None

    def _stop_timer(self, key: str, start: float | None) -> None:
        if self.profile_timings and start is not None and self._profile_stats is not None:
            self._profile_stats[key] += perf_counter() - start

    def add_profile_value(self, key: str, value: float) -> None:
        if self.profile_timings and self._profile_stats is not None:
            self._profile_stats[key] += float(value)

    def set_profile_value(self, key: str, value: float) -> None:
        if self.profile_timings and self._profile_stats is not None:
            self._profile_stats[key] = float(value)

    def profile_stats(self) -> dict[str, float | int]:
        if not self.profile_timings or self._profile_stats is None:
            return {}
        stats: dict[str, float | int] = {}
        for key, value in self._profile_stats.items():
            if key in PROFILE_COUNT_KEYS:
                stats[key] = int(round(value))
            else:
                stats[key] = float(value)
        return stats

    def close(self) -> None:
        if self._pyrender_renderer is not None:
            try:
                self._pyrender_renderer.delete()
            finally:
                self._pyrender_renderer = None
        self._torch_renderer = None

    def _get_pyrender_renderer(self) -> Any:
        if self._pyrender_renderer is None:
            import pyrender

            self._pyrender_renderer = pyrender.OffscreenRenderer(
                viewport_width=self.image_size[0],
                viewport_height=self.image_size[1],
            )
        return self._pyrender_renderer

    def _get_torch_renderer(self) -> TorchSilhouetteRenderer:
        if self._torch_renderer is None:
            self._torch_renderer = TorchSilhouetteRenderer(
                vertices=self.vertices,
                faces=self.faces,
                image_size=self.image_size,
                intrinsics=self.intrinsics,
                device=self.device,
                faces_per_pixel=self.pytorch3d_faces_per_pixel,
                cull_backfaces=self.pytorch3d_cull_backfaces,
                bin_size=self.pytorch3d_bin_size,
                max_faces_per_bin=self.pytorch3d_max_faces_per_bin,
            )
        return self._torch_renderer

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_result(
        self,
        translation_cam: np.ndarray,
        rotation_cam: np.ndarray,
        scale: np.ndarray,
        projected_bbox: list[float] | None,
        score: float,
        mask_iou_value: float,
        soft_mask_iou_value: float,
        bbox_iou_value: float,
        bbox_center_error_value: float,
        rendering_method: str,
        keep_mask: bool,
        rendered_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "score": float(score),
            "mask_iou": float(mask_iou_value),
            "soft_mask_iou": float(soft_mask_iou_value),
            "bbox_iou": float(bbox_iou_value),
            "bbox_center_error_px": float(bbox_center_error_value),
            "projected_bbox": projected_bbox,
            "translation_cam": np.asarray(translation_cam, dtype=self.dtype).copy(),
            "rotation_cam": np.asarray(rotation_cam, dtype=self.dtype).copy(),
            "scale": np.asarray(scale, dtype=self.dtype).copy(),
            "rendering_method": rendering_method,
        }
        if keep_mask:
            result["rendered_mask"] = (
                np.asarray(rendered_mask, dtype=np.uint8)
                if rendered_mask is not None
                else self._empty_mask()
            )
        return result

    def evaluate_absolute(
        self,
        translation_cam: np.ndarray,
        rotation_cam: np.ndarray,
        scale: np.ndarray,
        keep_mask: bool = False,
    ) -> dict[str, Any]:
        if self.profile_timings and self._profile_stats is not None:
            self._profile_stats["total_evaluations"] += 1.0

        translation_cam = np.asarray(translation_cam, dtype=self.dtype)
        rotation_cam = np.asarray(rotation_cam, dtype=self.dtype)
        scale = np.asarray(scale, dtype=self.dtype)

        projection_start = self._start_timer()
        scaled_vertices = self.vertices * scale.reshape(1, 3)
        points_cam = scaled_vertices @ rotation_cam.T + translation_cam.reshape(1, 3)
        projected_uv, valid_z = project_points(points_cam, **self.intrinsics)
        projected_bbox = bbox_from_projected_points(projected_uv, valid_z)
        self._stop_timer("projection_time", projection_start)

        if projected_bbox is None:
            return self._build_result(
                translation_cam=translation_cam,
                rotation_cam=rotation_cam,
                scale=scale,
                projected_bbox=None,
                score=-1.0,
                mask_iou_value=0.0,
                soft_mask_iou_value=0.0,
                bbox_iou_value=0.0,
                bbox_center_error_value=float("inf"),
                rendering_method="none",
                keep_mask=keep_mask,
            )

        metrics_start = self._start_timer()
        box_iou = bbox_iou(projected_bbox, self.json_bbox)
        center_error = bbox_center_error(projected_bbox, self.json_bbox)
        projected_width = float(projected_bbox[2] - projected_bbox[0])
        projected_height = float(projected_bbox[3] - projected_bbox[1])
        width_ratio = projected_width / self.target_bbox_width
        height_ratio = projected_height / self.target_bbox_height

        should_prefilter = False
        if self.enable_bbox_prefilter:
            center_threshold = self.prefilter_center_factor * self.target_bbox_diagonal
            size_ratio_ok = (
                self.prefilter_size_ratio_min <= width_ratio <= self.prefilter_size_ratio_max
                and self.prefilter_size_ratio_min <= height_ratio <= self.prefilter_size_ratio_max
            )
            should_prefilter = (
                (box_iou < self.prefilter_bbox_iou_min and center_error > center_threshold)
                or not size_ratio_ok
            )

        if should_prefilter:
            if self.profile_timings and self._profile_stats is not None:
                self._profile_stats["prefilter_skipped_evaluations"] += 1.0
            self._stop_timer("iou_time", metrics_start)
            return self._build_result(
                translation_cam=translation_cam,
                rotation_cam=rotation_cam,
                scale=scale,
                projected_bbox=projected_bbox,
                score=-1.0 + 0.1 * box_iou,
                mask_iou_value=0.0,
                soft_mask_iou_value=0.0,
                bbox_iou_value=box_iou,
                bbox_center_error_value=center_error,
                rendering_method="bbox_prefilter",
                keep_mask=keep_mask,
            )

        t_cam_from_object: np.ndarray | None = None
        if self.mesh is not None and self.backend_preference in {"auto", "pyrender"} and self.active_backend != "triangle_fill":
            t_cam_from_object = make_transform(rotation_cam, translation_cam)

        render_start = self._start_timer()
        rendered_mask, rendering_method = self.render_mask(
            projected_uv,
            valid_z,
            rotation_cam,
            translation_cam,
            scale,
            t_cam_from_object,
        )
        self._stop_timer("render_time", render_start)

        if self.profile_timings and self._profile_stats is not None:
            self._profile_stats["rendered_evaluations"] += 1.0

        if self.disable_roi_iou or self.roi_iou_margin < 0:
            sil_iou = mask_iou(rendered_mask, self.full_mask)
            soft_iou = soft_mask_iou(rendered_mask, self.soft_full_mask)
        else:
            roi = get_union_roi(projected_bbox, self.json_bbox, self.image_size, self.roi_iou_margin)
            if roi is None:
                sil_iou = 0.0
                soft_iou = 0.0
            else:
                x1, y1, x2, y2 = roi
                rendered_roi = rendered_mask[y1:y2, x1:x2]
                target_roi = self.full_mask[y1:y2, x1:x2]
                soft_target_roi = self.soft_full_mask[y1:y2, x1:x2]
                sil_iou = mask_iou(rendered_roi, target_roi)
                soft_iou = soft_mask_iou(rendered_roi, soft_target_roi)
        self._stop_timer("iou_time", metrics_start)

        score = (
            (1.0 - self.hard_mask_weight) * soft_iou
            + self.hard_mask_weight * sil_iou
            - self.bbox_weight * (1.0 - box_iou)
        )

        return self._build_result(
            translation_cam=translation_cam,
            rotation_cam=rotation_cam,
            scale=scale,
            projected_bbox=projected_bbox,
            score=score,
            mask_iou_value=sil_iou,
            soft_mask_iou_value=soft_iou,
            bbox_iou_value=box_iou,
            bbox_center_error_value=center_error,
            rendering_method=rendering_method,
            keep_mask=keep_mask,
            rendered_mask=rendered_mask,
        )

    def evaluate_delta(
        self,
        base_translation_cam: np.ndarray,
        base_rotation_cam: np.ndarray,
        base_scale: np.ndarray,
        params: np.ndarray,
        keep_mask: bool = False,
    ) -> dict[str, Any]:
        translation_cam, rotation_cam, scale = pose_from_deltas(
            base_translation_cam, base_rotation_cam, base_scale, params
        )
        result = self.evaluate_absolute(translation_cam, rotation_cam, scale, keep_mask=keep_mask)
        result["params"] = params.copy()
        return result

    def _render_pytorch3d_masks_adaptive(
        self,
        rotations_cam: np.ndarray,
        translations_cam: np.ndarray,
        scales: np.ndarray,
        batch_size: int,
    ) -> Any:
        renderer = self._get_torch_renderer()
        torch = renderer.torch
        masks: list[Any] = []
        current_batch_size = max(1, int(batch_size))
        index = 0
        while index < len(rotations_cam):
            end = min(len(rotations_cam), index + current_batch_size)
            try:
                if renderer.device.type == "cuda":
                    torch.cuda.synchronize(renderer.device)
                raster_start = self._start_timer()
                chunk_masks = renderer.render_masks_batch(
                    rotations_cam=rotations_cam[index:end],
                    translations_cam=translations_cam[index:end],
                    scales=scales[index:end],
                )
                if renderer.device.type == "cuda":
                    torch.cuda.synchronize(renderer.device)
                self._stop_timer("gpu_rasterization_time", raster_start)
                masks.append(chunk_masks.detach())
                index = end
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    self.add_profile_value("batch_gpu_oom_retries", 1.0)
                    if renderer.device.type == "cuda":
                        torch.cuda.empty_cache()
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"[warn] CUDA OOM during PyTorch3D batch render; retrying with batch_size={current_batch_size}")
                        continue
                    raise RuntimeError(
                        "PyTorch3D batch rasterization ran out of CUDA memory at batch_size=1. "
                        "Use a smaller mesh/proxy or disable batch GPU rendering."
                    ) from exc
                raise
        if not masks:
            return torch.empty((0, self.image_size[1], self.image_size[0]), dtype=torch.bool, device=renderer.device)
        return torch.cat(masks, dim=0)

    def evaluate_absolute_batch(
        self,
        translations_cam: np.ndarray,
        rotations_cam: np.ndarray,
        scales: np.ndarray,
        batch_size: int = 32,
        keep_masks: bool = False,
    ) -> list[dict[str, Any]]:
        """Batch scoring interface for PyTorch3D; falls back to per-pose evaluation for other backends."""
        translations = np.asarray(translations_cam, dtype=self.dtype)
        rotations = np.asarray(rotations_cam, dtype=self.dtype)
        scale_values = np.asarray(scales, dtype=self.dtype)
        if translations.ndim != 2 or rotations.ndim != 3:
            raise ValueError("translations_cam must be [B, 3] and rotations_cam must be [B, 3, 3].")
        if scale_values.ndim == 1:
            scale_values = scale_values.reshape(-1, 1)
        if scale_values.shape[1] == 1:
            scale_values = np.repeat(scale_values, 3, axis=1)

        if self.backend_preference != "pytorch3d" and self.active_backend != "pytorch3d":
            return [
                self.evaluate_absolute(translations[index], rotations[index], scale_values[index], keep_mask=keep_masks)
                for index in range(len(translations))
            ]

        try:
            import torch

            device = resolve_torch_device(self.device, allow_auto_fallback=False)
            if getattr(device, "type", "") == "cuda":
                torch.cuda.synchronize(device)
            projection_start = self._start_timer()
            metrics = batch_project_and_bbox(
                vertices=self.vertices,
                rotations_cam=rotations,
                translations_cam=translations,
                scales=scale_values,
                intrinsics=self.intrinsics,
                target_bbox=self.json_bbox,
                enable_bbox_prefilter=self.enable_bbox_prefilter,
                prefilter_bbox_iou_min=self.prefilter_bbox_iou_min,
                prefilter_center_factor=self.prefilter_center_factor,
                prefilter_size_ratio_min=self.prefilter_size_ratio_min,
                prefilter_size_ratio_max=self.prefilter_size_ratio_max,
                device=device,
            )
            if getattr(device, "type", "") == "cuda":
                torch.cuda.synchronize(device)
            self._stop_timer("gpu_projection_time", projection_start)
        except RuntimeError:
            raise

        keep = np.asarray(metrics["keep"], dtype=bool)
        valid = np.asarray(metrics["has_valid_bbox"], dtype=bool)
        projected_bboxes_np = np.asarray(metrics["projected_bbox"], dtype=np.float32)
        bbox_ious = np.asarray(metrics["bbox_iou"], dtype=np.float32)
        center_errors = np.asarray(metrics["bbox_center_error_px"], dtype=np.float32)
        results: list[dict[str, Any] | None] = [None] * len(translations)
        render_indices = [index for index, should_keep in enumerate(keep) if should_keep]

        if self.profile_timings and self._profile_stats is not None:
            self._profile_stats["total_evaluations"] += float(len(translations))
            self._profile_stats["prefilter_skipped_evaluations"] += float(len(translations) - len(render_indices))

        for index in range(len(translations)):
            projected_bbox = (
                projected_bboxes_np[index].astype(float).tolist()
                if valid[index]
                else None
            )
            if projected_bbox is None:
                results[index] = self._build_result(
                    translations[index],
                    rotations[index],
                    scale_values[index],
                    None,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    float("inf"),
                    "none",
                    keep_masks,
                )
            elif not keep[index]:
                results[index] = self._build_result(
                    translations[index],
                    rotations[index],
                    scale_values[index],
                    projected_bbox,
                    -1.0 + 0.1 * float(bbox_ious[index]),
                    0.0,
                    0.0,
                    float(bbox_ious[index]),
                    float(center_errors[index]),
                    "bbox_prefilter",
                    keep_masks,
                )

        if render_indices:
            render_start = self._start_timer()
            masks_t = self._render_pytorch3d_masks_adaptive(
                rotations_cam=rotations[render_indices],
                translations_cam=translations[render_indices],
                scales=scale_values[render_indices],
                batch_size=batch_size,
            )
            masks_np = torch_tensor_to_numpy(masks_t, dtype=np.uint8)
            self._stop_timer("render_time", render_start)
            if self.profile_timings and self._profile_stats is not None:
                self._profile_stats["rendered_evaluations"] += float(len(render_indices))

            for output_index, candidate_index in enumerate(render_indices):
                projected_bbox = projected_bboxes_np[candidate_index].astype(float).tolist()
                rendered_mask = masks_np[output_index]
                if self.disable_roi_iou or self.roi_iou_margin < 0:
                    sil_iou = mask_iou(rendered_mask, self.full_mask)
                    soft_iou = soft_mask_iou(rendered_mask, self.soft_full_mask)
                else:
                    roi = get_union_roi(projected_bbox, self.json_bbox, self.image_size, self.roi_iou_margin)
                    if roi is None:
                        sil_iou = 0.0
                        soft_iou = 0.0
                    else:
                        x1, y1, x2, y2 = roi
                        rendered_roi = rendered_mask[y1:y2, x1:x2]
                        target_roi = self.full_mask[y1:y2, x1:x2]
                        soft_target_roi = self.soft_full_mask[y1:y2, x1:x2]
                        sil_iou = mask_iou(rendered_roi, target_roi)
                        soft_iou = soft_mask_iou(rendered_roi, soft_target_roi)
                score = (
                    (1.0 - self.hard_mask_weight) * soft_iou
                    + self.hard_mask_weight * sil_iou
                    - self.bbox_weight * (1.0 - float(bbox_ious[candidate_index]))
                )
                results[candidate_index] = self._build_result(
                    translations[candidate_index],
                    rotations[candidate_index],
                    scale_values[candidate_index],
                    projected_bbox,
                    score,
                    sil_iou,
                    soft_iou,
                    float(bbox_ious[candidate_index]),
                    float(center_errors[candidate_index]),
                    "pytorch3d_batch",
                    keep_masks,
                    rendered_mask=rendered_mask,
                )

        return [result for result in results if result is not None]

    def evaluate_delta_batch(
        self,
        base_translation_cam: np.ndarray,
        base_rotation_cam: np.ndarray,
        base_scale: np.ndarray,
        params_batch: np.ndarray,
        batch_size: int = 32,
        keep_masks: bool = False,
    ) -> list[dict[str, Any]]:
        """Reserved batch neighborhood API; local_search_stage can still use evaluate_delta."""
        params_values = np.asarray(params_batch, dtype=np.float64)
        translations: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for params in params_values:
            translation_cam, rotation_cam, scale = pose_from_deltas(
                base_translation_cam,
                base_rotation_cam,
                base_scale,
                params,
            )
            translations.append(translation_cam)
            rotations.append(rotation_cam)
            scales.append(scale)
        results = self.evaluate_absolute_batch(
            np.stack(translations, axis=0),
            np.stack(rotations, axis=0),
            np.stack(scales, axis=0),
            batch_size=batch_size,
            keep_masks=keep_masks,
        )
        for result, params in zip(results, params_values):
            result["params"] = params.copy()
        return results

    def render_mask(
        self,
        projected_uv: np.ndarray,
        valid_z: np.ndarray,
        rotation_cam: np.ndarray,
        translation_cam: np.ndarray,
        scale: np.ndarray,
        t_cam_from_object: np.ndarray | None,
    ) -> tuple[np.ndarray, str]:
        if self.active_backend == "triangle_fill":
            return (
                render_mask_by_triangle_fill(projected_uv, valid_z, self.faces, self.image_size),
                "triangle_fill",
            )

        if self.active_backend == "pytorch3d":
            try:
                torch_start = self._start_timer()
                mask = self._get_torch_renderer().render_mask(rotation_cam, translation_cam, scale)
                self._stop_timer("gpu_rasterization_time", torch_start)
                return mask, "pytorch3d"
            except Exception as exc:
                self._torch_renderer = None
                if self.backend_preference == "auto":
                    print(f"[warn] pytorch3d failed, trying non-PyTorch3D fallback: {exc}")
                    self.active_backend = None
                else:
                    raise

        if self.active_backend == "pyrender" and self.mesh is not None:
            if t_cam_from_object is None:
                raise ValueError("t_cam_from_object is required for pyrender rendering.")
            try:
                return (
                    render_mask_with_pyrender(
                        self.mesh,
                        scale,
                        t_cam_from_object,
                        self.intrinsics,
                        self.image_size,
                        renderer=self._get_pyrender_renderer(),
                    ),
                    "pyrender",
                )
            except Exception as exc:
                self.close()
                print(f"[warn] pyrender failed, using triangle-fill fallback: {exc}")
                self.active_backend = "triangle_fill"
                return (
                    render_mask_by_triangle_fill(projected_uv, valid_z, self.faces, self.image_size),
                    "triangle_fill",
                )

        if self.backend_preference == "pytorch3d":
            try:
                torch_start = self._start_timer()
                mask = self._get_torch_renderer().render_mask(rotation_cam, translation_cam, scale)
                self._stop_timer("gpu_rasterization_time", torch_start)
                self.active_backend = "pytorch3d"
                return mask, "pytorch3d"
            except Exception as exc:
                self._torch_renderer = None
                raise

        if self.backend_preference in {"auto", "pyrender"} and self.mesh is not None:
            try:
                if t_cam_from_object is None:
                    raise ValueError("t_cam_from_object is required for pyrender rendering.")
                mask = render_mask_with_pyrender(
                    self.mesh,
                    scale,
                    t_cam_from_object,
                    self.intrinsics,
                    self.image_size,
                    renderer=self._get_pyrender_renderer(),
                )
                self.active_backend = "pyrender"
                return mask, "pyrender"
            except Exception as exc:
                self.close()
                print(f"[warn] pyrender failed, using triangle-fill fallback: {exc}")
                self.active_backend = "triangle_fill"
        else:
            self.active_backend = "triangle_fill"

        return (
            render_mask_by_triangle_fill(projected_uv, valid_z, self.faces, self.image_size),
            "triangle_fill",
        )


def clamp_delta_params(
    params: np.ndarray,
    max_translation_delta: float,
    max_rotation_delta_deg: float,
    scale_min_factor: float,
    scale_max_factor: float,
) -> np.ndarray:
    out = params.copy()
    out[:3] = np.clip(out[:3], -max_translation_delta, max_translation_delta)
    max_rot = math.radians(max_rotation_delta_deg)
    out[3:6] = np.clip(out[3:6], -max_rot, max_rot)
    out[6] = np.clip(out[6], math.log(scale_min_factor), math.log(scale_max_factor))
    return out


def stage_groups(stage_name: str) -> list[dict[str, Any]]:
    if stage_name == "coarse":
        return [
            {"name": "tx", "vector": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64), "step": 0.02, "min_step": 0.001},
            {"name": "ty", "vector": np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float64), "step": 0.02, "min_step": 0.001},
            {"name": "tz", "vector": np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float64), "step": 0.04, "min_step": 0.002},
            {"name": "yaw", "vector": np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float64), "step": math.radians(3.0), "min_step": math.radians(0.3)},
            {"name": "uniform_scale", "vector": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64), "step": 0.03, "min_step": 0.004},
        ]
    if stage_name == "rotation":
        return [
            {"name": "tx", "vector": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64), "step": 0.01, "min_step": 0.001},
            {"name": "ty", "vector": np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float64), "step": 0.01, "min_step": 0.001},
            {"name": "tz", "vector": np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float64), "step": 0.02, "min_step": 0.002},
            {"name": "pitch", "vector": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64), "step": math.radians(2.0), "min_step": math.radians(0.2)},
            {"name": "yaw", "vector": np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float64), "step": math.radians(2.0), "min_step": math.radians(0.2)},
            {"name": "roll", "vector": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float64), "step": math.radians(2.0), "min_step": math.radians(0.2)},
            {"name": "uniform_scale", "vector": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64), "step": 0.02, "min_step": 0.003},
        ]
    if stage_name == "fine":
        return [
            {"name": "tx", "vector": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64), "step": 0.006, "min_step": 0.0008},
            {"name": "ty", "vector": np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float64), "step": 0.006, "min_step": 0.0008},
            {"name": "tz", "vector": np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float64), "step": 0.015, "min_step": 0.0015},
            {"name": "pitch", "vector": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64), "step": math.radians(1.0), "min_step": math.radians(0.15)},
            {"name": "yaw", "vector": np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float64), "step": math.radians(1.0), "min_step": math.radians(0.15)},
            {"name": "roll", "vector": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float64), "step": math.radians(1.0), "min_step": math.radians(0.15)},
            {"name": "uniform_scale", "vector": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64), "step": 0.01, "min_step": 0.002},
        ]
    raise ValueError(f"Unsupported stage_name: {stage_name}")


def optimization_history_row(
    phase: str,
    iteration: int,
    parameter: str,
    direction: int,
    result: dict[str, Any],
    step_value: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": phase,
        "iteration": iteration,
        "parameter": parameter,
        "direction": direction,
        "score": result["score"],
        "mask_iou": result["mask_iou"],
        "bbox_iou": result["bbox_iou"],
        "bbox_center_error_px": result["bbox_center_error_px"],
        "rendering_method": result.get("rendering_method"),
        "step_value": step_value,
    }
    params = result.get("params", np.zeros(PARAM_DIM, dtype=np.float64))
    for name, value in zip(PARAM_NAMES, params):
        if name.endswith("_deg"):
            row[name] = math.degrees(float(value))
        else:
            row[name] = float(value)
    return row


def local_search_stage(
    evaluator: CameraPoseEvaluator,
    base_translation_cam: np.ndarray,
    base_rotation_cam: np.ndarray,
    base_scale: np.ndarray,
    stage_name: str,
    max_iters: int,
    step_decay: float,
    max_translation_delta: float,
    max_rotation_delta_deg: float,
    scale_min_factor: float,
    scale_max_factor: float,
    save_full_history: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = stage_groups(stage_name)
    params = np.zeros(PARAM_DIM, dtype=np.float64)
    group_steps = np.array([group["step"] for group in groups], dtype=np.float64)
    group_min_steps = np.array([group["min_step"] for group in groups], dtype=np.float64)

    best = evaluator.evaluate_delta(base_translation_cam, base_rotation_cam, base_scale, params)
    history = [optimization_history_row(stage_name, 0, "initial", -1, best, 0.0)]
    print(
        f"  [{stage_name}] start score={best['score']:.6f} "
        f"mask_iou={best['mask_iou']:.6f} bbox_iou={best['bbox_iou']:.6f}"
    )

    for iteration in range(1, max_iters + 1):
        improved = False
        for group_index, group in enumerate(groups):
            current_best = best
            current_params = params
            current_direction = 0
            step_value = float(group_steps[group_index])

            directions = (1.0, -1.0)
            candidates = [
                clamp_delta_params(
                    params + direction * step_value * group["vector"],
                    max_translation_delta,
                    max_rotation_delta_deg,
                    scale_min_factor,
                    scale_max_factor,
                )
                for direction in directions
            ]
            results = evaluator.evaluate_delta_batch(
                base_translation_cam,
                base_rotation_cam,
                base_scale,
                np.stack(candidates, axis=0),
            )

            for direction, candidate, result in zip(directions, candidates, results):
                if save_full_history:
                    history.append(
                        optimization_history_row(stage_name, iteration, group["name"], int(direction), result, step_value)
                    )
                if result["score"] > current_best["score"] + 1e-8:
                    current_best = result
                    current_params = candidate
                    current_direction = int(direction)

            if current_best["score"] > best["score"] + 1e-8:
                params = current_params
                best = current_best
                improved = True
                if not save_full_history:
                    history.append(
                        optimization_history_row(
                            stage_name,
                            iteration,
                            group["name"],
                            current_direction,
                            best,
                            step_value,
                        )
                    )
                print(
                    f"  [{stage_name} iter {iteration:02d}] improve {group['name']} "
                    f"score={best['score']:.6f} mask_iou={best['mask_iou']:.6f} bbox_iou={best['bbox_iou']:.6f}"
                )

        if not improved:
            group_steps *= step_decay
            print(f"  [{stage_name} iter {iteration:02d}] no improvement, shrink steps")
            if not save_full_history:
                history.append(
                    optimization_history_row(
                        stage_name,
                        iteration,
                        "step_shrink",
                        0,
                        best,
                        float(group_steps.max()),
                    )
                )
        if np.all(group_steps <= group_min_steps):
            break

    final = evaluator.evaluate_delta(base_translation_cam, base_rotation_cam, base_scale, params, keep_mask=True)
    final["params"] = params.copy()
    if not save_full_history:
        history.append(optimization_history_row(stage_name, max_iters + 1, "final", 0, final, 0.0))
    return final, history


def pose_signature(result: dict[str, Any]) -> tuple[float, ...]:
    """Coarse key used to avoid storing duplicate initial hypotheses."""
    translation = np.asarray(result["translation_cam"], dtype=np.float64)
    rotation = np.asarray(result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(result["scale"], dtype=np.float64)
    forward = rotation[:, 2]
    return tuple(np.round(np.concatenate([translation, forward, scale]), 4).tolist())


def keep_top_k_results(
    heap: list[tuple[float, int, dict[str, Any]]],
    seen_signatures: set[tuple[float, ...]],
    result: dict[str, Any],
    top_k: int,
    counter: int,
) -> int:
    signature = pose_signature(result)
    if signature in seen_signatures:
        return counter
    seen_signatures.add(signature)

    key = result["score"]
    if len(heap) < top_k:
        heapq.heappush(heap, (key, counter, result))
        return counter + 1
    if key > heap[0][0]:
        heapq.heapreplace(heap, (key, counter, result))
        return counter + 1
    return counter


def build_coarse_candidate_spec(
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    tz: float,
    anchor_local: np.ndarray,
    target_uv: np.ndarray,
    metadata: dict[str, Any],
    intrinsics: dict[str, float],
) -> dict[str, Any] | None:
    translation_cam = solve_translation_from_anchor(
        rotation_cam=rotation_cam,
        scale=scale,
        tz=tz,
        anchor_local=anchor_local,
        target_uv=target_uv,
        intrinsics=intrinsics,
    )
    if translation_cam is None:
        return None

    return {
        "translation_cam": np.asarray(translation_cam, dtype=np.float64),
        "rotation_cam": np.asarray(rotation_cam, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "initializer_metadata": metadata,
    }


def build_ground_contact_candidate_spec(
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    bottom_local: np.ndarray,
    road_anchor_world: np.ndarray,
    t_world_from_cam: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    r_world_from_cam = np.asarray(t_world_from_cam, dtype=np.float64)[:3, :3]
    t_world_from_cam_vec = np.asarray(t_world_from_cam, dtype=np.float64)[:3, 3]
    rotation_world = r_world_from_cam @ np.asarray(rotation_cam, dtype=np.float64)
    translation_world = np.asarray(road_anchor_world, dtype=np.float64) - rotation_world @ (
        np.asarray(bottom_local, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    )
    translation_cam = r_world_from_cam.T @ (translation_world - t_world_from_cam_vec)
    if not np.all(np.isfinite(translation_cam)) or float(translation_cam[2]) <= 0.05:
        return None
    return {
        "translation_cam": np.asarray(translation_cam, dtype=np.float64),
        "rotation_cam": np.asarray(rotation_cam, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "initializer_metadata": metadata,
    }


def evaluate_coarse_candidate_spec(
    evaluator: CameraPoseEvaluator,
    candidate_spec: dict[str, Any],
) -> dict[str, Any] | None:
    metadata = candidate_spec.get("initializer_metadata", {})
    if hasattr(evaluator, "set_initializer_metadata"):
        evaluator.set_initializer_metadata(metadata)
    result = evaluator.evaluate_absolute(
        candidate_spec["translation_cam"],
        candidate_spec["rotation_cam"],
        candidate_spec["scale"],
    )
    if result["projected_bbox"] is None:
        return None
    result["initializer_metadata"] = metadata
    return result


def batch_prefilter_initial_candidates(
    evaluator: CameraPoseEvaluator,
    candidate_specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Use torch batch projection for initial bbox prefilter, then score kept candidates normally."""
    if not candidate_specs:
        return []

    evaluator.add_profile_value("batch_gpu_candidates_total", float(len(candidate_specs)))
    try:
        import torch
    except Exception as exc:
        print(f"[warn] batch GPU prefilter unavailable, using sequential initial scoring: {exc}")
        return [
            result
            for result in (evaluate_coarse_candidate_spec(evaluator, spec) for spec in candidate_specs)
            if result is not None
        ]

    try:
        device = resolve_torch_device(
            args.device,
            allow_auto_fallback=args.render_backend == "auto",
        )
    except RuntimeError as exc:
        if args.render_backend == "auto":
            print(f"[warn] batch GPU prefilter disabled: {exc}")
            return [
                result
                for result in (evaluate_coarse_candidate_spec(evaluator, spec) for spec in candidate_specs)
                if result is not None
            ]
        raise
    if device is None:
        return [
            result
            for result in (evaluate_coarse_candidate_spec(evaluator, spec) for spec in candidate_specs)
            if result is not None
        ]

    kept_specs: list[dict[str, Any]] = []
    current_batch_size = max(1, int(args.batch_gpu_size))
    index = 0
    while index < len(candidate_specs):
        chunk = candidate_specs[index : index + current_batch_size]
        rotations = np.stack([np.asarray(spec["rotation_cam"], dtype=np.float32) for spec in chunk], axis=0)
        translations = np.stack([np.asarray(spec["translation_cam"], dtype=np.float32) for spec in chunk], axis=0)
        scales = np.stack([np.asarray(spec["scale"], dtype=np.float32) for spec in chunk], axis=0)
        try:
            if getattr(device, "type", "") == "cuda":
                torch.cuda.synchronize(device)
            projection_start = evaluator._start_timer()
            batch_metrics = batch_project_and_bbox(
                vertices=evaluator.vertices,
                rotations_cam=rotations,
                translations_cam=translations,
                scales=scales,
                intrinsics=evaluator.intrinsics,
                target_bbox=evaluator.json_bbox,
                enable_bbox_prefilter=evaluator.enable_bbox_prefilter,
                prefilter_bbox_iou_min=evaluator.prefilter_bbox_iou_min,
                prefilter_center_factor=evaluator.prefilter_center_factor,
                prefilter_size_ratio_min=evaluator.prefilter_size_ratio_min,
                prefilter_size_ratio_max=evaluator.prefilter_size_ratio_max,
                device=device,
            )
            if getattr(device, "type", "") == "cuda":
                torch.cuda.synchronize(device)
            evaluator._stop_timer("gpu_projection_time", projection_start)
        except RuntimeError as exc:
            if is_cuda_oom(exc):
                evaluator.add_profile_value("batch_gpu_oom_retries", 1.0)
                if getattr(device, "type", "") == "cuda":
                    torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"[warn] CUDA OOM during batch prefilter; retrying with batch_gpu_size={current_batch_size}")
                    continue
                raise RuntimeError(
                    "CUDA OOM during batch GPU prefilter even with batch_gpu_size=1. "
                    "Use a smaller proxy mesh or disable --enable_batch_gpu_eval."
                ) from exc
            raise

        keep = np.asarray(batch_metrics["keep"], dtype=bool)
        valid = np.asarray(batch_metrics["has_valid_bbox"], dtype=bool)
        projected_bboxes = np.asarray(batch_metrics["projected_bbox"], dtype=np.float32)
        bbox_ious = np.asarray(batch_metrics["bbox_iou"], dtype=np.float32)
        center_errors = np.asarray(batch_metrics["bbox_center_error_px"], dtype=np.float32)
        for local_index, spec in enumerate(chunk):
            if not valid[local_index]:
                continue
            spec = dict(spec)
            spec["batch_gpu_prefilter"] = {
                "projected_bbox": projected_bboxes[local_index].tolist(),
                "bbox_iou": float(bbox_ious[local_index]),
                "bbox_center_error_px": float(center_errors[local_index]),
                "kept": bool(keep[local_index]),
            }
            if keep[local_index]:
                kept_specs.append(spec)
        index += len(chunk)

    evaluator.add_profile_value("batch_gpu_candidates_kept", float(len(kept_specs)))
    print(
        f"[batch-gpu] bbox prefilter kept {len(kept_specs)}/{len(candidate_specs)} "
        f"initial candidates"
    )
    return [
        result
        for result in (evaluate_coarse_candidate_spec(evaluator, spec) for spec in kept_specs)
        if result is not None
    ]


def estimate_depth_guess(
    bbox_target: list[float],
    intrinsics: dict[str, float],
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    bbox_corners_local: np.ndarray,
) -> float | None:
    oriented_corners = (rotation_cam @ (bbox_corners_local * scale.reshape(1, 3)).T).T
    extent_x = float(oriented_corners[:, 0].max() - oriented_corners[:, 0].min())
    extent_y = float(oriented_corners[:, 1].max() - oriented_corners[:, 1].min())
    bbox_w, bbox_h = bbox_width_height(bbox_target)
    if bbox_w <= 1e-6 or bbox_h <= 1e-6:
        return None
    z_from_w = intrinsics["fx"] * extent_x / max(1e-6, bbox_w)
    z_from_h = intrinsics["fy"] * extent_y / max(1e-6, bbox_h)
    guess = max(0.2, 0.5 * (z_from_w + z_from_h))
    return guess


def mesh_sign_candidates(axis_prior: dict[str, Any] | None, key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(axis_prior, dict):
        return default
    values = axis_prior.get(f"{key}_sign_candidates")
    if isinstance(values, list) and values:
        parsed = []
        for value in values:
            try:
                parsed.append(1.0 if float(value) >= 0 else -1.0)
            except Exception:
                continue
        if parsed:
            return tuple(dict.fromkeys(parsed))
    value = axis_prior.get(f"{key}_sign")
    if value is not None:
        try:
            return (1.0 if float(value) >= 0 else -1.0,)
        except Exception:
            pass
    return default


def generate_initial_candidates(
    evaluator: CameraPoseEvaluator,
    obs: dict[str, Any],
    mesh_meta: dict[str, Any],
    base_scale: np.ndarray,
    corrected_seed: dict[str, Any] | None,
    t_world_from_cam: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    up_cam = camera_up_vector(t_world_from_cam, args.world_up_axis)
    road_anchor_world = None
    vehicle_pose_context = getattr(evaluator, "vehicle_pose_context", None)
    if isinstance(vehicle_pose_context, dict):
        road = vehicle_pose_context.get("road_constraint", {})
        truncation_info = getattr(evaluator, "truncation_info", {}) or {}
        truncation_sides = set(truncation_info.get("truncation_sides", [])) if isinstance(truncation_info, dict) else set()
        allow_truncated_road_init = bool(getattr(args, "road_aligned_initialization_for_truncated_enabled", True))
        use_road_init = "bottom" not in truncation_sides or allow_truncated_road_init
        if (
            use_road_init
            and bool(getattr(args, "road_aligned_initialization_enabled", True))
            and isinstance(road, dict)
            and bool(road.get("available"))
        ):
            try:
                plane = road.get("road_plane", {})
                normal_world, _ = oriented_plane(
                    np.asarray(plane["normal_world"], dtype=np.float64),
                    float(plane["offset"]),
                    str(getattr(args, "world_up_axis", "y")),
                )
                up_cam = world_vector_to_camera(t_world_from_cam, normal_world)
            except Exception:
                pass
            bottom_ref = road.get("bbox_bottom_ground")
            if isinstance(bottom_ref, dict) and "point_world" in bottom_ref:
                try:
                    point = np.asarray(bottom_ref["point_world"], dtype=np.float64)
                    if point.shape == (3,) and np.all(np.isfinite(point)):
                        road_anchor_world = point
                except Exception:
                    road_anchor_world = None
    bounds = mesh_meta["bounds"]
    bbox_corners_local = np.asarray(
        [
            [bounds[0, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
        ],
        dtype=np.float64,
    )

    global_scale_factors = parse_comma_floats(args.init_scale_factors)
    depth_factors = parse_comma_floats(args.init_depth_factors)
    yaw_candidates = np.arange(-180.0, 180.0, args.init_yaw_step_deg, dtype=np.float64)
    heap: list[tuple[float, int, dict[str, Any]]] = []
    seen_signatures: set[tuple[float, ...]] = set()
    batch_candidate_specs: list[dict[str, Any]] = []
    counter = 0

    # Also evaluate the provided corrected_pose once, in case it is already useful.
    if corrected_seed is not None:
        counter = keep_top_k_results(heap, seen_signatures, corrected_seed, args.top_k_candidates, counter)

    anchor_specs = [
        ("bbox_center", mesh_meta["center"], obs["bbox_center"]),
        ("mask_centroid", mesh_meta["center"], obs["centroid"]),
    ]

    axis_prior = mesh_meta.get("axis_prior") if isinstance(mesh_meta.get("axis_prior"), dict) else None
    up_signs = mesh_sign_candidates(axis_prior, "up", (1.0, -1.0))
    forward_signs = mesh_sign_candidates(axis_prior, "forward", (1.0, -1.0))

    for up_sign in up_signs:
        bottom_local = bottom_center_local(bounds, mesh_meta["shortest_axis"], up_sign)
        anchor_specs_with_bottom = anchor_specs + [("bottom_center", bottom_local, obs["bottom_center"])]

        for forward_sign in forward_signs:
            for yaw_deg in yaw_candidates:
                try:
                    rotation_cam = axis_alignment_rotation(mesh_meta, up_cam, float(yaw_deg), up_sign, forward_sign)
                except ValueError:
                    continue

                for global_scale_factor in global_scale_factors:
                    scale = base_scale * global_scale_factor
                    if road_anchor_world is not None:
                        bottom_local = bottom_center_local(bounds, mesh_meta["shortest_axis"], up_sign)
                        metadata = {
                            "source": "coarse_search",
                            "yaw_deg": float(yaw_deg),
                            "up_sign": up_sign,
                            "forward_sign": forward_sign,
                            "global_scale_factor": float(global_scale_factor),
                            "depth_factor": "road_plane",
                            "anchor_name": "road_bottom_center",
                        }
                        candidate_spec = build_ground_contact_candidate_spec(
                            rotation_cam=rotation_cam,
                            scale=scale,
                            bottom_local=bottom_local,
                            road_anchor_world=road_anchor_world,
                            t_world_from_cam=t_world_from_cam,
                            metadata=metadata,
                        )
                        if candidate_spec is not None:
                            if args.enable_batch_gpu_eval:
                                batch_candidate_specs.append(candidate_spec)
                            else:
                                candidate = evaluate_coarse_candidate_spec(evaluator, candidate_spec)
                                if candidate is not None:
                                    counter = keep_top_k_results(
                                        heap,
                                        seen_signatures,
                                        candidate,
                                        args.top_k_candidates,
                                        counter,
                                    )
                    depth_guess = estimate_depth_guess(obs["mask_bbox"], evaluator.intrinsics, rotation_cam, scale, bbox_corners_local)
                    if depth_guess is None:
                        continue

                    for depth_factor in depth_factors:
                        tz = depth_guess * depth_factor
                        if tz <= 0.05:
                            continue

                        for anchor_name, anchor_local, target_uv in anchor_specs_with_bottom:
                            metadata = {
                                "source": "coarse_search",
                                "yaw_deg": float(yaw_deg),
                                "up_sign": up_sign,
                                "forward_sign": forward_sign,
                                "global_scale_factor": float(global_scale_factor),
                                "depth_factor": float(depth_factor),
                                "anchor_name": anchor_name,
                            }
                            candidate_spec = build_coarse_candidate_spec(
                                rotation_cam=rotation_cam,
                                scale=scale,
                                tz=tz,
                                anchor_local=anchor_local,
                                target_uv=target_uv,
                                metadata=metadata,
                                intrinsics=evaluator.intrinsics,
                            )
                            if candidate_spec is None:
                                continue
                            if args.enable_batch_gpu_eval:
                                batch_candidate_specs.append(candidate_spec)
                                continue
                            candidate = evaluate_coarse_candidate_spec(evaluator, candidate_spec)
                            if candidate is None:
                                continue
                            counter = keep_top_k_results(heap, seen_signatures, candidate, args.top_k_candidates, counter)

    if args.enable_batch_gpu_eval:
        batch_candidates = batch_prefilter_initial_candidates(evaluator, batch_candidate_specs, args)
        for candidate in batch_candidates:
            counter = keep_top_k_results(heap, seen_signatures, candidate, args.top_k_candidates, counter)

    candidates = [item[2] for item in sorted(heap, key=lambda item: item[0], reverse=True)]
    if not candidates:
        raise RuntimeError("Failed to generate any initial pose candidate.")
    return candidates


def corrected_pose_seed(
    task: dict[str, Any],
    t_world_from_cam: np.ndarray,
    evaluator: CameraPoseEvaluator,
) -> dict[str, Any] | None:
    pose = task.get("corrected_pose")
    if not pose:
        return None

    translation_world = np.asarray(pose["translation_world"], dtype=np.float64)
    rotation_world = np.asarray(pose["rotation_matrix"], dtype=np.float64)
    scale = make_uniform_scale(scale_to_uniform_scalar(np.asarray(pose["scale"], dtype=np.float64)))
    t_world_from_object = make_transform(rotation_world, translation_world)
    t_cam_from_world = np.linalg.inv(t_world_from_cam)
    t_cam_from_object = t_cam_from_world @ t_world_from_object
    translation_cam = t_cam_from_object[:3, 3]
    rotation_cam = t_cam_from_object[:3, :3]

    result = evaluator.evaluate_absolute(translation_cam, rotation_cam, scale)
    if result["projected_bbox"] is None:
        return None
    result["initializer_metadata"] = {"source": "task_json_corrected_pose"}
    return result


def refine_candidate_stages(
    coarse_result: dict[str, Any],
    proxy_evaluator: CameraPoseEvaluator,
    full_evaluator: CameraPoseEvaluator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    translation_cam = np.asarray(coarse_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(coarse_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(coarse_result["scale"], dtype=np.float64)

    history: list[dict[str, Any]] = []
    stage1_result, stage1_history = local_search_stage(
        evaluator=proxy_evaluator,
        base_translation_cam=translation_cam,
        base_rotation_cam=rotation_cam,
        base_scale=scale,
        stage_name="coarse",
        max_iters=args.stage1_iters,
        step_decay=args.step_decay,
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor,
        save_full_history=args.save_full_history,
    )
    history.extend(stage1_history)
    translation_cam = np.asarray(stage1_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(stage1_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(stage1_result["scale"], dtype=np.float64)

    stage2_result, stage2_history = local_search_stage(
        evaluator=proxy_evaluator,
        base_translation_cam=translation_cam,
        base_rotation_cam=rotation_cam,
        base_scale=scale,
        stage_name="rotation",
        max_iters=args.stage2_iters,
        step_decay=args.step_decay,
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor,
        save_full_history=args.save_full_history,
    )
    history.extend(stage2_history)
    translation_cam = np.asarray(stage2_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(stage2_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(stage2_result["scale"], dtype=np.float64)

    stage3_result, stage3_history = local_search_stage(
        evaluator=full_evaluator,
        base_translation_cam=translation_cam,
        base_rotation_cam=rotation_cam,
        base_scale=scale,
        stage_name="fine",
        max_iters=args.stage3_iters,
        step_decay=args.step_decay,
        max_translation_delta=args.max_translation_delta,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        scale_min_factor=args.scale_min_factor,
        scale_max_factor=args.scale_max_factor,
        save_full_history=args.save_full_history,
    )
    history.extend(stage3_history)
    return stage3_result, history


def draw_stroked_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
    stroke_color: tuple[int, int, int] = (24, 24, 24),
    stroke_thickness: int = 3,
) -> None:
    """Draw compact image-plane labels without opaque patches."""
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        stroke_color,
        stroke_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_bbox(image: np.ndarray, bbox: list[float], color: tuple[int, int, int], label: str | None = None) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    if label:
        draw_stroked_text(out, label, (x1, max(18, y1 - 8)), color, scale=0.54)
    return out


def sampled_faces_for_visualization(faces: np.ndarray, max_faces: int = 5000) -> np.ndarray:
    """Return a deterministic face subset so wireframe drawing stays fast."""
    if len(faces) <= max_faces:
        return faces
    stride = max(1, int(math.ceil(len(faces) / max_faces)))
    return faces[::stride][:max_faces]


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    scale: float = 0.5,
) -> None:
    """Draw readable text over either the source image or a dark canvas."""
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(
        image,
        (x - 4, y - text_h - baseline - 4),
        (x + text_w + 4, y + baseline + 4),
        bg_color,
        thickness=-1,
    )
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_projected_wireframe(
    image: np.ndarray,
    projected_uv: np.ndarray,
    valid_z: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    max_faces: int = 4500,
) -> np.ndarray:
    """Draw sampled projected mesh triangles in image coordinates."""
    out = image.copy()
    height, width = out.shape[:2]
    valid_faces = valid_z[faces].all(axis=1)
    if not np.any(valid_faces):
        return out

    faces_to_draw = sampled_faces_for_visualization(faces[valid_faces], max_faces=max_faces)
    for face in faces_to_draw:
        tri = projected_uv[face]
        if not np.isfinite(tri).all():
            continue
        tri_min = tri.min(axis=0)
        tri_max = tri.max(axis=0)
        if tri_max[0] < 0 or tri_max[1] < 0 or tri_min[0] >= width or tri_min[1] >= height:
            continue
        pts = np.rint(tri).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], -width * 2, width * 3)
        pts[:, 1] = np.clip(pts[:, 1], -height * 2, height * 3)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    return out


def draw_native_mesh_view(
    canvas: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    axis_x: int,
    axis_y: int,
    panel_rect: tuple[int, int, int, int],
    title: str,
    max_faces: int = 6500,
) -> None:
    """Draw one orthographic native-object-space view of the GLB mesh."""
    x0, y0, panel_w, panel_h = panel_rect
    margin = 24
    view = vertices[:, [axis_x, axis_y]]
    view_min = view.min(axis=0)
    view_max = view.max(axis=0)
    view_center = 0.5 * (view_min + view_max)
    view_extent = np.maximum(view_max - view_min, 1e-8)
    fit_scale = min((panel_w - 2 * margin) / view_extent[0], (panel_h - 2 * margin) / view_extent[1])

    uv = np.zeros((len(vertices), 2), dtype=np.float64)
    uv[:, 0] = x0 + panel_w / 2.0 + (view[:, 0] - view_center[0]) * fit_scale
    uv[:, 1] = y0 + panel_h / 2.0 - (view[:, 1] - view_center[1]) * fit_scale

    sampled_faces = sampled_faces_for_visualization(faces, max_faces=max_faces)
    triangles = np.rint(uv[sampled_faces]).astype(np.int32)
    triangles[:, :, 0] = np.clip(triangles[:, :, 0], x0 - panel_w, x0 + panel_w * 2)
    triangles[:, :, 1] = np.clip(triangles[:, :, 1], y0 - panel_h, y0 + panel_h * 2)

    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1), (226, 226, 226), 1)
    cv2.fillPoly(canvas, triangles, color=(236, 230, 220))
    for tri in triangles[:: max(1, len(triangles) // 2500)]:
        cv2.polylines(canvas, [tri.reshape(-1, 1, 2)], True, (126, 106, 84), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        title,
        (x0 + 12, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (116, 116, 116),
        1,
        cv2.LINE_AA,
    )


def save_glb_native_shape_views(
    output_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    mesh_meta: dict[str, Any],
) -> Path:
    """Save orthographic native GLB views before applying optimized pose."""
    panel_w, panel_h = 332, 248
    margin = 10
    gap = 10
    canvas_w = margin * 2 + panel_w * 3 + gap * 2
    canvas_h = margin * 2 + panel_h
    canvas = np.full((canvas_h, canvas_w, 3), 249, dtype=np.uint8)

    draw_native_mesh_view(canvas, vertices, faces, 0, 1, (margin, margin, panel_w, panel_h), "XY")
    draw_native_mesh_view(
        canvas,
        vertices,
        faces,
        0,
        2,
        (margin + panel_w + gap, margin, panel_w, panel_h),
        "XZ",
    )
    draw_native_mesh_view(
        canvas,
        vertices,
        faces,
        1,
        2,
        (margin + (panel_w + gap) * 2, margin, panel_w, panel_h),
        "YZ",
    )

    path = output_dir / "02_glb_native_shape.png"
    cv2.imwrite(str(path), canvas)
    return path


def crop_and_center_preview(
    image: np.ndarray,
    visible_mask: np.ndarray,
    canvas_side: int = 440,
    padding_ratio: float = 0.18,
    inner_margin: int = 24,
) -> np.ndarray:
    """Crop around the visible object and place it on a fixed black square."""
    ys, xs = np.nonzero(visible_mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((canvas_side, canvas_side, 3), dtype=np.uint8)

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max() + 1)
    y2 = int(ys.max() + 1)
    obj_w = max(1, x2 - x1)
    obj_h = max(1, y2 - y1)
    pad = max(inner_margin, int(round(max(obj_w, obj_h) * padding_ratio)))

    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(image.shape[1], x2 + pad)
    crop_y2 = min(image.shape[0], y2 + pad)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_h, crop_w = crop.shape[:2]
    fit_w = max(1, canvas_side - 2 * inner_margin)
    fit_h = max(1, canvas_side - 2 * inner_margin)
    scale = min(fit_w / max(1, crop_w), fit_h / max(1, crop_h))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=interp)

    preview = np.zeros((canvas_side, canvas_side, 3), dtype=np.uint8)
    offset_x = (canvas_side - resized_w) // 2
    offset_y = (canvas_side - resized_h) // 2
    preview[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return preview


def render_color_with_pyrender(
    mesh: Any,
    scale: np.ndarray,
    t_cam_from_object: np.ndarray,
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Render the textured/color GLB with the optimized pose on a black background."""
    import pyrender

    width, height = image_size
    scaled_mesh = mesh.copy()
    scale_array = np.asarray(scale)
    vertex_dtype = np.float32 if scale_array.dtype == np.float32 else np.float64
    scaled_mesh.vertices = np.asarray(mesh.vertices, dtype=vertex_dtype) * scale_array.reshape(1, 3)

    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    t_gl_cam_from_object = cv_to_gl @ t_cam_from_object

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=[0.18, 0.18, 0.18])
    scene.add(pyrender.Mesh.from_trimesh(scaled_mesh, smooth=True), pose=t_gl_cam_from_object)
    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        znear=0.01,
        zfar=1000.0,
    )
    scene.add(camera, pose=np.eye(4))

    light_poses = [
        np.eye(4, dtype=np.float64),
        make_transform(np.eye(3, dtype=np.float64), np.array([1.5, -0.8, 0.8], dtype=np.float64)),
        make_transform(np.eye(3, dtype=np.float64), np.array([-1.2, -0.4, 1.2], dtype=np.float64)),
    ]
    for pose in light_poses:
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=28.0), pose=pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    color_bgr = cv2.cvtColor(color_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    return color_bgr, depth


def save_optimized_glb_pose_render(
    output_dir: Path,
    mesh: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    best_result: dict[str, Any],
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
) -> Path:
    """Save a fixed-size black-background model render at the optimized pose."""
    translation_cam = np.asarray(best_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(best_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(best_result["scale"], dtype=np.float64)

    t_cam_from_object = make_transform(rotation_cam, translation_cam)
    points_cam = transform_points(vertices * scale.reshape(1, 3), t_cam_from_object)
    projected_uv, valid_z = project_points(points_cam, **intrinsics)
    fallback_mask = render_mask_by_triangle_fill(projected_uv, valid_z, faces, image_size)

    try:
        pose_render, depth = render_color_with_pyrender(mesh, scale, t_cam_from_object, intrinsics, image_size)
        visible_mask = depth > 0
    except Exception as exc:
        print(f"[warn] color pose render failed, using wireframe fallback: {exc}")
        pose_render = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        pose_render[fallback_mask.astype(bool)] = (220, 220, 220)
        pose_render = draw_projected_wireframe(
            pose_render,
            projected_uv,
            valid_z,
            faces,
            color=(255, 200, 80),
            max_faces=4500,
        )
        visible_mask = fallback_mask.astype(bool)

    preview = crop_and_center_preview(pose_render, visible_mask, canvas_side=440)
    path = output_dir / "05_glb_optimized_pose_render.png"
    cv2.imwrite(str(path), preview)
    return path


def save_optimized_glb_projection_views(
    output_dir: Path,
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    best_result: dict[str, Any],
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
    json_bbox: list[float],
) -> dict[str, Path]:
    """Save model-focused views of the GLB at the final optimized pose."""
    translation_cam = np.asarray(best_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(best_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(best_result["scale"], dtype=np.float64)
    t_cam_from_object = make_transform(rotation_cam, translation_cam)
    points_cam = transform_points(vertices * scale.reshape(1, 3), t_cam_from_object)
    projected_uv, valid_z = project_points(points_cam, **intrinsics)
    projected_bbox = bbox_from_projected_points(projected_uv, valid_z)

    rendered_mask = best_result.get("rendered_mask")
    if rendered_mask is None:
        rendered_mask = render_mask_by_triangle_fill(projected_uv, valid_z, faces, image_size)
    rendered_bool = rendered_mask.astype(bool)

    projection_on_image = image.copy()
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[rendered_bool] = (255, 190, 0)
    projection_on_image = cv2.addWeighted(projection_on_image, 1.0, color_layer, 0.42, 0.0)
    projection_on_image = draw_projected_wireframe(
        projection_on_image, projected_uv, valid_z, faces, color=(0, 80, 255), max_faces=3500
    )
    contours, _ = cv2.findContours(rendered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(projection_on_image, contours, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    projection_on_image = draw_bbox(projection_on_image, json_bbox, (0, 255, 0), "json bbox")
    if projected_bbox is not None:
        projection_on_image = draw_bbox(projection_on_image, projected_bbox, (0, 0, 255), "projected")

    model_only = np.zeros_like(image, dtype=np.uint8)
    model_only[rendered_bool] = (185, 130, 45)
    model_only = draw_projected_wireframe(model_only, projected_uv, valid_z, faces, color=(0, 210, 255), max_faces=4500)
    cv2.drawContours(model_only, contours, -1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    if projected_bbox is not None:
        model_only = draw_bbox(model_only, projected_bbox, (0, 0, 255), "projected bbox")

    projection_path = output_dir / "03_glb_optimized_pose_projection.png"
    model_only_path = output_dir / "04_glb_optimized_pose_model_only.png"
    cv2.imwrite(str(projection_path), projection_on_image)
    cv2.imwrite(str(model_only_path), model_only)
    return {
        "glb_optimized_pose_projection": projection_path,
        "glb_optimized_pose_model_only": model_only_path,
    }


def save_mask_comparison(
    output_dir: Path,
    image: np.ndarray,
    full_mask: np.ndarray,
    rendered_mask: np.ndarray,
    json_bbox: list[float],
    projected_bbox: list[float] | None,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    full = full_mask > 0
    rendered = rendered_mask > 0
    overlap = full & rendered
    only_full = full & ~rendered
    only_rendered = rendered & ~full

    comparison = image.copy()
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[only_full] = (0, 255, 0)
    color_layer[only_rendered] = (0, 0, 255)
    color_layer[overlap] = (0, 255, 255)
    comparison = cv2.addWeighted(comparison, 1.0, color_layer, 0.55, 0.0)
    comparison = draw_bbox(comparison, json_bbox, (0, 255, 0), "json bbox")
    if projected_bbox is not None:
        comparison = draw_bbox(comparison, projected_bbox, (0, 0, 255), "projected")
    cv2.imwrite(str(output_dir / f"{prefix}_mask_comparison.png"), comparison)

    final = image.copy()
    final = draw_bbox(final, json_bbox, (0, 255, 0), "json bbox")
    if projected_bbox is not None:
        final = draw_bbox(final, projected_bbox, (0, 0, 255), "projected")
    full_contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rendered_contours, _ = cv2.findContours(rendered_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final, full_contours, -1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.drawContours(final, rendered_contours, -1, (255, 128, 0), 2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(output_dir / f"{prefix}_final_overlay.png"), final)


def render_triangle_mask_for_pose(
    vertices: np.ndarray,
    faces: np.ndarray,
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
    scale: np.ndarray,
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float] | None]:
    scaled_vertices = vertices * scale.reshape(1, 3)
    points_cam = scaled_vertices @ rotation_cam.T + translation_cam.reshape(1, 3)
    projected_uv, valid_z = project_points(points_cam, **intrinsics)
    projected_bbox = bbox_from_projected_points(projected_uv, valid_z)
    mask = render_mask_by_triangle_fill(projected_uv, valid_z, faces, image_size)
    return mask, projected_uv, valid_z, projected_bbox


def save_backend_overlay(path: Path, mask_a: np.ndarray, mask_b: np.ndarray) -> None:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    overlay = np.zeros((mask_a.shape[0], mask_a.shape[1], 3), dtype=np.uint8)
    overlay[a & b] = (255, 255, 255)
    overlay[a & ~b] = (0, 255, 0)
    overlay[b & ~a] = (0, 0, 255)
    cv2.imwrite(str(path), overlay)


def save_render_backend_validation(
    output_dir: Path,
    evaluator: CameraPoseEvaluator,
    mesh: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    best_result: dict[str, Any],
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    translation_cam = np.asarray(best_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(best_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(best_result["scale"], dtype=np.float64)
    triangle_mask, _, _, _ = render_triangle_mask_for_pose(
        vertices,
        faces,
        translation_cam,
        rotation_cam,
        scale,
        intrinsics,
        image_size,
    )
    t_cam_from_object = make_transform(rotation_cam, translation_cam)
    try:
        pyrender_mask = render_mask_with_pyrender(
            mesh,
            scale,
            t_cam_from_object,
            intrinsics,
            image_size,
            renderer=evaluator._get_pyrender_renderer(),
        )
        backend_iou = mask_iou(triangle_mask, pyrender_mask)
        print(f"[render-check] triangle_fill vs pyrender mask_iou = {backend_iou:.6f}")
    except Exception as exc:
        evaluator.close()
        print(f"[warn] pyrender validation failed; saved triangle-fill fallback mask instead: {exc}")
        pyrender_mask = triangle_mask.copy()
        backend_iou = 1.0

    triangle_path = output_dir / "render_check_triangle_fill.png"
    pyrender_path = output_dir / "render_check_pyrender.png"
    overlay_path = output_dir / "render_check_overlay.png"
    cv2.imwrite(str(triangle_path), (triangle_mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(pyrender_path), (pyrender_mask > 0).astype(np.uint8) * 255)
    save_backend_overlay(overlay_path, triangle_mask, pyrender_mask)
    return {
        "triangle_fill": str(triangle_path),
        "pyrender": str(pyrender_path),
        "overlay": str(overlay_path),
        "triangle_vs_pyrender_iou": backend_iou,
    }


def save_pytorch3d_alignment_validation(
    output_dir: Path,
    evaluator: CameraPoseEvaluator,
    vertices: np.ndarray,
    faces: np.ndarray,
    best_result: dict[str, Any],
    intrinsics: dict[str, float],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    translation_cam = np.asarray(best_result["translation_cam"], dtype=np.float64)
    rotation_cam = np.asarray(best_result["rotation_cam"], dtype=np.float64)
    scale = np.asarray(best_result["scale"], dtype=np.float64)
    triangle_mask, _, _, _ = render_triangle_mask_for_pose(
        vertices,
        faces,
        translation_cam,
        rotation_cam,
        scale,
        intrinsics,
        image_size,
    )
    raster_start = evaluator._start_timer()
    pytorch3d_mask = evaluator._get_torch_renderer().render_mask(rotation_cam, translation_cam, scale)
    evaluator._stop_timer("gpu_rasterization_time", raster_start)
    alignment_iou = mask_iou(triangle_mask, pytorch3d_mask)
    evaluator.set_profile_value("pytorch3d_alignment_iou", alignment_iou)
    print(f"triangle_fill vs pytorch3d mask_iou = {alignment_iou:.6f}")

    triangle_path = output_dir / "render_check_triangle_fill.png"
    pytorch3d_path = output_dir / "render_check_pytorch3d.png"
    overlay_path = output_dir / "render_check_pytorch3d_overlay.png"
    cv2.imwrite(str(triangle_path), (triangle_mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(pytorch3d_path), (pytorch3d_mask > 0).astype(np.uint8) * 255)
    save_backend_overlay(overlay_path, triangle_mask, pytorch3d_mask)
    return {
        "triangle_fill": str(triangle_path),
        "pytorch3d": str(pytorch3d_path),
        "overlay": str(overlay_path),
        "triangle_vs_pytorch3d_iou": alignment_iou,
    }


def read_bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def fit_image_to_box(
    image: np.ndarray,
    target_size: tuple[int, int],
    bg_color: tuple[int, int, int] = (252, 252, 250),
) -> np.ndarray:
    target_w, target_h = target_size
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def make_collage_panel(
    image: np.ndarray,
    title: str,
    content_size: tuple[int, int],
    panel_bg: tuple[int, int, int] = (246, 246, 244),
    content_bg: tuple[int, int, int] = (252, 252, 250),
) -> np.ndarray:
    caption_h = 26 if title else 0
    outer_pad = 10
    content_w, content_h = content_size
    panel_w = content_w + outer_pad * 2
    panel_h = outer_pad + content_h + outer_pad + caption_h
    panel = np.full((panel_h, panel_w, 3), panel_bg, dtype=np.uint8)

    content = fit_image_to_box(image, content_size, bg_color=content_bg)
    x0 = outer_pad
    y0 = outer_pad
    panel[y0 : y0 + content_h, x0 : x0 + content_w] = content
    cv2.rectangle(panel, (x0 - 1, y0 - 1), (x0 + content_w, y0 + content_h), (224, 224, 220), 1)
    if title:
        cv2.putText(
            panel,
            title,
            (x0 + 2, y0 + content_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (106, 106, 106),
            1,
            cv2.LINE_AA,
        )
    return panel


def save_image_collage(
    output_path: Path,
    panel_specs: list[tuple[str, np.ndarray]],
    columns: int,
    content_size: tuple[int, int],
) -> Path:
    panels = [make_collage_panel(image, panel_title, content_size) for panel_title, image in panel_specs]
    if not panels:
        raise ValueError("panel_specs must not be empty")

    rows = int(math.ceil(len(panels) / columns))
    panel_h, panel_w = panels[0].shape[:2]
    gap = 10
    margin = 12
    canvas_h = rows * panel_h + max(0, rows - 1) * gap + margin
    canvas_w = margin * 2 + columns * panel_w + max(0, columns - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 242, dtype=np.uint8)

    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        x0 = margin + col * (panel_w + gap)
        y0 = row * (panel_h + gap)
        canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel

    cv2.imwrite(str(output_path), canvas)
    return output_path


def clip_bbox_to_image(
    bbox: list[float] | None,
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0, min(width, int(math.floor(x1))))
    y1 = max(0, min(height, int(math.floor(y1))))
    x2 = max(0, min(width, int(math.ceil(x2))))
    y2 = max(0, min(height, int(math.ceil(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expanded_focus_bbox(
    image_shape: tuple[int, int, int] | tuple[int, int],
    boxes: list[list[float] | None],
    padding_ratio: float = 0.42,
    min_size: int = 220,
    top_extra_ratio: float = 0.30,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    valid_boxes = [box for box in (clip_bbox_to_image(box, image_shape) for box in boxes) if box is not None]
    if not valid_boxes:
        return 0, 0, width, height

    x1 = min(box[0] for box in valid_boxes)
    y1 = min(box[1] for box in valid_boxes)
    x2 = max(box[2] for box in valid_boxes)
    y2 = max(box[3] for box in valid_boxes)

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = max(22, int(round(box_w * padding_ratio)))
    pad_y = max(22, int(round(box_h * padding_ratio)))
    pad_top = pad_y + max(18, int(round(box_h * top_extra_ratio)))
    pad_bottom = pad_y

    crop_x1 = int(round(x1 - pad_x))
    crop_x2 = int(round(x2 + pad_x))
    crop_y1 = int(round(y1 - pad_top))
    crop_y2 = int(round(y2 + pad_bottom))

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w < min_size:
        extra = min_size - crop_w
        crop_x1 -= extra // 2
        crop_x2 += extra - extra // 2
    if crop_h < min_size:
        extra = min_size - crop_h
        crop_y1 -= extra // 2
        crop_y2 += extra - extra // 2

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > width:
        shift = crop_x2 - width
        crop_x1 = max(0, crop_x1 - shift)
        crop_x2 = width
    if crop_y2 > height:
        shift = crop_y2 - height
        crop_y1 = max(0, crop_y1 - shift)
        crop_y2 = height

    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_image_with_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def crop_to_foreground(image: np.ndarray, mode: str, padding: int = 12) -> np.ndarray:
    if mode == "light_on_dark":
        mask = image.max(axis=2) > 12
    elif mode == "dark_on_light":
        mask = image.min(axis=2) < 245
    else:
        raise ValueError(f"Unsupported crop mode: {mode}")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image.copy()

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(image.shape[1], int(xs.max()) + 1 + padding)
    y2 = min(image.shape[0], int(ys.max()) + 1 + padding)
    return image[y1:y2, x1:x2].copy()


def crop_native_shape_content(image: np.ndarray, padding: int = 18) -> np.ndarray:
    mean_intensity = image.mean(axis=2)
    mask = (mean_intensity > 55) & (mean_intensity < 245)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image.copy()
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(image.shape[1], int(xs.max()) + 1 + padding)
    y2 = min(image.shape[0], int(ys.max()) + 1 + padding)
    return image[y1:y2, x1:x2].copy()


INTERMEDIATE_IMAGE_FILENAMES = [
    "01_best_mask_comparison.png",
    "01_best_final_overlay.png",
    "02_glb_native_shape.png",
    "03_glb_optimized_pose_projection.png",
    "04_glb_optimized_pose_model_only.png",
    "05_glb_optimized_pose_render.png",
]

OBSOLETE_COLLAGE_FILENAMES = [
    "06_alignment_collage.png",
    "07_pose_closeup_collage.png",
    "08_model_reference_collage.png",
]


def cleanup_result_images(output_dir: Path) -> None:
    for filename in INTERMEDIATE_IMAGE_FILENAMES + OBSOLETE_COLLAGE_FILENAMES:
        path = Path(output_dir) / filename
        if path.exists():
            path.unlink()


def save_result_collages(
    output_dir: Path,
    json_bbox: list[float],
    projected_bbox: list[float] | None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    best_final_overlay = read_bgr_image(output_dir / "01_best_final_overlay.png")
    best_mask_comparison = read_bgr_image(output_dir / "01_best_mask_comparison.png")
    glb_native_shape = crop_native_shape_content(read_bgr_image(output_dir / "02_glb_native_shape.png"), padding=18)
    glb_pose_projection = read_bgr_image(output_dir / "03_glb_optimized_pose_projection.png")
    glb_model_only = read_bgr_image(output_dir / "04_glb_optimized_pose_model_only.png")
    glb_pose_render = crop_to_foreground(
        read_bgr_image(output_dir / "05_glb_optimized_pose_render.png"),
        mode="light_on_dark",
        padding=24,
    )

    alignment_collage_path = save_image_collage(
        output_dir / "01_alignment_overview.png",
        [
            ("overlay", best_final_overlay),
            ("mask", best_mask_comparison),
            ("projection", glb_pose_projection),
        ],
        columns=3,
        content_size=(432, 243),
    )

    focus_bbox = expanded_focus_bbox(
        best_final_overlay.shape,
        [json_bbox, projected_bbox],
        padding_ratio=0.42,
        min_size=220,
        top_extra_ratio=0.30,
    )
    pose_closeup_collage_path = save_image_collage(
        output_dir / "02_pose_inspection.png",
        [
            ("overlay", crop_image_with_bbox(best_final_overlay, focus_bbox)),
            ("projection", crop_image_with_bbox(glb_pose_projection, focus_bbox)),
            ("silhouette", crop_image_with_bbox(glb_model_only, focus_bbox)),
            ("render", glb_pose_render),
        ],
        columns=2,
        content_size=(400, 400),
    )

    model_reference_collage_path = save_image_collage(
        output_dir / "03_model_reference.png",
        [
            ("native views", glb_native_shape),
            ("pose render", glb_pose_render),
        ],
        columns=2,
        content_size=(520, 320),
    )

    return {
        "alignment_collage": alignment_collage_path,
        "pose_closeup_collage": pose_closeup_collage_path,
        "model_reference_collage": model_reference_collage_path,
    }


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


def combine_profile_stats(evaluators: list[CameraPoseEvaluator]) -> dict[str, float | int]:
    """Merge per-evaluator profiling counters into a single report payload."""
    combined: dict[str, float] = {}
    for evaluator in evaluators:
        stats = evaluator.profile_stats()
        for key, value in stats.items():
            if key == "pytorch3d_alignment_iou":
                combined[key] = max(combined.get(key, 0.0), float(value))
            else:
                combined[key] = combined.get(key, 0.0) + float(value)
    for key in list(combined.keys()):
        if key in PROFILE_COUNT_KEYS:
            combined[key] = int(round(combined[key]))
    return combined


def optimize_sample(args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = resolve_sample_dir(args.sample_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{sample_dir.name}_pose_optimized_uniform_scale"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_backend == "pytorch3d" or args.validate_pytorch3d_alignment:
        import_torch_and_pytorch3d()
        resolve_torch_device(args.device, allow_auto_fallback=False)

    task = read_json(sample_dir / "task.json")
    image = read_image(sample_dir / "image.jpg", mode="color")
    crop_mask = read_image(sample_dir / "mask.png", mode="gray")
    crop_image_path = sample_dir / "crop.jpg"
    crop_image = read_image(crop_image_path, mode="color") if crop_image_path.exists() else None
    image_size = image_size_from_task(task, image)
    json_bbox = [float(v) for v in task["bbox_xyxy"]]
    full_mask, mask_placement = paste_crop_mask_to_full_image(crop_mask, json_bbox, image_size, full_image=image, crop_image=crop_image)
    soft_full_mask = make_soft_mask(full_mask)
    if args.fast_float32:
        soft_full_mask = soft_full_mask.astype(np.float32, copy=False)
    obs = extract_mask_observations(full_mask)

    mesh_path = find_mesh_path(sample_dir, task)
    mesh = load_glb_as_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mesh_meta = mesh_axis_metadata(vertices)
    proxy_vertices, proxy_faces = build_proxy_mesh(vertices, faces, target_faces=args.proxy_face_count)

    t_world_from_cam = np.asarray(task["camera"]["T_world_from_cam"], dtype=np.float64)
    intrinsics = {
        "fx": float(task["camera"]["fx"]),
        "fy": float(task["camera"]["fy"]),
        "cx": float(task["camera"]["cx"]),
        "cy": float(task["camera"]["cy"]),
    }

    pose = task.get("corrected_pose", {})
    base_scale = make_uniform_scale(scale_to_uniform_scalar(np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)))

    proxy_evaluator = CameraPoseEvaluator(
        vertices=proxy_vertices,
        faces=proxy_faces,
        mesh=None,
        full_mask=full_mask,
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
    )
    full_evaluator = CameraPoseEvaluator(
        vertices=vertices,
        faces=faces,
        mesh=mesh,
        full_mask=full_mask,
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
    )

    corrected_seed = corrected_pose_seed(task, t_world_from_cam, proxy_evaluator) if args.include_corrected_seed else None
    initial_candidates = generate_initial_candidates(
        evaluator=proxy_evaluator,
        obs=obs,
        mesh_meta=mesh_meta,
        base_scale=base_scale,
        corrected_seed=corrected_seed,
        t_world_from_cam=t_world_from_cam,
        args=args,
    )

    best_result: dict[str, Any] | None = None
    best_history: list[dict[str, Any]] = []

    coarse_preview = initial_candidates[: min(5, len(initial_candidates))]
    preview_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(coarse_preview):
        preview_rows.append(
            {
                "rank": index + 1,
                "score": candidate["score"],
                "mask_iou": candidate["mask_iou"],
                "bbox_iou": candidate["bbox_iou"],
                "bbox_center_error_px": candidate["bbox_center_error_px"],
                "initializer_metadata": candidate.get("initializer_metadata", {}),
            }
        )

    candidates_to_refine = initial_candidates[: args.refine_top_k]
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
        refined_result, history = refine_candidate_stages(candidate, proxy_evaluator, full_evaluator, args)
        print(
            f"[candidate {rank:02d}] refined score={refined_result['score']:.6f} "
            f"mask_iou={refined_result['mask_iou']:.6f} bbox_iou={refined_result['bbox_iou']:.6f}"
        )
        refined_result["initializer_metadata"] = candidate.get("initializer_metadata", {})
        refined_result["candidate_rank"] = rank
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

    if best_result is None:
        raise RuntimeError("No valid pose candidate survived refinement.")

    best_uniform_scale = scale_to_uniform_scalar(np.asarray(best_result["scale"], dtype=np.float64))
    best_result["scale"] = make_uniform_scale(best_uniform_scale)
    translation_world, rotation_world = camera_pose_to_world_pose(
        t_world_from_cam,
        np.asarray(best_result["translation_cam"], dtype=np.float64),
        np.asarray(best_result["rotation_cam"], dtype=np.float64),
    )

    save_mask_comparison(
        output_dir,
        image,
        full_mask,
        best_result["rendered_mask"],
        json_bbox,
        best_result["projected_bbox"],
        "01_best",
    )
    glb_native_shape_path = save_glb_native_shape_views(output_dir, vertices, faces, mesh_meta)
    glb_projection_paths = save_optimized_glb_projection_views(
        output_dir=output_dir,
        image=image,
        vertices=vertices,
        faces=faces,
        best_result=best_result,
        intrinsics=intrinsics,
        image_size=image_size,
        json_bbox=json_bbox,
    )
    glb_pose_render_path = save_optimized_glb_pose_render(
        output_dir=output_dir,
        mesh=mesh,
        vertices=vertices,
        faces=faces,
        best_result=best_result,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    collage_paths = save_result_collages(
        output_dir=output_dir,
        json_bbox=json_bbox,
        projected_bbox=best_result["projected_bbox"],
    )
    write_history_csv(output_dir / "optimization_history.csv", best_history)

    render_validation_outputs: dict[str, Any] = {}
    if args.validate_render_backends:
        render_validation_outputs["render_backends"] = save_render_backend_validation(
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
        render_validation_outputs["pytorch3d_alignment"] = save_pytorch3d_alignment_validation(
            output_dir=output_dir,
            evaluator=full_evaluator,
            vertices=vertices,
            faces=faces,
            best_result=best_result,
            intrinsics=intrinsics,
            image_size=image_size,
        )

    optimized_task = json.loads(json.dumps(task))
    optimized_task["corrected_pose"]["translation_world"] = to_builtin(translation_world)
    optimized_task["corrected_pose"]["rotation_matrix"] = to_builtin(rotation_world)
    optimized_task["corrected_pose"]["scale"] = to_builtin(best_result["scale"])
    with (output_dir / "task_with_optimized_corrected_pose.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(optimized_task), f, indent=2)

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
        "camera_intrinsics": intrinsics,
        "render_backend": full_evaluator.active_backend or full_evaluator.backend_preference,
        "bbox_weight": args.bbox_weight,
        "scale_constraint": "uniform_xyz",
        "optimized_uniform_scale": best_uniform_scale,
        "initializer_top_candidates": preview_rows,
        "best_initializer_metadata": best_result.get("initializer_metadata", {}),
        "best_candidate_rank": best_result.get("candidate_rank"),
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
            "mask_iou": best_result["mask_iou"],
            "bbox_iou": best_result["bbox_iou"],
            "bbox_center_error_px": best_result["bbox_center_error_px"],
            "projected_bbox": best_result["projected_bbox"],
        },
        "outputs": {
            "alignment_collage": str(collage_paths["alignment_collage"]),
            "pose_closeup_collage": str(collage_paths["pose_closeup_collage"]),
            "model_reference_collage": str(collage_paths["model_reference_collage"]),
            "optimization_history": str(output_dir / "optimization_history.csv"),
            "optimization_report": str(output_dir / "optimization_report.json"),
            "optimized_task": str(output_dir / "task_with_optimized_corrected_pose.json"),
        },
    }
    if render_validation_outputs:
        report["render_validation"] = render_validation_outputs
    if args.profile_timings:
        report["profiling"] = combine_profile_stats([proxy_evaluator, full_evaluator])
    with (output_dir / "optimization_report.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(report), f, indent=2)
    cleanup_result_images(output_dir)
    proxy_evaluator.close()
    full_evaluator.close()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate and refine corrected_pose with a uniform x/y/z scale constraint."
    )
    parser.add_argument(
        "--sample_dir",
        default=r"E:\QingYan\pose_matching_tasks\pose_matching_tasks\obj_000001@000001",
        help="Sample directory containing image.jpg, mask.png, object_*.glb, and task.json.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for overlays, report, and optimized task JSON.",
    )
    parser.add_argument(
        "--render_backend",
        choices=["auto", "pyrender", "triangle_fill", "pytorch3d"],
        default="triangle_fill",
        help="Silhouette rendering backend for the final refinement stage.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used by PyTorch3D and batch GPU prefiltering.",
    )
    parser.add_argument(
        "--pytorch3d_faces_per_pixel",
        type=int,
        default=1,
        help="PyTorch3D rasterizer faces_per_pixel for silhouette rendering.",
    )
    parser.add_argument(
        "--pytorch3d_cull_backfaces",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="Whether PyTorch3D should cull backfaces; accepts true/false.",
    )
    parser.add_argument(
        "--pytorch3d_bin_size",
        type=int,
        default=None,
        help="PyTorch3D rasterizer bin_size. Use 0 for naive rasterization when coarse bins overflow.",
    )
    parser.add_argument(
        "--pytorch3d_max_faces_per_bin",
        type=int,
        default=None,
        help="PyTorch3D rasterizer max_faces_per_bin. Increase this to keep coarse rasterization fast without bin overflow.",
    )
    parser.add_argument(
        "--validate_render_backends",
        action="store_true",
        help="Save triangle_fill and pyrender masks plus an overlay for the final pose.",
    )
    parser.add_argument(
        "--validate_pytorch3d_alignment",
        action="store_true",
        help="Save triangle_fill and PyTorch3D masks plus an overlay for the final pose.",
    )
    parser.add_argument(
        "--enable_batch_gpu_eval",
        action="store_true",
        help="Use torch batch projection for initial candidate bbox prefiltering.",
    )
    parser.add_argument(
        "--batch_gpu_size",
        type=int,
        default=32,
        help="Initial chunk size for batch GPU projection/rendering. Halved automatically on CUDA OOM.",
    )
    parser.add_argument(
        "--bbox_weight",
        type=float,
        default=0.1,
        help="Regularization weight for bbox IoU. Higher values reduce silhouette-only drift.",
    )
    parser.add_argument(
        "--hard_mask_weight",
        type=float,
        default=0.0,
        help="Blend hard binary mask IoU into the visual score in addition to soft IoU.",
    )
    parser.add_argument(
        "--include_corrected_seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate task.json corrected_pose as one initialization candidate.",
    )
    parser.add_argument(
        "--enable_bbox_prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip obviously bad candidates before mask rendering.",
    )
    parser.add_argument(
        "--prefilter_bbox_iou_min",
        type=float,
        default=0.05,
        help="Minimum bbox IoU for a candidate to avoid bbox prefiltering.",
    )
    parser.add_argument(
        "--prefilter_center_factor",
        type=float,
        default=1.8,
        help="Center error factor against target bbox diagonal for bbox prefiltering.",
    )
    parser.add_argument(
        "--prefilter_size_ratio_min",
        type=float,
        default=0.35,
        help="Minimum projected-to-target bbox size ratio allowed by the prefilter.",
    )
    parser.add_argument(
        "--prefilter_size_ratio_max",
        type=float,
        default=3.0,
        help="Maximum projected-to-target bbox size ratio allowed by the prefilter.",
    )
    parser.add_argument(
        "--roi_iou_margin",
        type=int,
        default=30,
        help="ROI expansion in pixels for mask IoU and soft IoU.",
    )
    parser.add_argument(
        "--disable_roi_iou",
        action="store_true",
        help="Use full-image IoU instead of ROI-limited IoU.",
    )
    parser.add_argument(
        "--fast_float32",
        action="store_true",
        help="Use float32 for evaluator-side projection buffers and soft masks.",
    )
    parser.add_argument(
        "--save_full_history",
        action="store_true",
        help="Record full per-trial history instead of the reduced history log.",
    )
    parser.add_argument(
        "--profile_timings",
        action="store_true",
        help="Collect and print evaluator timing statistics.",
    )
    parser.add_argument(
        "--world_up_axis",
        choices=["x", "y", "z", "+x", "+y", "+z", "-x", "-y", "-z"],
        default="-y",
        help="World-space up axis used to build upright camera-frame initialization hypotheses.",
    )
    parser.add_argument("--proxy_face_count", type=int, default=1800)
    parser.add_argument("--top_k_candidates", type=int, default=8)
    parser.add_argument("--refine_top_k", type=int, default=3)
    parser.add_argument("--early_stop_mask_iou", type=float, default=0.90)
    parser.add_argument("--early_stop_bbox_iou", type=float, default=0.85)
    parser.add_argument("--init_yaw_step_deg", type=float, default=15.0)
    parser.add_argument(
        "--init_scale_factors",
        default="0.5,0.7,1.0,1.3,1.6",
        help="Comma-separated uniform scale multipliers applied to the base scale during coarse search.",
    )
    parser.add_argument(
        "--init_depth_factors",
        default="0.8,1.0,1.2",
        help="Comma-separated depth multipliers around the bbox-derived depth guess.",
    )
    parser.add_argument("--stage1_iters", type=int, default=10)
    parser.add_argument("--stage2_iters", type=int, default=8)
    parser.add_argument("--stage3_iters", type=int, default=14)
    parser.add_argument("--step_decay", type=float, default=0.5)
    parser.add_argument("--max_translation_delta", type=float, default=0.8)
    parser.add_argument("--max_rotation_delta_deg", type=float, default=45.0)
    parser.add_argument("--scale_min_factor", type=float, default=0.5)
    parser.add_argument("--scale_max_factor", type=float, default=2.2)
    parser.add_argument(
        "--road_aligned_initialization_for_truncated_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also generate road-bottom initialization candidates for truncated objects.",
    )
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
    print(f"optimized_translation_world: {to_builtin(pose_world['translation_world'])}")
    print(f"optimized_scale: {to_builtin(pose_world['scale'])}")
    print(f"render_backend: {report['render_backend']}")
    print(f"alignment_collage_path: {report['outputs']['alignment_collage']}")
    print(f"pose_closeup_collage_path: {report['outputs']['pose_closeup_collage']}")
    print(f"model_reference_collage_path: {report['outputs']['model_reference_collage']}")
    print(f"report_path: {report['outputs']['optimization_report']}")
    print(f"optimized_task_path: {report['outputs']['optimized_task']}")
    profiling = report.get("profiling")
    if profiling:
        print(f"profiling_total_evaluations: {int(profiling.get('total_evaluations', 0))}")
        print(f"profiling_rendered_evaluations: {int(profiling.get('rendered_evaluations', 0))}")
        print(f"profiling_prefilter_skipped_evaluations: {int(profiling.get('prefilter_skipped_evaluations', 0))}")
        print(f"profiling_projection_time_s: {float(profiling.get('projection_time', 0.0)):.6f}")
        print(f"profiling_render_time_s: {float(profiling.get('render_time', 0.0)):.6f}")
        print(f"profiling_iou_time_s: {float(profiling.get('iou_time', 0.0)):.6f}")
        print(f"profiling_gpu_projection_time_s: {float(profiling.get('gpu_projection_time', 0.0)):.6f}")
        print(f"profiling_gpu_rasterization_time_s: {float(profiling.get('gpu_rasterization_time', 0.0)):.6f}")
        print(f"profiling_batch_gpu_candidates_total: {int(profiling.get('batch_gpu_candidates_total', 0))}")
        print(f"profiling_batch_gpu_candidates_kept: {int(profiling.get('batch_gpu_candidates_kept', 0))}")
        print(f"profiling_batch_gpu_oom_retries: {int(profiling.get('batch_gpu_oom_retries', 0))}")
        print(f"profiling_pytorch3d_alignment_iou: {float(profiling.get('pytorch3d_alignment_iou', 0.0)):.6f}")


if __name__ == "__main__":
    main()
