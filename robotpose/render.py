from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RobotMesh:
    vertices: np.ndarray
    faces: np.ndarray
    link_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape [N, 3], got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape [M, 3], got {faces.shape}")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True)
class RenderResult:
    mask: np.ndarray
    projected_uv: np.ndarray
    valid_z: np.ndarray
    projected_bbox: list[float] | None


def normalize_camera_K(camera_K: np.ndarray | list[list[float]] | dict[str, float]) -> np.ndarray:
    if isinstance(camera_K, dict):
        return np.array(
            [
                [float(camera_K["fx"]), 0.0, float(camera_K["cx"])],
                [0.0, float(camera_K["fy"]), float(camera_K["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    K = np.asarray(camera_K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_K must be 3x3, got {K.shape}")
    return K


def scale_camera_K(camera_K: np.ndarray, scale_x: float, scale_y: float | None = None) -> np.ndarray:
    K = normalize_camera_K(camera_K).copy()
    sy = float(scale_x if scale_y is None else scale_y)
    sx = float(scale_x)
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def transform_points(points_B: np.ndarray, T_C_B: np.ndarray) -> np.ndarray:
    T = np.asarray(T_C_B, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_C_B must be 4x4, got {T.shape}")
    points = np.asarray(points_B, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3].reshape(1, 3)


def project_points(points_C: np.ndarray, camera_K: np.ndarray | list[list[float]] | dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    K = normalize_camera_K(camera_K)
    points = np.asarray(points_C, dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-8)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        x = points[valid, 0] / z[valid]
        y = points[valid, 1] / z[valid]
        uv[valid, 0] = K[0, 0] * x + K[0, 2]
        uv[valid, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def bbox_from_projected_points(
    projected_uv: np.ndarray,
    valid_z: np.ndarray,
    image_shape: tuple[int, int] | None = None,
) -> list[float] | None:
    valid = np.asarray(valid_z, dtype=bool) & np.isfinite(projected_uv).all(axis=1)
    if image_shape is not None:
        height, width = image_shape
        valid &= (
            (projected_uv[:, 0] >= 0.0)
            & (projected_uv[:, 0] < float(width))
            & (projected_uv[:, 1] >= 0.0)
            & (projected_uv[:, 1] < float(height))
        )
    pts = projected_uv[valid]
    if pts.size == 0:
        return None
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max() + 1), float(pts[:, 1].max() + 1)]


def render_mask_by_triangle_fill(
    projected_uv: np.ndarray,
    valid_z: np.ndarray,
    faces: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = [int(v) for v in image_shape]
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    faces_arr = np.asarray(faces, dtype=np.int32)
    if faces_arr.size == 0:
        return np.zeros((height, width), dtype=bool)
    valid_faces = np.asarray(valid_z, dtype=bool)[faces_arr].all(axis=1)
    for tri in projected_uv[faces_arr[valid_faces]]:
        if not np.isfinite(tri).all():
            continue
        if tri[:, 0].max() < 0 or tri[:, 1].max() < 0 or tri[:, 0].min() >= width or tri[:, 1].min() >= height:
            continue
        pts = [(float(x), float(y)) for x, y in tri]
        draw.polygon(pts, fill=1)
    return np.asarray(mask_img, dtype=np.uint8).astype(bool)


def render_robot_mask(
    mesh: RobotMesh,
    camera_K: np.ndarray | list[list[float]] | dict[str, float],
    T_C_B: np.ndarray,
    image_shape: tuple[int, int],
) -> RenderResult:
    points_C = transform_points(mesh.vertices, T_C_B)
    projected_uv, valid_z = project_points(points_C, camera_K)
    mask = render_mask_by_triangle_fill(projected_uv, valid_z, mesh.faces, image_shape)
    return RenderResult(
        mask=mask,
        projected_uv=projected_uv,
        valid_z=valid_z,
        projected_bbox=bbox_from_mask(mask),
    )


def render_color_with_pyrender(
    mesh: RobotMesh,
    camera_K: np.ndarray,
    T_C_B: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyrender  # type: ignore
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pyrender and trimesh are required for color rendering.") from exc

    height, width = image_shape
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.35, 0.35, 0.35])
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False), pose=cv_to_gl @ np.asarray(T_C_B, dtype=np.float64))
    K = normalize_camera_K(camera_K)
    scene.add(
        pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=1000.0),
        pose=np.eye(4),
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return color_rgba[:, :, :3], depth
