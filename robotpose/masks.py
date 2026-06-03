from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .losses import extract_contour


def read_rgb_image(path: str | Path) -> np.ndarray:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return np.asarray(Image.open(image_path).convert("RGB"))


def read_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask_path = Path(path).expanduser().resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if shape is not None and mask.shape != tuple(shape):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    if not mask.any():
        raise ValueError(f"Mask is empty: {mask_path}")
    return mask


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask).astype(np.uint8) * 255).save(out)


def draw_overlay(
    image_rgb: np.ndarray,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    *,
    alpha: float = 0.42,
) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    rendered = np.asarray(rendered_mask).astype(bool)
    observed = np.asarray(observed_mask).astype(bool)
    color = np.zeros_like(image)
    color[rendered] = (0, 190, 255)
    blended = image.copy()
    blended[rendered] = np.clip((1.0 - alpha) * image[rendered] + alpha * color[rendered], 0, 255).astype(np.uint8)

    pil = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil)
    _draw_contour_points(draw, extract_contour(observed, 0), fill=(255, 220, 0))
    _draw_contour_points(draw, extract_contour(rendered, 0), fill=(255, 40, 40))
    return np.asarray(pil)


def draw_mask_comparison(image_rgb: np.ndarray, observed_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    observed = np.asarray(observed_mask).astype(bool)
    rendered = np.asarray(rendered_mask).astype(bool)
    only_obs = observed & ~rendered
    only_ren = rendered & ~observed
    overlap = observed & rendered
    layer = np.zeros_like(image)
    layer[only_obs] = (0, 255, 0)
    layer[only_ren] = (255, 0, 0)
    layer[overlap] = (255, 255, 0)
    mask = observed | rendered
    out = image.copy()
    out[mask] = np.clip(0.45 * image[mask] + 0.55 * layer[mask], 0, 255).astype(np.uint8)
    return out


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(out)


def _draw_contour_points(draw: ImageDraw.ImageDraw, contour: np.ndarray, fill: tuple[int, int, int]) -> None:
    ys, xs = np.nonzero(np.asarray(contour).astype(bool))
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((int(x), int(y)), fill=fill)
