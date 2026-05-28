"""Image-only foreground/background appearance prior.

The prior deliberately uses the real image and detection mask only. It does not
read mesh material, texture, or vertex colors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AppearancePriorConfig:
    foreground_erode_kernel: int = 5
    foreground_min_pixels: int = 80
    background_inner_dilate_kernel: int = 9
    background_outer_dilate_kernel: int = 31
    background_bbox_expand_ratio: float = 0.15
    background_min_pixels: int = 120
    confidence_low_threshold: float = 0.40
    confidence_high_threshold: float = 1.20
    min_confidence: float = 0.30
    soft_iou_weight: float = 0.45
    precision_weight: float = 0.35
    recall_weight: float = 0.20
    leakage_weight: float = 0.25
    covariance_eps: float = 1e-3
    score_eps: float = 1e-6


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


def _odd_kernel_size(value: int) -> int:
    size = max(1, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _morph(mask: np.ndarray, op: str, kernel_size: int) -> np.ndarray:
    size = _odd_kernel_size(kernel_size)
    kernel = np.ones((size, size), dtype=np.uint8)
    if op == "erode":
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    if op == "dilate":
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    raise ValueError(f"Unsupported morphology op: {op!r}")


def _expanded_bbox_mask(
    shape: tuple[int, int],
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    expand_ratio: float,
) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    margin_x = bw * max(0.0, float(expand_ratio))
    margin_y = bh * max(0.0, float(expand_ratio))
    ix1 = max(0, int(np.floor(x1 - margin_x)))
    iy1 = max(0, int(np.floor(y1 - margin_y)))
    ix2 = min(width, int(np.ceil(x2 + margin_x)))
    iy2 = min(height, int(np.ceil(y2 + margin_y)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if ix2 > ix1 and iy2 > iy1:
        mask[iy1:iy2, ix1:ix2] = 1
    return mask


def _feature_image_lab_ab_hsv_s(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = np.stack(
        [
            lab[:, :, 1].astype(np.float32),
            lab[:, :, 2].astype(np.float32),
            hsv[:, :, 1].astype(np.float32),
        ],
        axis=-1,
    )
    return features


def _region_with_adaptive_erode(
    mask: np.ndarray,
    requested_kernel: int,
    min_pixels: int,
) -> tuple[np.ndarray, int]:
    binary = _as_binary_mask(mask)
    for kernel in range(_odd_kernel_size(requested_kernel), 0, -2):
        eroded = _morph(binary, "erode", kernel)
        if int(eroded.sum()) >= int(min_pixels) or kernel <= 1:
            return eroded.astype(bool), kernel
    return binary.astype(bool), 1


def _near_background_region(
    mask: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    config: AppearancePriorConfig,
    other_instance_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    binary = _as_binary_mask(mask)
    expanded = _expanded_bbox_mask(binary.shape, bbox_xyxy, config.background_bbox_expand_ratio).astype(bool)
    excluded = binary.astype(bool).copy()
    if other_instance_masks:
        for other in other_instance_masks:
            if other is None:
                continue
            other_mask = np.asarray(other)
            if other_mask.shape == binary.shape:
                excluded |= other_mask.astype(bool)

    attempts: list[dict[str, Any]] = []
    inner_start = _odd_kernel_size(config.background_inner_dilate_kernel)
    outer_start = _odd_kernel_size(config.background_outer_dilate_kernel)
    for outer in range(max(outer_start, inner_start + 2), inner_start, -2):
        inner = min(inner_start, outer - 2)
        outer_mask = _morph(binary, "dilate", outer).astype(bool)
        inner_mask = _morph(binary, "dilate", inner).astype(bool)
        ring = outer_mask & ~inner_mask & expanded & ~excluded
        attempts.append({"inner_kernel": inner, "outer_kernel": outer, "pixels": int(ring.sum())})
        if int(ring.sum()) >= int(config.background_min_pixels):
            return ring, {"attempts": attempts, "selected_inner_kernel": inner, "selected_outer_kernel": outer}

    # If the configured expanded bbox is too tight, keep the near-mask ring but
    # allow it to extend beyond the expanded bbox as a conservative fallback.
    outer = max(outer_start, inner_start + 2)
    inner = min(inner_start, outer - 2)
    ring = _morph(binary, "dilate", outer).astype(bool) & ~_morph(binary, "dilate", inner).astype(bool) & ~excluded
    attempts.append({"inner_kernel": inner, "outer_kernel": outer, "pixels": int(ring.sum()), "expanded_bbox": False})
    if int(ring.sum()) > 0:
        return ring, {
            "attempts": attempts,
            "selected_inner_kernel": inner,
            "selected_outer_kernel": outer,
            "used_unbounded_fallback": True,
        }

    # Last resort: pixels just outside the mask inside the expanded bbox.
    fallback = expanded & ~excluded
    attempts.append({"fallback": "expanded_bbox_without_instances", "pixels": int(fallback.sum())})
    return fallback, {"attempts": attempts, "used_expanded_bbox_fallback": True}


def _fit_gaussian(samples: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    if samples.size == 0:
        mean = np.zeros(3, dtype=np.float64)
        cov = np.eye(3, dtype=np.float64)
        return mean, cov
    samples64 = samples.astype(np.float64, copy=False)
    mean = samples64.mean(axis=0)
    if len(samples64) <= 1:
        cov = np.eye(samples64.shape[1], dtype=np.float64)
    else:
        cov = np.cov(samples64, rowvar=False)
        if cov.ndim == 0:
            cov = np.eye(samples64.shape[1], dtype=np.float64) * float(cov)
    cov = np.asarray(cov, dtype=np.float64)
    cov += np.eye(cov.shape[0], dtype=np.float64) * float(eps)
    return mean, cov


def _log_gaussian_pdf(features: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64, copy=False)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-2
        sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.pinv(cov)
    delta = flat - mean.reshape(1, -1)
    mahal = np.einsum("ij,jk,ik->i", delta, inv, delta)
    dim = mean.size
    logp = -0.5 * (mahal + logdet + dim * np.log(2.0 * np.pi))
    return logp.reshape(features.shape[:2])


def _softmax_fg_probability(log_fg: np.ndarray, log_bg: np.ndarray) -> np.ndarray:
    max_log = np.maximum(log_fg, log_bg)
    fg = np.exp(np.clip(log_fg - max_log, -80.0, 0.0))
    bg = np.exp(np.clip(log_bg - max_log, -80.0, 0.0))
    return (fg / np.maximum(fg + bg, 1e-12)).astype(np.float32)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class ImageAppearancePrior:
    """Scores rendered silhouettes against an image-derived soft color mask."""

    def __init__(
        self,
        image_bgr: np.ndarray,
        detection_mask: np.ndarray,
        bbox_xyxy: list[float] | tuple[float, float, float, float],
        *,
        config: AppearancePriorConfig | None = None,
        other_instance_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
    ) -> None:
        self.config = config or AppearancePriorConfig()
        self.image_bgr = np.asarray(image_bgr, dtype=np.uint8)
        self.detection_mask = _as_binary_mask(detection_mask)
        self.bbox_xyxy = [float(v) for v in bbox_xyxy]
        self.features = _feature_image_lab_ab_hsv_s(self.image_bgr)
        self.roi_mask = _expanded_bbox_mask(
            self.detection_mask.shape,
            self.bbox_xyxy,
            self.config.background_bbox_expand_ratio,
        ).astype(bool)

        fg_region, fg_kernel = _region_with_adaptive_erode(
            self.detection_mask,
            self.config.foreground_erode_kernel,
            self.config.foreground_min_pixels,
        )
        bg_region, bg_debug = _near_background_region(
            self.detection_mask,
            self.bbox_xyxy,
            self.config,
            other_instance_masks,
        )
        self.foreground_region = fg_region
        self.background_region = bg_region
        self.mean_fg, self.cov_fg = _fit_gaussian(self.features[fg_region], self.config.covariance_eps)
        self.mean_bg, self.cov_bg = _fit_gaussian(self.features[bg_region], self.config.covariance_eps)

        log_fg = _log_gaussian_pdf(self.features, self.mean_fg, self.cov_fg)
        log_bg = _log_gaussian_pdf(self.features, self.mean_bg, self.cov_bg)
        self.color_soft_mask = _softmax_fg_probability(log_fg, log_bg)

        fg_bg_distance = float(
            np.linalg.norm(self.mean_fg - self.mean_bg)
            / np.sqrt(np.trace(self.cov_fg) + np.trace(self.cov_bg) + self.config.score_eps)
        )
        denom = max(self.config.score_eps, self.config.confidence_high_threshold - self.config.confidence_low_threshold)
        appearance_confidence = _clamp01((fg_bg_distance - self.config.confidence_low_threshold) / denom)
        if int(fg_region.sum()) <= 0 or int(bg_region.sum()) <= 0:
            appearance_confidence = 0.0
        self.fg_bg_distance = fg_bg_distance
        self.appearance_confidence = appearance_confidence
        self.debug_info: dict[str, Any] = {
            "foreground_pixels": int(fg_region.sum()),
            "background_pixels": int(bg_region.sum()),
            "foreground_erode_kernel_used": int(fg_kernel),
            "background_region": bg_debug,
            "mean_fg": self.mean_fg.tolist(),
            "mean_bg": self.mean_bg.tolist(),
            "cov_trace_fg": float(np.trace(self.cov_fg)),
            "cov_trace_bg": float(np.trace(self.cov_bg)),
            "fg_bg_distance": self.fg_bg_distance,
            "appearance_confidence": self.appearance_confidence,
            "min_confidence": float(self.config.min_confidence),
        }

    def score_render_mask(
        self,
        render_mask: np.ndarray,
        visible_region: np.ndarray | None = None,
    ) -> dict[str, Any]:
        rendered = (np.asarray(render_mask) > 0).astype(np.float32)
        roi = self.roi_mask.copy()
        color_soft = np.where(roi, self.color_soft_mask.astype(np.float32), 0.0)
        rendered = np.where(roi, rendered, 0.0)
        if visible_region is not None:
            visible = np.asarray(visible_region).astype(bool)
            rendered = np.where(visible, rendered, 0.0)
            color_soft = np.where(visible, color_soft, 0.0)

        eps = float(self.config.score_eps)
        intersection = float((rendered * color_soft).sum())
        rendered_sum = float(rendered.sum())
        soft_sum = float(color_soft.sum())
        color_precision = 0.0 if rendered_sum <= eps else intersection / (rendered_sum + eps)
        color_recall = 0.0 if soft_sum <= eps else intersection / (soft_sum + eps)
        denom = rendered_sum + soft_sum - intersection
        color_soft_iou = 0.0 if denom <= eps else intersection / (denom + eps)
        background_leakage = 0.0 if rendered_sum <= eps else float((rendered * (1.0 - color_soft)).sum()) / (rendered_sum + eps)
        raw_score = (
            float(self.config.soft_iou_weight) * color_soft_iou
            + float(self.config.precision_weight) * color_precision
            + float(self.config.recall_weight) * color_recall
            - float(self.config.leakage_weight) * background_leakage
        )
        appearance_score = _clamp01(raw_score)
        confidence = float(self.appearance_confidence)
        if confidence < float(self.config.min_confidence):
            confidence = 0.0

        return {
            "appearance_score": appearance_score,
            "appearance_confidence": confidence,
            "color_soft_iou": float(color_soft_iou),
            "color_precision": float(color_precision),
            "color_recall": float(color_recall),
            "background_leakage": float(background_leakage),
            "fg_bg_distance": float(self.fg_bg_distance),
            "debug": dict(self.debug_info),
        }

    def save_debug_images(
        self,
        output_dir: str | Path,
        *,
        render_mask: np.ndarray | None = None,
        prefix: str = "",
    ) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        prefix_text = f"{prefix}_" if prefix else ""
        outputs: dict[str, str] = {}

        color_soft = np.clip(self.color_soft_mask * 255.0, 0, 255).astype(np.uint8)
        soft_path = output_path / f"{prefix_text}color_soft_mask.png"
        cv2.imwrite(str(soft_path), color_soft)
        outputs["color_soft_mask"] = str(soft_path)

        samples = self.image_bgr.copy()
        overlay = np.zeros_like(samples, dtype=np.uint8)
        overlay[self.background_region] = (255, 0, 0)
        overlay[self.foreground_region] = (0, 255, 0)
        sample_vis = cv2.addWeighted(samples, 1.0, overlay, 0.55, 0.0)
        fg_bg_path = output_path / f"{prefix_text}fg_bg_samples.png"
        cv2.imwrite(str(fg_bg_path), sample_vis)
        outputs["fg_bg_samples"] = str(fg_bg_path)

        if render_mask is not None:
            rendered = np.asarray(render_mask) > 0
            app_overlay = self.image_bgr.copy()
            color_layer = np.zeros_like(app_overlay, dtype=np.uint8)
            color_layer[rendered] = (0, 220, 255)
            leak = rendered & (self.color_soft_mask < 0.35)
            color_layer[leak] = (0, 0, 255)
            candidate_vis = cv2.addWeighted(app_overlay, 1.0, color_layer, 0.55, 0.0)
            candidate_path = output_path / f"{prefix_text}candidate_appearance_overlay.png"
            cv2.imwrite(str(candidate_path), candidate_vis)
            outputs["candidate_appearance_overlay"] = str(candidate_path)

        return outputs


def build_image_appearance_prior(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    *,
    config: AppearancePriorConfig | None = None,
    other_instance_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> ImageAppearancePrior:
    return ImageAppearancePrior(
        image,
        mask,
        bbox_xyxy,
        config=config,
        other_instance_masks=other_instance_masks,
    )
