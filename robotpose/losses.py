from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LossConfig:
    weight_obs2ren_mask: float = 1.0
    weight_obs2ren_contour: float = 1.0
    weight_ren2obs_weak: float = 0.05
    weight_prior: float = 0.01
    weight_area: float = 0.001
    border_margin: int = 10
    weak_reverse_max_distance_px: float = 30.0
    robust_eps: float = 1e-3
    prior_sigma_t: float = 0.30
    prior_sigma_R_deg: float = 20.0
    area_ratio_max: float = 8.0


@dataclass(frozen=True)
class LossResult:
    total: float
    terms: dict[str, float] = field(default_factory=dict)
    weighted_terms: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def robust_mean(values: np.ndarray, eps: float = 1e-3) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(arr * arr + float(eps) ** 2).mean())


def extract_contour(mask: np.ndarray, border_margin: int = 0) -> np.ndarray:
    binary = _as_bool(mask)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    contour = ndimage.binary_dilation(binary, structure=np.ones((3, 3))) ^ ndimage.binary_erosion(
        binary,
        structure=np.ones((3, 3)),
        border_value=0,
    )
    margin = max(0, int(border_margin))
    if margin > 0:
        contour[:margin, :] = False
        contour[-margin:, :] = False
        contour[:, :margin] = False
        contour[:, -margin:] = False
    return contour.astype(bool)


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    binary = _as_bool(mask)
    if binary.any():
        return ndimage.distance_transform_edt(~binary).astype(np.float32)
    return np.full(binary.shape, float(max(binary.shape) if binary.shape else 1), dtype=np.float32)


def make_trusted_region(shape: tuple[int, int], border_margin: int) -> np.ndarray:
    trusted = np.ones(shape, dtype=bool)
    margin = max(0, int(border_margin))
    if margin > 0:
        trusted[:margin, :] = False
        trusted[-margin:, :] = False
        trusted[:, :margin] = False
        trusted[:, -margin:] = False
    return trusted


def pose_prior_loss(T_C_B: np.ndarray, T0_C_B: np.ndarray, config: LossConfig) -> float:
    T = np.asarray(T_C_B, dtype=np.float64)
    T0 = np.asarray(T0_C_B, dtype=np.float64)
    sigma_t = max(1e-9, float(config.prior_sigma_t))
    sigma_R = max(1e-9, np.deg2rad(float(config.prior_sigma_R_deg)))
    delta_t = np.linalg.norm(T[:3, 3] - T0[:3, 3])
    delta_R = Rotation.from_matrix(T0[:3, :3].T @ T[:3, :3]).magnitude()
    return float((delta_t / sigma_t) ** 2 + (delta_R / sigma_R) ** 2)


def area_loss(observed_mask: np.ndarray, rendered_mask: np.ndarray, config: LossConfig) -> float:
    obs_area = float(_as_bool(observed_mask).sum())
    ren_area = float(_as_bool(rendered_mask).sum())
    if obs_area <= 0.0 or ren_area <= 0.0:
        return 1.0
    max_ratio = max(1.0, float(config.area_ratio_max))
    too_big = max(0.0, ren_area / obs_area - max_ratio)
    too_small = max(0.0, obs_area / ren_area - max_ratio)
    return float(too_big * too_big + too_small * too_small)


def compute_partial_observation_loss(
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    T_C_B: np.ndarray,
    T0_C_B: np.ndarray,
    *,
    config: LossConfig | None = None,
) -> LossResult:
    cfg = config or LossConfig()
    obs = _as_bool(observed_mask)
    ren = _as_bool(rendered_mask)

    D_ren = distance_to_mask(ren)
    obs2ren_mask = robust_mean(D_ren[obs], cfg.robust_eps) if obs.any() else float(max(obs.shape))

    obs_contour = extract_contour(obs, cfg.border_margin)
    ren_contour = extract_contour(ren, 0)
    D_ren_contour = distance_to_mask(ren_contour)
    obs2ren_contour = (
        robust_mean(D_ren_contour[obs_contour], cfg.robust_eps)
        if obs_contour.any()
        else obs2ren_mask
    )

    trusted = make_trusted_region(obs.shape, cfg.border_margin)
    ren_trust = ren & trusted
    D_obs = distance_to_mask(obs)
    if ren_trust.any():
        ren2obs_weak = robust_mean(np.clip(D_obs[ren_trust], 0.0, cfg.weak_reverse_max_distance_px), cfg.robust_eps)
    else:
        ren2obs_weak = 0.0

    prior = pose_prior_loss(T_C_B, T0_C_B, cfg)
    area = area_loss(obs, ren, cfg)

    terms = {
        "obs2ren_mask": float(obs2ren_mask),
        "obs2ren_contour": float(obs2ren_contour),
        "ren2obs_weak": float(ren2obs_weak),
        "prior": float(prior),
        "area": float(area),
    }
    weighted = {
        "obs2ren_mask": cfg.weight_obs2ren_mask * terms["obs2ren_mask"],
        "obs2ren_contour": cfg.weight_obs2ren_contour * terms["obs2ren_contour"],
        "ren2obs_weak": cfg.weight_ren2obs_weak * terms["ren2obs_weak"],
        "prior": cfg.weight_prior * terms["prior"],
        "area": cfg.weight_area * terms["area"],
    }
    total = float(sum(weighted.values()))

    overlap = float(np.logical_and(obs, ren).sum())
    obs_area = float(obs.sum())
    ren_area = float(ren.sum())
    union = float(np.logical_or(obs, ren).sum())
    metrics = {
        "overlap_ratio": 0.0 if obs_area <= 0.0 else overlap / obs_area,
        "mask_iou": 0.0 if union <= 0.0 else overlap / union,
        "area_obs": obs_area,
        "area_ren": ren_area,
        "area_ratio": float("inf") if obs_area <= 0.0 and ren_area > 0.0 else (0.0 if obs_area <= 0.0 else ren_area / obs_area),
        "obs_contour_pixels": float(obs_contour.sum()),
        "ren_contour_pixels": float(ren_contour.sum()),
    }
    return LossResult(total=total, terms=terms, weighted_terms=weighted, metrics=metrics)
