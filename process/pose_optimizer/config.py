"""Small dependency-free config helpers for optimizer variants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CONFIG_METADATA_KEYS = {
    "description",
    "mode",
    "notes",
    "variant",
}

NEGATED_BOOLEAN_OPTIONS = {
    "enable_bbox_prefilter",
    "enable_grabcut",
    "appearance_enabled",
    "bbox_area_trend_front_sign_enabled",
    "depth_enabled",
    "include_corrected_seed",
    "edge_score_enabled",
    "edge_topk_only",
    "edge_use_mask_erode",
    "partial_use_one_sided_distance",
    "partial_visibility_enabled",
    "heading_prior_enabled",
    "heading_prior_lock_front_sign",
    "front_sign_hard_gate_enabled",
    "tail_light_motion_consistency_flip_enabled",
    "front_sign_depth_trend_enabled",
    "ground_contact_hard_gate_enabled",
    "mesh_tail_light_front_sign_enabled",
    "road_constraint_enabled",
    "road_aligned_initialization_enabled",
    "road_aligned_initialization_for_truncated_enabled",
    "road_depth_fallback_enabled",
    "road_snap_candidate_enabled",
    "upright_hard_gate_enabled",
    "vehicle_mesh_axis_override_enabled",
    "visual_gate_enabled",
    "temporal_enabled",
    "temporal_anchor_visual_gate_enabled",
    "temporal_seed_enabled",
    "trusted_anchor_gate_enabled",
    "track_scale_prior_enabled",
    "truncated_bbox_constraint_enabled",
    "truncated_candidate_ground_gate_enabled",
    "truncated_final_visual_selection_enabled",
    "final_ground_constrained_selection_enabled",
    "non_truncated_visual_ground_rescue_enabled",
    "severe_truncation_final_gate_enabled",
    "generic_temporal_use_yaw_specific_term",
    "generic_coarse_scoring",
    "generic_lightweight_search_scoring",
    "generic_rotation_grid_enabled",
    "generic_heading_enabled",
    "save_color_soft_mask",
    "save_fg_bg_samples",
    "save_candidate_appearance_overlay",
    "save_score_breakdown",
    "support_aligned_seed_enabled",
    "depth_fallback_to_wildgs",
    "depth_gate_enabled",
    "depth_use_mask_erode",
    "support_fallback_to_wildgs",
    "support_fit_from_current_frame_depth",
    "support_exclude_object_masks",
    "support_exclude_other_instance_masks",
}


class ConfigError(ValueError):
    """Raised when a variant config cannot be loaded."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = parse_simple_yaml(text, config_path)

    if not isinstance(data, dict):
        raise ConfigError(f"Config must contain a mapping at top level: {config_path}")
    return data


NESTED_KEY_PREFIXES = {
    "depth": {
        "source": "depth_source",
        "type": "depth_type",
        "depth_type": "depth_type",
        "unit": "depth_unit",
        "depth_unit": "depth_unit",
        "fallback_to_wildgs": "depth_fallback_to_wildgs",
        "generic_depth_weight": "generic_depth_weight",
        "depth_sigma": "depth_sigma",
        "use_mask_erode": "depth_use_mask_erode",
        "depth_mask_erode_px": "depth_mask_erode_px",
        "min_valid_depth_ratio": "min_valid_depth_ratio",
        "depth_error_mode": "depth_error_mode",
    },
    "depth_gate": {
        "enabled": "depth_gate_enabled",
        "max_depth_error_m": "depth_gate_max_depth_error_m",
        "strict_depth_error_m": "depth_gate_strict_depth_error_m",
        "soft_penalty_start_m": "depth_gate_soft_penalty_start_m",
        "soft_penalty_weight": "depth_gate_soft_penalty_weight",
    },
    "support": {
        "source": "support_depth_source",
        "depth_source": "support_depth_source",
        "fallback_to_wildgs": "support_fallback_to_wildgs",
        "support_plane_weight": "support_plane_weight",
        "support_penalty_weight": "support_penalty_weight",
        "fit_from_current_frame_depth": "support_fit_from_current_frame_depth",
        "exclude_object_masks": "support_exclude_object_masks",
        "exclude_other_instance_masks": "support_exclude_other_instance_masks",
        "support_min_points": "support_min_points",
        "support_plane_min_points": "support_plane_min_points",
        "support_min_ransac_inlier_ratio": "support_min_ransac_inlier_ratio",
        "support_max_plane_fit_rmse": "support_max_plane_fit_rmse",
    },
}


def parse_simple_yaml(text: str, path: Path | None = None) -> dict[str, Any]:
    """Parse a deliberately small YAML subset.

    The optimizer configs are mostly flat. A tiny one-level nested form is also
    accepted for depth/support settings and flattened to the legacy CLI option
    names so older strategy entrypoints keep working.
    """

    data: dict[str, Any] = {}
    current_section: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            location = f"{path}:{line_number}" if path else f"line {line_number}"
            raise ConfigError(f"Expected 'key: value' in {location}: {raw_line!r}")

        key, value = raw_line.split(":", 1)
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key = key.strip()
        value = strip_inline_comment(value.strip())
        if not key:
            location = f"{path}:{line_number}" if path else f"line {line_number}"
            raise ConfigError(f"Empty config key in {location}")
        if indent == 0:
            current_section = key if value == "" and key in NESTED_KEY_PREFIXES else None
            if current_section is None:
                data[key] = parse_scalar(value)
            continue
        if current_section in NESTED_KEY_PREFIXES:
            flat_key = NESTED_KEY_PREFIXES[current_section].get(key, f"{current_section}_{key}")
            data[flat_key] = parse_scalar(value)
            continue
        data[key] = parse_scalar(value)
    return data


def strip_inline_comment(value: str) -> str:
    in_quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
            continue
        if char == "#" and in_quote is None:
            return value[:index].rstrip()
    return value


def parse_scalar(value: str) -> Any:
    if value == "":
        return ""

    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none"}:
        return None

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]

    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", value):
        return float(value)
    return value


def config_to_argv(config: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        if key in CONFIG_METADATA_KEYS or value is None:
            continue
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(option)
            elif key in NEGATED_BOOLEAN_OPTIONS:
                argv.append(f"--no-{key}")
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        value_text = str(value)
        if value_text.startswith("-"):
            argv.append(f"{option}={value_text}")
        else:
            argv.extend([option, value_text])
    return argv
