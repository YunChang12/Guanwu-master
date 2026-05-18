from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULT_CONFIG_PATH = Path.home() / ".guanwu" / "video.config.toml"
LEGACY_MODEL_BACKEND_CONFIG_PATH = Path.home() / ".guanwu" / "video.model-backend.config.toml"


class VLMDiscoveryConfig(BaseModel):
    enabled: bool = True
    periodic_interval: int = 5
    new_track_threshold: int = 1
    disappear_threshold: int = 1
    confidence_drop_threshold: float = 0.3
    open_vocab_enabled: bool = True
    open_vocab_cooldown_frames: int = 10
    image_change_threshold: float = 0.08
    image_change_enabled: bool = True


class RuntimeConfig(BaseModel):
    occluded_ttl_frames: int = 5
    removal_ttl_frames: int = 20
    vlm_discovery: VLMDiscoveryConfig = Field(default_factory=VLMDiscoveryConfig)
    video_source: str | None = None
    session_output_root: str | None = None
    save_intermediate: bool = True
    asset_materialization: str = "copy"  # copy | move | hardlink | symlink
    background_reconstruction: bool = True
    background_sample_frames: int = 30
    rerun_enabled: bool = False


class StorageConfig(BaseModel):
    world_id: str = "scene_main"


class IsaacConfig(BaseModel):
    stage_path: str = "data/demo_scene.usd"
    auto_save: bool = True


class PITConfig(BaseModel):
    camera_provider: str = "wildgs"  # colmap | wildgs | none
    colmap_model_dir: str | None = None
    # WildGS-SLAM provider — paths to pre-computed outputs (skip re-running if set)
    wildgs_camera_poses_jsonl: str | None = None
    wildgs_static_map_dir: str | None = None
    wildgs_dynamic_prior_dir: str | None = None
    wildgs_depth_maps_dir: str | None = None
    depth_provider: str = "wildgs"  # depth_anything_v2 | zaiwu_depth_anything3 | wildgs | none
    depth_model_path: str | None = None
    frame_dump_dir: str | None = None
    use_metric_scale: bool = False
    metric_scale_factor: float = 1.0
    # World calibration
    metric_scale_source: str = "manual"  # manual | floor_plane | known_object
    known_object_size_m: float | None = None
    floor_plane_z_offset: float = 0.0
    alignment_backend: str = "depth_icp"  # depth_icp | gotrack_visual
    visual_pose_mcp_url: str | None = None
    visual_pose_mcp_tool: str = "gotrack_refine_pose"
    visual_pose_command: str | None = None
    visual_pose_timeout_sec: float = 30.0
    visual_pose_min_score: float = 0.0
    visual_pose_max_translation_step_m: float = 1.5
    visual_pose_max_rotation_step_deg: float = 60.0

    @model_validator(mode="after")
    def _validate_provider_pairing(self) -> "PITConfig":
        camera_provider = str(self.camera_provider or "").strip().lower()
        depth_provider = str(self.depth_provider or "").strip().lower()
        if camera_provider not in {"none", "colmap", "wildgs"}:
            raise ValueError("pit.camera_provider must be one of: none, colmap, wildgs")
        if depth_provider not in {"none", "depth_anything_v2", "zaiwu_depth_anything3", "wildgs"}:
            raise ValueError("pit.depth_provider must be one of: none, depth_anything_v2, zaiwu_depth_anything3, wildgs")
        if (camera_provider == "none") != (depth_provider == "none"):
            raise ValueError("pit.camera_provider and pit.depth_provider must both be 'none' together")
        if (camera_provider == "wildgs") != (depth_provider == "wildgs"):
            raise ValueError("pit.camera_provider and pit.depth_provider must both be 'wildgs' together")
        if camera_provider == "wildgs" and self.wildgs_camera_poses_jsonl and not self.wildgs_depth_maps_dir:
            raise ValueError(
                "pit.wildgs_depth_maps_dir is required when using precomputed WildGS camera poses"
            )
        return self


class PIT2IsaacConfig(BaseModel):
    mode: str = "hybrid"  # replay | physics_ready | hybrid
    output_root: str | None = None
    usd_path: str | None = None
    physics_priors_json: str | None = None
    asset_mapping_json: str | None = None
    conversion_report_json: str | None = None
    use_category_assets: bool = True
    fallback_visual: str = "primitive"
    collision_strategy: str = "primitive"  # primitive | convex_hull | convex_decomp
    min_geom_quality: float = 0.5
    output_format: str = "usdc"  # usda | usdc | usdz


class GroundedSAM2Config(BaseModel):
    step: int = 20
    iou_threshold: float = 0.8
    box_threshold: float = 0.3
    text_threshold: float = 0.25


class Seg2TrackConfig(BaseModel):
    detect_interval: int = 5
    box_threshold: float = 0.3
    text_threshold: float = 0.25


class ProviderHTTPConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8103"
    timeout_sec: float = 30.0


class ZaiwuConfig(BaseModel):
    enabled: bool = False
    gateway_url: str = "http://127.0.0.1:8181"
    request_timeout_sec: float = 30.0
    job_timeout_sec: float = 1800.0
    job_poll_interval_sec: float = 1.0
    auto_start_workers: bool = True
    worker_run_group: str = "services"
    object_detection_backend: str = Field(
        default="seg2track_sam2",
        validation_alias=AliasChoices("object_detection_backend", "perception_backend"),
    )  # "sam3" | "grounding_dino_sam2" | "seg2track_sam2"
    sam3_service: str = "services.sam3"
    grounded_sam2_service: str = "services.grounding_dino_sam2"
    seg2track_sam2_service: str = "services.seg2track_sam2"
    sam3d_service: str = "services.sam3d"
    pose_optimizer_timeout_sec: float = 1800.0
    pose_optimize_min_bbox_area_px: float = 5000.0
    depth_service: str = "services.depth_anything3"
    wildgs_slam_service: str = "services.wildgs_slam"
    gotrack_service: str = "services.gotrack"
    grounded_sam2: GroundedSAM2Config = Field(default_factory=GroundedSAM2Config)
    seg2track_sam2: Seg2TrackConfig = Field(default_factory=Seg2TrackConfig)


class VLMConfig(BaseModel):
    mode: str = "embedded"  # embedded | http | disabled
    backend: str = "api"  # embedded mode only: api | command
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3.5-sonnet"
    max_retries: int = 3
    command_template: str | None = None
    service: ProviderHTTPConfig = Field(
        default_factory=lambda: ProviderHTTPConfig(
            base_url="http://127.0.0.1:8103",
            timeout_sec=30.0,
        )
    )

class SPWMSettings(BaseModel):
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    isaac: IsaacConfig = Field(default_factory=IsaacConfig)
    pit: PITConfig = Field(default_factory=PITConfig)
    pit2isaac: PIT2IsaacConfig = Field(default_factory=PIT2IsaacConfig)
    zaiwu: ZaiwuConfig = Field(
        default_factory=ZaiwuConfig,
        validation_alias=AliasChoices("zaiwu", "model_backend"),
    )


def load_settings(
    config_path: str | Path | None = None,
    *,
    require_session_output_root: bool = False,
) -> tuple[SPWMSettings, Path]:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        settings = SPWMSettings()
        _apply_legacy_vlm_settings(settings, legacy_path=_legacy_model_backend_config_path(path))
        _validate_settings(settings, require_session_output_root=require_session_output_root)
        return settings, path

    raw = _migrate_legacy_pit_providers(_normalize_empty_values(_read_toml(path)))
    settings = SPWMSettings.model_validate(raw)
    _normalize_paths(settings, base_dir=path.parent)
    _validate_settings(settings, require_session_output_root=require_session_output_root)
    return settings, path


def save_settings(settings: SPWMSettings, config_path: str | Path | None = None, create_backup: bool = True) -> Path:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if create_backup and path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    payload = settings.model_dump()
    text = _to_toml(payload)
    path.write_text(text, encoding="utf-8")
    return path


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _normalize_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_empty_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_empty_values(item) for item in value]
    if value == "":
        return None
    return value


def _migrate_legacy_pit_providers(raw: dict[str, Any]) -> dict[str, Any]:
    pit = raw.get("pit")
    if not isinstance(pit, dict):
        return raw
    camera_provider = str(pit.get("camera_provider") or "").strip().lower()
    depth_provider = str(pit.get("depth_provider") or "").strip().lower()
    if camera_provider == "synthetic" or depth_provider == "heuristic":
        migrated = dict(raw)
        migrated_pit = dict(pit)
        migrated_pit["camera_provider"] = "none"
        migrated_pit["depth_provider"] = "none"
        migrated["pit"] = migrated_pit
        return migrated
    return raw


def _legacy_model_backend_config_path(path: Path) -> Path:
    if path == DEFAULT_CONFIG_PATH:
        return LEGACY_MODEL_BACKEND_CONFIG_PATH
    return path.with_name("video.model-backend.config.toml")


def _apply_legacy_vlm_settings(settings: SPWMSettings, *, legacy_path: Path) -> None:
    if not legacy_path.exists():
        return

    raw = _read_toml(legacy_path)
    vlm_raw = raw.get("vlm")
    if not isinstance(vlm_raw, dict):
        return

    service_raw = vlm_raw.get("service")
    service: dict[str, Any] = {}
    if isinstance(service_raw, dict):
        service = {
            "base_url": service_raw.get("base_url", settings.vlm.service.base_url),
            "timeout_sec": service_raw.get("timeout_sec", settings.vlm.service.timeout_sec),
        }

    settings.vlm = VLMConfig.model_validate(
        {
            "mode": vlm_raw.get("mode", settings.vlm.mode),
            "backend": vlm_raw.get("backend", settings.vlm.backend),
            "api_key": vlm_raw.get("api_key", settings.vlm.api_key),
            "base_url": vlm_raw.get("base_url", settings.vlm.base_url),
            "model": vlm_raw.get("model", settings.vlm.model),
            "max_retries": settings.vlm.max_retries,
            "command_template": vlm_raw.get("command_template", settings.vlm.command_template),
            "service": service or settings.vlm.service.model_dump(mode="json"),
        }
    )


def _normalize_paths(settings: SPWMSettings, base_dir: Path) -> None:
    settings.isaac.stage_path = _resolve_path(settings.isaac.stage_path, base_dir)
    if settings.pit.colmap_model_dir:
        settings.pit.colmap_model_dir = _resolve_path(settings.pit.colmap_model_dir, base_dir)
    if settings.pit.wildgs_camera_poses_jsonl:
        settings.pit.wildgs_camera_poses_jsonl = _resolve_path(settings.pit.wildgs_camera_poses_jsonl, base_dir)
    if settings.pit.wildgs_static_map_dir:
        settings.pit.wildgs_static_map_dir = _resolve_path(settings.pit.wildgs_static_map_dir, base_dir)
    if settings.pit.wildgs_dynamic_prior_dir:
        settings.pit.wildgs_dynamic_prior_dir = _resolve_path(settings.pit.wildgs_dynamic_prior_dir, base_dir)
    if settings.pit.wildgs_depth_maps_dir:
        settings.pit.wildgs_depth_maps_dir = _resolve_path(settings.pit.wildgs_depth_maps_dir, base_dir)
    if settings.pit.depth_model_path:
        settings.pit.depth_model_path = _resolve_path(settings.pit.depth_model_path, base_dir)
    if settings.pit.frame_dump_dir:
        settings.pit.frame_dump_dir = _resolve_path(settings.pit.frame_dump_dir, base_dir)
    if settings.pit2isaac.output_root:
        settings.pit2isaac.output_root = _resolve_path(settings.pit2isaac.output_root, base_dir)
    if settings.pit2isaac.usd_path:
        settings.pit2isaac.usd_path = _resolve_path(settings.pit2isaac.usd_path, base_dir)
    if settings.pit2isaac.physics_priors_json:
        settings.pit2isaac.physics_priors_json = _resolve_path(settings.pit2isaac.physics_priors_json, base_dir)
    if settings.pit2isaac.asset_mapping_json:
        settings.pit2isaac.asset_mapping_json = _resolve_path(settings.pit2isaac.asset_mapping_json, base_dir)
    if settings.pit2isaac.conversion_report_json:
        settings.pit2isaac.conversion_report_json = _resolve_path(settings.pit2isaac.conversion_report_json, base_dir)
    if settings.runtime.video_source and not settings.runtime.video_source.isdigit():
        settings.runtime.video_source = _resolve_path(settings.runtime.video_source, base_dir)
    if settings.runtime.session_output_root:
        settings.runtime.session_output_root = _resolve_path(settings.runtime.session_output_root, base_dir)


def apply_session_output_root(settings: SPWMSettings, session_output_root: str | Path) -> None:
    root = Path(session_output_root).expanduser().resolve()
    settings.runtime.session_output_root = str(root)
    settings.isaac.stage_path = str(root / "runtime" / "demo_scene.usd")
    settings.pit.frame_dump_dir = str(root / "intermediate" / "frames")
    exports = root / "exports"
    settings.pit2isaac.output_root = str(exports)
    settings.pit2isaac.usd_path = str(exports / "pit_scene.usdc")
    settings.pit2isaac.physics_priors_json = str(exports / "physics_priors.json")
    settings.pit2isaac.asset_mapping_json = str(exports / "asset_mapping.json")
    settings.pit2isaac.conversion_report_json = str(exports / "conversion_report.json")


def _validate_settings(settings: SPWMSettings, *, require_session_output_root: bool) -> None:
    if require_session_output_root and not settings.runtime.session_output_root:
        raise ValueError("runtime.session_output_root is required")
    mode = (settings.runtime.asset_materialization or "").strip().lower()
    if mode and mode not in {"copy", "move", "hardlink", "symlink"}:
        raise ValueError(
            "runtime.asset_materialization must be one of: copy, move, hardlink, symlink"
        )
    alignment_backend = (settings.pit.alignment_backend or "").strip().lower()
    if alignment_backend and alignment_backend not in {"depth_icp", "gotrack_visual"}:
        raise ValueError("pit.alignment_backend must be one of: depth_icp, gotrack_visual")


def _resolve_path(value: str, base_dir: Path) -> str:
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _to_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    root_scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}

    for key, value in root_scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if root_scalars and tables:
        lines.append("")

    first = True
    for table, table_value in tables.items():
        if not first:
            lines.append("")
        first = False
        lines.append(f"[{table}]")
        for key, value in table_value.items():
            lines.append(f"{key} = {_toml_value(value)}")

    return "\n".join(lines) + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return '""'
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        parts = [f"{k} = {_toml_value(v)}" for k, v in value.items()]
        return "{ " + ", ".join(parts) + " }"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
