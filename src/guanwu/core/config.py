from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    raw_root: str = "raw"
    staging_root: str = "staging"
    canonical_root: str = "canonical"
    export_root: str = "exports"
    catalog_path: str = "catalog/catalog.duckdb"
    project_root: str = "projects"


class RemoteConfig(BaseModel):
    host: str | None = None
    conda_env: str | None = None
    work_dir: str = "/tmp/guanwu_remote"
    python: str = "python3"
    conda_init: str = "/root/miniconda3/etc/profile.d/conda.sh"


class RuntimeConfig(BaseModel):
    workers: int = 8
    fail_fast: bool = False
    log_json: bool = False
    resume: bool = True
    remote: RemoteConfig = RemoteConfig()


class VideoPipelineConfig(BaseModel):
    provider_mode: str = "mock"
    camera_provider: str = "wildgs"
    export_profile: str = "usd_full"
    default_dataset_id: str = "natural_video"
    zaiwu_gateway_url: str | None = None
    zaiwu_auto_start_workers: bool = True
    object_detection_backend: str = "seg2track_sam2"
    pose_optimizer_timeout_sec: float = 1800.0
    pose_optimize_min_bbox_area_px: float = 5000.0


class PoliciesConfig(BaseModel):
    default_up_axis: str = "Z"
    unit: str = "meters"
    fail_on_unknown_license: bool = False
    generate_glb_preview: bool = True
    generate_mesh_stats: bool = True
    generate_pointcloud_preview: bool = False
    keep_raw_forever: bool = True
    allow_proxy_mesh_generation: bool = True


class DatasetSourceConfig(BaseModel):
    mode: str = "local"
    path: str | None = None
    uri: str | None = None
    cache_dir: str | None = None
    object_ids_file: str | None = None


class DatasetConfig(BaseModel):
    enabled: bool = True
    source: DatasetSourceConfig = DatasetSourceConfig()
    splits: list[str] = []
    options: dict = {}
    filters: dict = {}


class WorkspaceConfig(BaseModel):
    workspace_root: str = "."
    random_seed: int = 42
    storage: StorageConfig = StorageConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    video_pipeline: VideoPipelineConfig = VideoPipelineConfig()
    policies: PoliciesConfig = PoliciesConfig()
    datasets: dict[str, DatasetConfig] = {}

    def resolve_paths(self) -> None:
        """Resolve relative paths against workspace_root."""
        root = Path(self.workspace_root)
        self.storage.raw_root = str(root / self.storage.raw_root)
        self.storage.staging_root = str(root / self.storage.staging_root)
        self.storage.canonical_root = str(root / self.storage.canonical_root)
        self.storage.export_root = str(root / self.storage.export_root)
        self.storage.catalog_path = str(root / self.storage.catalog_path)
        self.storage.project_root = str(root / self.storage.project_root)


def load_config(path: str | Path) -> WorkspaceConfig:
    """Load workspace config from YAML file."""
    p = Path(path)
    if not p.exists():
        from guanwu.core.errors import ConfigError

        raise ConfigError(f"Config file not found: {path}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    config = WorkspaceConfig(**data)
    config.resolve_paths()
    return config
