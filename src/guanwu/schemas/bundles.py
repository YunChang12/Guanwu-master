from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from guanwu.schemas.records import (
    ArticulationStateRecord,
    AssetRecord,
    DatasetRecord,
    EpisodeRecord,
    FrameRecord,
    InstanceRecord,
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
    SensorRecord,
    TrackStateRecord,
)


class AdapterConfig(BaseModel):
    dataset_id: str
    source_mode: str
    source_path: str | None = None
    source_uri: str | None = None
    cache_dir: str | None = None
    options: dict = {}
    filters: dict = {}
    splits: list[str] = []


class SourceItem(BaseModel):
    item_id: str
    dataset_id: str
    item_type: str  # "scene", "asset", "episode"
    source_path: str | None = None
    source_uri: str | None = None
    metadata: dict = {}


class RawRef(BaseModel):
    item_id: str
    raw_path: str  # path in raw store
    checksum_sha256: str | None = None
    size_bytes: int | None = None


class ParseBundle(BaseModel):
    dataset_id: str
    scenes: list[dict] = []
    assets: list[dict] = []
    sensors: list[dict] = []
    frames: list[dict] = []
    instances: list[dict] = []
    annotations: list[dict] = []
    articulations: list[dict] = []
    licenses: list[dict] = []
    raw_refs: list[RawRef] = []


class NormalizeBundle(BaseModel):
    dataset_id: str
    dataset_record: DatasetRecord | None = None
    scenes: list[SceneRecord] = []
    assets: list[AssetRecord] = []
    episodes: list[EpisodeRecord] = []
    sensors: list[SensorRecord] = []
    frames: list[FrameRecord] = []
    instances: list[InstanceRecord] = []
    track_states: list[TrackStateRecord] = []
    articulation_states: list[ArticulationStateRecord] = []
    licenses: list[LicenseRecord] = []
    provenance: list[ProvenanceRecord] = []


class ValidationIssue(BaseModel):
    severity: str  # "error", "warning", "info"
    check_name: str
    record_type: str | None = None
    record_id: str | None = None
    message: str
    details: dict = {}


class ValidationReport(BaseModel):
    dataset_id: str
    passed: bool
    issues: list[ValidationIssue] = []
    num_errors: int = 0
    num_warnings: int = 0


class EmitReport(BaseModel):
    dataset_id: str
    files_written: list[str] = []
    scenes_emitted: int = 0
    assets_emitted: int = 0
    episodes_emitted: int = 0


class JobContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    workspace_root: str
    raw_root: str
    staging_root: str
    canonical_root: str
    dry_run: bool = False
    resume: bool = False
    workers: int = 1
    fail_fast: bool = False
    limit: int | None = None
    scene_id: str | None = None
    asset_id: str | None = None
    remote_host: str | None = None
    remote_conda_env: str | None = None
    remote_work_dir: str = "/tmp/guanwu_remote"
    remote_python: str = "python3"
    remote_conda_init: str = "/root/miniconda3/etc/profile.d/conda.sh"
