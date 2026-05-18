from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from guanwu.schemas.enums import (
    AccessMode,
    GeometryLevel,
    RecordScope,
    SceneKind,
    SensorType,
    SourceType,
)


class DatasetRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    dataset_id: str
    dataset_name: str
    version: str | None = None
    source_type: SourceType
    source_uri: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    access_mode: AccessMode
    geometry_level_max: GeometryLevel
    created_at: datetime
    tags: list[str] = []


class SceneRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    scene_uid: str
    dataset_id: str
    source_scene_id: str | None = None
    scene_name: str | None = None
    scene_kind: SceneKind
    geometry_level: GeometryLevel
    world_units: Literal["meters"] = "meters"
    canonical_up_axis: Literal["Z"] = "Z"
    duration_sec: float | None = None
    num_frames: int | None = None
    num_sensors: int | None = None
    has_static_scene_mesh: bool
    has_dynamic_objects: bool
    has_humans: bool
    has_articulation: bool
    bbox_min_xyz: tuple[float, float, float] | None = None
    bbox_max_xyz: tuple[float, float, float] | None = None


class AssetRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    asset_uid: str
    dataset_id: str
    source_asset_id: str | None = None
    category: str | None = None
    supercategory: str | None = None
    geometry_level: GeometryLevel
    is_articulated: bool
    is_deformable: bool
    mesh_uri: str | None = None
    usd_uri: str | None = None
    glb_uri: str | None = None
    num_vertices: int | None = None
    num_faces: int | None = None
    watertight: bool | None = None
    manifold: bool | None = None
    material_count: int | None = None
    texture_count: int | None = None


class SensorRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    sensor_uid: str
    scene_uid: str | None = None
    episode_uid: str | None = None
    sensor_type: SensorType
    name: str
    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_model: str | None = None
    distortion_params: list[float] | None = None
    T_sensor_from_parent: list[float] | None = None  # 4x4 row-major
    parent_frame: str | None = None


class FrameRecord(BaseModel):
    frame_uid: str
    sensor_uid: str
    episode_uid: str | None = None
    scene_uid: str | None = None
    timestamp_ns: int
    image_uri: str | None = None
    depth_uri: str | None = None
    segmentation_uri: str | None = None
    pointcloud_uri: str | None = None
    T_world_from_sensor: list[float] | None = None
    exposure_time_ms: float | None = None


class EpisodeRecord(BaseModel):
    episode_uid: str
    dataset_id: str
    scene_uid: str | None = None
    source_episode_id: str | None = None
    duration_sec: float | None = None
    num_frames: int | None = None


class InstanceRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    instance_uid: str
    scene_uid: str | None = None
    episode_uid: str | None = None
    asset_uid: str | None = None
    category: str | None = None
    instance_name: str | None = None
    is_static: bool
    is_articulated: bool
    is_human: bool
    geometry_level: GeometryLevel


class TrackStateRecord(BaseModel):
    track_uid: str
    instance_uid: str
    timestamp_ns: int
    T_world_from_object: list[float] | None = None
    linear_velocity_xyz: tuple[float, float, float] | None = None
    angular_velocity_xyz: tuple[float, float, float] | None = None
    bbox3d_center_xyz: tuple[float, float, float] | None = None
    bbox3d_size_xyz: tuple[float, float, float] | None = None
    visibility: float | None = None


class ArticulationStateRecord(BaseModel):
    instance_uid: str
    asset_uid: str
    timestamp_ns: int
    joint_names: list[str]
    joint_positions: list[float]
    joint_velocities: list[float] | None = None


class LicenseRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    record_scope: RecordScope
    record_id: str
    license_name: str | None = None
    license_url: str | None = None
    commercial_use_allowed: bool | None = None
    redistribution_allowed: bool | None = None
    attribution_required: bool | None = None
    notes: str | None = None


class ProvenanceRecord(BaseModel):
    record_id: str
    dataset_id: str
    source_relpath: str | None = None
    source_sha256: str | None = None
    normalized_by_version: str
    normalized_at: datetime
    adapter_name: str
    adapter_version: str
    transform_log: list[dict] = []
