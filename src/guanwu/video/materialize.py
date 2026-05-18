from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from guanwu.core.ids import make_asset_uid, make_episode_uid, make_frame_uid, make_instance_uid, make_scene_uid, make_sensor_uid, make_track_uid
from guanwu.schemas.enums import AccessMode, GeometryLevel, RecordScope, SceneKind, SensorType, SourceType
from guanwu.schemas.records import AssetRecord, DatasetRecord, EpisodeRecord, FrameRecord, InstanceRecord, LicenseRecord, ProvenanceRecord, SceneRecord, SensorRecord, TrackStateRecord
from guanwu.storage.canonical_store import CanonicalStore
from guanwu.video.registry import NATURAL_VIDEO_DATASET_ID


@dataclass
class VideoMaterializeReport:
    dataset_id: str
    scene_uid: str
    episode_uid: str
    scenes_emitted: int = 0
    assets_emitted: int = 0
    episodes_emitted: int = 0

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "scene_uid": self.scene_uid,
            "episode_uid": self.episode_uid,
            "scenes_emitted": self.scenes_emitted,
            "assets_emitted": self.assets_emitted,
            "episodes_emitted": self.episodes_emitted,
        }


def _load_json_file(path: str | Path) -> dict | list | None:
    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _normalize_geometry_summary(summary: dict) -> dict:
    normalized = dict(summary)
    for key in ("camera_trajectory", "object_trajectories"):
        value = normalized.get(key)
        if isinstance(value, str):
            loaded = _load_json_file(value)
            if loaded is not None:
                normalized[key] = loaded
    if not isinstance(normalized.get("object_trajectories"), dict):
        normalized["object_trajectories"] = {}
    if not isinstance(normalized.get("camera_trajectory"), list):
        normalized["camera_trajectory"] = []
    return normalized


def _mesh_path_from_entry(entry: dict | None) -> str | None:
    if not isinstance(entry, dict):
        return None
    mesh_path = str(entry.get("mesh_path", "") or "").strip()
    if mesh_path:
        return mesh_path
    files = entry.get("files")
    if not isinstance(files, list):
        return None
    preferred_suffixes = [".usdc", ".usd", ".usda", ".glb", ".obj", ".ply"]
    preferred_paths: list[str] = []
    fallback_paths: list[str] = []
    for file_entry in files:
        if not isinstance(file_entry, dict):
            continue
        candidate = str(file_entry.get("path", "") or "").strip()
        if not candidate:
            continue
        fallback_paths.append(candidate)
        suffix = Path(candidate).suffix.lower()
        if suffix in preferred_suffixes:
            preferred_paths.append(candidate)
    for suffix in preferred_suffixes:
        for candidate in preferred_paths:
            if Path(candidate).suffix.lower() == suffix:
                return candidate
    return fallback_paths[0] if fallback_paths else None


def _valid_vec3_tuple(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out


def _track_center(point: dict) -> tuple[float, float, float] | None:
    return _valid_vec3_tuple(point.get("centroid_world")) or _valid_vec3_tuple(point.get("position_xyz"))


def _track_size(point: dict) -> tuple[float, float, float] | None:
    aabb = point.get("bbox_3d_aabb")
    if isinstance(aabb, dict):
        lo = _valid_vec3_tuple(aabb.get("min"))
        hi = _valid_vec3_tuple(aabb.get("max"))
        if lo is not None and hi is not None:
            size = tuple(max(0.0, hi[i] - lo[i]) for i in range(3))
            if any(v > 0.0 for v in size):
                return size  # type: ignore[return-value]
    return _valid_vec3_tuple(point.get("scale"))


def materialize_video_project(
    *,
    project_root: str | Path,
    canonical_root: str | Path,
    dataset_id: str = NATURAL_VIDEO_DATASET_ID,
    scene_export_path: str | Path,
    video_metadata: dict,
    frame_index: list[dict],
    object_index: list[dict],
    object_attrs: dict,
    geometry_summary: dict,
    mesh_manifest: dict,
) -> VideoMaterializeReport:
    now = datetime.now(timezone.utc)
    project_root = Path(project_root).expanduser().resolve()
    store = CanonicalStore(str(canonical_root))
    project_id = project_root.name
    geometry_summary = _normalize_geometry_summary(geometry_summary)

    source_scene_id = project_id
    source_episode_id = f"{project_id}:episode"
    scene_uid = make_scene_uid(dataset_id, source_scene_id)
    episode_uid = make_episode_uid(dataset_id, source_episode_id)
    sensor_uid = make_sensor_uid(dataset_id, episode_uid, "video_camera")

    has_mesh = bool(mesh_manifest)
    geometry_level = GeometryLevel.G3_PROXY_MESH if has_mesh else GeometryLevel.G2_POINT_OBS

    dataset_record = DatasetRecord(
        dataset_id=dataset_id,
        dataset_name="Natural Video",
        version="v1",
        source_type=SourceType.GENERATOR,
        access_mode=AccessMode.MANUAL,
        geometry_level_max=GeometryLevel.G3_PROXY_MESH,
        created_at=now,
        tags=["video", "proxy-mesh", "usdc"],
    )
    scene_record = SceneRecord(
        scene_uid=scene_uid,
        dataset_id=dataset_id,
        source_scene_id=source_scene_id,
        scene_name=project_id,
        scene_kind=SceneKind.MIXED,
        geometry_level=geometry_level,
        num_frames=int(video_metadata.get("frame_count", len(frame_index))),
        duration_sec=float(video_metadata.get("duration_sec", 0.0) or 0.0),
        has_static_scene_mesh=has_mesh,
        has_dynamic_objects=bool(object_index),
        has_humans=False,
        has_articulation=False,
    )
    episode_record = EpisodeRecord(
        episode_uid=episode_uid,
        dataset_id=dataset_id,
        scene_uid=scene_uid,
        source_episode_id=source_episode_id,
        duration_sec=float(video_metadata.get("duration_sec", 0.0) or 0.0),
        num_frames=int(video_metadata.get("frame_count", len(frame_index))),
    )
    sensor_record = SensorRecord(
        sensor_uid=sensor_uid,
        scene_uid=scene_uid,
        episode_uid=episode_uid,
        sensor_type=SensorType.CAMERA,
        name="video_camera",
        width=int(video_metadata.get("width", 0) or 0) or None,
        height=int(video_metadata.get("height", 0) or 0) or None,
        fx=float(video_metadata.get("width", 0) or 0) if video_metadata.get("width") else None,
        fy=float(video_metadata.get("height", 0) or 0) if video_metadata.get("height") else None,
        cx=float(video_metadata.get("width", 0) or 0) / 2 if video_metadata.get("width") else None,
        cy=float(video_metadata.get("height", 0) or 0) / 2 if video_metadata.get("height") else None,
    )

    frame_records: list[FrameRecord] = []
    for frame in frame_index:
        frame_idx = int(frame["frame_idx"])
        timestamp_ns = int(float(frame.get("timestamp", 0.0)) * 1_000_000_000)
        frame_records.append(FrameRecord(
            frame_uid=make_frame_uid(dataset_id, sensor_uid, str(frame_idx)),
            sensor_uid=sensor_uid,
            episode_uid=episode_uid,
            scene_uid=scene_uid,
            timestamp_ns=timestamp_ns,
            image_uri=frame.get("image_uri"),
        ))

    instances: list[InstanceRecord] = []
    tracks: list[TrackStateRecord] = []
    assets: list[AssetRecord] = []
    track_source = geometry_summary.get("object_trajectories", {})
    for obj in object_index:
        obj_id = obj["object_id"]
        label = obj.get("label", "object")
        instance_uid = make_instance_uid(dataset_id, episode_uid, obj_id)
        asset_uid = None
        mesh_entry = mesh_manifest.get(obj_id)
        mesh_path = _mesh_path_from_entry(mesh_entry)
        if mesh_path:
            asset_uid = make_asset_uid(dataset_id, obj_id)
            assets.append(AssetRecord(
                asset_uid=asset_uid,
                dataset_id=dataset_id,
                source_asset_id=obj_id,
                category=label,
                supercategory="object",
                geometry_level=GeometryLevel.G3_PROXY_MESH,
                is_articulated=False,
                is_deformable=False,
                mesh_uri=mesh_path,
            ))
        instances.append(InstanceRecord(
            instance_uid=instance_uid,
            scene_uid=scene_uid,
            episode_uid=episode_uid,
            asset_uid=asset_uid,
            category=label,
            instance_name=obj_id,
            is_static=False,
            is_articulated=False,
            is_human=False,
            geometry_level=GeometryLevel.G3_PROXY_MESH if asset_uid else GeometryLevel.G2_POINT_OBS,
        ))
        for point in track_source.get(obj_id, []):
            center = _track_center(point)
            size = _track_size(point)
            timestamp = point.get("timestamp_sec", point.get("timestamp", 0.0))
            timestamp_ns = int(float(timestamp) * 1_000_000_000)
            tracks.append(TrackStateRecord(
                track_uid=make_track_uid(dataset_id, instance_uid),
                instance_uid=instance_uid,
                timestamp_ns=timestamp_ns,
                bbox3d_center_xyz=center,
                bbox3d_size_xyz=size,
                visibility=1.0,
            ))

    license_record = LicenseRecord(
        record_scope=RecordScope.DATASET,
        record_id=dataset_id,
        license_name="User Provided / Manual",
        commercial_use_allowed=None,
        redistribution_allowed=False,
        attribution_required=None,
        notes="Natural-scene user video materialized locally; redistribution disabled by default.",
    )
    provenance_record = ProvenanceRecord(
        record_id=dataset_id,
        dataset_id=dataset_id,
        source_relpath=str(Path(scene_export_path).resolve()),
        normalized_by_version="0.1.0",
        normalized_at=now,
        adapter_name="video_project",
        adapter_version="0.1.0",
        transform_log=[{
            "step": "video.materialize",
            "project_root": str(project_root),
            "source_video": str(project_root / "input" / "video.mp4"),
        }],
    )

    store.write_dataset_record(dataset_id, dataset_record.model_dump(mode="json"))
    store.write_scene_meta(dataset_id, scene_uid, scene_record.model_dump(mode="json"))
    store.write_episode_meta(dataset_id, episode_uid, episode_record.model_dump(mode="json"))
    store.write_scene_sensors(dataset_id, scene_uid, [sensor_record.model_dump(mode="json")])
    store.write_scene_frames(dataset_id, scene_uid, [frame.model_dump(mode="json") for frame in frame_records])
    store.write_episode_sensor_frames(dataset_id, episode_uid, [frame.model_dump(mode="json") for frame in frame_records])
    store.write_scene_instances(dataset_id, scene_uid, [instance.model_dump(mode="json") for instance in instances])
    store.write_scene_tracks(dataset_id, scene_uid, [track.model_dump(mode="json") for track in tracks])
    store.write_episode_states(dataset_id, episode_uid, [track.model_dump(mode="json") for track in tracks])
    store.write_licenses(dataset_id, "dataset", dataset_id, [license_record.model_dump(mode="json")])
    store.write_provenance(dataset_id, "dataset", dataset_id, provenance_record.model_dump(mode="json"))

    scene_dir = store.scene_dir(dataset_id, scene_uid)
    exported_scene_path = Path(scene_export_path)
    target_scene_path = scene_dir / "scene.usdc"
    target_scene_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_scene_path, target_scene_path)

    for asset in assets:
        asset_dir = store.asset_dir(dataset_id, asset.asset_uid)
        store.write_asset_meta(dataset_id, asset.asset_uid, asset.model_dump(mode="json"))
        mesh_path = Path(asset.mesh_uri or "")
        if mesh_path.exists():
            shutil.copy2(mesh_path, asset_dir / mesh_path.name)
            if mesh_path.suffix != ".usdc":
                placeholder = asset_dir / "asset.usdc"
                placeholder.write_text("#usda 1.0\n", encoding="utf-8")

    report = VideoMaterializeReport(
        dataset_id=dataset_id,
        scene_uid=scene_uid,
        episode_uid=episode_uid,
        scenes_emitted=1,
        assets_emitted=len(assets),
        episodes_emitted=1,
    )
    return report
