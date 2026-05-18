"""Tests for canonical schemas (Pydantic models)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError as PydanticValidationError

from guanwu.schemas.enums import (
    AccessMode,
    GeometryLevel,
    RecordScope,
    SceneKind,
    SensorType,
    SourceType,
)
from guanwu.schemas.records import (
    ArticulationStateRecord,
    AssetRecord,
    DatasetRecord,
    FrameRecord,
    InstanceRecord,
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
    SensorRecord,
    TrackStateRecord,
)


class TestDatasetRecord:
    def test_create_minimal(self):
        r = DatasetRecord(
            dataset_id="test",
            dataset_name="Test Dataset",
            source_type=SourceType.LOCAL_FOLDER,
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G4_EXACT_MESH,
            created_at=datetime.now(timezone.utc),
        )
        assert r.dataset_id == "test"
        assert r.version is None
        assert r.tags == []

    def test_enum_values_serialized(self):
        r = DatasetRecord(
            dataset_id="test",
            dataset_name="Test",
            source_type=SourceType.LOCAL_FOLDER,
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G4_EXACT_MESH,
            created_at=datetime.now(timezone.utc),
        )
        d = r.model_dump()
        assert d["source_type"] == "local_folder"
        assert d["access_mode"] == "public"
        assert d["geometry_level_max"] == "G4_EXACT_MESH"


class TestSceneRecord:
    def test_create_full(self):
        r = SceneRecord(
            scene_uid="abc123",
            dataset_id="scannetpp",
            source_scene_id="scene0001_00",
            scene_name="Scene 1",
            scene_kind=SceneKind.INDOOR_STATIC,
            geometry_level=GeometryLevel.G4_EXACT_MESH,
            has_static_scene_mesh=True,
            has_dynamic_objects=False,
            has_humans=False,
            has_articulation=False,
            bbox_min_xyz=(-1.0, -1.0, 0.0),
            bbox_max_xyz=(5.0, 5.0, 3.0),
        )
        assert r.world_units == "meters"
        assert r.canonical_up_axis == "Z"

    def test_defaults(self):
        r = SceneRecord(
            scene_uid="s1",
            dataset_id="test",
            scene_kind=SceneKind.INDOOR_STATIC,
            geometry_level=GeometryLevel.G4_EXACT_MESH,
            has_static_scene_mesh=True,
            has_dynamic_objects=False,
            has_humans=False,
            has_articulation=False,
        )
        assert r.duration_sec is None
        assert r.num_frames is None


class TestAssetRecord:
    def test_articulated_asset(self):
        r = AssetRecord(
            asset_uid="a1",
            dataset_id="partnet_mobility",
            geometry_level=GeometryLevel.G5_ARTICULATED_MESH,
            is_articulated=True,
            is_deformable=False,
        )
        assert r.is_articulated

    def test_simple_asset(self):
        r = AssetRecord(
            asset_uid="a2",
            dataset_id="objaverse_xl",
            geometry_level=GeometryLevel.G4_EXACT_MESH,
            is_articulated=False,
            is_deformable=False,
            num_vertices=1000,
            num_faces=500,
        )
        assert r.num_vertices == 1000


class TestSensorRecord:
    def test_camera_sensor(self):
        r = SensorRecord(
            sensor_uid="s1",
            sensor_type=SensorType.CAMERA,
            name="camera_front",
            width=1920,
            height=1080,
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0,
        )
        assert r.sensor_type == "camera"


class TestFrameRecord:
    def test_frame(self):
        r = FrameRecord(
            frame_uid="f1",
            sensor_uid="s1",
            timestamp_ns=1000000000,
            image_uri="images/frame_001.jpg",
        )
        assert r.timestamp_ns == 1000000000


class TestTrackStateRecord:
    def test_with_bbox(self):
        r = TrackStateRecord(
            track_uid="t1",
            instance_uid="i1",
            timestamp_ns=0,
            bbox3d_center_xyz=(1.0, 2.0, 0.5),
            bbox3d_size_xyz=(0.5, 0.5, 1.0),
        )
        assert r.bbox3d_center_xyz == (1.0, 2.0, 0.5)


class TestLicenseRecord:
    def test_license(self):
        r = LicenseRecord(
            record_scope=RecordScope.DATASET,
            record_id="scannetpp",
            license_name="CC BY-NC-SA 4.0",
            commercial_use_allowed=False,
            redistribution_allowed=True,
            attribution_required=True,
        )
        d = r.model_dump()
        assert d["record_scope"] == "dataset"


class TestProvenanceRecord:
    def test_provenance(self):
        r = ProvenanceRecord(
            record_id="test",
            dataset_id="scannetpp",
            normalized_by_version="0.1.0",
            normalized_at=datetime.now(timezone.utc),
            adapter_name="scannetpp",
            adapter_version="0.1.0",
            transform_log=[{"step": "convert_up_axis", "from": "Y", "to": "Z"}],
        )
        assert len(r.transform_log) == 1
