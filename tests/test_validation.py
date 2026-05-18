"""Validation engine tests."""
from __future__ import annotations
from datetime import datetime, timezone
from guanwu.core.validation import validate_bundle
from guanwu.schemas.bundles import JobContext, NormalizeBundle
from guanwu.schemas.records import (
    DatasetRecord, SceneRecord, AssetRecord, LicenseRecord, ProvenanceRecord
)
from guanwu.schemas.enums import GeometryLevel, SceneKind, AccessMode, SourceType, RecordScope

def _make_ctx(tmp_path):
    return JobContext(
        job_id="test", workspace_root=str(tmp_path),
        raw_root=str(tmp_path / "raw"), staging_root=str(tmp_path / "staging"),
        canonical_root=str(tmp_path / "canonical"),
    )

def test_validate_empty_bundle(tmp_path):
    ctx = _make_ctx(tmp_path)
    bundle = NormalizeBundle(dataset_id="test")
    report = validate_bundle(bundle, ctx)
    assert not report.passed  # No dataset record = error

def test_validate_minimal_valid(tmp_path):
    ctx = _make_ctx(tmp_path)
    now = datetime.now(timezone.utc)
    bundle = NormalizeBundle(
        dataset_id="test",
        dataset_record=DatasetRecord(
            dataset_id="test", dataset_name="Test",
            source_type=SourceType.LOCAL_FOLDER,
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G4_EXACT_MESH,
            created_at=now,
        ),
        scenes=[SceneRecord(
            scene_uid="s1", dataset_id="test",
            scene_kind=SceneKind.INDOOR_STATIC,
            geometry_level=GeometryLevel.G4_EXACT_MESH,
            has_static_scene_mesh=True, has_dynamic_objects=False,
            has_humans=False, has_articulation=False,
        )],
        licenses=[LicenseRecord(
            record_scope=RecordScope.DATASET, record_id="test",
            license_name="Test License",
        )],
        provenance=[ProvenanceRecord(
            record_id="test", dataset_id="test",
            normalized_by_version="0.1.0", normalized_at=now,
            adapter_name="test", adapter_version="0.1.0",
        )],
    )
    report = validate_bundle(bundle, ctx)
    assert report.passed
    assert report.num_errors == 0

def test_validate_duplicate_scene_uid(tmp_path):
    ctx = _make_ctx(tmp_path)
    now = datetime.now(timezone.utc)
    scene = SceneRecord(
        scene_uid="s1", dataset_id="test",
        scene_kind=SceneKind.INDOOR_STATIC,
        geometry_level=GeometryLevel.G4_EXACT_MESH,
        has_static_scene_mesh=True, has_dynamic_objects=False,
        has_humans=False, has_articulation=False,
    )
    bundle = NormalizeBundle(
        dataset_id="test",
        dataset_record=DatasetRecord(
            dataset_id="test", dataset_name="Test",
            source_type=SourceType.LOCAL_FOLDER,
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G4_EXACT_MESH,
            created_at=now,
        ),
        scenes=[scene, scene],  # duplicate
        licenses=[LicenseRecord(record_scope=RecordScope.DATASET, record_id="test")],
        provenance=[ProvenanceRecord(
            record_id="test", dataset_id="test",
            normalized_by_version="0.1.0", normalized_at=now,
            adapter_name="test", adapter_version="0.1.0",
        )],
    )
    report = validate_bundle(bundle, ctx)
    assert not report.passed
    assert report.num_errors > 0
