from __future__ import annotations

import json
from pathlib import Path

import pytest

from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.project.artifacts import ArtifactRecord
from guanwu.video.project.config import ProjectConfig, ProjectMetadata
from guanwu.video.project.context import ProjectContext
from guanwu.video.project.executor import ProjectExecutor


def _build_mesh_reconstruct_executor(tmp_path: Path) -> ProjectExecutor:
    project_root = tmp_path / "demo_project"
    config = ProjectConfig(
        project=ProjectMetadata(
            project_id="demo_project",
            name="demo_project",
            input_video="/tmp/demo.mp4",
            root_dir=str(project_root),
            provider_mode="zaiwu",
            video_copy_mode="copy",
        ),
    )
    config.settings.zaiwu.enabled = True
    context = ProjectContext.create(project_root, config)
    executor = ProjectExecutor(context)

    geometry_dir = context.stage_output_dir("geometry.lift")
    attr_dir = context.stage_output_dir("object.attr")
    detections_path = geometry_dir / "frame_000001_detections.json"
    summary_path = geometry_dir / "summary.json"
    attrs_path = attr_dir / "object_attrs.json"

    instances: list[DetectedInstance] = []
    latest_objects: list[dict] = []
    object_attrs: dict[str, dict[str, bool]] = {}
    for idx, width in enumerate((10.0, 20.0, 30.0, 40.0, 50.0), start=1):
        object_id = f"obj_{idx:06d}"
        instances.append(
            DetectedInstance(
                mask_ref=f"mask://frame_00001/{object_id}",
                bbox=[0.0, 0.0, width, 10.0],
                object_id=object_id,
                concept_label="car",
                segment_kind="object",
                score=0.9,
            )
        )
        latest_objects.append(
            ObjectNode(
                object_id=object_id,
                label="car",
                confidence=0.9,
                segment_kind="object",
            ).model_dump(mode="json")
        )
        object_attrs[object_id] = {"is_movable": True, "is_rigid_body": True}

    detections_path.write_text(
        json.dumps(
            FrameDetections(
                frame_idx=1,
                timestamp=0.0,
                image_b64="ZmFrZQ==",
                instances=instances,
            ).model_dump(mode="json"),
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 1,
                        "timestamp": 0.0,
                        "detections": str(detections_path),
                    }
                ],
                "latest_objects": latest_objects,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    attrs_path.write_text(json.dumps(object_attrs, indent=2), encoding="utf-8")

    context.artifacts.set(
        ArtifactRecord(
            stage="geometry.lift",
            created_at="2026-04-23T00:00:00Z",
            inputs_hash="geometry",
            params_hash="geometry",
            outputs={"summary": str(summary_path)},
            summary={},
        )
    )
    context.artifacts.set(
        ArtifactRecord(
            stage="object.attr",
            created_at="2026-04-23T00:00:00Z",
            inputs_hash="attrs",
            params_hash="attrs",
            outputs={"object_attrs": str(attrs_path)},
            summary={},
        )
    )
    return executor


def test_mesh_reconstruct_attempts_all_zaiwu_movable_rigid_candidates(tmp_path: Path, monkeypatch) -> None:
    executor = _build_mesh_reconstruct_executor(tmp_path)
    attempted_ids: list[str] = []

    class _FakeAdapter:
        def reconstruct_object_meshes(self, best_frames, objects):  # type: ignore[no-untyped-def]
            _ = best_frames
            object_id = objects[0].object_id
            attempted_ids.append(object_id)
            if len(attempted_ids) == 1:
                return {
                    object_id: {
                        "instance_id": object_id,
                        "segment_kind": "object",
                        "mesh_path": f"/tmp/{object_id}.ply",
                        "files": [],
                    }
                }
            return {}

    monkeypatch.setattr(executor, "_assert_zaiwu_service_ready", lambda service_id, stage: None)
    monkeypatch.setattr(executor, "_get_zaiwu_sam3d", lambda: _FakeAdapter())

    result = executor._run_mesh_reconstruct()
    meshes_payload = json.loads(Path(result["outputs"]["sam3d_meshes"]).read_text(encoding="utf-8"))

    assert attempted_ids == ["obj_000005", "obj_000004", "obj_000003", "obj_000002", "obj_000001"]
    assert result["summary"]["mesh_count"] == 1
    assert result["summary"]["selected_count"] == 5
    assert result["summary"]["attempted_count"] == 5
    assert result["summary"]["failed_count"] == 4
    assert result["summary"]["skipped_count"] == 0
    assert set(meshes_payload) == {"obj_000005"}


def test_mesh_reconstruct_errors_when_sam3d_service_not_ready(tmp_path: Path, monkeypatch) -> None:
    executor = _build_mesh_reconstruct_executor(tmp_path)

    class _Gateway:
        gateway_url = "http://127.0.0.1:8181"

        def get_ready_service(self, service_id: str):  # type: ignore[no-untyped-def]
            _ = service_id
            return None

    monkeypatch.setattr(
        "guanwu.video.project.executor.build_zaiwu_gateway_client",
        lambda settings: _Gateway(),
    )

    with pytest.raises(RuntimeError, match="requires Zaiwu service services.sam3d to already be running"):
        executor._run_mesh_reconstruct()
