from __future__ import annotations

import json
from pathlib import Path

from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.project.artifacts import ArtifactRecord
from guanwu.video.project.config import ProjectConfig, ProjectMetadata
from guanwu.video.project.context import ProjectContext
from guanwu.video.project.executor import ProjectExecutor


def test_bbox_motion_summary_marks_small_jitter_static() -> None:
    frames = [
        {"bbox": [10.0, 10.0, 20.0, 20.0]},
        {"bbox": [10.5, 10.0, 20.5, 20.0]},
    ]

    summary = ProjectExecutor._bbox_motion_summary(
        frames,
        image_width=100,
        image_height=100,
        threshold=0.03,
    )

    assert summary["is_bbox_moving"] is False
    assert summary["center_span_norm"] < 0.03


def test_bbox_motion_summary_marks_large_center_span_moving() -> None:
    frames = [
        {"bbox": [10.0, 10.0, 20.0, 20.0]},
        {"bbox": [25.0, 10.0, 35.0, 20.0]},
    ]

    summary = ProjectExecutor._bbox_motion_summary(
        frames,
        image_width=100,
        image_height=100,
        threshold=0.03,
    )

    assert summary["is_bbox_moving"] is True
    assert summary["center_span_norm"] >= 0.03


def test_bbox_motion_summary_single_frame_is_static() -> None:
    summary = ProjectExecutor._bbox_motion_summary(
        [{"bbox": [10.0, 10.0, 20.0, 20.0]}],
        image_width=100,
        image_height=100,
        threshold=0.03,
    )

    assert summary["is_bbox_moving"] is False
    assert summary["center_span_norm"] == 0.0
    assert summary["frame_count"] == 1


def test_object_attr_merges_bbox_motion_fields(tmp_path: Path, monkeypatch) -> None:
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
    config.settings.runtime.bbox_motion_threshold = 0.03
    context = ProjectContext.create(project_root, config)
    executor = ProjectExecutor(context)

    detect_dir = context.stage_output_dir("object.detect")
    index_dir = context.stage_output_dir("object.index")
    inspect_dir = context.stage_output_dir("video.inspect")
    detect_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    inspect_dir.mkdir(parents=True, exist_ok=True)

    moving = ObjectNode(object_id="obj_000001", label="wooden block", segment_kind="object")
    static = ObjectNode(object_id="obj_000002", label="wooden board", segment_kind="object")
    objects_path = index_dir / "objects.json"
    objects_path.write_text(
        json.dumps(
            [
                {
                    "object_id": moving.object_id,
                    "label": moving.label,
                    "segment_kind": moving.segment_kind,
                    "frames": [
                        {"frame_idx": 1, "timestamp": 0.0, "bbox": [10.0, 10.0, 20.0, 20.0]},
                        {"frame_idx": 2, "timestamp": 0.1, "bbox": [25.0, 10.0, 35.0, 20.0]},
                    ],
                },
                {
                    "object_id": static.object_id,
                    "label": static.label,
                    "segment_kind": static.segment_kind,
                    "frames": [
                        {"frame_idx": 1, "timestamp": 0.0, "bbox": [40.0, 40.0, 55.0, 55.0]},
                        {"frame_idx": 2, "timestamp": 0.1, "bbox": [40.5, 40.0, 55.5, 55.0]},
                    ],
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    detections_paths: list[Path] = []
    for frame_idx, moving_bbox, static_bbox in (
        (1, [10.0, 10.0, 20.0, 20.0], [40.0, 40.0, 55.0, 55.0]),
        (2, [25.0, 10.0, 35.0, 20.0], [40.5, 40.0, 55.5, 55.0]),
    ):
        path = detect_dir / f"frame_{frame_idx:06d}_detections.json"
        path.write_text(
            json.dumps(
                FrameDetections(
                    frame_idx=frame_idx,
                    timestamp=(frame_idx - 1) * 0.1,
                    image_b64="ZmFrZQ==",
                    instances=[
                        DetectedInstance(
                            mask_ref=f"mask://frame_{frame_idx:05d}/{moving.object_id}",
                            bbox=moving_bbox,
                            object_id=moving.object_id,
                            concept_label=moving.label,
                            segment_kind=moving.segment_kind,
                            score=0.9,
                        ),
                        DetectedInstance(
                            mask_ref=f"mask://frame_{frame_idx:05d}/{static.object_id}",
                            bbox=static_bbox,
                            object_id=static.object_id,
                            concept_label=static.label,
                            segment_kind=static.segment_kind,
                            score=0.9,
                        ),
                    ],
                ).model_dump(mode="json"),
                indent=2,
            ),
            encoding="utf-8",
        )
        detections_paths.append(path)

    detect_summary_path = detect_dir / "summary.json"
    detect_summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_idx": 1, "timestamp": 0.0, "detections": str(detections_paths[0])},
                    {"frame_idx": 2, "timestamp": 0.1, "detections": str(detections_paths[1])},
                ],
                "latest_detections": str(detections_paths[-1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata_path = inspect_dir / "video_metadata.json"
    metadata_path.write_text(json.dumps({"width": 100, "height": 100}), encoding="utf-8")

    context.artifacts.set(
        ArtifactRecord(
            stage="object.detect",
            created_at="2026-06-02T00:00:00Z",
            inputs_hash="detect",
            params_hash="detect",
            outputs={"summary": str(detect_summary_path)},
            summary={},
        )
    )
    context.artifacts.set(
        ArtifactRecord(
            stage="object.index",
            created_at="2026-06-02T00:00:00Z",
            inputs_hash="index",
            params_hash="index",
            outputs={"objects": str(objects_path)},
            summary={},
        )
    )
    context.artifacts.set(
        ArtifactRecord(
            stage="video.inspect",
            created_at="2026-06-02T00:00:00Z",
            inputs_hash="inspect",
            params_hash="inspect",
            outputs={"video_metadata": str(metadata_path)},
            summary={},
        )
    )

    monkeypatch.setattr(
        executor,
        "_infer_object_physics_priors",
        lambda detections, objects: {
            obj.object_id: {
                "is_movable": True,
                "is_rigid_body": True,
                "class_name": obj.label,
            }
            for obj in objects
        },
    )

    result = executor._run_object_attr()
    payload = json.loads(Path(result["outputs"]["object_attrs"]).read_text(encoding="utf-8"))

    assert payload["obj_000001"]["is_rigid_body"] is True
    assert payload["obj_000001"]["is_bbox_moving"] is True
    assert payload["obj_000001"]["bbox_motion_score"] >= 0.03
    assert payload["obj_000001"]["bbox_motion_summary"]["threshold"] == 0.03
    assert payload["obj_000002"]["is_bbox_moving"] is False
    assert payload["obj_000002"]["bbox_motion_score"] < 0.03
    assert result["summary"]["bbox_moving_object_count"] == 1
