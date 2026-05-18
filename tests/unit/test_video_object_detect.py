from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.project.config import ProjectConfig, ProjectMetadata
from guanwu.video.project.context import ProjectContext
from guanwu.video.project.executor import ProjectExecutor
from guanwu.video.project.services import VideoFrameReader


def test_object_detect_deduplicates_prompt_expansion_duplicates(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "demo_project"
    config = ProjectConfig(
        project=ProjectMetadata(
            project_id="demo_project",
            name="demo_project",
            input_video="/tmp/demo.mp4",
            root_dir=str(project_root),
            provider_mode="mock",
            video_copy_mode="copy",
        ),
    )
    config.settings.runtime.vlm_discovery.enabled = False
    context = ProjectContext.create(project_root, config)
    executor = ProjectExecutor(context)

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    image_b64 = VideoFrameReader.encode_jpg(frame)
    detections = FrameDetections(
        frame_idx=1,
        timestamp=0.0,
        image_b64=image_b64,
        instances=[
            DetectedInstance(
                mask_ref="mask://frame_00001/s2t_1",
                bbox=[4.0, 5.0, 18.0, 16.0],
                object_id="s2t_1",
                concept_label="car",
                score=0.91,
                mask_rle={"size": [32, 32], "counts": "abc"},
            ),
            DetectedInstance(
                mask_ref="mask://frame_00001/s2t_2",
                bbox=[4.0, 5.0, 18.0, 16.0],
                object_id="s2t_2",
                concept_label="suv",
                score=0.72,
                mask_rle='{"size":[32,32],"counts":"abc"}',
            ),
        ],
    )

    monkeypatch.setattr(executor, "_services", lambda: SimpleNamespace(frames=[(1, 0.0, frame)]))
    monkeypatch.setattr(executor, "_detect_objects_in_frame", lambda frame, frame_idx, timestamp: detections)

    result = executor._run_object_detect()

    detections_path = Path(result["outputs"]["frames_dir"]) / "frame_000001" / "detections.json"
    payload = json.loads(detections_path.read_text(encoding="utf-8"))

    assert result["summary"]["latest_instance_count"] == 1
    assert len(payload["instances"]) == 1
    assert payload["instances"][0]["concept_label"] == "car"


def test_object_detect_repairs_seg2track_first_frame_warmup_gap(tmp_path: Path, monkeypatch) -> None:
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
    config.settings.runtime.vlm_discovery.enabled = False
    config.settings.zaiwu.enabled = True
    config.settings.zaiwu.object_detection_backend = "seg2track_sam2"
    config.settings.zaiwu.seg2track_sam2.detect_interval = 5
    context = ProjectContext.create(project_root, config)
    executor = ProjectExecutor(context)

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    image_b64 = VideoFrameReader.encode_jpg(frame)

    def _instance(object_id: str, bbox: list[float], *, label: str = "car", score: float = 0.9) -> DetectedInstance:
        return DetectedInstance(
            mask_ref=f"mask://frame_00001/{object_id}",
            bbox=bbox,
            object_id=object_id,
            concept_label=label,
            score=score,
            mask_rle={"size": [32, 32], "counts": object_id},
        )

    frame_detections = {
        1: FrameDetections(
            frame_idx=1,
            timestamp=0.0,
            image_b64=image_b64,
            instances=[_instance("s2t_1", [4.0, 5.0, 18.0, 16.0], score=0.91)],
        ),
        2: FrameDetections(
            frame_idx=2,
            timestamp=0.1,
            image_b64=image_b64,
            instances=[
                _instance("s2t_1", [4.0, 5.0, 18.0, 16.0], score=0.91),
                _instance("s2t_2", [19.0, 6.0, 28.0, 17.0], label="suv", score=0.85),
                _instance("s2t_3", [1.0, 1.0, 8.0, 8.0], label="person", score=0.8),
            ],
        ),
        3: FrameDetections(
            frame_idx=3,
            timestamp=0.2,
            image_b64=image_b64,
            instances=[
                _instance("s2t_1", [4.0, 5.0, 18.0, 16.0], score=0.91),
                _instance("s2t_2", [19.0, 6.0, 28.0, 17.0], label="suv", score=0.85),
                _instance("s2t_3", [1.0, 1.0, 8.0, 8.0], label="person", score=0.8),
            ],
        ),
        4: FrameDetections(
            frame_idx=4,
            timestamp=0.3,
            image_b64=image_b64,
            instances=[
                _instance("s2t_1", [4.0, 5.0, 18.0, 16.0], score=0.91),
                _instance("s2t_2", [19.0, 6.0, 28.0, 17.0], label="suv", score=0.85),
                _instance("s2t_3", [1.0, 1.0, 8.0, 8.0], label="person", score=0.8),
            ],
        ),
        5: FrameDetections(
            frame_idx=5,
            timestamp=0.4,
            image_b64=image_b64,
            instances=[
                _instance("s2t_1", [4.0, 5.0, 18.0, 16.0], score=0.91),
                _instance("s2t_2", [19.0, 6.0, 28.0, 17.0], label="suv", score=0.85),
                _instance("s2t_3", [1.0, 1.0, 8.0, 8.0], label="person", score=0.8),
            ],
        ),
    }

    monkeypatch.setattr(
        executor,
        "_services",
        lambda: SimpleNamespace(frames=[(idx, 0.1 * (idx - 1), frame) for idx in range(1, 6)]),
    )
    monkeypatch.setattr(
        executor,
        "_detect_objects_in_frame",
        lambda frame, frame_idx, timestamp: frame_detections[frame_idx].model_copy(deep=True),
    )

    result = executor._run_object_detect()

    first_path = Path(result["outputs"]["frames_dir"]) / "frame_000001" / "detections.json"
    summary_path = Path(result["outputs"]["summary"])
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(first_payload["instances"]) == 3
    assert [inst["object_id"] for inst in first_payload["instances"]] == [
        "obj_000001",
        "obj_000002",
        "obj_000003",
    ]
    assert summary_payload["frames"][0]["instance_count"] == 3
