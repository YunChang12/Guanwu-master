from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import guanwu.video.project.executor as executor_module
from guanwu.video.project.executor import ProjectExecutor


def _depth_bytes(depth: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, depth)
    return buf.getvalue()


class _FakeGateway:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth
        self.uploads: list[tuple[str, Path]] = []
        self.jobs: list[tuple[str, str, dict]] = []
        self.downloads: list[tuple[str, str]] = []

    def upload_file(self, service_id: str, path: str | Path) -> str:
        self.uploads.append((service_id, Path(path)))
        return "uploads/clean_target_rgb_for_depth.mp4"

    def run_service_job(self, service_id: str, operation: str, *, payload: dict, timeout_sec: float | None = None) -> dict:
        _ = timeout_sec
        self.jobs.append((service_id, operation, dict(payload)))
        return {"output_file_id": "outputs/clean_depth.npy"}

    def download_bytes(self, service_id: str, file_id: str) -> bytes:
        self.downloads.append((service_id, file_id))
        return _depth_bytes(self.depth)


class _FakeRoadGateway:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask.astype(np.uint8)
        self.jobs: list[tuple[str, str, dict]] = []

    def run_service_job(
        self,
        service_id: str,
        operation: str,
        payload: dict | None = None,
        *,
        timeout_sec: float | None = None,
        **kwargs,
    ) -> dict:
        _ = (timeout_sec, kwargs)
        payload = dict(payload or {})
        self.jobs.append((service_id, operation, payload))
        return {
            "frame_idx": payload.get("frame_idx", 0),
            "instances": [
                {
                    "object_id": "road_1",
                    "label": "road roadway asphalt road",
                    "score": 0.91,
                    "bbox": [0, 0, int(self.mask.shape[1]), int(self.mask.shape[0])],
                    "mask": json.dumps(_encode_uncompressed_rle(self.mask)),
                }
            ],
        }


def _encode_uncompressed_rle(mask: np.ndarray) -> dict:
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        if int(value) == current:
            run += 1
            continue
        counts.append(run)
        current = int(value)
        run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def test_clean_background_depth_estimator_calls_depth_anything_video_job(monkeypatch, tmp_path: Path) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[..., 0] = 96
    image_path = tmp_path / "clean_target_rgb.png"
    cv2.imwrite(str(image_path), image)
    fake_gateway = _FakeGateway(np.full((1, 24, 32), 7.5, dtype=np.float32))
    monkeypatch.setattr(executor_module, "build_zaiwu_gateway_client", lambda settings: fake_gateway)

    executor = object.__new__(ProjectExecutor)
    executor.context = SimpleNamespace(
        config=SimpleNamespace(
            project=SimpleNamespace(provider_mode="zaiwu"),
            settings=SimpleNamespace(
                zaiwu=SimpleNamespace(
                    enabled=True,
                    depth_service="services.depth_anything3",
                    job_timeout_sec=30.0,
                )
            ),
        )
    )

    result = executor._estimate_clean_background_depth_with_zaiwu(image_path, output_dir=tmp_path)

    assert result is not None
    assert result["source"] == "depth_anything3_clean_target_rgb"
    assert fake_gateway.uploads[0][0] == "services.depth_anything3"
    assert fake_gateway.jobs[0] == (
        "services.depth_anything3",
        "estimate_from_video",
        {"video_file_id": "uploads/clean_target_rgb_for_depth.mp4", "sample_every_n": 1},
    )
    saved = np.load(result["depth_path"])
    assert saved.shape == (24, 32)
    assert float(saved[0, 0]) == 7.5


def test_semantic_road_estimator_calls_grounded_sam2_frame_job(monkeypatch, tmp_path: Path) -> None:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :] = (80, 90, 100)
    image_path = tmp_path / "clean_target_rgb.png"
    cv2.imwrite(str(image_path), image)
    road = np.zeros((24, 32), dtype=np.uint8)
    road[4:24, 6:28] = 1
    fake_gateway = _FakeRoadGateway(road)
    monkeypatch.setattr(executor_module, "build_zaiwu_gateway_client", lambda settings: fake_gateway)

    executor = object.__new__(ProjectExecutor)
    executor.context = SimpleNamespace(
        config=SimpleNamespace(
            project=SimpleNamespace(provider_mode="zaiwu"),
            settings=SimpleNamespace(
                zaiwu=SimpleNamespace(
                    enabled=True,
                    grounded_sam2_service="services.grounding_dino_sam2",
                    job_timeout_sec=30.0,
                    grounded_sam2=SimpleNamespace(box_threshold=0.25, text_threshold=0.2),
                )
            ),
        )
    )

    result = executor._estimate_semantic_road_with_zaiwu(image_path, frame_id=3, output_dir=tmp_path)

    assert result is not None
    assert result["source"] == "grounding_dino_sam2_clean_target_rgb"
    assert fake_gateway.jobs[0][0] == "services.grounding_dino_sam2"
    assert fake_gateway.jobs[0][1] == "gsam2_parse_frame"
    assert fake_gateway.jobs[0][2]["frame_idx"] == 3
    assert fake_gateway.jobs[0][2]["text_prompt"].startswith("road.")
    assert "box_threshold" not in fake_gateway.jobs[0][2]
    assert "text_threshold" not in fake_gateway.jobs[0][2]
    saved = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
    assert saved[10, 10]
    assert not saved[2, 2]


def test_grounded_sam2_road_mask_falls_back_to_any_decoded_instance_for_road_prompt() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:22, 4:29] = 1
    payload = {
        "instances": [
            {
                "object_id": "trk_f1_0",
                "label": "",
                "score": 0.38,
                "bbox": [4.0, 5.0, 29.0, 22.0],
                "mask": json.dumps(_encode_uncompressed_rle(mask)),
            }
        ]
    }

    road = ProjectExecutor._road_mask_from_grounded_sam2_payload(payload, (24, 32))

    assert road is not None
    assert road[10, 10]
    assert not road[2, 2]


def test_grounded_sam2_road_mask_uses_bbox_when_mask_payload_is_missing() -> None:
    payload = {
        "instances": [
            {
                "object_id": "trk_f1_0",
                "label": "road roadway asphalt road",
                "score": 0.38,
                "bbox": [4.2, 5.1, 29.0, 22.0],
            }
        ]
    }

    road = ProjectExecutor._road_mask_from_grounded_sam2_payload(payload, (24, 32))

    assert road is not None
    assert road[10, 10]
    assert not road[2, 2]


def test_single_frame_depth_video_is_readable(tmp_path: Path) -> None:
    image = np.zeros((18, 26, 3), dtype=np.uint8)
    image[:, :] = (10, 40, 90)
    image_path = tmp_path / "clean_target_rgb.png"
    video_path = tmp_path / "clean_target_rgb_for_depth.mp4"
    cv2.imwrite(str(image_path), image)

    ProjectExecutor._write_single_frame_depth_video(image_path, video_path)

    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok
    assert frame.shape[:2] == (18, 26)
