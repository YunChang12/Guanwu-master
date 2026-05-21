from __future__ import annotations

import io
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
