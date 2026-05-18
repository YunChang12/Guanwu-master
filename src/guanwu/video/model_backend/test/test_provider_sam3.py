"""
Unit tests for SAM3 perception providers.
These tests do NOT require a real YOLO model or GPU.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from guanwu.video.model_backend.config import SAM3ProviderConfig
from guanwu.video.model_backend.providers.sam3 import DisabledSAM3Provider, EmbeddedSAM3Provider, build_sam3_provider
from guanwu.video.model_backend.test.conftest import extract_first_frame_b64


class TestDisabledSAM3Provider:
    def setup_method(self) -> None:
        self.provider = DisabledSAM3Provider()

    def test_mode_is_disabled(self) -> None:
        assert self.provider.mode == "disabled"

    def test_detect_objects_in_frame_returns_empty(self) -> None:
        detections = self.provider.detect_objects_in_frame(frame_idx=1, timestamp=0.5)
        assert detections.frame_idx == 1
        assert detections.timestamp == 0.5
        assert detections.instances == []

        dummy = base64.b64encode(b"data").decode()
        detections2 = self.provider.detect_objects_in_frame(frame_idx=2, timestamp=1.0, image_b64=dummy)
        assert detections2.instances == []

    def test_prompts_always_empty(self) -> None:
        assert self.provider.get_object_detection_prompts() == []
        assert self.provider.set_object_detection_prompts(["cup", "plate"]) == []

    def test_status(self) -> None:
        status = self.provider.detector_status()
        assert status["backend"] == "disabled"
        assert status["ready"] is False

    def test_first_frame_path_none(self) -> None:
        assert self.provider.get_first_frame_path() is None


class TestEmbeddedSAM3ProviderNoModel:
    def test_mode_is_embedded(self) -> None:
        cfg = SAM3ProviderConfig(mode="embedded", backend="mock")
        assert EmbeddedSAM3Provider(cfg).mode == "embedded"

    def test_mock_backend_raises(self) -> None:
        cfg = SAM3ProviderConfig(mode="embedded", backend="mock")
        provider = EmbeddedSAM3Provider(cfg)
        with pytest.raises(NotImplementedError, match="mock.*removed"):
            provider.detect_objects_in_frame(frame_idx=1, timestamp=0.0, image_b64=None)

        dummy = base64.b64encode(b"not-a-real-jpeg").decode()
        with pytest.raises(NotImplementedError):
            provider.detect_objects_in_frame(frame_idx=1, timestamp=0.0, image_b64=dummy)

    def test_ultralytics_missing_weights_raises(self) -> None:
        cfg = SAM3ProviderConfig(
            mode="embedded",
            backend="ultralytics",
            yolo_weights="/nonexistent/path/model.pt",
        )
        with pytest.raises(RuntimeError, match="requires existing yolo_weights"):
            EmbeddedSAM3Provider(cfg)

    def test_prompts(self) -> None:
        cfg = SAM3ProviderConfig(mode="embedded", backend="mock")
        provider = EmbeddedSAM3Provider(cfg)
        result = provider.set_object_detection_prompts(["cup", "  PLATE  ", ""])
        assert "" not in result
        assert all(p == p.strip().lower() for p in result)

        cfg2 = SAM3ProviderConfig(mode="embedded", backend="mock", prompts=["bowl"])
        assert isinstance(EmbeddedSAM3Provider(cfg2).get_object_detection_prompts(), list)


class TestBuildSAM3Provider:
    def test_disabled_and_unknown_mode(self) -> None:
        assert isinstance(build_sam3_provider(SAM3ProviderConfig(mode="disabled")), DisabledSAM3Provider)
        assert isinstance(build_sam3_provider(SAM3ProviderConfig(mode="totally_unknown")), DisabledSAM3Provider)

    def test_embedded_mode(self) -> None:
        cfg = SAM3ProviderConfig(mode="embedded", backend="mock")
        assert isinstance(build_sam3_provider(cfg), EmbeddedSAM3Provider)


class TestVideoFrameExtraction:
    def test_first_frame(self, video_path: Path) -> None:
        assert video_path.exists()
        assert video_path.suffix == ".mp4"

        first_frame_b64 = extract_first_frame_b64(video_path)
        assert isinstance(first_frame_b64, str)
        assert len(first_frame_b64) > 0

        raw = base64.b64decode(first_frame_b64)
        assert raw[:2] == b"\xff\xd8"
