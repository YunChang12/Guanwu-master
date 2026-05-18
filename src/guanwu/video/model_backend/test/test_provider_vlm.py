"""
Unit tests for VLM providers.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from guanwu.video.model_backend.config import VLMProviderConfig
from guanwu.video.model_backend.test.conftest import extract_first_frame_b64
from guanwu.video.model_backend.providers.vlm import DisabledVLMProvider, EmbeddedVLMProvider, build_vlm_provider
from guanwu.video.model_backend.schemas import FrameDetectionsModel


def _make_detections() -> FrameDetectionsModel:
    return FrameDetectionsModel(frame_idx=1, timestamp=0.0, instances=[])


def _object(object_id: str = "obj_001") -> dict:
    return {"object_id": object_id, "label": "cup", "geometry": {"scale_3d": [0.1, 0.1, 0.1]}}


class TestDisabledVLMProvider:
    def test_disabled(self) -> None:
        provider = DisabledVLMProvider()
        assert provider.mode == "disabled"
        assert provider.infer_object_physics_priors(_make_detections(), []) == {}
        assert provider.infer_object_physics_priors(_make_detections(), [_object()]) == {}
        assert provider.discover_movable_object_categories("dummy_b64") == []
        assert provider.discover_movable_object_categories("") == []


class TestEmbeddedVLMProviderNoKey:
    def _make_cfg(self, backend: str = "mock", api_key: str | None = None) -> VLMProviderConfig:
        return VLMProviderConfig(mode="embedded", backend=backend, api_key=api_key)

    def test_mode_is_embedded(self) -> None:
        assert EmbeddedVLMProvider(self._make_cfg()).mode == "embedded"

    def test_mock_backend_raises(self) -> None:
        provider = EmbeddedVLMProvider(self._make_cfg(backend="mock"))
        with pytest.raises(NotImplementedError):
            provider.infer_object_physics_priors(_make_detections(), [])
        dummy = base64.b64encode(b"not-real").decode()
        with pytest.raises(NotImplementedError):
            provider.discover_movable_object_categories(dummy)

    def test_api_backend_without_key_raises(self, video_path: Path) -> None:
        provider = EmbeddedVLMProvider(self._make_cfg(backend="api", api_key=None))
        with pytest.raises(NotImplementedError):
            provider.discover_movable_object_categories("any_b64")

        first_frame_b64 = extract_first_frame_b64(video_path)
        with pytest.raises(NotImplementedError):
            provider.discover_movable_object_categories(first_frame_b64)


class TestBuildVLMProvider:
    def test_disabled_and_unknown_mode(self) -> None:
        assert isinstance(build_vlm_provider(VLMProviderConfig(mode="disabled")), DisabledVLMProvider)
        assert isinstance(build_vlm_provider(VLMProviderConfig(mode="no_such_mode")), DisabledVLMProvider)

    def test_embedded_mode(self) -> None:
        cfg = VLMProviderConfig(mode="embedded", backend="mock")
        assert isinstance(build_vlm_provider(cfg), EmbeddedVLMProvider)
