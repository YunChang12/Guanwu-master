"""
Unit tests for SAM3D reconstruction providers.
"""
from __future__ import annotations

import pytest

from guanwu.video.model_backend.config import SAM3DProviderConfig
from guanwu.video.model_backend.providers.sam3d import DisabledSAM3DProvider, EmbeddedSAM3DProvider, build_sam3d_provider
from guanwu.video.model_backend.schemas import FrameDetectionsModel


def _make_detections(frame_idx: int = 1) -> FrameDetectionsModel:
    return FrameDetectionsModel(frame_idx=frame_idx, timestamp=0.0, instances=[])


def _make_object(object_id: str = "obj_001") -> dict:
    return {
        "object_id": object_id,
        "track_id": "trk_1",
        "segment_kind": "object",
        "geometry": {"bbox_2d": [0.0, 0.0, 100.0, 100.0]},
    }


class TestDisabledSAM3DProvider:
    def test_disabled(self) -> None:
        provider = DisabledSAM3DProvider()
        assert provider.mode == "disabled"
        assert provider.reconstruct_object_meshes(_make_detections(), []) == {}
        assert provider.reconstruct_object_meshes(_make_detections(), [_make_object("a"), _make_object("b")]) == {}


class TestEmbeddedSAM3DProviderNoCommand:
    def _make_cfg(self, command: str = "") -> SAM3DProviderConfig:
        return SAM3DProviderConfig(
            mode="embedded",
            backend="command",
            object_command=command,
            body_command=command,
        )

    def test_mode_is_embedded(self) -> None:
        assert EmbeddedSAM3DProvider(self._make_cfg()).mode == "embedded"

    def test_missing_command_raises(self) -> None:
        provider = EmbeddedSAM3DProvider(self._make_cfg(command=""))
        with pytest.raises(RuntimeError, match="Missing SAM3D command"):
            provider.reconstruct_object_meshes(_make_detections(), [_make_object()])

    def test_nonexistent_command_raises(self) -> None:
        provider = EmbeddedSAM3DProvider(self._make_cfg(command="__does_not_exist_cmd__"))
        with pytest.raises((RuntimeError, FileNotFoundError)):
            provider.reconstruct_object_meshes(_make_detections(), [_make_object()])


class TestBuildSAM3DProvider:
    def test_disabled_and_unknown_mode(self) -> None:
        assert isinstance(build_sam3d_provider(SAM3DProviderConfig(mode="disabled")), DisabledSAM3DProvider)
        assert isinstance(build_sam3d_provider(SAM3DProviderConfig(mode="nonexistent")), DisabledSAM3DProvider)

    def test_embedded_mode(self) -> None:
        cfg = SAM3DProviderConfig(mode="embedded", backend="command")
        assert isinstance(build_sam3d_provider(cfg), EmbeddedSAM3DProvider)
