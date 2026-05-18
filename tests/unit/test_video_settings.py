from __future__ import annotations

from pathlib import Path

import pytest

from guanwu.video.core.config import SPWMSettings, load_settings, save_settings


def test_spwm_settings_default_detector_backend_is_seg2track() -> None:
    settings = SPWMSettings()
    assert settings.zaiwu.object_detection_backend == "seg2track_sam2"


def test_load_settings_migrates_vlm_from_legacy_model_backend_config(tmp_path: Path) -> None:
    config_path = tmp_path / "video.config.toml"
    legacy_path = tmp_path / "video.model-backend.config.toml"
    legacy_path.write_text(
        """
[vlm]
mode = "embedded"
backend = "api"
api_key = "legacy-key"
base_url = "https://legacy.example/v1"
model = "legacy-model"
command_template = "legacy-vlm"
service = { base_url = "http://127.0.0.1:9103", timeout_sec = 45.0 }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    settings, loaded_path = load_settings(config_path=config_path)

    assert loaded_path == config_path
    assert settings.vlm.mode == "embedded"
    assert settings.vlm.backend == "api"
    assert settings.vlm.api_key == "legacy-key"
    assert settings.vlm.base_url == "https://legacy.example/v1"
    assert settings.vlm.model == "legacy-model"
    assert settings.vlm.command_template == "legacy-vlm"
    assert settings.vlm.service.base_url == "http://127.0.0.1:9103"
    assert settings.vlm.service.timeout_sec == 45.0
    assert settings.vlm.max_retries == 3


def test_save_settings_round_trips_blank_optional_values(tmp_path: Path) -> None:
    config_path = tmp_path / "video.config.toml"
    settings = SPWMSettings()
    settings.vlm.api_key = "system-key"

    save_settings(settings, config_path=config_path, create_backup=False)
    loaded, loaded_path = load_settings(config_path=config_path)

    assert loaded_path == config_path
    assert loaded.vlm.api_key == "system-key"
    assert loaded.pit.known_object_size_m is None
    assert loaded.runtime.video_source is None


def test_settings_reject_legacy_synthetic_heuristic_providers() -> None:
    with pytest.raises(ValueError, match="pit.camera_provider must be one of"):
        SPWMSettings.model_validate(
            {
                "pit": {
                    "camera_provider": "synthetic",
                    "depth_provider": "heuristic",
                }
            }
        )


def test_settings_require_wildgs_camera_and_depth_together() -> None:
    with pytest.raises(ValueError, match="must both be 'wildgs' together"):
        SPWMSettings.model_validate(
            {
                "pit": {
                    "camera_provider": "wildgs",
                    "depth_provider": "zaiwu_depth_anything3",
                }
            }
        )
