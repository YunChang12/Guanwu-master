from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from guanwu.video.clients.zaiwu import normalize_provider_mode
from guanwu.video.core.config import SPWMSettings, VLMConfig, _to_toml, load_settings

SYSTEM_MANAGED_VLM_FIELDS = set(VLMConfig.model_fields)


class ProjectMetadata(BaseModel):
    project_id: str
    name: str
    input_video: str
    root_dir: str
    provider_mode: str = "mock"  # mock | zaiwu
    video_copy_mode: str = "copy"  # copy | link


class ProjectConfig(BaseModel):
    project: ProjectMetadata
    settings: SPWMSettings = Field(default_factory=SPWMSettings)
    workspace: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)


def create_project_config(
    project_id: str,
    name: str,
    input_video: str,
    root_dir: str,
    provider_mode: str = "mock",
    video_copy_mode: str = "copy",
    workspace: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> ProjectConfig:
    settings, _ = load_settings()
    return ProjectConfig(
        project=ProjectMetadata(
            project_id=project_id,
            name=name,
            input_video=str(Path(input_video).expanduser().resolve()),
            root_dir=str(Path(root_dir).expanduser().resolve()),
            provider_mode=normalize_provider_mode(provider_mode),
            video_copy_mode=video_copy_mode,
        ),
        settings=settings,
        workspace=workspace or {},
        payload=payload or {},
    )


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = _normalize_empty_values(tomllib.load(handle))
    config = ProjectConfig.model_validate(raw)
    _apply_system_vlm_settings(config)
    return config


def save_project_config(config: ProjectConfig, path: str | Path) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_toml(_persisted_project_payload(config)), encoding="utf-8")
    return out_path


def project_config_payload(config: ProjectConfig) -> dict[str, Any]:
    return _persisted_project_payload(config)


def _apply_system_vlm_settings(config: ProjectConfig) -> None:
    system_settings, _ = load_settings()
    config.settings.vlm = system_settings.vlm.model_copy(deep=True)


def _persisted_project_payload(config: ProjectConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return payload
    vlm = settings.get("vlm")
    if not isinstance(vlm, dict):
        return payload
    for field in SYSTEM_MANAGED_VLM_FIELDS:
        vlm.pop(field, None)
    if not vlm:
        settings.pop("vlm", None)
    return payload


def _normalize_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_empty_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_empty_values(item) for item in value]
    if value == "":
        return None
    return value
