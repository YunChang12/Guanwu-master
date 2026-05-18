from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULT_CONFIG_PATH = Path.home() / ".guanwu" / "video.model-backend.config.toml"


class ProviderHTTPConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    timeout_sec: float = 30.0


class SAM3ProviderConfig(BaseModel):
    mode: str = "embedded"  # embedded | http | disabled
    backend: str = "mock"  # embedded mode only: mock | ultralytics
    yolo_weights: str | None = None
    confidence: float = 0.25
    frame_dump_dir: str | None = None
    prompts: list[str] = Field(default_factory=list)
    service: ProviderHTTPConfig = Field(
        default_factory=lambda: ProviderHTTPConfig(base_url="http://127.0.0.1:8101", timeout_sec=30.0)
    )


class SAM3DProviderConfig(BaseModel):
    mode: str = "embedded"  # embedded | http | disabled
    backend: str = "mock"  # embedded mode only: mock | command
    output_dir: str = "sam3d_meshes"
    object_command: str = "sam-3d-objects"
    body_command: str = "sam-3d-body"
    min_mesh_quality: float = 0.6
    service: ProviderHTTPConfig = Field(
        default_factory=lambda: ProviderHTTPConfig(base_url="http://127.0.0.1:8102", timeout_sec=30.0)
    )
    # Allow flat TOML keys http_base_url / http_timeout_sec as convenience aliases
    http_base_url: str | None = Field(default=None, exclude=True)
    http_timeout_sec: float | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _apply_flat_http_keys(self) -> "SAM3DProviderConfig":
        if self.http_base_url is not None:
            self.service.base_url = self.http_base_url
        if self.http_timeout_sec is not None:
            self.service.timeout_sec = self.http_timeout_sec
        return self



class VLMProviderConfig(BaseModel):
    mode: str = "embedded"  # embedded | http | disabled
    backend: str = "mock"  # embedded mode only: mock | command | api
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3.5-sonnet"
    command_template: str | None = None
    service: ProviderHTTPConfig = Field(
        default_factory=lambda: ProviderHTTPConfig(base_url="http://127.0.0.1:8103", timeout_sec=30.0)
    )


class ProviderSettings(BaseModel):
    sam3: SAM3ProviderConfig = Field(default_factory=SAM3ProviderConfig)
    sam3d: SAM3DProviderConfig = Field(default_factory=SAM3DProviderConfig)
    vlm: VLMProviderConfig = Field(default_factory=VLMProviderConfig)


def load_settings(config_path: str | Path | None = None) -> tuple[ProviderSettings, Path]:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _read_toml(path)
    settings = ProviderSettings.model_validate(raw)
    _normalize_paths(settings, path.parent)
    return settings, path


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _normalize_paths(settings: ProviderSettings, base_dir: Path) -> None:
    if settings.sam3.yolo_weights:
        settings.sam3.yolo_weights = _resolve_path(settings.sam3.yolo_weights, base_dir)
    if settings.sam3.frame_dump_dir:
        settings.sam3.frame_dump_dir = _resolve_path(settings.sam3.frame_dump_dir, base_dir)

    settings.sam3d.output_dir = _resolve_path(settings.sam3d.output_dir, base_dir)


def _resolve_path(value: str, base_dir: Path) -> str:
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())
