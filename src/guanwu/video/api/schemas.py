from __future__ import annotations

from pydantic import BaseModel, Field


class PromptsRequest(BaseModel):
    prompts: list[str] = Field(default_factory=list)


class BootstrapRequest(BaseModel):
    hints: list[str] = Field(default_factory=list)


class SimStepRequest(BaseModel):
    steps: int = 1


class HypothesisRequest(BaseModel):
    edit: dict


class HypothesisRevertRequest(BaseModel):
    snapshot_id: str | None = None
    object_id: str | None = None


class PIT2IsaacExportRequest(BaseModel):
    mode: str | None = None


class ConfigUpdateRequest(BaseModel):
    settings: dict = Field(default_factory=dict)
    reload_runtime: bool = True
    create_backup: bool = True


__all__ = [
    "PromptsRequest",
    "BootstrapRequest",
    "SimStepRequest",
    "HypothesisRequest",
    "HypothesisRevertRequest",
    "PIT2IsaacExportRequest",
    "ConfigUpdateRequest",
]
