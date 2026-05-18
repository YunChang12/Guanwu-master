from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectedInstanceModel(BaseModel):
    mask_ref: str
    bbox: list[float]
    track_id: str
    concept_label: str
    segment_kind: str = "object"
    score: float


class FrameDetectionsModel(BaseModel):
    frame_idx: int
    timestamp: float
    instances: list[DetectedInstanceModel] = Field(default_factory=list)
    image_b64: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ProviderCheck(BaseModel):
    name: str
    mode: str
    ok: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    ready: bool
    checks: list[ProviderCheck] = Field(default_factory=list)


class ProviderStatusResponse(BaseModel):
    providers: list[ProviderCheck] = Field(default_factory=list)


class FrameDetectionRequest(BaseModel):
    frame_idx: int
    timestamp: float
    prompts: list[str]
    image_b64: str | None = None


class FrameDetectionResponse(BaseModel):
    detections: FrameDetectionsModel


class ObjectMeshReconstructionRequest(BaseModel):
    detections: FrameDetectionsModel
    objects: list[dict[str, Any]] = Field(default_factory=list)


class ObjectMeshReconstructionResponse(BaseModel):
    meshes: dict[str, dict] = Field(default_factory=dict)


class ObjectPhysicsPriorRequest(BaseModel):
    detections: FrameDetectionsModel
    objects: list[dict[str, Any]] = Field(default_factory=list)
    sam3d_meshes: dict[str, dict] = Field(default_factory=dict)


class ObjectPhysicsPriorResponse(BaseModel):
    priors: dict[str, dict] = Field(default_factory=dict)


class MovableObjectDiscoveryRequest(BaseModel):
    image_b64: str


class MovableObjectDiscoveryResponse(BaseModel):
    categories: list[str] = Field(default_factory=list)


class SetPromptsRequest(BaseModel):
    prompts: list[str] = Field(default_factory=list)


class SetPromptsResponse(BaseModel):
    prompts: list[str] = Field(default_factory=list)


class DetectorStatusResponse(BaseModel):
    status: dict = Field(default_factory=dict)


