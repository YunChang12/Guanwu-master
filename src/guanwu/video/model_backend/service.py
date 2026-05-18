from __future__ import annotations

from guanwu.video.model_backend.config import ProviderSettings
from guanwu.video.model_backend.providers.registry import ProviderRegistry
from guanwu.video.model_backend.schemas import (
    DetectorStatusResponse,
    FrameDetectionRequest,
    FrameDetectionResponse,
    FrameDetectionsModel,
    HealthResponse,
    MovableObjectDiscoveryRequest,
    MovableObjectDiscoveryResponse,
    ObjectMeshReconstructionRequest,
    ObjectMeshReconstructionResponse,
    ObjectPhysicsPriorRequest,
    ObjectPhysicsPriorResponse,
    ProviderCheck,
    ProviderStatusResponse,
    ReadyResponse,
    SetPromptsRequest,
    SetPromptsResponse,
)


class ModelBackendService:
    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.providers = ProviderRegistry(settings)

    def health(self) -> HealthResponse:
        return HealthResponse()

    def ready(self) -> ReadyResponse:
        checks = [ProviderCheck.model_validate(x) for x in self.providers.checks()]
        return ReadyResponse(ready=all(c.ok for c in checks), checks=checks)

    def provider_status(self) -> ProviderStatusResponse:
        checks = [ProviderCheck.model_validate(x) for x in self.providers.checks()]
        return ProviderStatusResponse(providers=checks)

    def set_object_detection_prompts(self, req: SetPromptsRequest) -> SetPromptsResponse:
        prompts = self.providers.sam3.set_object_detection_prompts(req.prompts)
        return SetPromptsResponse(prompts=prompts)

    def get_object_detection_prompts(self) -> SetPromptsResponse:
        return SetPromptsResponse(prompts=self.providers.sam3.get_object_detection_prompts())

    def detector_status(self) -> DetectorStatusResponse:
        return DetectorStatusResponse(status=self.providers.sam3.detector_status())

    def detect_objects_in_frame(self, req: FrameDetectionRequest) -> FrameDetectionResponse:
        try:
            self.providers.sam3.set_object_detection_prompts(req.prompts)
            detections = self.providers.sam3.detect_objects_in_frame(
                frame_idx=req.frame_idx,
                timestamp=req.timestamp,
                image_b64=req.image_b64,
            )
            return FrameDetectionResponse(detections=detections)
        except Exception:
            empty = FrameDetectionsModel(
                frame_idx=req.frame_idx, timestamp=req.timestamp, instances=[]
            )
            return FrameDetectionResponse(detections=empty)

    def reconstruct_object_meshes(self, req: ObjectMeshReconstructionRequest) -> ObjectMeshReconstructionResponse:
        try:
            meshes = self.providers.sam3d.reconstruct_object_meshes(req.detections, req.objects)
            return ObjectMeshReconstructionResponse(meshes=meshes)
        except Exception:
            return ObjectMeshReconstructionResponse(meshes={})

    def infer_object_physics_priors(self, req: ObjectPhysicsPriorRequest) -> ObjectPhysicsPriorResponse:
        try:
            priors = self.providers.vlm.infer_object_physics_priors(
                req.detections, req.objects
            )
            return ObjectPhysicsPriorResponse(priors=priors)
        except Exception:
            return ObjectPhysicsPriorResponse(priors={})

    def discover_movable_object_categories(self, req: MovableObjectDiscoveryRequest) -> MovableObjectDiscoveryResponse:
        try:
            categories = self.providers.vlm.discover_movable_object_categories(
                req.image_b64
            )
            return MovableObjectDiscoveryResponse(categories=categories)
        except Exception:
            return MovableObjectDiscoveryResponse(categories=[])
