from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from guanwu.video.model_backend.config import load_settings
from guanwu.video.model_backend.schemas import (
    DetectorStatusResponse,
    FrameDetectionRequest,
    FrameDetectionResponse,
    HealthResponse,
    MovableObjectDiscoveryRequest,
    MovableObjectDiscoveryResponse,
    ObjectMeshReconstructionRequest,
    ObjectMeshReconstructionResponse,
    ObjectPhysicsPriorRequest,
    ObjectPhysicsPriorResponse,
    ProviderStatusResponse,
    ReadyResponse,
    SetPromptsRequest,
    SetPromptsResponse,
)
from guanwu.video.model_backend.service import ModelBackendService


def create_app(config_path: str | None = None) -> FastAPI:
    settings, _ = load_settings(config_path=config_path)
    service = ModelBackendService(settings)

    app = FastAPI(title="SPWM Model Backend", version="0.1.0")

    @app.exception_handler(Exception)
    async def custom_exception_handler(request, exc):
        # We catch exceptions to prevent FastAPI from dumping the raw giant base64 request into the console logger
        import logging
        from fastapi.responses import JSONResponse
        logger = logging.getLogger("spwm_agent.server")
        logger.exception(f"Internal Server Error for {request.url.path}")
        return JSONResponse(status_code=500, content={"message": str(exc)})

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.health()

    @app.get("/v1/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        return service.ready()

    @app.get("/v1/providers/status", response_model=ProviderStatusResponse)
    def provider_status() -> ProviderStatusResponse:
        return service.provider_status()

    @app.post("/v1/object-detection/prompts", response_model=SetPromptsResponse)
    def set_object_detection_prompts(req: SetPromptsRequest) -> SetPromptsResponse:
        return service.set_object_detection_prompts(req)

    @app.get("/v1/object-detection/prompts", response_model=SetPromptsResponse)
    def get_object_detection_prompts() -> SetPromptsResponse:
        return service.get_object_detection_prompts()

    @app.get("/v1/detector/status", response_model=DetectorStatusResponse)
    def detector_status() -> DetectorStatusResponse:
        return service.detector_status()

    @app.post("/v1/tasks/detect-objects-in-frame", response_model=FrameDetectionResponse)
    def detect_objects_in_frame(req: FrameDetectionRequest) -> FrameDetectionResponse:
        return service.detect_objects_in_frame(req)

    @app.post("/v1/tasks/reconstruct-object-meshes", response_model=ObjectMeshReconstructionResponse)
    def reconstruct_object_meshes(req: ObjectMeshReconstructionRequest) -> ObjectMeshReconstructionResponse:
        return service.reconstruct_object_meshes(req)

    @app.post("/v1/tasks/infer-object-physics-priors", response_model=ObjectPhysicsPriorResponse)
    def infer_object_physics_priors(req: ObjectPhysicsPriorRequest) -> ObjectPhysicsPriorResponse:
        return service.infer_object_physics_priors(req)

    @app.post("/v1/tasks/discover-movable-object-categories", response_model=MovableObjectDiscoveryResponse)
    def discover_movable_object_categories(req: MovableObjectDiscoveryRequest) -> MovableObjectDiscoveryResponse:
        return service.discover_movable_object_categories(req)

    return app

try:
    app = create_app(config_path=os.getenv("SPWM_MODEL_BACKEND_CONFIG"))
except Exception as exc:
    logging.getLogger(__name__).exception("Failed to create default model-backend app")
    app = FastAPI(title="SPWM Model Backend", version="0.1.0")

    @app.get("/v1/health")
    def health_fallback() -> dict[str, str]:
        return {"status": "error", "message": str(exc)}
