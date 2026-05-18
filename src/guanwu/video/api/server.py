from __future__ import annotations

import os
from threading import Lock

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from guanwu.video.app.planner import planner_mock_next_action as planner_next_action
from guanwu.video.app.runtime import WorldRuntime
from guanwu.video.api.schemas import (
    BootstrapRequest,
    ConfigUpdateRequest,
    HypothesisRequest,
    HypothesisRevertRequest,
    PIT2IsaacExportRequest,
    PromptsRequest,
    SimStepRequest,
)
from guanwu.video.api.ws_hub import ConnectionHub, VALID_TOPICS
from guanwu.video.core.config import SPWMSettings, save_settings
from guanwu.video.core.schema import WorldState


app = FastAPI(title="SPWM Agent API", version="0.1.0")


def _required_session_output_root() -> str:
    root = (os.getenv("SPWM_SESSION_OUTPUT_ROOT") or "").strip()
    if not root:
        raise RuntimeError("SPWM_SESSION_OUTPUT_ROOT is required to start guanwu.video API runtime")
    return root


runtime = WorldRuntime(session_output_root=_required_session_output_root())
runtime_lock = Lock()


hub = ConnectionHub()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _reload_runtime() -> None:
    global runtime
    with runtime_lock:
        runtime = WorldRuntime(session_output_root=_required_session_output_root())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/world/state", response_model=WorldState)
def get_world_state() -> WorldState:
    timestamp, sim_time, source_video_time = runtime.time_sync.tick(0.0, 0.0)
    return runtime.store.build_world_state(timestamp, sim_time, source_video_time)


@app.get("/world/objects/{object_id}")
def get_world_object(object_id: str) -> dict:
    obj = runtime.store.get_object(object_id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
    return {"object": obj, "isaac_prim_path": runtime.isaac_sync.get_prim_path(object_id)}


@app.get("/world/relations")
def get_world_relations(predicate: str | None = Query(default=None)) -> dict:
    return {"relations": runtime.store.get_relations(predicate=predicate)}


@app.get("/world/events")
def get_world_events(since: float | None = Query(default=None)) -> dict:
    return {"events": runtime.store.get_events_since(since=since)}


@app.get("/pit/trajectories")
def get_pit_trajectories() -> dict:
    return runtime.estimator.pit_snapshot()


@app.post("/world/prompts")
def set_world_prompts(req: PromptsRequest) -> dict:
    runtime.object_detector.set_object_detection_prompts(req.prompts)
    return {"prompts": runtime.object_detector.get_object_detection_prompts()}


@app.post("/world/track/bootstrap")
def bootstrap_tracking(req: BootstrapRequest) -> dict:
    return {
        "status": "accepted",
        "message": "Bootstrap hints recorded for operator workflow.",
        "hints": req.hints,
    }


@app.post("/sim/step")
async def sim_step(req: SimStepRequest) -> dict:
    steps = max(1, min(req.steps, 120))
    latest = None
    for _ in range(steps):
        latest = runtime.step_once()

    if latest is not None:
        await hub.publish("object.updated", {"objects": len(latest.objects), "timestamp": latest.timestamp})
        await hub.publish("relation.changed", {"relations": len(latest.relations), "timestamp": latest.timestamp})
        for evt in latest.events_recent[-8:]:
            await hub.publish("event.detected", evt.model_dump())
            if evt.type == "collision":
                await hub.publish("sim.collision", evt.model_dump())

    return {
        "status": "ok",
        "steps": steps,
        "world": latest,
        "sync_report": runtime.last_sync_report,
    }


@app.get("/sim/status")
def sim_status() -> dict:
    lifecycle = {
        "occluded_ttl_frames": runtime.occluded_ttl_frames,
        "removal_ttl_frames": runtime.removal_ttl_frames,
        "tracked_objects": len(runtime.object_registry),
        "missed_objects": len([1 for v in runtime.object_missed_frames.values() if v > 0]),
    }
    return {
        "config_path": str(runtime.config_path),
        "settings": runtime.settings.model_dump(),
        "sync": runtime.last_sync_report,
        "lifecycle": lifecycle,
        "detector": runtime.object_detector.detector_status(),
        "object_detector_init_error": runtime.object_detector_init_error,
    }


@app.get("/config")
def get_config() -> dict:
    return {
        "config_path": str(runtime.config_path),
        "settings": runtime.settings.model_dump(),
    }


@app.put("/config")
def update_config(req: ConfigUpdateRequest) -> dict:
    try:
        settings = SPWMSettings.model_validate(req.settings)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid settings: {exc}") from exc

    try:
        path = save_settings(settings, config_path=runtime.config_path, create_backup=req.create_backup)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {exc}") from exc

    if req.reload_runtime:
        _reload_runtime()

    return {
        "status": "ok",
        "config_path": str(path),
        "reloaded": req.reload_runtime,
        "settings": runtime.settings.model_dump(),
    }


@app.post("/world/hypothesis")
async def set_world_hypothesis(req: HypothesisRequest) -> dict:
    try:
        obj, evt, snapshot_id = runtime.apply_hypothesis(req.edit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Object not found: {req.edit.get('object_id')}") from None

    await hub.publish("object.updated", {"object_id": obj.object_id, "source": "hypothesis"})
    await hub.publish("event.detected", evt.model_dump())
    return {
        "status": "applied",
        "object": obj,
        "event": evt,
        "snapshot_id": snapshot_id,
        "sync_report": runtime.last_sync_report,
    }


@app.post("/world/hypothesis/revert")
async def revert_world_hypothesis(req: HypothesisRevertRequest) -> dict:
    try:
        obj, evt, snapshot_id = runtime.revert_hypothesis(snapshot_id=req.snapshot_id, object_id=req.object_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError:
        key = req.snapshot_id or req.object_id
        raise HTTPException(status_code=404, detail=f"Snapshot/object not found: {key}") from None

    await hub.publish("object.updated", {"object_id": obj.object_id, "source": "hypothesis_revert"})
    await hub.publish("event.detected", evt.model_dump())
    return {
        "status": "reverted",
        "object": obj,
        "event": evt,
        "snapshot_id": snapshot_id,
        "sync_report": runtime.last_sync_report,
    }


@app.post("/pit2isaac/export")
def export_pit2isaac(req: PIT2IsaacExportRequest) -> dict:
    try:
        return {"status": "ok", "export": runtime.export_pit2isaac(mode_override=req.mode)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pit2isaac export failed: {exc}") from exc


@app.get("/planner/mock/next_action")
def planner_mock_next_action(goal: str | None = Query(default=None)) -> dict:
    return planner_next_action(runtime.store, goal)


@app.websocket("/ws/{topic}")
async def websocket_topic(ws: WebSocket, topic: str) -> None:
    if topic not in VALID_TOPICS:
        await ws.accept()
        await ws.send_json({"error": f"Unsupported topic: {topic}"})
        await ws.close(code=1008)
        return

    await hub.connect(topic, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(topic, ws)
