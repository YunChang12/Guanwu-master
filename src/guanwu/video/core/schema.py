from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Visibility = Literal["visible", "occluded", "lost"]
InteractionState = Literal["idle", "held", "moving", "contact"]
RelationPredicate = Literal["on", "in", "next_to", "holding", "approaching", "contact_with", "on_floor", "against_wall"]
RelationStatus = Literal["active", "ended"]
EventType = Literal[
    "appeared",
    "disappeared",
    "picked_up",
    "placed_on",
    "entered_region",
    "collision",
]


class Pose3D(BaseModel):
    position: list[float] | None = Field(default=None, min_length=3, max_length=3)
    orientation_quat: list[float] | None = Field(default=None, min_length=4, max_length=4)
    frame: str = "world"


class Geometry(BaseModel):
    bbox_2d: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0], min_length=4, max_length=4)
    mask_ref: str = ""
    pose_3d: Pose3D = Field(default_factory=Pose3D)
    scale_3d: list[float] | None = Field(default=None, min_length=3, max_length=3)
    shape_proxy: Literal["box", "sphere", "capsule", "cylinder", "mesh"] = "box"


class PhysicsState(BaseModel):
    is_dynamic: bool = True
    mass: float | None = None
    friction: float | None = None
    restitution: float | None = None
    velocity_linear: list[float] | None = Field(default=None, min_length=3, max_length=3)
    velocity_angular: list[float] | None = Field(default=None, min_length=3, max_length=3)


class SemanticState(BaseModel):
    category: str = "unknown"
    attributes: list[str] = Field(default_factory=list)


class AffordanceState(BaseModel):
    graspable: bool = False
    openable: bool = False
    supportable: bool = False
    pourable: bool = False


class ObjectStatus(BaseModel):
    visibility: Visibility = "visible"
    interaction_state: InteractionState = "idle"
    last_seen_ts: float = 0.0


class Provenance(BaseModel):
    sensor: str = "cam_front"
    frame_idx: int = 0
    model: str = "sam3"


class ObjectNode(BaseModel):
    object_id: str
    label: str
    label_source: str = "sam3_text_prompt"
    confidence: float = 0.0
    segment_kind: str = "object"  # "object" | "body"
    geometry: Geometry = Field(default_factory=Geometry)
    physics: PhysicsState = Field(default_factory=PhysicsState)
    semantic: SemanticState = Field(default_factory=SemanticState)
    affordance: AffordanceState = Field(default_factory=AffordanceState)
    state: ObjectStatus = Field(default_factory=ObjectStatus)
    provenance: Provenance = Field(default_factory=Provenance)


class RelationTemporal(BaseModel):
    start_ts: float
    end_ts: float | None = None
    status: RelationStatus = "active"


class RelationEvidence(BaseModel):
    source: str = "rule_engine"
    frame_range: list[int] = Field(default_factory=lambda: [0, 0], min_length=2, max_length=2)


class RelationEdge(BaseModel):
    edge_id: str
    subject_id: str
    predicate: RelationPredicate
    object_id: str
    confidence: float
    temporal: RelationTemporal
    evidence: RelationEvidence = Field(default_factory=RelationEvidence)


class Event(BaseModel):
    event_id: str
    type: EventType
    timestamp: float
    actors: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)
    confidence: float = 0.0


class WorldMetadata(BaseModel):
    coordinate_frame: str = "isaac_world"
    sim_time: float = 0.0
    source_video_time: float = 0.0


class WorldState(BaseModel):
    world_id: str = "scene_main"
    timestamp: float = 0.0
    objects: list[ObjectNode] = Field(default_factory=list)
    relations: list[RelationEdge] = Field(default_factory=list)
    events_recent: list[Event] = Field(default_factory=list)
    metadata: WorldMetadata = Field(default_factory=WorldMetadata)
