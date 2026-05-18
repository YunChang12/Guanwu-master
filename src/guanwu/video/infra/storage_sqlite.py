from __future__ import annotations

from collections import deque

from guanwu.video.core.schema import Event, ObjectNode, RelationEdge, WorldMetadata, WorldState


class WorldStore:
    """
    In-memory world store.

    Persistence has been intentionally removed: runtime state only lives for the
    current process lifetime.
    """

    def __init__(self, world_id: str = "scene_main") -> None:
        self.world_id = world_id
        self._objects: dict[str, ObjectNode] = {}
        self._relations: dict[str, RelationEdge] = {}
        self._events_recent: deque[Event] = deque(maxlen=128)

    def upsert_objects(self, objects: list[ObjectNode], timestamp: float) -> None:
        for obj in objects:
            self._objects[obj.object_id] = obj

    def delete_objects(self, object_ids: list[str]) -> None:
        for obj_id in object_ids:
            self._objects.pop(obj_id, None)

    def replace_relations(self, relations: list[RelationEdge], timestamp: float) -> None:
        self._relations = {rel.edge_id: rel for rel in relations}

    def append_events(self, events: list[Event]) -> None:
        for evt in events:
            self._events_recent.append(evt)

    def get_object(self, object_id: str) -> ObjectNode | None:
        return self._objects.get(object_id)

    def get_objects(self) -> list[ObjectNode]:
        return list(self._objects.values())

    def get_relations(self, predicate: str | None = None) -> list[RelationEdge]:
        relations = list(self._relations.values())
        if predicate is None:
            return relations
        return [rel for rel in relations if rel.predicate == predicate]

    def get_events_since(self, since: float | None = None) -> list[Event]:
        if since is None:
            return list(self._events_recent)
        return [evt for evt in self._events_recent if evt.timestamp >= since]

    def build_world_state(self, timestamp: float, sim_time: float, source_video_time: float) -> WorldState:
        return WorldState(
            world_id=self.world_id,
            timestamp=timestamp,
            objects=self.get_objects(),
            relations=self.get_relations(),
            events_recent=list(self._events_recent),
            metadata=WorldMetadata(sim_time=sim_time, source_video_time=source_video_time),
        )
