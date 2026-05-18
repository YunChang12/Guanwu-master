from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

from guanwu.video.core.schema import Event, ObjectNode, RelationEdge

_BACKGROUND_LABEL_TOKENS = (
    "grass",
    "road",
    "ground",
    "floor",
    "wall",
    "ceiling",
    "track",
    "railing",
    "fence",
    "barrier",
)
_MIN_COLLISION_VOLUME = 5e-4


class EventEngine:
    """Detects lifecycle and relation-transition events."""

    def __init__(self) -> None:
        self._seen_objects: set[str] = set()
        self._prev_relation_keys: set[str] = set()
        self._event_counter = 0

    def infer(
        self,
        objects: list[ObjectNode],
        relations: list[RelationEdge],
        timestamp: float,
        removed_object_ids: Iterable[str] = (),
    ) -> list[Event]:
        events: list[Event] = []
        objects_by_id = {o.object_id: o for o in objects}

        current_ids = {o.object_id for o in objects}
        new_ids = current_ids - self._seen_objects
        for object_id in sorted(new_ids):
            events.append(self._event("appeared", timestamp, [object_id], [], 0.9))

        for object_id in sorted(set(removed_object_ids)):
            events.append(self._event("disappeared", timestamp, [object_id], [], 0.8))

        current_rel = {self._rel_key(r) for r in relations}
        started = current_rel - self._prev_relation_keys
        ended = self._prev_relation_keys - current_rel

        for rel_key in started:
            sub, pred, obj = rel_key.split("|")
            if pred == "contact_with" and self._allow_collision(objects_by_id.get(sub), objects_by_id.get(obj)):
                events.append(self._event("collision", timestamp, [sub], [obj], 0.65))

        for rel_key in ended:
            sub, pred, obj = rel_key.split("|")
            if pred == "on":
                held = any(k.startswith(f"{sub}|contact_with|") for k in current_rel)
                if held:
                    events.append(self._event("picked_up", timestamp, [sub], [obj], 0.8))

        for rel_key in started:
            sub, pred, obj = rel_key.split("|")
            if pred == "on":
                events.append(
                    self._event("placed_on", timestamp, [sub], [obj], 0.78, payload={"to_state": obj})
                )

        self._seen_objects = current_ids
        self._prev_relation_keys = current_rel
        return events

    def _event(
        self,
        event_type: str,
        timestamp: float,
        actors: list[str],
        targets: list[str],
        confidence: float,
        payload: dict | None = None,
    ) -> Event:
        self._event_counter += 1
        return Event(
            event_id=f"evt_{self._event_counter:06d}",
            type=event_type,
            timestamp=timestamp,
            actors=actors,
            targets=targets,
            payload=payload or {},
            confidence=confidence,
        )

    def _rel_key(self, relation: RelationEdge) -> str:
        return f"{relation.subject_id}|{relation.predicate}|{relation.object_id}"

    def _allow_collision(self, subject: ObjectNode | None, target: ObjectNode | None) -> bool:
        if subject is None or target is None:
            return False
        if self._is_background_like(subject) or self._is_background_like(target):
            return False
        return self._object_volume(subject) >= _MIN_COLLISION_VOLUME and self._object_volume(target) >= _MIN_COLLISION_VOLUME

    def _is_background_like(self, obj: ObjectNode) -> bool:
        label = str(obj.label or "").strip().lower()
        return any(token in label for token in _BACKGROUND_LABEL_TOKENS)

    def _object_volume(self, obj: ObjectNode) -> float:
        if not _valid_vec3(obj.geometry.scale_3d):
            return 0.0
        sx, sy, sz = obj.geometry.scale_3d
        return max(float(sx), 0.0) * max(float(sy), 0.0) * max(float(sz), 0.0)


def _valid_vec3(values: Any) -> bool:
    if not isinstance(values, list) or len(values) < 3:
        return False
    try:
        return all(math.isfinite(float(v)) for v in values[:3])
    except Exception:
        return False
