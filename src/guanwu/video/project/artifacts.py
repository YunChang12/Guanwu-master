from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


STAGE_ORDER = [
    "video.inspect",
    "frame.sample",
    "object.detect",
    "object.index",
    "object.attr",
    "geometry.lift",
    "mesh.reconstruct",
    "pose.optimize",
    "scene.compose",
    "physics.dynamics",
    "relation.infer",
    "event.infer",
    "world.compose",
    "world.align",
    "scene.export",
    "report.render",
    "materialize",
    "catalog",
]

PHASE_MAP: dict[str, tuple[str, str]] = {
    "parse":  ("video.inspect",    "object.attr"),
    "lift":   ("geometry.lift",    "pose.optimize"),
    "infer":  ("physics.dynamics", "event.infer"),
    "build":  ("world.compose",    "world.align"),
    "export": ("scene.export",     "report.render"),
    "publish": ("materialize",     "catalog"),
    "all":    ("video.inspect",    "catalog"),
}

STAGE_DEPENDENCIES: dict[str, list[str]] = {
    "video.inspect": [],
    "frame.sample": ["video.inspect"],
    "object.detect": ["frame.sample"],
    "object.index": ["object.detect"],
    "object.attr": ["object.index"],
    "geometry.lift": ["object.detect", "object.index", "object.attr"],
    "mesh.reconstruct": ["geometry.lift", "object.attr"],
    "pose.optimize": ["geometry.lift", "mesh.reconstruct"],
    "scene.compose": ["geometry.lift", "mesh.reconstruct", "pose.optimize"],
    "physics.dynamics": ["geometry.lift", "object.attr"],
    "relation.infer": ["geometry.lift", "object.attr"],
    "event.infer": ["geometry.lift", "relation.infer"],
    "world.compose": ["geometry.lift", "mesh.reconstruct", "object.attr", "physics.dynamics", "relation.infer", "event.infer"],
    "world.align": ["world.compose"],
    "scene.export": ["scene.compose"],
    "report.render": ["world.align"],
    "materialize": ["scene.export", "video.inspect", "frame.sample", "object.index", "object.attr", "geometry.lift", "mesh.reconstruct", "pose.optimize"],
    "catalog": ["materialize"],
    "validate": ["world.align"],
}

LEGACY_STAGE_ALIASES: dict[str, str] = {
    "object.track": "object.index",
    "physics.infer": "physics.dynamics",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ArtifactRecord(BaseModel):
    stage: str
    created_at: str
    inputs_hash: str
    params_hash: str
    outputs: dict[str, str] = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)


class StageStatus(BaseModel):
    stage: str
    status: str = "pending"  # pending | completed | failed
    last_run_at: str | None = None
    error: str | None = None
    inputs_hash: str | None = None
    params_hash: str | None = None


class ProjectManifest(BaseModel):
    project_id: str
    project_root: str
    input_video: str
    created_at: str
    updated_at: str


class ArtifactRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records = self._load()

    def _load(self) -> dict[str, ArtifactRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        normalized: dict[str, ArtifactRecord] = {}
        for stage, value in raw.items():
            canonical = LEGACY_STAGE_ALIASES.get(stage, stage)
            payload = dict(value)
            payload["stage"] = canonical
            normalized[canonical] = ArtifactRecord.model_validate(payload)
        return normalized

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {stage: record.model_dump(mode="json") for stage, record in self.records.items()}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, stage: str) -> ArtifactRecord | None:
        return self.records.get(stage)

    def set(self, record: ArtifactRecord) -> None:
        self.records[record.stage] = record
        self.save()

    def drop_many(self, stages: list[str]) -> None:
        changed = False
        for stage in stages:
            if stage in self.records:
                self.records.pop(stage, None)
                changed = True
        if changed:
            self.save()
