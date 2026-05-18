"""ProcTHOR-10K adapter for procedurally generated indoor scenes.

ProcTHOR-10K contains 10,000 procedurally generated houses with rooms,
objects, walls, doors, and windows. Data is stored as JSONL (one JSON
per house). Meshes are not included — only scene layouts with object
placement (assetId + position + rotation).

Expected local directory structure::

    <path>/
      train.jsonl.gz   (or train.jsonl)
      val.jsonl.gz
      test.jsonl.gz
"""
from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guanwu.adapters.base import DatasetAdapter, register_adapter
from guanwu.schemas.bundles import (
    AdapterConfig,
    EmitReport,
    JobContext,
    NormalizeBundle,
    ParseBundle,
    RawRef,
    SourceItem,
)
from guanwu.schemas.enums import (
    AccessMode,
    GeometryLevel,
    RecordScope,
    SceneKind,
    SourceType,
)
from guanwu.schemas.records import (
    DatasetRecord,
    InstanceRecord,
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
)

logger = logging.getLogger("guanwu")

DATASET_ID = "procthor_10k"


def _read_jsonl_gz(path: Path, limit: int | None = None) -> list[dict]:
    """Read a gzipped or plain JSONL file."""
    houses = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            houses.append(json.loads(line))
            if limit and len(houses) >= limit:
                break
    return houses


def _house_id(house: dict, index: int) -> str:
    """Generate a readable house ID."""
    spec = house.get("metadata", {}).get("roomSpecId", "")
    return f"house_{index:05d}_{spec}" if spec else f"house_{index:05d}"


def _count_by_type(objects: list[dict]) -> dict[str, int]:
    """Count objects by category prefix (e.g. Chair, Table)."""
    counts: dict[str, int] = {}
    for obj in objects:
        cat = obj.get("id", "").split("|")[0]
        counts[cat] = counts.get(cat, 0) + 1
    return counts


@register_adapter
class ProcTHOR10KAdapter(DatasetAdapter):
    """Adapter for the ProcTHOR-10K procedural house dataset."""

    name: str = "procthor_10k"
    version: str = "0.1.0"

    def capabilities(self) -> dict[str, bool]:
        return {
            "scene_mesh": False,
            "object_mesh": False,
            "articulation": False,
            "deformable_mesh": False,
            "camera": False,
            "depth": False,
            "lidar": False,
            "tracks": False,
            "videos": False,
            "sdk_required": False,
            "license_gated": False,
            "supports_local_ingest": True,
        }

    def inventory(
        self, config: AdapterConfig, ctx: JobContext
    ) -> list[SourceItem]:
        source = Path(config.source_path) if config.source_path else None
        if source is None or not source.is_dir():
            logger.warning("ProcTHOR-10K source_path not set or not a directory")
            return []

        items: list[SourceItem] = []
        for split_name in ["train", "val", "test"]:
            for ext in [".jsonl.gz", ".jsonl"]:
                split_path = source / f"{split_name}{ext}"
                if split_path.exists():
                    items.append(
                        SourceItem(
                            item_id=f"split:{split_name}",
                            dataset_id=DATASET_ID,
                            item_type="scene",
                            source_path=str(split_path),
                            metadata={"split": split_name},
                        )
                    )
                    break

        if config.splits:
            items = [it for it in items if it.metadata.get("split") in config.splits]

        if ctx.limit is not None:
            items = items[: ctx.limit]

        logger.info("ProcTHOR-10K inventory: %d splits found", len(items))
        return items

    def fetch(
        self, items: list[SourceItem], ctx: JobContext
    ) -> list[RawRef]:
        from guanwu.storage.raw_store import RawStore

        raw_store = RawStore(ctx.raw_root)
        refs = []
        for item in items:
            if not item.source_path:
                continue
            src = Path(item.source_path)
            dest = raw_store.dataset_dir(DATASET_ID) / src.name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(src.resolve())
            refs.append(
                RawRef(
                    item_id=item.item_id,
                    raw_path=str(dest),
                )
            )
        return refs

    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        bundle = ParseBundle(dataset_id=DATASET_ID, raw_refs=raw_refs)

        house_limit = ctx.limit or 100  # default: 100 houses per split

        for ref in raw_refs:
            raw_path = Path(ref.raw_path).resolve()
            if not raw_path.exists():
                continue
            split = ref.item_id.replace("split:", "")

            houses = _read_jsonl_gz(raw_path, limit=house_limit)
            logger.info(
                "ProcTHOR-10K: parsed %d houses from %s split", len(houses), split
            )

            for idx, house in enumerate(houses):
                house_id = _house_id(house, idx)
                rooms = house.get("rooms", [])
                objects = house.get("objects", [])
                walls = house.get("walls", [])
                doors = house.get("doors", [])
                windows = house.get("windows", [])

                bundle.scenes.append(
                    {
                        "source_scene_id": house_id,
                        "scene_name": house_id,
                        "split": split,
                        "num_rooms": len(rooms),
                        "num_objects": len(objects),
                        "num_walls": len(walls),
                        "num_doors": len(doors),
                        "num_windows": len(windows),
                        "room_types": [r.get("roomType", "") for r in rooms],
                        "object_counts": _count_by_type(objects),
                        "metadata": house.get("metadata", {}),
                    }
                )

                # Each placed object is an instance
                for obj in objects:
                    obj_id = obj.get("id", "")
                    asset_id = obj.get("assetId", "")
                    category = obj_id.split("|")[0] if "|" in obj_id else obj_id
                    pos = obj.get("position", {})
                    rot = obj.get("rotation", {})

                    bundle.instances.append(
                        {
                            "house_id": house_id,
                            "object_id": obj_id,
                            "asset_id": asset_id,
                            "category": category,
                            "position": pos,
                            "rotation": rot,
                            "kinematic": obj.get("kinematic", False),
                            "children": len(obj.get("children", [])),
                        }
                    )

        return bundle

    def normalize(
        self, bundle: ParseBundle, ctx: JobContext
    ) -> NormalizeBundle:
        now = datetime.now(timezone.utc)
        out = NormalizeBundle(dataset_id=DATASET_ID)

        out.dataset_record = DatasetRecord(
            dataset_id=DATASET_ID,
            dataset_name="ProcTHOR-10K",
            version="1.0.0",
            source_type=SourceType.LOCAL_FOLDER,
            license_name="Apache-2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G1_BBOX,
            created_at=now,
            tags=["indoor", "procedural", "embodied-ai", "procthor"],
        )

        # Scenes
        for scene_dict in bundle.scenes:
            house_id = scene_dict["source_scene_id"]
            room_types = scene_dict.get("room_types", [])

            out.scenes.append(
                SceneRecord(
                    scene_uid=house_id,
                    dataset_id=DATASET_ID,
                    source_scene_id=house_id,
                    scene_name=house_id,
                    scene_kind=SceneKind.INDOOR_STATIC,
                    geometry_level=GeometryLevel.G1_BBOX,
                    num_sensors=0,
                    has_static_scene_mesh=False,
                    has_dynamic_objects=True,
                    has_humans=False,
                    has_articulation=any(
                        d.get("openable") for d in bundle.scenes[0].get("doors", [])
                    ) if bundle.scenes else False,
                )
            )

        # Instances
        for inst_dict in bundle.instances:
            house_id = inst_dict["house_id"]
            obj_id = inst_dict["object_id"]
            category = inst_dict["category"]
            instance_uid = f"{house_id}_{obj_id}"

            out.instances.append(
                InstanceRecord(
                    instance_uid=instance_uid,
                    scene_uid=house_id,
                    category=category,
                    instance_name=obj_id,
                    is_static=inst_dict.get("kinematic", True),
                    is_articulated=False,
                    is_human=False,
                    geometry_level=GeometryLevel.G1_BBOX,
                )
            )

        # License
        out.licenses.append(
            LicenseRecord(
                record_scope=RecordScope.DATASET,
                record_id=DATASET_ID,
                license_name="Apache-2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
                commercial_use_allowed=True,
                redistribution_allowed=True,
                attribution_required=True,
                notes="ProcTHOR-10K by Allen AI, Apache 2.0 license.",
            )
        )

        # Provenance
        out.provenance.append(
            ProvenanceRecord(
                record_id=DATASET_ID,
                dataset_id=DATASET_ID,
                normalized_by_version=self.version,
                normalized_at=now,
                adapter_name=self.name,
                adapter_version=self.version,
                transform_log=[
                    {
                        "step": "normalize",
                        "scenes": len(out.scenes),
                        "instances": len(out.instances),
                    }
                ],
            )
        )

        return out
