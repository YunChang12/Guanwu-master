from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("guanwu")

class CanonicalStore:
    def __init__(self, canonical_root: str):
        self.root = Path(canonical_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset_id: str) -> Path:
        d = self.root / "datasets" / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def scene_dir(self, dataset_id: str, scene_uid: str) -> Path:
        d = self.dataset_dir(dataset_id) / "scenes" / scene_uid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def asset_dir(self, dataset_id: str, asset_uid: str) -> Path:
        d = self.dataset_dir(dataset_id) / "assets" / asset_uid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def episode_dir(self, dataset_id: str, episode_uid: str) -> Path:
        d = self.dataset_dir(dataset_id) / "episodes" / episode_uid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, path: Path, data: dict | list) -> None:
        """Write JSON using orjson for speed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        logger.debug(f"Wrote JSON: {path}")

    def write_parquet(self, path: Path, records: list[dict], schema: pa.Schema | None = None) -> None:
        """Write records to a Parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            logger.debug(f"No records to write for {path}")
            return
        table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, path)
        logger.debug(f"Wrote Parquet ({len(records)} rows): {path}")

    def write_dataset_record(self, dataset_id: str, record: dict) -> Path:
        path = self.dataset_dir(dataset_id) / "dataset.json"
        self.write_json(path, record)
        return path

    def write_scene_meta(self, dataset_id: str, scene_uid: str, record: dict) -> Path:
        path = self.scene_dir(dataset_id, scene_uid) / "scene_meta.json"
        self.write_json(path, record)
        return path

    def write_asset_meta(self, dataset_id: str, asset_uid: str, record: dict) -> Path:
        path = self.asset_dir(dataset_id, asset_uid) / "asset_meta.json"
        self.write_json(path, record)
        return path

    def write_episode_meta(self, dataset_id: str, episode_uid: str, record: dict) -> Path:
        path = self.episode_dir(dataset_id, episode_uid) / "episode_meta.json"
        self.write_json(path, record)
        return path

    def write_episode_states(self, dataset_id: str, episode_uid: str, states: list[dict]) -> Path:
        path = self.episode_dir(dataset_id, episode_uid) / "states.parquet"
        self.write_parquet(path, states)
        return path

    def write_episode_sensor_frames(self, dataset_id: str, episode_uid: str, frames: list[dict]) -> Path:
        path = self.episode_dir(dataset_id, episode_uid) / "sensor_frames.parquet"
        self.write_parquet(path, frames)
        return path

    def write_scene_sensors(self, dataset_id: str, scene_uid: str, sensors: list[dict]) -> Path:
        path = self.scene_dir(dataset_id, scene_uid) / "sensors.parquet"
        self.write_parquet(path, sensors)
        return path

    def write_scene_frames(self, dataset_id: str, scene_uid: str, frames: list[dict]) -> Path:
        path = self.scene_dir(dataset_id, scene_uid) / "frames.parquet"
        self.write_parquet(path, frames)
        return path

    def write_scene_instances(self, dataset_id: str, scene_uid: str, instances: list[dict]) -> Path:
        path = self.scene_dir(dataset_id, scene_uid) / "instances.parquet"
        self.write_parquet(path, instances)
        return path

    def write_scene_tracks(self, dataset_id: str, scene_uid: str, tracks: list[dict]) -> Path:
        path = self.scene_dir(dataset_id, scene_uid) / "tracks.parquet"
        self.write_parquet(path, tracks)
        return path

    def write_scene_articulation(self, dataset_id: str, scene_uid: str, records: list[dict]) -> Path:
        path = self.scene_dir(dataset_id, scene_uid) / "articulation.parquet"
        self.write_parquet(path, records)
        return path

    def write_licenses(self, dataset_id: str, entity_type: str, entity_uid: str, licenses: list[dict]) -> Path:
        if entity_type == "scene":
            base = self.scene_dir(dataset_id, entity_uid)
        elif entity_type == "asset":
            base = self.asset_dir(dataset_id, entity_uid)
        elif entity_type == "episode":
            base = self.episode_dir(dataset_id, entity_uid)
        else:
            base = self.dataset_dir(dataset_id)
        path = base / "licenses.parquet"
        self.write_parquet(path, licenses)
        return path

    def write_provenance(self, dataset_id: str, entity_type: str, entity_uid: str, provenance: dict) -> Path:
        if entity_type == "scene":
            base = self.scene_dir(dataset_id, entity_uid)
        elif entity_type == "asset":
            base = self.asset_dir(dataset_id, entity_uid)
        elif entity_type == "episode":
            base = self.episode_dir(dataset_id, entity_uid)
        else:
            base = self.dataset_dir(dataset_id)
        path = base / "provenance.json"
        self.write_json(path, provenance)
        return path

    def write_mesh_stats(self, dataset_id: str, entity_type: str, entity_uid: str, stats: dict) -> Path:
        if entity_type == "asset":
            base = self.asset_dir(dataset_id, entity_uid)
        else:
            base = self.scene_dir(dataset_id, entity_uid)
        path = base / "mesh_stats.json"
        self.write_json(path, stats)
        return path

    def list_scenes(self, dataset_id: str) -> list[str]:
        scenes_dir = self.dataset_dir(dataset_id) / "scenes"
        if not scenes_dir.exists():
            return []
        return [d.name for d in scenes_dir.iterdir() if d.is_dir()]

    def list_assets(self, dataset_id: str) -> list[str]:
        assets_dir = self.dataset_dir(dataset_id) / "assets"
        if not assets_dir.exists():
            return []
        return [d.name for d in assets_dir.iterdir() if d.is_dir()]

    def list_episodes(self, dataset_id: str) -> list[str]:
        episodes_dir = self.dataset_dir(dataset_id) / "episodes"
        if not episodes_dir.exists():
            return []
        return [d.name for d in episodes_dir.iterdir() if d.is_dir()]
