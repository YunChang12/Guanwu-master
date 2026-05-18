from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import duckdb
import orjson
import pyarrow.parquet as pq

logger = logging.getLogger("guanwu")

# SQL to create all catalog tables
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR PRIMARY KEY,
    dataset_name VARCHAR,
    version VARCHAR,
    source_type VARCHAR,
    source_uri VARCHAR,
    license_name VARCHAR,
    license_url VARCHAR,
    access_mode VARCHAR,
    geometry_level_max VARCHAR,
    created_at TIMESTAMP,
    tags VARCHAR[]
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_uid VARCHAR PRIMARY KEY,
    dataset_id VARCHAR,
    source_scene_id VARCHAR,
    scene_name VARCHAR,
    scene_kind VARCHAR,
    geometry_level VARCHAR,
    world_units VARCHAR DEFAULT 'meters',
    canonical_up_axis VARCHAR DEFAULT 'Z',
    duration_sec DOUBLE,
    num_frames INTEGER,
    num_sensors INTEGER,
    has_static_scene_mesh BOOLEAN,
    has_dynamic_objects BOOLEAN,
    has_humans BOOLEAN,
    has_articulation BOOLEAN,
    bbox_min_x DOUBLE, bbox_min_y DOUBLE, bbox_min_z DOUBLE,
    bbox_max_x DOUBLE, bbox_max_y DOUBLE, bbox_max_z DOUBLE
);

CREATE TABLE IF NOT EXISTS assets (
    asset_uid VARCHAR PRIMARY KEY,
    dataset_id VARCHAR,
    source_asset_id VARCHAR,
    category VARCHAR,
    supercategory VARCHAR,
    geometry_level VARCHAR,
    is_articulated BOOLEAN,
    is_deformable BOOLEAN,
    mesh_uri VARCHAR,
    usd_uri VARCHAR,
    glb_uri VARCHAR,
    num_vertices INTEGER,
    num_faces INTEGER,
    watertight BOOLEAN,
    manifold BOOLEAN,
    material_count INTEGER,
    texture_count INTEGER
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_uid VARCHAR PRIMARY KEY,
    dataset_id VARCHAR,
    scene_uid VARCHAR,
    source_episode_id VARCHAR,
    duration_sec DOUBLE,
    num_frames INTEGER
);

CREATE TABLE IF NOT EXISTS sensors (
    sensor_uid VARCHAR PRIMARY KEY,
    scene_uid VARCHAR,
    episode_uid VARCHAR,
    sensor_type VARCHAR,
    name VARCHAR,
    width INTEGER,
    height INTEGER,
    fx DOUBLE, fy DOUBLE, cx DOUBLE, cy DOUBLE,
    distortion_model VARCHAR,
    parent_frame VARCHAR
);

CREATE TABLE IF NOT EXISTS frames (
    frame_uid VARCHAR PRIMARY KEY,
    sensor_uid VARCHAR,
    episode_uid VARCHAR,
    scene_uid VARCHAR,
    timestamp_ns BIGINT,
    image_uri VARCHAR,
    depth_uri VARCHAR,
    segmentation_uri VARCHAR,
    pointcloud_uri VARCHAR,
    exposure_time_ms DOUBLE
);

CREATE TABLE IF NOT EXISTS instances (
    instance_uid VARCHAR PRIMARY KEY,
    scene_uid VARCHAR,
    episode_uid VARCHAR,
    asset_uid VARCHAR,
    category VARCHAR,
    instance_name VARCHAR,
    is_static BOOLEAN,
    is_articulated BOOLEAN,
    is_human BOOLEAN,
    geometry_level VARCHAR
);

CREATE TABLE IF NOT EXISTS track_states (
    track_uid VARCHAR,
    instance_uid VARCHAR,
    timestamp_ns BIGINT,
    bbox3d_center_x DOUBLE, bbox3d_center_y DOUBLE, bbox3d_center_z DOUBLE,
    bbox3d_size_x DOUBLE, bbox3d_size_y DOUBLE, bbox3d_size_z DOUBLE,
    visibility DOUBLE
);

CREATE TABLE IF NOT EXISTS articulation_states (
    instance_uid VARCHAR,
    asset_uid VARCHAR,
    timestamp_ns BIGINT,
    joint_names VARCHAR[],
    joint_positions DOUBLE[],
    joint_velocities DOUBLE[]
);

CREATE TABLE IF NOT EXISTS licenses (
    record_scope VARCHAR,
    record_id VARCHAR,
    license_name VARCHAR,
    license_url VARCHAR,
    commercial_use_allowed BOOLEAN,
    redistribution_allowed BOOLEAN,
    attribution_required BOOLEAN,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS provenance (
    record_id VARCHAR,
    dataset_id VARCHAR,
    source_relpath VARCHAR,
    source_sha256 VARCHAR,
    normalized_by_version VARCHAR,
    normalized_at TIMESTAMP,
    adapter_name VARCHAR,
    adapter_version VARCHAR
);
"""

class Catalog:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def initialize(self) -> None:
        """Create all tables."""
        self.conn.execute(_CREATE_TABLES_SQL)
        logger.info(f"Catalog initialized at {self.db_path}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def insert_dataset(self, record: dict) -> None:
        self._upsert("datasets", record, "dataset_id")

    def insert_scene(self, record: dict) -> None:
        # Flatten bbox tuples
        flat = dict(record)
        bbox_min = flat.pop("bbox_min_xyz", None)
        bbox_max = flat.pop("bbox_max_xyz", None)
        if bbox_min:
            flat["bbox_min_x"], flat["bbox_min_y"], flat["bbox_min_z"] = bbox_min
        if bbox_max:
            flat["bbox_max_x"], flat["bbox_max_y"], flat["bbox_max_z"] = bbox_max
        self._upsert("scenes", flat, "scene_uid")

    def insert_asset(self, record: dict) -> None:
        self._upsert("assets", record, "asset_uid")

    def insert_episode(self, record: dict) -> None:
        self._upsert("episodes", record, "episode_uid")

    def insert_sensor(self, record: dict) -> None:
        flat = dict(record)
        flat.pop("distortion_params", None)
        flat.pop("T_sensor_from_parent", None)
        self._upsert("sensors", flat, "sensor_uid")

    def insert_frame(self, record: dict) -> None:
        flat = dict(record)
        flat.pop("T_world_from_sensor", None)
        self._upsert("frames", flat, "frame_uid")

    def insert_instance(self, record: dict) -> None:
        self._upsert("instances", record, "instance_uid")

    def insert_track_state(self, record: dict) -> None:
        flat = dict(record)
        flat.pop("T_world_from_object", None)
        vel_lin = flat.pop("linear_velocity_xyz", None)
        vel_ang = flat.pop("angular_velocity_xyz", None)
        center = flat.pop("bbox3d_center_xyz", None)
        size = flat.pop("bbox3d_size_xyz", None)
        if center:
            flat["bbox3d_center_x"], flat["bbox3d_center_y"], flat["bbox3d_center_z"] = center
        if size:
            flat["bbox3d_size_x"], flat["bbox3d_size_y"], flat["bbox3d_size_z"] = size
        self._insert("track_states", flat)

    def insert_articulation_state(self, record: dict) -> None:
        self._insert("articulation_states", record)

    def insert_license(self, record: dict) -> None:
        self._insert("licenses", record)

    def insert_provenance(self, record: dict) -> None:
        flat = dict(record)
        flat.pop("transform_log", None)
        self._insert("provenance", flat)

    def query(self, sql: str) -> list[dict]:
        """Execute a SQL query and return results as list of dicts."""
        result = self.conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def _upsert(self, table: str, record: dict, pk: str) -> None:
        """Insert or replace a record by primary key."""
        # Remove None values and filter to known columns
        cols = self._table_columns(table)
        filtered = {k: v for k, v in record.items() if k in cols}
        if not filtered:
            return
        col_names = ", ".join(filtered.keys())
        placeholders = ", ".join(["?" for _ in filtered])
        values = list(filtered.values())
        # Use INSERT OR REPLACE
        self.conn.execute(
            f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})",
            values
        )

    def _insert(self, table: str, record: dict) -> None:
        """Insert a record (no upsert)."""
        cols = self._table_columns(table)
        filtered = {k: v for k, v in record.items() if k in cols}
        if not filtered:
            return
        col_names = ", ".join(filtered.keys())
        placeholders = ", ".join(["?" for _ in filtered])
        values = list(filtered.values())
        self.conn.execute(
            f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
            values
        )

    def _table_columns(self, table: str) -> set[str]:
        """Get column names for a table."""
        result = self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return {row[1] for row in result}

    def build_from_canonical(self, canonical_root: str) -> None:
        """Build/rebuild catalog from canonical store directory."""
        root = Path(canonical_root) / "datasets"
        if not root.exists():
            logger.warning(f"No canonical datasets found at {root}")
            return

        self.initialize()
        self._clear_tables()

        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue

            # Dataset record
            ds_json = dataset_dir / "dataset.json"
            if ds_json.exists():
                with open(ds_json, "rb") as f:
                    self.insert_dataset(orjson.loads(f.read()))

            # Scenes
            scenes_dir = dataset_dir / "scenes"
            if scenes_dir.exists():
                for scene_dir in scenes_dir.iterdir():
                    if not scene_dir.is_dir():
                        continue
                    meta = scene_dir / "scene_meta.json"
                    if meta.exists():
                        with open(meta, "rb") as f:
                            self.insert_scene(orjson.loads(f.read()))
                    self._load_parquet(scene_dir / "sensors.parquet", self.insert_sensor)
                    self._load_parquet(scene_dir / "frames.parquet", self.insert_frame)
                    self._load_parquet(scene_dir / "instances.parquet", self.insert_instance)
                    self._load_parquet(scene_dir / "tracks.parquet", self.insert_track_state)
                    self._load_parquet(scene_dir / "articulation.parquet", self.insert_articulation_state)
                    self._load_parquet(scene_dir / "licenses.parquet", self.insert_license)
                    self._load_json(scene_dir / "provenance.json", self.insert_provenance)

            # Assets
            assets_dir = dataset_dir / "assets"
            if assets_dir.exists():
                for asset_dir in assets_dir.iterdir():
                    if not asset_dir.is_dir():
                        continue
                    meta = asset_dir / "asset_meta.json"
                    if meta.exists():
                        with open(meta, "rb") as f:
                            self.insert_asset(orjson.loads(f.read()))
                    self._load_parquet(asset_dir / "licenses.parquet", self.insert_license)
                    self._load_json(asset_dir / "provenance.json", self.insert_provenance)

            # Episodes
            episodes_dir = dataset_dir / "episodes"
            if episodes_dir.exists():
                for episode_dir in episodes_dir.iterdir():
                    if not episode_dir.is_dir():
                        continue
                    meta = episode_dir / "episode_meta.json"
                    if meta.exists():
                        with open(meta, "rb") as f:
                            self.insert_episode(orjson.loads(f.read()))
                    self._load_parquet(episode_dir / "sensor_frames.parquet", self.insert_frame)
                    self._load_parquet(episode_dir / "states.parquet", self.insert_track_state)
                    self._load_parquet(episode_dir / "licenses.parquet", self.insert_license)
                    self._load_json(episode_dir / "provenance.json", self.insert_provenance)

        logger.info(f"Catalog rebuilt from {canonical_root}")

    def get_stats(self) -> dict:
        """Get catalog statistics."""
        stats = {}
        for table in ["datasets", "scenes", "assets", "episodes", "sensors", "frames",
                       "instances", "track_states", "articulation_states", "licenses", "provenance"]:
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[table] = result[0] if result else 0
            except Exception:
                stats[table] = 0
        return stats

    def _clear_tables(self) -> None:
        for table in [
            "datasets",
            "scenes",
            "assets",
            "episodes",
            "sensors",
            "frames",
            "instances",
            "track_states",
            "articulation_states",
            "licenses",
            "provenance",
        ]:
            self.conn.execute(f"DELETE FROM {table}")

    def _load_json(self, path: Path, loader: Any) -> None:
        if not path.exists():
            return
        with open(path, "rb") as f:
            payload = orjson.loads(f.read())
        if isinstance(payload, list):
            for record in payload:
                loader(record)
            return
        loader(payload)

    def _load_parquet(self, path: Path, loader: Any) -> None:
        if not path.exists():
            return
        table = pq.read_table(path)
        for record in table.to_pylist():
            loader(record)
