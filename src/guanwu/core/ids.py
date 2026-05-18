from __future__ import annotations

import hashlib


def _stable_hash(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def make_scene_uid(dataset_id: str, source_scene_id: str) -> str:
    return _stable_hash(f"{dataset_id}:scene:{source_scene_id}")


def make_asset_uid(dataset_id: str, source_asset_id: str) -> str:
    return _stable_hash(f"{dataset_id}:object:{source_asset_id}")


def make_episode_uid(dataset_id: str, source_episode_id: str) -> str:
    return _stable_hash(f"{dataset_id}:episode:{source_episode_id}")


def make_sensor_uid(dataset_id: str, parent_id: str, sensor_name: str) -> str:
    return _stable_hash(f"{dataset_id}:sensor:{parent_id}:{sensor_name}")


def make_frame_uid(dataset_id: str, sensor_id: str, timestamp_or_index: str) -> str:
    return _stable_hash(f"{dataset_id}:frame:{sensor_id}:{timestamp_or_index}")


def make_instance_uid(dataset_id: str, scene_or_ep_id: str, instance_id: str) -> str:
    return _stable_hash(f"{dataset_id}:instance:{scene_or_ep_id}:{instance_id}")


def make_track_uid(dataset_id: str, instance_uid: str) -> str:
    return _stable_hash(f"{dataset_id}:track:{instance_uid}")
