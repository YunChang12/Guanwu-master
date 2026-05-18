"""Tests for stable ID generation."""
from __future__ import annotations

from guanwu.core.ids import (
    make_asset_uid,
    make_episode_uid,
    make_frame_uid,
    make_instance_uid,
    make_scene_uid,
    make_sensor_uid,
    make_track_uid,
)


def test_scene_uid_stable():
    uid1 = make_scene_uid("scannetpp", "scene0001_00")
    uid2 = make_scene_uid("scannetpp", "scene0001_00")
    assert uid1 == uid2
    assert len(uid1) == 16


def test_scene_uid_different_inputs():
    uid1 = make_scene_uid("scannetpp", "scene0001_00")
    uid2 = make_scene_uid("scannetpp", "scene0002_00")
    assert uid1 != uid2


def test_asset_uid_stable():
    uid1 = make_asset_uid("objaverse_xl", "abc123")
    uid2 = make_asset_uid("objaverse_xl", "abc123")
    assert uid1 == uid2


def test_episode_uid_stable():
    uid = make_episode_uid("maniskill3", "ep001")
    assert len(uid) == 16


def test_sensor_uid_stable():
    uid = make_sensor_uid("scannetpp", "scene0001_00", "dslr")
    assert len(uid) == 16


def test_frame_uid_stable():
    uid = make_frame_uid("scannetpp", "sensor1", "1000000000")
    assert len(uid) == 16


def test_instance_uid_stable():
    uid = make_instance_uid("argoverse", "log1", "car_001")
    assert len(uid) == 16


def test_track_uid_stable():
    uid = make_track_uid("argoverse", "inst_001")
    assert len(uid) == 16


def test_different_datasets_different_uids():
    uid1 = make_scene_uid("scannetpp", "scene001")
    uid2 = make_scene_uid("arkitscenes", "scene001")
    assert uid1 != uid2
