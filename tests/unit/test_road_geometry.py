from __future__ import annotations

from guanwu.video.features.spatial.road_geometry import select_road_plane_for_frame


def test_select_road_plane_uses_global_policy_for_fixed_camera() -> None:
    road_geometry = {
        "available": True,
        "default_plane_policy": "global_for_fixed_camera",
        "global_plane": {
            "source": "weighted_keyframe_robust_global",
            "normal_world": [0.0, 1.0, 0.0],
            "offset": -2.5,
        },
        "keyframe_planes": [
            {"frame_id": 1, "depth_frame": 0, "normal_world": [0.0, 1.0, 0.0], "offset": -1.0},
            {"frame_id": 8, "depth_frame": 7, "normal_world": [0.0, 1.0, 0.0], "offset": -8.0},
        ],
    }

    plane = select_road_plane_for_frame(road_geometry, 8)

    assert plane is not None
    assert plane["offset"] == -2.5
    assert plane["selection"]["mode"] == "global"
    assert plane["selection"]["policy"] == "global_for_fixed_camera"
    assert plane["selection"]["target_frame_id"] == 8


def test_select_road_plane_keeps_nearest_keyframe_policy_when_requested() -> None:
    road_geometry = {
        "available": True,
        "default_plane_policy": "nearest_keyframe",
        "global_plane": {
            "source": "weighted_keyframe_robust_global",
            "normal_world": [0.0, 1.0, 0.0],
            "offset": -2.5,
        },
        "keyframe_planes": [
            {"frame_id": 1, "depth_frame": 0, "normal_world": [0.0, 1.0, 0.0], "offset": -1.0},
            {"frame_id": 8, "depth_frame": 7, "normal_world": [0.0, 1.0, 0.0], "offset": -8.0},
        ],
    }

    plane = select_road_plane_for_frame(road_geometry, 8)

    assert plane is not None
    assert plane["offset"] == -8.0
    assert plane["selection"]["mode"] == "nearest_keyframe"
