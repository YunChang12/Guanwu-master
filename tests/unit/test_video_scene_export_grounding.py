from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from guanwu.video.project.executor import ProjectExecutor


def _geometry_with_background_manifest(manifest_path: Path) -> SimpleNamespace:
    return SimpleNamespace(outputs={"background_assets_manifest": str(manifest_path)})


def test_scene_export_grounding_prefers_background_tabletop_reference(tmp_path: Path) -> None:
    tabletop_reference = tmp_path / "tabletop_reference.json"
    tabletop_reference.write_text(
        json.dumps(
            {
                "schema": "guanwu.tabletop_reference.v1",
                "source": "clean_depth_background",
                "target_frame_id": 7,
                "normal_world": [0.0, -2.0, 0.0],
                "offset": 0.42,
            }
        ),
        encoding="utf-8",
    )
    background_manifest = tmp_path / "background_manifest.json"
    background_manifest.write_text(
        json.dumps(
            {
                "target_frame_id": 7,
                "assets": {"tabletop_reference": str(tabletop_reference)},
            }
        ),
        encoding="utf-8",
    )
    road_geometry = tmp_path / "road_geometry.json"
    road_geometry.write_text(
        json.dumps(
            {
                "available": True,
                "default_plane_policy": "global_for_fixed_camera",
                "global_plane": {
                    "source": "road_global",
                    "normal_world": [1.0, 0.0, 0.0],
                    "offset": 99.0,
                },
            }
        ),
        encoding="utf-8",
    )

    frame_id, plane = ProjectExecutor._scene_export_fixed_camera_grounding_plane(
        _geometry_with_background_manifest(background_manifest),
        str(road_geometry),
        fallback_frame_id=1,
    )

    assert frame_id == 7
    assert plane is not None
    assert np.allclose(plane["normal_world"], [0.0, -1.0, 0.0])
    assert plane["offset"] == 0.42
    assert plane["source"] == "clean_depth_background"


def test_scene_export_grounding_falls_back_to_road_global_plane(tmp_path: Path) -> None:
    background_manifest = tmp_path / "background_manifest.json"
    background_manifest.write_text(json.dumps({"target_frame_id": 5, "assets": {}}), encoding="utf-8")
    road_geometry = tmp_path / "road_geometry.json"
    road_geometry.write_text(
        json.dumps(
            {
                "available": True,
                "default_plane_policy": "global_for_fixed_camera",
                "global_plane": {
                    "source": "road_global",
                    "normal_world": [0.0, -3.0, 0.0],
                    "offset": 0.12,
                },
            }
        ),
        encoding="utf-8",
    )

    frame_id, plane = ProjectExecutor._scene_export_fixed_camera_grounding_plane(
        _geometry_with_background_manifest(background_manifest),
        str(road_geometry),
        fallback_frame_id=1,
    )

    assert frame_id == 5
    assert plane is not None
    assert np.allclose(plane["normal_world"], [0.0, -1.0, 0.0])
    assert plane["offset"] == 0.12
    assert plane["source"] == "road_global"


def test_ground_pose_to_plane_uses_axis_roles_bottom_not_side_wall() -> None:
    # Local +Y is the cup's semantic up axis, so the physical bottom is the
    # low-Y band. Add a side-wall protrusion lower along world/table normal to
    # catch regressions that snap the globally lowest vertex instead.
    bottom = np.array(
        [
            [-0.2, 0.0, -0.2],
            [0.2, 0.0, -0.2],
            [-0.2, 0.0, 0.2],
            [0.2, 0.0, 0.2],
        ],
        dtype=np.float64,
    )
    wall = np.array(
        [
            [0.55, 0.4, -0.05],
            [0.55, 0.5, 0.05],
            [0.62, 0.45, 0.0],
        ],
        dtype=np.float64,
    )
    vertices = np.concatenate([bottom, wall], axis=0)
    theta = np.deg2rad(55.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    grounded_R, grounded_t, meta = ProjectExecutor._ground_pose_to_plane(
        rotation,
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        vertices,
        np.ones(3, dtype=np.float64),
        plane_normal,
        0.0,
        axis_roles={"up_axis_idx": 1, "up_axis_sign": 1.0},
    )

    transformed = (grounded_R @ vertices.T).T + grounded_t.reshape(1, 3)
    local_bottom = vertices[:, 1] <= np.percentile(vertices[:, 1], 3.0)
    side_wall = vertices[:, 0] > 0.5

    assert meta["contact_axis_source"] == "axis_roles"
    assert meta["contact_axis_index"] == 1
    assert meta["contact_axis_sign"] == 1.0
    assert np.dot(grounded_R[:, 1], plane_normal) > 0.999
    assert abs(float(np.min(transformed[local_bottom] @ plane_normal))) < 1e-9
    assert float(np.min(transformed[side_wall] @ plane_normal)) > 0.05
