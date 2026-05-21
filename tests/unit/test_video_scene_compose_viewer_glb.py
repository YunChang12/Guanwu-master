from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from guanwu.video.features.simulation.usd_coordinate_convention import (
    USDCoordinateConvention,
    convert_world_points_to_usd,
)
from guanwu.video.project.executor import ProjectExecutor


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[order]


def test_export_viewer_space_glb_uses_same_world_to_usd_point_conversion(tmp_path: Path) -> None:
    world_vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    mesh = trimesh.Trimesh(vertices=world_vertices, faces=np.array([[0, 1, 2]]), process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="triangle")

    out_path = tmp_path / "viewer.glb"
    ProjectExecutor._export_viewer_space_glb(scene, out_path)

    loaded = trimesh.load(str(out_path), force="mesh")
    convention = USDCoordinateConvention(
        R_usd_from_world=np.diag([1.0, -1.0, -1.0]).astype(np.float64),
        scene_up_world=np.array([0.0, -1.0, 0.0], dtype=np.float64),
        scene_forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ground_plane_offset_world=0.0,
    )
    expected = convert_world_points_to_usd(world_vertices, convention)

    assert out_path.exists()
    assert np.allclose(_sorted_rows(loaded.vertices), _sorted_rows(expected), atol=1e-5)
