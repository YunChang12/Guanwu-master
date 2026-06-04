from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from guanwu.video.project.config import ProjectConfig, ProjectMetadata
from guanwu.video.project.context import ProjectContext
from guanwu.video.project.executor import ProjectExecutor


def _build_executor(tmp_path: Path) -> ProjectExecutor:
    project_root = tmp_path / "demo_project"
    config = ProjectConfig(
        project=ProjectMetadata(
            project_id="demo_project",
            name="demo_project",
            input_video="/tmp/demo.mp4",
            root_dir=str(project_root),
            provider_mode="zaiwu",
            video_copy_mode="copy",
        ),
    )
    config.settings.zaiwu.mesh_proxy_mode = "auto"
    config.settings.zaiwu.mesh_proxy_target_faces = 24
    config.settings.zaiwu.mesh_proxy_use_for_pose = True
    context = ProjectContext.create(project_root, config)
    return ProjectExecutor(context)


def _write_basic_pose_optimizer_sample(
    executor: ProjectExecutor,
    task_dir: Path,
    source_mesh: Path,
    *,
    label: str = "wooden block",
) -> Path:
    frame = np.full((64, 96, 3), 180, dtype=np.uint8)
    full_mask = np.zeros((64, 96), dtype=bool)
    full_mask[20:44, 30:62] = True
    inst = {
        "bbox": [30.0, 20.0, 62.0, 44.0],
        "concept_label": label,
        "segment_kind": "object",
        "score": 0.9,
        "mask_ref": "mask://frame_00001/obj_000009",
    }
    camera = {
        "fx": 64.0,
        "fy": 64.0,
        "cx": 48.0,
        "cy": 32.0,
        "t": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "R": np.eye(3, dtype=np.float64),
    }
    return executor._write_pose_optimizer_sample(
        task_dir=task_dir,
        obj_id="obj_000009",
        frame_id=1,
        frame_image=frame,
        full_mask=full_mask,
        inst=inst,
        glb_path=source_mesh,
        camera=camera,
        object_node=None,
        object_track=[],
    )


def test_write_pose_optimizer_sample_uses_downsampled_mesh_proxy_for_wooden_block(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source_mesh = tmp_path / "source.glb"
    trimesh.creation.icosphere(subdivisions=2).export(source_mesh)

    task_path = _write_basic_pose_optimizer_sample(executor, task_dir, source_mesh)

    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["mesh_path"] == "object.glb"
    assert task["optimizer_mesh_path"] == "optimizer_object.glb"
    assert task["mesh_proxy"]["mode"] == "simplify"
    assert (task_dir / "object.glb").exists()
    assert (task_dir / "optimizer_object.glb").exists()
    assert cv2.imread(str(task_dir / "mask.png"), cv2.IMREAD_GRAYSCALE).shape == (64, 96)
    assert cv2.imread(str(task_dir / "crop_mask.png"), cv2.IMREAD_GRAYSCALE).shape == (24, 32)

    source = trimesh.load(source_mesh, force="mesh")
    proxy = trimesh.load(task_dir / "optimizer_object.glb", force="mesh")
    assert 12 < len(proxy.faces) <= executor.context.config.settings.zaiwu.mesh_proxy_target_faces * 2
    assert len(proxy.vertices) > 8
    assert np.all(proxy.bounds[0] >= source.bounds[0] - 1e-6)
    assert np.all(proxy.bounds[1] <= source.bounds[1] + 1e-6)


def test_pose_optimizer_cuboid_mode_is_redirected_to_downsampled_mesh(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path)
    executor.context.config.settings.zaiwu.mesh_proxy_mode = "cuboid"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source_mesh = tmp_path / "source.glb"
    trimesh.creation.icosphere(subdivisions=2).export(source_mesh)

    task_path = _write_basic_pose_optimizer_sample(executor, task_dir, source_mesh)

    task = json.loads(task_path.read_text(encoding="utf-8"))
    proxy = trimesh.load(task_dir / "optimizer_object.glb", force="mesh")
    assert task["mesh_proxy"]["mode"] == "simplify"
    assert len(proxy.vertices) > 8


def test_pose_optimizer_simplified_proxy_fallback_preserves_connected_surface(monkeypatch) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3)

    monkeypatch.setattr(type(mesh), "simplify_quadric_decimation", None, raising=False)
    monkeypatch.setattr(type(mesh), "simplify_quadratic_decimation", None, raising=False)

    proxy = ProjectExecutor._pose_optimizer_simplified_proxy(mesh, target_faces=80)

    assert len(proxy.faces) <= 160
    assert len(proxy.vertices) < len(mesh.vertices)
    assert len(proxy.faces) / max(1, len(proxy.vertices)) > 1.0
    assert proxy.euler_number <= 4
    assert proxy.area > mesh.area * 0.65


def test_write_pose_optimizer_sample_keeps_original_mesh_when_proxy_disabled(tmp_path: Path) -> None:
    executor = _build_executor(tmp_path)
    executor.context.config.settings.zaiwu.mesh_proxy_use_for_pose = False
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source_mesh = tmp_path / "source.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(source_mesh)

    frame = np.full((32, 48, 3), 180, dtype=np.uint8)
    full_mask = np.zeros((32, 48), dtype=np.uint8)
    full_mask[8:20, 10:24] = 255
    camera = {
        "fx": 32.0,
        "fy": 32.0,
        "cx": 24.0,
        "cy": 16.0,
        "t": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "R": np.eye(3, dtype=np.float64),
    }

    task_path = executor._write_pose_optimizer_sample(
        task_dir=task_dir,
        obj_id="obj_000001",
        frame_id=1,
        frame_image=frame,
        full_mask=full_mask,
        inst={"bbox": [10.0, 8.0, 24.0, 20.0], "concept_label": "toy"},
        glb_path=source_mesh,
        camera=camera,
        object_node=None,
        object_track=[],
    )

    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert "optimizer_mesh_path" not in task
    assert "mesh_proxy" not in task
