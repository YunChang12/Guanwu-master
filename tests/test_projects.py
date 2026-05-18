"""Project lifecycle tests for sim and video workflows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from guanwu.cli import app
from guanwu.core.config import DatasetConfig, DatasetSourceConfig, StorageConfig, WorkspaceConfig
from guanwu.projects import ProjectContext
from guanwu.sim.executor import SimProjectExecutor
from guanwu.video.executor import VideoProjectExecutor

runner = CliRunner()


def _workspace_for_fixture(tmp_workspace: Path, dataset_id: str, source_path: str) -> WorkspaceConfig:
    cfg = WorkspaceConfig(
        workspace_root=str(tmp_workspace),
        storage=StorageConfig(
            raw_root=str(tmp_workspace / "raw"),
            staging_root=str(tmp_workspace / "staging"),
            canonical_root=str(tmp_workspace / "canonical"),
            export_root=str(tmp_workspace / "exports"),
            catalog_path=str(tmp_workspace / "catalog" / "catalog.duckdb"),
            project_root=str(tmp_workspace / "projects"),
        ),
        datasets={
            dataset_id: DatasetConfig(
                enabled=True,
                source=DatasetSourceConfig(mode="local", path=source_path),
            )
        },
    )
    return cfg


def _write_workspace_yaml(path: Path, dataset_id: str, source_path: str) -> Path:
    config_path = path / "workspace.yaml"
    config_path.write_text(
        "\n".join([
            f"workspace_root: {path}",
            "storage:",
            f"  raw_root: {path / 'raw'}",
            f"  staging_root: {path / 'staging'}",
            f"  canonical_root: {path / 'canonical'}",
            f"  export_root: {path / 'exports'}",
            f"  catalog_path: {path / 'catalog' / 'catalog.duckdb'}",
            f"  project_root: {path / 'projects'}",
            "datasets:",
            f"  {dataset_id}:",
            "    enabled: true",
            "    source:",
            "      mode: local",
            f"      path: {source_path}",
        ]),
        encoding="utf-8",
    )
    return config_path


def _make_video(path: Path, frame_count: int = 4) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (96, 72),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create test video")
    try:
        for idx in range(frame_count):
            frame = np.zeros((72, 96, 3), dtype=np.uint8)
            frame[:, :, 0] = 30 + idx * 10
            frame[:, :, 1] = 90
            frame[:, :, 2] = 150
            cv2.putText(frame, f"F{idx}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_sim_project_lifecycle(tmp_workspace, scannetpp_fixture) -> None:
    cfg = _workspace_for_fixture(tmp_workspace, "scannetpp", str(scannetpp_fixture))
    project_root = tmp_workspace / "projects" / "sim" / "scannetpp" / "case"
    context = SimProjectExecutor.init_project(dataset_id="scannetpp", out_dir=project_root, workspace=cfg)
    executor = SimProjectExecutor(context)

    executor.run_range("inventory", "catalog")

    assert (project_root / "project.toml").exists()
    assert (project_root / "state" / "stage_status.json").exists()
    assert (project_root / "outputs" / "normalize" / "normalize_bundle.json").exists()
    assert (tmp_workspace / "canonical" / "datasets").exists()
    stats = json.loads((project_root / "outputs" / "catalog" / "catalog_stats.json").read_text(encoding="utf-8"))
    assert stats["datasets"] >= 1
    assert stats["scenes"] >= 1


def test_sim_project_force_rerun_invalidates_downstream(tmp_workspace, scannetpp_fixture) -> None:
    cfg = _workspace_for_fixture(tmp_workspace, "scannetpp", str(scannetpp_fixture))
    project_root = tmp_workspace / "projects" / "sim" / "scannetpp" / "invalidate"
    context = SimProjectExecutor.init_project(dataset_id="scannetpp", out_dir=project_root, workspace=cfg)
    executor = SimProjectExecutor(context)

    executor.run_range("inventory", "materialize")
    executor.run_stage("parse", force=True)

    statuses = ProjectContext(project_root).load_stage_statuses()
    assert statuses["normalize"].status == "pending"
    assert statuses["materialize"].status == "pending"


def test_legacy_ingest_command_uses_sim_project(tmp_workspace, scannetpp_fixture) -> None:
    config_path = _write_workspace_yaml(tmp_workspace, "scannetpp", str(scannetpp_fixture))
    result = runner.invoke(app, ["ingest", "scannetpp", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "project:" in result.output
    assert (tmp_workspace / "projects" / "sim" / "scannetpp" / "default" / "project.toml").exists()


def test_video_project_lifecycle(tmp_workspace) -> None:
    video_path = _make_video(tmp_workspace / "demo.mp4")
    cfg = WorkspaceConfig(
        workspace_root=str(tmp_workspace),
        storage=StorageConfig(
            raw_root=str(tmp_workspace / "raw"),
            staging_root=str(tmp_workspace / "staging"),
            canonical_root=str(tmp_workspace / "canonical"),
            export_root=str(tmp_workspace / "exports"),
            catalog_path=str(tmp_workspace / "catalog" / "catalog.duckdb"),
            project_root=str(tmp_workspace / "projects"),
        ),
    )
    project_root = tmp_workspace / "projects" / "video" / "demo"
    context = VideoProjectExecutor.init_project(video=str(video_path), out_dir=project_root, workspace=cfg)
    executor = VideoProjectExecutor(context)

    executor.run_range("video.inspect", "report.render")
    executor.run_stage("materialize")
    executor.run_stage("catalog")

    assert (executor.context.stage_output_dir("physics.dynamics") / "physics_dynamics.json").exists()
    assert (executor.context.stage_output_dir("relation.infer") / "relations.json").exists()
    assert (executor.context.stage_output_dir("event.infer") / "events.json").exists()
    assert (executor.context.stage_output_dir("scene.export") / "scene.usdc").exists()
    assert (executor.context.stage_output_dir("materialize") / "materialize_report.json").exists()
    stats = json.loads((executor.context.stage_output_dir("catalog") / "catalog_stats.json").read_text(encoding="utf-8"))
    assert stats["episodes"] >= 1
    assert stats["frames"] >= 1


def test_video_materialize_without_mesh_uses_point_obs(tmp_workspace) -> None:
    from guanwu.video.materialize import materialize_video_project

    project_root = tmp_workspace / "projects" / "video" / "meshless"
    (project_root / "input").mkdir(parents=True, exist_ok=True)
    scene_export = project_root / "scene.usdc"
    scene_export.write_text("#usda 1.0\n", encoding="utf-8")
    report = materialize_video_project(
        project_root=project_root,
        canonical_root=tmp_workspace / "canonical",
        scene_export_path=scene_export,
        video_metadata={"frame_count": 1, "duration_sec": 0.2, "width": 64, "height": 64},
        frame_index=[{"frame_idx": 0, "timestamp": 0.0, "image_uri": None}],
        object_index=[{"object_id": "obj_000001", "label": "object"}],
        object_attrs={"obj_000001": {"is_movable": False}},
        geometry_summary={"object_trajectories": {"obj_000001": [{"timestamp": 0.0, "position_xyz": [0, 0, 0.5]}]}},
        mesh_manifest={},
    )
    assert report.scenes_emitted == 1

    datasets_root = tmp_workspace / "canonical" / "datasets" / "natural_video" / "scenes"
    scene_dirs = list(datasets_root.iterdir())
    assert scene_dirs
    scene_meta = json.loads((scene_dirs[0] / "scene_meta.json").read_text(encoding="utf-8"))
    assert scene_meta["geometry_level"] == "G2_POINT_OBS"
