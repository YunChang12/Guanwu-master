"""Tests for configuration loading."""
from __future__ import annotations

import pytest
from pathlib import Path

from guanwu.core.config import WorkspaceConfig, load_config, StorageConfig
from process.pose_optimizer.config import config_to_argv


def test_default_config():
    cfg = WorkspaceConfig()
    assert cfg.workspace_root == "."
    assert cfg.random_seed == 42
    assert cfg.runtime.workers == 8
    assert cfg.video_pipeline.object_detection_backend == "seg2track_sam2"
    assert cfg.video_pipeline.mesh_reconstruct_object_ids == []
    assert cfg.video_pipeline.mesh_proxy_mode == "auto"
    assert cfg.video_pipeline.mesh_proxy_target_faces == 1500
    assert cfg.video_pipeline.mesh_proxy_use_for_pose is True
    assert cfg.video_pipeline.mesh_proxy_keep_original_for_export is True
    assert cfg.video_pipeline.background_mode == "auto"
    assert cfg.video_pipeline.background_disable_road_semantics is False
    assert cfg.video_pipeline.task_foreground_object_ids == []
    assert cfg.video_pipeline.background_cleaner == "temporal"
    assert cfg.video_pipeline.background_cleaner_config_path is None
    assert cfg.video_pipeline.background_cleaner_model == "gpt-image-2"
    assert cfg.video_pipeline.background_cleaner_reference_frame_id == 1


def test_resolve_paths():
    cfg = WorkspaceConfig(workspace_root="/data/ws")
    cfg.resolve_paths()
    assert cfg.storage.raw_root == "/data/ws/raw"
    assert cfg.storage.canonical_root == "/data/ws/canonical"
    assert cfg.storage.catalog_path == "/data/ws/catalog/catalog.duckdb"


def test_load_config_from_yaml(tmp_path):
    yaml_content = """
workspace_root: /tmp/test_ws
random_seed: 123
storage:
  raw_root: my_raw
runtime:
  workers: 4
  fail_fast: true
video_pipeline:
  mesh_reconstruct_object_ids:
    - obj_000007
    - obj_000012
  mesh_proxy_mode: simplify
  mesh_proxy_target_faces: 2400
  mesh_proxy_use_for_pose: false
  mesh_proxy_keep_original_for_export: true
  background_mode: tabletop_task
  background_disable_road_semantics: true
  task_foreground_object_ids:
    - obj_000009
  background_cleaner: openai_image_edit
  background_cleaner_config_path: /root/autodl-fs/Qcp/Guanwu-master/configs/openai-image-cleaner.yaml
  background_cleaner_model: gpt-image-2
  background_cleaner_reference_frame_id: 1
datasets:
  scannetpp:
    enabled: true
    source:
      mode: local
      path: /data/scannetpp
"""
    config_file = tmp_path / "workspace.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(config_file)
    assert cfg.workspace_root == "/tmp/test_ws"
    assert cfg.random_seed == 123
    assert cfg.runtime.workers == 4
    assert cfg.runtime.fail_fast is True
    assert cfg.video_pipeline.mesh_reconstruct_object_ids == ["obj_000007", "obj_000012"]
    assert cfg.video_pipeline.mesh_proxy_mode == "simplify"
    assert cfg.video_pipeline.mesh_proxy_target_faces == 2400
    assert cfg.video_pipeline.mesh_proxy_use_for_pose is False
    assert cfg.video_pipeline.mesh_proxy_keep_original_for_export is True
    assert cfg.video_pipeline.background_mode == "tabletop_task"
    assert cfg.video_pipeline.background_disable_road_semantics is True
    assert cfg.video_pipeline.task_foreground_object_ids == ["obj_000009"]
    assert cfg.video_pipeline.background_cleaner == "openai_image_edit"
    assert cfg.video_pipeline.background_cleaner_config_path == "/root/autodl-fs/Qcp/Guanwu-master/configs/openai-image-cleaner.yaml"
    assert cfg.video_pipeline.background_cleaner_model == "gpt-image-2"
    assert cfg.video_pipeline.background_cleaner_reference_frame_id == 1
    assert "scannetpp" in cfg.datasets
    assert cfg.datasets["scannetpp"].source.path == "/data/scannetpp"


def test_load_config_missing_file():
    from guanwu.core.errors import ConfigError
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path.yaml")


def test_dataset_config_defaults():
    cfg = WorkspaceConfig()
    # No datasets configured by default
    assert cfg.datasets == {}


def test_pose_optimizer_config_to_argv_keeps_negative_axis_values() -> None:
    assert config_to_argv({"world_up_axis": "-y"}) == ["--world_up_axis=-y"]
