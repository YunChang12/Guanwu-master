"""Tests for configuration loading."""
from __future__ import annotations

import pytest
from pathlib import Path

from guanwu.core.config import WorkspaceConfig, load_config, StorageConfig
from process.pose_optimizer.config import config_to_argv, parse_simple_yaml


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
    assert cfg.video_pipeline.task_foreground_object_ids == []
    assert cfg.video_pipeline.background_target_frame_id == 1
    assert cfg.video_pipeline.background_cleaner == "openai_image_edit"
    assert cfg.video_pipeline.background_cleaner_config_path is None
    assert cfg.video_pipeline.background_cleaner_model == "gpt-image-2"
    assert cfg.video_pipeline.background_cleaner_reference_frame_id == 1
    assert cfg.video_pipeline.background_scene_prompt_profile == "auto"


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
  task_foreground_object_ids:
    - obj_000009
  background_target_frame_id: 1
  background_cleaner: openai_image_edit
  background_cleaner_config_path: /root/autodl-fs/Qcp/Guanwu-master/configs/openai-image-cleaner.yaml
  background_cleaner_model: gpt-image-2
  background_cleaner_reference_frame_id: 1
  background_scene_prompt_profile: road
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
    assert cfg.video_pipeline.task_foreground_object_ids == ["obj_000009"]
    assert cfg.video_pipeline.background_target_frame_id == 1
    assert cfg.video_pipeline.background_cleaner == "openai_image_edit"
    assert cfg.video_pipeline.background_cleaner_config_path == "/root/autodl-fs/Qcp/Guanwu-master/configs/openai-image-cleaner.yaml"
    assert cfg.video_pipeline.background_cleaner_model == "gpt-image-2"
    assert cfg.video_pipeline.background_cleaner_reference_frame_id == 1
    assert cfg.video_pipeline.background_scene_prompt_profile == "road"
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


def test_pose_optimizer_config_to_argv_can_disable_support_aligned_seed() -> None:
    assert config_to_argv({"support_aligned_seed_enabled": False}) == ["--no-support_aligned_seed_enabled"]


def test_pose_optimizer_nested_depth_config_flattens_to_legacy_args() -> None:
    cfg = parse_simple_yaml(
        """
variant: generic_appearance_temporal
depth:
  source: depth_anything3
  fallback_to_wildgs: false
  depth_sigma: 0.75
  generic_depth_weight: 0.15
  use_mask_erode: false
support:
  source: depth_anything3
  fallback_to_wildgs: false
  support_plane_weight: 0.12
  fit_from_current_frame_depth: false
  exclude_object_masks: false
  exclude_other_instance_masks: false
""".strip()
    )

    argv = config_to_argv(cfg)

    assert "--depth_source" in argv
    assert argv[argv.index("--depth_source") + 1] == "depth_anything3"
    assert "--no-depth_fallback_to_wildgs" in argv
    assert "--support_depth_source" in argv
    assert argv[argv.index("--support_depth_source") + 1] == "depth_anything3"
    assert "--no-support_fallback_to_wildgs" in argv
    assert "--depth_sigma" in argv
    assert argv[argv.index("--depth_sigma") + 1] == "0.75"
    assert "--generic_depth_weight" in argv
    assert argv[argv.index("--generic_depth_weight") + 1] == "0.15"
    assert "--no-depth_use_mask_erode" in argv
    assert "--no-support_fit_from_current_frame_depth" in argv
    assert "--no-support_exclude_object_masks" in argv
    assert "--no-support_exclude_other_instance_masks" in argv


def test_fast_variant_accepts_depth_compat_config_args() -> None:
    from process.pose_optimizer.strategies import fast

    argv = config_to_argv(parse_simple_yaml(
        """
variant: fast
depth_source: wildgs
depth_type: metric
depth_unit: meter
depth_fallback_to_wildgs: true
support_depth_source: wildgs
support_fallback_to_wildgs: true
""".strip()
    ))

    parser = fast.argparse.ArgumentParser()
    fast.add_depth_compat_arguments(parser)
    parsed = parser.parse_args(argv)

    assert parsed.depth_source == "wildgs"
    assert parsed.depth_fallback_to_wildgs is True
    assert parsed.support_depth_source == "wildgs"
    assert parsed.support_fallback_to_wildgs is True
