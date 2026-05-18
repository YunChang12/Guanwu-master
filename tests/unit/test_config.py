"""Tests for configuration loading."""
from __future__ import annotations

import pytest
from pathlib import Path

from guanwu.core.config import WorkspaceConfig, load_config, StorageConfig


def test_default_config():
    cfg = WorkspaceConfig()
    assert cfg.workspace_root == "."
    assert cfg.random_seed == 42
    assert cfg.runtime.workers == 8
    assert cfg.video_pipeline.object_detection_backend == "seg2track_sam2"


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
