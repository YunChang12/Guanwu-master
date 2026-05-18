"""CLI smoke tests."""
from __future__ import annotations
import pytest
from typer.testing import CliRunner
from guanwu.cli import app

runner = CliRunner()

def test_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output

def test_registry_list():
    result = runner.invoke(app, ["registry", "list"])
    assert result.exit_code == 0
    assert "scannetpp" in result.output.lower() or "Datasets" in result.output

def test_registry_show_known():
    result = runner.invoke(app, ["registry", "show", "scannetpp"])
    assert result.exit_code == 0
    assert "ScanNet" in result.output

def test_registry_show_unknown():
    result = runner.invoke(app, ["registry", "show", "nonexistent_dataset"])
    assert result.exit_code == 1

def test_stats_no_config():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0

def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "d3u" in result.output.lower() or "guanwu" in result.output.lower()
