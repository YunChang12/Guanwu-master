"""Contract tests: every registered adapter must meet these."""
from __future__ import annotations
import importlib
import pytest
from guanwu.adapters.base import list_adapters, get_adapter, DatasetAdapter

ADAPTER_MODULES = [
    "guanwu.adapters.scannetpp",
    "guanwu.adapters.arkitscenes",
    "guanwu.adapters.objaverse_xl",
    "guanwu.adapters.partnet_mobility",
    "guanwu.adapters.maniskill3",
]

@pytest.fixture(autouse=True)
def _load_adapters():
    for mod in ADAPTER_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass

def test_all_p0_adapters_registered():
    adapters = list_adapters()
    expected = {"scannetpp", "arkitscenes", "objaverse_xl", "partnet_mobility", "maniskill3"}
    assert expected.issubset(set(adapters.keys())), f"Missing: {expected - set(adapters.keys())}"

@pytest.mark.parametrize("name", ["scannetpp", "arkitscenes", "objaverse_xl", "partnet_mobility", "maniskill3"])
def test_adapter_has_name_and_version(name):
    adapter = get_adapter(name)
    assert adapter.name == name
    assert adapter.version

@pytest.mark.parametrize("name", ["scannetpp", "arkitscenes", "objaverse_xl", "partnet_mobility", "maniskill3"])
def test_adapter_capabilities_keys(name):
    adapter = get_adapter(name)
    caps = adapter.capabilities()
    required_keys = {
        "scene_mesh", "object_mesh", "articulation", "deformable_mesh",
        "camera", "depth", "lidar", "tracks", "videos",
        "sdk_required", "license_gated", "supports_local_ingest",
    }
    assert required_keys.issubset(set(caps.keys())), f"Missing capability keys for {name}"
    for v in caps.values():
        assert isinstance(v, bool)

@pytest.mark.parametrize("name", ["scannetpp", "arkitscenes", "objaverse_xl", "partnet_mobility", "maniskill3"])
def test_adapter_is_dataset_adapter(name):
    adapter = get_adapter(name)
    assert isinstance(adapter, DatasetAdapter)
