"""Tests for taxonomy mapping."""
from __future__ import annotations

from guanwu.core.taxonomy import CanonicalCategory, map_category


def test_known_category():
    cat, _ = map_category("chair")
    assert cat == "furniture"


def test_case_insensitive():
    cat, _ = map_category("Chair")
    assert cat == "furniture"


def test_unknown_category():
    cat, _ = map_category("xyzzy_nonexistent")
    assert cat == "unknown"


def test_none_category():
    cat, _ = map_category(None)
    assert cat == "unknown"


def test_vehicle_mapping():
    cat, _ = map_category("car")
    assert cat == "vehicle"


def test_human_mapping():
    cat, _ = map_category("pedestrian")
    assert cat == "human"


def test_door_window():
    cat, _ = map_category("door")
    assert cat == "door_window"


def test_all_categories_are_valid():
    for c in CanonicalCategory:
        assert isinstance(c.value, str)
