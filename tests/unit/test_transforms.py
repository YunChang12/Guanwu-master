"""Tests for coordinate and time transforms."""
from __future__ import annotations

import numpy as np
import pytest

from guanwu.core.transforms import (
    convert_units,
    convert_up_axis,
    identity_4x4,
    is_valid_transform,
    normalize_transform,
    timestamp_from_frame_index,
    timestamp_from_seconds,
)


def test_identity():
    ident = identity_4x4()
    assert len(ident) == 16
    mat = np.array(ident).reshape(4, 4)
    np.testing.assert_array_almost_equal(mat, np.eye(4))


def test_convert_up_axis_z_noop():
    mat = np.eye(4)
    result = convert_up_axis(mat, "Z")
    np.testing.assert_array_almost_equal(result, mat)


def test_convert_up_axis_y():
    mat = np.eye(4)
    mat[1, 3] = 1.0  # translate 1 unit in Y
    result = convert_up_axis(mat, "Y")
    # Y-up to Z-up should swap Y and Z
    assert result.shape == (4, 4)


def test_convert_units_meters_noop():
    mat = np.eye(4)
    mat[:3, 3] = [1.0, 2.0, 3.0]
    result = convert_units(mat, "meters")
    np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])


def test_convert_units_centimeters():
    mat = np.eye(4)
    mat[:3, 3] = [100.0, 200.0, 300.0]
    result = convert_units(mat, "centimeters")
    np.testing.assert_array_almost_equal(result[:3, 3], [1.0, 2.0, 3.0])


def test_normalize_transform_list():
    mat_list = np.eye(4).flatten().tolist()
    result = normalize_transform(mat_list, source_up="Z", source_unit="meters")
    assert len(result) == 16


def test_is_valid_transform_identity():
    assert is_valid_transform(np.eye(4).flatten().tolist())


def test_is_valid_transform_none():
    assert is_valid_transform(None)


def test_is_valid_transform_bad_length():
    assert not is_valid_transform([1, 2, 3])


def test_is_valid_transform_bad_bottom_row():
    mat = np.eye(4)
    mat[3, 0] = 1.0
    assert not is_valid_transform(mat.flatten().tolist())


def test_timestamp_from_seconds():
    assert timestamp_from_seconds(1.0) == 1_000_000_000
    assert timestamp_from_seconds(0.5) == 500_000_000


def test_timestamp_from_frame_index():
    # 30 fps, frame 30 = 1 second
    assert timestamp_from_frame_index(30, 30.0) == 1_000_000_000
    assert timestamp_from_frame_index(0, 30.0) == 0
