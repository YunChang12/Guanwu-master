from __future__ import annotations

import numpy as np

# Convention: right-hand, meters, Z-up (canonical)
# OpenCV camera: +x right, +y down, +z forward

# Common up-axis transform matrices
_Y_UP_TO_Z_UP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

_X_UP_TO_Z_UP = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

UNIT_SCALES = {
    "meters": 1.0,
    "centimeters": 0.01,
    "millimeters": 0.001,
    "inches": 0.0254,
    "feet": 0.3048,
}


def convert_up_axis(transform: np.ndarray, source_up: str) -> np.ndarray:
    """Convert a 4x4 transform matrix from source up-axis to Z-up."""
    source_up = source_up.upper()
    if source_up == "Z":
        return transform.copy()
    elif source_up == "Y":
        return _Y_UP_TO_Z_UP @ transform
    elif source_up == "X":
        return _X_UP_TO_Z_UP @ transform
    else:
        raise ValueError(f"Unknown up axis: {source_up}")


def convert_units(transform: np.ndarray, source_unit: str) -> np.ndarray:
    """Scale the translation component of a 4x4 transform to meters."""
    source_unit = source_unit.lower()
    if source_unit not in UNIT_SCALES:
        raise ValueError(
            f"Unknown unit: {source_unit}. Known: {list(UNIT_SCALES.keys())}"
        )
    scale = UNIT_SCALES[source_unit]
    result = transform.copy()
    result[:3, 3] *= scale
    return result


def normalize_transform(
    transform: np.ndarray | list[float],
    source_up: str = "Z",
    source_unit: str = "meters",
) -> list[float]:
    """Normalize a transform matrix to canonical convention (Z-up, meters).

    Args:
        transform: 4x4 matrix as numpy array or 16-element list (row-major)
        source_up: Source up axis ("X", "Y", or "Z")
        source_unit: Source unit

    Returns:
        16-element list (row-major) of the canonical transform
    """
    if isinstance(transform, list):
        mat = np.array(transform, dtype=np.float64).reshape(4, 4)
    else:
        mat = np.array(transform, dtype=np.float64)

    mat = convert_units(mat, source_unit)
    mat = convert_up_axis(mat, source_up)
    return mat.flatten().tolist()


def is_valid_transform(matrix: list[float] | None) -> bool:
    """Check if a 4x4 transform matrix is valid."""
    if matrix is None:
        return True  # None is valid (means "not available")
    if len(matrix) != 16:
        return False
    mat = np.array(matrix).reshape(4, 4)
    # Bottom row should be [0, 0, 0, 1]
    if not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-6):
        return False
    # Rotation part should have determinant close to 1
    det = np.linalg.det(mat[:3, :3])
    if not np.isclose(abs(det), 1.0, atol=1e-3):
        return False
    return True


def timestamp_from_seconds(seconds: float) -> int:
    """Convert seconds to nanoseconds (int64)."""
    return int(seconds * 1_000_000_000)


def timestamp_from_frame_index(index: int, fps: float) -> int:
    """Convert frame index + fps to nanoseconds."""
    return int(index / fps * 1_000_000_000)


def identity_4x4() -> list[float]:
    """Return a 4x4 identity matrix as a 16-element list."""
    return np.eye(4, dtype=np.float64).flatten().tolist()
