"""Physics dynamics estimator.

Estimates kinematic quantities only from metric 3D trajectories produced by
geometry.lift. Unknown or non-metric inputs remain ``None`` instead of being
filled with pseudo-physical fallback values.

All quantities are in SI units:
  - positions: metres
  - velocities: m/s
  - accelerations: m/s²
  - angular velocities: rad/s when a trusted orientation track exists
  - mass: kg only when directly measured upstream
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)

# Savitzky-Golay-like simple smoothing window (box filter)
_SMOOTH_WINDOW = 3

# Motion class thresholds (scalar speed, m/s)
_SPEED_SLOW = 0.1
_SPEED_FAST = 5.0
_ACCEL_NOTABLE = 1.0


def _smooth(seq: list[float], window: int = _SMOOTH_WINDOW) -> list[float]:
    if len(seq) < window:
        return seq
    result = []
    half = window // 2
    for i in range(len(seq)):
        lo = max(0, i - half)
        hi = min(len(seq), i + half + 1)
        result.append(sum(seq[lo:hi]) / (hi - lo))
    return result


def _classify_motion(speeds: list[float], accels: list[float]) -> str:
    if not speeds:
        return "static"
    mean_speed = sum(speeds) / len(speeds)
    max_accel = max(abs(a) for a in accels) if accels else 0.0
    if mean_speed < _SPEED_SLOW:
        return "static"
    if max_accel >= _ACCEL_NOTABLE:
        # Determine dominant direction
        mean_accel = sum(accels) / len(accels)
        return "accelerating" if mean_accel > 0 else "decelerating"
    if mean_speed >= _SPEED_FAST:
        return "fast"
    return "slow"


def estimate_dynamics(
    trajectory: list[dict[str, Any]],
    attr: dict[str, Any],
) -> dict[str, Any]:
    """Estimate kinematics for a single object from its frame trajectory.

    Args:
        trajectory: list of {frame_idx, timestamp, centroid_3d: [x,y,z], ...}
                    sorted by timestamp.
        attr: object.attr entry for this object (used for mass calibration prior).

    Returns:
        Dict with velocity_mps, acceleration_mps2, angular_velocity_radps,
        speed_profile, motion_class, mass_kg_calibrated, static_friction,
        dynamic_friction, restitution.
    """
    _ = attr

    def _valid_point(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return False
        try:
            return all(math.isfinite(float(v)) for v in value[:3])
        except Exception:
            return False

    # Filter frames that have trusted metric centroids.
    frames = [f for f in trajectory if _valid_point(f.get("centroid_3d"))]
    frames.sort(key=lambda f: f.get("timestamp", f.get("frame_idx", 0)))

    n = len(frames)

    velocities: list[list[float]] | None = None
    accelerations: list[list[float]] | None = None
    speeds: list[float] = []
    accels_scalar: list[float] = []

    if n >= 2:
        velocities = []
        # First derivative: velocity
        for i in range(1, n):
            dt = float(frames[i].get("timestamp", i)) - float(frames[i - 1].get("timestamp", i - 1))
            if dt <= 0:
                continue
            p0 = np.array(frames[i - 1]["centroid_3d"][:3], dtype=float)
            p1 = np.array(frames[i]["centroid_3d"][:3], dtype=float)
            v = ((p1 - p0) / dt).tolist()
            velocities.append(v)
            speeds.append(float(np.linalg.norm(v)))

        # Smooth speeds
        speeds = _smooth(speeds)

        # Second derivative: acceleration
        if len(velocities) >= 2:
            accelerations = []
            for i in range(1, len(velocities)):
                dt = float(frames[i + 1].get("timestamp", i + 1)) - float(frames[i].get("timestamp", i))
                if dt <= 0:
                    continue
                v0 = np.array(velocities[i - 1])
                v1 = np.array(velocities[i])
                a = ((v1 - v0) / dt).tolist()
                accelerations.append(a)
                accels_scalar.append(float(np.linalg.norm(a)))

    # Smooth acceleration magnitudes
    accels_scalar = _smooth(accels_scalar)

    motion_class = _classify_motion(speeds, accels_scalar) if speeds else None

    return {
        "velocity_mps": velocities,
        "acceleration_mps2": accelerations,
        "angular_velocity_radps": None,
        "speed_profile": speeds or None,
        "motion_class": motion_class,
        "mass_kg_calibrated": None,
        "static_friction": None,
        "dynamic_friction": None,
        "restitution": None,
        "source": "metric_trajectory" if frames else None,
    }


class PhysicsDynamicsEstimator:
    """Estimates kinematic dynamics for all objects from geometry.lift trajectories."""

    def estimate(
        self,
        object_trajectories: dict[str, list[dict[str, Any]]],
        object_attrs: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Run dynamics estimation for all objects.

        Args:
            object_trajectories: {object_id: [frame_entry, ...]} from geometry.lift.
            object_attrs: {object_id: attr_dict} from object.attr.

        Returns:
            {object_id: dynamics_dict}
        """
        results: dict[str, dict[str, Any]] = {}
        for obj_id, trajectory in object_trajectories.items():
            attr = object_attrs.get(obj_id, {})
            try:
                results[obj_id] = estimate_dynamics(trajectory, attr)
                logger.info(
                    f"[PhysicsDynamics] {obj_id}: motion_class={results[obj_id]['motion_class']}, "
                    f"mass_calibrated={results[obj_id]['mass_kg_calibrated']}"
                )
            except Exception as e:
                logger.error(f"[PhysicsDynamics] Failed for {obj_id}: {e}")
        return results
