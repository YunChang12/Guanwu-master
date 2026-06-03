"""URDF + RGB robot base pose estimation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "RobotMesh",
    "RobotPoseResult",
    "OptimizerOptions",
    "estimate_robot_base_pose",
]


def __getattr__(name: str) -> Any:
    if name == "RobotMesh":
        from .render import RobotMesh

        return RobotMesh
    if name in {"RobotPoseResult", "OptimizerOptions"}:
        from .optimizer import OptimizerOptions, RobotPoseResult

        return {"OptimizerOptions": OptimizerOptions, "RobotPoseResult": RobotPoseResult}[name]
    if name == "estimate_robot_base_pose":
        from .estimate import estimate_robot_base_pose

        return estimate_robot_base_pose
    raise AttributeError(name)
