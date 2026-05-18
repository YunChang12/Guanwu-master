from __future__ import annotations

from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.features.spatial.state_estimator import StateEstimationAgent


def _sample_detections() -> FrameDetections:
    return FrameDetections(
        frame_idx=1,
        timestamp=0.0,
        image_b64="ZmFrZQ==",
        instances=[
            DetectedInstance(
                mask_ref="mask://frame_00001/obj_000001",
                bbox=[10.0, 10.0, 40.0, 40.0],
                object_id="obj_000001",
                concept_label="car",
                segment_kind="object",
                score=0.9,
            )
        ],
    )


def test_state_estimator_leaves_geometry_unknown_when_depth_provider_unavailable() -> None:
    estimator = StateEstimationAgent(camera_provider="none", depth_provider="zaiwu_depth_anything3")

    nodes = estimator.estimate(_sample_detections())

    assert len(nodes) == 1
    assert nodes[0].geometry.pose_3d.position is None
    assert nodes[0].geometry.scale_3d is None
    assert nodes[0].physics.velocity_linear is None
    assert estimator.pit_snapshot()["metric_enabled"] is False


def test_state_estimator_leaves_geometry_unknown_when_wildgs_assets_are_not_loaded() -> None:
    estimator = StateEstimationAgent(camera_provider="wildgs", depth_provider="wildgs")

    nodes = estimator.estimate(_sample_detections())

    assert len(nodes) == 1
    assert nodes[0].geometry.pose_3d.position is None
    assert nodes[0].geometry.scale_3d is None
    assert estimator.pit_snapshot()["metric_enabled"] is False
