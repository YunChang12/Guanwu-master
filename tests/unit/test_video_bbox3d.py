from __future__ import annotations

import pytest

from guanwu.video.core.schema import BBox3D, Geometry
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.features.spatial.state_estimator import CameraPose, StateEstimationAgent
from guanwu.video.materialize import _track_center, _track_size
from guanwu.video.project.executor import ProjectExecutor


class _FakeMetricDepthProvider:
    is_metric = True

    def depth_values(
        self,
        image_b64: str | None,
        samples_uv: list[tuple[float, float]],
        frame_idx: int = 0,
    ) -> list[float]:
        return [2.0 for _ in samples_uv]


def test_geometry_accepts_explicit_3d_bbox() -> None:
    bbox = BBox3D(
        type="aabb",
        center=[1.0, 2.0, 3.0],
        size=[0.5, 0.6, 0.7],
        corners=[
            [0.75, 1.7, 2.65],
            [1.25, 1.7, 2.65],
            [0.75, 2.3, 2.65],
            [1.25, 2.3, 2.65],
            [0.75, 1.7, 3.35],
            [1.25, 1.7, 3.35],
            [0.75, 2.3, 3.35],
            [1.25, 2.3, 3.35],
        ],
        frame="world",
        source="geometry_lift_depth",
        confidence=0.83,
    )

    geometry = Geometry(bbox_3d=bbox)
    dumped = geometry.model_dump(mode="json")

    assert dumped["bbox_3d"]["type"] == "aabb"
    assert dumped["bbox_3d"]["center"] == [1.0, 2.0, 3.0]
    assert dumped["bbox_3d"]["size"] == [0.5, 0.6, 0.7]
    assert len(dumped["bbox_3d"]["corners"]) == 8


def test_state_estimator_emits_bbox3d_for_metric_geometry() -> None:
    estimator = StateEstimationAgent(camera_provider="none", depth_provider="wildgs")
    estimator._depth_provider_impl = _FakeMetricDepthProvider()
    estimator._sam3d_camera = CameraPose(
        frame_id=1,
        timestamp_sec=0.0,
        K=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
        R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t=[0.0, 0.0, 0.0],
        pose_quality=0.9,
    )
    estimator._camera_mode = "sam3d_body"
    detections = FrameDetections(
        frame_idx=1,
        timestamp=0.0,
        image_b64="ZmFrZQ==",
        instances=[
            DetectedInstance(
                mask_ref="mask://frame_000001/obj_000001",
                bbox=[10.0, 20.0, 30.0, 50.0],
                object_id="src_1",
                concept_label="cup",
                score=0.9,
            )
        ],
    )

    nodes = estimator.estimate(detections)
    bbox = nodes[0].geometry.bbox_3d
    trajectory_bbox = estimator.pit_snapshot()["object_trajectories"]["obj_000001"][0]["bbox_3d"]

    assert bbox is not None
    assert bbox.type == "aabb"
    assert bbox.frame == "world"
    assert bbox.source == "geometry_lift_depth"
    assert len(bbox.corners or []) == 8
    assert trajectory_bbox["center"] == pytest.approx(bbox.center)
    assert trajectory_bbox["size"] == pytest.approx(bbox.size)
    for actual, expected in zip(trajectory_bbox["corners"], bbox.corners or [], strict=True):
        assert actual == pytest.approx(expected)


def test_materialize_prefers_standard_bbox3d_over_legacy_fields() -> None:
    point = {
        "centroid_world": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "bbox_3d": {
            "center": [1.5, 2.5, 3.5],
            "size": [0.4, 0.5, 0.6],
        },
    }

    assert _track_center(point) == (1.5, 2.5, 3.5)
    assert _track_size(point) == (0.4, 0.5, 0.6)


def test_pose_optimizer_track_frame_emits_oriented_bbox3d() -> None:
    frame = ProjectExecutor._edge_pose_track_frame(
        {
            "frame_id": 3,
            "timestamp_sec": 0.2,
            "pose": {
                "translation_world": [1.0, 2.0, 3.0],
                "rotation_matrix": [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "scale": [2.0, 4.0, 6.0],
            },
            "metrics": {
                "mask_iou": 0.9,
                "bbox_iou": 0.8,
                "bbox_center_error_px": 6.0,
            },
        },
        pose_source="edge_contour_fast_temporal",
    )

    assert frame is not None
    bbox = frame["bbox_3d"]
    assert bbox["type"] == "obb"
    assert bbox["center"] == [1.0, 2.0, 3.0]
    assert bbox["size"] == [2.0, 4.0, 6.0]
    assert bbox["orientation_quat"] == frame["orientation_quat"]
    assert bbox["frame"] == "world"
    assert bbox["source"] == "edge_contour_fast_temporal"
    assert len(bbox["corners"]) == 8
    assert bbox["corners"][0] == pytest.approx([3.0, 1.0, 0.0])


def test_refined_pose_trajectories_keep_bbox3d() -> None:
    refined = ProjectExecutor._refined_trajectories_from_pose_tracks(
        {
            "obj_000001": {
                "pose_source": "edge_contour_fast_temporal",
                "frames": [
                    {
                        "frame_id": 1,
                        "timestamp_sec": 0.1,
                        "centroid_world": [1.0, 2.0, 3.0],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                        "scale": [1.2, 1.4, 1.6],
                        "confidence": 0.9,
                        "source": "edge_contour_fast_temporal",
                    }
                ],
            }
        }
    )

    bbox = refined["obj_000001"][0]["bbox_3d"]

    assert bbox["type"] == "obb"
    assert bbox["center"] == [1.0, 2.0, 3.0]
    assert bbox["size"] == [1.2, 1.4, 1.6]
    assert len(bbox["corners"]) == 8


def test_corrected_trajectory_bbox3d_refreshes_from_current_pose() -> None:
    trajectories = {
        "obj_000001": {
            "frames": [
                {
                    "frame_id": 1,
                    "centroid_world": [4.0, 5.0, 6.0],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                    "scale": [2.0, 2.0, 2.0],
                    "trajectory_smoothing": {"applied": True},
                    "bbox_3d": {
                        "type": "obb",
                        "center": [0.0, 0.0, 0.0],
                        "size": [1.0, 1.0, 1.0],
                    },
                }
            ]
        }
    }

    ProjectExecutor._refresh_corrected_trajectory_bbox3d(trajectories)
    bbox = trajectories["obj_000001"]["frames"][0]["bbox_3d"]

    assert bbox["center"] == [4.0, 5.0, 6.0]
    assert bbox["size"] == [1.0, 1.0, 1.0]
    assert bbox["corners"][0] == pytest.approx([3.5, 4.5, 5.5])
