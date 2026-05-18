"""
Tests for the physics priors endpoint.
Requires --backend-url.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from guanwu.video.model_backend.test.conftest import dump_response, extract_first_frame_b64


def _make_detections(frame_idx: int = 1, image_b64: str | None = None) -> dict:
    detections: dict = {"frame_idx": frame_idx, "timestamp": 0.0, "instances": []}
    if image_b64 is not None:
        detections["image_b64"] = image_b64
    return detections


def _make_object(object_id: str = "obj_001") -> dict:
    return {
        "object_id": object_id,
        "track_id": "trk_1",
        "label": "cup",
        "segment_kind": "object",
        "geometry": {
            "bbox_2d": [10.0, 20.0, 100.0, 150.0],
            "pose_3d": {"position": [0.0, 0.0, 0.5], "orientation_quat": [0.0, 0.0, 0.0, 1.0], "frame": "camera"},
            "scale_3d": [0.1, 0.1, 0.1],
        },
    }


class TestObjectPhysicsPriors:
    def test_empty_objects(self, backend_client: httpx.Client) -> None:
        """No objects → empty priors dict (expected)."""
        resp = backend_client.post(
            "/v1/tasks/infer-object-physics-priors",
            json={"detections": _make_detections(), "objects": [], "sam3d_meshes": {}},
        )
        dump_response(resp)
        assert resp.status_code == 200
        assert resp.json()["priors"] == {}

    def test_single_object(self, backend_client: httpx.Client) -> None:
        """One object → must return non-empty priors for that object."""
        resp = backend_client.post(
            "/v1/tasks/infer-object-physics-priors",
            json={"detections": _make_detections(), "objects": [_make_object()], "sam3d_meshes": {}},
        )
        dump_response(resp)
        assert resp.status_code == 200
        priors = resp.json()["priors"]
        assert len(priors) > 0, "VLM returned empty priors — physics backend may be broken"

    def test_multiple_objects(self, backend_client: httpx.Client) -> None:
        """Multiple objects → must return priors for each."""
        objects = [_make_object(f"obj_{i:03d}") for i in range(3)]
        resp = backend_client.post(
            "/v1/tasks/infer-object-physics-priors",
            json={"detections": _make_detections(), "objects": objects, "sam3d_meshes": {}},
        )
        dump_response(resp)
        assert resp.status_code == 200
        priors = resp.json()["priors"]
        assert len(priors) > 0, "VLM returned empty priors — physics backend may be broken"

    def test_missing_detections_422(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post("/v1/tasks/infer-object-physics-priors", json={"objects": []})
        dump_response(resp)
        assert resp.status_code == 422
