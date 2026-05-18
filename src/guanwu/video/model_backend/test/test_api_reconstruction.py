"""
Tests for the reconstruction endpoint.
Requires --backend-url.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from guanwu.video.model_backend.test.conftest import dump_response, extract_first_frame_b64


def _make_detections(frame_idx: int = 1, timestamp: float = 0.0, image_b64: str | None = None) -> dict:
    detections: dict = {"frame_idx": frame_idx, "timestamp": timestamp, "instances": []}
    if image_b64 is not None:
        detections["image_b64"] = image_b64
    return detections


def _make_object(object_id: str = "obj_001", track_id: str = "trk_1", label: str = "cup") -> dict:
    return {
        "object_id": object_id,
        "track_id": track_id,
        "label": label,
        "segment_kind": "object",
        "geometry": {
            "bbox_2d": [100.0, 200.0, 300.0, 400.0],
            "pose_3d": {
                "position": [0.0, 0.0, 0.5],
                "orientation_quat": [0.0, 0.0, 0.0, 1.0],
                "frame": "camera",
            },
            "scale_3d": [0.1, 0.1, 0.1],
        },
    }


class TestObjectMeshReconstruction:
    def test_empty_objects(self, backend_client: httpx.Client) -> None:
        """No objects → empty meshes dict (expected)."""
        resp = backend_client.post(
            "/v1/tasks/reconstruct-object-meshes",
            json={"detections": _make_detections(), "objects": []},
        )
        dump_response(resp)
        assert resp.status_code == 200
        assert resp.json()["meshes"] == {}

    def test_with_real_image(self, backend_client: httpx.Client, video_path: Path) -> None:
        """Real image + object → must return non-empty meshes."""
        first_frame_b64 = extract_first_frame_b64(video_path)
        resp = backend_client.post(
            "/v1/tasks/reconstruct-object-meshes",
            json={
                "detections": _make_detections(image_b64=first_frame_b64),
                "objects": [_make_object()],
            },
        )
        dump_response(resp)
        assert resp.status_code == 200
        meshes = resp.json()["meshes"]
        assert isinstance(meshes, dict)
        assert len(meshes) > 0, "SAM3D returned empty meshes — reconstruction backend may be broken"

    def test_missing_detections_422(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post("/v1/tasks/reconstruct-object-meshes", json={"objects": []})
        dump_response(resp)
        assert resp.status_code == 422
