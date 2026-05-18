"""
Tests for perception endpoints.
Requires --backend-url.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from guanwu.video.model_backend.test.conftest import dump_response, extract_first_frame_b64

TEST_PROMPTS = ["dish rack", "plate", "carrot", "container"]


class TestObjectDetectionInFrame:
    def test_without_image(self, backend_client: httpx.Client) -> None:
        """No image -> returns valid empty detections."""
        resp = backend_client.post(
            "/v1/tasks/detect-objects-in-frame",
            json={"frame_idx": 1, "timestamp": 0.0, "prompts": TEST_PROMPTS},
        )
        dump_response(resp)
        assert resp.status_code == 200
        detections = resp.json()["detections"]
        assert detections["frame_idx"] == 1
        assert detections["timestamp"] == 0.0
        assert isinstance(detections["instances"], list)

    def test_with_real_image(self, backend_client: httpx.Client, video_path: Path) -> None:
        """Send a real video frame — YOLO must return detection instances."""
        first_frame_b64 = extract_first_frame_b64(video_path)
        resp = backend_client.post(
            "/v1/tasks/detect-objects-in-frame",
            json={
                "frame_idx": 1,
                "timestamp": 0.04,
                "prompts": TEST_PROMPTS,
                "image_b64": first_frame_b64,
            },
        )
        dump_response(resp)
        assert resp.status_code == 200
        instances = resp.json()["detections"]["instances"]
        assert isinstance(instances, list)
        assert len(instances) > 0, (
            "SAM3 returned zero instances — YOLO may not be detecting, "
            "or prompts don't match any YOLO labels"
        )

    def test_missing_prompts_422(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post(
            "/v1/tasks/detect-objects-in-frame",
            json={"frame_idx": 1, "timestamp": 0.0},
        )
        dump_response(resp)
        assert resp.status_code == 422

    def test_missing_required_fields_422(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post("/v1/tasks/detect-objects-in-frame", json={})
        dump_response(resp)
        assert resp.status_code == 422


class TestPrompts:
    def test_get_and_set_prompts(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post("/v1/object-detection/prompts", json={"prompts": TEST_PROMPTS})
        dump_response(resp)
        assert resp.status_code == 200
        assert isinstance(resp.json()["prompts"], list)

        resp = backend_client.get("/v1/object-detection/prompts")
        dump_response(resp)
        assert resp.status_code == 200
        assert isinstance(resp.json()["prompts"], list)

    def test_set_empty_prompts(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post("/v1/object-detection/prompts", json={"prompts": []})
        dump_response(resp)
        assert resp.status_code == 200


class TestPerceptionStatus:
    def test_status(self, backend_client: httpx.Client) -> None:
        resp = backend_client.get("/v1/detector/status")
        dump_response(resp)
        assert resp.status_code == 200
        assert isinstance(resp.json()["status"], dict)
