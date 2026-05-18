"""
Tests for the movable object discovery endpoint.
Requires --backend-url.
"""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

from guanwu.video.model_backend.test.conftest import dump_response, extract_first_frame_b64


class TestMovableObjectDiscovery:
    def test_with_real_frame(self, backend_client: httpx.Client, video_path: Path) -> None:
        """Real video frame → must return non-empty list of category strings."""
        first_frame_b64 = extract_first_frame_b64(video_path)
        resp = backend_client.post(
            "/v1/tasks/discover-movable-object-categories",
            json={"image_b64": first_frame_b64},
        )
        dump_response(resp)
        assert resp.status_code == 200
        cats = resp.json()["categories"]
        assert isinstance(cats, list)
        assert len(cats) > 0, "VLM returned empty categories — discovery backend may be broken"
        for item in cats:
            assert isinstance(item, str)

    def test_missing_image_b64_422(self, backend_client: httpx.Client) -> None:
        resp = backend_client.post("/v1/tasks/discover-movable-object-categories", json={})
        dump_response(resp)
        assert resp.status_code == 422
