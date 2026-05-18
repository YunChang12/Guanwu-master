"""
Tests for health / ready / provider-status API endpoints.
Requires --backend-url.
"""
from __future__ import annotations

import httpx

from guanwu.video.model_backend.test.conftest import dump_response


class TestHealth:
    def test_health(self, backend_client: httpx.Client) -> None:
        resp = backend_client.get("/v1/health")
        dump_response(resp)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestReady:
    def test_ready(self, backend_client: httpx.Client) -> None:
        resp = backend_client.get("/v1/ready")
        dump_response(resp)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["ready"], bool)
        assert isinstance(data["checks"], list)
        names = {c["name"] for c in data["checks"]}
        assert names == {"sam3", "sam3d", "vlm"}
        for check in data["checks"]:
            assert "mode" in check
            assert isinstance(check["ok"], bool)


class TestProviderStatus:
    def test_provider_status(self, backend_client: httpx.Client) -> None:
        resp = backend_client.get("/v1/providers/status")
        dump_response(resp)
        assert resp.status_code == 200
        data = resp.json()
        providers = data["providers"]
        assert isinstance(providers, list)
        assert len(providers) >= 1
        for p in providers:
            assert "name" in p
            assert "mode" in p
            assert isinstance(p["ok"], bool)
