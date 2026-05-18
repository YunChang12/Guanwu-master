"""
Shared fixtures for guanwu.video.model_backend tests.

All tests in this directory rely on:
  - `video_path`       – path to the bundled test video
  - `backend_client`   – httpx client for calling a running backend (requires --backend-url)

First frame is extracted from video via extract_first_frame_b64() when needed.

Run API tests against a live backend:
  pytest src/guanwu.video.model_backend/test/test_api_*.py --backend-url=http://localhost:8000
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).resolve().parent
VIDEO_PATH = TEST_DIR / "bridge_cropped_removed.mp4"


# ---------------------------------------------------------------------------
# Pytest hook
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--backend-url",
        action="store",
        default=None,
        help="Base URL of the model backend service to test (e.g. http://localhost:8000)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dump_response(resp: httpx.Response, *, truncate_b64: bool = True) -> None:
    """Print response status and JSON body. Long base64 fields are truncated for readability."""
    try:
        data = resp.json()
    except Exception:
        print(f"  [HTTP {resp.status_code}] (non-JSON body, {len(resp.content)} bytes)")
        return

    if truncate_b64:
        data = _truncate_b64_fields(data)
    print(f"  [HTTP {resp.status_code}] {json.dumps(data, indent=2, ensure_ascii=False)}")


def _truncate_b64_fields(obj, *, max_len: int = 60):
    """Recursively truncate long string values that look like base64."""
    if isinstance(obj, dict):
        return {k: _truncate_b64_fields(v, max_len=max_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_b64_fields(v, max_len=max_len) for v in obj]
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:40] + f"...({len(obj)} chars)"
    return obj


def extract_first_frame_b64(video_path: Path) -> str:
    """Extract the first video frame and return it as a base64-encoded JPEG string."""
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python not installed; skipping frame-based test")

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    assert ok, "Could not read first frame from test video"

    success, buf = cv2.imencode(".jpg", frame)
    assert success, "Could not encode frame as JPEG"
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def video_path() -> Path:
    """Return the path to the bundled test video."""
    assert VIDEO_PATH.exists(), f"Test video not found: {VIDEO_PATH}"
    return VIDEO_PATH


@pytest.fixture(scope="session")
def backend_url(request: pytest.FixtureRequest) -> str:
    """Backend base URL from --backend-url. Skips if not provided."""
    url = request.config.getoption("--backend-url", default=None)
    if not url:
        pytest.skip("API tests require --backend-url (e.g. --backend-url=http://localhost:8000)")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def backend_client(backend_url: str):
    """httpx client for calling the model backend service.

    trust_env=False bypasses HTTP_PROXY so localhost requests reach the backend directly.
    """
    import httpx

    return httpx.Client(base_url=backend_url, timeout=120.0, trust_env=False)
