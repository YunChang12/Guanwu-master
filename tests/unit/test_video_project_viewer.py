from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from guanwu.video.project.viewer import _load_state_payload, create_project_viewer_app


def _write_image(path: Path, *, size: tuple[int, int] = (32, 24), color: tuple[int, int, int] = (220, 180, 120)) -> str:
    image = Image.new("RGB", size, color)
    image.save(path, format="PNG")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project_root = tmp_path / "demo_project"
    state_dir = project_root / "state"
    detect_frame_dir = project_root / "outputs" / "03_object_detect" / "frames" / "frame_000001"
    inspect_dir = project_root / "outputs" / "01_video_inspect"
    frame_sample_dir = project_root / "outputs" / "02_frame_sample"

    detect_frame_dir.mkdir(parents=True)
    inspect_dir.mkdir(parents=True)
    frame_sample_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    first_frame_path = frame_sample_dir / "first_frame.png"
    overlay_path = detect_frame_dir / "overlay.jpg"
    summary_path = project_root / "outputs" / "03_object_detect" / "summary.json"
    detections_path = detect_frame_dir / "detections.json"
    metadata_path = inspect_dir / "video_metadata.json"

    image_b64 = _write_image(first_frame_path)
    _write_image(overlay_path, color=(140, 180, 220))

    metadata_path.write_text(
        json.dumps(
            {
                "video_path": "/tmp/demo.mp4",
                "frame_count": 1,
                "fps": 30.0,
                "width": 32,
                "height": 24,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    detections_path.write_text(
        json.dumps(
            {
                "frame_idx": 1,
                "timestamp": 0.0,
                "image_b64": image_b64,
                "instances": [
                    {
                        "object_id": "obj_000001",
                        "concept_label": "box",
                        "segment_kind": "object",
                        "score": 0.91,
                        "bbox": [4, 5, 18, 16],
                        "mask_ref": "mask://frame_00001/obj_000001",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 1,
                        "timestamp": 0.0,
                        "detections": str(detections_path),
                        "overlay": str(overlay_path),
                        "instance_count": 1,
                    }
                ],
                "latest_detections": str(detections_path),
                "latest_instance_count": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (project_root / "project.toml").write_text(
        f"""
[project]
project_id = "demo_project"
name = "demo_project"
input_video = "/tmp/demo.mp4"
root_dir = "{project_root}"
provider_mode = "zaiwu"
video_copy_mode = "copy"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (state_dir / "manifest.json").write_text(
        json.dumps(
            {
                "project_id": "demo_project",
                "project_root": str(project_root),
                "input_video": "/tmp/demo.mp4",
                "created_at": "2026-04-22T00:00:00+00:00",
                "updated_at": "2026-04-22T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (state_dir / "stage_status.json").write_text(
        json.dumps(
            {
                "video.inspect": {
                    "stage": "video.inspect",
                    "status": "completed",
                    "last_run_at": "2026-04-22T00:00:00+00:00",
                    "error": None,
                    "inputs_hash": "a",
                    "params_hash": "b",
                },
                "frame.sample": {
                    "stage": "frame.sample",
                    "status": "completed",
                    "last_run_at": "2026-04-22T00:00:01+00:00",
                    "error": None,
                    "inputs_hash": "a",
                    "params_hash": "b",
                },
                "object.detect": {
                    "stage": "object.detect",
                    "status": "completed",
                    "last_run_at": "2026-04-22T00:00:02+00:00",
                    "error": None,
                    "inputs_hash": "a",
                    "params_hash": "b",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (state_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "video.inspect": {
                    "stage": "video.inspect",
                    "created_at": "2026-04-22T00:00:00+00:00",
                    "inputs_hash": "a",
                    "params_hash": "b",
                    "outputs": {
                        "video_metadata": str(metadata_path),
                    },
                    "summary": {
                        "frame_count": 1,
                    },
                },
                "frame.sample": {
                    "stage": "frame.sample",
                    "created_at": "2026-04-22T00:00:01+00:00",
                    "inputs_hash": "a",
                    "params_hash": "b",
                    "outputs": {
                        "first_frame": str(first_frame_path),
                    },
                    "summary": {
                        "sample_count": 1,
                    },
                },
                "object.detect": {
                    "stage": "object.detect",
                    "created_at": "2026-04-22T00:00:02+00:00",
                    "inputs_hash": "a",
                    "params_hash": "b",
                    "outputs": {
                        "summary": str(summary_path),
                        "frames_dir": str(detect_frame_dir.parent),
                    },
                    "summary": {
                        "frame_count": 1,
                        "latest_instance_count": 1,
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return project_root, summary_path, detections_path, first_frame_path


def test_load_state_payload_includes_output_meta(tmp_path: Path) -> None:
    project_root, summary_path, _, _ = _build_project(tmp_path)

    payload = _load_state_payload(project_root)

    assert payload["project_name"] == "demo_project"
    assert payload["latest_completed_stage"] == "object.detect"
    assert payload["artifacts"]["object.detect"]["outputs_meta"]["summary"]["path"] == str(summary_path)
    assert payload["artifacts"]["object.detect"]["outputs_meta"]["summary"]["json_url"].startswith("/api/json?path=")


def test_project_viewer_endpoints(tmp_path: Path) -> None:
    project_root, summary_path, detections_path, first_frame_path = _build_project(tmp_path)
    client = TestClient(create_project_viewer_app(project_root))

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "Guanwu Project Viewer" in index_response.text

    state_response = client.get("/api/state")
    assert state_response.status_code == 200
    assert state_response.json()["project_root"] == str(project_root)

    json_response = client.get("/api/json", params={"path": str(summary_path)})
    assert json_response.status_code == 200
    assert json_response.json()["latest_instance_count"] == 1

    file_response = client.get("/api/file", params={"path": str(first_frame_path)})
    assert file_response.status_code == 200
    assert file_response.headers["content-type"].startswith("image/")

    render_response = client.get(
        "/api/object-detect/render",
        params={"path": str(detections_path)},
    )
    assert render_response.status_code == 200
    assert render_response.headers["content-type"] == "image/png"
    assert render_response.content.startswith(b"\x89PNG")
