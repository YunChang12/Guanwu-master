from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import test_grounded_sam2_road as road_script


class _FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def run_service_job(self, service_id: str, operation: str, payload: dict, **kwargs) -> dict:
        _ = kwargs
        self.calls.append((service_id, operation, payload))
        return {"frame_idx": payload["frame_idx"], "instances": []}


class _FakeContext:
    def __init__(self, project_root: Path) -> None:
        _ = project_root
        self.config = type(
            "Config",
            (),
            {
                "settings": type(
                    "Settings",
                    (),
                    {
                        "zaiwu": type(
                            "Zaiwu",
                            (),
                            {
                                "grounded_sam2_service": "services.grounding_dino_sam2",
                                "job_timeout_sec": 1800.0,
                            },
                        )()
                    },
                )()
            },
        )()


class _FakeZaiwuConfig:
    gateway_url = "http://zaiwu.local:8181"
    request_timeout_sec = 30.0
    job_timeout_sec = 1800.0
    job_poll_interval_sec = 1.0
    auto_start_workers = True
    worker_run_group = "services"
    grounded_sam2_service = "services.grounding_dino_sam2"


def _encode_rect_rle(mask: np.ndarray) -> dict:
    flat = mask.astype(np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run = 0
    for value in flat:
        if int(value) == current:
            run += 1
            continue
        counts.append(run)
        current = int(value)
        run = 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def test_build_road_mask_filters_road_labels_and_writes_overlay(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    road_mask = np.zeros((20, 30), dtype=np.uint8)
    road_mask[10:19, 3:25] = 1
    car_mask = np.zeros((20, 30), dtype=np.uint8)
    car_mask[12:17, 8:14] = 1
    payload = {
        "frame_idx": 1,
        "timestamp": 0.0,
        "instances": [
            {
                "object_id": "trk_road",
                "concept_label": "asphalt road",
                "bbox": [3.0, 10.0, 25.0, 19.0],
                "score": 0.82,
                "mask_rle": json.dumps(_encode_rect_rle(road_mask)),
            },
            {
                "object_id": "trk_car",
                "concept_label": "car",
                "bbox": [8.0, 12.0, 14.0, 17.0],
                "score": 0.91,
                "mask_rle": _encode_rect_rle(car_mask),
            },
        ],
    }

    result = road_script.write_outputs(payload, image, tmp_path)

    mask = cv2.imread(str(result["mask_path"]), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert int(mask[11, 5]) == 255
    assert int(mask[14, 10]) == 255
    assert int(mask[2, 2]) == 0
    assert result["summary"]["selected_count"] == 1
    assert result["summary"]["instances"][0]["label"] == "asphalt road"
    assert Path(result["overlay_path"]).is_file()


def test_call_grounded_sam2_frame_payload_omits_thresholds(monkeypatch, tmp_path: Path) -> None:
    gateway = _FakeGateway()

    monkeypatch.setattr(road_script, "load_zaiwu_config", lambda project_root: _FakeZaiwuConfig())
    monkeypatch.setattr(road_script, "make_zaiwu_gateway", lambda config: gateway)

    result = road_script.call_grounded_sam2(
        tmp_path,
        frame_idx=1,
        image_b64="ZmFrZQ==",
        prompt="road.",
        box_threshold=0.25,
        text_threshold=0.20,
    )

    assert result == {"frame_idx": 1, "instances": []}
    assert gateway.calls
    service_id, operation, payload = gateway.calls[0]
    assert service_id == "services.grounding_dino_sam2"
    assert operation == "gsam2_parse_frame"
    assert payload == {
        "frame_idx": 1,
        "timestamp": 0.0,
        "image_base64": "ZmFrZQ==",
        "text_prompt": "road.",
    }


def test_read_input_image_prefers_explicit_image_path(tmp_path: Path) -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[2:5, 3:9] = (10, 120, 240)
    image_path = tmp_path / "clean_target_rgb.png"
    assert cv2.imwrite(str(image_path), image)

    loaded, image_b64 = road_script.read_input_image(tmp_path, frame_idx=99, image_path=image_path)

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)
    decoded = road_script.decode_image_b64(image_b64)
    assert decoded.shape == image.shape
