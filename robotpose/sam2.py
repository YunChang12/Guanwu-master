from __future__ import annotations

import base64
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROBOT_LABEL_TERMS = ("robot", "arm", "manipulator", "gripper", "机械臂", "机器人")


@dataclass(frozen=True)
class GroundedSAM2Config:
    gateway_url: str
    request_timeout_sec: float = 30.0
    job_timeout_sec: float = 1800.0
    job_poll_interval_sec: float = 1.0
    auto_start_workers: bool = True
    worker_run_group: str = "services"
    service_name: str = "services.grounding_dino_sam2"


def decode_uncompressed_rle(rle: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    counts = rle.get("counts")
    size = rle.get("size")
    if not isinstance(counts, list):
        return None
    height, width = shape
    if isinstance(size, list) and len(size) >= 2:
        height, width = int(size[0]), int(size[1])
    total = int(height) * int(width)
    flat = np.zeros(total, dtype=np.uint8)
    index = 0
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            return None
        end = min(total, index + count)
        if value == 1 and end > index:
            flat[index:end] = 1
        index = end
        value = 1 - value
        if index >= total:
            break
    mask = flat.reshape((int(width), int(height))).T.astype(bool)
    if mask.shape != tuple(shape):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(mask_img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    return mask


def decode_compressed_rle(rle: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    counts_raw = rle.get("counts")
    size = rle.get("size")
    if not isinstance(counts_raw, str):
        return None
    if not (isinstance(size, list) and len(size) >= 2):
        return None
    height, width = int(size[0]), int(size[1])
    counts = _decode_coco_compressed_counts(counts_raw)
    if counts is None:
        return None
    return _mask_from_counts(counts, (height, width), shape)


def _decode_coco_compressed_counts(value: str) -> list[int] | None:
    counts: list[int] = []
    index = 0
    text = str(value)
    while index < len(text):
        shift = 0
        count = 0
        more = True
        while more:
            if index >= len(text):
                return None
            char_value = ord(text[index]) - 48
            index += 1
            count |= (char_value & 0x1F) << shift
            more = bool(char_value & 0x20)
            shift += 5
            if not more and (char_value & 0x10):
                count |= -1 << shift
        if len(counts) > 2:
            count += counts[-2]
        if count < 0:
            return None
        counts.append(int(count))
    return counts


def _mask_from_counts(counts: list[int], source_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray | None:
    height, width = [int(v) for v in source_shape]
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    index = 0
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0:
            return None
        end = min(total, index + count)
        if value == 1 and end > index:
            flat[index:end] = 1
        index = end
        value = 1 - value
        if index >= total:
            break
    mask = flat.reshape((width, height)).T.astype(bool)
    if mask.shape != tuple(target_shape):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(mask_img.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)) > 0
    return mask


def decode_instance_mask(inst: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    raw = inst.get("mask_rle") or inst.get("mask")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            raw = parsed
    if isinstance(raw, dict):
        decoded = decode_uncompressed_rle(raw, shape)
        if decoded is not None:
            return decoded
        decoded = decode_compressed_rle(raw, shape)
        if decoded is not None:
            return decoded
        try:
            from pycocotools import mask as mask_utils  # type: ignore

            rle = raw.copy()
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
            mask = mask_utils.decode(rle).astype(bool)
            if mask.shape != tuple(shape):
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
            return mask
        except Exception:
            return None
    if isinstance(raw, list):
        arr = np.asarray(raw)
        if arr.ndim == 2:
            mask = arr.astype(bool)
            if mask.shape != tuple(shape):
                mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
            return mask
    bbox = inst.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        height, width = shape
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask
    return None


def mask_from_grounded_sam2_payload(
    payload: dict[str, Any],
    shape: tuple[int, int],
    *,
    label_terms: tuple[str, ...] = ROBOT_LABEL_TERMS,
    include_all: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        instances = []
    selected: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    terms = tuple(term.lower() for term in label_terms)
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        label = str(inst.get("concept_label") or inst.get("label") or "").strip().lower()
        if not include_all and terms and not any(term in label for term in terms):
            continue
        mask = decode_instance_mask(inst, shape)
        if mask is None or not mask.any():
            continue
        masks.append(mask)
        selected.append(
            {
                "label": label,
                "score": float(inst.get("score", 0.0) or 0.0),
                "bbox": [float(v) for v in (inst.get("bbox") or [])[:4]],
                "area_px": int(mask.sum()),
            }
        )
    if not masks:
        return np.zeros(shape, dtype=bool), selected
    return np.logical_or.reduce(masks).astype(bool), selected


def load_grounded_sam2_config(project_root: str | Path) -> GroundedSAM2Config:
    config_path = Path(project_root).expanduser().resolve() / "project.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Project config not found for Grounded-SAM2: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    settings = raw.get("settings", {}) if isinstance(raw, dict) else {}
    zaiwu = settings.get("zaiwu", {}) if isinstance(settings, dict) else {}
    workspace = raw.get("workspace", {}) if isinstance(raw, dict) else {}
    video_pipeline = workspace.get("video_pipeline", {}) if isinstance(workspace, dict) else {}
    gateway_url = str(zaiwu.get("gateway_url") or video_pipeline.get("zaiwu_gateway_url") or "").strip()
    if not gateway_url:
        raise ValueError(f"No Zaiwu gateway URL found in {config_path}")
    return GroundedSAM2Config(
        gateway_url=gateway_url,
        request_timeout_sec=float(zaiwu.get("request_timeout_sec") or 30.0),
        job_timeout_sec=float(zaiwu.get("job_timeout_sec") or 1800.0),
        job_poll_interval_sec=float(zaiwu.get("job_poll_interval_sec") or 1.0),
        auto_start_workers=bool(zaiwu.get("auto_start_workers", True)),
        worker_run_group=str(zaiwu.get("worker_run_group") or "services"),
        service_name=str(zaiwu.get("grounded_sam2_service") or "services.grounding_dino_sam2"),
    )


def call_grounded_sam2(
    project_root: str | Path,
    image_rgb: np.ndarray,
    *,
    frame_idx: int = 0,
    prompt: str = "robot arm",
) -> dict[str, Any]:
    config = load_grounded_sam2_config(project_root)
    try:
        from guanwu.video.clients.zaiwu import ZaiwuGatewayClient
    except Exception as exc:
        raise RuntimeError("Guanwu Zaiwu client is required to call Grounded-SAM2.") from exc

    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    gateway = ZaiwuGatewayClient(
        gateway_url=config.gateway_url,
        request_timeout_sec=config.request_timeout_sec,
        job_timeout_sec=config.job_timeout_sec,
        job_poll_interval_sec=config.job_poll_interval_sec,
        auto_start_workers=config.auto_start_workers,
        worker_run_group=config.worker_run_group,
    )
    return gateway.run_service_job(
        config.service_name,
        "gsam2_parse_frame",
        {
            "frame_idx": int(frame_idx),
            "timestamp": 0.0,
            "image_base64": image_b64,
            "text_prompt": prompt,
        },
        timeout_sec=config.job_timeout_sec,
    )


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
