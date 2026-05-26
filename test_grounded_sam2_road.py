from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_PROJECT = PROJECT_ROOT / "workspace" / "projects" / "video" / "codex_allframes_20260521_1600"
DEFAULT_PROMPT = "road. roadway. asphalt road. driving lane. lane marking."
ROAD_TERMS = ("road", "roadway", "asphalt", "lane", "street", "pavement", "driveway")

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class ZaiwuProbeConfig:
    gateway_url: str
    request_timeout_sec: float = 30.0
    job_timeout_sec: float = 1800.0
    job_poll_interval_sec: float = 1.0
    auto_start_workers: bool = True
    worker_run_group: str = "services"
    grounded_sam2_service: str = "services.grounding_dino_sam2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe services.grounding_dino_sam2 for road surface segmentation on one video frame."
    )
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Video project root containing project.toml")
    parser.add_argument("--frame", type=int, default=1, help="1-based frame index to test")
    parser.add_argument("--image", default="", help="Explicit image path to test instead of reading a project video frame")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Grounding text prompt")
    parser.add_argument("--output-dir", default="", help="Directory for raw JSON, mask, overlay, and summary")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.20, help="GroundingDINO text threshold")
    parser.add_argument("--include-all", action="store_true", help="Use all returned instances even if labels are not road-like")
    return parser.parse_args()


def read_input_image(project_root: Path, frame_idx: int, image_path: str | Path | None = None) -> tuple[np.ndarray, str]:
    if image_path:
        path = Path(image_path).expanduser().resolve()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError(f"Failed to JPEG-encode image: {path}")
        return image, base64.b64encode(buffer.tobytes()).decode("ascii")
    return read_frame_image(project_root, frame_idx)


def read_frame_image(project_root: Path, frame_idx: int) -> tuple[np.ndarray, str]:
    detection_path = (
        project_root
        / "outputs"
        / "03_object_detect"
        / "frames"
        / f"frame_{int(frame_idx):06d}"
        / "detections.json"
    )
    if detection_path.is_file():
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        image_b64 = str(payload.get("image_b64") or "")
        if image_b64:
            image = decode_image_b64(image_b64)
            return image, image_b64

    video_path = project_root / "input" / "video.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find frame image or input video under {project_root}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx) - 1))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_idx} from {video_path}")
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode selected frame")
    return frame, base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_image_b64(value: str) -> np.ndarray:
    raw = str(value).split(",", 1)[1] if "," in str(value) else str(value)
    data = base64.b64decode(raw)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image_b64")
    return image


def call_grounded_sam2(
    project_root: Path,
    *,
    frame_idx: int,
    image_b64: str,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> dict[str, Any]:
    _ = (box_threshold, text_threshold)
    config = load_zaiwu_config(project_root)
    gateway = make_zaiwu_gateway(config)
    return gateway.run_service_job(
        config.grounded_sam2_service,
        "gsam2_parse_frame",
        {
            "frame_idx": int(frame_idx),
            "timestamp": 0.0,
            "image_base64": image_b64,
            "text_prompt": prompt,
        },
        timeout_sec=config.job_timeout_sec,
    )


def load_zaiwu_config(project_root: Path) -> ZaiwuProbeConfig:
    config_path = project_root / "project.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Project config not found: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    settings = raw.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    zaiwu = settings.get("zaiwu", {})
    if not isinstance(zaiwu, dict):
        zaiwu = {}
    workspace = raw.get("workspace", {})
    video_pipeline = workspace.get("video_pipeline", {}) if isinstance(workspace, dict) else {}
    if not isinstance(video_pipeline, dict):
        video_pipeline = {}

    gateway_url = str(zaiwu.get("gateway_url") or video_pipeline.get("zaiwu_gateway_url") or "").strip()
    if not gateway_url:
        raise ValueError(f"No Zaiwu gateway URL found in {config_path}")

    return ZaiwuProbeConfig(
        gateway_url=gateway_url,
        request_timeout_sec=float(zaiwu.get("request_timeout_sec") or 30.0),
        job_timeout_sec=float(zaiwu.get("job_timeout_sec") or 1800.0),
        job_poll_interval_sec=float(zaiwu.get("job_poll_interval_sec") or 1.0),
        auto_start_workers=bool(zaiwu.get("auto_start_workers", True)),
        worker_run_group=str(zaiwu.get("worker_run_group") or "services"),
        grounded_sam2_service=str(zaiwu.get("grounded_sam2_service") or "services.grounding_dino_sam2"),
    )


def make_zaiwu_gateway(config: ZaiwuProbeConfig):
    from guanwu.video.clients.zaiwu import ZaiwuGatewayClient

    return ZaiwuGatewayClient(
        gateway_url=config.gateway_url,
        request_timeout_sec=config.request_timeout_sec,
        job_timeout_sec=config.job_timeout_sec,
        job_poll_interval_sec=config.job_poll_interval_sec,
        auto_start_workers=config.auto_start_workers,
        worker_run_group=config.worker_run_group,
    )


def write_outputs(
    payload: dict[str, Any],
    image: np.ndarray,
    output_dir: str | Path,
    *,
    include_all: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "road_gsam2_raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    mask, selected, decoded_count = build_road_mask(payload, image.shape[:2], include_all=include_all)
    mask_path = out_dir / "road_mask.png"
    cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

    overlay = make_overlay(image, mask)
    overlay_path = out_dir / "road_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    summary = {
        "frame_idx": int(payload.get("frame_idx", 0) or 0),
        "returned_count": len(payload.get("instances", [])) if isinstance(payload.get("instances"), list) else 0,
        "decoded_mask_count": decoded_count,
        "selected_count": len(selected),
        "mask_area_px": int(mask.sum()),
        "mask_fraction": float(mask.mean()) if mask.size else 0.0,
        "instances": selected,
    }
    summary_path = out_dir / "road_debug_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "raw_path": str(raw_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def build_road_mask(
    payload: dict[str, Any],
    shape: tuple[int, int],
    *,
    include_all: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    height, width = shape
    instances = payload.get("instances", [])
    if not isinstance(instances, list):
        instances = []

    selected: list[dict[str, Any]] = []
    decoded_masks: list[np.ndarray] = []
    decoded_count = 0
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        label = str(inst.get("concept_label") or inst.get("label") or "").strip().lower()
        is_road = include_all or any(term in label for term in ROAD_TERMS)
        mask = decode_instance_mask(inst, (height, width))
        if mask is None:
            continue
        decoded_count += 1
        if not is_road:
            continue
        decoded_masks.append(mask)
        selected.append(
            {
                "object_id": str(inst.get("object_id") or inst.get("track_id") or ""),
                "label": label,
                "score": float(inst.get("score", 0.0) or 0.0),
                "bbox": [float(v) for v in (inst.get("bbox") or [])[:4]],
                "area_px": int(mask.sum()),
                "area_fraction": float(mask.mean()) if mask.size else 0.0,
            }
        )

    if not decoded_masks and not include_all:
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            mask = decode_instance_mask(inst, (height, width))
            if mask is None:
                continue
            decoded_masks.append(mask)
            label = str(inst.get("concept_label") or inst.get("label") or "").strip().lower()
            selected.append(
                {
                    "object_id": str(inst.get("object_id") or inst.get("track_id") or ""),
                    "label": label,
                    "score": float(inst.get("score", 0.0) or 0.0),
                    "bbox": [float(v) for v in (inst.get("bbox") or [])[:4]],
                    "area_px": int(mask.sum()),
                    "area_fraction": float(mask.mean()) if mask.size else 0.0,
                    "fallback_selected": True,
                }
            )

    if not decoded_masks:
        return np.zeros((height, width), dtype=bool), selected, decoded_count
    return np.logical_or.reduce(decoded_masks).astype(bool), selected, decoded_count


def decode_instance_mask(inst: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    raw = inst.get("mask_rle") or inst.get("mask")
    if raw:
        try:
            rle = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if isinstance(rle.get("counts"), list):
                mask = decode_uncompressed_rle(rle, shape)
                if mask is not None:
                    return mask
            counts = rle.get("counts")
            if isinstance(counts, str):
                rle["counts"] = counts.encode("ascii")
            from pycocotools import mask as mask_utils

            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            mask = decoded.astype(bool)
            if mask.shape == shape:
                return mask
        except Exception:
            pass

    bbox = inst.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        height, width = shape
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            return mask
    return None


def decode_uncompressed_rle(rle: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    size = rle.get("size")
    counts = rle.get("counts")
    if not (isinstance(size, list) and len(size) >= 2 and isinstance(counts, list)):
        return None
    height, width = int(size[0]), int(size[1])
    if (height, width) != shape:
        return None
    values: list[int] = []
    fill = 0
    for count in counts:
        try:
            run = int(count)
        except (TypeError, ValueError):
            return None
        if run < 0:
            return None
        values.extend([fill] * run)
        fill = 1 - fill
    expected = height * width
    if len(values) < expected:
        values.extend([0] * (expected - len(values)))
    if len(values) > expected:
        values = values[:expected]
    return np.asarray(values, dtype=np.uint8).reshape((height, width), order="F").astype(bool)


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    if mask.any():
        color = np.zeros_like(image)
        color[:, :] = (0, 220, 60)
        overlay[mask] = cv2.addWeighted(image[mask], 1.0 - alpha, color[mask], alpha, 0.0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def main() -> None:
    args = parse_args()
    project_root = Path(args.project).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve() if args.image else None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    elif image_path is not None:
        output_dir = image_path.parent / "grounded_sam2_road_probe"
    else:
        output_dir = project_root / "outputs" / "road_gsam2_probe" / f"frame_{int(args.frame):06d}"

    image, image_b64 = read_input_image(project_root, args.frame, image_path)
    payload = call_grounded_sam2(
        project_root,
        frame_idx=args.frame,
        image_b64=image_b64,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    result = write_outputs(payload, image, output_dir, include_all=args.include_all)

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"raw:     {result['raw_path']}")
    print(f"mask:    {result['mask_path']}")
    print(f"overlay: {result['overlay_path']}")
    print(f"summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
