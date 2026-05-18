"""ObjectVideoExtractor — extract per-object masked video clips from a processed video.

For each movable object tracked by SAM2, writes one video file containing only the
frames in which the object is visible, cropped to its bounding box and optionally
masked at pixel level (when mask_rle is available).

Usage::

    extractor = ObjectVideoExtractor()

    # Inside the main loop, after perception:
    extractor.collect_frame(batch, observed_objects)

    # After the loop:
    extractor.export(
        movable_registry=runtime.movable_registry,
        output_dir=session_output_root / "object_videos",
        fps=30.0,
    )
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from guanwu.video.core.logger import get_logger
from guanwu.video.core.instance_matching import match_instances_to_objects
from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import FrameDetections

logger = get_logger(__name__)


@dataclass
class _FrameRecord:
    frame_idx: int
    image_b64: str
    bbox: list[float]   # [x1, y1, x2, y2] in pixels
    mask_rle: str | None


class ObjectVideoExtractor:
    """Accumulates per-frame observations and exports one video per movable object."""

    def __init__(self) -> None:
        # object_id -> ordered list of frame records
        self._records: dict[str, list[_FrameRecord]] = {}
        # object_id -> label (for output filename)
        self._labels: dict[str, str] = {}

    def collect_frame(self, detections: FrameDetections, observed_objects: list[ObjectNode]) -> None:
        """Call once per frame after perception. Records all observed objects.

        Movable filtering happens at export time so we don't miss objects whose
        movability is determined asynchronously by the VLM physics agent.
        """
        if not detections.image_b64:
            return

        object_to_instance = match_instances_to_objects(observed_objects, detections.instances)

        for obj in observed_objects:
            inst = object_to_instance.get(obj.object_id)
            if inst is None:
                continue
            if obj.object_id not in self._records:
                self._records[obj.object_id] = []
                self._labels[obj.object_id] = obj.label
            self._records[obj.object_id].append(
                _FrameRecord(
                    frame_idx=detections.frame_idx,
                    image_b64=detections.image_b64,
                    bbox=list(inst.bbox),
                    mask_rle=inst.mask_rle,
                )
            )

    def export(
        self,
        movable_registry: dict[str, bool],
        output_dir: str | Path,
        fps: float = 30.0,
    ) -> dict[str, str]:
        """Write one video per movable object.

        Args:
            movable_registry: object_id -> is_movable (True/False/absent).
                              Only objects explicitly marked True are exported.
            output_dir: Directory to write ``{object_id}_{label}.mp4`` files.
            fps: Output video frame rate.

        Returns:
            Mapping of object_id -> output video path for successfully written files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        movable_ids = {oid for oid, is_mov in movable_registry.items() if is_mov}
        to_export = {
            oid: recs
            for oid, recs in self._records.items()
            if oid in movable_ids and recs
        }

        if not to_export:
            logger.info("[ObjectVideoExtractor] No movable objects with frames to export.")
            return {}

        logger.info(
            "[ObjectVideoExtractor] Exporting %d movable objects to %s",
            len(to_export),
            output_dir,
        )

        results: dict[str, str] = {}
        for object_id, records in to_export.items():
            label = self._labels.get(object_id, "unknown")
            out_path = output_dir / f"{object_id}_{label}.mp4"
            ok = _write_object_video(records, str(out_path), fps)
            if ok:
                results[object_id] = str(out_path)
                logger.info(
                    "[ObjectVideoExtractor] %s (%s): %d frames → %s",
                    object_id, label, len(records), out_path,
                )
            else:
                logger.warning(
                    "[ObjectVideoExtractor] Failed to write video for %s (%s)",
                    object_id, label,
                )

        return results


# ── internal helpers ──────────────────────────────────────────────────────────

def _write_object_video(
    records: list[_FrameRecord],
    out_path: str,
    fps: float,
) -> bool:
    """Render masked/cropped frames into a video file. Returns True on success."""
    writer: cv2.VideoWriter | None = None

    for rec in records:
        frame = _decode_b64_image(rec.image_b64)
        if frame is None:
            continue

        cropped = _crop_and_mask(frame, rec.bbox, rec.mask_rle)
        if cropped is None or cropped.size == 0:
            continue

        h, w = cropped.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(cropped)

    if writer is not None:
        writer.release()
        return True
    return False


def _decode_b64_image(image_b64: str) -> np.ndarray | None:
    try:
        data = base64.b64decode(image_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _crop_and_mask(
    frame: np.ndarray,
    bbox: list[float],
    mask_rle: str | None,
) -> np.ndarray | None:
    """Crop frame to bbox; apply pixel-level mask if mask_rle is available."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    if mask_rle is not None:
        mask = _decode_rle(mask_rle, h, w)
        if mask is not None:
            masked = frame.copy()
            masked[mask == 0] = 0  # black out background pixels
            return masked[y1:y2, x1:x2]

    return frame[y1:y2, x1:x2].copy()


def _decode_rle(rle_json: str, h: int, w: int) -> np.ndarray | None:
    """Decode a COCO RLE JSON string to a uint8 binary mask of shape (H, W)."""
    try:
        from pycocotools import mask as mask_util  # type: ignore[import]
        rle = json.loads(rle_json)
        return mask_util.decode(rle).astype(np.uint8)
    except Exception:
        return None
