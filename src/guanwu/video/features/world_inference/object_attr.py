from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import cv2
import numpy as np

from guanwu.video.core.config import VLMConfig
from guanwu.video.core.json_repair import parse_vlm_json
from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.core.logger import get_logger
from guanwu.video.core.instance_matching import match_instances_to_objects

logger = get_logger(__name__)

# Distinct colors (BGR) for up to 20 objects; cycles if more
_MASK_COLORS = [
    (255, 82, 82), (82, 255, 82), (82, 82, 255), (255, 255, 82),
    (255, 82, 255), (82, 255, 255), (255, 165, 0), (128, 0, 255),
    (0, 200, 100), (200, 50, 50), (50, 200, 200), (200, 200, 50),
    (150, 75, 0), (0, 128, 200), (200, 0, 128), (75, 150, 0),
    (0, 75, 150), (150, 0, 75), (100, 100, 200), (200, 100, 100),
]


def _decode_rle(rle_json: str, h: int, w: int) -> np.ndarray | None:
    """Decode COCO RLE JSON string to binary uint8 mask (H, W)."""
    try:
        from pycocotools import mask as mask_util  # type: ignore[import]
        rle = json.loads(rle_json)
        return mask_util.decode(rle).astype(np.uint8)
    except Exception:
        return None


def _render_annotated_image(
    image_b64: str,
    instances: list[tuple[str, DetectedInstance]],
) -> str:
    """Overlay pixel-level masks, object_id and label onto the image.

    Only the provided instances are annotated.
    Returns a base64-encoded JPEG string.
    """
    raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return image_b64

    h, w = frame.shape[:2]
    overlay = frame.copy()

    color_idx = 0
    for object_id, inst in instances:
        color = _MASK_COLORS[color_idx % len(_MASK_COLORS)]
        color_idx += 1

        # Draw pixel-level mask
        if inst.mask_rle:
            mask = _decode_rle(inst.mask_rle, h, w)
            if mask is not None:
                overlay[mask == 1] = (
                    overlay[mask == 1] * 0.45 + np.array(color) * 0.55
                ).astype(np.uint8)

        # Draw bbox
        x1, y1, x2, y2 = (int(round(v)) for v in inst.bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label: "object_id | label"
        text = f"{object_id} | {inst.concept_label}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + baseline)
        cv2.rectangle(overlay, (x1, ty - th - baseline), (x1 + tw + 2, ty + baseline), color, -1)
        cv2.putText(overlay, text, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    _, enc = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(enc.tobytes()).decode("ascii")

_PHYSICS_SYS = (
    "You are given an annotated image of a physical scene. "
    "Each detected object is highlighted with a colored mask and bounding box, "
    "and labeled with its object_id and class label in the format 'object_id | label' (pipe as separator). "
    "Use the visual appearance of each highlighted region (color, texture, transparency, "
    "surface finish, shape, deformation) together with the class label "
    "to describe semantic and material attributes. "
    "Return a JSON object keyed by object_id. Each value must contain: "
    '"is_movable" (bool or null — false for fixed infrastructure: road, building, wall, ground, '
    "floor, terrain, pavement, sidewalk, curb, fence, barrier, bridge, tunnel, grass, sky; "
    'true for objects a person could pick up or push; null if uncertain), '
    '"is_rigid_body" (bool or null), "class_name" (string), '
    '"material_candidates" (list of {"name": str, "prob": float}), '
    '"confidence" (float 0-1), "rationale" (string). '
    "Do not estimate mass, friction, restitution, dimensions, or any metric quantity. "
    "Only return JSON, no extra text."
)

# Fields that every prior entry should contain. Unknown values stay null/empty.
_REQUIRED_FIELDS: dict[str, Any] = {
    "is_movable": None,
    "is_rigid_body": None,
    "class_name": "unknown",
    "material_candidates": [],
    "confidence": None,
    "rationale": "",
}

_CORRECTION_SYS = (
    "You are an object attribute correction assistant. "
    "You will be given a partial JSON result from a previous inference step and a list of "
    "missing fields per object. Fill in ONLY the missing fields. "
    "Do not fill in metric or physical numeric quantities such as mass, friction, restitution, or dimensions. "
    "Return a JSON object keyed by object_id containing only the missing fields. "
    "No extra text."
)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _group_non_overlapping(instances: list[DetectedInstance]) -> list[list[DetectedInstance]]:
    """Greedy graph coloring: group instances so no two in the same group overlap.

    Two instances are considered overlapping when their bbox IoU > 0.
    Returns the minimum number of groups (colors) via greedy coloring.
    """
    n = len(instances)
    if n == 0:
        return []

    # Build adjacency: overlap[i][j] = True if bboxes overlap
    overlap = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_iou(instances[i].bbox, instances[j].bbox) > 0:
                overlap[i][j] = overlap[j][i] = True

    # Greedy coloring (largest-first ordering)
    order = sorted(range(n), key=lambda i: sum(overlap[i]), reverse=True)
    colors = [-1] * n
    for i in order:
        used = {colors[j] for j in range(n) if overlap[i][j] and colors[j] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[i] = c

    num_groups = max(colors) + 1
    groups: list[list[DetectedInstance]] = [[] for _ in range(num_groups)]
    for i, c in enumerate(colors):
        groups[c].append(instances[i])
    return groups


class ObjectAttrAgent:
    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = config.base_url
        self.model = config.model
        self.max_retries = config.max_retries
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except ImportError:
                logger.warning("VLM Physics Agent disabled: openai package is not installed.")

    def infer_object_physics_priors(
        self,
        detections: FrameDetections,
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        if not self.client or not objects:
            return {}

        matched_instances = match_instances_to_objects(objects, detections.instances)
        relevant = [(obj.object_id, matched_instances[obj.object_id]) for obj in objects if obj.object_id in matched_instances]

        # Group instances so each group is non-overlapping
        groups = _group_non_overlapping([inst for _, inst in relevant])
        logger.info(
            f"[ObjectAttrAgent] {len(objects)} objects split into {len(groups)} non-overlapping group(s)."
        )

        image_b64 = getattr(detections, "image_b64", None) or None
        obj_desc_by_id = {
            obj.object_id: {"object_id": obj.object_id, "label": str(obj.label or "unknown")}
            for obj in objects
        }

        result: dict[str, dict] = {}
        for g_idx, group in enumerate(groups):
            group_pairs = [(oid, inst) for oid, inst in relevant if inst in group]
            group_ids = {oid for oid, _ in group_pairs}
            group_descs = [obj_desc_by_id[oid] for oid in group_ids if oid in obj_desc_by_id]
            if not group_descs:
                continue

            image_part: dict[str, Any] | None = None
            if image_b64:
                annotated = _render_annotated_image(image_b64, group_pairs)
                image_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{annotated}"}}

            logger.info(f"[ObjectAttrAgent] Group {g_idx + 1}/{len(groups)}: {sorted(group_ids)}")
            group_result = self._infer_batch(group_descs, image_part)
            result.update(group_result)

        all_descs = list(obj_desc_by_id.values())

        # Validate and correct missing fields
        missing_by_obj = self._find_missing_fields(result, all_descs)
        if missing_by_obj:
            logger.warning(
                f"[ObjectAttrAgent] {len(missing_by_obj)} objects have missing fields: "
                + ", ".join(f"{oid}={fields}" for oid, fields in missing_by_obj.items())
            )
            corrections = self._correct_missing(result, missing_by_obj, all_descs, None)
            result = self._merge_corrections(result, corrections, missing_by_obj)

        result = self._apply_defaults(result, all_descs)

        logger.info(f"[ObjectAttrAgent] Inferred priors for {len(result)} objects across {len(groups)} group(s).")
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _find_missing_fields(self, result: dict, all_descs: list[dict]) -> dict[str, list[str]]:
        """Return {object_id: [missing_field, ...]} for objects missing required fields."""
        missing: dict[str, list[str]] = {}
        for desc in all_descs:
            oid = desc["object_id"]
            entry = result.get(oid, {})
            absent = [f for f in _REQUIRED_FIELDS if f not in entry]
            if absent:
                missing[oid] = absent
        return missing

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def _correct_missing(
        self,
        partial_result: dict,
        missing_by_obj: dict[str, list[str]],
        all_descs: list[dict],
        image_part: dict | None,
    ) -> dict:
        """Ask VLM to fill in only the missing fields."""
        descs_by_id = {d["object_id"]: d for d in all_descs}
        correction_input = {
            oid: {
                "existing": partial_result.get(oid, {}),
                "missing_fields": fields,
                "object_info": descs_by_id.get(oid, {}),
            }
            for oid, fields in missing_by_obj.items()
        }

        user_content: list[dict[str, Any]] = []
        if image_part:
            user_content.append(image_part)
        user_content.append({
            "type": "text",
            "text": (
                "The following objects are missing required fields in the object attributes. "
                "For each object, provide ONLY the missing fields listed.\n"
                f"{json.dumps(correction_input)}"
            ),
        })

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _CORRECTION_SYS},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                )
                raw = completion.choices[0].message.content or "{}"
                data = parse_vlm_json(raw, fallback="{}")
                if isinstance(data, dict) and data:
                    logger.info(f"[ObjectAttrAgent] Correction succeeded for {len(data)} objects.")
                    return data
            except Exception as e:
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        f"[ObjectAttrAgent] Correction attempt {attempt + 1}/{self.max_retries} "
                        f"failed: {e}. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                else:
                    logger.error(f"[ObjectAttrAgent] Correction inference failed: {e}")

        return {}

    def _merge_corrections(
        self,
        result: dict,
        corrections: dict,
        missing_by_obj: dict[str, list[str]],
    ) -> dict:
        """Merge correction fields into the original result."""
        merged = {oid: dict(entry) for oid, entry in result.items()}
        for oid, fields in missing_by_obj.items():
            correction_entry = corrections.get(oid, {})
            if oid not in merged:
                merged[oid] = {}
            for field in fields:
                if field in correction_entry:
                    merged[oid][field] = correction_entry[field]
        return merged

    # ------------------------------------------------------------------
    # Unknown defaults (last resort after correction)
    # ------------------------------------------------------------------

    def _apply_defaults(self, result: dict, all_descs: list[dict]) -> dict:
        """Fill any still-missing fields with null/empty values and log warnings."""
        for desc in all_descs:
            oid = desc["object_id"]
            if oid not in result:
                result[oid] = {}
            for field, default in _REQUIRED_FIELDS.items():
                if field not in result[oid]:
                    logger.warning(
                        f"[ObjectAttrAgent] {oid}: field '{field}' still missing after "
                        f"correction; applying default={default!r}"
                    )
                    result[oid][field] = default
        return result

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------
    def _infer_batch(self, all_descs: list[dict], image_part: dict | None) -> dict[str, dict]:
        """Infer physics priors for all objects in a single VLM call."""
        user_content: list[dict[str, Any]] = []
        if image_part:
            user_content.append(image_part)
        user_content.append({
            "type": "text",
            "text": (
                "The image above shows the scene with each object annotated by a colored mask and bounding box. "
                "Each label in the image reads 'object_id | class_label' (pipe as separator between object_id and label). "
                "Infer non-metric object attributes for the following objects based on their highlighted regions:\n"
                f"{json.dumps(all_descs)}"
            ),
        })

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _PHYSICS_SYS},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                )
                raw = completion.choices[0].message.content or "{}"
                data = parse_vlm_json(raw, fallback="{}")
                if isinstance(data, dict) and data:
                    return data
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        f"[ObjectAttrAgent] Batch attempt {attempt + 1}/{self.max_retries} "
                        f"failed: {e}. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)

        logger.error(f"[ObjectAttrAgent] Batch inference failed: {last_exc}")
        return {}
