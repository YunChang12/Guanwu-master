from __future__ import annotations

import json
import math
from typing import Any

from guanwu.video.core.schema import ObjectNode
from guanwu.video.core.types import DetectedInstance


def bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def label_matches(a: str, b: str) -> bool:
    """Check if two labels refer to the same object category."""
    left = a.strip().lower()
    right = b.strip().lower()
    if left == right:
        return True
    return left in right or right in left


def deduplicate_instances(
    instances: list[DetectedInstance],
    *,
    iou_threshold: float = 0.5,
    cross_label_iou_threshold: float = 0.85,
) -> list[DetectedInstance]:
    """Suppress duplicate detections while preserving original survivor order.

    Selection is score-first: the highest-confidence instance in an overlapping
    cluster is kept, but the returned list preserves the source order of the
    survivors so upstream track ordering remains stable.
    """
    ranked = sorted(
        enumerate(instances),
        key=lambda pair: (-float(pair[1].score or 0.0), pair[0]),
    )
    kept_ranked: list[tuple[int, DetectedInstance, str]] = []
    kept_indices: set[int] = set()

    for index, item in ranked:
        label = item.concept_label.lower()
        mask_key = _mask_signature(item.mask_rle)
        suppressed = False
        for _, kept, kept_mask_key in kept_ranked:
            if mask_key and kept_mask_key and mask_key == kept_mask_key:
                suppressed = True
                break
            if not (_bbox_is_valid(item.bbox) and _bbox_is_valid(kept.bbox)):
                continue
            iou = bbox_iou(item.bbox, kept.bbox)
            same_label = label_matches(label, kept.concept_label.lower())
            if same_label and iou > iou_threshold:
                suppressed = True
                break
            if not same_label and iou > cross_label_iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept_ranked.append((index, item, mask_key))
            kept_indices.add(index)

    return [item for index, item in enumerate(instances) if index in kept_indices]


def match_instances_to_objects(
    objects: list[ObjectNode],
    instances: list[DetectedInstance],
) -> dict[str, DetectedInstance]:
    """Match current-frame detections to objects using label and bbox overlap."""
    matches: dict[str, DetectedInstance] = {}
    used_indices: set[int] = set()

    for obj in objects:
        best_idx = -1
        best_score = -1.0
        obj_bbox = obj.geometry.bbox_2d
        obj_label = obj.label.strip().lower()

        for idx, inst in enumerate(instances):
            if idx in used_indices:
                continue
            iou = bbox_iou(obj_bbox, inst.bbox)
            label_bonus = 1.0 if inst.concept_label.strip().lower() == obj_label else 0.0
            score = (2.0 * iou) + label_bonus + (0.01 * float(inst.score))
            if iou <= 0.0 and label_bonus <= 0.0:
                continue
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0:
            used_indices.add(best_idx)
            matches[obj.object_id] = instances[best_idx]

    return matches


def _bbox_is_valid(bbox: list[float], min_size: float = 2.0) -> bool:
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return math.isfinite(width) and math.isfinite(height) and width >= min_size and height >= min_size


def _mask_signature(mask_rle: str | dict[str, Any] | None) -> str:
    if not mask_rle:
        return ""
    if isinstance(mask_rle, dict):
        return json.dumps(mask_rle, sort_keys=True, separators=(",", ":"))
    if isinstance(mask_rle, str):
        raw = mask_rle.strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return str(mask_rle)
