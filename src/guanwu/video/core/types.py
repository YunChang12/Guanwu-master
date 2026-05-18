from typing import Any

from pydantic import BaseModel


class DetectedInstance(BaseModel):
    mask_ref: str
    bbox: list[float]
    object_id: str
    concept_label: str
    segment_kind: str = "object"  # object | body
    score: float
    mask_rle: str | dict[str, Any] | None = None  # COCO RLE encoded segmentation mask


class FrameDetections(BaseModel):
    frame_idx: int
    timestamp: float
    instances: list[DetectedInstance]
    image_b64: str | None = None
