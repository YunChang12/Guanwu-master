from __future__ import annotations

import base64
import io

from guanwu.video.core.types import FrameDetections
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)


class KeyframeDetector:
    """Determines when to trigger VLM re-discovery based on scene signals.

    A frame is a keyframe if any of these conditions are met:
    - Periodic interval has elapsed
    - Number of new object_ids exceeds threshold (new objects entered scene)
    - Number of disappeared objects exceeds threshold (scene changed significantly)
    - Average detection confidence drops below threshold (detector may be missing things)
    - Image content changed significantly compared to the previous keyframe
    """

    def __init__(
        self,
        periodic_interval: int = 30,
        new_track_threshold: int = 1,
        disappear_threshold: int = 1,
        confidence_drop_threshold: float = 0.3,
        image_change_threshold: float = 0.08,
        image_change_enabled: bool = True,
    ) -> None:
        self.periodic_interval = periodic_interval
        self.new_track_threshold = new_track_threshold
        self.disappear_threshold = disappear_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.image_change_threshold = image_change_threshold
        self.image_change_enabled = image_change_enabled
        self._known_object_ids: set[str] = set()
        self._last_keyframe: int = 0
        self._prev_thumbnail: list[int] | None = None

    def check(
        self,
        frame_idx: int,
        detections: FrameDetections,
        observed_objects: list,
        removed_ids: list[str],
    ) -> bool:
        """Return True if this frame is a keyframe warranting VLM re-discovery."""
        reasons: list[str] = []

        # Periodic trigger
        if self.periodic_interval > 0 and (frame_idx - self._last_keyframe) >= self.periodic_interval:
            reasons.append("periodic")

        # New object IDs trigger
        current_object_ids = {inst.object_id for inst in detections.instances}
        new_objects = current_object_ids - self._known_object_ids
        if len(new_objects) >= self.new_track_threshold:
            reasons.append(f"new_objects({len(new_objects)})")
        self._known_object_ids.update(current_object_ids)

        # Disappearance trigger
        if len(removed_ids) >= self.disappear_threshold:
            reasons.append(f"disappearances({len(removed_ids)})")

        # Low confidence trigger
        if detections.instances:
            avg_conf = sum(inst.score for inst in detections.instances) / len(detections.instances)
            if avg_conf < self.confidence_drop_threshold:
                reasons.append(f"low_confidence({avg_conf:.2f})")

        # Image change trigger (only if at least half the periodic interval has
        # elapsed since the last keyframe, to avoid rapid-fire VLM calls)
        min_gap = max(3, self.periodic_interval // 2)
        if (
            self.image_change_enabled
            and detections.image_b64
            and (frame_idx - self._last_keyframe) >= min_gap
        ):
            score = self._image_change_score(detections.image_b64)
            if score is not None and score >= self.image_change_threshold:
                reasons.append(f"image_change({score:.3f})")

        if reasons:
            self._last_keyframe = frame_idx
            logger.info(f"[KeyframeDetector] Keyframe at frame {frame_idx} (reason: {', '.join(reasons)})")
            return True

        return False

    def _image_change_score(self, image_b64: str) -> float | None:
        """Compute mean absolute difference between current and previous thumbnail.

        Returns None if this is the first frame or PIL is unavailable.
        """
        try:
            from PIL import Image
        except ImportError:
            return None

        raw = image_b64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]

        try:
            img_data = base64.b64decode(raw)
            img = Image.open(io.BytesIO(img_data)).convert("L").resize((32, 32))
        except Exception:
            return None

        # get_flattened_data is the Pillow 14+ replacement for getdata
        if hasattr(img, "get_flattened_data"):
            pixels = list(img.get_flattened_data())
        else:
            pixels = list(img.getdata())

        if self._prev_thumbnail is None:
            self._prev_thumbnail = pixels
            return None

        total = sum(abs(a - b) for a, b in zip(pixels, self._prev_thumbnail))
        mad = total / (len(pixels) * 255.0)
        self._prev_thumbnail = pixels
        return mad
