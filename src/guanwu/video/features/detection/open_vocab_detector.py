from __future__ import annotations

from collections import deque

from guanwu.video.core.types import FrameDetections
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)


class OpenVocabDetector:
    """Lightweight per-frame heuristic that decides when to trigger VLM discovery.

    Two heuristics:
    1. Recent frames had objects but the current frame has zero → something may have
       entered / left the scene that the tracker lost.
    2. Tracked bbox coverage is extremely low while the scene previously had more
       objects → the detector may be missing a new category.

    A cooldown prevents consecutive triggers.
    """

    def __init__(self, cooldown_frames: int = 10) -> None:
        self.cooldown_frames = cooldown_frames
        self._last_trigger_frame: int = -cooldown_frames  # allow first trigger immediately
        self._recent_counts: deque[int] = deque(maxlen=5)

    def should_trigger_vlm(self, frame_idx: int, detections: FrameDetections) -> bool:
        """Return True if a VLM discovery call should be made for this frame."""
        current_count = len(detections.instances)
        self._recent_counts.append(current_count)

        if (frame_idx - self._last_trigger_frame) < self.cooldown_frames:
            return False

        # Heuristic 1: objects were present recently but current frame has none
        if current_count == 0 and len(self._recent_counts) >= 3:
            recent_avg = sum(list(self._recent_counts)[:-1]) / max(1, len(self._recent_counts) - 1)
            if recent_avg >= 1.0:
                logger.info(
                    f"[OpenVocabDetector] Trigger at frame {frame_idx}: "
                    f"current=0, recent_avg={recent_avg:.1f}"
                )
                self._last_trigger_frame = frame_idx
                return True

        # Heuristic 2: very low bbox coverage with few instances
        if 0 < current_count <= 1 and len(self._recent_counts) >= 3:
            recent_max = max(list(self._recent_counts)[:-1])
            if recent_max >= 3:
                logger.info(
                    f"[OpenVocabDetector] Trigger at frame {frame_idx}: "
                    f"current={current_count}, recent_max={recent_max}"
                )
                self._last_trigger_frame = frame_idx
                return True

        return False
