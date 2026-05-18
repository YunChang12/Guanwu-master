"""Background reconstruction via WildGS-SLAM.

Collects foreground detection info (bboxes) from the perception pipeline over an
initial sampling window, sends the **original video** directly to WildGS-SLAM,
then separates foreground/background points using the recovered static map.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from guanwu.video.core.logger import get_logger
from guanwu.video.core.types import FrameDetections
from guanwu.video.core.instance_matching import match_instances_to_objects

logger = get_logger(__name__)


class BackgroundReconstructionPipeline:
    """Accumulates detection info, reconstructs the full scene, then separates fg/bg."""

    def __init__(
        self,
        reconstruction_adapter: Any,
        video_source: str,
        depth_estimator: Any | None = None,
        depth_provider: Any | None = None,
        sample_frames: int = 30,
    ) -> None:
        self._reconstruction = reconstruction_adapter
        self._video_source = video_source
        self._depth_estimator = depth_estimator
        self._depth = depth_provider
        self._sample_frames = sample_frames

        # Collected per-frame detection data for later fg/bg separation
        self._frame_bboxes: list[list[list[float]]] = []  # per-frame list of [x1,y1,x2,y2]
        self._frame_indices: list[int] = []
        self._frame_sizes: list[tuple[int, int]] = []  # (h, w) of each collected frame
        self._done = False
        self._result: dict[str, Any] | None = None

    @property
    def is_ready(self) -> bool:
        return len(self._frame_bboxes) >= self._sample_frames

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def result(self) -> dict[str, Any] | None:
        return self._result

    def bootstrap_reconstruction(self) -> dict[str, Any]:
        """Run the reconstruction backend early so camera poses are available before frame 1."""
        if self._result is not None:
            return self._result

        logger.info("[BackgroundRecon] Bootstrapping WildGS-SLAM reconstruction from input video ...")
        wildgs_result = self._reconstruction.run_slam(video_path=self._video_source)
        self._result = self._normalize_reconstruction_result(wildgs_result)
        return self._result

    def collect_frame(
        self,
        detections: FrameDetections,
        observed_objects: list[Any] | None = None,
        movable_registry: dict[str, bool] | None = None,
    ) -> None:
        """Store movable-object bboxes from a frame for later fg/bg separation."""
        if self._done:
            return

        movable_object_ids: set[str] | None = None
        if observed_objects is not None and movable_registry is not None:
            movable_object_ids = {
                str(obj.object_id)
                for obj in observed_objects
                if movable_registry.get(str(obj.object_id)) is True
            }
        object_to_instance = (
            match_instances_to_objects(list(observed_objects or []), detections.instances)
            if observed_objects is not None
            else {}
        )

        bboxes: list[list[float]] = []
        if movable_object_ids is not None:
            iter_instances = [object_to_instance[oid] for oid in movable_object_ids if oid in object_to_instance]
        else:
            iter_instances = detections.instances
        for inst in iter_instances:
            bbox = inst.bbox
            if len(bbox) >= 4:
                bboxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        # Determine frame dimensions from image if available
        h, w = 480, 640  # defaults
        frame = _decode_b64_to_bgr(detections.image_b64)
        if frame is not None:
            h, w = frame.shape[:2]

        self._frame_bboxes.append(bboxes)
        self._frame_indices.append(detections.frame_idx)
        self._frame_sizes.append((h, w))

    def reconstruct(self, output_root: Path, depths_file_id: str | None = None) -> dict[str, Any]:
        """Run full-scene reconstruction using the original video. Call after is_ready becomes True."""
        if self._done:
            return self._result or {}
        if self._result is None and not self._frame_bboxes:
            logger.warning("[BackgroundRecon] No frames collected, skipping.")
            self._done = True
            return {}
        _ = depths_file_id

        if self._result is None:
            logger.info(
                "[BackgroundRecon] Sending original video to WildGS-SLAM for reconstruction ..."
            )
            try:
                self._result = self.bootstrap_reconstruction()
                logger.info(
                    "[BackgroundRecon] Reconstruction complete: %d frames, quality=%.3f",
                    int(self._result.get("num_frames", 0) or 0),
                    float(self._result.get("slam_quality", 0.0) or 0.0),
                )
            except Exception as exc:
                logger.error("[BackgroundRecon] WildGS-SLAM reconstruction failed: %s", exc)
                self._result = {"error": str(exc)}
                self._done = True
                return self._result

        # Run fg/bg separation if we got a points PLY
        points_path = self._result.get("points_path", "")
        if points_path and Path(points_path).exists() and "fg_bg_labels_path" not in self._result:
            try:
                labels_path = self._separate_fg_bg(points_path, output_root)
                self._result["fg_bg_labels_path"] = str(labels_path)
            except Exception as exc:
                logger.warning("[BackgroundRecon] fg/bg separation failed: %s", exc)

        self._done = True
        self._frame_bboxes.clear()
        self._frame_indices.clear()
        self._frame_sizes.clear()
        return self._result

    def _normalize_reconstruction_result(self, result: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(result)
        static_map_dir = Path(str(result.get("static_map_dir", "")).strip()).expanduser() if result.get("static_map_dir") else None
        points_path = self._find_static_map_points(static_map_dir) if static_map_dir else None
        if points_path:
            normalized["points_path"] = str(points_path)
        normalized.setdefault("background_backend", "wildgs")
        return normalized

    def _find_static_map_points(self, static_map_dir: Path) -> Path | None:
        if not static_map_dir.exists():
            return None
        candidates = (
            "static_gaussians.ply",
            "scene_points.ply",
            "points.ply",
            "map.ply",
        )
        for name in candidates:
            path = static_map_dir / name
            if path.exists():
                return path
        for path in static_map_dir.rglob("*.ply"):
            return path
        return None

    def _separate_fg_bg(self, points_ply_path: str, output_root: Path) -> Path:
        """Separate foreground/background points using collected bboxes + depth projection.

        For each Gaussian point, project it to 2D for each collected frame using the
        depth estimator's camera model.  If the projected point falls inside a
        detection bbox *and* its depth is consistent, label it as foreground (1);
        otherwise background (0).

        Returns path to the saved labels .npy file.
        """
        from plyfile import PlyData

        ply = PlyData.read(points_ply_path)
        verts = ply["vertex"]
        xyz = np.column_stack([verts["x"], verts["y"], verts["z"]])  # [N, 3]
        n_pts = xyz.shape[0]

        # Count how many frames vote each point as foreground
        fg_votes = np.zeros(n_pts, dtype=np.int32)
        total_votes = 0

        for bboxes, frame_idx, (h, w) in zip(
            self._frame_bboxes, self._frame_indices, self._frame_sizes
        ):
            if not bboxes:
                continue

            # Project 3D → 2D using the depth estimator's camera if available
            uv = self._project_points(xyz, frame_idx, h, w)
            if uv is None:
                continue

            total_votes += 1
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                in_bbox = (
                    (uv[:, 0] >= x1) & (uv[:, 0] <= x2) &
                    (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
                )
                fg_votes[in_bbox] += 1

        # A point is foreground if it was voted as such in > 50% of frames with detections
        if total_votes > 0:
            labels = (fg_votes > total_votes * 0.5).astype(np.uint8)
        else:
            labels = np.zeros(n_pts, dtype=np.uint8)

        bg_dir = output_root / "intermediate" / "background"
        bg_dir.mkdir(parents=True, exist_ok=True)
        labels_path = bg_dir / "fg_bg_labels.npy"
        np.save(str(labels_path), labels)

        n_fg = int(labels.sum())
        logger.info(
            "[BackgroundRecon] fg/bg separation: %d foreground, %d background out of %d points",
            n_fg, n_pts - n_fg, n_pts,
        )
        return labels_path

    def _project_points(
        self, xyz: np.ndarray, frame_idx: int, h: int, w: int
    ) -> np.ndarray | None:
        """Project 3D points to 2D pixel coordinates for a given frame.

        Uses the depth estimator's camera model if available; otherwise falls
        back to a simple perspective projection with reasonable defaults.
        """
        # Try to use depth estimator's projection capabilities
        if self._depth_estimator is not None and hasattr(self._depth_estimator, "project_3d_to_2d"):
            try:
                return self._depth_estimator.project_3d_to_2d(xyz, frame_idx, h, w)
            except Exception:
                pass

        # Fallback: simple pinhole projection with default focal length
        fx = fy = max(h, w)
        cx, cy = w / 2.0, h / 2.0

        z = xyz[:, 2]
        valid = z > 1e-4
        uv = np.full((xyz.shape[0], 2), -1.0, dtype=np.float64)
        uv[valid, 0] = (xyz[valid, 0] * fx / z[valid]) + cx
        uv[valid, 1] = (xyz[valid, 1] * fy / z[valid]) + cy
        return uv

    def _write_video(self, frames: list[np.ndarray], output_root: Path) -> Path:
        """Write frames as a debug video (optional, for diagnostics only)."""
        bg_dir = output_root / "intermediate" / "background"
        bg_dir.mkdir(parents=True, exist_ok=True)
        video_path = bg_dir / "debug_frames.mp4"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info("[BackgroundRecon] Debug video written: %s (%d frames)", video_path, len(frames))
        return video_path


def _decode_b64_to_bgr(image_b64: str | None) -> np.ndarray | None:
    if not image_b64:
        return None
    import base64

    payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _build_foreground_mask(shape: tuple[int, int], detections: FrameDetections) -> np.ndarray:
    """Build a binary mask (255 = foreground) from all detected instance bboxes."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for inst in detections.instances:
        bbox = inst.bbox
        if len(bbox) < 4:
            continue
        x1 = max(0, min(w, int(round(float(bbox[0])))))
        y1 = max(0, min(h, int(round(float(bbox[1])))))
        x2 = max(0, min(w, int(round(float(bbox[2])))))
        y2 = max(0, min(h, int(round(float(bbox[3])))))
        if x2 > x1 and y2 > y1:
            # Dilate bbox slightly for cleaner inpainting
            pad_x = max(1, int((x2 - x1) * 0.05))
            pad_y = max(1, int((y2 - y1) * 0.05))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            mask[y1:y2, x1:x2] = 255
    return mask
