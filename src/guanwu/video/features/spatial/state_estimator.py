from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guanwu.video.core.instance_matching import bbox_iou, deduplicate_instances, label_matches
from guanwu.video.core.schema import AffordanceState, Geometry, ObjectNode, PhysicsState, Pose3D, Provenance, SemanticState
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.core.logger import get_logger
from guanwu.video.features.spatial.alignment_utils import resolve_depth_map_path

logger = get_logger(__name__)


@dataclass
class CameraPose:
    frame_id: int
    timestamp_sec: float
    K: list[list[float]]
    R: list[list[float]]
    t: list[float]
    pose_quality: float


class ColmapPoseProvider:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)
        self._K = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        self._poses_by_name: dict[str, tuple[list[list[float]], list[float]]] = {}
        self._frame_keys: list[str] = []
        self._load()

    def _load(self) -> None:
        cam_file = self.model_dir / "cameras.txt"
        img_file = self.model_dir / "images.txt"
        if not cam_file.exists() or not img_file.exists():
            raise RuntimeError(f"COLMAP text model not found in {self.model_dir}")

        self._K = self._load_intrinsics(cam_file)
        self._poses_by_name = self._load_poses(img_file)
        self._frame_keys = sorted(self._poses_by_name.keys())
        if not self._frame_keys:
            raise RuntimeError("No camera poses loaded from COLMAP images.txt")

    def pose_for_frame(self, frame_id: int, timestamp_sec: float) -> CameraPose:
        name = f"{frame_id:06d}.jpg"
        key = name if name in self._poses_by_name else self._nearest_key(frame_id)
        R, t = self._poses_by_name[key]
        return CameraPose(frame_id=frame_id, timestamp_sec=timestamp_sec, K=self._K, R=R, t=t, pose_quality=0.95)

    def _nearest_key(self, frame_id: int) -> str:
        best = self._frame_keys[0]
        best_dist = 10**9
        for k in self._frame_keys:
            stem = Path(k).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            if not digits:
                continue
            d = abs(int(digits) - frame_id)
            if d < best_dist:
                best = k
                best_dist = d
        return best

    def _load_intrinsics(self, cam_file: Path) -> list[list[float]]:
        with cam_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                model = parts[1]
                width = float(parts[2])
                height = float(parts[3])
                params = [float(p) for p in parts[4:]]

                if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                    fxy, cx, cy = params[0], params[1], params[2]
                    fx, fy = fxy, fxy
                elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                else:
                    fx, fy = max(width, 1.0), max(height, 1.0)
                    cx, cy = width / 2.0, height / 2.0

                return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

        raise RuntimeError("No valid camera intrinsics found in cameras.txt")

    def _load_poses(self, img_file: Path) -> dict[str, tuple[list[list[float]], list[float]]]:
        poses: dict[str, tuple[list[list[float]], list[float]]] = {}
        with img_file.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

        # COLMAP images.txt uses two lines per image entry: pose+name and 2D points.
        for idx in range(0, len(lines), 2):
            row = lines[idx].split()
            if len(row) < 10:
                continue
            qw, qx, qy, qz = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            tx, ty, tz = float(row[5]), float(row[6]), float(row[7])
            name = row[9]

            R = _quat_to_rot(qw, qx, qy, qz)
            # COLMAP stores world->camera; convert to camera center in world coordinates.
            Rc = _transpose(R)
            t_world = _mat_vec_mul(Rc, [-tx, -ty, -tz])
            poses[name] = (Rc, t_world)

        return poses


class WildGSPoseProvider:
    """Load camera poses from a ``camera_poses.jsonl`` file produced by
    the WildGS-SLAM MCP server (or any tool that writes our standard format).

    Each line of the JSONL must contain at minimum::

        {
          "frame": <int>,
          "timestamp": <float>,
          "T_world_from_cam": [[...], [...], [...], [...]],   // 4×4
          "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...}
        }

    Also exposes paths to the static map and dynamic prior produced by
    WildGS-SLAM, so downstream modules (fusion, registration) can locate them.
    """

    def __init__(
        self,
        camera_poses_jsonl: str,
        static_map_dir: str | None = None,
        dynamic_prior_dir: str | None = None,
    ) -> None:
        self._jsonl_path = Path(camera_poses_jsonl).expanduser().resolve()
        self._static_map_dir = static_map_dir
        self._dynamic_prior_dir = dynamic_prior_dir
        self._records: list[dict] = []
        self._by_frame: dict[int, dict] = {}
        self._default_K = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        self._load()

    def _load(self) -> None:
        import json
        if not self._jsonl_path.exists():
            raise RuntimeError(f"camera_poses.jsonl not found: {self._jsonl_path}")
        with self._jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._records.append(record)
                self._by_frame[record["frame"]] = record
        if not self._records:
            raise RuntimeError(f"No poses loaded from {self._jsonl_path}")
        # Update default intrinsics from the first record
        if "intrinsics" in self._records[0]:
            K = self._records[0]["intrinsics"]
            self._default_K = [
                [K.get("fx", 600.0), 0.0, K.get("cx", 320.0)],
                [0.0, K.get("fy", 600.0), K.get("cy", 240.0)],
                [0.0, 0.0, 1.0],
            ]

    def pose_for_frame(self, frame_id: int, timestamp_sec: float) -> CameraPose:
        record = self._by_frame.get(frame_id) or self._nearest_record(timestamp_sec)
        T = record.get("T_world_from_cam", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        # Extract R (3×3) and t (3,) from the 4×4 transform
        R = [row[:3] for row in T[:3]]
        t = [T[0][3], T[1][3], T[2][3]]
        K = self._default_K
        if "intrinsics" in record:
            ki = record["intrinsics"]
            K = [
                [ki.get("fx", 600.0), 0.0, ki.get("cx", 320.0)],
                [0.0, ki.get("fy", 600.0), ki.get("cy", 240.0)],
                [0.0, 0.0, 1.0],
            ]
        return CameraPose(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            K=K,
            R=R,
            t=t,
            pose_quality=float(record.get("pose_quality", 0.9)),
        )

    def _nearest_record(self, timestamp_sec: float) -> dict:
        best = self._records[0]
        best_dt = abs(best.get("timestamp", 0.0) - timestamp_sec)
        for rec in self._records[1:]:
            dt = abs(rec.get("timestamp", 0.0) - timestamp_sec)
            if dt < best_dt:
                best = rec
                best_dt = dt
        return best

    @property
    def static_map_dir(self) -> str | None:
        return self._static_map_dir

    @property
    def dynamic_prior_dir(self) -> str | None:
        return self._dynamic_prior_dir

    def dynamic_prior_for_frame(self, frame_id: int) -> Any | None:
        """Load the per-frame uncertainty map (.npy) for a given frame, if available."""
        import numpy as np
        if not self._dynamic_prior_dir:
            return None
        npy_path = Path(self._dynamic_prior_dir) / f"{frame_id:06d}.npy"
        if npy_path.exists():
            return np.load(str(npy_path))
        return None


class DepthAnythingProvider:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._np = None
        self._Image = None
        self._torch = None
        self._load()

    def _load(self) -> None:
        try:
            import numpy as np
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything V2 backend requires transformers/torch/pillow/numpy. "
                "Install extras: uv pip install -e '.[pit]'"
            ) from exc

        self._np = np
        self._torch = torch
        self._Image = Image
        self._processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_path, local_files_only=True)
        self._model.to(self.device)
        self._model.eval()

    @property
    def is_metric(self) -> bool:
        return False

    def depth_values(
        self,
        image_b64: str,
        samples_uv: list[tuple[float, float]],
        frame_idx: int = 0,
    ) -> list[float | None]:
        _ = (image_b64, samples_uv, frame_idx)
        return [None] * len(samples_uv)

    def _relative_depth_values(self, image_b64: str, samples_uv: list[tuple[float, float]]) -> list[float]:
        import base64
        import io
        img_data = base64.b64decode(image_b64)
        image = self._Image.open(io.BytesIO(img_data)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._torch.no_grad():
            pred = self._model(**inputs).predicted_depth

        depth = pred[0].detach().cpu().numpy()
        h, w = depth.shape
        max_depth = float(depth.max()) if float(depth.max()) > 1e-6 else 1.0

        values: list[float] = []
        for u, v in samples_uv:
            x = int(max(0, min(w - 1, (u / 640.0) * w)))
            y = int(max(0, min(h - 1, (v / 480.0) * h)))
            values.append(float(depth[y, x] / max_depth))
        return values


class WildGSDepthProvider:
    """Reads per-frame metric depth maps produced by WildGS-SLAM.

    WildGS saves ``<run_dir>/mono_priors/depths/XXXXX.npy`` (float32, metres,
    shape [H, W]).  These are Depth Anything V2 metric estimates that have been
    jointly refined with camera poses via SLAM bundle adjustment, so they are
    geometrically consistent across frames and generally more accurate than
    running Depth Anything standalone.

    Args:
        depth_maps_dir: Local directory containing ``XXXXX.npy`` depth files.
    """

    def __init__(self, depth_maps_dir: str) -> None:
        import numpy as np
        self._np = np
        self._dir = Path(depth_maps_dir)
        self._cache: dict[int, Any] = {}

    def _load(self, frame_idx: int) -> Any:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        path = resolve_depth_map_path(self._dir, frame_idx)
        if path is None:
            return None
        depth = self._np.load(str(path))
        self._cache[frame_idx] = depth
        return depth

    def depth_values(
        self,
        image_b64: str,
        samples_uv: list[tuple[float, float]],
        frame_idx: int = 0,
    ) -> list[float]:
        depth = self._load(frame_idx)
        if depth is None:
            raise FileNotFoundError(f"WildGS depth map not found for frame {frame_idx} in {self._dir}")
        h, w = depth.shape
        values: list[float] = []
        for u, v in samples_uv:
            x = int(max(0, min(w - 1, (u / 640.0) * w)))
            y = int(max(0, min(h - 1, (v / 480.0) * h)))
            values.append(max(0.01, float(depth[y, x])))
        return values


@dataclass
class _GraveyardEntry:
    object_id: str
    label: str
    bbox_2d: list[float]
    removed_frame: int


class StateEstimationAgent:
    """
    PIT-based 3D estimator.

    Real providers (cross-platform: Linux/macOS):
    - Camera: COLMAP text model (cameras.txt/images.txt)
    - Depth: Depth Anything V2 via local transformers model, or Depth Anything 3
             via Zaiwu gateway tasks (depth_provider='zaiwu_depth_anything3')

    Provider misconfiguration or missing artifacts are treated as hard errors.
    """

    def __init__(
        self,
        camera_provider: str = "wildgs",
        colmap_model_dir: str | None = None,
        wildgs_camera_poses_jsonl: str | None = None,
        wildgs_static_map_dir: str | None = None,
        wildgs_dynamic_prior_dir: str | None = None,
        wildgs_depth_maps_dir: str | None = None,
        depth_provider: str = "wildgs",
        depth_model_path: str | None = None,
        zaiwu_gateway_url: str | None = None,
        zaiwu_depth_service: str | None = None,
        video_source: str | None = None,
        use_metric_scale: bool = False,
        metric_scale_factor: float = 1.0,
    ) -> None:
        self._last_centroids: dict[str, list[float]] = {}
        self._last_timestamps: dict[str, float] = {}
        self._object_points: dict[str, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=2000))
        self._camera_trajectory: list[dict] = []
        self._object_trajectory: dict[str, list[dict]] = defaultdict(list)
        self._frame_records: list[dict] = []
        self._source_to_object: dict[str, str] = {}
        self._object_to_source: dict[str, str] = {}
        self._graveyard: deque[_GraveyardEntry] = deque(maxlen=50)
        self._graveyard_ttl_frames = 60
        self._graveyard_grace_frames = 3  # don't match entries removed fewer than N frames ago
        self._K_default = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]

        self._camera_mode = camera_provider
        self._depth_mode = depth_provider
        self._camera_error: str | None = None
        self._depth_error: str | None = None
        self._use_metric_scale = use_metric_scale
        self._metric_scale_factor = metric_scale_factor

        # SAM3D body camera: updated externally via update_camera_from_sam3d()
        self._sam3d_camera: CameraPose | None = None
        # Per-object orientation overrides from SAM3D body global_rot
        self._object_orientations: dict[str, list[float]] = {}

        self._colmap: ColmapPoseProvider | None = None
        self._wildgs: WildGSPoseProvider | None = None
        self._depth_provider_impl: Any = None  # duck-typed: has depth_values(image_b64, samples_uv)

        if camera_provider == "colmap":
            if not colmap_model_dir:
                raise RuntimeError("colmap_model_dir is required when camera_provider='colmap'")
            self._colmap = ColmapPoseProvider(colmap_model_dir)

        elif camera_provider == "wildgs":
            if wildgs_camera_poses_jsonl:
                self._wildgs = WildGSPoseProvider(
                    wildgs_camera_poses_jsonl,
                    static_map_dir=wildgs_static_map_dir,
                    dynamic_prior_dir=wildgs_dynamic_prior_dir,
                )
            else:
                self._camera_error = "pending_wildgs_camera_poses"

        if depth_provider == "depth_anything_v2":
            if not depth_model_path:
                self._depth_error = "missing_depth_model_path"
            else:
                self._depth_provider_impl = DepthAnythingProvider(depth_model_path)

        elif depth_provider in {"mcp_unified", "zaiwu_depth_anything3"}:
            from guanwu.video.clients.zaiwu import ZaiwuDepthProvider, ZaiwuGatewayClient

            if not zaiwu_gateway_url:
                self._depth_error = "missing_zaiwu_gateway_url"
            else:
                gateway = ZaiwuGatewayClient(gateway_url=zaiwu_gateway_url)
                self._depth_provider_impl = ZaiwuDepthProvider(
                    gateway,
                    service_id=str(zaiwu_depth_service or "services.depth_anything3"),
                    video_path=video_source,
                )

        elif depth_provider == "wildgs":
            if wildgs_depth_maps_dir:
                self._depth_provider_impl = WildGSDepthProvider(wildgs_depth_maps_dir)
            else:
                self._depth_error = "pending_wildgs_depth_maps"

    def attach_wildgs_results(
        self,
        camera_poses_jsonl: str,
        static_map_dir: str | None = None,
        dynamic_prior_dir: str | None = None,
        depth_maps_dir: str | None = None,
    ) -> bool:
        """Switch camera pose recovery to WildGS outputs.

        If ``depth_maps_dir`` is provided, also switches depth estimation to use
        WildGS per-frame metric depth maps (geometrically consistent, SLAM-refined),
        replacing any previously configured depth provider.

        Returns ``True`` when the active camera provider changed and spatial
        history was reset.
        """
        provider = WildGSPoseProvider(
            camera_poses_jsonl,
            static_map_dir=static_map_dir,
            dynamic_prior_dir=dynamic_prior_dir,
        )

        current_jsonl = ""
        if self._wildgs is not None:
            current_jsonl = str(self._wildgs._jsonl_path)  # type: ignore[attr-defined]
        changed = (
            self._camera_mode != "wildgs"
            or current_jsonl != str(provider._jsonl_path)  # type: ignore[attr-defined]
        )

        self._wildgs = provider
        self._camera_error = None
        self._sam3d_camera = None

        if depth_maps_dir:
            self._depth_provider_impl = WildGSDepthProvider(depth_maps_dir)
            self._depth_mode = "wildgs"
            self._depth_error = None
            logger.info("[StateEstimator] Using WildGS depth maps from %s", depth_maps_dir)

        if changed:
            self._reset_spatial_history()
            self._camera_mode = "wildgs"

        return changed

    def estimate(self, detections: FrameDetections) -> list[ObjectNode]:
        pose = self._recover_camera_pose(detections.frame_idx, detections.timestamp)
        self._frame_records.append(
            {
                "frame_id": detections.frame_idx,
                "timestamp_sec": detections.timestamp,
                "image_b64": detections.image_b64,
                "instance_count": len(detections.instances),
                "source_ids": [x.object_id for x in detections.instances],
                "segment_kinds": {x.object_id: x.segment_kind for x in detections.instances},
            }
        )
        if pose is not None:
            self._camera_trajectory.append(
                {
                    "frame_id": pose.frame_id,
                    "timestamp_sec": pose.timestamp_sec,
                    "K": pose.K,
                    "R": pose.R,
                    "t": pose.t,
                    "pose_quality": pose.pose_quality,
                    "camera_provider": self._camera_mode,
                }
            )

        deduplicated = self._deduplicate_instances(detections.instances)
        metric_geometry_available = pose is not None and self._depth_provider_is_metric()

        objects: list[ObjectNode] = []
        for item in deduplicated:
            if not self._bbox_is_valid(item.bbox):
                continue
            object_id = self._resolve_object_id(item.object_id, instance=item, frame_idx=detections.frame_idx)

            centroid: list[float] | None = None
            bbox_min: list[float] | None = None
            bbox_max: list[float] | None = None
            scale: list[float] | None = None
            velocity: list[float] | None = None
            depth_samples: list[float | None] = []
            depth_consistency: float | None = None
            geom_quality = 0.0
            geometry_status = "unavailable"

            if metric_geometry_available and pose is not None:
                sampled_uv = self._sample_pixels_from_bbox(item, num_samples=49)
                depth_samples = self._estimate_depth_samples(detections.image_b64, item, sampled_uv, detections.frame_idx)
                valid_samples = [
                    ((u, v), float(d))
                    for (u, v), d in zip(sampled_uv, depth_samples)
                    if d is not None and math.isfinite(float(d)) and float(d) > 0.0
                ]
                if len(valid_samples) >= 3:
                    points_world = [self._project_to_world(pose, u, v, d) for (u, v), d in valid_samples]
                    centroid, bbox_min, bbox_max = self._geometry_from_points(points_world)
                    if self._use_metric_scale:
                        centroid = [c * self._metric_scale_factor for c in centroid]
                        bbox_min = [b * self._metric_scale_factor for b in bbox_min]
                        bbox_max = [b * self._metric_scale_factor for b in bbox_max]
                    scale = [
                        max(0.001, bbox_max[0] - bbox_min[0]),
                        max(0.001, bbox_max[1] - bbox_min[1]),
                        max(0.001, bbox_max[2] - bbox_min[2]),
                    ]

                    prev = self._last_centroids.get(object_id)
                    prev_ts = self._last_timestamps.get(object_id)
                    if prev is not None and prev_ts is not None:
                        dt = float(detections.timestamp) - float(prev_ts)
                        if dt > 1e-6:
                            velocity = [
                                (centroid[0] - prev[0]) / dt,
                                (centroid[1] - prev[1]) / dt,
                                (centroid[2] - prev[2]) / dt,
                            ]
                    self._last_centroids[object_id] = centroid
                    self._last_timestamps[object_id] = float(detections.timestamp)

                    depth_consistency = self._depth_consistency(depth_samples)
                    geom_quality = self._geom_quality(item.score, pose.pose_quality, depth_samples)
                    geometry_status = "metric"

            interaction = "moving" if velocity is not None and any(abs(v) > 0.05 for v in velocity) else "idle"
            label = item.concept_label.lower()
            orientation_quat = self._object_orientations.get(object_id)

            node = ObjectNode(
                object_id=object_id,
                label=label,
                confidence=round((item.score + geom_quality) / 2.0, 4) if centroid is not None else round(item.score, 4),
                segment_kind=item.segment_kind,
                geometry=Geometry(
                    bbox_2d=item.bbox,
                    mask_ref=item.mask_ref,
                    pose_3d=Pose3D(
                        position=centroid,
                        orientation_quat=orientation_quat,
                    ),
                    scale_3d=scale,
                    shape_proxy=self._shape_proxy_for_label(label),
                ),
                physics=PhysicsState(
                    is_dynamic=label not in {"table", "shelf", "desk"},
                    velocity_linear=velocity,
                    velocity_angular=None,
                ),
                semantic=SemanticState(
                    category=self._category_for_label(label),
                    attributes=["rigid"],
                ),
                affordance=self._affordance_prior(label),
                state={
                    "visibility": "visible",
                    "interaction_state": interaction,
                    "last_seen_ts": detections.timestamp,
                },
                provenance=Provenance(sensor="cam_front", frame_idx=detections.frame_idx, model="pit_real_v1"),
            )
            objects.append(node)

            self._object_trajectory[object_id].append(
                {
                    "frame_id": detections.frame_idx,
                    "timestamp_sec": detections.timestamp,
                    "centroid_world": centroid,
                    "orientation_quat": node.geometry.pose_3d.orientation_quat,
                    "bbox_3d_aabb": {"min": bbox_min, "max": bbox_max} if bbox_min is not None and bbox_max is not None else None,
                    "mask_quality": item.score,
                    "geom_quality": geom_quality,
                    "pose_quality": pose.pose_quality if pose is not None else None,
                    "depth_consistency_score": depth_consistency,
                    "depth_provider": self._depth_mode,
                    "geometry_status": geometry_status,
                }
            )

        return objects

    def pit_snapshot(self) -> dict:
        return {
            "camera_provider": self._camera_mode,
            "depth_provider": self._depth_mode,
            "camera_error": self._camera_error,
            "depth_error": self._depth_error,
            "frame_records": list(self._frame_records),
            "camera_trajectory": list(self._camera_trajectory),
            "object_trajectories": dict(self._object_trajectory),
            "metric_enabled": self._depth_provider_is_metric() and bool(self._camera_trajectory),
            "scene_to_meter": self._metric_scale_factor if self._use_metric_scale else 1.0,
        }

    def _recover_camera_pose(self, frame_id: int, timestamp_sec: float) -> CameraPose | None:
        if self._wildgs is not None:
            return self._wildgs.pose_for_frame(frame_id, timestamp_sec)
        if self._camera_mode == "wildgs":
            return None

        if self._colmap is not None:
            return self._colmap.pose_for_frame(frame_id, timestamp_sec)

        # SAM3D body camera: use if available and provider is set to sam3d_body
        if self._camera_mode == "sam3d_body" and self._sam3d_camera is not None:
            pose = self._sam3d_camera
            # Update frame/timestamp for current frame
            return CameraPose(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                K=pose.K,
                R=pose.R,
                t=pose.t,
                pose_quality=pose.pose_quality,
            )

        return None

    def update_camera_from_sam3d(self, sam3d_meshes: dict[str, dict], frame_id: int, timestamp_sec: float) -> None:
        """Extract camera intrinsics from SAM3D body results and update the camera pose.

        Called by the perception pipeline after SAM3D body reconstruction.
        When a body is detected, SAM3D estimates focal_length and camera_translation
        which provide better camera intrinsics than the default.
        """
        for _oid, mesh_info in sam3d_meshes.items():
            if mesh_info.get("segment_kind") != "body":
                continue
            focal = mesh_info.get("camera_focal_length")
            cam_t = mesh_info.get("camera_translation")
            if focal is None:
                continue

            # Build intrinsics from estimated focal length
            fl = float(focal)
            K = self._K_default
            # Update focal length, keep principal point from default
            K_updated = [[fl, 0.0, K[0][2]], [0.0, fl, K[1][2]], [0.0, 0.0, 1.0]]

            # Camera translation from SAM3D: [tx, ty, tz] in camera space
            t = [0.0, 0.0, 0.0]
            if cam_t and len(cam_t) >= 3:
                t = [float(cam_t[0]), float(cam_t[1]), float(cam_t[2])]

            self._sam3d_camera = CameraPose(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                K=K_updated,
                R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                t=t,
                pose_quality=0.92,
            )

            # Auto-switch to sam3d_body mode when body data becomes available
            if self._camera_mode == "none":
                self._reset_spatial_history()
                self._camera_mode = "sam3d_body"

            break  # use first body

    def _reset_spatial_history(self) -> None:
        """Drop cached trajectories when the active camera model changes.

        Historical 3D points and trajectories are expressed in the previous camera
        coordinate system. Keeping them would mix incompatible world frames and
        produce artificial jumps in centroid, velocity, and AABB size.
        """
        self._last_centroids.clear()
        self._last_timestamps.clear()
        self._object_points.clear()
        self._camera_trajectory.clear()
        self._object_trajectory.clear()
        self._frame_records.clear()

    def update_orientations_from_sam3d(self, sam3d_meshes: dict[str, dict]) -> None:
        """Extract body orientation from SAM3D results and store as orientation overrides.

        For body objects, SAM3D provides global_rot (rotation matrix or axis-angle).
        This converts it to a quaternion [qx, qy, qz, qw] for use in ObjectNode.
        """
        for oid, mesh_info in sam3d_meshes.items():
            rot = mesh_info.get("body_global_rotation")
            if rot is None:
                continue
            quat = _rotation_to_quat(rot)
            if quat is not None:
                self._object_orientations[oid] = quat

    def _sample_pixels_from_bbox(self, item: DetectedInstance, num_samples: int) -> list[tuple[float, float]]:
        # Try mask-based sampling first (more accurate — avoids background pixels)
        if item.mask_rle:
            samples = self._sample_pixels_from_mask(item, num_samples)
            if samples:
                return samples
        # Fallback: uniform grid within bbox
        x1, y1, x2, y2 = item.bbox
        w = max(2.0, x2 - x1)
        h = max(2.0, y2 - y1)
        grid_n = max(3, int(math.sqrt(num_samples)))
        return [
            (x1 + (gx + 0.5) * w / grid_n, y1 + (gy + 0.5) * h / grid_n)
            for gy in range(grid_n)
            for gx in range(grid_n)
        ]

    def _sample_pixels_from_mask(self, item: DetectedInstance, num_samples: int) -> list[tuple[float, float]]:
        """Uniformly sample pixel coordinates from within the segmentation mask.

        Decodes the COCO RLE mask and picks up to ``num_samples`` foreground
        pixels at roughly uniform spacing.  Returns an empty list if decoding
        fails or the mask is empty.
        """
        try:
            import json
            import numpy as np
            from pycocotools import mask as mask_util  # type: ignore[import]
        except Exception as exc:
            logger.debug("_sample_pixels_from_mask: pycocotools not available, falling back to bbox: %s", exc)
            return []

        try:
            rle = json.loads(item.mask_rle)
            binary = mask_util.decode(rle)  # uint8 [H, W]
        except Exception as exc:
            logger.warning("_sample_pixels_from_mask: failed to decode mask_rle for %s: %s", item.object_id, exc)
            return []

        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return []

        # Subsample evenly-spaced indices to get ~num_samples points
        total = len(xs)
        if total <= num_samples:
            indices = np.arange(total)
        else:
            indices = np.linspace(0, total - 1, num_samples, dtype=int)

        return [(float(xs[i]), float(ys[i])) for i in indices]

    def _estimate_depth_samples(
        self,
        image_b64: str | None,
        item: DetectedInstance,
        sampled_uv: list[tuple[float, float]],
        frame_idx: int = 0,
    ) -> list[float | None]:
        _ = item
        if self._depth_provider_impl is not None and image_b64:
            try:
                return self._depth_provider_impl.depth_values(image_b64, sampled_uv, frame_idx)
            except Exception as exc:
                logger.warning("Depth provider '%s' failed for frame %s: %s", self._depth_mode, frame_idx, exc)
        return [None] * len(sampled_uv)

    def _depth_provider_is_metric(self) -> bool:
        provider = self._depth_provider_impl
        if provider is None:
            return False
        if self._depth_mode == "wildgs":
            return True
        metric_attr = getattr(provider, "is_metric", None)
        if isinstance(metric_attr, bool):
            return metric_attr
        if callable(metric_attr):
            try:
                return bool(metric_attr())
            except Exception:
                return False
        return False

    def _project_to_world(self, pose: CameraPose, u: float, v: float, depth: float) -> tuple[float, float, float]:
        fx = pose.K[0][0]
        fy = pose.K[1][1]
        cx = pose.K[0][2]
        cy = pose.K[1][2]

        x_c = (u - cx) * depth / fx
        y_c = (v - cy) * depth / fy
        z_c = depth

        x_w = pose.R[0][0] * x_c + pose.R[0][1] * y_c + pose.R[0][2] * z_c + pose.t[0]
        y_w = pose.R[1][0] * x_c + pose.R[1][1] * y_c + pose.R[1][2] * z_c + pose.t[1]
        z_w = pose.R[2][0] * x_c + pose.R[2][1] * y_c + pose.R[2][2] * z_c + pose.t[2]
        return (x_w, y_w, z_w)

    def _geometry_from_points(self, points: list[tuple[float, float, float]]) -> tuple[list[float], list[float], list[float]]:
        if not points:
            return [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.1, 0.1, 1.1]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        centroid = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
        bbox_min = [min(xs), min(ys), min(zs)]
        bbox_max = [max(xs), max(ys), max(zs)]
        return centroid, bbox_min, bbox_max

    def _geom_quality(self, mask_quality: float, pose_quality: float, depth_samples: list[float | None]) -> float:
        depth_score = self._depth_consistency(depth_samples)
        if depth_score is None:
            return 0.0
        return max(0.0, min(1.0, 0.5 * mask_quality + 0.3 * pose_quality + 0.2 * depth_score))

    def _depth_consistency(self, depth_samples: list[float | None]) -> float | None:
        valid = [float(d) for d in depth_samples if d is not None and math.isfinite(float(d)) and float(d) > 0.0]
        if not valid:
            return None
        mean_d = sum(valid) / len(valid)
        if mean_d <= 1e-6:
            return None
        var = sum((d - mean_d) ** 2 for d in valid) / len(valid)
        coeff_var = math.sqrt(var) / mean_d
        return max(0.0, min(1.0, 1.0 - coeff_var))

    def _deduplicate_instances(
        self, instances: list[DetectedInstance], iou_threshold: float = 0.5,
    ) -> list[DetectedInstance]:
        return deduplicate_instances(instances, iou_threshold=iou_threshold)

    @staticmethod
    def _bbox_is_valid(bbox: list[float], min_size: float = 2.0) -> bool:
        if len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return math.isfinite(width) and math.isfinite(height) and width >= min_size and height >= min_size

    def _resolve_object_id(self, source_id: str, instance: DetectedInstance | None = None, frame_idx: int = 0) -> str:
        existing = self._source_to_object.get(source_id)
        if existing:
            self._object_to_source[existing] = source_id
            return existing

        # Search graveyard for a recently-removed object with matching label & bbox
        if instance is not None:
            label = instance.concept_label.lower()
            best_entry: _GraveyardEntry | None = None
            best_iou = 0.0
            for entry in self._graveyard:
                age = frame_idx - entry.removed_frame
                if age < self._graveyard_grace_frames:
                    continue  # too recently removed; avoid re-add in same step
                if age > self._graveyard_ttl_frames:
                    continue
                if not self._label_matches(label, entry.label):
                    continue
                iou = self._bbox_iou(instance.bbox, entry.bbox_2d)
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_entry = entry
            if best_entry is not None:
                self._source_to_object[source_id] = best_entry.object_id
                self._object_to_source[best_entry.object_id] = source_id
                return best_entry.object_id

        suffix = source_id.split("_")[-1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        object_id = f"obj_{digits.zfill(6)}" if digits else f"obj_{str(len(self._source_to_object) + 1).zfill(6)}"
        self._source_to_object[source_id] = object_id
        self._object_to_source[object_id] = source_id
        return object_id

    def notify_removed(self, object_id: str, label: str, bbox_2d: list[float], frame_idx: int) -> None:
        """Push a removed object into the graveyard for cross-track deduplication.

        Also cleans up the stale source-ID mapping so that if the same detector ID
        is later re-assigned by the perception backend, it will be treated as a new
        observation rather than silently reusing the removed object's ID.
        """
        self._graveyard.append(_GraveyardEntry(
            object_id=object_id,
            label=label,
            bbox_2d=list(bbox_2d),
            removed_frame=frame_idx,
        ))
        source_id = self._object_to_source.pop(object_id, None)
        if source_id and self._source_to_object.get(source_id) == object_id:
            del self._source_to_object[source_id]

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        return bbox_iou(a, b)

    @staticmethod
    def _label_matches(a: str, b: str) -> bool:
        return label_matches(a, b)

    def _shape_proxy_for_label(self, label: str) -> str:
        if label in {"cup", "bottle"}:
            return "cylinder"
        if label in {"robot arm", "robot gripper"}:
            return "capsule"
        return "box"

    def _affordance_prior(self, label: str) -> AffordanceState:
        if label in {"cup", "bottle"}:
            return AffordanceState(graspable=True, supportable=True, pourable=(label == "bottle"))
        if label in {"table", "desk", "shelf"}:
            return AffordanceState(supportable=True)
        return AffordanceState()

    def _category_for_label(self, label: str) -> str:
        mapping = {
            "cup": "container",
            "bottle": "container",
            "table": "furniture",
            "desk": "furniture",
            "robot gripper": "manipulator",
            "robot arm": "manipulator",
        }
        return mapping.get(label, "unknown")


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _mat_vec_mul(m: list[list[float]], v: list[float]) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _rotation_to_quat(rot: Any) -> list[float] | None:
    """Convert a rotation representation to quaternion [qx, qy, qz, qw].

    Accepts:
      - 3x3 rotation matrix (list of 3 lists of 3 floats)
      - 3-element axis-angle vector
      - Already a 4-element quaternion
    Returns None if the format is unrecognised.
    """
    if not isinstance(rot, list):
        return None

    # Already a quaternion
    if len(rot) == 4 and all(isinstance(v, (int, float)) for v in rot):
        return [float(v) for v in rot]

    # Axis-angle (3-vector)
    if len(rot) == 3 and all(isinstance(v, (int, float)) for v in rot):
        ax, ay, az = float(rot[0]), float(rot[1]), float(rot[2])
        angle = math.sqrt(ax * ax + ay * ay + az * az)
        if angle < 1e-8:
            return [0.0, 0.0, 0.0, 1.0]
        half = angle / 2.0
        s = math.sin(half) / angle
        return [ax * s, ay * s, az * s, math.cos(half)]

    # 3x3 rotation matrix
    if len(rot) == 3 and all(isinstance(row, list) and len(row) == 3 for row in rot):
        try:
            qw, qx, qy, qz = _rot_matrix_to_wxyz(rot)
            return [qx, qy, qz, qw]
        except Exception:
            return None

    return None


def _rot_matrix_to_wxyz(rot: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
    m00, m01, m02 = float(rot[0][0]), float(rot[0][1]), float(rot[0][2])
    m10, m11, m12 = float(rot[1][0]), float(rot[1][1]), float(rot[1][2])
    m20, m21, m22 = float(rot[2][0]), float(rot[2][1]), float(rot[2][2])

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        return (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
