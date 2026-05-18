from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from guanwu.video.core.config import apply_session_output_root
from guanwu.video.clients.zaiwu import build_zaiwu_visual_pose_tracker, normalize_provider_mode
from guanwu.video.infra.isaac_sync import IsaacSyncAgent
from guanwu.video.features.world_inference.event_engine import EventEngine
from guanwu.video.features.world_inference.relation_engine import RelationEngine
from guanwu.video.features.spatial.object_scene_alignment import ObjectSceneAlignmentRefiner
from guanwu.video.features.spatial.visual_pose_tracking import build_visual_pose_tracker
from guanwu.video.features.spatial.state_estimator import StateEstimationAgent
from guanwu.video.features.simulation.pit2isaac_exporter import PIT2IsaacExporter
from guanwu.video.features.simulation.runner import SimulationPipeline
from guanwu.video.project.config import ProjectConfig


class VideoFrameReader:
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = str(Path(video_path).expanduser().resolve())

    def metadata(self) -> dict[str, float | int | str]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        return {
            "video_path": self.video_path,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        }

    def iter_frames(self) -> list[tuple[int, float, np.ndarray]]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[tuple[int, float, np.ndarray]] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = (frame_idx - 1) / fps if fps > 1e-6 else float(frame_idx - 1)
            frames.append((frame_idx, timestamp, frame))
        cap.release()
        return frames

    @staticmethod
    def encode_jpg(frame: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return base64.b64encode(buffer.tobytes()).decode("ascii")


@dataclass
class PipelineServices:
    frames: list[tuple[int, float, np.ndarray]]
    estimator: StateEstimationAgent
    relation_engine: RelationEngine
    event_engine: EventEngine
    object_scene_alignment: ObjectSceneAlignmentRefiner
    simulation_pipeline: SimulationPipeline


def build_services(config: ProjectConfig, session_root: Path) -> PipelineServices:
    settings = config.settings.model_copy(deep=True)
    apply_session_output_root(settings, session_root)
    settings.zaiwu.enabled = settings.zaiwu.enabled or (
        normalize_provider_mode(config.project.provider_mode) == "zaiwu"
    )

    reader = VideoFrameReader(config.project.input_video)
    frames = reader.iter_frames()

    estimator = StateEstimationAgent(
        camera_provider=settings.pit.camera_provider,
        colmap_model_dir=settings.pit.colmap_model_dir,
        wildgs_camera_poses_jsonl=settings.pit.wildgs_camera_poses_jsonl,
        wildgs_static_map_dir=settings.pit.wildgs_static_map_dir,
        wildgs_dynamic_prior_dir=settings.pit.wildgs_dynamic_prior_dir,
        wildgs_depth_maps_dir=settings.pit.wildgs_depth_maps_dir,
        depth_provider=settings.pit.depth_provider,
        depth_model_path=settings.pit.depth_model_path,
        zaiwu_gateway_url=settings.zaiwu.gateway_url if settings.zaiwu.enabled else None,
        zaiwu_depth_service=settings.zaiwu.depth_service,
        video_source=config.project.input_video,
        use_metric_scale=settings.pit.use_metric_scale,
        metric_scale_factor=settings.pit.metric_scale_factor,
    )
    relation_engine = RelationEngine()
    event_engine = EventEngine()
    if settings.pit.visual_pose_mcp_url:
        visual_pose_tracker = build_visual_pose_tracker(
            backend=settings.pit.alignment_backend,
            prefer_mcp=True,
            mcp_url=settings.pit.visual_pose_mcp_url,
            mcp_tool=settings.pit.visual_pose_mcp_tool,
            command=settings.pit.visual_pose_command,
            timeout_sec=settings.pit.visual_pose_timeout_sec,
        )
    else:
        visual_pose_tracker = build_zaiwu_visual_pose_tracker(
            settings,
            timeout_sec=settings.pit.visual_pose_timeout_sec,
        ) or build_visual_pose_tracker(
            backend=settings.pit.alignment_backend,
            prefer_mcp=False,
            mcp_url=None,
            mcp_tool=settings.pit.visual_pose_mcp_tool,
            command=settings.pit.visual_pose_command,
            timeout_sec=settings.pit.visual_pose_timeout_sec,
        )
    simulation_pipeline = SimulationPipeline(
        IsaacSyncAgent(stage_path=settings.isaac.stage_path, auto_save=settings.isaac.auto_save),
        PIT2IsaacExporter(
            mode=settings.pit2isaac.mode,
            output_root=settings.pit2isaac.output_root,
            usd_path=settings.pit2isaac.usd_path,
            physics_priors_json=settings.pit2isaac.physics_priors_json,
            asset_mapping_json=settings.pit2isaac.asset_mapping_json,
            conversion_report_json=settings.pit2isaac.conversion_report_json,
            use_category_assets=settings.pit2isaac.use_category_assets,
            fallback_visual=settings.pit2isaac.fallback_visual,
            collision_strategy=settings.pit2isaac.collision_strategy,
            min_geom_quality=settings.pit2isaac.min_geom_quality,
            output_format=settings.pit2isaac.output_format,
        ),
    )
    return PipelineServices(
        frames=frames,
        estimator=estimator,
        relation_engine=relation_engine,
        event_engine=event_engine,
        object_scene_alignment=ObjectSceneAlignmentRefiner(
            alignment_backend=settings.pit.alignment_backend,
            visual_pose_tracker=visual_pose_tracker,
            visual_pose_min_score=settings.pit.visual_pose_min_score,
            visual_pose_max_translation_step_m=settings.pit.visual_pose_max_translation_step_m,
            visual_pose_max_rotation_step_deg=settings.pit.visual_pose_max_rotation_step_deg,
        ),
        simulation_pipeline=simulation_pipeline,
    )
