from __future__ import annotations

import copy
import math
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from guanwu.video.core.config import apply_session_output_root, load_settings
from guanwu.video.core.schema import Event, ObjectNode, RelationEdge, WorldState
from guanwu.video.core.time_sync import TimeSync
from guanwu.video.infra.isaac_sync import IsaacSyncAgent
from guanwu.video.infra.storage_sqlite import WorldStore
from guanwu.video.features.world_inference.relation_engine import RelationEngine
from guanwu.video.features.world_inference.event_engine import EventEngine
from guanwu.video.features.spatial.state_estimator import StateEstimationAgent
from guanwu.video.features.spatial.object_scene_alignment import ObjectSceneAlignmentRefiner
from guanwu.video.features.spatial.visual_pose_tracking import build_visual_pose_tracker
from guanwu.video.features.simulation.pit2isaac_exporter import PIT2IsaacExporter
from guanwu.video.features.simulation.runner import SimulationPipeline
from guanwu.video.clients.zaiwu import (
    build_zaiwu_object_detector,
    build_zaiwu_sam3d_adapter,
    build_zaiwu_visual_pose_tracker,
    build_zaiwu_wildgs_adapter,
)
from guanwu.video.features.detection.vlm_discovery import VLMDiscoveryAgent
from guanwu.video.features.detection.keyframe_detector import KeyframeDetector
from guanwu.video.features.detection.open_vocab_detector import OpenVocabDetector
from guanwu.video.features.world_inference.object_attr import ObjectAttrAgent
from guanwu.video.features.spatial.background_reconstruction import BackgroundReconstructionPipeline
from guanwu.video.features.spatial.object_video_extractor import ObjectVideoExtractor
from guanwu.video.viz.rerun_viz import HAS_RERUN, RerunVisualizer
from guanwu.video.core.logger import get_logger
from guanwu.video.core.instance_matching import match_instances_to_objects

logger = get_logger(__name__)


def _valid_vec3(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        out = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out


class WorldRuntime:
    def __init__(
        self,
        settings_override=None,
        *,
        session_output_root: str | Path | None = None,
        save_intermediate: bool | None = None,
        asset_materialization: str | None = None,
    ) -> None:
        self.settings, self.config_path = load_settings()
        if settings_override:
            self.settings = settings_override
        resolved_session_root = session_output_root or self.settings.runtime.session_output_root
        if not resolved_session_root:
            raise ValueError("runtime.session_output_root is required")
        apply_session_output_root(self.settings, resolved_session_root)
        if save_intermediate is not None:
            self.settings.runtime.save_intermediate = bool(save_intermediate)
        if asset_materialization is not None:
            self.settings.runtime.asset_materialization = str(asset_materialization).strip().lower()
        if self.settings.runtime.asset_materialization not in {"copy", "move", "hardlink", "symlink"}:
            raise ValueError(
                "runtime.asset_materialization must be one of: copy, move, hardlink, symlink"
            )
        self.session_output_root = Path(self.settings.runtime.session_output_root).resolve()
        self.object_detector_init_error: str | None = None
        self.frame_idx = 0
        self.occluded_ttl_frames = self.settings.runtime.occluded_ttl_frames
        self.removal_ttl_frames = self.settings.runtime.removal_ttl_frames
        self.session_output_root.mkdir(parents=True, exist_ok=True)
        self._ensure_session_dirs()
        self.settings.zaiwu.enabled = True
        try:
            backend_client = None
            object_detector = build_zaiwu_object_detector(
                self.settings,
                video_source=self.settings.runtime.video_source,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.object_detector_init_error = str(exc)
            logger.warning(f"Object detector initialization had issues: {exc}")
            backend_client = None
            object_detector = build_zaiwu_object_detector(
                self.settings,
                video_source=self.settings.runtime.video_source,
            )

        estimator = StateEstimationAgent(
            camera_provider=self.settings.pit.camera_provider,
            colmap_model_dir=self.settings.pit.colmap_model_dir,
            wildgs_camera_poses_jsonl=self.settings.pit.wildgs_camera_poses_jsonl,
            wildgs_static_map_dir=self.settings.pit.wildgs_static_map_dir,
            wildgs_dynamic_prior_dir=self.settings.pit.wildgs_dynamic_prior_dir,
            wildgs_depth_maps_dir=self.settings.pit.wildgs_depth_maps_dir,
            depth_provider=self.settings.pit.depth_provider,
            depth_model_path=self.settings.pit.depth_model_path,
            zaiwu_gateway_url=self.settings.zaiwu.gateway_url,
            zaiwu_depth_service=self.settings.zaiwu.depth_service,
            video_source=self.settings.runtime.video_source,
            use_metric_scale=self.settings.pit.use_metric_scale,
            metric_scale_factor=self.settings.pit.metric_scale_factor,
        )
        sam3d = build_zaiwu_sam3d_adapter(
            self.settings,
            materialization_root=str(self.session_output_root),
            materialization_mode=self.settings.runtime.asset_materialization,
        )
            
        relation_engine = RelationEngine()
        event_engine = EventEngine()
        vlm_physics = ObjectAttrAgent(self.settings.vlm)

        # Background scene reconstruction via WildGS-SLAM (MCP only)
        self.wildgs = None
        self.background_pipeline: BackgroundReconstructionPipeline | None = None
        if self.settings.runtime.background_reconstruction:
            self.wildgs = build_zaiwu_wildgs_adapter(
                self.settings,
                output_root=str(self.session_output_root),
            )
            self.background_pipeline = BackgroundReconstructionPipeline(
                reconstruction_adapter=self.wildgs,
                video_source=self.settings.runtime.video_source,
                depth_estimator=estimator,
                sample_frames=self.settings.runtime.background_sample_frames,
            )

        self.rigid_body_registry: dict[str, bool] = {}
        self.movable_registry: dict[str, bool] = {}
        self.object_video_extractor = ObjectVideoExtractor()
        self.object_detector_prefetched = False
        self.sam3d_agent = sam3d
        self._sam3d_cache: dict[str, dict] = {}
        if self.settings.pit.visual_pose_mcp_url:
            visual_pose_tracker = build_visual_pose_tracker(
                backend=self.settings.pit.alignment_backend,
                prefer_mcp=True,
                mcp_url=self.settings.pit.visual_pose_mcp_url,
                mcp_tool=self.settings.pit.visual_pose_mcp_tool,
                command=self.settings.pit.visual_pose_command,
                timeout_sec=self.settings.pit.visual_pose_timeout_sec,
            )
        else:
            visual_pose_tracker = build_zaiwu_visual_pose_tracker(
                self.settings,
                timeout_sec=self.settings.pit.visual_pose_timeout_sec,
            ) or build_visual_pose_tracker(
                backend=self.settings.pit.alignment_backend,
                prefer_mcp=False,
                mcp_url=None,
                mcp_tool=self.settings.pit.visual_pose_mcp_tool,
                command=self.settings.pit.visual_pose_command,
                timeout_sec=self.settings.pit.visual_pose_timeout_sec,
            )
        self.object_scene_alignment = ObjectSceneAlignmentRefiner(
            alignment_backend=self.settings.pit.alignment_backend,
            visual_pose_tracker=visual_pose_tracker,
            visual_pose_min_score=self.settings.pit.visual_pose_min_score,
            visual_pose_max_translation_step_m=self.settings.pit.visual_pose_max_translation_step_m,
            visual_pose_max_rotation_step_deg=self.settings.pit.visual_pose_max_rotation_step_deg,
        )
        self.relation_engine = relation_engine
        self.event_engine = event_engine
        self.vlm_physics = vlm_physics
        self._vlm_executor = ThreadPoolExecutor(max_workers=1)
        self._vlm_pending_future: Future | None = None

        isaac_sync = IsaacSyncAgent(
            stage_path=self.settings.isaac.stage_path,
            auto_save=self.settings.isaac.auto_save,
        )
        pit2isaac = PIT2IsaacExporter(
            mode=self.settings.pit2isaac.mode,
            output_root=self.settings.pit2isaac.output_root,
            usd_path=self.settings.pit2isaac.usd_path,
            physics_priors_json=self.settings.pit2isaac.physics_priors_json,
            asset_mapping_json=self.settings.pit2isaac.asset_mapping_json,
            conversion_report_json=self.settings.pit2isaac.conversion_report_json,
            use_category_assets=self.settings.pit2isaac.use_category_assets,
            fallback_visual=self.settings.pit2isaac.fallback_visual,
            collision_strategy=self.settings.pit2isaac.collision_strategy,
            min_geom_quality=self.settings.pit2isaac.min_geom_quality,
            output_format=self.settings.pit2isaac.output_format,
        )
        self.simulation_pipeline = SimulationPipeline(isaac_sync, pit2isaac)

        self.isaac_sync = isaac_sync
        self.service_client = backend_client
        self.object_detector = object_detector
        self.estimator = estimator
        self.store = WorldStore(world_id=self.settings.storage.world_id)
        self.time_sync = TimeSync()
        self.vlm_discovery = VLMDiscoveryAgent(self.settings.vlm)
        self.last_sync_report: dict = {
            "backend": self.isaac_sync.backend,
            "stage_path": self.isaac_sync.stage_path,
            "created": 0,
            "updated": 0,
            "removed": 0,
            "active_prims": 0,
            "prim_paths": {},
        }

        self.object_registry: dict[str, ObjectNode] = {}
        self.object_missed_frames: dict[str, int] = {}
        self._relation_trajectories: dict[str, list[dict]] = defaultdict(list)
        self._sam3d_meshes: dict[str, dict] = {}
        self._vlm_priors: dict[str, dict] = {}
        self._aligned_pit_snapshot: dict | None = None
        self._hypothesis_counter = 0
        self._snapshot_counter = 0
        self._hypothesis_snapshots: dict[str, dict] = {}
        self._snapshot_order_by_object: dict[str, list[str]] = defaultdict(list)
        self._vlm_discovery_done = False
        vlm_disc_cfg = self.settings.runtime.vlm_discovery
        self.keyframe_detector = KeyframeDetector(
            periodic_interval=vlm_disc_cfg.periodic_interval,
            new_track_threshold=vlm_disc_cfg.new_track_threshold,
            disappear_threshold=vlm_disc_cfg.disappear_threshold,
            confidence_drop_threshold=vlm_disc_cfg.confidence_drop_threshold,
            image_change_threshold=vlm_disc_cfg.image_change_threshold,
            image_change_enabled=vlm_disc_cfg.image_change_enabled,
        )

        # Per-frame open-vocab VLM trigger
        self.open_vocab_detector: OpenVocabDetector | None = None
        if vlm_disc_cfg.open_vocab_enabled:
            self.open_vocab_detector = OpenVocabDetector(
                cooldown_frames=vlm_disc_cfg.open_vocab_cooldown_frames,
            )

        # Rerun real-time visualizer (optional)
        self.rerun_viz: RerunVisualizer | None = None
        if self.settings.runtime.rerun_enabled and HAS_RERUN:
            self.rerun_viz = RerunVisualizer(application_id="spwm_scene")
        elif self.settings.runtime.rerun_enabled and not HAS_RERUN:
            logger.warning("[WorldRuntime] rerun_enabled=true but rerun-sdk is not installed. Skipping.")
        if self.background_pipeline is not None:
            self._bootstrap_wildgs_before_first_frame()

    def _reset_state_for_camera_provider_change(self) -> None:
        """Drop runtime caches tied to the previous world frame."""
        self.object_registry.clear()
        self.object_missed_frames.clear()
        self._relation_trajectories.clear()
        self._sam3d_meshes.clear()
        self._aligned_pit_snapshot = None
        self.last_sync_report = {
            "backend": self.isaac_sync.backend,
            "stage_path": self.isaac_sync.stage_path,
            "created": 0,
            "updated": 0,
            "removed": 0,
            "active_prims": 0,
            "prim_paths": {},
        }
        self.store = WorldStore(world_id=self.settings.storage.world_id)

    def _ensure_session_dirs(self) -> None:
        (self.session_output_root / "logs").mkdir(parents=True, exist_ok=True)
        (self.session_output_root / "exports").mkdir(parents=True, exist_ok=True)
        (self.session_output_root / "intermediate").mkdir(parents=True, exist_ok=True)
        (self.session_output_root / "runtime").mkdir(parents=True, exist_ok=True)

    def _bootstrap_wildgs_before_first_frame(self) -> None:
        """Recover camera poses before the first runtime step."""
        assert self.background_pipeline is not None
        try:
            bg_result = self.background_pipeline.bootstrap_reconstruction()
        except Exception as exc:
            logger.warning("[WorldRuntime] WildGS bootstrap failed; continuing without it: %s", exc)
            return

        camera_poses_path = str(bg_result.get("camera_poses_path", "")).strip()
        if not camera_poses_path:
            return

        switched = self.estimator.attach_wildgs_results(
            camera_poses_jsonl=camera_poses_path,
            static_map_dir=bg_result.get("static_map_dir"),
            dynamic_prior_dir=bg_result.get("dynamic_prior_dir"),
        )
        if switched:
            self._reset_state_for_camera_provider_change()
        logger.info("[WorldRuntime] Bootstrapped WildGS camera poses before frame 1: %s", camera_poses_path)

    def _bootstrap_first_frame_discovery(self) -> None:
        """First frame: run VLM discovery BEFORE perception so prompts are set for prefetch."""
        if self._vlm_discovery_done or not self.settings.runtime.vlm_discovery.enabled:
            return
        self._vlm_discovery_done = True
        image_b64 = self.object_detector.get_first_frame_b64()
        if not image_b64:
            return
        new_categories = self.vlm_discovery.discover_objects(image_b64)
        logger.info(f"[WorldRuntime] VLM discovery (first frame) found {len(new_categories)} categories: {new_categories}")
        if new_categories:
            current_prompts = self.object_detector.get_object_detection_prompts()
            merged = list(dict.fromkeys(current_prompts + new_categories))
            self.object_detector.set_object_detection_prompts(merged)
            logger.info(f"[WorldRuntime] Prompts updated: {merged}")

    def _prefetch_video_if_supported(self) -> None:
        if self.object_detector_prefetched:
            return
        fn = getattr(self.object_detector, "prefetch_video", None)
        if not callable(fn):
            self.object_detector_prefetched = True
            return
        logger.info("[WorldRuntime] Prefetching full video via object detector...")
        try:
            fn()
        finally:
            self.object_detector_prefetched = True

    def _run_detection(self, frame_idx: int, source_video_time: float):
        self._prefetch_video_if_supported()
        detections = self.object_detector.detect_objects_in_frame(frame_idx, source_video_time)
        observed_objects = self.estimator.estimate(detections)

        unknown = {o.object_id for o in observed_objects if o.object_id not in self.movable_registry}
        immovable = {o.object_id for o in observed_objects if self.movable_registry.get(o.object_id) is False}
        eligible = {o.object_id for o in observed_objects if self.movable_registry.get(o.object_id) is True}
        new_objects = [o for o in observed_objects if o.object_id in eligible and o.object_id not in self._sam3d_cache]
        non_rigid_objects = [
            o for o in observed_objects
            if o.object_id in self._sam3d_cache
            and o.object_id in eligible
            and not self.rigid_body_registry.get(o.object_id, True)
        ]
        to_reconstruct = new_objects + non_rigid_objects
        if to_reconstruct:
            matched_instances = match_instances_to_objects(to_reconstruct, detections.instances)
            best_frames = {
                object_id: (detections, inst)
                for object_id, inst in matched_instances.items()
            }
            new_meshes = self.sam3d_agent.reconstruct_object_meshes(best_frames, to_reconstruct)
            self._sam3d_cache.update(new_meshes)
        elif immovable or unknown:
            logger.debug(
                "[WorldRuntime] SAM3D skipped: %d immovable, %d unknown-movability objects.",
                len(immovable),
                len(unknown),
            )

        sam3d_meshes = {
            o.object_id: self._sam3d_cache[o.object_id]
            for o in observed_objects
            if o.object_id in self._sam3d_cache
        }
        if sam3d_meshes:
            self.estimator.update_camera_from_sam3d(sam3d_meshes, frame_idx, source_video_time)
            self.estimator.update_orientations_from_sam3d(sam3d_meshes)
        return detections, observed_objects, sam3d_meshes

    def _harvest_pending_vlm_priors(self) -> None:
        if self._vlm_pending_future is None or not self._vlm_pending_future.done():
            return
        try:
            new_priors = self._vlm_pending_future.result()
            self._vlm_priors.update(new_priors)
            for object_id, priors in new_priors.items():
                self.rigid_body_registry[object_id] = priors.get("is_rigid_body") is True
                self.movable_registry[object_id] = priors.get("is_movable") is True
        except Exception as exc:
            logger.error(f"[WorldRuntime] Async VLM failed: {exc}")
        finally:
            self._vlm_pending_future = None

    def _run_world_inference(
        self,
        frame_idx: int,
        source_video_time: float,
        active_objects: list[ObjectNode],
        observed_objects: list[ObjectNode],
        sam3d_meshes: dict[str, dict],
        detections,
        removed_object_ids: list[str],
    ):
        self._harvest_pending_vlm_priors()
        relation_objects = [obj for obj in active_objects if obj.state.visibility == "visible"]
        relations = self.relation_engine.infer(
            relation_objects,
            frame_idx,
            source_video_time,
            background_geometry=self._background_geometry(),
        )
        events = self.event_engine.infer(
            active_objects,
            relations,
            source_video_time,
            removed_object_ids=removed_object_ids,
        )
        new_objects = [o for o in observed_objects if o.object_id not in self._vlm_priors]
        if new_objects and self._vlm_pending_future is None:
            self._vlm_pending_future = self._vlm_executor.submit(
                self.vlm_physics.infer_object_physics_priors,
                copy.deepcopy(detections),
                copy.deepcopy(new_objects),
            )
        vlm_priors = {o.object_id: self._vlm_priors[o.object_id] for o in observed_objects if o.object_id in self._vlm_priors}
        return relations, events, vlm_priors

    def _maybe_vlm_rediscovery(self, frame_idx: int, detections, observed_objects: list, removed_ids: list[str]) -> None:
        """Run VLM incremental discovery on keyframes to detect new object categories."""
        if not self.settings.runtime.vlm_discovery.enabled:
            return
        if not self.keyframe_detector.check(frame_idx, detections, observed_objects, removed_ids):
            return

        image_b64 = getattr(detections, "image_b64", None)
        if not image_b64:
            return

        current_prompts = self.object_detector.get_object_detection_prompts()
        new_categories = self.vlm_discovery.discover_incremental(image_b64, current_prompts)
        if new_categories:
            logger.info(f"[WorldRuntime] VLM incremental discovery found {len(new_categories)} new categories: {new_categories}")
            merged = list(dict.fromkeys(current_prompts + new_categories))
            self.object_detector.set_object_detection_prompts(merged)
            logger.info(f"[WorldRuntime] Prompts updated: {merged}")

    def step_once(self) -> WorldState:
        if not self._vlm_discovery_done:
            self._bootstrap_first_frame_discovery()
        self.frame_idx += 1
        timestamp, sim_time, source_video_time = self.time_sync.tick()

        logger.debug(f"[WorldRuntime] Starting step_once for frame {self.frame_idx} (time: {source_video_time:.3f}s)")

        detections, observed_objects, sam3d_meshes = self._run_detection(self.frame_idx, source_video_time)
        self.object_video_extractor.collect_frame(detections, observed_objects)
        observed_ids: set[str] = set()
        removed_object_ids: list[str] = []

        logger.debug(f"[WorldRuntime] Detection step generated {len(observed_objects)} objects")

        for obj in observed_objects:
            obj.state.visibility = "visible"
            obj.state.last_seen_ts = source_video_time
            self.object_registry[obj.object_id] = obj
            self.object_missed_frames[obj.object_id] = 0
            mesh = sam3d_meshes.get(obj.object_id)
            if mesh:
                self._sam3d_meshes[obj.object_id] = mesh
            observed_ids.add(obj.object_id)

        for object_id, obj in list(self.object_registry.items()):
            if object_id in observed_ids:
                continue
            missed = self.object_missed_frames.get(object_id, 0) + 1
            self.object_missed_frames[object_id] = missed
            if missed <= self.occluded_ttl_frames:
                obj.state.visibility = "occluded"
                obj.state.interaction_state = "idle"
            elif missed <= self.removal_ttl_frames:
                obj.state.visibility = "lost"
                obj.state.interaction_state = "idle"
            else:
                # Notify estimator of removal for cross-track dedup
                self.estimator.notify_removed(
                    object_id=object_id,
                    label=obj.label,
                    bbox_2d=obj.geometry.bbox_2d,
                    frame_idx=self.frame_idx,
                )
                removed_object_ids.append(object_id)
                del self.object_registry[object_id]
                self.object_missed_frames.pop(object_id, None)
                self._sam3d_meshes.pop(object_id, None)
                self._vlm_priors.pop(object_id, None)
                self.rigid_body_registry.pop(object_id, None)
                self.movable_registry.pop(object_id, None)

        # VLM re-discovery on keyframes (updates prompts for next frame)
        # Always call keyframe_detector.check() to update internal state (track IDs),
        # but skip VLM calls on frame 1 since bootstrap already ran before perception.
        if self.frame_idx <= 1:
            self.keyframe_detector.check(
                self.frame_idx, detections, observed_objects, removed_object_ids,
            )
        else:
            self._maybe_vlm_rediscovery(
                self.frame_idx, detections, observed_objects, removed_object_ids,
            )

        # Per-frame open-vocab check (lightweight heuristic, VLM called only when triggered)
        if self.open_vocab_detector is not None and self.settings.runtime.vlm_discovery.enabled:
            if self.open_vocab_detector.should_trigger_vlm(self.frame_idx, detections):
                image_b64 = getattr(detections, "image_b64", None)
                if image_b64:
                    current_prompts = self.object_detector.get_object_detection_prompts()
                    new_cats = self.vlm_discovery.discover_incremental(image_b64, current_prompts)
                    if new_cats:
                        merged = list(dict.fromkeys(current_prompts + new_cats))
                        self.object_detector.set_object_detection_prompts(merged)
                        logger.info(f"[WorldRuntime] OpenVocab discovery: {new_cats}")

        active_objects = list(self.object_registry.values())
        relations, events, vlm_priors = self._run_world_inference(
            frame_idx=self.frame_idx,
            source_video_time=source_video_time,
            active_objects=active_objects,
            observed_objects=observed_objects,
            sam3d_meshes=sam3d_meshes,
            detections=detections,
            removed_object_ids=removed_object_ids,
        )
        self._append_relation_tracks(relations, source_video_time)

        for obj in observed_objects:
            if obj.object_id in vlm_priors:
                self._vlm_priors[obj.object_id] = vlm_priors[obj.object_id]

        # Background reconstruction: collect only movable-object boxes after
        # cognition has updated the movable registry.
        if self.background_pipeline is not None and not self.background_pipeline.is_done:
            self.background_pipeline.collect_frame(
                detections,
                observed_objects=observed_objects,
                movable_registry=self.movable_registry,
            )
            if self.background_pipeline.is_ready:
                depths_file_id = None
                if hasattr(self.estimator, '_depth_provider_impl') and hasattr(self.estimator._depth_provider_impl, '_depth_cache'):
                    # depth cache file_id is not used by the WildGS path
                    pass
                bg_result = self.background_pipeline.reconstruct(self.session_output_root, depths_file_id)
                if bg_result and not bg_result.get("error"):
                    camera_poses_path = str(bg_result.get("camera_poses_path", "")).strip()
                    if camera_poses_path:
                        switched = self.estimator.attach_wildgs_results(
                            camera_poses_jsonl=camera_poses_path,
                            static_map_dir=bg_result.get("static_map_dir"),
                            dynamic_prior_dir=bg_result.get("dynamic_prior_dir"),
                            depth_maps_dir=bg_result.get("depth_maps_dir"),
                        )
                        if switched:
                            self._reset_state_for_camera_provider_change()
                            logger.info(
                                "[WorldRuntime] Camera pose provider switched to WildGS: %s",
                                camera_poses_path,
                            )
                    logger.info(
                        "[WorldRuntime] Background scene reconstruction completed: poses=%s points=%s labels=%s",
                        bg_result.get("camera_poses_path", ""),
                        bg_result.get("points_path", ""),
                        bg_result.get("fg_bg_labels_path", ""),
                    )

        active_objects = list(self.object_registry.values())
        pit_snapshot = self.build_pit_snapshot(raw=True)
        refined_objects, refined_snapshot = self.object_scene_alignment.refine(active_objects, pit_snapshot)
        self._aligned_pit_snapshot = refined_snapshot
        self.object_registry = {obj.object_id: obj for obj in refined_objects}
        active_objects = refined_objects

        sim_bundle = self.simulation_pipeline.sync(active_objects)
        self.last_sync_report = sim_bundle.sync_report
        self.store.upsert_objects(active_objects, source_video_time)
        self.store.delete_objects(removed_object_ids)
        self.store.replace_relations(relations, source_video_time)
        self.store.append_events(events)

        world_state = self.store.build_world_state(timestamp, sim_time, source_video_time)

        # Stream frame to Rerun visualizer
        if self.rerun_viz is not None:
            image_b64 = getattr(detections, "image_b64", None)
            self.rerun_viz.log_frame(
                frame_idx=self.frame_idx,
                timestamp=source_video_time,
                image_b64=image_b64,
                objects=active_objects,
                relations=relations,
                events=events,
                sam3d_meshes=self._sam3d_meshes,
            )

        logger.debug(f"[WorldRuntime] step_once complete. Active objects: {len(world_state.objects)}, Relations: {len(world_state.relations)}")
        return world_state

    def _save_snapshot(self, obj: ObjectNode) -> str:
        self._snapshot_counter += 1
        snapshot_id = f"snap_{self._snapshot_counter:06d}"
        self._hypothesis_snapshots[snapshot_id] = {
            "object_id": obj.object_id,
            "object": obj.model_dump(),
            "missed_frames": self.object_missed_frames.get(obj.object_id, 0),
        }
        self._snapshot_order_by_object[obj.object_id].append(snapshot_id)
        return snapshot_id

    def _latest_snapshot_id_for_object(self, object_id: str) -> str | None:
        ordered = self._snapshot_order_by_object.get(object_id, [])
        if not ordered:
            return None
        return ordered[-1]

    def _sync_world(self, source_time: float) -> None:
        active_objects = list(self.object_registry.values())
        relation_objects = [o for o in active_objects if o.state.visibility == "visible"]
        relations = self.relation_engine.infer(
            relation_objects,
            self.frame_idx,
            source_time,
            background_geometry=self._background_geometry(),
        )
        self._append_relation_tracks(relations, source_time)
        sim_bundle = self.simulation_pipeline.sync(active_objects)
        self.last_sync_report = sim_bundle.sync_report
        self.store.upsert_objects(active_objects, source_time)
        self.store.replace_relations(relations, source_time)

    def export_object_videos(
        self,
        output_dir: str | Path | None = None,
        fps: float = 30.0,
    ) -> dict[str, str]:
        """Export one masked video clip per movable object tracked during the run.

        Args:
            output_dir: Where to write the ``.mp4`` files.
                        Defaults to ``{session_output_root}/object_videos``.
            fps: Output frame rate (should match the source video).

        Returns:
            ``{object_id: path}`` for each successfully written video.
        """
        out = Path(output_dir) if output_dir else self.session_output_root / "object_videos"
        return self.object_video_extractor.export(
            movable_registry=self.movable_registry,
            output_dir=out,
            fps=fps,
        )

    def export_pit2isaac(self, mode_override: str | None = None) -> dict:
        logger.debug("[WorldRuntime] Starting PIT2Isaac export...")
        objects = self.store.get_objects()
        relations = self.store.get_relations()
        pit_snapshot = self.build_pit_snapshot()
        
        result = self.simulation_pipeline.export(
            objects=objects,
            relations=relations,
            pit_snapshot=pit_snapshot,
            mode_override=mode_override,
        )
        logger.debug(f"[WorldRuntime] PIT2Isaac export completed: {result}")
        return result

    def build_pit_snapshot(self, raw: bool = False) -> dict:
        if not raw and self._aligned_pit_snapshot is not None:
            return copy.deepcopy(self._aligned_pit_snapshot)
        pit_snapshot = self.estimator.pit_snapshot()
        pit_snapshot["relation_trajectories"] = dict(self._relation_trajectories)
        pit_snapshot["sam3d_meshes"] = dict(self._sam3d_meshes)
        pit_snapshot["vlm_priors"] = dict(self._vlm_priors)
        if self.background_pipeline is not None and self.background_pipeline.result:
            pit_snapshot["background_reconstruction"] = self.background_pipeline.result
            depth_maps_dir = self.background_pipeline.result.get("depth_maps_dir")
            if depth_maps_dir:
                pit_snapshot["wildgs_depth_maps_dir"] = depth_maps_dir
        return pit_snapshot

    def _background_geometry(self) -> dict | None:
        if self.background_pipeline is None or not self.background_pipeline.result:
            return None
        points_path = str(self.background_pipeline.result.get("points_path", "")).strip()
        if not points_path:
            return None
        return {"ply_path": points_path}

    def _append_relation_tracks(self, relations: list[RelationEdge], timestamp: float) -> None:
        object_map = {obj.object_id: obj for obj in self.object_registry.values()}
        for rel in relations:
            s = object_map.get(rel.subject_id)
            o = object_map.get(rel.object_id)
            distance = None
            if s is not None and o is not None:
                sp = _valid_vec3(s.geometry.pose_3d.position)
                op = _valid_vec3(o.geometry.pose_3d.position)
                if sp is not None and op is not None:
                    sx, sy, sz = sp
                    ox, oy, oz = op
                    dx, dy, dz = sx - ox, sy - oy, sz - oz
                    distance = float((dx * dx + dy * dy + dz * dz) ** 0.5)
            key = f"{rel.subject_id}|{rel.predicate}|{rel.object_id}"
            self._relation_trajectories[key].append(
                {
                    "timestamp_sec": float(timestamp),
                    "distance": distance,
                    "relation_world": rel.predicate,
                    "relation_camera": rel.predicate,
                    "confidence": float(rel.confidence),
                }
            )

    def apply_hypothesis(self, edit: dict) -> tuple[ObjectNode, Event, str]:
        object_id = edit.get("object_id")
        if not object_id:
            raise ValueError("edit.object_id is required")

        obj = self.object_registry.get(object_id) or self.store.get_object(object_id)
        if obj is None:
            raise KeyError(object_id)
        snapshot_id = self._save_snapshot(obj)

        pose_3d = edit.get("pose_3d")
        if isinstance(pose_3d, dict):
            if "position" in pose_3d:
                obj.geometry.pose_3d.position = pose_3d["position"]
            if "orientation_quat" in pose_3d:
                obj.geometry.pose_3d.orientation_quat = pose_3d["orientation_quat"]
            if "frame" in pose_3d:
                obj.geometry.pose_3d.frame = pose_3d["frame"]

        velocity_linear = edit.get("velocity_linear")
        if velocity_linear is not None:
            obj.physics.velocity_linear = velocity_linear

        visibility = edit.get("visibility")
        if visibility in {"visible", "occluded", "lost"}:
            obj.state.visibility = visibility

        interaction_state = edit.get("interaction_state")
        if interaction_state in {"idle", "held", "moving", "contact"}:
            obj.state.interaction_state = interaction_state

        obj.state.last_seen_ts = self.time_sync.tick(0.0, 0.0)[2]
        self.object_registry[object_id] = obj
        self.object_missed_frames[object_id] = 0

        self._sync_world(obj.state.last_seen_ts)

        self._hypothesis_counter += 1
        evt = Event(
            event_id=f"evt_hyp_{self._hypothesis_counter:06d}",
            type="entered_region",
            timestamp=obj.state.last_seen_ts,
            actors=[object_id],
            targets=[],
            payload={"source": "hypothesis", "edit": edit},
            confidence=0.7,
        )
        self.store.append_events([evt])
        return obj, evt, snapshot_id

    def revert_hypothesis(self, snapshot_id: str | None = None, object_id: str | None = None) -> tuple[ObjectNode, Event, str]:
        resolved_snapshot_id = snapshot_id
        if resolved_snapshot_id is None:
            if not object_id:
                raise ValueError("snapshot_id or object_id is required")
            resolved_snapshot_id = self._latest_snapshot_id_for_object(object_id)
            if resolved_snapshot_id is None:
                raise KeyError(object_id)

        snapshot = self._hypothesis_snapshots.get(resolved_snapshot_id)
        if snapshot is None:
            raise KeyError(resolved_snapshot_id)

        restored = ObjectNode.model_validate(snapshot["object"])
        self.object_registry[restored.object_id] = restored
        self.object_missed_frames[restored.object_id] = int(snapshot.get("missed_frames", 0))
        restore_time = self.time_sync.tick(0.0, 0.0)[2]
        restored.state.last_seen_ts = restore_time

        self._sync_world(restore_time)
        self._hypothesis_counter += 1
        evt = Event(
            event_id=f"evt_hyp_{self._hypothesis_counter:06d}",
            type="entered_region",
            timestamp=restore_time,
            actors=[restored.object_id],
            targets=[],
            payload={"source": "hypothesis_revert", "snapshot_id": resolved_snapshot_id},
            confidence=0.75,
        )
        self.store.append_events([evt])
        return restored, evt, resolved_snapshot_id
