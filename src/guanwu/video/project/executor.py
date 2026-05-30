import logging as _logging
_logger = _logging.getLogger(__name__)

import base64
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from guanwu.core.config import WorkspaceConfig
from guanwu.storage.catalog import Catalog
from guanwu.video.clients.zaiwu import (
    build_zaiwu_gateway_client,
    build_zaiwu_object_detector,
    build_zaiwu_sam3d_adapter,
    build_zaiwu_wildgs_adapter,
    normalize_provider_mode,
)
from guanwu.video.core.instance_matching import deduplicate_instances
from guanwu.video.core.schema import Event, ObjectNode, RelationEdge, WorldState
from guanwu.video.core.types import DetectedInstance, FrameDetections
from guanwu.video.features.spatial.alignment_utils import (
    build_depth_point_cloud,
    compute_axis_scale,
    trim_point_cloud_outliers,
)
from guanwu.video.features.spatial.road_geometry import (
    estimate_road_geometry,
    intersect_camera_ray_with_plane,
    select_road_plane_for_frame,
)
from guanwu.video.features.spatial.scene_background_assets import (
    generate_target_frame_background_assets,
    load_background_asset_meshes,
)
from guanwu.video.features.temporal.trajectory_smoothing import smooth_object_trajectories
from guanwu.video.features.simulation.usd_coordinate_convention import (
    USDCoordinateConvention,
    build_coordinate_report,
    build_world_to_usd_basis,
    convert_cv_camera_pose_to_usd,
    convert_world_normals_to_usd,
    convert_world_points_to_usd,
    convert_world_pose_to_usd,
)
from guanwu.video.infra.storage_sqlite import WorldStore
from guanwu.video.materialize import materialize_video_project
from guanwu.video.project.artifacts import (
    ArtifactRecord,
    LEGACY_STAGE_ALIASES,
    PHASE_MAP,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    StageStatus,
    stable_hash,
    utc_now,
)
from guanwu.video.project.config import ProjectConfig, create_project_config, project_config_payload, save_project_config
from guanwu.video.project.context import ProjectContext
from guanwu.video.project.services import PipelineServices, VideoFrameReader, build_services
from guanwu.video.registry import NATURAL_VIDEO_DATASET_ID


_ZAIWU_SAM3D_PER_OBJECT_TIMEOUT_SEC = 300.0
_POSE_OPTIMIZE_MIN_BBOX_AREA_PX = 800.0
_POSE_MATCH_MIN_BBOX_AREA_PX = 800.0
_POSE_TRACK_SCALE_PRIOR_MIN_FRAMES = 2
_EDGE_POSE_HEADING_METRIC_KEYS = (
    "heading_prior_score",
    "heading_prior_angle_error_deg",
    "heading_prior_confidence",
    "heading_planar_score",
    "heading_front_sign_enabled",
    "heading_front_sign_confidence",
    "heading_candidate_forward_sign",
    "heading_semantic_front_sign",
    "heading_tail_light_front_sign",
    "heading_tail_light_flipped",
    "heading_front_sign_hard_rejected",
    "heading_front_angle_penalty",
    "heading_front_sign_penalty",
    "heading_depth_trend_score",
    "heading_depth_trend_direction",
    "heading_depth_trend_confidence",
    "heading_front_depth_cam",
    "heading_prior_projected_vector_image",
    "heading_prior_target_vector_image",
    "effective_heading_prior_weight",
    "effective_front_sign_penalty_weight",
)
_EDGE_POSE_TRUNCATION_METRIC_KEYS = (
    "truncation_severity",
    "low_observability",
    "truncation_observability_score",
    "truncation_observability_reasons",
    "visible_target_fraction",
    "visible_target_area_px",
    "visible_contour_mean_distance_px",
    "visible_profile_mean_distance_px",
    "truncated_visual_quality_gate",
    "truncated_visual_quality_reason",
    "severe_truncation_gate_passed",
    "severe_truncation_gate_reasons",
)
_GENERIC_POSE_METRIC_KEYS = (
    "soft_mask_iou",
    "mask_blend_score",
    "contour_score",
    "edge_score",
    "edge_confidence",
    "depth_score",
    "depth_confidence",
    "depth_error",
    "valid_depth_ratio",
    "appearance_score",
    "appearance_confidence",
    "color_soft_iou",
    "color_precision",
    "color_recall",
    "background_leakage",
    "fg_bg_distance",
    "temporal_score",
    "generic_temporal_loss",
    "scale_prior_score",
    "optional_prior_score",
    "support_plane_confidence",
    "observation_score",
    "optional_prior_gate",
    "projection_valid_ratio",
    "visible_ratio",
    "truncation_ratio",
    "acceptance_status",
    "reject_reasons",
)


class ProjectExecutor:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self._cached_zaiwu_detector = None
        self._cached_zaiwu_sam3d = None

    def _canonical_stage(self, stage: str) -> str:
        return LEGACY_STAGE_ALIASES.get(stage, stage)

    @classmethod
    def init_project(
        cls,
        *,
        video: str,
        out_dir: str,
        provider_mode: str = "mock",
        video_copy_mode: str = "copy",
        workspace: dict | None = None,
        payload: dict | None = None,
    ) -> ProjectContext:
        out_path = Path(out_dir).expanduser().resolve()
        project_id = out_path.name
        config = create_project_config(
            project_id=project_id,
            name=project_id,
            input_video=video,
            root_dir=str(out_path),
            provider_mode=provider_mode,
            video_copy_mode=video_copy_mode,
            workspace=workspace,
            payload=payload,
        )
        context = ProjectContext.create(out_path, config)
        save_project_config(config, context.paths.config)
        source = Path(video).expanduser().resolve()
        if video_copy_mode == "link":
            context.paths.input_video.symlink_to(source)
        else:
            shutil.copy2(source, context.paths.input_video)
        return ProjectContext(out_path)

    def status(self) -> dict:
        statuses = self.context.load_stage_statuses()
        steps = {name: status.model_dump(mode="json") for name, status in statuses.items()}
        return {
            "project": project_config_payload(self.context.config)["project"],
            "steps": steps,
            "stages": steps,
        }

    def inspect(self) -> dict:
        manifest = self.context.load_manifest()
        return {
            "manifest": manifest.model_dump(mode="json"),
            "config": project_config_payload(self.context.config),
            "artifacts": {stage: record.model_dump(mode="json") for stage, record in self.context.artifacts.records.items()},
        }

    def run_stage(self, stage: str, force: bool = False, **kwargs) -> dict:
        stage = self._canonical_stage(stage)
        if stage not in STAGE_ORDER and stage != "validate":
            raise ValueError(f"Unsupported stage: {stage}")
        self.context.acquire_lock()
        try:
            if stage == "validate":
                self.ensure_dependencies(stage, force=False)
                return self.validate()
            self.ensure_dependencies(stage, force=force)
            if force:
                self.invalidate_downstream(stage)
                self._clear_stage_output(stage)
            statuses = self.context.load_stage_statuses()
            current = statuses.get(stage)
            if current and current.status == "completed" and not force:
                record = self.context.artifacts.get(stage)
                return {
                    "status": "cached",
                    "stage": stage,
                    "summary": record.summary if record else {},
                    "outputs": record.outputs if record else {},
                }
            runner = getattr(self, self._runner_name(stage))
            result = runner(**kwargs)
            statuses[stage] = StageStatus(
                stage=stage,
                status="completed",
                last_run_at=utc_now(),
                inputs_hash=result["inputs_hash"],
                params_hash=result["params_hash"],
            )
            self.context.save_stage_statuses(statuses)
            self.context.artifacts.set(
                ArtifactRecord(
                    stage=stage,
                    created_at=utc_now(),
                    inputs_hash=result["inputs_hash"],
                    params_hash=result["params_hash"],
                    outputs=result["outputs"],
                    summary=result["summary"],
                )
            )
            return {"status": "ok", "stage": stage, **result}
        except Exception as exc:
            statuses = self.context.load_stage_statuses()
            statuses[stage] = StageStatus(stage=stage, status="failed", last_run_at=utc_now(), error=str(exc))
            self.context.save_stage_statuses(statuses)
            raise
        finally:
            self.context.release_lock()

    def run_phase(self, phase: str, force: bool = False) -> list[dict]:
        if phase not in PHASE_MAP:
            raise ValueError(f"Unknown phase: {phase!r}. Available: {list(PHASE_MAP)}")
        from_stage, to_stage = PHASE_MAP[phase]
        return self.run_range(from_stage, to_stage, force=force)

    def run_range(self, from_stage: str, to_stage: str, force: bool = False) -> list[dict]:
        from_stage = self._canonical_stage(from_stage)
        to_stage = self._canonical_stage(to_stage)
        start = STAGE_ORDER.index(from_stage)
        end = STAGE_ORDER.index(to_stage)
        if start > end:
            raise ValueError("from_stage must come before to_stage")
        results = []
        for stage in STAGE_ORDER[start : end + 1]:
            results.append(self.run_stage(stage, force=force))
            force = False
        return results

    def ensure_dependencies(self, stage: str, force: bool) -> None:
        stage = self._canonical_stage(stage)
        for dependency in STAGE_DEPENDENCIES.get(stage, []):
            statuses = self.context.load_stage_statuses()
            dep_status = statuses.get(dependency)
            if dep_status is None or dep_status.status != "completed":
                self.run_stage(dependency, force=False)

    def invalidate_downstream(self, stage: str) -> None:
        stage = self._canonical_stage(stage)
        statuses = self.context.load_stage_statuses()
        start = STAGE_ORDER.index(stage)
        downstream = STAGE_ORDER[start + 1 :]
        for name in downstream:
            statuses[name] = StageStatus(stage=name)
        self.context.save_stage_statuses(statuses)
        self.context.artifacts.drop_many(downstream)
        for name in downstream:
            self._clear_stage_output(name)

    def _clear_stage_output(self, stage: str) -> None:
        path = self.context.stage_output_dir(stage)
        if stage == "pose.optimize" and self._pose_resume_results():
            path.mkdir(parents=True, exist_ok=True)
            return
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _services(self) -> PipelineServices:
        return build_services(self.context.config, self.context.paths.root)

    def _workspace_config(self) -> WorkspaceConfig:
        workspace = self.context.config.workspace or {}
        if workspace:
            return WorkspaceConfig.model_validate(workspace)
        return WorkspaceConfig(workspace_root=str(self.context.paths.root))

    def _dataset_id(self) -> str:
        dataset_id = str(self.context.config.payload.get("dataset_id", "") or "").strip()
        return dataset_id or NATURAL_VIDEO_DATASET_ID

    def _provider_mode(self) -> str:
        return normalize_provider_mode(self.context.config.project.provider_mode)

    def _base_result(self, stage: str, summary: dict, outputs: dict, params: dict | None = None) -> dict:
        upstream_hashes = {
            name: (self.context.artifacts.get(name).inputs_hash if self.context.artifacts.get(name) else "")
            for name in STAGE_DEPENDENCIES.get(stage, [])
        }
        inputs_hash = stable_hash({"stage": stage, "project": self.context.config.project.project_id, "upstream": upstream_hashes})
        params_hash = stable_hash(params or {})
        return {
            "summary": summary,
            "outputs": outputs,
            "inputs_hash": inputs_hash,
            "params_hash": params_hash,
        }

    def _runner_name(self, stage: str) -> str:
        return f"_run_{stage.replace('.', '_')}"

    def _json_load(self, path: str | Path) -> dict | list:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _json_dump(self, path: str | Path, payload: dict | list) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out)

    def _get_zaiwu_detector(self):
        if self._cached_zaiwu_detector is None:
            detector = build_zaiwu_object_detector(
                self.context.config.settings,
                video_source=self.context.config.project.input_video,
            )
            detector.set_object_detection_prompts([])
            self._cached_zaiwu_detector = detector
        return self._cached_zaiwu_detector

    def _get_zaiwu_sam3d(self):
        if self._cached_zaiwu_sam3d is None:
            self._cached_zaiwu_sam3d = build_zaiwu_sam3d_adapter(
                self.context.config.settings,
                materialization_root=str(self.context.paths.root),
                materialization_mode=self.context.config.settings.runtime.asset_materialization or "copy",
                per_object_timeout_sec=_ZAIWU_SAM3D_PER_OBJECT_TIMEOUT_SEC,
            )
        return self._cached_zaiwu_sam3d

    def _pose_optimizer_timeout_sec(self) -> float:
        value = getattr(self.context.config.settings.zaiwu, "pose_optimizer_timeout_sec", None)
        try:
            return max(1.0, float(value))
        except (TypeError, ValueError):
            return 1800.0

    def _pose_optimize_min_bbox_area_px(self) -> float:
        value = getattr(self.context.config.settings.zaiwu, "pose_optimize_min_bbox_area_px", None)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return _POSE_OPTIMIZE_MIN_BBOX_AREA_PX

    @staticmethod
    def _bbox_area_px(bbox: object) -> float:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return 0.0
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _detect_objects_in_frame_mock(self, frame, frame_idx: int, timestamp: float) -> FrameDetections:  # noqa: ANN001
        image_b64 = VideoFrameReader.encode_jpg(frame)
        height, width = frame.shape[:2]
        prompts = ["object"]
        instances: list[DetectedInstance] = []
        box_w = max(width // 5, 32)
        box_h = max(height // 5, 32)
        for idx, prompt in enumerate(prompts):
            x1 = int((idx * 73 + frame_idx * 7) % max(width - box_w, 1))
            y1 = int((idx * 47 + frame_idx * 5) % max(height - box_h, 1))
            bbox = [float(x1), float(y1), float(x1 + box_w), float(y1 + box_h)]
            label = prompt.strip().lower() or f"object_{idx+1}"
            segment_kind = "body" if any(term in label for term in ("person", "human", "body")) else "object"
            instances.append(
                DetectedInstance(
                    mask_ref=f"mask://frame_{frame_idx:05d}/obj_{idx+1:06d}",
                    bbox=bbox,
                    object_id=f"obj_{idx+1:06d}",
                    concept_label=label,
                    segment_kind=segment_kind,
                    score=max(0.5, 0.95 - idx * 0.08),
                )
            )
        return FrameDetections(frame_idx=frame_idx, timestamp=timestamp, instances=instances, image_b64=image_b64)

    def _detect_objects_in_frame_zaiwu(self, frame_idx: int, timestamp: float) -> FrameDetections:
        detector = self._get_zaiwu_detector()
        if hasattr(detector, "prefetch_video"):
            detector.prefetch_video()
        return detector.detect_objects_in_frame(frame_idx=frame_idx, timestamp=timestamp)

    def _detect_objects_in_frame(self, frame, frame_idx: int, timestamp: float) -> FrameDetections:  # noqa: ANN001
        mode = self._provider_mode()
        if mode == "mock":
            return self._detect_objects_in_frame_mock(frame, frame_idx, timestamp)
        if mode == "zaiwu":
            return self._detect_objects_in_frame_zaiwu(frame_idx, timestamp)
        return self._detect_objects_in_frame_mock(frame, frame_idx, timestamp)

    def _mock_meshes(self, objects: list[ObjectNode]) -> dict[str, dict]:
        return {
            obj.object_id: {
                "instance_id": obj.object_id,
                "segment_kind": obj.segment_kind,
                "quality": 0.5,
                "mesh_path": "",
                "source": "mock",
            }
            for obj in objects
        }

    def _find_best_frame_per_object(
        self,
        object_ids: set[str],
    ) -> dict[str, tuple[FrameDetections, DetectedInstance]]:
        """Return the frame with highest visibility score per object.

        Score favors confident, large-enough, centered, non-truncated object views.
        Iterates all geometry.lift frames and picks the best for each object.
        """
        geometry = self.context.artifacts.get("geometry.lift")
        if geometry is None:
            raise RuntimeError("geometry.lift outputs are required")
        summary = self._json_load(geometry.outputs["summary"])

        best: dict[str, tuple[tuple[int, float], FrameDetections, DetectedInstance]] = {}
        for entry in summary.get("frames", []):
            det_path = entry.get("detections")
            if not det_path:
                continue
            try:
                detections = FrameDetections.model_validate(self._json_load(det_path))
            except Exception as exc:
                _logger.warning(f"[mesh.reconstruct] Could not load detections {det_path}: {exc}")
                continue
            for inst in detections.instances:
                obj_id = inst.object_id
                if obj_id not in object_ids:
                    continue
                score_info = self._mesh_frame_selection_score(inst)
                score = (0 if score_info["truncated"] else 1, float(score_info["score"]))
                prev = best.get(obj_id)
                if prev is None or score > prev[0]:
                    best[obj_id] = (score, detections, inst)

        return {oid: (fd, inst) for oid, (_, fd, inst) in best.items()}

    @staticmethod
    def _mesh_frame_image_size(inst: DetectedInstance | dict) -> tuple[float | None, float | None]:
        def get_value(key: str) -> object:
            if isinstance(inst, dict):
                return inst.get(key)
            return getattr(inst, key, None)

        width = get_value("image_width") or get_value("width")
        height = get_value("image_height") or get_value("height")
        if width is not None and height is not None:
            try:
                return float(width), float(height)
            except (TypeError, ValueError):
                return None, None

        mask_rle = get_value("mask_rle")
        if mask_rle is None:
            return None, None
        try:
            rle = json.loads(mask_rle) if isinstance(mask_rle, str) else mask_rle
        except Exception:
            return None, None
        if not isinstance(rle, dict):
            return None, None
        size = rle.get("size")
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            return None, None
        try:
            return float(size[1]), float(size[0])
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _mesh_frame_selection_score(inst: DetectedInstance | dict) -> dict[str, float | bool]:
        def get_value(key: str) -> object:
            if isinstance(inst, dict):
                return inst.get(key)
            return getattr(inst, key, None)

        bbox = get_value("bbox_xyxy") or get_value("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return {
                "score": 0.0,
                "truncated": True,
                "area_px": 0.0,
                "edge_margin_px": 0.0,
                "edge_score": 0.1,
                "center_score": 0.1,
                "size_score": 0.0,
            }
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return {
                "score": 0.0,
                "truncated": True,
                "area_px": 0.0,
                "edge_margin_px": 0.0,
                "edge_score": 0.1,
                "center_score": 0.1,
                "size_score": 0.0,
            }

        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        area = width * height
        if area <= 0.0:
            return {
                "score": 0.0,
                "truncated": True,
                "area_px": 0.0,
                "edge_margin_px": 0.0,
                "edge_score": 0.1,
                "center_score": 0.1,
                "size_score": 0.0,
            }

        image_width, image_height = ProjectExecutor._mesh_frame_image_size(inst)
        edge_margin = min(x1, y1)
        right_gap = bottom_gap = None
        if image_width is not None:
            right_gap = image_width - 1.0 - x2
            edge_margin = min(edge_margin, right_gap)
        if image_height is not None:
            bottom_gap = image_height - 1.0 - y2
            edge_margin = min(edge_margin, bottom_gap)
        edge_margin = max(float(edge_margin), 0.0)

        explicit_truncated = bool(get_value("is_truncated") or get_value("truncated"))
        truncation_info = get_value("truncation_info")
        if isinstance(truncation_info, dict):
            explicit_truncated = explicit_truncated or bool(
                truncation_info.get("is_truncated") or truncation_info.get("touches_image_border")
            )
        touches_border = (
            x1 <= 1.0
            or y1 <= 1.0
            or (right_gap is not None and right_gap <= 1.0)
            or (bottom_gap is not None and bottom_gap <= 1.0)
        )
        truncated = bool(explicit_truncated or touches_border)

        size_score = min(area / 12000.0, 1.0)
        if area < 2500.0:
            size_score *= max(area / 2500.0, 0.1)

        edge_score = max(0.1, min(edge_margin / 24.0, 1.0))
        if truncated:
            edge_score *= 0.2

        center_score = 1.0
        if image_width is not None and image_height is not None and image_width > 0.0 and image_height > 0.0:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            dx = abs(cx - 0.5 * image_width) / max(0.5 * image_width, 1.0)
            dy = abs(cy - 0.5 * image_height) / max(0.5 * image_height, 1.0)
            center_score = max(0.25, 1.0 - 0.5 * (dx + dy))

        confidence_score = max(float(get_value("score") or 0.0), 0.05)
        score = confidence_score * size_score * edge_score * center_score
        return {
            "score": float(score),
            "truncated": truncated,
            "area_px": float(area),
            "edge_margin_px": float(edge_margin),
            "edge_score": float(edge_score),
            "center_score": float(center_score),
            "size_score": float(size_score),
        }

    def _reconstruct_object_meshes(
        self,
        best_frames: dict[str, tuple[FrameDetections, DetectedInstance]],
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        mode = self._provider_mode()
        if mode == "mock":
            return self._mock_meshes(objects)
        if mode == "zaiwu":
            return self._get_zaiwu_sam3d().reconstruct_object_meshes(best_frames, objects)
        return self._mock_meshes(objects)

    def _assert_zaiwu_service_ready(self, service_id: str, *, stage: str) -> None:
        gateway = build_zaiwu_gateway_client(self.context.config.settings)
        try:
            endpoint = gateway.get_ready_service(service_id)
        except Exception as exc:
            raise RuntimeError(
                f"{stage} could not verify Zaiwu service {service_id}: {exc}"
            ) from exc
        if endpoint is None:
            raise RuntimeError(
                f"{stage} requires Zaiwu service {service_id} to already be running, "
                f"but no ready worker was found via {gateway.gateway_url}"
            )

    def _mesh_reconstruction_priority(
        self,
        obj: ObjectNode,
        frame_data: tuple[FrameDetections, DetectedInstance],
    ) -> float:
        _ = obj
        _, inst = frame_data
        return float(self._mesh_frame_selection_score(inst)["score"])

    def _select_zaiwu_mesh_candidates(
        self,
        objects: list[ObjectNode],
        best_frames: dict[str, tuple[FrameDetections, DetectedInstance]],
    ) -> tuple[list[ObjectNode], dict[str, int]]:
        ranked: list[tuple[float, ObjectNode]] = []
        missing_best_frame_count = 0
        for obj in objects:
            frame_data = best_frames.get(obj.object_id)
            if frame_data is None:
                missing_best_frame_count += 1
                continue
            ranked.append((self._mesh_reconstruction_priority(obj, frame_data), obj))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [obj for _, obj in ranked]
        return selected, {
            "candidate_count": len(ranked),
            "selected_count": len(selected),
            "missing_best_frame_count": missing_best_frame_count,
            "skipped_unselected_count": 0,
        }

    def _run_zaiwu_mesh_reconstruct(
        self,
        objects_to_reconstruct: list[ObjectNode],
        best_frames: dict[str, tuple[FrameDetections, DetectedInstance]],
        meshes_path: Path,
    ) -> tuple[dict[str, dict], dict]:
        adapter = self._get_zaiwu_sam3d()
        selected, selection_stats = self._select_zaiwu_mesh_candidates(objects_to_reconstruct, best_frames)
        total_selected = len(selected)

        meshes: dict[str, dict] = {}
        attempted_count = 0
        failed_count = 0

        for idx, obj in enumerate(selected, start=1):
            frame_data = best_frames.get(obj.object_id)
            if frame_data is None:
                continue

            attempted_count += 1
            start = time.perf_counter()
            _logger.info(
                "[mesh.reconstruct] %d/%d start %s label=%s frame=%s",
                idx,
                total_selected,
                obj.object_id,
                obj.label,
                frame_data[0].frame_idx,
            )
            result = adapter.reconstruct_object_meshes(
                {obj.object_id: frame_data},
                [obj],
            )
            elapsed = time.perf_counter() - start
            if obj.object_id in result:
                meshes[obj.object_id] = result[obj.object_id]
                self._json_dump(meshes_path, meshes)
                _logger.info(
                    "[mesh.reconstruct] %d/%d done %s in %.2fs",
                    idx,
                    total_selected,
                    obj.object_id,
                    elapsed,
                )
            else:
                failed_count += 1
                _logger.warning(
                    "[mesh.reconstruct] %d/%d no mesh for %s after %.2fs",
                    idx,
                    total_selected,
                    obj.object_id,
                    elapsed,
                )

        skipped_selected_count = max(0, total_selected - attempted_count)
        skipped_count = (
            selection_stats["missing_best_frame_count"]
            + selection_stats["skipped_unselected_count"]
            + skipped_selected_count
        )
        stats: dict[str, int | str] = {
            **selection_stats,
            "attempted_count": attempted_count,
            "failed_count": failed_count,
            "skipped_selected_count": skipped_selected_count,
            "skipped_count": skipped_count,
        }
        return meshes, stats

    def _infer_object_physics_priors(
        self,
        detections: FrameDetections,
        objects: list[ObjectNode],
    ) -> dict[str, dict]:
        mode = self._provider_mode()
        if mode in ("mock", "zaiwu"):
            if mode == "zaiwu":
                from guanwu.video.features.world_inference.object_attr import ObjectAttrAgent
                from guanwu.video.core.config import VLMConfig
                vlm_cfg = self.context.config.settings.vlm
                agent = ObjectAttrAgent(VLMConfig(
                    api_key=vlm_cfg.api_key,
                    base_url=vlm_cfg.base_url,
                    model=vlm_cfg.model,
                    max_retries=vlm_cfg.max_retries,
                ))
                return agent.infer_object_physics_priors(detections, objects)
            priors: dict[str, dict[str, object]] = {}
            for obj in objects:
                label = str(obj.label).lower()
                movable = label not in {"table", "wall", "floor", "ceiling", "road"}
                priors[obj.object_id] = {
                    "is_movable": movable,
                    "is_rigid_body": movable,
                    "material": "rigid" if movable else "static",
                }
            return priors
        return {}

    def _latest_frame_detections(self) -> FrameDetections:
        detect = self.context.artifacts.get("object.detect")
        if detect is None:
            raise RuntimeError("object.detect outputs are required")
        summary = self._json_load(detect.outputs["summary"])
        latest_detections_path = summary["latest_detections"]
        return FrameDetections.model_validate(self._json_load(latest_detections_path))

    def _all_objects(self) -> list[ObjectNode]:
        geometry = self.context.artifacts.get("geometry.lift")
        if geometry is None:
            raise RuntimeError("geometry.lift outputs are required")
        summary = self._json_load(geometry.outputs["summary"])

        # Collect best observation per object across ALL frames so that objects
        # whose bbox degrades in late frames are still included.
        best: dict[str, tuple[float, dict]] = {}
        for entry in summary.get("frames", []):
            obs_path = entry.get("observed_objects")
            if not obs_path:
                continue
            try:
                obs_list = self._json_load(obs_path)
            except Exception:
                continue
            for obj_d in obs_list:
                oid = obj_d.get("object_id", "")
                score = obj_d.get("confidence", 0.0) or 0.0
                prev = best.get(oid)
                if prev is None or score > prev[0]:
                    best[oid] = (score, obj_d)

        if best:
            return [ObjectNode.model_validate(d) for _, d in best.values()]

        # Fallback: use summary's latest_objects (legacy / already aggregated)
        return [ObjectNode.model_validate(obj) for obj in summary.get("latest_objects", [])]

    def _run_video_inspect(self) -> dict:
        out_dir = self.context.stage_output_dir("video.inspect")
        reader = VideoFrameReader(self.context.paths.input_video)
        metadata = reader.metadata()
        metadata["resolved_input_video"] = str(self.context.paths.input_video)
        metadata_path = out_dir / "video_metadata.json"
        self._json_dump(metadata_path, metadata)
        summary = {
            "frame_count": metadata["frame_count"],
            "fps": metadata["fps"],
            "resolution": [metadata["width"], metadata["height"]],
        }
        outputs = {"video_metadata": str(metadata_path)}
        return self._base_result("video.inspect", summary, outputs)

    def _run_frame_sample(self) -> dict:
        out_dir = self.context.stage_output_dir("frame.sample")
        reader = VideoFrameReader(self.context.paths.input_video)
        frames = reader.iter_frames()
        first_frame_path = out_dir / "first_frame.jpg"
        if frames:
            cv2.imwrite(str(first_frame_path), frames[0][2])
        frame_index = [{"frame_idx": frame_idx, "timestamp": timestamp} for frame_idx, timestamp, _ in frames]
        sample_index_path = out_dir / "frame_index.json"
        self._json_dump(sample_index_path, frame_index)
        summary = {"frame_count": len(frame_index), "sampled_frames": min(10, len(frame_index))}
        outputs = {"frame_index": str(sample_index_path), "first_frame": str(first_frame_path)}
        return self._base_result("frame.sample", summary, outputs)

    def _run_object_detect(self) -> dict:
        out_dir = self.context.stage_output_dir("object.detect")
        frames_root = out_dir / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)
        services = self._services()

        mode = self._provider_mode()
        vlm_cfg = self.context.config.settings.runtime.vlm_discovery
        vlm_agent = None
        keyframe_detector = None
        discovered_prompts: list[str] = []

        if mode == "zaiwu" and vlm_cfg.enabled:
            from guanwu.video.features.detection.vlm_discovery import VLMDiscoveryAgent
            from guanwu.video.features.detection.keyframe_detector import KeyframeDetector
            vlm_agent = VLMDiscoveryAgent(self.context.config.settings.vlm)
            keyframe_detector = KeyframeDetector(
                periodic_interval=vlm_cfg.periodic_interval,
                new_track_threshold=vlm_cfg.new_track_threshold,
                disappear_threshold=vlm_cfg.disappear_threshold,
                confidence_drop_threshold=vlm_cfg.confidence_drop_threshold,
                image_change_threshold=vlm_cfg.image_change_threshold,
                image_change_enabled=vlm_cfg.image_change_enabled,
            )
            # 第一帧全量发现
            frame_sample = self.context.artifacts.get("frame.sample")
            first_frame_path = frame_sample.outputs.get("first_frame") if frame_sample else None
            if first_frame_path and Path(first_frame_path).exists():
                import base64 as _b64
                image_b64 = _b64.b64encode(Path(first_frame_path).read_bytes()).decode("ascii")
                discovered_prompts = vlm_agent.discover_objects(image_b64)
                if discovered_prompts:
                    self._get_zaiwu_detector().set_object_detection_prompts(discovered_prompts)
                    _logger.info("object.detect: VLM discovered prompts: %s", discovered_prompts)
                else:
                    _logger.warning(
                        "object.detect: VLM discovery returned no prompts; "
                        "detector will fall back to its backend default prompt."
                    )

        frame_index: list[dict] = []
        latest_detections_path = ""
        latest_instances = 0
        prev_object_ids: set[str] = set()
        # Canonical ID mapping: raw tracker id → obj_NNNNNN
        _track_to_obj: dict[str, str] = {}
        _next_obj_seq = 1

        def _canonicalize_id(raw_id: str) -> str:
            """Map any raw tracker id to a stable obj_NNNNNN id."""
            nonlocal _next_obj_seq
            if raw_id in _track_to_obj:
                return _track_to_obj[raw_id]
            # Try to extract a numeric suffix to keep numbering intuitive
            digits = "".join(ch for ch in raw_id.split("_")[-1] if ch.isdigit())
            if digits:
                obj_id = f"obj_{digits.zfill(6)}"
            else:
                obj_id = f"obj_{_next_obj_seq:06d}"
            # Avoid collisions
            existing_ids = set(_track_to_obj.values())
            while obj_id in existing_ids:
                _next_obj_seq += 1
                obj_id = f"obj_{_next_obj_seq:06d}"
            _track_to_obj[raw_id] = obj_id
            _next_obj_seq = max(_next_obj_seq, int(obj_id.split("_")[1]) + 1)
            return obj_id

        for frame_idx, timestamp, frame in services.frames:
            detections = self._detect_objects_in_frame(frame, frame_idx, timestamp)
            deduplicated_instances = deduplicate_instances(detections.instances)
            if len(deduplicated_instances) != len(detections.instances):
                _logger.info(
                    "object.detect: deduplicated frame %s instances %s -> %s",
                    frame_idx,
                    len(detections.instances),
                    len(deduplicated_instances),
                )
                detections = detections.model_copy(update={"instances": deduplicated_instances})

            # Normalize all detection IDs to obj_NNNNNN format so every downstream
            # stage uses one canonical ID scheme.
            for inst in detections.instances:
                inst.object_id = _canonicalize_id(inst.object_id)

            # 增量 VLM discovery：由 KeyframeDetector 决定触发时机
            if vlm_agent is not None:
                current_object_ids = {inst.object_id for inst in detections.instances}
                removed_ids = sorted(prev_object_ids - current_object_ids)
                prev_object_ids = current_object_ids
                if keyframe_detector.check(frame_idx, detections, [], removed_ids):
                    new_cats = vlm_agent.discover_incremental(
                        VideoFrameReader.encode_jpg(frame), discovered_prompts
                    )
                    if new_cats:
                        discovered_prompts = discovered_prompts + new_cats
                        self._get_zaiwu_detector().set_object_detection_prompts(discovered_prompts)
            frame_dir = frames_root / f"frame_{frame_idx:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            detections_path = frame_dir / "detections.json"
            self._json_dump(detections_path, detections.model_dump(mode="json"))
            latest_detections_path = str(detections_path)
            latest_instances = len(detections.instances)
            overlay_path = frame_dir / "overlay.jpg"
            self._write_detection_overlay(overlay_path, detections)
            frame_index.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "detections": str(detections_path),
                    "overlay": str(overlay_path),
                    "instance_count": len(detections.instances),
                }
            )
        self._repair_seg2track_warmup_gap(frame_index)
        if frame_index:
            latest_entry = frame_index[-1]
            latest_detections_path = str(latest_entry["detections"])
            latest_instances = int(latest_entry["instance_count"])
        summary_path = out_dir / "summary.json"
        summary_payload = {
            "frames": frame_index,
            "latest_detections": latest_detections_path,
            "latest_instance_count": latest_instances,
        }
        self._json_dump(summary_path, summary_payload)
        outputs = {"summary": str(summary_path), "frames_dir": str(frames_root)}
        summary = {"frame_count": len(frame_index), "latest_instance_count": latest_instances}
        return self._base_result("object.detect", summary, outputs)

    def _write_detection_overlay(self, overlay_path: str | Path, detections: FrameDetections) -> None:
        if not detections.image_b64:
            return
        jpg = cv2.imdecode(np_from_b64(detections.image_b64), cv2.IMREAD_COLOR)
        if jpg is None:
            return
        overlay = jpg.copy()
        for inst in detections.instances:
            x1, y1, x2, y2 = [int(v) for v in inst.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 170, 255), 2)
            cv2.putText(
                overlay,
                inst.concept_label,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (70, 170, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlay_path), overlay)

    def _repair_seg2track_warmup_gap(self, frame_index: list[dict]) -> None:
        if self._provider_mode() != "zaiwu":
            return
        backend = str(self.context.config.settings.zaiwu.object_detection_backend or "").strip().lower()
        if backend != "seg2track_sam2":
            return
        if len(frame_index) < 3:
            return

        warmup_window = min(
            len(frame_index),
            max(3, int(self.context.config.settings.zaiwu.seg2track_sam2.detect_interval or 0)),
        )
        warmup_entries = frame_index[:warmup_window]
        warmup_detections = [
            (entry, FrameDetections.model_validate(self._json_load(entry["detections"])))
            for entry in warmup_entries
        ]

        first_entry, first_detections = warmup_detections[0]
        tail = warmup_detections[1:]
        if not tail:
            return

        max_count = max(len(detections.instances) for _, detections in tail)
        if max_count < 3:
            return
        if len(first_detections.instances) >= max(2, int(math.floor(max_count * 0.5))):
            return

        anchor_candidates = [
            (entry, detections)
            for entry, detections in tail
            if len(detections.instances) >= max(3, int(math.ceil(max_count * 0.9)))
        ]
        if len(anchor_candidates) < 2:
            return

        stable_ids = {
            inst.object_id
            for inst in anchor_candidates[0][1].instances
        }
        for _, detections in anchor_candidates[1:]:
            stable_ids &= {inst.object_id for inst in detections.instances}

        if len(stable_ids) < max(3, int(math.floor(max_count * 0.7))):
            return

        donor_by_id: dict[str, DetectedInstance] = {}
        for _, detections in anchor_candidates:
            for inst in detections.instances:
                if inst.object_id in stable_ids and inst.object_id not in donor_by_id:
                    donor_by_id[inst.object_id] = inst.model_copy(deep=True)

        first_ids = {inst.object_id for inst in first_detections.instances}
        missing_ids = [object_id for object_id in sorted(stable_ids) if object_id not in first_ids]
        if not missing_ids:
            return

        repaired_instances = [inst.model_copy(deep=True) for inst in first_detections.instances]
        for object_id in missing_ids:
            donor = donor_by_id.get(object_id)
            if donor is not None:
                repaired_instances.append(donor.model_copy(deep=True))
        repaired_instances = deduplicate_instances(repaired_instances)
        if len(repaired_instances) <= len(first_detections.instances):
            return

        repaired = first_detections.model_copy(update={"instances": repaired_instances})
        self._json_dump(first_entry["detections"], repaired.model_dump(mode="json"))
        self._write_detection_overlay(first_entry["overlay"], repaired)
        first_entry["instance_count"] = len(repaired.instances)
        _logger.info(
            "object.detect: repaired seg2track warmup gap on frame %s using %s stable ids (%s -> %s)",
            repaired.frame_idx,
            len(stable_ids),
            len(first_detections.instances),
            len(repaired.instances),
        )

    def _run_object_index(self) -> dict:
        detect = self.context.artifacts.get("object.detect")
        if detect is None:
            raise RuntimeError("object.detect outputs are required")
        out_dir = self.context.stage_output_dir("object.index")
        summary = self._json_load(detect.outputs["summary"])
        object_frames: list[dict] = []
        objects: dict[str, dict] = {}
        for entry in summary["frames"]:
            detections = FrameDetections.model_validate(self._json_load(entry["detections"]))
            per_frame = []
            for inst in detections.instances:
                objects.setdefault(
                    inst.object_id,
                    {"object_id": inst.object_id, "label": inst.concept_label, "segment_kind": inst.segment_kind, "frames": []},
                )
                point = {"frame_idx": detections.frame_idx, "timestamp": detections.timestamp, "bbox": inst.bbox}
                objects[inst.object_id]["frames"].append(point)
                per_frame.append({"object_id": inst.object_id, "label": inst.concept_label, "bbox": inst.bbox})
            object_frames.append({"frame_idx": detections.frame_idx, "timestamp": detections.timestamp, "objects": per_frame})
        objects_path = out_dir / "objects.json"
        frames_path = out_dir / "object_frames.json"
        self._json_dump(objects_path, list(objects.values()))
        self._json_dump(frames_path, object_frames)
        outputs = {"objects": str(objects_path), "object_frames": str(frames_path)}
        summary_payload = {"object_count": len(objects), "frame_count": len(object_frames)}
        return self._base_result("object.index", summary_payload, outputs)

    def _run_geometry_lift(self) -> dict:
        detect = self.context.artifacts.get("object.detect")
        if detect is None:
            raise RuntimeError("object.detect outputs are required")
        out_dir = self.context.stage_output_dir("geometry.lift")
        frames_root = out_dir / "frames"
        frames_root.mkdir(parents=True, exist_ok=True)
        services = self._services()

        # Run WildGS-SLAM via Zaiwu-managed service if camera_provider is "wildgs"
        wildgs_outputs: dict = {}
        pit_cfg = self.context.config.settings.pit
        if pit_cfg.camera_provider == "wildgs" and not pit_cfg.wildgs_camera_poses_jsonl:
            if self._provider_mode() == "zaiwu":
                slam_root = out_dir / "wildgs"
                adapter = build_zaiwu_wildgs_adapter(
                    self.context.config.settings,
                    output_root=str(slam_root),
                )
                slam_result = adapter.run_slam(
                    video_path=self.context.config.project.input_video,
                    export_depth_every_frame=True,
                    depth_export_stride=1,
                    pose_export_stride=1,
                    extract_every_input_frame=True,
                    frame_stride=1,
                )
                if pit_cfg.depth_provider == "wildgs" and not slam_result.get("depth_maps_dir"):
                    raise RuntimeError(
                        "WildGS-SLAM did not return depth_maps_dir while pit.depth_provider='wildgs'"
                    )
                wildgs_outputs = slam_result
                if slam_result.get("camera_poses_path"):
                    services.estimator.attach_wildgs_results(
                        slam_result["camera_poses_path"],
                        static_map_dir=slam_result.get("static_map_dir"),
                        dynamic_prior_dir=slam_result.get("dynamic_prior_dir"),
                        depth_maps_dir=slam_result.get("depth_maps_dir"),
                    )
                if slam_result.get("static_map_file_id"):
                    try:
                        mesh_result = adapter.reconstruct_background_mesh(
                            slam_result["static_map_file_id"]
                        )
                        wildgs_outputs = {**wildgs_outputs, **{"mesh_result": mesh_result}}
                    except Exception as exc:
                        print(f"[geometry.lift] Background mesh reconstruction failed (non-fatal): {exc}")

        summary = self._json_load(detect.outputs["summary"])
        frame_entries: list[dict] = []
        latest_snapshot: dict = {}
        # Track the best observation per object across ALL frames (not just the
        # last one), so objects that leave the view or whose bbox degrades are
        # still included in the final latest_objects list.
        _best_object_obs: dict[str, tuple[float, dict]] = {}  # object_id → (score, obj_dict)
        for entry in summary["frames"]:
            detections = FrameDetections.model_validate(self._json_load(entry["detections"]))
            observed_objects = services.estimator.estimate(detections)
            pit_snapshot = services.estimator.pit_snapshot()
            frame_dir = frames_root / f"frame_{detections.frame_idx:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            objects_path = frame_dir / "observed_objects.json"
            snapshot_path = frame_dir / "pit_snapshot.json"
            self._json_dump(objects_path, [obj.model_dump(mode="json") for obj in observed_objects])
            self._json_dump(snapshot_path, pit_snapshot)
            frame_entries.append(
                {
                    "frame_idx": detections.frame_idx,
                    "timestamp": detections.timestamp,
                    "detections": entry["detections"],
                    "observed_objects": str(objects_path),
                    "pit_snapshot": str(snapshot_path),
                }
            )
            latest_snapshot = pit_snapshot
            for obj in observed_objects:
                obj_d = obj.model_dump(mode="json")
                score = obj.confidence or 0.0
                prev = _best_object_obs.get(obj.object_id)
                if prev is None or score > prev[0]:
                    _best_object_obs[obj.object_id] = (score, obj_d)
        latest_objects = [obj_d for _, obj_d in _best_object_obs.values()]
        camera_path = out_dir / "camera_trajectory.json"
        object_traj_path = out_dir / "object_trajectories.json"
        self._json_dump(camera_path, latest_snapshot.get("camera_trajectory", []))
        self._json_dump(object_traj_path, latest_snapshot.get("object_trajectories", {}))
        summary_path = out_dir / "summary.json"
        summary_payload = {
            "frames": frame_entries,
            "latest_pit_snapshot": latest_snapshot,
            "latest_objects": latest_objects,
            "camera_trajectory": str(camera_path),
            "object_trajectories": str(object_traj_path),
        }
        self._json_dump(summary_path, summary_payload)
        background_assets: dict = {}
        try:
            background_assets = generate_target_frame_background_assets(
                summary_path=summary_path,
                output_dir=out_dir / "background_assets",
                target_frame_id=3,
                depth_maps_dir=wildgs_outputs.get("depth_maps_dir"),
                camera_trajectory_path=camera_path,
                clean_depth_estimator=self._build_clean_background_depth_estimator(out_dir / "background_assets"),
                semantic_road_estimator=self._build_semantic_road_estimator(out_dir / "background_assets"),
                grid_stride=4,
            )
            _logger.info(
                "[geometry.lift] Generated target-frame background assets: %s",
                background_assets.get("manifest_path"),
            )
        except Exception as exc:
            _logger.warning("[geometry.lift] Target-frame background asset generation failed (non-fatal): %s", exc)
        outputs = {
            "summary": str(summary_path),
            "frames_dir": str(frames_root),
            "camera_trajectory": str(camera_path),
            "object_trajectories": str(object_traj_path),
        }
        if wildgs_outputs.get("camera_poses_path"):
            outputs["wildgs_camera_poses"] = wildgs_outputs["camera_poses_path"]
        if wildgs_outputs.get("static_map_dir"):
            outputs["wildgs_static_map"] = wildgs_outputs["static_map_dir"]
        if wildgs_outputs.get("dynamic_prior_dir"):
            outputs["wildgs_dynamic_prior"] = wildgs_outputs["dynamic_prior_dir"]
        if wildgs_outputs.get("depth_maps_dir"):
            outputs["wildgs_depth_maps"] = wildgs_outputs["depth_maps_dir"]
        if wildgs_outputs.get("plots_dir"):
            outputs["wildgs_plots"] = wildgs_outputs["plots_dir"]
        if wildgs_outputs.get("static_map_dir"):
            outputs["wildgs_background_mesh"] = wildgs_outputs["static_map_dir"]
        elif wildgs_outputs.get("mesh_result", {}).get("mesh_dir"):
            outputs["wildgs_background_mesh"] = wildgs_outputs["mesh_result"]["mesh_dir"]
        if background_assets.get("manifest_path"):
            outputs["background_assets_manifest"] = str(background_assets["manifest_path"])
            outputs["background_assets_mesh_dir"] = str(background_assets["mesh_dir"])
        result_summary = {
            "frame_count": len(frame_entries),
            "object_count": len(latest_objects),
            "camera_samples": len(latest_snapshot.get("camera_trajectory", [])),
            "camera_provider": latest_snapshot.get("camera_provider", pit_cfg.camera_provider),
            "wildgs_slam_quality": wildgs_outputs.get("slam_quality"),
            "background_assets_available": bool(background_assets.get("manifest_path")),
        }
        return self._base_result("geometry.lift", result_summary, outputs)

    def _build_clean_background_depth_estimator(self, output_dir: Path):
        if self._provider_mode() != "zaiwu":
            return None
        settings = self.context.config.settings
        if not getattr(settings.zaiwu, "enabled", False):
            return None

        def estimate(clean_rgb_path: Path) -> dict[str, Any] | None:
            return self._estimate_clean_background_depth_with_zaiwu(clean_rgb_path, output_dir=output_dir)

        return estimate

    def _build_semantic_road_estimator(self, output_dir: Path):
        if self._provider_mode() != "zaiwu":
            return None
        settings = self.context.config.settings
        if not getattr(settings.zaiwu, "enabled", False):
            return None

        def estimate(clean_rgb_path: Path, *, frame_id: int) -> dict[str, Any] | None:
            return self._estimate_semantic_road_with_zaiwu(clean_rgb_path, frame_id=frame_id, output_dir=output_dir)

        return estimate

    def _estimate_clean_background_depth_with_zaiwu(self, clean_rgb_path: Path, *, output_dir: Path) -> dict[str, Any] | None:
        service_id = str(self.context.config.settings.zaiwu.depth_service or "services.depth_anything3")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "_clean_target_rgb_for_depth.mp4"
        depth_path = output_dir / "clean_target_depth_depth_anything3.npy"
        try:
            self._write_single_frame_depth_video(clean_rgb_path, video_path)
            gateway = build_zaiwu_gateway_client(self.context.config.settings)
            video_file_id = gateway.upload_file(service_id, video_path)
            result = gateway.run_service_job(
                service_id,
                "estimate_from_video",
                payload={"video_file_id": video_file_id, "sample_every_n": 1},
                timeout_sec=max(1800.0, float(self.context.config.settings.zaiwu.job_timeout_sec or 0.0)),
            )
            output_file_id = str(result.get("output_file_id") or result.get("result_file_id") or "")
            if not output_file_id:
                _logger.warning("[geometry.lift] Depth Anything3 returned no depth artifact for clean background: %s", result)
                return None
            data = gateway.download_bytes(service_id, output_file_id)
            depth_arr = self._decode_depth_anything_result(data)
            if depth_arr is None:
                _logger.warning("[geometry.lift] Depth Anything3 clean background artifact is not a valid depth array")
                return None
            np.save(depth_path, depth_arr.astype(np.float32))
            return {
                "depth_path": str(depth_path),
                "source": "depth_anything3_clean_target_rgb",
                "quality": {
                    "depth_service": service_id,
                    "depth_frame_count": int(1 if depth_arr.ndim == 2 else depth_arr.shape[0]),
                    "depth_artifact_file_id": output_file_id,
                },
            }
        except Exception as exc:
            _logger.warning("[geometry.lift] Depth Anything3 clean background depth failed; falling back to WildGS depth: %s", exc)
            return None

    def _estimate_semantic_road_with_zaiwu(
        self,
        clean_rgb_path: Path,
        *,
        frame_id: int,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        service_id = str(self.context.config.settings.zaiwu.grounded_sam2_service or "services.grounding_dino_sam2")
        mask_path = output_dir / "road_gsam2_mask.png"
        raw_path = output_dir / "road_gsam2_raw.json"
        try:
            image = cv2.imread(str(clean_rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read clean background RGB for road segmentation: {clean_rgb_path}")
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError(f"Failed to encode clean background RGB for road segmentation: {clean_rgb_path}")
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            payload = {
                "frame_idx": int(frame_id),
                "timestamp": 0.0,
                "image_base64": image_b64,
                "text_prompt": "road. roadway. asphalt road. driving lane. lane marking.",
            }
            gateway = build_zaiwu_gateway_client(self.context.config.settings)
            result = gateway.run_service_job(
                service_id,
                "gsam2_parse_frame",
                payload=payload,
                timeout_sec=max(1800.0, float(self.context.config.settings.zaiwu.job_timeout_sec or 0.0)),
            )
            raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            mask = self._road_mask_from_grounded_sam2_payload(result, image.shape[:2])
            if mask is None or not mask.any():
                _logger.warning("[geometry.lift] GroundedSAM2 returned no usable road mask for clean background")
                return None
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            return {
                "mask": mask,
                "mask_path": str(mask_path),
                "source": "grounding_dino_sam2_clean_target_rgb",
                "quality": {
                    "road_service": service_id,
                    "road_mask_fraction": float(np.mean(mask)),
                    "raw_result_path": str(raw_path),
                },
            }
        except Exception as exc:
            _logger.warning("[geometry.lift] GroundedSAM2 clean road segmentation failed; falling back to detection road masks: %s", exc)
            return None

    @staticmethod
    def _road_mask_from_grounded_sam2_payload(payload: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
        road_masks: list[np.ndarray] = []
        decoded_masks: list[np.ndarray] = []
        for inst in payload.get("instances", []) or []:
            if not isinstance(inst, dict):
                continue
            label = str(inst.get("concept_label") or inst.get("label") or "").lower()
            mask = ProjectExecutor._decode_grounded_sam2_mask(inst, shape)
            if mask is None:
                continue
            decoded_masks.append(mask)
            if any(token in label for token in ("road", "roadway", "asphalt", "lane", "street", "pavement", "driveway")):
                road_masks.append(mask)
        masks = road_masks or decoded_masks
        if not masks:
            return None
        return np.logical_or.reduce(masks).astype(bool)

    @staticmethod
    def _decode_grounded_sam2_mask(inst: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
        raw = inst.get("mask_rle") or inst.get("mask")
        if raw:
            try:
                rle = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if isinstance(rle.get("counts"), list):
                    return ProjectExecutor._decode_uncompressed_rle_mask(rle, shape)
                counts = rle.get("counts")
                if isinstance(counts, str):
                    rle["counts"] = counts.encode("ascii")
                from pycocotools import mask as mask_utils

                decoded = mask_utils.decode(rle)
                if decoded.ndim == 3:
                    decoded = decoded[:, :, 0]
                mask = decoded.astype(bool)
                if mask.shape == shape:
                    return mask
            except Exception:
                pass
        bbox = inst.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                height, width = shape
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
            except Exception:
                return None
            if x2 > x1 and y2 > y1:
                mask = np.zeros((height, width), dtype=bool)
                mask[y1:y2, x1:x2] = True
                return mask
        return None

    @staticmethod
    def _decode_uncompressed_rle_mask(rle: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
        size = rle.get("size")
        counts = rle.get("counts")
        if not (isinstance(size, list) and len(size) >= 2 and isinstance(counts, list)):
            return None
        height, width = int(size[0]), int(size[1])
        if (height, width) != shape:
            return None
        values: list[int] = []
        fill = 0
        for count in counts:
            try:
                run = int(count)
            except (TypeError, ValueError):
                return None
            if run < 0:
                return None
            values.extend([fill] * run)
            fill = 1 - fill
        expected = height * width
        if len(values) < expected:
            values.extend([0] * (expected - len(values)))
        if len(values) > expected:
            values = values[:expected]
        return np.asarray(values, dtype=np.uint8).reshape((height, width), order="F").astype(bool)

    @staticmethod
    def _write_single_frame_depth_video(image_path: Path, video_path: Path, *, fps: float = 1.0) -> None:
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Failed to read clean background RGB for depth: {image_path}")
        height, width = rgb.shape[:2]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open temporary depth video writer: {video_path}")
        try:
            writer.write(rgb)
        finally:
            writer.release()

    @staticmethod
    def _decode_depth_anything_result(data: bytes):
        import io

        try:
            arr = np.load(io.BytesIO(data))
        except Exception:
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            tmp_path = Path(tmp.name)
            try:
                tmp.write(data)
                tmp.close()
                arr = np.load(str(tmp_path))
            finally:
                try:
                    tmp.close()
                except Exception:
                    pass
                tmp_path.unlink(missing_ok=True)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            return None
        return arr

    def _run_mesh_reconstruct(self) -> dict:
        geometry = self.context.artifacts.get("geometry.lift")
        if geometry is None:
            raise RuntimeError("geometry.lift outputs are required")
        attr_artifact = self.context.artifacts.get("object.attr")
        if attr_artifact is None:
            raise RuntimeError("object.attr outputs are required")
        out_dir = self.context.stage_output_dir("mesh.reconstruct")
        objects = self._all_objects()
        object_attrs: dict = self._json_load(attr_artifact.outputs["object_attrs"])
        # Only reconstruct objects that are movable rigid bodies
        objects_to_reconstruct = [
            obj for obj in objects
            if object_attrs.get(obj.object_id, {}).get("is_movable") is True
            and object_attrs.get(obj.object_id, {}).get("is_rigid_body") is True
        ]
        _logger.info(
            f"mesh.reconstruct: {len(objects_to_reconstruct)}/{len(objects)} objects "
            f"selected (is_movable=True, is_rigid_body=True)"
        )
        object_ids = {obj.object_id for obj in objects_to_reconstruct}
        best_frames = self._find_best_frame_per_object(object_ids)
        _logger.info(
            f"mesh.reconstruct: best frames found for {len(best_frames)}/{len(object_ids)} objects"
        )
        meshes_path = out_dir / "sam3d_meshes.json"
        meshes: dict[str, dict] = {}
        self._json_dump(meshes_path, meshes)

        mode = self._provider_mode()
        if mode == "zaiwu":
            self._assert_zaiwu_service_ready(
                self.context.config.settings.zaiwu.sam3d_service,
                stage="mesh.reconstruct",
            )
            meshes, mesh_stats = self._run_zaiwu_mesh_reconstruct(
                objects_to_reconstruct,
                best_frames,
                meshes_path,
            )
        else:
            meshes = self._reconstruct_object_meshes(best_frames, objects_to_reconstruct)
            self._json_dump(meshes_path, meshes)
            mesh_stats = {
                "candidate_count": len(best_frames),
                "selected_count": len(objects_to_reconstruct),
                "missing_best_frame_count": max(0, len(objects_to_reconstruct) - len(best_frames)),
                "skipped_unselected_count": 0,
                "attempted_count": len(objects_to_reconstruct),
                "skipped_selected_count": 0,
                "skipped_count": max(0, len(objects_to_reconstruct) - len(best_frames)),
            }
        # Annotate each mesh entry with reconstruction metadata needed for world-space alignment
        for obj_id, entry in meshes.items():
            frame_data = best_frames.get(obj_id)
            if frame_data is not None:
                fd, inst = frame_data
                entry["reconstruction_frame_idx"] = fd.frame_idx
                entry["mask_rle"] = inst.mask_rle
                selection = self._mesh_frame_selection_score(inst)
                entry["mesh_frame_selection"] = {
                    "frame_idx": fd.frame_idx,
                    "score": selection["score"],
                    "truncated": selection["truncated"],
                    "area_px": selection["area_px"],
                    "edge_margin_px": selection["edge_margin_px"],
                    "edge_score": selection["edge_score"],
                    "center_score": selection["center_score"],
                    "size_score": selection["size_score"],
                }
        self._json_dump(meshes_path, meshes)
        outputs = {"sam3d_meshes": str(meshes_path)}
        summary = {
            "mesh_count": len(meshes),
            "object_count": len(objects),
            "reconstructed_count": len(objects_to_reconstruct),
            **mesh_stats,
        }
        return self._base_result("mesh.reconstruct", summary, outputs)

    def _run_pose_optimize(self) -> dict:
        import numpy as np

        geometry = self.context.artifacts.get("geometry.lift")
        mesh_art = self.context.artifacts.get("mesh.reconstruct")
        if geometry is None or mesh_art is None:
            raise RuntimeError("geometry.lift and mesh.reconstruct outputs are required")

        out_dir = self.context.stage_output_dir("pose.optimize")
        out_dir.mkdir(parents=True, exist_ok=True)

        sam3d_meshes: dict = self._json_load(mesh_art.outputs["sam3d_meshes"])
        cam_traj: list = self._json_load(geometry.outputs["camera_trajectory"])
        obj_traj: dict = self._json_load(geometry.outputs["object_trajectories"])
        geo_summary = self._json_load(geometry.outputs["summary"])
        detection_frames = geo_summary.get("frames", [])
        depth_maps_dir = self._resolve_wildgs_depth_maps_dir(geometry)
        wildgs_poses, wildgs_K = self._load_wildgs_poses(geometry)
        road_geometry = estimate_road_geometry(
            depth_maps_dir=depth_maps_dir,
            wildgs_poses=wildgs_poses,
            wildgs_K=wildgs_K,
            detection_frames=detection_frames,
            world_up_axis="-y",
        )
        road_geometry = self._road_geometry_with_background_fallback(road_geometry, geometry)
        road_geometry_path = out_dir / "road_geometry.json"
        self._json_dump(road_geometry_path, road_geometry)

        scene_up = self._pose_track_scene_up(road_geometry)
        target_frame_id = self._pose_target_frame_id()
        target_window_radius = self._pose_target_window_radius()
        pose_strategy = self._pose_optimizer_mode()
        if pose_strategy in {"edge_contour_fast_temporal", "generic_appearance_temporal"}:
            return self._run_edge_contour_temporal_pose_optimize(
                out_dir=out_dir,
                sam3d_meshes=sam3d_meshes,
                cam_traj=cam_traj,
                obj_traj=obj_traj,
                detection_frames=detection_frames,
                depth_maps_dir=depth_maps_dir,
                wildgs_poses=wildgs_poses,
                wildgs_K=wildgs_K,
                road_geometry=road_geometry,
                road_geometry_path=road_geometry_path,
                scene_up=scene_up,
                target_frame_id=target_frame_id,
                target_window_radius=target_window_radius,
                pose_strategy=pose_strategy,
            )
        object_pose_tracks: dict[str, dict] = {}
        per_frame_object_poses: dict[str, dict] = {}
        mesh_canonicalization: dict[str, dict] = {}
        pose_quality_report: dict[str, dict] = {}
        manifest: dict[str, dict] = {
            "schema": "guanwu.pose_track.v1",
            "strategy": "depth_icp_temporal",
            "target_frame_id": target_frame_id,
            "target_window_radius": target_window_radius,
            "objects": {},
        }
        tracked = 0
        skipped = 0
        interpolated_frames = 0
        depth_frames = 0
        target_refined = 0

        for obj_id, mesh_entry in sam3d_meshes.items():
            glb_path = self._find_glb(mesh_entry)
            if not glb_path:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "missing_glb"}
                skipped += 1
                continue

            obj_mesh = self._load_trimesh(glb_path)
            if obj_mesh is None or len(obj_mesh.vertices) < 8:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "invalid_mesh"}
                skipped += 1
                continue
            verts = np.asarray(obj_mesh.vertices, dtype=np.float64)

            track_records = [
                rec for rec in obj_traj.get(obj_id, [])
                if isinstance(rec, dict) and int(rec.get("frame_id") or 0) > 0
            ]
            if not track_records:
                frame_id = int(mesh_entry.get("reconstruction_frame_idx") or 0)
                if frame_id > 0:
                    track_records = [{"frame_id": frame_id, "timestamp_sec": self._timestamp_for_frame([], frame_id)}]
            if not track_records:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "missing_object_track"}
                skipped += 1
                continue
            track_records = sorted(track_records, key=lambda rec: int(rec.get("frame_id") or 0))
            if target_frame_id is not None and self._get_instance_for_frame(obj_id, target_frame_id, detection_frames):
                if not any(int(rec.get("frame_id") or 0) == int(target_frame_id) for rec in track_records):
                    track_records.append({
                        "frame_id": int(target_frame_id),
                        "timestamp_sec": self._timestamp_for_frame(track_records, int(target_frame_id)),
                    })
                    track_records = sorted(track_records, key=lambda rec: int(rec.get("frame_id") or 0))

            observations = self._build_pose_track_observations(
                obj_id=obj_id,
                track_records=track_records,
                detection_frames=detection_frames,
                depth_maps_dir=depth_maps_dir,
                cam_traj=cam_traj,
                wildgs_poses=wildgs_poses,
                wildgs_K=wildgs_K,
            )
            depth_observations = [obs for obs in observations if obs.get("points") is not None]
            if not depth_observations:
                manifest["objects"][obj_id] = {
                    "status": "skipped",
                    "reason": "missing_depth_observations",
                    "frame_count": len(track_records),
                }
                skipped += 1
                continue

            rotation, mesh_basis, world_basis, axis_roles, heading_info = self._pose_track_object_rotation(
                verts,
                depth_observations,
                scene_up=scene_up,
            )
            scale = self._pose_track_shared_scale(
                verts,
                depth_observations,
                mesh_basis=mesh_basis,
                world_basis=world_basis,
            )
            if target_frame_id is not None:
                window_depth = [
                    obs for obs in depth_observations
                    if abs(int(obs.get("frame_id") or 0) - int(target_frame_id)) <= target_window_radius
                ]
                if window_depth:
                    scale = self._pose_track_shared_scale(
                        verts,
                        window_depth,
                        mesh_basis=mesh_basis,
                        world_basis=world_basis,
                    )

            frames = self._pose_track_frames_from_observations(
                verts=verts,
                observations=observations,
                rotation=rotation,
                scale=scale,
                scene_up=scene_up,
                road_geometry=road_geometry,
            )
            if not frames:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "empty_pose_track"}
                skipped += 1
                continue

            target_refinement = None
            if target_frame_id is not None:
                target_refinement = self._optimize_target_frame_vehicle_pose(
                    obj_id=obj_id,
                    verts=verts,
                    observations=observations,
                    base_rotation=rotation,
                    base_scale=scale,
                    mesh_basis=mesh_basis,
                    axis_roles=axis_roles,
                    scene_up=scene_up,
                    road_geometry=road_geometry,
                    target_frame_id=target_frame_id,
                    window_radius=target_window_radius,
                    cam_traj=cam_traj,
                    wildgs_poses=wildgs_poses,
                    wildgs_K=wildgs_K,
                    detection_frames=detection_frames,
                )
                if target_refinement:
                    frames = self._merge_pose_track_frames(frames, target_refinement["frames"])
                    target_refined += 1

            object_pose_tracks[obj_id] = {
                "schema": "guanwu.object_pose_track.v1",
                "object_id": obj_id,
                "mesh_path": str(glb_path),
                "mesh_basis": mesh_basis.tolist(),
                "axis_roles": axis_roles,
                "scale": [float(v) for v in scale],
                "rotation_matrix": rotation.tolist(),
                "orientation_quat": self._rotation_matrix_to_quat_xyzw(rotation),
                "heading": heading_info,
                "target_frame_refinement": (target_refinement or {}).get("summary"),
                "frames": frames,
            }
            mesh_canonicalization[obj_id] = {
                "mesh_path": str(glb_path),
                "mesh_basis": mesh_basis.tolist(),
                "axis_roles": axis_roles,
                "local_up_axis_idx": axis_roles.get("up_axis_idx"),
                "local_forward_axis_idx": axis_roles.get("forward_axis_idx"),
            }
            depth_count = sum(1 for frame in frames if frame.get("source") == "depth_temporal")
            interp_count = sum(1 for frame in frames if frame.get("source") != "depth_temporal")
            depth_frames += depth_count
            interpolated_frames += interp_count
            tracked += 1
            manifest["objects"][obj_id] = {
                "status": "tracked",
                "frame_count": len(frames),
                "depth_frame_count": depth_count,
                "interpolated_frame_count": interp_count,
                "scale": [float(v) for v in scale],
                "heading": heading_info,
                "target_frame_refinement": (target_refinement or {}).get("summary"),
            }
            pose_quality_report[obj_id] = {
                "frame_count": len(frames),
                "depth_frame_count": depth_count,
                "interpolated_frame_count": interp_count,
                "mean_confidence": float(np.mean([frame.get("confidence", 0.0) for frame in frames])),
                "target_frame_refined": bool(target_refinement),
            }
            for frame in frames:
                frame_key = f"frame_{int(frame['frame_id']):06d}"
                per_frame_object_poses.setdefault(frame_key, {})[obj_id] = {
                    "object_id": obj_id,
                    "mesh_path": str(glb_path),
                    "T_world_from_object": frame["T_world_from_object"],
                    "centroid_world": frame["centroid_world"],
                    "rotation_matrix": frame["rotation_matrix"],
                    "orientation_quat": frame["orientation_quat"],
                    "scale": frame["scale"],
                    "confidence": frame["confidence"],
                    "source": frame["source"],
                    "quality": frame.get("quality", {}),
                }

        task_summary = self._write_pose_track_task_result_artifacts(
            out_dir=out_dir,
            object_pose_tracks=object_pose_tracks,
            per_frame_object_poses=per_frame_object_poses,
            sam3d_meshes=sam3d_meshes,
            detection_frames=detection_frames,
            cam_traj=cam_traj,
            wildgs_poses=wildgs_poses,
            wildgs_K=wildgs_K,
            road_geometry=road_geometry,
            target_frame_id=target_frame_id,
        )
        manifest["task_result_artifacts"] = task_summary
        object_tracks_path = out_dir / "object_pose_tracks.json"
        per_frame_path = out_dir / "per_frame_object_poses.json"
        mesh_canon_path = out_dir / "mesh_canonicalization.json"
        quality_path = out_dir / "pose_quality_report.json"
        refined_traj_path = out_dir / "refined_object_trajectories.json"
        manifest_path = out_dir / "pose_optimizer_manifest.json"
        refined_object_trajectories = self._refined_trajectories_from_pose_tracks(object_pose_tracks)
        self._json_dump(object_tracks_path, object_pose_tracks)
        self._json_dump(per_frame_path, per_frame_object_poses)
        self._json_dump(mesh_canon_path, mesh_canonicalization)
        self._json_dump(quality_path, pose_quality_report)
        self._json_dump(refined_traj_path, refined_object_trajectories)
        self._json_dump(manifest_path, manifest)
        outputs = {
            "pose_optimizer_manifest": str(manifest_path),
            "object_pose_tracks": str(object_tracks_path),
            "per_frame_object_poses": str(per_frame_path),
            "mesh_canonicalization": str(mesh_canon_path),
            "pose_quality_report": str(quality_path),
            "refined_object_trajectories": str(refined_traj_path),
            "road_geometry": str(road_geometry_path),
            "tasks_dir": str(out_dir / "tasks"),
            "results_dir": str(out_dir / "results"),
        }
        summary = {
            "object_count": len(sam3d_meshes),
            "tracked_count": tracked,
            "skipped_count": skipped,
            "depth_pose_frame_count": depth_frames,
            "interpolated_pose_frame_count": interpolated_frames,
            "per_frame_count": len(per_frame_object_poses),
            "road_geometry_available": bool(road_geometry.get("available")),
            "strategy": "depth_icp_temporal",
            "target_frame_id": target_frame_id,
            "target_window_radius": target_window_radius,
            "target_refined_objects": target_refined,
            "pose_task_count": task_summary.get("task_count", 0),
            "pose_result_count": task_summary.get("result_count", 0),
        }
        return self._base_result(
            "pose.optimize",
            summary,
            outputs,
            params={"target_frame_id": target_frame_id, "target_window_radius": target_window_radius},
        )

    @staticmethod
    def _pose_optimizer_mode() -> str:
        raw = os.environ.get("GUANWU_POSE_OPTIMIZER_MODE", "edge_contour_fast_temporal")
        value = str(raw or "").strip().lower().replace("-", "_")
        if value in {"generic", "generic_appearance", "generic_appearance_temporal"}:
            return "generic_appearance_temporal"
        if value in {"depth", "depth_temporal", "depth_icp", "depth_icp_temporal"}:
            return "depth_icp_temporal"
        return "edge_contour_fast_temporal"

    @staticmethod
    def _refined_trajectories_from_pose_tracks(object_pose_tracks: dict) -> dict[str, list[dict]]:
        refined: dict[str, list[dict]] = {}
        if not isinstance(object_pose_tracks, dict):
            return refined
        for obj_id, track in object_pose_tracks.items():
            if not isinstance(track, dict):
                continue
            pose_source = str(track.get("pose_source") or "pose_optimize")
            frames_out: list[dict] = []
            for frame in track.get("frames", []) or []:
                if not isinstance(frame, dict):
                    continue
                if not ProjectExecutor._valid_vec3_like(frame.get("centroid_world")):
                    continue
                if not ProjectExecutor._valid_vec3_like(frame.get("scale")):
                    continue
                rotation = frame.get("rotation_matrix") or track.get("rotation_matrix")
                if not isinstance(rotation, list) or len(rotation) != 3:
                    continue
                center = [float(v) for v in frame["centroid_world"]]
                scale = [float(v) for v in frame["scale"]]
                try:
                    frame_id = int(frame.get("frame_id") or 0)
                except Exception:
                    frame_id = 0
                if frame_id <= 0:
                    continue
                refined_frame = {
                    "frame_id": frame_id,
                    "timestamp_sec": float(frame.get("timestamp_sec", 0.0) or 0.0),
                    "centroid_world": center,
                    "position_xyz": center,
                    "rotation_matrix": rotation,
                    "orientation_quat": frame.get("orientation_quat")
                    or ProjectExecutor._rotation_matrix_to_quat_xyzw(rotation),
                    "scale": scale,
                    "confidence": float(frame.get("confidence", 0.0) or 0.0),
                    "trajectory_source": "pose_optimize",
                    "pose_source": str(frame.get("source") or pose_source),
                    "geometry_status": frame.get("geometry_status", frame.get("source", pose_source)),
                    "quality": frame.get("quality", {}),
                }
                if frame.get("T_world_from_object") is not None:
                    refined_frame["T_world_from_object"] = frame.get("T_world_from_object")
                frames_out.append(refined_frame)
            if frames_out:
                frames_out.sort(key=lambda item: int(item.get("frame_id") or 0))
                refined[str(obj_id)] = frames_out
        return refined

    @staticmethod
    def _merge_refined_object_trajectories(coarse: dict | None, refined: dict | None) -> dict[str, list[dict]]:
        merged: dict[str, dict[int, dict]] = {}
        for source_name, payload in (("geometry_lift", coarse), ("pose_optimize", refined)):
            if not isinstance(payload, dict):
                continue
            for obj_id, records in payload.items():
                if not isinstance(records, list):
                    continue
                obj_frames = merged.setdefault(str(obj_id), {})
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    try:
                        frame_id = int(record.get("frame_id") or record.get("frame_idx") or 0)
                    except Exception:
                        continue
                    if frame_id <= 0:
                        continue
                    normalized = dict(record)
                    normalized["frame_id"] = frame_id
                    if "timestamp_sec" not in normalized and "timestamp" in normalized:
                        normalized["timestamp_sec"] = normalized.get("timestamp")
                    if source_name == "pose_optimize":
                        normalized["trajectory_source"] = "pose_optimize"
                    else:
                        normalized.setdefault("trajectory_source", "geometry_lift")
                    obj_frames[frame_id] = normalized
        return {
            obj_id: [frames[fid] for fid in sorted(frames)]
            for obj_id, frames in sorted(merged.items())
            if frames
        }

    @staticmethod
    def _prepare_usd_object_mesh_vertices(vertices):
        import numpy as np

        verts = np.asarray(vertices, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            return verts.copy()
        # Pose optimizer rotations are solved in the mesh's native local
        # axes and translation is the transform of that same local origin.
        # Re-centering here changes the solved ground contact in scene export.
        return verts.copy()

    @staticmethod
    def _pose_resume_results() -> bool:
        raw = os.environ.get("GUANWU_POSE_RESUME_RESULTS", "")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _pose_target_object_ids() -> set[str]:
        raw = os.environ.get("GUANWU_POSE_TARGET_OBJECT_IDS", "")
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    @staticmethod
    def _pose_max_target_objects() -> int | None:
        raw = os.environ.get("GUANWU_POSE_MAX_TARGET_OBJECTS", "")
        if str(raw).strip() == "":
            return None
        try:
            value = int(str(raw).strip())
        except ValueError:
            return None
        return max(0, value)

    def _run_edge_contour_temporal_pose_optimize(
        self,
        *,
        out_dir: Path,
        sam3d_meshes: dict,
        cam_traj: list[dict],
        obj_traj: dict,
        detection_frames: list[dict],
        depth_maps_dir: str | None,
        wildgs_poses: list[dict],
        wildgs_K: dict | None,
        road_geometry: dict | None,
        road_geometry_path: Path,
        scene_up,
        target_frame_id: int | None,
        target_window_radius: int,
        pose_strategy: str = "edge_contour_fast_temporal",
    ) -> dict:
        import numpy as np

        pose_strategy = str(pose_strategy or "edge_contour_fast_temporal")
        generic_mode = pose_strategy == "generic_appearance_temporal"
        target_frame_mode = self._pose_target_frame_mode()
        all_frames_mode = target_frame_mode == "all_frames"
        dynamic_window_enabled = self._pose_env_bool("GUANWU_POSE_DYNAMIC_WINDOW", default=all_frames_mode)
        dynamic_window_min_radius, dynamic_window_max_radius = self._pose_dynamic_window_bounds(target_window_radius)
        if target_frame_id is None and not all_frames_mode:
            target_frame_id = self._select_dense_detection_frame(detection_frames)
        target_frame_id = int(target_frame_id or 1) if target_frame_id is not None else None
        target_window_radius = int(max(0, target_window_radius))
        target_object_ids = self._pose_target_object_ids()
        max_target_objects = self._pose_max_target_objects()
        resume_results = self._pose_resume_results()
        selected_target_objects = 0

        tasks_dir = out_dir / "tasks"
        results_dir = out_dir / "results"
        if tasks_dir.exists() and not resume_results:
            shutil.rmtree(tasks_dir)
        if results_dir.exists() and not resume_results:
            shutil.rmtree(results_dir)
        tasks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        object_nodes = {obj.object_id: obj for obj in self._all_objects()}
        object_pose_tracks: dict[str, dict] = {}
        per_frame_object_poses: dict[str, dict] = {}
        mesh_canonicalization: dict[str, dict] = {}
        pose_quality_report: dict[str, dict] = {}
        manifest: dict[str, dict] = {
            "schema": "guanwu.pose_track.v1",
            "strategy": pose_strategy,
            "target_frame_mode": target_frame_mode,
            "target_frame_id": target_frame_id,
            "target_window_radius": target_window_radius,
            "dynamic_window": {
                "enabled": dynamic_window_enabled,
                "min_radius": dynamic_window_min_radius,
                "max_radius": dynamic_window_max_radius,
            },
            "target_object_ids": sorted(target_object_ids) if target_object_ids else None,
            "max_target_objects": max_target_objects,
            "resume_results": resume_results,
            "acceptance": (
                {
                    "min_visible_mask_iou": 0.12,
                    "min_bbox_iou": 0.10,
                    "max_center_error_px": "max(120, 0.35*bbox_diag)",
                    "min_projection_valid_ratio": 0.50,
                    "depth_gate_when_confident": True,
                    "road_heading_ground_upright_hard_gates": False,
                }
                if generic_mode
                else {
                    "bbox_area_hard_filter": False,
                    "mask_iou_min": 0.20,
                    "bbox_iou_min": 0.20,
                    "bbox_center_error_px_max": 120.0,
                    "temporal_jump_rejection": True,
                    "upright_angle_error_deg_max": 35.0,
                    "ground_contact_max_abs_m_max": 0.65,
                    "bbox_bottom_ground_distance_m_max": 1.25,
                    "lock_mesh_up_sign": True,
                    "lock_mesh_forward_sign": False,
                }
            ),
            "objects": {},
        }

        attempted_frames = 0
        accepted_frames = 0
        rejected_frames = 0
        failed_frames = 0
        reused_frames = 0
        tracked_objects = 0
        skipped_objects = 0
        target_accepted_objects = 0

        for obj_id, mesh_entry in sam3d_meshes.items():
            if target_object_ids and obj_id not in target_object_ids:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "not_in_target_object_ids"}
                skipped_objects += 1
                continue

            glb_path = self._find_glb(mesh_entry)
            if not glb_path:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "missing_glb"}
                skipped_objects += 1
                continue

            obj_mesh = self._load_trimesh(glb_path)
            if obj_mesh is None or len(obj_mesh.vertices) < 8:
                manifest["objects"][obj_id] = {"status": "skipped", "reason": "invalid_mesh"}
                skipped_objects += 1
                continue
            verts = np.asarray(obj_mesh.vertices, dtype=np.float64)

            if all_frames_mode:
                frame_ids = self._pose_all_frame_candidate_frame_ids(
                    obj_id=obj_id,
                    detection_frames=detection_frames,
                    min_bbox_area_px=_POSE_MATCH_MIN_BBOX_AREA_PX,
                    generic_mode=generic_mode,
                )
                if not frame_ids:
                    manifest["objects"][obj_id] = {
                        "status": "skipped",
                        "reason": (
                            "missing_generic_observations_above_bbox_threshold"
                            if generic_mode
                            else "missing_vehicle_observations_above_bbox_threshold"
                        ),
                        "target_frame_mode": target_frame_mode,
                        "min_bbox_area_px": _POSE_MATCH_MIN_BBOX_AREA_PX,
                    }
                    skipped_objects += 1
                    continue
                target_inst = None
                target_obs = {"bbox": None, "bbox_area_px": 0.0}
            else:
                target_inst = self._get_instance_for_frame(obj_id, int(target_frame_id), detection_frames)
                target_bbox = (target_inst or {}).get("bbox_xyxy") or (target_inst or {}).get("bbox")
                target_obs = {"bbox": target_bbox, "bbox_area_px": self._bbox_area_px(target_bbox)}
                if not self._is_target_pose_candidate(target_inst, target_obs, generic_mode=generic_mode):
                    manifest["objects"][obj_id] = {
                        "status": "skipped",
                        "reason": "missing_target_generic_observation" if generic_mode else "missing_target_vehicle_observation",
                        "target_frame_id": target_frame_id,
                        "target_bbox_area_px": target_obs["bbox_area_px"],
                    }
                    skipped_objects += 1
                    continue
                if target_obs["bbox_area_px"] < _POSE_MATCH_MIN_BBOX_AREA_PX:
                    manifest["objects"][obj_id] = {
                        "status": "skipped",
                        "reason": "pose_match_bbox_area_too_small",
                        "target_frame_id": target_frame_id,
                        "target_bbox_area_px": target_obs["bbox_area_px"],
                        "min_bbox_area_px": _POSE_MATCH_MIN_BBOX_AREA_PX,
                    }
                    skipped_objects += 1
                    continue

            if max_target_objects is not None and selected_target_objects >= max_target_objects:
                manifest["objects"][obj_id] = {
                    "status": "skipped",
                    "reason": "target_object_limit_reached",
                    "max_target_objects": max_target_objects,
                }
                skipped_objects += 1
                continue

            if not all_frames_mode:
                frame_ids = self._pose_temporal_window_frame_ids(
                    obj_id=obj_id,
                    target_frame_id=int(target_frame_id),
                    window_radius=target_window_radius,
                    detection_frames=detection_frames,
                )
                if int(target_frame_id) not in frame_ids:
                    frame_ids.append(int(target_frame_id))
                    frame_ids = sorted(set(frame_ids))
            if not frame_ids:
                manifest["objects"][obj_id] = {
                    "status": "skipped",
                    "reason": "empty_temporal_window",
                    "target_frame_id": target_frame_id,
                }
                skipped_objects += 1
                continue
            first_pose_frame_id = int(min(frame_ids))
            first_pose_inst = self._get_instance_for_frame(obj_id, first_pose_frame_id, detection_frames)
            first_truncation_decision = self._pose_first_frame_truncation_skip_decision(
                first_pose_inst,
                frame_id=first_pose_frame_id,
            )
            if first_truncation_decision.get("skip_object") and not generic_mode:
                manifest["objects"][obj_id] = {
                    "status": "skipped",
                    "reason": first_truncation_decision.get("reason"),
                    "target_frame_mode": target_frame_mode,
                    "target_frame_id": target_frame_id,
                    "first_pose_frame_id": first_pose_frame_id,
                    "truncated_sides": first_truncation_decision.get("truncated_sides", []),
                    "truncation_severity": first_truncation_decision.get("truncation_severity"),
                    "low_observability": first_truncation_decision.get("low_observability"),
                    "frame_ids": frame_ids,
                    "frames": {
                        f"frame_{first_pose_frame_id:06d}": {
                            "status": "skipped",
                            **first_truncation_decision,
                        }
                    },
                }
                skipped_objects += 1
                continue
            selected_target_objects += 1

            seed_track, seed_meta = self._build_edge_pose_seed_track(
                obj_id=obj_id,
                frame_ids=frame_ids,
                verts=verts,
                obj_traj=obj_traj,
                detection_frames=detection_frames,
                depth_maps_dir=depth_maps_dir,
                cam_traj=cam_traj,
                wildgs_poses=wildgs_poses,
                wildgs_K=wildgs_K,
                scene_up=scene_up,
                road_geometry=road_geometry,
            )
            # Seed tracks come from depth/object-track bootstrapping and can be
            # very noisy for small or truncated vehicles.  Keep them as pose
            # initializers only; scale priors are promoted later from accepted
            # high-quality pose optimizer results.
            track_scale_prior = None
            mesh_basis = seed_meta.get("mesh_basis")
            if mesh_basis is None:
                mesh_basis = np.eye(3, dtype=np.float64).tolist()
            axis_roles = seed_meta.get("axis_roles") or self._infer_source_axis_roles(verts)
            mesh_axis_prior = self._mesh_axis_prior_for_pose_optimizer(verts, axis_roles=axis_roles)
            heading_info = seed_meta.get("heading") or {"source": pose_strategy}

            accepted_records: list[dict] = []
            object_attempted = 0
            object_rejected = 0
            object_failed = 0
            object_fail_fast: dict | None = None
            previous_accepted: dict | None = None
            previous_candidate_prior: dict | None = None
            previous_anchor: dict | None = None
            stable_temporal_streak = 0
            all_frame_prior_records: list[dict] = []
            frame_records: dict[str, dict] = {}
            candidate_records_by_frame: dict[int, list[dict]] = {}
            trajectory_selection_summary: dict = {
                "enabled": target_window_radius > 0,
                "applied": False,
                "reason": "not_evaluated",
            }

            for frame_id in frame_ids:
                inst = self._get_instance_for_frame(obj_id, int(frame_id), detection_frames)
                if not isinstance(inst, dict):
                    frame_records[f"frame_{int(frame_id):06d}"] = {"status": "skipped", "reason": "missing_detection"}
                    continue
                image = self._load_frame_image_for_detection(inst)
                if image is None:
                    image = self._read_project_video_frame(int(frame_id))
                if image is None:
                    frame_records[f"frame_{int(frame_id):06d}"] = {"status": "skipped", "reason": "missing_frame_image"}
                    continue
                mask = self._mask_from_detection(inst, image.shape[:2])
                if mask is None or int(mask.sum()) <= 0:
                    frame_records[f"frame_{int(frame_id):06d}"] = {"status": "skipped", "reason": "missing_mask"}
                    continue
                camera = self._projection_camera(int(frame_id), cam_traj, wildgs_poses, wildgs_K)
                if camera is None:
                    frame_records[f"frame_{int(frame_id):06d}"] = {"status": "skipped", "reason": "missing_camera"}
                    continue

                task_id = f"{obj_id}@{int(frame_id):06d}"
                task_dir = tasks_dir / task_id
                result_dir = results_dir / task_id
                task_dir.mkdir(parents=True, exist_ok=True)
                report_path = result_dir / "optimization_report.json"
                can_reuse_result = resume_results and report_path.exists()
                if result_dir.exists() and not can_reuse_result:
                    shutil.rmtree(result_dir)
                result_dir.mkdir(parents=True, exist_ok=True)
                bbox = inst.get("bbox_xyxy") or inst.get("bbox")
                obs = {"bbox": bbox, "bbox_area_px": self._bbox_area_px(bbox)}
                frame_window_radius = target_window_radius
                if dynamic_window_enabled:
                    frame_window_radius = self._pose_dynamic_window_radius(
                        inst,
                        obs,
                        base_radius=target_window_radius,
                        min_radius=dynamic_window_min_radius,
                        max_radius=dynamic_window_max_radius,
                    )
                local_seed_track = seed_track
                local_seed_meta = seed_meta
                if all_frames_mode:
                    local_seed_frame_ids = self._pose_local_seed_frame_ids(
                        frame_ids=frame_ids,
                        current_frame_id=int(frame_id),
                        window_radius=frame_window_radius,
                    )
                    local_seed_track, local_seed_meta = self._build_edge_pose_seed_track(
                        obj_id=obj_id,
                        frame_ids=local_seed_frame_ids,
                        verts=verts,
                        obj_traj=obj_traj,
                        detection_frames=detection_frames,
                        depth_maps_dir=depth_maps_dir,
                        cam_traj=cam_traj,
                        wildgs_poses=wildgs_poses,
                        wildgs_K=wildgs_K,
                        scene_up=scene_up,
                        road_geometry=road_geometry,
                    )
                vehicle_pose_context = self._vehicle_pose_context_for_task(
                    obj_id=obj_id,
                    frame_id=int(frame_id),
                    bbox_xyxy=bbox,
                    camera=camera,
                    detection_frames=detection_frames,
                    road_geometry=road_geometry,
                    mesh_axis_prior=self._mesh_axis_prior_for_pose_optimizer(
                        verts,
                        axis_roles=(
                            local_seed_meta.get("axis_roles")
                            if isinstance(local_seed_meta, dict)
                            else axis_roles
                        ),
                    ),
                    target_window_radius=target_window_radius,
                )
                if generic_mode:
                    vehicle_pose_context = {
                        "schema": "generic_pose_context.v1",
                        "object_id": obj_id,
                        "frame_id": int(frame_id),
                        "support_plane": "auto",
                    }
                    depth_map_path = self._resolve_depth_map_for_frame(depth_maps_dir, int(frame_id))
                    if depth_map_path:
                        vehicle_pose_context["depth_map_path"] = str(depth_map_path)
                vehicle_pose_context["temporal_window"] = {
                    "mode": target_frame_mode,
                    "base_radius": int(target_window_radius),
                    "radius": int(frame_window_radius),
                    "dynamic": bool(dynamic_window_enabled),
                }
                if track_scale_prior:
                    vehicle_pose_context["track_scale_prior"] = track_scale_prior
                trusted_anchor_prior = self._edge_pose_candidate_temporal_prior_payload(
                    previous_anchor,
                    all_frames_mode=all_frames_mode,
                )
                if trusted_anchor_prior:
                    trusted_anchor_prior["source"] = "trusted_temporal_anchor_pose"
                    vehicle_pose_context["trusted_temporal_anchor_pose"] = trusted_anchor_prior
                task_path = self._write_pose_optimizer_sample(
                    task_dir=task_dir,
                    obj_id=obj_id,
                    frame_id=int(frame_id),
                    frame_image=image,
                    full_mask=mask,
                    inst=inst,
                    glb_path=glb_path,
                    camera=camera,
                    object_node=object_nodes.get(obj_id),
                    object_track=local_seed_track,
                    vehicle_pose_context=vehicle_pose_context,
                    temporal_prior_pose=self._edge_pose_candidate_temporal_prior_payload(
                        previous_candidate_prior if all_frames_mode else previous_accepted,
                        all_frames_mode=all_frames_mode,
                    ),
                )

                attempted_frames += 1
                object_attempted += 1
                if can_reuse_result:
                    reused_frames += 1
                    run_info = {
                        "returncode": 0,
                        "stdout_tail": "",
                        "stderr_tail": "",
                        "reused_optimizer_result": True,
                    }
                else:
                    run_info = (
                        self._run_generic_appearance_temporal(task_dir, result_dir)
                        if generic_mode
                        else self._run_edge_contour_fast(task_dir, result_dir)
                    )
                    report_path = result_dir / "optimization_report.json"
                record_base = {
                    "frame_id": int(frame_id),
                    "task": str(task_path),
                    "output_dir": str(result_dir),
                    **run_info,
                }
                if run_info.get("returncode") != 0 or not report_path.exists():
                    failed_frames += 1
                    object_failed += 1
                    failed_record = {
                        "status": "failed",
                        "reason": "optimizer_failed" if run_info.get("returncode") != 0 else "missing_optimization_report",
                        **record_base,
                    }
                    frame_records[f"frame_{int(frame_id):06d}"] = failed_record
                    object_fail_fast = (
                        {"skip_object": False, "reason": "generic_mode"}
                        if generic_mode
                        else self._pose_truncated_object_fail_fast_decision(
                            failed_record,
                            inst=inst,
                            frame_id=int(frame_id),
                        )
                    )
                    if object_fail_fast.get("skip_object"):
                        break
                    continue

                try:
                    report = self._json_load(report_path)
                    if not isinstance(report, dict):
                        raise ValueError("optimization report payload is not a JSON object")
                except Exception as exc:
                    failed_frames += 1
                    object_failed += 1
                    failed_record = {
                        "status": "failed",
                        "reason": "invalid_optimization_report",
                        "error": str(exc),
                        **record_base,
                    }
                    frame_records[f"frame_{int(frame_id):06d}"] = failed_record
                    object_fail_fast = (
                        {"skip_object": False, "reason": "generic_mode"}
                        if generic_mode
                        else self._pose_truncated_object_fail_fast_decision(
                            failed_record,
                            inst=inst,
                            frame_id=int(frame_id),
                        )
                    )
                    if object_fail_fast.get("skip_object"):
                        break
                    continue

                decision = (
                    self._generic_pose_optimizer_acceptance(report)
                    if generic_mode
                    else self._pose_optimizer_acceptance(report)
                )
                if decision.get("accepted"):
                    jump_decision = self._pose_optimizer_temporal_jump_acceptance(
                        report,
                        previous_accepted,
                        generic_mode=generic_mode,
                    )
                    if not jump_decision.get("accepted"):
                        decision = jump_decision

                pose_record = self._edge_pose_record_from_report(
                    obj_id=obj_id,
                    frame_id=int(frame_id),
                    report=report,
                    report_path=report_path,
                    result_dir=result_dir,
                    task_path=task_path,
                    timestamp_sec=self._timestamp_for_frame(seed_track, int(frame_id)),
                    run_info=run_info,
                )
                candidate_records = self._edge_pose_candidate_records_from_report(
                    obj_id=obj_id,
                    frame_id=int(frame_id),
                    report=report,
                    report_path=report_path,
                    result_dir=result_dir,
                    task_path=task_path,
                    timestamp_sec=self._timestamp_for_frame(seed_track, int(frame_id)),
                    run_info=run_info,
                )
                accepted_candidate_records = []
                for candidate_record in candidate_records:
                    candidate_report = self._pose_optimizer_report_from_candidate(report, candidate_record)
                    candidate_decision = (
                        self._generic_pose_optimizer_acceptance(candidate_report)
                        if generic_mode
                        else self._pose_optimizer_acceptance(candidate_report)
                    )
                    candidate_record["base_acceptance"] = candidate_decision
                    if candidate_decision.get("accepted"):
                        accepted_candidate_records.append(candidate_record)
                if accepted_candidate_records:
                    candidate_records_by_frame[int(frame_id)] = accepted_candidate_records
                pose_record["status"] = "accepted" if decision.get("accepted") else "rejected"
                pose_record["reason"] = decision.get("reason", "")

                if decision.get("accepted"):
                    anchor_decision = (
                        {"accepted": True, "reason": "generic_mode"}
                        if generic_mode
                        else self._pose_anchor_temporal_gate(pose_record, previous_anchor)
                    )
                    pose_record["anchor_temporal_gate"] = anchor_decision
                    if not anchor_decision.get("accepted"):
                        decision = anchor_decision
                        pose_record["status"] = "rejected"
                        pose_record["reason"] = str(anchor_decision.get("reason") or "anchor_temporal_gate_rejected")
                    else:
                        pose_record["metrics"].update(
                            {
                                key: anchor_decision.get(key)
                                for key in (
                                    "anchor_frame_id",
                                    "yaw_jump_deg",
                                    "rotation_jump_deg",
                                    "scale_ratio",
                                    "mask_drop",
                                    "bbox_drop",
                                )
                                if anchor_decision.get(key) is not None
                            }
                        )
                if decision.get("accepted"):
                    anchor_kind = self._pose_temporal_anchor_kind(pose_record)
                    if anchor_kind:
                        pose_record["temporal_anchor_kind"] = anchor_kind
                    if all_frames_mode:
                        if self._pose_record_updates_temporal_anchor(pose_record):
                            previous_candidate_prior = pose_record
                            stable_temporal_streak = 0
                        elif anchor_kind == "stable":
                            stable_temporal_streak += 1
                            if self._pose_record_promotes_temporal_candidate(
                                pose_record,
                                stable_streak_count=stable_temporal_streak,
                            ):
                                previous_candidate_prior = pose_record
                        else:
                            stable_temporal_streak = 0
                        if anchor_kind:
                            previous_anchor = pose_record
                        else:
                            stable_temporal_streak = 0
                        all_frame_prior_records.append(pose_record)
                        updated_scale_prior = self._pose_track_scale_prior(
                            all_frame_prior_records,
                            source="accepted_track_median_scale",
                            require_high_quality=True,
                            max_frame_id=int(frame_id),
                        )
                        if updated_scale_prior:
                            track_scale_prior = updated_scale_prior
                    else:
                        accepted_frames += 1
                        accepted_records.append(pose_record)
                        previous_accepted = pose_record
                        updated_scale_prior = self._pose_track_scale_prior(
                            accepted_records,
                            source="accepted_track_median_scale",
                            require_high_quality=True,
                            max_frame_id=int(frame_id),
                        )
                        if updated_scale_prior:
                            track_scale_prior = updated_scale_prior
                    frame_records[f"frame_{int(frame_id):06d}"] = pose_record
                else:
                    rejected_frames += 1
                    object_rejected += 1
                    frame_records[f"frame_{int(frame_id):06d}"] = pose_record
                    object_fail_fast = (
                        {"skip_object": False, "reason": "generic_mode"}
                        if generic_mode
                        else self._pose_truncated_object_fail_fast_decision(
                            pose_record,
                            inst=inst,
                            frame_id=int(frame_id),
                        )
                    )
                    if object_fail_fast.get("skip_object"):
                        break

            if object_fail_fast and object_fail_fast.get("skip_object"):
                fail_fast_summary = self._apply_truncated_object_fail_fast(
                    object_fail_fast,
                    frame_ids=frame_ids,
                    frame_records=frame_records,
                    accepted_records=accepted_records,
                )
                if fail_fast_summary.get("skip_entire_object"):
                    manifest["objects"][obj_id] = {
                        "status": "skipped_after_truncated_failure",
                        "reason": object_fail_fast.get("reason"),
                        "failed_frame_id": object_fail_fast.get("frame_id"),
                        "truncation_severity": object_fail_fast.get("truncation_severity"),
                        "low_observability": object_fail_fast.get("low_observability"),
                        "attempted_frame_count": object_attempted,
                        "accepted_frame_count": len(accepted_records),
                        "rejected_frame_count": object_rejected,
                        "failed_frame_count": object_failed,
                        "remaining_frame_count": fail_fast_summary.get("remaining_frame_count", 0),
                        "frames": frame_records,
                    }
                    skipped_objects += 1
                    continue

            if candidate_records_by_frame:
                old_accepted_count = len(accepted_records)
                old_rejected_count = object_rejected
                selected_records, trajectory_selection_summary = self._select_edge_pose_candidate_trajectory(
                    candidate_records_by_frame,
                    target_frame_id=target_frame_id,
                    generic_mode=generic_mode,
                )
                has_required_target = all_frames_mode or any(
                    int(rec.get("frame_id") or 0) == int(target_frame_id) for rec in selected_records
                )
                if selected_records and has_required_target:
                    if generic_mode:
                        anchor_gate_summary = {"applied": False, "reason": "generic_mode"}
                    else:
                        selected_records, anchor_gate_summary = self._apply_anchor_temporal_gate_to_selected_records(
                            selected_records,
                            frame_ids=frame_ids,
                            frame_records=frame_records,
                        )
                    if anchor_gate_summary.get("applied"):
                        trajectory_selection_summary["anchor_temporal_gate"] = anchor_gate_summary
                    accepted_records = selected_records
                    for selected in accepted_records:
                        frame_key = f"frame_{int(selected.get('frame_id') or 0):06d}"
                        selected["status"] = "accepted"
                        selected["reason"] = "trajectory_selected"
                        frame_records[frame_key] = selected
                    new_accepted_count = len(accepted_records)
                    new_rejected_count = max(0, object_attempted - object_failed - new_accepted_count)
                    accepted_frames += new_accepted_count - old_accepted_count
                    rejected_frames += new_rejected_count - old_rejected_count
                    object_rejected = new_rejected_count

            target_record = None
            if target_frame_id is not None:
                target_record = next(
                    (record for record in accepted_records if int(record.get("frame_id") or 0) == int(target_frame_id)),
                    None,
                )
            if target_record is None:
                if all_frames_mode and accepted_records:
                    target_record = max(
                        accepted_records,
                        key=lambda record: (
                            float((record.get("metrics") or {}).get("mask_iou") or 0.0),
                            float((record.get("metrics") or {}).get("bbox_iou") or 0.0),
                            -float((record.get("metrics") or {}).get("bbox_center_error_px") or 1e9),
                        ),
                    )
                else:
                    manifest["objects"][obj_id] = {
                        "status": "skipped",
                        "reason": "missing_accepted_target_pose",
                        "target_frame_id": target_frame_id,
                        "attempted_frame_count": object_attempted,
                        "accepted_frame_count": len(accepted_records),
                        "rejected_frame_count": object_rejected,
                        "failed_frame_count": object_failed,
                        "frames": frame_records,
                    }
                    skipped_objects += 1
                    continue

            accepted_records, stabilization_summary = self._stabilize_edge_pose_records(accepted_records)
            trajectory_refinement = {
                "selection": trajectory_selection_summary,
                "stabilization": stabilization_summary,
            }
            frames = [self._edge_pose_track_frame(record, pose_source=pose_strategy) for record in accepted_records]
            frames = [frame for frame in frames if frame is not None]
            frames.sort(key=lambda item: int(item.get("frame_id") or 0))
            if not frames:
                manifest["objects"][obj_id] = {
                    "status": "skipped",
                    "reason": "empty_accepted_track",
                    "target_frame_id": target_frame_id,
                    "frames": frame_records,
                }
                skipped_objects += 1
                continue

            target_frame = None
            if target_frame_id is not None:
                target_frame = next((frame for frame in frames if int(frame["frame_id"]) == int(target_frame_id)), None)
            if target_frame is None:
                target_frame = max(frames, key=lambda frame: float(frame.get("confidence", 0.0) or 0.0))
            track_scale = self._median_pose_scale([frame.get("scale") for frame in frames])
            track_rotation = target_frame["rotation_matrix"]
            object_pose_tracks[obj_id] = {
                "schema": "guanwu.object_pose_track.v1",
                "object_id": obj_id,
                "mesh_path": str(glb_path),
                "pose_source": pose_strategy,
                "mesh_basis": mesh_basis,
                "axis_roles": axis_roles,
                "scale": track_scale,
                "rotation_matrix": track_rotation,
                "orientation_quat": target_frame["orientation_quat"],
                "heading": heading_info,
                "target_frame_refinement": {
                    "status": pose_strategy,
                    "mode": target_frame_mode,
                    "target_frame_id": target_frame_id,
                    "window_radius": target_window_radius,
                    "accepted_frame_count": len(frames),
                    "representative_frame_id": int(target_frame.get("frame_id") or 0),
                    "target_metrics": target_record.get("metrics", {}),
                    "trajectory_refinement": trajectory_refinement,
                },
                "frames": frames,
            }
            mesh_canonicalization[obj_id] = {
                "mesh_path": str(glb_path),
                "mesh_basis": mesh_basis,
                "axis_roles": axis_roles,
                "mesh_axis_prior": mesh_axis_prior,
                "local_up_axis_idx": axis_roles.get("up_axis_idx") if isinstance(axis_roles, dict) else None,
                "local_forward_axis_idx": axis_roles.get("forward_axis_idx") if isinstance(axis_roles, dict) else None,
            }
            for frame in frames:
                frame_key = f"frame_{int(frame['frame_id']):06d}"
                per_frame_object_poses.setdefault(frame_key, {})[obj_id] = {
                    "object_id": obj_id,
                    "mesh_path": str(glb_path),
                    "T_world_from_object": frame["T_world_from_object"],
                    "centroid_world": frame["centroid_world"],
                    "rotation_matrix": frame["rotation_matrix"],
                    "orientation_quat": frame["orientation_quat"],
                    "scale": frame["scale"],
                    "confidence": frame["confidence"],
                    "source": frame["source"],
                    "quality": frame.get("quality", {}),
                }

            tracked_objects += 1
            target_accepted_objects += 1
            manifest["objects"][obj_id] = {
                "status": "tracked",
                "target_frame_mode": target_frame_mode,
                "target_frame_id": target_frame_id,
                "attempted_frame_count": object_attempted,
                "accepted_frame_count": len(frames),
                "rejected_frame_count": object_rejected,
                "failed_frame_count": object_failed,
                "target_report": target_record.get("report"),
                "target_output_dir": target_record.get("output_dir"),
                "target_metrics": target_record.get("metrics", {}),
                "mesh_axis_prior": mesh_axis_prior,
                "frames": frame_records,
            }
            pose_quality_report[obj_id] = {
                "frame_count": len(frames),
                "accepted_frame_count": len(frames),
                "mean_confidence": float(np.mean([frame.get("confidence", 0.0) for frame in frames])),
                "target_frame_mode": target_frame_mode,
                "target_frame_id": target_frame_id,
                "target_confidence": float(target_frame.get("confidence", 0.0) or 0.0),
                "target_metrics": target_record.get("metrics", {}),
                "pose_source": pose_strategy,
            }

        object_tracks_path = out_dir / "object_pose_tracks.json"
        per_frame_path = out_dir / "per_frame_object_poses.json"
        mesh_canon_path = out_dir / "mesh_canonicalization.json"
        quality_path = out_dir / "pose_quality_report.json"
        refined_traj_path = out_dir / "refined_object_trajectories.json"
        manifest_path = out_dir / "pose_optimizer_manifest.json"
        refined_object_trajectories = self._refined_trajectories_from_pose_tracks(object_pose_tracks)
        self._json_dump(object_tracks_path, object_pose_tracks)
        self._json_dump(per_frame_path, per_frame_object_poses)
        self._json_dump(mesh_canon_path, mesh_canonicalization)
        self._json_dump(quality_path, pose_quality_report)
        self._json_dump(refined_traj_path, refined_object_trajectories)
        self._json_dump(manifest_path, manifest)

        result_dir_count = sum(1 for item in results_dir.iterdir() if item.is_dir()) if results_dir.exists() else 0
        outputs = {
            "pose_optimizer_manifest": str(manifest_path),
            "object_pose_tracks": str(object_tracks_path),
            "per_frame_object_poses": str(per_frame_path),
            "mesh_canonicalization": str(mesh_canon_path),
            "pose_quality_report": str(quality_path),
            "refined_object_trajectories": str(refined_traj_path),
            "road_geometry": str(road_geometry_path),
            "tasks_dir": str(tasks_dir),
            "results_dir": str(results_dir),
        }
        summary = {
            "object_count": len(sam3d_meshes),
            "tracked_count": tracked_objects,
            "skipped_count": skipped_objects,
            "attempted_pose_frame_count": attempted_frames,
            "accepted_pose_frame_count": accepted_frames,
            "rejected_pose_frame_count": rejected_frames,
            "failed_pose_frame_count": failed_frames,
            "reused_pose_frame_count": reused_frames,
            "per_frame_count": len(per_frame_object_poses),
            "road_geometry_available": bool((road_geometry or {}).get("available")),
            "strategy": pose_strategy,
            "target_frame_mode": target_frame_mode,
            "target_frame_id": target_frame_id,
            "target_window_radius": target_window_radius,
            "dynamic_window_enabled": dynamic_window_enabled,
            "dynamic_window_min_radius": dynamic_window_min_radius,
            "dynamic_window_max_radius": dynamic_window_max_radius,
            "selected_target_objects": selected_target_objects,
            "target_object_ids": sorted(target_object_ids) if target_object_ids else None,
            "max_target_objects": max_target_objects,
            "resume_results": resume_results,
            "target_accepted_objects": target_accepted_objects,
            "pose_task_count": sum(1 for item in tasks_dir.iterdir() if item.is_dir()) if tasks_dir.exists() else 0,
            "pose_result_count": result_dir_count,
            "bbox_area_hard_filter": False,
        }
        return self._base_result(
            "pose.optimize",
            summary,
            outputs,
            params={
                "strategy": pose_strategy,
                "target_frame_mode": target_frame_mode,
                "target_frame_id": target_frame_id,
                "target_window_radius": target_window_radius,
                "dynamic_window_enabled": dynamic_window_enabled,
                "target_object_ids": sorted(target_object_ids) if target_object_ids else None,
                "max_target_objects": max_target_objects,
                "resume_results": resume_results,
            },
        )

    @staticmethod
    def _pose_track_scene_up(road_geometry: dict | None):
        import numpy as np

        candidates = []
        if isinstance(road_geometry, dict):
            for plane in road_geometry.get("planes", []) or []:
                normal = plane.get("normal_world")
                if isinstance(normal, list) and len(normal) == 3:
                    try:
                        candidates.append(np.asarray(normal, dtype=np.float64))
                    except Exception:
                        pass
            normal = road_geometry.get("normal_world")
            if isinstance(normal, list) and len(normal) == 3:
                try:
                    candidates.append(np.asarray(normal, dtype=np.float64))
                except Exception:
                    pass
        if candidates:
            up = np.mean(candidates, axis=0)
        else:
            up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(up))
        if norm < 1e-6:
            up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        else:
            up = up / norm
        return up

    @staticmethod
    def _pose_target_frame_id() -> int | None:
        for name in ("GUANWU_POSE_TARGET_FRAME_ID", "GUANWU_SCENE_FRAME_ID", "GUANWU_TARGET_FRAME_ID"):
            raw = os.environ.get(name)
            if raw is None or str(raw).strip() == "":
                continue
            try:
                value = int(str(raw).strip())
            except ValueError:
                continue
            return value if value > 0 else None
        return None

    @staticmethod
    def _pose_target_window_radius() -> int:
        raw = os.environ.get("GUANWU_POSE_TARGET_WINDOW_RADIUS", "2")
        try:
            return max(0, min(int(str(raw).strip()), 12))
        except ValueError:
            return 2

    @staticmethod
    def _pose_target_frame_mode() -> str:
        if str(os.environ.get("GUANWU_POSE_ALL_FRAMES", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return "all_frames"
        raw = str(os.environ.get("GUANWU_POSE_TARGET_FRAME_MODE", "")).strip().lower().replace("-", "_")
        if raw in {"all", "all_frame", "all_frames", "full", "full_video", "sequence"}:
            return "all_frames"
        return "single_frame"

    @staticmethod
    def _background_road_geometry_from_manifest(geometry) -> dict | None:
        manifest_path = None
        try:
            manifest_path = (geometry.outputs or {}).get("background_assets_manifest")
        except Exception:
            manifest_path = None
        if not manifest_path:
            return None
        path = Path(manifest_path)
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        plane = manifest.get("road_plane")
        if not isinstance(plane, dict):
            return None
        try:
            normal = [float(v) for v in plane.get("normal_world", [])[:3]]
            offset = float(plane.get("offset"))
        except Exception:
            return None
        if len(normal) != 3 or not all(math.isfinite(v) for v in normal) or not math.isfinite(offset):
            return None
        plane = dict(plane)
        plane["normal_world"] = normal
        plane["offset"] = offset
        plane.setdefault("source", "background_assets_manifest")
        plane.setdefault(
            "selection",
            {
                "mode": "global",
                "policy": "global_for_fixed_camera",
                "target_frame_id": int(manifest.get("target_frame_id") or 0),
            },
        )
        return {
            "available": True,
            "source": "background_assets_manifest",
            "background_assets_manifest": str(path),
            "default_plane_policy": "global_for_fixed_camera",
            "keyframe_planes": [],
            "planes": [],
            "global_plane": plane,
        }

    @staticmethod
    def _road_geometry_with_background_fallback(road_geometry: dict | None, geometry) -> dict:
        if isinstance(road_geometry, dict) and road_geometry.get("available") and road_geometry.get("global_plane"):
            return road_geometry
        fallback = ProjectExecutor._background_road_geometry_from_manifest(geometry)
        if fallback is None:
            return road_geometry if isinstance(road_geometry, dict) else {"available": False, "reason": "missing_road_geometry"}
        if isinstance(road_geometry, dict):
            fallback["fallback_from"] = {
                key: road_geometry.get(key)
                for key in ("available", "reason", "source", "depth_maps_dir")
                if key in road_geometry
            }
        return fallback

    @staticmethod
    def _pose_env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _pose_dynamic_window_bounds(base_radius: int) -> tuple[int, int]:
        def read_int(name: str, default: int) -> int:
            try:
                return int(str(os.environ.get(name, default)).strip())
            except ValueError:
                return int(default)

        min_radius = max(0, read_int("GUANWU_POSE_DYNAMIC_WINDOW_MIN_RADIUS", 1))
        max_radius = max(min_radius, read_int("GUANWU_POSE_DYNAMIC_WINDOW_MAX_RADIUS", max(4, int(base_radius))))
        return min_radius, min(max_radius, 12)

    @staticmethod
    def _pose_detection_entry_frame_id(entry: dict) -> int:
        try:
            return int(entry.get("frame_idx") or entry.get("frame_id") or 0)
        except Exception:
            return 0

    @staticmethod
    def _pose_all_frame_candidate_frame_ids(
        *,
        obj_id: str,
        detection_frames: list[dict],
        min_bbox_area_px: float,
        generic_mode: bool = False,
    ) -> list[int]:
        frame_range = ProjectExecutor._pose_frame_id_range_filter()
        frame_ids: list[int] = []
        for entry in detection_frames or []:
            if not isinstance(entry, dict):
                continue
            frame_id = ProjectExecutor._pose_detection_entry_frame_id(entry)
            if frame_id <= 0:
                continue
            if frame_range is not None and not (frame_range[0] <= frame_id <= frame_range[1]):
                continue
            instances = entry.get("instances")
            if instances is None:
                det_path = entry.get("detections")
                if det_path and Path(det_path).exists():
                    try:
                        with open(det_path, "r", encoding="utf-8") as f:
                            det = json.load(f)
                        instances = det.get("instances", []) if isinstance(det, dict) else []
                    except Exception:
                        instances = []
            for inst in instances or []:
                if not isinstance(inst, dict) or inst.get("object_id") != obj_id:
                    continue
                bbox = inst.get("bbox_xyxy") or inst.get("bbox")
                obs = {"bbox": bbox, "bbox_area_px": ProjectExecutor._bbox_area_px(bbox)}
                if obs["bbox_area_px"] < float(min_bbox_area_px):
                    continue
                if not ProjectExecutor._is_target_pose_candidate(inst, obs, generic_mode=generic_mode):
                    continue
                frame_ids.append(frame_id)
                break
        return sorted(set(frame_ids))

    @staticmethod
    def _pose_frame_id_range_filter() -> tuple[int, int] | None:
        raw = str(os.environ.get("GUANWU_POSE_FRAME_ID_RANGE", "") or "").strip()
        if not raw:
            return None
        text = raw.replace(":", "-").replace(",", "-")
        parts = [part.strip() for part in text.split("-") if part.strip()]
        if len(parts) != 2:
            return None
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            return None
        if start <= 0 or end <= 0:
            return None
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def _pose_dynamic_window_radius(
        inst: dict | None,
        obs: dict | None,
        *,
        base_radius: int,
        min_radius: int,
        max_radius: int,
    ) -> int:
        radius = int(base_radius)
        inst = inst if isinstance(inst, dict) else {}
        obs = obs if isinstance(obs, dict) else {}
        bbox = obs.get("bbox") or inst.get("bbox_xyxy") or inst.get("bbox")
        try:
            area = float(obs.get("bbox_area_px") or ProjectExecutor._bbox_area_px(bbox))
        except Exception:
            area = 0.0

        truncated = bool(inst.get("is_truncated") or inst.get("truncated"))
        truncation_info = inst.get("truncation_info")
        if isinstance(truncation_info, dict):
            truncated = truncated or bool(truncation_info.get("is_truncated") or truncation_info.get("touches_image_border"))
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                width = inst.get("image_width") or inst.get("width")
                height = inst.get("image_height") or inst.get("height")
                if (width is None or height is None) and inst.get("mask_rle"):
                    try:
                        rle = json.loads(inst["mask_rle"]) if isinstance(inst["mask_rle"], str) else inst["mask_rle"]
                        size = rle.get("size") if isinstance(rle, dict) else None
                        if isinstance(size, (list, tuple)) and len(size) >= 2:
                            height, width = int(size[0]), int(size[1])
                    except Exception:
                        pass
                if width is not None and height is not None:
                    w = float(width)
                    h = float(height)
                    truncated = truncated or x1 <= 1.0 or y1 <= 1.0 or x2 >= w - 1.0 or y2 >= h - 1.0
            except Exception:
                pass

        if area >= 12000.0 and not truncated:
            radius -= 1
        if area > 0.0 and area < 2500.0:
            radius += 1
        if truncated:
            radius += 1
        return max(int(min_radius), min(int(max_radius), int(radius)))

    @staticmethod
    def _pose_first_frame_truncation_skip_decision(inst: dict | None, *, frame_id: int) -> dict:
        inst = inst if isinstance(inst, dict) else {}
        truncated = bool(inst.get("is_truncated") or inst.get("truncated"))
        low_observability = False
        severity = ""
        sides: set[str] = set()

        truncation_info = inst.get("truncation_info")
        if isinstance(truncation_info, dict):
            truncated = truncated or bool(
                truncation_info.get("is_truncated")
                or truncation_info.get("touches_image_border")
                or truncation_info.get("low_observability")
            )
            low_observability = bool(truncation_info.get("low_observability"))
            severity = str(
                truncation_info.get("truncation_severity")
                or truncation_info.get("severity")
                or ""
            ).lower()
            raw_sides = truncation_info.get("truncation_sides") or truncation_info.get("sides") or []
            if isinstance(raw_sides, str):
                raw_sides = [raw_sides]
            if isinstance(raw_sides, (list, tuple, set)):
                sides.update(str(side).strip().lower() for side in raw_sides if str(side).strip())

        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                width = inst.get("image_width") or inst.get("width")
                height = inst.get("image_height") or inst.get("height")
                if (width is None or height is None) and inst.get("mask_rle"):
                    try:
                        rle = json.loads(inst["mask_rle"]) if isinstance(inst["mask_rle"], str) else inst["mask_rle"]
                        size = rle.get("size") if isinstance(rle, dict) else None
                        if isinstance(size, (list, tuple)) and len(size) >= 2:
                            height, width = int(size[0]), int(size[1])
                    except Exception:
                        pass
                if width is not None and height is not None:
                    w = float(width)
                    h = float(height)
                    if x1 <= 1.0:
                        sides.add("left")
                    if y1 <= 1.0:
                        sides.add("top")
                    if x2 >= w - 1.0:
                        sides.add("right")
                    if y2 >= h - 1.0:
                        sides.add("bottom")
                    truncated = truncated or bool(sides)
            except Exception:
                pass

        if not severity:
            if low_observability or len(sides) >= 2 or "bottom" in sides:
                severity = "severe"
            elif sides:
                severity = "light"
            elif truncated:
                severity = "unknown"

        if not truncated:
            return {"skip_object": False, "reason": "first_pose_frame_observable", "frame_id": int(frame_id)}
        return {
            "skip_object": True,
            "reason": "first_pose_frame_truncated",
            "frame_id": int(frame_id),
            "truncated_sides": sorted(sides),
            "truncation_severity": severity or "unknown",
            "low_observability": bool(low_observability),
        }

    def _build_pose_track_observations(
        self,
        *,
        obj_id: str,
        track_records: list[dict],
        detection_frames: list[dict],
        depth_maps_dir: str | None,
        cam_traj: list[dict],
        wildgs_poses: list[dict],
        wildgs_K: dict | None,
    ) -> list[dict]:
        observations: list[dict] = []
        for rec in track_records:
            frame_id = int(rec.get("frame_id") or 0)
            if frame_id <= 0:
                continue
            inst = self._get_instance_for_frame(obj_id, frame_id, detection_frames)
            mask_rle = inst.get("mask_rle") if inst else None
            points = None
            if depth_maps_dir and mask_rle:
                points = build_depth_point_cloud(
                    depth_maps_dir,
                    frame_id,
                    mask_rle,
                    cam_traj=cam_traj,
                    wildgs_poses=wildgs_poses,
                    wildgs_K=wildgs_K,
                )
            bbox = (inst or {}).get("bbox_xyxy") or (inst or {}).get("bbox")
            observations.append({
                "frame_id": frame_id,
                "timestamp_sec": float(rec.get("timestamp_sec", self._timestamp_for_frame(track_records, frame_id)) or 0.0),
                "bbox": bbox,
                "bbox_area_px": self._bbox_area_px(bbox),
                "mask_rle": mask_rle,
                "points": points,
                "depth_valid_points": int(len(points)) if points is not None else 0,
            })
        return observations

    @staticmethod
    def _pose_track_object_rotation(verts, observations: list[dict], *, scene_up):
        import numpy as np

        roles = dict(ProjectExecutor._infer_source_axis_roles(verts))
        extents = np.ptp(np.asarray(verts, dtype=np.float64), axis=0)
        order = np.argsort(extents)
        # The only reliable semantic prior for current SAM3D vehicle meshes:
        # local +Y points toward the roof/up. Keep this fixed everywhere.
        up_idx = 1
        forward_idx = int(roles.get("forward_axis_idx", int(order[-1])))
        if forward_idx == up_idx:
            forward_idx = int(order[-1] if int(order[-1]) != up_idx else order[-2])
        roles["up_axis_idx"] = up_idx
        roles["up_axis_sign"] = 1.0
        roles["forward_axis_idx"] = forward_idx
        # Front/back cannot be inferred reliably from bbox motion alone.  Use
        # the positive long axis as the primary hypothesis and keep the
        # opposite sign available for the optimizer.
        forward_sign = 1.0
        roles["forward_axis_sign"] = forward_sign

        local_up = np.zeros(3, dtype=np.float64)
        local_up[up_idx] = 1.0
        local_forward = np.zeros(3, dtype=np.float64)
        local_forward[forward_idx] = forward_sign
        local_right = np.cross(local_up, local_forward)
        right_norm = float(np.linalg.norm(local_right))
        if right_norm < 1e-6:
            local_right = np.eye(3, dtype=np.float64)[:, int([i for i in range(3) if i not in {up_idx, forward_idx}][0])]
        else:
            local_right = local_right / right_norm
        mesh_basis = np.stack([local_right, local_up, local_forward], axis=1)

        up = np.asarray(scene_up, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        centers = []
        for obs in observations:
            pts = obs.get("points")
            if pts is None:
                continue
            arr = trim_point_cloud_outliers(pts, min_keep=16)
            if arr.ndim == 2 and arr.shape[0] >= 16:
                centers.append(np.mean(arr, axis=0))
        heading = None
        source = "fallback"
        if len(centers) >= 2:
            delta = np.asarray(centers[-1], dtype=np.float64) - np.asarray(centers[0], dtype=np.float64)
            delta = delta - up * float(delta @ up)
            if float(np.linalg.norm(delta)) > 1e-4:
                heading = delta / float(np.linalg.norm(delta))
                source = "depth_centroid_motion"
        if heading is None:
            heading = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            heading = heading - up * float(heading @ up)
            if float(np.linalg.norm(heading)) < 1e-6:
                heading = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            heading = heading / float(np.linalg.norm(heading))
        right = np.cross(up, heading)
        right = right / max(float(np.linalg.norm(right)), 1e-8)
        heading = np.cross(right, up)
        heading = heading / max(float(np.linalg.norm(heading)), 1e-8)
        world_basis = np.stack([right, up, heading], axis=1)
        rotation = world_basis @ mesh_basis.T
        if np.linalg.det(rotation) < 0.0:
            world_basis[:, 0] *= -1.0
            rotation = world_basis @ mesh_basis.T
        heading_info = {
            "source": source,
            "world_forward": [float(v) for v in heading],
            "world_up": [float(v) for v in up],
        }
        return rotation, mesh_basis, world_basis, roles, heading_info

    @staticmethod
    def _pose_track_shared_scale(verts, observations: list[dict], *, mesh_basis, world_basis):
        import numpy as np

        source = np.asarray(verts, dtype=np.float64)
        source_local = source @ np.asarray(mesh_basis, dtype=np.float64)
        mesh_extents = np.maximum(np.ptp(source_local, axis=0), np.array([1e-6, 1e-6, 1e-6]))
        candidates = []
        for obs in observations:
            pts = obs.get("points")
            if pts is None:
                continue
            arr = trim_point_cloud_outliers(pts, min_keep=16)
            if arr.ndim != 2 or arr.shape[0] < 16:
                continue
            target_local = (arr - np.mean(arr, axis=0, keepdims=True)) @ np.asarray(world_basis, dtype=np.float64)
            target_extents = np.maximum(np.ptp(target_local, axis=0), np.array([1e-6, 1e-6, 1e-6]))
            ratios = target_extents / mesh_extents
            finite = ratios[np.isfinite(ratios) & (ratios > 0.01) & (ratios < 20.0)]
            if finite.size:
                candidates.append(float(np.median(finite)))
        if not candidates:
            return [1.0, 1.0, 1.0]
        scale = float(np.median(candidates))
        scale = min(max(scale, 0.08), 15.0)
        return [scale, scale, scale]

    def _pose_track_frames_from_observations(
        self,
        *,
        verts,
        observations: list[dict],
        rotation,
        scale,
        scene_up,
        road_geometry: dict | None,
    ) -> list[dict]:
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        scale_arr = np.asarray(scale, dtype=np.float64).reshape(3)
        up = np.asarray(scene_up, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        transformed_centered = (rot @ (np.asarray(verts, dtype=np.float64) * scale_arr[None, :]).T).T
        bottom_rel = float(np.min(transformed_centered @ up))

        frames: list[dict] = []
        valid_centers: list[tuple[int, np.ndarray]] = []
        for obs in observations:
            pts = obs.get("points")
            center = None
            confidence = 0.0
            quality = {
                "depth_valid_points": int(obs.get("depth_valid_points") or 0),
                "bbox_area_px": float(obs.get("bbox_area_px") or 0.0),
            }
            source = "interpolated"
            if pts is not None:
                arr = trim_point_cloud_outliers(pts, min_keep=16)
                if arr.ndim == 2 and arr.shape[0] >= 16:
                    center = np.mean(arr, axis=0)
                    low = float(np.percentile(arr @ up, 5.0))
                    center += up * (low - (float(center @ up) + bottom_rel))
                    road_plane = select_road_plane_for_frame(road_geometry, int(obs["frame_id"])) if road_geometry else None
                    if road_plane:
                        normal = np.asarray(road_plane.get("normal_world", up), dtype=np.float64)
                        normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
                        offset = float(road_plane.get("offset", 0.0))
                        bottom_plane_distance = float(normal @ center + offset + bottom_rel)
                        center = center - normal * bottom_plane_distance
                        quality["ground_error_m"] = abs(bottom_plane_distance)
                    depth_score = min(float(arr.shape[0]) / 512.0, 1.0)
                    area_score = min(float(obs.get("bbox_area_px") or 0.0) / 4000.0, 1.0)
                    confidence = float(max(0.05, 0.65 * depth_score + 0.35 * area_score))
                    source = "depth_temporal"
            if center is not None:
                valid_centers.append((int(obs["frame_id"]), center))
            frames.append({
                "frame_id": int(obs["frame_id"]),
                "timestamp_sec": float(obs.get("timestamp_sec", 0.0) or 0.0),
                "_center": center,
                "confidence": confidence,
                "source": source,
                "quality": quality,
            })

        if not valid_centers:
            return []
        for frame in frames:
            if frame["_center"] is None:
                frame["_center"] = self._interpolate_pose_track_center(int(frame["frame_id"]), valid_centers)
                frame["confidence"] = 0.25
                frame["source"] = "temporal_interpolated"
        quat = self._rotation_matrix_to_quat_xyzw(rot)
        for frame in frames:
            center = np.asarray(frame.pop("_center"), dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot
            T[:3, 3] = center
            frame["centroid_world"] = [float(v) for v in center]
            frame["rotation_matrix"] = rot.tolist()
            frame["orientation_quat"] = quat
            frame["scale"] = [float(v) for v in scale_arr]
            frame["T_world_from_object"] = T.tolist()
            frame["geometry_status"] = "depth_icp_temporal"
        return frames

    def _optimize_target_frame_vehicle_pose(
        self,
        *,
        obj_id: str,
        verts,
        observations: list[dict],
        base_rotation,
        base_scale,
        mesh_basis,
        axis_roles: dict,
        scene_up,
        road_geometry: dict | None,
        target_frame_id: int,
        window_radius: int,
        cam_traj: list[dict],
        wildgs_poses: list[dict],
        wildgs_K: dict | None,
        detection_frames: list[dict],
    ) -> dict | None:
        import math
        import numpy as np

        target_obs = next(
            (obs for obs in observations if int(obs.get("frame_id") or 0) == int(target_frame_id)),
            None,
        )
        if not target_obs or target_obs.get("points") is None:
            return None
        target_inst = self._get_instance_for_frame(obj_id, target_frame_id, detection_frames)
        if not self._is_target_frame_vehicle_candidate(target_inst, target_obs):
            return None

        target_points = trim_point_cloud_outliers(target_obs["points"], min_keep=16)
        if target_points.ndim != 2 or target_points.shape[0] < 16:
            return None

        window_obs = [
            obs for obs in observations
            if obs.get("points") is not None
            and abs(int(obs.get("frame_id") or 0) - int(target_frame_id)) <= int(window_radius)
        ]
        if not window_obs:
            window_obs = [target_obs]

        base_rot = np.asarray(base_rotation, dtype=np.float64)
        base_scale_arr = np.asarray(base_scale, dtype=np.float64).reshape(3)
        up = np.asarray(scene_up, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        source = np.asarray(verts, dtype=np.float64)
        source = source - source.mean(axis=0, keepdims=True)
        if source.shape[0] > 1200:
            source_fit = source[np.linspace(0, source.shape[0] - 1, 1200, dtype=int)]
        else:
            source_fit = source

        prior_candidates = []
        for obs in window_obs:
            pts = trim_point_cloud_outliers(obs["points"], min_keep=16)
            if pts.ndim != 2 or pts.shape[0] < 16:
                continue
            candidate = self._resolve_frame_alignment(
                source_fit,
                pts,
                anchor_rotation=base_rot,
                anchor_scale=base_scale_arr,
                traj=observations,
                scene_up=up,
                frame_id=int(obs["frame_id"]),
                mask_rle=obs.get("mask_rle"),
                cam_traj=cam_traj,
                wildgs_poses=wildgs_poses,
                wildgs_K=wildgs_K,
                source_axis_roles=axis_roles,
            )
            prior_candidates.append(candidate)

        if prior_candidates:
            finite_scales = [
                np.asarray(c.get("scale"), dtype=np.float64).reshape(3)
                for c in prior_candidates
                if self._valid_vec3_like(list(np.asarray(c.get("scale"), dtype=np.float64).reshape(-1)[:3]))
            ]
            if finite_scales:
                shared_scale = np.median(np.stack(finite_scales, axis=0), axis=0)
                shared_scale = np.clip(shared_scale, 0.08, 15.0)
            else:
                shared_scale = base_scale_arr
            target_prior = min(
                prior_candidates,
                key=lambda c: abs(int(target_frame_id) - int(c.get("frame_id", target_frame_id) or target_frame_id))
                if c.get("frame_id") is not None else 0,
            )
            seed_rotation = np.asarray(target_prior.get("rotation", base_rot), dtype=np.float64)
        else:
            shared_scale = base_scale_arr
            seed_rotation = base_rot

        seed_rotation = self._limit_rotation_to_vehicle_dofs(seed_rotation, base_rot, up, max_tilt_deg=10.0)
        yaw_candidates = self._target_pose_yaw_candidates(seed_rotation, up, degrees=(-16, -8, -4, 0, 4, 8, 16))
        pitch_roll_candidates = [
            np.eye(3, dtype=np.float64),
            self._axis_angle_rotation(self._horizontal_unit(seed_rotation[:, 0], scene_up=up), math.radians(3.0)),
            self._axis_angle_rotation(self._horizontal_unit(seed_rotation[:, 0], scene_up=up), math.radians(-3.0)),
            self._axis_angle_rotation(self._horizontal_unit(seed_rotation[:, 2], scene_up=up), math.radians(3.0)),
            self._axis_angle_rotation(self._horizontal_unit(seed_rotation[:, 2], scene_up=up), math.radians(-3.0)),
        ]

        best = None
        best_score = float("inf")
        for yaw_rot in yaw_candidates:
            for tilt_rot in pitch_roll_candidates:
                rotation = self._orthonormalize_rotation(tilt_rot @ yaw_rot)
                rotation = self._limit_rotation_to_vehicle_dofs(rotation, base_rot, up, max_tilt_deg=12.0)
                center = self._pose_center_from_depth_and_ground(
                    source,
                    target_points,
                    rotation,
                    shared_scale,
                    up,
                    road_geometry,
                    target_frame_id,
                )
                candidate = {
                    "center": center,
                    "rotation": rotation,
                    "scale": shared_scale,
                }
                losses = self._target_vehicle_pose_losses(
                    source,
                    candidate,
                    target_points,
                    target_obs=target_obs,
                    frame_id=target_frame_id,
                    cam_traj=cam_traj,
                    wildgs_poses=wildgs_poses,
                    wildgs_K=wildgs_K,
                    base_rotation=base_rot,
                    base_scale=base_scale_arr,
                    scene_up=up,
                    road_geometry=road_geometry,
                )
                weighted = (
                    1.0 * losses["depth_alignment"]
                    + 0.75 * losses["silhouette"]
                    + 0.45 * losses["ground_contact"]
                    + 0.35 * losses["temporal_smoothness"]
                    + 0.35 * losses["dimension_prior"]
                )
                if weighted < best_score:
                    best_score = weighted
                    best = {"candidate": candidate, "losses": losses, "weighted_loss": float(weighted)}

        if best is None:
            return None

        candidate = best["candidate"]
        rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        scale = np.asarray(candidate["scale"], dtype=np.float64).reshape(3)
        center = np.asarray(candidate["center"], dtype=np.float64).reshape(3)
        quat = self._rotation_matrix_to_quat_xyzw(rotation)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = center
        confidence = self._target_pose_confidence(best["losses"], int(target_points.shape[0]), target_obs)
        frame = {
            "frame_id": int(target_frame_id),
            "timestamp_sec": float(target_obs.get("timestamp_sec", 0.0) or 0.0),
            "centroid_world": [float(v) for v in center],
            "rotation_matrix": rotation.tolist(),
            "orientation_quat": quat,
            "scale": [float(v) for v in scale],
            "T_world_from_object": T.tolist(),
            "confidence": float(confidence),
            "source": "target_frame_temporal_refined",
            "geometry_status": "target_frame_temporal_refined",
            "quality": {
                "depth_valid_points": int(target_points.shape[0]),
                "bbox_area_px": float(target_obs.get("bbox_area_px") or 0.0),
                "losses": {key: float(value) for key, value in best["losses"].items()},
                "weighted_loss": float(best["weighted_loss"]),
                "window_radius": int(window_radius),
                "window_depth_frames": int(len(window_obs)),
            },
        }
        return {
            "frames": [frame],
            "summary": {
                "status": "refined",
                "target_frame_id": int(target_frame_id),
                "window_radius": int(window_radius),
                "window_depth_frames": int(len(window_obs)),
                "confidence": float(confidence),
                "weighted_loss": float(best["weighted_loss"]),
                "losses": {key: float(value) for key, value in best["losses"].items()},
            },
        }

    @staticmethod
    def _merge_pose_track_frames(frames: list[dict], replacements: list[dict]) -> list[dict]:
        by_id = {int(frame.get("frame_id") or 0): frame for frame in frames if int(frame.get("frame_id") or 0) > 0}
        for frame in replacements:
            fid = int(frame.get("frame_id") or 0)
            if fid > 0:
                by_id[fid] = frame
        return [by_id[fid] for fid in sorted(by_id)]

    @staticmethod
    def _is_target_frame_vehicle_candidate(inst: dict | None, obs: dict) -> bool:
        if not isinstance(inst, dict):
            return False
        label = str(
            inst.get("concept_label")
            or inst.get("label")
            or inst.get("class_name")
            or inst.get("category")
            or ""
        ).lower()
        vehicle_terms = ("car", "truck", "van", "bus", "vehicle", "pickup")
        static_terms = ("fence", "guardrail", "railroad", "streetlight", "road", "lane")
        if any(term in label for term in static_terms):
            return False
        if any(term in label for term in vehicle_terms):
            return True
        bbox = obs.get("bbox") or inst.get("bbox_xyxy") or inst.get("bbox")
        if not bbox or len(bbox) < 4:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        area = width * height
        if area < 250.0:
            return False
        if width <= 0.0 or height <= 0.0:
            return False
        aspect = width / max(height, 1.0)
        touches_top = y1 <= 1.0 and height <= 24.0
        wide_static = aspect > 4.2 and height < 55.0
        return not (touches_top or wide_static)

    @staticmethod
    def _is_target_pose_candidate(inst: dict | None, obs: dict, *, generic_mode: bool = False) -> bool:
        if not generic_mode:
            return ProjectExecutor._is_target_frame_vehicle_candidate(inst, obs)
        if not isinstance(inst, dict):
            return False
        label = str(
            inst.get("concept_label")
            or inst.get("label")
            or inst.get("class_name")
            or inst.get("category")
            or ""
        ).lower()
        static_terms = ("road", "lane", "sidewalk", "sky", "wall", "floor", "ceiling")
        if any(term in label for term in static_terms):
            return False
        bbox = obs.get("bbox") or inst.get("bbox_xyxy") or inst.get("bbox")
        if not bbox or len(bbox) < 4:
            return False
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return False
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width <= 0.0 or height <= 0.0:
            return False
        return width * height >= 250.0

    @staticmethod
    def _pose_center_from_depth_and_ground(source, target_points, rotation, scale, scene_up, road_geometry, frame_id):
        import numpy as np

        up = np.asarray(scene_up, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        source = np.asarray(source, dtype=np.float64)
        target = trim_point_cloud_outliers(target_points, min_keep=16)
        center = np.mean(target, axis=0)
        transformed = (np.asarray(rotation, dtype=np.float64) @ (source * np.asarray(scale, dtype=np.float64)[None, :]).T).T
        bottom_rel = float(np.min(transformed @ up))
        low = float(np.percentile(target @ up, 5.0))
        center = center + up * (low - (float(center @ up) + bottom_rel))
        road_plane = select_road_plane_for_frame(road_geometry, int(frame_id)) if road_geometry else None
        if road_plane:
            normal = np.asarray(road_plane.get("normal_world", up), dtype=np.float64)
            normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
            offset = float(road_plane.get("offset", 0.0))
            bottom_plane_distance = float(normal @ center + offset + bottom_rel)
            center = center - normal * bottom_plane_distance
        return center

    @staticmethod
    def _target_pose_yaw_candidates(seed_rotation, scene_up, *, degrees):
        import math

        candidates = []
        seen: set[tuple[float, ...]] = set()
        for deg in degrees:
            rot = ProjectExecutor._axis_angle_rotation(scene_up, math.radians(float(deg))) @ seed_rotation
            rot = ProjectExecutor._orthonormalize_rotation(rot)
            key = tuple(round(float(v), 6) for v in rot.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(rot)
        return candidates

    @staticmethod
    def _axis_angle_rotation(axis, angle_rad: float):
        import math
        import numpy as np

        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return np.eye(3, dtype=np.float64)
        x, y, z = axis / norm
        c = math.cos(float(angle_rad))
        s = math.sin(float(angle_rad))
        C = 1.0 - c
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ], dtype=np.float64)

    @staticmethod
    def _orthonormalize_rotation(rotation):
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        try:
            u, _, vt = np.linalg.svd(rot)
            out = u @ vt
            if np.linalg.det(out) < 0.0:
                u[:, -1] *= -1.0
                out = u @ vt
            return out
        except np.linalg.LinAlgError:
            return np.eye(3, dtype=np.float64)

    @staticmethod
    def _limit_rotation_to_vehicle_dofs(rotation, reference_rotation, scene_up, *, max_tilt_deg: float):
        import math
        import numpy as np

        rot = ProjectExecutor._orthonormalize_rotation(rotation)
        ref = ProjectExecutor._orthonormalize_rotation(reference_rotation)
        up = np.asarray(scene_up, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        forward = ProjectExecutor._horizontal_unit(rot[:, 2], scene_up=up)
        if float(np.linalg.norm(forward)) < 1e-6:
            forward = ProjectExecutor._horizontal_unit(ref[:, 2], scene_up=up)
        right = np.cross(up, forward)
        right = right / max(float(np.linalg.norm(right)), 1e-8)
        yaw_forward = np.cross(right, up)
        yaw_forward = yaw_forward / max(float(np.linalg.norm(yaw_forward)), 1e-8)
        yaw_only = np.stack([right, up, yaw_forward], axis=1)

        max_tilt = math.radians(float(max_tilt_deg))
        cur_up = rot[:, 1]
        tilt_axis = np.cross(up, cur_up)
        tilt_norm = float(np.linalg.norm(tilt_axis))
        if tilt_norm < 1e-8:
            return yaw_only
        tilt_angle = math.asin(max(-1.0, min(1.0, tilt_norm)))
        tilt_angle = max(-max_tilt, min(max_tilt, tilt_angle))
        tilt = ProjectExecutor._axis_angle_rotation(tilt_axis / tilt_norm, tilt_angle)
        return ProjectExecutor._orthonormalize_rotation(tilt @ yaw_only)

    @staticmethod
    def _target_vehicle_pose_losses(
        source,
        candidate: dict,
        target_points,
        *,
        target_obs: dict,
        frame_id: int,
        cam_traj: list[dict],
        wildgs_poses: list[dict],
        wildgs_K: dict | None,
        base_rotation,
        base_scale,
        scene_up,
        road_geometry: dict | None,
    ) -> dict[str, float]:
        import math
        import numpy as np
        from scipy.spatial import KDTree

        source_arr = np.asarray(source, dtype=np.float64)
        if source_arr.shape[0] > 900:
            source_fit = source_arr[np.linspace(0, source_arr.shape[0] - 1, 900, dtype=int)]
        else:
            source_fit = source_arr
        target = trim_point_cloud_outliers(target_points, min_keep=16)
        rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        scale = np.asarray(candidate["scale"], dtype=np.float64).reshape(3)
        center = np.asarray(candidate["center"], dtype=np.float64).reshape(3)
        world = (rotation @ (source_fit * scale[None, :]).T).T + center[None, :]

        if target.ndim == 2 and target.shape[0] >= 16:
            tree = KDTree(target)
            dists, _ = tree.query(world)
            depth_loss = float(np.median(np.clip(dists, 0.0, 5.0)))
        else:
            depth_loss = 5.0

        projection_score = ProjectExecutor._candidate_projection_score(
            source_arr,
            candidate,
            frame_id=frame_id,
            mask_rle=target_obs.get("mask_rle"),
            cam_traj=cam_traj,
            wildgs_poses=wildgs_poses,
            wildgs_K=wildgs_K,
        )
        silhouette_loss = 1.0 if not math.isfinite(float(projection_score)) else float(1.0 - max(0.0, min(1.0, projection_score)))

        transformed_full = (rotation @ (source_arr * scale[None, :]).T).T + center[None, :]
        up = np.asarray(scene_up, dtype=np.float64)
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        bottom = float(np.min(transformed_full @ up))
        ground_loss = 0.0
        road_plane = select_road_plane_for_frame(road_geometry, int(frame_id)) if road_geometry else None
        if road_plane:
            normal = np.asarray(road_plane.get("normal_world", up), dtype=np.float64)
            normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
            offset = float(road_plane.get("offset", 0.0))
            plane_values = transformed_full @ normal + offset
            ground_loss = abs(float(np.percentile(plane_values, 2.0)))
        else:
            target_low = float(np.percentile(target @ up, 5.0)) if target.ndim == 2 and target.shape[0] else bottom
            ground_loss = abs(bottom - target_low)

        temporal_loss = ProjectExecutor._rotation_geodesic_deg(rotation, base_rotation) / 45.0
        temporal_loss = min(max(float(temporal_loss), 0.0), 3.0)

        base_scale_arr = np.asarray(base_scale, dtype=np.float64).reshape(3)
        ratio = scale / np.maximum(base_scale_arr, 1e-6)
        dimension_loss = float(np.mean(np.abs(np.log(np.clip(ratio, 1e-3, 1e3)))))
        scale_ratio = float(np.max(scale) / max(float(np.min(scale)), 1e-6))
        dimension_loss += max(0.0, math.log(scale_ratio) - math.log(5.5))

        return {
            "depth_alignment": depth_loss,
            "silhouette": silhouette_loss,
            "ground_contact": min(float(ground_loss), 5.0),
            "temporal_smoothness": temporal_loss,
            "dimension_prior": dimension_loss,
            "projection_score": float(projection_score) if math.isfinite(float(projection_score)) else -1.0,
        }

    @staticmethod
    def _target_pose_confidence(losses: dict, point_count: int, obs: dict) -> float:
        import math

        weighted = (
            float(losses.get("depth_alignment", 1.0))
            + 0.75 * float(losses.get("silhouette", 1.0))
            + 0.45 * float(losses.get("ground_contact", 1.0))
            + 0.35 * float(losses.get("temporal_smoothness", 1.0))
            + 0.35 * float(losses.get("dimension_prior", 1.0))
        )
        depth_score = min(max(float(point_count) / 512.0, 0.0), 1.0)
        area_score = min(max(float(obs.get("bbox_area_px") or 0.0) / 4000.0, 0.0), 1.0)
        fit_score = math.exp(-max(weighted, 0.0))
        return max(0.05, min(0.99, 0.45 * fit_score + 0.35 * depth_score + 0.20 * area_score))

    @staticmethod
    def _interpolate_pose_track_center(frame_id: int, centers: list[tuple[int, object]]):
        import numpy as np

        ordered = sorted((int(fid), np.asarray(center, dtype=np.float64)) for fid, center in centers)
        if frame_id <= ordered[0][0]:
            return ordered[0][1].copy()
        if frame_id >= ordered[-1][0]:
            return ordered[-1][1].copy()
        for (left_id, left_center), (right_id, right_center) in zip(ordered, ordered[1:]):
            if left_id <= frame_id <= right_id:
                if right_id == left_id:
                    return left_center.copy()
                alpha = (frame_id - left_id) / float(right_id - left_id)
                return left_center * (1.0 - alpha) + right_center * alpha
        return ordered[-1][1].copy()


    # ------------------------------------------------------------------
    # scene.compose — anchor objects to world-space via ICP + correct trajectories
    # ------------------------------------------------------------------

    def _run_scene_compose(self) -> dict:
        import numpy as np
        import trimesh

        geometry = self.context.artifacts.get("geometry.lift")
        mesh_art = self.context.artifacts.get("mesh.reconstruct")
        if geometry is None or mesh_art is None:
            raise RuntimeError("geometry.lift and mesh.reconstruct outputs are required")

        out_dir = self.context.stage_output_dir("scene.compose")

        sam3d_meshes: dict = self._json_load(mesh_art.outputs["sam3d_meshes"])
        cam_traj: list = self._json_load(geometry.outputs["camera_trajectory"])
        obj_traj: dict = self._json_load(geometry.outputs["object_trajectories"])
        depth_maps_dir = geometry.outputs.get("wildgs_depth_maps")

        wildgs_poses, wildgs_K = self._load_wildgs_poses(geometry)
        bg_mesh_path = self._find_bg_mesh(geometry.outputs.get("wildgs_background_mesh"))
        pose_opt_artifact = self.context.artifacts.get("pose.optimize")
        pose_road_geometry_path = None if pose_opt_artifact is None else pose_opt_artifact.outputs.get("road_geometry")
        bg_meshes = load_background_asset_meshes(
            geometry.outputs.get("background_assets_manifest"),
            road_geometry_path=pose_road_geometry_path,
            camera_trajectory_path=geometry.outputs.get("camera_trajectory"),
        )
        if not bg_meshes:
            bg_meshes = self._find_bg_meshes(geometry.outputs.get("wildgs_background_mesh"))

        geo_summary = self._json_load(geometry.outputs["summary"])
        detection_frames = geo_summary.get("frames", [])
        object_nodes = {obj.object_id: obj for obj in self._all_objects()}
        pose_opt_manifest = self._load_pose_optimizer_manifest()

        scene = trimesh.Scene()
        corrected_trajectories: dict[str, list[dict]] = {}
        manifest: list[dict] = []
        traj_path = out_dir / "corrected_trajectories.json"
        raw_traj_path = out_dir / "corrected_trajectories.raw.json"
        smoothing_report_path = out_dir / "trajectory_smoothing_report.json"
        manifest_path = out_dir / "scene_manifest.json"
        frame_scene_manifest_path = out_dir / "frame_scene_manifest.json"

        scene_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if bg_meshes:
            for bg_name, bg_path in bg_meshes:
                bg = trimesh.load(str(bg_path), force="mesh")
                scene.add_geometry(bg, node_name=f"background_{bg_name}")
                if bg_name == "road":
                    scene_up = self._estimate_scene_up(np.asarray(bg.vertices, dtype=np.float64))
                _logger.info("[scene.compose] Background %s: %d verts", bg_name, len(bg.vertices))
        elif bg_mesh_path:
            bg = trimesh.load(str(bg_mesh_path), force="mesh")
            scene.add_geometry(bg, node_name="background")
            scene_up = self._estimate_scene_up(np.asarray(bg.vertices, dtype=np.float64))
            _logger.info("[scene.compose] Background: %d verts", len(bg.vertices))

        pose_tracks = self._load_pose_track_outputs()
        if pose_tracks:
            selected_frame_id = self._pose_target_frame_id()
            background_target_frame_id = self._background_assets_target_frame_id(geometry)
            if background_target_frame_id is not None:
                selected_frame_id = background_target_frame_id
            if selected_frame_id is None:
                selected_frame_id = self._select_pose_track_scene_frame(pose_tracks.get("per_frame", {}))
            scene_manifest = self._compose_scene_from_pose_tracks(
                scene=scene,
                sam3d_meshes=sam3d_meshes,
                pose_tracks=pose_tracks,
                selected_frame_id=selected_frame_id,
            )
            corrected_trajectories = scene_manifest["corrected_trajectories"]
            manifest = scene_manifest["manifest"]
            placed = scene_manifest["placed"]
            corrected_trajectories, smoothing_report = self._smooth_scene_corrected_trajectories(
                corrected_trajectories,
                raw_traj_path=raw_traj_path,
                smoothed_traj_path=traj_path,
                smoothing_report_path=smoothing_report_path,
            )
            self._json_dump(manifest_path, manifest)
            self._json_dump(frame_scene_manifest_path, scene_manifest["frame_manifest"])

            glb_path = out_dir / "composed_scene.glb"
            viewer_glb_path = out_dir / "composed_scene_viewer.glb"
            if len(scene.geometry) > 0:
                scene.export(str(glb_path))
                self._export_viewer_space_glb(scene, viewer_glb_path)
            else:
                glb_path.write_bytes(b"")
                viewer_glb_path.write_bytes(b"")
            outputs = {
                "composed_scene_glb": str(glb_path),
                "composed_scene_viewer_glb": str(viewer_glb_path),
                "corrected_trajectories": str(traj_path),
                "corrected_trajectories_raw": str(raw_traj_path),
                "trajectory_smoothing_report": str(smoothing_report_path),
                "scene_manifest": str(manifest_path),
                "frame_scene_manifest": str(frame_scene_manifest_path),
            }
            summary = {
                "placed_objects": placed,
                "total_objects": len(sam3d_meshes),
                "has_background": bool(bg_meshes) or bg_mesh_path is not None,
                "background_source": "background_assets" if bg_meshes else "wildgs",
                "pose_source": self._pose_source_from_tracks(pose_tracks),
                "selected_frame_id": selected_frame_id,
                "trajectory_smoothing": {
                    "corrected_translation_outliers": smoothing_report.get("corrected_translation_outliers", 0),
                    "corrected_rotation_outliers": smoothing_report.get("corrected_rotation_outliers", 0),
                    "max_translation_adjust_m": smoothing_report.get("max_translation_adjust_m", 0.0),
                    "max_rotation_adjust_deg": smoothing_report.get("max_rotation_adjust_deg", 0.0),
                },
            }
            return self._base_result(
                "scene.compose",
                summary,
                outputs,
                params={"selected_frame_id": selected_frame_id},
            )

        placed = 0
        if bg_mesh_path is None and not bg_meshes:
            bg_mesh_path = self._find_bg_mesh(geometry.outputs.get("wildgs_background_mesh"))
            if bg_mesh_path:
                bg = trimesh.load(str(bg_mesh_path), force="mesh")
                scene.add_geometry(bg, node_name="background")
                scene_up = self._estimate_scene_up(np.asarray(bg.vertices, dtype=np.float64))
                _logger.info("[scene.compose] Background: %d verts", len(bg.vertices))
        for obj_id, entry in sam3d_meshes.items():
            obj_node = object_nodes.get(obj_id)
            glb_path = self._find_glb(entry)
            if not glb_path:
                continue
            obj_mesh = self._load_trimesh(glb_path)
            if obj_mesh is None or len(obj_mesh.vertices) < 8:
                continue

            traj = obj_traj.get(obj_id, [])
            if self._track_is_low_quality_edge_fragment(obj_id, traj, detection_frames):
                _logger.info("[scene.compose] skip %s: low-quality edge-fragment track", obj_id)
                continue
            recon_frame = entry.get("reconstruction_frame_idx")

            verts = np.asarray(obj_mesh.vertices, dtype=np.float64)

            pose_opt = self._accepted_pose_optimizer_record(pose_opt_manifest, obj_id)
            if pose_opt is not None:
                center = [float(v) for v in pose_opt["translation_world"]]
                scale = [float(v) for v in pose_opt["scale"]]
                rotation = np.asarray(pose_opt["rotation_matrix"], dtype=np.float64)
                frame_id = int(pose_opt.get("frame_id") or recon_frame or 0)
                quat = self._rotation_matrix_to_quat_xyzw(rotation)
                source_traj = [rec for rec in (traj or []) if isinstance(rec, dict)]
                if not source_traj:
                    source_traj = [{
                        "frame_id": frame_id,
                        "timestamp_sec": self._timestamp_for_frame(traj, frame_id),
                        "centroid_world": center,
                    }]
                corrected = []
                for rec in source_traj:
                    try:
                        fid = int(rec.get("frame_id") or frame_id)
                    except Exception:
                        fid = frame_id
                    rec_center = rec.get("centroid_world")
                    frame_center = [float(v) for v in rec_center] if self._valid_vec3_like(rec_center) else center
                    try:
                        timestamp = float(rec.get("timestamp_sec", self._timestamp_for_frame(traj, fid)) or 0.0)
                    except Exception:
                        timestamp = self._timestamp_for_frame(traj, fid)
                    corrected.append({
                        "frame_id": fid,
                        "timestamp_sec": timestamp,
                        "centroid_world": frame_center,
                        "orientation_quat": quat,
                        "rotation_matrix": rotation.tolist(),
                        "scale": scale,
                        "geometry_status": "silhouette_pose_optimized",
                        "pose_optimizer": {
                            "report": pose_opt.get("report"),
                            "metrics": pose_opt.get("metrics", {}),
                            "reason": pose_opt.get("reason"),
                        },
                    })
                corrected.sort(key=lambda item: int(item.get("frame_id") or 0))
                corrected_trajectories[obj_id] = {
                    "frames": corrected,
                    "anchor_R": rotation.tolist(),
                    "mesh_basis": np.eye(3, dtype=np.float64).tolist(),
                    "axis_roles": self._infer_source_axis_roles(verts),
                    "pose_source": "edge_contour_fast",
                }
                verts_world = (rotation @ (verts * np.asarray(scale, dtype=np.float64)[None, :]).T).T + np.asarray(center, dtype=np.float64)
                obj_mesh.vertices = verts_world.astype(np.float32)
                scene.add_geometry(obj_mesh, node_name=obj_id)
                placed += 1
                manifest.append({
                    "object_id": obj_id,
                    "anchor_frame": frame_id,
                    "anchor_scale": [float(x) for x in scale],
                    "anchor_centroid": [float(x) for x in center],
                    "depth_frames_used": 0,
                    "num_trajectory_frames": len(corrected),
                    "vertex_count": len(obj_mesh.vertices),
                    "face_count": len(obj_mesh.faces),
                    "mesh_basis": np.eye(3, dtype=np.float64).tolist(),
                    "pose_source": "edge_contour_fast",
                    "pose_optimizer_report": pose_opt.get("report"),
                    "pose_optimizer_metrics": pose_opt.get("metrics", {}),
                })
                self._json_dump(traj_path, corrected_trajectories)
                self._json_dump(manifest_path, manifest)
                _logger.info("[scene.compose] placed %s with edge_contour_fast pose", obj_id)
                continue

            R_recon = self._get_wildgs_R(wildgs_poses, cam_traj, recon_frame)
            if R_recon is None:
                R_recon = np.eye(3)
            mesh_basis = np.eye(3, dtype=np.float64)
            verts_canonical = verts
            base_source_roles = self._infer_source_axis_roles(verts)

            # Find best anchor frame for scale estimation
            anchor_frame, anchor_pts = self._find_anchor_frame(
                obj_id, traj, detection_frames, depth_maps_dir,
                cam_traj, wildgs_poses, wildgs_K,
            )

            anchor_mask_rle = self._get_mask_rle_for_frame(obj_id, anchor_frame, detection_frames) if anchor_frame else None
            if anchor_pts is None or len(anchor_pts) < 32:
                _logger.info("[scene.compose] skip %s: no metric depth anchor", obj_id)
                manifest.append({
                    "object_id": obj_id,
                    "skipped": True,
                    "reason": "missing_metric_depth_anchor",
                })
                continue

            mesh_variants = [
                {
                    "name": "raw",
                    "basis": np.eye(3, dtype=np.float64),
                    "verts": verts,
                    "seed_rotations": [R_recon],
                },
            ]

            anchor_candidates = []
            for variant in mesh_variants:
                solution = self._solve_icp_alignment(
                    variant["verts"],
                    anchor_pts,
                    traj=traj,
                    scene_up=scene_up,
                    seed_rotations=variant.get("seed_rotations"),
                )
                candidate = dict(solution)
                candidate["center"] = np.asarray(solution["center"], dtype=np.float64)
                candidate["mesh_basis"] = np.asarray(variant["basis"], dtype=np.float64)
                candidate["mesh_verts"] = np.asarray(variant["verts"], dtype=np.float64)
                candidate["variant_name"] = variant["name"]
                candidate["projection_score"] = self._candidate_projection_score(
                    variant["verts"],
                    candidate,
                    frame_id=anchor_frame,
                    mask_rle=anchor_mask_rle,
                    cam_traj=cam_traj,
                    wildgs_poses=wildgs_poses,
                    wildgs_K=wildgs_K,
                )
                anchor_candidates.append(candidate)

            chosen = self._select_alignment_candidate(
                anchor_candidates,
                scene_up=scene_up,
                source_axis_roles=base_source_roles,
            )
            mesh_basis = np.asarray(chosen["mesh_basis"], dtype=np.float64)
            verts_canonical = np.asarray(chosen["mesh_verts"], dtype=np.float64)
            anchor_center = np.asarray(chosen["center"], dtype=np.float64).tolist()
            anchor_scale = np.asarray(chosen["scale"], dtype=np.float64).tolist()
            anchor_R = np.asarray(chosen["rotation"], dtype=np.float64)
            _logger.info(
                "[scene.compose] %s: anchor frame=%s variant=%s proj=%.4f scale=%s",
                obj_id,
                anchor_frame,
                chosen.get("variant_name"),
                float(chosen.get("projection_score", float("-inf"))),
                anchor_scale,
            )

            anchor_axis_roles = dict(base_source_roles)
            if anchor_axis_roles:
                up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
                up_norm = float(np.linalg.norm(up_ref))
                if up_norm < 1e-6:
                    up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                else:
                    up_ref = up_ref / up_norm
                up_axis_idx = anchor_axis_roles.get("up_axis_idx")
                if up_axis_idx is not None:
                    up_axis = anchor_R[:, int(up_axis_idx)]
                    anchor_axis_roles["up_axis_sign"] = 1.0 if float(up_axis @ up_ref) >= 0.0 else -1.0
            else:
                anchor_axis_roles = self._reference_axis_roles_from_rotation(
                    anchor_R,
                    scene_up=scene_up,
                    source_points=verts_canonical,
                )

            anchor_point_cloud = None
            if anchor_pts is not None and len(anchor_pts) >= 32:
                anchor_point_cloud = trim_point_cloud_outliers(
                    np.asarray(anchor_pts, dtype=np.float64) - np.asarray(anchor_center, dtype=np.float64),
                    min_keep=32,
                )

            # Per-frame trajectory: use depth+mask point cloud only where it is
            # available. Missing frames remain unknown instead of receiving
            # synthetic/interpolated poses.
            corrected = []
            depth_rotations: dict[int, list[list[float]]] = {}  # frame_id → world rotation from depth
            previous_depth_rotation = None
            previous_depth_points = None
            for rec in traj:
                fid = rec.get("frame_id")
                mask_rle = self._get_mask_rle_for_frame(obj_id, fid, detection_frames)
                if mask_rle and depth_maps_dir:
                    pts = build_depth_point_cloud(
                        depth_maps_dir,
                        fid,
                        mask_rle,
                        cam_traj=cam_traj,
                        wildgs_poses=wildgs_poses,
                        wildgs_K=wildgs_K,
                    )
                    if pts is not None and len(pts) >= 16:
                        frame_alignment = self._resolve_frame_alignment(
                            verts_canonical,
                            pts,
                            anchor_rotation=anchor_R,
                            anchor_scale=anchor_scale,
                            anchor_points=anchor_point_cloud,
                            traj=traj,
                            scene_up=scene_up,
                            frame_id=fid,
                            mask_rle=mask_rle,
                            cam_traj=cam_traj,
                            wildgs_poses=wildgs_poses,
                            wildgs_K=wildgs_K,
                            previous_rotation=previous_depth_rotation,
                            previous_points=previous_depth_points,
                            source_axis_roles=anchor_axis_roles,
                        )
                        center = frame_alignment["center"].tolist()
                        scale = frame_alignment["scale"].tolist()
                        depth_rotations[fid] = frame_alignment["rotation"].tolist()
                        previous_depth_rotation = np.asarray(frame_alignment["rotation"], dtype=np.float64)
                        previous_depth_points = np.asarray(pts, dtype=np.float64)
                        rotation = depth_rotations[fid]
                        quat = self._rotation_matrix_to_quat_xyzw(rotation)
                        corrected.append({
                            "frame_id": fid,
                            "timestamp_sec": rec.get("timestamp_sec", 0.0),
                            "centroid_world": center,
                            "orientation_quat": quat,
                            "rotation_matrix": rotation,
                            "scale": scale,
                            "geometry_status": "metric_depth",
                        })

            if not corrected:
                _logger.info("[scene.compose] skip %s: no metric trajectory frames", obj_id)
                manifest.append({
                    "object_id": obj_id,
                    "skipped": True,
                    "reason": "missing_metric_trajectory",
                })
                continue

            corrected_trajectories[obj_id] = {
                "frames": corrected,
                "anchor_R": anchor_R.tolist(),
                "mesh_basis": np.asarray(mesh_basis, dtype=np.float64).tolist(),
                "axis_roles": anchor_axis_roles,
            }

            # Place at anchor for GLB snapshot (apply ICP rotation + scale + translation)
            verts_world = (anchor_R @ (verts_canonical * np.asarray(anchor_scale)[None, :]).T).T + np.array(anchor_center)
            obj_mesh.vertices = verts_world.astype(np.float32)
            scene.add_geometry(obj_mesh, node_name=obj_id)
            placed += 1
            manifest.append({
                "object_id": obj_id,
                "anchor_frame": anchor_frame,
                "anchor_scale": [float(x) for x in anchor_scale],
                "anchor_centroid": [float(x) for x in anchor_center],
                "depth_frames_used": len(corrected),
                "num_trajectory_frames": len(corrected),
                "vertex_count": len(obj_mesh.vertices),
                "face_count": len(obj_mesh.faces),
                "mesh_basis": np.asarray(mesh_basis, dtype=np.float64).tolist(),
            })
            self._json_dump(traj_path, corrected_trajectories)
            self._json_dump(manifest_path, manifest)
            _logger.info("[scene.compose] placed %s (%d/%d)", obj_id, placed, len(sam3d_meshes))

        # Export GLB
        glb_path = out_dir / "composed_scene.glb"
        viewer_glb_path = out_dir / "composed_scene_viewer.glb"
        if len(scene.geometry) > 0:
            scene.export(str(glb_path))
            self._export_viewer_space_glb(scene, viewer_glb_path)
        else:
            glb_path.write_bytes(b"")
            viewer_glb_path.write_bytes(b"")

        corrected_trajectories, smoothing_report = self._smooth_scene_corrected_trajectories(
            corrected_trajectories,
            raw_traj_path=raw_traj_path,
            smoothed_traj_path=traj_path,
            smoothing_report_path=smoothing_report_path,
        )

        self._json_dump(manifest_path, manifest)

        outputs = {
            "composed_scene_glb": str(glb_path),
            "composed_scene_viewer_glb": str(viewer_glb_path),
            "corrected_trajectories": str(traj_path),
            "corrected_trajectories_raw": str(raw_traj_path),
            "trajectory_smoothing_report": str(smoothing_report_path),
            "scene_manifest": str(manifest_path),
        }
        summary = {
            "placed_objects": placed,
            "total_objects": len(sam3d_meshes),
            "has_background": bg_mesh_path is not None,
            "trajectory_smoothing": {
                "corrected_translation_outliers": smoothing_report.get("corrected_translation_outliers", 0),
                "corrected_rotation_outliers": smoothing_report.get("corrected_rotation_outliers", 0),
                "max_translation_adjust_m": smoothing_report.get("max_translation_adjust_m", 0.0),
                "max_rotation_adjust_deg": smoothing_report.get("max_rotation_adjust_deg", 0.0),
            },
        }
        return self._base_result("scene.compose", summary, outputs)

    # ── scene.compose helpers ──

    def _smooth_scene_corrected_trajectories(
        self,
        corrected_trajectories: dict,
        *,
        raw_traj_path: Path,
        smoothed_traj_path: Path,
        smoothing_report_path: Path,
    ) -> tuple[dict, dict]:
        self._json_dump(raw_traj_path, corrected_trajectories)
        smoothed, report = smooth_object_trajectories(corrected_trajectories)
        self._json_dump(smoothed_traj_path, smoothed)
        self._json_dump(smoothing_report_path, report)
        _logger.info(
            "[scene.compose] trajectory smoothing: translation_outliers=%s rotation_outliers=%s max_translation=%.3fm max_rotation=%.2fdeg",
            report.get("corrected_translation_outliers", 0),
            report.get("corrected_rotation_outliers", 0),
            float(report.get("max_translation_adjust_m", 0.0) or 0.0),
            float(report.get("max_rotation_adjust_deg", 0.0) or 0.0),
        )
        return smoothed, report

    @staticmethod
    def _scene_glb_viewer_coordinate_convention() -> USDCoordinateConvention:
        return USDCoordinateConvention(
            R_usd_from_world=np.diag([1.0, -1.0, -1.0]).astype(np.float64),
            scene_up_world=np.array([0.0, -1.0, 0.0], dtype=np.float64),
            scene_forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            ground_plane_offset_world=0.0,
        )

    @staticmethod
    def _export_viewer_space_glb(scene, output_path: str | Path) -> None:
        import trimesh

        output_path = Path(output_path)
        convention = ProjectExecutor._scene_glb_viewer_coordinate_convention()
        viewer_scene = trimesh.Scene()
        for name, geometry in scene.geometry.items():
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            mesh = geometry.copy()
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if verts.ndim == 2 and verts.shape[1] == 3 and len(verts) > 0:
                mesh.vertices = convert_world_points_to_usd(verts, convention).astype(np.float32)
            normals = getattr(mesh, "vertex_normals", None)
            if normals is not None and len(normals) == len(mesh.vertices):
                try:
                    mesh.vertex_normals = convert_world_normals_to_usd(normals, convention)
                except Exception:
                    pass
            viewer_scene.add_geometry(mesh, node_name=str(name))
        if len(viewer_scene.geometry) > 0:
            viewer_scene.export(str(output_path))
        else:
            output_path.write_bytes(b"")

    def _load_pose_track_outputs(self) -> dict | None:
        artifact = self.context.artifacts.get("pose.optimize")
        if artifact is None:
            return None
        object_tracks_path = artifact.outputs.get("object_pose_tracks")
        per_frame_path = artifact.outputs.get("per_frame_object_poses")
        if not object_tracks_path or not per_frame_path:
            return None
        if not Path(object_tracks_path).exists() or not Path(per_frame_path).exists():
            return None
        try:
            object_tracks = self._json_load(object_tracks_path)
            per_frame = self._json_load(per_frame_path)
        except Exception:
            return None
        if not isinstance(object_tracks, dict) or not isinstance(per_frame, dict):
            return None
        return {"object_tracks": object_tracks, "per_frame": per_frame}

    @staticmethod
    def _select_pose_track_scene_frame(per_frame: dict) -> int:
        best_frame = 1
        best_count = -1
        for key, value in per_frame.items():
            if not isinstance(value, dict):
                continue
            try:
                frame_id = int(str(key).split("_")[-1])
            except Exception:
                continue
            count = sum(
                1
                for pose in value.values()
                if isinstance(pose, dict) and float(pose.get("confidence", 0.0) or 0.0) > 0.0
            )
            if count > best_count or (count == best_count and frame_id < best_frame):
                best_frame = frame_id
                best_count = count
        return best_frame

    @staticmethod
    def _pose_source_from_tracks(pose_tracks: dict) -> str:
        object_tracks = pose_tracks.get("object_tracks", {}) if isinstance(pose_tracks, dict) else {}
        for track in object_tracks.values():
            if isinstance(track, dict):
                source = track.get("pose_source")
                if source:
                    return str(source)
                frames = track.get("frames")
                if isinstance(frames, list):
                    for frame in frames:
                        if isinstance(frame, dict) and frame.get("source"):
                            return str(frame["source"])
        return "depth_icp_temporal"

    def _compose_scene_from_pose_tracks(self, *, scene, sam3d_meshes: dict, pose_tracks: dict, selected_frame_id: int) -> dict:
        import numpy as np

        object_tracks: dict = pose_tracks.get("object_tracks", {})
        per_frame: dict = pose_tracks.get("per_frame", {})
        frame_key = f"frame_{int(selected_frame_id):06d}"
        frame_poses: dict = per_frame.get(frame_key, {})
        pose_source = self._pose_source_from_tracks(pose_tracks)

        corrected_trajectories: dict[str, dict] = {}
        manifest: list[dict] = []
        frame_manifest = {
            "schema": "guanwu.frame_scene.v1",
            "frame_id": int(selected_frame_id),
            "pose_source": pose_source,
            "objects": [],
        }
        placed = 0

        for obj_id, track in object_tracks.items():
            if not isinstance(track, dict):
                continue
            entry = sam3d_meshes.get(obj_id)
            if not isinstance(entry, dict):
                continue
            glb_path = self._find_glb(entry)
            if not glb_path:
                continue
            obj_mesh = self._load_trimesh(glb_path)
            if obj_mesh is None or len(obj_mesh.vertices) < 8:
                continue

            frames = [frame for frame in track.get("frames", []) if isinstance(frame, dict)]
            compat_frames = []
            for frame in frames:
                if not self._valid_vec3_like(frame.get("centroid_world")) or not self._valid_vec3_like(frame.get("scale")):
                    continue
                rotation = frame.get("rotation_matrix") or track.get("rotation_matrix")
                if not isinstance(rotation, list) or len(rotation) != 3:
                    rotation = np.eye(3, dtype=np.float64).tolist()
                compat_frames.append({
                    "frame_id": int(frame.get("frame_id") or 0),
                    "timestamp_sec": float(frame.get("timestamp_sec", 0.0) or 0.0),
                    "centroid_world": [float(v) for v in frame["centroid_world"]],
                    "orientation_quat": frame.get("orientation_quat") or self._rotation_matrix_to_quat_xyzw(rotation),
                    "rotation_matrix": rotation,
                    "scale": [float(v) for v in frame["scale"]],
                    "geometry_status": frame.get("geometry_status", "depth_icp_temporal"),
                    "confidence": float(frame.get("confidence", 0.0) or 0.0),
                    "source": frame.get("source", "depth_icp_temporal"),
                    "quality": frame.get("quality", {}),
                })
            if not compat_frames:
                continue

            corrected_trajectories[obj_id] = {
                "frames": compat_frames,
                "anchor_R": track.get("rotation_matrix"),
                "mesh_basis": track.get("mesh_basis"),
                "axis_roles": track.get("axis_roles", {}),
                "pose_source": track.get("pose_source", pose_source),
            }

            frame_pose = frame_poses.get(obj_id)
            if not isinstance(frame_pose, dict):
                if frame_key in per_frame:
                    continue
                frame_pose = min(
                    compat_frames,
                    key=lambda item: abs(int(item.get("frame_id") or 0) - int(selected_frame_id)),
                )
            center = np.asarray(frame_pose.get("centroid_world"), dtype=np.float64).reshape(3)
            rotation = np.asarray(frame_pose.get("rotation_matrix", track.get("rotation_matrix", np.eye(3))), dtype=np.float64)
            if rotation.shape != (3, 3):
                rotation = np.eye(3, dtype=np.float64)
            scale = np.asarray(frame_pose.get("scale", track.get("scale", [1.0, 1.0, 1.0])), dtype=np.float64).reshape(3)

            verts = np.asarray(obj_mesh.vertices, dtype=np.float64)
            verts_world = (rotation @ (verts * scale[None, :]).T).T + center[None, :]
            obj_mesh.vertices = verts_world.astype(np.float32)
            scene.add_geometry(obj_mesh, node_name=obj_id)
            placed += 1

            manifest.append({
                "object_id": obj_id,
                "anchor_frame": int(frame_pose.get("frame_id", selected_frame_id) or selected_frame_id),
                "anchor_scale": [float(v) for v in scale],
                "anchor_centroid": [float(v) for v in center],
                "depth_frames_used": sum(1 for frame in compat_frames if frame.get("source") == "depth_temporal"),
                "num_trajectory_frames": len(compat_frames),
                "vertex_count": len(obj_mesh.vertices),
                "face_count": len(obj_mesh.faces),
                "mesh_basis": track.get("mesh_basis"),
                "pose_source": frame_pose.get("source", track.get("pose_source", pose_source)),
                "confidence": float(frame_pose.get("confidence", 0.0) or 0.0),
            })
            frame_manifest["objects"].append({
                "object_id": obj_id,
                "mesh_path": str(glb_path),
                "centroid_world": [float(v) for v in center],
                "rotation_matrix": rotation.tolist(),
                "scale": [float(v) for v in scale],
                "confidence": float(frame_pose.get("confidence", 0.0) or 0.0),
                "source": frame_pose.get("source", "depth_icp_temporal"),
            })

        return {
            "corrected_trajectories": corrected_trajectories,
            "manifest": manifest,
            "frame_manifest": frame_manifest,
            "placed": placed,
        }

    def _best_pose_optimize_frame(self, obj_id: str, traj: list[dict], detection_frames: list[dict]) -> int:
        best_frame = 0
        best_score = -1.0
        for rec in traj or []:
            fid = int(rec.get("frame_id") or 0)
            if fid <= 0:
                continue
            inst = self._get_instance_for_frame(obj_id, fid, detection_frames)
            if not inst:
                continue
            bbox = inst.get("bbox_xyxy") or inst.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            if len(bbox) < 4:
                continue
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            score = float(inst.get("score", rec.get("geom_quality", 0.0)) or 0.0) * max(area, 1.0)
            if score > best_score:
                best_frame = fid
                best_score = score
        return best_frame

    def _select_dense_detection_frame(self, detection_frames: list[dict]) -> int:
        best_frame = 1
        best_score = -1.0
        for entry in detection_frames or []:
            frame_id = int(entry.get("frame_idx") or 0)
            det_path = entry.get("detections")
            if frame_id <= 0 or not det_path or not Path(det_path).exists():
                continue
            try:
                det = self._json_load(det_path)
            except Exception:
                continue
            score = 0.0
            for inst in det.get("instances", []) or []:
                bbox = inst.get("bbox_xyxy") or inst.get("bbox")
                obs = {"bbox": bbox, "bbox_area_px": self._bbox_area_px(bbox)}
                if self._is_target_frame_vehicle_candidate(inst, obs):
                    score += max(1.0, self._bbox_area_px(bbox))
            if score > best_score:
                best_frame = frame_id
                best_score = score
        return best_frame

    def _pose_temporal_window_frame_ids(
        self,
        *,
        obj_id: str,
        target_frame_id: int,
        window_radius: int,
        detection_frames: list[dict],
    ) -> list[int]:
        start = max(1, int(target_frame_id) - int(window_radius))
        end = int(target_frame_id) + int(window_radius)
        frame_ids = []
        for frame_id in range(start, end + 1):
            inst = self._get_instance_for_frame(obj_id, frame_id, detection_frames)
            if not isinstance(inst, dict):
                continue
            bbox = inst.get("bbox_xyxy") or inst.get("bbox")
            if self._bbox_area_px(bbox) <= 0.0:
                continue
            frame_ids.append(frame_id)
        return sorted(set(frame_ids))

    def _build_edge_pose_seed_track(
        self,
        *,
        obj_id: str,
        frame_ids: list[int],
        verts,
        obj_traj: dict,
        detection_frames: list[dict],
        depth_maps_dir: str | None,
        cam_traj: list[dict],
        wildgs_poses: list[dict],
        wildgs_K: dict | None,
        scene_up,
        road_geometry: dict | None,
    ) -> tuple[list[dict], dict]:
        import numpy as np

        source_track = [
            rec for rec in obj_traj.get(obj_id, [])
            if isinstance(rec, dict) and int(rec.get("frame_id") or 0) in set(frame_ids)
        ]
        if not source_track:
            source_track = [{"frame_id": int(fid), "timestamp_sec": self._timestamp_for_frame([], int(fid))} for fid in frame_ids]

        observations = self._build_pose_track_observations(
            obj_id=obj_id,
            track_records=source_track,
            detection_frames=detection_frames,
            depth_maps_dir=depth_maps_dir,
            cam_traj=cam_traj,
            wildgs_poses=wildgs_poses,
            wildgs_K=wildgs_K,
        )
        depth_observations = [obs for obs in observations if obs.get("points") is not None]
        axis_roles = self._infer_source_axis_roles(verts)
        mesh_basis = np.eye(3, dtype=np.float64)
        heading_info = {"source": "fallback", "world_up": [float(v) for v in np.asarray(scene_up, dtype=np.float64)]}
        seed_frames: list[dict] = []
        if depth_observations:
            rotation, mesh_basis, _world_basis, axis_roles, heading_info = self._pose_track_object_rotation(
                verts,
                depth_observations,
                scene_up=scene_up,
            )
            scale = self._pose_track_shared_scale(
                verts,
                depth_observations,
                mesh_basis=mesh_basis,
                world_basis=_world_basis,
            )
            seed_frames = self._pose_track_frames_from_observations(
                verts=verts,
                observations=observations,
                rotation=rotation,
                scale=scale,
                scene_up=scene_up,
                road_geometry=road_geometry,
            )

        seed_by_frame = {int(frame.get("frame_id") or 0): frame for frame in seed_frames if isinstance(frame, dict)}
        fallback_scale = [1.0, 1.0, 1.0]
        fallback_rotation = np.eye(3, dtype=np.float64).tolist()
        out_track = []
        for fid in frame_ids:
            seed = seed_by_frame.get(int(fid))
            source = next((rec for rec in source_track if int(rec.get("frame_id") or 0) == int(fid)), {})
            if seed:
                out_track.append({
                    "frame_id": int(fid),
                    "timestamp_sec": float(seed.get("timestamp_sec", self._timestamp_for_frame(source_track, int(fid))) or 0.0),
                    "centroid_world": seed.get("centroid_world"),
                    "scale": seed.get("scale", fallback_scale),
                    "orientation_quat": seed.get("orientation_quat"),
                    "rotation_matrix": seed.get("rotation_matrix", fallback_rotation),
                })
                fallback_scale = seed.get("scale", fallback_scale)
                fallback_rotation = seed.get("rotation_matrix", fallback_rotation)
                continue
            centroid = source.get("centroid_world")
            if not self._valid_vec3_like(centroid):
                camera = self._projection_camera(int(fid), cam_traj, wildgs_poses, wildgs_K)
                centroid = camera["t"].tolist() if camera is not None else [0.0, 0.0, 0.0]
            out_track.append({
                "frame_id": int(fid),
                "timestamp_sec": float(source.get("timestamp_sec", self._timestamp_for_frame(source_track, int(fid))) or 0.0),
                "centroid_world": [float(v) for v in centroid],
                "scale": source.get("scale") if self._valid_vec3_like(source.get("scale")) else fallback_scale,
                "orientation_quat": source.get("orientation_quat"),
                "rotation_matrix": fallback_rotation,
            })
        return out_track, {
            "mesh_basis": np.asarray(mesh_basis, dtype=np.float64).tolist(),
            "axis_roles": axis_roles,
            "heading": heading_info,
        }

    @staticmethod
    def _pose_optimizer_temporal_jump_acceptance(
        report: dict,
        previous: dict | None,
        *,
        generic_mode: bool = False,
    ) -> dict:
        import numpy as np

        if previous is None:
            return {"accepted": True, "reason": "accepted"}
        pose = report.get("optimized_corrected_pose_world", {})
        prev_pose = previous.get("pose", {}) if isinstance(previous, dict) else {}
        if not ProjectExecutor._valid_vec3_like(pose.get("translation_world")):
            return {"accepted": False, "reason": "invalid_translation"}
        if not ProjectExecutor._valid_vec3_like(prev_pose.get("translation_world")):
            return {"accepted": True, "reason": "accepted"}
        frame_id = int(report.get("frame_idx") or report.get("frame_id") or 0)
        prev_frame_id = int(previous.get("frame_id") or frame_id)
        dt = max(1, abs(frame_id - prev_frame_id))
        translation = np.asarray(pose.get("translation_world"), dtype=np.float64)
        prev_translation = np.asarray(prev_pose.get("translation_world"), dtype=np.float64)
        jump_m = float(np.linalg.norm(translation - prev_translation))
        translation_limit = max(4.0, 2.2 * dt) if not generic_mode else max(4.0, 2.5 * dt)
        if jump_m > translation_limit:
            return {"accepted": False, "reason": f"temporal_translation_jump:{jump_m:.3f}m"}

        rotation = pose.get("rotation_matrix")
        prev_rotation = prev_pose.get("rotation_matrix")
        if isinstance(rotation, list) and isinstance(prev_rotation, list):
            jump_deg = ProjectExecutor._rotation_geodesic_deg(rotation, prev_rotation)
            rotation_limit = max(55.0, 35.0 * dt) if not generic_mode else max(90.0, 55.0 * dt)
            if math.isfinite(jump_deg) and jump_deg > rotation_limit:
                return {"accepted": False, "reason": f"temporal_rotation_jump:{jump_deg:.1f}deg"}
            if not generic_mode:
                up_axis = ProjectExecutor._pose_report_axis_idx(report, "up_axis_idx", "shortest_axis")
                if up_axis is not None:
                    try:
                        cur = np.asarray(rotation, dtype=np.float64)[:, int(up_axis)]
                        prev = np.asarray(prev_rotation, dtype=np.float64)[:, int(up_axis)]
                        dot = float(cur @ prev) / max(float(np.linalg.norm(cur) * np.linalg.norm(prev)), 1e-8)
                        dot = max(-1.0, min(1.0, dot))
                        up_jump_deg = math.degrees(math.acos(dot))
                        if up_jump_deg > max(40.0, 25.0 * dt):
                            return {"accepted": False, "reason": f"temporal_up_flip:{up_jump_deg:.1f}deg"}
                    except Exception:
                        return {"accepted": False, "reason": "invalid_temporal_up_axis"}

        scale = pose.get("scale")
        prev_scale = prev_pose.get("scale")
        if ProjectExecutor._valid_vec3_like(scale) and ProjectExecutor._valid_vec3_like(prev_scale):
            scale_arr = np.asarray(scale, dtype=np.float64)
            prev_scale_arr = np.asarray(prev_scale, dtype=np.float64)
            ratio = scale_arr / np.maximum(prev_scale_arr, 1e-6)
            max_ratio = float(max(np.max(ratio), np.max(1.0 / np.maximum(ratio, 1e-6))))
            max_scale_ratio = 2.2 if generic_mode else 1.8
            if max_ratio > max_scale_ratio:
                return {"accepted": False, "reason": f"temporal_scale_jump:{max_ratio:.3f}x"}
        return {"accepted": True, "reason": "accepted"}

    def _edge_pose_record_from_report(
        self,
        *,
        obj_id: str,
        frame_id: int,
        report: dict,
        report_path: Path,
        result_dir: Path,
        task_path: Path,
        timestamp_sec: float,
        run_info: dict,
    ) -> dict:
        pose = report.get("optimized_corrected_pose_world", {})
        metrics = report.get("metrics", {})
        record_metrics = {
            "score": metrics.get("score"),
            "mask_iou": metrics.get("mask_iou"),
            "bbox_iou": metrics.get("bbox_iou"),
            "bbox_center_error_px": metrics.get("bbox_center_error_px"),
            "ground_contact_max_abs_m": metrics.get("ground_contact_max_abs_m"),
            "top_distance_mean_m": metrics.get("top_distance_mean_m"),
            "top_distance_min_m": metrics.get("top_distance_min_m"),
            "roof_above_ground": metrics.get("roof_above_ground"),
            "bbox_bottom_distance_m": metrics.get("bbox_bottom_distance_m"),
            "upright_angle_error_deg": metrics.get("upright_angle_error_deg"),
            "ground_hard_reject": metrics.get("ground_hard_reject"),
            "projected_bbox": metrics.get("projected_bbox"),
        }
        record_metrics.update({key: metrics.get(key) for key in _EDGE_POSE_HEADING_METRIC_KEYS if key in metrics})
        record_metrics.update({key: metrics.get(key) for key in _EDGE_POSE_TRUNCATION_METRIC_KEYS if key in metrics})
        record_metrics.update({key: metrics.get(key) for key in _GENERIC_POSE_METRIC_KEYS if key in metrics})
        return {
            "object_id": obj_id,
            "frame_id": int(frame_id),
            "timestamp_sec": float(timestamp_sec),
            "task": str(task_path),
            "output_dir": str(result_dir),
            "report": str(report_path),
            "pose": {
                "translation_world": pose.get("translation_world"),
                "rotation_matrix": pose.get("rotation_matrix"),
                "scale": pose.get("scale"),
            },
            "metrics": record_metrics,
            **run_info,
        }

    def _edge_pose_candidate_records_from_report(
        self,
        *,
        obj_id: str,
        frame_id: int,
        report: dict,
        report_path: Path,
        result_dir: Path,
        task_path: Path,
        timestamp_sec: float,
        run_info: dict,
    ) -> list[dict]:
        candidates = report.get("refined_pose_candidates")
        if not isinstance(candidates, list) or not candidates:
            return [
                self._edge_pose_record_from_report(
                    obj_id=obj_id,
                    frame_id=frame_id,
                    report=report,
                    report_path=report_path,
                    result_dir=result_dir,
                    task_path=task_path,
                    timestamp_sec=timestamp_sec,
                    run_info=run_info,
                )
            ]

        records: list[dict] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            pose = candidate.get("optimized_corrected_pose_world")
            metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
            if not isinstance(pose, dict):
                continue
            record_metrics = {
                "score": metrics.get("score"),
                "mask_iou": metrics.get("mask_iou"),
                "bbox_iou": metrics.get("bbox_iou"),
                "bbox_center_error_px": metrics.get("bbox_center_error_px"),
                "ground_contact_max_abs_m": metrics.get("ground_contact_max_abs_m"),
                "ground_contact_mean_abs_m": metrics.get("ground_contact_mean_abs_m"),
                "top_distance_mean_m": metrics.get("top_distance_mean_m"),
                "top_distance_min_m": metrics.get("top_distance_min_m"),
                "roof_above_ground": metrics.get("roof_above_ground"),
                "bbox_bottom_distance_m": metrics.get("bbox_bottom_distance_m"),
                "upright_angle_error_deg": metrics.get("upright_angle_error_deg"),
                "ground_hard_reject": metrics.get("ground_hard_reject"),
                "projected_bbox": metrics.get("projected_bbox"),
                "temporal_score": metrics.get("temporal_score"),
                "temporal_loss": metrics.get("temporal_loss"),
                "visible_mask_iou": metrics.get("visible_mask_iou"),
                "visible_bbox_iou": metrics.get("visible_bbox_iou"),
                "visible_contour_score": metrics.get("visible_contour_score"),
                "final_selection_mode": metrics.get("final_selection_mode"),
            }
            record_metrics.update({key: metrics.get(key) for key in _EDGE_POSE_HEADING_METRIC_KEYS if key in metrics})
            record_metrics.update({key: metrics.get(key) for key in _EDGE_POSE_TRUNCATION_METRIC_KEYS if key in metrics})
            record_metrics.update({key: metrics.get(key) for key in _GENERIC_POSE_METRIC_KEYS if key in metrics})
            record = {
                "object_id": obj_id,
                "frame_id": int(frame_id),
                "timestamp_sec": float(timestamp_sec),
                "task": str(task_path),
                "output_dir": str(result_dir),
                "report": str(report_path),
                "pose": {
                    "translation_world": pose.get("translation_world"),
                    "rotation_matrix": pose.get("rotation_matrix"),
                    "scale": pose.get("scale"),
                },
                "metrics": record_metrics,
                "candidate_index": int(index),
                "candidate_rank": candidate.get("candidate_rank"),
                "initializer_metadata": candidate.get("initializer_metadata", {}),
                **run_info,
            }
            records.append(record)
        return records

    @staticmethod
    def _pose_optimizer_report_from_candidate(base_report: dict, candidate_record: dict) -> dict:
        report = dict(base_report)
        metrics = dict(base_report.get("metrics", {}) if isinstance(base_report.get("metrics"), dict) else {})
        metrics.update(candidate_record.get("metrics", {}) if isinstance(candidate_record.get("metrics"), dict) else {})
        report["metrics"] = metrics
        report["optimized_corrected_pose_world"] = candidate_record.get("pose", {})
        if candidate_record.get("initializer_metadata"):
            report["best_initializer_metadata"] = candidate_record.get("initializer_metadata")
        return report

    @staticmethod
    def _edge_pose_track_frame(record: dict, pose_source: str = "edge_contour_fast_temporal") -> dict | None:
        import numpy as np

        pose = record.get("pose", {})
        if not ProjectExecutor._valid_vec3_like(pose.get("translation_world")):
            return None
        if not ProjectExecutor._valid_vec3_like(pose.get("scale")):
            return None
        rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
        if rotation.shape != (3, 3):
            return None
        center = np.asarray(pose["translation_world"], dtype=np.float64).reshape(3)
        scale = np.asarray(pose["scale"], dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = center
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        confidence = ProjectExecutor._edge_pose_confidence(metrics)
        return {
            "frame_id": int(record.get("frame_id") or 0),
            "timestamp_sec": float(record.get("timestamp_sec", 0.0) or 0.0),
            "centroid_world": [float(v) for v in center],
            "rotation_matrix": rotation.tolist(),
            "orientation_quat": ProjectExecutor._rotation_matrix_to_quat_xyzw(rotation),
            "scale": [float(v) for v in scale],
            "T_world_from_object": T.tolist(),
            "confidence": confidence,
            "source": pose_source,
            "geometry_status": pose_source,
            "quality": {
                "pose_optimizer_report": record.get("report"),
                "pose_optimizer_output_dir": record.get("output_dir"),
                "metrics": metrics,
                "reason": record.get("reason"),
            },
        }

    @staticmethod
    def _edge_pose_confidence(metrics: dict) -> float:
        try:
            mask_iou = float(metrics.get("mask_iou") or 0.0)
            bbox_iou = float(metrics.get("bbox_iou") or 0.0)
            center_error = float(metrics.get("bbox_center_error_px") or 120.0)
        except Exception:
            return 0.05
        center_score = max(0.0, 1.0 - center_error / 120.0)
        confidence = 0.45 * max(0.0, min(1.0, mask_iou)) + 0.35 * max(0.0, min(1.0, bbox_iou)) + 0.20 * center_score
        return max(0.05, min(0.99, float(confidence)))

    @staticmethod
    def _edge_pose_temporal_prior_payload(previous: dict | None) -> dict | None:
        if not isinstance(previous, dict):
            return None
        pose = previous.get("pose") if isinstance(previous.get("pose"), dict) else {}
        if not ProjectExecutor._valid_vec3_like(pose.get("translation_world")):
            return None
        if not ProjectExecutor._valid_vec3_like(pose.get("scale")):
            return None
        rotation = pose.get("rotation_matrix")
        if not isinstance(rotation, list):
            return None
        metrics = previous.get("metrics") if isinstance(previous.get("metrics"), dict) else {}
        quality = {
            "mask_iou": metrics.get("visible_mask_iou", metrics.get("mask_iou")),
            "bbox_iou": metrics.get("visible_bbox_iou", metrics.get("bbox_iou")),
            "bbox_center_error_px": metrics.get("bbox_center_error_px"),
            "truncation_severity": metrics.get("truncation_severity"),
            "low_observability": metrics.get("low_observability"),
        }
        return {
            "source": "previous_accepted_pose_in_memory",
            "frame_id": int(previous.get("frame_id") or 0),
            "output_dir": previous.get("output_dir"),
            "path": previous.get("report"),
            "quality": quality,
            "pose": {
                "translation_world": pose.get("translation_world"),
                "rotation_matrix": rotation,
                "scale": pose.get("scale"),
            },
        }

    @staticmethod
    def _edge_pose_candidate_temporal_prior_payload(
        previous: dict | None,
        *,
        all_frames_mode: bool,
    ) -> dict | None:
        if previous is None:
            return None
        # all_frames still must not read stale result folders from disk during
        # candidate generation, but the current run's last accepted pose is a
        # useful trust-region seed and mirrors the target-window path.
        return ProjectExecutor._edge_pose_temporal_prior_payload(previous)

    @staticmethod
    def _pose_record_updates_temporal_anchor(record: dict) -> bool:
        if ProjectExecutor._pose_record_is_generic(record):
            return ProjectExecutor._pose_record_is_generic_anchor(record)
        return ProjectExecutor._pose_record_is_high_quality_anchor(record)

    @staticmethod
    def _pose_record_promotes_temporal_candidate(record: dict, *, stable_streak_count: int = 0) -> bool:
        if ProjectExecutor._pose_record_is_generic(record):
            return ProjectExecutor._pose_record_is_generic_anchor(record)
        if ProjectExecutor._pose_record_is_high_quality_anchor(record):
            return True
        return (
            stable_streak_count >= 2
            and ProjectExecutor._pose_record_is_stable_temporal_anchor(record)
        )

    @staticmethod
    def _pose_record_is_generic(record: dict) -> bool:
        metrics = record.get("metrics") if isinstance(record, dict) and isinstance(record.get("metrics"), dict) else {}
        return any(key in metrics for key in ("appearance_score", "depth_score", "projection_valid_ratio", "observation_score"))

    @staticmethod
    def _pose_record_is_generic_anchor(record: dict) -> bool:
        if not isinstance(record, dict):
            return False
        if str(record.get("status") or "accepted") != "accepted":
            return False
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        try:
            mask_iou = float(metrics.get("visible_mask_iou") or metrics.get("soft_mask_iou") or metrics.get("mask_iou") or 0.0)
            bbox_iou = float(metrics.get("visible_bbox_iou") or metrics.get("bbox_iou") or 0.0)
            center_error = float(metrics.get("bbox_center_error_px") or 1e9)
            projection_valid_ratio = float(metrics.get("projection_valid_ratio") or 0.0)
            appearance = float(metrics.get("appearance_score") or 0.0) * float(metrics.get("appearance_confidence") or 0.0)
            depth = float(metrics.get("depth_score") or 0.0) * float(metrics.get("depth_confidence") or 0.0)
        except Exception:
            return False
        pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
        return (
            mask_iou >= 0.35
            and bbox_iou >= 0.30
            and center_error <= 100.0
            and projection_valid_ratio >= 0.50
            and (appearance >= 0.12 or depth >= 0.20 or mask_iou >= 0.55)
            and ProjectExecutor._valid_vec3_like(pose.get("translation_world"))
            and ProjectExecutor._valid_vec3_like(pose.get("scale"))
            and isinstance(pose.get("rotation_matrix"), list)
        )

    @staticmethod
    def _pose_record_is_stable_temporal_anchor(record: dict) -> bool:
        if ProjectExecutor._pose_record_is_generic(record):
            return ProjectExecutor._pose_record_is_generic_anchor(record)
        if not isinstance(record, dict):
            return False
        if str(record.get("status") or "accepted") != "accepted":
            return False
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        severity = str(metrics.get("truncation_severity") or "").lower()
        if severity in {"moderate", "severe", "critical"}:
            return False
        if bool(metrics.get("low_observability")) and severity in {"severe", "critical"}:
            return False
        anchor_gate = record.get("anchor_temporal_gate") if isinstance(record.get("anchor_temporal_gate"), dict) else {}
        if anchor_gate and not bool(anchor_gate.get("accepted", False)):
            return False
        try:
            mask_iou = float(metrics.get("visible_mask_iou") or metrics.get("mask_iou") or 0.0)
            bbox_iou = float(metrics.get("visible_bbox_iou") or metrics.get("bbox_iou") or 0.0)
            ground = float(metrics.get("ground_contact_max_abs_m") or 0.0)
            heading_err = float(metrics.get("heading_prior_angle_error_deg") or 0.0)
            contour_mean = metrics.get("visible_contour_mean_distance_px")
        except Exception:
            return False
        if not math.isfinite(mask_iou) or not math.isfinite(bbox_iou):
            return False
        if mask_iou < 0.80 or bbox_iou < 0.75:
            return False
        if math.isfinite(ground) and ground > 0.20:
            return False
        if math.isfinite(heading_err) and heading_err > 60.0:
            return False
        if contour_mean is not None:
            try:
                if float(contour_mean) > 6.5:
                    return False
            except Exception:
                return False
        if anchor_gate:
            try:
                yaw_jump = float(anchor_gate.get("yaw_jump_deg") or 0.0)
            except Exception:
                yaw_jump = 0.0
            try:
                scale_ratio = float(anchor_gate.get("scale_ratio") or 1.0)
            except Exception:
                scale_ratio = 1.0
            try:
                mask_drop = float(anchor_gate.get("mask_drop") or 0.0)
            except Exception:
                mask_drop = 0.0
            try:
                bbox_drop = float(anchor_gate.get("bbox_drop") or 0.0)
            except Exception:
                bbox_drop = 0.0
            combined_drop = max(0.0, mask_drop) + max(0.0, bbox_drop)
            source_anchor_kind = str(anchor_gate.get("anchor_kind") or "").lower()
            normal_observation = severity in {"", "none", "normal"}
            if math.isfinite(yaw_jump) and yaw_jump > 18.0:
                return False
            if math.isfinite(scale_ratio) and scale_ratio > 1.10:
                return False
            if (
                normal_observation
                and source_anchor_kind == "high_quality"
                and combined_drop > 0.20
                and (mask_drop > 0.08 or bbox_drop > 0.10)
                and (mask_iou < 0.86 or bbox_iou < 0.82)
            ):
                return False
            if (
                math.isfinite(yaw_jump)
                and yaw_jump > 6.0
                and combined_drop > 0.18
                and (mask_drop > 0.08 or bbox_drop > 0.10)
                and (bbox_iou < 0.82 or mask_iou < 0.84)
            ):
                return False
            if mask_drop > 0.12 or bbox_drop > 0.15:
                return False
        return True

    @staticmethod
    def _pose_temporal_anchor_kind(record: dict) -> str | None:
        if ProjectExecutor._pose_record_is_generic_anchor(record):
            return "generic_visual"
        if ProjectExecutor._pose_record_is_high_quality_anchor(record):
            return "high_quality"
        if ProjectExecutor._pose_record_is_stable_temporal_anchor(record):
            return "stable"
        return None

    @staticmethod
    def _pose_visual_iou(metrics: dict, *, kind: str) -> float:
        if kind == "mask":
            keys = ("visible_mask_iou", "mask_iou")
        else:
            keys = ("visible_bbox_iou", "bbox_iou")
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                score = float(value)
            except Exception:
                continue
            if math.isfinite(score) and score > 0.0:
                return max(0.0, min(1.0, score))
        return 0.0

    @staticmethod
    def _pose_record_is_high_quality_anchor(record: dict) -> bool:
        if not isinstance(record, dict):
            return False
        if str(record.get("status") or "accepted") != "accepted":
            return False
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        severity = str(metrics.get("truncation_severity") or "").lower()
        if bool(metrics.get("low_observability")) or severity in {"moderate", "severe", "critical"}:
            return False
        mask_iou = ProjectExecutor._pose_visual_iou(metrics, kind="mask")
        bbox_iou = ProjectExecutor._pose_visual_iou(metrics, kind="bbox")
        if mask_iou < 0.88 or bbox_iou < 0.85:
            return False
        try:
            ground = float(metrics.get("ground_contact_max_abs_m") or 0.0)
        except Exception:
            ground = 0.0
        if math.isfinite(ground) and ground > 0.18:
            return False
        pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
        return (
            ProjectExecutor._valid_vec3_like(pose.get("translation_world"))
            and ProjectExecutor._valid_vec3_like(pose.get("scale"))
            and isinstance(pose.get("rotation_matrix"), list)
        )

    @staticmethod
    def _pose_scale_ratio(record: dict, anchor: dict) -> float | None:
        import numpy as np

        pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
        anchor_pose = anchor.get("pose") if isinstance(anchor.get("pose"), dict) else {}
        if not ProjectExecutor._valid_vec3_like(pose.get("scale")):
            return None
        if not ProjectExecutor._valid_vec3_like(anchor_pose.get("scale")):
            return None
        cur = float(np.median(np.asarray(pose["scale"], dtype=np.float64).reshape(3)))
        ref = float(np.median(np.asarray(anchor_pose["scale"], dtype=np.float64).reshape(3)))
        if not math.isfinite(cur) or not math.isfinite(ref) or cur <= 1e-8 or ref <= 1e-8:
            return None
        return max(cur, ref) / max(1e-8, min(cur, ref))

    @staticmethod
    def _pose_anchor_temporal_gate(record: dict, anchor: dict | None) -> dict:
        if not isinstance(anchor, dict):
            return {"accepted": True, "reason": "accepted"}
        pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
        anchor_pose = anchor.get("pose") if isinstance(anchor.get("pose"), dict) else {}
        rotation = pose.get("rotation_matrix")
        anchor_rotation = anchor_pose.get("rotation_matrix")
        if not isinstance(rotation, list) or not isinstance(anchor_rotation, list):
            return {"accepted": True, "reason": "accepted"}

        rotation_jump = ProjectExecutor._rotation_geodesic_deg(rotation, anchor_rotation)
        yaw_jump = rotation_jump
        scale_ratio = ProjectExecutor._pose_scale_ratio(record, anchor)
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        anchor_metrics = anchor.get("metrics") if isinstance(anchor.get("metrics"), dict) else {}
        mask_iou = ProjectExecutor._pose_visual_iou(metrics, kind="mask")
        bbox_iou = ProjectExecutor._pose_visual_iou(metrics, kind="bbox")
        anchor_mask = ProjectExecutor._pose_visual_iou(anchor_metrics, kind="mask")
        anchor_bbox = ProjectExecutor._pose_visual_iou(anchor_metrics, kind="bbox")
        mask_drop = max(0.0, anchor_mask - mask_iou)
        bbox_drop = max(0.0, anchor_bbox - bbox_iou)
        combined_visual_drop = mask_drop + bbox_drop
        severity = str(metrics.get("truncation_severity") or "").lower()
        low_observability = bool(metrics.get("low_observability"))
        visible_fraction = None
        try:
            if metrics.get("visible_target_fraction") is not None:
                visible_fraction = float(metrics.get("visible_target_fraction"))
        except Exception:
            visible_fraction = None
        try:
            heading_err = float(metrics.get("heading_prior_angle_error_deg"))
        except Exception:
            heading_err = 0.0
        if not math.isfinite(heading_err):
            heading_err = 0.0
        anchor_kind = str(anchor.get("temporal_anchor_kind") or "").lower()
        if not anchor_kind:
            anchor_kind = ProjectExecutor._pose_temporal_anchor_kind(anchor) or ""
        stable_anchor = anchor_kind == "stable"

        decision_base = {
            "anchor_frame_id": int(anchor.get("frame_id") or 0),
            "anchor_kind": anchor_kind or None,
            "yaw_jump_deg": float(yaw_jump) if math.isfinite(yaw_jump) else None,
            "rotation_jump_deg": float(rotation_jump) if math.isfinite(rotation_jump) else None,
            "scale_ratio": scale_ratio,
            "mask_drop": float(mask_drop),
            "bbox_drop": float(bbox_drop),
            "combined_visual_drop": float(combined_visual_drop),
            "mask_iou": float(mask_iou),
            "bbox_iou": float(bbox_iou),
            "anchor_mask_iou": float(anchor_mask),
            "anchor_bbox_iou": float(anchor_bbox),
            "heading_prior_angle_error_deg": heading_err,
            "truncation_severity": severity or None,
            "low_observability": bool(low_observability),
            "visible_target_fraction": visible_fraction,
        }

        severe_observation_loss = low_observability or severity in {"severe", "critical"} or (
            visible_fraction is not None and visible_fraction < 0.40
        )
        moderate_observation_loss = severity == "moderate" or (
            visible_fraction is not None and visible_fraction < 0.72
        )
        if severe_observation_loss:
            yaw_limit = 22.0 if stable_anchor else 30.0
            scale_limit = 1.18 if stable_anchor else 1.12
            heading_limit = 70.0 if stable_anchor else 60.0
            if math.isfinite(yaw_jump) and yaw_jump > yaw_limit:
                return {"accepted": False, "reason": f"anchor_low_observability_yaw_jump:{yaw_jump:.1f}deg", **decision_base}
            if scale_ratio is not None and scale_ratio > scale_limit:
                return {"accepted": False, "reason": f"anchor_low_observability_scale_jump:{scale_ratio:.3f}x", **decision_base}
            if heading_err > heading_limit:
                return {"accepted": False, "reason": f"anchor_low_observability_heading_error:{heading_err:.1f}deg", **decision_base}

        if moderate_observation_loss:
            yaw_limit = 36.0 if stable_anchor else 45.0
            scale_limit = 1.30 if stable_anchor else 1.25
            heading_limit = 80.0 if stable_anchor else 75.0
            if math.isfinite(yaw_jump) and yaw_jump > yaw_limit:
                return {"accepted": False, "reason": f"anchor_truncated_yaw_jump:{yaw_jump:.1f}deg", **decision_base}
            if scale_ratio is not None and scale_ratio > scale_limit:
                return {"accepted": False, "reason": f"anchor_truncated_scale_jump:{scale_ratio:.3f}x", **decision_base}
            if heading_err > heading_limit:
                return {"accepted": False, "reason": f"anchor_truncated_heading_error:{heading_err:.1f}deg", **decision_base}

        visual_mask_drop_limit = 0.14 if stable_anchor else 0.10
        visual_bbox_drop_limit = 0.16 if stable_anchor else 0.12
        visual_degraded = mask_drop > visual_mask_drop_limit or bbox_drop > visual_bbox_drop_limit
        high_quality_anchor = anchor_kind == "high_quality" or (anchor_mask >= 0.88 and anchor_bbox >= 0.85)
        coupled_visual_degraded = (
            combined_visual_drop > (0.22 if stable_anchor else 0.18)
            and (mask_drop > 0.08 or bbox_drop > 0.10)
            and (mask_iou < 0.86 or bbox_iou < 0.82)
        )
        if (
            high_quality_anchor
            and severity in {"", "none", "normal"}
            and coupled_visual_degraded
            and math.isfinite(yaw_jump)
            and yaw_jump > (10.0 if stable_anchor else 6.0)
        ):
            return {"accepted": False, "reason": f"anchor_visual_degradation_yaw_jump_from_high_quality:{yaw_jump:.1f}deg", **decision_base}
        if coupled_visual_degraded and math.isfinite(yaw_jump):
            yaw_limit = 8.0 if stable_anchor else 6.0
            if yaw_jump > yaw_limit:
                return {"accepted": False, "reason": f"anchor_visual_degradation_yaw_jump_coupled:{yaw_jump:.1f}deg", **decision_base}
        if visual_degraded:
            yaw_limit = 12.0 if stable_anchor else 8.0
            low_mask = 0.84 if stable_anchor else 0.86
            low_bbox = 0.82 if stable_anchor else 0.84
            if math.isfinite(yaw_jump) and yaw_jump > yaw_limit and (mask_iou < low_mask or bbox_iou < low_bbox):
                return {"accepted": False, "reason": f"anchor_visual_degradation_yaw_jump:{yaw_jump:.1f}deg", **decision_base}
            rotation_limit = 24.0 if stable_anchor else 18.0
            if math.isfinite(yaw_jump) and yaw_jump > rotation_limit:
                return {"accepted": False, "reason": f"anchor_visual_degradation_rotation_jump:{yaw_jump:.1f}deg", **decision_base}
            scale_limit = 1.20 if stable_anchor else 1.15
            if scale_ratio is not None and scale_ratio > scale_limit:
                return {"accepted": False, "reason": f"anchor_visual_degradation_scale_jump:{scale_ratio:.3f}x", **decision_base}

        return {"accepted": True, "reason": "accepted", **decision_base}

    @staticmethod
    def _apply_anchor_temporal_gate_to_selected_records(
        selected_records: list[dict],
        *,
        frame_ids: list[int],
        frame_records: dict[str, dict],
    ) -> tuple[list[dict], dict]:
        kept: list[dict] = []
        anchor: dict | None = None
        sorted_records = sorted(
            [record for record in selected_records if isinstance(record, dict)],
            key=lambda record: int(record.get("frame_id") or 0),
        )
        for record in sorted_records:
            decision = ProjectExecutor._pose_anchor_temporal_gate(record, anchor)
            record["anchor_temporal_gate"] = decision
            if not decision.get("accepted"):
                failed_frame_id = int(record.get("frame_id") or 0)
                record["status"] = "rejected"
                record["reason"] = str(decision.get("reason") or "anchor_temporal_gate_rejected")
                frame_records[f"frame_{failed_frame_id:06d}"] = record
                remaining_count = 0
                for remaining_frame_id in sorted({int(fid) for fid in frame_ids if int(fid) > failed_frame_id}):
                    remaining_count += 1
                    frame_records[f"frame_{remaining_frame_id:06d}"] = {
                        "status": "skipped",
                        "reason": "skipped_after_anchor_temporal_gate_failure",
                        "failed_frame_id": failed_frame_id,
                    }
                return kept, {
                    "applied": True,
                    "reason": record["reason"],
                    "failed_frame_id": failed_frame_id,
                    "remaining_frame_count": remaining_count,
                    "accepted_frame_count_before_failure": len(kept),
                    "gate": decision,
                }
            record["status"] = "accepted"
            kept.append(record)
            anchor_kind = ProjectExecutor._pose_temporal_anchor_kind(record)
            if anchor_kind:
                record["temporal_anchor_kind"] = anchor_kind
            if anchor_kind:
                anchor = record
        return kept, {"applied": False, "reason": "all_selected_records_passed"}

    @staticmethod
    def _pose_truncated_object_fail_fast_decision(record: dict, *, inst: dict | None, frame_id: int) -> dict:
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        status = str(record.get("status") or "").lower()
        reason = str(record.get("reason") or "")
        failure_status = status in {"failed", "rejected"}
        if not failure_status:
            return {"skip_object": False, "reason": "not_failed_or_rejected"}

        severity = str(metrics.get("truncation_severity") or "").lower()
        low_observability = bool(metrics.get("low_observability"))
        truncated = low_observability or severity in {"severe", "critical"}
        inst = inst if isinstance(inst, dict) else {}
        truncation_info = inst.get("truncation_info")
        if isinstance(truncation_info, dict):
            info_severity = str(truncation_info.get("truncation_severity") or truncation_info.get("severity") or "").lower()
            low_observability = low_observability or bool(truncation_info.get("low_observability"))
            truncated = truncated or bool(
                truncation_info.get("is_truncated")
                or truncation_info.get("touches_image_border")
                or info_severity in {"severe", "critical"}
            )
            if not severity and info_severity:
                severity = info_severity
        truncated = truncated or bool(inst.get("is_truncated") or inst.get("truncated"))

        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                width = inst.get("image_width") or inst.get("width")
                height = inst.get("image_height") or inst.get("height")
                if (width is None or height is None) and inst.get("mask_rle"):
                    try:
                        rle = json.loads(inst["mask_rle"]) if isinstance(inst["mask_rle"], str) else inst["mask_rle"]
                        size = rle.get("size") if isinstance(rle, dict) else None
                        if isinstance(size, (list, tuple)) and len(size) >= 2:
                            height, width = int(size[0]), int(size[1])
                    except Exception:
                        pass
                if width is not None and height is not None:
                    w = float(width)
                    h = float(height)
                    touches_bottom = y2 >= h - 1.0
                    touches_side = x1 <= 1.0 or x2 >= w - 1.0
                    if touches_bottom and touches_side:
                        severity = severity or "severe"
                        truncated = True
                    elif touches_bottom and status == "failed":
                        severity = severity or "severe"
                        truncated = True
            except Exception:
                pass

        if not truncated:
            return {"skip_object": False, "reason": "not_truncated"}
        if severity not in {"severe", "critical"} and not low_observability and status != "failed":
            return {"skip_object": False, "reason": "truncation_not_severe"}
        return {
            "skip_object": True,
            "reason": reason or status,
            "frame_id": int(frame_id),
            "status": status,
            "truncation_severity": severity or ("severe" if low_observability else "unknown"),
            "low_observability": bool(low_observability),
        }

    @staticmethod
    def _apply_truncated_object_fail_fast(
        fail_fast: dict,
        *,
        frame_ids: list[int],
        frame_records: dict[str, dict],
        accepted_records: list[dict],
    ) -> dict:
        failed_frame_id = int(fail_fast.get("frame_id") or 0)
        remaining_count = 0
        for remaining_frame_id in frame_ids:
            remaining_frame_id = int(remaining_frame_id)
            if failed_frame_id and remaining_frame_id <= failed_frame_id:
                continue
            remaining_count += 1
            frame_key = f"frame_{remaining_frame_id:06d}"
            if frame_key in frame_records:
                continue
            frame_records[frame_key] = {
                "status": "skipped",
                "reason": "skipped_after_truncated_object_failure",
                "failed_frame_id": failed_frame_id,
            }
        accepted_count = len([record for record in accepted_records if isinstance(record, dict)])
        if accepted_count <= 0:
            accepted_count = sum(
                1
                for record in frame_records.values()
                if isinstance(record, dict) and str(record.get("status") or "").lower() == "accepted"
            )
        return {
            "skip_entire_object": accepted_count <= 0,
            "failed_frame_id": failed_frame_id,
            "remaining_frame_count": remaining_count,
            "accepted_frame_count_before_failure": accepted_count,
        }

    @staticmethod
    def _pose_local_seed_frame_ids(
        *,
        frame_ids: list[int],
        current_frame_id: int,
        window_radius: int,
    ) -> list[int]:
        frames = sorted({int(fid) for fid in frame_ids if int(fid) > 0})
        if not frames:
            return [int(current_frame_id)]
        radius = max(0, int(window_radius))
        current = int(current_frame_id)
        desired_count = max(1, radius * 2 + 1)
        local = sorted(fid for fid in frames if abs(fid - current) <= radius)
        if current in frames and current not in local:
            local.append(current)
            local.sort()
        if local and len(local) < desired_count:
            selected = set(local)
            for fid in sorted(frames, key=lambda item: (abs(item - current), item)):
                selected.add(fid)
                if len(selected) >= desired_count:
                    break
            local = sorted(selected)
        if local:
            return local
        nearest = min(frames, key=lambda fid: abs(fid - current))
        return [nearest]

    @staticmethod
    def _pose_track_scale_prior(
        records: list[dict],
        *,
        source: str,
        require_high_quality: bool = False,
        max_frame_id: int | None = None,
    ) -> dict | None:
        import numpy as np

        valid = []
        frame_ids = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            if max_frame_id is not None:
                try:
                    if int(record.get("frame_id") or 0) > int(max_frame_id):
                        continue
                except Exception:
                    continue
            if require_high_quality and not ProjectExecutor._pose_scale_prior_record_is_high_quality(record):
                continue
            scale = record.get("scale")
            if scale is None and isinstance(record.get("pose"), dict):
                scale = record["pose"].get("scale")
            if not ProjectExecutor._valid_vec3_like(scale):
                continue
            arr = np.asarray(scale, dtype=np.float64).reshape(3)
            scalar = float(np.median(arr))
            if not math.isfinite(scalar) or scalar <= 1e-8:
                continue
            valid.append(scalar)
            try:
                frame_ids.append(int(record.get("frame_id") or 0))
            except Exception:
                pass
        if len(valid) < _POSE_TRACK_SCALE_PRIOR_MIN_FRAMES:
            return None
        value = float(np.median(np.asarray(valid, dtype=np.float64)))
        if not math.isfinite(value) or value <= 1e-8:
            return None
        return {
            "available": True,
            "source": source,
            "scale": [value, value, value],
            "sample_count": len(valid),
            "frame_ids": sorted({fid for fid in frame_ids if fid > 0}),
        }

    @staticmethod
    def _pose_scale_prior_record_is_high_quality(record: dict) -> bool:
        if record.get("status") not in {None, "accepted"}:
            return False
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        try:
            mask_iou = float(metrics.get("visible_mask_iou") or metrics.get("mask_iou") or 0.0)
            bbox_iou = float(metrics.get("visible_bbox_iou") or metrics.get("bbox_iou") or 0.0)
            center_error = float(metrics.get("bbox_center_error_px") or 1e9)
        except Exception:
            return False
        if bool(metrics.get("low_observability")) and str(metrics.get("truncation_severity") or "") == "severe":
            return False
        contour_mean = metrics.get("visible_contour_mean_distance_px")
        if str(metrics.get("truncation_severity") or "") == "severe" and contour_mean is not None:
            try:
                if float(contour_mean) > 7.0:
                    return False
            except Exception:
                return False
        return mask_iou >= 0.55 and bbox_iou >= 0.55 and center_error <= 80.0

    @staticmethod
    def _pose_optimizer_temporal_fallback_acceptance(report: dict, rejection: dict | None) -> dict:
        reason = str((rejection or {}).get("reason") or "")
        if "temporal_" not in reason:
            return {"accepted": False, "reason": reason or "not_temporal_rejection"}
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
        try:
            mask_iou = float(metrics.get("visible_mask_iou") or metrics.get("mask_iou") or 0.0)
            bbox_iou = float(metrics.get("visible_bbox_iou") or metrics.get("bbox_iou") or 0.0)
            center_error = float(metrics.get("bbox_center_error_px") or 1e9)
        except Exception:
            return {"accepted": False, "reason": "invalid_visual_fallback_metrics"}
        if mask_iou >= 0.55 and bbox_iou >= 0.55 and center_error <= 80.0:
            return {
                "accepted": True,
                "reason": f"visual_fallback_after_{reason}",
                "low_confidence": True,
            }
        return {
            "accepted": False,
            "reason": f"visual_fallback_rejected_after_{reason}",
            "low_confidence": True,
        }

    @staticmethod
    def _edge_pose_heading_cost(metrics: dict) -> float:
        try:
            angle = float(metrics.get("heading_prior_angle_error_deg"))
        except Exception:
            return 0.0
        if not math.isfinite(angle):
            return 0.0
        if not bool(metrics.get("heading_front_sign_enabled")):
            return 0.0
        try:
            confidence = float(metrics.get("heading_front_sign_confidence") or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if confidence <= 0.0:
            return 0.0
        if bool(metrics.get("heading_front_sign_hard_rejected")):
            return 10.0
        try:
            candidate_sign = float(metrics.get("heading_candidate_forward_sign"))
            semantic_sign = float(metrics.get("heading_semantic_front_sign"))
        except Exception:
            candidate_sign = 0.0
            semantic_sign = 0.0
        sign_flip_penalty = 1.0 if candidate_sign * semantic_sign < 0.0 else 0.0
        wrong_sign_penalty = max(0.0, angle - 90.0) / 90.0
        soft_angle_penalty = min(angle, 90.0) / 180.0
        return confidence * (sign_flip_penalty + 4.0 * wrong_sign_penalty + 0.3 * soft_angle_penalty)

    @staticmethod
    def _select_edge_pose_candidate_trajectory(
        frame_candidates: dict[int, list[dict]],
        *,
        target_frame_id: int | None,
        generic_mode: bool = False,
    ) -> tuple[list[dict], dict]:
        import copy
        import numpy as np

        frames = sorted(int(fid) for fid, candidates in frame_candidates.items() if candidates)
        if not frames:
            return [], {"enabled": True, "applied": False, "reason": "no_candidates"}

        def has_strong_truncated_visual_evidence(candidate: dict) -> bool:
            metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
            severity = str(metrics.get("truncation_severity") or "")
            if severity not in {"light", "moderate", "severe"} and not bool(metrics.get("low_observability")):
                return False
            try:
                mask = float(metrics.get("visible_mask_iou") or metrics.get("mask_iou") or 0.0)
                contour = float(metrics.get("visible_contour_score") or 0.0)
                contour_mean = float(metrics.get("visible_contour_mean_distance_px"))
                profile_mean = float(metrics.get("visible_profile_mean_distance_px"))
            except Exception:
                return False
            return mask >= 0.90 and contour >= 0.60 and contour_mean <= 2.5 and profile_mean <= 4.0

        def unary_cost(candidate: dict) -> float:
            metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
            score = float(metrics.get("score") or 0.0)
            mask = float(metrics.get("visible_mask_iou") or metrics.get("mask_iou") or 0.0)
            bbox = float(metrics.get("visible_bbox_iou") or metrics.get("bbox_iou") or 0.0)
            if generic_mode:
                appearance = float(metrics.get("appearance_score") or 0.0) * float(metrics.get("appearance_confidence") or 0.0)
                depth = float(metrics.get("depth_score") or 0.0) * float(metrics.get("depth_confidence") or 0.0)
                temporal = float(metrics.get("temporal_score") or 0.0)
                center = float(metrics.get("bbox_center_error_px") or 120.0)
                return (
                    -score
                    -0.55 * mask
                    -0.20 * bbox
                    -0.18 * appearance
                    -0.18 * depth
                    -0.10 * temporal
                    + 0.003 * min(center, 240.0)
                )
            severity = str(metrics.get("truncation_severity") or "")
            severe = severity == "severe" or bool(metrics.get("low_observability"))
            truncated = severe or severity in {"light", "moderate"}
            contour = float(metrics.get("visible_contour_score") or 0.0)
            contour_mean = metrics.get("visible_contour_mean_distance_px")
            profile_mean = metrics.get("visible_profile_mean_distance_px")
            center = float(metrics.get("bbox_center_error_px") or 120.0)
            ground = metrics.get("ground_contact_max_abs_m")
            ground_penalty = 0.0
            if ground is not None:
                ground_penalty = min(2.0, max(0.0, float(ground) - 0.12) * 2.0)
            heading_penalty = ProjectExecutor._edge_pose_heading_cost(metrics)
            contour_penalty = 0.0
            if severe:
                if contour_mean is not None:
                    contour_penalty += min(4.0, max(0.0, float(contour_mean) - 5.0) * 0.45)
                if profile_mean is not None:
                    contour_penalty += min(3.0, max(0.0, float(profile_mean) - 8.0) * 0.30)
                return (
                    -0.20 * score
                    -0.95 * mask
                    -0.50 * contour
                    -0.03 * bbox
                    + 0.004 * min(center, 200.0)
                    + ground_penalty
                    + heading_penalty
                    + contour_penalty
                )
            if truncated:
                if contour_mean is not None:
                    contour_penalty += min(2.5, max(0.0, float(contour_mean) - 3.5) * 0.25)
                if profile_mean is not None:
                    contour_penalty += min(2.0, max(0.0, float(profile_mean) - 6.0) * 0.10)
                return (
                    -0.18 * score
                    -1.05 * mask
                    -0.75 * contour
                    -0.10 * bbox
                    + 0.002 * min(center, 200.0)
                    + ground_penalty
                    + heading_penalty
                    + contour_penalty
                )
            return -score - 0.7 * mask - 0.35 * bbox + 0.004 * min(center, 200.0) + ground_penalty + heading_penalty

        def transition_cost(prev: dict, cur: dict) -> float:
            prev_pose = prev.get("pose") if isinstance(prev.get("pose"), dict) else {}
            cur_pose = cur.get("pose") if isinstance(cur.get("pose"), dict) else {}
            cost = 0.0
            dt = max(1, abs(int(cur.get("frame_id") or 0) - int(prev.get("frame_id") or 0)))
            strong_truncated_visual = (
                has_strong_truncated_visual_evidence(cur)
                or has_strong_truncated_visual_evidence(prev)
            )
            if ProjectExecutor._valid_vec3_like(prev_pose.get("translation_world")) and ProjectExecutor._valid_vec3_like(cur_pose.get("translation_world")):
                prev_t = np.asarray(prev_pose["translation_world"], dtype=np.float64)
                cur_t = np.asarray(cur_pose["translation_world"], dtype=np.float64)
                jump = float(np.linalg.norm(cur_t - prev_t)) / float(dt)
                translation_sigma_m = 1.2 if strong_truncated_visual else 0.8
                cost += min(12.0, (jump / translation_sigma_m) ** 2)
            prev_r = prev_pose.get("rotation_matrix")
            cur_r = cur_pose.get("rotation_matrix")
            if isinstance(prev_r, list) and isinstance(cur_r, list):
                jump_deg = ProjectExecutor._rotation_geodesic_deg(cur_r, prev_r)
                if math.isfinite(jump_deg):
                    rotation_sigma_deg = 45.0 if strong_truncated_visual else 25.0
                    cost += min(16.0, (jump_deg / rotation_sigma_deg) ** 2)
                    if jump_deg > 90.0:
                        cost += 6.0
            if ProjectExecutor._valid_vec3_like(prev_pose.get("scale")) and ProjectExecutor._valid_vec3_like(cur_pose.get("scale")):
                prev_s = float(np.median(np.asarray(prev_pose["scale"], dtype=np.float64)))
                cur_s = float(np.median(np.asarray(cur_pose["scale"], dtype=np.float64)))
                ratio = max(prev_s, cur_s) / max(1e-8, min(prev_s, cur_s))
                scale_sigma_log = 0.24 if strong_truncated_visual else 0.14
                cost += min(9.0, (math.log(max(ratio, 1e-8)) / scale_sigma_log) ** 2)
            return cost

        dp: list[list[float]] = []
        parent: list[list[int | None]] = []
        first = frames[0]
        first_candidates = frame_candidates[first]
        dp.append([unary_cost(candidate) for candidate in first_candidates])
        parent.append([None for _ in first_candidates])
        for frame_index, frame_id in enumerate(frames[1:], start=1):
            candidates = frame_candidates[frame_id]
            previous_candidates = frame_candidates[frames[frame_index - 1]]
            row: list[float] = []
            parent_row: list[int | None] = []
            for candidate in candidates:
                costs = [
                    dp[frame_index - 1][prev_index] + transition_cost(previous, candidate)
                    for prev_index, previous in enumerate(previous_candidates)
                ]
                best_prev = int(np.argmin(costs)) if costs else 0
                row.append(unary_cost(candidate) + (costs[best_prev] if costs else 0.0))
                parent_row.append(best_prev)
            dp.append(row)
            parent.append(parent_row)

        last_index = int(np.argmin(dp[-1]))
        selected_indices = [last_index]
        for frame_index in range(len(frames) - 1, 0, -1):
            prev_index = parent[frame_index][selected_indices[-1]]
            selected_indices.append(0 if prev_index is None else int(prev_index))
        selected_indices.reverse()

        selected: list[dict] = []
        for frame_id, index in zip(frames, selected_indices):
            record = copy.deepcopy(frame_candidates[frame_id][index])
            record["trajectory_selection"] = {
                "enabled": True,
                "candidate_index": int(index),
                "dp_cost": float(dp[frames.index(frame_id)][index]),
            }
            selected.append(record)

        return selected, {
            "enabled": True,
            "applied": True,
            "method": "viterbi_candidate_trajectory",
            "target_frame_id": int(target_frame_id) if target_frame_id is not None else None,
            "frame_ids": frames,
            "selected_frame_count": len(selected),
            "total_cost": float(min(dp[-1])),
        }

    @staticmethod
    def _stabilize_edge_pose_records(records: list[dict]) -> tuple[list[dict], dict]:
        import copy
        import numpy as np

        if len(records) < 2:
            return records, {"enabled": True, "applied": False, "reason": "fewer_than_two_frames"}
        stabilized = copy.deepcopy(records)
        scales = []
        for record in stabilized:
            pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
            scale = pose.get("scale")
            if ProjectExecutor._valid_vec3_like(scale):
                scales.append(np.asarray(scale, dtype=np.float64).reshape(3))
        if not scales:
            return stabilized, {"enabled": True, "applied": False, "reason": "missing_scales"}
        shared_scalar = float(np.median([float(np.median(scale)) for scale in scales]))
        if not math.isfinite(shared_scalar) or shared_scalar <= 1e-8:
            return stabilized, {"enabled": True, "applied": False, "reason": "invalid_shared_scale"}
        shared_scale = [shared_scalar, shared_scalar, shared_scalar]
        for record in stabilized:
            pose = record.get("pose") if isinstance(record.get("pose"), dict) else {}
            if pose:
                pose["scale"] = list(shared_scale)
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            metrics["track_shared_scale"] = list(shared_scale)
        return stabilized, {
            "enabled": True,
            "applied": True,
            "method": "track_median_shared_scale",
            "shared_scale": shared_scale,
            "frame_count": len(stabilized),
        }

    @staticmethod
    def _median_pose_scale(scales: list[object]) -> list[float]:
        import numpy as np

        valid = []
        for scale in scales:
            if ProjectExecutor._valid_vec3_like(scale):
                valid.append(np.asarray(scale, dtype=np.float64).reshape(3))
        if not valid:
            return [1.0, 1.0, 1.0]
        median = np.median(np.stack(valid, axis=0), axis=0)
        return [float(v) for v in median]

    @staticmethod
    def _rename_rejected_pose_result_dir(results_dir: Path, result_dir: Path, task_id: str) -> Path:
        rejected_dir = results_dir / f"{task_id}__rejected"
        if rejected_dir.exists():
            shutil.rmtree(rejected_dir)
        try:
            result_dir.rename(rejected_dir)
            return rejected_dir
        except Exception:
            return result_dir

    @staticmethod
    def _load_frame_image_for_detection(inst: dict):
        image_b64 = inst.get("image_b64")
        if not image_b64:
            return None
        return cv2.imdecode(np_from_b64(str(image_b64)), cv2.IMREAD_COLOR)

    def _read_project_video_frame(self, frame_id: int):
        cap = cv2.VideoCapture(str(self.context.paths.input_video))
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id) - 1))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    @staticmethod
    def _mask_from_detection(inst: dict, image_shape_hw: tuple[int, int]):
        import numpy as np

        height, width = [int(v) for v in image_shape_hw]
        mask_rle = inst.get("mask_rle")
        if mask_rle:
            try:
                from pycocotools import mask as mask_util

                rle = json.loads(mask_rle) if isinstance(mask_rle, str) else mask_rle
                mask = mask_util.decode(rle).astype(np.uint8)
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                return (mask > 0).astype(np.uint8) * 255
            except Exception as exc:
                _logger.warning("[pose.optimize] failed to decode mask_rle: %s", exc)

        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def _crop_to_bbox(image, bbox: list[float]):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return image.copy()
        return image[y1:y2, x1:x2].copy()

    def _motion_heading_prior_for_track(
        self,
        *,
        obj_id: str,
        frame_id: int,
        detection_frames: list[dict],
        window: int = 8,
    ) -> dict | None:
        samples: list[tuple[int, float, float, float]] = []
        start = max(1, int(frame_id) - int(window))
        end = int(frame_id) + int(window)
        for fid in range(start, end + 1):
            inst = self._get_instance_for_frame(obj_id, fid, detection_frames)
            if not inst:
                continue
            bbox = inst.get("bbox_xyxy") or inst.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            except Exception:
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 1.0:
                continue
            samples.append((int(fid), 0.5 * (x1 + x2), 0.5 * (y1 + y2), area))

        if len(samples) < 2:
            return None

        before = [sample for sample in samples if sample[0] < int(frame_id)]
        after = [sample for sample in samples if sample[0] > int(frame_id)]
        if before and after:
            left = before[0]
            right = after[-1]
        else:
            left = samples[0]
            right = samples[-1]
        dt = max(1, abs(right[0] - left[0]))
        dx = float(right[1] - left[1])
        dy = float(right[2] - left[2])
        displacement_px = math.hypot(dx, dy)

        anchor = next((sample for sample in samples if sample[0] == int(frame_id)), samples[len(samples) // 2])
        bbox_diag = math.sqrt(max(1.0, float(anchor[3])))
        min_displacement_px = max(8.0, 0.12 * bbox_diag)
        if displacement_px < min_displacement_px:
            return {
                "enabled": False,
                "source": "bbox_track_motion",
                "reason": "insufficient_motion",
                "frame_window": [samples[0][0], samples[-1][0]],
                "sample_count": len(samples),
                "displacement_px": displacement_px,
                "min_displacement_px": min_displacement_px,
            }

        raw_confidence = min(1.0, displacement_px / max(1e-6, 0.5 * bbox_diag))
        # Bbox-center motion gives a useful line direction, but it is not a
        # reliable semantic front cue under perspective/camera motion.  Keep it
        # weak so front/back selection is not dominated by this prior.
        confidence = min(0.35, 0.35 * raw_confidence)
        return {
            "enabled": True,
            "source": "bbox_track_motion",
            "frame_window": [samples[0][0], samples[-1][0]],
            "sample_count": len(samples),
            "from_frame": left[0],
            "to_frame": right[0],
            "vector_image": [dx / displacement_px, dy / displacement_px],
            "displacement_px": displacement_px,
            "per_frame_px": displacement_px / dt,
            "raw_confidence": raw_confidence,
            "confidence": confidence,
        }

    def _vehicle_pose_context_for_task(
        self,
        *,
        obj_id: str,
        frame_id: int,
        bbox_xyxy: object,
        camera: dict,
        detection_frames: list[dict],
        road_geometry: dict | None,
        mesh_axis_prior: dict | None = None,
        target_window_radius: int | None = None,
    ) -> dict:
        context: dict = {
            "schema": "vehicle_pose_context.v1",
            "object_id": obj_id,
            "frame_id": int(frame_id),
        }
        if isinstance(mesh_axis_prior, dict) and mesh_axis_prior.get("available"):
            context["mesh_axis_prior"] = mesh_axis_prior
        if target_window_radius is not None and int(target_window_radius) <= 0:
            context["heading_prior"] = {
                "enabled": False,
                "source": "bbox_track_motion",
                "reason": "single_frame_window",
                "target_window_radius": int(target_window_radius),
            }
        else:
            heading_window = 8
            if target_window_radius is not None:
                heading_window = max(8, int(target_window_radius))
            heading_prior = self._motion_heading_prior_for_track(
                obj_id=obj_id,
                frame_id=frame_id,
                detection_frames=detection_frames,
                window=heading_window,
            )
            if heading_prior:
                heading_prior["target_window_radius"] = int(heading_window)
                context["heading_prior"] = heading_prior

        road_plane = select_road_plane_for_frame(road_geometry, int(frame_id))
        if road_plane:
            context["road_plane"] = road_plane
            if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
                try:
                    x1, _y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
                    bottom_uv = ((x1 + x2) * 0.5, y2)
                    bottom_intersection = intersect_camera_ray_with_plane(
                        camera=camera,
                        uv=bottom_uv,
                        plane=road_plane,
                    )
                    if bottom_intersection:
                        context["bbox_bottom_ground"] = bottom_intersection
                except Exception:
                    pass
        return context

    def _write_pose_track_task_result_artifacts(
        self,
        *,
        out_dir: Path,
        object_pose_tracks: dict[str, dict],
        per_frame_object_poses: dict[str, dict],
        sam3d_meshes: dict,
        detection_frames: list[dict],
        cam_traj: list[dict],
        wildgs_poses: list[dict],
        wildgs_K: dict | None,
        road_geometry: dict | None,
        target_frame_id: int | None,
    ) -> dict:
        import numpy as np

        tasks_dir = out_dir / "tasks"
        results_dir = out_dir / "results"
        if tasks_dir.exists():
            shutil.rmtree(tasks_dir)
        if results_dir.exists():
            shutil.rmtree(results_dir)
        tasks_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        object_nodes = {obj.object_id: obj for obj in self._all_objects()}
        written_tasks = 0
        written_results = 0
        skipped: dict[str, str] = {}

        for obj_id, track in object_pose_tracks.items():
            if not isinstance(track, dict):
                skipped[obj_id] = "invalid_track"
                continue
            frame = self._select_pose_task_frame(track, per_frame_object_poses, obj_id, target_frame_id)
            if not frame:
                skipped[obj_id] = "missing_pose_frame"
                continue
            frame_id = int(frame.get("frame_id") or 0)
            if frame_id <= 0:
                skipped[obj_id] = "invalid_frame"
                continue

            mesh_entry = sam3d_meshes.get(obj_id, {})
            glb_path = self._find_glb(mesh_entry)
            if not glb_path:
                skipped[obj_id] = "missing_glb"
                continue
            inst = self._get_instance_for_frame(obj_id, frame_id, detection_frames)
            if not isinstance(inst, dict):
                skipped[obj_id] = "missing_detection"
                continue
            image = self._load_frame_image_for_detection(inst)
            if image is None:
                image = self._read_project_video_frame(frame_id)
            if image is None:
                skipped[obj_id] = "missing_frame_image"
                continue
            mask = self._mask_from_detection(inst, image.shape[:2])
            if mask is None:
                skipped[obj_id] = "missing_mask"
                continue
            camera = self._projection_camera(frame_id, cam_traj, wildgs_poses, wildgs_K)
            if camera is None:
                skipped[obj_id] = "missing_camera"
                continue

            task_id = f"{obj_id}@{frame_id:06d}"
            task_dir = tasks_dir / task_id
            result_dir = results_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            bbox = inst.get("bbox_xyxy") or inst.get("bbox")
            obj_mesh = self._load_trimesh(glb_path)
            mesh_axis_prior = self._mesh_axis_prior_for_pose_optimizer(
                np.asarray(obj_mesh.vertices, dtype=np.float64) if obj_mesh is not None else None,
                axis_roles=(track.get("axis_roles") if isinstance(track, dict) else None),
            )
            vehicle_pose_context = self._vehicle_pose_context_for_task(
                obj_id=obj_id,
                frame_id=frame_id,
                bbox_xyxy=bbox,
                camera=camera,
                detection_frames=detection_frames,
                road_geometry=road_geometry,
                mesh_axis_prior=mesh_axis_prior,
            )
            self._write_pose_optimizer_sample(
                task_dir=task_dir,
                obj_id=obj_id,
                frame_id=frame_id,
                frame_image=image,
                full_mask=mask,
                inst=inst,
                glb_path=glb_path,
                camera=camera,
                object_node=object_nodes.get(obj_id),
                object_track=track.get("frames", []),
                vehicle_pose_context=vehicle_pose_context,
            )
            written_tasks += 1
            self._write_pose_track_result_artifact(
                result_dir=result_dir,
                task_dir=task_dir,
                task_id=task_id,
                obj_id=obj_id,
                frame=frame,
                track=track,
                inst=inst,
                frame_image=image,
                full_mask=mask,
                camera=camera,
                image_shape_hw=image.shape[:2],
                source_mesh_path=glb_path,
            )
            written_results += 1

        return {
            "schema": "guanwu.pose_optimizer.task_result_artifacts.v1",
            "tasks_dir": str(tasks_dir),
            "results_dir": str(results_dir),
            "task_count": written_tasks,
            "result_count": written_results,
            "skipped_count": len(skipped),
            "skipped": skipped,
        }

    @staticmethod
    def _select_pose_task_frame(
        track: dict,
        per_frame_object_poses: dict[str, dict],
        obj_id: str,
        target_frame_id: int | None,
    ) -> dict | None:
        frames = [frame for frame in track.get("frames", []) if isinstance(frame, dict)]
        if not frames:
            return None
        if target_frame_id is not None:
            frame_key = f"frame_{int(target_frame_id):06d}"
            pose = per_frame_object_poses.get(frame_key, {}).get(obj_id)
            if isinstance(pose, dict):
                out = dict(pose)
                out["frame_id"] = int(target_frame_id)
                return out
            return None
        return max(
            frames,
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                float((item.get("quality") or {}).get("bbox_area_px", 0.0) or 0.0),
            ),
        )

    def _write_pose_track_result_artifact(
        self,
        *,
        result_dir: Path,
        task_dir: Path,
        task_id: str,
        obj_id: str,
        frame: dict,
        track: dict,
        inst: dict,
        frame_image,
        full_mask,
        camera: dict,
        image_shape_hw: tuple[int, int],
        source_mesh_path: Path,
    ) -> None:
        import csv

        task_path = task_dir / "task.json"
        task = self._json_load(task_path)
        pose = {
            "translation_world": [float(v) for v in frame.get("centroid_world", [0.0, 0.0, 0.0])],
            "rotation_matrix": frame.get("rotation_matrix", track.get("rotation_matrix")),
            "scale": [float(v) for v in frame.get("scale", track.get("scale", [1.0, 1.0, 1.0]))],
        }
        task["corrected_pose"] = pose
        quality = frame.get("quality", {}) if isinstance(frame.get("quality"), dict) else {}
        losses = quality.get("losses", {}) if isinstance(quality.get("losses"), dict) else {}
        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        report = {
            "task_id": task_id,
            "object_id": obj_id,
            "label": task.get("label", "object"),
            "sample_dir": str(task_dir),
            "mesh_path": str(task_dir / "object.glb"),
            "image_size": [int(image_shape_hw[1]), int(image_shape_hw[0])],
            "json_bbox": [float(v) for v in bbox[:4]] if bbox and len(bbox) >= 4 else None,
            "optimized_corrected_pose_world": pose,
            "pose_track": {
                "schema": track.get("schema", "guanwu.object_pose_track.v1"),
                "frame_id": int(frame.get("frame_id") or 0),
                "source": frame.get("source", "depth_temporal"),
                "confidence": float(frame.get("confidence", 0.0) or 0.0),
                "geometry_status": frame.get("geometry_status", "depth_icp_temporal"),
                "axis_roles": track.get("axis_roles", {}),
                "mesh_basis": track.get("mesh_basis"),
                "heading": track.get("heading"),
                "target_frame_refinement": track.get("target_frame_refinement"),
            },
            "metrics": {
                "confidence": float(frame.get("confidence", 0.0) or 0.0),
                "depth_valid_points": int(quality.get("depth_valid_points", 0) or 0),
                "bbox_area_px": float(quality.get("bbox_area_px", 0.0) or 0.0),
                "weighted_loss": float(quality.get("weighted_loss", 0.0) or 0.0),
                "depth_alignment_loss": float(losses.get("depth_alignment", 0.0) or 0.0),
                "silhouette_loss": float(losses.get("silhouette", 0.0) or 0.0),
                "ground_contact_loss": float(losses.get("ground_contact", 0.0) or 0.0),
                "temporal_smoothness_loss": float(losses.get("temporal_smoothness", 0.0) or 0.0),
                "dimension_prior_loss": float(losses.get("dimension_prior", 0.0) or 0.0),
                "projection_score": float(losses.get("projection_score", 0.0) or 0.0),
            },
            "outputs": {
                "optimization_history": str(result_dir / "optimization_history.csv"),
                "optimization_report": str(result_dir / "optimization_report.json"),
                "optimized_task": str(result_dir / "task_with_optimized_corrected_pose.json"),
            },
            "run_metadata": {
                "variant": "depth_icp_temporal_target_frame",
                "source_mesh_path": str(source_mesh_path),
                "generated_by": "ProjectExecutor._write_pose_track_task_result_artifacts",
            },
        }
        viz_outputs = self._write_pose_track_visualizations(
            result_dir=result_dir,
            frame_image=frame_image,
            full_mask=full_mask,
            bbox=bbox,
            pose=pose,
            camera=camera,
            mesh_path=task_dir / "object.glb",
            metrics=report["metrics"],
            task_id=task_id,
        )
        report["outputs"].update(viz_outputs)
        self._json_dump(result_dir / "optimization_report.json", report)
        self._json_dump(result_dir / "task_with_optimized_corrected_pose.json", task)
        with open(result_dir / "optimization_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss", "confidence", "source"])
            writer.writeheader()
            writer.writerow({
                "step": 0,
                "loss": report["metrics"]["weighted_loss"],
                "confidence": report["metrics"]["confidence"],
                "source": report["pose_track"]["source"],
            })

    def _write_pose_track_visualizations(
        self,
        *,
        result_dir: Path,
        frame_image,
        full_mask,
        bbox,
        pose: dict,
        camera: dict,
        mesh_path: Path,
        metrics: dict,
        task_id: str,
    ) -> dict[str, str]:
        import numpy as np

        image = frame_image.copy()
        height, width = image.shape[:2]
        mask_bool = full_mask > 0
        overlay = image.copy()
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([0, 180, 255])).astype(np.uint8)
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)

        pixels = self._project_pose_mesh_pixels(mesh_path, pose, camera, width=width, height=height)
        if pixels.shape[0] > 0:
            sample = pixels
            if sample.shape[0] > 2500:
                sample = sample[np.linspace(0, sample.shape[0] - 1, 2500, dtype=int)]
            for x, y in np.rint(sample).astype(int):
                cv2.circle(overlay, (int(x), int(y)), 1, (60, 255, 80), -1)
            proj_bbox = self._bbox_from_pixels(pixels, width=width, height=height)
            if proj_bbox:
                cv2.rectangle(
                    overlay,
                    (int(proj_bbox[0]), int(proj_bbox[1])),
                    (int(proj_bbox[2]), int(proj_bbox[3])),
                    (60, 255, 80),
                    2,
                )
        text = f"{task_id} conf={float(metrics.get('confidence', 0.0)):.2f}"
        cv2.putText(overlay, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)

        mask_bgr = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
        inspection = np.concatenate([image, overlay], axis=1)
        reference = np.concatenate([image, mask_bgr], axis=1)
        debug = overlay.copy()
        lines = [
            f"loss={float(metrics.get('weighted_loss', 0.0)):.3f}",
            f"depth={float(metrics.get('depth_alignment_loss', 0.0)):.3f}",
            f"sil={float(metrics.get('silhouette_loss', 0.0)):.3f}",
            f"ground={float(metrics.get('ground_contact_loss', 0.0)):.3f}",
            f"proj={float(metrics.get('projection_score', 0.0)):.3f}",
        ]
        y = 52
        for line in lines:
            cv2.putText(debug, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            y += 24

        paths = {
            "alignment_collage": str(result_dir / "01_alignment_overview.png"),
            "pose_closeup_collage": str(result_dir / "02_pose_inspection.png"),
            "model_reference_collage": str(result_dir / "03_model_reference.png"),
            "temporal_edge_debug": str(result_dir / "04_temporal_edge_debug.png"),
        }
        cv2.imwrite(paths["alignment_collage"], overlay)
        cv2.imwrite(paths["pose_closeup_collage"], inspection)
        cv2.imwrite(paths["model_reference_collage"], reference)
        cv2.imwrite(paths["temporal_edge_debug"], debug)
        return paths

    def _project_pose_mesh_pixels(self, mesh_path: Path, pose: dict, camera: dict, *, width: int, height: int):
        import numpy as np

        obj_mesh = self._load_trimesh(mesh_path)
        if obj_mesh is None or len(obj_mesh.vertices) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        verts = np.asarray(obj_mesh.vertices, dtype=np.float64)
        if verts.shape[0] > 4000:
            verts = verts[np.linspace(0, verts.shape[0] - 1, 4000, dtype=int)]
        rotation = np.asarray(pose.get("rotation_matrix"), dtype=np.float64)
        if rotation.shape != (3, 3):
            rotation = np.eye(3, dtype=np.float64)
        scale = np.asarray(pose.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
        center = np.asarray(pose.get("translation_world", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        world = (rotation @ (verts * scale[None, :]).T).T + center[None, :]
        return self._project_points_to_image(world, camera, width=width, height=height)

    def _write_pose_optimizer_sample(
        self,
        *,
        task_dir: Path,
        obj_id: str,
        frame_id: int,
        frame_image,
        full_mask,
        inst: dict,
        glb_path: Path,
        camera: dict,
        object_node: ObjectNode | None,
        object_track: list[dict],
        vehicle_pose_context: dict | None = None,
        temporal_prior_pose: dict | None = None,
    ) -> Path:
        import numpy as np

        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        if bbox and len(bbox) >= 4:
            bbox_xyxy = [float(v) for v in bbox[:4]]
        else:
            ys, xs = np.nonzero(full_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                raise ValueError(f"Cannot derive bbox for {obj_id}@{frame_id}: empty mask")
            bbox_xyxy = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        crop_image = self._crop_to_bbox(frame_image, bbox_xyxy)
        crop_mask = self._crop_to_bbox(full_mask, bbox_xyxy)
        cv2.imwrite(str(task_dir / "image.jpg"), frame_image)
        cv2.imwrite(str(task_dir / "crop.jpg"), crop_image)
        cv2.imwrite(str(task_dir / "mask.png"), crop_mask)
        shutil.copy2(glb_path, task_dir / "object.glb")

        translation = None
        scale = None
        orientation_quat = None
        timestamp = 0.0
        for rec in object_track or []:
            if int(rec.get("frame_id") or -1) == int(frame_id):
                translation = rec.get("centroid_world")
                scale = rec.get("scale")
                orientation_quat = rec.get("orientation_quat")
                timestamp = float(rec.get("timestamp_sec", 0.0) or 0.0)
                break
        if translation is None and object_node is not None:
            translation = object_node.geometry.pose_3d.position
        if scale is None and object_node is not None:
            scale = object_node.geometry.scale_3d
        if orientation_quat is None and object_node is not None:
            orientation_quat = object_node.geometry.pose_3d.orientation_quat

        translation = translation if self._valid_vec3_like(translation) else camera["t"].tolist()
        scale = scale if self._valid_vec3_like(scale) else [1.0, 1.0, 1.0]
        rotation_matrix = (
            self._quat_xyzw_to_rotation_matrix(orientation_quat).tolist()
            if isinstance(orientation_quat, list) and len(orientation_quat) == 4
            else np.eye(3, dtype=np.float64).tolist()
        )

        task = {
            "task_id": f"{obj_id}@{int(frame_id):06d}",
            "object_id": obj_id,
            "label": str(inst.get("concept_label") or (object_node.label if object_node else "object")),
            "frame_idx": int(frame_id),
            "timestamp_sec": timestamp,
            "segment_kind": str(inst.get("segment_kind") or (object_node.segment_kind if object_node else "object")),
            "bbox_xyxy": bbox_xyxy,
            "mask_ref": str(inst.get("mask_ref") or ""),
            "detection_score": float(inst.get("score", 0.0) or 0.0),
            "image_size": [int(frame_image.shape[1]), int(frame_image.shape[0])],
            "mesh_path": "object.glb",
            "camera": {
                "fx": float(camera["fx"]),
                "fy": float(camera["fy"]),
                "cx": float(camera["cx"]),
                "cy": float(camera["cy"]),
                "T_world_from_cam": self._camera_dict_to_transform(camera).tolist(),
            },
            "corrected_pose": {
                "translation_world": [float(v) for v in translation],
                "rotation_matrix": rotation_matrix,
                "scale": [float(v) for v in scale],
            },
        }
        if vehicle_pose_context:
            task["vehicle_pose_context"] = vehicle_pose_context
        if temporal_prior_pose:
            task["temporal_prior_pose"] = temporal_prior_pose
        task_path = task_dir / "task.json"
        self._json_dump(task_path, task)
        return task_path

    @staticmethod
    def _camera_dict_to_transform(camera: dict):
        import numpy as np

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(camera["R"], dtype=np.float64)
        T[:3, 3] = np.asarray(camera["t"], dtype=np.float64)
        return T

    @staticmethod
    def _valid_vec3_like(value: object) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return False
        try:
            vals = [float(v) for v in value]
        except Exception:
            return False
        return all(math.isfinite(v) for v in vals)

    def _run_edge_contour_fast(self, task_dir: Path, result_dir: Path) -> dict:
        return self._run_pose_optimizer_cli(
            task_dir,
            result_dir,
            config_filename="edge_contour_fast_quick.yaml",
        )

    def _run_generic_appearance_temporal(self, task_dir: Path, result_dir: Path) -> dict:
        return self._run_pose_optimizer_cli(
            task_dir,
            result_dir,
            config_filename="generic_appearance_temporal.yaml",
            extra_args=self._generic_temporal_speedup_args_for_task(task_dir),
        )

    @staticmethod
    def _generic_temporal_speedup_args_for_task(task_dir: Path) -> list[str]:
        task_path = Path(task_dir) / "task.json"
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        temporal_prior = task.get("temporal_prior_pose")
        if not isinstance(temporal_prior, dict) or not temporal_prior:
            return []
        return [
            "--top_k_candidates",
            "8",
            "--refine_top_k",
            "1",
            "--stage1_iters",
            "4",
            "--stage2_iters",
            "3",
            "--stage3_iters",
            "6",
        ]

    def _run_pose_optimizer_cli(
        self,
        task_dir: Path,
        result_dir: Path,
        *,
        config_filename: str,
        extra_args: list[str] | None = None,
    ) -> dict:
        guanwu_root = Path(__file__).resolve().parents[4]
        bundled_optimizer_root = guanwu_root / "process" / "pose_optimizer"
        workspace_root = guanwu_root if bundled_optimizer_root.exists() else guanwu_root.parent
        optimizer_root = workspace_root / "process" / "pose_optimizer"
        config_path = optimizer_root / "configs" / config_filename
        command = [
            sys.executable,
            "-m",
            "process.pose_optimizer.cli",
            "--config",
            str(config_path),
            "--sample_dir",
            str(task_dir),
            "--output_dir",
            str(result_dir),
        ]
        if extra_args:
            command.extend([str(item) for item in extra_args])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace_root),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._pose_optimizer_timeout_sec(),
                check=False,
            )
        except Exception as exc:
            return {"returncode": -1, "stdout_tail": "", "stderr_tail": str(exc)}
        return {
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    @staticmethod
    def _pose_optimizer_acceptance(report: dict) -> dict:
        metrics = report.get("metrics", {})
        try:
            mask_iou = float(metrics.get("mask_iou") or 0.0)
            bbox_iou = float(metrics.get("bbox_iou") or 0.0)
            center_error = float(metrics.get("bbox_center_error_px") or 1e9)
        except Exception:
            return {"accepted": False, "reason": "invalid_metrics"}
        if mask_iou < 0.20:
            return {"accepted": False, "reason": f"mask_iou_below_threshold:{mask_iou:.3f}"}
        if bbox_iou < 0.20:
            return {"accepted": False, "reason": f"bbox_iou_below_threshold:{bbox_iou:.3f}"}
        if center_error > 120.0:
            return {"accepted": False, "reason": f"center_error_too_large:{center_error:.1f}"}
        sign_decision = ProjectExecutor._pose_optimizer_axis_sign_acceptance(report)
        if not sign_decision.get("accepted", False):
            return sign_decision
        road = report.get("road_constraint", {})
        if isinstance(road, dict) and bool(road.get("available")):
            if bool(road.get("hard_reject", False)):
                return {"accepted": False, "reason": "road_ground_hard_reject"}
            try:
                max_ground_error = road.get("ground_contact_max_abs_m")
                roof_above_ground = road.get("roof_above_ground")
                upright_error = road.get("upright_angle_error_deg")
                if roof_above_ground is False:
                    return {"accepted": False, "reason": "roof_below_ground"}
                if upright_error is None:
                    return {"accepted": False, "reason": "missing_upright_angle_error"}
                if float(upright_error) > 35.0:
                    return {"accepted": False, "reason": f"upright_error_too_large:{float(upright_error):.1f}deg"}
                if max_ground_error is None:
                    return {"accepted": False, "reason": "missing_ground_contact_error"}
                if max_ground_error is not None and float(max_ground_error) > 0.65:
                    return {"accepted": False, "reason": f"ground_error_too_large:{float(max_ground_error):.3f}m"}
                bottom_distance = road.get("bbox_bottom_distance_m")
                if bottom_distance is None:
                    bottom_distance = metrics.get("bbox_bottom_distance_m")
                if bottom_distance is not None and float(bottom_distance) > 1.25:
                    return {"accepted": False, "reason": f"bbox_bottom_ground_distance_too_large:{float(bottom_distance):.3f}m"}
            except Exception:
                return {"accepted": False, "reason": "invalid_road_constraint_metrics"}
        pose = report.get("optimized_corrected_pose_world", {})
        if not ProjectExecutor._valid_vec3_like(pose.get("translation_world")):
            return {"accepted": False, "reason": "invalid_translation"}
        if not ProjectExecutor._valid_vec3_like(pose.get("scale")):
            return {"accepted": False, "reason": "invalid_scale"}
        return {"accepted": True, "reason": "accepted"}

    @staticmethod
    def _generic_pose_optimizer_acceptance(report: dict) -> dict:
        metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
        if metrics.get("acceptance_status") == "rejected":
            reasons = metrics.get("reject_reasons")
            if isinstance(reasons, list) and reasons:
                return {"accepted": False, "reason": ",".join(str(item) for item in reasons)}
        try:
            visible_iou = float(
                metrics.get("visible_mask_iou")
                or metrics.get("visible_soft_mask_iou")
                or metrics.get("soft_mask_iou")
                or metrics.get("mask_iou")
                or 0.0
            )
            bbox_iou = float(metrics.get("bbox_iou") or 0.0)
            center_error = float(metrics.get("bbox_center_error_px") or 1e9)
            projection_ratio = float(metrics.get("projection_valid_ratio") or 0.0)
        except Exception:
            return {"accepted": False, "reason": "invalid_generic_metrics"}
        bbox = report.get("json_bbox") or []
        bbox_diag = 0.0
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                bbox_diag = math.hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
            except Exception:
                bbox_diag = 0.0
        if visible_iou < 0.12:
            return {"accepted": False, "reason": f"visible_mask_or_soft_iou_below_threshold:{visible_iou:.3f}"}
        if bbox_iou < 0.10:
            return {"accepted": False, "reason": f"bbox_iou_below_threshold:{bbox_iou:.3f}"}
        if center_error > max(120.0, 0.35 * bbox_diag):
            return {"accepted": False, "reason": f"center_error_too_large:{center_error:.1f}"}
        if projection_ratio < 0.50:
            return {"accepted": False, "reason": f"projection_valid_ratio_below_threshold:{projection_ratio:.3f}"}
        try:
            depth_confidence = float(metrics.get("depth_confidence") or 0.0)
            depth_score = float(metrics.get("depth_score") or 0.0)
        except Exception:
            depth_confidence = 0.0
            depth_score = 0.0
        if depth_confidence >= 0.70 and depth_score < 0.25:
            return {"accepted": False, "reason": f"depth_score_below_threshold:{depth_score:.3f}"}
        pose = report.get("optimized_corrected_pose_world", {})
        if not ProjectExecutor._valid_vec3_like(pose.get("translation_world")):
            return {"accepted": False, "reason": "invalid_translation"}
        if not ProjectExecutor._valid_vec3_like(pose.get("scale")):
            return {"accepted": False, "reason": "invalid_scale"}
        return {"accepted": True, "reason": "accepted"}

    @staticmethod
    def _pose_report_axis_idx(report: dict, canonical_key: str, fallback_key: str) -> int | None:
        mesh_meta = report.get("mesh_axis_metadata", {})
        if not isinstance(mesh_meta, dict):
            return None
        prior = mesh_meta.get("canonical_axis_prior")
        if isinstance(prior, dict) and prior.get(canonical_key) is not None:
            try:
                axis = int(prior.get(canonical_key))
                return axis if axis in (0, 1, 2) else None
            except Exception:
                return None
        if mesh_meta.get(fallback_key) is not None:
            try:
                axis = int(mesh_meta.get(fallback_key))
                return axis if axis in (0, 1, 2) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _pose_report_expected_sign(report: dict, canonical_key: str) -> float | None:
        mesh_meta = report.get("mesh_axis_metadata", {})
        if not isinstance(mesh_meta, dict):
            return None
        prior = mesh_meta.get("canonical_axis_prior")
        if isinstance(prior, dict) and prior.get(canonical_key) is not None:
            if canonical_key == "up_sign" and prior.get("lock_up_sign") is False:
                return None
            if canonical_key == "forward_sign" and prior.get("lock_forward_sign") is False:
                return None
            try:
                return -1.0 if float(prior.get(canonical_key)) < 0.0 else 1.0
            except Exception:
                return None
        return None

    @staticmethod
    def _pose_report_allowed_signs(report: dict, canonical_key: str) -> set[float]:
        mesh_meta = report.get("mesh_axis_metadata", {})
        if not isinstance(mesh_meta, dict):
            return set()
        prior = mesh_meta.get("canonical_axis_prior")
        if not isinstance(prior, dict):
            return set()
        candidate_key = "up_sign_candidates" if canonical_key == "up_sign" else "forward_sign_candidates"
        raw = prior.get(candidate_key)
        if not isinstance(raw, list):
            return set()
        signs: set[float] = set()
        for value in raw:
            try:
                signs.add(-1.0 if float(value) < 0.0 else 1.0)
            except Exception:
                continue
        return signs

    @staticmethod
    def _pose_optimizer_axis_sign_acceptance(report: dict) -> dict:
        meta = report.get("best_initializer_metadata", {})
        if not isinstance(meta, dict):
            return {"accepted": True, "reason": "accepted"}

        expected_up = ProjectExecutor._pose_report_expected_sign(report, "up_sign")
        if expected_up is not None and meta.get("up_sign") is not None:
            try:
                actual_up = -1.0 if float(meta.get("up_sign")) < 0.0 else 1.0
            except Exception:
                return {"accepted": False, "reason": "invalid_up_sign"}
            if actual_up != expected_up:
                return {"accepted": False, "reason": f"unexpected_up_sign:{actual_up:+.0f}"}
        elif meta.get("up_sign") is not None:
            try:
                actual_up = -1.0 if float(meta.get("up_sign")) < 0.0 else 1.0
            except Exception:
                return {"accepted": False, "reason": "invalid_up_sign"}
            allowed = ProjectExecutor._pose_report_allowed_signs(report, "up_sign")
            if allowed and actual_up not in allowed:
                return {"accepted": False, "reason": f"disallowed_up_sign:{actual_up:+.0f}"}

        expected_forward = ProjectExecutor._pose_report_expected_sign(report, "forward_sign")
        if expected_forward is not None and meta.get("forward_sign") is not None:
            try:
                actual_forward = -1.0 if float(meta.get("forward_sign")) < 0.0 else 1.0
            except Exception:
                return {"accepted": False, "reason": "invalid_forward_sign"}
            if actual_forward != expected_forward:
                return {"accepted": False, "reason": f"unexpected_forward_sign:{actual_forward:+.0f}"}
        elif meta.get("forward_sign") is not None:
            try:
                actual_forward = -1.0 if float(meta.get("forward_sign")) < 0.0 else 1.0
            except Exception:
                return {"accepted": False, "reason": "invalid_forward_sign"}
            allowed = ProjectExecutor._pose_report_allowed_signs(report, "forward_sign")
            if allowed and actual_forward not in allowed:
                return {"accepted": False, "reason": f"disallowed_forward_sign:{actual_forward:+.0f}"}

        return {"accepted": True, "reason": "accepted"}

    def _load_pose_optimizer_manifest(self) -> dict:
        artifact = self.context.artifacts.get("pose.optimize")
        if artifact is None:
            return {}
        path = artifact.outputs.get("pose_optimizer_manifest")
        if not path or not Path(path).exists():
            return {}
        try:
            payload = self._json_load(path)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _accepted_pose_optimizer_record(manifest: dict, obj_id: str) -> dict | None:
        record = manifest.get(obj_id)
        if not isinstance(record, dict) or record.get("status") != "accepted":
            return None
        if not ProjectExecutor._valid_vec3_like(record.get("translation_world")):
            return None
        if not ProjectExecutor._valid_vec3_like(record.get("scale")):
            return None
        rotation = record.get("rotation_matrix")
        if not isinstance(rotation, list) or len(rotation) != 3:
            return None
        return record

    @staticmethod
    def _timestamp_for_frame(traj: list[dict], frame_id: int) -> float:
        for rec in traj or []:
            if int(rec.get("frame_id") or -1) == int(frame_id):
                return float(rec.get("timestamp_sec", 0.0) or 0.0)
        return 0.0

    def _load_wildgs_poses(self, geometry) -> tuple[list[dict], dict | None]:
        wildgs_poses: list[dict] = []
        wildgs_K: dict | None = None
        cam_poses_path = geometry.outputs.get("wildgs_camera_poses")
        if not cam_poses_path:
            gl_dir = Path(str(geometry.outputs["summary"])).parent
            for d in [gl_dir] + list(gl_dir.parent.iterdir()):
                cp = d / "wildgs" / "exports" / "camera_poses.jsonl"
                if cp.exists():
                    cam_poses_path = str(cp)
                    break
        if cam_poses_path and Path(cam_poses_path).exists():
            with open(cam_poses_path) as f:
                for line in f:
                    if line.strip():
                        wildgs_poses.append(json.loads(line))
            if wildgs_poses:
                wildgs_K = wildgs_poses[0].get("intrinsics")
        return wildgs_poses, wildgs_K

    @staticmethod
    def _resolve_wildgs_depth_maps_dir(geometry) -> str | None:
        raw = geometry.outputs.get("wildgs_depth_maps")
        if raw and Path(str(raw)).exists():
            return str(raw)
        summary_path = geometry.outputs.get("summary")
        if not summary_path:
            return str(raw) if raw else None
        gl_dir = Path(str(summary_path)).parent
        candidates = [
            gl_dir / "wildgs" / "exports" / "depth_maps" / "depth_maps",
            gl_dir / "wildgs" / "exports" / "depth_maps",
        ]
        try:
            candidates.extend(
                sibling / "wildgs" / "exports" / "depth_maps" / "depth_maps"
                for sibling in gl_dir.parent.iterdir()
                if sibling.is_dir()
            )
        except OSError:
            pass
        for candidate in candidates:
            try:
                if not candidate.exists():
                    continue
                try:
                    has_depth = any(candidate.glob("*.npy"))
                except OSError:
                    has_depth = False
                if not has_depth:
                    for index in range(0, 16):
                        probe = candidate / f"{index:05d}.npy"
                        try:
                            exists = probe.exists()
                        except PermissionError:
                            exists = True
                        if exists:
                            has_depth = True
                            break
                if has_depth:
                    return str(candidate)
            except OSError:
                continue
        return str(raw) if raw else None

    @staticmethod
    def _resolve_depth_map_for_frame(depth_maps_dir: str | Path | None, frame_id: int) -> Path | None:
        if not depth_maps_dir:
            return None
        root = Path(depth_maps_dir)
        frame_id = int(frame_id)
        depth_frame_name = f"{max(frame_id - 1, 0):05d}.npy"
        roots = [
            root,
            root / "depth_maps",
            root / "depth_maps" / "depth_maps",
        ]
        candidates = [candidate_root / depth_frame_name for candidate_root in roots]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue
        return None

    @staticmethod
    def _find_bg_mesh(bg_mesh_dir: str | None) -> "Path | None":
        if not bg_mesh_dir:
            return None
        raw = Path(bg_mesh_dir)
        if raw.is_file() and raw.exists():
            return raw
        for name in ("background_mesh.ply", "background_mesh.obj"):
            p = raw / name
            if p.exists():
                return p
        return None

    @staticmethod
    def _find_bg_meshes(bg_mesh_dir: str | None) -> list[tuple[str, "Path"]]:
        if not bg_mesh_dir:
            return []
        raw = Path(bg_mesh_dir)
        if not raw.exists() or raw.is_file():
            return []
        ordered = [
            ("road", raw / "road_mesh.obj"),
            ("structures", raw / "structures_mesh.obj"),
            ("far", raw / "far_mesh.obj"),
        ]
        out = [(name, path) for name, path in ordered if path.exists()]
        return out if len(out) >= 2 else []

    @staticmethod
    def _background_assets_target_frame_id(geometry) -> int | None:
        manifest = None if geometry is None else geometry.outputs.get("background_assets_manifest")
        if not manifest:
            return None
        path = Path(manifest)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("target_frame_id")
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _find_glb(entry: dict) -> "Path | None":
        for f in entry.get("files", []):
            if f.get("format") == "glb":
                p = Path(f["path"])
                if p.exists():
                    return p
        return None

    @staticmethod
    def _load_trimesh(path):
        import trimesh
        loaded = trimesh.load(str(path))
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            return geoms[0] if geoms else None
        if isinstance(loaded, trimesh.Trimesh):
            return loaded
        return None

    @staticmethod
    def _get_wildgs_R(wildgs_poses, cam_traj, frame_idx):
        import numpy as np
        if frame_idx is None:
            return None
        wgs_frame = frame_idx - 1
        if wildgs_poses:
            wpose = next((p for p in wildgs_poses if p.get("frame") == wgs_frame), None)
            if wpose is None and wildgs_poses:
                wpose = min(wildgs_poses, key=lambda p: abs(p.get("frame", -999) - wgs_frame))
            T = (wpose or {}).get("T_world_from_cam")
            if T is not None:
                return np.asarray(T, dtype=np.float64)[:3, :3]
        if cam_traj:
            pose = next((p for p in cam_traj if p.get("frame_id") == frame_idx), cam_traj[-1])
            R = pose.get("R")
            if R:
                return np.asarray(R, dtype=np.float64)
        return None

    def _find_anchor_frame(self, obj_id, traj, detection_frames, depth_maps_dir, cam_traj, wildgs_poses, wildgs_K):
        if not depth_maps_dir or not traj:
            return None, None
        depth_dir = Path(depth_maps_dir)
        avail_depths = sorted(int(p.stem) for p in depth_dir.glob("*.npy"))
        if not avail_depths:
            return None, None

        best_frame, best_pts, best_score = None, None, -1
        for rec in traj:
            fid = rec.get("frame_id")
            if fid is None:
                continue
            wgs_frame = fid - 1
            closest_depth = min(avail_depths, key=lambda x: abs(x - wgs_frame))
            dist = abs(closest_depth - wgs_frame)
            if dist > 3:
                continue

            inst = self._get_instance_for_frame(obj_id, fid, detection_frames)
            if not inst:
                continue
            mask_rle = inst.get("mask_rle")
            if not mask_rle:
                continue

            quality = rec.get("geom_quality", 0.5)
            pts = build_depth_point_cloud(
                depth_maps_dir,
                fid,
                mask_rle,
                cam_traj=cam_traj,
                wildgs_poses=wildgs_poses,
                wildgs_K=wildgs_K,
            )
            if pts is None or len(pts) < 32:
                continue

            visibility = self._instance_anchor_visibility(inst)
            density = min(float(len(pts)) / 256.0, 1.5)
            score = quality * visibility * density / (1.0 + dist)
            if score <= best_score:
                continue

            best_frame, best_pts, best_score = fid, pts, score

        return best_frame, best_pts

    def _get_instance_for_frame(self, obj_id, frame_id, detection_frames):
        for entry in detection_frames:
            if entry.get("frame_idx") != frame_id:
                continue
            det_path = entry.get("detections")
            if not det_path or not Path(det_path).exists():
                continue
            try:
                det = self._json_load(det_path)
            except Exception:
                continue
            for inst in det.get("instances", []):
                if inst.get("object_id") == obj_id:
                    return inst
            break
        return None

    def _get_mask_rle_for_frame(self, obj_id, frame_id, detection_frames):
        inst = self._get_instance_for_frame(obj_id, frame_id, detection_frames)
        if inst is None:
            return None
        return inst.get("mask_rle")

    @staticmethod
    def _instance_anchor_visibility(inst: dict) -> float:
        bbox = inst.get("bbox_xyxy") or inst.get("bbox")
        if not bbox or len(bbox) < 4:
            return 1.0
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        area = width * height
        size_score = min(area / 4000.0, 1.0)

        margin = min(x1, y1)
        edge_penalty = 1.0
        if x1 <= 2.0 or y1 <= 2.0:
            edge_penalty *= 0.35
        if width < 12.0 or height < 12.0:
            edge_penalty *= 0.5
        return max(0.1, size_score * edge_penalty)

    def _track_is_low_quality_edge_fragment(self, obj_id, traj, detection_frames) -> bool:
        boxes = []
        for rec in traj or []:
            fid = rec.get("frame_id")
            if fid is None:
                continue
            inst = self._get_instance_for_frame(obj_id, fid, detection_frames)
            if inst is None:
                continue
            bbox = inst.get("bbox_xyxy") or inst.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            boxes.append((x1, y1, width, height))
        if len(boxes) < 3:
            return False
        max_area = max(width * height for _, _, width, height in boxes)
        edge_ratio = sum(1 for x1, y1, _, _ in boxes if x1 <= 1.0 or y1 <= 1.0) / float(len(boxes))
        return max_area < 300.0 and edge_ratio >= 0.8

    @staticmethod
    def _icp_anchor(mesh_verts, target_pts, traj=None, scene_up=None):
        """Find R, t, s that aligns mesh_verts to target_pts.

        Initialize rotation from source/target principal frames, choose the
        lowest-residual signed axis mapping, then run rotation-only ICP while
        solving per-axis scale in the object's local frame.
        Returns (center, scale_xyz, R_3x3).
        """
        import numpy as np

        solution = ProjectExecutor._solve_icp_alignment(
            mesh_verts,
            target_pts,
            traj=traj,
            scene_up=scene_up,
        )
        center = np.asarray(solution["center"], dtype=np.float64).tolist()
        scale = np.asarray(solution["scale"], dtype=np.float64).tolist()
        rotation = np.asarray(solution["rotation"], dtype=np.float64)
        return center, scale, rotation

    @staticmethod
    def _solve_icp_alignment(mesh_verts, target_pts, traj=None, scene_up=None, seed_rotations=None):
        import numpy as np

        if len(mesh_verts) > 1000:
            src = mesh_verts[np.linspace(0, len(mesh_verts) - 1, 1000, dtype=int)]
        else:
            src = np.asarray(mesh_verts, dtype=np.float64).copy()
        target_pts = trim_point_cloud_outliers(target_pts, min_keep=32)
        if len(target_pts) > 2000:
            tgt = target_pts[np.linspace(0, len(target_pts) - 1, 2000, dtype=int)]
        else:
            tgt = np.asarray(target_pts, dtype=np.float64).copy()

        src_c = src - src.mean(axis=0)
        center = tgt.mean(axis=0)
        tgt_c = tgt - center
        src_basis = ProjectExecutor._principal_axes(src_c)
        tgt_basis = ProjectExecutor._principal_axes(tgt_c)

        init_rotations: list[tuple[str, np.ndarray]] = []
        seen: set[tuple[float, ...]] = set()
        for idx, rot in enumerate(seed_rotations or [], start=1):
            arr = np.asarray(rot, dtype=np.float64)
            key = tuple(np.round(arr.reshape(-1), 6))
            if key in seen:
                continue
            seen.add(key)
            init_rotations.append((f"seed_{idx}", arr))

        for name, rot in [("identity", np.eye(3, dtype=np.float64))]:
            key = tuple(np.round(rot.reshape(-1), 6))
            if key in seen:
                continue
            seen.add(key)
            init_rotations.append((name, rot))

        for idx, axis_map in enumerate(ProjectExecutor._signed_permutation_rotations(), start=1):
            rot = tgt_basis @ axis_map @ src_basis.T
            key = tuple(np.round(rot.reshape(-1), 6))
            if key in seen:
                continue
            seen.add(key)
            init_rotations.append((f"perm_{idx}", rot))

        best_solution = None
        best_score = float("inf")
        for _, R_init in init_rotations:
            target_local = (R_init.T @ tgt_c.T).T
            scale = compute_axis_scale(src_c, target_local)
            current = (R_init @ (src_c * scale[None, :]).T).T
            current, R_delta, _ = ProjectExecutor._refine_icp_rotation(current, tgt_c, iters=6)
            R_total = R_delta @ R_init
            target_local = (R_total.T @ tgt_c.T).T
            scale = compute_axis_scale(src_c, target_local)
            current = (R_total @ (src_c * scale[None, :]).T).T
            current, R_delta_final, residual = ProjectExecutor._refine_icp_rotation(current, tgt_c, iters=4)
            R_total = R_delta_final @ R_total
            target_local = (R_total.T @ tgt_c.T).T
            scale = compute_axis_scale(src_c, target_local)
            current = (R_total @ (src_c * scale[None, :]).T).T
            penalty = ProjectExecutor._alignment_prior_penalty(current, traj=traj, scene_up=scene_up)
            penalty += ProjectExecutor._alignment_target_penalty(current, tgt_c, scale=scale)
            score = float(residual + penalty)
            if score < best_score:
                best_solution = {
                    "rotation": R_total,
                    "scale": scale,
                    "aligned": current,
                    "residual": float(residual),
                    "penalty": float(penalty),
                    "score": score,
                }
                best_score = score

        if best_solution is None:
            best_solution = {
                "rotation": np.eye(3, dtype=np.float64),
                "scale": np.ones(3, dtype=np.float64),
                "aligned": src_c.copy(),
                "residual": float("inf"),
                "penalty": 0.0,
                "score": float("inf"),
            }

        current = (best_solution["rotation"] @ (src_c * best_solution["scale"][None, :]).T).T
        current, R_delta, residual = ProjectExecutor._refine_icp_rotation(current, tgt_c, iters=10)
        R_final = R_delta @ best_solution["rotation"]
        target_local = (R_final.T @ tgt_c.T).T
        scale_final = compute_axis_scale(src_c, target_local)
        aligned_final = (R_final @ (src_c * scale_final[None, :]).T).T
        penalty_final = ProjectExecutor._alignment_prior_penalty(aligned_final, traj=traj, scene_up=scene_up)
        penalty_final += ProjectExecutor._alignment_target_penalty(aligned_final, tgt_c, scale=scale_final)
        return {
            "center": center,
            "rotation": R_final,
            "scale": scale_final,
            "aligned": aligned_final,
            "residual": float(residual),
            "penalty": float(penalty_final),
            "score": float(residual + penalty_final),
            "source_points": src_c,
            "target_points": tgt_c,
        }

    @staticmethod
    def _resolve_frame_alignment(
        mesh_verts,
        target_pts,
        *,
        anchor_rotation,
        anchor_scale=None,
        anchor_points=None,
        traj=None,
        scene_up=None,
        frame_id=None,
        mask_rle=None,
        cam_traj=None,
        wildgs_poses=None,
        wildgs_K=None,
        previous_rotation=None,
        previous_points=None,
        source_axis_roles=None,
    ):
        import numpy as np

        target_arr = trim_point_cloud_outliers(target_pts, min_keep=32)
        if target_arr.ndim != 2 or target_arr.shape[0] < 16:
            center = np.asarray(target_pts, dtype=np.float64).mean(axis=0)
            return {
                "center": center,
                "rotation": np.asarray(anchor_rotation, dtype=np.float64),
                "scale": np.ones(3, dtype=np.float64),
                "residual": float("inf"),
                "score": float("inf"),
            }
        if target_arr.shape[0] > 512:
            target_arr = target_arr[np.linspace(0, target_arr.shape[0] - 1, 512, dtype=int)]

        center = target_arr.mean(axis=0)
        target_centered = target_arr - center
        source_arr = np.asarray(mesh_verts, dtype=np.float64)
        source_centered = source_arr - source_arr.mean(axis=0)
        if source_centered.shape[0] > 1000:
            source_centered = source_centered[np.linspace(0, source_centered.shape[0] - 1, 1000, dtype=int)]

        if source_axis_roles:
            merged_axis_roles = dict(ProjectExecutor._infer_source_axis_roles(source_centered))
            merged_axis_roles.update(source_axis_roles)
            source_axis_roles = merged_axis_roles
        else:
            source_axis_roles = ProjectExecutor._infer_source_axis_roles(source_centered)

        anchor_rot = np.asarray(anchor_rotation, dtype=np.float64)
        seed_rotations: list[np.ndarray] = [anchor_rot]
        previous_rot = None if previous_rotation is None else np.asarray(previous_rotation, dtype=np.float64)
        if previous_rot is not None:
            seed_rotations.insert(0, previous_rot)

        source_basis = ProjectExecutor._alignment_basis(source_centered, scene_up=scene_up)
        target_basis = ProjectExecutor._alignment_basis(target_centered, scene_up=scene_up)
        seed_rotations.append(target_basis @ source_basis.T)
        seed_rotations.append(
            ProjectExecutor._principal_axes(target_centered) @ ProjectExecutor._principal_axes(source_centered).T
        )

        seen: set[tuple[float, ...]] = set()
        candidates = []
        for rotation in seed_rotations:
            rot = np.asarray(rotation, dtype=np.float64)
            key = tuple(np.round(rot.reshape(-1), 6))
            if key in seen:
                continue
            seen.add(key)
            candidate = ProjectExecutor._refine_alignment_candidate(
                source_centered,
                target_centered,
                rot,
                traj=traj,
                scene_up=scene_up,
            )
            candidate["center"] = center
            if mask_rle and frame_id is not None:
                candidate["projection_score"] = ProjectExecutor._candidate_projection_score(
                    source_centered,
                    candidate,
                    frame_id=frame_id,
                    mask_rle=mask_rle,
                    cam_traj=cam_traj,
                    wildgs_poses=wildgs_poses,
                    wildgs_K=wildgs_K,
                )
            candidates.append(candidate)

        if candidates:
            best_solution = ProjectExecutor._select_alignment_candidate(
                candidates,
                scene_up=scene_up,
                source_axis_roles=source_axis_roles,
                previous_rotation=previous_rot,
            )
        else:
            best_solution = {
                "rotation": anchor_rot,
                "scale": np.ones(3, dtype=np.float64),
                "aligned": source_centered.copy(),
                "residual": float("inf"),
                "penalty": 0.0,
                "score": float("inf"),
                "center": center,
            }
        return best_solution

    @staticmethod
    def _candidate_from_rotation(
        mesh_verts,
        target_pts,
        rotation,
        *,
        traj=None,
        scene_up=None,
        coarse_iters=0,
        fine_iters=0,
    ):
        import numpy as np

        target_arr = trim_point_cloud_outliers(target_pts, min_keep=32)
        center = np.asarray(target_arr, dtype=np.float64).mean(axis=0)
        source_centered = np.asarray(mesh_verts, dtype=np.float64)
        source_centered = source_centered - source_centered.mean(axis=0, keepdims=True)
        target_centered = target_arr - center
        candidate = ProjectExecutor._refine_alignment_candidate(
            source_centered,
            target_centered,
            rotation,
            traj=traj,
            scene_up=scene_up,
            coarse_iters=coarse_iters,
            fine_iters=fine_iters,
        )
        candidate["center"] = center
        return candidate

    @staticmethod
    def _basis_transfer_alignment(
        mesh_verts,
        anchor_pts,
        target_pts,
        *,
        anchor_rotation,
        traj=None,
        scene_up=None,
    ):
        import numpy as np

        anchor_arr = trim_point_cloud_outliers(anchor_pts, min_keep=32)
        target_arr = trim_point_cloud_outliers(target_pts, min_keep=32)
        anchor_centered = anchor_arr - anchor_arr.mean(axis=0, keepdims=True)
        target_centered = target_arr - target_arr.mean(axis=0, keepdims=True)
        anchor_basis = ProjectExecutor._alignment_basis(anchor_centered, traj=traj, scene_up=scene_up)
        target_basis = ProjectExecutor._alignment_basis(target_centered, traj=traj, scene_up=scene_up)
        rotation = target_basis @ anchor_basis.T @ np.asarray(anchor_rotation, dtype=np.float64)
        return ProjectExecutor._candidate_from_rotation(
            mesh_verts,
            target_arr,
            rotation,
            traj=traj,
            scene_up=scene_up,
            coarse_iters=0,
            fine_iters=0,
        )

    @staticmethod
    def _refine_alignment_candidate(
        source_centered,
        target_centered,
        rotation,
        *,
        traj=None,
        scene_up=None,
        coarse_iters=3,
        fine_iters=2,
    ):
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        target_local = (rot.T @ target_centered.T).T
        scale = compute_axis_scale(source_centered, target_local)
        current = (rot @ (source_centered * scale[None, :]).T).T
        current, delta_rot, _ = ProjectExecutor._refine_icp_rotation(current, target_centered, iters=coarse_iters)
        rot = delta_rot @ rot
        target_local = (rot.T @ target_centered.T).T
        scale = compute_axis_scale(source_centered, target_local)
        current = (rot @ (source_centered * scale[None, :]).T).T
        current, delta_rot, residual = ProjectExecutor._refine_icp_rotation(current, target_centered, iters=fine_iters)
        rot = delta_rot @ rot
        target_local = (rot.T @ target_centered.T).T
        scale = compute_axis_scale(source_centered, target_local)
        current = (rot @ (source_centered * scale[None, :]).T).T
        penalty = ProjectExecutor._alignment_prior_penalty(current, traj=traj, scene_up=scene_up)
        penalty += ProjectExecutor._alignment_target_penalty(current, target_centered, scale=scale)
        return {
            "rotation": rot,
            "scale": scale,
            "aligned": current,
            "residual": float(residual),
            "penalty": float(penalty),
            "score": float(residual + penalty),
        }

    @staticmethod
    def _projection_camera(frame_id, cam_traj, wildgs_poses, wildgs_K):
        import numpy as np

        pose = next((p for p in (cam_traj or []) if int(p.get("frame_id", -1)) == int(frame_id)), None)
        K = pose.get("K") if pose else None
        if K is not None:
            fx, fy = float(K[0][0]), float(K[1][1])
            cx, cy = float(K[0][2]), float(K[1][2])
        elif wildgs_K:
            fx, fy = float(wildgs_K["fx"]), float(wildgs_K["fy"])
            cx, cy = float(wildgs_K["cx"]), float(wildgs_K["cy"])
        else:
            return None

        T = None
        target_frame = max(int(frame_id) - 1, 0)
        if wildgs_poses:
            wpose = next((p for p in wildgs_poses if int(p.get("frame", -10**9)) == target_frame), None)
            if wpose is None:
                wpose = min(wildgs_poses, key=lambda p: abs(int(p.get("frame", -10**9)) - target_frame))
            T = wpose.get("T_world_from_cam")
        if T is None and pose is not None and pose.get("R") is not None:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(pose["R"], dtype=np.float64)
            T[:3, 3] = np.asarray(pose.get("t", [0.0, 0.0, 0.0]), dtype=np.float64)
        if T is None:
            return None
        T = np.asarray(T, dtype=np.float64)
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "R": T[:3, :3],
            "t": T[:3, 3],
        }

    @staticmethod
    def _project_points_to_image(points_world, camera, *, width, height):
        import numpy as np

        arr = np.asarray(points_world, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        pts_cam = (camera["R"].T @ (arr - camera["t"]).T).T
        keep = pts_cam[:, 2] > 1e-6
        pts_cam = pts_cam[keep]
        if pts_cam.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        x = camera["fx"] * pts_cam[:, 0] / pts_cam[:, 2] + camera["cx"]
        y = camera["fy"] * pts_cam[:, 1] / pts_cam[:, 2] + camera["cy"]
        keep = (x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))
        if not np.any(keep):
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([x[keep], y[keep]], axis=1)

    @staticmethod
    def _bbox_from_pixels(pixels, *, width, height):
        import numpy as np

        arr = np.asarray(pixels, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        xs = np.clip(np.rint(arr[:, 0]).astype(int), 0, max(int(width) - 1, 0))
        ys = np.clip(np.rint(arr[:, 1]).astype(int), 0, max(int(height) - 1, 0))
        if xs.size == 0 or ys.size == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @staticmethod
    def _bbox_iou(a, b):
        if a is None or b is None:
            return None
        x1 = max(int(a[0]), int(b[0]))
        y1 = max(int(a[1]), int(b[1]))
        x2 = min(int(a[2]), int(b[2]))
        y2 = min(int(a[3]), int(b[3]))
        inter_w = max(0, x2 - x1 + 1)
        inter_h = max(0, y2 - y1 + 1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0, int(a[2]) - int(a[0]) + 1) * max(0, int(a[3]) - int(a[1]) + 1)
        area_b = max(0, int(b[2]) - int(b[0]) + 1) * max(0, int(b[3]) - int(b[1]) + 1)
        union = area_a + area_b - inter
        return float(inter / max(union, 1))

    @staticmethod
    def _candidate_projection_score(mesh_verts, candidate, *, frame_id, mask_rle, cam_traj, wildgs_poses, wildgs_K):
        import numpy as np

        if not mask_rle:
            return float("-inf")
        try:
            from pycocotools import mask as mask_util

            rle = json.loads(mask_rle) if isinstance(mask_rle, str) else mask_rle
            bbox_xywh = mask_util.toBbox(rle)
            mask = mask_util.decode(rle).astype(bool)
        except Exception:
            return float("-inf")
        if bbox_xywh is None:
            return float("-inf")

        camera = ProjectExecutor._projection_camera(frame_id, cam_traj, wildgs_poses, wildgs_K)
        if camera is None:
            return float("-inf")

        source = np.asarray(mesh_verts, dtype=np.float64)
        if source.ndim != 2 or source.shape[0] == 0:
            return float("-inf")
        if source.shape[0] > 768:
            source = source[np.linspace(0, source.shape[0] - 1, 768, dtype=int)]

        rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        scale = np.asarray(candidate["scale"], dtype=np.float64)
        center = np.asarray(candidate["center"], dtype=np.float64)
        world = (rotation @ (source * scale[None, :]).T).T + center[None, :]
        height, width = mask.shape[:2]
        pixels = ProjectExecutor._project_points_to_image(world, camera, width=width, height=height)
        if pixels.shape[0] == 0:
            return float("-inf")

        proj_bbox = ProjectExecutor._bbox_from_pixels(pixels, width=width, height=height)
        mask_bbox = [
            int(round(float(bbox_xywh[0]))),
            int(round(float(bbox_xywh[1]))),
            int(round(float(bbox_xywh[0] + bbox_xywh[2]))),
            int(round(float(bbox_xywh[1] + bbox_xywh[3]))),
        ]
        bbox_iou = ProjectExecutor._bbox_iou(mask_bbox, proj_bbox) or 0.0

        xs = np.clip(np.rint(pixels[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(pixels[:, 1]).astype(int), 0, height - 1)
        if xs.size == 0 or ys.size == 0:
            return float(bbox_iou)

        in_mask = float(mask[ys, xs].mean())
        unique_pixels = np.unique(np.stack([xs, ys], axis=1), axis=0)
        proj_area = max(float(unique_pixels.shape[0]), 1.0)
        mask_area = max(float(mask.sum()), 1.0)
        fill_ratio = min(proj_area / mask_area, 1.0)
        return float(0.5 * in_mask + 0.35 * bbox_iou + 0.15 * fill_ratio)

    @staticmethod
    def _infer_source_axis_roles(source_points):
        import numpy as np

        arr = np.asarray(source_points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return {}
        extents = np.ptp(arr, axis=0)
        order = np.argsort(extents)
        roles = {"extents": extents.tolist()}
        if float(extents[order[0]]) > 1e-6 and float(extents[order[1]]) / float(extents[order[0]]) > 1.05:
            roles["up_axis_idx"] = int(order[0])
        if float(extents[order[1]]) > 1e-6 and float(extents[order[2]]) / float(extents[order[1]]) > 1.05:
            roles["forward_axis_idx"] = int(order[2])
        return roles

    @staticmethod
    def _mesh_axis_prior_for_pose_optimizer(source_points, *, axis_roles: dict | None = None) -> dict:
        import numpy as np

        roles = dict(axis_roles or {})
        arr = np.asarray(source_points, dtype=np.float64) if source_points is not None else np.zeros((0, 3), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 3:
            return {"available": False, "reason": "missing_mesh_points"}

        extents = np.ptp(arr, axis=0)
        order = np.argsort(extents)
        # Current SAM3D vehicle GLBs encode local +Y as roof/up.
        # Keep the axis fixed to Y and lock the sign to the verified semantic up.
        up_axis = 1

        try:
            forward_axis = int(roles.get("forward_axis_idx", int(order[-1])))
        except Exception:
            forward_axis = int(order[-1])
        if forward_axis not in (0, 1, 2) or forward_axis == up_axis:
            remaining = [int(axis) for axis in order[::-1] if int(axis) != up_axis]
            forward_axis = remaining[0] if remaining else int(order[-1])

        remaining = [axis for axis in (0, 1, 2) if axis not in {up_axis, forward_axis}]
        right_axis = remaining[0] if remaining else None

        bounds = np.stack([arr.min(axis=0), arr.max(axis=0)], axis=0)
        low = float(bounds[0, up_axis])
        high = float(bounds[1, up_axis])
        span = max(1e-6, high - low)
        low_count = int(np.count_nonzero(arr[:, up_axis] <= low + 0.28 * span))
        high_count = int(np.count_nonzero(arr[:, up_axis] >= high - 0.28 * span))
        up_sign = 1.0
        up_sign_source = "sam3d_vehicle_local_positive_y_roof_prior"
        try:
            if roles.get("up_axis_sign") is not None:
                role_up_sign = -1.0 if float(roles.get("up_axis_sign")) < 0.0 else 1.0
                up_sign_source = (
                    "axis_roles_positive_y"
                    if role_up_sign > 0.0
                    else "semantic_positive_y_overrides_axis_roles"
                )
        except Exception:
            pass

        if roles.get("forward_axis_sign") is not None:
            try:
                forward_sign = -1.0 if float(roles.get("forward_axis_sign")) < 0.0 else 1.0
            except Exception:
                forward_sign = 1.0
        else:
            forward_sign = 1.0

        return {
            "available": True,
            "source": "mesh_bbox_axis_roles",
            "up_axis_idx": int(up_axis),
            "up_sign": float(up_sign),
            "up_sign_candidates": [float(up_sign)],
            "lock_up_sign": True,
            "up_sign_source": up_sign_source,
            "up_sign_density": {
                "low_count": low_count,
                "high_count": high_count,
                "low_value": low,
                "high_value": high,
            },
            "forward_axis_idx": int(forward_axis),
            "forward_sign": float(forward_sign),
            "forward_sign_candidates": [float(forward_sign), float(-forward_sign)],
            "forward_sign_source": "positive_long_axis_primary",
            "lock_forward_sign": False,
            "right_axis_idx": int(right_axis) if right_axis is not None else None,
            "extents": [float(v) for v in extents.tolist()],
            "confidence": 0.75,
        }

    @staticmethod
    def _reference_axis_roles_from_rotation(rotation, *, scene_up=None, source_points=None):
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        roles = dict(ProjectExecutor._infer_source_axis_roles(source_points))
        if rot.shape != (3, 3):
            return roles

        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm

        up_axis_idx = roles.get("up_axis_idx")
        if up_axis_idx is None:
            up_dots = [float(rot[:, i] @ up_ref) for i in range(3)]
            up_axis_idx = int(np.argmax(np.abs(up_dots)))
            roles["up_axis_idx"] = up_axis_idx
            roles["up_axis_sign"] = 1.0 if up_dots[up_axis_idx] >= 0.0 else -1.0
        else:
            up_axis = rot[:, int(up_axis_idx)]
            roles["up_axis_sign"] = 1.0 if float(up_axis @ up_ref) >= 0.0 else -1.0

        if roles.get("forward_axis_idx") is None:
            remaining = [i for i in range(3) if i != int(up_axis_idx)]
            if remaining:
                extents = np.asarray(roles.get("extents", [1.0, 1.0, 1.0]), dtype=np.float64)
                forward_axis_idx = max(
                    remaining,
                    key=lambda idx: (
                        float(extents[idx]) if idx < extents.shape[0] else 0.0,
                        float(np.linalg.norm(ProjectExecutor._horizontal_unit(rot[:, idx], scene_up=up_ref))),
                    ),
                )
                roles["forward_axis_idx"] = int(forward_axis_idx)
        return roles

    @staticmethod
    def _candidate_orientation_penalty(rotation, *, scene_up=None, source_axis_roles=None, previous_rotation=None):
        import math
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape != (3, 3):
            return 0.0
        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm

        penalty = 0.0
        roles = source_axis_roles or {}
        up_axis_idx = roles.get("up_axis_idx")
        if up_axis_idx is not None:
            up_axis = rot[:, int(up_axis_idx)]
            up_align = abs(float(up_axis @ up_ref))
            vertical_scores = [abs(float(rot[:, idx] @ up_ref)) for idx in range(3)]
            dominant_up_idx = int(np.argmax(vertical_scores))
            if dominant_up_idx != int(up_axis_idx):
                penalty += 0.65
            penalty += max(0.0, 0.92 - up_align) * 0.55
            up_axis_sign = roles.get("up_axis_sign")
            if up_axis_sign is not None and float(up_axis @ up_ref) * float(up_axis_sign) < 0.0:
                penalty += 0.45

        forward_axis_idx = roles.get("forward_axis_idx")
        if forward_axis_idx is not None:
            forward_axis = rot[:, int(forward_axis_idx)]
            penalty += max(0.0, abs(float(forward_axis @ up_ref)) - 0.28) * 0.25

        if previous_rotation is not None:
            prev = np.asarray(previous_rotation, dtype=np.float64)
            if prev.shape == (3, 3):
                delta = prev.T @ rot
                cos_theta = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) * 0.5))
                angle_deg = math.degrees(math.acos(cos_theta))
                penalty += max(0.0, angle_deg - 25.0) / 65.0 * 0.28
                if angle_deg > 90.0:
                    penalty += min((angle_deg - 90.0) / 90.0, 1.0) * 0.35
                if up_axis_idx is not None:
                    prev_up = prev[:, int(up_axis_idx)]
                    cur_up = rot[:, int(up_axis_idx)]
                    prev_sign = float(prev_up @ up_ref)
                    cur_sign = float(cur_up @ up_ref)
                    if prev_sign * cur_sign < 0.0:
                        penalty += 0.35
                    up_consistency = max(-1.0, min(1.0, float(prev_up @ cur_up)))
                    up_angle = math.degrees(math.acos(up_consistency))
                    penalty += max(0.0, up_angle - 20.0) / 70.0 * 0.22
                if forward_axis_idx is not None:
                    prev_forward = ProjectExecutor._horizontal_unit(prev[:, int(forward_axis_idx)], scene_up=up_ref)
                    cur_forward = ProjectExecutor._horizontal_unit(rot[:, int(forward_axis_idx)], scene_up=up_ref)
                    if float(np.linalg.norm(prev_forward)) > 1e-6 and float(np.linalg.norm(cur_forward)) > 1e-6:
                        forward_dot = max(-1.0, min(1.0, float(prev_forward @ cur_forward)))
                        forward_angle = math.degrees(math.acos(forward_dot))
                        penalty += max(0.0, forward_angle - 45.0) / 90.0 * 0.18
        return float(penalty)

    @staticmethod
    def _rotation_geodesic_deg(lhs, rhs):
        import math
        import numpy as np

        a = np.asarray(lhs, dtype=np.float64)
        b = np.asarray(rhs, dtype=np.float64)
        if a.shape != (3, 3) or b.shape != (3, 3):
            return float("inf")
        delta = a.T @ b
        cos_theta = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) * 0.5))
        return float(math.degrees(math.acos(cos_theta)))

    @staticmethod
    def _rotation_up_role(rotation, *, scene_up=None):
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape != (3, 3):
            return None, 0.0
        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm
        scores = [abs(float(rot[:, idx] @ up_ref)) for idx in range(3)]
        role = int(np.argmax(scores))
        return role, float(scores[role])

    @staticmethod
    def _stabilize_depth_rotations(depth_rotations, *, scene_up=None, source_axis_roles=None):
        import numpy as np

        if not depth_rotations:
            return depth_rotations

        roles = source_axis_roles or {}
        stable_up_idx = roles.get("up_axis_idx")
        stabilized = {
            int(fid): np.asarray(rot, dtype=np.float64)
            for fid, rot in depth_rotations.items()
        }
        sorted_fids = sorted(stabilized.keys())

        def _interp_between(fid0, rot0, fid1, rot1, fid):
            if fid1 == fid0:
                return np.asarray(rot0, dtype=np.float64)
            t = (fid - fid0) / float(fid1 - fid0)
            q0 = ProjectExecutor._rotation_matrix_to_quat_xyzw(rot0)
            q1 = ProjectExecutor._rotation_matrix_to_quat_xyzw(rot1)
            return np.asarray(
                ProjectExecutor._quat_xyzw_to_rotation_matrix(
                    ProjectExecutor._slerp_quat(q0, q1, t)
                ),
                dtype=np.float64,
            )

        if stable_up_idx is not None and len(sorted_fids) >= 3:
            idx = 0
            while idx < len(sorted_fids):
                fid = sorted_fids[idx]
                role, _ = ProjectExecutor._rotation_up_role(stabilized[fid], scene_up=scene_up)
                if role == stable_up_idx:
                    idx += 1
                    continue
                start = idx
                current_role = role
                while idx + 1 < len(sorted_fids):
                    next_role, _ = ProjectExecutor._rotation_up_role(
                        stabilized[sorted_fids[idx + 1]],
                        scene_up=scene_up,
                    )
                    if next_role != current_role:
                        break
                    idx += 1
                end = idx
                seg_len = end - start + 1
                prev_idx = start - 1
                next_idx = end + 1
                if seg_len <= 3 and prev_idx >= 0 and next_idx < len(sorted_fids):
                    prev_fid = sorted_fids[prev_idx]
                    next_fid = sorted_fids[next_idx]
                    prev_role, _ = ProjectExecutor._rotation_up_role(stabilized[prev_fid], scene_up=scene_up)
                    next_role, _ = ProjectExecutor._rotation_up_role(stabilized[next_fid], scene_up=scene_up)
                    if prev_role == stable_up_idx and next_role == stable_up_idx:
                        prev_rot = stabilized[prev_fid]
                        next_rot = stabilized[next_fid]
                        for seg_idx in range(start, end + 1):
                            cur_fid = sorted_fids[seg_idx]
                            stabilized[cur_fid] = _interp_between(prev_fid, prev_rot, next_fid, next_rot, cur_fid)
                idx += 1

        for _ in range(2):
            for idx in range(1, len(sorted_fids) - 1):
                prev_fid = sorted_fids[idx - 1]
                cur_fid = sorted_fids[idx]
                next_fid = sorted_fids[idx + 1]
                prev_rot = stabilized[prev_fid]
                cur_rot = stabilized[cur_fid]
                next_rot = stabilized[next_fid]
                prev_angle = ProjectExecutor._rotation_geodesic_deg(prev_rot, cur_rot)
                next_angle = ProjectExecutor._rotation_geodesic_deg(cur_rot, next_rot)
                bridge_angle = ProjectExecutor._rotation_geodesic_deg(prev_rot, next_rot)
                if prev_angle > 55.0 and next_angle > 55.0 and bridge_angle < 35.0:
                    stabilized[cur_fid] = _interp_between(prev_fid, prev_rot, next_fid, next_rot, cur_fid)

        if len(sorted_fids) >= 2 and stable_up_idx is not None:
            first_fid, second_fid = sorted_fids[0], sorted_fids[1]
            first_role, _ = ProjectExecutor._rotation_up_role(stabilized[first_fid], scene_up=scene_up)
            second_role, _ = ProjectExecutor._rotation_up_role(stabilized[second_fid], scene_up=scene_up)
            if (
                first_role != stable_up_idx
                and second_role == stable_up_idx
                and ProjectExecutor._rotation_geodesic_deg(stabilized[first_fid], stabilized[second_fid]) > 70.0
            ):
                stabilized[first_fid] = stabilized[second_fid].copy()

            prev_fid, last_fid = sorted_fids[-2], sorted_fids[-1]
            prev_role, _ = ProjectExecutor._rotation_up_role(stabilized[prev_fid], scene_up=scene_up)
            last_role, _ = ProjectExecutor._rotation_up_role(stabilized[last_fid], scene_up=scene_up)
            if (
                last_role != stable_up_idx
                and prev_role == stable_up_idx
                and ProjectExecutor._rotation_geodesic_deg(stabilized[prev_fid], stabilized[last_fid]) > 70.0
            ):
                stabilized[last_fid] = stabilized[prev_fid].copy()

        max_step_deg_per_frame = 35.0
        for idx in range(1, len(sorted_fids)):
            prev_fid = sorted_fids[idx - 1]
            cur_fid = sorted_fids[idx]
            prev_rot = stabilized[prev_fid]
            cur_rot = stabilized[cur_fid]
            gap = max(int(cur_fid - prev_fid), 1)
            max_step = max_step_deg_per_frame * float(gap)
            angle = ProjectExecutor._rotation_geodesic_deg(prev_rot, cur_rot)
            if angle > max_step + 1e-6:
                t = max_step / max(angle, 1e-6)
                stabilized[cur_fid] = _interp_between(prev_fid, prev_rot, cur_fid, cur_rot, prev_fid + t * gap)

        return {fid: stabilized[fid].tolist() for fid in sorted_fids}

    @staticmethod
    def _select_alignment_candidate(candidates, *, scene_up=None, source_axis_roles=None, previous_rotation=None):
        import math

        if not candidates:
            raise ValueError("alignment candidate list is empty")
        best = None
        best_effective = float("-inf")
        for candidate in candidates:
            projection = float(candidate.get("projection_score", float("-inf")))
            projection_finite = math.isfinite(projection)
            penalty = ProjectExecutor._candidate_orientation_penalty(
                candidate.get("rotation"),
                scene_up=scene_up,
                source_axis_roles=source_axis_roles,
                previous_rotation=previous_rotation,
            )
            candidate["orientation_penalty"] = float(penalty)
            penalty = float(candidate.get("orientation_penalty", 0.0))
            effective = projection - penalty if projection_finite else float("-inf")
            best_finite = math.isfinite(best_effective)
            if best is None:
                best = candidate
                best_effective = effective
                continue
            if projection_finite and (not best_finite or effective > best_effective + 1e-6):
                best = candidate
                best_effective = effective
                continue
            if projection_finite == best_finite and (
                (not projection_finite)
                or abs(effective - best_effective) <= 1e-6
            ):
                lhs = float(candidate.get("score", float("inf"))) + float(candidate.get("orientation_penalty", 0.0))
                rhs = float(best.get("score", float("inf"))) + float(best.get("orientation_penalty", 0.0))
                if lhs < rhs:
                    best = candidate
                    best_effective = effective
        return best

    @staticmethod
    def _estimate_scene_up(points):
        import numpy as np

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 32 or arr.shape[1] != 3:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if len(arr) > 20000:
            arr = arr[np.linspace(0, len(arr) - 1, 20000, dtype=int)]
        arr = trim_point_cloud_outliers(arr, min_keep=128)
        centered = arr - arr.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = np.asarray(vh[-1], dtype=np.float64)
        norm = float(np.linalg.norm(up))
        if norm < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = up / norm
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if float(up @ world_up) < 0.0:
            up = -up
        return up

    @staticmethod
    def _alignment_residual(current, target_pts):
        import numpy as np
        from scipy.spatial import KDTree

        cur = np.asarray(current, dtype=np.float64)
        tgt = np.asarray(target_pts, dtype=np.float64)
        if cur.ndim != 2 or tgt.ndim != 2 or len(cur) == 0 or len(tgt) == 0:
            return float("inf")
        tree = KDTree(tgt)
        dists, _ = tree.query(cur)
        return float(np.mean(dists))

    @staticmethod
    def _principal_axes(points):
        import numpy as np

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 3:
            return np.eye(3, dtype=np.float64)
        centered = arr - arr.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.eye(3, dtype=np.float64)
        basis = np.asarray(vh.T, dtype=np.float64)
        if np.linalg.det(basis) < 0:
            basis[:, 2] *= -1.0
        return basis

    @staticmethod
    def _alignment_basis(points, traj=None, scene_up=None):
        import numpy as np

        pca = ProjectExecutor._principal_axes(points)
        axes = [pca[:, idx].copy() for idx in range(3)]
        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm
        up_idx = max(range(3), key=lambda idx: abs(float(axes[idx] @ up_ref)))
        up = axes[up_idx]
        if float(up @ up_ref) < 0.0:
            up = -up

        remaining = [idx for idx in range(3) if idx != up_idx]
        forward = None
        if remaining:
            forward_idx = max(
                remaining,
                key=lambda idx: float(np.linalg.norm(ProjectExecutor._horizontal_unit(axes[idx], scene_up=up_ref))),
            )
            candidate = ProjectExecutor._horizontal_unit(axes[forward_idx], scene_up=up_ref)
            if np.linalg.norm(candidate) > 1e-6:
                forward = candidate
        if forward is None:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        right = np.cross(up, forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right = right / np.linalg.norm(right)
        forward = np.cross(right, up)
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            forward = forward / np.linalg.norm(forward)
        basis = np.stack([forward, right, up], axis=1)
        if np.linalg.det(basis) < 0:
            basis[:, 1] *= -1.0
        return basis

    @staticmethod
    def _horizontal_unit(vec, scene_up=None):
        import numpy as np

        out = np.asarray(vec, dtype=np.float64).copy()
        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm
        out = out - float(out @ up_ref) * up_ref
        norm = float(np.linalg.norm(out))
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float64)
        return out / norm

    @staticmethod
    def _alignment_prior_penalty(points, traj=None, scene_up=None):
        import numpy as np

        basis = ProjectExecutor._principal_axes(points)
        short_axis = basis[:, 2]
        long_axis = basis[:, 0]
        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm
        short_axis_up_penalty = max(0.0, 0.75 - abs(float(short_axis @ up_ref))) * 0.2
        long_axis_tilt_penalty = max(0.0, abs(float(long_axis @ up_ref)) - 0.35) * 0.1
        return short_axis_up_penalty + long_axis_tilt_penalty

    @staticmethod
    def _alignment_target_penalty(points, target_points, *, scale, rotation=None, source_axis_roles=None, scene_up=None):
        import numpy as np

        cur_basis = ProjectExecutor._principal_axes(points)
        tgt_basis = ProjectExecutor._principal_axes(target_points)
        short_align = abs(float(cur_basis[:, 2] @ tgt_basis[:, 2]))
        long_align = max(
            abs(float(cur_basis[:, 0] @ tgt_basis[:, 0])),
            abs(float(cur_basis[:, 0] @ tgt_basis[:, 1])),
        )

        semantic_forward_penalty = 0.0
        up_ref = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up_ref))
        if up_norm < 1e-6:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up_ref = up_ref / up_norm

        cur_semantic = ProjectExecutor._target_semantic_basis(points, scene_up=up_ref)
        tgt_semantic = ProjectExecutor._target_semantic_basis(target_points, scene_up=up_ref)
        if cur_semantic is not None and tgt_semantic is not None:
            semantic_align = abs(float(cur_semantic[:, 0] @ tgt_semantic[:, 0]))
            semantic_forward_penalty += max(0.0, 0.85 - semantic_align) * 0.18

        roles = source_axis_roles or {}
        if rotation is not None and tgt_semantic is not None:
            rot = np.asarray(rotation, dtype=np.float64)
            if rot.shape == (3, 3):
                forward_axis_idx = roles.get("forward_axis_idx")
                if forward_axis_idx is not None:
                    src_forward = ProjectExecutor._horizontal_unit(rot[:, int(forward_axis_idx)], scene_up=up_ref)
                    tgt_forward = ProjectExecutor._horizontal_unit(tgt_semantic[:, 0], scene_up=up_ref)
                    if float(np.linalg.norm(src_forward)) > 1e-6 and float(np.linalg.norm(tgt_forward)) > 1e-6:
                        semantic_forward_penalty += max(0.0, 0.92 - abs(float(src_forward @ tgt_forward))) * 0.45
                up_axis_idx = roles.get("up_axis_idx")
                if up_axis_idx is not None:
                    src_up = rot[:, int(up_axis_idx)]
                    semantic_forward_penalty += max(0.0, 0.92 - abs(float(src_up @ up_ref))) * 0.22

        scale_arr = np.asarray(scale, dtype=np.float64)
        scale_ratio = float(scale_arr.max() / max(scale_arr.min(), 1e-6))
        scale_penalty = max(0.0, np.log(scale_ratio) - np.log(4.5)) * 0.2

        return (
            max(0.0, 0.9 - short_align) * 0.8
            + max(0.0, 0.65 - long_align) * 0.1
            + semantic_forward_penalty
            + scale_penalty
        )

    @staticmethod
    def _semantic_sign_matrices():
        import numpy as np

        return (
            np.diag([1.0, 1.0, 1.0]).astype(np.float64),
            np.diag([-1.0, -1.0, 1.0]).astype(np.float64),
            np.diag([1.0, -1.0, -1.0]).astype(np.float64),
            np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
        )

    @staticmethod
    def _source_semantic_basis(points):
        import numpy as np

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 8:
            return None
        extents = np.ptp(arr, axis=0)
        order = np.argsort(extents)
        up_idx = int(order[0])
        remaining = [idx for idx in range(3) if idx != up_idx]
        if len(remaining) != 2:
            return None
        forward_idx = max(remaining, key=lambda idx: float(extents[idx]))
        right_idx = next(idx for idx in remaining if idx != forward_idx)
        if float(extents[forward_idx]) <= max(float(extents[right_idx]) * 1.05, 1e-6):
            return None
        basis = np.eye(3, dtype=np.float64)[:, [forward_idx, right_idx, up_idx]]
        if np.linalg.det(basis) < 0:
            basis[:, 1] *= -1.0
        return basis

    @staticmethod
    def _target_semantic_basis(points, scene_up=None):
        import numpy as np

        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 8:
            return None
        up = np.asarray(scene_up if scene_up is not None else [0.0, 1.0, 0.0], dtype=np.float64)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1e-6:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            up = up / up_norm

        centered = arr - arr.mean(axis=0, keepdims=True)
        flat = centered - np.outer(centered @ up, up)
        horiz_norms = np.linalg.norm(flat, axis=1)
        keep = horiz_norms > 1e-6
        if int(np.count_nonzero(keep)) < 4:
            return None
        flat = flat[keep]
        try:
            _, singular, vh = np.linalg.svd(flat, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if singular.shape[0] < 2 or float(singular[0]) <= max(float(singular[1]) * 1.05, 1e-6):
            return None
        forward = vh[0]
        forward = ProjectExecutor._horizontal_unit(forward, scene_up=up)
        if float(np.linalg.norm(forward)) < 1e-6:
            return None
        right = np.cross(up, forward)
        if float(np.linalg.norm(right)) < 1e-6:
            return None
        right = right / np.linalg.norm(right)
        forward = np.cross(right, up)
        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
        basis = np.stack([forward, right, up], axis=1)
        if np.linalg.det(basis) < 0:
            basis[:, 1] *= -1.0
        return basis

    @staticmethod
    def _signed_permutation_rotations():
        import itertools
        import numpy as np

        mats = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                mat = np.zeros((3, 3), dtype=np.float64)
                for row, col in enumerate(perm):
                    mat[row, col] = signs[row]
                if np.linalg.det(mat) > 0.5:
                    mats.append(mat)
        return mats

    @staticmethod
    def _refine_icp_rotation(current, target_pts, *, iters):
        import numpy as np
        from scipy.spatial import KDTree

        cur = np.asarray(current, dtype=np.float64)
        tgt = np.asarray(target_pts, dtype=np.float64)
        tree = KDTree(tgt)
        R_accum = np.eye(3, dtype=np.float64)
        for _ in range(max(int(iters), 0)):
            _, indices = tree.query(cur)
            matched = tgt[indices]
            H = cur.T @ matched
            try:
                U, _, Vt = np.linalg.svd(H)
            except np.linalg.LinAlgError:
                break
            d = np.linalg.det(Vt.T @ U.T)
            R_step = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
            cur = (R_step @ cur.T).T
            R_accum = R_step @ R_accum
        _, indices = tree.query(cur)
        residual = float(np.mean(np.linalg.norm(cur - tgt[indices], axis=1)))
        return cur, R_accum, residual

    @staticmethod
    def _rotation_matrix_to_quat_xyzw(rotation):
        import math
        import numpy as np

        rot = np.asarray(rotation, dtype=np.float64)
        if rot.shape != (3, 3):
            return [0.0, 0.0, 0.0, 1.0]

        m00, m01, m02 = float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2])
        m10, m11, m12 = float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2])
        m20, m21, m22 = float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s
        quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
        quat /= max(float(np.linalg.norm(quat)), 1e-8)
        return quat.tolist()

    @staticmethod
    def _quat_xyzw_to_rotation_matrix(quat):
        import numpy as np

        q = np.asarray(quat, dtype=np.float64)
        if q.shape != (4,):
            return np.eye(3, dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-8)
        x, y, z, w = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _slerp_quat(q0, q1, t: float):
        import math
        import numpy as np

        qa = np.asarray(q0, dtype=np.float64)
        qb = np.asarray(q1, dtype=np.float64)
        qa /= max(float(np.linalg.norm(qa)), 1e-8)
        qb /= max(float(np.linalg.norm(qb)), 1e-8)
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot
        dot = max(min(dot, 1.0), -1.0)
        if dot > 0.9995:
            out = qa + float(t) * (qb - qa)
            out /= max(float(np.linalg.norm(out)), 1e-8)
            return out.tolist()
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * float(t)
        sin_theta = math.sin(theta)
        s0 = math.sin(theta_0 - theta) / max(sin_theta_0, 1e-8)
        s1 = sin_theta / max(sin_theta_0, 1e-8)
        out = s0 * qa + s1 * qb
        out /= max(float(np.linalg.norm(out)), 1e-8)
        return out.tolist()

    @staticmethod
    def _trajectory_visibility_segments(frame_ids: list[int]) -> list[tuple[int, int]]:
        frames = sorted({int(frame_id) for frame_id in frame_ids if int(frame_id) > 0})
        if not frames:
            return []

        segments: list[tuple[int, int]] = []
        start = frames[0]
        prev = frames[0]
        for frame_id in frames[1:]:
            if frame_id == prev + 1:
                prev = frame_id
                continue
            segments.append((start, prev))
            start = frame_id
            prev = frame_id
        segments.append((start, prev))
        return segments

    @staticmethod
    def _apply_trajectory_visibility_samples(
        imageable,
        frame_ids: list[int],
        *,
        stage_start_frame: int,
        stage_end_frame: int,
    ) -> list[tuple[int, int]]:
        from pxr import Usd, UsdGeom

        segments = ProjectExecutor._trajectory_visibility_segments(frame_ids)
        vis_attr = imageable.GetVisibilityAttr()
        vis_attr.Set(UsdGeom.Tokens.inherited)
        if not segments:
            vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(float(stage_start_frame)))
            return []

        if segments[0][0] > int(stage_start_frame):
            vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(float(stage_start_frame)))

        for start, end in segments:
            vis_attr.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(float(start)))
            if end < int(stage_end_frame):
                vis_attr.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(float(end + 1)))
        return segments

    def _export_usdc(
        self,
        usdc_path,
        sam3d_meshes,
        corrected_trajectories,
        bg_mesh_path,
        wildgs_poses,
        cam_traj,
        bg_meshes=None,
        conversion_report_path=None,
        fixed_camera_reference_frame_id=None,
        fixed_camera_road_plane=None,
    ):
        import numpy as np
        from pxr import Usd, UsdGeom, Gf, Sdf, Vt

        def _usd_points(points):
            arr = np.asarray(points, dtype=np.float64)
            return Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in arr])

        def _set_mesh_orientation(mesh_prim):
            mesh_prim.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            mesh_prim.CreateDoubleSidedAttr(True)

        def _set_display_colors(mesh_prim, vertex_colors):
            vc = np.asarray(vertex_colors)
            if vc.ndim != 2 or vc.shape[0] == 0:
                return
            colors = Vt.Vec3fArray(
                [Gf.Vec3f(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in vc]
            )
            UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            ).Set(colors)

        def _set_normals(mesh_prim, normals):
            arr = np.asarray(normals, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return
            mesh_prim.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in arr]))
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        def _set_quat(op, rotation_matrix, time_code=None):
            quat = ProjectExecutor._rotation_matrix_to_quat_xyzw(np.asarray(rotation_matrix, dtype=np.float64))
            value = Gf.Quatf(float(quat[3]), Gf.Vec3f(float(quat[0]), float(quat[1]), float(quat[2])))
            if time_code is None:
                op.Set(value)
            else:
                op.Set(value, time_code)

        def _valid_vec3(value) -> bool:
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(3)
            except Exception:
                return False
            return bool(np.all(np.isfinite(arr)))
        def _camera_pose_record_for_frame(frame_id: int):
            target_wildgs_frame = int(frame_id) - 1
            if wildgs_poses:
                exact = next((p for p in wildgs_poses if int(p.get("frame", -999999)) == target_wildgs_frame), None)
                pose = exact or min(wildgs_poses, key=lambda p: abs(int(p.get("frame", -999999)) - target_wildgs_frame))
                T = pose.get("T_world_from_cam")
                if T is not None:
                    try:
                        matrix = np.asarray(T, dtype=np.float64).reshape(4, 4)
                    except Exception:
                        matrix = None
                    if matrix is not None and np.all(np.isfinite(matrix)):
                        return pose, matrix
            if cam_traj:
                exact = next((p for p in cam_traj if int(p.get("frame_id", -999999)) == int(frame_id)), None)
                pose = exact or min(cam_traj, key=lambda p: abs(int(p.get("frame_id", -999999)) - int(frame_id)))
                try:
                    rotation = np.asarray(pose.get("R"), dtype=np.float64).reshape(3, 3)
                    translation = np.asarray(pose.get("t"), dtype=np.float64).reshape(3)
                except Exception:
                    return None, None
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, :3] = rotation
                matrix[:3, 3] = translation
                if np.all(np.isfinite(matrix)):
                    return pose, matrix
            return None, None

        fixed_camera_reference_frame_id = (
            int(fixed_camera_reference_frame_id)
            if fixed_camera_reference_frame_id is not None and int(fixed_camera_reference_frame_id) > 0
            else None
        )
        fixed_camera_reference_pose = None
        fixed_camera_T_ref = None
        fixed_camera_road_normal = None
        fixed_camera_road_offset = None
        if isinstance(fixed_camera_road_plane, dict):
            try:
                fixed_camera_road_normal = np.asarray(fixed_camera_road_plane.get("normal_world"), dtype=np.float64).reshape(3)
                fixed_camera_road_norm = float(np.linalg.norm(fixed_camera_road_normal))
                if fixed_camera_road_norm > 1e-8:
                    fixed_camera_road_normal = fixed_camera_road_normal / fixed_camera_road_norm
                    fixed_camera_road_offset = float(fixed_camera_road_plane.get("offset", 0.0))
                else:
                    fixed_camera_road_normal = None
            except Exception:
                fixed_camera_road_normal = None
                fixed_camera_road_offset = None
        if fixed_camera_reference_frame_id is not None:
            fixed_camera_reference_pose, fixed_camera_T_ref = _camera_pose_record_for_frame(fixed_camera_reference_frame_id)
            if fixed_camera_T_ref is None:
                fixed_camera_reference_frame_id = None

        def _fixed_camera_grounded_pose(frame_id: int, rotation, translation, local_vertices, scale):
            rot_world, trans_world = _fixed_camera_world_pose(frame_id, rotation, translation)
            if fixed_camera_T_ref is None or fixed_camera_road_normal is None or fixed_camera_road_offset is None:
                return rot_world, trans_world
            try:
                local = np.asarray(local_vertices, dtype=np.float64)
                scale_arr = np.asarray(scale, dtype=np.float64).reshape(3)
                transformed = (rot_world @ (local * scale_arr[None, :]).T).T
            except Exception:
                return rot_world, trans_world
            if transformed.ndim != 2 or transformed.shape[0] == 0 or transformed.shape[1] != 3:
                return rot_world, trans_world
            bottom_rel = float(np.min(transformed @ fixed_camera_road_normal))
            if not np.isfinite(bottom_rel):
                return rot_world, trans_world
            bottom_distance = float(fixed_camera_road_normal @ trans_world + fixed_camera_road_offset + bottom_rel)
            if not np.isfinite(bottom_distance):
                return rot_world, trans_world
            trans_world = trans_world - fixed_camera_road_normal * bottom_distance
            return rot_world, trans_world

        def _fixed_camera_world_pose(frame_id: int, rotation, translation):
            rot_world = np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            trans_world = np.asarray(translation, dtype=np.float64).reshape(3)
            if fixed_camera_T_ref is None:
                return rot_world, trans_world
            _, T_frame = _camera_pose_record_for_frame(int(frame_id))
            if T_frame is None:
                return rot_world, trans_world
            try:
                correction = fixed_camera_T_ref @ np.linalg.inv(T_frame)
            except np.linalg.LinAlgError:
                return rot_world, trans_world
            out_rot = correction[:3, :3] @ rot_world
            out_trans = correction[:3, :3] @ trans_world + correction[:3, 3]
            return out_rot, out_trans


        stage = Usd.Stage.CreateNew(str(usdc_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Determine FPS from trajectory timestamps
        all_frames = []
        for val in corrected_trajectories.values():
            flist = val.get("frames", val) if isinstance(val, dict) else val
            all_frames.extend(
                rec for rec in flist
                if _valid_vec3(rec.get("centroid_world")) and _valid_vec3(rec.get("scale"))
            )
        if len(all_frames) >= 2:
            sorted_f = sorted(all_frames, key=lambda f: f.get("frame_id", 0))
            dt_list = [b.get("timestamp_sec", 0) - a.get("timestamp_sec", 0)
                       for a, b in zip(sorted_f, sorted_f[1:])
                       if b.get("timestamp_sec", 0) - a.get("timestamp_sec", 0) > 0]
            fps = 1.0 / (sum(dt_list) / len(dt_list)) if dt_list else 30.0
        else:
            fps = 30.0
        stage.SetTimeCodesPerSecond(fps)
        max_frame = max((f.get("frame_id", 0) for f in all_frames), default=1)
        stage.SetStartTimeCode(1)
        stage.SetEndTimeCode(max_frame)

        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Objects")

        bg_meshes = list(bg_meshes or [])
        bg = None
        bg_verts_world = None
        scene_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if bg_meshes:
            import trimesh

            road_path = next((path for name, path in bg_meshes if name == "road"), bg_meshes[0][1])
            bg = trimesh.load(str(road_path), force="mesh")
            bg_verts_world = np.asarray(bg.vertices, dtype=np.float64)
            if bg_verts_world.ndim == 2 and bg_verts_world.shape[0] >= 3:
                scene_up = self._estimate_scene_up(bg_verts_world)
        elif bg_mesh_path:
            import trimesh

            bg = trimesh.load(str(bg_mesh_path), force="mesh")
            bg_verts_world = np.asarray(bg.vertices, dtype=np.float64)
            if bg_verts_world.ndim == 2 and bg_verts_world.shape[0] >= 3:
                scene_up = self._estimate_scene_up(bg_verts_world)

        # WildGS world uses CV camera convention: X=right, Y=DOWN, Z=forward.
        # USD uses Y-up convention: X=right, Y=UP, Z=back.
        # Transform for positions/points: x'=x, y'=-y, z'=-z
        # SAM3D mesh vertices are in their own Y=up space which already matches
        # USD Y-up, so mesh vertices are NOT transformed.
        coord_convention = USDCoordinateConvention(
            R_usd_from_world=np.diag([1.0, -1.0, -1.0]).astype(np.float64),
            scene_up_world=np.array([0.0, -1.0, 0.0], dtype=np.float64),
            scene_forward_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            ground_plane_offset_world=0.0,
            stage_up_axis="Y",
        )
        report_camera_track = [fixed_camera_reference_pose] if fixed_camera_reference_pose is not None else (wildgs_poses or cam_traj)
        coord_report = build_coordinate_report(coord_convention, camera_track=report_camera_track)
        coord_report["fixed_camera"] = {
            "enabled": fixed_camera_reference_frame_id is not None,
            "reference_frame_id": fixed_camera_reference_frame_id,
        }
        if conversion_report_path is not None:
            Path(conversion_report_path).write_text(json.dumps(coord_report, indent=2), encoding="utf-8")

        # Background mesh
        if bg_meshes:
            import trimesh

            UsdGeom.Xform.Define(stage, "/World/Background")
            for bg_name, bg_part_path in bg_meshes:
                bg_part = trimesh.load(str(bg_part_path), force="mesh")
                verts_world = np.asarray(bg_part.vertices, dtype=np.float64)
                if verts_world.ndim != 2 or len(verts_world) == 0:
                    continue
                bg_verts = convert_world_points_to_usd(verts_world, coord_convention)
                prim_name = "".join(part.capitalize() for part in str(bg_name).replace("-", "_").split("_")) or "Mesh"
                mesh_prim = UsdGeom.Mesh.Define(stage, f"/World/Background/{prim_name}")
                mesh_prim.CreatePointsAttr(_usd_points(bg_verts))
                mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(bg_part.faces)))
                mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(bg_part.faces.flatten().tolist()))
                _set_mesh_orientation(mesh_prim)
                vc = getattr(bg_part.visual, "vertex_colors", None)
                if vc is not None and len(vc) == len(bg_verts):
                    _set_display_colors(mesh_prim, vc)
                normals = getattr(bg_part, "vertex_normals", None)
                if normals is not None and len(normals) == len(bg_verts):
                    _set_normals(mesh_prim, convert_world_normals_to_usd(normals, coord_convention))
        elif bg is not None and bg_verts_world is not None:
            bg_verts = convert_world_points_to_usd(bg_verts_world, coord_convention)
            UsdGeom.Xform.Define(stage, "/World/Background")
            mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Background/Mesh")
            mesh_prim.CreatePointsAttr(_usd_points(bg_verts))
            mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(bg.faces)))
            mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray(bg.faces.flatten().tolist()))
            _set_mesh_orientation(mesh_prim)
            vc = getattr(bg.visual, "vertex_colors", None)
            if vc is not None and len(vc) == len(bg_verts):
                _set_display_colors(mesh_prim, vc)
            normals = getattr(bg, "vertex_normals", None)
            if normals is not None and len(normals) == len(bg_verts):
                _set_normals(mesh_prim, convert_world_normals_to_usd(normals, coord_convention))

        # Camera animation
        if wildgs_poses or cam_traj:
            UsdGeom.Xform.Define(stage, "/World/Cameras")
            cam = UsdGeom.Camera.Define(stage, "/World/Cameras/MainCamera")
            cam_xf = UsdGeom.Xformable(cam.GetPrim())
            cam_translate = cam_xf.AddTranslateOp()
            cam_orient = cam_xf.AddOrientOp()
            pose_with_k = next((p for p in (cam_traj or []) if p.get("K") is not None), None)
            if pose_with_k is not None:
                K = np.asarray(pose_with_k["K"], dtype=np.float64)
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
            elif wildgs_poses and wildgs_poses[0].get("intrinsics") is not None:
                intr = wildgs_poses[0]["intrinsics"]
                fx = float(intr["fx"])
                fy = float(intr["fy"])
                cx = float(intr["cx"])
                cy = float(intr["cy"])
            else:
                fx = fy = cx = cy = None
            if fx is not None and fy is not None and cx is not None and cy is not None:
                width_px = max(float(cx) * 2.0, 1.0)
                height_px = max(float(cy) * 2.0, 1.0)
                h_aperture = 20.955
                v_aperture = h_aperture * height_px / width_px
                focal_h = fx * h_aperture / width_px
                focal_v = fy * v_aperture / height_px
                cam.CreateHorizontalApertureAttr().Set(float(h_aperture))
                cam.CreateVerticalApertureAttr().Set(float(v_aperture))
                cam.CreateFocalLengthAttr().Set(float((focal_h + focal_v) * 0.5))
            if fixed_camera_T_ref is not None:
                T_usd = convert_cv_camera_pose_to_usd(fixed_camera_T_ref, coord_convention)
                cam_translate.Set(Gf.Vec3d(float(T_usd[0, 3]), float(T_usd[1, 3]), float(T_usd[2, 3])))
                _set_quat(cam_orient, T_usd[:3, :3])
            else:
                for wp in wildgs_poses:
                    frame_num = float(wp.get("frame", 0)) + 1
                    T = wp.get("T_world_from_cam")
                    if T:
                        T_usd = convert_cv_camera_pose_to_usd(T, coord_convention)
                        cam_translate.Set(
                            Gf.Vec3d(float(T_usd[0, 3]), float(T_usd[1, 3]), float(T_usd[2, 3])),
                            frame_num,
                        )
                        _set_quat(cam_orient, T_usd[:3, :3], frame_num)
                if not wildgs_poses and cam_traj:
                    for pose in cam_traj:
                        frame_num = float(pose.get("frame_id", 0) or 0)
                        if frame_num <= 0:
                            continue
                        _, T = _camera_pose_record_for_frame(int(frame_num))
                        if T is None:
                            continue
                        T_usd = convert_cv_camera_pose_to_usd(T, coord_convention)
                        cam_translate.Set(
                            Gf.Vec3d(float(T_usd[0, 3]), float(T_usd[1, 3]), float(T_usd[2, 3])),
                            frame_num,
                        )
                        _set_quat(cam_orient, T_usd[:3, :3], frame_num)

        # Load object attrs + labels for metadata
        attr_artifact = self.context.artifacts.get("object.attr")
        object_attrs = {}
        if attr_artifact:
            try:
                object_attrs = self._json_load(attr_artifact.outputs["object_attrs"])
            except Exception:
                pass
        obj_labels = {}
        try:
            for obj in self._all_objects():
                obj_labels[obj.object_id] = {"label": obj.label, "segment_kind": obj.segment_kind}
        except Exception:
            pass

        # Objects with keyframe animation
        for obj_id, entry in sam3d_meshes.items():
            traj_data = corrected_trajectories.get(obj_id, {})
            if isinstance(traj_data, dict):
                frames = traj_data.get("frames", [])
                anchor_R_list = traj_data.get("anchor_R")
                mesh_basis_list = traj_data.get("mesh_basis")
            else:
                frames = traj_data
                anchor_R_list = None
                mesh_basis_list = None
            frames = [
                rec for rec in frames
                if _valid_vec3(rec.get("centroid_world")) and _valid_vec3(rec.get("scale"))
            ]
            if not frames:
                continue
            glb_path = self._find_glb(entry)
            if not glb_path:
                continue
            obj_mesh = self._load_trimesh(glb_path)
            if obj_mesh is None:
                continue

            token = obj_id.replace("-", "_")
            obj_path = f"/World/Objects/{token}"
            xform = UsdGeom.Xform.Define(stage, obj_path)
            prim = xform.GetPrim()

            prim.CreateAttribute("pit:object_id", Sdf.ValueTypeNames.String).Set(obj_id)
            info = obj_labels.get(obj_id, {})
            attrs = object_attrs.get(obj_id, {})
            label = attrs.get("class_name") or info.get("label", "")
            if label:
                prim.CreateAttribute("pit:class_name", Sdf.ValueTypeNames.String).Set(label)
            prim.CreateAttribute("pit:segment_kind", Sdf.ValueTypeNames.String).Set(
                info.get("segment_kind", entry.get("segment_kind", "object")))
            if attrs.get("is_movable") is not None:
                prim.CreateAttribute("pit:is_movable", Sdf.ValueTypeNames.Bool).Set(bool(attrs["is_movable"]))
            if attrs.get("measured_mass_kg") is not None:
                prim.CreateAttribute("pit:mass_kg", Sdf.ValueTypeNames.Float).Set(float(attrs["measured_mass_kg"]))

            xf = UsdGeom.Xformable(prim)
            tr_op = xf.AddTranslateOp()
            imageable = UsdGeom.Imageable(prim)

            # Mesh vertices: SAM3D uses OpenGL convention (Y=up, -Z=forward).
            # USD also uses Y=up. But we flipped Z for world positions (WildGS Z=forward
            # → USD Z=-forward), so mesh Z must also be flipped to match the scene.
            verts = self._prepare_usd_object_mesh_vertices(obj_mesh.vertices)

            visual = UsdGeom.Xform.Define(stage, f"{obj_path}/Visual")
            vis_xf = UsdGeom.Xformable(visual.GetPrim())
            vis_orient = vis_xf.AddOrientOp()
            vis_scale = vis_xf.AddScaleOp()
            self._apply_trajectory_visibility_samples(
                imageable,
                [int(float(rec.get("frame_id", 0))) for rec in frames],
                stage_start_frame=1,
                stage_end_frame=int(max_frame),
            )
            for rec in frames:
                frame_num = float(rec.get("frame_id", 0))
                cx, cy, cz = rec["centroid_world"]
                scale = np.asarray(rec.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)

                rot_world, trans_world = _fixed_camera_grounded_pose(
                    int(frame_num),
                    rec.get("rotation_matrix"),
                    [cx, cy, cz],
                    verts,
                    scale,
                )
                rot_usd, trans_usd = convert_world_pose_to_usd(
                    rot_world,
                    trans_world,
                    coord_convention,
                )
                tr_op.Set(
                    Gf.Vec3d(float(trans_usd[0]), float(trans_usd[1]), float(trans_usd[2])),
                    frame_num,
                )
                vis_scale.Set(
                    Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])),
                    frame_num,
                )
                _set_quat(vis_orient, rot_usd, frame_num)

            mesh_path = f"{obj_path}/Visual/Mesh"
            usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            usd_mesh.CreatePointsAttr(_usd_points(verts))
            usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(obj_mesh.faces)))
            usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(obj_mesh.faces.flatten().tolist()))
            _set_mesh_orientation(usd_mesh)

            vc = getattr(obj_mesh.visual, "vertex_colors", None)
            if vc is not None and len(vc) == len(verts):
                _set_display_colors(usd_mesh, vc)
            normals = getattr(obj_mesh, "vertex_normals", None)
            if normals is not None and len(normals) == len(verts):
                _set_normals(usd_mesh, normals)

        stage.GetRootLayer().Save()
        _logger.info("[scene.export] USDC exported: %s", usdc_path)
        return coord_report

    def _run_object_attr(self) -> dict:
        out_dir = self.context.stage_output_dir("object.attr")

        # Collect ALL unique objects across ALL detection frames (not just the
        # last one) so every object gets physics attributes.
        detect = self.context.artifacts.get("object.detect")
        if detect is None:
            raise RuntimeError("object.detect outputs are required")
        summary = self._json_load(detect.outputs["summary"])

        # For each object, keep the detection frame where it has highest score * bbox area
        best_per_obj: dict[str, tuple[float, "FrameDetections", "DetectedInstance"]] = {}
        for entry in summary.get("frames", []):
            det_path = entry.get("detections")
            if not det_path or not Path(det_path).exists():
                continue
            try:
                detections = FrameDetections.model_validate(self._json_load(det_path))
            except Exception:
                continue
            for inst in detections.instances:
                b = inst.bbox
                area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                score = inst.score * area
                prev = best_per_obj.get(inst.object_id)
                if prev is None or score > prev[0]:
                    best_per_obj[inst.object_id] = (score, detections, inst)

        # Infer attrs per object using its best frame
        all_attrs: dict = {}
        for obj_id, (_, det, inst) in best_per_obj.items():
            obj = ObjectNode(
                object_id=obj_id,
                label=inst.concept_label,
                segment_kind=inst.segment_kind,
            )
            try:
                result = self._infer_object_physics_priors(det, [obj])
                all_attrs.update(result)
            except Exception as exc:
                _logger.warning("[object.attr] Failed for %s: %s", obj_id, exc)

        attrs_path = out_dir / "object_attrs.json"
        self._json_dump(attrs_path, all_attrs)
        outputs = {"object_attrs": str(attrs_path)}
        summary = {"attr_count": len(all_attrs), "object_count": len(best_per_obj)}
        return self._base_result("object.attr", summary, outputs)

    def _run_physics_dynamics(self) -> dict:
        from guanwu.video.features.world_inference.physics_dynamics import PhysicsDynamicsEstimator
        geometry = self.context.artifacts.get("geometry.lift")
        if geometry is None:
            raise RuntimeError("geometry.lift outputs are required")
        attr_artifact = self.context.artifacts.get("object.attr")
        if attr_artifact is None:
            raise RuntimeError("object.attr outputs are required")
        out_dir = self.context.stage_output_dir("physics.dynamics")
        geometry_summary = self._json_load(geometry.outputs["summary"])
        object_attrs: dict = self._json_load(attr_artifact.outputs["object_attrs"])

        object_trajectories: dict[str, list[dict]] = {}
        trajectory_path = geometry.outputs.get("object_trajectories")
        pose_artifact = self.context.artifacts.get("pose.optimize")
        refined_path = pose_artifact.outputs.get("refined_object_trajectories") if pose_artifact else None
        refined_traj = self._json_load(refined_path) if refined_path and Path(refined_path).exists() else {}
        if trajectory_path:
            raw_traj = self._json_load(trajectory_path)
            if isinstance(raw_traj, dict):
                raw_traj = self._merge_refined_object_trajectories(raw_traj, refined_traj)
                for oid, track in raw_traj.items():
                    if not isinstance(track, list):
                        continue
                    for rec in track:
                        if not isinstance(rec, dict):
                            continue
                        object_trajectories.setdefault(oid, []).append({
                            "frame_idx": rec.get("frame_id"),
                            "timestamp": rec.get("timestamp_sec"),
                            "centroid_3d": rec.get("centroid_world"),
                        })

        estimator = PhysicsDynamicsEstimator()
        dynamics = estimator.estimate(object_trajectories, object_attrs)
        dynamics_path = out_dir / "physics_dynamics.json"
        self._json_dump(dynamics_path, dynamics)
        outputs = {"physics_dynamics": str(dynamics_path)}
        summary = {"object_count": len(dynamics)}
        return self._base_result("physics.dynamics", summary, outputs)

    def _run_relation_infer(self) -> dict:
        out_dir = self.context.stage_output_dir("relation.infer")
        services = self._services()
        objects = self._all_objects()
        geometry = self.context.artifacts.get("geometry.lift")
        geometry_summary = self._json_load(geometry.outputs["summary"]) if geometry else {}
        latest_timestamp = 0.0
        if geometry_summary.get("frames"):
            latest_timestamp = float(geometry_summary["frames"][-1]["timestamp"])

        # Pass WildGS static map as background geometry for floor/wall relation inference
        background_geometry: dict | None = None
        if geometry and geometry.outputs.get("wildgs_static_map"):
            static_map_dir = geometry.outputs["wildgs_static_map"]
            import os
            ply_path = os.path.join(static_map_dir, "final_gs.ply")
            if not os.path.exists(ply_path):
                # Some versions name it static_gaussians.ply
                ply_path = os.path.join(static_map_dir, "static_gaussians.ply")
            if os.path.exists(ply_path):
                background_geometry = {"ply_path": ply_path}

        relations = services.relation_engine.infer(
            objects,
            frame_idx=len(geometry_summary.get("frames", [])),
            timestamp=latest_timestamp,
            background_geometry=background_geometry,
        )
        relations_path = out_dir / "relations.json"
        self._json_dump(relations_path, [rel.model_dump(mode="json") for rel in relations])
        outputs = {"relations": str(relations_path)}
        summary = {"relation_count": len(relations), "object_count": len(objects)}
        return self._base_result("relation.infer", summary, outputs)

    def _run_event_infer(self) -> dict:
        geometry = self.context.artifacts.get("geometry.lift")
        if geometry is None:
            raise RuntimeError("geometry.lift outputs are required")
        out_dir = self.context.stage_output_dir("event.infer")
        services = self._services()
        geometry_summary = self._json_load(geometry.outputs["summary"])

        # Resolve background geometry once for all frames
        background_geometry: dict | None = None
        if geometry.outputs.get("wildgs_static_map"):
            import os
            static_map_dir = geometry.outputs["wildgs_static_map"]
            ply_path = os.path.join(static_map_dir, "final_gs.ply")
            if not os.path.exists(ply_path):
                ply_path = os.path.join(static_map_dir, "static_gaussians.ply")
            if os.path.exists(ply_path):
                background_geometry = {"ply_path": ply_path}

        engine = services.event_engine.__class__()
        previous_ids: set[str] = set()
        events: list[Event] = []
        frames_out: list[dict] = []
        for entry in geometry_summary["frames"]:
            objects = [ObjectNode.model_validate(obj) for obj in self._json_load(entry["observed_objects"])]
            relations = services.relation_engine.infer(
                objects,
                frame_idx=int(entry["frame_idx"]),
                timestamp=float(entry["timestamp"]),
                background_geometry=background_geometry,
            )
            current_ids = {obj.object_id for obj in objects}
            removed_ids = sorted(previous_ids - current_ids)
            frame_events = engine.infer(objects, relations, float(entry["timestamp"]), removed_object_ids=removed_ids)
            previous_ids = current_ids
            events.extend(frame_events)
            frames_out.append(
                {
                    "frame_idx": entry["frame_idx"],
                    "timestamp": entry["timestamp"],
                    "event_count": len(frame_events),
                    "events": [evt.model_dump(mode="json") for evt in frame_events],
                }
            )
        events_path = out_dir / "events.json"
        frames_path = out_dir / "event_frames.json"
        self._json_dump(events_path, [evt.model_dump(mode="json") for evt in events])
        self._json_dump(frames_path, frames_out)
        outputs = {"events": str(events_path), "event_frames": str(frames_path)}
        summary = {"event_count": len(events), "frame_count": len(frames_out)}
        return self._base_result("event.infer", summary, outputs)

    def _run_world_compose(self) -> dict:
        geometry = self.context.artifacts.get("geometry.lift")
        mesh = self.context.artifacts.get("mesh.reconstruct")
        attr_artifact = self.context.artifacts.get("object.attr")
        dynamics = self.context.artifacts.get("physics.dynamics")
        relation = self.context.artifacts.get("relation.infer")
        event = self.context.artifacts.get("event.infer")
        if any(item is None for item in (geometry, mesh, attr_artifact, dynamics, relation, event)):
            raise RuntimeError("world.compose requires geometry, mesh, object.attr, physics.dynamics, relation, and event outputs")
        out_dir = self.context.stage_output_dir("world.compose")
        geometry_summary = self._json_load(geometry.outputs["summary"])
        objects = [ObjectNode.model_validate(obj) for obj in geometry_summary["latest_objects"]]
        relations = [RelationEdge.model_validate(rel) for rel in self._json_load(relation.outputs["relations"])]
        events = [Event.model_validate(evt) for evt in self._json_load(event.outputs["events"])]
        pit_snapshot = dict(geometry_summary["latest_pit_snapshot"])
        pit_snapshot["sam3d_meshes"] = self._json_load(mesh.outputs["sam3d_meshes"])
        pit_snapshot["object_attrs"] = self._json_load(attr_artifact.outputs["object_attrs"])
        pit_snapshot["vlm_priors"] = pit_snapshot["object_attrs"]
        pit_snapshot["physics_dynamics"] = self._json_load(dynamics.outputs["physics_dynamics"])
        pit_snapshot["alignment_frames"] = list(geometry_summary.get("frames", []))
        pose_artifact = self.context.artifacts.get("pose.optimize")
        refined_path = pose_artifact.outputs.get("refined_object_trajectories") if pose_artifact else None
        if refined_path and Path(refined_path).exists():
            refined_traj = self._json_load(refined_path)
            raw_traj_path = geometry.outputs.get("object_trajectories")
            raw_traj = self._json_load(raw_traj_path) if raw_traj_path else {}
            pit_snapshot["object_trajectories"] = self._merge_refined_object_trajectories(raw_traj, refined_traj)
            pit_snapshot["trajectory_source"] = "pose_optimize_refined"

        if geometry.outputs.get("wildgs_static_map"):
            import os
            static_map_dir = geometry.outputs["wildgs_static_map"]
            ply_path = os.path.join(static_map_dir, "final_gs.ply")
            if not os.path.exists(ply_path):
                ply_path = os.path.join(static_map_dir, "static_gaussians.ply")
            if os.path.exists(ply_path):
                pit_snapshot["background_reconstruction"] = {
                    "points_path": ply_path,
                    "source": "wildgs",
                }
        if geometry.outputs.get("wildgs_depth_maps"):
            pit_snapshot["wildgs_depth_maps_dir"] = geometry.outputs["wildgs_depth_maps"]
        world_payload = {
            "timestamp": geometry_summary["frames"][-1]["timestamp"] if geometry_summary["frames"] else 0.0,
            "active_objects": [obj.model_dump(mode="json") for obj in objects],
            "relations": [rel.model_dump(mode="json") for rel in relations],
            "events": [evt.model_dump(mode="json") for evt in events],
        }
        world_path = out_dir / "world_state.raw.json"
        pit_path = out_dir / "pit_snapshot.raw.json"
        self._json_dump(world_path, world_payload)
        self._json_dump(pit_path, pit_snapshot)
        outputs = {"world_state_raw": str(world_path), "pit_snapshot_raw": str(pit_path)}
        summary = {
            "object_count": len(objects),
            "relation_count": len(relations),
            "event_count": len(events),
        }
        return self._base_result("world.compose", summary, outputs)

    def _run_world_align(self) -> dict:
        compose = self.context.artifacts.get("world.compose")
        if compose is None:
            raise RuntimeError("world.compose outputs are required")
        out_dir = self.context.stage_output_dir("world.align")
        services = self._services()
        latest_world = self._json_load(compose.outputs["world_state_raw"])
        pit_snapshot = self._json_load(compose.outputs["pit_snapshot_raw"])
        objects = [ObjectNode.model_validate(obj) for obj in latest_world.get("active_objects", [])]
        refined_objects, refined_snapshot = services.object_scene_alignment.refine(objects, pit_snapshot)
        relations = [RelationEdge.model_validate(rel) for rel in latest_world.get("relations", [])]
        events = [Event.model_validate(evt) for evt in latest_world.get("events", [])]
        store = WorldStore(world_id=self.context.config.settings.storage.world_id)
        source_time = float(latest_world.get("timestamp", 0.0))
        store.upsert_objects(refined_objects, source_time)
        store.replace_relations(relations, source_time)
        store.append_events(events)
        world_state = store.build_world_state(source_time, source_time, source_time)
        aligned_path = out_dir / "world_state.aligned.json"
        snapshot_path = out_dir / "pit_snapshot.aligned.json"
        self._json_dump(aligned_path, world_state.model_dump(mode="json"))
        self._json_dump(snapshot_path, refined_snapshot)
        self.context.paths.latest_world_state.write_text(
            json.dumps(world_state.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        self._write_world_db(world_state)
        outputs = {
            "world_state_aligned": str(aligned_path),
            "pit_snapshot_aligned": str(snapshot_path),
            "latest_world_state": str(self.context.paths.latest_world_state),
            "world_db": str(self.context.paths.world_db),
        }
        summary = {
            "object_count": len(world_state.objects),
            "relation_count": len(world_state.relations),
            "event_count": len(world_state.events_recent),
        }
        return self._base_result("world.align", summary, outputs)

    def _run_scene_export(self) -> dict:
        out_dir = self.context.stage_output_dir("scene.export")
        compose = self.context.artifacts.get("scene.compose")
        if compose is not None:
            geometry = self.context.artifacts.get("geometry.lift")
            mesh_art = self.context.artifacts.get("mesh.reconstruct")
            sam3d_meshes = self._json_load(mesh_art.outputs["sam3d_meshes"])
            corrected_traj = self._json_load(compose.outputs["corrected_trajectories"])
            bg_mesh_path = self._find_bg_mesh(geometry.outputs.get("wildgs_background_mesh")) if geometry else None
            pose_opt_artifact = self.context.artifacts.get("pose.optimize")
            pose_road_geometry_path = None if pose_opt_artifact is None else pose_opt_artifact.outputs.get("road_geometry")
            bg_meshes = (
                load_background_asset_meshes(
                    geometry.outputs.get("background_assets_manifest"),
                    road_geometry_path=pose_road_geometry_path,
                    camera_trajectory_path=geometry.outputs.get("camera_trajectory"),
                )
                if geometry
                else []
            )
            if geometry and not bg_meshes:
                bg_meshes = self._find_bg_meshes(geometry.outputs.get("wildgs_background_mesh"))
            cam_traj = self._json_load(geometry.outputs["camera_trajectory"]) if geometry else []
            wildgs_poses, _ = self._load_wildgs_poses(geometry) if geometry else ([], None)

            fixed_camera_reference_frame_id = None
            fixed_camera_road_plane = None
            if pose_road_geometry_path and Path(pose_road_geometry_path).exists():
                try:
                    road_geometry = self._json_load(pose_road_geometry_path)
                except Exception:
                    road_geometry = {}
                if str(road_geometry.get("default_plane_policy", "")).strip().lower() == "global_for_fixed_camera":
                    fixed_camera_reference_frame_id = self._background_assets_target_frame_id(geometry)
                    if fixed_camera_reference_frame_id is None:
                        fixed_camera_reference_frame_id = self._pose_target_frame_id() or 1
                    fixed_camera_road_plane = select_road_plane_for_frame(road_geometry, int(fixed_camera_reference_frame_id), policy="global_for_fixed_camera")

            usdc_path = out_dir / "scene.usdc"
            conversion_report_path = out_dir / "conversion_report.json"
            coord_report = self._export_usdc(
                usdc_path,
                sam3d_meshes,
                corrected_traj,
                bg_mesh_path,
                wildgs_poses,
                cam_traj,
                bg_meshes=bg_meshes,
                conversion_report_path=conversion_report_path,
                fixed_camera_reference_frame_id=fixed_camera_reference_frame_id,
                fixed_camera_road_plane=fixed_camera_road_plane,
            )

            outputs = {"usdc": str(usdc_path), "conversion_report": str(conversion_report_path)}
            summary = {
                "object_count": len(sam3d_meshes),
                "format": "usdc",
                "stage_up_axis": coord_report.get("stage_up_axis", "Z"),
            }
            return self._base_result("scene.export", summary, outputs)

        align = self.context.artifacts.get("world.align")
        if align is None:
            raise RuntimeError("world.align outputs are required")
        services = self._services()
        world_state = WorldState.model_validate(self._json_load(align.outputs["world_state_aligned"]))
        pit_snapshot = self._json_load(align.outputs["pit_snapshot_aligned"])
        export = services.simulation_pipeline.export(
            objects=world_state.objects,
            relations=world_state.relations,
            pit_snapshot=pit_snapshot,
        )
        export_path = out_dir / "export.json"
        self._json_dump(export_path, export)
        outputs = {"export": str(export_path), **{k: str(v) for k, v in export.items() if isinstance(v, str)}}
        summary = {"mode": export.get("mode"), "object_count": export.get("object_count", 0)}
        return self._base_result("scene.export", summary, outputs)

    def _run_report_render(self) -> dict:
        align = self.context.artifacts.get("world.align")
        if align is None:
            raise RuntimeError("world.align outputs are required")
        out_dir = self.context.stage_output_dir("report.render")
        world_state = WorldState.model_validate(self._json_load(align.outputs["world_state_aligned"]))
        html_path = out_dir / "index.html"
        summary_path = out_dir / "summary.json"
        rows = "\n".join(
            f"<tr><td>{obj.object_id}</td><td>{obj.label}</td><td>{obj.state.visibility}</td><td>{obj.geometry.pose_3d.position}</td></tr>"
            for obj in world_state.objects
        )
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{self.context.config.project.name}</title>
<style>body{{font-family:ui-monospace,Menlo,monospace;padding:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}</style>
</head><body>
<h1>{self.context.config.project.name}</h1>
<p>Objects: {len(world_state.objects)} | Relations: {len(world_state.relations)} | Events: {len(world_state.events_recent)}</p>
<table><thead><tr><th>ID</th><th>Label</th><th>Visibility</th><th>Position</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
        html_path.write_text(html, encoding="utf-8")
        self._json_dump(
            summary_path,
            {
                "object_count": len(world_state.objects),
                "relation_count": len(world_state.relations),
                "event_count": len(world_state.events_recent),
                "html_path": str(html_path),
            },
        )
        summary = {"html_path": str(html_path), "object_count": len(world_state.objects)}
        outputs = {"index_html": str(html_path), "summary": str(summary_path)}
        return self._base_result("report.render", summary, outputs)

    def _run_materialize(self) -> dict:
        scene_export = self.context.artifacts.get("scene.export")
        inspect_artifact = self.context.artifacts.get("video.inspect")
        frame_sample = self.context.artifacts.get("frame.sample")
        object_index = self.context.artifacts.get("object.index")
        object_attr = self.context.artifacts.get("object.attr")
        geometry = self.context.artifacts.get("geometry.lift")
        mesh = self.context.artifacts.get("mesh.reconstruct")
        if any(item is None for item in (scene_export, inspect_artifact, frame_sample, object_index, object_attr, geometry, mesh)):
            raise RuntimeError(
                "materialize requires scene.export, video.inspect, frame.sample, "
                "object.index, object.attr, geometry.lift, and mesh.reconstruct outputs"
            )

        out_dir = self.context.stage_output_dir("materialize")
        workspace = self._workspace_config()
        geometry_summary = self._json_load(geometry.outputs["summary"])
        pose_artifact = self.context.artifacts.get("pose.optimize")
        refined_path = pose_artifact.outputs.get("refined_object_trajectories") if pose_artifact else None
        if refined_path and Path(refined_path).exists():
            refined_traj = self._json_load(refined_path)
            raw_traj_path = geometry.outputs.get("object_trajectories")
            raw_traj = self._json_load(raw_traj_path) if raw_traj_path else {}
            geometry_summary = dict(geometry_summary)
            geometry_summary["object_trajectories"] = self._merge_refined_object_trajectories(raw_traj, refined_traj)
            geometry_summary["object_trajectories_source"] = "pose_optimize_refined"
        report = materialize_video_project(
            project_root=self.context.paths.root,
            canonical_root=workspace.storage.canonical_root,
            dataset_id=self._dataset_id(),
            scene_export_path=scene_export.outputs["usdc"],
            video_metadata=self._json_load(inspect_artifact.outputs["video_metadata"]),
            frame_index=self._json_load(frame_sample.outputs["frame_index"]),
            object_index=self._json_load(object_index.outputs["objects"]),
            object_attrs=self._json_load(object_attr.outputs["object_attrs"]),
            geometry_summary=geometry_summary,
            mesh_manifest=self._json_load(mesh.outputs["sam3d_meshes"]),
        )
        report_path = out_dir / "materialize_report.json"
        self._json_dump(report_path, report.to_dict())
        outputs = {"materialize_report": str(report_path)}
        return self._base_result("materialize", report.to_dict(), outputs)

    def _run_catalog(self) -> dict:
        out_dir = self.context.stage_output_dir("catalog")
        workspace = self._workspace_config()
        catalog = Catalog(workspace.storage.catalog_path)
        catalog.build_from_canonical(workspace.storage.canonical_root)
        stats = catalog.get_stats()
        catalog.close()
        stats_path = out_dir / "catalog_stats.json"
        self._json_dump(stats_path, stats)
        return self._base_result("catalog", stats, {"catalog_stats": str(stats_path)})

    def validate(self) -> dict:
        statuses = self.context.load_stage_statuses()
        errors: list[str] = []
        for stage in (
            "video.inspect",
            "frame.sample",
            "object.detect",
            "object.index",
            "object.attr",
            "geometry.lift",
            "mesh.reconstruct",
            "physics.dynamics",
            "relation.infer",
            "event.infer",
            "world.compose",
            "world.align",
        ):
            status = statuses.get(stage)
            if status is None or status.status != "completed":
                errors.append(f"step {stage} is not completed")
            artifact = self.context.artifacts.get(stage)
            if artifact is None:
                errors.append(f"artifact {stage} is missing")
                continue
            for label, path in artifact.outputs.items():
                if path and not Path(path).exists():
                    errors.append(f"{stage}:{label} missing at {path}")
        for optional in ("scene.export", "report.render", "materialize", "catalog"):
            artifact = self.context.artifacts.get(optional)
            if artifact is None:
                continue
            for label, path in artifact.outputs.items():
                if not self._looks_like_path(path):
                    continue
                if path and not Path(path).exists():
                    errors.append(f"{optional}:{label} missing at {path}")
        if not self.context.paths.latest_world_state.exists():
            errors.append("latest_world_state.json is missing")
        return {"status": "ok" if not errors else "failed", "errors": errors}

    def _write_world_db(self, world_state: WorldState) -> None:
        conn = sqlite3.connect(self.context.paths.world_db)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS world_state (kind TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("DELETE FROM world_state")
            conn.execute(
                "INSERT INTO world_state(kind, payload) VALUES(?, ?)",
                ("latest", json.dumps(world_state.model_dump(mode="json"))),
            )
            conn.commit()
        finally:
            conn.close()

    def _looks_like_path(self, value: str) -> bool:
        if not value:
            return False
        text = str(value)
        return "/" in text or text.endswith((".json", ".jsonl", ".html", ".usd", ".usdc", ".usdz"))

    def _materialize_backend_meshes(self, meshes: dict[str, dict], frame_idx: int) -> dict[str, dict]:
        mode = (self.context.config.settings.runtime.asset_materialization or "copy").strip().lower()
        frame_root = self.context.root / "intermediate" / f"frame_{int(frame_idx):06d}" / "objects"
        out: dict[str, dict] = {}
        for object_id, entry in meshes.items():
            if not isinstance(entry, dict):
                out[object_id] = entry
                continue
            src = str(entry.get("mesh_path", "")).strip()
            if not src:
                out[object_id] = entry
                continue
            src_path = Path(src).expanduser()
            if not src_path.exists():
                out[object_id] = entry
                continue
            object_root = frame_root / _safe_name(object_id) / "assets"
            object_root.mkdir(parents=True, exist_ok=True)
            ext = src_path.suffix or ".bin"
            dst = _unique_path(object_root / f"object{ext}")
            _materialize_file(src_path, dst, mode)
            updated = dict(entry)
            updated["mesh_path"] = str(dst)
            updated.setdefault("files", []).append({"format": ext.lstrip("."), "path": str(dst)})
            out[object_id] = updated
        return out


def np_from_b64(image_b64: str):  # type: ignore[no-untyped-def]
    import base64
    import numpy as np

    payload = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    return np.frombuffer(base64.b64decode(payload), dtype=np.uint8)


def _materialize_file(src: Path, dst: Path, mode: str) -> None:
    mode = (mode or "copy").strip().lower()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return safe or "unknown"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
    def _canonical_stage(self, stage: str) -> str:
        return LEGACY_STAGE_ALIASES.get(stage, stage)
