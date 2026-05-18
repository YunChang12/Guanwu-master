from __future__ import annotations

from pathlib import Path
from typing import Any

from guanwu.core.config import WorkspaceConfig
from guanwu.video.clients.zaiwu import normalize_provider_mode
from guanwu.video.project.artifacts import STAGE_DEPENDENCIES as VIDEO_STAGE_DEPENDENCIES
from guanwu.video.project.artifacts import STAGE_ORDER as VIDEO_STAGE_ORDER
from guanwu.video.project.config import save_project_config
from guanwu.video.project.context import ProjectContext
from guanwu.video.project.executor import ProjectExecutor as MigratedVideoProjectExecutor


def _coerce_context(context: ProjectContext | str | Path | Any) -> ProjectContext:
    if isinstance(context, ProjectContext):
        return context
    if hasattr(context, "root"):
        return ProjectContext(getattr(context, "root"))
    return ProjectContext(context)


def ensure_video_project(
    *,
    project_root: str | Path,
    workspace: WorkspaceConfig | None = None,
    video_path: str | Path | None = None,
) -> ProjectContext:
    root = Path(project_root)
    if (root / "project.toml").exists():
        return ProjectContext(root)
    if workspace is None or video_path is None:
        raise ValueError("workspace and video_path are required to initialize a video project")
    return VideoProjectExecutor.init_project(
        video=str(video_path),
        out_dir=root,
        workspace=workspace,
    )


class VideoProjectExecutor:
    def __init__(self, context: ProjectContext | str | Path | Any) -> None:
        self.context = _coerce_context(context)
        self._executor = MigratedVideoProjectExecutor(self.context)

    @property
    def workspace(self) -> WorkspaceConfig:
        workspace = self.context.config.workspace or {}
        if workspace:
            return WorkspaceConfig.model_validate(workspace)
        return WorkspaceConfig(workspace_root=str(self.context.root))

    @classmethod
    def init_project(
        cls,
        *,
        video: str,
        out_dir: str | Path,
        workspace: WorkspaceConfig,
    ) -> ProjectContext:
        camera_provider = (workspace.video_pipeline.camera_provider or "wildgs").strip().lower()
        if camera_provider == "synthetic":
            camera_provider = "none"
        context = MigratedVideoProjectExecutor.init_project(
            video=str(Path(video).expanduser().resolve()),
            out_dir=str(Path(out_dir).expanduser().resolve()),
            provider_mode=normalize_provider_mode(workspace.video_pipeline.provider_mode),
            video_copy_mode="copy",
            workspace=workspace.model_dump(mode="json"),
            payload={
                "dataset_id": workspace.video_pipeline.default_dataset_id,
                "export_profile": workspace.video_pipeline.export_profile,
                "camera_provider": camera_provider,
            },
        )
        context.config.settings.pit.camera_provider = camera_provider
        if camera_provider == "wildgs":
            context.config.settings.pit.depth_provider = "wildgs"
        elif camera_provider == "none":
            context.config.settings.pit.depth_provider = "none"
        context.config.settings.zaiwu.enabled = normalize_provider_mode(workspace.video_pipeline.provider_mode) == "zaiwu"
        if workspace.video_pipeline.zaiwu_gateway_url:
            context.config.settings.zaiwu.gateway_url = workspace.video_pipeline.zaiwu_gateway_url
        context.config.settings.zaiwu.auto_start_workers = workspace.video_pipeline.zaiwu_auto_start_workers
        context.config.settings.zaiwu.object_detection_backend = workspace.video_pipeline.object_detection_backend
        context.config.settings.zaiwu.pose_optimizer_timeout_sec = workspace.video_pipeline.pose_optimizer_timeout_sec
        context.config.settings.zaiwu.pose_optimize_min_bbox_area_px = workspace.video_pipeline.pose_optimize_min_bbox_area_px
        save_project_config(context.config, context.paths.config)
        return ProjectContext(context.paths.root)

    def status(self) -> dict[str, Any]:
        return self._executor.status()

    def inspect(self) -> dict[str, Any]:
        return self._executor.inspect()

    def run_stage(self, stage: str, force: bool = False) -> dict[str, Any]:
        return self._executor.run_stage(stage, force=force)

    def run_range(self, from_stage: str, to_stage: str, force: bool = False) -> list[dict[str, Any]]:
        return self._executor.run_range(from_stage, to_stage, force=force)

    def run_phase(self, phase: str, force: bool = False) -> list[dict[str, Any]]:
        return self._executor.run_phase(phase, force=force)

    def validate(self) -> dict[str, Any]:
        return self._executor.validate()

    def invalidate_downstream(self, stage: str) -> None:
        self._executor.invalidate_downstream(stage)
