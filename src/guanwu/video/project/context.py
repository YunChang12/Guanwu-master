from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from guanwu.video.project.artifacts import ArtifactRegistry, LEGACY_STAGE_ALIASES, ProjectManifest, STAGE_ORDER, StageStatus, utc_now
from guanwu.video.project.config import ProjectConfig, load_project_config, save_project_config


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    input_dir: Path
    input_video: Path
    state_dir: Path
    outputs_dir: Path
    cache_dir: Path
    logs_dir: Path
    manifest: Path
    stage_status: Path
    artifacts: Path
    latest_world_state: Path
    world_db: Path
    lock_file: Path


class ProjectContext:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.paths = ProjectPaths(
            root=self.root,
            config=self.root / "project.toml",
            input_dir=self.root / "input",
            input_video=self.root / "input" / "video.mp4",
            state_dir=self.root / "state",
            outputs_dir=self.root / "outputs",
            cache_dir=self.root / "cache",
            logs_dir=self.root / "logs",
            manifest=self.root / "state" / "manifest.json",
            stage_status=self.root / "state" / "stage_status.json",
            artifacts=self.root / "state" / "artifacts.json",
            latest_world_state=self.root / "state" / "latest_world_state.json",
            world_db=self.root / "state" / "world.db",
            lock_file=self.root / ".project.lock",
        )
        self.config = load_project_config(self.paths.config)
        self.artifacts = ArtifactRegistry(self.paths.artifacts)

    @classmethod
    def create(cls, root: str | Path, config: ProjectConfig) -> "ProjectContext":
        root_path = Path(root).expanduser().resolve()
        paths = ProjectPaths(
            root=root_path,
            config=root_path / "project.toml",
            input_dir=root_path / "input",
            input_video=root_path / "input" / "video.mp4",
            state_dir=root_path / "state",
            outputs_dir=root_path / "outputs",
            cache_dir=root_path / "cache",
            logs_dir=root_path / "logs",
            manifest=root_path / "state" / "manifest.json",
            stage_status=root_path / "state" / "stage_status.json",
            artifacts=root_path / "state" / "artifacts.json",
            latest_world_state=root_path / "state" / "latest_world_state.json",
            world_db=root_path / "state" / "world.db",
            lock_file=root_path / ".project.lock",
        )
        for path in (
            paths.root,
            paths.input_dir,
            paths.state_dir,
            paths.outputs_dir,
            paths.cache_dir,
            paths.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        manifest = ProjectManifest(
            project_id=config.project.project_id,
            project_root=str(paths.root),
            input_video=config.project.input_video,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        save_project_config(config, paths.config)
        paths.manifest.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
        statuses = {stage: StageStatus(stage=stage).model_dump(mode="json") for stage in STAGE_ORDER}
        paths.stage_status.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
        paths.artifacts.write_text("{}", encoding="utf-8")
        return cls(paths.root)

    def load_manifest(self) -> ProjectManifest:
        return ProjectManifest.model_validate(json.loads(self.paths.manifest.read_text(encoding="utf-8")))

    def save_manifest(self, manifest: ProjectManifest) -> None:
        manifest.updated_at = utc_now()
        self.paths.manifest.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")

    def load_stage_statuses(self) -> dict[str, StageStatus]:
        raw = json.loads(self.paths.stage_status.read_text(encoding="utf-8"))
        normalized: dict[str, StageStatus] = {}
        for stage, value in raw.items():
            canonical = LEGACY_STAGE_ALIASES.get(stage, stage)
            payload = dict(value)
            payload["stage"] = canonical
            normalized[canonical] = StageStatus.model_validate(payload)
        return normalized

    def save_stage_statuses(self, statuses: dict[str, StageStatus]) -> None:
        payload = {stage: status.model_dump(mode="json") for stage, status in statuses.items()}
        self.paths.stage_status.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def stage_output_dir(self, stage: str) -> Path:
        stage = LEGACY_STAGE_ALIASES.get(stage, stage)
        index = STAGE_ORDER.index(stage) + 1
        safe_stage = stage.replace(".", "_")
        path = self.paths.outputs_dir / f"{index:02d}_{safe_stage}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def acquire_lock(self) -> None:
        try:
            fd = os.open(self.paths.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Reentrant: if the lock is held by this process, allow it
            try:
                owner_pid = int(self.paths.lock_file.read_text().strip())
            except (ValueError, OSError):
                owner_pid = -1
            if owner_pid == os.getpid():
                return  # same process, allow reentry
            raise RuntimeError(f"Project is already locked: {self.paths.lock_file}")
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)

    def release_lock(self) -> None:
        self.paths.lock_file.unlink(missing_ok=True)
