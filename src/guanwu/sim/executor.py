from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import orjson

from guanwu.adapters.base import get_adapter
from guanwu.core.config import WorkspaceConfig
from guanwu.exporters.profiles import run_export
from guanwu.projects import (
    ArtifactRecord,
    ProjectConfig,
    ProjectContext,
    StageStatus,
    create_project_config,
    stable_hash,
    utc_now,
)
from guanwu.schemas.bundles import AdapterConfig, EmitReport, JobContext, NormalizeBundle, ParseBundle, RawRef, SourceItem, ValidationReport
from guanwu.sim.materialize import materialize_bundle
from guanwu.storage.catalog import Catalog

SIM_STAGE_ORDER = [
    "inventory",
    "fetch",
    "parse",
    "normalize",
    "derived",
    "validate",
    "materialize",
    "catalog",
    "export",
]

SIM_STAGE_DEPENDENCIES = {
    "inventory": [],
    "fetch": ["inventory"],
    "parse": ["fetch"],
    "normalize": ["parse"],
    "derived": ["normalize"],
    "validate": ["normalize"],
    "materialize": ["normalize"],
    "catalog": ["materialize"],
    "export": ["materialize"],
}


def ensure_sim_project(
    *,
    dataset_id: str,
    workspace: WorkspaceConfig,
    project_root: str | Path | None = None,
) -> ProjectContext:
    root = Path(project_root or Path(workspace.storage.project_root) / "sim" / dataset_id / "default")
    if (root / "project.toml").exists():
        return ProjectContext(root)
    return SimProjectExecutor.init_project(
        dataset_id=dataset_id,
        out_dir=root,
        workspace=workspace,
    )


class SimProjectExecutor:
    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.workspace = WorkspaceConfig.model_validate(self.context.config.workspace)
        self.dataset_id = str(self.context.config.payload.get("dataset_id", ""))
        if not self.dataset_id:
            raise ValueError("sim project payload.dataset_id is required")
        self.adapter = get_adapter(self.dataset_id)

    @classmethod
    def init_project(
        cls,
        *,
        dataset_id: str,
        out_dir: str | Path,
        workspace: WorkspaceConfig,
    ) -> ProjectContext:
        out_path = Path(out_dir).expanduser().resolve()
        project_id = out_path.name
        config = create_project_config(
            project_id=project_id,
            kind="sim",
            name=f"sim:{dataset_id}",
            root_dir=str(out_path),
            workspace=workspace.model_dump(mode="json"),
            payload={
                "dataset_id": dataset_id,
                "default_export_profile": "mesh_preview",
            },
        )
        return ProjectContext.create(out_path, config, stages=SIM_STAGE_ORDER)

    def status(self) -> dict[str, Any]:
        statuses = self.context.load_stage_statuses()
        return {
            "project": self.context.config.model_dump(mode="json")["project"],
            "steps": {
                name: status.model_dump(mode="json")
                for name, status in statuses.items()
            },
        }

    def inspect(self) -> dict[str, Any]:
        manifest = self.context.load_manifest()
        return {
            "manifest": manifest.model_dump(mode="json"),
            "config": self.context.config.model_dump(mode="json"),
            "artifacts": {
                stage: record.model_dump(mode="json")
                for stage, record in self.context.artifacts.records.items()
            },
        }

    def run_stage(self, stage: str, force: bool = False) -> dict[str, Any]:
        if stage not in SIM_STAGE_ORDER:
            raise ValueError(f"Unsupported stage: {stage}")
        self.context.acquire_lock()
        try:
            self._ensure_dependencies(stage)
            if force:
                self.invalidate_downstream(stage)
                self._clear_stage_output(stage)
            statuses = self.context.load_stage_statuses()
            current = statuses.get(stage)
            if current and current.status == "completed" and not force:
                artifact = self.context.artifacts.get(stage)
                return {
                    "status": "cached",
                    "stage": stage,
                    "summary": artifact.summary if artifact else {},
                    "outputs": artifact.outputs if artifact else {},
                }
            runner = getattr(self, f"_run_{stage}")
            result = runner()
            statuses[stage] = StageStatus(
                stage=stage,
                status="completed",
                last_run_at=utc_now(),
                inputs_hash=result["inputs_hash"],
                params_hash=result["params_hash"],
            )
            self.context.save_stage_statuses(statuses)
            self.context.artifacts.set(ArtifactRecord(
                stage=stage,
                created_at=utc_now(),
                inputs_hash=result["inputs_hash"],
                params_hash=result["params_hash"],
                outputs=result["outputs"],
                summary=result["summary"],
            ))
            return {"status": "ok", "stage": stage, **result}
        except Exception as exc:
            statuses = self.context.load_stage_statuses()
            statuses[stage] = StageStatus(
                stage=stage,
                status="failed",
                last_run_at=utc_now(),
                error=str(exc),
            )
            self.context.save_stage_statuses(statuses)
            raise
        finally:
            self.context.release_lock()

    def run_range(self, from_stage: str, to_stage: str, force: bool = False) -> list[dict[str, Any]]:
        start = SIM_STAGE_ORDER.index(from_stage)
        end = SIM_STAGE_ORDER.index(to_stage)
        if start > end:
            raise ValueError("from_stage must come before to_stage")
        results = []
        for stage in SIM_STAGE_ORDER[start:end + 1]:
            results.append(self.run_stage(stage, force=force))
            force = False
        return results

    def invalidate_downstream(self, stage: str) -> None:
        statuses = self.context.load_stage_statuses()
        start = SIM_STAGE_ORDER.index(stage)
        downstream = SIM_STAGE_ORDER[start + 1:]
        for name in downstream:
            statuses[name] = StageStatus(stage=name)
        self.context.save_stage_statuses(statuses)
        self.context.artifacts.drop_many(downstream)
        for name in downstream:
            self._clear_stage_output(name)

    def _clear_stage_output(self, stage: str) -> None:
        path = self.context.stage_output_dir(stage)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def _ensure_dependencies(self, stage: str) -> None:
        for dependency in SIM_STAGE_DEPENDENCIES.get(stage, []):
            status = self.context.load_stage_statuses().get(dependency)
            if status is None or status.status != "completed":
                self.run_stage(dependency, force=False)

    def _adapter_config(self) -> AdapterConfig:
        ds_cfg = self.workspace.datasets.get(self.dataset_id)
        if ds_cfg is None:
            return AdapterConfig(dataset_id=self.dataset_id, source_mode="local")
        return AdapterConfig(
            dataset_id=self.dataset_id,
            source_mode=ds_cfg.source.mode,
            source_path=ds_cfg.source.path,
            source_uri=ds_cfg.source.uri,
            cache_dir=ds_cfg.source.cache_dir,
            options=ds_cfg.options,
            filters=ds_cfg.filters,
            splits=ds_cfg.splits,
        )

    def _job_context(self) -> JobContext:
        remote = self.workspace.runtime.remote
        return JobContext(
            job_id=f"{self.dataset_id}-{self.context.config.project.project_id}",
            workspace_root=self.workspace.workspace_root,
            raw_root=self.workspace.storage.raw_root,
            staging_root=self.workspace.storage.staging_root,
            canonical_root=self.workspace.storage.canonical_root,
            dry_run=False,
            resume=self.workspace.runtime.resume,
            workers=self.workspace.runtime.workers,
            fail_fast=self.workspace.runtime.fail_fast,
            remote_host=remote.host,
            remote_conda_env=remote.conda_env,
            remote_work_dir=remote.work_dir,
            remote_python=remote.python,
            remote_conda_init=remote.conda_init,
        )

    def _load_inventory(self) -> list[SourceItem]:
        artifact = self.context.artifacts.get("inventory")
        if artifact is None:
            raise RuntimeError("inventory outputs are required")
        path = artifact.outputs["inventory"]
        return [
            SourceItem.model_validate(item)
            for item in self._json_load(path)
        ]

    def _load_raw_refs(self) -> list[RawRef]:
        artifact = self.context.artifacts.get("fetch")
        if artifact is None:
            raise RuntimeError("fetch outputs are required")
        return [
            RawRef.model_validate(item)
            for item in self._json_load(artifact.outputs["raw_refs"])
        ]

    def _load_parse_bundle(self) -> ParseBundle:
        artifact = self.context.artifacts.get("parse")
        if artifact is None:
            raise RuntimeError("parse outputs are required")
        return ParseBundle.model_validate(self._json_load(artifact.outputs["parse_bundle"]))

    def _load_normalize_bundle(self) -> NormalizeBundle:
        artifact = self.context.artifacts.get("normalize")
        if artifact is None:
            raise RuntimeError("normalize outputs are required")
        return NormalizeBundle.model_validate(self._json_load(artifact.outputs["normalize_bundle"]))

    def _base_result(self, stage: str, summary: dict[str, Any], outputs: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        upstream_hashes = {
            dep: (self.context.artifacts.get(dep).inputs_hash if self.context.artifacts.get(dep) else "")
            for dep in SIM_STAGE_DEPENDENCIES.get(stage, [])
        }
        inputs_hash = stable_hash({
            "stage": stage,
            "project": self.context.config.project.project_id,
            "upstream": upstream_hashes,
        })
        params_hash = stable_hash(params or {})
        return {
            "summary": summary,
            "outputs": outputs,
            "inputs_hash": inputs_hash,
            "params_hash": params_hash,
        }

    def _json_load(self, path: str | Path) -> Any:
        return orjson.loads(Path(path).read_bytes())

    def _json_dump(self, path: str | Path, payload: Any) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        return str(out)

    def _run_inventory(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("inventory")
        items = self.adapter.inventory(self._adapter_config(), self._job_context())
        inventory_path = out_dir / "inventory.json"
        self._json_dump(inventory_path, [item.model_dump(mode="json") for item in items])
        return self._base_result(
            "inventory",
            {"item_count": len(items)},
            {"inventory": str(inventory_path)},
        )

    def _run_fetch(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("fetch")
        raw_refs = self.adapter.fetch(self._load_inventory(), self._job_context())
        refs_path = out_dir / "raw_refs.json"
        self._json_dump(refs_path, [ref.model_dump(mode="json") for ref in raw_refs])
        return self._base_result(
            "fetch",
            {"raw_ref_count": len(raw_refs)},
            {"raw_refs": str(refs_path)},
        )

    def _run_parse(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("parse")
        bundle = self.adapter.parse_raw(self._load_raw_refs(), self._job_context())
        bundle_path = out_dir / "parse_bundle.json"
        self._json_dump(bundle_path, bundle.model_dump(mode="json"))
        return self._base_result(
            "parse",
            {
                "scene_count": len(bundle.scenes),
                "asset_count": len(bundle.assets),
            },
            {"parse_bundle": str(bundle_path)},
        )

    def _run_normalize(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("normalize")
        bundle = self.adapter.normalize(self._load_parse_bundle(), self._job_context())
        bundle_path = out_dir / "normalize_bundle.json"
        self._json_dump(bundle_path, bundle.model_dump(mode="json"))
        return self._base_result(
            "normalize",
            {
                "scene_count": len(bundle.scenes),
                "asset_count": len(bundle.assets),
                "episode_count": len(bundle.episodes),
            },
            {"normalize_bundle": str(bundle_path)},
        )

    def _run_derived(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("derived")
        report_path = out_dir / "derived_report.json"
        payload = {
            "status": "ok",
            "note": "Derived products are currently emitted during materialize/export.",
        }
        self._json_dump(report_path, payload)
        return self._base_result(
            "derived",
            {"status": "ok"},
            {"derived_report": str(report_path)},
        )

    def _run_validate(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("validate")
        report: ValidationReport = self.adapter.validate(self._load_normalize_bundle(), self._job_context())
        report_path = out_dir / "validation_report.json"
        self._json_dump(report_path, report.model_dump(mode="json"))
        return self._base_result(
            "validate",
            {
                "passed": report.passed,
                "num_errors": report.num_errors,
                "num_warnings": report.num_warnings,
            },
            {"validation_report": str(report_path)},
        )

    def _run_materialize(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("materialize")
        bundle = self._load_normalize_bundle()
        prepare_emit = getattr(self.adapter, "prepare_emit", None)
        if callable(prepare_emit):
            prepare_emit(self._adapter_config(), self._load_parse_bundle(), bundle)
        report: EmitReport = materialize_bundle(self.adapter, bundle, self._job_context())
        report_path = out_dir / "materialize_report.json"
        self._json_dump(report_path, report.model_dump(mode="json"))
        return self._base_result(
            "materialize",
            {
                "scenes_emitted": report.scenes_emitted,
                "assets_emitted": report.assets_emitted,
                "episodes_emitted": report.episodes_emitted,
            },
            {"materialize_report": str(report_path)},
        )

    def _run_catalog(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("catalog")
        catalog = Catalog(self.workspace.storage.catalog_path)
        catalog.build_from_canonical(self.workspace.storage.canonical_root)
        stats = catalog.get_stats()
        catalog.close()
        stats_path = out_dir / "catalog_stats.json"
        self._json_dump(stats_path, stats)
        return self._base_result(
            "catalog",
            stats,
            {"catalog_stats": str(stats_path)},
        )

    def _run_export(self) -> dict[str, Any]:
        out_dir = self.context.stage_output_dir("export")
        profile = str(self.context.config.payload.get("default_export_profile", "mesh_preview"))
        report = run_export(
            dataset_id=self.dataset_id,
            profile=profile,
            canonical_root=self.workspace.storage.canonical_root,
            export_root=self.workspace.storage.export_root,
            catalog_path=self.workspace.storage.catalog_path,
        )
        report_path = out_dir / "export_report.json"
        self._json_dump(report_path, report)
        return self._base_result(
            "export",
            {
                "profile": profile,
                "files_written": len(report.get("files_written", [])),
            },
            {"export_report": str(report_path)},
            params={"profile": profile},
        )
