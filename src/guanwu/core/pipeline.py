"""Pipeline engine - orchestrates adapter stages."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from guanwu.adapters.base import get_adapter
from guanwu.core.bb_logging import new_job_id
from guanwu.core.config import WorkspaceConfig
from guanwu.core.errors import BlueBirdError
from guanwu.schemas.bundles import AdapterConfig, EmitReport, JobContext, ValidationReport
from guanwu.schemas.enums import PipelineStage
from guanwu.storage.catalog import Catalog

logger = logging.getLogger("guanwu")


class JobSummary:
    """Summary of a pipeline job."""

    def __init__(self, job_id: str, dataset_id: str):
        self.job_id = job_id
        self.dataset_id = dataset_id
        self.start_time = datetime.now(timezone.utc)
        self.end_time: datetime | None = None
        self.stages_run: list[str] = []
        self.success_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.errors: list[dict] = []
        self.emit_report: EmitReport | None = None
        self.validation_report: ValidationReport | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_sec": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "stages_run": self.stages_run,
            "success_count": self.success_count,
            "skip_count": self.skip_count,
            "fail_count": self.fail_count,
            "errors": self.errors,
        }


# Default stage order
STAGE_ORDER = [
    PipelineStage.INVENTORY,
    PipelineStage.FETCH,
    PipelineStage.PARSE,
    PipelineStage.NORMALIZE,
    PipelineStage.DERIVED,
    PipelineStage.VALIDATE,
    PipelineStage.CATALOG,
]


def _make_job_context(config: WorkspaceConfig, **overrides) -> JobContext:
    """Create a JobContext from workspace config."""
    remote = config.runtime.remote
    return JobContext(
        job_id=overrides.pop("job_id", new_job_id()),
        workspace_root=config.workspace_root,
        raw_root=config.storage.raw_root,
        staging_root=config.storage.staging_root,
        canonical_root=config.storage.canonical_root,
        dry_run=overrides.pop("dry_run", False),
        resume=overrides.pop("resume", config.runtime.resume),
        workers=overrides.pop("workers", config.runtime.workers),
        fail_fast=overrides.pop("fail_fast", config.runtime.fail_fast),
        limit=overrides.pop("limit", None),
        scene_id=overrides.pop("scene_id", None),
        asset_id=overrides.pop("asset_id", None),
        remote_host=remote.host,
        remote_conda_env=remote.conda_env,
        remote_work_dir=remote.work_dir,
        remote_python=remote.python,
        remote_conda_init=remote.conda_init,
    )


def _make_adapter_config(dataset_id: str, config: WorkspaceConfig) -> AdapterConfig:
    """Create an AdapterConfig from workspace config."""
    ds_cfg = config.datasets.get(dataset_id)
    if ds_cfg is None:
        return AdapterConfig(
            dataset_id=dataset_id,
            source_mode="local",
        )
    return AdapterConfig(
        dataset_id=dataset_id,
        source_mode=ds_cfg.source.mode,
        source_path=ds_cfg.source.path,
        source_uri=ds_cfg.source.uri,
        cache_dir=ds_cfg.source.cache_dir,
        options=ds_cfg.options,
        filters=ds_cfg.filters,
        splits=ds_cfg.splits,
    )


def run_pipeline(
    dataset_id: str,
    config: WorkspaceConfig,
    stages: list[str] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    scene_id: str | None = None,
    asset_id: str | None = None,
    resume: bool = False,
    workers: int | None = None,
    fail_fast: bool = False,
) -> JobSummary:
    """Run the processing pipeline for a dataset."""
    job_id = new_job_id()
    summary = JobSummary(job_id, dataset_id)

    ctx = _make_job_context(
        config,
        job_id=job_id,
        dry_run=dry_run,
        limit=limit,
        scene_id=scene_id,
        asset_id=asset_id,
        resume=resume,
        workers=workers or config.runtime.workers,
        fail_fast=fail_fast,
    )

    adapter_config = _make_adapter_config(dataset_id, config)

    # Resolve stages
    if stages:
        requested = [PipelineStage(s) for s in stages]
    else:
        requested = list(STAGE_ORDER)

    logger.info(f"[{job_id}] Starting pipeline for {dataset_id}, stages: {[s.value for s in requested]}")

    adapter = get_adapter(dataset_id)

    items = []
    raw_refs = []
    parse_bundle = None
    normalize_bundle = None

    try:
        # Inventory
        if PipelineStage.INVENTORY in requested:
            logger.info(f"[{job_id}] Stage: inventory")
            items = adapter.inventory(adapter_config, ctx)
            summary.stages_run.append("inventory")
            summary.success_count += len(items)
            logger.info(f"[{job_id}] Inventory found {len(items)} items")

            if dry_run:
                logger.info(f"[{job_id}] Dry-run: stopping after inventory")
                summary.end_time = datetime.now(timezone.utc)
                return summary

        # Fetch
        if PipelineStage.FETCH in requested:
            logger.info(f"[{job_id}] Stage: fetch")
            if not items:
                items = adapter.inventory(adapter_config, ctx)
            raw_refs = adapter.fetch(items, ctx)
            summary.stages_run.append("fetch")
            logger.info(f"[{job_id}] Fetched {len(raw_refs)} raw refs")

        # Parse
        if PipelineStage.PARSE in requested:
            logger.info(f"[{job_id}] Stage: parse")
            if not raw_refs:
                if not items:
                    items = adapter.inventory(adapter_config, ctx)
                raw_refs = adapter.fetch(items, ctx)
            parse_bundle = adapter.parse_raw(raw_refs, ctx)
            summary.stages_run.append("parse")
            logger.info(
                f"[{job_id}] Parsed: {len(parse_bundle.scenes)} scenes, "
                f"{len(parse_bundle.assets)} assets"
            )

        # Normalize
        if PipelineStage.NORMALIZE in requested:
            logger.info(f"[{job_id}] Stage: normalize")
            if parse_bundle is None:
                if not raw_refs:
                    if not items:
                        items = adapter.inventory(adapter_config, ctx)
                    raw_refs = adapter.fetch(items, ctx)
                parse_bundle = adapter.parse_raw(raw_refs, ctx)
            normalize_bundle = adapter.normalize(parse_bundle, ctx)
            summary.stages_run.append("normalize")
            logger.info(
                f"[{job_id}] Normalized: {len(normalize_bundle.scenes)} scenes, "
                f"{len(normalize_bundle.assets)} assets"
            )

        # Validate
        if PipelineStage.VALIDATE in requested:
            logger.info(f"[{job_id}] Stage: validate")
            if normalize_bundle is None:
                logger.warning(f"[{job_id}] No normalized data to validate")
            else:
                report = adapter.validate(normalize_bundle, ctx)
                summary.validation_report = report
                summary.stages_run.append("validate")
                if not report.passed:
                    logger.warning(
                        f"[{job_id}] Validation: {report.num_errors} errors, "
                        f"{report.num_warnings} warnings"
                    )
                else:
                    logger.info(f"[{job_id}] Validation passed")

        # Emit (covers both DERIVED and main emit)
        if PipelineStage.NORMALIZE in requested or PipelineStage.CATALOG in requested:
            if normalize_bundle is not None:
                logger.info(f"[{job_id}] Stage: emit")
                emit_report = adapter.emit(normalize_bundle, ctx)
                summary.emit_report = emit_report
                summary.stages_run.append("emit")
                logger.info(
                    f"[{job_id}] Emitted: {emit_report.scenes_emitted} scenes, "
                    f"{emit_report.assets_emitted} assets"
                )

        # Catalog
        if PipelineStage.CATALOG in requested:
            logger.info(f"[{job_id}] Stage: catalog")
            catalog = Catalog(config.storage.catalog_path)
            catalog.build_from_canonical(config.storage.canonical_root)
            catalog.close()
            summary.stages_run.append("catalog")
            logger.info(f"[{job_id}] Catalog updated")

    except BlueBirdError as e:
        summary.fail_count += 1
        summary.errors.append({"type": type(e).__name__, "message": str(e), "details": e.details})
        logger.error(f"[{job_id}] Pipeline error: {e}")
        if fail_fast:
            raise
    except Exception as e:
        summary.fail_count += 1
        summary.errors.append({"type": type(e).__name__, "message": str(e)})
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)
        if fail_fast:
            raise

    summary.end_time = datetime.now(timezone.utc)
    logger.info(f"[{job_id}] Pipeline complete: {summary.to_dict()}")
    return summary


def run_pipeline_all(
    config: WorkspaceConfig,
    group: str | None = None,
    **kwargs,
) -> list[JobSummary]:
    """Run pipeline for all enabled datasets, optionally filtered by group."""
    from guanwu.registry.manager import get_datasets_by_group

    if group:
        dataset_ids = get_datasets_by_group(group)
    else:
        dataset_ids = [
            did for did, dcfg in config.datasets.items() if dcfg.enabled
        ]

    summaries = []
    for dataset_id in dataset_ids:
        try:
            s = run_pipeline(dataset_id, config, **kwargs)
            summaries.append(s)
        except Exception as e:
            logger.error(f"Pipeline failed for {dataset_id}: {e}")
            if kwargs.get("fail_fast"):
                raise
    return summaries
