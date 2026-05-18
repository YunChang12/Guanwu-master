"""Export profile definitions and engine."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import orjson

from guanwu.storage.canonical_store import CanonicalStore

logger = logging.getLogger("guanwu")


class ExportProfile(str, Enum):
    MESH_PREVIEW = "mesh_preview"
    USD_FULL = "usd_full"
    ML_MINIMAL = "ml_minimal"
    RESEARCH_SAFE = "research_safe"


_PROFILE_DESCRIPTIONS = {
    ExportProfile.MESH_PREVIEW: "Lightweight GLB previews, thumbnails, mesh stats, metadata JSON",
    ExportProfile.USD_FULL: "Complete USD scenes/assets with parquet metadata and license/provenance",
    ExportProfile.ML_MINIMAL: "Parquet indices, frame URIs, poses, calibration, bboxes/tracks (no large meshes)",
    ExportProfile.RESEARCH_SAFE: "Only redistributable assets; restricted data exports metadata only",
}


def list_profiles() -> list[dict]:
    """List available export profiles."""
    return [
        {"name": p.value, "description": _PROFILE_DESCRIPTIONS[p]}
        for p in ExportProfile
    ]


def run_export(
    dataset_id: str,
    profile: str,
    canonical_root: str,
    export_root: str,
    catalog_path: str | None = None,
) -> dict:
    """Run an export for a dataset with the given profile."""
    profile_enum = ExportProfile(profile)
    store = CanonicalStore(canonical_root)
    export_dir = Path(export_root) / profile / dataset_id
    export_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_id": dataset_id,
        "profile": profile,
        "export_dir": str(export_dir),
        "files_written": [],
    }

    dataset_dir = store.dataset_dir(dataset_id)
    ds_json = dataset_dir / "dataset.json"
    if ds_json.exists():
        _copy_json(ds_json, export_dir / "dataset.json")
        report["files_written"].append("dataset.json")

    if profile_enum == ExportProfile.MESH_PREVIEW:
        _export_mesh_preview(store, dataset_id, export_dir, report)
    elif profile_enum == ExportProfile.USD_FULL:
        _export_usd_full(store, dataset_id, export_dir, report)
    elif profile_enum == ExportProfile.ML_MINIMAL:
        _export_ml_minimal(store, dataset_id, export_dir, report)
    elif profile_enum == ExportProfile.RESEARCH_SAFE:
        _export_research_safe(store, dataset_id, export_dir, report, catalog_path)

    logger.info(f"Export complete: {profile} for {dataset_id} -> {export_dir}")
    return report


def _copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)


def _export_mesh_preview(
    store: CanonicalStore, dataset_id: str, export_dir: Path, report: dict
) -> None:
    """Export lightweight previews: GLB, thumbnails, mesh stats, metadata."""
    # Scenes
    for scene_uid in store.list_scenes(dataset_id):
        scene_export = export_dir / "scenes" / scene_uid
        scene_export.mkdir(parents=True, exist_ok=True)

        scene_meta = store.scene_dir(dataset_id, scene_uid) / "scene_meta.json"
        if scene_meta.exists():
            _copy_json(scene_meta, scene_export / "scene_meta.json")
            report["files_written"].append(f"scenes/{scene_uid}/scene_meta.json")

        mesh_stats = store.scene_dir(dataset_id, scene_uid) / "mesh_stats.json"
        if mesh_stats.exists():
            _copy_json(mesh_stats, scene_export / "mesh_stats.json")
            report["files_written"].append(f"scenes/{scene_uid}/mesh_stats.json")

    # Assets
    for asset_uid in store.list_assets(dataset_id):
        asset_export = export_dir / "assets" / asset_uid
        asset_export.mkdir(parents=True, exist_ok=True)

        asset_meta = store.asset_dir(dataset_id, asset_uid) / "asset_meta.json"
        if asset_meta.exists():
            _copy_json(asset_meta, asset_export / "asset_meta.json")
            report["files_written"].append(f"assets/{asset_uid}/asset_meta.json")

        mesh_stats = store.asset_dir(dataset_id, asset_uid) / "mesh_stats.json"
        if mesh_stats.exists():
            _copy_json(mesh_stats, asset_export / "mesh_stats.json")
            report["files_written"].append(f"assets/{asset_uid}/mesh_stats.json")


def _export_usd_full(
    store: CanonicalStore, dataset_id: str, export_dir: Path, report: dict
) -> None:
    """Export full USD scenes/assets with all metadata."""
    import shutil

    # Copy entire canonical dataset directory
    src = store.dataset_dir(dataset_id)
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            dst = export_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            report["files_written"].append(str(rel))


def _export_ml_minimal(
    store: CanonicalStore, dataset_id: str, export_dir: Path, report: dict
) -> None:
    """Export parquet indices and metadata only - no large meshes."""
    import shutil

    src = store.dataset_dir(dataset_id)
    for item in src.rglob("*"):
        if item.is_file() and item.suffix in (".json", ".parquet"):
            rel = item.relative_to(src)
            dst = export_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            report["files_written"].append(str(rel))


def _export_research_safe(
    store: CanonicalStore,
    dataset_id: str,
    export_dir: Path,
    report: dict,
    catalog_path: str | None = None,
) -> None:
    """Export only redistributable data; restricted data gets metadata only."""
    # Check license info
    redistributable = True
    if catalog_path:
        try:
            from guanwu.storage.catalog import Catalog
            catalog = Catalog(catalog_path)
            catalog.initialize()
            result = catalog.query(
                f"SELECT redistribution_allowed FROM licenses WHERE record_id = '{dataset_id}'"
            )
            if result and result[0].get("redistribution_allowed") is False:
                redistributable = False
            catalog.close()
        except Exception:
            pass

    if redistributable:
        _export_usd_full(store, dataset_id, export_dir, report)
    else:
        _export_ml_minimal(store, dataset_id, export_dir, report)
        logger.info(f"Research-safe export: {dataset_id} is restricted, exported metadata only")
