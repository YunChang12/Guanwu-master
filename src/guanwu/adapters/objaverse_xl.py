"""Objaverse-XL adapter for large-scale 3D object assets."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guanwu.adapters.base import DatasetAdapter, register_adapter
from guanwu.core.ids import make_asset_uid
from guanwu.schemas.bundles import (
    AdapterConfig,
    JobContext,
    NormalizeBundle,
    ParseBundle,
    RawRef,
    SourceItem,
)
from guanwu.schemas.enums import (
    AccessMode,
    GeometryLevel,
    RecordScope,
    SourceType,
)
from guanwu.schemas.records import (
    AssetRecord,
    DatasetRecord,
    LicenseRecord,
    ProvenanceRecord,
)

logger = logging.getLogger("guanwu")

_MESH_EXTENSIONS = {".glb", ".gltf", ".obj"}


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_glb_files(cache_dir: str) -> dict[str, str]:
    """Scan the hf-objaverse-v1 cache layout and return uid -> file path mapping."""
    objects: dict[str, str] = {}
    glbs_root = os.path.join(cache_dir, "hf-objaverse-v1", "glbs")
    if not os.path.isdir(glbs_root):
        # Fall back: check if cache_dir itself contains sharded dirs directly
        glbs_root = cache_dir

    if not os.path.isdir(glbs_root):
        return objects

    for shard in sorted(os.listdir(glbs_root)):
        shard_path = os.path.join(glbs_root, shard)
        if not os.path.isdir(shard_path):
            continue
        for fname in sorted(os.listdir(shard_path)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in _MESH_EXTENSIONS:
                objects[stem] = os.path.join(shard_path, fname)
    return objects


def _load_object_ids_file(path: str) -> list[str]:
    """Read a newline-delimited file of object UIDs."""
    ids: list[str] = []
    with open(path, "r") as f:
        for line in f:
            uid = line.strip()
            if uid and not uid.startswith("#"):
                ids.append(uid)
    return ids


@register_adapter
class ObjaverseXLAdapter(DatasetAdapter):
    """Adapter for the Objaverse-XL dataset.

    Supports two local modes:
      - cache_dir: pre-downloaded objects under hf-objaverse-v1/glbs/<shard>/<uid>.glb
      - object_ids_file: a text file listing UIDs to process (resolved against cache)

    Also supports the ``objaverse`` Python API when the package is installed.
    """

    name: str = "objaverse_xl"
    version: str = "0.1.0"

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool]:
        return {
            "scene_mesh": False,
            "object_mesh": True,
            "articulation": False,
            "deformable_mesh": False,
            "camera": False,
            "depth": False,
            "lidar": False,
            "tracks": False,
            "videos": False,
            "sdk_required": False,
            "license_gated": False,
            "supports_local_ingest": True,
        }

    # ------------------------------------------------------------------
    # inventory
    # ------------------------------------------------------------------

    def inventory(
        self, config: AdapterConfig, ctx: JobContext
    ) -> list[SourceItem]:
        max_objects = config.filters.get("max_objects")
        object_ids_file = config.options.get("object_ids_file")

        # Determine the cache directory
        cache_dir = config.cache_dir or config.source_path or ""

        # Collect candidate UIDs and their file paths
        uid_to_path: dict[str, str] = {}

        if object_ids_file and os.path.isfile(object_ids_file):
            requested_uids = _load_object_ids_file(object_ids_file)
            # Resolve against cache if available
            if cache_dir:
                all_cached = _find_glb_files(cache_dir)
                for uid in requested_uids:
                    if uid in all_cached:
                        uid_to_path[uid] = all_cached[uid]
                    else:
                        # Mark as needing fetch (path will be resolved later)
                        uid_to_path[uid] = ""
            else:
                for uid in requested_uids:
                    uid_to_path[uid] = ""
        elif cache_dir:
            uid_to_path = _find_glb_files(cache_dir)
        else:
            logger.warning(
                "ObjaverseXL: no source_path, cache_dir, or object_ids_file "
                "specified; inventory is empty."
            )
            return []

        # Sort for deterministic ordering
        sorted_uids = sorted(uid_to_path.keys())

        # Resumability: if resume is set, skip items already in raw store
        if ctx.resume:
            existing: set[str] = set()
            raw_dataset_dir = os.path.join(ctx.raw_root, config.dataset_id)
            if os.path.isdir(raw_dataset_dir):
                for name in os.listdir(raw_dataset_dir):
                    existing.add(name)
            sorted_uids = [u for u in sorted_uids if u not in existing]

        # Apply limit from context or config filter
        limit = ctx.limit
        if max_objects is not None:
            limit = min(limit, int(max_objects)) if limit else int(max_objects)
        if limit is not None:
            sorted_uids = sorted_uids[:limit]

        items: list[SourceItem] = []
        for uid in sorted_uids:
            path = uid_to_path[uid]
            items.append(
                SourceItem(
                    item_id=uid,
                    dataset_id=config.dataset_id,
                    item_type="asset",
                    source_path=path or None,
                    metadata={"uid": uid},
                )
            )

        logger.info(
            "ObjaverseXL inventory: %d objects found, %d selected",
            len(uid_to_path),
            len(items),
        )
        return items

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    def fetch(
        self, items: list[SourceItem], ctx: JobContext
    ) -> list[RawRef]:
        refs: list[RawRef] = []
        for item in items:
            if item.source_path and os.path.isfile(item.source_path):
                # Local file exists -- create a symlink in the raw store
                raw_dir = os.path.join(
                    ctx.raw_root, item.dataset_id, item.item_id
                )
                os.makedirs(raw_dir, exist_ok=True)
                fname = os.path.basename(item.source_path)
                link_path = os.path.join(raw_dir, fname)
                if not os.path.exists(link_path):
                    os.symlink(os.path.abspath(item.source_path), link_path)
                checksum = _sha256_file(item.source_path)
                size_bytes = os.path.getsize(item.source_path)
                refs.append(
                    RawRef(
                        item_id=item.item_id,
                        raw_path=raw_dir,
                        checksum_sha256=checksum,
                        size_bytes=size_bytes,
                    )
                )
            elif item.source_path == "" or item.source_path is None:
                # API / deferred mode: record path that will be populated
                refs.append(
                    RawRef(
                        item_id=item.item_id,
                        raw_path="",
                    )
                )
                logger.debug(
                    "ObjaverseXL: object %s has no local file; "
                    "requires API fetch or manual download.",
                    item.item_id,
                )
            else:
                logger.warning(
                    "ObjaverseXL: file not found for object %s at %s",
                    item.item_id,
                    item.source_path,
                )
        return refs

    # ------------------------------------------------------------------
    # parse_raw
    # ------------------------------------------------------------------

    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        assets: list[dict] = []
        licenses: list[dict] = []

        for ref in raw_refs:
            if not ref.raw_path or not os.path.isdir(ref.raw_path):
                continue

            mesh_files: list[str] = []
            for fname in os.listdir(ref.raw_path):
                _, ext = os.path.splitext(fname)
                if ext.lower() in _MESH_EXTENSIONS:
                    mesh_files.append(os.path.join(ref.raw_path, fname))

            if not mesh_files:
                logger.warning(
                    "ObjaverseXL: no mesh files found for object %s in %s",
                    ref.item_id,
                    ref.raw_path,
                )
                continue

            # Use the first mesh file as the primary mesh
            primary_mesh = mesh_files[0]
            primary_ext = os.path.splitext(primary_mesh)[1].lower()

            asset_dict: dict[str, Any] = {
                "uid": ref.item_id,
                "mesh_files": mesh_files,
                "primary_mesh": primary_mesh,
                "primary_ext": primary_ext,
                "checksum_sha256": ref.checksum_sha256,
                "size_bytes": ref.size_bytes,
            }
            assets.append(asset_dict)

            # Per-object license: default to unknown
            licenses.append(
                {
                    "uid": ref.item_id,
                    "license_name": "unknown",
                    "license_url": None,
                }
            )

        return ParseBundle(
            dataset_id=ctx.job_id.split(":")[0] if ":" in ctx.job_id else ctx.job_id,
            assets=assets,
            licenses=licenses,
            raw_refs=raw_refs,
        )

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(
        self, bundle: ParseBundle, ctx: JobContext
    ) -> NormalizeBundle:
        dataset_id = bundle.dataset_id
        now = datetime.now(timezone.utc)

        assets: list[AssetRecord] = []
        licenses: list[LicenseRecord] = []
        provenance: list[ProvenanceRecord] = []

        for asset_dict in bundle.assets:
            uid = asset_dict["uid"]
            asset_uid = make_asset_uid(dataset_id, uid)
            primary_mesh = asset_dict.get("primary_mesh")
            primary_ext = asset_dict.get("primary_ext", "")

            glb_uri = primary_mesh if primary_ext == ".glb" else None
            mesh_uri = primary_mesh

            assets.append(
                AssetRecord(
                    asset_uid=asset_uid,
                    dataset_id=dataset_id,
                    source_asset_id=uid,
                    category=None,
                    supercategory=None,
                    geometry_level=GeometryLevel.G4_EXACT_MESH,
                    is_articulated=False,
                    is_deformable=False,
                    mesh_uri=mesh_uri,
                    glb_uri=glb_uri,
                )
            )

            provenance.append(
                ProvenanceRecord(
                    record_id=asset_uid,
                    dataset_id=dataset_id,
                    source_relpath=primary_mesh,
                    source_sha256=asset_dict.get("checksum_sha256"),
                    normalized_by_version=self.version,
                    normalized_at=now,
                    adapter_name=self.name,
                    adapter_version=self.version,
                )
            )

        # Per-object licenses
        for lic_dict in bundle.licenses:
            uid = lic_dict["uid"]
            asset_uid = make_asset_uid(dataset_id, uid)
            licenses.append(
                LicenseRecord(
                    record_scope=RecordScope.ASSET,
                    record_id=asset_uid,
                    license_name=lic_dict.get("license_name", "unknown"),
                    license_url=lic_dict.get("license_url"),
                    commercial_use_allowed=None,
                    redistribution_allowed=None,
                    attribution_required=None,
                    notes="Per-object license; default unknown for Objaverse-XL.",
                )
            )

        dataset_record = DatasetRecord(
            dataset_id=dataset_id,
            dataset_name="Objaverse-XL",
            version="1",
            source_type=SourceType.LOCAL_FOLDER,
            license_name="mixed",
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G4_EXACT_MESH,
            created_at=now,
            tags=["3d", "objects", "mesh", "objaverse"],
        )

        return NormalizeBundle(
            dataset_id=dataset_id,
            dataset_record=dataset_record,
            assets=assets,
            licenses=licenses,
            provenance=provenance,
        )
