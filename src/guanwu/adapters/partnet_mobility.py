"""PartNet-Mobility adapter for articulated object assets."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import xml.etree.ElementTree as ET
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

_URDF_FILENAMES = ("mobility.urdf", "mobility_v2.urdf")


# ------------------------------------------------------------------
# URDF parsing helpers
# ------------------------------------------------------------------


def _parse_urdf(urdf_path: str) -> dict[str, Any]:
    """Parse a URDF file and extract links and joints.

    Returns a dict with:
      - links: list of {name, mesh_filename}
      - joints: list of {name, type, parent_link, child_link, axis, limits}
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: list[dict[str, Any]] = []
    for link_el in root.findall("link"):
        link_name = link_el.get("name", "")
        mesh_filename: str | None = None

        # Look for visual/geometry/mesh
        visual = link_el.find("visual")
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    mesh_filename = mesh_el.get("filename")

        links.append({"name": link_name, "mesh_filename": mesh_filename})

    joints: list[dict[str, Any]] = []
    for joint_el in root.findall("joint"):
        joint_name = joint_el.get("name", "")
        joint_type = joint_el.get("type", "fixed")

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        parent_link = parent_el.get("link", "") if parent_el is not None else ""
        child_link = child_el.get("link", "") if child_el is not None else ""

        # Axis
        axis_el = joint_el.find("axis")
        axis: list[float] | None = None
        if axis_el is not None:
            xyz_str = axis_el.get("xyz", "")
            if xyz_str:
                try:
                    axis = [float(v) for v in xyz_str.split()]
                except ValueError:
                    axis = None

        # Limits
        limit_el = joint_el.find("limit")
        limits: dict[str, float] | None = None
        if limit_el is not None:
            limits = {}
            for attr in ("lower", "upper", "effort", "velocity"):
                val = limit_el.get(attr)
                if val is not None:
                    try:
                        limits[attr] = float(val)
                    except ValueError:
                        pass

        joints.append(
            {
                "name": joint_name,
                "type": joint_type,
                "parent_link": parent_link,
                "child_link": child_link,
                "axis": axis,
                "limits": limits,
            }
        )

    return {"links": links, "joints": joints}


def _find_link_meshes(obj_dir: str) -> dict[str, str]:
    """Find textured OBJ meshes for links in textured_objs/.

    Returns a mapping of link/mesh name (stem) to absolute path.
    """
    meshes: dict[str, str] = {}
    tex_dir = os.path.join(obj_dir, "textured_objs")
    if not os.path.isdir(tex_dir):
        return meshes
    for fname in sorted(os.listdir(tex_dir)):
        if fname.lower().endswith(".obj"):
            stem = os.path.splitext(fname)[0]
            meshes[stem] = os.path.join(tex_dir, fname)
    return meshes


def _load_meta_json(obj_dir: str) -> dict[str, Any]:
    """Load meta.json from an object directory, returning {} on failure."""
    meta_path = os.path.join(obj_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("PartNet-Mobility: failed to read %s: %s", meta_path, exc)
        return {}


def _find_urdf(obj_dir: str) -> str | None:
    """Return the path to the mobility URDF if present, else None."""
    for name in _URDF_FILENAMES:
        candidate = os.path.join(obj_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


@register_adapter
class PartNetMobilityAdapter(DatasetAdapter):
    """Adapter for the PartNet-Mobility dataset.

    Expects a local directory tree where each sub-directory is an object
    identified by its numeric ID and contains a ``mobility.urdf`` (or
    ``mobility_v2.urdf``), ``meta.json``, and ``textured_objs/`` meshes.
    """

    name: str = "partnet_mobility"
    version: str = "0.1.0"

    def __init__(self) -> None:
        # Articulation data carried from normalize to emit, keyed by asset_uid
        self._articulation_data: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, bool]:
        return {
            "scene_mesh": False,
            "object_mesh": True,
            "articulation": True,
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
        source_path = config.source_path or config.cache_dir or ""
        if not source_path or not os.path.isdir(source_path):
            logger.warning(
                "PartNet-Mobility: source_path %r is not a valid directory.",
                source_path,
            )
            return []

        # Support newer layout with a dataset/ sub-directory
        dataset_subdir = os.path.join(source_path, "dataset")
        if os.path.isdir(dataset_subdir):
            scan_root = dataset_subdir
        else:
            scan_root = source_path

        # Scan for object directories that contain a mobility URDF
        object_ids: list[str] = []
        for entry in sorted(os.listdir(scan_root)):
            entry_path = os.path.join(scan_root, entry)
            if not os.path.isdir(entry_path):
                continue
            if _find_urdf(entry_path) is not None:
                object_ids.append(entry)

        # Resumability: skip objects whose raw directories already exist
        if ctx.resume:
            existing: set[str] = set()
            raw_dataset_dir = os.path.join(ctx.raw_root, config.dataset_id)
            if os.path.isdir(raw_dataset_dir):
                for name in os.listdir(raw_dataset_dir):
                    existing.add(name)
            object_ids = [oid for oid in object_ids if oid not in existing]

        # Apply limit
        max_objects = config.filters.get("max_objects")
        limit = ctx.limit
        if max_objects is not None:
            limit = min(limit, int(max_objects)) if limit else int(max_objects)
        if limit is not None:
            object_ids = object_ids[:limit]

        items: list[SourceItem] = []
        for oid in object_ids:
            obj_path = os.path.join(scan_root, oid)
            meta = _load_meta_json(obj_path)
            items.append(
                SourceItem(
                    item_id=oid,
                    dataset_id=config.dataset_id,
                    item_type="asset",
                    source_path=obj_path,
                    metadata={
                        "object_id": oid,
                        "category": meta.get("model_cat", ""),
                        "anno_id": meta.get("anno_id", oid),
                    },
                )
            )

        logger.info(
            "PartNet-Mobility inventory: %d objects found, %d selected",
            len(object_ids),
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
            if not item.source_path or not os.path.isdir(item.source_path):
                logger.warning(
                    "PartNet-Mobility: source directory missing for object %s",
                    item.item_id,
                )
                continue

            raw_dir = os.path.join(ctx.raw_root, item.dataset_id, item.item_id)
            os.makedirs(os.path.dirname(raw_dir), exist_ok=True)

            # Symlink the entire object directory into raw store
            if not os.path.exists(raw_dir):
                os.symlink(os.path.abspath(item.source_path), raw_dir)

            # Checksum the URDF as the representative file
            urdf_path = _find_urdf(item.source_path)
            checksum = _sha256_file(urdf_path) if urdf_path else None

            refs.append(
                RawRef(
                    item_id=item.item_id,
                    raw_path=raw_dir,
                    checksum_sha256=checksum,
                )
            )

        return refs

    # ------------------------------------------------------------------
    # parse_raw
    # ------------------------------------------------------------------

    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        assets: list[dict] = []
        articulations: list[dict] = []
        licenses: list[dict] = []

        for ref in raw_refs:
            obj_dir = ref.raw_path
            if not obj_dir or not os.path.isdir(obj_dir):
                continue

            urdf_path = _find_urdf(obj_dir)
            if urdf_path is None:
                logger.warning(
                    "PartNet-Mobility: no URDF found for object %s in %s",
                    ref.item_id,
                    obj_dir,
                )
                continue

            # Parse URDF
            urdf_data = _parse_urdf(urdf_path)

            # Find link meshes
            link_meshes = _find_link_meshes(obj_dir)

            # Read metadata
            meta = _load_meta_json(obj_dir)
            category = meta.get("model_cat", "")
            anno_id = meta.get("anno_id", ref.item_id)

            # Read optional bounding box
            bbox_path = os.path.join(obj_dir, "bounding_box.json")
            bbox_data: dict[str, Any] | None = None
            if os.path.isfile(bbox_path):
                try:
                    with open(bbox_path, "r") as f:
                        bbox_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            # Extract joint info
            joint_names: list[str] = []
            joint_types: list[str] = []
            joint_limits: list[dict[str, float] | None] = []
            for jnt in urdf_data["joints"]:
                joint_names.append(jnt["name"])
                joint_types.append(jnt["type"])
                joint_limits.append(jnt.get("limits"))

            asset_dict: dict[str, Any] = {
                "object_id": ref.item_id,
                "category": category,
                "anno_id": anno_id,
                "urdf_path": urdf_path,
                "link_meshes": link_meshes,
                "links": urdf_data["links"],
                "joints": urdf_data["joints"],
                "joint_names": joint_names,
                "joint_types": joint_types,
                "joint_limits": joint_limits,
                "bbox": bbox_data,
                "checksum_sha256": ref.checksum_sha256,
            }
            assets.append(asset_dict)

            articulations.append(
                {
                    "object_id": ref.item_id,
                    "joint_names": joint_names,
                    "joint_types": joint_types,
                    "joint_limits": joint_limits,
                    "links": urdf_data["links"],
                    "joints": urdf_data["joints"],
                }
            )

            licenses.append(
                {
                    "object_id": ref.item_id,
                    "license_name": None,
                    "license_url": None,
                }
            )

        return ParseBundle(
            dataset_id="partnet_mobility",
            assets=assets,
            articulations=articulations,
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
            object_id = asset_dict["object_id"]
            category = asset_dict.get("category") or "unknown"

            # Use readable uid: category_objectid (e.g. Cabinet_7128)
            safe_cat = category.replace(" ", "_")
            asset_uid = f"{safe_cat}_{object_id}"

            # Determine mesh URI: prefer the first textured OBJ, fall back to
            # whatever the URDF references.
            link_meshes: dict[str, str] = asset_dict.get("link_meshes", {})
            mesh_uri: str | None = None
            if link_meshes:
                mesh_uri = next(iter(link_meshes.values()))
            urdf_path = asset_dict.get("urdf_path")

            # Collect joint metadata for storage alongside the asset
            joint_names = asset_dict.get("joint_names", [])
            joint_types = asset_dict.get("joint_types", [])
            joint_limits = asset_dict.get("joint_limits", [])

            assets.append(
                AssetRecord(
                    asset_uid=asset_uid,
                    dataset_id=dataset_id,
                    source_asset_id=object_id,
                    category=category,
                    geometry_level=GeometryLevel.G5_ARTICULATED_MESH,
                    is_articulated=True,
                    is_deformable=False,
                    mesh_uri=mesh_uri,
                )
            )

            # Stash articulation data for emit phase
            self._articulation_data[asset_uid] = {
                "link_meshes": link_meshes,
                "joints": asset_dict.get("joints", []),
            }

            # Per-object license
            lic_dict = _find_license_for(object_id, bundle.licenses)
            licenses.append(
                LicenseRecord(
                    record_scope=RecordScope.ASSET,
                    record_id=asset_uid,
                    license_name=lic_dict.get("license_name") if lic_dict else None,
                    license_url=lic_dict.get("license_url") if lic_dict else None,
                    notes="PartNet-Mobility per-object license.",
                )
            )

            provenance.append(
                ProvenanceRecord(
                    record_id=asset_uid,
                    dataset_id=dataset_id,
                    source_relpath=urdf_path,
                    source_sha256=asset_dict.get("checksum_sha256"),
                    normalized_by_version=self.version,
                    normalized_at=now,
                    adapter_name=self.name,
                    adapter_version=self.version,
                    transform_log=[
                        {
                            "step": "urdf_parse",
                            "joint_names": joint_names,
                            "joint_types": joint_types,
                            "joint_limits": joint_limits,
                        }
                    ],
                )
            )

        dataset_record = DatasetRecord(
            dataset_id=dataset_id,
            dataset_name="PartNet-Mobility",
            version="1",
            source_type=SourceType.LOCAL_FOLDER,
            license_name="PartNet-Mobility License",
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G5_ARTICULATED_MESH,
            created_at=now,
            tags=["3d", "articulated", "urdf", "partnet"],
        )

        return NormalizeBundle(
            dataset_id=dataset_id,
            dataset_record=dataset_record,
            assets=assets,
            licenses=licenses,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # emit  (override to write articulated USDC)
    # ------------------------------------------------------------------

    def emit(
        self, bundle: NormalizeBundle, ctx: JobContext
    ) -> EmitReport:
        from guanwu.schemas.bundles import EmitReport

        # Run default emit first (writes JSON/Parquet)
        report = super().emit(bundle, ctx)

        # Then generate articulated USDC for each asset
        try:
            from guanwu.exporters.usd import articulated_asset_to_usdc, animated_asset_to_usdc
        except ImportError:
            return report

        from guanwu.storage.canonical_store import CanonicalStore

        store = CanonicalStore(ctx.canonical_root)

        for asset in bundle.assets:
            art_data = self._articulation_data.get(asset.asset_uid)
            if not art_data:
                continue

            link_meshes = art_data.get("link_meshes", {})
            joints = art_data.get("joints", [])

            if not link_meshes:
                continue

            asset_dir = store.asset_dir(bundle.dataset_id, asset.asset_uid)

            # Static articulated USDC
            articulated_asset_to_usdc(link_meshes, joints, asset_dir / "asset.usdc")

            # Animated USDC: synthetic open-close trajectory (60 frames)
            if joints:
                num_frames = 60
                joint_trajectories = {}
                for j in joints:
                    limits = j.get("limits", {})
                    lower = float(limits.get("lower", 0))
                    upper = float(limits.get("upper", 1.0))
                    traj = []
                    for f in range(num_frames):
                        t = f / 29.0 if f < 30 else 1.0 - (f - 30) / 29.0
                        traj.append(lower + t * (upper - lower))
                    joint_trajectories[j["name"]] = traj

                animated_asset_to_usdc(
                    link_meshes, joints, joint_trajectories,
                    asset_dir / "animated.usdc", fps=24.0,
                )

        return report


def _find_license_for(
    object_id: str, license_dicts: list[dict]
) -> dict[str, Any] | None:
    """Find the license dict matching a given object_id."""
    for ld in license_dicts:
        if ld.get("object_id") == object_id:
            return ld
    return None
