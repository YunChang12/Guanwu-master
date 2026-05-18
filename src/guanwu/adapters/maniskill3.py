"""ManiSkill 3 adapter for simulation scenes, trajectories, and environment states.

Expected local directory structure::

    <path>/
      demos/
        <task_name>/
          trajectory.h5   OR  *.h5
          ...
      scene_datasets/  (optional)
        ai2thor/
        replica_cad/
        ...
      assets/  (optional)
        <asset_type>/
          <asset_id>/
            model.glb or model.obj
            ...
      robots/ (optional)
        <robot_name>/
          ...

Trajectory HDF5 structure (typical)::

    trajectory.h5
      env_states/     # list of env state dicts
      actions/        # action arrays
      obs/            # observation arrays (optional)
      success/        # bool success flags
      env_kwargs/     # environment config
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guanwu.adapters.base import DatasetAdapter, register_adapter
from guanwu.core.ids import (
    make_asset_uid,
    make_episode_uid,
    make_instance_uid,
    make_scene_uid,
    make_track_uid,
)
from guanwu.schemas.bundles import (
    AdapterConfig,
    EmitReport,
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
    SceneKind,
    SourceType,
)
from guanwu.schemas.records import (
    ArticulationStateRecord,
    AssetRecord,
    DatasetRecord,
    InstanceRecord,
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
    TrackStateRecord,
)
from guanwu.storage.raw_store import RawStore

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

logger = logging.getLogger("guanwu")

DATASET_ID = "maniskill3"

_MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl", ".ply", ".dae", ".fbx"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h5_files(root: Path) -> list[Path]:
    """Return all .h5 files under *root* (recursive)."""
    return sorted(root.rglob("*.h5")) if root.is_dir() else []


def _opt(config: AdapterConfig, key: str, default: Any = None) -> Any:
    return config.options.get(key, default)


def _read_h5_trajectory(h5_path: Path) -> dict[str, Any]:
    """Read a ManiSkill trajectory HDF5 file.

    Returns a dict with keys like ``env_states``, ``actions``, ``success``,
    ``env_kwargs``, and the list of ``episode_keys`` found in the file.
    """
    if h5py is None:
        raise RuntimeError(
            "h5py is required to parse ManiSkill trajectory files. "
            "Install it with: pip install h5py"
        )

    result: dict[str, Any] = {"episodes": [], "env_kwargs": {}}

    with h5py.File(str(h5_path), "r") as f:
        # Top-level env_kwargs (stored as JSON string attribute sometimes)
        if "env_kwargs" in f.attrs:
            import json

            try:
                result["env_kwargs"] = json.loads(f.attrs["env_kwargs"])
            except (json.JSONDecodeError, TypeError):
                result["env_kwargs"] = dict(f.attrs["env_kwargs"])
        elif "env_kwargs" in f:
            result["env_kwargs"] = {
                k: v[()] if hasattr(v, "shape") else v
                for k, v in f["env_kwargs"].items()
            }

        # Each top-level group starting with "traj" is an episode
        episode_keys = sorted(
            k for k in f.keys() if k.startswith("traj")
        )

        for ep_key in episode_keys:
            ep_group = f[ep_key]
            ep_data: dict[str, Any] = {"key": ep_key}

            # actions
            if "actions" in ep_group:
                import numpy as np

                ep_data["actions"] = np.array(ep_group["actions"])
                ep_data["num_steps"] = ep_data["actions"].shape[0]

            # env_states – may be a group of datasets or a single dataset
            if "env_states" in ep_group:
                env_states_grp = ep_group["env_states"]
                if isinstance(env_states_grp, h5py.Dataset):
                    import numpy as np

                    ep_data["env_states_raw"] = np.array(env_states_grp)
                else:
                    # It is a group – read each child
                    ep_data["env_states_keys"] = list(env_states_grp.keys())

            # success flag
            if "success" in ep_group:
                import numpy as np

                ep_data["success"] = bool(np.array(ep_group["success"]).any())
            else:
                ep_data["success"] = None

            # info dict (optional)
            if "info" in ep_group:
                info_grp = ep_group["info"]
                if isinstance(info_grp, h5py.Group):
                    ep_data["info_keys"] = list(info_grp.keys())

            result["episodes"].append(ep_data)

    return result


def _discover_scene_datasets(root: Path) -> list[dict[str, Any]]:
    """Discover scene datasets under ``scene_datasets/``."""
    scene_ds_dir = root / "scene_datasets"
    if not scene_ds_dir.is_dir():
        return []
    scenes: list[dict[str, Any]] = []
    for provider_dir in sorted(scene_ds_dir.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider_name = provider_dir.name
        for scene_dir in sorted(provider_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            mesh_files = [
                p for p in scene_dir.rglob("*")
                if p.suffix.lower() in _MESH_EXTENSIONS
            ]
            scenes.append(
                {
                    "provider": provider_name,
                    "scene_id": f"{provider_name}/{scene_dir.name}",
                    "scene_name": scene_dir.name,
                    "scene_dir": str(scene_dir),
                    "mesh_files": [str(m) for m in mesh_files],
                }
            )
    return scenes


def _discover_assets(root: Path) -> list[dict[str, Any]]:
    """Discover object / robot assets under ``assets/`` and ``robots/``."""
    assets: list[dict[str, Any]] = []
    for subdir_name in ("assets", "robots"):
        assets_dir = root / subdir_name
        if not assets_dir.is_dir():
            continue
        for category_dir in sorted(assets_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for asset_dir in sorted(category_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                mesh_files = [
                    p for p in asset_dir.rglob("*")
                    if p.suffix.lower() in _MESH_EXTENSIONS
                ]
                glb_files = [
                    m for m in mesh_files if m.suffix.lower() in (".glb", ".gltf")
                ]
                obj_files = [
                    m for m in mesh_files if m.suffix.lower() == ".obj"
                ]
                is_robot = subdir_name == "robots"
                assets.append(
                    {
                        "source_asset_id": f"{subdir_name}/{category}/{asset_dir.name}",
                        "category": category,
                        "asset_name": asset_dir.name,
                        "asset_dir": str(asset_dir),
                        "mesh_files": [str(m) for m in mesh_files],
                        "glb_path": str(glb_files[0]) if glb_files else None,
                        "obj_path": str(obj_files[0]) if obj_files else None,
                        "is_robot": is_robot,
                        "is_articulated": is_robot,
                    }
                )
    return assets


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_adapter
class ManiSkill3Adapter(DatasetAdapter):
    """Adapter for ManiSkill 3 manipulation environments and demonstrations."""

    name: str = "maniskill3"
    version: str = "0.1.0"

    def __init__(self) -> None:
        # Stashed during parse for use in emit:
        # {h5_path: {env_id, control_mode, traj_keys}}
        self._h5_info: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> dict[str, bool]:
        return {
            "scene_mesh": True,
            "object_mesh": True,
            "articulation": True,
            "deformable_mesh": False,
            "camera": False,
            "depth": False,
            "lidar": False,
            "tracks": True,
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
        source = Path(config.source_path) if config.source_path else None
        if source is None or not source.is_dir():
            logger.warning("ManiSkill3 source_path is not set or not a directory")
            return []

        dataset_id = config.dataset_id or DATASET_ID
        ingest_trajectories = _opt(config, "ingest_trajectories", True)
        ingest_scene_assets = _opt(config, "ingest_scene_assets", True)
        items: list[SourceItem] = []

        # --- trajectories ---
        if ingest_trajectories:
            demos_dir = source / "demos"
            h5_files = _h5_files(demos_dir)
            for h5_path in h5_files:
                rel = h5_path.relative_to(source)
                items.append(
                    SourceItem(
                        item_id=str(rel),
                        dataset_id=dataset_id,
                        item_type="episode",
                        source_path=str(h5_path),
                        metadata={"h5_relpath": str(rel)},
                    )
                )

        # --- scene datasets ---
        if ingest_scene_assets:
            scenes = _discover_scene_datasets(source)
            for s in scenes:
                items.append(
                    SourceItem(
                        item_id=f"scene:{s['scene_id']}",
                        dataset_id=dataset_id,
                        item_type="scene",
                        source_path=s["scene_dir"],
                        metadata=s,
                    )
                )

        # --- assets (objects + robots) ---
        if ingest_scene_assets:
            assets = _discover_assets(source)
            for a in assets:
                items.append(
                    SourceItem(
                        item_id=f"asset:{a['source_asset_id']}",
                        dataset_id=dataset_id,
                        item_type="asset",
                        source_path=a["asset_dir"],
                        metadata=a,
                    )
                )

        if ctx.limit is not None:
            items = items[: ctx.limit]

        logger.info(
            "ManiSkill3 inventory: %d items (trajectories=%s, scenes=%s)",
            len(items),
            ingest_trajectories,
            ingest_scene_assets,
        )
        return items

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    def fetch(
        self, items: list[SourceItem], ctx: JobContext
    ) -> list[RawRef]:
        if not items:
            return []

        dataset_id = items[0].dataset_id
        raw_store = RawStore(ctx.raw_root)

        # For local ingestion we symlink the entire source directory once.
        # Determine the common source root from the first item.
        first_source = Path(items[0].source_path) if items[0].source_path else None
        if first_source is None:
            logger.warning("No source_path on inventory items; nothing to fetch")
            return []

        # Walk up to the common root (the top-level ManiSkill directory).
        # Items may point into demos/, scene_datasets/, assets/ so we look
        # for the lowest directory that contains all items.
        all_source_paths = [
            Path(it.source_path).resolve()
            for it in items
            if it.source_path
        ]
        if not all_source_paths:
            return []

        common_root = all_source_paths[0]
        if not common_root.is_dir():
            common_root = common_root.parent
        for p in all_source_paths[1:]:
            # Find common ancestor directory
            while not str(p).startswith(str(common_root)):
                common_root = common_root.parent

        linked_dir = raw_store.link_directory(dataset_id, common_root)

        refs: list[RawRef] = []
        for item in items:
            if item.source_path:
                src = Path(item.source_path).resolve()
                try:
                    rel = src.relative_to(common_root)
                except ValueError:
                    rel = Path(src.name)
                refs.append(
                    RawRef(
                        item_id=item.item_id,
                        raw_path=str(linked_dir / rel),
                    )
                )
        return refs

    # ------------------------------------------------------------------
    # parse_raw
    # ------------------------------------------------------------------

    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        if not raw_refs:
            return ParseBundle(dataset_id=DATASET_ID)

        dataset_id = DATASET_ID
        bundle = ParseBundle(dataset_id=dataset_id, raw_refs=raw_refs)

        for ref in raw_refs:
            raw_path = Path(ref.raw_path)
            if ref.item_id.startswith("scene:"):
                # Scene dataset directory
                if raw_path.is_dir():
                    mesh_files = [
                        p for p in raw_path.rglob("*")
                        if p.suffix.lower() in _MESH_EXTENSIONS
                    ]
                    scene_id = ref.item_id.removeprefix("scene:")
                    bundle.scenes.append(
                        {
                            "source_scene_id": scene_id,
                            "scene_name": raw_path.name,
                            "scene_dir": str(raw_path),
                            "mesh_files": [str(m) for m in mesh_files],
                            "has_mesh": len(mesh_files) > 0,
                        }
                    )

            elif ref.item_id.startswith("asset:"):
                # Object or robot asset directory
                if raw_path.is_dir():
                    mesh_files = [
                        p for p in raw_path.rglob("*")
                        if p.suffix.lower() in _MESH_EXTENSIONS
                    ]
                    glb_files = [
                        m for m in mesh_files
                        if m.suffix.lower() in (".glb", ".gltf")
                    ]
                    obj_files = [
                        m for m in mesh_files if m.suffix.lower() == ".obj"
                    ]
                    source_asset_id = ref.item_id.removeprefix("asset:")
                    parts = source_asset_id.split("/")
                    is_robot = parts[0] == "robots" if parts else False
                    category = parts[1] if len(parts) > 1 else None
                    bundle.assets.append(
                        {
                            "source_asset_id": source_asset_id,
                            "category": category,
                            "asset_name": raw_path.name,
                            "asset_dir": str(raw_path),
                            "mesh_files": [str(m) for m in mesh_files],
                            "glb_path": str(glb_files[0]) if glb_files else None,
                            "obj_path": str(obj_files[0]) if obj_files else None,
                            "is_robot": is_robot,
                            "is_articulated": is_robot,
                        }
                    )

            else:
                # Trajectory H5 file
                raw_file = Path(ref.raw_path)
                if raw_file.is_file() and raw_file.suffix == ".h5":
                    if h5py is None:
                        logger.warning(
                            "h5py not installed; skipping trajectory %s",
                            raw_file,
                        )
                        continue
                    try:
                        traj_data = _read_h5_trajectory(raw_file)
                    except Exception:
                        logger.exception(
                            "Failed to read trajectory file %s", raw_file
                        )
                        continue

                    # Derive task name from parent directory
                    # Resolve symlinks to get the real path for env_id detection
                    real_file = raw_file.resolve()
                    task_name = real_file.parent.name
                    # env_id: demos/<env_id>/<subdir>/trajectory.h5 → grandparent
                    env_id_candidate = real_file.parent.parent.name
                    env_id = env_id_candidate if env_id_candidate != "demos" else task_name

                    # Read control_mode from companion JSON if available
                    control_mode = "pd_joint_pos"
                    json_path = raw_file.with_suffix(".json")
                    if json_path.is_file():
                        try:
                            import json as _json
                            with open(json_path) as _jf:
                                _meta = _json.load(_jf)
                            control_mode = (
                                _meta.get("env_info", {})
                                .get("env_kwargs", {})
                                .get("control_mode", control_mode)
                            )
                        except Exception:
                            pass

                    # Stash H5 info for remote replay in emit
                    h5_key = str(raw_file)
                    if h5_key not in self._h5_info:
                        self._h5_info[h5_key] = {
                            "env_id": env_id,
                            "control_mode": control_mode,
                            "traj_keys": [],
                        }

                    for ep in traj_data["episodes"]:
                        ep_id = f"{task_name}/{ep['key']}"
                        self._h5_info[h5_key]["traj_keys"].append(ep["key"])
                        ep_dict: dict[str, Any] = {
                            "source_episode_id": ep_id,
                            "task_name": task_name,
                            "h5_path": str(raw_file),
                            "h5_key": ep["key"],
                            "num_steps": ep.get("num_steps"),
                            "success": ep.get("success"),
                            "has_actions": "actions" in ep,
                            "has_env_states": (
                                "env_states_raw" in ep
                                or "env_states_keys" in ep
                            ),
                            "env_states_keys": ep.get("env_states_keys", []),
                        }
                        bundle.instances.append(ep_dict)

                    # Store env_kwargs as annotation metadata
                    if traj_data.get("env_kwargs"):
                        bundle.annotations.append(
                            {
                                "type": "env_kwargs",
                                "task_name": task_name,
                                "h5_path": str(raw_file),
                                "data": _safe_serialize(traj_data["env_kwargs"]),
                            }
                        )

                    # Articulation metadata from env_states keys
                    for ep in traj_data["episodes"]:
                        if ep.get("env_states_keys"):
                            bundle.articulations.append(
                                {
                                    "source_episode_id": f"{task_name}/{ep['key']}",
                                    "env_states_keys": ep["env_states_keys"],
                                }
                            )

        return bundle

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(
        self, bundle: ParseBundle, ctx: JobContext
    ) -> NormalizeBundle:
        dataset_id = bundle.dataset_id
        now = datetime.now(tz=timezone.utc)

        out = NormalizeBundle(dataset_id=dataset_id)

        # --- DatasetRecord ---
        has_scenes = len(bundle.scenes) > 0
        has_assets = len(bundle.assets) > 0
        has_episodes = len(bundle.instances) > 0

        max_geom = GeometryLevel.G0_NONE
        if has_scenes or has_assets:
            max_geom = GeometryLevel.G4_EXACT_MESH
        if any(a.get("is_articulated") for a in bundle.assets):
            max_geom = GeometryLevel.G5_ARTICULATED_MESH

        out.dataset_record = DatasetRecord(
            dataset_id=dataset_id,
            dataset_name="ManiSkill 3",
            version="0.1.0",
            source_type=SourceType.LOCAL_FOLDER,
            license_name="Apache-2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=max_geom,
            created_at=now,
            tags=["manipulation", "robotics", "simulation", "maniskill"],
        )

        # --- SceneRecords ---
        for s in bundle.scenes:
            scene_uid = make_scene_uid(dataset_id, s["source_scene_id"])
            has_mesh = s.get("has_mesh", False)
            out.scenes.append(
                SceneRecord(
                    scene_uid=scene_uid,
                    dataset_id=dataset_id,
                    source_scene_id=s["source_scene_id"],
                    scene_name=s.get("scene_name"),
                    scene_kind=SceneKind.INDOOR_STATIC,
                    geometry_level=(
                        GeometryLevel.G4_EXACT_MESH
                        if has_mesh
                        else GeometryLevel.G0_NONE
                    ),
                    has_static_scene_mesh=has_mesh,
                    has_dynamic_objects=False,
                    has_humans=False,
                    has_articulation=False,
                )
            )

        # --- AssetRecords ---
        for a in bundle.assets:
            asset_uid = make_asset_uid(dataset_id, a["source_asset_id"])
            is_art = a.get("is_articulated", False)
            geom = (
                GeometryLevel.G5_ARTICULATED_MESH
                if is_art
                else GeometryLevel.G4_EXACT_MESH
            )
            out.assets.append(
                AssetRecord(
                    asset_uid=asset_uid,
                    dataset_id=dataset_id,
                    source_asset_id=a["source_asset_id"],
                    category=a.get("category"),
                    geometry_level=geom,
                    is_articulated=is_art,
                    is_deformable=False,
                    glb_uri=a.get("glb_path"),
                    mesh_uri=a.get("obj_path") or a.get("glb_path"),
                )
            )

        # --- Episode / trajectory records ---
        # ManiSkill episodes are in bundle.instances from parse step
        for ep_dict in bundle.instances:
            ep_source_id = ep_dict["source_episode_id"]
            episode_uid = make_episode_uid(dataset_id, ep_source_id)
            task_name = ep_dict.get("task_name", "unknown")
            num_steps = ep_dict.get("num_steps", 0) or 0

            # Create a scene record for the episode (manipulation scene)
            ep_scene_uid = make_scene_uid(
                dataset_id, f"episode:{ep_source_id}"
            )
            out.scenes.append(
                SceneRecord(
                    scene_uid=ep_scene_uid,
                    dataset_id=dataset_id,
                    source_scene_id=f"episode:{ep_source_id}",
                    scene_name=f"{task_name}/{ep_dict.get('h5_key', '')}",
                    scene_kind=SceneKind.SYNTHETIC_MANIPULATION,
                    geometry_level=GeometryLevel.G0_NONE,
                    num_frames=num_steps,
                    has_static_scene_mesh=False,
                    has_dynamic_objects=True,
                    has_humans=False,
                    has_articulation=ep_dict.get("has_env_states", False),
                )
            )

            # Create an instance for the robot / manipulated object
            robot_instance_uid = make_instance_uid(
                dataset_id, episode_uid, "robot"
            )
            out.instances.append(
                InstanceRecord(
                    instance_uid=robot_instance_uid,
                    scene_uid=ep_scene_uid,
                    episode_uid=episode_uid,
                    category="robot",
                    instance_name="robot",
                    is_static=False,
                    is_articulated=True,
                    is_human=False,
                    geometry_level=GeometryLevel.G0_NONE,
                )
            )

            # Create track state records from env_states
            if num_steps > 0 and ep_dict.get("has_env_states", False):
                track_uid = make_track_uid(dataset_id, robot_instance_uid)
                # We emit per-step track states if env_states exist.
                # In v0.1 we record the step indices as timestamps.
                for step_idx in range(num_steps):
                    out.track_states.append(
                        TrackStateRecord(
                            track_uid=track_uid,
                            instance_uid=robot_instance_uid,
                            timestamp_ns=step_idx * 1_000_000,  # 1ms per step
                        )
                    )

            # Create articulation state placeholders from env_states keys
            env_states_keys = ep_dict.get("env_states_keys", [])
            if env_states_keys and num_steps > 0:
                joint_names = [
                    k for k in env_states_keys if "qpos" in k or "joint" in k
                ]
                if not joint_names:
                    # Use all keys as potential state dimensions
                    joint_names = env_states_keys

                for step_idx in range(num_steps):
                    out.articulation_states.append(
                        ArticulationStateRecord(
                            instance_uid=robot_instance_uid,
                            asset_uid="",  # no asset link in v0.1
                            timestamp_ns=step_idx * 1_000_000,
                            joint_names=joint_names,
                            joint_positions=[0.0] * len(joint_names),
                        )
                    )

        # --- LicenseRecord ---
        out.licenses.append(
            LicenseRecord(
                record_scope=RecordScope.DATASET,
                record_id=dataset_id,
                license_name="Apache-2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
                commercial_use_allowed=True,
                redistribution_allowed=True,
                attribution_required=True,
                notes="ManiSkill 3 is released under the Apache 2.0 license.",
            )
        )

        # --- ProvenanceRecord ---
        out.provenance.append(
            ProvenanceRecord(
                record_id=dataset_id,
                dataset_id=dataset_id,
                normalized_by_version=self.version,
                normalized_at=now,
                adapter_name=self.name,
                adapter_version=self.version,
                transform_log=[
                    {
                        "step": "normalize",
                        "scenes": len(out.scenes),
                        "assets": len(out.assets),
                        "instances": len(out.instances),
                        "track_states": len(out.track_states),
                    }
                ],
            )
        )

        return out

    # ------------------------------------------------------------------
    # emit  (override: auto remote replay + animated USDC)
    # ------------------------------------------------------------------

    def emit(
        self, bundle: NormalizeBundle, ctx: JobContext
    ) -> EmitReport:
        from guanwu.schemas.bundles import EmitReport

        # Run default emit (writes JSON/Parquet)
        report = super().emit(bundle, ctx)

        # If remote is configured, replay trajectories to get real FK
        if not ctx.remote_host or not self._h5_info:
            return report

        if ctx.dry_run:
            logger.info("Dry-run: skipping remote replay")
            return report

        logger.info("Remote GPU configured (%s), running trajectory replay...", ctx.remote_host)

        try:
            from guanwu.core.config import RemoteConfig
            from guanwu.core.remote import get_remote_executor
            from guanwu.core.remote_tasks import replay_trajectory

            remote_cfg = RemoteConfig(
                host=ctx.remote_host,
                conda_env=ctx.remote_conda_env,
                work_dir=ctx.remote_work_dir,
                python=ctx.remote_python,
                conda_init=ctx.remote_conda_init,
            )
            executor = get_remote_executor(remote_cfg)
            if executor is None:
                return report

            from guanwu.storage.canonical_store import CanonicalStore
            store = CanonicalStore(ctx.canonical_root)
            anim_dir = Path(ctx.canonical_root) / "datasets" / bundle.dataset_id / "animated_scenes"
            anim_dir.mkdir(parents=True, exist_ok=True)

            for h5_path_str, info in self._h5_info.items():
                h5_path = Path(h5_path_str).resolve()  # resolve symlinks
                if not h5_path.exists():
                    logger.warning("H5 file not found: %s", h5_path)
                    continue

                env_id = info["env_id"]
                control_mode = info["control_mode"]
                traj_keys = info["traj_keys"]

                # Limit trajectories for replay (default: 5 to avoid long runs)
                max_replay = min(ctx.limit or 5, 5)
                traj_keys = traj_keys[:max_replay]

                logger.info(
                    "Replaying %s (env=%s, %d trajs) on %s...",
                    h5_path.name, env_id, len(traj_keys), ctx.remote_host,
                )

                replay_out = Path(ctx.staging_root) / "replay" / bundle.dataset_id / env_id
                replay_out.mkdir(parents=True, exist_ok=True)

                result = replay_trajectory(
                    executor,
                    h5_path=h5_path,
                    env_id=env_id,
                    control_mode=control_mode,
                    traj_keys=traj_keys,
                    output_dir=replay_out,
                    max_trajs=ctx.limit,
                )

                # Generate animated USDC per trajectory
                for traj_key, npz_path in result.get("output_files", {}).items():
                    try:
                        self._generate_animated_usdc(
                            npz_path, env_id, traj_key, anim_dir, bundle,
                        )
                        report.files_written.append(
                            f"animated_scenes/{env_id}_{traj_key}.usdc"
                        )
                    except Exception as e:
                        logger.error("Failed to generate USDC for %s/%s: %s", env_id, traj_key, e)

        except Exception as e:
            logger.error("Remote replay failed: %s", e, exc_info=True)

        return report

    def _generate_animated_usdc(
        self,
        npz_path: str,
        env_id: str,
        traj_key: str,
        output_dir: Path,
        bundle: NormalizeBundle,
    ) -> None:
        """Generate FK animated USDC from replay npz data."""
        try:
            import numpy as np
            from pxr import Usd, UsdGeom, Gf, Vt
            import trimesh
        except ImportError:
            logger.warning("usd-core or trimesh not installed, skipping USDC generation")
            return

        import xml.etree.ElementTree as ET

        data = np.load(npz_path)
        link_names = sorted(set(k.replace("pos_", "") for k in data.files if k.startswith("pos_")))
        num_frames = len(data["qpos"])

        # Find Panda URDF for mesh paths
        try:
            import mani_skill
            ms_root = Path(mani_skill.__file__).parent
        except ImportError:
            # Fallback: try local venv
            import sys
            for p in sys.path:
                candidate = Path(p) / "mani_skill"
                if candidate.is_dir():
                    ms_root = candidate
                    break
            else:
                logger.warning("Cannot find mani_skill package, skipping USDC")
                return

        urdf_path = ms_root / "assets/robots/panda/panda_v3.urdf"
        if not urdf_path.exists():
            logger.warning("Panda URDF not found: %s", urdf_path)
            return

        urdf_dir = urdf_path.parent
        tree = ET.parse(str(urdf_path))
        urdf_root = tree.getroot()

        link_mesh_path = {}
        link_vis_rpy = {}
        for le in urdf_root.findall("link"):
            n = le.get("name", "")
            me = le.find(".//visual//mesh")
            if me is not None:
                p = urdf_dir / me.get("filename", "")
                if p.exists():
                    link_mesh_path[n] = str(p)
            vo = le.find(".//visual/origin")
            if vo is not None:
                link_vis_rpy[n] = [float(x) for x in vo.get("rpy", "0 0 0").split()]

        def _q2m(q):
            w, x, y, z = q
            return np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
            ])

        def _rpy2m(rpy):
            r, p, y = rpy
            cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
            return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
                    @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
                    @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))

        def _n2g(m):
            g = Gf.Matrix4d(); t = m.T
            for r in range(4):
                g.SetRow(r, Gf.Vec4d(*t[r].tolist()))
            return g

        def _load_glb(path, vis_rpy=None):
            scene = trimesh.load(str(path), process=False)
            if not isinstance(scene, trimesh.Scene):
                return (scene, None) if isinstance(scene, trimesh.Trimesh) else (None, None)
            g2t = {}
            for node in scene.graph.nodes:
                T, gn = scene.graph.get(node)
                if gn is not None:
                    g2t[gn] = T
            all_v, all_f, all_fc, off = [], [], [], 0
            for gn, geom in scene.geometry.items():
                if not isinstance(geom, trimesh.Trimesh):
                    continue
                T = g2t.get(gn, np.eye(4))
                v = (T @ np.hstack([geom.vertices, np.ones((len(geom.vertices), 1))]).T).T[:, :3]
                fc = np.array([0.7, 0.7, 0.7])
                mat = getattr(getattr(geom, "visual", None), "material", None)
                mc = getattr(mat, "main_color", None) if mat else None
                if mc is not None:
                    fc = np.array([mc[0]/255, mc[1]/255, mc[2]/255])
                all_v.append(v)
                all_f.append(geom.faces + off)
                all_fc.extend([fc] * len(geom.faces))
                off += len(v)
            if not all_v:
                return None, None
            mesh = trimesh.Trimesh(vertices=np.vstack(all_v), faces=np.vstack(all_f), process=False)
            if vis_rpy and not np.allclose(vis_rpy, [0, 0, 0]):
                T = np.eye(4); T[:3, :3] = _rpy2m(vis_rpy)
                mesh.apply_transform(T)
            return mesh, np.array(all_fc)

        # Build USDC
        out_path = output_dir / f"{env_id}_{traj_key}.usdc"
        if out_path.exists():
            out_path.unlink()

        stage = Usd.Stage.CreateNew(str(out_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(num_frames - 1)
        stage.SetFramesPerSecond(24.0)
        stage.SetTimeCodesPerSecond(24.0)

        # Robot links
        for ln in link_names:
            if ln not in link_mesh_path:
                continue
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in ln)
            pp = f"/Scene/Robot/{safe}"
            xf = UsdGeom.Xform.Define(stage, pp)

            mesh, fc = _load_glb(link_mesh_path[ln], link_vis_rpy.get(ln))
            if mesh is None:
                continue

            um = UsdGeom.Mesh.Define(stage, f"{pp}/Geometry")
            um.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh.vertices.tolist()]))
            um.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(mesh.faces)))
            um.GetFaceVertexIndicesAttr().Set(Vt.IntArray(mesh.faces.flatten().tolist()))
            um.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*mesh.bounds[0]), Gf.Vec3f(*mesh.bounds[1])]))
            if fc is not None and len(fc) == len(mesh.faces):
                pv = um.GetDisplayColorPrimvar()
                pv.Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in fc.tolist()]))
                pv.SetInterpolation(UsdGeom.Tokens.uniform)

            pk, qk = f"pos_{ln}", f"quat_{ln}"
            if pk not in data or qk not in data:
                continue
            pos, qua = data[pk], data[qk]
            op = xf.AddTransformOp()
            for fi in range(num_frames):
                wT = np.eye(4)
                wT[:3, 3] = pos[fi]
                wT[:3, :3] = _q2m(qua[fi])
                op.Set(_n2g(wT), Usd.TimeCode(fi))

        # Cube (from env_states in the original H5)
        cube_key = "cube_pos" if "cube_pos" in data else None
        if cube_key is None:
            # Try to get from original H5
            pass

        # Table
        table_mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
        txf = UsdGeom.Xform.Define(stage, "/Scene/Table")
        txf.AddTransformOp().Set(_n2g(np.diag([1.0, 1, 1, 1])))
        ut = UsdGeom.Mesh.Define(stage, "/Scene/Table/Geometry")
        ut.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in table_mesh.vertices.tolist()]))
        ut.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(table_mesh.faces)))
        ut.GetFaceVertexIndicesAttr().Set(Vt.IntArray(table_mesh.faces.flatten().tolist()))
        pv = ut.GetDisplayColorPrimvar()
        pv.Set(Vt.Vec3fArray([Gf.Vec3f(0.76, 0.60, 0.42)]))
        pv.SetInterpolation(UsdGeom.Tokens.constant)

        stage.GetRootLayer().Save()
        logger.info("Generated animated USDC: %s (%d frames)", out_path, num_frames)


def _safe_serialize(obj: Any) -> Any:
    """Best-effort serialization of HDF5-derived values to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj
