"""ARKitScenes adapter for RGB-D indoor scene understanding."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from guanwu.adapters.base import DatasetAdapter, register_adapter
from guanwu.core.ids import (
    make_frame_uid,
    make_instance_uid,
    make_scene_uid,
    make_sensor_uid,
    make_track_uid,
)
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
    PipelineStage,
    RecordScope,
    SceneKind,
    SensorType,
    SourceType,
)
from guanwu.schemas.records import (
    DatasetRecord,
    FrameRecord,
    InstanceRecord,
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
    SensorRecord,
    TrackStateRecord,
)

logger = logging.getLogger("guanwu")

DATASET_ID = "arkitscenes"
DATASET_NAME = "ARKitScenes"
DATASET_VERSION = "1.0"


@register_adapter
class ARKitScenesAdapter(DatasetAdapter):
    """Adapter for ARKitScenes RGB-D indoor scene understanding dataset.

    Expected local directory structure::

        <path>/
          3dod/
            Training/ or Validation/
              <video_id>/
                <video_id>_3dod_mesh.ply
                <video_id>_3dod_annotation.json
                <video_id>.mov or lowres_wide/
                lowres_depth/
                lowres_wide_intrinsics/
                wide/
                wide_intrinsics/
    """

    name: str = DATASET_ID
    version: str = "0.1.0"

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    def capabilities(self) -> dict[str, bool]:
        return {
            "scene_mesh": True,
            "camera": True,
            "depth": True,
            "object_mesh": False,
            "articulation": False,
            "deformable_mesh": False,
            "lidar": False,
            "tracks": True,
            "videos": True,
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
        source_path = config.source_path
        if source_path is None:
            raise ValueError("ARKitScenes adapter requires source_path (local folder)")

        root = Path(source_path)
        three_dod = root / "3dod"
        if not three_dod.is_dir():
            logger.warning("ARKitScenes 3dod/ directory not found at %s", three_dod)
            return []

        items: list[SourceItem] = []
        for split_name in ("Training", "Validation"):
            split_dir = three_dod / split_name
            if not split_dir.is_dir():
                continue

            # Apply optional split filter.
            if config.splits and split_name.lower() not in [
                s.lower() for s in config.splits
            ]:
                continue

            for entry in sorted(split_dir.iterdir()):
                if not entry.is_dir():
                    continue
                video_id = entry.name

                # Apply optional scene_id filter from context.
                if ctx.scene_id and video_id != ctx.scene_id:
                    continue

                items.append(
                    SourceItem(
                        item_id=video_id,
                        dataset_id=config.dataset_id,
                        item_type="scene",
                        source_path=str(entry),
                        metadata={
                            "video_id": video_id,
                            "split": split_name.lower(),
                        },
                    )
                )

                if ctx.limit and len(items) >= ctx.limit:
                    break

            if ctx.limit and len(items) >= ctx.limit:
                break

        logger.info(
            "ARKitScenes inventory: found %d scenes under %s",
            len(items),
            three_dod,
        )
        return items

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------
    def fetch(
        self, items: list[SourceItem], ctx: JobContext
    ) -> list[RawRef]:
        raw_refs: list[RawRef] = []
        for item in items:
            src = item.source_path
            if src is None:
                logger.warning("Skipping item %s: no source_path", item.item_id)
                continue

            dest = Path(ctx.raw_root) / DATASET_ID / item.item_id
            if ctx.dry_run:
                logger.info("[dry-run] Would symlink %s -> %s", dest, src)
                raw_refs.append(
                    RawRef(item_id=item.item_id, raw_path=str(dest))
                )
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() or dest.is_symlink():
                logger.debug("Raw path already exists: %s", dest)
            else:
                os.symlink(os.path.abspath(src), str(dest))
                logger.info("Symlinked %s -> %s", dest, src)

            raw_refs.append(
                RawRef(item_id=item.item_id, raw_path=str(dest))
            )

        return raw_refs

    # ------------------------------------------------------------------
    # parse_raw
    # ------------------------------------------------------------------
    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        bundle = ParseBundle(dataset_id=DATASET_ID)
        bundle.raw_refs = list(raw_refs)

        for ref in raw_refs:
            scene_dir = Path(ref.raw_path)
            video_id = ref.item_id

            scene_entry: dict = {
                "video_id": video_id,
                "mesh_path": None,
                "annotation_path": None,
                "has_mesh": False,
            }

            # --- mesh ---
            mesh_path = scene_dir / f"{video_id}_3dod_mesh.ply"
            if mesh_path.exists():
                scene_entry["mesh_path"] = str(mesh_path)
                scene_entry["has_mesh"] = True

            # --- annotations ---
            annotation_path = scene_dir / f"{video_id}_3dod_annotation.json"
            if annotation_path.exists():
                scene_entry["annotation_path"] = str(annotation_path)
                self._parse_annotations(annotation_path, video_id, bundle)

            bundle.scenes.append(scene_entry)

            # --- sensors: lowres wide camera ---
            lowres_wide_dir = scene_dir / "lowres_wide"
            lowres_depth_dir = scene_dir / "lowres_depth"
            lowres_intrinsics_dir = scene_dir / "lowres_wide_intrinsics"

            if lowres_wide_dir.is_dir() or lowres_depth_dir.is_dir():
                self._parse_lowres_data(
                    video_id,
                    lowres_wide_dir,
                    lowres_depth_dir,
                    lowres_intrinsics_dir,
                    bundle,
                )

            # --- sensors: wide (high-res) camera ---
            wide_dir = scene_dir / "wide"
            wide_intrinsics_dir = scene_dir / "wide_intrinsics"

            if wide_dir.is_dir():
                self._parse_wide_data(
                    video_id, wide_dir, wide_intrinsics_dir, bundle
                )

        return bundle

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------
    def normalize(
        self, bundle: ParseBundle, ctx: JobContext
    ) -> NormalizeBundle:
        now = datetime.now(tz=timezone.utc)
        dataset_id = bundle.dataset_id

        norm = NormalizeBundle(
            dataset_id=dataset_id,
            dataset_record=DatasetRecord(
                dataset_id=dataset_id,
                dataset_name=DATASET_NAME,
                version=DATASET_VERSION,
                source_type=SourceType.LOCAL_FOLDER,
                access_mode=AccessMode.PUBLIC,
                geometry_level_max=GeometryLevel.G4_EXACT_MESH,
                created_at=now,
                tags=["indoor", "rgb-d", "3d-detection", "arkitscenes"],
            ),
        )

        # --- scenes ---
        for scene_dict in bundle.scenes:
            video_id: str = scene_dict["video_id"]
            scene_uid = make_scene_uid(dataset_id, video_id)
            has_mesh: bool = scene_dict.get("has_mesh", False)

            geometry_level = (
                GeometryLevel.G4_EXACT_MESH
                if has_mesh
                else GeometryLevel.G2_POINT_OBS
            )

            norm.scenes.append(
                SceneRecord(
                    scene_uid=scene_uid,
                    dataset_id=dataset_id,
                    source_scene_id=video_id,
                    scene_name=video_id,
                    scene_kind=SceneKind.INDOOR_STATIC,
                    geometry_level=geometry_level,
                    has_static_scene_mesh=has_mesh,
                    has_dynamic_objects=True,
                    has_humans=False,
                    has_articulation=False,
                )
            )

        # --- sensors ---
        for sensor_dict in bundle.sensors:
            video_id = sensor_dict["video_id"]
            scene_uid = make_scene_uid(dataset_id, video_id)
            sensor_name: str = sensor_dict["name"]
            sensor_uid = make_sensor_uid(dataset_id, video_id, sensor_name)

            sensor_type = SensorType.CAMERA
            if "depth" in sensor_name.lower():
                sensor_type = SensorType.DEPTH_CAMERA

            norm.sensors.append(
                SensorRecord(
                    sensor_uid=sensor_uid,
                    scene_uid=scene_uid,
                    sensor_type=sensor_type,
                    name=sensor_name,
                    width=sensor_dict.get("width"),
                    height=sensor_dict.get("height"),
                    fx=sensor_dict.get("fx"),
                    fy=sensor_dict.get("fy"),
                    cx=sensor_dict.get("cx"),
                    cy=sensor_dict.get("cy"),
                )
            )

        # --- frames ---
        for frame_dict in bundle.frames:
            video_id = frame_dict["video_id"]
            scene_uid = make_scene_uid(dataset_id, video_id)
            sensor_name = frame_dict["sensor_name"]
            sensor_uid = make_sensor_uid(dataset_id, video_id, sensor_name)
            frame_idx = str(frame_dict.get("frame_index", 0))
            frame_uid = make_frame_uid(dataset_id, sensor_uid, frame_idx)

            norm.frames.append(
                FrameRecord(
                    frame_uid=frame_uid,
                    sensor_uid=sensor_uid,
                    scene_uid=scene_uid,
                    timestamp_ns=frame_dict.get("timestamp_ns", 0),
                    image_uri=frame_dict.get("image_uri"),
                    depth_uri=frame_dict.get("depth_uri"),
                    T_world_from_sensor=frame_dict.get("T_world_from_sensor"),
                )
            )

        # --- instances and tracks from annotations ---
        for ann in bundle.annotations:
            video_id = ann["video_id"]
            scene_uid = make_scene_uid(dataset_id, video_id)
            obj_id = str(ann["object_id"])
            instance_uid = make_instance_uid(dataset_id, video_id, obj_id)

            # Determine geometry level from annotation data.
            has_bbox = ann.get("center") is not None
            instance_geom = GeometryLevel.G1_BBOX if has_bbox else GeometryLevel.G0_NONE

            norm.instances.append(
                InstanceRecord(
                    instance_uid=instance_uid,
                    scene_uid=scene_uid,
                    category=ann.get("label"),
                    instance_name=f"{ann.get('label', 'unknown')}_{obj_id}",
                    is_static=True,
                    is_articulated=False,
                    is_human=False,
                    geometry_level=instance_geom,
                )
            )

            # Generate a track state if bbox info is available.
            center = ann.get("center")
            size = ann.get("size")
            if center and size:
                track_uid = make_track_uid(dataset_id, instance_uid)
                norm.track_states.append(
                    TrackStateRecord(
                        track_uid=track_uid,
                        instance_uid=instance_uid,
                        timestamp_ns=0,
                        bbox3d_center_xyz=(
                            float(center[0]),
                            float(center[1]),
                            float(center[2]),
                        ),
                        bbox3d_size_xyz=(
                            float(size[0]),
                            float(size[1]),
                            float(size[2]),
                        ),
                    )
                )

        # --- license (CC BY-NC-SA 4.0) ---
        norm.licenses.append(
            LicenseRecord(
                record_scope=RecordScope.DATASET,
                record_id=dataset_id,
                license_name="CC BY-NC-SA 4.0",
                license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
                commercial_use_allowed=False,
                redistribution_allowed=True,
                attribution_required=True,
            )
        )

        # --- provenance ---
        norm.provenance.append(
            ProvenanceRecord(
                record_id=dataset_id,
                dataset_id=dataset_id,
                normalized_by_version=self.version,
                normalized_at=now,
                adapter_name=self.name,
                adapter_version=self.version,
                transform_log=[
                    {
                        "stage": PipelineStage.NORMALIZE.value,
                        "description": (
                            "Parsed ARKitScenes local directory; extracted meshes, "
                            "3D OBB annotations, lowres/wide camera data, and depth maps."
                        ),
                    }
                ],
            )
        )

        return norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_annotations(
        self, annotation_path: Path, video_id: str, bundle: ParseBundle
    ) -> None:
        """Parse 3D oriented bounding box annotations."""
        try:
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read annotation at %s: %s", annotation_path, exc
            )
            return

        # The annotation file may be a list or a dict with a "data" key.
        objects: list[dict] = []
        if isinstance(data, list):
            objects = data
        elif isinstance(data, dict):
            objects = data.get("data", data.get("objects", []))

        for obj in objects:
            ann_entry: dict = {
                "video_id": video_id,
                "object_id": obj.get("uid", obj.get("id", id(obj))),
                "label": obj.get("label", obj.get("category")),
                "center": None,
                "size": None,
                "rotation": None,
            }

            # ARKitScenes uses oriented bounding boxes.
            transform = obj.get("transform")
            dimensions = obj.get("dimensions")

            if transform and isinstance(transform, list) and len(transform) >= 3:
                ann_entry["center"] = transform[:3]
            elif obj.get("position"):
                ann_entry["center"] = obj["position"][:3]

            if dimensions and isinstance(dimensions, list) and len(dimensions) >= 3:
                ann_entry["size"] = dimensions[:3]
            elif obj.get("size"):
                ann_entry["size"] = obj["size"][:3]

            if obj.get("rotation"):
                ann_entry["rotation"] = obj["rotation"]

            bundle.annotations.append(ann_entry)

    def _parse_lowres_data(
        self,
        video_id: str,
        lowres_wide_dir: Path,
        lowres_depth_dir: Path,
        lowres_intrinsics_dir: Path,
        bundle: ParseBundle,
    ) -> None:
        """Parse lowres wide images, depth maps, and per-frame intrinsics."""
        sensor_name = "lowres_wide"
        sensor_info: dict = {"video_id": video_id, "name": sensor_name}

        # Try to read one intrinsics file for default camera params.
        if lowres_intrinsics_dir.is_dir():
            intrinsics = self._read_first_intrinsics(lowres_intrinsics_dir)
            if intrinsics:
                sensor_info.update(intrinsics)

        bundle.sensors.append(sensor_info)

        # Also add a depth sensor entry.
        if lowres_depth_dir.is_dir():
            bundle.sensors.append(
                {"video_id": video_id, "name": "lowres_depth"}
            )

        # Pair images and depth by stem.
        image_files: dict[str, Path] = {}
        if lowres_wide_dir.is_dir():
            for f in sorted(lowres_wide_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_files[f.stem] = f

        depth_files: dict[str, Path] = {}
        if lowres_depth_dir.is_dir():
            for f in sorted(lowres_depth_dir.iterdir()):
                if f.suffix.lower() in (".png", ".npy", ".npz", ".depth"):
                    depth_files[f.stem] = f

        all_keys = sorted(set(image_files.keys()) | set(depth_files.keys()))
        for idx, key in enumerate(all_keys):
            # Try to extract timestamp from filename (ARKitScenes convention:
            # <video_id>_<timestamp>.png or just <timestamp>.png).
            ts_ns = self._filename_to_timestamp_ns(key)

            bundle.frames.append(
                {
                    "video_id": video_id,
                    "sensor_name": sensor_name,
                    "frame_index": idx,
                    "timestamp_ns": ts_ns if ts_ns else idx * 33_333_333,
                    "image_uri": (
                        str(image_files[key]) if key in image_files else None
                    ),
                    "depth_uri": (
                        str(depth_files[key]) if key in depth_files else None
                    ),
                    "T_world_from_sensor": None,
                }
            )

    def _parse_wide_data(
        self,
        video_id: str,
        wide_dir: Path,
        wide_intrinsics_dir: Path,
        bundle: ParseBundle,
    ) -> None:
        """Parse high-res wide images and intrinsics."""
        sensor_name = "wide"
        sensor_info: dict = {"video_id": video_id, "name": sensor_name}

        if wide_intrinsics_dir.is_dir():
            intrinsics = self._read_first_intrinsics(wide_intrinsics_dir)
            if intrinsics:
                sensor_info.update(intrinsics)

        bundle.sensors.append(sensor_info)

        if wide_dir.is_dir():
            for idx, f in enumerate(sorted(wide_dir.iterdir())):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                ts_ns = self._filename_to_timestamp_ns(f.stem)
                bundle.frames.append(
                    {
                        "video_id": video_id,
                        "sensor_name": sensor_name,
                        "frame_index": idx,
                        "timestamp_ns": ts_ns if ts_ns else idx * 33_333_333,
                        "image_uri": str(f),
                        "depth_uri": None,
                        "T_world_from_sensor": None,
                    }
                )

    @staticmethod
    def _read_first_intrinsics(intrinsics_dir: Path) -> dict | None:
        """Read the first intrinsics file to extract camera parameters.

        ARKitScenes stores per-frame intrinsics as files containing a 3x3
        matrix (one row per line, space-separated).
        """
        for f in sorted(intrinsics_dir.iterdir()):
            if not f.is_file():
                continue
            try:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) >= 3:
                    row0 = [float(v) for v in lines[0].split()]
                    row1 = [float(v) for v in lines[1].split()]
                    # row0: [fx, 0, cx], row1: [0, fy, cy]
                    if len(row0) >= 3 and len(row1) >= 3:
                        return {
                            "fx": row0[0],
                            "cx": row0[2],
                            "fy": row1[1],
                            "cy": row1[2],
                        }
            except (ValueError, OSError):
                continue
        return None

    @staticmethod
    def _filename_to_timestamp_ns(stem: str) -> int | None:
        """Try to parse a timestamp from an ARKitScenes filename stem.

        Filenames typically look like ``<video_id>_<timestamp>`` or just a
        float timestamp.  Returns nanoseconds or None.
        """
        # Try the part after the last underscore.
        parts = stem.rsplit("_", 1)
        candidate = parts[-1] if len(parts) > 1 else stem
        try:
            ts_sec = float(candidate)
            return int(ts_sec * 1_000_000_000)
        except ValueError:
            return None
