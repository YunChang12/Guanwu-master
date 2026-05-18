"""ScanNet++ adapter for high-fidelity indoor scene reconstructions."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from guanwu.adapters.base import DatasetAdapter, register_adapter
from guanwu.core.ids import make_frame_uid, make_scene_uid, make_sensor_uid
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
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
    SensorRecord,
)

logger = logging.getLogger("guanwu")

DATASET_ID = "scannetpp"
DATASET_NAME = "ScanNet++"
DATASET_VERSION = "1.0"


@register_adapter
class ScanNetPPAdapter(DatasetAdapter):
    """Adapter for ScanNet++ high-fidelity indoor scene reconstructions.

    Expected local directory structure::

        <path>/
          data/
            <scene_id>/
              scans/
                mesh_aligned_0.05.ply
                transform.txt
              dslr/
                colmap/
                nerfstudio/
                  transforms.json
                resized_images/
                  *.JPG
                undistorted_images/
                undistorted_anon_images/
              iphone/
                rgb/
                depth/
          metadata/
            semantic/
          splits/
            nvs_sem_train.txt
            nvs_sem_val.txt
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
            "tracks": False,
            "videos": False,
            "sdk_required": False,
            "license_gated": True,
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
            raise ValueError("ScanNet++ adapter requires source_path (local folder)")

        data_dir = Path(source_path) / "data"
        if not data_dir.is_dir():
            logger.warning("ScanNet++ data/ directory not found at %s", data_dir)
            return []

        items: list[SourceItem] = []
        for entry in sorted(data_dir.iterdir()):
            if not entry.is_dir():
                continue
            scene_id = entry.name

            # Apply optional scene_id filter from context.
            if ctx.scene_id and scene_id != ctx.scene_id:
                continue

            items.append(
                SourceItem(
                    item_id=scene_id,
                    dataset_id=config.dataset_id,
                    item_type="scene",
                    source_path=str(entry),
                    metadata={"scene_id": scene_id},
                )
            )

            if ctx.limit and len(items) >= ctx.limit:
                break

        logger.info(
            "ScanNet++ inventory: found %d scenes under %s", len(items), data_dir
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
            scene_id = ref.item_id

            scene_entry: dict = {
                "scene_id": scene_id,
                "mesh_path": None,
                "transform_path": None,
            }

            # --- scans ---
            mesh_path = scene_dir / "scans" / "mesh_aligned_0.05.ply"
            if mesh_path.exists():
                scene_entry["mesh_path"] = str(mesh_path)

            transform_path = scene_dir / "scans" / "transform.txt"
            if transform_path.exists():
                scene_entry["transform_path"] = str(transform_path)

            bundle.scenes.append(scene_entry)

            # --- DSLR sensors / frames ---
            dslr_transforms = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
            if dslr_transforms.exists():
                self._parse_nerfstudio_transforms(
                    dslr_transforms, scene_id, "dslr", bundle
                )

            # Also discover raw DSLR images even if transforms.json is absent.
            resized_dir = scene_dir / "dslr" / "resized_images"
            if resized_dir.is_dir():
                for img in sorted(resized_dir.glob("*.JPG")):
                    # Record discovered images in frames if not already covered
                    # by transforms.json parsing above.
                    pass  # frames are created from transforms; images noted in scene entry

            # --- iPhone depth ---
            iphone_depth_dir = scene_dir / "iphone" / "depth"
            iphone_rgb_dir = scene_dir / "iphone" / "rgb"
            if iphone_rgb_dir.is_dir() or iphone_depth_dir.is_dir():
                self._parse_iphone_data(
                    scene_dir / "iphone", scene_id, bundle
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
                access_mode=AccessMode.GATED,
                geometry_level_max=GeometryLevel.G4_EXACT_MESH,
                created_at=now,
                tags=["indoor", "reconstruction", "mesh", "scannet++"],
            ),
        )

        for scene_dict in bundle.scenes:
            scene_id: str = scene_dict["scene_id"]
            scene_uid = make_scene_uid(dataset_id, scene_id)
            has_mesh = scene_dict.get("mesh_path") is not None

            norm.scenes.append(
                SceneRecord(
                    scene_uid=scene_uid,
                    dataset_id=dataset_id,
                    source_scene_id=scene_id,
                    scene_name=scene_id,
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

        # Sensors and frames from parsed data.
        for sensor_dict in bundle.sensors:
            scene_id = sensor_dict["scene_id"]
            scene_uid = make_scene_uid(dataset_id, scene_id)
            sensor_name: str = sensor_dict["name"]
            sensor_uid = make_sensor_uid(dataset_id, scene_id, sensor_name)

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

        for frame_dict in bundle.frames:
            scene_id = frame_dict["scene_id"]
            scene_uid = make_scene_uid(dataset_id, scene_id)
            sensor_name = frame_dict["sensor_name"]
            sensor_uid = make_sensor_uid(dataset_id, scene_id, sensor_name)
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

        # License record — ScanNet++ is gated, no auto-acceptance.
        norm.licenses.append(
            LicenseRecord(
                record_scope=RecordScope.DATASET,
                record_id=dataset_id,
                license_name="ScanNet++ Terms of Use",
                license_url="https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf",
                commercial_use_allowed=False,
                redistribution_allowed=False,
                attribution_required=True,
                notes="License gated. Users must accept terms before access.",
            )
        )

        # Provenance record.
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
                            "Parsed ScanNet++ local directory; extracted scene meshes, "
                            "DSLR and iPhone camera data, and depth maps."
                        ),
                    }
                ],
            )
        )

        return norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_nerfstudio_transforms(
        self,
        transforms_path: Path,
        scene_id: str,
        modality: str,
        bundle: ParseBundle,
    ) -> None:
        """Parse a nerfstudio-format transforms.json into sensors and frames."""
        try:
            data = json.loads(transforms_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read transforms.json at %s: %s", transforms_path, exc
            )
            return

        sensor_name = f"{modality}_camera"

        # Extract intrinsics from top-level keys (nerfstudio convention).
        sensor_info: dict = {
            "scene_id": scene_id,
            "name": sensor_name,
            "width": data.get("w"),
            "height": data.get("h"),
            "fx": data.get("fl_x"),
            "fy": data.get("fl_y"),
            "cx": data.get("cx"),
            "cy": data.get("cy"),
        }
        bundle.sensors.append(sensor_info)

        frames_list = data.get("frames", [])
        for idx, frame_data in enumerate(frames_list):
            transform_matrix = frame_data.get("transform_matrix")
            T_flat: list[float] | None = None
            if transform_matrix and isinstance(transform_matrix, list):
                T_flat = [v for row in transform_matrix for v in row]

            file_path = frame_data.get("file_path")
            image_uri: str | None = None
            if file_path:
                # Resolve relative to transforms.json parent.
                img = transforms_path.parent / file_path
                if img.exists():
                    image_uri = str(img)

            bundle.frames.append(
                {
                    "scene_id": scene_id,
                    "sensor_name": sensor_name,
                    "frame_index": idx,
                    "timestamp_ns": idx * 33_333_333,  # ~30 fps placeholder
                    "image_uri": image_uri,
                    "depth_uri": None,
                    "T_world_from_sensor": T_flat,
                }
            )

    def _parse_iphone_data(
        self, iphone_dir: Path, scene_id: str, bundle: ParseBundle
    ) -> None:
        """Discover iPhone RGB and depth data."""
        rgb_dir = iphone_dir / "rgb"
        depth_dir = iphone_dir / "depth"

        sensor_name = "iphone_camera"
        bundle.sensors.append(
            {
                "scene_id": scene_id,
                "name": sensor_name,
            }
        )

        # If both rgb and depth exist, pair them by stem name.
        rgb_files: dict[str, Path] = {}
        if rgb_dir.is_dir():
            for f in sorted(rgb_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rgb_files[f.stem] = f

        depth_files: dict[str, Path] = {}
        if depth_dir.is_dir():
            for f in sorted(depth_dir.iterdir()):
                if f.suffix.lower() in (".png", ".npy", ".npz"):
                    depth_files[f.stem] = f

        all_keys = sorted(set(rgb_files.keys()) | set(depth_files.keys()))
        for idx, key in enumerate(all_keys):
            frame_entry: dict = {
                "scene_id": scene_id,
                "sensor_name": sensor_name,
                "frame_index": idx,
                "timestamp_ns": idx * 33_333_333,
                "image_uri": str(rgb_files[key]) if key in rgb_files else None,
                "depth_uri": str(depth_files[key]) if key in depth_files else None,
                "T_world_from_sensor": None,
            }
            bundle.frames.append(frame_entry)

        if sensor_name == "iphone_camera" and depth_files:
            # Add a separate depth sensor record for clarity.
            bundle.sensors.append(
                {
                    "scene_id": scene_id,
                    "name": "iphone_depth",
                }
            )
