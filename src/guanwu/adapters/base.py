"""Adapter base class and registry for dataset adapters."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from guanwu.schemas.bundles import (
    AdapterConfig,
    EmitReport,
    JobContext,
    NormalizeBundle,
    ParseBundle,
    RawRef,
    SourceItem,
    ValidationReport,
)

logger = logging.getLogger("guanwu")


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Each adapter must implement the full pipeline:
    inventory -> fetch -> parse_raw -> normalize -> validate -> emit
    """

    name: str = ""
    version: str = "0.1.0"

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Declare adapter capabilities.

        Must return a dict with at least these keys:
            scene_mesh, object_mesh, articulation, deformable_mesh,
            camera, depth, lidar, tracks, videos,
            sdk_required, license_gated, supports_local_ingest
        """
        ...

    @abstractmethod
    def inventory(self, config: AdapterConfig, ctx: JobContext) -> list[SourceItem]:
        """Scan data source and return list of items to process.

        Must not modify canonical store. Must support dry-run.
        """
        ...

    @abstractmethod
    def fetch(
        self, items: list[SourceItem], ctx: JobContext
    ) -> list[RawRef]:
        """Fetch/link raw data into the raw store.

        For local sources, this typically creates symlinks.
        Must not silently overwrite existing raw files.
        Must generate checksums for downloaded files.
        """
        ...

    @abstractmethod
    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        """Parse raw directory structure into a standard intermediate bundle.

        Must discover: scenes, assets, sensors, frames, annotations,
        articulation metadata, licenses.
        """
        ...

    @abstractmethod
    def normalize(
        self, bundle: ParseBundle, ctx: JobContext
    ) -> NormalizeBundle:
        """Normalize parsed data to canonical schema.

        Must: unify units to meters, coordinate system to Z-up right-hand,
        generate stable IDs, split scene/asset/episode/state layers,
        map categories to canonical taxonomy, generate provenance.
        """
        ...

    def validate(
        self, bundle: NormalizeBundle, ctx: JobContext
    ) -> ValidationReport:
        """Validate normalized data.

        Default implementation runs basic schema checks.
        Adapters can override to add dataset-specific validation.
        """
        from guanwu.core.validation import validate_bundle

        return validate_bundle(bundle, ctx)

    def emit(
        self, bundle: NormalizeBundle, ctx: JobContext
    ) -> EmitReport:
        """Write canonical outputs to the canonical store.

        Default implementation writes JSON/Parquet files per the spec directory layout.
        """
        from guanwu.storage.canonical_store import CanonicalStore

        store = CanonicalStore(ctx.canonical_root)
        report = EmitReport(dataset_id=bundle.dataset_id)

        if bundle.dataset_record:
            store.write_dataset_record(
                bundle.dataset_id, bundle.dataset_record.model_dump()
            )

        for scene in bundle.scenes:
            d = scene.model_dump()
            store.write_scene_meta(bundle.dataset_id, scene.scene_uid, d)
            report.scenes_emitted += 1

        for episode in bundle.episodes:
            d = episode.model_dump()
            store.write_episode_meta(bundle.dataset_id, episode.episode_uid, d)
            report.episodes_emitted += 1

        scene_sensors: dict[str, list[dict]] = {}
        for sensor in bundle.sensors:
            key = sensor.scene_uid or "__global__"
            scene_sensors.setdefault(key, []).append(sensor.model_dump())
        for scene_uid, sensors in scene_sensors.items():
            if scene_uid != "__global__":
                store.write_scene_sensors(bundle.dataset_id, scene_uid, sensors)

        episode_frames: dict[str, list[dict]] = {}
        scene_frames: dict[str, list[dict]] = {}
        for frame in bundle.frames:
            key = frame.scene_uid or "__global__"
            scene_frames.setdefault(key, []).append(frame.model_dump())
            if frame.episode_uid:
                episode_frames.setdefault(frame.episode_uid, []).append(frame.model_dump())
        for scene_uid, frames in scene_frames.items():
            if scene_uid != "__global__":
                store.write_scene_frames(bundle.dataset_id, scene_uid, frames)
        for episode_uid, frames in episode_frames.items():
            store.write_episode_sensor_frames(bundle.dataset_id, episode_uid, frames)

        scene_instances: dict[str, list[dict]] = {}
        for inst in bundle.instances:
            key = inst.scene_uid or "__global__"
            scene_instances.setdefault(key, []).append(inst.model_dump())
        for scene_uid, instances in scene_instances.items():
            if scene_uid != "__global__":
                store.write_scene_instances(
                    bundle.dataset_id, scene_uid, instances
                )

        scene_tracks: dict[str, list[dict]] = {}
        episode_tracks: dict[str, list[dict]] = {}
        for ts in bundle.track_states:
            for inst in bundle.instances:
                if inst.instance_uid == ts.instance_uid:
                    key = inst.scene_uid or "__global__"
                    scene_tracks.setdefault(key, []).append(ts.model_dump())
                    if inst.episode_uid:
                        episode_tracks.setdefault(inst.episode_uid, []).append(ts.model_dump())
                    break
        for scene_uid, tracks in scene_tracks.items():
            if scene_uid != "__global__":
                store.write_scene_tracks(bundle.dataset_id, scene_uid, tracks)
        for episode_uid, tracks in episode_tracks.items():
            store.write_episode_states(bundle.dataset_id, episode_uid, tracks)

        instance_by_uid = {
            inst.instance_uid: inst
            for inst in bundle.instances
        }
        scene_articulation: dict[str, list[dict]] = {}
        for state in bundle.articulation_states:
            inst = instance_by_uid.get(state.instance_uid)
            if inst is None or not inst.scene_uid:
                continue
            scene_articulation.setdefault(inst.scene_uid, []).append(
                state.model_dump()
            )
        for scene_uid, records in scene_articulation.items():
            store.write_scene_articulation(bundle.dataset_id, scene_uid, records)

        for asset in bundle.assets:
            d = asset.model_dump()
            store.write_asset_meta(bundle.dataset_id, asset.asset_uid, d)
            report.assets_emitted += 1

            # Generate USDC if mesh data is available
            if asset.mesh_uri:
                self._try_emit_asset_usdc(
                    store, bundle.dataset_id, asset, ctx
                )

        for lic in bundle.licenses:
            store.write_licenses(
                bundle.dataset_id,
                lic.record_scope,
                lic.record_id,
                [lic.model_dump()],
            )

        for prov in bundle.provenance:
            store.write_provenance(
                bundle.dataset_id,
                "dataset",
                prov.record_id,
                prov.model_dump(),
            )

        return report

    # ── USD helpers (best-effort, no failure if pxr missing) ────────

    def _try_emit_asset_usdc(
        self,
        store: Any,
        dataset_id: str,
        asset: Any,
        ctx: Any,
    ) -> None:
        """Best-effort USDC generation for an asset."""
        try:
            from guanwu.exporters.usd import mesh_to_usdc
        except ImportError:
            return

        from pathlib import Path

        mesh_path = Path(asset.mesh_uri)
        if not mesh_path.exists():
            return

        usdc_path = store.asset_dir(dataset_id, asset.asset_uid) / "asset.usdc"
        mesh_to_usdc(mesh_path, usdc_path)


_ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Decorator to register an adapter class."""
    _ADAPTER_REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> DatasetAdapter:
    """Get an adapter instance by dataset name."""
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
        raise KeyError(
            f"Unknown adapter: {name}. Available: {available}"
        )
    return _ADAPTER_REGISTRY[name]()


def list_adapters() -> dict[str, type[DatasetAdapter]]:
    """Return all registered adapters."""
    return dict(_ADAPTER_REGISTRY)
