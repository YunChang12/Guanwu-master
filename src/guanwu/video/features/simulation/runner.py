from __future__ import annotations

from guanwu.video.features.simulation.output import SimulationBundle
from guanwu.video.core.logger import get_logger

logger = get_logger(__name__)


class SimulationPipeline:
    def __init__(self, isaac_sync_agent, pit2isaac_exporter) -> None:
        self.isaac_sync_agent = isaac_sync_agent
        self.pit2isaac_exporter = pit2isaac_exporter

    def sync(self, active_objects) -> SimulationBundle:
        logger.debug(f"[SimulationPipeline] Syncing {len(active_objects)} active objects to Isaac")
        report = self.isaac_sync_agent.sync(active_objects).model_dump()
        logger.debug(f"[SimulationPipeline] Sync report: {report}")
        return SimulationBundle(sync_report=report)

    def export(self, objects, relations, pit_snapshot: dict, mode_override: str | None = None) -> dict:
        if mode_override:
            self.pit2isaac_exporter.mode = mode_override
        result = self.pit2isaac_exporter.export(objects=objects, relations=relations, pit_snapshot=pit_snapshot)
        return {
            "mode": result.mode,
            "usd_backend": result.usd_backend,
            "usd_path": result.usd_path,
            "physics_priors_json": result.physics_priors_json,
            "asset_mapping_json": result.asset_mapping_json,
            "conversion_report_json": result.conversion_report_json,
            "object_count": result.object_count,
            "relation_count": result.relation_count,
        }
