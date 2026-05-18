from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationBundle:
    sync_report: dict
    export: dict | None = None
