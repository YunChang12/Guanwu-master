from __future__ import annotations

from guanwu.adapters.base import DatasetAdapter
from guanwu.schemas.bundles import EmitReport, JobContext, NormalizeBundle


def materialize_bundle(
    adapter: DatasetAdapter,
    bundle: NormalizeBundle,
    ctx: JobContext,
) -> EmitReport:
    return adapter.emit(bundle, ctx)
