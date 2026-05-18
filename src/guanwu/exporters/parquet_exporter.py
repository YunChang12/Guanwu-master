"""Parquet export utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("guanwu")


def export_records_to_parquet(
    records: list[dict],
    output_path: str | Path,
    schema: pa.Schema | None = None,
) -> int:
    """Export a list of record dicts to a Parquet file.

    Returns the number of records written.
    """
    if not records:
        return 0
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path)
    logger.debug(f"Exported {len(records)} records to {path}")
    return len(records)
