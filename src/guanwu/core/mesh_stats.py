"""Mesh statistics computation module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from guanwu.utils.mesh import compute_mesh_stats
from guanwu.storage.canonical_store import CanonicalStore

logger = logging.getLogger("guanwu")


def compute_and_write_mesh_stats(
    canonical_root: str,
    dataset_id: str,
    entity_type: str,
    entity_uid: str,
    mesh_path: str | Path,
) -> dict[str, Any]:
    """Compute mesh stats and write to canonical store."""
    stats = compute_mesh_stats(mesh_path)
    store = CanonicalStore(canonical_root)
    store.write_mesh_stats(dataset_id, entity_type, entity_uid, stats)
    return stats
