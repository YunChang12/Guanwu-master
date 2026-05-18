"""GLB export utilities."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("guanwu")


def export_mesh_to_glb(
    mesh_path: str | Path,
    output_path: str | Path,
) -> bool:
    """Convert a mesh file to GLB format for preview."""
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, cannot export GLB")
        return False

    try:
        mesh = trimesh.load(str(mesh_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path), file_type="glb")
        logger.debug(f"Exported GLB: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to export GLB from {mesh_path}: {e}")
        return False
