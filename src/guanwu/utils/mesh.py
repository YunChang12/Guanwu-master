"""Mesh utility functions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("guanwu")


def compute_mesh_stats(mesh_path: str | Path) -> dict[str, Any]:
    """Compute mesh QA statistics from a mesh file.

    Returns dict with: num_vertices, num_faces, surface_area, bbox_extent,
    has_normals, has_uv, num_materials, degenerate_face_count,
    non_manifold_edge_count, watertight.
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not installed, returning minimal mesh stats")
        return {"error": "trimesh not installed"}

    path = Path(mesh_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    try:
        loaded = trimesh.load(str(path), force="mesh", process=False)
    except Exception as e:
        logger.warning(f"Failed to load mesh {path}: {e}")
        return {"error": str(e)}

    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            return {"error": "No geometry in scene"}
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        return {"error": f"Unexpected type: {type(loaded)}"}

    stats: dict[str, Any] = {
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "surface_area": float(mesh.area),
        "bbox_extent": mesh.bounding_box.extents.tolist(),
        "has_normals": mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0,
        "has_uv": hasattr(mesh.visual, "uv") and mesh.visual.uv is not None,
        "num_materials": 0,
        "degenerate_face_count": int(trimesh.triangles.degenerate(mesh.triangles).sum()),
        "watertight": bool(mesh.is_watertight),
    }

    # Non-manifold edges
    try:
        edges = mesh.edges_sorted
        from collections import Counter
        edge_counts = Counter(map(tuple, edges.tolist()))
        stats["non_manifold_edge_count"] = sum(1 for c in edge_counts.values() if c > 2)
    except Exception:
        stats["non_manifold_edge_count"] = None

    # Material count
    if hasattr(mesh.visual, "material"):
        stats["num_materials"] = 1
    if isinstance(loaded, trimesh.Scene):
        materials = set()
        for geom in loaded.geometry.values():
            if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                materials.add(id(geom.visual.material))
        stats["num_materials"] = len(materials)

    return stats


def load_mesh_basic_info(mesh_path: str | Path) -> dict:
    """Load basic mesh info without full stats computation."""
    try:
        import trimesh
    except ImportError:
        return {}

    path = Path(mesh_path)
    if not path.exists():
        return {}

    try:
        loaded = trimesh.load(str(path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Trimesh):
            return {
                "num_vertices": len(loaded.vertices),
                "num_faces": len(loaded.faces),
            }
        elif isinstance(loaded, trimesh.Scene):
            total_v = sum(len(g.vertices) for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh))
            total_f = sum(len(g.faces) for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh))
            return {"num_vertices": total_v, "num_faces": total_f}
    except Exception:
        pass
    return {}
