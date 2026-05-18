"""OpenUSD export utilities.

Converts mesh files and articulated assets to USDC format using the OpenUSD
Python bindings (``pxr``).  All functions gracefully handle a missing ``pxr``
and return ``False`` / ``None`` with a logged warning.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("guanwu")

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    _HAS_USD = True
except ImportError:
    _HAS_USD = False

try:
    import trimesh
    import numpy as np
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


def _check_deps(what: str = "USD export") -> bool:
    if not _HAS_USD:
        logger.warning("pxr (usd-core) not installed; %s skipped", what)
        return False
    if not _HAS_TRIMESH:
        logger.warning("trimesh not installed; %s skipped", what)
        return False
    return True


# ── Single mesh → USDC ─────────────────────────────────────────────────


def mesh_to_usdc(mesh_path: str | Path, output_path: str | Path) -> bool:
    """Convert a mesh file (OBJ / PLY / GLB / STL) to a binary USDC file.

    Returns True on success.
    """
    if not _check_deps("mesh_to_usdc"):
        return False

    mesh_path = Path(mesh_path)
    output_path = Path(output_path)

    if not mesh_path.exists():
        logger.warning("Mesh file not found: %s", mesh_path)
        return False

    try:
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    except Exception:
        # Might be a scene with multiple geometries
        try:
            loaded = trimesh.load(str(mesh_path), process=False)
        except Exception as e:
            logger.warning("Cannot load mesh %s: %s", mesh_path, e)
            return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    if isinstance(loaded, trimesh.Scene):
        root = UsdGeom.Xform.Define(stage, "/Root")
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                safe_name = _sanitize_name(name)
                _add_trimesh_to_stage(stage, f"/Root/{safe_name}", geom)
    elif isinstance(loaded, trimesh.Trimesh):
        _add_trimesh_to_stage(stage, "/Root/Mesh", loaded)
    else:
        logger.warning("Unsupported mesh type %s from %s", type(loaded), mesh_path)
        return False

    stage.GetRootLayer().Save()
    logger.debug("Wrote USDC: %s", output_path)
    return True


# ── Articulated asset → USDC ───────────────────────────────────────────


def articulated_asset_to_usdc(
    link_meshes: dict[str, str | Path],
    joints: list[dict[str, Any]],
    output_path: str | Path,
    root_link: str | None = None,
) -> bool:
    """Write an articulated asset as a USDC scene graph.

    Args:
        link_meshes: mapping ``{link_name: mesh_file_path}``
        joints: list of dicts, each with keys:
            name, type, parent_link, child_link, axis (xyz list), limits (dict with lower/upper)
        output_path: destination ``.usdc`` file
        root_link: name of the root link (auto-detected if None)

    Returns True on success.
    """
    if not _check_deps("articulated_asset_to_usdc"):
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Build parent map to find root
    child_to_parent: dict[str, str] = {}
    for j in joints:
        child_to_parent[j["child_link"]] = j["parent_link"]

    if root_link is None:
        all_links = set(link_meshes.keys())
        child_links = set(child_to_parent.keys())
        roots = all_links - child_links
        root_link = sorted(roots)[0] if roots else sorted(link_meshes.keys())[0]

    # Create link prims with mesh geometry
    link_paths: dict[str, str] = {}
    for link_name, mesh_file in link_meshes.items():
        safe_name = _sanitize_name(link_name)
        prim_path = f"/Asset/{safe_name}"
        link_paths[link_name] = prim_path

        xform = UsdGeom.Xform.Define(stage, prim_path)

        mesh_file = Path(mesh_file)
        if mesh_file.exists():
            try:
                loaded = trimesh.load(str(mesh_file), force="mesh", process=False)
                if isinstance(loaded, trimesh.Trimesh):
                    _add_trimesh_to_stage(stage, f"{prim_path}/Geometry", loaded)
            except Exception as e:
                logger.debug("Cannot load link mesh %s: %s", mesh_file, e)

    # Create joint definitions as custom attributes on the child xform
    for j in joints:
        child_name = j.get("child_link", "")
        if child_name not in link_paths:
            continue
        child_prim_path = link_paths[child_name]
        child_prim = stage.GetPrimAtPath(child_prim_path)
        if not child_prim:
            continue

        joint_type = j.get("type", "fixed")
        joint_name = j.get("name", "")
        axis = j.get("axis", [0, 0, 1])
        limits = j.get("limits", {})

        child_prim.SetCustomDataByKey("joint:name", joint_name)
        child_prim.SetCustomDataByKey("joint:type", joint_type)
        child_prim.SetCustomDataByKey("joint:parent", j.get("parent_link", ""))
        child_prim.SetCustomDataByKey("joint:axis", Gf.Vec3f(*[float(a) for a in axis]))
        if limits:
            child_prim.SetCustomDataByKey("joint:lower", float(limits.get("lower", 0)))
            child_prim.SetCustomDataByKey("joint:upper", float(limits.get("upper", 0)))

    stage.GetRootLayer().Save()
    logger.debug("Wrote articulated USDC: %s", output_path)
    return True


# ── Scene → USDC ───────────────────────────────────────────────────────


def scene_to_usdc(
    mesh_paths: list[str | Path],
    output_path: str | Path,
    transforms: list[list[float]] | None = None,
    names: list[str] | None = None,
) -> bool:
    """Write a scene USDC with one or more meshes.

    Args:
        mesh_paths: list of mesh file paths
        output_path: destination ``.usdc``
        transforms: optional per-mesh 4x4 transforms (16-element row-major lists)
        names: optional per-mesh names
    """
    if not _check_deps("scene_to_usdc"):
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    for i, mesh_path in enumerate(mesh_paths):
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            continue

        name = _sanitize_name(names[i]) if names and i < len(names) else f"Mesh_{i}"
        prim_path = f"/Scene/{name}"

        xform = UsdGeom.Xform.Define(stage, prim_path)

        # Apply transform if provided
        if transforms and i < len(transforms) and transforms[i]:
            mat = np.array(transforms[i]).reshape(4, 4)
            gf_mat = _np_to_gf_matrix(mat)
            xform.AddTransformOp().Set(gf_mat)

        try:
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Trimesh):
                _add_trimesh_to_stage(stage, f"{prim_path}/Geometry", loaded)
        except Exception as e:
            logger.debug("Cannot load scene mesh %s: %s", mesh_path, e)

    stage.GetRootLayer().Save()
    logger.debug("Wrote scene USDC: %s", output_path)
    return True


# ── Animated articulated asset → USDC ──────────────────────────────────


def animated_asset_to_usdc(
    link_meshes: dict[str, str | Path],
    joints: list[dict[str, Any]],
    joint_trajectories: dict[str, list[float]],
    output_path: str | Path,
    fps: float = 24.0,
    root_link: str | None = None,
) -> bool:
    """Write an articulated asset with joint-state animation as a single USDC.

    Each link's mesh is written once as static geometry.  Joint motion is
    encoded as ``timeSamples`` on the child-link ``Xform`` so that a USD
    viewer can play back the animation.

    Args:
        link_meshes: ``{link_name: mesh_file_path}``
        joints: list of dicts with keys *name*, *type* (``revolute`` /
            ``prismatic``), *parent_link*, *child_link*, *axis* ``[x,y,z]``,
            *limits* ``{lower, upper}``
        joint_trajectories: ``{joint_name: [value_frame0, value_frame1, ...]}``
            For revolute joints the values are radians; for prismatic, meters.
        output_path: destination ``.usdc``
        fps: frames per second for the time codes
        root_link: auto-detected if *None*

    Returns:
        ``True`` on success.
    """
    if not _check_deps("animated_asset_to_usdc"):
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Determine frame range from trajectories
    num_frames = max((len(v) for v in joint_trajectories.values()), default=1)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames - 1)
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)

    # Build parent map
    child_to_parent: dict[str, str] = {}
    joint_by_child: dict[str, dict] = {}
    for j in joints:
        child_to_parent[j["child_link"]] = j["parent_link"]
        joint_by_child[j["child_link"]] = j

    if root_link is None:
        all_links = set(link_meshes.keys())
        child_links = set(child_to_parent.keys())
        roots = all_links - child_links
        root_link = sorted(roots)[0] if roots else sorted(link_meshes.keys())[0]

    # Create link prims with mesh + animated transforms
    link_paths: dict[str, str] = {}

    for link_name, mesh_file in link_meshes.items():
        safe_name = _sanitize_name(link_name)
        prim_path = f"/Asset/{safe_name}"
        link_paths[link_name] = prim_path

        xform = UsdGeom.Xform.Define(stage, prim_path)

        # Load and write static mesh geometry
        mesh_file = Path(mesh_file)
        if mesh_file.exists():
            try:
                loaded = trimesh.load(str(mesh_file), force="mesh", process=False)
                if isinstance(loaded, trimesh.Trimesh):
                    _add_trimesh_to_stage(stage, f"{prim_path}/Geometry", loaded)
            except Exception as e:
                logger.debug("Cannot load link mesh %s: %s", mesh_file, e)

        # If this link has a joint, write animated transform timeSamples
        jinfo = joint_by_child.get(link_name)
        if jinfo is None:
            continue

        joint_name = jinfo["name"]
        traj = joint_trajectories.get(joint_name)
        if traj is None:
            continue

        joint_type = jinfo.get("type", "fixed")
        axis = np.array(jinfo.get("axis", [0, 0, 1]), dtype=np.float64)
        origin = jinfo.get("origin", [0, 0, 0])
        if isinstance(origin, str):
            origin = [float(x) for x in origin.split()]

        xform_op = xform.AddTransformOp()

        for frame_idx, value in enumerate(traj):
            mat = np.eye(4, dtype=np.float64)

            # Base translation from joint origin
            if origin:
                mat[:3, 3] = np.array(origin[:3], dtype=np.float64)

            if joint_type == "revolute":
                mat[:3, :3] = _rotation_matrix(axis, float(value))
            elif joint_type == "prismatic":
                mat[:3, 3] += axis * float(value)

            gf_mat = _np_to_gf_matrix(mat)
            xform_op.Set(gf_mat, Usd.TimeCode(frame_idx))

        # Store joint metadata
        prim = stage.GetPrimAtPath(prim_path)
        prim.SetCustomDataByKey("joint:name", joint_name)
        prim.SetCustomDataByKey("joint:type", joint_type)
        prim.SetCustomDataByKey("joint:parent", jinfo.get("parent_link", ""))
        prim.SetCustomDataByKey("joint:axis", Gf.Vec3f(*axis.tolist()))

    stage.GetRootLayer().Save()
    logger.info("Wrote animated USDC (%d frames): %s", num_frames, output_path)
    return True


def _np_to_gf_matrix(mat: np.ndarray) -> Any:
    """Convert a 4x4 numpy matrix (column-vector convention) to Gf.Matrix4d
    (row-vector convention).  USD uses row-vectors so translation lives in
    row 3 of Gf.Matrix4d whereas numpy / OpenGL puts it in column 3."""
    gf = Gf.Matrix4d()
    m = mat.T  # transpose: col-vector → row-vector
    for r in range(4):
        gf.SetRow(r, Gf.Vec4d(*m[r].tolist()))
    return gf


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation: 3x3 rotation matrix around *axis* by *angle* rad."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


# ── Internal helpers ───────────────────────────────────────────────────


def _sanitize_name(name: str) -> str:
    """Make a name safe for use as a USD prim path token."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "N" + safe
    return safe or "Unnamed"


def _add_trimesh_to_stage(
    stage: Any,  # Usd.Stage
    prim_path: str,
    mesh: Any,  # trimesh.Trimesh
) -> None:
    """Add a trimesh.Trimesh as a UsdGeom.Mesh at the given prim path."""
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

    vertices = mesh.vertices.tolist()
    faces = mesh.faces
    face_count = len(faces)

    # Points
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices]))

    # Face vertex counts (all triangles)
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * face_count))

    # Face vertex indices
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))

    # Normals if available
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        normals = mesh.vertex_normals.tolist()
        usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Extent (bounding box)
    bounds = mesh.bounds
    usd_mesh.GetExtentAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*bounds[0]), Gf.Vec3f(*bounds[1])])
    )
