from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .render import RobotMesh


@dataclass
class VisualSpec:
    link_name: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class ParsedURDF:
    root_link: str
    visuals: list[VisualSpec] = field(default_factory=list)
    joints: list[JointSpec] = field(default_factory=list)


def load_robot_mesh_from_urdf(
    urdf_path: str | Path,
    joint_angles: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> RobotMesh:
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    try:
        return _load_with_yourdfpy(path, joint_angles)
    except ModuleNotFoundError:
        return _load_with_builtin_parser(path, joint_angles)
    except Exception:
        return _load_with_builtin_parser(path, joint_angles)


def _load_with_yourdfpy(
    urdf_path: Path,
    joint_angles: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None,
) -> RobotMesh:
    import trimesh  # type: ignore
    from yourdfpy import URDF  # type: ignore

    robot = URDF.load(str(urdf_path))
    cfg = _joint_mapping_from_model(robot, joint_angles)
    scene = robot.scene
    robot.update_cfg(cfg)
    meshes: list[Any] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry.get(geometry_name)
        if geometry is None:
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No visual mesh geometry found in URDF: {urdf_path}")
    combined = trimesh.util.concatenate(meshes)
    return RobotMesh(vertices=np.asarray(combined.vertices, dtype=np.float64), faces=np.asarray(combined.faces, dtype=np.int32))


def _joint_mapping_from_model(robot: Any, joint_angles: Any) -> dict[str, float]:
    if joint_angles is None:
        return {}
    if isinstance(joint_angles, dict):
        return {str(k): float(v) for k, v in joint_angles.items()}
    names: list[str] = []
    for joint in getattr(robot, "actuated_joints", []) or []:
        names.append(str(getattr(joint, "name", "")))
    values = [float(v) for v in joint_angles]
    return {name: values[index] for index, name in enumerate(names[: len(values)])}


def _load_with_builtin_parser(
    urdf_path: Path,
    joint_angles: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None,
) -> RobotMesh:
    parsed = parse_minimal_urdf(urdf_path)
    q = normalize_joint_angles(parsed.joints, joint_angles)
    link_T = compute_link_transforms(parsed, q)
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    link_names: list[str] = []
    vertex_offset = 0
    for visual in parsed.visuals:
        vertices, faces = load_obj_mesh(visual.mesh_path)
        vertices = vertices * visual.scale.reshape(1, 3)
        T = link_T.get(visual.link_name, np.eye(4)) @ visual.origin
        vertices_B = vertices @ T[:3, :3].T + T[:3, 3].reshape(1, 3)
        vertices_all.append(vertices_B)
        faces_all.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]
        link_names.append(visual.link_name)
    if not vertices_all:
        raise ValueError(f"No visual mesh geometry found in URDF: {urdf_path}")
    return RobotMesh(vertices=np.vstack(vertices_all), faces=np.vstack(faces_all), link_names=tuple(link_names))


def parse_minimal_urdf(urdf_path: Path) -> ParsedURDF:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_names = [str(link.attrib["name"]) for link in root.findall("link") if "name" in link.attrib]
    child_links: set[str] = set()
    visuals: list[VisualSpec] = []
    joints: list[JointSpec] = []
    for link in root.findall("link"):
        link_name = str(link.attrib.get("name", ""))
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None or not mesh.attrib.get("filename"):
                continue
            scale = parse_xyz(mesh.attrib.get("scale"), default=(1.0, 1.0, 1.0))
            origin = parse_origin(visual.find("origin"))
            visuals.append(
                VisualSpec(
                    link_name=link_name,
                    mesh_path=resolve_mesh_path(urdf_path.parent, str(mesh.attrib["filename"])),
                    origin=origin,
                    scale=scale,
                )
            )
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        child_name = str(child.attrib.get("link", ""))
        child_links.add(child_name)
        axis_el = joint.find("axis")
        axis = parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None, default=(1.0, 0.0, 0.0))
        norm = float(np.linalg.norm(axis))
        if norm > 1e-12:
            axis = axis / norm
        joints.append(
            JointSpec(
                name=str(joint.attrib.get("name", "")),
                joint_type=str(joint.attrib.get("type", "fixed")),
                parent=str(parent.attrib.get("link", "")),
                child=child_name,
                origin=parse_origin(joint.find("origin")),
                axis=axis,
            )
        )
    roots = [name for name in link_names if name not in child_links]
    root_link = roots[0] if roots else (link_names[0] if link_names else "")
    return ParsedURDF(root_link=root_link, visuals=visuals, joints=joints)


def compute_link_transforms(parsed: ParsedURDF, joint_angles: dict[str, float]) -> dict[str, np.ndarray]:
    children_by_parent: dict[str, list[JointSpec]] = {}
    for joint in parsed.joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)
    transforms: dict[str, np.ndarray] = {parsed.root_link: np.eye(4)}
    stack = [parsed.root_link]
    while stack:
        parent = stack.pop()
        T_parent = transforms[parent]
        for joint in children_by_parent.get(parent, []):
            q = float(joint_angles.get(joint.name, 0.0))
            transforms[joint.child] = T_parent @ joint.origin @ joint_motion_transform(joint, q)
            stack.append(joint.child)
    return transforms


def normalize_joint_angles(joints: list[JointSpec], joint_angles: Any) -> dict[str, float]:
    if joint_angles is None:
        return {}
    if isinstance(joint_angles, dict):
        return {str(key): float(value) for key, value in joint_angles.items()}
    values = [float(v) for v in joint_angles]
    movable = [joint for joint in joints if joint.joint_type not in {"fixed", "floating", "planar"}]
    return {joint.name: values[index] for index, joint in enumerate(movable[: len(values)])}


def joint_motion_transform(joint: JointSpec, value: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if joint.joint_type in {"revolute", "continuous"}:
        T[:3, :3] = Rotation.from_rotvec(joint.axis * value).as_matrix()
    elif joint.joint_type == "prismatic":
        T[:3, 3] = joint.axis * value
    return T


def parse_origin(origin_el: ET.Element | None) -> np.ndarray:
    xyz = parse_xyz(origin_el.attrib.get("xyz") if origin_el is not None else None, default=(0.0, 0.0, 0.0))
    rpy = parse_xyz(origin_el.attrib.get("rpy") if origin_el is not None else None, default=(0.0, 0.0, 0.0))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def parse_xyz(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    parts = [float(item) for item in value.replace(",", " ").split()]
    if len(parts) == 1:
        parts = [parts[0], parts[0], parts[0]]
    if len(parts) != 3:
        raise ValueError(f"Expected three numeric values, got {value!r}")
    return np.asarray(parts, dtype=np.float64)


def resolve_mesh_path(urdf_dir: Path, filename: str) -> Path:
    if filename.startswith("file://"):
        return Path(filename[7:]).expanduser().resolve()
    if filename.startswith("package://"):
        parts = filename[len("package://") :].split("/", 1)
        rel = parts[1] if len(parts) == 2 else parts[0]
        return (urdf_dir / rel).resolve()
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (urdf_dir / path).resolve()


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if mesh_path.suffix.lower() != ".obj":
        try:
            import trimesh  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"Non-OBJ mesh requires trimesh: {mesh_path}") from exc
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw_line in mesh_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            indices = []
            for part in line.split()[1:]:
                token = part.split("/")[0]
                index = int(token)
                if index < 0:
                    index = len(vertices) + index + 1
                indices.append(index - 1)
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no triangles: {mesh_path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)
