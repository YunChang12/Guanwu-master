#!/usr/bin/env python3
"""SAPIEN-based episode replay for RoboTwin 2.0 → USDC export.

Requirements:
  - Linux x86_64 + NVIDIA GPU with CUDA
  - ``pip install sapien==3.0.*`` (inside an existing conda env)
  - RoboTwin source code: ``pip install -e /path/to/RoboTwin``
  - ``pip install usd-core h5py numpy trimesh pycollada``

Usage as standalone script (for development / quick testing)::

    cd /tmp/robotwin_replay
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate sam3d

    export PYTHONPATH=/tmp/robotwin_replay/RoboTwin:$PYTHONPATH
    python replay_episode.py \\
        --h5 dataset/handover_block/aloha-agilex_clean_50/data/episode0.hdf5 \\
        --task handover_block \\
        --seed 0 \\
        --robotwin-root /tmp/robotwin_replay/RoboTwin \\
        --out /tmp/replay_out/episode0.usdc
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("guanwu.replay")

_CAMERA_NAMES = ("front_camera", "head_camera", "left_camera", "right_camera")
SceneActorRecord = tuple[str, str, int, Any]
Vec3 = tuple[float, float, float]

_GENERATED_CAMERA_MOTIONS = ("random-static", "trajectory")
_TRAJECTORY_KINDS = (
    "orbit360",
    "up_down",
    "left_right",
    "swing",
    "shake",
    "dolly",
    "spiral",
)

try:
    import imageio
    _IMAGEIO_OK = True
except ImportError:  # pragma: no cover
    _IMAGEIO_OK = False

# ---------------------------------------------------------------------------
# Availability check (safe to call on macOS — just returns False)
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if SAPIEN + RoboTwin envs are importable."""
    try:
        import sapien  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Core replay logic
# ---------------------------------------------------------------------------

def replay_episode(
    h5_path: Path,
    task_name: str,
    seed: int,
    robotwin_root: Path,
    out_usdc: Path | None = None,
    out_usdz: Path | None = None,
    keep_usdc_assets: bool = True,
    renders_dir: Path | None = None,
    views_dir: Path | None = None,
    view_specs: list[dict[str, Any]] | None = None,
    generated_camera_motion: str = "random-static",
    generated_static_camera_count: int | None = None,
    generated_trajectory_camera_count: int | None = None,
    camera_trajectory_kind: str = "orbit360",
    trajectory_kind_mode: str = "fixed",
    camera_seed: int | None = None,
    view_width_px: int = 768,
    view_height_px: int = 768,
    embodiment: str = "aloha-agilex",
    fps: float = 30.0,
    video_fps: int = 10,
    headless: bool = True,
) -> dict[str, Any] | None:
    """Replay one HDF5 episode in SAPIEN and optionally write a USDC file.

    Parameters
    ----------
    h5_path : Path to the episode HDF5 (e.g. ``data/episode0.hdf5``).
    task_name : RoboTwin task module name (e.g. ``"handover_block"``).
    seed : Episode random seed (from ``seed.txt``).
    robotwin_root : Root of the RoboTwin repository clone.
    out_usdc : If given, write the animated USDC to this path.
    out_usdz : If given, package the animated scene and texture assets into
        one USDZ file. When ``out_usdc`` is absent, a temporary sibling USDC is
        created as the package source.
    keep_usdc_assets : If False after ``out_usdz`` is written, remove the
        temporary USDC and sibling ``textures/`` directory.
    renders_dir : If given, render per-camera MP4 files into this directory.
        This legacy path may write HDF5 observation videos and is not used for
        generated learning views.
    views_dir : If given, render generated-camera live SAPIEN view videos into
        ``views_dir/<view_id>/video.mp4``. These videos never use HDF5 RGB.
    view_specs : Optional explicit generated camera specs. When absent,
        ``generated_*`` options create random-static and/or trajectory views.
    embodiment : Robot embodiment name (default ``"aloha-agilex"``).
    fps : Frames per second for the USDC timeline.
    video_fps : Frames per second for rendered MP4 videos.
    headless : If True (default), run without a GUI viewer.

    Returns
    -------
    dict with keys:
        ``"T"``           – number of frames
        ``"actor_states"``– {actor_name: (T, 7) ndarray [x,y,z, qw,qx,qy,qz]}
        ``"actor_meshes"``– {actor_name: {"verts": Nx3, "faces": Mx3, "color": (r,g,b)}}
        ``"robot_link_states"``– {link_name: (T, 4, 4) world transforms}
        ``"robot_link_meshes"``– {link_name: {"verts": Nx3, "faces": Mx3, "color": (r,g,b)}}

    Raises if prerequisites are missing or scene geometry cannot be exported.
    """
    if not is_available():
        raise RuntimeError("SAPIEN is not importable; cannot replay RoboTwin episode")

    import h5py
    import yaml

    # Resolve paths to absolute BEFORE any chdir happens later.
    h5_path = Path(h5_path).resolve()
    robotwin_root = Path(robotwin_root).resolve()
    usdz_path = Path(out_usdz).resolve() if out_usdz is not None else None
    usdc_path = Path(out_usdc).resolve() if out_usdc is not None else None
    if usdc_path is None and usdz_path is not None:
        usdc_path = usdz_path.with_suffix(".usdc")
    texture_exporter = (
        TextureExportContext(usdc_path.parent)
        if usdc_path is not None
        else None
    )

    # ── Ensure RoboTwin is on sys.path ───────────────────────────────────
    rt_root = robotwin_root
    rt_str = str(rt_root)
    if rt_str not in sys.path:
        sys.path.insert(0, rt_str)

    # ── Resolve embodiment file path + config ────────────────────────────
    emb_config_path = rt_root / "task_config" / "_embodiment_config.yml"
    with open(emb_config_path) as f:
        emb_cfg = yaml.safe_load(f)
    robot_file = emb_cfg[embodiment]["file_path"]

    # robot_file is relative to the RoboTwin root (e.g. "./assets/embodiments/aloha-agilex/")
    robot_file_abs = str((rt_root / robot_file.lstrip("./")).resolve()) + "/"
    embodiment_root = Path(robot_file_abs)
    with open(embodiment_root / "config.yml") as f:
        embodiment_args = yaml.safe_load(f)

    # ── Load task config (clean variant by default) ──────────────────────
    task_config_path = rt_root / "task_config" / "demo_clean.yml"
    with open(task_config_path) as f:
        args: dict[str, Any] = yaml.safe_load(f)

    args["task_name"] = task_name
    args["seed"] = seed
    args["now_ep_num"] = 0
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = embodiment_args
    args["right_embodiment_config"] = embodiment_args
    args["dual_arm_embodied"] = True
    args["save_data"] = False
    args["need_plan"] = False
    args["render_freq"] = 0   # no viewer

    # Load pre-planned trajectory from _traj_data so the robot knows the path
    variant_dir = h5_path.resolve().parent.parent
    traj_pkl = variant_dir / "_traj_data" / f"{h5_path.stem}.pkl"
    if traj_pkl.is_file():
        with open(traj_pkl, "rb") as fp:
            traj_data = pickle.load(fp)
        args["left_joint_path"] = traj_data.get("left_joint_path", [])
        args["right_joint_path"] = traj_data.get("right_joint_path", [])
    else:
        args["left_joint_path"] = []
        args["right_joint_path"] = []

    with h5py.File(str(h5_path), "r") as f:
        joint_action = np.array(f["joint_action/vector"])
    camera_observations = _read_hdf5_camera_observations(
        h5_path,
        read_rgb=renders_dir is not None,
    )

    renders_path = Path(renders_dir) if renders_dir is not None else None
    if renders_path is not None:
        renders_path.mkdir(parents=True, exist_ok=True)
        if not _IMAGEIO_OK:
            logger.warning(
                "imageio is not installed; skipping video export for %s",
                h5_path,
            )
            renders_path = None
    simulator_renders_path = renders_path
    if renders_path is not None:
        try:
            if _write_hdf5_camera_videos(
                camera_observations,
                renders_path,
                fps=video_fps,
            ):
                simulator_renders_path = None
                logger.info("Wrote authoritative HDF5 observation videos to %s", renders_path)
        except Exception:
            logger.exception(
                "Failed to export HDF5 observation videos for %s; "
                "falling back to simulator replay capture.",
                h5_path,
            )

    # ── Initialize the task scene ────────────────────────────────────────
    logger.info("Initializing %s scene with seed=%d …", task_name, seed)
    task_env = _build_task_env(rt_root, task_name, args)
    logger.info("Scene ready — %d actors.", len(task_env.scene.get_all_actors()))

    # ── Discover actors + extract initial mesh data ──────────────────────
    all_actors = list(task_env.scene.get_all_actors())
    actor_records = _scene_actor_records(all_actors)
    actor_metadata = _actor_metadata(actor_records)
    actor_meshes: dict[str, dict] = {}
    for actor_key, _source_name, _duplicate_index, actor in actor_records:
        actor_meshes[actor_key] = _extract_entity_mesh(
            actor,
            texture_exporter=texture_exporter,
            texture_prefix=f"actor_{actor_key}",
        )

    robot_links = _get_robot_links(task_env)
    urdf_path = embodiment_root / embodiment_args["urdf_path"]
    robot_link_meshes = _load_urdf_link_meshes(
        urdf_path,
        texture_exporter=texture_exporter,
        texture_prefix="robot",
    )
    robot_link_metadata = _load_urdf_link_metadata(urdf_path)
    scene_dt = _get_scene_timestep(task_env.scene)
    native_replay_ready = bool(
        (args["left_joint_path"] or args["right_joint_path"])
        and hasattr(task_env, "play_once")
    )

    views_path = Path(views_dir) if views_dir is not None else None
    view_recorders: list[dict[str, Any]] = []
    if views_path is not None or view_specs:
        if not _IMAGEIO_OK:
            raise RuntimeError(
                "imageio is required for generated RoboTwin view export. "
                "Install with: pip install imageio imageio-ffmpeg"
            )
        scene_bounds = _scene_bounds_from_initial_state(
            actor_records=actor_records,
            actor_meshes=actor_meshes,
            robot_links=robot_links,
            robot_link_meshes=robot_link_meshes,
        )
        planned_views = _build_generated_view_specs(
            scene_bounds=scene_bounds,
            frame_count=max(int(joint_action.shape[0]), 1),
            fps=float(fps),
            explicit_view_specs=view_specs,
            generated_camera_motion=generated_camera_motion,
            generated_static_camera_count=generated_static_camera_count,
            generated_trajectory_camera_count=generated_trajectory_camera_count,
            camera_trajectory_kind=camera_trajectory_kind,
            trajectory_kind_mode=trajectory_kind_mode,
            camera_seed=seed if camera_seed is None else int(camera_seed),
            width_px=int(view_width_px),
            height_px=int(view_height_px),
        )
        if planned_views:
            if views_path is None:
                raise ValueError("views_dir is required when view_specs are provided")
            view_recorders = _open_scene_view_recorders(
                task_env=task_env,
                views_dir=views_path,
                view_specs=planned_views,
                source_fps=float(fps),
                video_fps=float(video_fps),
            )

    if native_replay_ready:
        logger.info(
            "Replaying via task-native play_once() at dt=%.6f s for %s",
            scene_dt,
            h5_path.name,
        )
        try:
            result = _replay_episode_via_native_play_once(
                task_env=task_env,
                actor_records=actor_records,
                actor_meshes=actor_meshes,
                actor_metadata=actor_metadata,
                robot_links=robot_links,
                robot_link_meshes=robot_link_meshes,
                fps=fps,
                scene_dt=scene_dt,
                view_recorders=view_recorders,
            )
        finally:
            view_records = _close_scene_view_recorders(view_recorders)
            task_env.close_env()

        if simulator_renders_path is not None:
            video_env = _build_task_env(rt_root, task_name, args)
            try:
                _render_episode_via_native_play_once(
                    task_env=video_env,
                    renders_dir=simulator_renders_path,
                    scene_dt=scene_dt,
                    video_fps=video_fps,
                )
                _write_render_manifest(
                    simulator_renders_path,
                    video_source="simulator_replay_rgb",
                    cameras=camera_observations,
                    fps=video_fps,
                    camera_pose_source="simulator task_env.cameras (not serialized)",
                    camera_intrinsic_source="simulator task_env.cameras (not serialized)",
                )
            finally:
                video_env.close_env()
    else:
        logger.warning(
            "No cached native RoboTwin trajectory found for %s; "
            "falling back to qpos replay.",
            h5_path,
        )
        try:
            result = _replay_episode_via_qpos(
                task_env=task_env,
                joint_action=joint_action,
                actor_records=actor_records,
                actor_meshes=actor_meshes,
                actor_metadata=actor_metadata,
                robot_links=robot_links,
                robot_link_meshes=robot_link_meshes,
                fps=fps,
                renders_dir=simulator_renders_path,
                video_fps=video_fps,
                view_recorders=view_recorders,
            )
            if simulator_renders_path is not None:
                _write_render_manifest(
                    simulator_renders_path,
                    video_source="simulator_replay_rgb",
                    cameras=camera_observations,
                    fps=video_fps,
                    camera_pose_source="simulator task_env.cameras (not serialized)",
                    camera_intrinsic_source="simulator task_env.cameras (not serialized)",
                )
        finally:
            view_records = _close_scene_view_recorders(view_recorders)
            task_env.close_env()

    # ── Write USDC if requested ──────────────────────────────────────────
    result["robot_link_metadata"] = robot_link_metadata
    result["camera_observations"] = {
        name: {
            key: value
            for key, value in data.items()
            if key in {"cam2world_gl", "intrinsic_cv", "image_size"}
        }
        for name, data in camera_observations.items()
    }
    if usdc_path is not None:
        usdc_path.parent.mkdir(parents=True, exist_ok=True)
        _write_replay_usdc(usdc_path, result)
        logger.info("Wrote USDC: %s (%d KB)", usdc_path, usdc_path.stat().st_size // 1024)
        result["scene_usdc"] = str(usdc_path)

    if usdz_path is not None:
        if usdc_path is None or not usdc_path.is_file():
            raise RuntimeError("USDC source is required before writing USDZ")
        usdz_path.parent.mkdir(parents=True, exist_ok=True)
        _write_replay_usdz(usdc_path, usdz_path)
        logger.info("Wrote USDZ: %s (%d KB)", usdz_path, usdz_path.stat().st_size // 1024)
        result["scene_usdz"] = str(usdz_path)
        if not keep_usdc_assets:
            try:
                usdc_path.unlink()
            except FileNotFoundError:
                pass
            shutil.rmtree(usdc_path.parent / "textures", ignore_errors=True)

    if view_records:
        result["view_records"] = view_records

    return result


def _build_task_env(rt_root: Path, task_name: str, args: dict[str, Any]) -> Any:
    """Instantiate and set up one RoboTwin task environment."""
    os.chdir(str(rt_root))  # RoboTwin refs relative paths for assets
    import importlib

    task_mod = importlib.import_module(f"envs.{task_name}")
    task_cls = getattr(task_mod, task_name)
    task_env = task_cls()
    task_env.setup_demo(**dict(args))
    return task_env


def _get_scene_timestep(scene: Any) -> float:
    """Best-effort lookup of the simulator timestep."""
    get_timestep = getattr(scene, "get_timestep", None)
    if callable(get_timestep):
        try:
            dt = float(get_timestep())
        except Exception:
            dt = 0.0
        if dt > 0.0:
            return dt
    return 1.0 / 240.0


def _scene_actor_records(all_actors: list[Any]) -> list[SceneActorRecord]:
    """Assign stable unique keys to actors, even when SAPIEN names collide."""
    counts: dict[str, int] = {}
    used_keys: set[str] = set()
    records: list[SceneActorRecord] = []
    duplicates: dict[str, int] = {}

    for actor_index, actor in enumerate(all_actors):
        try:
            source_name = str(actor.get_name() or "")
        except Exception:
            source_name = ""
        if not source_name:
            source_name = f"Actor_{actor_index}"

        duplicate_index = counts.get(source_name, 0) + 1
        counts[source_name] = duplicate_index
        actor_key = source_name if duplicate_index == 1 else f"{source_name}_{duplicate_index}"
        while actor_key in used_keys:
            duplicate_index += 1
            counts[source_name] = duplicate_index
            actor_key = f"{source_name}_{duplicate_index}"

        used_keys.add(actor_key)
        records.append((actor_key, source_name, duplicate_index, actor))
        if duplicate_index > 1:
            duplicates[source_name] = duplicate_index

    if duplicates:
        logger.info(
            "Disambiguated duplicate RoboTwin actor names: %s",
            ", ".join(f"{name} x{count}" for name, count in sorted(duplicates.items())),
        )

    return records


def _actor_metadata(actor_records: list[SceneActorRecord]) -> dict[str, dict[str, Any]]:
    """Metadata for USD provenance/debugging after actor-name disambiguation."""
    return {
        actor_key: {
            "source_name": source_name,
            "duplicate_index": int(duplicate_index),
        }
        for actor_key, source_name, duplicate_index, _actor in actor_records
    }


class _TimedCaptureSampler:
    """Capture scene state or camera frames at a fixed real-time cadence."""

    def __init__(self, capture_hz: float, scene_dt: float, capture_fn: Any) -> None:
        self.capture_hz = max(float(capture_hz), 0.0)
        self.scene_dt = max(float(scene_dt), 1e-6)
        self.capture_fn = capture_fn
        self.current_time = 0.0
        self.last_capture_time = -1.0
        self.frames_captured = 0
        self._period = 0.0 if self.capture_hz <= 0.0 else 1.0 / self.capture_hz
        self._next_capture_time = 0.0

    def capture_initial(self) -> None:
        self.capture_fn()
        self.frames_captured += 1
        self.last_capture_time = self.current_time
        self._next_capture_time = self._period

    def on_step(self) -> None:
        self.current_time += self.scene_dt
        if self.capture_hz <= 0.0:
            self._capture()
            return

        epsilon = self.scene_dt * 0.5 + 1e-9
        if self.current_time + epsilon < self._next_capture_time:
            return
        self._capture()
        while self._next_capture_time <= self.current_time + epsilon:
            self._next_capture_time += self._period

    def capture_final(self) -> None:
        if self.current_time - self.last_capture_time > self.scene_dt * 0.5:
            self._capture()

    def _capture(self) -> None:
        self.capture_fn()
        self.frames_captured += 1
        self.last_capture_time = self.current_time


def _run_native_play_once(
    task_env: Any,
    *,
    capture_hz: float,
    scene_dt: float,
    capture_fn: Any,
) -> int:
    """Run task_env.play_once() while sampling scene state from the true replay."""
    scene = task_env.scene
    original_step = scene.step
    sampler = _TimedCaptureSampler(capture_hz=capture_hz, scene_dt=scene_dt, capture_fn=capture_fn)

    def wrapped_step(*args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        sampler.on_step()
        return result

    sampler.capture_initial()
    scene.step = wrapped_step
    try:
        task_env.play_once()
        sampler.capture_final()
    finally:
        scene.step = original_step
    return sampler.frames_captured


def _record_actor_pose_frame(
    actor_records: list[SceneActorRecord],
    actor_states: dict[str, list[np.ndarray]],
) -> None:
    """Append one world-pose snapshot for every scene actor."""
    for actor_key, _source_name, _duplicate_index, actor in actor_records:
        pose = actor.get_pose()
        actor_states[actor_key].append(
            np.array(
                [*pose.p, pose.q[0], pose.q[1], pose.q[2], pose.q[3]],
                dtype=np.float64,
            )
        )


def _record_robot_link_frame(
    robot_links: list[Any],
    robot_link_states: dict[str, list[np.ndarray]],
) -> None:
    """Append one world transform snapshot for every robot link."""
    for link in robot_links:
        pose = link.get_pose()
        T_mat = np.eye(4)
        T_mat[:3, :3] = _sapien_quat_to_rot(pose.q)
        T_mat[:3, 3] = pose.p
        robot_link_states[link.get_name()].append(T_mat)


def _stack_pose_history(history: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Convert per-frame history lists into numpy arrays."""
    out: dict[str, np.ndarray] = {}
    for name, frames in history.items():
        if not frames:
            continue
        out[name] = np.stack(frames)
    return out


def _scene_bounds_from_initial_state(
    *,
    actor_records: list[SceneActorRecord],
    actor_meshes: dict[str, dict[str, Any]],
    robot_links: list[Any],
    robot_link_meshes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Estimate manipulation-scene bounds for generated camera placement."""
    all_points: list[np.ndarray] = []
    skipped_scene_shells = {"ground", "wall"}

    for actor_key, source_name, _duplicate_index, actor in actor_records:
        if str(source_name).lower() in skipped_scene_shells:
            continue
        mesh_info = actor_meshes.get(actor_key)
        if not mesh_info:
            continue
        try:
            local_points = _mesh_info_vertices(mesh_info)
            if local_points.size == 0:
                continue
            if not _mesh_extent_is_reasonable(local_points, max_abs_coord=20.0, max_span=20.0):
                continue
            all_points.append(_apply_sapien_pose(local_points, actor.get_pose()))
        except Exception:
            logger.debug("Skipping actor %s while estimating camera bounds", actor_key, exc_info=True)

    for link in robot_links:
        try:
            mesh_info = robot_link_meshes.get(link.get_name())
            if not mesh_info:
                continue
            local_points = _mesh_info_vertices(mesh_info)
            if local_points.size == 0:
                continue
            all_points.append(_apply_sapien_pose(local_points, link.get_pose()))
        except Exception:
            logger.debug(
                "Skipping robot link while estimating camera bounds",
                exc_info=True,
            )

    if not all_points:
        return {
            "center_W_m": (0.0, -0.45, 0.75),
            "extent_m": (1.6, 1.6, 1.2),
            "radius_m": 1.0,
            "up_axis": "Z",
        }

    points = np.concatenate(all_points, axis=0)
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    center = (min_corner + max_corner) * 0.5
    extent = np.maximum(max_corner - min_corner, 1e-3)
    radius = max(float(np.linalg.norm(extent) * 0.5), 0.25)
    return {
        "center_W_m": tuple(float(v) for v in center),
        "extent_m": tuple(float(v) for v in extent),
        "radius_m": float(radius),
        "up_axis": "Z",
    }


def _mesh_info_vertices(mesh_info: dict[str, Any]) -> np.ndarray:
    """Return representative local vertices from a mesh-info payload."""
    parts = mesh_info.get("parts")
    if isinstance(parts, list) and parts:
        arrays = []
        for part in parts:
            verts = part.get("verts") if isinstance(part, dict) else None
            if verts is None:
                continue
            arr = np.asarray(verts, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.size:
                arrays.append(arr)
        if arrays:
            return np.concatenate(arrays, axis=0)

    verts = mesh_info.get("verts")
    if verts is None:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.asarray(verts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr


def _build_generated_view_specs(
    *,
    scene_bounds: dict[str, Any],
    frame_count: int,
    fps: float,
    explicit_view_specs: list[dict[str, Any]] | None = None,
    generated_camera_motion: str = "random-static",
    generated_static_camera_count: int | None = None,
    generated_trajectory_camera_count: int | None = None,
    camera_trajectory_kind: str = "orbit360",
    trajectory_kind_mode: str = "fixed",
    camera_seed: int | None = None,
    width_px: int = 768,
    height_px: int = 768,
) -> list[dict[str, Any]]:
    """Create normalized generated-camera view specs for live SAPIEN rendering."""
    if explicit_view_specs:
        return [
            _normalize_view_spec(
                dict(spec),
                view_index=index,
                width_px=width_px,
                height_px=height_px,
                fps=fps,
            )
            for index, spec in enumerate(explicit_view_specs)
        ]

    motion = generated_camera_motion
    if motion not in _GENERATED_CAMERA_MOTIONS:
        raise ValueError(
            "generated_camera_motion must be one of: "
            + ", ".join(_GENERATED_CAMERA_MOTIONS)
        )
    if camera_trajectory_kind not in _TRAJECTORY_KINDS:
        raise ValueError(
            "camera_trajectory_kind must be one of: "
            + ", ".join(_TRAJECTORY_KINDS)
        )
    if trajectory_kind_mode not in {"fixed", "random"}:
        raise ValueError("trajectory_kind_mode must be one of: fixed, random")

    explicit_counts = (
        generated_static_camera_count is not None
        or generated_trajectory_camera_count is not None
    )
    if explicit_counts:
        static_count = int(generated_static_camera_count or 0)
        trajectory_count = int(generated_trajectory_camera_count or 0)
    else:
        static_count = 1 if motion == "random-static" else 0
        trajectory_count = 1 if motion == "trajectory" else 0
    if static_count < 0 or trajectory_count < 0:
        raise ValueError("generated camera counts must be >= 0")

    specs: list[dict[str, Any]] = []
    view_index = 0
    for local_index in range(static_count):
        view_seed = _seed_for_generated_view(camera_seed, view_index)
        camera = _sample_random_view_camera(
            scene_bounds,
            seed=view_seed,
            width_px=width_px,
            height_px=height_px,
        )
        specs.append(
            _normalize_view_spec(
                {
                    **camera,
                    "view_id": f"view_{view_index:03d}",
                    "observation_id": f"generated_camera_static_{local_index:03d}",
                    "camera_strategy": "random-static",
                    "seed": view_seed,
                    "view_index": view_index,
                    "view_local_index": local_index,
                    "scene_bounds": scene_bounds,
                },
                view_index=view_index,
                width_px=width_px,
                height_px=height_px,
                fps=fps,
            )
        )
        view_index += 1

    for local_index in range(trajectory_count):
        view_seed = _seed_for_generated_view(camera_seed, view_index)
        trajectory_kind = camera_trajectory_kind
        if trajectory_kind_mode == "random":
            trajectory_kind = random.Random(view_seed).choice(_TRAJECTORY_KINDS)
        camera = _sample_trajectory_view_camera(
            scene_bounds,
            seed=view_seed,
            frame_count=max(int(frame_count), 1),
            trajectory_kind=trajectory_kind,
            width_px=width_px,
            height_px=height_px,
        )
        specs.append(
            _normalize_view_spec(
                {
                    **camera,
                    "view_id": f"view_{view_index:03d}",
                    "observation_id": f"generated_camera_trajectory_{local_index:03d}",
                    "camera_strategy": "trajectory",
                    "camera_trajectory_kind": trajectory_kind,
                    "trajectory_kind_mode": trajectory_kind_mode,
                    "seed": view_seed,
                    "view_index": view_index,
                    "view_local_index": local_index,
                    "scene_bounds": scene_bounds,
                },
                view_index=view_index,
                width_px=width_px,
                height_px=height_px,
                fps=fps,
            )
        )
        view_index += 1

    return specs


def _normalize_view_spec(
    spec: dict[str, Any],
    *,
    view_index: int,
    width_px: int,
    height_px: int,
    fps: float,
) -> dict[str, Any]:
    view_id = _safe_view_id(spec.get("view_id") or f"view_{view_index:03d}")
    camera = dict(spec.get("camera") or spec)
    mode = str(camera.get("mode") or spec.get("mode") or "fixed_orbit")
    camera["mode"] = mode
    camera["fov_deg"] = float(camera.get("fov_deg", 55.0))
    camera["clip_near_m"] = float(camera.get("clip_near_m", 0.01))
    camera["clip_far_m"] = float(camera.get("clip_far_m", 8.0))
    camera["width_px"] = int(camera.get("width_px", spec.get("width_px", width_px)))
    camera["height_px"] = int(camera.get("height_px", spec.get("height_px", height_px)))
    if mode == "camera_trajectory":
        keyframes = camera.get("keyframes") or spec.get("keyframes") or []
        if not keyframes:
            raise ValueError("camera_trajectory view requires keyframes")
        camera["keyframes"] = [
            {
                "time_code": float(frame.get("time_code", index)),
                "eye_W_m": _as_vec3(frame.get("eye_W_m"), (0.0, -2.0, 1.0)),
                "target_W_m": _as_vec3(frame.get("target_W_m"), (0.0, 0.0, 0.5)),
                "up_W": _as_vec3(frame.get("up_W"), (0.0, 0.0, 1.0)),
            }
            for index, frame in enumerate(keyframes)
        ]
    else:
        camera["eye_W_m"] = _as_vec3(camera.get("eye_W_m"), (0.0, -2.0, 1.0))
        camera["target_W_m"] = _as_vec3(camera.get("target_W_m"), (0.0, 0.0, 0.5))
        camera["up_W"] = _as_vec3(camera.get("up_W"), (0.0, 0.0, 1.0))

    return {
        **spec,
        "view_id": view_id,
        "camera": camera,
        "fps": float(spec.get("fps", fps)),
        "width_px": int(camera["width_px"]),
        "height_px": int(camera["height_px"]),
        "uses_hdf5_rgb": False,
    }


def _seed_for_generated_view(seed: int | None, view_index: int) -> int:
    base = 0 if seed is None else int(seed)
    return int(base + 1009 * int(view_index))


def _sample_random_view_camera(
    bounds: dict[str, Any],
    *,
    seed: int,
    width_px: int,
    height_px: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    center = _as_vec3(bounds.get("center_W_m"), (0.0, -0.45, 0.75))
    up = _up_vector(str(bounds.get("up_axis", "Z")))
    right, forward_ref = _basis_from_up(up)
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    elevation = math.radians(rng.uniform(15.0, 50.0))
    radius = max(float(bounds.get("radius_m", 1.0)), 0.25)
    fov_deg = 55.0
    distance = max(radius * 2.4, 0.5)
    eye = _orbit_eye(center, up, forward_ref, right, azimuth, elevation, distance)
    return {
        "mode": "fixed_orbit",
        "eye_W_m": eye,
        "target_W_m": center,
        "up_W": up,
        "fov_deg": fov_deg,
        "clip_near_m": 0.01,
        "clip_far_m": max(1.01, radius * 8.0),
        "width_px": int(width_px),
        "height_px": int(height_px),
    }


def _sample_trajectory_view_camera(
    bounds: dict[str, Any],
    *,
    seed: int,
    frame_count: int,
    trajectory_kind: str,
    width_px: int,
    height_px: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    center = _as_vec3(bounds.get("center_W_m"), (0.0, -0.45, 0.75))
    up = _up_vector(str(bounds.get("up_axis", "Z")))
    right, forward_ref = _basis_from_up(up)
    radius = max(float(bounds.get("radius_m", 1.0)), 0.25)
    distance = max(radius * 2.4, 0.5)
    base_azimuth = rng.uniform(0.0, 2.0 * math.pi)
    base_elevation = math.radians(32.5)
    n = max(int(frame_count) - 1, 1)
    keyframes: list[dict[str, Any]] = []
    for index in range(max(int(frame_count), 1)):
        phase = index / n
        azimuth = base_azimuth
        elevation = base_elevation
        local_distance = distance
        local_target = center
        jitter = (0.0, 0.0, 0.0)

        if trajectory_kind == "orbit360":
            azimuth = base_azimuth + 2.0 * math.pi * phase
        elif trajectory_kind == "up_down":
            elevation = base_elevation + 0.35 * math.sin(2.0 * math.pi * phase)
        elif trajectory_kind == "left_right":
            lateral = 0.35 * radius * math.sin(2.0 * math.pi * phase)
            local_target = _add(center, _scale(right, 0.25 * lateral))
            jitter = _scale(right, lateral)
        elif trajectory_kind == "swing":
            azimuth = base_azimuth + math.radians(30.0) * math.sin(2.0 * math.pi * phase)
        elif trajectory_kind == "shake":
            jitter = (
                rng.uniform(-1.0, 1.0) * 0.03 * radius,
                rng.uniform(-1.0, 1.0) * 0.03 * radius,
                rng.uniform(-1.0, 1.0) * 0.03 * radius,
            )
        elif trajectory_kind == "dolly":
            local_distance = distance * (1.0 + 0.35 * math.sin(2.0 * math.pi * phase))
        elif trajectory_kind == "spiral":
            azimuth = base_azimuth + 2.0 * math.pi * phase
            elevation = base_elevation + 0.8 * (phase - 0.5)

        eye = _orbit_eye(local_target, up, forward_ref, right, azimuth, elevation, local_distance)
        eye = _add(eye, jitter)
        keyframes.append(
            {
                "time_code": float(index),
                "eye_W_m": eye,
                "target_W_m": local_target,
                "up_W": up,
            }
        )

    return {
        "mode": "camera_trajectory",
        "keyframes": keyframes,
        "fov_deg": 55.0,
        "clip_near_m": 0.01,
        "clip_far_m": max(1.01, radius * 8.0),
        "width_px": int(width_px),
        "height_px": int(height_px),
    }


def _open_scene_view_recorders(
    *,
    task_env: Any,
    views_dir: Path,
    view_specs: list[dict[str, Any]],
    source_fps: float,
    video_fps: float,
) -> list[dict[str, Any]]:
    """Create SAPIEN cameras and MP4 writers for generated live views."""
    views_dir.mkdir(parents=True, exist_ok=True)
    recorders: list[dict[str, Any]] = []
    for index, spec in enumerate(view_specs):
        view_id = _safe_view_id(spec.get("view_id") or f"view_{index:03d}")
        view_dir = views_dir / view_id
        if view_dir.exists():
            shutil.rmtree(view_dir)
        view_dir.mkdir(parents=True, exist_ok=True)

        camera_spec = dict(spec["camera"])
        width = int(camera_spec.get("width_px") or spec.get("width_px") or 768)
        height = int(camera_spec.get("height_px") or spec.get("height_px") or 768)
        scene_camera = task_env.scene.add_camera(
            name=f"guanwu_{view_id}",
            width=width,
            height=height,
            fovy=math.radians(float(camera_spec.get("fov_deg", 55.0))),
            near=float(camera_spec.get("clip_near_m", 0.01)),
            far=float(camera_spec.get("clip_far_m", 8.0)),
        )
        _set_sapien_camera_pose(scene_camera, _camera_sample_for_time(camera_spec, 0.0))
        writer_fps = max(int(round(video_fps)), 1)
        writer = _open_video_writer(view_dir / "video.mp4", fps=writer_fps)
        camera_json = {
            "view_id": view_id,
            "observation_id": spec.get("observation_id", view_id),
            "coordinate_frame": "CanonicalWorld",
            "camera": camera_spec,
            "width_px": width,
            "height_px": height,
            "fps": float(writer_fps),
            "source_fps": float(source_fps),
            "uses_hdf5_rgb": False,
            "renderer": "sapien_live",
            "source": "robotwin_live_replay_generated_camera",
            "camera_strategy": spec.get("camera_strategy", camera_spec.get("mode")),
            "seed": spec.get("seed"),
            "scene_bounds": spec.get("scene_bounds"),
        }
        (view_dir / "camera.json").write_text(json.dumps(camera_json, indent=2), encoding="utf-8")
        recorders.append(
            {
                "view_id": view_id,
                "view_dir": view_dir,
                "camera_spec": camera_spec,
                "scene_camera": scene_camera,
                "writer": writer,
                "fps": float(writer_fps),
                "source_fps": float(source_fps),
                "next_capture_time_code": 0.0,
                "frame_mapping": [],
                "frame_count": 0,
                "camera_json": camera_json,
            }
        )
    return recorders


def _capture_scene_view_recorders(
    task_env: Any,
    recorders: list[dict[str, Any]],
    *,
    time_code: float,
    scene_frame_index: int,
) -> None:
    if not recorders:
        return
    due_recorders = [
        recorder
        for recorder in recorders
        if float(time_code) + 1e-6 >= float(recorder.get("next_capture_time_code", 0.0))
    ]
    if not due_recorders:
        return

    for recorder in due_recorders:
        sample = _camera_sample_for_time(recorder["camera_spec"], time_code)
        _set_sapien_camera_pose(recorder["scene_camera"], sample)

    if hasattr(task_env, "_update_render"):
        task_env._update_render()
    elif hasattr(task_env.scene, "update_render"):
        task_env.scene.update_render()

    for recorder in due_recorders:
        camera = recorder["scene_camera"]
        camera.take_picture()
        rgba = camera.get_picture("Color")
        frame = _camera_color_to_uint8(rgba)
        recorder["writer"].append_data(frame)
        video_frame_index = int(recorder["frame_count"])
        recorder["frame_mapping"].append(
            {
                "video_frame_index": video_frame_index,
                "scene_frame_index": int(scene_frame_index),
                "time_code": float(time_code),
                "time_sec": float(time_code) / max(float(recorder.get("source_fps", 1.0)), 1e-6),
            }
        )
        recorder["frame_count"] = video_frame_index + 1
        source_fps = max(float(recorder.get("source_fps", 1.0)), 1e-6)
        video_fps = max(float(recorder.get("fps", 1.0)), 1e-6)
        period = source_fps / video_fps
        next_time = float(recorder.get("next_capture_time_code", 0.0))
        while next_time <= float(time_code) + 1e-6:
            next_time += period
        recorder["next_capture_time_code"] = next_time


def _close_scene_view_recorders(recorders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Close generated-view writers and return manifest-ready view records."""
    records: list[dict[str, Any]] = []
    for recorder in list(recorders):
        view_dir = Path(recorder["view_dir"])
        writer = recorder.get("writer")
        try:
            if writer is not None:
                writer.close()
        except Exception:
            logger.debug("Failed to close generated view writer %s", view_dir, exc_info=True)

        mapping = recorder.get("frame_mapping", [])
        (view_dir / "frame_mapping.json").write_text(
            json.dumps(mapping, indent=2),
            encoding="utf-8",
        )
        render_meta = {
            "view_id": recorder["view_id"],
            "video": "video.mp4",
            "camera": "camera.json",
            "frame_mapping": "frame_mapping.json",
            "renderer": "sapien_live",
            "source": "robotwin_live_replay_generated_camera",
            "uses_hdf5_rgb": False,
            "fps": float(recorder.get("fps", 0.0)),
            "source_fps": float(recorder.get("source_fps", 0.0)),
            "frame_count": int(recorder.get("frame_count", 0)),
            "width_px": int(recorder["camera_json"].get("width_px", 0)),
            "height_px": int(recorder["camera_json"].get("height_px", 0)),
            "camera_strategy": recorder["camera_json"].get("camera_strategy"),
            "observation_id": recorder["camera_json"].get("observation_id"),
        }
        (view_dir / "render_meta.json").write_text(
            json.dumps(render_meta, indent=2),
            encoding="utf-8",
        )
        records.append(
            {
                "view_id": recorder["view_id"],
                "video": f"views/{recorder['view_id']}/video.mp4",
                "camera": f"views/{recorder['view_id']}/camera.json",
                "render_meta": f"views/{recorder['view_id']}/render_meta.json",
                "frame_mapping": f"views/{recorder['view_id']}/frame_mapping.json",
                "uses_hdf5_rgb": False,
                "frame_count": int(recorder.get("frame_count", 0)),
            }
        )
    recorders.clear()
    return records


def _camera_sample_for_time(camera_spec: dict[str, Any], time_code: float) -> dict[str, Any]:
    if camera_spec.get("mode") != "camera_trajectory":
        return {
            "eye_W_m": _as_vec3(camera_spec.get("eye_W_m"), (0.0, -2.0, 1.0)),
            "target_W_m": _as_vec3(camera_spec.get("target_W_m"), (0.0, 0.0, 0.5)),
            "up_W": _as_vec3(camera_spec.get("up_W"), (0.0, 0.0, 1.0)),
        }

    keyframes = sorted(
        list(camera_spec.get("keyframes") or []),
        key=lambda item: float(item.get("time_code", 0.0)),
    )
    if not keyframes:
        raise ValueError("camera_trajectory requires at least one keyframe")
    if len(keyframes) == 1 or time_code <= float(keyframes[0].get("time_code", 0.0)):
        first = keyframes[0]
        return {
            "eye_W_m": _as_vec3(first.get("eye_W_m"), (0.0, -2.0, 1.0)),
            "target_W_m": _as_vec3(first.get("target_W_m"), (0.0, 0.0, 0.5)),
            "up_W": _as_vec3(first.get("up_W"), (0.0, 0.0, 1.0)),
        }
    for left, right in zip(keyframes[:-1], keyframes[1:]):
        t0 = float(left.get("time_code", 0.0))
        t1 = float(right.get("time_code", t0))
        if time_code <= t1:
            alpha = 0.0 if t1 <= t0 else (float(time_code) - t0) / (t1 - t0)
            return {
                "eye_W_m": _lerp_vec3(
                    _as_vec3(left.get("eye_W_m"), (0.0, -2.0, 1.0)),
                    _as_vec3(right.get("eye_W_m"), (0.0, -2.0, 1.0)),
                    alpha,
                ),
                "target_W_m": _lerp_vec3(
                    _as_vec3(left.get("target_W_m"), (0.0, 0.0, 0.5)),
                    _as_vec3(right.get("target_W_m"), (0.0, 0.0, 0.5)),
                    alpha,
                ),
                "up_W": _lerp_vec3(
                    _as_vec3(left.get("up_W"), (0.0, 0.0, 1.0)),
                    _as_vec3(right.get("up_W"), (0.0, 0.0, 1.0)),
                    alpha,
                ),
            }
    last = keyframes[-1]
    return {
        "eye_W_m": _as_vec3(last.get("eye_W_m"), (0.0, -2.0, 1.0)),
        "target_W_m": _as_vec3(last.get("target_W_m"), (0.0, 0.0, 0.5)),
        "up_W": _as_vec3(last.get("up_W"), (0.0, 0.0, 1.0)),
    }


def _set_sapien_camera_pose(scene_camera: Any, sample: dict[str, Any]) -> None:
    pose = _look_at_sapien_pose(
        sample["eye_W_m"],
        sample["target_W_m"],
        sample["up_W"],
    )
    target = getattr(scene_camera, "entity", scene_camera)
    setter = getattr(target, "set_pose", None)
    if not callable(setter):
        setter = getattr(scene_camera, "set_pose", None)
    if not callable(setter):
        raise RuntimeError(f"SAPIEN camera {scene_camera!r} has no set_pose method")
    setter(pose)


def _look_at_sapien_pose(eye: Vec3, target: Vec3, up_hint: Vec3) -> Any:
    """Build a SAPIEN camera pose matching the live-render probe convention."""
    try:
        import sapien.core as sapien_core  # type: ignore[import]
    except ImportError:
        import sapien as sapien_core  # type: ignore[no-redef]

    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_arr = np.asarray(up_hint, dtype=np.float64)
    forward = target_arr - eye_arr
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    left = np.cross(up_arr, forward)
    if float(np.linalg.norm(left)) < 1e-8:
        left = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float64), forward)
    left /= max(float(np.linalg.norm(left)), 1e-12)
    up = np.cross(forward, left)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    mat44 = np.eye(4, dtype=np.float64)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = eye_arr
    return sapien_core.Pose(mat44)


def _camera_color_to_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise RuntimeError(f"Invalid SAPIEN camera color shape: {arr.shape}")
    rgb = arr[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    else:
        rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8, copy=False)


def _safe_view_id(value: Any) -> str:
    raw = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("._")
    return safe or "view_000"


def _as_vec3(value: Any, fallback: Vec3) -> Vec3:
    if value is None:
        return fallback
    try:
        vals = list(value)
    except TypeError:
        return fallback
    if len(vals) < 3:
        return fallback
    try:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    except (TypeError, ValueError):
        return fallback


def _up_vector(up_axis: str) -> Vec3:
    axis = up_axis.upper()
    if axis == "X":
        return (1.0, 0.0, 0.0)
    if axis == "Y":
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _basis_from_up(up: Vec3) -> tuple[Vec3, Vec3]:
    reference = (0.0, 0.0, 1.0) if abs(_dot(up, (0.0, 0.0, 1.0))) < 0.9 else (0.0, 1.0, 0.0)
    right = _normalize(_cross(reference, up))
    forward = _normalize(_cross(up, right))
    return right, forward


def _orbit_eye(
    target: Vec3,
    up: Vec3,
    forward_ref: Vec3,
    right: Vec3,
    azimuth: float,
    elevation: float,
    distance: float,
) -> Vec3:
    forward = _normalize(
        _add(_scale(forward_ref, math.cos(azimuth)), _scale(right, math.sin(azimuth)))
    )
    direction = _normalize(
        _add(_scale(forward, math.cos(elevation)), _scale(up, math.sin(elevation)))
    )
    return _add(target, _scale(direction, distance))


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(a: Vec3) -> Vec3:
    length = max(math.sqrt(_dot(a, a)), 1e-8)
    return (a[0] / length, a[1] / length, a[2] / length)


def _lerp_vec3(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    t = min(max(float(alpha), 0.0), 1.0)
    return (
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    )


def _replay_episode_via_native_play_once(
    *,
    task_env: Any,
    actor_records: list[SceneActorRecord],
    actor_meshes: dict[str, dict[str, Any]],
    actor_metadata: dict[str, dict[str, Any]],
    robot_links: list[Any],
    robot_link_meshes: dict[str, dict[str, Any]],
    fps: float,
    scene_dt: float,
    view_recorders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay using RoboTwin's cached native planner paths via play_once()."""
    actor_states: dict[str, list[np.ndarray]] = {
        actor_key: [] for actor_key, _source_name, _duplicate_index, _actor in actor_records
    }
    robot_link_states: dict[str, list[np.ndarray]] = {
        link.get_name(): [] for link in robot_links
    }

    def capture_state() -> None:
        frame_index = len(next(iter(actor_states.values()))) if actor_states else 0
        _record_actor_pose_frame(actor_records, actor_states)
        _record_robot_link_frame(robot_links, robot_link_states)
        if view_recorders:
            _capture_scene_view_recorders(
                task_env,
                view_recorders,
                time_code=float(frame_index),
                scene_frame_index=frame_index,
            )

    T = _run_native_play_once(
        task_env,
        capture_hz=fps,
        scene_dt=scene_dt,
        capture_fn=capture_state,
    )

    return {
        "T": T,
        "fps": fps,
        "actor_states": _stack_pose_history(actor_states),
        "actor_meshes": actor_meshes,
        "actor_metadata": actor_metadata,
        "robot_link_states": _stack_pose_history(robot_link_states),
        "robot_link_meshes": robot_link_meshes,
        "robot_link_names": [link.get_name() for link in robot_links],
    }


def _render_episode_via_native_play_once(
    *,
    task_env: Any,
    renders_dir: Path,
    scene_dt: float,
    video_fps: int,
) -> None:
    """Render RGB videos while replaying the true native RoboTwin episode."""
    video_writers: dict[str, Any] = {}
    atexit.register(_close_video_writers, video_writers)

    def capture_frame() -> None:
        camera_frames = _capture_camera_frames(task_env)
        _write_camera_frame_set(
            video_writers,
            renders_dir,
            camera_frames,
            fps=video_fps,
        )

    try:
        _run_native_play_once(
            task_env,
            capture_hz=video_fps,
            scene_dt=scene_dt,
            capture_fn=capture_frame,
        )
    finally:
        _close_video_writers(video_writers)
        _ensure_multiview_video(renders_dir)


def _replay_episode_via_qpos(
    *,
    task_env: Any,
    joint_action: np.ndarray,
    actor_records: list[SceneActorRecord],
    actor_meshes: dict[str, dict[str, Any]],
    actor_metadata: dict[str, dict[str, Any]],
    robot_links: list[Any],
    robot_link_meshes: dict[str, dict[str, Any]],
    fps: float,
    renders_dir: Path | None,
    video_fps: int,
    view_recorders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fallback replay path using direct qpos control when cached paths are absent."""
    actor_states: dict[str, list[np.ndarray]] = {
        actor_key: [] for actor_key, _source_name, _duplicate_index, _actor in actor_records
    }
    robot_link_states: dict[str, list[np.ndarray]] = {
        link.get_name(): [] for link in robot_links
    }

    video_writers: dict[str, Any] = {}
    if renders_dir is not None:
        atexit.register(_close_video_writers, video_writers)

    logger.info("Falling back to qpos replay across %d action frames …", len(joint_action))
    try:
        for t, action in enumerate(joint_action):
            task_env.take_action(action, action_type="qpos")
            _record_actor_pose_frame(actor_records, actor_states)
            _record_robot_link_frame(robot_links, robot_link_states)
            if view_recorders:
                _capture_scene_view_recorders(
                    task_env,
                    view_recorders,
                    time_code=float(t),
                    scene_frame_index=t,
                )

            if renders_dir is not None:
                try:
                    camera_frames = _capture_camera_frames(task_env)
                    _write_camera_frame_set(
                        video_writers,
                        renders_dir,
                        camera_frames,
                        fps=video_fps,
                    )
                except Exception:
                    logger.exception(
                        "Failed to render simulator cameras for frame %d", t
                    )

            if (t + 1) % 50 == 0:
                logger.info("  frame %d / %d", t + 1, len(joint_action))
    finally:
        _close_video_writers(video_writers)
        if renders_dir is not None:
            _ensure_multiview_video(renders_dir)

    return {
        "T": len(joint_action),
        "fps": fps,
        "actor_states": _stack_pose_history(actor_states),
        "actor_meshes": actor_meshes,
        "actor_metadata": actor_metadata,
        "robot_link_states": _stack_pose_history(robot_link_states),
        "robot_link_meshes": robot_link_meshes,
        "robot_link_names": [link.get_name() for link in robot_links],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sapien_quat_to_rot(q: Any) -> np.ndarray:
    """SAPIEN quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = max((w*w + x*x + y*y + z*z) ** 0.5, 1e-12)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _euler_xyz_to_rot(rpy: tuple[float, float, float]) -> np.ndarray:
    """ROS-style fixed-axis XYZ Euler → 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _apply_sapien_pose(points: np.ndarray, pose: Any) -> np.ndarray:
    """Apply a SAPIEN pose [p, q] to an Nx3 point cloud."""
    if points.size == 0:
        return points
    R = _sapien_quat_to_rot(pose.q)
    t = np.asarray(pose.p, dtype=np.float64)
    return points @ R.T + t[None, :]


def _box_mesh_from_half_size(half_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a triangle mesh for a box centered at the origin."""
    hx, hy, hz = [float(v) for v in half_size]
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return verts, faces


def _plane_mesh_from_scale(scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a finite quad mesh for a SAPIEN render plane."""
    sx, sy = float(scale[0]), float(scale[1])
    verts = np.array(
        [
            [0.0, -sx, -sy],
            [0.0, sx, -sy],
            [0.0, sx, sy],
            [0.0, -sx, sy],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


class MeshExtractionError(RuntimeError):
    """Raised when a SAPIEN visual shape cannot be converted to mesh data."""


class TextureExportContext:
    """Export texture images beside a USD file and return USD-relative paths."""

    def __init__(self, usd_dir: Path, asset_dir_name: str = "textures") -> None:
        self.usd_dir = Path(usd_dir)
        self.asset_dir_name = asset_dir_name
        self.texture_dir = self.usd_dir / asset_dir_name
        self._used_names: set[str] = set()

    def export_image(self, image: Any, name_hint: str, *, extension: str = ".png") -> str | None:
        if image is None:
            return None
        safe_stem = _safe_texture_stem(name_hint)
        ext = extension if extension.startswith(".") else f".{extension}"
        filename = self._unique_filename(f"{safe_stem}{ext.lower()}")
        out_path = self.texture_dir / filename
        self.texture_dir.mkdir(parents=True, exist_ok=True)
        try:
            _save_texture_image(image, out_path)
        except Exception:
            logger.debug("Failed to export texture image %s", out_path, exc_info=True)
            return None
        return f"{self.asset_dir_name}/{filename}"

    def copy_file(self, source_path: Path, name_hint: str) -> str | None:
        if not source_path.is_file():
            return None
        suffix = source_path.suffix.lower()
        if suffix not in _TEXTURE_EXTENSIONS:
            suffix = ".png"
        filename = self._unique_filename(f"{_safe_texture_stem(name_hint)}{suffix}")
        out_path = self.texture_dir / filename
        self.texture_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_path, out_path)
        except Exception:
            logger.debug("Failed to copy texture file %s", source_path, exc_info=True)
            return None
        return f"{self.asset_dir_name}/{filename}"

    def _unique_filename(self, filename: str) -> str:
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".png"
        candidate = f"{stem}{suffix}"
        index = 2
        while candidate in self._used_names:
            candidate = f"{stem}_{index}{suffix}"
            index += 1
        self._used_names.add(candidate)
        return candidate


def _scaled_vertices(points: np.ndarray, shape: Any) -> np.ndarray:
    scale = getattr(shape, "scale", None)
    if scale is None:
        return points
    scale_arr = np.asarray(scale, dtype=np.float64)
    if scale_arr.shape != (3,):
        raise MeshExtractionError(f"Invalid render shape scale {scale_arr!r}")
    return points * scale_arr[None, :]


def _trimesh_primitive_mesh(kind: str, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Create a primitive mesh via trimesh when SAPIEN exposes shape params only."""
    try:
        import trimesh
    except ImportError as exc:
        raise MeshExtractionError("trimesh is required for primitive mesh export") from exc

    try:
        if kind == "icosphere":
            mesh = trimesh.creation.icosphere(**kwargs)
        elif kind == "capsule":
            mesh = trimesh.creation.capsule(**kwargs)
        elif kind == "cylinder":
            mesh = trimesh.creation.cylinder(**kwargs)
        else:
            raise MeshExtractionError(f"Unsupported trimesh primitive: {kind}")
    except Exception as exc:
        raise MeshExtractionError(
            f"Failed to create trimesh primitive {kind}: {exc}"
        ) from exc

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        raise MeshExtractionError(f"Empty trimesh primitive: {kind}")
    return verts, faces


_MATERIAL_FIELD_HINTS = (
    "base",
    "color",
    "colour",
    "diffuse",
    "emiss",
    "metal",
    "normal",
    "opacity",
    "rough",
    "specular",
    "texture",
    "transmission",
    "alpha",
)


def _jsonable_material_value(value: Any) -> Any | None:
    """Convert a material attribute to a compact JSON-safe value."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        if len(value) <= 32:
            out = []
            for item in value:
                converted = _jsonable_material_value(item)
                if converted is None and item is not None:
                    return None
                out.append(converted)
            return out
        return {"type": type(value).__name__, "length": len(value)}

    for attr_name in ("filename", "file_path", "path", "uri", "name"):
        try:
            attr_value = getattr(value, attr_name)
        except Exception:
            continue
        converted = _jsonable_material_value(attr_value)
        if converted is not None:
            return {attr_name: converted, "type": type(value).__name__}

    return None


def _material_getter_value(material: Any, name: str) -> Any | None:
    """Read a material attribute or SAPIEN-style zero-argument getter."""
    try:
        value = getattr(material, name)
    except Exception:
        value = None
    if value is not None and not callable(value):
        return value

    try:
        getter = getattr(material, f"get_{name}")
    except Exception:
        return None
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _safe_texture_stem(value: Any, fallback: str = "texture") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("._-")
    if not safe:
        safe = fallback
    if safe[0].isdigit():
        safe = f"_{safe}"
    return safe[:120]


def _download_texture_image(texture: Any) -> Any | None:
    download = getattr(texture, "download", None)
    if not callable(download):
        return None
    try:
        return download()
    except Exception:
        return None


def _save_texture_image(image: Any, path: Path) -> None:
    save = getattr(image, "save", None)
    if callable(save) and not isinstance(image, np.ndarray):
        save(str(path))
        return

    if not _IMAGEIO_OK:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("imageio or Pillow is required to export textures") from exc
        Image.fromarray(_texture_image_to_uint8(image)).save(str(path))
        return

    imageio.imwrite(str(path), _texture_image_to_uint8(image))


def _texture_image_to_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 0:
        raise ValueError("Texture image is scalar")
    if arr.ndim == 3 and arr.shape[-1] > 4:
        arr = arr[..., :4]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _image_representative_color(image: Any) -> tuple[float, float, float] | None:
    arr = np.asarray(image)
    if arr.ndim < 3 or arr.shape[-1] < 3 or arr.size == 0:
        return None

    rgb = arr[..., :3].astype(np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    if arr.shape[-1] >= 4:
        alpha = arr[..., 3].astype(np.float32)
        if alpha.max(initial=0.0) > 1.0:
            alpha = alpha / 255.0
        mask = alpha > 0.05
    else:
        mask = np.ones(rgb.shape[:2], dtype=bool)

    if not bool(mask.any()):
        return None
    pixels = rgb[mask]
    if pixels.size == 0:
        return None

    color = np.clip(pixels.mean(axis=0), 0.0, 1.0)
    return tuple(float(v) for v in color[:3])


def _image_metadata(image: Any | None) -> dict[str, Any]:
    if image is None:
        return {}
    info: dict[str, Any] = {"image_type": type(image).__name__}
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        info["width"] = int(size[0])
        info["height"] = int(size[1])
    mode = getattr(image, "mode", None)
    if mode:
        info["mode"] = str(mode)
    arr = np.asarray(image)
    if arr.size:
        if "height" not in info and arr.ndim >= 2:
            info["height"] = int(arr.shape[0])
            info["width"] = int(arr.shape[1])
        if arr.ndim >= 3:
            info["channels"] = int(arr.shape[-1])
        info["dtype"] = str(arr.dtype)
    return info


def _texture_representative_color(
    texture: Any,
    image: Any | None = None,
) -> tuple[float, float, float] | None:
    """Estimate a compact albedo color for embedded SAPIEN textures.

    GLB textures imported by SAPIEN commonly arrive as RenderTexture2D objects
    with an empty filename.  USDC consumers still need a displayColor fallback,
    so we download the texture array and keep a representative RGB.
    """
    if image is None:
        image = _download_texture_image(texture)
    if image is None:
        return None
    return _image_representative_color(image)


def _texture_metadata(
    texture: Any | None,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_name_hint: str | None = None,
) -> dict[str, Any] | None:
    """Summarize a texture without embedding image bytes in material metadata."""
    if texture is None:
        return None

    info: dict[str, Any] = {"type": type(texture).__name__}
    for attr_name in ("filename", "format", "width", "height", "channels"):
        converted = _jsonable_material_value(_material_getter_value(texture, attr_name))
        if converted not in (None, ""):
            info[attr_name] = converted

    image = _download_texture_image(texture)
    if image is not None:
        info.update(_image_metadata(image))
        if texture_exporter is not None and texture_name_hint:
            exported = texture_exporter.export_image(image, texture_name_hint)
            if exported:
                info["file"] = exported

    representative = _texture_representative_color(texture, image=image)
    if representative is not None:
        info["representative_color"] = representative
    return info


def _numeric_tuple(value: Any, *, max_len: int = 4) -> tuple[float, ...] | None:
    converted = _jsonable_material_value(value)
    if isinstance(converted, (int, float)):
        return (float(converted),)
    if not isinstance(converted, list):
        return None
    if len(converted) > max_len:
        return None
    out: list[float] = []
    for item in converted:
        if not isinstance(item, (int, float)):
            return None
        out.append(float(item))
    return tuple(out)


def _material_property(material: Any, name: str) -> Any | None:
    return _material_getter_value(material, name)


def _extract_material_info(
    material: Any | None,
    *,
    default_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> dict[str, Any]:
    """Capture renderer material fields without depending on one SAPIEN version."""
    info: dict[str, Any] = {
        "base_color": tuple(float(v) for v in default_color),
        "source_type": type(material).__name__ if material is not None else None,
        "properties": {},
        "textures": {},
    }
    if material is None:
        return info

    base_color = _numeric_tuple(_material_property(material, "base_color"))
    if base_color and len(base_color) >= 3:
        info["base_color"] = tuple(float(v) for v in base_color[:3])
        if len(base_color) >= 4:
            info["alpha"] = float(base_color[3])

    texture_cache: dict[int, dict[str, Any] | None] = {}

    def metadata_for_texture(texture_name: str, texture_value: Any) -> dict[str, Any] | None:
        if texture_value is None:
            return None
        cache_key = id(texture_value)
        if cache_key in texture_cache:
            cached = texture_cache[cache_key]
            return dict(cached) if isinstance(cached, dict) else None
        hint = f"{texture_prefix}_{texture_name}" if texture_prefix else texture_name
        texture_info = _texture_metadata(
            texture_value,
            texture_exporter=texture_exporter,
            texture_name_hint=hint,
        )
        texture_cache[cache_key] = dict(texture_info) if isinstance(texture_info, dict) else None
        return texture_info

    for texture_name in (
        "base_color_texture",
        "diffuse_texture",
        "albedo_texture",
        "baseColorTexture",
    ):
        texture_info = metadata_for_texture(
            texture_name,
            _material_property(material, texture_name),
        )
        if texture_info is None:
            continue
        info["textures"][texture_name] = texture_info
        representative = texture_info.get("representative_color")
        if representative is not None:
            info["base_color"] = tuple(
                float(base) * float(tex)
                for base, tex in zip(info["base_color"][:3], representative[:3])
            )
            info["texture_representative_color"] = tuple(
                float(value) for value in representative[:3]
            )
        break

    for scalar_name in ("roughness", "metallic", "metalness", "specular", "opacity"):
        value = _numeric_tuple(_material_property(material, scalar_name), max_len=1)
        if value:
            canonical = "metallic" if scalar_name == "metalness" else scalar_name
            info[canonical] = float(value[0])

    for color_name in ("emission", "emission_color", "emissive", "diffuse_color"):
        value = _numeric_tuple(_material_property(material, color_name))
        if value and len(value) >= 3:
            canonical = "emissive_color" if "emiss" in color_name else color_name
            info[canonical] = tuple(float(v) for v in value[:3])

    properties: dict[str, Any] = {}
    textures: dict[str, Any] = dict(info.get("textures") or {})
    for attr_name in dir(material):
        if attr_name.startswith("_"):
            continue
        lower = attr_name.lower()
        if not any(hint in lower for hint in _MATERIAL_FIELD_HINTS):
            continue
        value = _material_property(material, attr_name)
        converted = _jsonable_material_value(value)
        if converted is None and value is not None:
            converted = {"type": type(value).__name__}
        if converted is None:
            continue
        properties[attr_name] = converted
        if "texture" in lower or lower in {"normal_map", "albedo_map"}:
            if attr_name not in textures:
                textures[attr_name] = metadata_for_texture(attr_name, value) or converted

    info["properties"] = properties
    info["textures"] = textures
    return info


def _mesh_like_uvs(mesh_like: Any) -> np.ndarray | None:
    """Best-effort lookup for per-vertex UV coordinates on SAPIEN mesh objects."""
    for attr_name in ("uvs", "uv", "texcoords", "tex_coords", "texture_coordinates"):
        try:
            value = getattr(mesh_like, attr_name)
        except Exception:
            continue
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.size:
            return arr[:, :2]
    for method_name in (
        "get_vertex_uv",
        "get_vertex_uvs",
        "get_uv",
        "get_uvs",
        "get_texcoords",
        "get_texture_coordinates",
    ):
        try:
            method = getattr(mesh_like, method_name)
        except Exception:
            continue
        if not callable(method):
            continue
        try:
            value = method()
        except Exception:
            continue
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.size:
            return arr[:, :2]
    return None


def _extract_part_material(part: Any) -> Any | None:
    material = getattr(part, "material", None)
    if material is None and hasattr(part, "get_material"):
        try:
            material = part.get_material()
        except Exception:
            material = None
    return material


def _extract_render_shape_material(shape: Any) -> Any | None:
    """Best-effort access to the first material carried by a render shape."""
    parts = getattr(shape, "parts", None)
    if parts:
        for part in parts:
            material = _extract_part_material(part)
            if material is not None:
                return material
    return None


def _shape_source_mesh_path(shape: Any) -> Path | None:
    """Resolve a SAPIEN render shape source mesh path when it is available."""
    filename = None
    for attr_name in ("filename",):
        try:
            value = getattr(shape, attr_name)
        except Exception:
            continue
        if value:
            filename = value
            break
    if filename is None:
        try:
            getter = getattr(shape, "get_filename")
        except Exception:
            getter = None
        if callable(getter):
            try:
                filename = getter()
            except Exception:
                filename = None
    if not filename:
        return None
    path = Path(str(filename))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _extract_source_mesh_geometry_parts(
    shape: Any,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> list[dict[str, Any]] | None:
    """Load original visual mesh data for file-backed SAPIEN triangle meshes.

    SAPIEN can expose the correct runtime vertices while its downloaded
    RenderTexture2D is not the best source of material semantics for USD.
    When a render shape points back to a GLB/DAE, prefer that file for
    geometry, UVs, and PBR textures, then apply the SAPIEN shape scale.
    """
    source_mesh = _shape_source_mesh_path(shape)
    if source_mesh is None or not source_mesh.is_file():
        return None
    if source_mesh.suffix.lower() not in {".glb", ".gltf", ".dae", ".obj"}:
        return None
    try:
        parts = _load_mesh_geometry_parts(
            source_mesh,
            texture_exporter=texture_exporter,
            texture_prefix=f"{texture_prefix or source_mesh.stem}_source",
        )
    except MeshExtractionError:
        logger.debug(
            "Falling back to SAPIEN runtime mesh for %s",
            source_mesh,
            exc_info=True,
        )
        return None

    out: list[dict[str, Any]] = []
    type_name = type(shape).__name__
    for part_idx, part in enumerate(parts):
        out.append(
            {
                **part,
                "verts": _scaled_vertices(np.asarray(part["verts"], dtype=np.float64), shape),
                "source": {
                    **dict(part.get("source") or {}),
                    "shape_type": type_name,
                    "part_index": part_idx,
                    "source_mesh_loader": "trimesh_original_visual",
                },
            }
        )
    return out


def _extract_render_shape_geometry_parts(
    shape: Any,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Extract all exportable geometry/material parts from one render shape."""
    type_name = type(shape).__name__
    source_parts = _extract_source_mesh_geometry_parts(
        shape,
        texture_exporter=texture_exporter,
        texture_prefix=texture_prefix,
    )
    if source_parts:
        return source_parts

    parts = getattr(shape, "parts", None) or []
    out: list[dict[str, Any]] = []

    for part_idx, part in enumerate(parts):
        mesh = getattr(part, "mesh", None)
        if mesh is not None:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
            if verts.size == 0 or faces.size == 0:
                raise MeshExtractionError(
                    f"Empty mesh part for render shape {type(shape).__name__}"
                )
            out.append(
                {
                    "verts": _scaled_vertices(verts, shape),
                    "faces": faces,
                    "uvs": _mesh_like_uvs(mesh),
                    "material": _extract_material_info(
                        _extract_part_material(part),
                        texture_exporter=texture_exporter,
                        texture_prefix=f"{texture_prefix or type_name}_part{part_idx:03d}",
                    ),
                    "source": {
                        "shape_type": type_name,
                        "part_type": type(part).__name__,
                        "part_index": part_idx,
                    },
                }
            )
            continue

        if (
            type(part).__name__ == "RenderShapeTriangleMeshPart"
            and type_name == "RenderShapeTriangleMesh"
            and hasattr(part, "vertices")
            and hasattr(part, "triangles")
        ):
            verts = np.asarray(part.vertices, dtype=np.float64)
            faces = np.asarray(part.triangles, dtype=np.int32).reshape(-1, 3)
            if verts.size == 0 or faces.size == 0:
                raise MeshExtractionError(
                    f"Empty triangle mesh part for render shape {type(shape).__name__}"
                )
            out.append(
                {
                    "verts": _scaled_vertices(verts, shape),
                    "faces": faces,
                    "uvs": _mesh_like_uvs(part),
                    "material": _extract_material_info(
                        _extract_part_material(part),
                        texture_exporter=texture_exporter,
                        texture_prefix=f"{texture_prefix or type_name}_part{part_idx:03d}",
                    ),
                    "source": {
                        "shape_type": type_name,
                        "part_type": type(part).__name__,
                        "part_index": part_idx,
                    },
                }
            )

    if out:
        return out

    if type_name == "RenderShapeBox" and hasattr(shape, "half_size"):
        verts, faces = _box_mesh_from_half_size(np.asarray(shape.half_size, dtype=np.float64))
    elif type_name == "RenderShapePlane" and hasattr(shape, "scale"):
        verts, faces = _plane_mesh_from_scale(np.asarray(shape.scale, dtype=np.float64))
    elif type_name == "RenderShapeSphere" and hasattr(shape, "radius"):
        verts, faces = _trimesh_primitive_mesh("icosphere", subdivisions=2, radius=float(shape.radius))
    elif type_name == "RenderShapeCapsule":
        radius = getattr(shape, "radius", None)
        half_length = getattr(shape, "half_length", None)
        if radius is None or half_length is None:
            raise MeshExtractionError(f"Unsupported render shape type: {type_name}")
        verts, faces = _trimesh_primitive_mesh(
            "capsule",
            radius=float(radius),
            height=float(half_length) * 2.0,
            count=[16, 16],
        )
    elif type_name == "RenderShapeCylinder":
        radius = getattr(shape, "radius", None)
        half_length = getattr(shape, "half_length", getattr(shape, "length", None))
        if radius is None or half_length is None:
            raise MeshExtractionError(f"Unsupported render shape type: {type_name}")
        height = float(half_length) * (2.0 if hasattr(shape, "half_length") else 1.0)
        verts, faces = _trimesh_primitive_mesh(
            "cylinder",
            radius=float(radius),
            height=height,
            sections=24,
        )
    else:
        raise MeshExtractionError(f"Unsupported render shape type: {type_name}")

    return [
        {
            "verts": verts,
            "faces": faces,
            "uvs": None,
            "material": _extract_material_info(
                _extract_render_shape_material(shape),
                texture_exporter=texture_exporter,
                texture_prefix=f"{texture_prefix or type_name}_part000",
            ),
            "source": {"shape_type": type_name, "part_type": None, "part_index": 0},
        }
    ]


def _extract_render_shape_geometry(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract one render shape as vertices/faces in shape-local coordinates."""
    first_part = _extract_render_shape_geometry_parts(shape)[0]
    return first_part["verts"], first_part["faces"]


def _extract_entity_mesh(
    entity: Any,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> dict:
    """Extract visual mesh data from a SAPIEN actor or link entity.

    Returns ``{"verts": Nx3, "faces": Mx3, "color": (r,g,b)}``.
    """
    try:
        import sapien
        render_owner = entity
        if not hasattr(render_owner, "find_component_by_type"):
            render_owner = getattr(entity, "entity", None)
        if render_owner is None or not hasattr(render_owner, "find_component_by_type"):
            raise MeshExtractionError(
                f"Entity {getattr(entity, 'get_name', lambda: '<entity>')()} "
                "does not expose SAPIEN components"
            )

        visual_bodies = render_owner.find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        if visual_bodies is None:
            raise MeshExtractionError(
                f"Entity {getattr(entity, 'get_name', lambda: '<entity>')()} "
                "has no RenderBodyComponent"
            )

        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        mesh_parts: list[dict[str, Any]] = []
        color = (0.7, 0.7, 0.7)
        vert_offset = 0

        for shape_idx, shape in enumerate(visual_bodies.render_shapes):
            local_pose = getattr(shape, "local_pose", None)
            shape_prefix = f"{texture_prefix or 'entity'}_shape{shape_idx:03d}"
            for part_idx, part in enumerate(
                _extract_render_shape_geometry_parts(
                    shape,
                    texture_exporter=texture_exporter,
                    texture_prefix=shape_prefix,
                )
            ):
                verts = part["verts"]
                faces = part["faces"]
                uvs = part.get("uvs")
                if local_pose is not None:
                    verts = _apply_sapien_pose(verts, local_pose)

                material_info = dict(part.get("material") or {})
                part_color = material_info.get("base_color") or color
                color = tuple(float(v) for v in part_color[:3])

                mesh_parts.append(
                    {
                        "verts": verts,
                        "faces": faces,
                        "uvs": uvs,
                        "color": color,
                        "material": material_info,
                        "source": {
                            **dict(part.get("source") or {}),
                            "render_shape_index": shape_idx,
                            "shape_part_index": part_idx,
                        },
                    }
                )
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

        if not all_verts:
            raise MeshExtractionError(
                f"Entity {getattr(entity, 'get_name', lambda: '<entity>')()} "
                "has a RenderBodyComponent but no exportable render shapes"
            )
        return {
            "verts": np.concatenate(all_verts),
            "faces": np.concatenate(all_faces),
            "color": color,
            "parts": mesh_parts,
        }
    except MeshExtractionError:
        raise
    except Exception as exc:
        name = getattr(entity, "get_name", lambda: "<entity>")()
        raise MeshExtractionError(f"Failed to extract mesh from {name}: {exc}") from exc


def _get_robot_links(task_env: Any) -> list[Any]:
    """Discover RoboTwin robot links across the wrapper variants used in practice."""
    robot = getattr(task_env, "robot", None)
    if robot is None:
        raise MeshExtractionError("Task environment has no robot")

    articulations: list[Any] = []
    for attr in ("left_entity", "right_entity", "_entity", "robot"):
        entity = getattr(robot, attr, None)
        if entity is None:
            continue
        if any(entity is existing for existing in articulations):
            continue
        articulations.append(entity)

    links: list[Any] = []
    seen: set[int] = set()
    for articulation in articulations:
        get_links = getattr(articulation, "get_links", None)
        if get_links is None:
            continue
        try:
            art_links = list(get_links())
        except Exception as exc:
            raise MeshExtractionError(
                f"Failed to get links from robot articulation {articulation}: {exc}"
            ) from exc
        for link in art_links:
            ident = id(link)
            if ident in seen:
                continue
            seen.add(ident)
            links.append(link)
    if not links:
        raise MeshExtractionError("No robot links discovered from task environment")
    return links


def _parse_urdf_vector(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Parse a URDF xyz/rpy/scale vector string into a 3-tuple."""
    if not text:
        return default
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return default


def _parse_urdf_rgba(text: str | None) -> tuple[float, float, float, float] | None:
    """Parse a URDF color vector, returning None for absent or malformed values."""
    if not text:
        return None
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) not in (3, 4):
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 3:
        values.append(1.0)
    return (values[0], values[1], values[2], values[3])


def _extract_urdf_visual_material(
    visual_elem: Any,
    *,
    urdf_dir: Path | None = None,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> dict[str, Any]:
    """Capture URDF visual material color/texture hints when present."""
    info = _extract_material_info(None, default_color=(0.72, 0.72, 0.74))
    material_elem = visual_elem.find("material")
    if material_elem is None:
        return info

    info.setdefault("properties", {})["urdf_material_present"] = True
    name = material_elem.get("name")
    if name:
        info["name"] = name
    color_elem = material_elem.find("color")
    rgba = _parse_urdf_rgba(color_elem.get("rgba") if color_elem is not None else None)
    if rgba is not None:
        info["base_color"] = tuple(float(v) for v in rgba[:3])
        info["alpha"] = float(rgba[3])
        info.setdefault("properties", {})["urdf_color_present"] = True
    texture_elem = material_elem.find("texture")
    filename = texture_elem.get("filename") if texture_elem is not None else None
    if filename:
        texture_info: dict[str, Any] = {"filename": filename}
        if urdf_dir is not None and texture_exporter is not None:
            resolved = (urdf_dir / filename).resolve()
            exported = texture_exporter.copy_file(
                resolved,
                f"{texture_prefix or 'urdf'}_texture",
            )
            if exported:
                texture_info["file"] = exported
        info.setdefault("textures", {})["urdf_texture"] = texture_info
        info.setdefault("properties", {})["urdf_texture_present"] = True
    info["source_type"] = "urdf_visual_material"
    return info


def _numeric_color_to_unit_rgb(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 3:
        return None
    rgb = arr[:3]
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return tuple(float(v) for v in rgb)


def _numeric_alpha_to_unit(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 4:
        return None
    alpha = float(arr[3])
    if alpha > 1.0:
        alpha = alpha / 255.0
    return float(np.clip(alpha, 0.0, 1.0))


def _image_texture_metadata(
    image: Any | None,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_name_hint: str | None = None,
) -> dict[str, Any] | None:
    if image is None:
        return None
    info: dict[str, Any] = {"type": type(image).__name__, **_image_metadata(image)}
    representative = _image_representative_color(image)
    if representative is not None:
        info["representative_color"] = representative
    if texture_exporter is not None and texture_name_hint:
        exported = texture_exporter.export_image(image, texture_name_hint)
        if exported:
            info["file"] = exported
    return info


def _extract_trimesh_material_info(
    visual: Any,
    mesh_path: Path,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> dict[str, Any]:
    """Extract material/texture hints from a trimesh visual object."""
    material = getattr(visual, "material", None)
    info = _extract_material_info(None, default_color=(0.72, 0.72, 0.74))
    info["source_type"] = type(material).__name__ if material is not None else type(visual).__name__
    info["properties"] = {
        "source_mesh": str(mesh_path),
        "source_visual_type": type(visual).__name__,
    }

    color = None
    alpha = None
    if material is not None:
        for attr_name in ("baseColorFactor", "main_color", "diffuse", "ambient"):
            try:
                value = getattr(material, attr_name)
            except Exception:
                continue
            color = _numeric_color_to_unit_rgb(value)
            alpha = _numeric_alpha_to_unit(value)
            if color is not None:
                info["properties"]["source_color_attr"] = attr_name
                break
        name = getattr(material, "name", None)
        if name:
            info["name"] = str(name)
            info["properties"]["material_name"] = str(name)

    if color is None:
        for attr_name in ("main_color", "vertex_colors", "face_colors"):
            try:
                value = getattr(visual, attr_name)
            except Exception:
                continue
            arr = np.asarray(value) if value is not None else np.asarray([])
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                value = np.mean(arr.reshape(-1, arr.shape[-1]), axis=0)
            color = _numeric_color_to_unit_rgb(value)
            alpha = _numeric_alpha_to_unit(value)
            if color is not None:
                info["properties"]["source_color_attr"] = attr_name
                break

    if color is not None:
        info["base_color"] = color
    if alpha is not None:
        info["alpha"] = alpha

    textures: dict[str, Any] = {}
    if material is not None:
        for attr_name in ("baseColorTexture", "image"):
            try:
                image = getattr(material, attr_name)
            except Exception:
                continue
            texture_info = _image_texture_metadata(
                image,
                texture_exporter=texture_exporter,
                texture_name_hint=f"{texture_prefix or mesh_path.stem}_{attr_name}",
            )
            if texture_info is None:
                continue
            texture_key = "base_color_texture" if attr_name == "baseColorTexture" else attr_name
            textures[texture_key] = texture_info
            representative = texture_info.get("representative_color")
            if representative is not None:
                base = info.get("base_color") or (1.0, 1.0, 1.0)
                info["base_color"] = tuple(
                    float(base_value) * float(tex_value)
                    for base_value, tex_value in zip(base[:3], representative[:3])
                )
                info["texture_representative_color"] = tuple(
                    float(value) for value in representative[:3]
                )
            break

    info["textures"] = textures
    return info


def _trimesh_visual_uvs(visual: Any) -> np.ndarray | None:
    for attr_name in ("uv", "uvs"):
        try:
            value = getattr(visual, attr_name)
        except Exception:
            continue
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.size:
            return arr[:, :2]
    return None


def _apply_transform_matrix(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        return points
    homo = np.ones((points.shape[0], 4), dtype=np.float64)
    homo[:, :3] = points
    return (homo @ transform.T)[:, :3]


def _iter_trimesh_scene_geometry(loaded: Any) -> list[tuple[str, Any, np.ndarray]]:
    if hasattr(loaded, "geometry"):
        geometry = getattr(loaded, "geometry", {}) or {}
        graph = getattr(loaded, "graph", None)
        items: list[tuple[str, Any, np.ndarray]] = []
        nodes = list(getattr(graph, "nodes_geometry", []) or []) if graph is not None else []
        for node_name in nodes:
            try:
                transform, geom_name = graph.get(frame_to=node_name)
            except Exception:
                continue
            geom = geometry.get(geom_name)
            if geom is None:
                continue
            items.append((str(node_name), geom, np.asarray(transform, dtype=np.float64)))
        if items:
            return items
        return [
            (str(name), geom, np.eye(4, dtype=np.float64))
            for name, geom in geometry.items()
        ]
    return [(Path(str(getattr(loaded, "metadata", {}).get("file_name", "mesh"))).stem, loaded, np.eye(4, dtype=np.float64))]


def _load_trimesh_geometry_parts(
    mesh_path: Path,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> list[dict[str, Any]]:
    try:
        import trimesh
    except ImportError as exc:
        raise MeshExtractionError(
            f"trimesh is required to load robot mesh materials from {mesh_path}"
        ) from exc

    try:
        loaded = trimesh.load(str(mesh_path), force="scene")
    except Exception as exc:
        raise MeshExtractionError(f"Failed to load mesh {mesh_path}: {exc}") from exc

    parts: list[dict[str, Any]] = []
    for geom_idx, (geom_name, geom, transform) in enumerate(_iter_trimesh_scene_geometry(loaded)):
        if not hasattr(geom, "vertices") or not hasattr(geom, "faces"):
            continue
        verts = np.asarray(geom.vertices, dtype=np.float64)
        faces = np.asarray(geom.faces, dtype=np.int32)
        if verts.size == 0 or faces.size == 0:
            continue
        verts = _apply_transform_matrix(verts, transform)
        visual = getattr(geom, "visual", None)
        material_info = _extract_trimesh_material_info(
            visual,
            mesh_path,
            texture_exporter=texture_exporter,
            texture_prefix=f"{texture_prefix or mesh_path.stem}_{geom_idx:03d}",
        )
        color = tuple(float(v) for v in material_info.get("base_color", (0.72, 0.72, 0.74))[:3])
        parts.append(
            {
                "verts": verts,
                "faces": faces.reshape(-1, 3),
                "uvs": _trimesh_visual_uvs(visual),
                "color": color,
                "material": material_info,
                "source": {
                    "source_mesh": str(mesh_path),
                    "geometry_name": geom_name,
                    "geometry_index": geom_idx,
                },
            }
        )

    if not parts:
        raise MeshExtractionError(f"Mesh scene {mesh_path} has no geometry")
    return parts


def _load_mesh_geometry_parts(
    mesh_path: Path,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Load mesh parts with best-effort per-part material and UV metadata."""
    try:
        return _load_trimesh_geometry_parts(
            mesh_path,
            texture_exporter=texture_exporter,
            texture_prefix=texture_prefix,
        )
    except MeshExtractionError:
        logger.debug("Falling back to geometry-only mesh loader for %s", mesh_path, exc_info=True)

    verts, faces = _load_mesh_geometry(mesh_path)
    material = _extract_material_info(None, default_color=(0.72, 0.72, 0.74))
    material.setdefault("properties", {})["source_mesh"] = str(mesh_path)
    return [
        {
            "verts": verts,
            "faces": faces,
            "uvs": None,
            "color": tuple(float(v) for v in material["base_color"][:3]),
            "material": material,
            "source": {"source_mesh": str(mesh_path), "geometry_index": 0},
        }
    ]


def _merge_mesh_and_urdf_material_info(
    mesh_material: dict[str, Any],
    urdf_material: dict[str, Any],
) -> dict[str, Any]:
    if urdf_material.get("source_type") != "urdf_visual_material":
        return dict(mesh_material)

    merged = dict(mesh_material)
    merged["properties"] = {
        **dict(mesh_material.get("properties") or {}),
        **dict(urdf_material.get("properties") or {}),
    }
    merged["textures"] = {
        **dict(mesh_material.get("textures") or {}),
        **dict(urdf_material.get("textures") or {}),
    }
    if "name" in urdf_material:
        merged["name"] = urdf_material["name"]
    if urdf_material.get("properties", {}).get("urdf_color_present"):
        merged["base_color"] = urdf_material.get("base_color", merged.get("base_color"))
        if "alpha" in urdf_material:
            merged["alpha"] = urdf_material["alpha"]
    if merged.get("source_type") and merged["source_type"] != "urdf_visual_material":
        merged["source_type"] = f"{merged['source_type']}+urdf_visual_material"
    else:
        merged["source_type"] = "urdf_visual_material"
    return merged


def _load_collada_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Collada mesh while respecting its scene graph transforms."""
    try:
        import collada
    except ImportError as exc:
        raise MeshExtractionError(
            f"pycollada is required to load Collada robot mesh {mesh_path}"
        ) from exc

    try:
        doc = collada.Collada(str(mesh_path))
    except Exception as exc:
        raise MeshExtractionError(f"Failed to parse Collada mesh {mesh_path}: {exc}") from exc

    bound_geometries: list[Any] = []
    if getattr(doc, "scene", None) is not None:
        try:
            bound_geometries = list(doc.scene.objects("geometry"))
        except Exception as exc:
            raise MeshExtractionError(
                f"Failed to walk Collada scene {mesh_path}: {exc}"
            ) from exc

    if not bound_geometries:
        for geom in getattr(doc, "geometries", []):
            bind = getattr(geom, "bind", None)
            if bind is None:
                continue
            try:
                bound_geometries.append(geom.bind(np.eye(4, dtype=np.float64), {}))
            except Exception as exc:
                raise MeshExtractionError(
                    f"Failed to bind Collada geometry from {mesh_path}: {exc}"
                ) from exc

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for bound_geom in bound_geometries:
        try:
            primitives = list(bound_geom.primitives())
        except Exception as exc:
            raise MeshExtractionError(
                f"Failed to iterate Collada primitives from {mesh_path}: {exc}"
            ) from exc

        for primitive in primitives:
            tri_primitive = primitive
            if hasattr(tri_primitive, "triangleset"):
                try:
                    tri_primitive = tri_primitive.triangleset()
                except Exception as exc:
                    raise MeshExtractionError(
                        f"Failed to triangulate Collada primitive from {mesh_path}: {exc}"
                    ) from exc

            verts = getattr(tri_primitive, "vertex", None)
            faces = getattr(tri_primitive, "vertex_index", None)
            if verts is None or faces is None:
                raise MeshExtractionError(
                    f"Collada primitive from {mesh_path} has no vertices/faces"
                )

            verts = np.asarray(verts, dtype=np.float64)
            faces = np.asarray(faces, dtype=np.int32)
            if verts.size == 0 or faces.size == 0:
                raise MeshExtractionError(f"Collada primitive from {mesh_path} is empty")

            faces = faces.reshape(-1, 3)
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            vert_offset += len(verts)

    if not all_verts:
        raise MeshExtractionError(f"Collada mesh {mesh_path} has no exportable geometry")

    return np.concatenate(all_verts), np.concatenate(all_faces)


def _load_mesh_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one mesh file into vertices/faces arrays."""
    if mesh_path.suffix.lower() == ".dae":
        return _load_collada_geometry(mesh_path)

    try:
        import trimesh
    except ImportError as exc:
        raise MeshExtractionError(
            f"trimesh is required to load robot mesh {mesh_path}"
        ) from exc

    try:
        loaded = trimesh.load(str(mesh_path), force="scene")
    except Exception as exc:
        raise MeshExtractionError(f"Failed to load mesh {mesh_path}: {exc}") from exc

    if hasattr(loaded, "geometry"):
        geometries = [
            geom for geom in loaded.geometry.values()
            if hasattr(geom, "vertices") and hasattr(geom, "faces")
        ]
        if not geometries:
            raise MeshExtractionError(f"Mesh scene {mesh_path} has no geometry")
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise MeshExtractionError(f"Mesh {mesh_path} has no vertices/faces")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        raise MeshExtractionError(f"Mesh {mesh_path} is empty")
    return verts, faces


def _mesh_extent_is_reasonable(
    verts: np.ndarray,
    *,
    max_abs_coord: float = 5.0,
    max_span: float = 5.0,
) -> bool:
    """Reject obviously broken robot meshes with absurd local coordinates."""
    if verts.size == 0:
        return False
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    span = np.max(maxs - mins)
    abs_coord = np.max(np.abs(verts))
    return bool(abs_coord <= max_abs_coord and span <= max_span)


def _load_urdf_link_meshes(
    urdf_path: Path,
    *,
    texture_exporter: TextureExportContext | None = None,
    texture_prefix: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load visual meshes for each URDF link, baked into link-local coordinates."""
    if not urdf_path.is_file():
        raise MeshExtractionError(f"URDF not found for robot mesh export: {urdf_path}")

    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        raise MeshExtractionError(f"Failed to parse URDF {urdf_path}: {exc}") from exc

    out: dict[str, dict[str, Any]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        if not link_name:
            raise MeshExtractionError(f"URDF {urdf_path} contains a link without name")

        all_verts: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        mesh_parts: list[dict[str, Any]] = []
        vert_offset = 0

        for visual_idx, visual_elem in enumerate(link_elem.findall("visual")):
            mesh_elem = visual_elem.find("./geometry/mesh")
            if mesh_elem is None:
                if visual_elem.find("geometry") is not None:
                    raise MeshExtractionError(
                        f"Unsupported non-mesh visual geometry in URDF link {link_name}"
                    )
                continue

            filename = mesh_elem.get("filename") or mesh_elem.get("url")
            if not filename:
                raise MeshExtractionError(
                    f"URDF visual mesh in link {link_name} has no filename/url"
                )
            mesh_path = (urdf_path.parent / filename).resolve()
            part_prefix = (
                f"{texture_prefix or 'robot'}_{link_name}_visual{visual_idx:03d}_{mesh_path.stem}"
            )
            loaded_parts = _load_mesh_geometry_parts(
                mesh_path,
                texture_exporter=texture_exporter,
                texture_prefix=part_prefix,
            )

            scale = np.asarray(
                _parse_urdf_vector(mesh_elem.get("scale"), (1.0, 1.0, 1.0)),
                dtype=np.float64,
            )
            origin_elem = visual_elem.find("origin")
            xyz = _parse_urdf_vector(
                origin_elem.get("xyz") if origin_elem is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_urdf_vector(
                origin_elem.get("rpy") if origin_elem is not None else None,
                (0.0, 0.0, 0.0),
            )
            R = _euler_xyz_to_rot(rpy)
            urdf_material_info = _extract_urdf_visual_material(
                visual_elem,
                urdf_dir=urdf_path.parent,
                texture_exporter=texture_exporter,
                texture_prefix=part_prefix,
            )

            for local_part_idx, mesh_part in enumerate(loaded_parts):
                verts = np.asarray(mesh_part["verts"], dtype=np.float64)
                faces = np.asarray(mesh_part["faces"], dtype=np.int32)

                if not _mesh_extent_is_reasonable(verts):
                    raise MeshExtractionError(
                        "Anomalous robot mesh "
                        f"{mesh_path} for link {link_name} "
                        f"(absmax={float(np.max(np.abs(verts))):.3f}, "
                        f"span={float(np.max(np.max(verts, axis=0) - np.min(verts, axis=0))):.3f})"
                    )

                verts = verts * scale[None, :]
                verts = verts @ R.T + np.asarray(xyz, dtype=np.float64)

                material_info = _merge_mesh_and_urdf_material_info(
                    dict(mesh_part.get("material") or {}),
                    urdf_material_info,
                )
                material_info.setdefault("properties", {})["source_mesh"] = str(mesh_path)
                material_info.setdefault("properties", {})["visual_index"] = visual_idx
                material_info.setdefault("properties", {})["mesh_part_index"] = local_part_idx
                color = tuple(float(v) for v in material_info.get("base_color", (0.72, 0.72, 0.74))[:3])
                mesh_parts.append(
                    {
                        "verts": verts,
                        "faces": faces,
                        "uvs": mesh_part.get("uvs"),
                        "color": color,
                        "material": material_info,
                        "source": {
                            **dict(mesh_part.get("source") or {}),
                            "source_mesh": str(mesh_path),
                            "visual_index": visual_idx,
                            "mesh_part_index": local_part_idx,
                            "link_name": link_name,
                        },
                    }
                )
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

        if all_verts:
            out[link_name] = {
                "verts": np.concatenate(all_verts),
                "faces": np.concatenate(all_faces),
                "color": mesh_parts[-1]["color"] if mesh_parts else (0.72, 0.72, 0.74),
                "parts": mesh_parts,
            }

    logger.info("Loaded %d URDF robot link meshes from %s", len(out), urdf_path)
    return out


def _load_urdf_link_metadata(urdf_path: Path) -> dict[str, dict[str, Any]]:
    """Return per-link renderability metadata from the robot URDF."""
    if not urdf_path.is_file():
        raise MeshExtractionError(f"URDF not found for robot link metadata: {urdf_path}")

    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        raise MeshExtractionError(f"Failed to parse URDF {urdf_path}: {exc}") from exc

    out: dict[str, dict[str, Any]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        if not link_name:
            raise MeshExtractionError(f"URDF {urdf_path} contains a link without name")

        visual_elems = link_elem.findall("visual")
        collision_elems = link_elem.findall("collision")
        inertial_elems = link_elem.findall("inertial")
        visual_mesh_count = 0
        for visual_elem in visual_elems:
            if visual_elem.find("./geometry/mesh") is not None:
                visual_mesh_count += 1

        renderable = visual_mesh_count > 0
        if renderable:
            semantic_role = "renderable_link"
        elif inertial_elems and not collision_elems:
            semantic_role = "inertial_frame"
        elif not visual_elems and not collision_elems and not inertial_elems:
            semantic_role = "root_frame" if link_name == "footprint" else "kinematic_frame"
        else:
            semantic_role = "non_renderable_link"

        out[link_name] = {
            "renderable": renderable,
            "has_geometry": renderable,
            "has_visual": bool(visual_elems),
            "visual_mesh_count": visual_mesh_count,
            "has_collision": bool(collision_elems),
            "has_inertial": bool(inertial_elems),
            "semantic_role": semantic_role,
        }
    return out


def _is_encoded_rgb_dataset(rgb_ds: Any) -> bool:
    dtype = getattr(rgb_ds, "dtype", None)
    if dtype is None:
        return False
    return dtype.kind in ("S", "V", "O")


def _decode_rgb_bytes(data: bytes) -> np.ndarray:
    import io

    try:
        import cv2

        encoded = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass

    if _IMAGEIO_OK:
        return np.asarray(imageio.imread(io.BytesIO(data)), dtype=np.uint8)

    raise RuntimeError(
        "No RGB frame decoder found. Install imageio, Pillow, or opencv-python."
    )


def _decode_hdf5_rgb_frames(rgb_ds: Any) -> np.ndarray:
    if _is_encoded_rgb_dataset(rgb_ds):
        frames: list[np.ndarray] = []
        for raw in rgb_ds:
            frame = _decode_rgb_bytes(bytes(raw))
            frames.append(np.asarray(frame, dtype=np.uint8))
        if not frames:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return np.stack(frames, axis=0)
    return np.asarray(rgb_ds, dtype=np.uint8)


def _camera_image_size_from_intrinsics(intrinsic: Any) -> tuple[int, int] | None:
    if intrinsic is None:
        return None
    arr = np.asarray(intrinsic, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != (3, 3):
        return None
    cx = float(arr[0, 2])
    cy = float(arr[1, 2])
    if cx <= 0.0 or cy <= 0.0:
        return None
    return int(round(cx * 2.0)), int(round(cy * 2.0))


def _read_hdf5_camera_observations(
    h5_path: Path,
    *,
    read_rgb: bool = False,
) -> dict[str, dict[str, Any]]:
    """Read RoboTwin camera metadata, optionally decoding HDF5 RGB frames."""
    import h5py

    cameras: dict[str, dict[str, Any]] = {}
    with h5py.File(str(h5_path), "r") as file_obj:
        obs = file_obj.get("observation")
        if obs is None:
            return cameras
        for cam_name in _CAMERA_NAMES:
            if cam_name not in obs:
                continue
            cam_grp = obs[cam_name]
            camera: dict[str, Any] = {}
            if "cam2world_gl" in cam_grp:
                camera["cam2world_gl"] = np.asarray(cam_grp["cam2world_gl"], dtype=np.float64)
            if "intrinsic_cv" in cam_grp:
                camera["intrinsic_cv"] = np.asarray(cam_grp["intrinsic_cv"], dtype=np.float64)
            if read_rgb and "rgb" in cam_grp:
                frames = _decode_hdf5_rgb_frames(cam_grp["rgb"])
                if frames.ndim == 4 and frames.shape[-1] == 3:
                    camera["rgb"] = frames
                    camera["image_size"] = (int(frames.shape[2]), int(frames.shape[1]))
            if "image_size" not in camera:
                image_size = _camera_image_size_from_intrinsics(camera.get("intrinsic_cv"))
                if image_size is not None:
                    camera["image_size"] = image_size
            if camera:
                cameras[cam_name] = camera
    return cameras


def _write_rgb_video(frames: np.ndarray, path: Path, fps: int) -> None:
    writer = _open_video_writer(path, fps=fps)
    try:
        for frame in np.asarray(frames, dtype=np.uint8):
            writer.append_data(frame)
    finally:
        writer.close()


def _write_render_manifest(
    renders_dir: Path,
    *,
    video_source: str,
    cameras: dict[str, dict[str, Any]],
    fps: int,
    camera_pose_source: str = "hdf5:observation/<camera>/cam2world_gl",
    camera_intrinsic_source: str = "hdf5:observation/<camera>/intrinsic_cv",
) -> None:
    manifest = {
        "video_source": video_source,
        "fps": fps,
        "camera_pose_source": camera_pose_source,
        "camera_intrinsic_source": camera_intrinsic_source,
        "cameras": {
            name: {
                "mp4": f"{name}.mp4",
                "usd_camera": f"/Scene/Cameras/{name}",
                "frames": int(data["rgb"].shape[0]) if "rgb" in data else None,
                "image_size": list(data["image_size"]) if "image_size" in data else None,
            }
            for name, data in sorted(cameras.items())
        },
    }
    (renders_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _write_hdf5_camera_videos(
    cameras: dict[str, dict[str, Any]],
    renders_dir: Path,
    *,
    fps: int,
) -> bool:
    """Write per-camera MP4s from HDF5 RGB observations.

    Returns True when at least one authoritative HDF5 camera video was written.
    """
    wrote_any = False
    camera_arrays: dict[str, np.ndarray] = {}
    for cam_name, data in cameras.items():
        frames = data.get("rgb")
        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
            continue
        _write_rgb_video(frames, renders_dir / f"{cam_name}.mp4", fps=fps)
        camera_arrays[cam_name] = frames
        wrote_any = True

    if not wrote_any:
        return False

    composite_frames = _make_composite_video(
        camera_arrays,
        ["front_camera", "head_camera", "left_camera", "right_camera"],
    )
    if composite_frames is not None:
        _write_rgb_video(composite_frames, renders_dir / "composite.mp4", fps=fps)
        _ensure_multiview_video(renders_dir)
    _write_render_manifest(
        renders_dir,
        video_source="hdf5_observation_rgb",
        cameras={name: data for name, data in cameras.items() if name in camera_arrays},
        fps=fps,
    )
    return True


def _make_composite_video(
    camera_arrays: dict[str, np.ndarray],
    cam_order: list[str],
) -> np.ndarray | None:
    available = [
        camera_arrays[c] for c in cam_order if c in camera_arrays
    ]
    if not available:
        return None

    T = min(a.shape[0] for a in available)
    H, W = available[0].shape[1], available[0].shape[2]
    blank = np.zeros((T, H, W, 3), dtype=np.uint8)
    slots = (available + [blank] * 4)[:4]
    top = np.concatenate([slots[0][:T], slots[1][:T]], axis=2)
    bottom = np.concatenate([slots[2][:T], slots[3][:T]], axis=2)
    return np.concatenate([top, bottom], axis=1)


def _capture_camera_frames(task_env: Any) -> dict[str, np.ndarray]:
    """Capture one RGB frame from each task camera after a replay step."""
    task_env._update_render()
    task_env.cameras.update_picture()
    rgb = task_env.cameras.get_rgb()
    frames: dict[str, np.ndarray] = {}
    for cam_name, cam_data in rgb.items():
        frame = np.asarray(cam_data.get("rgb"))
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frames[cam_name] = frame.astype(np.uint8, copy=False)
    return frames


def _open_video_writer(path: Path, fps: int) -> Any:
    """Open an MP4 writer using imageio-ffmpeg."""
    if not _IMAGEIO_OK:
        raise RuntimeError(
            "imageio is required for simulator video export. "
            "Install with: pip install imageio imageio-ffmpeg"
        )
    return imageio.get_writer(
        str(path),
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        pixelformat="yuv420p",
    )


def _close_video_writers(writers: dict[str, Any]) -> None:
    """Close all open video writers without failing the caller."""
    for key, writer in list(writers.items()):
        try:
            writer.close()
        except Exception:
            logger.debug("Failed to close video writer %s", key, exc_info=True)
    writers.clear()


def _ensure_multiview_video(renders_dir: Path) -> None:
    """Mirror the composite render to RoboTwin's conventional multi_view filename."""
    composite = renders_dir / "composite.mp4"
    multiview = renders_dir / "multi_view.mp4"
    if not composite.is_file():
        return
    try:
        shutil.copy2(composite, multiview)
    except Exception:
        logger.debug("Failed to create multi_view.mp4 from %s", composite, exc_info=True)


def _make_composite_frame(
    camera_frames: dict[str, np.ndarray],
    cam_order: list[str],
) -> np.ndarray | None:
    """Arrange up to 4 camera views into one 2x2 RGB frame."""
    available = [camera_frames[c] for c in cam_order if c in camera_frames]
    if not available:
        return None

    h, w = available[0].shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    slots = (available + [blank] * 4)[:4]
    top = np.concatenate([slots[0], slots[1]], axis=1)
    bottom = np.concatenate([slots[2], slots[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _write_camera_frame_set(
    video_writers: dict[str, Any],
    renders_dir: Path,
    camera_frames: dict[str, np.ndarray],
    *,
    fps: int,
) -> None:
    """Append one simulator frame to per-camera and composite MP4 files."""
    composite_order = [
        "front_camera",
        "head_camera",
        "left_camera",
        "right_camera",
    ]
    for cam_name, frame in camera_frames.items():
        writer = video_writers.get(cam_name)
        if writer is None:
            writer = _open_video_writer(renders_dir / f"{cam_name}.mp4", fps=fps)
            video_writers[cam_name] = writer
        writer.append_data(np.asarray(frame, dtype=np.uint8))

    composite = _make_composite_frame(camera_frames, composite_order)
    if composite is not None:
        writer = video_writers.get("composite")
        if writer is None:
            writer = _open_video_writer(renders_dir / "composite.mp4", fps=fps)
            video_writers["composite"] = writer
        writer.append_data(composite)


# ---------------------------------------------------------------------------
# USDC writer (uses extracted replay data — no heuristics needed)
# ---------------------------------------------------------------------------


_USD_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


def _usd_child_name(name: Any, fallback: str, used: set[str]) -> str:
    """Return a non-empty USD child prim name unique within one parent."""
    raw = str(name or "").strip()
    safe = _USD_IDENTIFIER_RE.sub("_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = fallback
    if safe[0].isdigit():
        safe = f"_{safe}"

    candidate = safe
    idx = 1
    while candidate in used:
        idx += 1
        candidate = f"{safe}_{idx}"
    used.add(candidate)
    return candidate


def _is_non_renderable_link(link_name: Any, metadata: dict[str, Any]) -> bool:
    link_meta = metadata.get(str(link_name), {})
    return bool(link_meta) and link_meta.get("renderable") is False


def _set_custom_attr(prim: Any, name: str, type_name: Any, value: Any) -> None:
    prim.CreateAttribute(name, type_name).Set(value)


def _mesh_info_parts(mesh_info: dict[str, Any]) -> list[dict[str, Any]]:
    parts = mesh_info.get("parts")
    if parts:
        return list(parts)
    return [
        {
            "verts": mesh_info["verts"],
            "faces": mesh_info["faces"],
            "uvs": mesh_info.get("uvs"),
            "color": mesh_info.get("color", (0.7, 0.7, 0.7)),
            "material": {
                "base_color": mesh_info.get("color", (0.7, 0.7, 0.7)),
                "source_type": None,
                "properties": {},
                "textures": {},
            },
            "source": {},
        }
    ]


def _material_rgb(material_info: dict[str, Any], fallback: Any) -> tuple[float, float, float]:
    color = material_info.get("base_color") or fallback or (0.7, 0.7, 0.7)
    if len(color) < 3:
        color = (0.7, 0.7, 0.7)
    return (float(color[0]), float(color[1]), float(color[2]))


def _material_scalar(material_info: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = material_info.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


_TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr")


def _find_texture_asset_path(value: Any) -> str | None:
    """Find a plausible texture filename/path in a nested material value."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.lower().endswith(_TEXTURE_EXTENSIONS):
            return text
        return None
    if isinstance(value, dict):
        for key in ("file", "filename", "file_path", "path", "uri", "base_color_texture"):
            if key in value:
                found = _find_texture_asset_path(value[key])
                if found:
                    return found
        for item in value.values():
            found = _find_texture_asset_path(item)
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_texture_asset_path(item)
            if found:
                return found
    return None


def _author_usd_material(
    stage: Any,
    mesh_prim: Any,
    *,
    material_info: dict[str, Any],
    fallback_color: Any,
    name_hint: str,
    used_material_names: set[str],
    has_uvs: bool = False,
) -> None:
    """Create and bind a UsdPreviewSurface material plus RoboTwin metadata."""
    from pxr import Gf, Sdf, UsdShade, Vt

    safe_name = _usd_child_name(name_hint, "Material", used_material_names)
    material_path = f"/Scene/Materials/{safe_name}"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")

    rgb = _material_rgb(material_info, fallback_color)
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    texture_hints = material_info.get("textures") or {}
    texture_path = _find_texture_asset_path(texture_hints)
    if texture_path and has_uvs:
        st_reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader_st")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        texture = UsdShade.Shader.Define(stage, f"{material_path}/BaseColorTexture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(),
            "result",
        )
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        diffuse_input.ConnectToSource(texture.ConnectableAPI(), "rgb")
    else:
        diffuse_input.Set(Gf.Vec3f(*rgb))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
        float(material_info.get("alpha", 1.0))
    )
    roughness = _material_scalar(material_info, "roughness")
    if roughness is not None:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    metallic = _material_scalar(material_info, "metallic")
    if metallic is not None:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    emissive = material_info.get("emissive_color") or material_info.get("emission_color")
    if isinstance(emissive, (list, tuple)) and len(emissive) >= 3:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(emissive[0]), float(emissive[1]), float(emissive[2]))
        )

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)

    material_json = json.dumps(material_info, sort_keys=True, default=str)
    _set_custom_attr(
        material.GetPrim(),
        "robotwin:materialJson",
        Sdf.ValueTypeNames.String,
        material_json,
    )
    _set_custom_attr(
        material.GetPrim(),
        "robotwin:sourceMaterialType",
        Sdf.ValueTypeNames.String,
        str(material_info.get("source_type") or ""),
    )
    material.GetPrim().CreateAttribute(
        "robotwin:baseColor",
        Sdf.ValueTypeNames.FloatArray,
    ).Set(Vt.FloatArray([float(v) for v in rgb]))
    _set_custom_attr(
        material.GetPrim(),
        "robotwin:alpha",
        Sdf.ValueTypeNames.Float,
        float(material_info.get("alpha", 1.0)),
    )

    if texture_hints:
        _set_custom_attr(
            material.GetPrim(),
            "robotwin:texturesJson",
            Sdf.ValueTypeNames.String,
            json.dumps(texture_hints, sort_keys=True, default=str),
        )
        if texture_path:
            _set_custom_attr(
                material.GetPrim(),
                "robotwin:baseColorTexture",
                Sdf.ValueTypeNames.Asset,
                Sdf.AssetPath(texture_path),
            )


def _author_mesh_prim(
    stage: Any,
    prim_path: str,
    part: dict[str, Any],
    *,
    material_name_hint: str,
    used_material_names: set[str],
) -> Any:
    from pxr import Gf, Sdf, UsdGeom, Vt

    verts = np.asarray(part["verts"], dtype=np.float64)
    faces = np.asarray(part["faces"], dtype=np.int32)
    material_info = dict(part.get("material") or {})
    color = _material_rgb(material_info, part.get("color"))

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([
            Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
            for v in verts
        ])
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    uvs = part.get("uvs")
    has_uvs = False
    if uvs is not None:
        uv_arr = np.asarray(uvs, dtype=np.float32)
        if uv_arr.ndim == 2 and uv_arr.shape[1] >= 2 and uv_arr.shape[0] == len(verts):
            primvars = UsdGeom.PrimvarsAPI(mesh)
            st = primvars.CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex,
            )
            st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uv_arr[:, :2]]))
            has_uvs = True

    source = part.get("source") or {}
    if source:
        _set_custom_attr(
            mesh.GetPrim(),
            "robotwin:sourceGeometryJson",
            Sdf.ValueTypeNames.String,
            json.dumps(source, sort_keys=True, default=str),
        )
    _set_custom_attr(
        mesh.GetPrim(),
        "robotwin:materialJson",
        Sdf.ValueTypeNames.String,
        json.dumps(material_info, sort_keys=True, default=str),
    )

    _author_usd_material(
        stage,
        mesh.GetPrim(),
        material_info=material_info,
        fallback_color=color,
        name_hint=material_name_hint,
        used_material_names=used_material_names,
        has_uvs=has_uvs,
    )
    return mesh


def _apply_pose_samples_to_xform(xf: Any, poses: np.ndarray, *, is_static: bool) -> None:
    if is_static:
        _set_pose_on_xform(xf, poses[0])
        return
    tr_op = xf.AddTranslateOp()
    or_op = xf.AddOrientOp()
    for t in range(len(poses)):
        _set_pose_animated(tr_op, or_op, poses[t], t)


def _write_usd_cameras(
    stage: Any,
    *,
    camera_observations: dict[str, dict[str, Any]],
    frame_count: int,
) -> None:
    """Author HDF5 observation cameras as USD camera prims."""
    if not camera_observations:
        return

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    UsdGeom.Xform.Define(stage, "/Scene/Cameras")
    used_names: set[str] = set()
    for cam_name, data in sorted(camera_observations.items()):
        poses = data.get("cam2world_gl")
        if poses is None:
            continue
        poses_arr = np.asarray(poses, dtype=np.float64)
        if poses_arr.ndim == 2:
            poses_arr = poses_arr[None, ...]
        if poses_arr.ndim != 3 or poses_arr.shape[1:] != (4, 4):
            logger.warning(
                "Skipping camera %s with invalid pose shape %s",
                cam_name,
                poses_arr.shape,
            )
            continue

        prim_name = _usd_child_name(cam_name, "Camera", used_names)
        cam = UsdGeom.Camera.Define(stage, f"/Scene/Cameras/{prim_name}")
        cam.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        _set_custom_attr(
            cam.GetPrim(),
            "robotwin:sourceCameraName",
            Sdf.ValueTypeNames.String,
            str(cam_name),
        )
        _set_custom_attr(
            cam.GetPrim(),
            "robotwin:cameraPoseSource",
            Sdf.ValueTypeNames.String,
            "hdf5:observation/<camera>/cam2world_gl",
        )

        intrinsic = data.get("intrinsic_cv")
        if intrinsic is not None:
            intrinsic = np.asarray(intrinsic, dtype=np.float64)
            if intrinsic.ndim == 3:
                intrinsic = intrinsic[0]
            if intrinsic.shape == (3, 3):
                cam.GetPrim().CreateAttribute(
                    "robotwin:intrinsicCv",
                    Sdf.ValueTypeNames.FloatArray,
                ).Set(Vt.FloatArray(intrinsic.reshape(-1).astype(float).tolist()))
            else:
                intrinsic = None

        image_size = data.get("image_size")
        if image_size is None:
            image_size = _camera_image_size_from_intrinsics(intrinsic)
        if image_size is not None:
            width, height = int(image_size[0]), int(image_size[1])
            _set_custom_attr(cam.GetPrim(), "robotwin:imageWidth", Sdf.ValueTypeNames.Int, width)
            _set_custom_attr(cam.GetPrim(), "robotwin:imageHeight", Sdf.ValueTypeNames.Int, height)
            if intrinsic is not None and width > 0 and height > 0:
                fx = float(intrinsic[0, 0])
                fy = float(intrinsic[1, 1])
                cx = float(intrinsic[0, 2])
                cy = float(intrinsic[1, 2])
                if fx > 0.0 and fy > 0.0:
                    horizontal_aperture = 36.0
                    focal_length = fx * horizontal_aperture / float(width)
                    vertical_aperture = focal_length * float(height) / fy
                    cam.GetFocalLengthAttr().Set(float(focal_length))
                    cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
                    cam.GetVerticalApertureAttr().Set(float(vertical_aperture))
                    cam.GetHorizontalApertureOffsetAttr().Set(
                        float((cx - width * 0.5) * horizontal_aperture / width)
                    )
                    cam.GetVerticalApertureOffsetAttr().Set(
                        float((cy - height * 0.5) * vertical_aperture / height)
                    )

        xformable = UsdGeom.Xformable(cam.GetPrim())
        tr_op = xformable.AddTranslateOp()
        or_op = xformable.AddOrientOp()
        for t in range(min(frame_count, poses_arr.shape[0])):
            T = poses_arr[t]
            tr_op.Set(
                Gf.Vec3d(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
                Usd.TimeCode(t),
            )
            qw, qx, qy, qz = _rot_to_quat(T[:3, :3])
            or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))


def _write_replay_usdc(path: Path, data: dict) -> None:
    """Write a ground-truth animated USDC from replay data."""
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:
        raise RuntimeError("usd-core is required to write replay USDC") from exc

    T = data["T"]
    fps = data["fps"]
    actor_states = data.get("actor_states", {})
    actor_meshes = data.get("actor_meshes", {})
    actor_metadata = data.get("actor_metadata", {})
    robot_link_states = data.get("robot_link_states", {})
    robot_link_meshes = data.get("robot_link_meshes", {})
    robot_link_metadata = data.get("robot_link_metadata", {})

    missing_actor_meshes = [
        str(actor_name) for actor_name in actor_states if actor_name not in actor_meshes
    ]
    if missing_actor_meshes:
        raise MeshExtractionError(
            "Cannot write replay USDC because actor meshes are missing for: "
            + ", ".join(missing_actor_meshes)
        )

    missing_robot_meshes = [
        str(link_name)
        for link_name in robot_link_states
        if link_name not in robot_link_meshes
        and not _is_non_renderable_link(link_name, robot_link_metadata)
    ]
    if missing_robot_meshes:
        raise MeshExtractionError(
            "Cannot write replay USDC because robot link meshes are missing for: "
            + ", ".join(missing_robot_meshes)
        )

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(T - 1)
    stage.SetFramesPerSecond(fps)
    stage.SetTimeCodesPerSecond(fps)
    root_xf = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root_xf.GetPrim())

    _write_usd_cameras(
        stage,
        camera_observations=data.get("camera_observations", {}),
        frame_count=T,
    )
    UsdGeom.Scope.Define(stage, "/Scene/Materials")
    used_material_names: set[str] = set()

    # ── Static scene actors (table, wall, ground, etc.) ──────────────────
    # Actors that barely move across frames are rendered as static meshes
    # at their t=0 pose. Everything else is animated.
    STATIC_ACTORS = {"table", "wall", "ground"}

    used_actor_names: set[str] = set()
    for actor_idx, (actor_name, poses) in enumerate(data["actor_states"].items()):
        actor_name_str = str(actor_name or "")
        actor_meta = dict(actor_metadata.get(actor_name, {}))
        source_actor_name = str(actor_meta.get("source_name") or actor_name_str)

        safe_name = _usd_child_name(actor_name_str, f"Actor_{actor_idx}", used_actor_names)
        prim_path = f"/Scene/{safe_name}"
        mesh_info = data["actor_meshes"].get(actor_name)

        if mesh_info is None:
            raise MeshExtractionError(
                f"Actor {actor_name_str!r} has pose samples but no mesh data"
            )

        parts = _mesh_info_parts(mesh_info)
        is_static = source_actor_name in STATIC_ACTORS
        if len(parts) == 1:
            mesh = _author_mesh_prim(
                stage,
                prim_path,
                parts[0],
                material_name_hint=f"{safe_name}_material",
                used_material_names=used_material_names,
            )
            prim = mesh.GetPrim()
            _set_custom_attr(
                prim,
                "robotwin:actorKey",
                Sdf.ValueTypeNames.String,
                actor_name_str,
            )
            _set_custom_attr(
                prim,
                "robotwin:sourceActorName",
                Sdf.ValueTypeNames.String,
                source_actor_name,
            )
            duplicate_index = actor_meta.get("duplicate_index")
            if duplicate_index is not None:
                _set_custom_attr(
                    prim,
                    "robotwin:duplicateActorIndex",
                    Sdf.ValueTypeNames.Int,
                    int(duplicate_index),
                )
            _apply_pose_samples_to_xform(
                UsdGeom.Xformable(prim),
                poses,
                is_static=is_static,
            )
        else:
            actor_xf = UsdGeom.Xform.Define(stage, prim_path)
            _set_custom_attr(
                actor_xf.GetPrim(),
                "robotwin:meshPartCount",
                Sdf.ValueTypeNames.Int,
                len(parts),
            )
            _set_custom_attr(
                actor_xf.GetPrim(),
                "robotwin:actorKey",
                Sdf.ValueTypeNames.String,
                actor_name_str,
            )
            _set_custom_attr(
                actor_xf.GetPrim(),
                "robotwin:sourceActorName",
                Sdf.ValueTypeNames.String,
                source_actor_name,
            )
            duplicate_index = actor_meta.get("duplicate_index")
            if duplicate_index is not None:
                _set_custom_attr(
                    actor_xf.GetPrim(),
                    "robotwin:duplicateActorIndex",
                    Sdf.ValueTypeNames.Int,
                    int(duplicate_index),
                )
            _apply_pose_samples_to_xform(
                UsdGeom.Xformable(actor_xf.GetPrim()),
                poses,
                is_static=is_static,
            )
            used_part_names: set[str] = set()
            for part_idx, part in enumerate(parts):
                part_name = _usd_child_name(
                    f"Part_{part_idx:03d}",
                    f"Part_{part_idx:03d}",
                    used_part_names,
                )
                _author_mesh_prim(
                    stage,
                    f"{prim_path}/{part_name}",
                    part,
                    material_name_hint=f"{safe_name}_{part_name}_material",
                    used_material_names=used_material_names,
                )

    # ── Robot links: renderable links become Mesh; frame-only links become Xform ──
    if robot_link_states:
        UsdGeom.Xform.Define(stage, "/Scene/Robot")
        used_link_names: set[str] = set()
        for link_idx, (link_name, transforms) in enumerate(robot_link_states.items()):
            safe = _usd_child_name(link_name, f"Link_{link_idx}", used_link_names)
            prim_path = f"/Scene/Robot/{safe}"
            mesh_info = robot_link_meshes.get(link_name)
            if mesh_info is None:
                link_meta = robot_link_metadata.get(str(link_name), {})
                if not link_meta or link_meta.get("renderable") is not False:
                    raise MeshExtractionError(
                        f"Robot link {link_name!r} has pose samples but no mesh data"
                    )

                frame = UsdGeom.Xform.Define(stage, prim_path)
                prim = frame.GetPrim()
                _set_custom_attr(prim, "gewu:entityType", Sdf.ValueTypeNames.String, "kinematic_frame")
                _set_custom_attr(prim, "gewu:renderable", Sdf.ValueTypeNames.Bool, False)
                _set_custom_attr(prim, "gewu:hasGeometry", Sdf.ValueTypeNames.Bool, False)
                _set_custom_attr(
                    prim,
                    "gewu:semanticRole",
                    Sdf.ValueTypeNames.String,
                    str(link_meta.get("semantic_role") or "kinematic_frame"),
                )
                _set_custom_attr(prim, "gewu:sourceLinkName", Sdf.ValueTypeNames.String, str(link_name))
                _set_custom_attr(prim, "gewu:hasVisual", Sdf.ValueTypeNames.Bool, bool(link_meta.get("has_visual")))
                _set_custom_attr(prim, "gewu:hasCollision", Sdf.ValueTypeNames.Bool, bool(link_meta.get("has_collision")))
                _set_custom_attr(prim, "gewu:hasInertial", Sdf.ValueTypeNames.Bool, bool(link_meta.get("has_inertial")))

                xf = UsdGeom.Xformable(prim)
                tr_op = xf.AddTranslateOp()
                or_op = xf.AddOrientOp()
                for t in range(T):
                    T_link = transforms[t]
                    p = T_link[:3, 3]
                    tr_op.Set(
                        Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
                        Usd.TimeCode(t),
                    )
                    qw, qx, qy, qz = _rot_to_quat(T_link[:3, :3])
                    or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))
                continue

            link_meta = robot_link_metadata.get(str(link_name), {})
            if link_meta.get("renderable") is False:
                raise MeshExtractionError(
                    f"Robot link {link_name!r} is marked non-renderable but has mesh data"
                )

            parts = _mesh_info_parts(mesh_info)
            if len(parts) == 1:
                mesh = _author_mesh_prim(
                    stage,
                    prim_path,
                    parts[0],
                    material_name_hint=f"Robot_{safe}_material",
                    used_material_names=used_material_names,
                )
                xf = UsdGeom.Xformable(mesh.GetPrim())
                tr_op = xf.AddTranslateOp()
                or_op = xf.AddOrientOp()
                for t in range(T):
                    T_link = transforms[t]
                    p = T_link[:3, 3]
                    tr_op.Set(
                        Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
                        Usd.TimeCode(t),
                    )
                    qw, qx, qy, qz = _rot_to_quat(T_link[:3, :3])
                    or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))
            else:
                link_xf = UsdGeom.Xform.Define(stage, prim_path)
                _set_custom_attr(
                    link_xf.GetPrim(),
                    "robotwin:meshPartCount",
                    Sdf.ValueTypeNames.Int,
                    len(parts),
                )
                xf = UsdGeom.Xformable(link_xf.GetPrim())
                tr_op = xf.AddTranslateOp()
                or_op = xf.AddOrientOp()
                for t in range(T):
                    T_link = transforms[t]
                    p = T_link[:3, 3]
                    tr_op.Set(
                        Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
                        Usd.TimeCode(t),
                    )
                    qw, qx, qy, qz = _rot_to_quat(T_link[:3, :3])
                    or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))

                used_part_names: set[str] = set()
                for part_idx, part in enumerate(parts):
                    part_name = _usd_child_name(
                        f"Part_{part_idx:03d}",
                        f"Part_{part_idx:03d}",
                        used_part_names,
                    )
                    _author_mesh_prim(
                        stage,
                        f"{prim_path}/{part_name}",
                        part,
                        material_name_hint=f"Robot_{safe}_{part_name}_material",
                        used_material_names=used_material_names,
                    )

    stage.GetRootLayer().Save()


def _write_replay_usdz(usdc_path: Path, usdz_path: Path) -> None:
    """Package a replay USDC and its texture dependencies into one USDZ."""
    try:
        from pxr import UsdUtils
    except ImportError as exc:
        raise RuntimeError("usd-core with UsdUtils is required to write replay USDZ") from exc

    usdc_path = Path(usdc_path)
    usdz_path = Path(usdz_path)
    if not usdc_path.is_file():
        raise RuntimeError(f"Cannot package missing USDC: {usdc_path}")
    if usdz_path.exists():
        usdz_path.unlink()
    ok = UsdUtils.CreateNewUsdzPackage(str(usdc_path), str(usdz_path))
    if not ok or not usdz_path.is_file():
        raise RuntimeError(f"Failed to package USDZ: {usdz_path}")


def _set_pose_on_xform(xf: Any, pose7: np.ndarray) -> None:
    """Set a static translate + orient on a UsdGeom.Xformable from a 7-vec."""
    from pxr import Gf, Usd
    xf.AddTranslateOp().Set(
        Gf.Vec3d(float(pose7[0]), float(pose7[1]), float(pose7[2]))
    )
    xf.AddOrientOp().Set(
        Gf.Quatf(float(pose7[3]), float(pose7[4]), float(pose7[5]), float(pose7[6]))
    )


def _set_pose_animated(tr_op: Any, or_op: Any, pose7: np.ndarray, t: int) -> None:
    """Set a time-sampled translate + orient from a 7-vec."""
    from pxr import Gf, Usd
    tr_op.Set(
        Gf.Vec3d(float(pose7[0]), float(pose7[1]), float(pose7[2])),
        Usd.TimeCode(t),
    )
    or_op.Set(
        Gf.Quatf(float(pose7[3]), float(pose7[4]), float(pose7[5]), float(pose7[6])),
        Usd.TimeCode(t),
    )


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


# ---------------------------------------------------------------------------
# CLI entry point (for standalone testing)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Replay a RoboTwin 2.0 episode in SAPIEN → USDC")
    parser.add_argument("--h5", required=True, help="Path to episode HDF5 file")
    parser.add_argument("--task", required=True, help="Task name (e.g. handover_block)")
    parser.add_argument("--seed", type=int, required=True, help="Episode seed")
    parser.add_argument("--robotwin-root", required=True, help="Path to RoboTwin repo root")
    parser.add_argument("--out", required=True, help="Output USDC path")
    parser.add_argument("--out-usdz", default=None, help="Optional output USDZ package path")
    parser.add_argument(
        "--drop-usdc-assets",
        action="store_true",
        help="After writing USDZ, remove sibling scene.usdc and textures/ assets",
    )
    parser.add_argument("--renders-dir", default=None, help="Directory to write rendered MP4s")
    parser.add_argument("--views-dir", default=None, help="Directory to write generated-camera view folders")
    parser.add_argument("--view-specs-json", default=None, help="JSON file with explicit generated view specs")
    parser.add_argument(
        "--generated-camera-motion",
        choices=_GENERATED_CAMERA_MOTIONS,
        default="random-static",
    )
    parser.add_argument("--generated-static-camera-count", type=int, default=None)
    parser.add_argument("--generated-trajectory-camera-count", type=int, default=None)
    parser.add_argument(
        "--camera-trajectory-kind",
        choices=_TRAJECTORY_KINDS,
        default="orbit360",
    )
    parser.add_argument("--trajectory-kind-mode", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--camera-seed", type=int, default=None)
    parser.add_argument("--view-width", type=int, default=768)
    parser.add_argument("--view-height", type=int, default=768)
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()
    view_specs = None
    if args.view_specs_json:
        view_specs = json.loads(Path(args.view_specs_json).read_text(encoding="utf-8"))

    result = replay_episode(
        h5_path=Path(args.h5),
        task_name=args.task,
        seed=args.seed,
        robotwin_root=Path(args.robotwin_root),
        out_usdc=Path(args.out),
        out_usdz=Path(args.out_usdz) if args.out_usdz else None,
        keep_usdc_assets=not bool(args.drop_usdc_assets),
        renders_dir=Path(args.renders_dir) if args.renders_dir else None,
        views_dir=Path(args.views_dir) if args.views_dir else None,
        view_specs=view_specs,
        generated_camera_motion=args.generated_camera_motion,
        generated_static_camera_count=args.generated_static_camera_count,
        generated_trajectory_camera_count=args.generated_trajectory_camera_count,
        camera_trajectory_kind=args.camera_trajectory_kind,
        trajectory_kind_mode=args.trajectory_kind_mode,
        camera_seed=args.camera_seed,
        view_width_px=args.view_width,
        view_height_px=args.view_height,
        embodiment=args.embodiment,
        fps=args.fps,
        video_fps=args.video_fps,
    )
    if result:
        print(f"OK: {result['T']} frames, {len(result['actor_states'])} actors, "
              f"{len(result['robot_link_states'])} robot links")
    else:
        print("FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
