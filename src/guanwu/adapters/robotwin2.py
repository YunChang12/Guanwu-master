"""RoboTwin 2.0 adapter for bimanual manipulation trajectories.

RoboTwin 2.0 is a scalable data generator and benchmark for robust bimanual
robotic manipulation with strong domain randomization.

Expected local directory structure (after downloading from HuggingFace)::

    <root>/
      <task_name>/                        # e.g. handover_block
        <robot>_clean_<N>/               # e.g. aloha-agilex_clean_50
          data/
            episode0.hdf5
            episode1.hdf5
            ...
          instructions/
            episode0.json               # {"seen": [...], "unseen": [...]}
            ...
          video/
            episode0.mp4               # pre-rendered multi-view video
            ...
        <robot>_randomized_<N>/
          data/ instructions/ video/
          ...

Real HDF5 episode structure::

    episode*.hdf5
      observation/
        front_camera/
          rgb/             # (T,) |S<N> — JPEG-encoded bytes per frame
          cam2world_gl/   # (T, 4, 4) float32
          extrinsic_cv/   # (T, 3, 4) float32
          intrinsic_cv/   # (T, 3, 3) float32
        head_camera/  left_camera/  right_camera/   # same layout
        pointcloud/      # (T, 0) — empty in most releases
      endpose/
        left_endpose/    # (T, 7) float64 — [x, y, z, qw, qx, qy, qz]
        left_gripper/    # (T,)   float64
        right_endpose/   # (T, 7) float64
        right_gripper/   # (T,)   float64
      joint_action/
        vector/          # (T, 14) float64 — concatenated 7-DOF per arm
        left_arm/        # (T, 6)  float64
        right_arm/       # (T, 6)  float64
        left_gripper/    # (T,)
        right_gripper/   # (T,)

Quaternion convention (SAPIEN):  [x, y, z, qw, qx, qy, qz]
                                        ↑ w is 4th element (index 3)

Reference: https://robotwin-platform.github.io/
Dataset:   https://huggingface.co/datasets/TianxingChen/RoboTwin2.0
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guanwu.adapters.base import DatasetAdapter, register_adapter
from guanwu.core.ids import (
    make_episode_uid,
    make_instance_uid,
    make_scene_uid,
    make_sensor_uid,
    make_track_uid,
)

try:
    import imageio
    _IMAGEIO_OK = True
except ImportError:  # pragma: no cover
    _IMAGEIO_OK = False

try:
    import trimesh  # noqa: F401  (used inside _load_aloha_meshes)
    _TRIMESH_OK = True
except ImportError:  # pragma: no cover
    _TRIMESH_OK = False

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # type: ignore[import]
    _USD_OK = True
except ImportError:  # pragma: no cover
    _USD_OK = False

from guanwu.schemas.bundles import (
    AdapterConfig,
    EmitReport,
    JobContext,
    NormalizeBundle,
    ParseBundle,
    RawRef,
    SourceItem,
)
from guanwu.schemas.enums import (
    AccessMode,
    GeometryLevel,
    RecordScope,
    SceneKind,
    SensorType,
    SourceType,
)
from guanwu.schemas.records import (
    ArticulationStateRecord,
    DatasetRecord,
    EpisodeRecord,
    InstanceRecord,
    LicenseRecord,
    ProvenanceRecord,
    SceneRecord,
    SensorRecord,
    TrackStateRecord,
)
from guanwu.storage.raw_store import RawStore

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

logger = logging.getLogger("guanwu")

DATASET_ID = "robotwin2"

_SCENE_VIEWS_FORMAT = "guanwu.robotwin2.scene_views.v0.1"
_RENDER_PIPELINE_VERSION = "robotwin2_replay_usdz_views.v0.2"

# All cameras present in real RoboTwin 2.0 HDF5 episodes
_CAMERA_NAMES = ("front_camera", "head_camera", "left_camera", "right_camera")

# Total joint DOF: 7 per arm × 2 arms
_ARM_DOFS = 14

_USD_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_usd_name(name: Any, fallback: str = "Camera") -> str:
    safe = _USD_IDENTIFIER_RE.sub("_", str(name or "").strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = fallback
    if safe[0].isdigit():
        safe = f"_{safe}"
    return safe


# ---------------------------------------------------------------------------
# ALOHA-Agilex (ARX5) kinematic chain — extracted from
# embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
# ---------------------------------------------------------------------------
# Each entry describes one revolute joint of the front arm:
#   xyz : translation from previous link, in parent frame (metres)
#   rpy : fixed Euler-XYZ rotation of the joint frame relative to parent (rad)
#   axis: rotation axis in the joint frame
# Both fl_jointN and fr_jointN share identical chains; only the base differs.

_ARX5_FRONT_CHAIN = (
    {"xyz": (0.0, 0.0, 0.058),               "rpy": (0.0, 0.0, 0.0),       "axis": (0, 0, 1)},
    {"xyz": (0.025013, 0.00060169, 0.042),   "rpy": (0.0, 0.0, 0.0),       "axis": (0, 1, 0)},
    {"xyz": (-0.26396, 0.0044548, 0.0),      "rpy": (-3.1416, 0.0, -0.015928), "axis": (0, 1, 0)},
    {"xyz": (0.246, -0.00025, -0.06),        "rpy": (0.0, 0.0, 0.0),       "axis": (0, 1, 0)},
    {"xyz": (0.06775, 0.0015, -0.0855),      "rpy": (0.0, 0.0, -0.015928), "axis": (0, 0, 1)},
    {"xyz": (0.03095, 0.0, 0.0855),          "rpy": (-3.1416, 0.0, 0.0),   "axis": (1, 0, 0)},
)

# Robot world placement from task_config: robot_pose = [0, -0.65, 0, 0.707, 0, 0, 0.707]
_ROBOT_WORLD_POS  = (0.0, -0.65, 0.0)
_ROBOT_WORLD_QUAT = (0.7071067811865476, 0.0, 0.0, 0.7071067811865475)  # (w, x, y, z) → 90° about Z

# Front-arm base offsets from robot footprint (in robot-local frame)
_LEFT_ARM_BASE_XYZ  = (0.2305, 0.297, 0.782)
_RIGHT_ARM_BASE_XYZ = (0.2315, -0.3063, 0.781)

# Mobile-base bounding box (centre + half-extents) in robot-local frame.
# Approximated from the tracer + box1/box2 links so the arms have a body to attach to.
_BASE_BOX_CENTRE   = (0.0, 0.0, 0.39)   # half-height of the body
_BASE_BOX_HALF_EXT = (0.22, 0.21, 0.39)  # half-extents (x, y, z)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hdf5_files(root: Path) -> list[Path]:
    """Return all .hdf5 / .h5 files under *root* (recursive)."""
    files: list[Path] = []
    for ext in ("*.hdf5", "*.h5"):
        files.extend(sorted(root.rglob(ext)))
    return sorted(files)


def _opt(config: AdapterConfig, key: str, default: Any = None) -> Any:
    return config.options.get(key, default)


def _as_name_set(value: Any) -> set[str] | None:
    """Normalize a config scalar/list into a set of names."""
    if value is None:
        return None
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        names = [str(part).strip() for part in value]
    else:
        names = [str(value).strip()]
    return {name for name in names if name}


def _as_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _as_resolution(value: Any, default: tuple[int, int] = (768, 768)) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.lower().replace("x", ",")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return default
    if len(parts) != 2:
        return default
    try:
        width = int(parts[0])
        height = int(parts[1])
    except (TypeError, ValueError):
        return default
    if width <= 0 or height <= 0:
        return default
    return width, height


def _load_view_specs(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        views = value.get("views") or value.get("view_specs")
        if isinstance(views, list):
            return [dict(item) for item in views if isinstance(item, dict)]
        return [dict(value)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        candidate = Path(text)
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
        data = json.loads(text)
        return _load_view_specs(data)
    raise TypeError(f"Unsupported view_specs value: {type(value).__name__}")


def _load_instruction(instructions_dir: Path, episode_stem: str) -> str | None:
    """Load the natural language instruction for an episode.

    Handles both real RoboTwin 2.0 format ``{"seen": [...], "unseen": [...]}``
    and simpler fixture formats ``{"instruction": "..."}``.
    """
    for suffix in (".json", ".txt"):
        cand = instructions_dir / f"{episode_stem}{suffix}"
        if not cand.is_file():
            continue
        try:
            if suffix == ".json":
                with open(cand) as f:
                    data = json.load(f)
                if isinstance(data, str):
                    return data
                if isinstance(data, list) and data:
                    return str(data[0])
                if isinstance(data, dict):
                    # Real format: {"seen": [...100 variations...], "unseen": [...]}
                    if "seen" in data and isinstance(data["seen"], list) and data["seen"]:
                        return str(data["seen"][0])
                    # Fallback for other dict shapes
                    for key in ("instruction", "text", "task", "description"):
                        if key in data:
                            val = data[key]
                            if isinstance(val, list) and val:
                                return str(val[0])
                            return str(val)
            else:
                return cand.read_text().strip()
        except Exception:
            logger.debug("Failed to load instruction from %s", cand)
    return None


def _load_variant_seed(variant_dir: Path) -> int | None:
    """Load the replay seed for a variant directory from seed.txt if present."""
    seed_path = variant_dir / "seed.txt"
    if not seed_path.is_file():
        return None
    try:
        return int(seed_path.read_text().strip().splitlines()[0])
    except Exception:
        logger.debug("Failed to parse RoboTwin seed from %s", seed_path)
        return None


def _episode_identity_from_ref(raw_path: Path, ref: RawRef) -> tuple[str, str, str, str | None]:
    """Recover RoboTwin task identity after fetch has linked files into raw store."""
    relpath = ref.item_id or ""
    rel_parts = Path(relpath).parts
    if (
        len(rel_parts) >= 4
        and rel_parts[-2] == "data"
        and raw_path.suffix in (".hdf5", ".h5")
    ):
        return rel_parts[-4], rel_parts[-3], Path(rel_parts[-1]).stem, relpath

    # Fallback for callers that pass raw paths directly in the original layout:
    # .../<task_name>/<variant>/data/episode*.hdf5
    parts = raw_path.parts
    ep_stem = raw_path.stem
    task_name = parts[-4] if len(parts) >= 4 else "unknown"
    variant = parts[-3] if len(parts) >= 3 else "unknown"
    return task_name, variant, ep_stem, None


def _read_hdf5_episode(h5_path: Path) -> dict[str, Any]:
    """Read metadata from a RoboTwin 2.0 HDF5 episode file.

    Supports both the real on-disk format (JPEG bytes, endpose/, joint_action/)
    and the synthetic test-fixture format (raw uint8 arrays, observation/joint_state).

    Returns a dict with:
        num_steps:       int   — number of timesteps T
        has_cameras:     list  — camera names found in observation/
        has_depth:       list  — cameras with a depth dataset (fixture only)
        action_dim:      int | None
        has_joint_state: bool
        has_ee_pose:     bool
        success:         bool | None
    """
    if h5py is None:
        raise RuntimeError(
            "h5py is required to parse RoboTwin 2.0 HDF5 files. "
            "Install it with: pip install h5py"
        )

    result: dict[str, Any] = {
        "num_steps": 0,
        "has_cameras": [],
        "has_depth": [],
        "action_dim": None,
        "has_joint_state": False,
        "has_ee_pose": False,
        "success": None,
    }

    with h5py.File(str(h5_path), "r") as f:
        obs = f.get("observation")

        # ── Camera channels ───────────────────────────────────────────────
        if obs is not None:
            for cam in _CAMERA_NAMES:
                if cam not in obs:
                    continue
                result["has_cameras"].append(cam)
                cam_grp = obs[cam]
                if "rgb" in cam_grp:
                    rgb_ds = cam_grp["rgb"]
                    if hasattr(rgb_ds, "shape") and rgb_ds.ndim >= 1:
                        result["num_steps"] = max(
                            result["num_steps"], int(rgb_ds.shape[0])
                        )
                if "depth" in cam_grp:
                    # Real data has no depth; fixture does — keep for compat
                    result["has_depth"].append(cam)

        # ── Joint states ─────────────────────────────────────────────────
        # Real format: joint_action/vector (T, 14)
        # Fixture format: observation/joint_state (T, 14)
        if "joint_action" in f and "vector" in f["joint_action"]:
            js_ds = f["joint_action"]["vector"]
            result["has_joint_state"] = True
            if hasattr(js_ds, "shape"):
                if js_ds.ndim > 1:
                    result["action_dim"] = int(js_ds.shape[-1])
                result["num_steps"] = max(result["num_steps"], int(js_ds.shape[0]))
        elif obs is not None and "joint_state" in obs:
            result["has_joint_state"] = True
            js_ds = obs["joint_state"]
            if hasattr(js_ds, "shape") and js_ds.ndim >= 1:
                result["num_steps"] = max(result["num_steps"], int(js_ds.shape[0]))

        # ── End-effector poses ────────────────────────────────────────────
        # Real format: endpose/left_endpose + endpose/right_endpose
        # Fixture format: observation/end_effector
        if "endpose" in f and "left_endpose" in f["endpose"]:
            result["has_ee_pose"] = True
        elif obs is not None and "end_effector" in obs:
            result["has_ee_pose"] = True

        # ── Action dim fallback (fixture uses top-level "action" dataset) ─
        if result["action_dim"] is None and "action" in f:
            action_ds = f["action"]
            if hasattr(action_ds, "shape"):
                result["action_dim"] = (
                    int(action_ds.shape[-1]) if action_ds.ndim > 1 else 1
                )
                result["num_steps"] = max(
                    result["num_steps"], int(action_ds.shape[0])
                )

        # ── Success flag (fixture only — real data omits this) ────────────
        if "success" in f:
            import numpy as np
            try:
                result["success"] = bool(np.array(f["success"]).any())
            except Exception:
                pass

    return result


def _read_joint_position_series(h5_path: Path) -> list[list[float]] | None:
    """Read per-step joint positions from the source HDF5 for canonical states."""
    if h5py is None:
        return None

    try:
        import numpy as np

        with h5py.File(str(h5_path), "r") as f:
            obs = f.get("observation")
            if "joint_action" in f and "vector" in f["joint_action"]:
                data = f["joint_action"]["vector"]
            elif obs is not None and "joint_state" in obs:
                data = obs["joint_state"]
            elif "action" in f:
                data = f["action"]
            else:
                return None

            arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        return arr.astype(float).tolist()
    except Exception:
        logger.exception("Failed to read RoboTwin joint positions from %s", h5_path)
        return None


def _decode_jpeg_bytes(data: bytes) -> Any:
    """Decode raw JPEG bytes into an (H, W, 3) uint8 numpy array.

    Tries OpenCV → Pillow → imageio in order. OpenCV is much faster for the
    RoboTwin fixed-length JPEG byte datasets on the remote replay machines.
    """
    import io as _io
    import numpy as np

    try:
        import cv2
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    try:
        from PIL import Image
        img = Image.open(_io.BytesIO(data))
        return np.asarray(img.convert("RGB"))
    except ImportError:
        pass

    if _IMAGEIO_OK:
        import imageio as _imageio
        return np.asarray(_imageio.imread(_io.BytesIO(data)))

    raise RuntimeError(
        "No JPEG decoder found. Install one of: imageio[ffmpeg], Pillow, opencv-python"
    )


def _decode_jpeg_frames(rgb_ds: Any) -> Any:
    """Decode a (T,) array of JPEG byte strings into (T, H, W, 3) uint8."""
    import numpy as np

    frames = []
    for raw in rgb_ds:
        try:
            frames.append(_decode_jpeg_bytes(bytes(raw)))
        except Exception as exc:
            logger.debug("Skipping bad JPEG frame: %s", exc)
            if frames:
                frames.append(np.zeros_like(frames[-1]))
            else:
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))

    if not frames:
        return np.zeros((0, 240, 320, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def _is_jpeg_dataset(rgb_ds: Any) -> bool:
    """Return True if *rgb_ds* stores JPEG bytes (real format) vs raw pixel arrays."""
    # Fixed-length byte strings: dtype like |S12984, itemsize >> 3 bytes
    if hasattr(rgb_ds, "dtype"):
        d = rgb_ds.dtype
        if d.kind in ("S", "V"):          # fixed-length bytes / void
            return True
        if d.kind == "O":                  # variable-length bytes
            return True
    return False


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read JSON from %s", path, exc_info=True)
        return None


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _scene_relative(path: Path, scene_dir: Path) -> str:
    return path.relative_to(scene_dir).as_posix()


def _scene_view_required_files(view_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        view_dir / "video.mp4",
        view_dir / "camera.json",
        view_dir / "render_meta.json",
        view_dir / "frame_mapping.json",
    )


def _is_scene_view_complete(view_dir: Path) -> bool:
    return all(path.is_file() for path in _scene_view_required_files(view_dir))


def _discover_scene_views(scene_dir: Path) -> list[dict[str, Any]]:
    views_root = scene_dir / "views"
    if not views_root.is_dir():
        return []

    views: list[dict[str, Any]] = []
    for view_dir in sorted(path for path in views_root.iterdir() if path.is_dir()):
        video = view_dir / "video.mp4"
        camera = view_dir / "camera.json"
        render_meta = view_dir / "render_meta.json"
        frame_mapping = view_dir / "frame_mapping.json"
        if not video.is_file():
            continue
        meta = _read_json_if_exists(render_meta) or {}
        camera_payload = _read_json_if_exists(camera) or {}
        record: dict[str, Any] = {
            "view_id": view_dir.name,
            "video": _scene_relative(video, scene_dir),
            "camera": _scene_relative(camera, scene_dir) if camera.is_file() else None,
            "render_meta": (
                _scene_relative(render_meta, scene_dir)
                if render_meta.is_file()
                else None
            ),
            "frame_mapping": (
                _scene_relative(frame_mapping, scene_dir)
                if frame_mapping.is_file()
                else None
            ),
            "uses_hdf5_rgb": bool(meta.get("uses_hdf5_rgb", False)),
            "frame_count": meta.get("frame_count"),
            "fps": meta.get("fps"),
            "camera_strategy": (
                meta.get("camera_strategy")
                or camera_payload.get("camera_strategy")
                or (camera_payload.get("camera") or {}).get("mode")
            ),
            "observation_id": (
                meta.get("observation_id")
                or camera_payload.get("observation_id")
                or view_dir.name
            ),
        }
        views.append({key: value for key, value in record.items() if value is not None})
    return views


def _expected_scene_view_count(
    *,
    render_views: bool,
    explicit_view_specs: list[dict[str, Any]] | None,
    generated_camera_motion: str,
    generated_static_camera_count: int | None,
    generated_trajectory_camera_count: int | None,
) -> int:
    if not render_views:
        return 0
    if explicit_view_specs:
        return len(explicit_view_specs)
    if (
        generated_static_camera_count is not None
        or generated_trajectory_camera_count is not None
    ):
        return int(generated_static_camera_count or 0) + int(
            generated_trajectory_camera_count or 0
        )
    if generated_camera_motion == "trajectory":
        return 1
    return 1


def _build_render_options(
    options: dict[str, Any],
    *,
    explicit_view_specs: list[dict[str, Any]] | None,
    replay_seed: int,
) -> dict[str, Any]:
    generated_static_camera_count = _as_nonnegative_int(
        options.get("generated_static_camera_count")
    )
    generated_trajectory_camera_count = _as_nonnegative_int(
        options.get("generated_trajectory_camera_count")
    )
    render_views = _as_bool(
        options.get(
            "render_views",
            options.get("remote_render_views"),
        ),
        False,
    )
    if explicit_view_specs:
        render_views = True
    if (generated_static_camera_count or 0) > 0 or (
        generated_trajectory_camera_count or 0
    ) > 0:
        render_views = True

    view_width, view_height = _as_resolution(
        options.get("view_resolution")
        or options.get("generated_view_resolution")
        or options.get("resolution"),
        (768, 768),
    )
    generated_camera_motion = str(
        options.get("generated_camera_motion", "random-static")
    )
    video_fps = int(options.get("remote_video_fps", 10))
    usdc_fps = float(options.get("remote_usdc_fps", 30.0))
    camera_seed = options.get("camera_seed")

    render_options = {
        "pipeline_version": _RENDER_PIPELINE_VERSION,
        "render_views": render_views,
        "generated_camera_motion": generated_camera_motion,
        "generated_static_camera_count": generated_static_camera_count,
        "generated_trajectory_camera_count": generated_trajectory_camera_count,
        "camera_trajectory_kind": str(
            options.get("camera_trajectory_kind", "orbit360")
        ),
        "trajectory_kind_mode": str(options.get("trajectory_kind_mode", "fixed")),
        "camera_seed": camera_seed,
        "replay_seed": replay_seed,
        "view_width_px": view_width,
        "view_height_px": view_height,
        "view_specs_count": len(explicit_view_specs or []),
        "view_specs_hash": _json_hash(explicit_view_specs or []),
        "expected_view_count": _expected_scene_view_count(
            render_views=render_views,
            explicit_view_specs=explicit_view_specs,
            generated_camera_motion=generated_camera_motion,
            generated_static_camera_count=generated_static_camera_count,
            generated_trajectory_camera_count=generated_trajectory_camera_count,
        ),
        "embodiment": str(options.get("remote_embodiment", "aloha-agilex")),
        "usdc_fps": usdc_fps,
        "video_fps": video_fps,
    }
    render_options["render_options_hash"] = _json_hash(render_options)
    return render_options


def _scene_export_complete(scene_dir: Path, render_options: dict[str, Any]) -> bool:
    if not (scene_dir / "scene.usdz").is_file():
        return False

    manifest = _read_json_if_exists(scene_dir / "manifest.json")
    if not isinstance(manifest, dict):
        return False
    if manifest.get("format") != _SCENE_VIEWS_FORMAT:
        return False

    render = manifest.get("render") or {}
    if render.get("render_options_hash") != render_options.get("render_options_hash"):
        return False
    if bool(render.get("uses_hdf5_rgb", True)):
        return False

    if not render_options.get("render_views"):
        return True

    expected = int(render_options.get("expected_view_count") or 0)
    views_root = scene_dir / "views"
    if not views_root.is_dir():
        return expected == 0
    complete_view_dirs = [
        path
        for path in sorted(views_root.iterdir())
        if path.is_dir() and _is_scene_view_complete(path)
    ]
    if len(complete_view_dirs) < expected:
        return False

    manifest_views = manifest.get("views") or []
    if len(manifest_views) < expected:
        return False
    return not any(bool(view.get("uses_hdf5_rgb")) for view in manifest_views)


def _append_scene_artifacts_to_report(
    report: EmitReport,
    *,
    scene_uid: str,
    scene_dir: Path,
) -> None:
    reported = set(report.files_written)

    def add(rel_path: str) -> None:
        entry = f"scenes/{scene_uid}/{rel_path}"
        if entry not in reported:
            report.files_written.append(entry)
            reported.add(entry)

    for rel in ("scene.usdz", "scene.usdc", "scene_source.json", "manifest.json"):
        if (scene_dir / rel).is_file():
            add(rel)
    for view in _discover_scene_views(scene_dir):
        for key in ("video", "camera", "render_meta", "frame_mapping"):
            rel = view.get(key)
            if rel:
                add(str(rel))


def _write_robotwin_scene_manifest(
    scene_dir: Path,
    *,
    dataset_id: str,
    scene_uid: str,
    episode_uid: str,
    info: dict[str, Any],
    export_result: dict[str, Any],
    render_options: dict[str, Any],
) -> list[Path]:
    summary = export_result.get("summary") or {}
    source_payload = {
        "dataset_id": dataset_id,
        "scene_uid": scene_uid,
        "episode_uid": episode_uid,
        "task_name": info.get("task_name"),
        "variant": info.get("variant"),
        "episode_stem": info.get("episode_stem"),
        "hdf5_path": info.get("hdf5_path"),
        "source_relpath": info.get("source_relpath"),
        "seed": info.get("seed"),
        "instruction": info.get("instruction"),
        "num_frames": summary.get("num_frames"),
    }
    source_path = _write_json(scene_dir / "scene_source.json", source_payload)
    views = _discover_scene_views(scene_dir)
    scene_payload = {
        "source": "scene_source.json",
    }
    if (scene_dir / "scene.usdz").is_file():
        scene_payload["usdz"] = "scene.usdz"
    if (scene_dir / "scene.usdc").is_file():
        scene_payload["usdc"] = "scene.usdc"

    manifest = {
        "format": _SCENE_VIEWS_FORMAT,
        "dataset_id": dataset_id,
        "scene_uid": scene_uid,
        "episode_uid": episode_uid,
        "scene": scene_payload,
        "source": source_payload,
        "render": {
            "source": "robotwin_live_replay",
            "scene_asset_source": "same_replay_as_views",
            "views_source": "sapien_live_generated_camera",
            "uses_hdf5_rgb": False,
            **render_options,
        },
        "views": views,
    }
    manifest_path = _write_json(scene_dir / "manifest.json", manifest)
    return [source_path, manifest_path]


# ---------------------------------------------------------------------------
# ALOHA-Agilex visual meshes
# ---------------------------------------------------------------------------
# Each entry maps a logical link name → mesh filename + visual offset (xyz, rpy)
# extracted from the URDF <visual> blocks of arx5_description_isaac.urdf.
# Vertices are baked into the link-local frame at load time so that applying
# the FK link transform places them correctly in world space.

_ALOHA_LINK_MESHES: dict[str, dict[str, Any]] = {
    # Front-arm fl_/fr_ chain links (shared mesh files for both arms)
    "base_arm": {"file": "base_arm.STL", "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link1":    {"file": "link1.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link2":    {"file": "link2.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link3":    {"file": "link3.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link4":    {"file": "link4.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link5":    {"file": "link5.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link6":    {"file": "link6.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link7":    {"file": "link7.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "link8":    {"file": "link8.STL",    "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    # Body / torso
    "box1_Link":        {"file": "box1_Link.STL",        "xyz": (0, 0, 0), "rpy": (-0.05, 0, 0)},
    "box2_Link":        {"file": "box2_Link.STL",        "xyz": (0, 0, 0), "rpy": (0, 0, 3.141592653589793)},
    "tracer_base_link": {"file": "tracer_base_link.STL", "xyz": (0, 0, 0), "rpy": (1.57, 0, 0)},
    # Head camera tower
    "camera_base_link": {"file": "camera_base_link.STL", "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "camera_link1":     {"file": "camera_link1.STL",     "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
    "camera_link2":     {"file": "camera_link2.STL",     "xyz": (0, 0, 0), "rpy": (0, 0, 0)},
}


def _find_embodiments_dir(source_path: Path | None = None) -> Path | None:
    """Locate the RoboTwin embodiments asset directory.

    Search order:
      1. ``ROBOTWIN2_EMBODIMENTS_PATH`` environment variable
      2. ``<source_path>/embodiments`` and ``<source_path>/assets/embodiments``
         (and the same one or two levels above the source dir)
      3. ``/tmp/robotwin2_assets/embodiments_extracted/embodiments``
    """
    import os
    candidates: list[Path] = []
    env = os.environ.get("ROBOTWIN2_EMBODIMENTS_PATH")
    if env:
        candidates.append(Path(env))

    if source_path:
        bases = [source_path, source_path.parent, source_path.parent.parent]
        for b in bases:
            candidates.extend([b / "embodiments", b / "assets" / "embodiments"])

    candidates.append(Path("/tmp/robotwin2_assets/embodiments_extracted/embodiments"))

    for c in candidates:
        if c.is_dir() and (c / "aloha-agilex").is_dir():
            return c
    return None


def _load_aloha_meshes(emb_dir: Path) -> dict[str, tuple[Any, Any]]:
    """Load all ALOHA-Agilex link meshes once.

    Returns ``{link_name: (vertices_Nx3, faces_Mx3)}`` with the visual
    rpy/xyz offsets already baked into the vertices (so the result is in the
    URDF link's local frame and FK transforms apply directly).
    Returns an empty dict if trimesh is unavailable or the directory is empty.
    """
    if not _TRIMESH_OK:
        return {}

    import numpy as np

    mesh_dir = (
        emb_dir / "aloha-agilex" / "urdf" / "aloha_maniskill_sim" / "meshes"
    )
    if not mesh_dir.is_dir():
        return {}

    out: dict[str, tuple[Any, Any]] = {}
    for link_name, info in _ALOHA_LINK_MESHES.items():
        path = mesh_dir / info["file"]
        if not path.is_file():
            continue
        try:
            m = trimesh.load(str(path), force="mesh")
            verts = np.asarray(m.vertices, dtype=np.float64)
            faces = np.asarray(m.faces,    dtype=np.int32)
            R = _euler_xyz_to_rot(info["rpy"])
            verts = verts @ R.T + np.asarray(info["xyz"], dtype=np.float64)
            out[link_name] = (verts, faces)
        except Exception as exc:
            logger.debug("Failed to load mesh %s: %s", path, exc)
    logger.info("Loaded %d ALOHA-Agilex meshes from %s", len(out), mesh_dir)
    return out


def _rot_to_quat(R: Any) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → quaternion (w, x, y, z) — Shepperd's method."""
    import numpy as np
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return (float(w), float(x), float(y), float(z))


def _add_static_mesh(
    stage: Any,
    prim_path: str,
    vertices: Any,
    faces: Any,
    color: Any,
    transform_T: Any | None = None,
) -> Any:
    """Create a UsdGeom.Mesh prim at *prim_path*; optional fixed transform."""
    import numpy as np
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                       for v in vertices])
    )
    counts = [3] * len(faces)
    indices = np.asarray(faces, dtype=np.int32).flatten().tolist()
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
    if transform_T is not None:
        T = np.asarray(transform_T, dtype=np.float64)
        xf = UsdGeom.Xformable(mesh.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        ))
        qw, qx, qy, qz = _rot_to_quat(T[:3, :3])
        xf.AddOrientOp().Set(Gf.Quatf(qw, qx, qy, qz))
    return mesh


def _add_animated_mesh(
    stage: Any,
    prim_path: str,
    vertices: Any,
    faces: Any,
    color: Any,
    times_T: dict[int, Any],
) -> Any:
    """Create an animated UsdGeom.Mesh prim with per-frame translate+orient."""
    import numpy as np
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                       for v in vertices])
    )
    counts = [3] * len(faces)
    indices = np.asarray(faces, dtype=np.int32).flatten().tolist()
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))

    xf = UsdGeom.Xformable(mesh.GetPrim())
    tr_op = xf.AddTranslateOp()
    or_op = xf.AddOrientOp()
    for t in sorted(times_T.keys()):
        T = np.asarray(times_T[t], dtype=np.float64)
        tr_op.Set(
            Gf.Vec3d(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
            Usd.TimeCode(t),
        )
        qw, qx, qy, qz = _rot_to_quat(T[:3, :3])
        or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))
    return mesh


# ---------------------------------------------------------------------------
# Forward kinematics — pure-numpy implementation of the URDF chain
# ---------------------------------------------------------------------------


def _euler_xyz_to_rot(rpy: tuple[float, float, float]) -> Any:
    """ROS-style fixed-axis XYZ Euler → 3×3 rotation matrix (Rz·Ry·Rx)."""
    import numpy as np
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr,  cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_to_rot(axis: tuple[float, float, float], angle: float) -> Any:
    """Rodrigues rotation: rotation matrix from axis+angle."""
    import numpy as np
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = a
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _make_T(xyz: tuple[float, float, float],
            rpy: tuple[float, float, float] = (0, 0, 0)) -> Any:
    """4×4 homogeneous transform from translation + Euler-XYZ rotation."""
    import numpy as np
    T = np.eye(4)
    T[:3, :3] = _euler_xyz_to_rot(rpy)
    T[:3, 3]  = xyz
    return T


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> Any:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    import numpy as np
    n = max((qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5, 1e-12)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def _robot_world_T() -> Any:
    """4×4 transform that places the robot footprint in world space."""
    import numpy as np
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(*_ROBOT_WORLD_QUAT)
    T[:3, 3]  = _ROBOT_WORLD_POS
    return T


def _arm_world_base_T(side: str) -> Any:
    """World-frame transform of a front-arm base ('left' or 'right')."""
    offset = _LEFT_ARM_BASE_XYZ if side == "left" else _RIGHT_ARM_BASE_XYZ
    return _robot_world_T() @ _make_T(offset)


def _forward_kinematics(base_T: Any, joint_angles: Any) -> list[Any]:
    """Run FK on the ARX5 front-arm chain.

    Returns a list of (N+1) 4×4 world-frame transforms:
        [base, link1, link2, link3, link4, link5, link6 (EE)]
    """
    import numpy as np
    T = base_T.copy()
    out = [T.copy()]
    for joint, angle in zip(_ARX5_FRONT_CHAIN, joint_angles):
        T_origin = _make_T(joint["xyz"], joint["rpy"])
        R = np.eye(4)
        R[:3, :3] = _axis_angle_to_rot(joint["axis"], float(angle))
        T = T @ T_origin @ R
        out.append(T.copy())
    return out


def _z_to_vec_quat(direction: Any) -> Any:
    """Quaternion that rotates the +Z axis to align with *direction*.

    Returns a Gf.Quatf(w, x, y, z). Identity if *direction* is too small.
    """
    import numpy as np
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9 or not _USD_OK:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0) if _USD_OK else None
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if dot >  0.999999:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    if dot < -0.999999:
        # 180° around any axis perpendicular to Z — pick X
        return Gf.Quatf(0.0, 1.0, 0.0, 0.0)
    axis = np.cross(z, d)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(dot))
    half = angle * 0.5
    s = float(np.sin(half))
    return Gf.Quatf(float(np.cos(half)),
                    float(axis[0]) * s,
                    float(axis[1]) * s,
                    float(axis[2]) * s)


# ---------------------------------------------------------------------------
# Held-object heuristic — recover block + target-pad poses from EE trajectory
# ---------------------------------------------------------------------------
#
# RoboTwin 2.0 episodes don't persist manipulated-object state to disk (the
# `pointcloud` dataset is empty, the pickled `_traj_data` only has joint-plan
# data, and `scene_info.json` describes textures/clutter, not tracked objects).
# But we *do* know which arm has its gripper closed at each frame, so we can
# reconstruct a good-enough block trajectory:
#
#   - gripper < 0.5  →  that arm is grasping; block pose follows its EE
#   - both grippers closed simultaneously → handover, block is at midpoint
#   - before any grasp → block is stationary where it is first grasped
#   - after the final release → block stays where it was placed
#
# The final release location is also a good proxy for the blue target-pad
# position (handover_block, place_a2b_* and similar tasks all drop the block
# on the target at the end of the episode).
#
# Gripper convention in RoboTwin 2.0 HDF5:
#     0.0 = fully CLOSED (fingers together, object gripped)
#     1.0 = fully OPEN   (fingers spread, no contact)

_GRIPPER_CLOSED_THRESHOLD = 0.5

# The URDF EE frame (fl_link6 / fr_link6) is the wrist, not the grip point.
# Midpoint of joint7 / joint8 in link6-local frame ≈ (0.08457, 0, -0.0001).
# Apply this offset (rotated by the EE orientation) to get the actual
# between-fingers position where the manipulated object is centred.
_GRIP_OFFSET_IN_EE = (0.08457, 0.0, -0.0001)


def _apply_grip_offset(ee_pose_7: Any) -> Any:
    """Transform an EE pose [x,y,z, qw,qx,qy,qz] to the grip-centre pose.

    Position is translated by R(q) × grip_offset; orientation is preserved.
    """
    import numpy as np
    pos = np.asarray(ee_pose_7[:3], dtype=np.float64)
    qw, qx, qy, qz = (
        float(ee_pose_7[3]), float(ee_pose_7[4]),
        float(ee_pose_7[5]), float(ee_pose_7[6]),
    )
    R = _quat_to_rot(qw, qx, qy, qz)
    grip_pos = pos + R @ np.asarray(_GRIP_OFFSET_IN_EE, dtype=np.float64)
    out = np.empty(7, dtype=np.float64)
    out[:3] = grip_pos
    out[3:7] = ee_pose_7[3:7]
    return out


def _estimate_object_trajectory(
    T: int,
    ee_poses: Any,               # (T, 2, 7) — [left, right] × [x,y,z,qw,qx,qy,qz]
    gripper_states: Any,         # {"left": (T,), "right": (T,)} or None
) -> tuple[Any, Any] | tuple[None, None]:
    """Heuristically reconstruct (block_pose_per_frame, target_pad_pose).

    Returns ``(None, None)`` when the input lacks gripper data or when no
    grasp is ever detected (implying the episode doesn't manipulate an object).
    """
    import numpy as np

    if ee_poses is None or not gripper_states:
        return None, None
    lg = gripper_states.get("left")
    rg = gripper_states.get("right")
    if lg is None or rg is None:
        return None, None

    lg = np.asarray(lg, dtype=np.float64)
    rg = np.asarray(rg, dtype=np.float64)

    # A smooth closure weight in [0, 1]: 0 when the gripper is fully open
    # (g >= 0.5), ramps linearly to 1 as it closes (g → 0). Using this as a
    # blend weight (instead of a hard threshold) eliminates the "teleport"
    # artefact during handover, when left opens while right closes.
    def _closure_weight(g: Any) -> Any:
        return np.clip(1.0 - 2.0 * np.asarray(g, dtype=np.float64), 0.0, 1.0)

    w_left  = _closure_weight(lg)
    w_right = _closure_weight(rg)
    any_closed = (lg < _GRIPPER_CLOSED_THRESHOLD) | (rg < _GRIPPER_CLOSED_THRESHOLD)
    if not any_closed.any():
        return None, None  # no manipulation detected

    # Pre-compute the grip-centre pose for every frame per arm.
    left_grip  = np.stack([_apply_grip_offset(ee_poses[t, 0]) for t in range(T)])
    right_grip = np.stack([_apply_grip_offset(ee_poses[t, 1]) for t in range(T)])

    first_grasp = int(np.argmax(any_closed))

    block_pose = np.zeros((T, 7), dtype=np.float64)
    last_pose: Any = None

    for t in range(T):
        wl, wr = float(w_left[t]), float(w_right[t])
        total = wl + wr
        if total > 1e-6:
            # Smooth weighted blend between the two arms' grip points.
            al = wl / total
            ar = wr / total
            pos = al * left_grip[t, :3] + ar * right_grip[t, :3]
            # Orientation from the dominantly-closed gripper
            orient_src = left_grip[t] if wl >= wr else right_grip[t]
            block_pose[t, :3] = pos
            block_pose[t, 3:7] = orient_src[3:7]
            last_pose = block_pose[t].copy()
        else:
            # Neither arm is closing — block is either pre-grasp or post-release
            if last_pose is not None:
                block_pose[t] = last_pose
            else:
                # Pre-grasp: block rests at the first-grasp grip point
                anchor = (
                    left_grip[first_grasp]
                    if w_left[first_grasp] >= w_right[first_grasp]
                    else right_grip[first_grasp]
                )
                block_pose[t] = anchor

    target_pad_pose = block_pose[-1].copy()
    return block_pose, target_pad_pose


def _make_box_mesh(size: tuple[float, float, float]) -> tuple[Any, Any]:
    """Axis-aligned box centred at origin.

    Returns ``(vertices_Nx3, faces_Nx3)`` (triangulated, right-handed outward).
    """
    import numpy as np
    sx, sy, sz = (s * 0.5 for s in size)
    verts = np.array([
        [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
        [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz],
    ], dtype=np.float64)
    # Triangles with outward-facing normals
    faces = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom (−Z)
        [4, 5, 6], [4, 6, 7],   # top    (+Z)
        [0, 1, 5], [0, 5, 4],   # −Y
        [1, 2, 6], [1, 6, 5],   # +X
        [2, 3, 7], [2, 7, 6],   # +Y
        [3, 0, 4], [3, 4, 7],   # −X
    ], dtype=np.int32)
    return verts, faces


def _add_box_mesh(
    stage: Any,
    prim_path: str,
    centre: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    color: Any,
) -> None:
    """Define a static axis-aligned box mesh at *centre* with given half-extents."""
    if not _USD_OK:
        return
    size = (half_extents[0] * 2.0, half_extents[1] * 2.0, half_extents[2] * 2.0)
    verts, faces = _make_box_mesh(size)
    import numpy as np
    T = np.eye(4)
    T[:3, 3] = centre
    _add_static_mesh(stage, prim_path, verts, faces, color, T)


def _build_table(
    stage: Any,
    *,
    pose_xyz: tuple[float, float, float],
    length: float,
    width: float,
    height: float,
    thickness: float,
) -> None:
    """Recreate the SAPIEN create_table() geometry in USD (tabletop + 4 legs).

    Geometry (relative to *pose_xyz*), matching ``envs/utils/create_actor.py``:
      * tabletop at ``(0, 0, -thickness/2)``, half-size ``(L/2, W/2, t/2)``
      * four legs at ``(±(L/2 − 0.05), ±(W/2 − 0.05))``, each half-size
        ``(thickness/2, thickness/2, height/2 − 0.002)`` centred at
        ``z = -height/2 − 0.002`` relative to the pose.
    """
    if not _USD_OK:
        return

    px, py, pz = pose_xyz
    leg_spacing = 0.1      # matches create_table()
    leg_len     = height - 0.004

    UsdGeom.Xform.Define(stage, "/Scene/Table")

    # Tabletop
    _add_box_mesh(
        stage, "/Scene/Table/Top",
        centre=(px, py, pz - thickness / 2.0),
        half_extents=(length / 2.0, width / 2.0, thickness / 2.0),
        color=Gf.Vec3f(0.82, 0.74, 0.60),   # warm beige
    )

    # Legs
    for i, sx in enumerate((-1, 1)):
        for j, sy in enumerate((-1, 1)):
            lx = px + sx * (length / 2.0 - leg_spacing / 2.0)
            ly = py + sy * (width  / 2.0 - leg_spacing / 2.0)
            lz = pz - height / 2.0 - 0.002
            _add_box_mesh(
                stage, f"/Scene/Table/Leg{i * 2 + j}",
                centre=(lx, ly, lz),
                half_extents=(thickness / 2.0, thickness / 2.0, leg_len / 2.0),
                color=Gf.Vec3f(0.45, 0.38, 0.30),
            )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter
class RoboTwin2Adapter(DatasetAdapter):
    """Adapter for RoboTwin 2.0 bimanual manipulation dataset."""

    name: str = "robotwin2"
    version: str = "0.1.0"

    def __init__(self) -> None:
        # Populated during parse_raw; keyed by source_episode_id.
        # value: {"hdf5_path": str, "task_name": str, "variant": str,
        #         "episode_stem": str, "instruction": str | None}
        self._episode_map: dict[str, dict[str, Any]] = {}

        # Populated during normalize; maps source_episode_id → episode_uid
        self._uid_map: dict[str, str] = {}
        self._options: dict[str, Any] = {}
        self._source_root: Path | None = None

    def prepare_emit(
        self,
        config: AdapterConfig,
        parse_bundle: ParseBundle,
        normalize_bundle: NormalizeBundle,
    ) -> None:
        """Restore emit-only state when materialize runs from saved artifacts."""
        self._options = dict(config.options)
        source = Path(config.source_path) if config.source_path else None
        self._source_root = source.resolve() if source is not None and source.exists() else None

        self._episode_map = {}
        for ep_dict in parse_bundle.instances:
            source_episode_id = ep_dict.get("source_episode_id")
            hdf5_path = ep_dict.get("hdf5_path")
            if not source_episode_id or not hdf5_path:
                continue
            self._episode_map[str(source_episode_id)] = {
                "hdf5_path": str(hdf5_path),
                "task_name": ep_dict.get("task_name", "unknown"),
                "variant": ep_dict.get("variant", "unknown"),
                "episode_stem": ep_dict.get("episode_stem") or Path(str(hdf5_path)).stem,
                "instruction": ep_dict.get("instruction"),
                "seed": ep_dict.get("seed"),
                "source_relpath": ep_dict.get("source_relpath"),
            }

        self._uid_map = {
            str(episode.source_episode_id): episode.episode_uid
            for episode in normalize_bundle.episodes
            if episode.source_episode_id
        }

    def capabilities(self) -> dict[str, bool]:
        return {
            "scene_mesh": False,
            "object_mesh": False,
            "articulation": True,       # dual-arm joint states
            "deformable_mesh": False,
            "camera": True,             # front + head + wrist RGB cameras
            "depth": True,              # depth present in some releases
            "lidar": False,
            "tracks": True,             # joint state trajectories
            "videos": True,             # generated-camera live SAPIEN views
            "sdk_required": True,        # native SAPIEN/RoboTwin replay is required
            "license_gated": False,
            "supports_local_ingest": True,
        }

    # ------------------------------------------------------------------
    # inventory
    # ------------------------------------------------------------------

    def inventory(
        self, config: AdapterConfig, ctx: JobContext
    ) -> list[SourceItem]:
        source = Path(config.source_path) if config.source_path else None
        self._options = dict(config.options)
        self._source_root = source.resolve() if source is not None and source.exists() else None
        if source is None or not source.is_dir():
            logger.warning(
                "RoboTwin2 source_path is not set or not a directory: %s",
                config.source_path,
            )
            return []

        dataset_id = config.dataset_id or DATASET_ID
        ingest_clean = _opt(config, "ingest_clean", True)
        ingest_randomized = _opt(config, "ingest_randomized", True)
        task_names = _as_name_set(_opt(config, "task_names"))
        variant_names = _as_name_set(_opt(config, "variant_names"))
        task_limit = _as_positive_int(_opt(config, "task_limit"))
        episodes_per_task = _as_positive_int(
            _opt(
                config,
                "episodes_per_task",
                _opt(config, "trajectories_per_task"),
            )
        )
        items: list[SourceItem] = []
        selected_task_count = 0

        # Walk: source/<task_name>/<variant>/data/episode*.hdf5
        for task_dir in sorted(source.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            if task_names is not None and task_name not in task_names:
                continue

            task_items: list[SourceItem] = []

            for variant_dir in sorted(task_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                variant_name = variant_dir.name
                if variant_names is not None and variant_name not in variant_names:
                    continue

                is_clean = "clean" in variant_name.lower()
                is_rand = (
                    "randomized" in variant_name.lower()
                    or "rand" in variant_name.lower()
                )

                if is_clean and not ingest_clean:
                    continue
                if is_rand and not ingest_randomized:
                    continue

                data_dir = variant_dir / "data"
                if not data_dir.is_dir():
                    data_dir = variant_dir  # fallback: HDF5 directly in variant_dir

                instructions_dir = variant_dir / "instructions"
                seed = _load_variant_seed(variant_dir)

                for ep_path in sorted(_hdf5_files(data_dir)):
                    ep_stem = ep_path.stem  # e.g. "episode0"
                    instruction = None
                    if instructions_dir.is_dir():
                        instruction = _load_instruction(instructions_dir, ep_stem)

                    rel = ep_path.relative_to(source)
                    item_id = str(rel)
                    task_items.append(
                        SourceItem(
                            item_id=item_id,
                            dataset_id=dataset_id,
                            item_type="episode",
                            source_path=str(ep_path),
                            metadata={
                                "task_name": task_name,
                                "variant": variant_name,
                                "episode_stem": ep_stem,
                                "is_clean": is_clean,
                                "is_randomized": is_rand,
                                "instruction": instruction,
                                "hdf5_relpath": str(rel),
                                "seed": seed,
                            },
                        )
                    )

            if episodes_per_task is not None:
                task_items = task_items[:episodes_per_task]
            if task_items:
                items.extend(task_items)
                selected_task_count += 1
                if task_limit is not None and selected_task_count >= task_limit:
                    break

        if ctx.limit is not None:
            items = items[: ctx.limit]

        logger.info(
            "RoboTwin2 inventory: %d episodes found under %s",
            len(items),
            source,
        )
        return items

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------

    def fetch(
        self, items: list[SourceItem], ctx: JobContext
    ) -> list[RawRef]:
        if not items:
            return []

        dataset_id = items[0].dataset_id
        raw_store = RawStore(ctx.raw_root)

        all_source_paths = [
            Path(it.source_path).resolve()
            for it in items
            if it.source_path
        ]
        if not all_source_paths:
            return []

        # Find the common ancestor directory
        common_root = all_source_paths[0].parent
        while common_root != common_root.parent:
            if all(str(p).startswith(str(common_root)) for p in all_source_paths):
                break
            common_root = common_root.parent

        linked_dir = raw_store.link_directory(dataset_id, common_root)

        refs: list[RawRef] = []
        for item in items:
            if item.source_path:
                src = Path(item.source_path).resolve()
                try:
                    rel = src.relative_to(common_root)
                except ValueError:
                    rel = Path(src.name)
                refs.append(
                    RawRef(
                        item_id=item.item_id,
                        raw_path=str(linked_dir / rel),
                    )
                )
        return refs

    # ------------------------------------------------------------------
    # parse_raw
    # ------------------------------------------------------------------

    def parse_raw(
        self, raw_refs: list[RawRef], ctx: JobContext
    ) -> ParseBundle:
        if not raw_refs:
            return ParseBundle(dataset_id=DATASET_ID)

        bundle = ParseBundle(dataset_id=DATASET_ID, raw_refs=raw_refs)

        for ref in raw_refs:
            raw_path = Path(ref.raw_path)
            if not raw_path.is_file():
                logger.warning("RoboTwin2: raw file not found: %s", raw_path)
                continue
            if raw_path.suffix not in (".hdf5", ".h5"):
                continue

            if h5py is None:
                logger.warning(
                    "h5py not installed; skipping episode %s", raw_path
                )
                continue

            try:
                ep_data = _read_hdf5_episode(raw_path)
            except Exception:
                logger.exception("Failed to read HDF5: %s", raw_path)
                continue

            task_name, variant, ep_stem, source_relpath = _episode_identity_from_ref(
                raw_path,
                ref,
            )
            seed = _load_variant_seed(raw_path.parent.parent)
            if self._source_root is not None:
                if source_relpath is None:
                    try:
                        source_relpath = str(raw_path.resolve().relative_to(self._source_root))
                    except ValueError:
                        source_relpath = None

            source_episode_id = f"{task_name}/{variant}/{ep_stem}"

            ep_dict: dict[str, Any] = {
                "source_episode_id": source_episode_id,
                "task_name": task_name,
                "variant": variant,
                "episode_stem": ep_stem,
                "hdf5_path": str(raw_path),
                "num_steps": ep_data["num_steps"],
                "has_cameras": ep_data["has_cameras"],
                "has_depth": ep_data["has_depth"],
                "action_dim": ep_data["action_dim"],
                "has_joint_state": ep_data["has_joint_state"],
                "has_ee_pose": ep_data["has_ee_pose"],
                "success": ep_data["success"],
                "seed": seed,
                "source_relpath": source_relpath,
            }

            # Load instruction from instructions/ sibling directory
            instructions_dir = raw_path.parent.parent / "instructions"
            if instructions_dir.is_dir():
                ep_dict["instruction"] = _load_instruction(
                    instructions_dir, ep_stem
                )

            bundle.instances.append(ep_dict)

            # Stash full info for emit()
            self._episode_map[source_episode_id] = {
                "hdf5_path": str(raw_path),
                "task_name": task_name,
                "variant": variant,
                "episode_stem": ep_stem,
                "instruction": ep_dict.get("instruction"),
                "seed": seed,
                "source_relpath": source_relpath,
            }

            # Camera sensor entries
            for cam_name in ep_data["has_cameras"]:
                bundle.sensors.append(
                    {
                        "sensor_type": "camera",
                        "name": cam_name,
                        "episode_source_id": source_episode_id,
                        "has_depth": cam_name in ep_data["has_depth"],
                    }
                )

        logger.info(
            "RoboTwin2 parse: %d episodes, %d sensor entries",
            len(bundle.instances),
            len(bundle.sensors),
        )
        return bundle

    # ------------------------------------------------------------------
    # normalize
    # ------------------------------------------------------------------

    def normalize(
        self, bundle: ParseBundle, ctx: JobContext
    ) -> NormalizeBundle:
        dataset_id = bundle.dataset_id
        now = datetime.now(tz=timezone.utc)
        out = NormalizeBundle(dataset_id=dataset_id)

        out.dataset_record = DatasetRecord(
            dataset_id=dataset_id,
            dataset_name="RoboTwin 2.0",
            version="2.0",
            source_type=SourceType.LOCAL_FOLDER,
            source_uri="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0",
            license_name="Apache-2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
            access_mode=AccessMode.PUBLIC,
            geometry_level_max=GeometryLevel.G0_NONE,
            created_at=now,
            tags=[
                "manipulation",
                "robotics",
                "simulation",
                "bimanual",
                "dual-arm",
                "domain-randomization",
                "sapien",
            ],
        )

        # Sensor registry: one record per (scene, camera) pair
        seen_sensors: set[str] = set()

        for ep_dict in bundle.instances:
            source_episode_id = ep_dict["source_episode_id"]
            task_name = ep_dict.get("task_name", "unknown")
            variant = ep_dict.get("variant", "unknown")
            num_steps = ep_dict.get("num_steps", 0) or 0

            episode_uid = make_episode_uid(dataset_id, source_episode_id)
            scene_uid = make_scene_uid(dataset_id, f"episode:{source_episode_id}")

            # --- Scene record (one per episode) ---
            out.scenes.append(
                SceneRecord(
                    scene_uid=scene_uid,
                    dataset_id=dataset_id,
                    source_scene_id=f"episode:{source_episode_id}",
                    scene_name=(
                        f"{task_name}/{variant}/{ep_dict.get('episode_stem', '')}"
                    ),
                    scene_kind=SceneKind.SYNTHETIC_MANIPULATION,
                    geometry_level=GeometryLevel.G0_NONE,
                    num_frames=num_steps,
                    has_static_scene_mesh=False,
                    has_dynamic_objects=True,
                    has_humans=False,
                    has_articulation=True,
                )
            )

            out.episodes.append(
                EpisodeRecord(
                    episode_uid=episode_uid,
                    dataset_id=dataset_id,
                    scene_uid=scene_uid,
                    source_episode_id=source_episode_id,
                    num_frames=num_steps,
                )
            )

            # --- Camera sensors ---
            for cam_name in ep_dict.get("has_cameras", []):
                sensor_key = f"{scene_uid}:{cam_name}"
                if sensor_key not in seen_sensors:
                    seen_sensors.add(sensor_key)
                    sensor_uid = make_sensor_uid(dataset_id, scene_uid, cam_name)
                    has_depth = cam_name in ep_dict.get("has_depth", [])
                    out.sensors.append(
                        SensorRecord(
                            sensor_uid=sensor_uid,
                            scene_uid=scene_uid,
                            episode_uid=episode_uid,
                            sensor_type=(
                                SensorType.DEPTH_CAMERA
                                if has_depth
                                else SensorType.CAMERA
                            ),
                            name=cam_name,
                        )
                    )

            # --- Robot instance ---
            robot_uid = make_instance_uid(dataset_id, episode_uid, "robot")
            out.instances.append(
                InstanceRecord(
                    instance_uid=robot_uid,
                    scene_uid=scene_uid,
                    episode_uid=episode_uid,
                    category="robot",
                    instance_name="dual_arm_robot",
                    is_static=False,
                    is_articulated=True,
                    is_human=False,
                    geometry_level=GeometryLevel.G0_NONE,
                )
            )

            # --- Track + articulation states ---
            if num_steps > 0 and ep_dict.get("has_joint_state", False):
                track_uid = make_track_uid(dataset_id, robot_uid)
                joint_positions = _read_joint_position_series(
                    Path(str(ep_dict.get("hdf5_path", "")))
                )
                joint_dim = (
                    len(joint_positions[0])
                    if joint_positions
                    else int(ep_dict.get("action_dim") or _ARM_DOFS)
                )
                joint_names = [f"joint_{i}" for i in range(joint_dim)]
                for step_idx in range(num_steps):
                    ts_ns = step_idx * 1_000_000  # 1 ms per step
                    out.track_states.append(
                        TrackStateRecord(
                            track_uid=track_uid,
                            instance_uid=robot_uid,
                            timestamp_ns=ts_ns,
                        )
                    )
                    out.articulation_states.append(
                        ArticulationStateRecord(
                            instance_uid=robot_uid,
                            asset_uid="",
                            timestamp_ns=ts_ns,
                            joint_names=joint_names,
                            joint_positions=(
                                joint_positions[step_idx]
                                if joint_positions and step_idx < len(joint_positions)
                                else [0.0] * joint_dim
                            ),
                        )
                    )

            self._uid_map[source_episode_id] = episode_uid

        # --- License ---
        out.licenses.append(
            LicenseRecord(
                record_scope=RecordScope.DATASET,
                record_id=dataset_id,
                license_name="Apache-2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
                commercial_use_allowed=True,
                redistribution_allowed=True,
                attribution_required=True,
                notes=(
                    "RoboTwin 2.0 is released under the Apache 2.0 license. "
                    "See https://github.com/RoboTwin-Platform/RoboTwin for details."
                ),
            )
        )

        # --- Provenance ---
        out.provenance.append(
            ProvenanceRecord(
                record_id=dataset_id,
                dataset_id=dataset_id,
                normalized_by_version=self.version,
                normalized_at=now,
                adapter_name=self.name,
                adapter_version=self.version,
                transform_log=[
                    {
                        "step": "normalize",
                        "scenes": len(out.scenes),
                        "sensors": len(out.sensors),
                        "instances": len(out.instances),
                        "track_states": len(out.track_states),
                        "articulation_states": len(out.articulation_states),
                    }
                ],
            )
        )

        return out

    # ------------------------------------------------------------------
    # emit  — writes canonical records then generates remote-native scene views
    # ------------------------------------------------------------------

    def emit(self, bundle: NormalizeBundle, ctx: JobContext) -> EmitReport:
        if ctx.dry_run:
            logger.info("Dry-run: skipping RoboTwin2 materialization")
            return EmitReport(dataset_id=bundle.dataset_id)

        self._require_remote_replay_config(ctx)
        if bundle.episodes and not self._episode_map:
            raise RuntimeError(
                "RoboTwin2 materialize requires parsed episode source paths. "
                "Run through the sim project stages or call prepare_emit before emit()."
            )

        report = super().emit(bundle, ctx)

        from guanwu.storage.canonical_store import CanonicalStore

        store = CanonicalStore(ctx.canonical_root)
        episode_scene_uids = {
            episode.episode_uid: episode.scene_uid
            for episode in bundle.episodes
            if episode.scene_uid
        }

        for source_ep_id, info in self._episode_map.items():
            episode_uid = self._uid_map.get(source_ep_id)
            if episode_uid is None:
                continue

            scene_uid = episode_scene_uids.get(episode_uid) or make_scene_uid(
                bundle.dataset_id,
                f"episode:{source_ep_id}",
            )
            scene_dir = store.scene_dir(bundle.dataset_id, scene_uid)
            self._emit_episode_remote(
                bundle=bundle,
                ctx=ctx,
                info=info,
                episode_uid=episode_uid,
                scene_uid=scene_uid,
                scene_dir=scene_dir,
                report=report,
            )

        return report

    def _require_remote_replay_config(self, ctx: JobContext) -> None:
        if not ctx.remote_host:
            raise RuntimeError(
                "RoboTwin2 materialize requires a remote GPU host for native replay "
                "(runtime.remote.host)."
            )
        if not self._options.get("remote_engine_export", True):
            raise RuntimeError(
                "RoboTwin2 local fallback export has been removed; "
                "robotwin2.options.remote_engine_export must stay true."
            )
        if not self._options.get("remote_robotwin_root"):
            raise RuntimeError(
                "RoboTwin2 materialize requires "
                "robotwin2.options.remote_robotwin_root for native replay."
            )

    def _emit_episode_remote(
        self,
        *,
        bundle: NormalizeBundle,
        ctx: JobContext,
        info: dict[str, Any],
        episode_uid: str,
        scene_uid: str,
        scene_dir: Path,
        report: EmitReport,
    ) -> bool:
        """Run authoritative RoboTwin replay on the remote GPU and download outputs."""
        h5_path = Path(info["hdf5_path"])
        if not h5_path.is_file():
            raise RuntimeError(f"HDF5 not found for RoboTwin2 remote export: {h5_path}")

        explicit_view_specs = _load_view_specs(
            self._options.get("view_specs")
            or self._options.get("camera_view_specs")
            or self._options.get("generated_view_specs")
        )
        replay_seed = int(info.get("seed") or self._options.get("remote_default_seed", 0))
        render_options = _build_render_options(
            self._options,
            explicit_view_specs=explicit_view_specs,
            replay_seed=replay_seed,
        )
        force_export = _as_bool(
            self._options.get(
                "force_remote_export",
                self._options.get("force_render_views"),
            ),
            False,
        )
        if not force_export and _scene_export_complete(scene_dir, render_options):
            _append_scene_artifacts_to_report(
                report,
                scene_uid=scene_uid,
                scene_dir=scene_dir,
            )
            logger.info(
                "Skipping complete RoboTwin scene export for %s: %s",
                episode_uid,
                render_options["render_options_hash"],
            )
            return False

        try:
            from guanwu.core.config import RemoteConfig
            from guanwu.core.remote import get_remote_executor
            from guanwu.core.remote_tasks import export_robotwin_episode

            remote_cfg = RemoteConfig(
                host=ctx.remote_host,
                conda_env=ctx.remote_conda_env,
                work_dir=ctx.remote_work_dir,
                python=ctx.remote_python,
                conda_init=ctx.remote_conda_init,
            )
            executor = get_remote_executor(remote_cfg)
            if executor is None:
                raise RuntimeError(f"Remote executor is unavailable for {ctx.remote_host!r}")

            result = export_robotwin_episode(
                executor,
                h5_path=h5_path,
                task_name=info.get("task_name", "unknown"),
                variant_name=info.get("variant", "unknown"),
                episode_stem=info.get("episode_stem", h5_path.stem),
                seed=replay_seed,
                robotwin_root=str(self._options["remote_robotwin_root"]),
                output_dir=scene_dir,
                source_relpath=info.get("source_relpath"),
                remote_source_root=self._options.get("remote_source_root"),
                render_videos=False,
                render_views=bool(render_options["render_views"]),
                view_specs=explicit_view_specs,
                generated_camera_motion=render_options["generated_camera_motion"],
                generated_static_camera_count=render_options["generated_static_camera_count"],
                generated_trajectory_camera_count=render_options["generated_trajectory_camera_count"],
                camera_trajectory_kind=render_options["camera_trajectory_kind"],
                trajectory_kind_mode=render_options["trajectory_kind_mode"],
                camera_seed=(
                    int(render_options["camera_seed"])
                    if render_options["camera_seed"] is not None
                    else None
                ),
                view_width_px=int(render_options["view_width_px"]),
                view_height_px=int(render_options["view_height_px"]),
                embodiment=str(render_options["embodiment"]),
                fps=float(render_options["usdc_fps"]),
                video_fps=int(render_options["video_fps"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Remote native RoboTwin replay export failed for {h5_path}"
            ) from exc

        scene_usdz = scene_dir / "scene.usdz"
        if not scene_usdz.is_file():
            raise RuntimeError(
                f"Remote native RoboTwin replay did not produce {scene_usdz}"
            )
        renders_dir = scene_dir / "renders"
        if renders_dir.exists():
            shutil.rmtree(renders_dir)

        manifest_paths = _write_robotwin_scene_manifest(
            scene_dir,
            dataset_id=bundle.dataset_id,
            scene_uid=scene_uid,
            episode_uid=episode_uid,
            info=info,
            export_result=result,
            render_options=render_options,
        )

        files_written = result.get("files_written", [])
        reported_scene = False
        reported: set[str] = set(report.files_written)
        for rel_path in files_written:
            rel = rel_path.replace("\\", "/")
            if rel.startswith("renders/"):
                continue
            if rel == "scene.usdz":
                reported_scene = True
            entry = f"scenes/{scene_uid}/{rel}"
            if entry not in reported:
                report.files_written.append(entry)
                reported.add(entry)
        if not reported_scene:
            entry = f"scenes/{scene_uid}/scene.usdz"
            if entry not in reported:
                report.files_written.append(entry)
                reported.add(entry)
        for path in manifest_paths:
            entry = f"scenes/{scene_uid}/{path.relative_to(scene_dir).as_posix()}"
            if entry not in reported:
                report.files_written.append(entry)
                reported.add(entry)
        logger.info(
            "Remote RoboTwin export complete for %s: %d files",
            episode_uid,
            len([p for p in files_written if not str(p).startswith("renders/")]),
        )
        return True


def _camera_image_size_from_intrinsics(intrinsic: Any) -> tuple[int, int] | None:
    import numpy as np

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


def _write_usd_cameras(
    stage: Any,
    *,
    camera_poses: Any,
    camera_intrinsics: Any | None,
    camera_image_sizes: Any | None,
    frame_count: int,
) -> None:
    """Author HDF5 observation cameras as USD camera prims.

    RoboTwin stores ``cam2world_gl`` in an OpenGL-style camera basis. USD cameras
    use the same visual basis (-Z forward, +Y up), so the matrix can be authored
    directly as the camera-to-world transform.
    """
    import numpy as np

    if not camera_poses:
        return

    UsdGeom.Xform.Define(stage, "/Scene/Cameras")
    for cam_name in sorted(camera_poses):
        poses = np.asarray(camera_poses[cam_name], dtype=np.float64)
        if poses.ndim == 2:
            poses = poses[None, ...]
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            logger.warning("Skipping camera %s with invalid pose shape %s", cam_name, poses.shape)
            continue

        prim_name = _sanitize_usd_name(cam_name)
        cam = UsdGeom.Camera.Define(stage, f"/Scene/Cameras/{prim_name}")
        cam.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        cam.GetPrim().CreateAttribute(
            "robotwin:sourceCameraName",
            Sdf.ValueTypeNames.String,
        ).Set(str(cam_name))
        cam.GetPrim().CreateAttribute(
            "robotwin:cameraPoseSource",
            Sdf.ValueTypeNames.String,
        ).Set("hdf5:observation/<camera>/cam2world_gl")

        intrinsic = None
        if camera_intrinsics and cam_name in camera_intrinsics:
            intrinsic = np.asarray(camera_intrinsics[cam_name], dtype=np.float64)
            if intrinsic.ndim == 3:
                intrinsic = intrinsic[0]
            if intrinsic.shape == (3, 3):
                cam.GetPrim().CreateAttribute(
                    "robotwin:intrinsicCv",
                    Sdf.ValueTypeNames.FloatArray,
                ).Set(Vt.FloatArray(intrinsic.reshape(-1).astype(float).tolist()))
            else:
                intrinsic = None

        image_size = None
        if camera_image_sizes and cam_name in camera_image_sizes:
            image_size = camera_image_sizes[cam_name]
        if image_size is None:
            image_size = _camera_image_size_from_intrinsics(intrinsic)
        if image_size is not None:
            width, height = int(image_size[0]), int(image_size[1])
            cam.GetPrim().CreateAttribute(
                "robotwin:imageWidth",
                Sdf.ValueTypeNames.Int,
            ).Set(width)
            cam.GetPrim().CreateAttribute(
                "robotwin:imageHeight",
                Sdf.ValueTypeNames.Int,
            ).Set(height)
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

        xf = UsdGeom.Xformable(cam.GetPrim())
        tr_op = xf.AddTranslateOp()
        or_op = xf.AddOrientOp()
        samples = min(frame_count, poses.shape[0])
        for t in range(samples):
            T = poses[t]
            tr_op.Set(
                Gf.Vec3d(float(T[0, 3]), float(T[1, 3]), float(T[2, 3])),
                Usd.TimeCode(t),
            )
            qw, qx, qy, qz = _rot_to_quat(T[:3, :3])
            or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))


# ---------------------------------------------------------------------------
# Animated USDC writer
# ---------------------------------------------------------------------------


def _write_animated_usdc(
    path: Path,
    *,
    T: int,
    ee_poses: Any,          # (T, 2, 7) — [left, right] × [x,y,z, qw,qx,qy,qz]
    joint_states: Any,      # (T, 14) or None
    gripper_states: Any,    # dict {"left": (T,), "right": (T,)} or None
    camera_poses: Any,      # dict {cam_name: (T,4,4) float32} or None
    arm_joint_angles: Any,  # dict {"left": (T,6), "right": (T,6)} or None — for FK
    mesh_data: Any,         # dict {link_name: (verts, faces)} or None — real STL meshes
    instruction: str | None,
    camera_intrinsics: Any | None = None,  # dict {cam_name: (T,3,3) float32}
    camera_image_sizes: Any | None = None,  # dict {cam_name: (width, height)}
    fps: float = 30.0,
) -> None:
    """Write an animated USD stage of a dual-arm ALOHA-Agilex episode.

    When *mesh_data* is provided (real STL meshes for the ARX5 links and
    body), the robot is rendered with its actual geometry — no primitive
    proxies. The arm links are animated via FK on *arm_joint_angles*; the
    grippers (link7/link8) are animated via the gripper aperture in
    *gripper_states*; the body / camera tower are placed statically per the
    URDF chain.

    When *mesh_data* is empty, falls back to cylinder-skeleton + EE-sphere
    visualisation (used by the test fixture, which has no robot meshes).

    EE quaternion convention (SAPIEN):
        ee_poses[t, arm, 3] = qw, [4]=qx, [5]=qy, [6]=qz
    Gripper convention: 0 = open (fingers spread), 1 = closed (fingers shut).
    """
    if not _USD_OK:
        raise RuntimeError(
            "usd-core is required for USDC export. "
            "Install with: pip install usd-core"
        )

    import numpy as np

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
        camera_poses=camera_poses,
        camera_intrinsics=camera_intrinsics,
        camera_image_sizes=camera_image_sizes,
        frame_count=T,
    )

    # ── Floor plane ──────────────────────────────────────────────────────
    floor_pts: list[tuple[float, float, float]] = [
        (-2.0, -2.0, 0.0), (2.0, -2.0, 0.0),
        (2.0,  2.0, 0.0), (-2.0,  2.0, 0.0),
    ]
    floor_mesh = UsdGeom.Mesh.Define(stage, "/Scene/Floor")
    floor_mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(*p) for p in floor_pts])
    )
    floor_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    floor_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    floor_mesh.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.55)])
    )

    # ── Back wall removed —————————————————————————————————————————————
    # The SAPIEN scene has a 6 × 1.2 × 3 m physics-boundary "wall" at
    # (0, 1, 1.5) but it's off-frame in every rendered camera and only
    # clutters the USD view, so we don't re-create it here.

    # ── Table (authoritative dimensions from envs/_base_task.py) ───────
    # create_table(pose(0,0,0.74), length=1.2, width=0.7, height=0.74,
    #              thickness=0.05) — tabletop + 4 legs
    _build_table(
        stage,
        pose_xyz=(0.0, 0.0, 0.74),
        length=1.2, width=0.7, height=0.74, thickness=0.05,
    )

    # ── Robot ────────────────────────────────────────────────────────────
    UsdGeom.Xform.Define(stage, "/Scene/Robot")
    use_real_mesh = bool(mesh_data) and arm_joint_angles is not None
    poses_arr = np.asarray(ee_poses, dtype=np.float64) if ee_poses is not None else None

    if use_real_mesh:
        _render_robot_meshes(
            stage,
            T=T,
            mesh_data=mesh_data,
            arm_joint_angles=arm_joint_angles,
            gripper_states=gripper_states,
            ee_poses=poses_arr,
        )
    else:
        _render_robot_primitives(
            stage,
            T=T,
            arm_joint_angles=arm_joint_angles,
            ee_poses=poses_arr,
            gripper_states=gripper_states,
        )

    # ── Manipulated objects (block + target pad) ─────────────────────────
    # RoboTwin 2.0 doesn't persist object state; reconstruct from gripper + EE.
    block_pose, target_pad_pose = _estimate_object_trajectory(
        T, poses_arr, gripper_states
    )
    if block_pose is not None:
        UsdGeom.Xform.Define(stage, "/Scene/Objects")

        # Red block — create_box in handover_block.py uses half_size=(0.03,0.03,0.1)
        block_verts, block_faces = _make_box_mesh((0.06, 0.06, 0.20))
        times_T_block: dict[int, Any] = {}
        for t in range(T):
            pos = block_pose[t, :3]
            qw, qx, qy, qz = block_pose[t, 3:7]
            M = np.eye(4)
            M[:3, :3] = _quat_to_rot(
                float(qw), float(qx), float(qy), float(qz)
            )
            M[:3, 3] = pos
            times_T_block[t] = M
        _add_animated_mesh(
            stage, "/Scene/Objects/Block",
            block_verts, block_faces,
            Gf.Vec3f(0.88, 0.12, 0.12),          # red
            times_T_block,
        )

        # Blue target pad — flat pad on the table at the final block XY.
        # Task default: half_size=(0.05, 0.05, 0.005); top surface at z=0.76.
        if target_pad_pose is not None:
            pad_verts, pad_faces = _make_box_mesh((0.10, 0.10, 0.01))
            pad_T = np.eye(4)
            pad_T[0, 3] = float(target_pad_pose[0])
            pad_T[1, 3] = float(target_pad_pose[1])
            pad_T[2, 3] = 0.76 + 0.005     # sit on the table
            _add_static_mesh(
                stage, "/Scene/Objects/TargetPad",
                pad_verts, pad_faces,
                Gf.Vec3f(0.12, 0.25, 0.95),      # blue
                pad_T,
            )

    # ── Joint state as custom time-sampled attribute on /Scene ───────────
    scene_prim = root_xf.GetPrim()
    if joint_states is not None:
        js = np.asarray(joint_states, dtype=np.float32)
        joint_attr = scene_prim.CreateAttribute(
            "robotwin:jointState", Sdf.ValueTypeNames.FloatArray,
        )
        for t in range(T):
            joint_attr.Set(Vt.FloatArray(js[t].tolist()), Usd.TimeCode(t))

    # ── String metadata ───────────────────────────────────────────────────
    if instruction:
        scene_prim.CreateAttribute(
            "robotwin:instruction", Sdf.ValueTypeNames.String
        ).Set(instruction)
    scene_prim.CreateAttribute(
        "robotwin:jointDim", Sdf.ValueTypeNames.Int
    ).Set(_ARM_DOFS)

    stage.GetRootLayer().Save()


# ---------------------------------------------------------------------------
# Robot rendering — real-mesh path (uses URDF + STL geometry)
# ---------------------------------------------------------------------------


def _render_robot_meshes(
    stage: Any,
    *,
    T: int,
    mesh_data: dict,
    arm_joint_angles: dict,
    gripper_states: Any,
    ee_poses: Any,
) -> None:
    """Render the ALOHA-Agilex robot using real STL meshes + FK transforms."""
    import numpy as np

    robot_T = _robot_world_T()

    # ── Static body parts (parented through fixed joints) ────────────────
    # box_joint:  footprint → base_link  (xyz=0,0,0.15)
    # box1:       footprint → box1_Link  (xyz=0,0,0.15)
    # box2:       box1_Link → box2_Link  (xyz=0.158, -0.385, -0.135)
    # camera_to_box1: box1_Link → camera_base_link (xyz=0.18, 0, 0.626)
    base_link_T = robot_T @ _make_T((0.0, 0.0, 0.15))
    box1_T      = robot_T @ _make_T((0.0, 0.0, 0.15))
    box2_T      = box1_T  @ _make_T((0.158, -0.385, -0.135))
    cam_base_T  = box1_T  @ _make_T((0.18,   0.0,    0.626))
    cam1_T      = cam_base_T @ _make_T((0.071198, 0.0, 0.10384), (0.0, 0.26931, 0.0))
    cam2_T      = cam1_T     @ _make_T((-0.14,    0.0, 0.5),     (0.0, 0.011358, 0.0))

    body_color   = Gf.Vec3f(0.20, 0.22, 0.26)   # dark grey
    panel_color  = Gf.Vec3f(0.85, 0.86, 0.88)   # light grey
    metal_color  = Gf.Vec3f(0.35, 0.36, 0.40)   # medium grey
    arm_color    = Gf.Vec3f(0.90, 0.91, 0.93)   # off-white
    finger_color = Gf.Vec3f(0.20, 0.20, 0.22)

    UsdGeom.Xform.Define(stage, "/Scene/Robot/Body")
    if "tracer_base_link" in mesh_data:
        v, f = mesh_data["tracer_base_link"]
        _add_static_mesh(stage, "/Scene/Robot/Body/Tracer", v, f, body_color, base_link_T)
    if "box1_Link" in mesh_data:
        v, f = mesh_data["box1_Link"]
        _add_static_mesh(stage, "/Scene/Robot/Body/Box1", v, f, panel_color, box1_T)
    # NOTE: box2_Link is intentionally skipped. The URDF places it at
    # (0.158, -0.385, -0.135) relative to box1, which after the robot's
    # 90° world-Z rotation lands as a 0.7×0.7×0.76 m asymmetric compartment
    # in the world +X region — it visually reads as "the robot body is
    # skewed to one side" even though the placement is URDF-faithful. Until
    # we do full SAPIEN replay (which gives authoritative actor poses), we
    # add a symmetric shoulder pillar below instead so the arms don't look
    # detached from the mobile base.

    # Symmetric shoulder pillar (robot-local frame): connects box1 top at
    # z≈0.22 to the arm-base plane at z=0.78, spanning the full left/right
    # arm separation. Drawn in robot-local frame and transformed by robot_T.
    import numpy as np
    pillar_verts, pillar_faces = _make_box_mesh((0.28, 0.68, 0.58))
    pillar_T_local = np.eye(4)
    pillar_T_local[:3, 3] = (0.18, 0.0, 0.22 + 0.29)   # centred x=0.18, z=0.51
    pillar_T = robot_T @ pillar_T_local
    _add_static_mesh(
        stage, "/Scene/Robot/Body/ShoulderPillar",
        pillar_verts, pillar_faces, panel_color, pillar_T,
    )

    UsdGeom.Xform.Define(stage, "/Scene/Robot/Head")
    if "camera_base_link" in mesh_data:
        v, f = mesh_data["camera_base_link"]
        _add_static_mesh(stage, "/Scene/Robot/Head/Pole", v, f, metal_color, cam_base_T)
    if "camera_link1" in mesh_data:
        v, f = mesh_data["camera_link1"]
        _add_static_mesh(stage, "/Scene/Robot/Head/Mount", v, f, metal_color, cam1_T)
    if "camera_link2" in mesh_data:
        v, f = mesh_data["camera_link2"]
        _add_static_mesh(stage, "/Scene/Robot/Head/Camera", v, f, body_color, cam2_T)

    # ── Per-arm articulated meshes ───────────────────────────────────────
    _ARMS = (("LeftArm", "left", 0), ("RightArm", "right", 1))
    GRIPPER_OPEN_M = 0.045   # prismatic position when open
    GRIPPER_CLOSED_M = 0.0   # prismatic position when closed

    for arm_name, side, arm_idx in _ARMS:
        if side not in arm_joint_angles:
            continue
        UsdGeom.Xform.Define(stage, f"/Scene/Robot/{arm_name}")

        base_T = _arm_world_base_T(side)
        angles = np.asarray(arm_joint_angles[side], dtype=np.float64)
        grippers = (
            np.asarray(gripper_states[side], dtype=np.float64)
            if gripper_states and side in gripper_states else None
        )

        # Pre-compute FK link transforms for every frame
        link_world_T: dict[int, list[Any]] = {}
        for t in range(T):
            link_world_T[t] = _forward_kinematics(base_T, angles[t])
            # _forward_kinematics returns 7 transforms: base + 6 joints

        # Static base mount
        if "base_arm" in mesh_data:
            v, f = mesh_data["base_arm"]
            _add_static_mesh(
                stage, f"/Scene/Robot/{arm_name}/BaseMount",
                v, f, metal_color, base_T,
            )

        # Animated link meshes (link1 .. link6)
        for li in range(1, 7):
            key = f"link{li}"
            if key not in mesh_data:
                continue
            v, f = mesh_data[key]
            times_T = {t: link_world_T[t][li] for t in range(T)}
            _add_animated_mesh(
                stage, f"/Scene/Robot/{arm_name}/Link{li}",
                v, f, arm_color, times_T,
            )

        # Gripper fingers (prismatic mimic joints attached to link6)
        # joint7: xyz=(0.08457,  0.024493, -0.00010349), axis=(0,  1, 0)
        # joint8: xyz=(0.08457, -0.024496, -0.00010354), axis=(0, -1, 0)
        # Gripper convention: 0.0 = CLOSED (fingers together),
        #                      1.0 = OPEN   (fingers fully spread)
        if "link7" in mesh_data and "link8" in mesh_data:
            v7, f7 = mesh_data["link7"]
            v8, f8 = mesh_data["link8"]
            j7_origin_T = _make_T((0.08457,  0.024493, -0.00010349))
            j8_origin_T = _make_T((0.08457, -0.024496, -0.00010354))

            times_T7: dict[int, Any] = {}
            times_T8: dict[int, Any] = {}
            for t in range(T):
                ee_T = link_world_T[t][6]
                if grippers is not None:
                    g = float(np.clip(grippers[t], 0.0, 1.0))
                    # g = 0 → closed (fingers meet, pos = 0)
                    # g = 1 → open   (fingers apart, pos = GRIPPER_OPEN_M)
                    finger_pos = g * GRIPPER_OPEN_M
                else:
                    finger_pos = GRIPPER_OPEN_M
                # joint7 axis = +Y, so translation along +Y
                j7_T = ee_T @ j7_origin_T @ _make_T((0.0,  finger_pos, 0.0))
                # joint8 axis = -Y, so translation along -Y
                j8_T = ee_T @ j8_origin_T @ _make_T((0.0, -finger_pos, 0.0))
                times_T7[t] = j7_T
                times_T8[t] = j8_T

            _add_animated_mesh(
                stage, f"/Scene/Robot/{arm_name}/FingerL",
                v7, f7, finger_color, times_T7,
            )
            _add_animated_mesh(
                stage, f"/Scene/Robot/{arm_name}/FingerR",
                v8, f8, finger_color, times_T8,
            )

        # Small EE pose marker (gripper state → colour: closed = red, open = green)
        if ee_poses is not None:
            indicator = UsdGeom.Sphere.Define(
                stage, f"/Scene/Robot/{arm_name}/GraspIndicator"
            )
            indicator.GetRadiusAttr().Set(0.012)
            ind_xf = UsdGeom.Xformable(indicator.GetPrim())
            tr_op = ind_xf.AddTranslateOp()
            color_attr = indicator.GetDisplayColorAttr()
            for t in range(T):
                p = ee_poses[t, arm_idx, :3]
                tr_op.Set(
                    Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
                    Usd.TimeCode(t),
                )
                if grippers is not None:
                    g = float(np.clip(grippers[t], 0.0, 1.0))
                    # g=0 closed → red; g=1 open → green
                    color_attr.Set(
                        Vt.Vec3fArray([Gf.Vec3f(1.0 - g, g, 0.05)]),
                        Usd.TimeCode(t),
                    )
                else:
                    color_attr.Set(
                        Vt.Vec3fArray([Gf.Vec3f(0.9, 0.4, 0.1)]),
                        Usd.TimeCode(t),
                    )


# ---------------------------------------------------------------------------
# Robot rendering — primitive fallback (used when mesh data is unavailable)
# ---------------------------------------------------------------------------


def _render_robot_primitives(
    stage: Any,
    *,
    T: int,
    arm_joint_angles: Any,
    ee_poses: Any,
    gripper_states: Any,
) -> None:
    """Cylinder + sphere stick-figure used when mesh assets aren't available."""
    import numpy as np

    # Body box (mobile base + torso) placed at robot world pose.
    body_xform = UsdGeom.Xform.Define(stage, "/Scene/Robot/Base")
    body_prim_xf = UsdGeom.Xformable(body_xform.GetPrim())
    body_prim_xf.AddTranslateOp().Set(Gf.Vec3d(*_ROBOT_WORLD_POS))
    body_prim_xf.AddOrientOp().Set(Gf.Quatf(*_ROBOT_WORLD_QUAT))

    bx, by, bz = _BASE_BOX_HALF_EXT
    cx, cy, cz = _BASE_BOX_CENTRE
    body_pts = [
        (cx - bx, cy - by, cz - bz), (cx + bx, cy - by, cz - bz),
        (cx + bx, cy + by, cz - bz), (cx - bx, cy + by, cz - bz),
        (cx - bx, cy - by, cz + bz), (cx + bx, cy - by, cz + bz),
        (cx + bx, cy + by, cz + bz), (cx - bx, cy + by, cz + bz),
    ]
    body_faces = [
        4, 4, 5, 6, 7,
        4, 0, 1, 5, 4,   4, 1, 2, 6, 5,
        4, 2, 3, 7, 6,   4, 3, 0, 4, 7,
        4, 0, 3, 2, 1,
    ]
    body_mesh = UsdGeom.Mesh.Define(stage, "/Scene/Robot/Base/Body")
    body_mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(*p) for p in body_pts])
    )
    counts2, indices2 = [], []
    i = 0
    while i < len(body_faces):
        n = body_faces[i]; i += 1
        counts2.append(n)
        indices2.extend(body_faces[i:i + n]); i += n
    body_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts2))
    body_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices2))
    body_mesh.GetDisplayColorAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(0.25, 0.27, 0.32)])
    )

    poses_arr = ee_poses

    _ARMS = {
        "LeftArm":  {"side": "left",  "arm_idx": 0,
                     "link_color": Gf.Vec3f(0.60, 0.62, 0.68),
                     "joint_color": Gf.Vec3f(0.20, 0.45, 0.85)},
        "RightArm": {"side": "right", "arm_idx": 1,
                     "link_color": Gf.Vec3f(0.60, 0.62, 0.68),
                     "joint_color": Gf.Vec3f(0.85, 0.42, 0.18)},
    }

    for arm_name, cfg in _ARMS.items():
        side = cfg["side"]
        arm_idx = cfg["arm_idx"]
        link_color = cfg["link_color"]
        joint_color = cfg["joint_color"]
        UsdGeom.Xform.Define(stage, f"/Scene/Robot/{arm_name}")

        grippers = None
        if gripper_states and side in gripper_states:
            grippers = np.asarray(gripper_states[side], dtype=np.float64)

        have_fk = (
            arm_joint_angles is not None
            and side in arm_joint_angles
            and arm_joint_angles[side].shape[1] == len(_ARX5_FRONT_CHAIN)
        )

        joint_world_xyz = None
        if have_fk:
            base_T = _arm_world_base_T(side)
            angles_arr = np.asarray(arm_joint_angles[side], dtype=np.float64)
            joint_world_xyz = np.zeros((T, len(_ARX5_FRONT_CHAIN) + 1, 3))
            for t in range(T):
                T_chain = _forward_kinematics(base_T, angles_arr[t])
                for k, Tk in enumerate(T_chain):
                    joint_world_xyz[t, k, :] = Tk[:3, 3]

        if joint_world_xyz is not None:
            for k in range(joint_world_xyz.shape[1]):
                jm = UsdGeom.Sphere.Define(
                    stage, f"/Scene/Robot/{arm_name}/Joint{k}"
                )
                jm.GetRadiusAttr().Set(0.02)
                jm.GetDisplayColorAttr().Set(
                    Vt.Vec3fArray([joint_color * 0.7])
                )
                jm_xf = UsdGeom.Xformable(jm.GetPrim()).AddTranslateOp()
                for t in range(T):
                    p = joint_world_xyz[t, k]
                    jm_xf.Set(
                        Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
                        Usd.TimeCode(t),
                    )

            n_links = joint_world_xyz.shape[1] - 1
            for li in range(n_links):
                cyl = UsdGeom.Cylinder.Define(
                    stage, f"/Scene/Robot/{arm_name}/Link{li + 1}"
                )
                cyl.GetRadiusAttr().Set(0.022)
                cyl.GetHeightAttr().Set(1.0)
                cyl.GetAxisAttr().Set(UsdGeom.Tokens.z)
                cyl.GetDisplayColorAttr().Set(Vt.Vec3fArray([link_color]))
                cyl_xf = UsdGeom.Xformable(cyl.GetPrim())
                tr_op = cyl_xf.AddTranslateOp()
                or_op = cyl_xf.AddOrientOp()
                sc_op = cyl_xf.AddScaleOp()
                for t in range(T):
                    p_a = joint_world_xyz[t, li]
                    p_b = joint_world_xyz[t, li + 1]
                    mid = (p_a + p_b) * 0.5
                    direction = p_b - p_a
                    length = float(np.linalg.norm(direction))
                    tr_op.Set(
                        Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])),
                        Usd.TimeCode(t),
                    )
                    or_op.Set(_z_to_vec_quat(direction), Usd.TimeCode(t))
                    sc_op.Set(Gf.Vec3f(1.0, 1.0, max(length, 1e-4)),
                              Usd.TimeCode(t))

        ee_sph = UsdGeom.Sphere.Define(stage, f"/Scene/Robot/{arm_name}/EE")
        ee_sph.GetRadiusAttr().Set(0.045)
        ee_xf = UsdGeom.Xformable(ee_sph.GetPrim())
        tr_op = ee_xf.AddTranslateOp()
        or_op = ee_xf.AddOrientOp()
        color_attr = ee_sph.GetDisplayColorAttr()

        if poses_arr is not None:
            for t in range(T):
                x = float(poses_arr[t, arm_idx, 0])
                y = float(poses_arr[t, arm_idx, 1])
                z = float(poses_arr[t, arm_idx, 2])
                tr_op.Set(Gf.Vec3d(x, y, z), Usd.TimeCode(t))
                qw = float(poses_arr[t, arm_idx, 3])
                qx = float(poses_arr[t, arm_idx, 4])
                qy = float(poses_arr[t, arm_idx, 5])
                qz = float(poses_arr[t, arm_idx, 6])
                or_op.Set(Gf.Quatf(qw, qx, qy, qz), Usd.TimeCode(t))
                if grippers is not None:
                    g = float(np.clip(grippers[t], 0.0, 1.0))
                    # g=0 closed → red; g=1 open → green
                    ee_color = Gf.Vec3f(1.0 - g, g, 0.05)
                else:
                    ee_color = joint_color
                color_attr.Set(Vt.Vec3fArray([ee_color]), Usd.TimeCode(t))
        elif joint_world_xyz is not None:
            for t in range(T):
                p = joint_world_xyz[t, -1]
                tr_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])),
                          Usd.TimeCode(t))
            color_attr.Set(Vt.Vec3fArray([joint_color]))
        else:
            try:
                base_T = _arm_world_base_T(side)
                p = base_T[:3, 3]
                tr_op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            except Exception:
                tr_op.Set(Gf.Vec3d(0.0, 0.0, 0.9))
            color_attr.Set(Vt.Vec3fArray([joint_color]))
