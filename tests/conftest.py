"""Shared test fixtures and configuration."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory structure."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    for d in ["raw", "staging", "canonical", "exports", "catalog", "logs", "cache", "projects"]:
        (ws / d).mkdir()
    return ws


@pytest.fixture
def workspace_config(tmp_workspace):
    """Create a WorkspaceConfig pointing to temp directories."""
    from guanwu.core.config import WorkspaceConfig, StorageConfig

    return WorkspaceConfig(
        workspace_root=str(tmp_workspace),
        storage=StorageConfig(
            raw_root=str(tmp_workspace / "raw"),
            staging_root=str(tmp_workspace / "staging"),
            canonical_root=str(tmp_workspace / "canonical"),
            export_root=str(tmp_workspace / "exports"),
            catalog_path=str(tmp_workspace / "catalog" / "catalog.duckdb"),
            project_root=str(tmp_workspace / "projects"),
        ),
    )


@pytest.fixture
def job_context(tmp_workspace):
    """Create a JobContext for testing."""
    from guanwu.schemas.bundles import JobContext

    return JobContext(
        job_id="test_job_001",
        workspace_root=str(tmp_workspace),
        raw_root=str(tmp_workspace / "raw"),
        staging_root=str(tmp_workspace / "staging"),
        canonical_root=str(tmp_workspace / "canonical"),
        dry_run=False,
        resume=False,
        workers=1,
    )


# ── Synthetic fixture helpers ──────────────────────────────────────────

@pytest.fixture
def scannetpp_fixture(tmp_path):
    """Create a minimal ScanNet++ fixture directory."""
    root = tmp_path / "scannetpp"
    data = root / "data"

    # Create a scene
    scene = data / "scene0001_00" / "scans"
    scene.mkdir(parents=True)
    # Minimal PLY file (header only, no real geometry)
    (scene / "mesh_aligned_0.05.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"
    )

    # Camera poses
    dslr = data / "scene0001_00" / "dslr" / "nerfstudio"
    dslr.mkdir(parents=True)
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": 1000.0,
        "fl_y": 1000.0,
        "cx": 500.0,
        "cy": 375.0,
        "w": 1000,
        "h": 750,
        "frames": [
            {
                "file_path": "frame_00001.JPG",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
        ],
    }
    (dslr / "transforms.json").write_text(json.dumps(transforms))

    # Images dir
    imgs = data / "scene0001_00" / "dslr" / "resized_images"
    imgs.mkdir(parents=True)
    (imgs / "frame_00001.JPG").write_bytes(b"\xff\xd8\xff")

    # Splits
    splits = root / "splits"
    splits.mkdir(parents=True)
    (splits / "nvs_sem_train.txt").write_text("scene0001_00\n")
    (splits / "nvs_sem_val.txt").write_text("")

    return root


@pytest.fixture
def objaverse_fixture(tmp_path):
    """Create a minimal Objaverse-XL fixture directory."""
    root = tmp_path / "objaverse"
    glbs = root / "hf-objaverse-v1" / "glbs" / "000-000"
    glbs.mkdir(parents=True)

    # Minimal GLB file (just header bytes)
    glb_header = b"glTF\x02\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00"
    (glbs / "abc123def456.glb").write_bytes(glb_header + b"\x00" * 12)

    # Object IDs file
    (root / "object_ids.txt").write_text("abc123def456\n")

    return root


@pytest.fixture
def partnet_fixture(tmp_path):
    """Create a minimal PartNet-Mobility fixture directory."""
    root = tmp_path / "partnet"

    obj_dir = root / "7236"
    obj_dir.mkdir(parents=True)

    # meta.json
    (obj_dir / "meta.json").write_text(json.dumps({
        "model_cat": "Table",
        "anno_id": "7236",
    }))

    # Minimal URDF
    urdf_content = """<?xml version="1.0" ?>
<robot name="table">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="textured_objs/base.obj"/>
      </geometry>
    </visual>
  </link>
  <link name="drawer">
    <visual>
      <geometry>
        <mesh filename="textured_objs/drawer.obj"/>
      </geometry>
    </visual>
  </link>
  <joint name="drawer_joint" type="prismatic">
    <parent link="base"/>
    <child link="drawer"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.5" effort="100" velocity="1"/>
  </joint>
</robot>"""
    (obj_dir / "mobility.urdf").write_text(urdf_content)

    # Mesh files
    meshes = obj_dir / "textured_objs"
    meshes.mkdir()
    for name in ["base.obj", "drawer.obj"]:
        (meshes / name).write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        )

    return root


@pytest.fixture
def arkitscenes_fixture(tmp_path):
    """Create a minimal ARKitScenes fixture directory."""
    root = tmp_path / "arkitscenes"

    scene = root / "3dod" / "Training" / "47331606"
    scene.mkdir(parents=True)

    # Annotation
    ann = {
        "data": [
            {
                "uid": "obj001",
                "label": "chair",
                "dimensions": {"length": 0.5, "width": 0.5, "height": 0.8},
                "position": {"x": 1.0, "y": 0.0, "z": 0.4},
                "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            }
        ]
    }
    (scene / "47331606_3dod_annotation.json").write_text(json.dumps(ann))

    # Depth dir
    (scene / "lowres_depth").mkdir()
    # Intrinsics
    (scene / "lowres_wide_intrinsics").mkdir()

    return root


@pytest.fixture
def maniskill_fixture(tmp_path):
    """Create a minimal ManiSkill 3 fixture directory."""
    root = tmp_path / "maniskill"
    demos = root / "demos" / "PickCube-v1"
    demos.mkdir(parents=True)

    # Create a minimal HDF5 file if h5py is available
    try:
        import h5py
        import numpy as np

        h5_path = demos / "trajectory.h5"
        with h5py.File(h5_path, "w") as f:
            # env_states group with transforms
            env_states = f.create_group("traj_0")
            env_states.create_dataset("actions", data=np.zeros((10, 7)))
            env_states.create_dataset("success", data=np.array([True] * 10))
    except ImportError:
        # Create a placeholder
        (demos / "trajectory.h5").write_bytes(b"")

    # Assets dir
    assets = root / "assets" / "cube"
    assets.mkdir(parents=True)
    (assets / "model.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    return root


@pytest.fixture
def robotwin2_fixture(tmp_path):
    """Create a minimal RoboTwin 2.0 fixture directory.

    Mimics the on-disk layout:
        <root>/
          <task_name>/
            <task_name>_clean_50/
              data/
                episode0.hdf5
                episode1.hdf5
              instructions/
                episode0.json
                episode1.json
            <task_name>_randomized_50/
              data/
                episode0.hdf5
              instructions/
                episode0.json
    """
    root = tmp_path / "robotwin2"

    # Two tasks, one clean variant + one randomized variant each
    tasks = [
        ("block_handover", "block_handover_clean_50", True),
        ("block_handover", "block_handover_randomized_50", False),
        ("pour_water", "pour_water_clean_50", True),
    ]

    T = 20      # timesteps per episode
    H, W = 64, 64  # image height/width (small for tests)
    N_EPISODES = 2   # episodes per variant

    for task_name, variant_name, is_clean in tasks:
        variant_dir = root / task_name / variant_name
        data_dir = variant_dir / "data"
        instr_dir = variant_dir / "instructions"
        data_dir.mkdir(parents=True)
        instr_dir.mkdir(parents=True)

        for ep_idx in range(N_EPISODES):
            ep_stem = f"episode{ep_idx}"

            # ── Instruction JSON ──────────────────────────────────────
            instruction_text = (
                f"Pick up the block and hand it over"
                if task_name == "block_handover"
                else f"Pour water from the cup into the bowl"
            )
            (instr_dir / f"{ep_stem}.json").write_text(
                json.dumps({"instruction": instruction_text})
            )

            # ── HDF5 episode ─────────────────────────────────────────
            try:
                import h5py
                import numpy as np

                h5_path = data_dir / f"{ep_stem}.hdf5"
                with h5py.File(str(h5_path), "w") as f:
                    obs = f.create_group("observation")

                    # Three cameras: head, left wrist, right wrist
                    for cam_name in ("head_camera", "left_camera", "right_camera"):
                        cam = obs.create_group(cam_name)
                        # RGB: (T, H, W, 3) uint8
                        rgb = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
                        cam.create_dataset("rgb", data=rgb, compression="gzip")
                        # Depth: (T, H, W) float32, values in metres
                        depth = np.random.uniform(0.3, 1.5, (T, H, W)).astype(np.float32)
                        cam.create_dataset("depth", data=depth, compression="gzip")

                    # Joint state: (T, 14) — 7 DOF per arm
                    joint_state = np.random.uniform(-3.14, 3.14, (T, 14)).astype(np.float32)
                    obs.create_dataset("joint_state", data=joint_state)

                    # End-effector pose: (T, 2, 7) — left/right × (xyz + quat)
                    ee_pose = np.random.randn(T, 2, 7).astype(np.float32)
                    obs.create_dataset("end_effector", data=ee_pose)

                    # Action: (T, 14) — target joint positions
                    action = np.random.uniform(-3.14, 3.14, (T, 14)).astype(np.float32)
                    f.create_dataset("action", data=action)

                    # Success flag
                    f.create_dataset("success", data=np.bool_(ep_idx == 0))

            except ImportError:
                # h5py not available — write a stub file so inventory still runs
                (data_dir / f"{ep_stem}.hdf5").write_bytes(b"")

    return root
