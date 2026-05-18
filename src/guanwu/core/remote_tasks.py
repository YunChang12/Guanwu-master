"""Remote GPU tasks for simulation data processing.

Concrete tasks that use ``RemoteExecutor`` to run on a GPU machine:
- Trajectory replay: extract real joint states / link poses from ManiSkill
- RGB-D rendering: render frames from simulation trajectories
"""
from __future__ import annotations

import json
import logging
import shutil
import tarfile
import textwrap
from pathlib import Path

from guanwu.core.remote import RemoteExecutor

logger = logging.getLogger("guanwu")


def _extract_tarball(tar_path: Path, output_dir: Path) -> None:
    """Extract a tarball using safe defaults when supported by Python."""
    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            tar.extractall(output_dir, filter="data")
        except TypeError:  # pragma: no cover - older Python without tar filter support
            tar.extractall(output_dir)


def replay_trajectory(
    executor: RemoteExecutor,
    h5_path: str | Path,
    env_id: str,
    control_mode: str = "pd_joint_pos",
    traj_keys: list[str] | None = None,
    output_dir: str | Path = ".",
    max_trajs: int | None = None,
) -> dict:
    """Replay a ManiSkill trajectory on the remote GPU to extract real link poses.

    1. Uploads the H5 file to the remote machine
    2. Runs a replay script inside ManiSkill/SAPIEN
    3. Downloads the resulting npz files (link poses + qpos per frame)

    Args:
        executor: configured RemoteExecutor
        h5_path: local path to trajectory .h5 file
        env_id: ManiSkill env ID, e.g. "PickCube-v1"
        control_mode: control mode used in the demo
        traj_keys: specific trajectory keys to replay (e.g. ["traj_0"]);
                   None = replay all
        output_dir: local directory to save results
        max_trajs: max number of trajectories to replay

    Returns:
        dict with keys: traj_keys, output_files, num_frames_per_traj
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Upload H5
    remote_h5 = executor.upload(h5_path)

    # Also upload trajectory.json if it exists alongside
    json_path = h5_path.parent / (h5_path.stem + ".json")
    if json_path.exists():
        executor.upload(json_path)

    # Ensure mani_skill is installed
    executor.ensure_package("mani_skill")

    # Build replay script
    traj_filter = json.dumps(traj_keys) if traj_keys else "None"
    max_trajs_val = max_trajs if max_trajs is not None else "None"

    script = textwrap.dedent(f"""\
        import numpy as np
        import h5py
        import json
        import gymnasium as gym
        import mani_skill.envs

        h5_path = "{remote_h5}"
        env_id = "{env_id}"
        control_mode = "{control_mode}"
        traj_filter = {traj_filter}
        max_trajs = {max_trajs_val}
        work_dir = "{executor.work_dir}"

        env = gym.make(env_id, obs_mode="none", control_mode=control_mode, render_mode=None)

        f = h5py.File(h5_path, "r")

        # Determine which trajectories to replay
        all_keys = sorted([k for k in f.keys() if k.startswith("traj")],
                          key=lambda x: int(x.split("_")[1]))
        if traj_filter is not None:
            all_keys = [k for k in all_keys if k in traj_filter]
        if max_trajs is not None:
            all_keys = all_keys[:max_trajs]

        results = {{}}
        robot = env.unwrapped.agent.robot

        for traj_key in all_keys:
            traj = f[traj_key]
            actions = np.array(traj["actions"])
            num_steps = len(actions)

            obs, info = env.reset()

            all_qpos = []
            all_link_poses = {{}}

            for step in range(num_steps + 1):
                qpos = robot.get_qpos().cpu().numpy().flatten()
                all_qpos.append(qpos)

                for link in robot.get_links():
                    name = link.name
                    if name not in all_link_poses:
                        all_link_poses[name] = {{"pos": [], "quat": []}}
                    pose = link.pose
                    all_link_poses[name]["pos"].append(pose.p.cpu().numpy().flatten())
                    all_link_poses[name]["quat"].append(pose.q.cpu().numpy().flatten())

                if step < num_steps:
                    obs, _, _, _, _ = env.step(actions[step])

            # Save per-trajectory npz
            save_dict = {{"qpos": np.array(all_qpos)}}
            for ln, data in all_link_poses.items():
                save_dict[f"pos_{{ln}}"] = np.array(data["pos"])
                save_dict[f"quat_{{ln}}"] = np.array(data["quat"])

            out_path = f"{{work_dir}}/{{traj_key}}_replay.npz"
            np.savez_compressed(out_path, **save_dict)

            results[traj_key] = {{
                "num_frames": len(all_qpos),
                "qpos_dim": all_qpos[0].shape[0],
                "link_names": list(all_link_poses.keys()),
                "output": out_path,
            }}
            print(f"{{traj_key}}: {{len(all_qpos)}} frames, {{len(all_link_poses)}} links")

        f.close()
        env.close()

        # Write summary
        with open(f"{{work_dir}}/replay_summary.json", "w") as sf:
            json.dump(results, sf, indent=2)
        print("DONE")
    """)

    logger.info("Replaying %s on %s (env=%s)...", h5_path.name, executor.host, env_id)
    stdout = executor.run_script(script, timeout=600)
    logger.info("Remote replay output:\n%s", stdout.strip())

    # Download results
    output_files = {}
    try:
        summary_local = output_dir / "replay_summary.json"
        executor.download(f"{executor.work_dir}/replay_summary.json", summary_local)
        with open(summary_local) as f:
            summary = json.load(f)

        for traj_key, info in summary.items():
            remote_npz = info["output"]
            local_npz = output_dir / f"{traj_key}_replay.npz"
            executor.download(remote_npz, local_npz)
            output_files[traj_key] = str(local_npz)
            logger.info("Downloaded %s (%d frames)", local_npz.name, info["num_frames"])

    except Exception as e:
        logger.error("Failed to download replay results: %s", e)

    return {
        "traj_keys": list(output_files.keys()),
        "output_files": output_files,
        "stdout": stdout,
    }


def render_trajectory(
    executor: RemoteExecutor,
    h5_path: str | Path,
    env_id: str,
    control_mode: str = "pd_joint_pos",
    traj_keys: list[str] | None = None,
    output_dir: str | Path = ".",
    resolution: tuple[int, int] = (512, 512),
    max_trajs: int | None = None,
    frame_step: int = 1,
) -> dict:
    """Render RGB-D frames from a ManiSkill trajectory on the remote GPU.

    Args:
        executor: configured RemoteExecutor
        h5_path: local path to trajectory .h5 file
        env_id: ManiSkill env ID
        control_mode: control mode
        traj_keys: trajectories to render
        output_dir: local directory to save rendered images
        resolution: (width, height)
        max_trajs: max trajectories
        frame_step: render every N-th frame (1 = all frames)

    Returns:
        dict with rendered file info
    """
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    remote_h5 = executor.upload(h5_path)
    executor.ensure_package("mani_skill")

    traj_filter = json.dumps(traj_keys) if traj_keys else "None"
    max_trajs_val = max_trajs if max_trajs is not None else "None"
    w, h = resolution

    script = textwrap.dedent(f"""\
        import numpy as np
        import h5py
        import json
        import os
        import gymnasium as gym
        import mani_skill.envs
        from PIL import Image

        h5_path = "{remote_h5}"
        env_id = "{env_id}"
        control_mode = "{control_mode}"
        traj_filter = {traj_filter}
        max_trajs = {max_trajs_val}
        frame_step = {frame_step}
        work_dir = "{executor.work_dir}"
        W, H = {w}, {h}

        env = gym.make(env_id, obs_mode="rgbd", control_mode=control_mode,
                       render_mode="rgb_array",
                       sensor_configs=dict(width=W, height=H))

        f = h5py.File(h5_path, "r")
        all_keys = sorted([k for k in f.keys() if k.startswith("traj")],
                          key=lambda x: int(x.split("_")[1]))
        if traj_filter is not None:
            all_keys = [k for k in all_keys if k in traj_filter]
        if max_trajs is not None:
            all_keys = all_keys[:max_trajs]

        results = {{}}
        for traj_key in all_keys:
            traj = f[traj_key]
            actions = np.array(traj["actions"])
            num_steps = len(actions)

            obs, info = env.reset()
            out_dir = os.path.join(work_dir, "renders", traj_key)
            os.makedirs(out_dir, exist_ok=True)
            rendered = []

            for step in range(num_steps + 1):
                if step % frame_step == 0:
                    img = env.render()
                    if isinstance(img, np.ndarray) and img.size > 0:
                        fname = f"frame_{{step:05d}}.png"
                        Image.fromarray(img).save(os.path.join(out_dir, fname))
                        rendered.append(fname)
                if step < num_steps:
                    obs, _, _, _, _ = env.step(actions[step])

            results[traj_key] = {{
                "num_rendered": len(rendered),
                "render_dir": out_dir,
            }}
            print(f"{{traj_key}}: rendered {{len(rendered)}} frames")

        f.close()
        env.close()

        with open(os.path.join(work_dir, "render_summary.json"), "w") as sf:
            json.dump(results, sf, indent=2)

        # Pack renders into a tar
        import tarfile
        tar_path = os.path.join(work_dir, "renders.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(os.path.join(work_dir, "renders"), arcname="renders")
        print(f"Packed to {{tar_path}}")
        print("DONE")
    """)

    logger.info("Rendering %s on %s (env=%s, %dx%d)...",
                h5_path.name, executor.host, env_id, w, h)
    stdout = executor.run_script(script, timeout=1200)
    logger.info("Remote render output:\n%s", stdout.strip())

    # Download tar
    output_files = {}
    try:
        tar_local = output_dir / "renders.tar.gz"
        executor.download(f"{executor.work_dir}/renders.tar.gz", tar_local)

        _extract_tarball(tar_local, output_dir)

        logger.info("Extracted renders to %s", output_dir / "renders")
        output_files["renders_dir"] = str(output_dir / "renders")
    except Exception as e:
        logger.error("Failed to download renders: %s", e)

    return {
        "output_files": output_files,
        "stdout": stdout,
    }


def _robotwin_replay_module_path() -> Path:
    """Return the local RoboTwin replay module path to upload to the remote host."""
    return Path(__file__).resolve().parents[1] / "adapters" / "robotwin2_replay.py"


def _upload_robotwin_episode_inputs(
    executor: RemoteExecutor,
    *,
    h5_path: Path,
    task_name: str,
    variant_name: str,
    episode_stem: str,
) -> str:
    """Upload a minimal RoboTwin episode bundle preserving sibling layout."""
    remote_variant_dir = (
        Path(executor.work_dir)
        / "robotwin2_input"
        / task_name
        / variant_name
    )
    remote_h5 = str(remote_variant_dir / "data" / f"{episode_stem}{h5_path.suffix}")
    executor.upload(h5_path, remote_h5)

    traj_pkl = h5_path.parent.parent / "_traj_data" / f"{episode_stem}.pkl"
    if traj_pkl.exists():
        executor.upload(
            traj_pkl,
            str(remote_variant_dir / "_traj_data" / traj_pkl.name),
        )

    seed_file = h5_path.parent.parent / "seed.txt"
    if seed_file.exists():
        executor.upload(seed_file, str(remote_variant_dir / "seed.txt"))

    return remote_h5


def export_robotwin_episode(
    executor: RemoteExecutor,
    *,
    h5_path: str | Path,
    task_name: str,
    variant_name: str,
    episode_stem: str,
    seed: int,
    robotwin_root: str,
    output_dir: str | Path,
    source_relpath: str | None = None,
    remote_source_root: str | None = None,
    render_videos: bool = False,
    render_views: bool | None = None,
    view_specs: list[dict] | None = None,
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
) -> dict:
    """Replay one RoboTwin 2.0 episode on the remote GPU and download exports."""
    render_views_enabled = bool(render_views) if render_views is not None else bool(render_videos)
    if view_specs:
        render_views_enabled = True
    if render_videos and render_views is None:
        logger.info(
            "RoboTwin render_videos=True maps to generated-camera views; HDF5 RGB is not used"
        )

    h5_path = Path(h5_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    executor.ensure_package("numpy")
    executor.ensure_package("h5py")
    executor.ensure_package("imageio")
    executor.ensure_package("trimesh")
    executor.ensure_package("pycollada", import_name="collada")
    if render_views_enabled:
        executor.ensure_package("imageio-ffmpeg", import_name="imageio_ffmpeg")
    executor.ensure_package("usd-core", import_name="pxr")

    local_module = _robotwin_replay_module_path()
    remote_module = executor.upload(
        local_module,
        f"{executor.work_dir}/scripts/{local_module.name}",
    )

    remote_h5: str | None = None
    if remote_source_root and source_relpath:
        candidate = str(Path(remote_source_root) / source_relpath)
        if executor.path_exists(candidate):
            remote_h5 = candidate
            logger.info("Using mirrored RoboTwin episode on %s: %s", executor.host, candidate)

    if remote_h5 is None:
        remote_h5 = _upload_robotwin_episode_inputs(
            executor,
            h5_path=h5_path,
            task_name=task_name,
            variant_name=variant_name,
            episode_stem=episode_stem,
        )

    remote_output_dir = str(
        Path(executor.work_dir) / "robotwin2_out" / task_name / variant_name / episode_stem
    )
    remote_tar = str(
        Path(executor.work_dir) / "robotwin2_out" / f"{task_name}_{variant_name}_{episode_stem}.tar.gz"
    )
    remote_summary = str(
        Path(executor.work_dir) / "robotwin2_out" / f"{task_name}_{variant_name}_{episode_stem}.json"
    )

    script = textwrap.dedent(
        f"""\
        import importlib.util
        import json
        import shutil
        import tarfile
        from pathlib import Path

        module_path = Path({json.dumps(remote_module)})
        spec = importlib.util.spec_from_file_location("robotwin2_replay_remote", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load replay module: {{module_path}}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        h5_path = Path({json.dumps(remote_h5)})
        output_dir = Path({json.dumps(remote_output_dir)})
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        views_dir = output_dir / "views"
        if not {repr(bool(render_views_enabled))}:
            views_dir = None
        view_specs = json.loads({json.dumps(json.dumps(view_specs or []))})
        if view_specs and views_dir is None:
            views_dir = output_dir / "views"

        result = module.replay_episode(
            h5_path=h5_path,
            task_name={json.dumps(task_name)},
            seed={int(seed)},
            robotwin_root=Path({json.dumps(robotwin_root)}),
            out_usdc=output_dir / "scene.usdc",
            out_usdz=output_dir / "scene.usdz",
            keep_usdc_assets=False,
            renders_dir=None,
            views_dir=views_dir,
            view_specs=view_specs or None,
            generated_camera_motion={json.dumps(generated_camera_motion)},
            generated_static_camera_count={repr(generated_static_camera_count)},
            generated_trajectory_camera_count={repr(generated_trajectory_camera_count)},
            camera_trajectory_kind={json.dumps(camera_trajectory_kind)},
            trajectory_kind_mode={json.dumps(trajectory_kind_mode)},
            camera_seed={repr(camera_seed)},
            view_width_px={int(view_width_px)},
            view_height_px={int(view_height_px)},
            embodiment={json.dumps(embodiment)},
            fps={float(fps)},
            video_fps={int(video_fps)},
        )
        if result is None:
            raise RuntimeError("RoboTwin replay returned no result")

        files = []
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                files.append(str(file_path.relative_to(output_dir)))

        tar_path = Path({json.dumps(remote_tar)})
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "w:gz") as tar:
            for file_path in sorted(output_dir.rglob("*")):
                if file_path.is_file():
                    tar.add(file_path, arcname=str(file_path.relative_to(output_dir)))

        summary = {{
            "output_dir": str(output_dir),
            "tar_path": str(tar_path),
            "files_written": files,
            "num_frames": int(result["T"]),
            "num_actors": len(result["actor_states"]),
            "num_robot_links": len(result["robot_link_states"]),
            "num_robot_link_meshes": len(result["robot_link_meshes"]),
            "view_records": result.get("view_records", []),
        }}
        with open({json.dumps(remote_summary)}, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary))
        """
    )

    logger.info(
        "Replaying RoboTwin episode %s/%s/%s on %s...",
        task_name,
        variant_name,
        episode_stem,
        executor.host,
    )
    stdout = executor.run_script(script, timeout=3600)
    logger.info("Remote RoboTwin export output:\n%s", stdout.strip())

    summary_local = output_dir / "robotwin_export_summary.json"
    executor.download(remote_summary, summary_local)
    with open(summary_local) as f:
        summary = json.load(f)
    try:
        summary_local.unlink()
    except OSError:
        pass

    local_scene = output_dir / "scene.usdc"
    if local_scene.exists():
        local_scene.unlink()
    local_scene_usdz = output_dir / "scene.usdz"
    if local_scene_usdz.exists():
        local_scene_usdz.unlink()
    local_renders = output_dir / "renders"
    if local_renders.exists():
        shutil.rmtree(local_renders)
    local_views = output_dir / "views"
    if local_views.exists():
        shutil.rmtree(local_views)
    local_textures = output_dir / "textures"
    if local_textures.exists():
        shutil.rmtree(local_textures)

    tar_local = output_dir / "robotwin_export.tar.gz"
    executor.download(remote_tar, tar_local)
    _extract_tarball(tar_local, output_dir)

    try:
        tar_local.unlink()
    except OSError:
        pass

    return {
        "output_dir": str(output_dir),
        "files_written": summary.get("files_written", []),
        "stdout": stdout,
        "summary": summary,
    }
