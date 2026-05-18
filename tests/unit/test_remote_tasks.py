from __future__ import annotations

import json
import tarfile
from pathlib import Path

from guanwu.core.remote_tasks import export_robotwin_episode


class _FakeRemoteExecutor:
    def __init__(self, tmp_path: Path) -> None:
        self.host = "fake-remote"
        self.work_dir = "/remote/work"
        self._tmp_path = tmp_path
        self.last_script = ""
        self.packages: list[tuple[str, str | None]] = []
        self.uploads: list[tuple[str, str]] = []

    def ensure_package(self, package: str, import_name: str | None = None) -> None:
        self.packages.append((package, import_name))

    def upload(self, local_path: str | Path, remote_path: str | None = None) -> str:
        local_path = Path(local_path)
        remote_path = remote_path or f"{self.work_dir}/{local_path.name}"
        self.uploads.append((str(local_path), remote_path))
        return remote_path

    def path_exists(self, remote_path: str) -> bool:
        return remote_path == (
            "/remote/datasets/robotwin2/"
            "handover_block/aloha-agilex_clean_50/data/episode0.hdf5"
        )

    def run_script(self, script: str, timeout: int = 300) -> str:
        self.last_script = script
        return "fake remote replay ok"

    def download(self, remote_path: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if remote_path.endswith(".json"):
            summary = {
                "output_dir": f"{self.work_dir}/robotwin2_out/handover_block/aloha-agilex_clean_50/episode0",
                "tar_path": f"{self.work_dir}/robotwin2_out/handover_block_aloha-agilex_clean_50_episode0.tar.gz",
                "files_written": [
                    "scene.usdz",
                    "views/view_000/video.mp4",
                    "views/view_000/camera.json",
                    "views/view_000/render_meta.json",
                    "views/view_000/frame_mapping.json",
                ],
                "num_frames": 20,
                "num_actors": 3,
                "num_robot_links": 10,
                "view_records": [
                    {
                        "view_id": "view_000",
                        "video": "views/view_000/video.mp4",
                        "uses_hdf5_rgb": False,
                    }
                ],
            }
            local_path.write_text(json.dumps(summary))
            return local_path

        payload_root = self._tmp_path / "remote_payload"
        payload_root.mkdir(parents=True, exist_ok=True)
        (payload_root / "scene.usdz").write_bytes(b"fake-usdz")
        view_dir = payload_root / "views" / "view_000"
        view_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "video.mp4").write_bytes(b"fake-video")
        (view_dir / "camera.json").write_text(
            json.dumps({"view_id": "view_000", "uses_hdf5_rgb": False})
        )
        (view_dir / "render_meta.json").write_text(
            json.dumps({"frame_count": 20, "uses_hdf5_rgb": False})
        )
        (view_dir / "frame_mapping.json").write_text(json.dumps([]))

        with tarfile.open(local_path, "w:gz") as tar:
            tar.add(payload_root / "scene.usdz", arcname="scene.usdz")
            tar.add(view_dir / "video.mp4", arcname="views/view_000/video.mp4")
            tar.add(view_dir / "camera.json", arcname="views/view_000/camera.json")
            tar.add(view_dir / "render_meta.json", arcname="views/view_000/render_meta.json")
            tar.add(view_dir / "frame_mapping.json", arcname="views/view_000/frame_mapping.json")
        return local_path


def test_export_robotwin_episode_extracts_scene_and_generated_views(tmp_path):
    h5_path = tmp_path / "episode0.hdf5"
    h5_path.write_bytes(b"placeholder")

    executor = _FakeRemoteExecutor(tmp_path)
    output_dir = tmp_path / "out"
    stale = output_dir / "renders" / "stale.txt"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("old")
    stale_view = output_dir / "views" / "view_000" / "old.txt"
    stale_view.parent.mkdir(parents=True, exist_ok=True)
    stale_view.write_text("old-view")

    result = export_robotwin_episode(
        executor,
        h5_path=h5_path,
        task_name="handover_block",
        variant_name="aloha-agilex_clean_50",
        episode_stem="episode0",
        seed=0,
        robotwin_root="/remote/RoboTwin",
        output_dir=output_dir,
        source_relpath="handover_block/aloha-agilex_clean_50/data/episode0.hdf5",
        remote_source_root="/remote/datasets/robotwin2",
        render_videos=True,
    )

    assert "if not True:" in executor.last_script
    assert "if not true:" not in executor.last_script
    assert ("imageio-ffmpeg", "imageio_ffmpeg") in executor.packages
    assert "renders_dir=None" in executor.last_script
    assert "views_dir=views_dir" in executor.last_script
    assert "out_usdz=output_dir / \"scene.usdz\"" in executor.last_script
    assert "keep_usdc_assets=False" in executor.last_script
    assert (output_dir / "scene.usdz").is_file()
    assert not (output_dir / "scene.usdc").exists()
    assert not (output_dir / "textures").exists()
    assert (output_dir / "views" / "view_000" / "video.mp4").is_file()
    assert (output_dir / "views" / "view_000" / "camera.json").is_file()
    assert not (output_dir / "renders").exists()
    assert not stale.exists()
    assert not stale_view.exists()
    assert result["files_written"] == [
        "scene.usdz",
        "views/view_000/video.mp4",
        "views/view_000/camera.json",
        "views/view_000/render_meta.json",
        "views/view_000/frame_mapping.json",
    ]
