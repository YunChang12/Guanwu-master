from __future__ import annotations

from pathlib import Path

from guanwu.video.core.config import SPWMSettings, VLMConfig
from guanwu.video.project.config import (
    ProjectConfig,
    ProjectMetadata,
    load_project_config,
    project_config_payload,
    save_project_config,
)


def _system_settings(api_key: str) -> tuple[SPWMSettings, Path]:
    settings = SPWMSettings(
        vlm=VLMConfig(
            mode="embedded",
            backend="api",
            api_key=api_key,
            base_url="https://system.example/v1",
            model="system-model",
            max_retries=7,
            command_template="system-vlm",
        )
    )
    return settings, Path("/tmp/video.config.toml")


def test_project_config_payload_omits_system_managed_vlm_fields(monkeypatch) -> None:
    monkeypatch.setattr("guanwu.video.project.config.load_settings", lambda: _system_settings("system-key"))

    config = ProjectConfig(
        project=ProjectMetadata(
            project_id="demo",
            name="demo",
            input_video="/tmp/demo.mp4",
            root_dir="/tmp/project",
        ),
        settings=SPWMSettings(vlm=VLMConfig(api_key="system-key")),
    )

    payload = project_config_payload(config)

    assert "vlm" not in payload["settings"]


def test_save_project_config_omits_system_managed_vlm_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "project.toml"
    config = ProjectConfig(
        project=ProjectMetadata(
            project_id="demo",
            name="demo",
            input_video="/tmp/demo.mp4",
            root_dir="/tmp/project",
        ),
        settings=SPWMSettings(vlm=VLMConfig(api_key="system-key")),
    )

    save_project_config(config, config_path)
    saved = config_path.read_text(encoding="utf-8")

    assert "vlm =" not in saved
    assert "api_key" not in saved


def test_load_project_config_overlays_vlm_settings_from_system(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        """
[project]
project_id = "demo"
name = "demo"
input_video = "/tmp/demo.mp4"
root_dir = "/tmp/project"

[settings]
vlm = { mode = "embedded", backend = "api", api_key = "project-key", base_url = "https://project.example/v1", model = "project-model", max_retries = 3, command_template = "", service = { base_url = "http://127.0.0.1:8103", timeout_sec = 30.0 } }
runtime = { occluded_ttl_frames = 5, removal_ttl_frames = 20, vlm_discovery = { enabled = true, periodic_interval = 5, new_track_threshold = 1, disappear_threshold = 1, confidence_drop_threshold = 0.3, open_vocab_enabled = true, open_vocab_cooldown_frames = 10, image_change_threshold = 0.08, image_change_enabled = true }, video_source = "", session_output_root = "", save_intermediate = true, asset_materialization = "copy", background_reconstruction = true, background_sample_frames = 30, rerun_enabled = false }
storage = { world_id = "scene_main" }
isaac = { stage_path = "data/demo_scene.usd", auto_save = true }
pit = { camera_provider = "wildgs", colmap_model_dir = "", wildgs_camera_poses_jsonl = "", wildgs_static_map_dir = "", wildgs_dynamic_prior_dir = "", wildgs_depth_maps_dir = "", depth_provider = "wildgs", depth_model_path = "", frame_dump_dir = "", use_metric_scale = false, metric_scale_factor = 1.0, metric_scale_source = "manual", known_object_size_m = "", floor_plane_z_offset = 0.0, alignment_backend = "depth_icp", visual_pose_mcp_url = "", visual_pose_mcp_tool = "gotrack_refine_pose", visual_pose_command = "", visual_pose_timeout_sec = 30.0, visual_pose_min_score = 0.0, visual_pose_max_translation_step_m = 1.5, visual_pose_max_rotation_step_deg = 60.0 }
pit2isaac = { mode = "hybrid", output_root = "", usd_path = "", physics_priors_json = "", asset_mapping_json = "", conversion_report_json = "", use_category_assets = true, fallback_visual = "primitive", collision_strategy = "primitive", min_geom_quality = 0.5, output_format = "usdc" }
zaiwu = { enabled = false, gateway_url = "http://127.0.0.1:8181", request_timeout_sec = 30.0, job_timeout_sec = 1800.0, job_poll_interval_sec = 1.0, auto_start_workers = true, worker_run_group = "services", object_detection_backend = "seg2track_sam2", sam3_service = "services.sam3", grounded_sam2_service = "services.grounding_dino_sam2", seg2track_sam2_service = "services.seg2track_sam2", sam3d_service = "services.sam3d", depth_service = "services.depth_anything3", wildgs_slam_service = "services.wildgs_slam", gotrack_service = "services.gotrack", grounded_sam2 = { step = 20, iou_threshold = 0.8, box_threshold = 0.3, text_threshold = 0.25 }, seg2track_sam2 = { detect_interval = 5, box_threshold = 0.3, text_threshold = 0.25 } }

[workspace]

[payload]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("guanwu.video.project.config.load_settings", lambda: _system_settings("system-key"))

    config = load_project_config(config_path)

    assert config.settings.vlm.api_key == "system-key"
    assert config.settings.vlm.base_url == "https://system.example/v1"
    assert config.settings.vlm.model == "system-model"
    assert config.settings.vlm.command_template == "system-vlm"
    assert config.settings.vlm.max_retries == 7
