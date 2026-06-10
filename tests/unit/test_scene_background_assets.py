from __future__ import annotations

import base64
import io
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import openai
import trimesh
from PIL import Image

from guanwu.video.features.spatial.scene_background_assets import (
    _fill_low_candidate_dynamic_regions,
    _resolve_openai_image_edit_size,
    _resolve_depth_for_frame,
    build_dynamic_mask,
    build_static_guard_mask,
    generate_depth_background_mesh_assets,
    generate_target_frame_background_assets,
    load_background_asset_meshes,
)


def _image_b64(rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _zmask(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8).reshape(-1), bitorder="little").tobytes()
    return base64.b64encode(zlib.compress(packed)).decode("ascii")


def _write_frame(path: Path, frame_idx: int, rgb: np.ndarray, instances: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 30.0,
        "image_b64": _image_b64(rgb),
        "instances": instances,
    }
    out = path / "detections.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def _mask_instance(object_id: str, label: str, mask: np.ndarray, bbox: list[float]) -> dict:
    return {
        "object_id": object_id,
        "concept_label": label,
        "bbox": bbox,
        "score": 0.9,
        "mask_rle": json.dumps({"encoding": "zlib_packbits", "size": list(mask.shape), "counts": _zmask(mask)}),
    }


def test_resolve_openai_image_edit_size_uses_original_image_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "reference.png"
    Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8)).save(image_path)

    assert _resolve_openai_image_edit_size(image_path, "original") == "64x36"
    assert _resolve_openai_image_edit_size(image_path, "same") == "64x36"
    assert _resolve_openai_image_edit_size(image_path, "1024x1024") == "1024x1024"


def test_resolve_depth_for_frame_maps_pipeline_frame_to_zero_based_wildgs_depth(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth_maps"
    nested_depth_dir = depth_dir / "depth_maps"
    nested_depth_dir.mkdir(parents=True)
    expected = nested_depth_dir / "00002.npy"
    stale_one_based = nested_depth_dir / "00003.npy"
    np.save(expected, np.zeros((2, 2), dtype=np.float32))
    np.save(stale_one_based, np.ones((2, 2), dtype=np.float32))

    assert _resolve_depth_for_frame(depth_dir, 3) == expected


def test_generate_background_assets_defaults_to_first_frame_and_uses_da3_clean_depth_directly(tmp_path: Path) -> None:
    rgb = np.full((24, 32, 3), 96, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[10:18, 12:20] = True
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 1,
                        "detections": str(
                            _write_frame(
                                tmp_path / "frame_000001",
                                1,
                                rgb,
                                [_mask_instance("obj_000001", "object", mask, [12, 10, 20, 18])],
                            )
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    camera_trajectory = tmp_path / "camera_trajectory.json"
    camera_trajectory.write_text(
        json.dumps(
            [
                {
                    "frame_id": 1,
                    "K": [[28.0, 0.0, 16.0], [0.0, 28.0, 12.0], [0.0, 0.0, 1.0]],
                    "R": np.eye(3).tolist(),
                    "t": [0.0, 0.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    wildgs_depth_dir = tmp_path / "wildgs_depth"
    wildgs_depth_dir.mkdir()
    np.save(wildgs_depth_dir / "00000.npy", np.full((24, 32), 6.0, dtype=np.float32))
    external_depth = tmp_path / "external_depth.npy"
    np.save(external_depth, np.full((24, 32), 9.0, dtype=np.float32))

    result = generate_target_frame_background_assets(
        summary_path=summary_path,
        output_dir=tmp_path / "background_assets",
        depth_maps_dir=wildgs_depth_dir,
        camera_trajectory_path=camera_trajectory,
        clean_depth_estimator=lambda _path: {
            "depth_path": external_depth,
            "source": "depth_anything3_clean_rgb",
        },
        grid_stride=4,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["target_frame_id"] == 1
    assert manifest["quality"]["depth_calibration_source"] == "da3_metric_direct"
    assert "depth_calibration_reference" not in manifest["quality"]
    assert "wildgs_depth_index" not in manifest["quality"]
    assert float(np.load(manifest["assets"]["clean_depth"])[0, 0]) == 9.0


def test_openai_image_cleaner_uses_api_size_and_saves_original_dimensions(tmp_path: Path) -> None:
    from guanwu.video.features.spatial.scene_background_assets import run_openai_image_edit_background_cleaner

    reference_path = tmp_path / "reference.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "clean.png"
    Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8)).save(reference_path)
    Image.fromarray(np.zeros((36, 64, 4), dtype=np.uint8)).save(mask_path)
    captured: dict[str, str] = {}

    class FakeImages:
        def edit(self, **kwargs):
            captured["size"] = kwargs["size"]
            edited = Image.fromarray(np.full((1024, 1536, 3), 180, dtype=np.uint8))
            buffer = io.BytesIO()
            edited.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            return SimpleNamespace(data=[SimpleNamespace(b64_json=b64)])

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.images = FakeImages()

    original = openai.OpenAI
    openai.OpenAI = FakeOpenAI
    try:
        run_openai_image_edit_background_cleaner(
            image_path=reference_path,
            mask_path=mask_path,
            output_path=output_path,
            config={"api_key": "test-key", "size": "original", "api_size": "1536x1024"},
        )
    finally:
        openai.OpenAI = original

    assert captured["size"] == "1536x1024"
    with Image.open(output_path) as image:
        assert image.size == (64, 36)


def test_openai_image_cleaner_prompt_only_omits_mask(tmp_path: Path) -> None:
    from guanwu.video.features.spatial.scene_background_assets import run_openai_image_edit_background_cleaner

    reference_path = tmp_path / "reference.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "clean.png"
    Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8)).save(reference_path)
    Image.fromarray(np.zeros((36, 64, 4), dtype=np.uint8)).save(mask_path)
    captured: dict[str, object] = {}

    class FakeImages:
        def edit(self, **kwargs):
            captured.update(kwargs)
            edited = Image.fromarray(np.full((1024, 1536, 3), 180, dtype=np.uint8))
            buffer = io.BytesIO()
            edited.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            return SimpleNamespace(data=[SimpleNamespace(b64_json=b64)])

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.images = FakeImages()

    original = openai.OpenAI
    openai.OpenAI = FakeOpenAI
    try:
        run_openai_image_edit_background_cleaner(
            image_path=reference_path,
            mask_path=mask_path,
            output_path=output_path,
            config={"api_key": "test-key", "api_size": "1536x1024", "mask_mode": "none"},
        )
    finally:
        openai.OpenAI = original

    assert "mask" not in captured
    assert captured["size"] == "1536x1024"
    with Image.open(output_path) as image:
        assert image.size == (64, 36)


def test_openai_image_cleaner_uses_system_vlm_config_when_api_key_missing(monkeypatch, tmp_path: Path) -> None:
    from guanwu.video.core import config as core_config
    from guanwu.video.features.spatial.scene_background_assets import run_openai_image_edit_background_cleaner

    system_config = tmp_path / "video.config.toml"
    system_config.write_text(
        """
[vlm]
mode = "embedded"
backend = "api"
api_key = "system-key"
base_url = "https://system.example/v1"
model = "gpt-5.5"
max_retries = 3
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(core_config, "DEFAULT_CONFIG_PATH", system_config)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    reference_path = tmp_path / "reference.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "clean.png"
    Image.fromarray(np.zeros((36, 64, 3), dtype=np.uint8)).save(reference_path)
    Image.fromarray(np.zeros((36, 64, 4), dtype=np.uint8)).save(mask_path)
    captured: dict[str, object] = {}

    class FakeImages:
        def edit(self, **kwargs):
            edited = Image.fromarray(np.full((36, 64, 3), 180, dtype=np.uint8))
            buffer = io.BytesIO()
            edited.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            return SimpleNamespace(data=[SimpleNamespace(b64_json=b64)])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.images = FakeImages()

    original = openai.OpenAI
    openai.OpenAI = FakeOpenAI
    try:
        run_openai_image_edit_background_cleaner(
            image_path=reference_path,
            mask_path=mask_path,
            output_path=output_path,
            config={"api_key": "", "base_url": "", "api_size": "64x36", "mask_mode": "prompt_only"},
        )
    finally:
        openai.OpenAI = original

    assert captured["api_key"] == "system-key"
    assert captured["base_url"] == "https://system.example/v1"
    assert output_path.exists()


def test_build_dynamic_mask_uses_only_movable_categories_and_expands_shadow() -> None:
    car = np.zeros((24, 32), dtype=bool)
    car[8:14, 10:18] = True
    fence = np.zeros((24, 32), dtype=bool)
    fence[3:7, 2:30] = True
    detections = {
        "instances": [
            _mask_instance("car_1", "car", car, [10, 8, 18, 14]),
            _mask_instance("static_1", "fence railing", fence, [2, 3, 30, 7]),
        ]
    }

    dynamic = build_dynamic_mask(detections, (24, 32), foreground_expand_px=2, shadow_expand_px=4)

    assert dynamic[10, 14]
    assert dynamic[17, 14]
    assert not dynamic[4, 3]


def test_build_static_guard_mask_uses_static_boundary_categories_not_road() -> None:
    fence = np.zeros((24, 32), dtype=bool)
    fence[3:7, 2:30] = True
    road = np.zeros((24, 32), dtype=bool)
    road[14:, :] = True
    car = np.zeros((24, 32), dtype=bool)
    car[8:14, 10:18] = True
    detections = {
        "instances": [
            _mask_instance("static_1", "fence railing", fence, [2, 3, 30, 7]),
            _mask_instance("road_1", "road", road, [0, 14, 32, 24]),
            _mask_instance("car_1", "car", car, [10, 8, 18, 14]),
        ]
    }

    guard = build_static_guard_mask(detections, (24, 32), expand_px=0)

    assert guard[4, 3]
    assert not guard[18, 16]
    assert not guard[10, 14]


def test_build_dynamic_mask_uses_smaller_expansion_for_tiny_objects() -> None:
    tiny = np.zeros((48, 80), dtype=bool)
    tiny[12:16, 20:26] = True
    detections = {"instances": [_mask_instance("tiny_car", "car", tiny, [20, 12, 26, 16])]}

    dynamic = build_dynamic_mask(detections, (48, 80), foreground_expand_px=8, shadow_expand_px=0)

    assert dynamic[12, 16]
    assert not dynamic[12, 12]


def test_generate_road_clean_background_uses_openai_prompt_only_and_da3_depth(tmp_path: Path) -> None:
    height, width = 36, 64
    road_color = np.array([84, 86, 88], dtype=np.uint8)
    car_color = np.array([230, 30, 20], dtype=np.uint8)
    clean_color = np.array([92, 94, 96], dtype=np.uint8)
    car_mask = np.zeros((height, width), dtype=bool)
    car_mask[18:29, 24:42] = True
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = road_color
    rgb[car_mask] = car_color
    detections = _write_frame(
        tmp_path / "frame_000003",
        3,
        rgb,
        [_mask_instance("car_1", "car", car_mask, [24, 18, 42, 29])],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": [{"frame_idx": 3, "detections": str(detections)}]}), encoding="utf-8")
    camera_trajectory = tmp_path / "camera_trajectory.json"
    camera_trajectory.write_text(
        json.dumps(
            [
                {
                    "frame_id": 3,
                    "K": [[28.0, 0.0, 32.0], [0.0, 28.0, 18.0], [0.0, 0.0, 1.0]],
                    "R": np.eye(3).tolist(),
                    "t": [0.0, 0.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    clean_depth = tmp_path / "clean_road_depth.npy"
    np.save(clean_depth, np.full((height, width), 7.0, dtype=np.float32))
    calls: list[dict] = []

    def fake_cleaner(**kwargs):
        calls.append(kwargs)
        mask_rgba = Image.open(kwargs["mask_path"]).convert("RGBA")
        alpha = np.asarray(mask_rgba.getchannel("A"))
        assert int(np.count_nonzero(alpha)) == 0
        assert kwargs["config"]["mask_mode"] == "prompt_only"
        assert kwargs["config"]["use_full_image_output"] is True
        assert "Remove all vehicles" in kwargs["config"]["prompt"]
        assert "lane width" in kwargs["config"]["prompt"]
        edited = np.zeros((height, width, 3), dtype=np.uint8)
        edited[:] = clean_color
        Image.fromarray(edited).save(kwargs["output_path"])
        return {
            "clean_rgb_path": str(kwargs["output_path"]),
            "raw_output_path": str(kwargs["output_path"]),
            "model": kwargs["config"]["model"],
            "prompt": kwargs["config"]["prompt"],
        }

    def estimate(clean_rgb_path: Path) -> dict:
        assert clean_rgb_path.name == "clean_target_rgb.png"
        return {
            "depth_path": clean_depth,
            "source": "depth_anything3_clean_target_rgb",
            "quality": {"depth_service": "fake_depth_anything3"},
        }

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        camera_trajectory_path=camera_trajectory,
        clean_depth_estimator=estimate,
        background_mode="road_clean_background",
        background_cleaner="openai_image_edit",
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert manifest["schema"] == "guanwu.target_frame_background_assets.road_clean_depth.v2"
    assert manifest["quality"]["background_mode"] == "road_clean_background"
    assert manifest["quality"]["source_frame_count"] == 0
    assert manifest["quality"]["clean_rgb_source"] == "openai_image_edit"
    assert manifest["quality"]["clean_rgb_mask_mode"] == "prompt_only"
    assert manifest["quality"]["clean_rgb_full_image_output"] is True
    assert manifest["quality"]["depth_background_source"] == "depth_anything3_clean_target_rgb"
    assert manifest["quality"]["depth_calibration_source"] == "da3_metric_direct"
    assert "depth_background_glb" in manifest["assets"]
    assert "clean_depth" in manifest["assets"]
    assert set(manifest["assets"]) >= {
        "clean_rgb",
        "dynamic_mask",
        "openai_image_edit_reference",
        "depth_background_glb",
        "clean_depth",
    }
    assert load_background_asset_meshes(result["manifest_path"])[0][0] == "depth_background"


def test_generate_default_non_tabletop_background_uses_road_clean_openai_path(tmp_path: Path) -> None:
    height, width = 36, 64
    car_mask = np.zeros((height, width), dtype=bool)
    car_mask[18:29, 24:42] = True
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = (84, 86, 88)
    rgb[car_mask] = (230, 30, 20)
    detections = _write_frame(
        tmp_path / "frame_000001",
        1,
        rgb,
        [_mask_instance("car_1", "car", car_mask, [24, 18, 42, 29])],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": [{"frame_idx": 1, "detections": str(detections)}]}), encoding="utf-8")
    calls: list[dict] = []

    def fake_cleaner(**kwargs):
        calls.append(kwargs)
        edited = np.zeros((height, width, 3), dtype=np.uint8)
        edited[:] = (92, 94, 96)
        Image.fromarray(edited).save(kwargs["output_path"])
        return {
            "clean_rgb_path": str(kwargs["output_path"]),
            "raw_output_path": str(kwargs["output_path"]),
            "model": kwargs["config"]["model"],
            "prompt": kwargs["config"]["prompt"],
        }

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert manifest["quality"]["background_mode"] == "road_clean_background"
    assert manifest["quality"]["clean_rgb_source"] == "openai_image_edit"
    assert manifest["quality"]["clean_rgb_mask_mode"] == "prompt_only"
    assert "openai_image_edit_reference" in manifest["assets"]


def test_generate_tabletop_task_background_assets_only_masks_target_object(tmp_path: Path) -> None:
    frames = []
    height, width = 36, 64
    block_mask = np.zeros((height, width), dtype=bool)
    block_mask[17:25, 27:39] = True
    arm_mask = np.zeros((height, width), dtype=bool)
    arm_mask[5:18, 8:22] = True
    table_color = np.array([122, 112, 94], dtype=np.uint8)
    block_color = np.array([188, 128, 62], dtype=np.uint8)
    arm_color = np.array([42, 54, 68], dtype=np.uint8)
    for frame_idx in [1, 2, 3, 4, 5]:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = table_color
        rgb[arm_mask] = arm_color
        instances = [_mask_instance("obj_arm", "robot arm gripper", arm_mask, [8, 5, 22, 18])]
        if frame_idx == 3:
            rgb[block_mask] = block_color
            instances.append(_mask_instance("obj_000009", "wooden block", block_mask, [27, 17, 39, 25]))
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(_write_frame(tmp_path / f"frame_{frame_idx:06d}", frame_idx, rgb, instances)),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_000009"],
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "guanwu.target_frame_background_assets.tabletop.v1"
    assert manifest["quality"]["background_mode"] == "tabletop_task"
    assert manifest["quality"]["target_foreground_object_ids"] == ["obj_000009"]
    assert "tabletop_mesh" in manifest["assets"]

    foreground = cv2.imread(manifest["assets"]["dynamic_mask"], cv2.IMREAD_GRAYSCALE) > 0
    assert foreground[20, 31]
    assert not foreground[10, 12]

    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    assert np.linalg.norm(clean[20, 31].astype(np.float32) - table_color.astype(np.float32)) < 10.0
    assert np.linalg.norm(clean[10, 12].astype(np.float32) - arm_color.astype(np.float32)) < 10.0

    meshes = load_background_asset_meshes(result["manifest_path"])
    assert [(name, path.name) for name, path in meshes] == [("tabletop", "tabletop_background.obj")]


def test_generate_tabletop_task_background_assets_writes_tabletop_reference_from_clean_depth(tmp_path: Path) -> None:
    height, width = 24, 32
    rgb = np.full((height, width, 3), (122, 112, 94), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[10:16, 13:20] = True
    rgb[mask] = (188, 128, 62)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 1,
                        "detections": str(
                            _write_frame(
                                tmp_path / "frame_000001",
                                1,
                                rgb,
                                [_mask_instance("obj_000009", "wooden block", mask, [13, 10, 20, 16])],
                            )
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    camera_trajectory = tmp_path / "camera_trajectory.json"
    camera_trajectory.write_text(
        json.dumps(
            [
                {
                    "frame_id": 1,
                    "K": [[24.0, 0.0, 16.0], [0.0, 24.0, 12.0], [0.0, 0.0, 1.0]],
                    "R": np.eye(3).tolist(),
                    "t": [0.0, 0.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    yy, xx = np.mgrid[0:height, 0:width]
    external_depth = tmp_path / "external_depth.npy"
    np.save(external_depth, (4.0 + 0.01 * xx + 0.02 * yy).astype(np.float32))

    result = generate_target_frame_background_assets(
        summary_path=summary_path,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_000009"],
        camera_trajectory_path=camera_trajectory,
        clean_depth_estimator=lambda _path: {
            "depth_path": external_depth,
            "source": "depth_anything3_clean_rgb",
        },
        grid_stride=4,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    reference_path = Path(manifest["assets"]["tabletop_reference"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    geometry_reference_path = Path(manifest["assets"]["background_geometry_reference"])
    geometry_reference = json.loads(geometry_reference_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "guanwu.target_frame_background_assets.tabletop_depth.v2"
    assert reference["source"] == "clean_depth_background"
    assert len(reference["normal_world"]) == 3
    assert np.isclose(np.linalg.norm(reference["normal_world"]), 1.0)
    assert np.isfinite(float(reference["offset"]))
    assert manifest["tabletop_reference"]["path"] == str(reference_path)
    assert manifest["background_geometry_reference"]["path"] == str(geometry_reference_path)
    assert geometry_reference["schema"] == "guanwu.background_geometry_reference.v1"
    assert geometry_reference["reference_type"] == "support_surface"
    assert geometry_reference["source"] == "clean_background_depth"
    assert geometry_reference["support_surfaces"][0]["type"] == "plane"
    assert geometry_reference["support_surfaces"][0]["normal_world"] == reference["normal_world"]
    assert geometry_reference["support_surfaces"][0]["offset"] == reference["offset"]
    assert Path(geometry_reference["exclusion"]["foreground_mask_path"]).exists()
    assert load_background_asset_meshes(result["manifest_path"])[0][0] == "depth_background"


def test_generate_tabletop_task_background_assets_uses_openai_image_cleaner_on_reference_frame(tmp_path: Path) -> None:
    frames = []
    height, width = 36, 64
    block_mask_frame1 = np.zeros((height, width), dtype=bool)
    block_mask_frame1[15:23, 22:34] = True
    block_mask_frame3 = np.zeros((height, width), dtype=bool)
    block_mask_frame3[17:25, 28:40] = True
    arm_mask = np.zeros((height, width), dtype=bool)
    arm_mask[5:18, 8:22] = True
    table_color = np.array([122, 112, 94], dtype=np.uint8)
    block_color = np.array([188, 128, 62], dtype=np.uint8)
    arm_color = np.array([42, 54, 68], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = table_color
        rgb[arm_mask] = arm_color
        instances = [_mask_instance("obj_arm", "robot arm gripper", arm_mask, [8, 5, 22, 18])]
        if frame_idx == 1:
            rgb[block_mask_frame1] = block_color
            instances.append(_mask_instance("obj_000009", "wooden block", block_mask_frame1, [22, 15, 34, 23]))
        if frame_idx == 3:
            rgb[block_mask_frame3] = block_color
            instances.append(_mask_instance("obj_000009", "wooden block", block_mask_frame3, [28, 17, 40, 25]))
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(_write_frame(tmp_path / f"frame_{frame_idx:06d}", frame_idx, rgb, instances)),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    calls: list[dict] = []

    def fake_cleaner(**kwargs):
        calls.append(kwargs)
        reference = cv2.cvtColor(cv2.imread(str(kwargs["image_path"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(kwargs["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
        edited = reference.copy()
        edited[mask] = table_color
        Image.fromarray(edited).save(kwargs["output_path"])
        return {
            "clean_rgb_path": str(kwargs["output_path"]),
            "raw_output_path": str(kwargs["output_path"]),
            "model": kwargs["config"]["model"],
            "prompt": kwargs["config"]["prompt"],
        }

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_000009"],
        background_cleaner="openai_image_edit",
        background_cleaner_config={
            "model": "gpt-image-2",
            "prompt": "remove only the wooden block",
            "mask_mode": "target",
            "use_full_image_output": False,
        },
        background_cleaner_reference_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert calls[0]["reference_frame_id"] == 1
    assert calls[0]["config"]["model"] == "gpt-image-2"
    assert manifest["quality"]["clean_rgb_source"] == "openai_image_edit"
    assert manifest["quality"]["clean_rgb_reference_frame_id"] == 1
    assert manifest["quality"]["clean_rgb_model"] == "gpt-image-2"
    assert "openai_image_edit_mask" in manifest["assets"]
    assert "openai_image_edit_raw" in manifest["assets"]

    cleaner_mask = cv2.imread(manifest["assets"]["openai_image_edit_mask"], cv2.IMREAD_GRAYSCALE) > 0
    assert cleaner_mask[18, 26]
    assert cleaner_mask[21, 34]
    assert not cleaner_mask[10, 12]

    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    assert np.linalg.norm(clean[18, 26].astype(np.float32) - table_color.astype(np.float32)) < 10.0
    assert np.linalg.norm(clean[10, 12].astype(np.float32) - arm_color.astype(np.float32)) < 10.0


def test_generate_tabletop_task_background_assets_can_clean_full_image(tmp_path: Path) -> None:
    frames = []
    height, width = 36, 64
    block_mask = np.zeros((height, width), dtype=bool)
    block_mask[15:23, 22:34] = True
    arm_mask = np.zeros((height, width), dtype=bool)
    arm_mask[5:18, 8:22] = True
    table_color = np.array([122, 112, 94], dtype=np.uint8)
    block_color = np.array([188, 128, 62], dtype=np.uint8)
    arm_color = np.array([42, 54, 68], dtype=np.uint8)
    clean_background_color = np.array([150, 132, 96], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = table_color
        rgb[arm_mask] = arm_color
        rgb[block_mask] = block_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [
                            _mask_instance("obj_arm", "robot arm gripper", arm_mask, [8, 5, 22, 18]),
                            _mask_instance("obj_000009", "wooden block", block_mask, [22, 15, 34, 23]),
                        ],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    def fake_cleaner(**kwargs):
        mask_rgba = Image.open(kwargs["mask_path"]).convert("RGBA")
        alpha = np.asarray(mask_rgba.getchannel("A"))
        assert int(np.count_nonzero(alpha == 0)) == height * width
        edited = np.zeros((height, width, 3), dtype=np.uint8)
        edited[:] = clean_background_color
        Image.fromarray(edited).save(kwargs["output_path"])
        return {
            "clean_rgb_path": str(kwargs["output_path"]),
            "raw_output_path": str(kwargs["output_path"]),
            "model": kwargs["config"]["model"],
            "prompt": kwargs["config"]["prompt"],
        }

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_000009"],
        background_cleaner="openai_image_edit",
        background_cleaner_config={
            "model": "gpt-image-2",
            "prompt": "generate a clean empty tabletop background",
            "mask_mode": "full_image",
            "use_full_image_output": True,
        },
        background_cleaner_reference_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    cleaner_mask = Image.open(manifest["assets"]["openai_image_edit_mask"]).convert("RGBA")
    alpha = np.asarray(cleaner_mask.getchannel("A"))
    assert int(np.count_nonzero(alpha == 0)) == height * width
    assert manifest["quality"]["clean_rgb_mask_mode"] == "full_image"
    assert manifest["quality"]["clean_rgb_full_image_output"] is True

    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    assert np.linalg.norm(clean[10, 12].astype(np.float32) - clean_background_color.astype(np.float32)) < 1.0
    assert np.linalg.norm(clean[18, 26].astype(np.float32) - clean_background_color.astype(np.float32)) < 1.0


def test_generate_tabletop_task_background_assets_can_clean_foreground_objects_only(tmp_path: Path) -> None:
    frames = []
    height, width = 36, 64
    block_mask = np.zeros((height, width), dtype=bool)
    block_mask[15:23, 22:34] = True
    arm_mask = np.zeros((height, width), dtype=bool)
    arm_mask[5:18, 8:22] = True
    paper_mask = np.zeros((height, width), dtype=bool)
    paper_mask[25:34, 45:60] = True
    table_color = np.array([122, 112, 94], dtype=np.uint8)
    block_color = np.array([188, 128, 62], dtype=np.uint8)
    arm_color = np.array([42, 54, 68], dtype=np.uint8)
    paper_color = np.array([220, 220, 210], dtype=np.uint8)
    clean_background_color = np.array([150, 132, 96], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = table_color
        rgb[arm_mask] = arm_color
        rgb[block_mask] = block_color
        rgb[paper_mask] = paper_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [
                            _mask_instance("obj_arm", "robot arm gripper", arm_mask, [8, 5, 22, 18]),
                            _mask_instance("obj_000009", "wooden block", block_mask, [22, 15, 34, 23]),
                            _mask_instance("obj_paper", "paper sheet", paper_mask, [45, 25, 60, 34]),
                        ],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    def fake_cleaner(**kwargs):
        reference = cv2.cvtColor(cv2.imread(str(kwargs["image_path"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(kwargs["mask_path"]), cv2.IMREAD_UNCHANGED)
        alpha = mask[:, :, 3]
        edit_mask = alpha == 0
        assert edit_mask[10, 12]
        assert edit_mask[18, 26]
        assert edit_mask[29, 50]
        assert not edit_mask[2, 30]
        edited = reference.copy()
        edited[edit_mask] = clean_background_color
        Image.fromarray(edited).save(kwargs["output_path"])
        return {
            "clean_rgb_path": str(kwargs["output_path"]),
            "raw_output_path": str(kwargs["output_path"]),
            "model": kwargs["config"]["model"],
            "prompt": kwargs["config"]["prompt"],
        }

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_000009"],
        background_cleaner="openai_image_edit",
        background_cleaner_config={
            "model": "gpt-image-2",
            "prompt": "remove foreground objects and preserve visible background",
            "mask_mode": "foreground_objects",
            "use_full_image_output": False,
            "mask_expand_px": 0,
        },
        background_cleaner_reference_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["quality"]["clean_rgb_mask_mode"] == "foreground_objects"
    assert manifest["quality"]["clean_rgb_full_image_output"] is False

    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    assert np.linalg.norm(clean[10, 12].astype(np.float32) - clean_background_color.astype(np.float32)) < 1.0
    assert np.linalg.norm(clean[18, 26].astype(np.float32) - clean_background_color.astype(np.float32)) < 1.0
    assert np.linalg.norm(clean[29, 50].astype(np.float32) - clean_background_color.astype(np.float32)) < 1.0
    assert np.linalg.norm(clean[2, 30].astype(np.float32) - table_color.astype(np.float32)) < 4.0


def test_generate_tabletop_task_background_assets_preserves_background_instances_in_foreground_mode(tmp_path: Path) -> None:
    height, width = 36, 64
    board_mask = np.zeros((height, width), dtype=bool)
    board_mask[2:34, 18:44] = True
    metal_mask = np.zeros((height, width), dtype=bool)
    metal_mask[:, 44:64] = True
    arm_mask = np.zeros((height, width), dtype=bool)
    arm_mask[5:18, 8:22] = True
    block_mask = np.zeros((height, width), dtype=bool)
    block_mask[15:23, 28:36] = True
    board_color = np.array([122, 112, 94], dtype=np.uint8)
    metal_color = np.array([160, 170, 172], dtype=np.uint8)
    clean_background_color = np.array([150, 132, 96], dtype=np.uint8)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = np.array([20, 30, 40], dtype=np.uint8)
    rgb[board_mask] = board_color
    rgb[metal_mask] = metal_color
    rgb[arm_mask] = np.array([42, 54, 68], dtype=np.uint8)
    rgb[block_mask] = np.array([188, 128, 62], dtype=np.uint8)
    frame_path = _write_frame(
        tmp_path / "frame_000001",
        1,
        rgb,
        [
            _mask_instance("obj_arm", "robotic arm", arm_mask, [8, 5, 22, 18]),
            _mask_instance("obj_block", "wooden block", block_mask, [28, 15, 36, 23]),
            _mask_instance("obj_board", "wooden board", board_mask, [18, 2, 44, 34]),
            _mask_instance("obj_metal", "metal table", metal_mask, [44, 0, 64, 36]),
        ],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": [{"frame_idx": 1, "detections": str(frame_path)}]}), encoding="utf-8")

    def fake_cleaner(**kwargs):
        reference = cv2.cvtColor(cv2.imread(str(kwargs["image_path"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(kwargs["mask_path"]), cv2.IMREAD_UNCHANGED)
        edit_mask = mask[:, :, 3] == 0
        assert edit_mask[10, 12]
        assert edit_mask[18, 31]
        assert not edit_mask[4, 30]
        assert not edit_mask[10, 50]
        edited = reference.copy()
        edited[edit_mask] = clean_background_color
        Image.fromarray(edited).save(kwargs["output_path"])
        return {"raw_output_path": str(kwargs["output_path"]), "model": kwargs["config"]["model"], "prompt": kwargs["config"]["prompt"]}

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_block"],
        background_cleaner="openai_image_edit",
        background_cleaner_config={
            "model": "gpt-image-2",
            "prompt": "remove foreground objects and preserve visible background",
            "mask_mode": "foreground_objects",
            "use_full_image_output": False,
            "mask_expand_px": 0,
        },
        background_cleaner_reference_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["quality"]["clean_rgb_mask_mode"] == "foreground_objects"
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    assert np.linalg.norm(clean[4, 30].astype(np.float32) - board_color.astype(np.float32)) < 4.0
    assert np.linalg.norm(clean[10, 50].astype(np.float32) - metal_color.astype(np.float32)) < 4.0


def test_generate_tabletop_task_background_assets_uses_reference_frame_mask_by_default(tmp_path: Path) -> None:
    height, width = 36, 64
    table_color = np.array([122, 112, 94], dtype=np.uint8)
    clean_background_color = np.array([150, 132, 96], dtype=np.uint8)
    frame_entries = []
    for frame_idx, x1 in [(1, 10), (2, 40)]:
        obj_mask = np.zeros((height, width), dtype=bool)
        obj_mask[12:22, x1 : x1 + 10] = True
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = table_color
        rgb[obj_mask] = np.array([42, 54, 68], dtype=np.uint8)
        frame_entries.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [_mask_instance(f"obj_arm_{frame_idx}", "robotic arm", obj_mask, [x1, 12, x1 + 10, 22])],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frame_entries}), encoding="utf-8")

    def fake_cleaner(**kwargs):
        reference = cv2.cvtColor(cv2.imread(str(kwargs["image_path"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(kwargs["mask_path"]), cv2.IMREAD_UNCHANGED)
        edit_mask = mask[:, :, 3] == 0
        assert edit_mask[16, 15]
        assert not edit_mask[16, 45]
        edited = reference.copy()
        edited[edit_mask] = clean_background_color
        Image.fromarray(edited).save(kwargs["output_path"])
        return {"raw_output_path": str(kwargs["output_path"]), "model": kwargs["config"]["model"], "prompt": kwargs["config"]["prompt"]}

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_arm_1"],
        background_cleaner="openai_image_edit",
        background_cleaner_config={
            "model": "gpt-image-2",
            "prompt": "remove foreground objects and preserve visible background",
            "mask_mode": "foreground_objects",
            "use_full_image_output": False,
            "mask_expand_px": 0,
        },
        background_cleaner_reference_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["quality"]["clean_rgb_mask_frame_mode"] == "reference_frame"
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    assert np.linalg.norm(clean[16, 15].astype(np.float32) - clean_background_color.astype(np.float32)) < 1.0
    assert np.linalg.norm(clean[16, 45].astype(np.float32) - table_color.astype(np.float32)) < 4.0


def test_generate_tabletop_task_background_assets_reads_full_image_mode_from_config_path(tmp_path: Path) -> None:
    height, width = 18, 32
    block_mask = np.zeros((height, width), dtype=bool)
    block_mask[7:11, 12:18] = True
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = np.array([122, 112, 94], dtype=np.uint8)
    frame_path = _write_frame(
        tmp_path / "frame_000001",
        1,
        rgb,
        [_mask_instance("obj_000009", "wooden block", block_mask, [12, 7, 18, 11])],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": [{"frame_idx": 1, "detections": str(frame_path)}]}), encoding="utf-8")
    cleaner_config_path = tmp_path / "cleaner.yaml"
    cleaner_config_path.write_text(
        """
openai:
  api_key: test-key
image_edit:
  model: gpt-image-2
  mask_mode: full_image
  use_full_image_output: true
  prompt: clean empty tabletop
""".strip(),
        encoding="utf-8",
    )

    def fake_cleaner(**kwargs):
        mask_rgba = Image.open(kwargs["mask_path"]).convert("RGBA")
        alpha = np.asarray(mask_rgba.getchannel("A"))
        assert int(np.count_nonzero(alpha == 0)) == height * width
        edited = np.full((height, width, 3), 140, dtype=np.uint8)
        Image.fromarray(edited).save(kwargs["output_path"])
        return {"raw_output_path": str(kwargs["output_path"]), "model": kwargs["config"]["model"], "prompt": kwargs["config"]["prompt"]}

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=1,
        background_mode="tabletop_task",
        task_foreground_object_ids=["obj_000009"],
        background_cleaner="openai_image_edit",
        background_cleaner_config={"config_path": str(cleaner_config_path), "model": "gpt-image-2"},
        background_cleaner_reference_frame_id=1,
        background_image_cleaner=fake_cleaner,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["quality"]["clean_rgb_mask_mode"] == "full_image"
    assert manifest["quality"]["clean_rgb_full_image_output"] is True


def test_low_candidate_far_fill_does_not_pull_diagonal_structure_into_vehicle_hole(tmp_path: Path) -> None:
    clean = np.zeros((80, 160, 3), dtype=np.uint8)
    for y in range(80):
        for x in range(160):
            clean[y, x] = (80 + x // 6 + y // 20, 86 + x // 8, 92 + x // 10)
    for i in range(20):
        y = 8 + i
        x = 48 + i
        clean[y : y + 2, x : x + 18] = (150, 150, 150)
    object_mask = np.zeros((80, 160), dtype=bool)
    object_mask[12:30, 50:88] = True
    target = clean.copy()
    target[object_mask] = (245, 245, 245)
    source_count = np.full((80, 160), 5, dtype=np.uint16)
    source_count[object_mask] = 0

    filled = _fill_low_candidate_dynamic_regions(clean.copy(), target, object_mask, source_count)

    patch = filled[12:30, 50:88].astype(np.float32)
    center = filled[21, 69].astype(np.float32)
    expected_center = (filled[21, 49].astype(np.float32) + filled[21, 88].astype(np.float32)) * 0.5
    assert np.linalg.norm(center - expected_center) < 10.0
    assert float(np.percentile(patch[..., 0], 95)) < 120.0
    assert float(np.abs(np.diff(patch, axis=1)).max()) < 12.0


def test_low_candidate_far_fill_does_not_spread_lane_markings_across_hole() -> None:
    clean = np.zeros((90, 160, 3), dtype=np.uint8)
    clean[:] = (92, 96, 100)
    clean[18:22, 49] = (238, 238, 238)
    clean[18:22, 88] = (238, 238, 238)
    object_mask = np.zeros((90, 160), dtype=bool)
    object_mask[12:30, 50:88] = True
    target = clean.copy()
    target[object_mask] = (245, 245, 245)
    source_count = np.full((90, 160), 5, dtype=np.uint16)
    source_count[object_mask] = 0

    filled = _fill_low_candidate_dynamic_regions(clean.copy(), target, object_mask, source_count)

    patch = filled[12:30, 50:88].astype(np.float32)
    bright_row = patch[6:10]
    assert float(np.percentile(bright_row, 95)) < 130.0
    assert float(np.abs(np.diff(patch, axis=1)).max()) < 20.0
    assert np.linalg.norm(patch.mean(axis=(0, 1)) - np.array([245, 245, 245], dtype=np.float32)) > 180.0


def test_low_candidate_fill_preserves_sparse_real_donor_pixels() -> None:
    clean = np.zeros((80, 160, 3), dtype=np.uint8)
    clean[:] = (92, 96, 100)
    object_mask = np.zeros((80, 160), dtype=bool)
    object_mask[12:30, 50:88] = True
    clean[object_mask] = (104, 108, 112)
    clean[18:22, 50:88] = (68, 72, 76)
    target = clean.copy()
    target[object_mask] = (245, 245, 245)
    source_count = np.full((80, 160), 5, dtype=np.uint16)
    source_count[object_mask] = 1

    filled = _fill_low_candidate_dynamic_regions(clean.copy(), target, object_mask, source_count)

    assert np.array_equal(filled[object_mask], clean[object_mask])


def test_load_background_asset_meshes_prefers_explicit_background_mesh(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    background = assets / "background.glb"
    background.write_bytes(b"glb")
    manifest = assets / "background_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "guanwu.target_frame_background_assets.v2",
                "assets": {
                    "background_mesh": str(background),
                },
            }
        ),
        encoding="utf-8",
    )

    meshes = load_background_asset_meshes(str(manifest))

    assert [(name, path.name) for name, path in meshes] == [("background", "background.glb")]


def test_generate_depth_background_mesh_assets_writes_colored_glb_and_manifest(tmp_path: Path) -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(32, dtype=np.uint8)[None, :] * 4
    rgb[..., 1] = np.arange(24, dtype=np.uint8)[:, None] * 6
    rgb[..., 2] = 120
    rgb_path = tmp_path / "clean_target_rgb.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth = np.ones((24, 32), dtype=np.float32) * 8.0
    depth_path = tmp_path / "clean_target_depth.npy"
    np.save(depth_path, depth)

    result = generate_depth_background_mesh_assets(
        clean_rgb_path=rgb_path,
        depth_path=depth_path,
        output_dir=tmp_path / "depth_background",
        camera={
            "fx": 24.0,
            "fy": 24.0,
            "cx": 16.0,
            "cy": 12.0,
            "R": np.eye(3).tolist(),
            "t": [0.0, 0.0, 0.0],
        },
        grid_stride=4,
        target_frame_id=3,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["assets"]["depth_background_glb"].endswith("depth_background.glb")
    assert Path(manifest["assets"]["depth_background_glb"]).exists()
    assert Path(manifest["assets"]["clean_depth"]).exists()
    assert manifest["quality"]["vertex_count"] > 0
    assert manifest["quality"]["face_count"] > 0
    meshes = load_background_asset_meshes(str(result["manifest_path"]))
    assert [(name, path.name) for name, path in meshes] == [("depth_background", "depth_background.glb")]


def test_generate_depth_background_mesh_assets_drops_large_depth_discontinuity_faces(tmp_path: Path) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :] = (80, 100, 120)
    rgb_path = tmp_path / "clean_target_rgb.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth = np.ones((8, 8), dtype=np.float32)
    depth[:, 4:] = 8.0
    depth_path = tmp_path / "clean_target_depth.npy"
    np.save(depth_path, depth)

    result = generate_depth_background_mesh_assets(
        clean_rgb_path=rgb_path,
        depth_path=depth_path,
        output_dir=tmp_path / "depth_background",
        camera={
            "fx": 8.0,
            "fy": 8.0,
            "cx": 4.0,
            "cy": 4.0,
            "R": np.eye(3).tolist(),
            "t": [0.0, 0.0, 0.0],
        },
        grid_stride=1,
        target_frame_id=3,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    mesh = trimesh.load(manifest["assets"]["depth_background_glb"], force="mesh")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    face_depths = vertices[faces, 2]
    assert not np.any((np.min(face_depths, axis=1) < 2.0) & (np.max(face_depths, axis=1) > 6.0))
    assert manifest["quality"]["discontinuity_faces_removed"] > 0


def test_load_background_asset_meshes_prefers_depth_background_over_tabletop_proxy(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    tabletop = assets / "tabletop_background.obj"
    tabletop.write_text("o tabletop\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    depth_background = assets / "depth_background.glb"
    depth_background.write_bytes(b"glb")
    manifest = assets / "background_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "guanwu.target_frame_background_assets.tabletop.v2",
                "assets": {
                    "tabletop_mesh": str(tabletop),
                    "task_background_mesh": str(tabletop),
                    "depth_background_glb": str(depth_background),
                },
            }
        ),
        encoding="utf-8",
    )

    meshes = load_background_asset_meshes(str(manifest))

    assert [(name, path.name) for name, path in meshes] == [("depth_background", "depth_background.glb")]


def test_load_background_asset_meshes_uses_road_clean_depth_background_directly(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    clean_rgb = assets / "clean_target_rgb.png"
    Image.fromarray(np.full((24, 32, 3), 90, dtype=np.uint8)).save(clean_rgb)
    depth_background = assets / "depth_background.glb"
    depth_background.write_bytes(b"glb")
    manifest = assets / "background_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_frame_id": 1,
                "assets": {
                    "clean_rgb": str(clean_rgb),
                    "depth_background_glb": str(depth_background),
                },
                "quality": {"background_mode": "road_clean_background"},
            }
        ),
        encoding="utf-8",
    )
    meshes = load_background_asset_meshes(str(manifest))

    assert [(name, path.name) for name, path in meshes] == [("depth_background", "depth_background.glb")]


def test_generate_target_frame_background_assets_prefers_clean_depth_estimator(tmp_path: Path) -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :] = (88, 96, 104)
    mask = np.zeros((24, 32), dtype=bool)
    mask[12:21, 8:22] = True
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 3,
                        "detections": str(
                            _write_frame(
                                tmp_path / "frame_000003",
                                3,
                                rgb,
                                [_mask_instance("car_1", "car", mask, [8, 12, 22, 21])],
                            )
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    camera_trajectory = tmp_path / "camera_trajectory.json"
    camera_trajectory.write_text(
        json.dumps(
            [
                {
                    "frame_id": 3,
                    "K": [[28.0, 0.0, 16.0], [0.0, 28.0, 12.0], [0.0, 0.0, 1.0]],
                    "R": np.eye(3).tolist(),
                    "t": [0.0, 0.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    wildgs_depth_dir = tmp_path / "wildgs_depth"
    wildgs_depth_dir.mkdir()
    np.save(wildgs_depth_dir / "00003.npy", np.full((24, 32), 6.0, dtype=np.float32))
    external_depth = tmp_path / "external_depth.npy"
    np.save(external_depth, np.full((24, 32), 9.0, dtype=np.float32))

    def estimate(clean_rgb_path: Path) -> dict:
        assert clean_rgb_path.name == "clean_target_rgb.png"
        return {
            "depth_path": external_depth,
            "source": "depth_anything3_clean_rgb",
            "quality": {"depth_service": "fake_depth_anything3"},
        }

    result = generate_target_frame_background_assets(
        summary_path=summary_path,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        depth_maps_dir=wildgs_depth_dir,
        camera_trajectory_path=camera_trajectory,
        clean_depth_estimator=estimate,
        grid_stride=4,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "guanwu.target_frame_background_assets.road_clean_depth.v2"
    assert manifest["quality"]["depth_background_source"] == "depth_anything3_clean_rgb"
    assert manifest["quality"]["depth_service"] == "fake_depth_anything3"
    depth = np.load(manifest["assets"]["clean_depth"])
    assert float(depth[0, 0]) == 9.0
    assert manifest["quality"]["depth_calibration_source"] == "da3_metric_direct"
    assert load_background_asset_meshes(result["manifest_path"])[0][0] == "depth_background"


def test_clean_depth_estimator_depth_uses_da3_metric_directly_without_wildgs_calibration(tmp_path: Path) -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :] = (92, 100, 108)
    mask = np.zeros((24, 32), dtype=bool)
    mask[12:21, 8:22] = True
    road = np.zeros((24, 32), dtype=bool)
    road[10:24, 4:28] = True
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 3,
                        "detections": str(
                            _write_frame(
                                tmp_path / "frame_000003",
                                3,
                                rgb,
                                [
                                    _mask_instance("road_1", "asphalt road", road, [4, 10, 28, 24]),
                                    _mask_instance("car_1", "car", mask, [8, 12, 22, 21]),
                                ],
                            )
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    camera_trajectory = tmp_path / "camera_trajectory.json"
    camera_trajectory.write_text(
        json.dumps(
            [
                {
                    "frame_id": 3,
                    "K": [[28.0, 0.0, 16.0], [0.0, 28.0, 12.0], [0.0, 0.0, 1.0]],
                    "R": np.eye(3).tolist(),
                    "t": [0.0, 0.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    yy, xx = np.mgrid[0:24, 0:32]
    external_depth_values = (0.5 + xx * 0.01 + yy * 0.02).astype(np.float32)
    wildgs_metric_depth = (external_depth_values * 7.0 + 2.5).astype(np.float32)
    wildgs_depth_dir = tmp_path / "wildgs_depth"
    wildgs_depth_dir.mkdir()
    np.save(wildgs_depth_dir / "00003.npy", wildgs_metric_depth)
    external_depth = tmp_path / "external_depth.npy"
    np.save(external_depth, external_depth_values)

    result = generate_target_frame_background_assets(
        summary_path=summary_path,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        depth_maps_dir=wildgs_depth_dir,
        camera_trajectory_path=camera_trajectory,
        clean_depth_estimator=lambda _path: {
            "depth_path": external_depth,
            "source": "depth_anything3_clean_rgb",
        },
        grid_stride=4,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean_depth = np.load(manifest["assets"]["clean_depth"])
    assert manifest["quality"]["depth_background_source"] == "depth_anything3_clean_rgb"
    assert manifest["quality"]["depth_calibration_source"] == "da3_metric_direct"
    assert np.allclose(clean_depth, external_depth_values)
