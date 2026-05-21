from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import cv2
import numpy as np

from guanwu.video.features.spatial.scene_background_assets import (
    _fill_low_candidate_dynamic_regions,
    build_dynamic_mask,
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


def test_build_dynamic_mask_uses_smaller_expansion_for_tiny_objects() -> None:
    tiny = np.zeros((48, 80), dtype=bool)
    tiny[12:16, 20:26] = True
    detections = {"instances": [_mask_instance("tiny_car", "car", tiny, [20, 12, 26, 16])]}

    dynamic = build_dynamic_mask(detections, (48, 80), foreground_expand_px=8, shadow_expand_px=0)

    assert dynamic[12, 16]
    assert not dynamic[12, 12]


def test_generate_target_frame_background_assets_writes_split_meshes_and_manifest(tmp_path: Path) -> None:
    frames = []
    object_mask = np.zeros((36, 64), dtype=bool)
    object_mask[14:24, 24:38] = True
    for frame_idx, color in [(1, (60, 80, 100)), (2, (90, 110, 130)), (3, (120, 140, 160))]:
        rgb = np.zeros((36, 64, 3), dtype=np.uint8)
        rgb[:, :] = color
        rgb[20:, :] = (80 + frame_idx * 10, 80 + frame_idx * 10, 80 + frame_idx * 10)
        rgb[object_mask] = (220, 20, 20)
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [_mask_instance("obj_car", "car", object_mask, [24, 14, 38, 24])],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    road_geometry = tmp_path / "road_geometry.json"
    road_geometry.write_text(
        json.dumps(
            {
                "available": True,
                "keyframe_planes": [
                    {"frame_id": 3, "normal_world": [0.0, 1.0, 0.0], "offset": 0.0, "quality": {"inlier_ratio": 0.9}}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        road_geometry_path=road_geometry,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == "guanwu.target_frame_background_assets.v1"
    assert manifest["target_frame_id"] == 3
    for key in ("road_mesh", "structures_mesh", "far_mesh", "clean_rgb", "dynamic_mask"):
        assert Path(manifest["assets"][key]).exists()
    assert manifest["quality"]["source_frame_count"] == 3
    assert manifest["quality"]["target_dynamic_fraction"] > 0.0


def test_generate_background_replaces_target_frame_vehicle_pixels_with_donor_road(tmp_path: Path) -> None:
    frames = []
    object_mask = np.zeros((36, 64), dtype=bool)
    object_mask[18:28, 20:36] = True
    road_color = np.array([96, 96, 96], dtype=np.uint8)
    vehicle_color = np.array([230, 20, 20], dtype=np.uint8)
    for frame_idx in [1, 2, 3, 4, 5]:
        rgb = np.zeros((36, 64, 3), dtype=np.uint8)
        rgb[:] = road_color
        instances = []
        if frame_idx == 3:
            rgb[object_mask] = vehicle_color
            instances = [_mask_instance("obj_car", "car", object_mask, [20, 18, 36, 28])]
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
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    replaced = clean[object_mask].mean(axis=0)
    assert np.linalg.norm(replaced - road_color.astype(np.float32)) < 8.0
    assert np.linalg.norm(replaced - vehicle_color.astype(np.float32)) > 120.0


def test_generate_background_does_not_preserve_unmasked_target_frame_vehicle(tmp_path: Path) -> None:
    frames = []
    object_region = np.zeros((36, 64), dtype=bool)
    object_region[6:14, 44:56] = True
    road_color = np.array([104, 104, 104], dtype=np.uint8)
    vehicle_color = np.array([235, 235, 235], dtype=np.uint8)
    for frame_idx in [1, 2, 3, 4, 5]:
        rgb = np.zeros((36, 64, 3), dtype=np.uint8)
        rgb[:] = road_color
        if frame_idx == 3:
            rgb[object_region] = vehicle_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(_write_frame(tmp_path / f"frame_{frame_idx:06d}", frame_idx, rgb, [])),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    repaired = clean[object_region].mean(axis=0)
    assert np.linalg.norm(repaired - road_color.astype(np.float32)) < 8.0
    assert np.linalg.norm(repaired - vehicle_color.astype(np.float32)) > 150.0


def test_object_index_masks_small_vehicle_without_detection_mask(tmp_path: Path) -> None:
    frames = []
    object_region = np.zeros((36, 64), dtype=bool)
    object_region[5:11, 44:56] = True
    road_color = np.array([90, 90, 90], dtype=np.uint8)
    vehicle_color = np.array([240, 240, 240], dtype=np.uint8)
    for frame_idx in [1, 2, 3, 4, 5]:
        rgb = np.zeros((36, 64, 3), dtype=np.uint8)
        rgb[:] = road_color
        if frame_idx == 3:
            rgb[object_region] = vehicle_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(_write_frame(tmp_path / f"frame_{frame_idx:06d}", frame_idx, rgb, [])),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(
        json.dumps(
            [
                {
                    "object_id": "small_car",
                    "label": "car",
                    "frames": [{"frame_idx": 3, "bbox": [44.0, 5.0, 56.0, 11.0]}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=3,
        object_index_path=objects,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    dynamic = cv2.imread(manifest["assets"]["dynamic_mask"], cv2.IMREAD_GRAYSCALE)
    assert dynamic[7, 48] > 0
    assert np.linalg.norm(clean[object_region].mean(axis=0) - road_color.astype(np.float32)) < 8.0


def test_low_candidate_dynamic_region_is_filled_from_neighbors(tmp_path: Path) -> None:
    frames = []
    object_mask = np.zeros((36, 64), dtype=bool)
    object_mask[8:14, 44:56] = True
    road_color = np.array([112, 112, 112], dtype=np.uint8)
    vehicle_color = np.array([245, 245, 245], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((36, 64, 3), dtype=np.uint8)
        rgb[:] = road_color
        rgb[object_mask] = vehicle_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [_mask_instance("obj_car", "car", object_mask, [44, 8, 56, 14])],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=2,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    repaired = clean[object_mask].mean(axis=0)
    assert np.linalg.norm(repaired - road_color.astype(np.float32)) < 16.0
    assert np.linalg.norm(repaired - vehicle_color.astype(np.float32)) > 150.0


def test_low_candidate_far_region_uses_horizontal_neighbor_texture(tmp_path: Path) -> None:
    frames = []
    object_mask = np.zeros((36, 64), dtype=bool)
    object_mask[1:8, 4:12] = True
    left_texture = np.array([70, 84, 92], dtype=np.uint8)
    right_texture = np.array([78, 88, 96], dtype=np.uint8)
    vehicle_color = np.array([240, 240, 240], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((36, 64, 3), dtype=np.uint8)
        rgb[:, :20] = left_texture
        rgb[:, 20:] = right_texture
        rgb[object_mask] = vehicle_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [_mask_instance("obj_car", "car", object_mask, [4, 1, 12, 8])],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=2,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    repaired = clean[object_mask].mean(axis=0)
    assert np.linalg.norm(repaired - left_texture.astype(np.float32)) < 18.0
    assert np.linalg.norm(repaired - vehicle_color.astype(np.float32)) > 180.0


def test_low_candidate_fill_has_local_texture_variation(tmp_path: Path) -> None:
    frames = []
    object_mask = np.zeros((40, 72), dtype=bool)
    object_mask[8:18, 30:44] = True
    vehicle_color = np.array([240, 240, 240], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((40, 72, 3), dtype=np.uint8)
        for x in range(72):
            rgb[:, x] = (80 + x // 3, 88 + x // 4, 96 + x // 5)
        rgb[object_mask] = vehicle_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [_mask_instance("obj_car", "car", object_mask, [30, 8, 44, 18])],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=2,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    patch = clean[object_mask]
    assert float(patch.std()) > 1.0
    assert np.linalg.norm(patch.mean(axis=0) - vehicle_color.astype(np.float32)) > 180.0


def test_low_candidate_fill_avoids_sharp_inpaint_spikes(tmp_path: Path) -> None:
    frames = []
    object_mask = np.zeros((40, 72), dtype=bool)
    object_mask[6:18, 8:26] = True
    vehicle_color = np.array([245, 245, 245], dtype=np.uint8)
    for frame_idx in [1, 2, 3]:
        rgb = np.zeros((40, 72, 3), dtype=np.uint8)
        for x in range(72):
            rgb[:, x] = (84 + x // 4, 92 + x // 5, 98 + x // 6)
        rgb[object_mask] = vehicle_color
        frames.append(
            {
                "frame_idx": frame_idx,
                "detections": str(
                    _write_frame(
                        tmp_path / f"frame_{frame_idx:06d}",
                        frame_idx,
                        rgb,
                        [_mask_instance("obj_car", "car", object_mask, [8, 6, 26, 18])],
                    )
                ),
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"frames": frames}), encoding="utf-8")

    result = generate_target_frame_background_assets(
        summary_path=summary,
        output_dir=tmp_path / "background_assets",
        target_frame_id=2,
        grid_stride=8,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    clean = cv2.cvtColor(cv2.imread(manifest["assets"]["clean_rgb"]), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = 8, 6, 26, 18
    patch = clean[y1:y2, x1:x2].astype(np.float32)
    grad_x = np.abs(np.diff(patch, axis=1)).max()
    grad_y = np.abs(np.diff(patch, axis=0)).max()
    assert max(float(grad_x), float(grad_y)) < 35.0
    assert np.linalg.norm(patch.mean(axis=(0, 1)) - vehicle_color.astype(np.float32)) > 180.0


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


def test_load_background_asset_meshes_prefers_manifest_split_order(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("road_mesh.obj", "structures_mesh.obj", "far_mesh.obj"):
        (assets / name).write_text("o x\nv 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n", encoding="utf-8")
    manifest = assets / "background_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "guanwu.target_frame_background_assets.v1",
                "assets": {
                    "road_mesh": str(assets / "road_mesh.obj"),
                    "structures_mesh": str(assets / "structures_mesh.obj"),
                    "far_mesh": str(assets / "far_mesh.obj"),
                },
            }
        ),
        encoding="utf-8",
    )

    meshes = load_background_asset_meshes(str(manifest))

    assert [(name, path.name) for name, path in meshes] == [
        ("road", "road_mesh.obj"),
        ("structures", "structures_mesh.obj"),
        ("far", "far_mesh.obj"),
    ]


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
    assert manifest["schema"] == "guanwu.target_frame_background_assets.v2"
    assert manifest["quality"]["depth_background_source"] == "depth_anything3_clean_rgb"
    assert manifest["quality"]["depth_service"] == "fake_depth_anything3"
    depth = np.load(manifest["assets"]["clean_depth"])
    assert float(depth[0, 0]) == 6.0
    assert manifest["quality"]["depth_calibration_source"] == "wildgs_metric_depth_affine"
    assert load_background_asset_meshes(result["manifest_path"])[0][0] == "depth_background"


def test_clean_depth_estimator_depth_is_calibrated_to_wildgs_metric_depth(tmp_path: Path) -> None:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :] = (92, 100, 108)
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
    calibrated = np.load(manifest["assets"]["clean_depth"])
    assert manifest["quality"]["depth_background_source"] == "depth_anything3_clean_rgb"
    assert manifest["quality"]["depth_calibration_source"] == "wildgs_metric_depth_affine"
    road_mask = cv2.imread(manifest["assets"]["road_mask"], cv2.IMREAD_GRAYSCALE) > 0
    assert float(np.median(np.abs(calibrated[road_mask] - wildgs_metric_depth[road_mask]))) < 0.05
