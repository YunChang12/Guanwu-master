from __future__ import annotations

import sys
import types
import zipfile

import numpy as np
import pytest

from guanwu.adapters.robotwin2_replay import (
    MeshExtractionError,
    TextureExportContext,
    _TimedCaptureSampler,
    _actor_metadata,
    _apply_sapien_pose,
    _build_generated_view_specs,
    _camera_color_to_uint8,
    _camera_sample_for_time,
    _ensure_multiview_video,
    _extract_entity_mesh,
    _extract_material_info,
    _get_robot_links,
    _load_mesh_geometry,
    _load_mesh_geometry_parts,
    _load_urdf_link_metadata,
    _load_urdf_link_meshes,
    _plane_mesh_from_scale,
    _mesh_extent_is_reasonable,
    _record_actor_pose_frame,
    _scene_actor_records,
    _stack_pose_history,
    _usd_child_name,
    _write_replay_usdc,
    _write_replay_usdz,
)


class _FakeArticulation:
    def __init__(self, links):
        self._links = links

    def get_links(self):
        return list(self._links)


class _FakeRobotWrapper:
    def __init__(self, left_entity=None, right_entity=None, robot=None):
        self.left_entity = left_entity
        self.right_entity = right_entity
        self.robot = robot


class _FakeTaskEnv:
    def __init__(self, robot):
        self.robot = robot


def test_get_robot_links_prefers_robotwin_entities_and_deduplicates_shared_articulation():
    shared_links = [object(), object(), object()]
    articulation = _FakeArticulation(shared_links)
    task_env = _FakeTaskEnv(
        _FakeRobotWrapper(
            left_entity=articulation,
            right_entity=articulation,
        )
    )

    links = _get_robot_links(task_env)

    assert links == shared_links


def test_ensure_multiview_video_overwrites_stale_copy(tmp_path):
    renders_dir = tmp_path / "renders"
    renders_dir.mkdir()
    (renders_dir / "composite.mp4").write_bytes(b"fresh")
    (renders_dir / "multi_view.mp4").write_bytes(b"stale")

    _ensure_multiview_video(renders_dir)

    assert (renders_dir / "multi_view.mp4").read_bytes() == b"fresh"


def test_extract_entity_mesh_uses_link_entity_render_owner(monkeypatch):
    class FakeRenderBodyComponent:
        pass

    fake_sapien = types.SimpleNamespace(
        render=types.SimpleNamespace(RenderBodyComponent=FakeRenderBodyComponent)
    )
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)

    mesh = types.SimpleNamespace(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices=[0, 1, 2],
    )
    material = types.SimpleNamespace(base_color=(0.1, 0.2, 0.3, 1.0))
    part = types.SimpleNamespace(mesh=mesh, material=material)
    render_shape = types.SimpleNamespace(parts=[part])
    render_body = types.SimpleNamespace(render_shapes=[render_shape])

    class FakeEntity:
        def find_component_by_type(self, _component_type):
            return render_body

    fake_link = types.SimpleNamespace(entity=FakeEntity())

    mesh_info = _extract_entity_mesh(fake_link)

    assert mesh_info is not None
    assert mesh_info["verts"].shape == (3, 3)
    assert mesh_info["faces"].shape == (1, 3)
    assert mesh_info["color"] == (0.1, 0.2, 0.3)


def test_extract_entity_mesh_builds_box_primitive_and_applies_local_pose(monkeypatch):
    class FakeRenderBodyComponent:
        pass

    fake_sapien = types.SimpleNamespace(
        render=types.SimpleNamespace(RenderBodyComponent=FakeRenderBodyComponent)
    )
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)

    material = types.SimpleNamespace(base_color=(0.4, 0.5, 0.6, 1.0))
    part = types.SimpleNamespace(material=material)

    class RenderShapeBox:
        def __init__(self):
            self.parts = [part]
            self.half_size = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            self.local_pose = types.SimpleNamespace(
                p=np.array([10.0, 20.0, 30.0], dtype=np.float64),
                q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )

    render_body = types.SimpleNamespace(render_shapes=[RenderShapeBox()])

    class FakeEntity:
        def find_component_by_type(self, _component_type):
            return render_body

    mesh_info = _extract_entity_mesh(FakeEntity())

    assert mesh_info is not None
    assert mesh_info["faces"].shape == (12, 3)
    assert np.allclose(mesh_info["verts"].min(axis=0), [9.0, 18.0, 27.0])
    assert np.allclose(mesh_info["verts"].max(axis=0), [11.0, 22.0, 33.0])
    assert mesh_info["color"] == (0.4, 0.5, 0.6)


def test_extract_entity_mesh_handles_sapien_triangle_mesh_part(monkeypatch):
    class FakeRenderBodyComponent:
        pass

    fake_sapien = types.SimpleNamespace(
        render=types.SimpleNamespace(RenderBodyComponent=FakeRenderBodyComponent)
    )
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)

    class RenderShapeTriangleMeshPart:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        material = types.SimpleNamespace(base_color=(0.2, 0.3, 0.4, 1.0))

        def get_vertex_uv(self):
            return np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )

    class RenderShapeTriangleMesh:
        parts = [RenderShapeTriangleMeshPart()]
        scale = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        local_pose = types.SimpleNamespace(
            p=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

    render_body = types.SimpleNamespace(render_shapes=[RenderShapeTriangleMesh()])

    class FakeEntity:
        def find_component_by_type(self, _component_type):
            return render_body

    mesh_info = _extract_entity_mesh(FakeEntity())

    assert mesh_info["faces"].shape == (1, 3)
    assert np.allclose(mesh_info["parts"][0]["uvs"], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(mesh_info["verts"].min(axis=0), [1.0, 1.0, 1.0])
    assert np.allclose(mesh_info["verts"].max(axis=0), [3.0, 7.0, 1.0])
    assert mesh_info["color"] == (0.2, 0.3, 0.4)


def test_extract_entity_mesh_preserves_per_part_materials(monkeypatch):
    class FakeRenderBodyComponent:
        pass

    fake_sapien = types.SimpleNamespace(
        render=types.SimpleNamespace(RenderBodyComponent=FakeRenderBodyComponent)
    )
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)

    mesh_a = types.SimpleNamespace(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices=[0, 1, 2],
    )
    mesh_b = types.SimpleNamespace(
        vertices=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        indices=[0, 1, 2],
    )
    part_a = types.SimpleNamespace(
        mesh=mesh_a,
        material=types.SimpleNamespace(base_color=(0.1, 0.2, 0.3, 0.8), roughness=0.4),
    )
    part_b = types.SimpleNamespace(
        mesh=mesh_b,
        material=types.SimpleNamespace(base_color=(0.8, 0.7, 0.6, 1.0), metallic=0.2),
    )
    render_body = types.SimpleNamespace(
        render_shapes=[
            types.SimpleNamespace(parts=[part_a]),
            types.SimpleNamespace(parts=[part_b]),
        ]
    )

    class FakeEntity:
        def find_component_by_type(self, _component_type):
            return render_body

    mesh_info = _extract_entity_mesh(FakeEntity())

    assert len(mesh_info["parts"]) == 2
    assert mesh_info["parts"][0]["color"] == (0.1, 0.2, 0.3)
    assert mesh_info["parts"][0]["material"]["alpha"] == 0.8
    assert mesh_info["parts"][0]["material"]["roughness"] == 0.4
    assert mesh_info["parts"][1]["color"] == (0.8, 0.7, 0.6)
    assert mesh_info["parts"][1]["material"]["metallic"] == 0.2


def test_extract_entity_mesh_prefers_file_backed_visual_mesh(monkeypatch, tmp_path):
    class FakeRenderBodyComponent:
        pass

    fake_sapien = types.SimpleNamespace(
        render=types.SimpleNamespace(RenderBodyComponent=FakeRenderBodyComponent)
    )
    monkeypatch.setitem(sys.modules, "sapien", fake_sapien)
    monkeypatch.chdir(tmp_path)
    source_mesh = tmp_path / "assets" / "objects" / "bottle.glb"
    source_mesh.parent.mkdir(parents=True)
    source_mesh.write_bytes(b"glb")

    source_verts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    source_faces = np.array([[0, 1, 2]], dtype=np.int32)
    source_uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_load_mesh_geometry_parts(path, **_kwargs):
        assert path == source_mesh.resolve()
        return [
            {
                "verts": source_verts,
                "faces": source_faces,
                "uvs": source_uvs,
                "color": (0.2, 0.7, 0.3),
                "material": {
                    "base_color": (0.2, 0.7, 0.3),
                    "textures": {"base_color_texture": {"file": "textures/bottle.png"}},
                    "properties": {},
                },
                "source": {"source_mesh": str(path)},
            }
        ]

    monkeypatch.setattr(
        "guanwu.adapters.robotwin2_replay._load_mesh_geometry_parts",
        fake_load_mesh_geometry_parts,
    )

    runtime_mesh = types.SimpleNamespace(
        vertices=[[9.0, 9.0, 9.0], [9.0, 8.0, 9.0], [8.0, 9.0, 9.0]],
        indices=[0, 1, 2],
    )
    runtime_part = types.SimpleNamespace(
        mesh=runtime_mesh,
        material=types.SimpleNamespace(base_color=(1.0, 0.0, 0.0, 1.0)),
    )
    render_shape = types.SimpleNamespace(
        filename="assets/objects/bottle.glb",
        scale=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        parts=[runtime_part],
    )
    render_body = types.SimpleNamespace(render_shapes=[render_shape])

    class FakeEntity:
        def find_component_by_type(self, _component_type):
            return render_body

    mesh_info = _extract_entity_mesh(FakeEntity())

    assert np.allclose(mesh_info["verts"], source_verts * 2.0)
    assert mesh_info["color"] == (0.2, 0.7, 0.3)
    assert np.allclose(mesh_info["parts"][0]["uvs"], source_uvs)
    assert mesh_info["parts"][0]["source"]["source_mesh_loader"] == "trimesh_original_visual"


def test_extract_material_info_reads_sapien_getters_and_embedded_texture_color():
    class FakeTexture:
        filename = ""
        format = "R8G8B8A8Unorm"
        width = 2
        height = 2

        def download(self):
            return np.array(
                [
                    [[0, 255, 0, 255], [0, 255, 0, 255]],
                    [[0, 128, 0, 255], [0, 128, 0, 255]],
                ],
                dtype=np.uint8,
            )

    class FakeMaterial:
        roughness = 0.25

        def get_base_color(self):
            return [1.0, 0.5, 1.0, 0.75]

        def get_base_color_texture(self):
            return FakeTexture()

    info = _extract_material_info(FakeMaterial())

    assert np.allclose(info["base_color"], (0.0, 0.3754902, 0.0))
    assert info["alpha"] == 0.75
    assert info["roughness"] == 0.25
    assert info["texture_representative_color"] == pytest.approx((0.0, 0.7509804, 0.0))
    assert info["textures"]["base_color_texture"]["format"] == "R8G8B8A8Unorm"


def test_extract_material_info_exports_embedded_texture_file(tmp_path):
    class FakeTexture:
        filename = ""
        width = 2
        height = 2

        def download(self):
            return np.array(
                [
                    [[255, 0, 0, 255], [0, 255, 0, 255]],
                    [[0, 0, 255, 255], [255, 255, 255, 255]],
                ],
                dtype=np.uint8,
            )

    class FakeMaterial:
        base_color = (1.0, 1.0, 1.0, 1.0)
        base_color_texture = FakeTexture()

    exporter = TextureExportContext(tmp_path)
    info = _extract_material_info(
        FakeMaterial(),
        texture_exporter=exporter,
        texture_prefix="object_part",
    )

    texture_info = info["textures"]["base_color_texture"]
    assert texture_info["file"].startswith("textures/object_part_base_color_texture")
    assert (tmp_path / texture_info["file"]).is_file()


def test_plane_mesh_from_scale_matches_sapien_plane_local_frame():
    verts, faces = _plane_mesh_from_scale(np.array([10.0, 10.0, 10.0], dtype=np.float64))
    pose = types.SimpleNamespace(
        p=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        q=np.array([0.7071068, 0.0, -0.7071068, 0.0], dtype=np.float64),
    )

    world_verts = _apply_sapien_pose(verts, pose)

    assert faces.shape == (2, 3)
    assert np.allclose(world_verts[:, 2], 0.0, atol=1e-5)
    assert np.isclose(world_verts[:, 0].min(), -10.0, atol=1e-5)
    assert np.isclose(world_verts[:, 0].max(), 10.0, atol=1e-5)
    assert np.isclose(world_verts[:, 1].min(), -10.0, atol=1e-5)
    assert np.isclose(world_verts[:, 1].max(), 10.0, atol=1e-5)


def test_timed_capture_sampler_samples_at_requested_rate():
    captured: list[float] = []
    sampler = _TimedCaptureSampler(
        capture_hz=2.0,
        scene_dt=0.1,
        capture_fn=lambda: captured.append(round(sampler.current_time, 3)),
    )

    sampler.capture_initial()
    for _ in range(10):
        sampler.on_step()
    sampler.capture_final()

    assert captured == [0.0, 0.5, 1.0]


def test_generated_view_specs_expand_static_and_trajectory_without_usdc_duplication():
    specs = _build_generated_view_specs(
        scene_bounds={
            "center_W_m": (0.0, -0.4, 0.7),
            "extent_m": (1.0, 1.0, 1.0),
            "radius_m": 1.0,
            "up_axis": "Z",
        },
        frame_count=4,
        fps=30.0,
        generated_static_camera_count=1,
        generated_trajectory_camera_count=1,
        camera_seed=11,
        width_px=320,
        height_px=240,
    )

    assert [spec["view_id"] for spec in specs] == ["view_000", "view_001"]
    assert specs[0]["camera"]["mode"] == "fixed_orbit"
    assert specs[1]["camera"]["mode"] == "camera_trajectory"
    assert len(specs[1]["camera"]["keyframes"]) == 4
    assert all("scene.usdc" not in spec for spec in specs)
    assert specs[0]["uses_hdf5_rgb"] is False


def test_camera_sample_for_time_interpolates_trajectory_keyframes():
    sample = _camera_sample_for_time(
        {
            "mode": "camera_trajectory",
            "keyframes": [
                {
                    "time_code": 0.0,
                    "eye_W_m": (0.0, 0.0, 0.0),
                    "target_W_m": (0.0, 0.0, 1.0),
                    "up_W": (0.0, 0.0, 1.0),
                },
                {
                    "time_code": 2.0,
                    "eye_W_m": (2.0, 0.0, 0.0),
                    "target_W_m": (0.0, 2.0, 1.0),
                    "up_W": (0.0, 0.0, 1.0),
                },
            ],
        },
        1.0,
    )

    assert sample["eye_W_m"] == pytest.approx((1.0, 0.0, 0.0))
    assert sample["target_W_m"] == pytest.approx((0.0, 1.0, 1.0))


def test_camera_color_to_uint8_handles_float_rgba():
    image = np.array(
        [
            [[0.0, 0.5, 1.0, 1.0], [2.0, -1.0, 0.25, 1.0]],
        ],
        dtype=np.float32,
    )

    rgb = _camera_color_to_uint8(image)

    assert rgb.dtype == np.uint8
    assert rgb.tolist() == [[[0, 127, 255], [255, 0, 63]]]


def test_mesh_extent_is_reasonable_rejects_absurd_geometry():
    huge = np.array(
        [[-600.0, 0.0, 0.0], [700.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    normal = np.array(
        [[-0.1, -0.1, -0.1], [0.2, 0.1, 0.3], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    assert _mesh_extent_is_reasonable(normal) is True
    assert _mesh_extent_is_reasonable(huge) is False


def test_usd_child_name_handles_empty_invalid_and_duplicate_names():
    used: set[str] = set()

    assert _usd_child_name("", "Actor_0", used) == "Actor_0"
    assert _usd_child_name("123 bad-name", "Actor_1", used) == "_123_bad_name"
    assert _usd_child_name("123 bad-name", "Actor_2", used) == "_123_bad_name_2"


def test_scene_actor_records_keep_duplicate_actor_tracks_separate():
    class FakeActor:
        def __init__(self, name, positions):
            self._name = name
            self._positions = list(positions)
            self._frame = 0

        def get_name(self):
            return self._name

        def get_pose(self):
            return types.SimpleNamespace(
                p=np.asarray(self._positions[self._frame], dtype=np.float64),
                q=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )

        def advance(self):
            self._frame += 1

    first = FakeActor("075_bread", [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    second = FakeActor("075_bread", [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])

    records = _scene_actor_records([first, second])
    states = {actor_key: [] for actor_key, *_rest in records}
    _record_actor_pose_frame(records, states)
    first.advance()
    second.advance()
    _record_actor_pose_frame(records, states)

    stacked = _stack_pose_history(states)
    metadata = _actor_metadata(records)

    assert [actor_key for actor_key, *_rest in records] == ["075_bread", "075_bread_2"]
    assert np.allclose(stacked["075_bread"][:, :3], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert np.allclose(stacked["075_bread_2"][:, :3], [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    assert metadata["075_bread_2"] == {
        "source_name": "075_bread",
        "duplicate_index": 2,
    }


def test_write_replay_usdc_authors_hdf5_cameras(tmp_path):
    pytest.importorskip("pxr", reason="usd-core required for USD structure checks")
    from pxr import Usd, UsdGeom

    path = tmp_path / "scene.usdc"
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, 3] = [1.0, 2.0, 3.0]
    intrinsic = np.array(
        [
            [358.64218, 0.0, 160.0],
            [0.0, 358.64218, 120.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    _write_replay_usdc(
        path,
        {
            "T": 2,
            "fps": 30.0,
            "actor_states": {},
            "actor_meshes": {},
            "robot_link_states": {},
            "robot_link_meshes": {},
            "camera_observations": {
                "front_camera": {
                    "cam2world_gl": np.stack([pose0, pose1], axis=0),
                    "intrinsic_cv": np.stack([intrinsic, intrinsic], axis=0),
                    "image_size": (320, 240),
                }
            },
        },
    )

    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/Scene/Cameras/front_camera")
    assert prim.IsA(UsdGeom.Camera)
    assert prim.GetAttribute("robotwin:imageWidth").Get() == 320
    assert prim.GetAttribute("robotwin:imageHeight").Get() == 240
    assert prim.GetAttribute("robotwin:sourceCameraName").Get() == "front_camera"

    world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(1)
    translation = world_xf.ExtractTranslation()
    assert np.allclose([translation[0], translation[1], translation[2]], [1.0, 2.0, 3.0])


def test_write_replay_usdc_authors_per_part_preview_materials(tmp_path):
    pytest.importorskip("pxr", reason="usd-core required for USD structure checks")
    from pxr import Usd, UsdGeom, UsdShade

    path = tmp_path / "scene.usdc"
    poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    verts0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    verts1 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    _write_replay_usdc(
        path,
        {
            "T": 1,
            "fps": 30.0,
            "actor_states": {"object": poses},
            "actor_meshes": {
                "object": {
                    "verts": np.concatenate([verts0, verts1], axis=0),
                    "faces": np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
                    "color": (0.8, 0.7, 0.6),
                    "parts": [
                        {
                            "verts": verts0,
                            "faces": faces,
                            "uvs": np.array(
                                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                                dtype=np.float32,
                            ),
                            "color": (0.1, 0.2, 0.3),
                            "material": {
                                "base_color": (0.1, 0.2, 0.3),
                                "alpha": 0.75,
                                "roughness": 0.25,
                                "properties": {"source": "part0"},
                                "textures": {"base_color_texture": "textures/albedo.png"},
                            },
                            "source": {"part": 0},
                        },
                        {
                            "verts": verts1,
                            "faces": faces,
                            "color": (0.8, 0.7, 0.6),
                            "material": {
                                "base_color": (0.8, 0.7, 0.6),
                                "metallic": 0.5,
                                "properties": {"source": "part1"},
                                "textures": {},
                            },
                            "source": {"part": 1},
                        },
                    ],
                }
            },
            "robot_link_states": {},
            "robot_link_meshes": {},
        },
    )

    stage = Usd.Stage.Open(str(path))
    parent = stage.GetPrimAtPath("/Scene/object")
    assert parent.IsA(UsdGeom.Xform)

    part0 = stage.GetPrimAtPath("/Scene/object/Part_000")
    assert part0.IsA(UsdGeom.Mesh)
    assert UsdGeom.Mesh(part0).GetDisplayColorAttr().Get()[0] == (0.1, 0.2, 0.3)

    material, _rel = UsdShade.MaterialBindingAPI(part0).ComputeBoundMaterial()
    assert material.GetPrim().IsValid()
    shader = UsdShade.Shader(
        stage.GetPrimAtPath(material.GetPath().AppendChild("PreviewSurface"))
    )
    assert shader.GetInput("opacity").Get() == 0.75
    assert shader.GetInput("roughness").Get() == 0.25
    assert "part0" in material.GetPrim().GetAttribute("robotwin:materialJson").Get()
    assert (
        material.GetPrim().GetAttribute("robotwin:baseColorTexture").Get().path
        == "textures/albedo.png"
    )


def test_write_replay_usdz_packages_texture_dependencies(tmp_path):
    pytest.importorskip("pxr", reason="usd-core required for USDZ packaging checks")

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    (textures_dir / "albedo.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05"
        b"\xfe\x02\xfeA\x89\x81\x9b\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    usdc_path = tmp_path / "scene.usdc"
    poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    _write_replay_usdc(
        usdc_path,
        {
            "T": 1,
            "fps": 30.0,
            "actor_states": {"object": poses},
            "actor_meshes": {
                "object": {
                    "verts": verts,
                    "faces": faces,
                    "color": (1.0, 1.0, 1.0),
                    "parts": [
                        {
                            "verts": verts,
                            "faces": faces,
                            "uvs": np.array(
                                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                                dtype=np.float32,
                            ),
                            "color": (1.0, 1.0, 1.0),
                            "material": {
                                "base_color": (1.0, 1.0, 1.0),
                                "properties": {},
                                "textures": {"base_color_texture": "textures/albedo.png"},
                            },
                            "source": {},
                        }
                    ],
                }
            },
            "robot_link_states": {},
            "robot_link_meshes": {},
        },
    )

    usdz_path = tmp_path / "scene.usdz"
    _write_replay_usdz(usdc_path, usdz_path)

    with zipfile.ZipFile(usdz_path) as archive:
        names = archive.namelist()
    assert "scene.usdc" in names
    assert any(name.endswith("albedo.png") for name in names)


def test_load_mesh_geometry_prefers_collada_scene_graph(monkeypatch, tmp_path):
    class FakePrimitive:
        def __init__(self):
            self.vertex = np.array(
                [[1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [1.0, 5.0, 3.0]],
                dtype=np.float64,
            )
            self.vertex_index = np.array([[0, 1, 2]], dtype=np.int32)

    class FakeBoundGeometry:
        def primitives(self):
            return [FakePrimitive()]

    class FakeScene:
        def objects(self, tipo):
            assert tipo == "geometry"
            return [FakeBoundGeometry()]

    class FakeColladaDoc:
        def __init__(self):
            self.scene = FakeScene()
            self.geometries = []

    fake_collada = types.SimpleNamespace(Collada=lambda _path: FakeColladaDoc())
    monkeypatch.setitem(sys.modules, "collada", fake_collada)

    def _unexpected_trimesh_load(*_args, **_kwargs):
        raise AssertionError("trimesh fallback should not be used for Collada")

    fake_trimesh = types.SimpleNamespace(load=_unexpected_trimesh_load)
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    mesh_path = tmp_path / "link.dae"
    mesh_path.write_text("placeholder")

    verts, faces = _load_mesh_geometry(mesh_path)  # type: ignore[misc]

    assert np.allclose(verts, [[1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [1.0, 5.0, 3.0]])
    assert np.array_equal(faces, [[0, 1, 2]])


def test_load_mesh_geometry_parts_preserves_trimesh_material_texture(monkeypatch, tmp_path):
    class FakeMaterial:
        name = "label"
        main_color = np.array([255, 255, 255, 255], dtype=np.uint8)
        baseColorTexture = np.array(
            [
                [[255, 0, 0, 255], [255, 0, 0, 255]],
                [[0, 0, 255, 255], [0, 0, 255, 255]],
            ],
            dtype=np.uint8,
        )

    class FakeVisual:
        material = FakeMaterial()
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    class FakeGeom:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        visual = FakeVisual()

    class FakeScene:
        geometry = {"geom": FakeGeom()}
        graph = None

    fake_trimesh = types.SimpleNamespace(load=lambda *_args, **_kwargs: FakeScene())
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    mesh_path = tmp_path / "textured.dae"
    mesh_path.write_text("placeholder")
    exporter = TextureExportContext(tmp_path)

    parts = _load_mesh_geometry_parts(
        mesh_path,
        texture_exporter=exporter,
        texture_prefix="robot_link",
    )

    assert len(parts) == 1
    assert parts[0]["uvs"].shape == (3, 2)
    texture_info = parts[0]["material"]["textures"]["base_color_texture"]
    assert texture_info["file"].startswith("textures/robot_link_000_baseColorTexture")
    assert (tmp_path / texture_info["file"]).is_file()
    assert parts[0]["material"]["texture_representative_color"] == pytest.approx(
        (0.5, 0.0, 0.5)
    )


def test_load_urdf_link_meshes_bakes_origin_and_scale(tmp_path, monkeypatch):
    urdf_path = tmp_path / "robot.urdf"
    mesh_path = tmp_path / "meshes" / "link1.dae"
    mesh_path.parent.mkdir()
    mesh_path.write_text("placeholder")
    urdf_path.write_text(
        """
        <robot name="test_robot">
          <link name="link1">
            <visual>
              <origin xyz="1 2 3" rpy="0 0 0" />
              <geometry>
                <mesh filename="meshes/link1.dae" scale="2 2 2" />
              </geometry>
            </visual>
          </link>
        </robot>
        """
    )

    monkeypatch.setattr(
        "guanwu.adapters.robotwin2_replay._load_mesh_geometry",
        lambda _path: (
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0, 1, 1]], dtype=np.int32),
        ),
    )

    mesh_map = _load_urdf_link_meshes(urdf_path)

    assert "link1" in mesh_map
    verts = mesh_map["link1"]["verts"]
    assert np.allclose(verts[0], [1.0, 2.0, 3.0])
    assert np.allclose(verts[1], [3.0, 2.0, 3.0])


def test_load_urdf_link_metadata_marks_frame_only_links(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """
        <robot name="test_robot">
          <link name="footprint" />
          <link name="inertial_link">
            <inertial>
              <mass value="1" />
              <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1" />
            </inertial>
          </link>
          <link name="link1">
            <visual>
              <geometry>
                <mesh filename="meshes/link1.dae" />
              </geometry>
            </visual>
          </link>
        </robot>
        """
    )

    metadata = _load_urdf_link_metadata(urdf_path)

    assert metadata["footprint"]["renderable"] is False
    assert metadata["footprint"]["semantic_role"] == "root_frame"
    assert metadata["inertial_link"]["renderable"] is False
    assert metadata["inertial_link"]["semantic_role"] == "inertial_frame"
    assert metadata["link1"]["renderable"] is True


def test_load_urdf_link_meshes_raises_on_anomalous_geometry(tmp_path, monkeypatch):
    urdf_path = tmp_path / "robot.urdf"
    mesh_path = tmp_path / "meshes" / "bad.dae"
    mesh_path.parent.mkdir()
    mesh_path.write_text("placeholder")
    urdf_path.write_text(
        """
        <robot name="test_robot">
          <link name="bad_link">
            <visual>
              <geometry>
                <mesh filename="meshes/bad.dae" />
              </geometry>
            </visual>
          </link>
        </robot>
        """
    )

    monkeypatch.setattr(
        "guanwu.adapters.robotwin2_replay._load_mesh_geometry",
        lambda _path: (
            np.array([[-600.0, 0.0, 0.0], [700.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
            np.array([[0, 1, 2]], dtype=np.int32),
        ),
    )

    with pytest.raises(MeshExtractionError, match="Anomalous robot mesh"):
        _load_urdf_link_meshes(urdf_path)
