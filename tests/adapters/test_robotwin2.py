"""Golden fixture tests for the RoboTwin 2.0 adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("h5py", reason="h5py required for RoboTwin 2.0 tests")

import h5py  # noqa: E402  (after importorskip)
from guanwu.adapters.robotwin2 import RoboTwin2Adapter, _write_animated_usdc
from guanwu.schemas.bundles import AdapterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(source_path: str, dataset_id: str = "robotwin2") -> AdapterConfig:
    return AdapterConfig(
        dataset_id=dataset_id,
        source_mode="local",
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

class TestInventory:
    def test_returns_episodes(self, robotwin2_fixture, job_context):
        adapter = RoboTwin2Adapter()
        config = _make_config(str(robotwin2_fixture))
        items = adapter.inventory(config, job_context)
        # 3 variants × 2 episodes = 6 items
        assert len(items) == 6

    def test_item_type_is_episode(self, robotwin2_fixture, job_context):
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(_make_config(str(robotwin2_fixture)), job_context)
        assert all(it.item_type == "episode" for it in items)

    def test_metadata_has_task_and_variant(self, robotwin2_fixture, job_context):
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(_make_config(str(robotwin2_fixture)), job_context)
        for it in items:
            assert "task_name" in it.metadata
            assert "variant" in it.metadata
            assert "episode_stem" in it.metadata

    def test_instruction_loaded(self, robotwin2_fixture, job_context):
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(_make_config(str(robotwin2_fixture)), job_context)
        instructions = [it.metadata.get("instruction") for it in items]
        assert any(instr is not None for instr in instructions), (
            "Expected at least one episode to have an instruction"
        )

    def test_limit_respected(self, robotwin2_fixture, job_context):
        from guanwu.schemas.bundles import JobContext
        ctx = job_context.model_copy(update={"limit": 2})
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(_make_config(str(robotwin2_fixture)), ctx)
        assert len(items) == 2

    def test_empty_source_returns_empty(self, tmp_path, job_context):
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(_make_config(str(tmp_path / "nonexistent")), job_context)
        assert items == []

    def test_ingest_clean_only(self, robotwin2_fixture, job_context):
        config = AdapterConfig(
            dataset_id="robotwin2",
            source_mode="local",
            source_path=str(robotwin2_fixture),
            options={"ingest_clean": True, "ingest_randomized": False},
        )
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(config, job_context)
        for it in items:
            assert it.metadata.get("is_randomized") is False, (
                f"Randomized episode leaked: {it.item_id}"
            )

    def test_ingest_randomized_only(self, robotwin2_fixture, job_context):
        config = AdapterConfig(
            dataset_id="robotwin2",
            source_mode="local",
            source_path=str(robotwin2_fixture),
            options={"ingest_clean": False, "ingest_randomized": True},
        )
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(config, job_context)
        assert len(items) > 0
        for it in items:
            assert it.metadata.get("is_clean") is False

    def test_task_limit_and_episodes_per_task(self, robotwin2_fixture, job_context):
        config = AdapterConfig(
            dataset_id="robotwin2",
            source_mode="local",
            source_path=str(robotwin2_fixture),
            options={
                "ingest_clean": True,
                "ingest_randomized": False,
                "task_limit": 2,
                "episodes_per_task": 1,
            },
        )
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(config, job_context)
        assert len(items) == 2
        assert {it.metadata["task_name"] for it in items} == {
            "block_handover",
            "pour_water",
        }
        assert all(it.metadata["episode_stem"] == "episode0" for it in items)

    def test_task_and_variant_name_filters(self, robotwin2_fixture, job_context):
        config = AdapterConfig(
            dataset_id="robotwin2",
            source_mode="local",
            source_path=str(robotwin2_fixture),
            options={
                "task_names": ["block_handover"],
                "variant_names": ["block_handover_randomized_50"],
            },
        )
        adapter = RoboTwin2Adapter()
        items = adapter.inventory(config, job_context)
        assert len(items) == 2
        assert {it.metadata["task_name"] for it in items} == {"block_handover"}
        assert {it.metadata["variant"] for it in items} == {
            "block_handover_randomized_50"
        }


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

class TestParseRaw:
    def _parse(self, robotwin2_fixture, job_context):
        adapter = RoboTwin2Adapter()
        config = _make_config(str(robotwin2_fixture))
        items = adapter.inventory(config, job_context)
        raw_refs = adapter.fetch(items, job_context)
        return adapter.parse_raw(raw_refs, job_context)

    def test_episodes_parsed(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        assert len(bundle.instances) == 6

    def test_episode_has_num_steps(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        for ep in bundle.instances:
            assert ep["num_steps"] == 20, f"Expected 20 steps, got {ep['num_steps']}"

    def test_cameras_detected(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        for ep in bundle.instances:
            assert set(ep["has_cameras"]) == {"head_camera", "left_camera", "right_camera"}, (
                f"Wrong cameras: {ep['has_cameras']}"
            )

    def test_depth_detected(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        for ep in bundle.instances:
            assert len(ep["has_depth"]) == 3  # all three cameras have depth

    def test_joint_state_detected(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        for ep in bundle.instances:
            assert ep["has_joint_state"] is True

    def test_action_dim_is_14(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        for ep in bundle.instances:
            assert ep["action_dim"] == 14, f"Expected 14-DOF actions, got {ep['action_dim']}"

    def test_success_flag(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        # episode0 → success=True, episode1 → success=False (per fixture)
        successes = {ep["episode_stem"]: ep["success"] for ep in bundle.instances}
        assert any(v is True for v in successes.values())
        assert any(v is False for v in successes.values())

    def test_sensors_populated(self, robotwin2_fixture, job_context):
        bundle = self._parse(robotwin2_fixture, job_context)
        assert len(bundle.sensors) > 0
        sensor_types = {s["sensor_type"] for s in bundle.sensors}
        assert "camera" in sensor_types


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def _normalize(self, robotwin2_fixture, job_context):
        adapter = RoboTwin2Adapter()
        config = _make_config(str(robotwin2_fixture))
        items = adapter.inventory(config, job_context)
        raw_refs = adapter.fetch(items, job_context)
        bundle = adapter.parse_raw(raw_refs, job_context)
        return adapter.normalize(bundle, job_context)

    def test_dataset_record_created(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        assert out.dataset_record is not None
        assert out.dataset_record.dataset_id == "robotwin2"
        assert out.dataset_record.dataset_name == "RoboTwin 2.0"

    def test_scenes_created(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        assert len(out.scenes) == 6

    def test_episodes_created_and_bound_to_scenes(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        scene_uids = {scene.scene_uid for scene in out.scenes}

        assert len(out.episodes) == 6
        for episode in out.episodes:
            assert episode.scene_uid in scene_uids
            assert episode.source_episode_id

    def test_scene_has_manipulation_kind(self, robotwin2_fixture, job_context):
        from guanwu.schemas.enums import SceneKind
        out = self._normalize(robotwin2_fixture, job_context)
        for sc in out.scenes:
            assert sc.scene_kind == SceneKind.SYNTHETIC_MANIPULATION

    def test_sensors_per_episode(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        # 3 cameras × 6 episodes = 18 sensor records
        assert len(out.sensors) == 18

    def test_sensor_types(self, robotwin2_fixture, job_context):
        from guanwu.schemas.enums import SensorType
        out = self._normalize(robotwin2_fixture, job_context)
        for s in out.sensors:
            assert s.sensor_type in (SensorType.CAMERA, SensorType.DEPTH_CAMERA)

    def test_robot_instances_created(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        robot_insts = [i for i in out.instances if i.category == "robot"]
        assert len(robot_insts) == 6

    def test_robot_is_articulated(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        for inst in out.instances:
            assert inst.is_articulated is True

    def test_track_states_generated(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        # 20 steps × 6 episodes
        assert len(out.track_states) == 6 * 20

    def test_articulation_states_generated(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        assert len(out.articulation_states) == 6 * 20

    def test_joint_names_are_14(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        for art in out.articulation_states:
            assert len(art.joint_names) == 14

    def test_articulation_states_use_hdf5_joint_positions(
        self, robotwin2_fixture, job_context
    ):
        import numpy as np
        from guanwu.core.ids import make_episode_uid, make_instance_uid

        adapter = RoboTwin2Adapter()
        config = _make_config(str(robotwin2_fixture))
        items = adapter.inventory(config, job_context)
        raw_refs = adapter.fetch(items, job_context)
        parsed = adapter.parse_raw(raw_refs, job_context)
        out = adapter.normalize(parsed, job_context)

        first_ep = parsed.instances[0]
        episode_uid = make_episode_uid(out.dataset_id, first_ep["source_episode_id"])
        robot_uid = make_instance_uid(out.dataset_id, episode_uid, "robot")
        first_state = next(
            state
            for state in out.articulation_states
            if state.instance_uid == robot_uid and state.timestamp_ns == 0
        )

        with h5py.File(first_ep["hdf5_path"], "r") as f:
            expected = np.array(f["observation"]["joint_state"][0])
        assert np.allclose(first_state.joint_positions, expected)

    def test_license_apache(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        assert len(out.licenses) == 1
        lic = out.licenses[0]
        assert lic.license_name == "Apache-2.0"
        assert lic.commercial_use_allowed is True
        assert lic.redistribution_allowed is True

    def test_provenance_present(self, robotwin2_fixture, job_context):
        out = self._normalize(robotwin2_fixture, job_context)
        assert len(out.provenance) == 1
        prov = out.provenance[0]
        assert prov.adapter_name == "robotwin2"

    def test_stable_scene_uids(self, robotwin2_fixture, job_context):
        """Running normalize twice on the same data must produce identical UIDs."""
        out1 = self._normalize(robotwin2_fixture, job_context)
        out2 = self._normalize(robotwin2_fixture, job_context)
        uids1 = sorted(s.scene_uid for s in out1.scenes)
        uids2 = sorted(s.scene_uid for s in out2.scenes)
        assert uids1 == uids2


# ---------------------------------------------------------------------------
# Capabilities contract
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_required_keys_present(self):
        adapter = RoboTwin2Adapter()
        caps = adapter.capabilities()
        required = {
            "scene_mesh", "object_mesh", "articulation", "deformable_mesh",
            "camera", "depth", "lidar", "tracks", "videos",
            "sdk_required", "license_gated", "supports_local_ingest",
        }
        assert required.issubset(set(caps.keys()))

    def test_all_values_are_bool(self):
        adapter = RoboTwin2Adapter()
        for k, v in adapter.capabilities().items():
            assert isinstance(v, bool), f"capability '{k}' is not bool: {v}"


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

class TestEmit:
    def _prepare(self, robotwin2_fixture, job_context, options=None):
        ctx = job_context.model_copy(update={"limit": 1})
        adapter = RoboTwin2Adapter()
        config = AdapterConfig(
            dataset_id="robotwin2",
            source_mode="local",
            source_path=str(robotwin2_fixture),
            options=options or {},
        )
        items = adapter.inventory(config, ctx)
        raw_refs = adapter.fetch(items, ctx)
        parsed = adapter.parse_raw(raw_refs, ctx)
        normalized = adapter.normalize(parsed, ctx)
        return adapter, normalized, ctx

    def test_emit_uses_remote_export_when_configured(
        self, robotwin2_fixture, job_context, monkeypatch
    ):
        adapter, normalized, ctx = self._prepare(
            robotwin2_fixture,
            job_context,
            options={
                "remote_engine_export": True,
                "remote_robotwin_root": "/remote/RoboTwin",
                "remote_source_root": "/remote/datasets/robotwin2",
                "render_views": True,
                "generated_static_camera_count": 1,
                "generated_trajectory_camera_count": 1,
                "view_resolution": [320, 240],
            },
        )
        ctx = ctx.model_copy(update={"remote_host": "qingyan-mcps"})

        import guanwu.core.remote as remote_mod
        import guanwu.core.remote_tasks as remote_tasks_mod

        def fake_get_remote_executor(_config):
            return object()

        def fake_export(_executor, **kwargs):
            assert kwargs["render_videos"] is False
            assert kwargs["render_views"] is True
            assert kwargs["generated_static_camera_count"] == 1
            assert kwargs["generated_trajectory_camera_count"] == 1
            assert kwargs["view_width_px"] == 320
            assert kwargs["view_height_px"] == 240
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "scene.usdz").write_bytes(b"remote-usdz")
            files_written = ["scene.usdz"]
            for idx, strategy in enumerate(("random-static", "trajectory")):
                view_id = f"view_{idx:03d}"
                view_dir = output_dir / "views" / view_id
                view_dir.mkdir(parents=True, exist_ok=True)
                (view_dir / "video.mp4").write_bytes(b"video")
                (view_dir / "camera.json").write_text(
                    json.dumps({
                        "view_id": view_id,
                        "camera_strategy": strategy,
                        "uses_hdf5_rgb": False,
                    })
                )
                (view_dir / "render_meta.json").write_text(
                    json.dumps({
                        "frame_count": 20,
                        "fps": 30.0,
                        "uses_hdf5_rgb": False,
                    })
                )
                (view_dir / "frame_mapping.json").write_text("[]")
                files_written.extend([
                    f"views/{view_id}/video.mp4",
                    f"views/{view_id}/camera.json",
                    f"views/{view_id}/render_meta.json",
                    f"views/{view_id}/frame_mapping.json",
                ])
            return {
                "files_written": files_written,
                "summary": {"num_frames": 20},
            }

        monkeypatch.setattr(remote_mod, "get_remote_executor", fake_get_remote_executor)
        monkeypatch.setattr(remote_tasks_mod, "export_robotwin_episode", fake_export)

        report = adapter.emit(normalized, ctx)

        episode_uid = next(iter(adapter._uid_map.values()))
        scene_uid = next(
            ep.scene_uid for ep in normalized.episodes
            if ep.episode_uid == episode_uid
        )
        scene_dir = (
            Path(ctx.canonical_root)
            / "datasets"
            / normalized.dataset_id
            / "scenes"
            / scene_uid
        )
        episode_dir = (
            Path(ctx.canonical_root)
            / "datasets"
            / normalized.dataset_id
            / "episodes"
            / episode_uid
        )
        assert (scene_dir / "scene.usdz").is_file()
        assert not (scene_dir / "scene.usdc").exists()
        assert (scene_dir / "views" / "view_000" / "video.mp4").is_file()
        assert not (scene_dir / "views" / "view_000" / "scene.usdc").exists()
        assert not (scene_dir / "renders").exists()
        manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format"] == "guanwu.robotwin2.scene_views.v0.1"
        assert manifest["scene"]["usdz"] == "scene.usdz"
        assert manifest["render"]["expected_view_count"] == 2
        assert manifest["render"]["render_options_hash"]
        assert manifest["views"][0]["video"] == "views/view_000/video.mp4"
        assert manifest["views"][1]["video"] == "views/view_001/video.mp4"
        assert manifest["views"][0]["uses_hdf5_rgb"] is False
        assert "learning_view" not in manifest
        assert (scene_dir / "articulation.parquet").is_file()
        assert not (episode_dir / "scene.usdc").exists()
        assert f"scenes/{scene_uid}/scene.usdz" in report.files_written
        assert f"scenes/{scene_uid}/views/view_000/video.mp4" in report.files_written
        assert f"scenes/{scene_uid}/views/view_001/video.mp4" in report.files_written
        assert f"scenes/{scene_uid}/manifest.json" in report.files_written

        def explode_get_remote_executor(_config):
            raise AssertionError("complete scene should skip remote executor setup")

        monkeypatch.setattr(remote_mod, "get_remote_executor", explode_get_remote_executor)
        resumed = adapter.emit(normalized, ctx)
        assert f"scenes/{scene_uid}/scene.usdz" in resumed.files_written
        assert f"scenes/{scene_uid}/views/view_001/video.mp4" in resumed.files_written

    def test_emit_raises_when_remote_export_fails(
        self, robotwin2_fixture, job_context, monkeypatch
    ):
        adapter, normalized, ctx = self._prepare(
            robotwin2_fixture,
            job_context,
            options={
                "remote_engine_export": True,
                "remote_robotwin_root": "/remote/RoboTwin",
            },
        )
        ctx = ctx.model_copy(update={"remote_host": "qingyan-mcps"})

        import guanwu.core.remote as remote_mod
        import guanwu.core.remote_tasks as remote_tasks_mod

        def fake_get_remote_executor(_config):
            return object()

        def fake_export(_executor, **kwargs):
            raise RuntimeError(f"remote replay unavailable for {kwargs['episode_stem']}")

        monkeypatch.setattr(remote_mod, "get_remote_executor", fake_get_remote_executor)
        monkeypatch.setattr(remote_tasks_mod, "export_robotwin_episode", fake_export)

        with pytest.raises(RuntimeError, match="Remote native RoboTwin replay export failed"):
            adapter.emit(normalized, ctx)

        episode_uid = next(iter(adapter._uid_map.values()))
        scene_uid = next(
            ep.scene_uid for ep in normalized.episodes
            if ep.episode_uid == episode_uid
        )
        scene_dir = (
            Path(ctx.canonical_root)
            / "datasets"
            / normalized.dataset_id
            / "scenes"
            / scene_uid
        )
        assert not (scene_dir / "scene.usdc").exists()

    def test_emit_requires_remote_gpu_configuration(
        self, robotwin2_fixture, job_context
    ):
        adapter, normalized, ctx = self._prepare(
            robotwin2_fixture,
            job_context,
            options={
                "remote_engine_export": True,
                "remote_robotwin_root": "/remote/RoboTwin",
            },
        )

        with pytest.raises(RuntimeError, match="remote GPU host"):
            adapter.emit(normalized, ctx)

    def test_emit_rejects_disabled_remote_engine(
        self, robotwin2_fixture, job_context
    ):
        adapter, normalized, ctx = self._prepare(
            robotwin2_fixture,
            job_context,
            options={
                "remote_engine_export": False,
                "remote_robotwin_root": "/remote/RoboTwin",
            },
        )
        ctx = ctx.model_copy(update={"remote_host": "qingyan-mcps"})

        with pytest.raises(RuntimeError, match="remote_engine_export must stay true"):
            adapter.emit(normalized, ctx)

    def test_camera_and_depth_declared(self):
        caps = RoboTwin2Adapter().capabilities()
        assert caps["camera"] is True
        assert caps["depth"] is True
        assert caps["articulation"] is True

    def test_no_mesh_capabilities(self):
        caps = RoboTwin2Adapter().capabilities()
        assert caps["scene_mesh"] is False
        assert caps["object_mesh"] is False


def test_write_animated_usdc_uses_shoulder_pillar_instead_of_box2(tmp_path):
    pytest.importorskip("pxr", reason="usd-core required for USD structure checks")
    from pxr import Usd
    import numpy as np

    path = tmp_path / "scene.usdc"
    tri_verts = np.array(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]],
        dtype=np.float64,
    )
    tri_faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh_keys = [
        "tracer_base_link",
        "box1_Link",
        "camera_base_link",
        "camera_link1",
        "camera_link2",
        "base_arm",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link8",
    ]
    mesh_data = {key: (tri_verts, tri_faces) for key in mesh_keys}

    _write_animated_usdc(
        path,
        T=2,
        ee_poses=np.zeros((2, 2, 7), dtype=np.float64),
        joint_states=np.zeros((2, 14), dtype=np.float64),
        gripper_states={
            "left": np.zeros(2, dtype=np.float64),
            "right": np.zeros(2, dtype=np.float64),
        },
        camera_poses=None,
        arm_joint_angles={
            "left": np.zeros((2, 6), dtype=np.float64),
            "right": np.zeros((2, 6), dtype=np.float64),
        },
        mesh_data=mesh_data,
        instruction=None,
        fps=30.0,
    )

    stage = Usd.Stage.Open(str(path))
    body = stage.GetPrimAtPath("/Scene/Robot/Body")

    assert stage.GetDefaultPrim().GetPath().pathString == "/Scene"
    assert body.GetChild("ShoulderPillar").IsValid()
    assert not body.GetChild("Box2").IsValid()


def test_write_animated_usdc_authors_hdf5_cameras(tmp_path):
    pytest.importorskip("pxr", reason="usd-core required for USD structure checks")
    from pxr import Usd, UsdGeom
    import numpy as np

    path = tmp_path / "scene.usdc"
    pose0 = np.eye(4, dtype=np.float64)
    pose1 = np.eye(4, dtype=np.float64)
    pose1[:3, 3] = [0.25, -0.5, 1.25]
    intrinsic = np.array(
        [
            [358.64218, 0.0, 160.0],
            [0.0, 358.64218, 120.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    _write_animated_usdc(
        path,
        T=2,
        ee_poses=np.zeros((2, 2, 7), dtype=np.float64),
        joint_states=np.zeros((2, 14), dtype=np.float64),
        gripper_states=None,
        camera_poses={"front_camera": np.stack([pose0, pose1], axis=0)},
        arm_joint_angles=None,
        mesh_data=None,
        instruction=None,
        camera_intrinsics={"front_camera": np.stack([intrinsic, intrinsic], axis=0)},
        camera_image_sizes={"front_camera": (320, 240)},
        fps=30.0,
    )

    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/Scene/Cameras/front_camera")
    assert prim.IsA(UsdGeom.Camera)
    assert prim.GetAttribute("robotwin:imageWidth").Get() == 320
    assert prim.GetAttribute("robotwin:imageHeight").Get() == 240
    assert prim.GetAttribute("robotwin:sourceCameraName").Get() == "front_camera"

    world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(1)
    translation = world_xf.ExtractTranslation()
    assert np.allclose([translation[0], translation[1], translation[2]], [0.25, -0.5, 1.25])
