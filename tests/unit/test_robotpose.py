from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation


def _square_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


def test_project_points_and_delta_pose_use_opencv_camera_convention() -> None:
    from robotpose.optimizer import compose_delta_pose
    from robotpose.render import project_points

    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.5, 2.0], [0.0, 0.0, -1.0]])

    uv, valid = project_points(points, K)

    assert np.allclose(uv[:2], [[50.0, 40.0], [100.0, 65.0]])
    assert valid.tolist() == [True, True, False]

    base = np.eye(4)
    base[:3, 3] = [0.1, 0.2, 1.0]
    delta = np.array([0.0, 0.0, np.pi / 2.0, 0.3, -0.1, 0.2])

    updated = compose_delta_pose(base, delta)

    assert np.allclose(updated[:3, 3], [0.1, 0.0, 1.2], atol=1e-9)
    assert np.allclose(updated[:3, :3], Rotation.from_rotvec(delta[:3]).as_matrix(), atol=1e-9)


def test_contour_suppression_and_one_sided_loss_terms() -> None:
    from robotpose.losses import LossConfig, compute_partial_observation_loss, extract_contour

    observed = np.zeros((40, 50), dtype=bool)
    observed[10:25, 12:32] = True
    rendered = observed.copy()

    contour = extract_contour(observed, border_margin=5)
    assert contour.sum() > 0
    assert not contour[:5, :].any()
    assert not contour[:, :5].any()

    good = compute_partial_observation_loss(
        observed,
        rendered,
        np.eye(4),
        np.eye(4),
        config=LossConfig(),
    )
    shifted = np.zeros_like(observed)
    shifted[10:25, 20:40] = True
    bad = compute_partial_observation_loss(
        observed,
        shifted,
        np.eye(4),
        np.eye(4),
        config=LossConfig(),
    )

    assert good.total < bad.total
    assert good.metrics["overlap_ratio"] == pytest.approx(1.0)
    assert bad.terms["obs2ren_mask"] > good.terms["obs2ren_mask"]


def test_triangle_renderer_and_optimizer_reduce_synthetic_pose_error() -> None:
    from robotpose.optimizer import OptimizerOptions, optimize_robot_pose
    from robotpose.render import RobotMesh, render_robot_mask

    vertices, faces = _square_mesh()
    mesh = RobotMesh(vertices=vertices, faces=faces)
    K = np.array([[80.0, 0.0, 48.0], [0.0, 80.0, 36.0], [0.0, 0.0, 1.0]])
    image_shape = (72, 96)
    true_pose = np.eye(4)
    true_pose[:3, 3] = [0.0, 0.0, 4.0]
    observed = render_robot_mask(mesh, K, true_pose, image_shape).mask

    init_pose = np.eye(4)
    init_pose[:3, 3] = [0.35, -0.25, 4.2]
    options = OptimizerOptions(
        coarse_depths=(4.0,),
        coarse_yaws_deg=(0.0,),
        coarse_pitch_deg=(0.0,),
        coarse_roll_deg=(0.0,),
        top_k=1,
        stage_scales=(1.0,),
        max_iterations=60,
        optimizer_method="Powell",
    )

    result = optimize_robot_pose(mesh, K, observed, image_shape, init_poses=[init_pose], options=options)

    assert result.loss.total < result.initial_loss.total
    assert np.linalg.norm(result.T_C_B[:3, 3] - true_pose[:3, 3]) < 0.18
    assert result.rendered_mask.shape == observed.shape


def test_decode_uncompressed_grounded_sam2_rle_mask() -> None:
    from robotpose.sam2 import decode_instance_mask, mask_from_grounded_sam2_payload

    payload = {
        "instances": [
            {
                "label": "robot arm",
                "score": 0.91,
                "mask_rle": {"size": [3, 4], "counts": [2, 3, 7]},
            },
            {"label": "table", "score": 0.99, "mask_rle": {"size": [3, 4], "counts": [0, 1, 11]}},
        ]
    }

    decoded = decode_instance_mask(payload["instances"][0], (3, 4))
    mask, selected = mask_from_grounded_sam2_payload(payload, (3, 4), label_terms=("robot",))

    assert decoded is not None
    assert decoded.sum() == 3
    assert mask.sum() == 3
    assert selected[0]["label"] == "robot arm"


def test_decode_compressed_grounded_sam2_rle_json_string_without_bbox_fallback() -> None:
    from robotpose.sam2 import decode_instance_mask

    expected_flat = np.zeros(12, dtype=bool)
    expected_flat[2:5] = True
    expected = expected_flat.reshape((3, 4), order="F")
    inst = {
        "label": "robotic arm",
        "bbox": [0, 0, 4, 3],
        "mask_rle": json.dumps({"size": [3, 4], "counts": "237"}),
    }

    decoded = decode_instance_mask(inst, expected.shape)

    assert decoded is not None
    assert np.array_equal(decoded, expected)
    assert int(decoded.sum()) == 3


def test_minimal_urdf_loader_builds_current_configuration_mesh(tmp_path: Path) -> None:
    from robotpose.urdf_model import load_robot_mesh_from_urdf

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "base.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    (mesh_dir / "link.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="tiny">
  <link name="base">
    <visual><geometry><mesh filename="meshes/base.obj"/></geometry></visual>
  </link>
  <link name="link1">
    <visual><geometry><mesh filename="meshes/link.obj"/></geometry></visual>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    mesh = load_robot_mesh_from_urdf(urdf, {"slide": 0.5})

    assert mesh.vertices.shape == (6, 3)
    assert mesh.faces.shape == (2, 3)
    assert np.allclose(mesh.vertices[3], [1.0, 0.5, 0.0])


def test_cli_writes_result_and_visual_outputs(tmp_path: Path) -> None:
    image = np.full((40, 50, 3), 240, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:24, 14:31] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "base.obj").write_text(
        "v -0.5 -0.4 0\nv 0.5 -0.4 0\nv 0.5 0.4 0\nv -0.5 0.4 0\nf 1 2 3\nf 1 3 4\n",
        encoding="utf-8",
    )
    (tmp_path / "robot.urdf").write_text(
        """<?xml version="1.0"?>
<robot name="tiny">
  <link name="base">
    <visual><geometry><mesh filename="meshes/base.obj"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    (tmp_path / "joints.json").write_text("{}", encoding="utf-8")
    (tmp_path / "camera.json").write_text(
        json.dumps({"K": [[40.0, 0.0, 25.0], [0.0, 40.0, 20.0], [0.0, 0.0, 1.0]]}),
        encoding="utf-8",
    )
    init_pose = np.eye(4)
    init_pose[:3, 3] = [0.0, 0.0, 3.0]
    (tmp_path / "init_pose.json").write_text(json.dumps({"T_C_B": init_pose.tolist()}), encoding="utf-8")
    out_dir = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robotpose.estimate",
            "--image",
            str(tmp_path / "image.png"),
            "--urdf",
            str(tmp_path / "robot.urdf"),
            "--joints",
            str(tmp_path / "joints.json"),
            "--camera",
            str(tmp_path / "camera.json"),
            "--mask",
            str(tmp_path / "mask.png"),
            "--init-pose",
            str(tmp_path / "init_pose.json"),
            "--output-dir",
            str(out_dir),
            "--max-iterations",
            "3",
            "--no-coarse-search",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    result_path = out_dir / "result.json"
    assert "result_path" in completed.stdout
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert np.asarray(result["T_C_B"]).shape == (4, 4)
    assert "translation" in result
    assert "rotation_vector" in result
    assert "quaternion_xyzw" in result
    assert (out_dir / "overlay.png").is_file()
    assert (out_dir / "mask_comparison.png").is_file()
    assert (out_dir / "rendered_mask.png").is_file()
    assert (out_dir / "loss_history.csv").is_file()
