from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .losses import LossConfig
from .masks import draw_mask_comparison, draw_overlay, read_mask, read_rgb_image, save_mask, save_rgb
from .optimizer import OptimizerOptions, RobotPoseResult, optimize_robot_pose, write_history_csv
from .render import RobotMesh, normalize_camera_K
from .sam2 import call_grounded_sam2, mask_from_grounded_sam2_payload, save_payload
from .urdf_model import load_robot_mesh_from_urdf


def estimate_robot_base_pose(
    image_rgb: np.ndarray,
    camera_K: np.ndarray | list[list[float]] | dict[str, float],
    urdf_path: str | Path,
    joint_angles: dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None,
    *,
    init_poses: list[np.ndarray] | None = None,
    robot_mask: np.ndarray | None = None,
    options: OptimizerOptions | None = None,
) -> RobotPoseResult:
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H, W, 3], got {image.shape}")
    if robot_mask is None:
        raise ValueError("robot_mask is required for the public API. The CLI can obtain one via Grounded-SAM2.")
    mask = np.asarray(robot_mask).astype(bool)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"robot_mask shape {mask.shape} does not match image shape {image.shape[:2]}")
    mesh = load_robot_mesh_from_urdf(urdf_path, joint_angles)
    return optimize_robot_pose(
        mesh,
        normalize_camera_K(camera_K),
        mask,
        image.shape[:2],
        init_poses=init_poses,
        options=options,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate robot base pose T_C_B from URDF, joint angles, RGB image, camera intrinsics, and mask/SAM2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="RGB image path")
    parser.add_argument("--urdf", required=True, help="Robot URDF path")
    parser.add_argument("--joints", required=True, help="Joint angles JSON path: mapping or list")
    parser.add_argument("--camera", required=True, help="Camera JSON path containing K or fx/fy/cx/cy")
    parser.add_argument("--output-dir", required=True, help="Directory for result.json and visual outputs")
    parser.add_argument("--mask", default=None, help="Binary robot mask PNG. If omitted, --sam2-project is required.")
    parser.add_argument("--sam2-project", default=None, help="Guanwu video project root with project.toml for Zaiwu Grounded-SAM2")
    parser.add_argument("--prompt", default="robot arm", help="Grounded-SAM2 text prompt")
    parser.add_argument("--sam2-payload", default=None, help="Use saved Grounded-SAM2 JSON instead of calling the service")
    parser.add_argument("--init-pose", default=None, help="JSON containing T_C_B or a raw 4x4 matrix")
    parser.add_argument("--max-iterations", type=int, default=90)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--stage-scales", default="0.35,0.65,1.0")
    parser.add_argument("--coarse-depths", default="0.8,1.2,1.8,2.5,3.5,5.0")
    parser.add_argument("--coarse-yaws-deg", default="0:360:30")
    parser.add_argument("--no-coarse-search", action="store_true")
    parser.add_argument("--render-backend", choices=["triangle_fill"], default="triangle_fill")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = read_rgb_image(args.image)
    camera_K = load_camera_K(args.camera)
    joints = load_json(args.joints)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mask:
        robot_mask = read_mask(args.mask, image.shape[:2])
    else:
        payload = load_or_call_sam2(args, image)
        save_payload(out_dir / "grounded_sam2_raw.json", payload)
        robot_mask, selected = mask_from_grounded_sam2_payload(payload, image.shape[:2])
        (out_dir / "grounded_sam2_selected.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not robot_mask.any():
            raise RuntimeError("Grounded-SAM2 did not return a non-empty robot mask. Provide --mask or adjust --prompt.")

    init_poses = load_init_poses(args.init_pose) if args.init_pose else None
    options = OptimizerOptions(
        loss_config=LossConfig(),
        coarse_depths=parse_float_tuple(args.coarse_depths),
        coarse_yaws_deg=parse_angle_grid(args.coarse_yaws_deg),
        top_k=int(args.top_k),
        stage_scales=parse_float_tuple(args.stage_scales),
        max_iterations=int(args.max_iterations),
        coarse_search=not bool(args.no_coarse_search),
    )
    if args.no_coarse_search and not init_poses:
        raise ValueError("--init-pose is required with --no-coarse-search")

    result = estimate_robot_base_pose(
        image,
        camera_K,
        args.urdf,
        joints,
        init_poses=init_poses,
        robot_mask=robot_mask,
        options=options,
    )
    write_outputs(out_dir, image, robot_mask, result, options)
    print(f"result_path: {out_dir / 'result.json'}")
    print(f"overlay_path: {out_dir / 'overlay.png'}")
    print(f"loss_total: {result.loss.total:.6f}")
    return 0


def load_or_call_sam2(args: argparse.Namespace, image: np.ndarray) -> dict[str, Any]:
    if args.sam2_payload:
        return load_json(args.sam2_payload)
    if not args.sam2_project:
        raise ValueError("Either --mask or --sam2-project/--sam2-payload must be provided.")
    return call_grounded_sam2(args.sam2_project, image, prompt=args.prompt)


def write_outputs(
    out_dir: Path,
    image_rgb: np.ndarray,
    observed_mask: np.ndarray,
    result: RobotPoseResult,
    options: OptimizerOptions,
) -> None:
    save_mask(out_dir / "rendered_mask.png", result.rendered_mask)
    save_mask(out_dir / "observed_mask.png", observed_mask)
    save_rgb(out_dir / "overlay.png", draw_overlay(image_rgb, observed_mask, result.rendered_mask))
    save_rgb(out_dir / "mask_comparison.png", draw_mask_comparison(image_rgb, observed_mask, result.rendered_mask))
    write_history_csv(out_dir / "loss_history.csv", result.history)
    payload = {
        "T_C_B": result.T_C_B.tolist(),
        "translation": result.translation.tolist(),
        "rotation_vector": result.rotation_vector.tolist(),
        "quaternion_xyzw": result.quaternion_xyzw.tolist(),
        "loss_total": result.loss.total,
        "loss_terms": result.loss.terms,
        "weighted_loss_terms": result.loss.weighted_terms,
        "metrics": result.loss.metrics,
        "initial_loss_total": result.initial_loss.total,
        "initial_loss_terms": result.initial_loss.terms,
        "selected_init_index": result.selected_init_index,
        "projected_bbox": result.projected_bbox,
        "options": {
            "max_iterations": options.max_iterations,
            "top_k": options.top_k,
            "stage_scales": list(options.stage_scales),
            "coarse_depths": list(options.coarse_depths),
            "coarse_yaws_deg": list(options.coarse_yaws_deg),
            "loss_config": asdict(options.loss_config),
        },
        "outputs": {
            "overlay": str(out_dir / "overlay.png"),
            "mask_comparison": str(out_dir / "mask_comparison.png"),
            "rendered_mask": str(out_dir / "rendered_mask.png"),
            "observed_mask": str(out_dir / "observed_mask.png"),
            "loss_history": str(out_dir / "loss_history.csv"),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_camera_K(path: str | Path) -> np.ndarray:
    data = load_json(path)
    if isinstance(data, list):
        return normalize_camera_K(data)
    if not isinstance(data, dict):
        raise ValueError("Camera JSON must be a mapping or 3x3 matrix list.")
    if "K" in data:
        return normalize_camera_K(data["K"])
    return normalize_camera_K(data)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def load_init_poses(path: str | Path) -> list[np.ndarray]:
    data = load_json(path)
    raw = data.get("T_C_B") if isinstance(data, dict) else data
    if raw is None and isinstance(data, dict):
        raw = data.get("poses")
    if isinstance(raw, list) and len(raw) == 4 and all(isinstance(row, list) for row in raw):
        return [np.asarray(raw, dtype=np.float64).reshape(4, 4)]
    poses = [np.asarray(item.get("T_C_B", item), dtype=np.float64).reshape(4, 4) for item in raw]
    return poses


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def parse_angle_grid(value: str) -> tuple[float, ...]:
    text = str(value).strip()
    if ":" in text:
        start, stop, step = [float(item) for item in text.split(":")]
        values = []
        current = start
        while current < stop:
            values.append(float(current))
            current += step
        return tuple(values)
    return parse_float_tuple(text)


if __name__ == "__main__":
    raise SystemExit(main())
