# URDF RGB Robot Base Pose Estimation

This module estimates the robot base pose `T_C_B` from:

- RGB image
- camera intrinsics `K`
- robot URDF
- known joint angles
- either a binary robot mask or a Grounded-SAM2 prompt

It uses a first-version CPU pipeline: URDF/FK mesh assembly, OpenCV-style projection, triangle-fill silhouette rendering, partial-observation mask/contour losses, multi-start initialization, Powell local optimization, and mask-moment refinement.

## Install Extras

The minimal tested path works with `numpy`, `scipy`, and `Pillow`. For real URDF assets and SAM2 mask decoding, install:

```bash
pip install -e ".[robotpose]"
```

`yourdfpy`/`trimesh` are used when available for full URDF mesh loading. A small fallback parser supports simple `.obj` visual meshes.

## CLI

Use a provided mask:

```bash
python -m robotpose.estimate \
  --image image.png \
  --urdf robot.urdf \
  --joints joints.json \
  --camera camera.json \
  --mask robot_mask.png \
  --init-pose init_pose.json \
  --output-dir outputs/robotpose_run
```

Use Grounded-SAM2 through the Guanwu/Zaiwu service:

```bash
python -m robotpose.estimate \
  --image image.png \
  --urdf robot.urdf \
  --joints joints.json \
  --camera camera.json \
  --sam2-project /path/to/video_project \
  --prompt "robot arm" \
  --output-dir outputs/robotpose_run
```

You can also replay a saved service payload:

```bash
python -m robotpose.estimate \
  --image image.png \
  --urdf robot.urdf \
  --joints joints.json \
  --camera camera.json \
  --sam2-payload grounded_sam2_raw.json \
  --output-dir outputs/robotpose_run
```

## Input Formats

`camera.json`:

```json
{
  "K": [[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]]
}
```

or:

```json
{"fx": 640.0, "fy": 640.0, "cx": 320.0, "cy": 240.0}
```

`joints.json` can be a mapping:

```json
{"shoulder_pan_joint": 0.1, "elbow_joint": -1.2}
```

or a list in URDF actuated-joint order when using `yourdfpy`.

`init_pose.json`:

```json
{
  "T_C_B": [
    [1, 0, 0, 0.0],
    [0, 1, 0, 0.0],
    [0, 0, 1, 2.0],
    [0, 0, 0, 1.0]
  ]
}
```

Coordinate convention is OpenCV camera coordinates: `x` right, `y` down, `z` forward. The URDF root link is the robot base frame `B`.

## Outputs

The output directory contains:

- `result.json`: final `T_C_B`, translation, rotation vector, quaternion `xyzw`, loss terms, metrics, selected initialization.
- `overlay.png`: rendered robot silhouette over the RGB image, with observed and rendered contours.
- `mask_comparison.png`: observed mask, rendered mask, and overlap.
- `rendered_mask.png`: optimized rendered silhouette.
- `observed_mask.png`: mask used for optimization.
- `loss_history.csv`: per-evaluation loss history.
- `grounded_sam2_raw.json` / `grounded_sam2_selected.json` when SAM2 is used.

## Useful Options

```bash
--coarse-depths 0.8,1.2,1.8,2.5,3.5,5.0
--coarse-yaws-deg 0:360:30
--stage-scales 0.35,0.65,1.0
--top-k 8
--max-iterations 90
--no-coarse-search
```

If `--no-coarse-search` is used, provide `--init-pose`.

## Python API

```python
from robotpose import estimate_robot_base_pose

result = estimate_robot_base_pose(
    image_rgb=image,
    camera_K=K,
    urdf_path="robot.urdf",
    joint_angles={"joint_1": 0.0},
    init_poses=[T0_C_B],
    robot_mask=mask,
)

print(result.T_C_B)
print(result.translation)
print(result.rotation_vector)
```

## Notes

- This is a silhouette-based optimizer, so a reasonable mask and plausible initial depth range matter.
- Partial observation is handled by one-sided observed-to-render losses plus weak reverse and area regularization.
- If the robot is heavily occluded, lower the reliance on reverse/area terms by constructing `OptimizerOptions(loss_config=...)` in Python.
