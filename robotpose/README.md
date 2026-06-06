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

## End-to-End Pipeline

`robotpose.estimate` is a single-frame estimator. For an RGB video, first choose
one representative frame, prepare the matching camera intrinsics and joint
angles for that frame, then run pose estimation on that frame. The current
recommended flow is:

1. Extract one RGB frame from the video.
2. Prepare `camera.json` at the extracted frame resolution.
3. Prepare `joints.json` for the same frame timestamp.
4. Obtain a binary robot mask, either from Grounded-SAM2 or a saved mask.
5. Run a broad coarse search.
6. Run a high-resolution local refinement initialized from the coarse result.
7. Inspect `overlay.png`, `mask_comparison.png`, and metrics in `result.json`.

### 1. Prepare a Frame

The CLI expects an image, not a video path. For a video, extract a frame with
OpenCV:

```bash
python - <<'PY'
from pathlib import Path
import cv2

video = Path("/path/to/cam_xxx/color.mp4")
frame_idx = 192
out = Path("inputs/frame_000192.png")
out.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, frame_bgr = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Could not read frame {frame_idx} from {video}")
cv2.imwrite(str(out), frame_bgr)
PY
```

Keep the frame index and timestamp in a small run summary so that the camera,
joints, mask, and output can be audited later.

### 2. Prepare Camera Intrinsics

The camera file must contain intrinsics for the actual image resolution passed
to the estimator. If the calibration was stored for `1280x720` but the video
frame is `640x360`, scale the first row of `K` by `640 / 1280` and the second
row by `360 / 720`.

```json
{
  "K": [
    [454.8904724121094, 0.0, 316.56683349609375],
    [0.0, 454.3556823730469, 168.76495361328125],
    [0.0, 0.0, 1.0]
  ],
  "original_K": [
    [909.7809448242188, 0.0, 633.1336669921875],
    [0.0, 908.7113647460938, 337.5299072265625],
    [0.0, 0.0, 1.0]
  ],
  "source_size": [1280, 720],
  "frame_size": [640, 360],
  "camera_id": "104122063678"
}
```

Only `K` is required by the estimator. The extra fields are useful for
debugging and reproducing a run.

### 3. Prepare Joint Angles

`joints.json` can be a mapping from URDF joint name to joint angle in radians.
For the Franka Panda URDF used in the current dataset:

```json
{
  "panda_joint1": 0.057901822024818514,
  "panda_joint2": 0.4961554474341375,
  "panda_joint3": -0.0006500427416632046,
  "panda_joint4": -1.940181299186304,
  "panda_joint5": -0.039590449931045345,
  "panda_joint6": 2.473358170371896,
  "panda_joint7": 0.9951655984979417,
  "panda_finger_joint1": 0.01,
  "panda_finger_joint2": 0.01
}
```

For RH20T-style data, `timestamps.npy` may be stored as a pickled dictionary
with `color` and `depth` timestamp lists, while `transformed/joint.npy` may be a
pickled dictionary keyed by camera id and timestamp. In that case, use the
selected color-frame timestamp and choose the nearest available joint
timestamp. Record the timestamp delta; it should normally be small.

### 4. Prepare the Robot Mask

The optimizer aligns the URDF silhouette to a binary observed robot mask. There
are three supported ways to provide it:

- `--mask robot_mask.png`: use an existing binary mask.
- `--sam2-project /path/to/project --prompt "robot arm"`: call the Zaiwu
  Grounded-SAM2 service and use all returned robot-like instances.
- `--sam2-payload grounded_sam2_raw.json`: replay a saved Grounded-SAM2 payload.

When the scene contains multiple robot instances or distractors, first run a
SAM2 probe, inspect `grounded_sam2_selected.json` and the mask overlays, then
pass the selected instance mask back through `--mask` for the actual pose run.
This avoids optimizing against a union mask that includes unrelated robot parts.

### 5. Broad Coarse Search

The current practical coarse pass uses the `pattern` optimizer, a depth/yaw/
pitch/roll grid, image-space lateral offsets, and multiple stage scales:

```bash
python -m robotpose.estimate \
  --image inputs/frame_000192.png \
  --urdf /path/to/franka_panda/panda_robot_only_visual.urdf \
  --joints inputs/joints_frame_000192.json \
  --camera inputs/camera_scaled.json \
  --mask inputs/robot_mask.png \
  --output-dir outputs/coarse_pattern_lowres \
  --optimizer-method pattern \
  --top-k 6 \
  --stage-scales 0.3,0.6,1.0 \
  --coarse-depths 0.5,0.6,0.8,1.0,1.2,1.5,1.8 \
  --coarse-yaws-deg 0:360:30 \
  --coarse-pitches-deg -90,-60,-45,-30,0,30,45,60,90 \
  --coarse-rolls-deg -180,-150,-135,-120,-90,-60,-45,-30,0,30,45,60,90,120,135,150 \
  --coarse-lateral-offsets-px '0,0;0,160;0,200;0,240;0,280;0,320;60,220;-60,220;80,260;-80,260;120,260;-120,260' \
  --min-depth 0.25 \
  --max-depth 2.2 \
  --max-iterations 55
```

This writes the selected coarse pose to
`outputs/coarse_pattern_lowres/result.json`.

### 6. High-Resolution Refinement

Use the coarse `result.json` as the initialization for a focused refinement:

```bash
python -m robotpose.estimate \
  --image inputs/frame_000192.png \
  --urdf /path/to/franka_panda/panda_robot_only_visual.urdf \
  --joints inputs/joints_frame_000192.json \
  --camera inputs/camera_scaled.json \
  --mask inputs/robot_mask.png \
  --init-pose outputs/coarse_pattern_lowres/result.json \
  --output-dir outputs/highres_refine \
  --optimizer-method pattern \
  --top-k 1 \
  --stage-scales 0.6,1.0 \
  --min-depth 0.25 \
  --max-depth 2.2 \
  --max-iterations 90
```

The final pose is in `outputs/highres_refine/result.json`.

### 7. Recommended Run Layout

Use a fresh output root for each run rather than overwriting old results:

```text
robotpose_task0035_cam_104122063678_current_method_YYYYMMDD_HHMMSS/
  frame_000192.png
  camera_104122063678_scaled_from_rh20t.json
  joints_frame_000192.json
  robot_mask.png
  run_inputs_summary.json
  coarse_pattern_lowres/
    result.json
    overlay.png
    mask_comparison.png
    rendered_mask.png
    observed_mask.png
    loss_history.csv
    initial_candidates.csv
  highres_refine/
    result.json
    overlay.png
    mask_comparison.png
    rendered_mask.png
    observed_mask.png
    loss_history.csv
    initial_candidates.csv
```

### 8. Reading the Result

Important fields in `result.json`:

- `T_C_B`: final camera-to-robot-base transform.
- `translation`: base translation in OpenCV camera coordinates.
- `rotation_vector` / `quaternion_xyzw`: base rotation.
- `loss_total`: weighted optimization loss; lower is better for the same mask
  and settings.
- `metrics.mask_iou`: IoU between observed and rendered silhouettes.
- `metrics.overlap_ratio`: fraction of observed mask covered by the render.
- `metrics.area_ratio`: rendered mask area divided by observed mask area.
- `projected_bbox`: final rendered robot bbox in image coordinates.
- `initial_candidate_summary`: how many coarse candidates were generated,
  prefiltered, and rendered.

Always inspect `overlay.png` and `mask_comparison.png`. A high overlap with a
large `area_ratio` can mean the rendered robot is too large but still covers the
observed mask; a low IoU can also come from a poor SAM2 mask, an incorrect
camera scale, or joint angles from the wrong timestamp.

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
