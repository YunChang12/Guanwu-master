# Guanwu Project Memory

Updated: 2026-06-20 01:31:00 UTC

This document is a handoff note for a new Codex window. The user is iterating on Guanwu video pipeline reconstruction, pose optimization, background generation, and batched UA-DETRAC processing. Prefer Chinese when replying to the user.

## Repository And Workspace

- Main repo: `/root/autodl-fs/Qcp/Guanwu-master`
- Current git branch seen in this session: `depth-constraint-version`
- Important batch workspace: `/root/autodl-tmp/guanwu_ua_detrac_batch_workspace`
- UA-DETRAC source videos: `/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2`
- Current staged batch config: `/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/car-video2-edge-quick-staged-batch.yaml`
- Current staged batch state: `/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_state`
- Current staged batch logs: `/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_logs`

## User Goals And Preferences

- The user wants Guanwu to run complete video mainlines robustly for tabletop and road scenes.
- For road scenes, the desired 08 pose strategy is `edge_contour_fast_quick`, full-frame mode.
- For tabletop scenes, the user has often used `generic_appearance_temporal`, full-frame mode, with strong depth consistency when appropriate.
- The user prefers simple, effective, generalizable fixes instead of object-specific hacks.
- The user cares about:
  - Correct pose in USDC/catalog.
  - Consistent background depth and pose optimization depth.
  - Clean but not distorted background images.
  - Efficient multi-video throughput.
  - Avoiding wasted Zaiwu worker time.
- The user often asks to use `PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0` or GPU-specific EGL rendering.
- Do not delete running pipeline outputs unless the user explicitly asks and the relevant process has been stopped.

## Zaiwu Service Context

The user moved/repaired Zaiwu services. The current gateway URL used in project configs is:

```text
https://u785484-7r1j-d4195f69.westd.seetacloud.com:8443
```

Services the user reported:

- `depth_anything3`: running on port `20001`
- `sam3d`: running on port `19003`
- `wildgs_slam`: running on port `20011`
- `seg2track_sam2`: running on port `20010`
- `grounding_dino`: running on port `20005`, dependency of `seg2track_sam2`

Important preference from the user:

- The original calling style may remain direct-worker/gateway-compatible, but the goal is simply that the client still works after moving machines.
- For SAM3D ordinary object reconstruction, add/use:

```json
{
  "quality": "balanced",
  "export_glb": true,
  "export_ply": false
}
```

- Do not pass object SAM3D `quality` payload to `reconstruct_body`.
- Guanwu should not assume SAM3D returns PLY. Prefer returned primary/preferred mesh, then GLB, then PLY fallback.

## Current Road Batch Strategy

The current UA-DETRAC batch was changed from "6 full pipelines in parallel" to staged scheduling:

- `premesh`: `video.inspect -> geometry.lift`
- `mesh`: `mesh.reconstruct -> mesh.reconstruct`
- `postmesh`: `pose.optimize -> catalog`

Reason:

- Zaiwu services are shared and expensive.
- Same service type should be serialized/locked where needed, especially SAM3D.
- Different service types can run concurrently.
- A video that completes 07 can enter 08 immediately instead of waiting for all videos in a batch.

Current config snapshot:

```yaml
workspace_root: /root/autodl-tmp/guanwu_ua_detrac_batch_workspace
runtime:
  workers: 8
video_pipeline:
  provider_mode: zaiwu
  camera_provider: wildgs
  object_detection_backend: seg2track_sam2
  background_mode: road
  background_cleaner: openai_image_edit
  background_cleaner_model: gpt-image-2
  background_cleaner_reference_frame_id: 1
  background_scene_prompt_profile: road
  pose_optimizer_timeout_sec: 1800.0
  pose_optimize_min_bbox_area_px: 5000.0
  mesh_proxy_mode: auto
  mesh_proxy_target_faces: 15000
  mesh_reconstruct_top_k: 8
  mesh_proxy_use_for_pose: true
  mesh_proxy_keep_original_for_export: true
```

## 2026-06-20 Handoff Update From Latest Window

This section is the newest handoff. Prefer it over older "current state" notes below.

### What Happened In This Window

- The user asked this window to read this memory document, monitor the UA-DETRAC staged batch, stop/restart workers, clean bad outputs/projects, fix vehicle heading flips in 08 pose optimization, and keep the batch moving.
- The user prefers Chinese replies.
- Do not stop workers, delete project outputs, remove failed records, or retry failed samples unless the user explicitly asks.
- The batch was changed to process only already-existing/current projects instead of adding new projects. The restart path uses `current_existing_project_videos.txt`.
- Previous failed records were cleared once and the staged batch was restarted with:

```bash
BATCH_ROOT=/root/autodl-tmp/guanwu_ua_detrac_batch_workspace \
SESSION_NAME=guanwu_ua_detrac_staged_batch \
FORCE=1 \
bash scripts/run_ua_detrac_selected_staged_batch_tmux.sh --resume
```

- Backups made during that retry:

```text
/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_state/premesh_failed.txt.bak_retry_failed_20260619_104434
/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_state/premesh_claimed.txt.bak_retry_failed_20260619_104434
```

- User requested deletion of projects whose `05_object_attr` became `unknown`/`null` because of VLM/API issues. That cleanup was performed in this window according to the prior assistant state. In a new window, verify current filesystem state before assuming more deletion is needed.

### Important 08 Pose Orientation Fix

The user reported sudden vehicle heading reversal in `08_pose_optimize`, including examples around:

```text
ua_detrac_detrac_10_MVI_20062_detrac_10_MVI_20062_part001_edge_quick_balanced_allframes
obj_000001@000005 -> obj_000001@000006
obj_000003@000049 -> obj_000003@000050
```

Root cause found:

- Task generation only wrote top-level `temporal_prior_pose` from `previous_candidate_prior`.
- When that candidate prior was stale/missing, the optimizer could receive no orientation prior even though a previous accepted frame existed.
- Some generated tasks, e.g. around `000050`, had no `prior`, indicating a previous run used old logic or an old process/code state.

Implemented fix:

- File: `src/guanwu/video/project/executor.py`
- Added `_edge_pose_temporal_prior_payload_for_task`.
- In all-frames mode, temporal prior selection is now:
  1. candidate prior,
  2. fallback to previous accepted continuity seed,
  3. fallback to trusted anchor.
- In target-window mode, previous accepted pose remains preferred.
- Task metadata now writes `vehicle_pose_context["temporal_prior_selection"]` so later inspection can explain which prior source was used.

Tests/verification from this window:

```text
pytest tests/unit/test_video_pose_temporal.py -q
131 passed, 6 skipped

python -m py_compile src/guanwu/video/project/executor.py
exit 0
```

Important follow-up:

- If inspecting old `08_pose_optimize/tasks`, remember old tasks may not contain the new prior fields.
- To use the fix on an existing project, remove stale `08_pose_optimize` outputs/tasks for that project and rerun `pose.optimize -> catalog` after stopping any process that is using that project.

### Severe Truncation Discussion

- A proposed bottom-truncation heuristic was considered: bbox touches bottom and visible bbox height <= 25% of image height, mark `severe + low_observability`.
- User rejected this as unreliable and asked to refer back to the original truncation logic.
- New window should inspect current code before assuming the heuristic is active or inactive. Do not reintroduce this simple 25% bottom heuristic without user confirmation.

### Latest Batch State Checked 2026-06-20 UTC

Command used:

```bash
BATCH_ROOT=/root/autodl-tmp/guanwu_ua_detrac_batch_workspace \
SESSION_NAME=guanwu_ua_detrac_staged_batch \
bash scripts/run_ua_detrac_selected_staged_batch_tmux.sh --status
```

Latest observed counts:

```text
queued: 32
premesh.claimed: 22
premesh.done: 29
premesh.failed: 3
mesh.claimed: 29
mesh.done: 29
mesh.failed: 0
postmesh.claimed: 29
postmesh.done: 26
postmesh.failed: 1
```

Active tmux session:

```text
guanwu_ua_detrac_staged_batch
```

It currently has 6 windows, all `postmesh-*`.

Active long-running postmesh pipelines at latest check:

```text
detrac_53_MVI_63525_part010
detrac_53_MVI_63525_part016
```

Their active optimizer child tasks at latest check:

```text
part010: obj_000016@000039
part016: obj_000014@000052
```

### Current Known Failures

Three `premesh` failures remain. These failed again after retry and are not key-quota/VLM failures. They fail in `06_geometry_lift` / WildGS SLAM:

```text
/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2/detrac_47_MVI_40981/detrac_47_MVI_40981_part034.mp4
/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2/detrac_53_MVI_63525/detrac_53_MVI_63525_part017.mp4
/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2/detrac_10_MVI_20062/detrac_10_MVI_20062_part014.mp4
```

Observed WildGS error:

```text
ZeroDivisionError: float division by zero
wildgs_slam/mapper.py line 1102
```

One `postmesh` failure:

```text
/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2/detrac_10_MVI_20062/detrac_10_MVI_20062_part012.mp4
```

Cause:

```text
DuckDB catalog lock conflict on /root/autodl-tmp/guanwu_ua_detrac_batch_workspace/catalog/catalog.duckdb
```

This is not a pose failure. For `part012`, `08_pose_optimize` was complete at latest check:

```text
tasks: 460
results: 460
```

Likely recovery after active writers finish:

- Confirm no process holds `catalog.duckdb`.
- Remove/reconcile only the `postmesh_failed` record for `part012`.
- Rerun catalog or `pose.optimize -> catalog` for that project. Since pose results already exist, avoid deleting 08 unless there is a separate pose-quality reason.

### Recommended Next Step For A New Window

1. Re-check status first; do not trust the counts above if time has passed:

```bash
BATCH_ROOT=/root/autodl-tmp/guanwu_ua_detrac_batch_workspace \
SESSION_NAME=guanwu_ua_detrac_staged_batch \
bash scripts/run_ua_detrac_selected_staged_batch_tmux.sh --status
```

2. Check active processes:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg -i "run_video_pipeline|pose_optimizer|staged_worker|guanwu_ua_detrac" | rg -v "rg -i"
```

3. If the user wants final cleanup/retry:
   - Let the two active postmesh jobs finish or stop them only with explicit permission.
   - Retry `part012` catalog-lock failure separately.
   - Treat the three WildGS `ZeroDivisionError` premesh failures as a separate root-cause/debugging task.

4. For future orientation-flip investigations:
   - Inspect the task JSON for `temporal_prior_pose` and `vehicle_pose_context.temporal_prior_selection`.
   - Distinguish old generated tasks from tasks generated after the executor fix.

## Historical Running State From 2026-06-18 (Stale)

There is an active tmux session:

```text
guanwu_ua_detrac_staged_batch
```

It has 13 windows:

- 6 premesh windows
- 1 mesh window
- 6 postmesh windows

Queue counts at last check:

```text
videos                 57
premesh_done           11
premesh_claimed        8
premesh_failed         1
mesh_done              2
mesh_claimed           3
mesh_failed            0
postmesh_done          0
postmesh_claimed       2
postmesh_failed        0
```

Active pipeline examples at last check:

- `detrac_10_MVI_20062_part001`: running `pose.optimize -> catalog`
- `detrac_10_MVI_20062_part002`: running `pose.optimize -> catalog`
- `detrac_10_MVI_20062_part003`: running `mesh.reconstruct`
- `detrac_10_MVI_20062_part008`: running `video.inspect -> geometry.lift`
- `detrac_10_MVI_20062_part013`: running `video.inspect -> geometry.lift`
- `detrac_45_MVI_40962_part001`: running `video.inspect -> geometry.lift`
- `detrac_45_MVI_40962_part002`: running `video.inspect -> geometry.lift`
- `detrac_45_MVI_40962_part003`: running `video.inspect -> geometry.lift`
- `detrac_45_MVI_40962_part004`: running `video.inspect -> geometry.lift`

Do not assume this is still current in a new window. Re-check with the commands below.

## Important Part001 Finding

The user noticed old-looking results under:

```text
/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/projects/video/ua_detrac_detrac_10_MVI_20062_detrac_10_MVI_20062_part001_edge_quick_balanced_allframes/intermediate
/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/projects/video/ua_detrac_detrac_10_MVI_20062_detrac_10_MVI_20062_part001_edge_quick_balanced_allframes/outputs/07_mesh_reconstruct
```

Investigation result:

- `outputs/07_mesh_reconstruct` was regenerated on 2026-06-18:
  - `mesh_candidate_selection.json`: 10:16
  - `sam3d_meshes.json`: 10:28
- `intermediate/` directory itself has older timestamps because the directory structure was created earlier.
- The mesh files referenced by `sam3d_meshes.json` are new and were generated on 2026-06-18 10:17-10:28.
- Example referenced meshes:
  - `obj_000003`: `intermediate/frame_000041/objects/obj_000003/assets/object_1.glb`
  - `obj_000001`: `intermediate/frame_000001/objects/obj_000001/assets/object_1.glb`
  - `obj_000044`: `intermediate/frame_000011/objects/obj_000044/assets/object_1.glb`

Conclusion:

- Do not delete part001 `intermediate/` or `outputs/07_mesh_reconstruct` while part001 postmesh is running.
- `pose.optimize -> catalog` may depend on these mesh paths.
- If the user requests a strict clean rerun, stop or wait for the relevant postmesh process, then clean 07+ and the specific intermediate object assets safely.

## Background Generation Requirements

The user wanted old road-specific background generation removed and unified with the tabletop-style pipeline:

- `tabletop_task` style OpenAI image edit cleaner.
- DA3 clean depth.
- Support/background reference geometry.
- Road scenes should use a road-specific prompt profile.
- OpenAI should select prompt profile by scene when possible, but the confirmed implementation direction was "方案 A": explicit scene prompt profile selection.

Road prompt requirements:

- Remove all vehicles, road users, distant cars, shadows, reflections, and ghosting.
- Preserve camera perspective, road geometry, lane width, lane markings, curbs, buildings, lighting, and texture.
- Only fill regions occluded by vehicles.
- Avoid over-cleaning and avoid changing environment/geometry.

Important background mesh lesson:

- Visual background mesh should prioritize continuity.
- Do not remove background mesh triangles by absolute long-edge length.
- Keep only lightweight discontinuity filtering by depth ratio, e.g. skip grid cell if `max(z_values) / max(min(z_values), 1e-6) > 1.8`.
- Use linear resize for DA3 clean depth.
- USDC/background/08 support should use consistent DA3 clean background geometry where possible.

## Pose Optimization Lessons

Road scene:

- `edge_contour_fast_quick` is used for vehicles.
- Mesh proxy faces were increased to `15000` because too sparse proxies caused poor contours.
- Issue seen: vehicle pose lagged behind true motion in later frames. Recommended/fixed direction was to strengthen temporal continuity without preventing real motion, and avoid stale prior dominance.
- Issue seen: vehicle head direction flipped suddenly. Check whether temporal orientation prior is active and whether symmetric orientation candidates are being scored without continuity penalty.

Tabletop scene:

- `generic_appearance_temporal` uses mask/bbox/edge/appearance/depth/temporal constraints depending on config.
- DA3 depth alignment was critical: RGB, mask, bbox, K, and depth must be in the same image coordinate system.
- Do not use low-resolution DA3 depth directly with original K/masks.
- Aligned DA3 depth should be stored separately under `depth_anything3/depth_maps_aligned`.
- Depth constraints should generally compare visible/front-facing rendered mesh depth to observed DA3 depth, not blindly the whole mesh.
- Strong depth alignment helps resolve mask/bbox scale ambiguity, but contact/support constraints may need relaxation during grasping/occlusion.
- For generic pose, avoid overfitting to object-specific orientation hacks; prefer simple general constraints like temporal orientation continuity, gravity/up priors when reliable, depth consistency, and better candidate coverage.

## Mesh Reconstruction Selection

The user wanted 07 mesh reconstruction to reconstruct only important foreground movable rigid objects.

Current minimum landing:

- Limit 07 reconstruction object count with Top-K.
- Current road batch Top-K is `8`.
- Selection should prioritize high-confidence, large, persistent foreground/movable rigid objects.
- Avoid reconstructing background/static irrelevant detections.
- Previous cleanup removed projects where `outputs/05_object_attr/object_attrs.json` contained `"class_name": "unknown"`, indicating VLM failure.

## Known Code Changes / Dirty Worktree

At last check the repo had modified files:

```text
configs/car-video2-edge-quick.yaml
src/guanwu/core/config.py
src/guanwu/video/clients/zaiwu.py
src/guanwu/video/core/config.py
src/guanwu/video/executor.py
src/guanwu/video/features/spatial/scene_background_assets.py
src/guanwu/video/infra/isaac_sync.py
src/guanwu/video/project/executor.py
tests/unit/test_config.py
tests/unit/test_video_mesh_reconstruct.py
tests/unit/test_video_pose_temporal.py
tests/unit/test_video_project_config.py
tests/unit/test_zaiwu_gateway.py
scripts/
tests/unit/test_ua_detrac_staged_batch_script.py
```

One important fix already applied:

- `src/guanwu/video/infra/isaac_sync.py`
- `IsaacSyncAgent._init_stage()` was adjusted to handle an existing USD layer registry by using `Sdf.Layer.Find(str(path))`, clearing existing layer contents, and opening/reusing the existing layer rather than always `Usd.Stage.CreateNew`.
- This was meant to avoid USD layer reuse/create conflicts during repeated exports.

Do not run destructive git commands. The user has repeatedly asked to preserve or push current work in some contexts, but also sometimes asks to reset. Confirm before any reset/checkout.

## Useful Monitoring Commands

Check tmux:

```bash
tmux ls
tmux list-windows -t guanwu_ua_detrac_staged_batch
```

Check active Guanwu processes:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'run_video_pipeline.py|staged_worker.sh' | grep -v grep
```

Check queue counts:

```bash
STATE_DIR=/root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_state
for f in videos premesh_done premesh_claimed premesh_failed mesh_done mesh_claimed mesh_failed postmesh_done postmesh_claimed postmesh_failed; do
  p="$STATE_DIR/${f}.txt"
  printf '%-22s ' "$f"
  [ -f "$p" ] && wc -l < "$p" || echo 0
done
```

Tail a project log:

```bash
tail -n 100 /root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_logs/<project_name>.premesh.log
tail -n 100 /root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_logs/<project_name>.mesh.log
tail -n 100 /root/autodl-tmp/guanwu_ua_detrac_batch_workspace/batch_logs/<project_name>.postmesh.log
```

Inspect project stage status:

```bash
sed -n '1,260p' /root/autodl-tmp/guanwu_ua_detrac_batch_workspace/projects/video/<project_name>/state/stage_status.json
```

Inspect SAM3D meshes:

```bash
python - <<'PY'
import json, pathlib, datetime
p = pathlib.Path('/path/to/project/outputs/07_mesh_reconstruct/sam3d_meshes.json')
j = json.loads(p.read_text())
for obj, it in j.items():
    mp = pathlib.Path(it.get('mesh_path', ''))
    if mp.exists():
        st = mp.stat()
        print(obj, datetime.datetime.fromtimestamp(st.st_mtime), st.st_size, mp)
    else:
        print(obj, 'MISSING', mp)
PY
```

## Safe Cleanup Rules

- Never delete `intermediate/` or `outputs/07_mesh_reconstruct/` for a project currently running postmesh.
- Before cleaning a project, verify no process references its project name:

```bash
ps -eo pid,args | grep '<project_name>' | grep -v grep
```

- If strict rerun from 07 is needed:
  - Stop/wait for that project's mesh/postmesh process.
  - Remove `outputs/07_mesh_reconstruct` and later outputs.
  - Remove only relevant `intermediate/frame_*/objects/<obj_id>` assets if needed.
  - Update batch queue state carefully; preserve `videos.txt` and completed state files unless intentionally resetting.

## Recommended Next Step In New Window

1. Re-check current running processes and queue counts.
2. For any user concern about "old results", inspect timestamps of files referenced by manifests/JSON, not just parent directory mtime.
3. If the user asks to stop batch tasks, stop tmux session or selected worker processes deliberately, then reconcile queue claimed/done files.
4. If the user asks to continue monitoring, tail relevant `.postmesh.log`, `.mesh.log`, and `.premesh.log` files and report which projects are running/completed/failed.
