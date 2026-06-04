# Background Geometry Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pose optimization and scene composition share one generic clean-background geometry reference instead of independently trusting per-frame WildGS depth and clean-background depth.

**Architecture:** Keep existing tabletop fields for compatibility, but add a generic `background_geometry_reference` asset and manifest section. Scene compose and pose task context should prefer this reference; older `tabletop_reference` paths remain fallback.

**Tech Stack:** Python, NumPy, OpenCV, pytest, existing Guanwu video pipeline.

---

### Task 1: Manifest Reference Schema

**Files:**
- Modify: `tests/unit/test_scene_background_assets.py`
- Modify: `src/guanwu/video/features/spatial/scene_background_assets.py`

- [ ] **Step 1: Write the failing test**

Add expectations to the clean-depth tabletop task test that a generic background geometry reference is written alongside the legacy tabletop reference.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_scene_background_assets.py::test_generate_tabletop_task_background_assets_writes_tabletop_reference_from_clean_depth -q
```

Expected: FAIL because `assets.background_geometry_reference` is missing.

- [ ] **Step 3: Write minimal implementation**

Change the reference writer to output `background_geometry_reference.json` with schema `guanwu.background_geometry_reference.v1`, a `support_surfaces` list, source metadata, and legacy top-level `normal_world`/`offset` fields for existing consumers.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command and expect PASS.

### Task 2: Executor Reference Loading

**Files:**
- Modify: `tests/unit/test_video_pose_temporal.py`
- Modify: `src/guanwu/video/project/executor.py`

- [ ] **Step 1: Write the failing test**

Add a test manifest with both `background_geometry_reference` and `tabletop_reference`, and assert `ProjectExecutor._tabletop_reference_from_geometry()` returns the generic reference.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/unit/test_video_pose_temporal.py::test_scene_compose_prefers_background_geometry_reference_from_manifest -q
```

Expected: FAIL because the loader still only checks `tabletop_reference`.

- [ ] **Step 3: Write minimal implementation**

Update the loader to check `assets.background_geometry_reference` and manifest-level `background_geometry_reference.path` before legacy fields. Normalize either a top-level plane or the first `support_surfaces` plane into the existing `normal_world`/`offset` shape.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command and expect PASS.

### Task 3: Pose Optimizer Context and Consumption

**Files:**
- Modify: `src/guanwu/video/project/executor.py`
- Modify: `process/pose_optimizer/strategies/generic_appearance_temporal.py`
- Modify: `tests/unit/test_video_pose_temporal.py`

- [ ] **Step 1: Write failing tests**

Add tests for a helper that attaches `background_geometry_reference` into generic pose context and for optimizer support-plane loading from task context.

- [ ] **Step 2: Run tests to verify failure**

Run targeted tests and expect missing helper / unsupported context failures.

- [ ] **Step 3: Write minimal implementation**

Add executor helper to copy the reference into generic task context. Add optimizer loader that converts world-space reference plane into camera-space support plane with confidence metadata, and prefer it over per-frame observed-depth plane when available.

- [ ] **Step 4: Run targeted tests**

Run scene background, pose temporal, and generic pose prior tests.

### Task 4: Verification

**Files:**
- No additional production files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/unit/test_scene_background_assets.py::test_generate_tabletop_task_background_assets_writes_tabletop_reference_from_clean_depth tests/unit/test_video_pose_temporal.py::test_scene_compose_prefers_background_geometry_reference_from_manifest -q
```

- [ ] **Step 2: Run broader affected suites**

Run:

```bash
pytest tests/unit/test_scene_background_assets.py tests/unit/test_video_pose_temporal.py tests/unit/test_video_trajectory_smoothing.py tests/unit/test_generic_pose_priors.py -q
```

- [ ] **Step 3: Inspect generated task context if pipeline is rerun**

Check that `outputs/08_pose_optimize/tasks/obj_000009@000001/task.json` includes `vehicle_pose_context.background_geometry_reference`.
