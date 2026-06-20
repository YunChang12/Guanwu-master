from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_ua_detrac_selected_staged_batch_tmux.sh"


def test_staged_batch_script_splits_pipeline_around_mesh_reconstruct() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'PREMESH_FROM_STAGE="${PREMESH_FROM_STAGE:-video.inspect}"' in script
    assert 'PREMESH_TO_STAGE="${PREMESH_TO_STAGE:-geometry.lift}"' in script
    assert 'MESH_FROM_STAGE="${MESH_FROM_STAGE:-mesh.reconstruct}"' in script
    assert 'MESH_TO_STAGE="${MESH_TO_STAGE:-mesh.reconstruct}"' in script
    assert 'POSTMESH_FROM_STAGE="${POSTMESH_FROM_STAGE:-pose.optimize}"' in script
    assert 'POSTMESH_TO_STAGE="${POSTMESH_TO_STAGE:-catalog}"' in script


def test_staged_batch_script_runs_mesh_with_single_stage_worker_and_sam3d_stage_lock() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'MESH_WORKER_COUNT="${MESH_WORKER_COUNT:-1}"' in script
    assert 'GUANWU_SAM3D_STAGE_LOCK=1' in script
    assert "bash '$worker_file' mesh" in script
    assert '"$PYTHON_BIN" "$REPO_ROOT/run_video_pipeline.py"' in script


def test_staged_batch_script_serializes_zaiwu_services_used_by_premesh_workers() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'PREMESH_WORKER_COUNT="${PREMESH_WORKER_COUNT:-6}"' in script
    assert 'GUANWU_ZAIWU_SERVICE_LOCKS=1' in script
    assert 'GUANWU_ZAIWU_SERVICE_LOCK_DIR="$SERVICE_LOCK_DIR"' in script
    assert 'GUANWU_OPENAI_IMAGE_EDIT_LOCK=1' in script
    match = re.search(r'GUANWU_ZAIWU_LOCK_SERVICES="([^"]+)"', script)
    assert match is not None
    locked_services = {item.strip() for item in match.group(1).split(",")}
    for service_id in (
        "services.sam3",
        "services.seg2track_sam2",
        "services.grounding_dino_sam2",
        "services.wildgs_slam",
        "services.depth_anything3",
    ):
        assert service_id in locked_services


def test_staged_batch_script_uses_independent_session_and_workspace() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'BATCH_ROOT="${BATCH_ROOT:-/root/autodl-tmp/guanwu_ua_detrac_staged_batch_workspace}"' in script
    assert 'SESSION_NAME="${SESSION_NAME:-guanwu_ua_detrac_staged_batch}"' in script
    assert "guanwu_ua_detrac_batch" not in script


def test_staged_batch_script_can_resume_without_resetting_stage_state() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--resume" in script
    assert 'RESET_STATE="${RESET_STATE:-1}"' in script
    assert 'if [[ "$RESET_STATE" == "1" ]]; then' in script


def test_staged_batch_script_can_seed_postmesh_from_existing_projects() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'SEED_POSTMESH_PROJECTS_FILE="${SEED_POSTMESH_PROJECTS_FILE:-}"' in script
    assert "seed_postmesh_from_projects_file" in script
    assert 'printf "%s\\n" "$video_path" >> "$STATE_DIR/mesh_done.txt"' in script


def test_staged_batch_script_can_use_explicit_video_list_file() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-}"' in script
    assert 'if [[ -n "$VIDEO_LIST_FILE" ]]; then' in script
    assert 'sed \'/^[[:space:]]*$/d\' "$VIDEO_LIST_FILE" | sort -u > "$VIDEOS_FILE"' in script
    assert 'elif [[ -f "$STATE_DIR/current_existing_project_videos.txt" ]]; then' in script


def test_staged_batch_script_can_retry_seeded_postmesh_failures() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "remove_video_from_stage_state" in script
    assert 'remove_video_from_stage_state "postmesh" "$video_path"' in script
    assert 'failed_tmp="$STATE_DIR/${stage}_failed.txt.tmp"' in script


def test_staged_batch_worker_cleans_only_dead_project_locks() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "clear_stale_project_lock" in script
    assert 'lock_pid="$(cat "$lock_path" 2>/dev/null || true)"' in script
    assert 'kill -0 "$lock_pid"' in script
    assert 'rm -f "$lock_path"' in script


def test_staged_batch_resume_releases_claimed_only_items() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "release_incomplete_claims" in script
    assert 'release_incomplete_claims "$stage"' in script
    assert 'done_or_failed="$STATE_DIR/${stage}_done_or_failed.tmp"' in script
