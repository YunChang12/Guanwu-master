#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-fs/Qcp/Guanwu-master}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/3d_env/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2}"
BATCH_ROOT="${BATCH_ROOT:-/root/autodl-tmp/guanwu_ua_detrac_staged_batch_workspace}"
SESSION_NAME="${SESSION_NAME:-guanwu_ua_detrac_staged_batch}"
PREMESH_WORKER_COUNT="${PREMESH_WORKER_COUNT:-6}"
MESH_WORKER_COUNT="${MESH_WORKER_COUNT:-1}"
POSTMESH_WORKER_COUNT="${POSTMESH_WORKER_COUNT:-6}"
TO_STAGE="${TO_STAGE:-catalog}"
FORCE="${FORCE:-1}"
RESET_STATE="${RESET_STATE:-1}"
SEED_POSTMESH_PROJECTS_FILE="${SEED_POSTMESH_PROJECTS_FILE:-}"
VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-}"

PREMESH_FROM_STAGE="${PREMESH_FROM_STAGE:-video.inspect}"
PREMESH_TO_STAGE="${PREMESH_TO_STAGE:-geometry.lift}"
MESH_FROM_STAGE="${MESH_FROM_STAGE:-mesh.reconstruct}"
MESH_TO_STAGE="${MESH_TO_STAGE:-mesh.reconstruct}"
POSTMESH_FROM_STAGE="${POSTMESH_FROM_STAGE:-pose.optimize}"
POSTMESH_TO_STAGE="${POSTMESH_TO_STAGE:-catalog}"

CONFIG_SRC="${CONFIG_SRC:-$REPO_ROOT/configs/car-video2-edge-quick.yaml}"
CONFIG_DST="$BATCH_ROOT/car-video2-edge-quick-staged-batch.yaml"
STATE_DIR="$BATCH_ROOT/batch_state"
LOG_DIR="$BATCH_ROOT/batch_logs"
SERVICE_LOCK_DIR="$BATCH_ROOT/service_locks"
VIDEOS_FILE="$STATE_DIR/videos.txt"
QUEUE_LOCK="$STATE_DIR/queue.lock"

SELECTED_DIRS=(
  "detrac_10_MVI_20062"
  "detrac_45_MVI_40962"
  "detrac_47_MVI_40981"
  "detrac_53_MVI_63525"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [--prepare-only|--start|--resume|--status]

Staged scheduling:
  premesh:  $PREMESH_FROM_STAGE -> $PREMESH_TO_STAGE, workers=$PREMESH_WORKER_COUNT
  mesh:     $MESH_FROM_STAGE -> $MESH_TO_STAGE, workers=$MESH_WORKER_COUNT
  postmesh: $POSTMESH_FROM_STAGE -> $POSTMESH_TO_STAGE, workers=$POSTMESH_WORKER_COUNT

Environment overrides:
  BATCH_ROOT=$BATCH_ROOT
  SESSION_NAME=$SESSION_NAME
  PREMESH_WORKER_COUNT=$PREMESH_WORKER_COUNT
  MESH_WORKER_COUNT=$MESH_WORKER_COUNT
  POSTMESH_WORKER_COUNT=$POSTMESH_WORKER_COUNT
  FORCE=$FORCE
  RESET_STATE=$RESET_STATE
  SEED_POSTMESH_PROJECTS_FILE=$SEED_POSTMESH_PROJECTS_FILE
  VIDEO_LIST_FILE=$VIDEO_LIST_FILE
EOF
}

prepare_workspace() {
  mkdir -p "$BATCH_ROOT" "$STATE_DIR" "$LOG_DIR" "$SERVICE_LOCK_DIR"
  mkdir -p "$BATCH_ROOT/projects" "$BATCH_ROOT/raw" "$BATCH_ROOT/staging" "$BATCH_ROOT/canonical" "$BATCH_ROOT/exports" "$BATCH_ROOT/catalog"
  : > "$QUEUE_LOCK"

  python - "$CONFIG_SRC" "$CONFIG_DST" "$BATCH_ROOT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
batch_root = Path(sys.argv[3])
data = yaml.safe_load(src.read_text(encoding="utf-8"))
data["workspace_root"] = str(batch_root)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

  if [[ -n "$VIDEO_LIST_FILE" ]]; then
    if [[ ! -f "$VIDEO_LIST_FILE" ]]; then
      echo "[staged-batch] Missing VIDEO_LIST_FILE: $VIDEO_LIST_FILE" >&2
      exit 1
    fi
    sed '/^[[:space:]]*$/d' "$VIDEO_LIST_FILE" | sort -u > "$VIDEOS_FILE"
  elif [[ -f "$STATE_DIR/current_existing_project_videos.txt" ]]; then
    sed '/^[[:space:]]*$/d' "$STATE_DIR/current_existing_project_videos.txt" | sort -u > "$VIDEOS_FILE"
  else
    : > "$VIDEOS_FILE"
    for dir_name in "${SELECTED_DIRS[@]}"; do
      if [[ ! -d "$INPUT_ROOT/$dir_name" ]]; then
        echo "[staged-batch] Missing selected input dir: $INPUT_ROOT/$dir_name" >&2
        exit 1
      fi
      find "$INPUT_ROOT/$dir_name" -type f -iname '*.mp4' | sort >> "$VIDEOS_FILE"
    done
    sort -u "$VIDEOS_FILE" -o "$VIDEOS_FILE"
  fi

  if [[ "$RESET_STATE" == "1" ]]; then
    for stage in premesh mesh postmesh; do
      : > "$STATE_DIR/${stage}_claimed.txt"
      : > "$STATE_DIR/${stage}_done.txt"
      : > "$STATE_DIR/${stage}_failed.txt"
    done
  else
    for stage in premesh mesh postmesh; do
      touch "$STATE_DIR/${stage}_claimed.txt" "$STATE_DIR/${stage}_done.txt" "$STATE_DIR/${stage}_failed.txt"
      release_incomplete_claims "$stage"
    done
  fi

  if [[ -n "$SEED_POSTMESH_PROJECTS_FILE" ]]; then
    seed_postmesh_from_projects_file "$SEED_POSTMESH_PROJECTS_FILE"
  fi

  echo "[staged-batch] Prepared $BATCH_ROOT"
  echo "[staged-batch] Config: $CONFIG_DST"
  echo "[staged-batch] Videos: $(wc -l < "$VIDEOS_FILE")"
}

project_name_for_video_path() {
  local video="$1"
  local parent stem safe_parent safe_stem
  parent="$(basename "$(dirname "$video")")"
  stem="$(basename "$video" .mp4)"
  safe_parent="$(printf "%s" "$parent" | tr -c "A-Za-z0-9._-" "_")"
  safe_stem="$(printf "%s" "$stem" | tr -c "A-Za-z0-9._-" "_")"
  printf "ua_detrac_%s_%s_edge_quick_balanced_allframes" "$safe_parent" "$safe_stem"
}

seed_postmesh_from_projects_file() {
  local seed_file="$1"
  if [[ ! -f "$seed_file" ]]; then
    echo "[staged-batch] Missing SEED_POSTMESH_PROJECTS_FILE: $seed_file" >&2
    exit 1
  fi

  while IFS= read -r project_name; do
    [[ -z "$project_name" ]] && continue
    while IFS= read -r video_path; do
      [[ -z "$video_path" ]] && continue
      if [[ "$(project_name_for_video_path "$video_path")" != "$project_name" ]]; then
        continue
      fi
      grep -Fxq "$video_path" "$STATE_DIR/premesh_done.txt" 2>/dev/null || printf "%s\n" "$video_path" >> "$STATE_DIR/premesh_done.txt"
      grep -Fxq "$video_path" "$STATE_DIR/mesh_done.txt" 2>/dev/null || printf "%s\n" "$video_path" >> "$STATE_DIR/mesh_done.txt"
      remove_video_from_stage_state "postmesh" "$video_path"
      break
    done < "$VIDEOS_FILE"
  done < "$seed_file"
}

remove_video_from_stage_state() {
  local stage="$1"
  local video_path="$2"
  local claimed_file="$STATE_DIR/${stage}_claimed.txt"
  local failed_file="$STATE_DIR/${stage}_failed.txt"
  local claimed_tmp="$STATE_DIR/${stage}_claimed.txt.tmp"
  local failed_tmp="$STATE_DIR/${stage}_failed.txt.tmp"

  if [[ -f "$claimed_file" ]]; then
    grep -Fxv "$video_path" "$claimed_file" > "$claimed_tmp" || true
    mv "$claimed_tmp" "$claimed_file"
  fi
  if [[ -f "$failed_file" ]]; then
    awk -F '\t' -v video="$video_path" '$1 != video { print }' "$failed_file" > "$failed_tmp"
    mv "$failed_tmp" "$failed_file"
  fi
}

release_incomplete_claims() {
  local stage="$1"
  local claimed_file="$STATE_DIR/${stage}_claimed.txt"
  local done_file="$STATE_DIR/${stage}_done.txt"
  local failed_file="$STATE_DIR/${stage}_failed.txt"
  local done_or_failed="$STATE_DIR/${stage}_done_or_failed.tmp"
  local claimed_tmp="$STATE_DIR/${stage}_claimed.txt.tmp"

  [[ -f "$claimed_file" ]] || return 0
  {
    cat "$done_file" 2>/dev/null || true
    cut -f1 "$failed_file" 2>/dev/null || true
  } | sed '/^$/d' | sort -u > "$done_or_failed"
  grep -Fxf "$done_or_failed" "$claimed_file" > "$claimed_tmp" || true
  mv "$claimed_tmp" "$claimed_file"
  rm -f "$done_or_failed"
}

worker_script() {
  cat <<'EOS'
set -euo pipefail

STAGE_KIND="$1"
WORKER_ID="$2"
GPU_ID="$3"
FROM_STAGE="$4"
TO_STAGE="$5"
UPSTREAM_KIND="${6:-}"
export STAGE_KIND WORKER_ID GPU_ID FROM_STAGE TO_STAGE UPSTREAM_KIND

project_name_for_video() {
  local video="$1"
  local parent stem safe_parent safe_stem
  parent="$(basename "$(dirname "$video")")"
  stem="$(basename "$video" .mp4)"
  safe_parent="$(printf "%s" "$parent" | tr -c "A-Za-z0-9._-" "_")"
  safe_stem="$(printf "%s" "$stem" | tr -c "A-Za-z0-9._-" "_")"
  printf "ua_detrac_%s_%s_edge_quick_balanced_allframes" "$safe_parent" "$safe_stem"
}

count_file_lines() {
  local path="$1"
  wc -l < "$path" 2>/dev/null || echo 0
}

upstream_total() {
  if [[ -z "$UPSTREAM_KIND" ]]; then
    count_file_lines "$VIDEOS_FILE"
  else
    local done failed
    done="$(count_file_lines "$STATE_DIR/${UPSTREAM_KIND}_done.txt")"
    failed="$(count_file_lines "$STATE_DIR/${UPSTREAM_KIND}_failed.txt")"
    echo $(( done + failed ))
  fi
}

upstream_closed() {
  local total upstream
  total="$(count_file_lines "$VIDEOS_FILE")"
  upstream="$(upstream_total)"
  [[ "$upstream" -ge "$total" ]]
}

clear_stale_project_lock() {
  local project_name="$1"
  local lock_path
  lock_path="$BATCH_ROOT/projects/video/${project_name}/.project.lock"
  [[ -f "$lock_path" ]] || return 0

  local lock_pid
  lock_pid="$(cat "$lock_path" 2>/dev/null || true)"
  if [[ "$lock_pid" =~ ^[0-9]+$ ]] && kill -0 "$lock_pid" 2>/dev/null; then
    return 0
  fi

  echo "[${STAGE_KIND} worker $WORKER_ID] removing stale project lock: $lock_path pid=${lock_pid:-unknown}"
  rm -f "$lock_path"
}

next_video() {
  flock "$QUEUE_LOCK" bash -c '
    set -euo pipefail
    stage_kind="$1"
    upstream_kind="$2"
    if [[ -z "$upstream_kind" ]]; then
      source_file="$VIDEOS_FILE"
    else
      source_file="$STATE_DIR/${upstream_kind}_done.txt"
    fi
    claimed_file="$STATE_DIR/${stage_kind}_claimed.txt"
    done_file="$STATE_DIR/${stage_kind}_done.txt"
    failed_file="$STATE_DIR/${stage_kind}_failed.txt"

    while IFS= read -r video; do
      [[ -z "$video" ]] && continue
      grep -Fxq "$video" "$done_file" 2>/dev/null && continue
      cut -f1 "$failed_file" 2>/dev/null | grep -Fxq "$video" && continue
      grep -Fxq "$video" "$claimed_file" 2>/dev/null && continue
      printf "%s\n" "$video" >> "$claimed_file"
      printf "%s\n" "$video"
      exit 0
    done < "$source_file"
  ' _ "$STAGE_KIND" "$UPSTREAM_KIND"
}

run_stage_worker() {
  echo "[${STAGE_KIND} worker $WORKER_ID][gpu $GPU_ID] range ${FROM_STAGE} -> ${TO_STAGE}"
  while true; do
    video_path="$(next_video || true)"
    if [[ -z "${video_path:-}" ]]; then
      if upstream_closed; then
        echo "[${STAGE_KIND} worker $WORKER_ID] queue drained; exiting"
        exit 0
      fi
      sleep 10
      continue
    fi

    project_name="$(project_name_for_video "$video_path")"
    log_path="$LOG_DIR/${project_name}.${STAGE_KIND}.log"
    echo "[${STAGE_KIND} worker $WORKER_ID][gpu $GPU_ID] start $project_name"
    clear_stale_project_lock "$project_name"
    {
      echo "stage_kind=$STAGE_KIND"
      echo "worker=$WORKER_ID"
      echo "gpu=$GPU_ID"
      echo "range=$FROM_STAGE -> $TO_STAGE"
      echo "video=$video_path"
    } > "$log_path"
    start_ts="$(date +%s)"

    force_args=()
    if [[ "${FORCE:-1}" == "1" ]]; then
      force_args=(--force)
    fi

    # These locks apply to all stages, including 6-way premesh workers.
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYOPENGL_PLATFORM=egl \
    EGL_DEVICE_ID=0 \
    PYTHONUNBUFFERED=1 \
    GUANWU_POSE_OPTIMIZER_MODE=edge_contour_fast_quick \
    GUANWU_POSE_TARGET_FRAME_MODE=all_frames \
    GUANWU_ZAIWU_SERVICE_LOCKS=1 \
    GUANWU_SAM3D_STAGE_LOCK=1 \
    GUANWU_ZAIWU_SERVICE_LOCK_DIR="$SERVICE_LOCK_DIR" \
    GUANWU_ZAIWU_LOCK_SERVICES="services.sam3,services.sam3d,services.depth_anything3,services.wildgs_slam,services.seg2track_sam2,services.grounding_dino_sam2,services.gotrack" \
    GUANWU_OPENAI_IMAGE_EDIT_LOCK=1 \
    GUANWU_OPENAI_IMAGE_EDIT_LOCK_PATH="$SERVICE_LOCK_DIR/openai_image_edit.lock" \
    "$PYTHON_BIN" "$REPO_ROOT/run_video_pipeline.py" \
      --video "$video_path" \
      --config "$CONFIG_DST" \
      --project-name "$project_name" \
      --from-stage "$FROM_STAGE" \
      --to-stage "$TO_STAGE" \
      "${force_args[@]}" >> "$log_path" 2>&1
    status=$?
    set -e

    elapsed=$(( $(date +%s) - start_ts ))
    flock "$QUEUE_LOCK" bash -c '
      status="$1"
      video="$2"
      elapsed="$3"
      stage_kind="$4"
      if [[ "$status" == "0" ]]; then
        printf "%s\n" "$video" >> "$STATE_DIR/${stage_kind}_done.txt"
      else
        printf "%s\t%s\t%s\n" "$video" "$status" "$elapsed" >> "$STATE_DIR/${stage_kind}_failed.txt"
      fi
    ' _ "$status" "$video_path" "$elapsed" "$STAGE_KIND"

    if [[ "$status" == "0" ]]; then
      echo "[${STAGE_KIND} worker $WORKER_ID] done $project_name in ${elapsed}s"
    else
      echo "[${STAGE_KIND} worker $WORKER_ID] failed $project_name status=$status elapsed=${elapsed}s; see $log_path"
    fi
  done
}

run_stage_worker
EOS
}

start_tmux() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[staged-batch] tmux session already exists: $SESSION_NAME" >&2
    echo "[staged-batch] Attach: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi

  local worker_file="$STATE_DIR/staged_worker.sh"
  worker_script > "$worker_file"
  chmod +x "$worker_file"

  local window_index=0
  local i gpu

  for ((i=0; i<PREMESH_WORKER_COUNT; i++)); do
    gpu=$(( i % 6 ))
    if [[ "$window_index" -eq 0 ]]; then
      tmux new-session -d -s "$SESSION_NAME" -n "premesh-$i" \
        "cd '$REPO_ROOT' && REPO_ROOT='$REPO_ROOT' BATCH_ROOT='$BATCH_ROOT' PYTHON_BIN='$PYTHON_BIN' CONFIG_DST='$CONFIG_DST' STATE_DIR='$STATE_DIR' LOG_DIR='$LOG_DIR' SERVICE_LOCK_DIR='$SERVICE_LOCK_DIR' VIDEOS_FILE='$VIDEOS_FILE' QUEUE_LOCK='$QUEUE_LOCK' FORCE='$FORCE' bash '$worker_file' premesh '$i' '$gpu' '$PREMESH_FROM_STAGE' '$PREMESH_TO_STAGE' ''"
    else
      tmux new-window -t "$SESSION_NAME" -n "premesh-$i" \
        "cd '$REPO_ROOT' && REPO_ROOT='$REPO_ROOT' BATCH_ROOT='$BATCH_ROOT' PYTHON_BIN='$PYTHON_BIN' CONFIG_DST='$CONFIG_DST' STATE_DIR='$STATE_DIR' LOG_DIR='$LOG_DIR' SERVICE_LOCK_DIR='$SERVICE_LOCK_DIR' VIDEOS_FILE='$VIDEOS_FILE' QUEUE_LOCK='$QUEUE_LOCK' FORCE='$FORCE' bash '$worker_file' premesh '$i' '$gpu' '$PREMESH_FROM_STAGE' '$PREMESH_TO_STAGE' ''"
    fi
    window_index=$((window_index + 1))
  done

  for ((i=0; i<MESH_WORKER_COUNT; i++)); do
    gpu=$(( (PREMESH_WORKER_COUNT + i) % 6 ))
    tmux new-window -t "$SESSION_NAME" -n "mesh-$i" \
      "cd '$REPO_ROOT' && REPO_ROOT='$REPO_ROOT' BATCH_ROOT='$BATCH_ROOT' PYTHON_BIN='$PYTHON_BIN' CONFIG_DST='$CONFIG_DST' STATE_DIR='$STATE_DIR' LOG_DIR='$LOG_DIR' SERVICE_LOCK_DIR='$SERVICE_LOCK_DIR' VIDEOS_FILE='$VIDEOS_FILE' QUEUE_LOCK='$QUEUE_LOCK' FORCE='$FORCE' bash '$worker_file' mesh '$i' '$gpu' '$MESH_FROM_STAGE' '$MESH_TO_STAGE' premesh"
  done

  for ((i=0; i<POSTMESH_WORKER_COUNT; i++)); do
    gpu=$(( i % 6 ))
    tmux new-window -t "$SESSION_NAME" -n "postmesh-$i" \
      "cd '$REPO_ROOT' && REPO_ROOT='$REPO_ROOT' BATCH_ROOT='$BATCH_ROOT' PYTHON_BIN='$PYTHON_BIN' CONFIG_DST='$CONFIG_DST' STATE_DIR='$STATE_DIR' LOG_DIR='$LOG_DIR' SERVICE_LOCK_DIR='$SERVICE_LOCK_DIR' VIDEOS_FILE='$VIDEOS_FILE' QUEUE_LOCK='$QUEUE_LOCK' FORCE='$FORCE' bash '$worker_file' postmesh '$i' '$gpu' '$POSTMESH_FROM_STAGE' '$POSTMESH_TO_STAGE' mesh"
  done

  echo "[staged-batch] Started tmux session: $SESSION_NAME"
  echo "[staged-batch] Attach: tmux attach -t $SESSION_NAME"
  echo "[staged-batch] Logs: $LOG_DIR"
}

status_batch() {
  echo "[staged-batch] session:"
  tmux ls 2>/dev/null | grep -F "$SESSION_NAME" || true
  echo "[staged-batch] queued: $(wc -l < "$VIDEOS_FILE" 2>/dev/null || echo 0)"
  for stage in premesh mesh postmesh; do
    echo "[staged-batch] ${stage}.claimed: $(wc -l < "$STATE_DIR/${stage}_claimed.txt" 2>/dev/null || echo 0)"
    echo "[staged-batch] ${stage}.done: $(wc -l < "$STATE_DIR/${stage}_done.txt" 2>/dev/null || echo 0)"
    echo "[staged-batch] ${stage}.failed: $(wc -l < "$STATE_DIR/${stage}_failed.txt" 2>/dev/null || echo 0)"
  done
}

cmd="${1:---start}"
case "$cmd" in
  --prepare-only)
    prepare_workspace
    ;;
  --start)
    RESET_STATE="${RESET_STATE:-1}"
    prepare_workspace
    start_tmux
    ;;
  --resume)
    RESET_STATE=0
    prepare_workspace
    start_tmux
    ;;
  --status)
    status_batch
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
