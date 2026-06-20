#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-fs/Qcp/Guanwu-master}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda/envs/3d_env/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-/root/autodl-tmp/mydataset/UA-DETRAC/videos_train_h264_like_car_video2}"
BATCH_ROOT="${BATCH_ROOT:-/root/autodl-tmp/guanwu_ua_detrac_batch_workspace}"
SESSION_NAME="${SESSION_NAME:-guanwu_ua_detrac_batch}"
WORKER_COUNT="${WORKER_COUNT:-6}"
TO_STAGE="${TO_STAGE:-materialize}"
FORCE="${FORCE:-1}"

CONFIG_SRC="${CONFIG_SRC:-$REPO_ROOT/configs/car-video2-edge-quick.yaml}"
CONFIG_DST="$BATCH_ROOT/car-video2-edge-quick-batch.yaml"
STATE_DIR="$BATCH_ROOT/batch_state"
LOG_DIR="$BATCH_ROOT/batch_logs"
SERVICE_LOCK_DIR="$BATCH_ROOT/service_locks"
VIDEOS_FILE="$STATE_DIR/videos.txt"
DONE_FILE="$STATE_DIR/done.txt"
FAILED_FILE="$STATE_DIR/failed.txt"
CLAIMED_FILE="$STATE_DIR/claimed.txt"
QUEUE_LOCK="$STATE_DIR/queue.lock"

SELECTED_DIRS=(
  "detrac_10_MVI_20062"
  "detrac_45_MVI_40962"
  "detrac_47_MVI_40981"
  "detrac_53_MVI_63525"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [--prepare-only|--start|--status]

Environment overrides:
  BATCH_ROOT=$BATCH_ROOT
  SESSION_NAME=$SESSION_NAME
  WORKER_COUNT=$WORKER_COUNT
  TO_STAGE=$TO_STAGE
  FORCE=$FORCE
EOF
}

prepare_workspace() {
  mkdir -p "$BATCH_ROOT" "$STATE_DIR" "$LOG_DIR" "$SERVICE_LOCK_DIR"
  mkdir -p "$BATCH_ROOT/projects" "$BATCH_ROOT/raw" "$BATCH_ROOT/staging" "$BATCH_ROOT/canonical" "$BATCH_ROOT/exports" "$BATCH_ROOT/catalog"
  : > "$DONE_FILE"
  : > "$FAILED_FILE"
  : > "$CLAIMED_FILE"
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

  : > "$VIDEOS_FILE"
  for dir_name in "${SELECTED_DIRS[@]}"; do
    if [[ ! -d "$INPUT_ROOT/$dir_name" ]]; then
      echo "[batch] Missing selected input dir: $INPUT_ROOT/$dir_name" >&2
      exit 1
    fi
    find "$INPUT_ROOT/$dir_name" -type f -iname '*.mp4' | sort >> "$VIDEOS_FILE"
  done
  sort -u "$VIDEOS_FILE" -o "$VIDEOS_FILE"
  echo "[batch] Prepared $BATCH_ROOT"
  echo "[batch] Config: $CONFIG_DST"
  echo "[batch] Videos: $(wc -l < "$VIDEOS_FILE")"
}

worker_script() {
  cat <<'EOS'
set -euo pipefail
WORKER_ID="$1"
GPU_ID="$2"
export WORKER_ID GPU_ID

next_video() {
  flock "$QUEUE_LOCK" bash -c '
    set -euo pipefail
    while IFS= read -r video; do
      [[ -z "$video" ]] && continue
      grep -Fxq "$video" "$DONE_FILE" 2>/dev/null && continue
      cut -f1 "$FAILED_FILE" 2>/dev/null | grep -Fxq "$video" && continue
      grep -Fxq "$video" "$CLAIMED_FILE" 2>/dev/null && continue
      printf "%s\n" "$video" >> "$CLAIMED_FILE"
      printf "%s\n" "$video"
      exit 0
    done < "$VIDEOS_FILE"
  '
}

project_name_for_video() {
  local video="$1"
  local parent stem safe_parent safe_stem
  parent="$(basename "$(dirname "$video")")"
  stem="$(basename "$video" .mp4)"
  safe_parent="$(printf "%s" "$parent" | tr -c "A-Za-z0-9._-" "_")"
  safe_stem="$(printf "%s" "$stem" | tr -c "A-Za-z0-9._-" "_")"
  printf "ua_detrac_%s_%s_edge_quick_balanced_allframes" "$safe_parent" "$safe_stem"
}

while true; do
  video_path="$(next_video || true)"
  if [[ -z "${video_path:-}" ]]; then
    echo "[worker $WORKER_ID] queue empty; exiting"
    exit 0
  fi

  project_name="$(project_name_for_video "$video_path")"
  log_path="$LOG_DIR/${project_name}.log"
  echo "[worker $WORKER_ID][gpu $GPU_ID] start $project_name"
  echo "[worker $WORKER_ID] video=$video_path" > "$log_path"
  start_ts="$(date +%s)"

  force_args=()
  if [[ "${FORCE:-1}" == "1" ]]; then
    force_args=(--force)
  fi

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
  GUANWU_ZAIWU_LOCK_SERVICES="services.sam3d,services.depth_anything3,services.wildgs_slam,services.seg2track_sam2,services.grounding_dino_sam2,services.gotrack" \
  GUANWU_OPENAI_IMAGE_EDIT_LOCK=1 \
  GUANWU_OPENAI_IMAGE_EDIT_LOCK_PATH="$SERVICE_LOCK_DIR/openai_image_edit.lock" \
  "$PYTHON_BIN" "$REPO_ROOT/run_video_pipeline.py" \
    --video "$video_path" \
    --config "$CONFIG_DST" \
    --project-name "$project_name" \
    --from-stage video.inspect \
    --to-stage "$TO_STAGE" \
    "${force_args[@]}" >> "$log_path" 2>&1
  status=$?
  set -e

  elapsed=$(( $(date +%s) - start_ts ))
  flock "$QUEUE_LOCK" bash -c '
    if [[ "$0" == "0" ]]; then
      printf "%s\n" "$1" >> "$DONE_FILE"
    else
      printf "%s\t%s\t%s\n" "$1" "$0" "$2" >> "$FAILED_FILE"
    fi
  ' "$status" "$video_path" "$elapsed"

  if [[ "$status" == "0" ]]; then
    echo "[worker $WORKER_ID] done $project_name in ${elapsed}s"
  else
    echo "[worker $WORKER_ID] failed $project_name status=$status elapsed=${elapsed}s; see $log_path"
  fi
done
EOS
}

start_tmux() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[batch] tmux session already exists: $SESSION_NAME" >&2
    echo "[batch] Attach: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi

  local worker_file="$STATE_DIR/worker.sh"
  worker_script > "$worker_file"
  chmod +x "$worker_file"

  tmux new-session -d -s "$SESSION_NAME" -n "worker-0" \
    "cd '$REPO_ROOT' && REPO_ROOT='$REPO_ROOT' PYTHON_BIN='$PYTHON_BIN' CONFIG_DST='$CONFIG_DST' STATE_DIR='$STATE_DIR' LOG_DIR='$LOG_DIR' SERVICE_LOCK_DIR='$SERVICE_LOCK_DIR' VIDEOS_FILE='$VIDEOS_FILE' DONE_FILE='$DONE_FILE' FAILED_FILE='$FAILED_FILE' CLAIMED_FILE='$CLAIMED_FILE' QUEUE_LOCK='$QUEUE_LOCK' TO_STAGE='$TO_STAGE' FORCE='$FORCE' bash '$worker_file' 0 0"

  local i
  for ((i=1; i<WORKER_COUNT; i++)); do
    tmux new-window -t "$SESSION_NAME" -n "worker-$i" \
      "cd '$REPO_ROOT' && REPO_ROOT='$REPO_ROOT' PYTHON_BIN='$PYTHON_BIN' CONFIG_DST='$CONFIG_DST' STATE_DIR='$STATE_DIR' LOG_DIR='$LOG_DIR' SERVICE_LOCK_DIR='$SERVICE_LOCK_DIR' VIDEOS_FILE='$VIDEOS_FILE' DONE_FILE='$DONE_FILE' FAILED_FILE='$FAILED_FILE' CLAIMED_FILE='$CLAIMED_FILE' QUEUE_LOCK='$QUEUE_LOCK' TO_STAGE='$TO_STAGE' FORCE='$FORCE' bash '$worker_file' '$i' '$i'"
  done

  echo "[batch] Started tmux session: $SESSION_NAME"
  echo "[batch] Attach: tmux attach -t $SESSION_NAME"
  echo "[batch] Logs: $LOG_DIR"
}

status_batch() {
  echo "[batch] session:"
  tmux ls 2>/dev/null | grep -F "$SESSION_NAME" || true
  echo "[batch] queued: $(wc -l < "$VIDEOS_FILE" 2>/dev/null || echo 0)"
  echo "[batch] done: $(wc -l < "$DONE_FILE" 2>/dev/null || echo 0)"
  echo "[batch] failed: $(wc -l < "$FAILED_FILE" 2>/dev/null || echo 0)"
  echo "[batch] claimed: $(wc -l < "$CLAIMED_FILE" 2>/dev/null || echo 0)"
}

cmd="${1:---start}"
case "$cmd" in
  --prepare-only)
    prepare_workspace
    ;;
  --start)
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
