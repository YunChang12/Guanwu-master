"""
One-click video pipeline runner.

Usage:
    python run_video_pipeline.py                          # use default video (workspace/raw/video.mp4)
    python run_video_pipeline.py --video path/to/video.mp4
    python run_video_pipeline.py --project-name my_run    # custom project name
    python run_video_pipeline.py --from-stage video.inspect --to-stage scene.export
    python run_video_pipeline.py --materialize            # also run materialize + catalog after scene.export
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "workspace.yaml"
DEFAULT_VIDEO = PROJECT_ROOT / "workspace" / "raw" / "video.mp4"
DEFAULT_FROM = "video.inspect"
DEFAULT_TO = "scene.export"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Guanwu video pipeline end-to-end")
    p.add_argument("--video", type=str, default=str(DEFAULT_VIDEO),
                    help="Input video file path")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                    help="Workspace config YAML")
    p.add_argument("--project-name", type=str, default="demo",
                    help="Project name / ID")
    p.add_argument("--from-stage", type=str, default=DEFAULT_FROM,
                    help="Start stage (default: video.inspect)")
    p.add_argument("--to-stage", type=str, default=DEFAULT_TO,
                    help="End stage (default: scene.export)")
    p.add_argument("--materialize", action="store_true",
                    help="Also run materialize + catalog stages after the main range")
    p.add_argument("--force", action="store_true",
                    help="Force rerun all stages from scratch")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── validate inputs ─────────────────────────────────────────────────────
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    # ── load workspace config ───────────────────────────────────────────────
    from guanwu.core.config import load_config
    cfg = load_config(str(config_path))

    project_root = Path(cfg.storage.project_root) / "video" / args.project_name

    # ── step 1: init project ────────────────────────────────────────────────
    from guanwu.video.executor import VideoProjectExecutor, ensure_video_project

    print("=" * 60)
    print(f"  Guanwu Video Pipeline - One Click Runner")
    print("=" * 60)
    print(f"  Video:      {video_path}")
    print(f"  Config:     {config_path}")
    print(f"  Project:    {project_root}")
    print(f"  Stages:     {args.from_stage} -> {args.to_stage}")
    print(f"  Provider:   {cfg.video_pipeline.provider_mode}")
    print(f"  Detection:  {cfg.video_pipeline.object_detection_backend}")
    print(f"  Camera:     {cfg.video_pipeline.camera_provider}")
    print(f"  Force:      {args.force}")
    print(f"  Materialize: {args.materialize}")
    print("=" * 60)

    # init or reuse project
    context = ensure_video_project(
        project_root=str(project_root),
        workspace=cfg,
        video_path=str(video_path),
    )
    executor = VideoProjectExecutor(context)
    print(f"\n[OK] Project ready: {context.root}\n")

    # ── step 2: run pipeline stages ─────────────────────────────────────────
    t0 = time.time()
    print(f"[RUN] Running stages: {args.from_stage} -> {args.to_stage} ...")
    try:
        results = executor.run_range(args.from_stage, args.to_stage, force=args.force)
        for r in results:
            stage = r.get("stage", "?")
            status = r.get("status", "?")
            elapsed = r.get("elapsed_sec", "")
            elapsed_str = f" ({elapsed:.1f}s)" if isinstance(elapsed, (int, float)) else ""
            print(f"  [{status.upper():>9}] {stage}{elapsed_str}")
    except Exception as exc:
        print(f"\n[FAIL] Pipeline error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    t1 = time.time()
    print(f"\n[OK] Main pipeline done in {t1 - t0:.1f}s")

    # ── step 3: materialize + catalog (optional) ────────────────────────────
    if args.materialize:
        print(f"\n[RUN] Running materialize ...")
        try:
            r = executor.run_stage("materialize", force=args.force)
            print(f"  [{r.get('status', '?').upper():>9}] materialize")
        except Exception as exc:
            print(f"  [FAIL] materialize: {exc}")

        print(f"[RUN] Running catalog ...")
        try:
            r = executor.run_stage("catalog", force=args.force)
            print(f"  [{r.get('status', '?').upper():>9}] catalog")
        except Exception as exc:
            print(f"  [FAIL] catalog: {exc}")

        t2 = time.time()
        print(f"\n[OK] Full pipeline done in {t2 - t0:.1f}s")

    # ── step 4: print final status ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Project Status")
    print("=" * 60)
    status = executor.status()
    for stage_name, stage_info in status.get("stages", status.get("steps", {})).items():
        s = stage_info if isinstance(stage_info, str) else stage_info.get("status", "?")
        icon = "OK" if s == "completed" else "  " if s == "pending" else s.upper()[:4]
        print(f"  [{icon:>4}] {stage_name}")
    print("=" * 60)
    print(f"\n  Project root: {context.root}")
    print(f"  Done!\n")


if __name__ == "__main__":
    main()
