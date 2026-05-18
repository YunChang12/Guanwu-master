"""Unified CLI for running versioned pose optimizer variants."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, config_to_argv, load_config
from .variants import VARIANTS, Variant


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a managed pose optimizer variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default=None,
        help="Optimizer variant to run. If omitted, the config variant is used, then baseline.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Flat YAML/JSON config file. Defaults to the registered config for the selected variant.",
    )
    parser.add_argument(
        "--list_variants",
        action="store_true",
        help="List registered variants and exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved variant and forwarded optimizer arguments without running.",
    )
    parser.add_argument(
        "--no_default_config",
        action="store_true",
        help="Run the variant without loading its registered default config.",
    )
    return parser


def list_variants() -> None:
    for name in sorted(VARIANTS):
        variant = VARIANTS[name]
        print(f"{variant.name}: {variant.description}")
        print(f"  module: {variant.module_name}")
        print(f"  config: {variant.config_path}")


def normalize_passthrough_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def resolve_variant_and_config(
    requested_variant: str | None,
    requested_config: str | None,
    no_default_config: bool,
) -> tuple[Variant, Path | None, dict[str, Any]]:
    explicit_config = Path(requested_config).resolve() if requested_config else None
    config: dict[str, Any] = {}

    if explicit_config is not None:
        config = load_config(explicit_config)

    config_variant = config.get("variant")
    if config_variant is not None and str(config_variant) not in VARIANTS:
        raise ConfigError(f"Unknown variant in config: {config_variant!r}")

    variant_name = requested_variant or (str(config_variant) if config_variant else "baseline")
    if requested_variant and config_variant and requested_variant != str(config_variant):
        raise ConfigError(
            f"Config variant {config_variant!r} does not match requested variant {requested_variant!r}."
        )

    variant = VARIANTS[variant_name]
    config_path: Path | None = explicit_config
    if config_path is None and not no_default_config:
        config_path = variant.config_path
        config = load_config(config_path)
        config_variant = config.get("variant")
        if config_variant is not None and str(config_variant) != variant.name:
            raise ConfigError(
                f"Default config {config_path} declares variant {config_variant!r}, "
                f"but it is registered for {variant.name!r}."
            )

    return variant, config_path, config


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def git_dirty_status() -> str:
    return git_value("status", "--short") or ""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def append_run_metadata(
    report: dict[str, Any],
    variant: Variant,
    config_path: Path | None,
    forwarded_args: list[str],
) -> dict[str, Any]:
    metadata = {
        "variant": variant.name,
        "variant_module": variant.module_name,
        "config_path": str(config_path) if config_path else None,
        "forwarded_args": forwarded_args,
        "git_commit": git_value("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git_dirty_status()),
        "managed_entrypoint": "process.pose_optimizer.cli",
    }
    report["run_metadata"] = metadata

    report_path = report.get("outputs", {}).get("optimization_report")
    if report_path:
        path = Path(report_path)
        path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    pose_world = report["optimized_corrected_pose_world"]

    print(f"task_id: {report['task_id']}")
    print(f"best_mask_iou: {metrics['mask_iou']:.6f}")
    print(f"best_bbox_iou: {metrics['bbox_iou']:.6f}")
    print(f"best_bbox_center_error_px: {metrics['bbox_center_error_px']:.6f}")
    print(f"best_projected_bbox: {metrics['projected_bbox']}")
    print(f"optimized_translation_world: {json_safe(pose_world['translation_world'])}")
    print(f"optimized_scale: {json_safe(pose_world['scale'])}")
    print(f"render_backend: {report['render_backend']}")
    print(f"alignment_collage_path: {report['outputs']['alignment_collage']}")
    print(f"pose_closeup_collage_path: {report['outputs']['pose_closeup_collage']}")
    print(f"model_reference_collage_path: {report['outputs']['model_reference_collage']}")
    print(f"report_path: {report['outputs']['optimization_report']}")
    print(f"optimized_task_path: {report['outputs']['optimized_task']}")

    profiling = report.get("profiling")
    if profiling:
        for key in sorted(profiling):
            print(f"profiling_{key}: {profiling[key]}")

    metadata = report.get("run_metadata")
    if metadata:
        print(f"variant: {metadata['variant']}")
        print(f"git_commit: {metadata['git_commit']}")
        print(f"git_dirty: {metadata['git_dirty']}")


def run_variant(
    variant: Variant,
    config_path: Path | None,
    config: dict[str, Any],
    passthrough_args: list[str],
    dry_run: bool,
) -> int:
    forwarded_args = config_to_argv(config) + passthrough_args
    if dry_run:
        print(f"variant: {variant.name}")
        print(f"module: {variant.module_name}")
        print(f"config: {config_path}")
        print("forwarded_args:")
        print(" ".join(forwarded_args))
        return 0

    module = importlib.import_module(variant.module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [variant.module_name, *forwarded_args]
        args = module.parse_args()
    finally:
        sys.argv = old_argv

    try:
        report = module.optimize_sample(args)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1

    append_run_metadata(report, variant, config_path, forwarded_args)
    print_summary(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    known_args, unknown_args = parser.parse_known_args(argv)
    passthrough_args = normalize_passthrough_args(unknown_args)

    if known_args.list_variants:
        list_variants()
        return 0

    try:
        variant, config_path, config = resolve_variant_and_config(
            known_args.variant,
            known_args.config,
            known_args.no_default_config,
        )
    except ConfigError as exc:
        parser.error(str(exc))

    return run_variant(
        variant=variant,
        config_path=config_path,
        config=config,
        passthrough_args=passthrough_args,
        dry_run=known_args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
