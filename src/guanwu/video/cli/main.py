from __future__ import annotations

import argparse
import json
import shlex
import sys
from cmd import Cmd

from guanwu.video.project import ProjectContext, ProjectExecutor
from guanwu.video.project.artifacts import LEGACY_STAGE_ALIASES, PHASE_MAP, STAGE_ORDER


class AgentShell(Cmd):
    intro = "SPWM agent shell. Type 'help' or 'exit'."
    prompt = "spwm> "

    def __init__(self, project_root: str) -> None:
        super().__init__()
        self.context = ProjectContext(project_root)
        self.executor = ProjectExecutor(self.context)

    def do_status(self, arg: str) -> bool | None:
        _ = arg
        print(json.dumps(self.executor.status(), indent=2))
        return None

    def do_run(self, arg: str) -> bool | None:
        stage = arg.strip()
        if not stage:
            print("usage: run <stage>")
            return None
        print(json.dumps(self.executor.run_stage(stage), indent=2))
        return None

    def do_viz(self, arg: str) -> bool | None:
        _ = arg
        print(json.dumps(self.executor.run_stage("report.render"), indent=2))
        return None

    def do_validate(self, arg: str) -> bool | None:
        _ = arg
        print(json.dumps(self.executor.validate(), indent=2))
        return None

    def do_phase(self, arg: str) -> bool | None:
        parts = arg.strip().split()
        if not parts or parts[0] == "list":
            for name, (from_s, to_s) in PHASE_MAP.items():
                print(f"  {name:<8}  {from_s} → {to_s}")
            return None
        phase = parts[0]
        force = "--force" in parts
        print(json.dumps(self.executor.run_phase(phase, force=force), indent=2))
        return None

    def do_steps(self, arg: str) -> bool | None:
        _ = arg
        print("\n".join(STAGE_ORDER))
        return None

    def default(self, line: str) -> bool | None:
        lowered = line.strip().lower()
        if lowered in {"exit", "quit"}:
            return True
        if lowered.startswith("run "):
            return self.do_run(line.split(" ", 1)[1])
        if "status" in lowered:
            return self.do_status("")
        if "visual" in lowered or lowered.startswith("viz"):
            return self.do_viz("")
        print(f"Unknown command: {line}")
        return None

    def do_exit(self, arg: str) -> bool:
        _ = arg
        return True

    def do_quit(self, arg: str) -> bool:
        _ = arg
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spwm", description="Project-based SPWM CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    project_cmd = sub.add_parser("project")
    project_sub = project_cmd.add_subparsers(dest="project_command", required=True)
    init_cmd = project_sub.add_parser("init")
    init_cmd.add_argument("--video", required=True)
    init_cmd.add_argument("--out", required=True)
    init_cmd.add_argument("--provider-mode", default="mock", choices=["mock", "zaiwu"])
    init_cmd.add_argument("--video-copy-mode", default="copy", choices=["copy", "link"])
    init_cmd.add_argument("--prompts", nargs="*")

    status_cmd = project_sub.add_parser("status")
    status_cmd.add_argument("project")

    inspect_cmd = project_sub.add_parser("inspect")
    inspect_cmd.add_argument("project")

    step_cmd = sub.add_parser("step")
    step_sub = step_cmd.add_subparsers(dest="stage", required=True)
    aliases_by_stage: dict[str, list[str]] = {}
    for legacy, canonical in LEGACY_STAGE_ALIASES.items():
        aliases_by_stage.setdefault(canonical, []).append(legacy)
    for stage in STAGE_ORDER:
        stage_cmd = step_sub.add_parser(stage, aliases=aliases_by_stage.get(stage, []))
        stage_cmd.add_argument("project")
        stage_cmd.add_argument("--force", action="store_true")

    viz_cmd = sub.add_parser("viz")
    viz_cmd.add_argument("project")

    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("project")

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("project")
    run_cmd.add_argument("--from", dest="from_stage", required=True)
    run_cmd.add_argument("--to", dest="to_stage", required=True)
    run_cmd.add_argument("--force", action="store_true")

    shell_cmd = sub.add_parser("shell")
    shell_cmd.add_argument("project")

    phase_cmd = sub.add_parser("phase")
    phase_sub = phase_cmd.add_subparsers(dest="phase_name", required=True)
    list_cmd = phase_sub.add_parser("list")  # noqa: F841
    for phase in PHASE_MAP:
        p = phase_sub.add_parser(phase)
        p.add_argument("project")
        p.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "project" and args.project_command == "init":
        context = ProjectExecutor.init_project(
            video=args.video,
            out_dir=args.out,
            provider_mode=args.provider_mode,
            video_copy_mode=args.video_copy_mode,
        )
        print(json.dumps({"status": "ok", "project_root": str(context.paths.root)}, indent=2))
        return 0

    if args.command == "project":
        executor = ProjectExecutor(ProjectContext(args.project))
        if args.project_command == "status":
            print(json.dumps(executor.status(), indent=2))
            return 0
        if args.project_command == "inspect":
            print(json.dumps(executor.inspect(), indent=2))
            return 0

    if args.command == "step":
        executor = ProjectExecutor(ProjectContext(args.project))
        print(json.dumps(executor.run_stage(LEGACY_STAGE_ALIASES.get(args.stage, args.stage), force=args.force), indent=2))
        return 0

    if args.command == "viz":
        executor = ProjectExecutor(ProjectContext(args.project))
        print(json.dumps(executor.run_stage("report.render"), indent=2))
        return 0

    if args.command == "validate":
        executor = ProjectExecutor(ProjectContext(args.project))
        result = executor.validate()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ok" else 1

    if args.command == "run":
        executor = ProjectExecutor(ProjectContext(args.project))
        print(
            json.dumps(
                executor.run_range(
                    LEGACY_STAGE_ALIASES.get(args.from_stage, args.from_stage),
                    LEGACY_STAGE_ALIASES.get(args.to_stage, args.to_stage),
                    force=args.force,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "phase":
        if args.phase_name == "list":
            for name, (from_s, to_s) in PHASE_MAP.items():
                print(f"  {name:<8}  {from_s} → {to_s}")
            return 0
        executor = ProjectExecutor(ProjectContext(args.project))
        print(json.dumps(executor.run_phase(args.phase_name, force=args.force), indent=2))
        return 0

    if args.command == "shell":
        shell = AgentShell(args.project)
        shell.cmdloop()
        return 0

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
