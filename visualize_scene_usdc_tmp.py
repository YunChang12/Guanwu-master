from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENE = Path(
    r"E:\QingYan\Guanwu-master2\workspace\projects\video\demo_master2_video_main"
    r"\outputs\15_scene_export\scene.usdc"
)
DEFAULT_CONDA = Path(r"D:\AnacondaPackage\Anaconda\Scripts\conda.exe")
DEFAULT_ENV = "3d_env"
DEFAULT_CAMERA = "/World/Cameras/MainCamera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly inspect and visualize the exported OpenUSD scene."
    )
    parser.add_argument(
        "scene",
        nargs="?",
        type=Path,
        default=DEFAULT_SCENE,
        help="USD/USDC/USDZ file to open.",
    )
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--env", default=DEFAULT_ENV, help="Conda environment name.")
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="Initial usdview camera. Use an empty string to disable.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=1,
        help="Initial frame/current time code.",
    )
    parser.add_argument(
        "--complexity",
        choices=("low", "medium", "high", "veryhigh"),
        default="medium",
        help="Initial usdview mesh refinement complexity.",
    )
    parser.add_argument(
        "--dump-first-image",
        type=Path,
        default=None,
        help="Write a PNG preview and exit instead of opening the interactive viewer.",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print scene metadata; do not launch usdview.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the scene metadata pass before launching usdview.",
    )
    parser.add_argument(
        "--norender",
        action="store_true",
        help="Open usdview's hierarchy browser without Hydra rendering.",
    )
    return parser.parse_args()


def conda_run_prefix(conda: Path, env: str) -> list[str]:
    if conda.is_file():
        return [str(conda), "run", "-n", env]
    found = shutil.which("conda")
    if found:
        return [found, "run", "-n", env]
    raise FileNotFoundError(f"Cannot find conda.exe: {conda}")


def run_in_env(prefix: list[str], argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PXR_USDVIEW_SUPPRESS_STATE_SAVING", "1")
    return subprocess.run(
        [*prefix, *argv],
        env=env,
        text=True,
        check=check,
    )


def print_scene_summary(prefix: list[str], scene: Path) -> None:
    scene_literal = repr(str(scene))
    code = (
        "from pathlib import Path; from pxr import Usd, UsdGeom; "
        f"scene=Path({scene_literal}); stage=Usd.Stage.Open(str(scene)); "
        "assert stage is not None, f'Could not open USD stage: {scene}'; "
        "roots=[str(child.GetPath()) for child in stage.GetPseudoRoot().GetChildren()]; "
        "prims=list(stage.Traverse()); "
        "cameras=[str(prim.GetPath()) for prim in prims if prim.IsA(UsdGeom.Camera)]; "
        "meshes=sum(1 for prim in prims if prim.IsA(UsdGeom.Mesh)); "
        "xforms=sum(1 for prim in prims if prim.IsA(UsdGeom.Xformable)); "
        "default_prim=stage.GetDefaultPrim(); "
        "print(f'Scene: {scene}'); "
        "print(f'Size: {scene.stat().st_size / (1024 * 1024):.2f} MiB'); "
        "print(f'Time: {stage.GetStartTimeCode()} -> {stage.GetEndTimeCode()} fps={stage.GetFramesPerSecond()}'); "
        "print(f'Default prim: {default_prim.GetPath() if default_prim else \"<missing>\"}'); "
        "print(f'Root prims: {\", \".join(roots) if roots else \"<none>\"}'); "
        "print(f'Cameras ({len(cameras)}): {\", \".join(cameras) if cameras else \"<none>\"}'); "
        "print(f'Meshes: {meshes}'); "
        "print(f'Xformables: {xforms}')"
    )
    run_in_env(prefix, ["python", "-c", code])


def usdview_args(args: argparse.Namespace) -> list[str]:
    argv = ["usdview", str(args.scene), "--clearsettings", "--cf", str(args.frame)]
    argv.extend(["--complexity", args.complexity])
    if args.camera:
        argv.extend(["--camera", args.camera])
    if args.norender:
        argv.append("--norender")
    if args.dump_first_image is not None:
        args.dump_first_image = args.dump_first_image.resolve()
        args.dump_first_image.parent.mkdir(parents=True, exist_ok=True)
        argv.extend(["--dumpFirstImage", str(args.dump_first_image), "--quitAfterStartup"])
    return argv


def main() -> int:
    args = parse_args()
    args.scene = args.scene.resolve()
    if not args.scene.is_file():
        print(f"Scene file does not exist: {args.scene}", file=sys.stderr)
        return 2

    try:
        prefix = conda_run_prefix(args.conda, args.env)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not args.no_summary:
        print_scene_summary(prefix, args.scene)
        print()

    if args.info_only:
        return 0

    if args.dump_first_image is None:
        print("Launching usdview. Close the usdview window to return to this console.")
    else:
        print(f"Writing preview: {args.dump_first_image}")
    try:
        run_in_env(prefix, usdview_args(args))
    except subprocess.CalledProcessError as exc:
        print(f"usdview failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
