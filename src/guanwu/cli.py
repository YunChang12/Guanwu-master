"""QingNiao CLI - guanwu command."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="guanwu",
    help="QingNiao - dual-track data factory for sim datasets and natural-scene videos",
    no_args_is_help=True,
)
console = Console()

sim_app = typer.Typer(help="Simulation and dataset-ingest commands")
sim_registry_app = typer.Typer(help="Simulation dataset registry commands")
sim_project_app = typer.Typer(help="Simulation project commands")
video_app = typer.Typer(help="Natural video parsing commands")
video_project_app = typer.Typer(help="Natural video project commands")
registry_app = typer.Typer(help="Compatibility registry commands")
pipeline_app = typer.Typer(help="Compatibility pipeline execution commands")
catalog_app = typer.Typer(help="Catalog query commands")
remote_app = typer.Typer(help="Remote GPU execution commands")

app.add_typer(sim_app, name="sim")
sim_app.add_typer(sim_registry_app, name="registry")
sim_app.add_typer(sim_project_app, name="project")
app.add_typer(video_app, name="video")
video_app.add_typer(video_project_app, name="project")
app.add_typer(registry_app, name="registry")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(catalog_app, name="catalog")
app.add_typer(remote_app, name="remote")

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Path to workspace.yaml config file")
LOG_JSON_OPTION = typer.Option(False, "--log-json", help="Output logs in JSONL format")


def _load_config(config_path: str | None):
    from guanwu.core.config import WorkspaceConfig, load_config

    if config_path:
        return load_config(config_path)
    for default in ["configs/workspace.yaml", "workspace.yaml"]:
        if Path(default).exists():
            return load_config(default)
    return WorkspaceConfig()


def _ensure_sim_loaded() -> None:
    import guanwu.sim.adapters  # noqa: F401


def _project_table(title: str, payload: dict) -> Table:
    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, str(value))
    return table


def _ensure_compat_sim_executor(dataset_id: str, cfg):
    _ensure_sim_loaded()
    from guanwu.sim import ensure_sim_project, SimProjectExecutor

    context = ensure_sim_project(dataset_id=dataset_id, workspace=cfg)
    return SimProjectExecutor(context), context


def _set_sim_export_profile(executor, profile: str) -> None:
    from guanwu.projects.config import save_project_config

    executor.context.config.payload["default_export_profile"] = profile
    save_project_config(executor.context.config, executor.context.paths.config)
    executor.context.config = executor.context.config


def _compat_run_dataset(dataset_id: str, cfg, stages: list[str]) -> tuple[object, object, list[dict]]:
    executor, context = _ensure_compat_sim_executor(dataset_id, cfg)
    results = []
    for stage in stages:
        results.append(executor.run_stage(stage))
    return executor, context, results


def _show_sim_project_status(executor, context) -> None:
    status = executor.status()
    console.print(f"[green]Project:[/green] {context.root}")
    table = Table(title=f"sim project {executor.dataset_id}")
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    for name, item in status["steps"].items():
        table.add_row(name, item["status"])
    console.print(table)


def _show_video_project_status(executor, context) -> None:
    status = executor.status()
    console.print(f"[green]Project:[/green] {context.root}")
    table = Table(title=f"video project {context.root.name}")
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    for name, item in status["steps"].items():
        table.add_row(name, item["status"])
    console.print(table)


@sim_registry_app.command("list")
def sim_registry_list(
    group: Optional[str] = typer.Option(None, "--group", help="Filter by priority group (p0, p1, p2)"),
):
    _ensure_sim_loaded()
    from guanwu.sim.registry.manager import list_datasets

    datasets = list_datasets(group=group)
    table = Table(title="Simulation Datasets")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Geometry Max")
    table.add_column("Access")
    table.add_column("Group", style="green")
    table.add_column("Adapter", style="bold")
    for ds in datasets:
        table.add_row(
            ds["dataset_id"],
            ds["name"],
            ds.get("geometry_level_max", ""),
            ds.get("access_mode", ""),
            ds.get("group", ""),
            "yes" if ds.get("has_adapter") else "no",
        )
    console.print(table)


@sim_registry_app.command("show")
def sim_registry_show(dataset_id: str = typer.Argument(..., help="Dataset ID to show")):
    _ensure_sim_loaded()
    from guanwu.sim.registry.manager import show_dataset

    info = show_dataset(dataset_id)
    if not info:
        console.print(f"[red]Unknown dataset: {dataset_id}[/red]")
        raise typer.Exit(1)
    console.print(f"\n[bold cyan]{info['name']}[/bold cyan] ({dataset_id})")
    console.print(f"  URL: {info.get('url', 'N/A')}")
    console.print(f"  Geometry Max: {info.get('geometry_level_max', 'N/A')}")
    console.print(f"  Access: {info.get('access_mode', 'N/A')}")
    console.print(f"  License: {info.get('license', 'N/A')}")
    console.print(f"  Group: {info.get('group', 'N/A')}")
    console.print(f"  Adapter: {'v' + info['adapter_version'] if info.get('has_adapter') else 'not available'}")


@sim_project_app.command("init")
def sim_project_init(
    dataset: str = typer.Option(..., "--dataset", help="Dataset ID"),
    out: Optional[str] = typer.Option(None, "--out", help="Project output directory"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    _ensure_sim_loaded()
    from guanwu.sim.executor import SimProjectExecutor

    out_dir = out or str(Path(cfg.storage.project_root) / "sim" / dataset / "default")
    context = SimProjectExecutor.init_project(dataset_id=dataset, out_dir=out_dir, workspace=cfg)
    console.print(f"[green]sim project created:[/green] {context.root}")


@sim_project_app.command("status")
def sim_project_status(project: str = typer.Argument(..., help="Project root")):
    from guanwu.sim.executor import SimProjectExecutor
    from guanwu.projects import ProjectContext

    context = ProjectContext(project)
    executor = SimProjectExecutor(context)
    _show_sim_project_status(executor, context)


@sim_project_app.command("inspect")
def sim_project_inspect(project: str = typer.Argument(..., help="Project root")):
    from guanwu.sim.executor import SimProjectExecutor
    from guanwu.projects import ProjectContext

    context = ProjectContext(project)
    executor = SimProjectExecutor(context)
    console.print_json(data=executor.inspect())


@sim_app.command("step")
def sim_step(
    stage: str = typer.Argument(..., help="Stage name"),
    project: str = typer.Argument(..., help="Project root"),
    force: bool = typer.Option(False, "--force", help="Force rerun stage"),
):
    from guanwu.sim.executor import SimProjectExecutor
    from guanwu.projects import ProjectContext

    context = ProjectContext(project)
    executor = SimProjectExecutor(context)
    result = executor.run_stage(stage, force=force)
    console.print_json(data=result)


@sim_app.command("run")
def sim_run(
    project: str = typer.Argument(..., help="Project root"),
    from_stage: str = typer.Option("inventory", "--from", help="Start stage"),
    to_stage: str = typer.Option("catalog", "--to", help="End stage"),
    force: bool = typer.Option(False, "--force", help="Force rerun from the first stage"),
):
    from guanwu.sim.executor import SimProjectExecutor
    from guanwu.projects import ProjectContext

    context = ProjectContext(project)
    executor = SimProjectExecutor(context)
    result = executor.run_range(from_stage, to_stage, force=force)
    console.print_json(data=result)


@video_project_app.command("init")
def video_project_init(
    video: str = typer.Option(..., "--video", help="Input video"),
    out: Optional[str] = typer.Option(None, "--out", help="Project output directory"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    from guanwu.video.executor import VideoProjectExecutor

    video_path = Path(video).expanduser().resolve()
    out_dir = out or str(Path(cfg.storage.project_root) / "video" / video_path.stem)
    context = VideoProjectExecutor.init_project(video=str(video_path), out_dir=out_dir, workspace=cfg)
    console.print(f"[green]video project created:[/green] {context.root}")


@video_project_app.command("status")
def video_project_status(project: str = typer.Argument(..., help="Project root")):
    from guanwu.video.executor import VideoProjectExecutor

    executor = VideoProjectExecutor(project)
    _show_video_project_status(executor, executor.context)


@video_project_app.command("inspect")
def video_project_inspect(project: str = typer.Argument(..., help="Project root")):
    from guanwu.video.executor import VideoProjectExecutor

    executor = VideoProjectExecutor(project)
    console.print_json(data=executor.inspect())


@video_project_app.command("view")
@video_app.command("view")
def video_project_view(
    project: str = typer.Argument(..., help="Project root"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for the local viewer server"),
    port: int = typer.Option(8811, "--port", help="Port for the local viewer server"),
):
    from guanwu.video.project.viewer import serve_project_viewer

    project_root = str(Path(project).expanduser().resolve())
    console.print(f"[green]viewer:[/green] http://{host}:{port}")
    console.print(f"  project: {project_root}")
    serve_project_viewer(project_root, host=host, port=port)


@video_app.command("step")
def video_step(
    stage: str = typer.Argument(..., help="Stage name"),
    project: str = typer.Argument(..., help="Project root"),
    force: bool = typer.Option(False, "--force", help="Force rerun stage"),
):
    from guanwu.video.executor import VideoProjectExecutor

    executor = VideoProjectExecutor(project)
    result = executor.run_stage(stage, force=force)
    console.print_json(data=result)


@video_app.command("run")
def video_run(
    project: str = typer.Argument(..., help="Project root"),
    from_stage: str = typer.Option("video.inspect", "--from", help="Start stage"),
    to_stage: str = typer.Option("scene.export", "--to", help="End stage"),
    force: bool = typer.Option(False, "--force", help="Force rerun from the first stage"),
):
    from guanwu.video.executor import VideoProjectExecutor

    executor = VideoProjectExecutor(project)
    result = executor.run_range(from_stage, to_stage, force=force)
    console.print_json(data=result)


@video_app.command("materialize")
def video_materialize(
    project: str = typer.Argument(..., help="Project root"),
    force: bool = typer.Option(False, "--force", help="Force rerun materialize stage"),
):
    from guanwu.video.executor import VideoProjectExecutor

    executor = VideoProjectExecutor(project)
    result = executor.run_stage("materialize", force=force)
    console.print_json(data=result)


@registry_app.command("list")
def registry_list(
    group: Optional[str] = typer.Option(None, "--group", help="Filter by priority group (p0, p1, p2)"),
):
    sim_registry_list(group=group)


@registry_app.command("show")
def registry_show(dataset_id: str = typer.Argument(..., help="Dataset ID to show")):
    sim_registry_show(dataset_id=dataset_id)


@app.command()
def inventory(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    executor, context, _ = _compat_run_dataset(dataset_id, cfg, ["inventory"])
    console.print(f"[green]Done:[/green] inventory complete for {dataset_id}")
    console.print(f"  project: {context.root}")


@app.command()
def fetch(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    executor, context, _ = _compat_run_dataset(dataset_id, cfg, ["inventory", "fetch"])
    console.print(f"[green]Done:[/green] fetch complete for {dataset_id}")
    console.print(f"  project: {context.root}")


@app.command()
def ingest(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    executor, context, _ = _compat_run_dataset(
        dataset_id,
        cfg,
        ["inventory", "fetch", "parse", "normalize", "materialize"],
    )
    console.print(f"[green]Done:[/green] materialize complete for {dataset_id}")
    console.print(f"  project: {context.root}")


@app.command()
def validate(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    _, context, results = _compat_run_dataset(
        dataset_id,
        cfg,
        ["inventory", "fetch", "parse", "normalize", "validate"],
    )
    console.print(f"[green]Done:[/green] validate complete for {dataset_id}")
    console.print(f"  project: {context.root}")
    console.print_json(data=results[-1]["summary"])


@app.command()
def export(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    profile: str = typer.Option("mesh_preview", "--profile", "-p", help="Export profile"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    executor, context = _ensure_compat_sim_executor(dataset_id, cfg)
    _set_sim_export_profile(executor, profile)
    executor.run_stage("materialize")
    result = executor.run_stage("export", force=True)
    console.print(f"[green]Export complete:[/green] {profile} for {dataset_id}")
    console.print(f"  project: {context.root}")
    console.print_json(data=result["summary"])


@pipeline_app.command("run")
def pipeline_run(
    target: str = typer.Argument(..., help="Dataset ID or 'all'"),
    stages: Optional[str] = typer.Option(None, "--stages", help="Comma-separated stages"),
    group: Optional[str] = typer.Option(None, "--group", help="Priority group when target is 'all'"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    _ensure_sim_loaded()
    if target == "all":
        from guanwu.sim.registry.manager import get_datasets_by_group

        dataset_ids = get_datasets_by_group(group) if group else [
            dataset_id
            for dataset_id, dataset_cfg in cfg.datasets.items()
            if dataset_cfg.enabled
        ]
        for dataset_id in dataset_ids:
            stage_list = stages.split(",") if stages else ["inventory", "fetch", "parse", "normalize", "materialize", "catalog"]
            _, context, _ = _compat_run_dataset(dataset_id, cfg, stage_list)
            console.print(f"  {dataset_id}: {context.root}")
        return
    stage_list = stages.split(",") if stages else ["inventory", "fetch", "parse", "normalize", "materialize", "catalog"]
    _, context, _ = _compat_run_dataset(target, cfg, stage_list)
    console.print(f"[green]Pipeline complete:[/green] {target}")
    console.print(f"  project: {context.root}")


@catalog_app.command("build")
def catalog_build(config: Optional[str] = CONFIG_OPTION):
    cfg = _load_config(config)
    from guanwu.storage.catalog import Catalog

    catalog = Catalog(cfg.storage.catalog_path)
    catalog.build_from_canonical(cfg.storage.canonical_root)
    stats = catalog.get_stats()
    catalog.close()
    table = Table(title="Catalog Statistics")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    for table_name, count in stats.items():
        table.add_row(table_name, str(count))
    console.print(table)


@catalog_app.command("query")
def catalog_query(
    sql: str = typer.Argument(..., help="SQL query to execute"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    from guanwu.storage.catalog import Catalog

    catalog = Catalog(cfg.storage.catalog_path)
    catalog.initialize()
    try:
        results = catalog.query(sql)
        if not results:
            console.print("(no results)")
            return
        table = Table()
        for column in results[0].keys():
            table.add_column(column, style="cyan")
        for row in results:
            table.add_row(*[str(value) for value in row.values()])
        console.print(table)
    finally:
        catalog.close()


@remote_app.command("test")
def remote_test(config: Optional[str] = CONFIG_OPTION):
    cfg = _load_config(config)
    from guanwu.core.remote import get_remote_executor

    executor = get_remote_executor(cfg.runtime.remote)
    if executor is None:
        console.print("[red]Remote not configured.[/red] Add runtime.remote.host to workspace.yaml")
        raise typer.Exit(1)
    result = executor.test_connection()
    if result["ok"]:
        console.print(f"[green]Connected to {result['hostname']}[/green]")
        console.print(f"  GPU: {result.get('gpu', 'N/A')}")
        console.print(f"  Python: {result.get('python', 'N/A')}")
    else:
        console.print(f"[red]Connection failed:[/red] {result.get('error', 'unknown')}")
        raise typer.Exit(1)


@remote_app.command("replay")
def remote_replay(
    dataset_id: str = typer.Argument(..., help="Dataset ID (e.g. maniskill3)"),
    h5_file: str = typer.Option(..., "--h5", help="Path to trajectory .h5 file"),
    env_id: str = typer.Option(..., "--env", help="ManiSkill env ID"),
    traj: Optional[str] = typer.Option(None, "--traj", help="Comma-separated traj keys"),
    control_mode: str = typer.Option("pd_joint_pos", "--control-mode"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    max_trajs: Optional[int] = typer.Option(None, "--max-trajs"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    from guanwu.core.remote import get_remote_executor
    from guanwu.core.remote_tasks import replay_trajectory

    executor = get_remote_executor(cfg.runtime.remote)
    if executor is None:
        console.print("[red]Remote not configured.[/red]")
        raise typer.Exit(1)
    traj_keys = traj.split(",") if traj else None
    out = output_dir or str(Path(h5_file).parent)
    result = replay_trajectory(
        executor,
        h5_file,
        env_id,
        control_mode,
        traj_keys=traj_keys,
        output_dir=out,
        max_trajs=max_trajs,
    )
    console.print(f"[green]Replay complete:[/green] {len(result['traj_keys'])} trajectories")


@remote_app.command("render")
def remote_render(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    h5_file: str = typer.Option(..., "--h5", help="Path to trajectory .h5 file"),
    env_id: str = typer.Option(..., "--env", help="ManiSkill env ID"),
    traj: Optional[str] = typer.Option(None, "--traj", help="Comma-separated traj keys"),
    control_mode: str = typer.Option("pd_joint_pos", "--control-mode"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o"),
    width: int = typer.Option(512, "--width"),
    height: int = typer.Option(512, "--height"),
    frame_step: int = typer.Option(1, "--frame-step"),
    max_trajs: Optional[int] = typer.Option(None, "--max-trajs"),
    config: Optional[str] = CONFIG_OPTION,
):
    cfg = _load_config(config)
    from guanwu.core.remote import get_remote_executor
    from guanwu.core.remote_tasks import render_trajectory

    executor = get_remote_executor(cfg.runtime.remote)
    if executor is None:
        console.print("[red]Remote not configured.[/red]")
        raise typer.Exit(1)
    traj_keys = traj.split(",") if traj else None
    out = output_dir or str(Path(h5_file).parent / "renders")
    result = render_trajectory(
        executor,
        h5_file,
        env_id,
        control_mode,
        traj_keys=traj_keys,
        output_dir=out,
        resolution=(width, height),
        max_trajs=max_trajs,
        frame_step=frame_step,
    )
    renders_dir = result.get("output_files", {}).get("renders_dir")
    if renders_dir:
        console.print(f"[green]Renders saved to:[/green] {renders_dir}")


@app.command()
def doctor():
    console.print("[bold]QingNiao System Check[/bold]\n")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _print_check(f"Python {py_ver}", sys.version_info >= (3, 11), "Requires 3.11+")
    for dep in ["typer", "pydantic", "yaml", "fsspec", "pyarrow", "duckdb", "orjson", "numpy", "scipy", "pandas", "tqdm", "tenacity", "rich", "networkx"]:
        try:
            __import__(dep)
            _print_check(dep, True)
        except ImportError:
            _print_check(dep, False, "not installed")
    console.print("\n[bold]Optional geometry dependencies:[/bold]")
    for dep in ["trimesh", "cv2", "imageio", "PIL", "h5py"]:
        try:
            __import__(dep)
            _print_check(dep, True)
        except ImportError:
            _print_check(dep, None, "not installed (optional)")
    console.print("\n[bold]Simulation adapters:[/bold]")
    _ensure_sim_loaded()
    from guanwu.adapters.base import list_adapters

    for name, adapter_cls in sorted(list_adapters().items()):
        _print_check(f"adapter:{name}", True, f"v{adapter_cls().version}")
    console.print("\n[bold]Video sources:[/bold]")
    from guanwu.video.registry import list_sources

    for source in list_sources():
        _print_check(f"video:{source['dataset_id']}", True, source["geometry_level_max"])


@app.command()
def stats(config: Optional[str] = CONFIG_OPTION):
    cfg = _load_config(config)
    console.print(f"[bold]Workspace:[/bold] {cfg.workspace_root}")
    console.print(f"  Raw store: {cfg.storage.raw_root}")
    console.print(f"  Canonical store: {cfg.storage.canonical_root}")
    console.print(f"  Project root: {cfg.storage.project_root}")
    console.print(f"  Catalog: {cfg.storage.catalog_path}")
    cat_path = Path(cfg.storage.catalog_path)
    if cat_path.exists():
        from guanwu.storage.catalog import Catalog

        catalog = Catalog(str(cat_path))
        catalog.initialize()
        stats_payload = catalog.get_stats()
        catalog.close()
        console.print("\n[bold]Catalog Statistics:[/bold]")
        for table_name, count in stats_payload.items():
            if count > 0:
                console.print(f"  {table_name}: {count}")
    else:
        console.print("\n  (no catalog built yet)")


def _print_check(name: str, ok: bool | None, detail: str = "") -> None:
    if ok is True:
        icon = "[green]OK[/green]"
    elif ok is False:
        icon = "[red]FAIL[/red]"
    else:
        icon = "[yellow]SKIP[/yellow]"
    detail_str = f" - {detail}" if detail else ""
    console.print(f"  {icon}  {name}{detail_str}")


if __name__ == "__main__":
    app()
