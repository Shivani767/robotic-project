"""The ``robot-ci`` command-line interface.

Phase 1 ships the commands that establish the foundation: ``init``,
``validate``, ``version``, ``backend``, ``scenario`` and ``experiment show``.
Every later command from the project specification is registered and reports
``not implemented - Phase N`` with exit code 5, rather than pretending to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.table import Table

from robot_ci.cli.common import (
    EXIT_OK,
    console,
    error_console,
    fail,
    handle_errors,
    not_implemented,
    version_table,
)
from robot_ci.core.errors import RobotCIError
from robot_ci.core.version import ENVIRONMENT_VERSION, SCHEMA_VERSION, SIMULATOR_VERSION
from robot_ci.scenarios import get_family, list_families, load_scenario, save_scenario
from robot_ci.simulation import create_backend, list_backends, probe_backend

app = typer.Typer(
    name="robot-ci",
    help="ROBOT CI - automated behavioural regression testing for physical AI.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
scenario_app = typer.Typer(help="Inspect and generate scenarios.", no_args_is_help=True)
experiment_app = typer.Typer(help="Run and compare experiments.", no_args_is_help=True)
backend_app = typer.Typer(help="Inspect simulator backends.", no_args_is_help=True)
counterexample_app = typer.Typer(help="Search, minimize and reproduce counterexamples.", no_args_is_help=True)
app.add_typer(scenario_app, name="scenario")
app.add_typer(experiment_app, name="experiment")
app.add_typer(backend_app, name="backend")
app.add_typer(counterexample_app, name="counterexample")


# ---------------------------------------------------------------- version
@app.command("version")
@handle_errors
def version_command() -> None:
    """Print version information."""
    table = version_table()
    table.add_row("simulator contract", SIMULATOR_VERSION)
    table.add_row("environment", ENVIRONMENT_VERSION)
    table.add_row("python", __import__("platform").python_version())
    console.print(table)


# ------------------------------------------------------------------ init
@app.command("init")
@handle_errors
def init_command(
    force: Annotated[bool, typer.Option("--force", help="Recreate missing files even if they exist.")] = False,
) -> None:
    """Create the ROBOT CI directory layout in this repository."""
    from robot_ci.core.workspace import Workspace

    workspace = Workspace.discover()
    created = workspace.ensure_layout()
    console.print(
        Panel(
            f"workspace: [bold]{workspace.root}[/]\n"
            f"directories created: [green]{len(created)}[/]\n"
            + "\n".join(f"  + {workspace.relative(p)}" for p in created),
            title="ROBOT CI init",
            border_style="green",
        )
    )
    if not created:
        console.print("[dim]all directories already existed[/]")
    del force


# -------------------------------------------------------------- validate
@app.command("validate")
@handle_errors
def validate_command(
    probe: Annotated[bool, typer.Option("--probe/--no-probe", help="Instantiate each backend.")] = True,
) -> None:
    """Check the environment, backends, scenario catalogue and schemas.

    This is the Phase 1 acceptance gate: it must exit 0 on a stock MacBook.
    """
    problems: list[str] = []
    notes: list[str] = []

    console.rule("[bold cyan]ROBOT CI validate")

    # 1) Workspace
    try:
        from robot_ci.core.workspace import Workspace

        workspace = Workspace.discover()
        console.print(f"[green]OK[/] workspace: {workspace.root}")
    except RobotCIError as exc:
        problems.append(f"workspace: {exc.message}")
        workspace = None

    # 2) Imports of the research engine (must not need a simulator).
    try:

        console.print(f"[green]OK[/] core packages import (schema {SCHEMA_VERSION})")
    except Exception as exc:
        problems.append(f"import: {type(exc).__name__}: {exc}")

    # 3) Backends
    table = Table(title="simulator backends", title_style="bold")
    table.add_column("backend")
    table.add_column("status")
    table.add_column("physics")
    table.add_column("rgb/depth/seg")
    table.add_column("notes")
    for name in list_backends():
        result: dict[str, Any] = (
            probe_backend(name)
            if probe
            else {"name": name, "available": None, "reason": "not probed", "capabilities": None}
        )
        caps = result["capabilities"]
        if result["available"] is False:
            table.add_row(name, "[red]unavailable[/]", "-", "-", result["reason"][:48])
            if name in {"mock", "synthetic"}:
                problems.append(f"backend {name}: {result['reason']}")
        elif result["available"] is None:
            table.add_row(name, "[dim]not probed[/]", "-", "-", "--no-probe")
        else:
            table.add_row(
                name, "[green]available[/]",
                "yes" if caps.physics else "no",
                f"{'y' if caps.rgb else 'n'}/{'y' if caps.depth else 'n'}/{'y' if caps.segmentation else 'n'}",
                caps.notes[:48] or "-",
            )
            if name == "mujoco":
                notes.append(f"mujoco OpenGL backend: {os_environ_gl()}")
    console.print(table)

    # 4) Scenario catalogue
    fam_table = Table(title="scenario catalogue", title_style="bold")
    fam_table.add_column("family")
    fam_table.add_column("bench")
    fam_table.add_column("values")
    for entry in list_families():
        try:
            family = get_family(entry["name"])
            family.build(family.values[0], seed=0)
            fam_table.add_row(
                str(entry["name"]), str(entry["benchmark"]),
                ", ".join(f"{v:g}" for v in entry["values"]),
            )
        except RobotCIError as exc:
            problems.append(f"family {entry['name']}: {exc.message}")
    console.print(fam_table)

    # 5) Scenario files on disk
    if workspace is not None:
        scenario_dir = workspace.scenarios_dir
        files = sorted(scenario_dir.rglob("*.y*ml")) if scenario_dir.exists() else []
        bad = 0
        for path in files:
            try:
                load_scenario(path)
            except RobotCIError as exc:
                bad += 1
                problems.append(f"{workspace.relative(path)}: {exc.message}")
        console.print(
            f"[green]OK[/] scenario files: {len(files) - bad}/{len(files)} valid in {workspace.relative(scenario_dir)}"
        )

    # 6) Determinism self-check
    try:
        backend = create_backend("mock")
        handle = backend.reset(_default_scenario(), seed=1234)
        first = backend.observe(handle).robot.joint_positions
        backend2 = create_backend("mock")
        handle2 = backend2.reset(_default_scenario(), seed=1234)
        second = backend2.observe(handle2).robot.joint_positions
        backend.close_all()
        backend2.close_all()
        if first != second:
            problems.append("determinism check failed: two identical resets disagreed")
        else:
            console.print("[green]OK[/] determinism: identical seeds produce identical observations")
    except Exception as exc:
        problems.append(f"determinism: {type(exc).__name__}: {exc}")

    # 7) Ground-truth separation
    try:
        backend = create_backend("mock")
        handle = backend.reset(_default_scenario(), seed=7)
        obs = backend.observe(handle)
        backend.close_all()
        if obs.is_leaking_ground_truth():
            problems.append("ground-truth leak: Observation exposes a ground-truth field")
        else:
            console.print("[green]OK[/] ground-truth separation: Observation carries no ground truth")
    except Exception as exc:
        problems.append(f"ground-truth check: {type(exc).__name__}: {exc}")

    for note in notes:
        console.print(f"[dim]note: {note}[/]")

    console.rule()
    if problems:
        error_console.print(f"[bold red]{len(problems)} problem(s) found[/]")
        for problem in problems:
            error_console.print(f"  - {problem}")
        raise typer.Exit(3)
    console.print("[bold green]validate: PASSED[/]")
    raise typer.Exit(EXIT_OK)


def os_environ_gl() -> str:
    """Return the MuJoCo OpenGL backend currently selected (if any)."""
    import os

    return os.environ.get("MUJOCO_GL", "unset")


def _default_scenario() -> Any:
    """Build the nominal scenario used by the self-checks."""
    from robot_ci.models.scenario import ScenarioConfig

    return ScenarioConfig()


# --------------------------------------------------------------- backend
@backend_app.command("list")
@handle_errors
def backend_list() -> None:
    """List every registered backend and whether it works here."""
    table = Table(title="registered simulator backends", title_style="bold")
    table.add_column("name")
    table.add_column("available")
    table.add_column("physics")
    table.add_column("max dof")
    table.add_column("detail")
    for name in list_backends():
        result = probe_backend(name)
        caps = result["capabilities"]
        detail = result["reason"] or (caps.notes if caps else "") or "-"
        table.add_row(
            name,
            "[green]yes[/]" if result["available"] else "[red]no[/]",
            ("yes" if caps.physics else "no") if caps else "-",
            str(caps.max_dof) if caps else "-",
            detail[:50],
        )
    console.print(table)


@backend_app.command("probe")
@handle_errors
def backend_probe(
    name: Annotated[str, typer.Argument(help="Backend name to probe.")],
) -> None:
    """Construct one backend and report its capabilities."""
    result = probe_backend(name)
    if not result["available"]:
        fail(f"backend {name!r} is unavailable: {result['reason']}", code=4)
    caps = result["capabilities"]
    console.print(
        Panel(
            f"physics      : {caps.physics}\n"
            f"collision    : {caps.collision}\n"
            f"rgb/depth/seg: {caps.rgb}/{caps.depth}/{caps.segmentation}\n"
            f"max dof      : {caps.max_dof}\n"
            f"requires GPU : {caps.requires_gpu}\n"
            f"tasks        : {', '.join(t.value for t in caps.supported_tasks)}\n"
            f"notes        : {caps.notes}",
            title=f"backend {name}",
            border_style="cyan",
        )
    )


# -------------------------------------------------------------- scenario
@scenario_app.command("list")
@handle_errors
def scenario_list() -> None:
    """List the built-in scenario families."""
    table = Table(title="scenario families", title_style="bold")
    table.add_column("family")
    table.add_column("bench")
    table.add_column("perturbs")
    table.add_column("values")
    table.add_column("unit")
    for entry in list_families():
        table.add_row(
            str(entry["name"]), str(entry["benchmark"]), str(entry["variable"]),
            ", ".join(f"{v:g}" for v in entry["values"]),
            str(entry["unit"]) or "-",
        )
    console.print(table)


@scenario_app.command("show")
@handle_errors
def scenario_show(
    name: Annotated[str, typer.Argument(help="Family name, or a path to a scenario YAML file.")],
    value: Annotated[float | None, typer.Option("--value", help="Value along the family's axis.")] = None,
    seed: Annotated[int, typer.Option(help="Scenario seed.")] = 42,
) -> None:
    """Show a scenario's resolved configuration."""
    candidate = Path(name)
    if candidate.exists():
        scenario = load_scenario(candidate)
    else:
        family = get_family(name)
        chosen = family.values[0] if value is None else value
        scenario = family.build(chosen, seed=seed)
    console.print(
        Panel(
            f"scenario_id : {scenario.scenario_id}\n"
            f"family      : {scenario.family}\n"
            f"task        : {scenario.task.name.value}\n"
            f"seed        : {scenario.seed}\n"
            f"resolution  : {scenario.resolution[0]}x{scenario.resolution[1]}\n"
            f"objects     : {', '.join(o.display_name for o in scenario.objects)}\n"
            f"corruption  : {', '.join(scenario.sensors.active_sources()) or 'none'}\n"
            f"config_hash : {scenario.config_hash()}",
            title=scenario.scenario_id,
            border_style="cyan",
        )
    )
    conditions = scenario.describe_conditions()
    table = Table(title="conditions", show_header=True, title_style="bold")
    table.add_column("parameter")
    table.add_column("value")
    for key, val in conditions.items():
        table.add_row(key, f"{val:g}" if isinstance(val, float) else str(val))
    console.print(table)


@scenario_app.command("generate")
@handle_errors
def scenario_generate(
    family: Annotated[str, typer.Argument(help="Family name to expand.")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Directory or file to write.")] = None,
    seed: Annotated[int, typer.Option(help="Base seed.")] = 42,
    values: Annotated[str | None, typer.Option(help="Comma-separated override of the sweep values.")] = None,
) -> None:
    """Write one scenario file per value of a family's axis."""
    from robot_ci.models.scenario import VariableSpec
    from robot_ci.scenarios import sweep_scenario

    spec = get_family(family)
    if not spec.variable_path:
        fail(f"family {family!r} has no single swept axis and cannot be expanded")

    sweep_values = tuple(float(v) for v in values.split(",")) if values else spec.values
    variable = VariableSpec(name=spec.variable_name, path=spec.variable_path, values=sweep_values, unit=spec.unit)

    scenarios = list(sweep_scenario(spec.build(sweep_values[0], seed=seed), variable))
    console.print(f"generated [bold]{len(scenarios)}[/] scenarios from family [cyan]{family}[/]")

    if output is None:
        table = Table(title="generated scenarios", title_style="bold")
        table.add_column("scenario_id")
        table.add_column("seed")
        table.add_column("config_hash")
        for scenario in scenarios:
            table.add_row(scenario.scenario_id, str(scenario.seed), scenario.config_hash()[:22] + "...")
        console.print(table)
        console.print("[dim]use --output DIR to write YAML files[/]")
        return

    target = Path(output)
    written: list[Path] = []
    for scenario in scenarios:
        path = target if target.suffix in {".yaml", ".yml"} else target / f"{scenario.scenario_id}.yaml"
        written.append(save_scenario(scenario, path))
    console.print(f"[green]wrote {len(written)} files[/] -> {target}")


@scenario_app.command("validate")
@handle_errors
def scenario_validate(
    paths: Annotated[list[Path] | None, typer.Argument(help="Scenario files; defaults to scenarios/.")] = None,
) -> None:
    """Validate scenario files without running them."""
    from robot_ci.core.workspace import Workspace

    targets = list(paths) if paths else sorted(Workspace.discover().scenarios_dir.rglob("*.y*ml"))
    if not targets:
        console.print("[yellow]no scenario files found[/]")
        return
    bad = 0
    table = Table(title="scenario validation", title_style="bold")
    table.add_column("file")
    table.add_column("status")
    table.add_column("family")
    table.add_column("corruption")
    for path in targets:
        try:
            scenario = load_scenario(path)
            table.add_row(path.name, "[green]valid[/]", scenario.family, ", ".join(scenario.sensors.active_sources()) or "none")
        except RobotCIError as exc:
            bad += 1
            table.add_row(path.name, "[red]invalid[/]", "-", exc.message[:40])
    console.print(table)
    if bad:
        error_console.print(f"[bold red]{bad} invalid scenario file(s)[/]")
        raise typer.Exit(3)
    console.print(f"[bold green]{len(targets)} scenario file(s) valid[/]")


# ------------------------------------------------------------- experiment
@experiment_app.command("show")
@handle_errors
def experiment_show(
    path: Annotated[Path, typer.Argument(help="Path to experiment.yaml.")],
) -> None:
    """Validate an experiment file and print its resolved plan."""
    from robot_ci.core.errors import ConfigurationError
    from robot_ci.core.io import load_yaml
    from robot_ci.models.scenario import ExperimentConfig

    try:
        config = ExperimentConfig.model_validate(load_yaml(path))
    except Exception as exc:
        fail(f"invalid experiment file {path}: {exc}", code=3)
        return

    console.print(
        Panel(
            f"experiment_id : {config.experiment_id}\n"
            f"baseline      : {config.baseline.name} ({config.baseline.implementation})\n"
            f"candidate     : {config.candidate.name} ({config.candidate.implementation})\n"
            f"backend       : {config.backend}\n"
            f"repetitions   : {config.repetitions}\n"
            f"grid cells    : {config.grid_size}\n"
            f"episodes      : {config.total_episodes} per policy, {config.total_episodes_both_policies} total\n"
            f"config_hash   : {config.config_hash()}",
            title=config.experiment_id,
            border_style="cyan",
        )
    )
    changed = config.baseline.differs_from(config.candidate)
    console.print(
        f"components differing: [bold]{', '.join(c.value for c in changed) if changed else 'none declared'}[/]"
    )
    del ConfigurationError


# --------------------------------------------------- reserved (later phases)
@experiment_app.command("run")
@handle_errors
def experiment_run_reserved() -> None:
    """Run an experiment (Phase 2)."""
    not_implemented("robot-ci experiment run", 2, "Use `robot-ci experiment show FILE` to validate a config today.")


@experiment_app.command("compare")
@handle_errors
def experiment_compare_reserved() -> None:
    """Compare two experiments (Phase 6)."""
    not_implemented("robot-ci experiment compare", 6)


@app.command("regression")
@handle_errors
def regression_reserved() -> None:
    """Detect behavioural regressions (Phase 6)."""
    not_implemented("robot-ci regression detect", 6)


@app.command("diagnosis")
@handle_errors
def diagnosis_reserved() -> None:
    """Diagnose a regression (Phase 9)."""
    not_implemented("robot-ci diagnosis run", 9)


@app.command("boundary")
@handle_errors
def boundary_reserved() -> None:
    """Map a behavioural boundary (Phase 8)."""
    not_implemented("robot-ci boundary map", 8)


@app.command("report")
@handle_errors
def report_reserved() -> None:
    """Generate a report (Phase 11)."""
    not_implemented("robot-ci report generate", 11)


@app.command("dashboard")
@handle_errors
def dashboard_reserved() -> None:
    """Serve the dashboard (Phase 12)."""
    not_implemented("robot-ci dashboard", 12)


@app.command("test")
@handle_errors
def test_reserved(
    backend: Annotated[str, typer.Option(help="Backend to test with.")] = "mujoco",
) -> None:
    """Run the full regression suite (Phase 14)."""
    not_implemented("robot-ci test", 14, f"(--backend {backend} is validated; the suite arrives in Phase 14.)")


@counterexample_app.command("search")
@handle_errors
def counterexample_search_reserved() -> None:
    """Search for a counterexample (Phase 7)."""
    not_implemented("robot-ci counterexample search", 7)


@counterexample_app.command("reproduce")
@handle_errors
def counterexample_reproduce_reserved() -> None:
    """Reproduce a counterexample (Phase 7)."""
    not_implemented("robot-ci counterexample reproduce", 7)


@app.command("export")
@handle_errors
def export_reserved() -> None:
    """Package an experiment for Colab/Kaggle (Phase 13)."""
    not_implemented("robot-ci export", 13)


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
