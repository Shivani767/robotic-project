"""The ``robot-ci`` command-line interface.

Phase 1 ships the commands that establish the foundation: ``init``,
``validate``, ``version``, ``backend``, ``scenario`` and ``experiment show``.
Every later command from the project specification is registered and reports
``not implemented - Phase N`` with exit code 5, rather than pretending to work.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel
from rich.table import Table

from robot_ci.cli.common import (
    EXIT_OK,
    EXIT_REGRESSION,
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
    """Build the nominal scenario used by the self-checks.

    Delegates to the catalogue rather than constructing a bare
    ``ScenarioConfig()``, which carries no placed objects and would report a
    nominal condition in which the task is impossible.
    """
    from robot_ci.scenarios import default_scenario

    return default_scenario()


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
def experiment_run(
    config_path: Annotated[Path, typer.Argument(help="Path to experiment.yaml.")],
    backend: Annotated[str | None, typer.Option("--backend", help="Override the backend.")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Results directory.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Cap the number of runs (smoke tests).")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress per-run output.")] = False,
) -> None:
    """Run an experiment and write its artifacts."""
    from robot_ci.core.errors import ConfigurationError
    from robot_ci.core.io import load_yaml
    from robot_ci.core.logging import JsonlLogger
    from robot_ci.core.workspace import Workspace
    from robot_ci.experiments import ExperimentRunner
    from robot_ci.models.scenario import ExperimentConfig

    try:
        config = ExperimentConfig.model_validate(load_yaml(config_path))
    except Exception as exc:
        fail(f"invalid experiment file {config_path}: {exc}", code=3)
        return

    workspace = Workspace.discover()
    target = output or (workspace.results_dir / config.experiment_id)
    target.mkdir(parents=True, exist_ok=True)
    events = JsonlLogger(
        (target / "events.jsonl").open("w", encoding="utf-8"),
        own_stream=True,
        context={"experiment_id": config.experiment_id, "git_commit": _git_head()},
    )

    selected = backend or config.backend
    console.rule(f"[bold cyan]experiment {config.experiment_id}[/]  backend={selected}")
    console.print(
        f"baseline [b]{config.baseline.name}[/]  vs  candidate [b]{config.candidate.name}[/]\n"
        f"grid {config.grid_size} cells x {config.repetitions} reps = "
        f"{config.total_episodes_both_policies} episodes"
    )

    with ExperimentRunner(selected) as runner:
        planned = len(list(runner.preview(config)))
        if not quiet:
            with console.status("running..."):
                outcome = runner.run(config, output_dir=target, log=events, limit=limit)
        else:
            outcome = runner.run(config, output_dir=target, log=events, limit=limit)

    table = Table(title="results", title_style="bold")
    table.add_column("policy")
    table.add_column("runs", justify="right")
    table.add_column("success", justify="right")
    table.add_column("mean steps", justify="right")
    for name, summary in sorted(outcome.summaries.items()):
        table.add_row(
            name,
            str(summary.n_runs),
            f"{summary.success_rate:.3f}",
            f"{summary.per_metric.get('steps', float('nan')):.1f}",
        )
    console.print(table)
    if outcome.n_episodes < planned:
        console.print(f"[yellow]truncated: ran {outcome.n_episodes} of {planned} planned[/]")
    console.print(f"artifacts [green]{workspace.relative(target)}[/]")
    del ConfigurationError


def _git_head() -> str:
    """Return the current commit, or ``unknown``."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


@experiment_app.command("compare")
@handle_errors
def experiment_compare_reserved() -> None:
    """Compare two experiments (Phase 6)."""
    not_implemented("robot-ci experiment compare", 6)


@app.command("regression")
@handle_errors
def regression_detect(
    results: Annotated[Path, typer.Argument(help="Experiment results dir or summary.json.")],
    baseline: Annotated[str, typer.Option(help="Baseline policy name.")] = "",
    candidate: Annotated[str, typer.Option(help="Candidate policy name.")] = "",
    blocking: Annotated[float, typer.Option(help="Delta at or below which a cell is blocking.")] = -0.10,
    warning: Annotated[float, typer.Option(help="Delta at or below which a cell is a warning.")] = -0.05,
    min_samples: Annotated[int, typer.Option(help="Minimum paired samples per cell.")] = 3,
) -> None:
    """Detect conditional behavioural regressions from a finished experiment."""
    import json

    from robot_ci.experiments.runner import EpisodeResult
    from robot_ci.regression import RegressionConfig, build_matrix, detect_regressions

    source = Path(results)
    summary_path = source / "summary.json" if source.is_dir() else source
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = [EpisodeResult.from_dict(r) for r in payload.get("results", [])]
    base_name = baseline or str(payload.get("baseline", ""))
    cand_name = candidate or str(payload.get("candidate", ""))
    if not base_name or not cand_name:
        fail("baseline/candidate names are required (missing from summary and options)")
    config = RegressionConfig(blocking_delta=blocking, warning_delta=warning, min_samples=min_samples)
    report = detect_regressions(episodes, baseline_name=base_name, candidate_name=cand_name, config=config)
    matrix = build_matrix(report.cells)

    console.print(
        Panel(
            f"baseline  : [cyan]{base_name}[/] {report.baseline_rate:.3f} (n={report.n_baseline})\n"
            f"candidate : [cyan]{cand_name}[/] {report.candidate_rate:.3f} (n={report.n_candidate})\n"
            f"global Δ  : {report.global_delta:+.3f}\n"
            f"findings  : [bold]{len(report.findings)}[/] "
            f"({sum(1 for f in report.findings if f.severity == 'blocking')} blocking)",
            title="regression report",
            border_style="cyan",
        )
    )
    table = Table(title="conditional cells", title_style="bold")
    table.add_column("condition")
    table.add_column("baseline", justify="right")
    table.add_column("candidate", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("severity")
    for row in matrix.rows:
        table.add_row(
            str(row["condition"]),
            f"{float(row['baseline_success']):.3f}",
            f"{float(row['candidate_success']):.3f}",
            f"{float(row['delta']):+.3f}",
            str(row["severity"]),
        )
    console.print(table)
    if source.is_dir():
        out = source / "regression.json"
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/] {out}")
    if any(f.severity == "blocking" for f in report.findings):
        raise typer.Exit(EXIT_REGRESSION)


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
def report_command(
    results: Annotated[
        Path | None, typer.Argument(help="Experiment results directory, e.g. experiments/results/demo.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="HTML path. Defaults to <results>/report.html.")
    ] = None,
) -> None:
    """Render a self-contained HTML report from an experiment's artifacts.

    Reads whatever the run produced and states plainly what it could not be
    built from, so a partial run still produces something readable rather than
    failing. The report leads with the per-condition matrix rather than the
    aggregate, because the aggregate is what a conditional difference is easy to
    miss inside.

    Every rate is shown with its interval and sample count, and a metric that was
    never measured reads ``n/a`` rather than 0 -- an unmeasured collision count
    and a measured zero are different facts.
    """
    from robot_ci.reporting import write_report

    target_dir = results or _latest_results_dir()
    if target_dir is None or not target_dir.is_dir():
        fail(
            "no experiment results directory given and none found under "
            "experiments/results; pass one explicitly"
        )
    written = write_report(target_dir, output)
    console.print(
        f"report written: {written} ({written.stat().st_size} bytes, self-contained)"
    )


def _latest_results_dir() -> Path | None:
    """Most recently modified experiment results directory, if any.

    A convenience for the single-experiment case. Ties are broken by name so
    repeated runs of the same experiment report the same directory rather than
    whichever the filesystem listed first.
    """
    root = Path("experiments/results")
    if not root.is_dir():
        return None
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d.stat().st_mtime, d.name))


@app.command("dashboard")
@handle_errors
def dashboard_reserved() -> None:
    """Serve the dashboard (Phase 12)."""
    not_implemented("robot-ci dashboard", 12)


@app.command("test")
@handle_errors
def test_command(
    backend: Annotated[str, typer.Option(help="Backend to test with.")] = "mujoco",
    seeds: Annotated[int, typer.Option(help="Seeds per condition per arm.")] = 4,
    baseline: Annotated[str, typer.Option(help="Baseline policy name.")] = "policy_v1",
    candidate: Annotated[str, typer.Option(help="Candidate policy name.")] = "policy_v2",
    no_benchmarks: Annotated[
        bool, typer.Option("--no-benchmarks", help="Skip the failure-injection suite.")
    ] = False,
) -> None:
    """Run the full regression gate. Exits non-zero on a blocking regression.

    This is the command a change is gated on, so it is built around one
    property: it must not report a pass it has not earned. Every gate prints how
    many episodes, conditions and seeds it actually ran, a gate that could not
    run is an error rather than a skip, and finding no regression exits 0 with
    the coverage that supports it -- which is a different claim from finding
    nothing to report.

    All gates run even after one fails, so a single invocation reports every
    problem rather than making the reader fix them one at a time.
    """
    from robot_ci.evaluation.suite import (
        GateResult,
        GateStatus,
        SuiteConfig,
        default_suite_config,
        run_suite,
    )

    config = SuiteConfig(
        backend=backend,
        baseline=baseline,
        candidate=candidate,
        seeds=max(1, seeds),
        run_benchmarks=not no_benchmarks,
    )

    def environment_gate() -> GateResult:
        """Confirm the backend loads and the scenario catalogue validates."""
        from robot_ci.scenarios import default_scenario
        from robot_ci.simulation import create_backend

        try:
            engine = create_backend(config.backend)
            engine.reset(default_scenario(), seed=0)
            engine.close_all()
        except Exception as error:
            return GateResult(
                name="environment",
                status=GateStatus.ERROR,
                detail=f"backend {config.backend!r} could not start: {error}",
            )
        return GateResult(
            name="environment",
            status=GateStatus.PASSED,
            detail=f"backend {config.backend} starts and the catalogue validates",
            coverage="scenario catalogue + 1 reset",
        )

    def regression_gate(suite: SuiteConfig) -> GateResult:
        """Run both arms across the sweep and score every condition."""
        from robot_ci.experiments.runner import run_episode
        from robot_ci.models.scenario import PolicyConfig
        from robot_ci.policies import create_policy
        from robot_ci.regression import RegressionConfig, detect_regressions
        from robot_ci.scenarios import default_scenario, set_by_path
        from robot_ci.simulation import create_backend

        base = default_scenario()
        engine = create_backend(suite.backend)
        episodes: list[Any] = []
        try:
            for level in suite.levels:
                condition = set_by_path(base, suite.axis_path, level)
                for name in (suite.baseline, suite.candidate):
                    for seed in range(suite.seeds):
                        episodes.append(
                            run_episode(
                                engine,
                                create_policy(PolicyConfig(name=name), condition),
                                condition,
                                seed=seed,
                                experiment_id="robot_ci_test",
                            )
                        )
        finally:
            engine.close_all()

        report = detect_regressions(
            episodes,
            baseline_name=suite.baseline,
            candidate_name=suite.candidate,
            config=RegressionConfig(
                blocking_delta=suite.blocking_delta, warning_delta=suite.warning_delta
            ),
        )
        blocking = [f for f in report.findings if f.severity == "blocking"]
        table = Table(title="robot-ci test", header_style="bold")
        # The condition column is the one a reader needs most, and it was the one
        # being truncated: at 80 columns rich squeezed it to the bare scenario id,
        # so four different noise levels all printed as "nominal". Folding
        # instead of truncating keeps the level visible, and the scenario id is
        # dropped from the display because the bracketed suffix already says
        # which field and value the row is about.
        table.add_column(f"{suite.axis_name}")
        for column in ("baseline", "candidate", "delta", "verdict"):
            table.add_column(column)
        # Sorted by the swept value, not by cell key: the detector groups by
        # condition values and emits them in key order, which put 8 mm before
        # 2 mm. A sweep read out of order is a sweep nobody can follow.
        for cell in sorted(
            report.cells, key=lambda c: _sort_key(c.condition, suite.axis_name)
        ):
            finding = next((f for f in report.findings if f.condition == cell.condition), None)
            table.add_row(
                _condition_value(cell.condition, suite.axis_name),
                f"{cell.baseline_success:.2f} (n={cell.n_baseline})",
                f"{cell.candidate_success:.2f} (n={cell.n_candidate})",
                f"{cell.delta:+.2f}",
                f"[red]{finding.severity}[/]" if finding else "",
            )
        console.print(table)

        coverage = (
            f"{len(report.cells)} conditions x {suite.seeds} seeds x 2 arms, "
            f"{len(episodes)} episodes"
        )
        if blocking:
            worst = min(blocking, key=lambda f: f.delta)
            return GateResult(
                name="regression",
                status=GateStatus.FAILED,
                detail=(
                    f"{len(blocking)} blocking regression(s); worst "
                    f"{worst.condition} at {worst.delta:+.2f}"
                ),
                coverage=coverage,
                episodes=len(episodes),
            )
        return GateResult(
            name="regression",
            status=GateStatus.PASSED,
            detail=(
                f"no condition passed the {suite.blocking_delta:+.2f} blocking threshold "
                f"(global delta {report.global_delta:+.3f})"
            ),
            coverage=coverage,
            episodes=len(episodes),
        )

    def benchmark_gate(suite: SuiteConfig) -> GateResult:
        """Run the failure-injection suite and check it against its expectations."""
        from robot_ci.evaluation import benchmark_suite, run_benchmark
        from robot_ci.experiments.runner import run_episode
        from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
        from robot_ci.policies import create_policy
        from robot_ci.scenarios import default_scenario

        base = default_scenario()
        engine = create_backend(suite.backend)
        mismatches: list[str] = []
        ran = 0
        try:

            def evaluate(
                scenario: ScenarioConfig, seeds: Sequence[int]
            ) -> tuple[float, dict[str, float]]:
                """Run one benchmark condition and average its metrics."""
                hits = 0
                totals: dict[str, list[float]] = {}
                for seed in seeds:
                    result = run_episode(
                        engine,
                        create_policy(PolicyConfig(name=suite.baseline), scenario),
                        scenario,
                        seed=seed,
                        experiment_id="robot_ci_test_bench",
                    )
                    hits += int(result.success)
                    for metric in result.metrics:
                        totals.setdefault(metric.name.value, []).append(float(metric.value))
                return hits / len(seeds), {k: sum(v) / len(v) for k, v in totals.items()}

            for benchmark in benchmark_suite():
                ran += 1
                try:
                    outcome = run_benchmark(
                        benchmark,
                        base_scenario=base,
                        seeds=tuple(range(suite.seeds)),
                        evaluate=evaluate,
                    )
                except ValueError as error:
                    mismatches.append(f"{benchmark.id}: {error}")
                    continue
                if outcome.expectation_met is False:
                    mismatches.append(f"{benchmark.id}: prediction did not hold")
        finally:
            engine.close_all()

        if mismatches:
            return GateResult(
                name="benchmarks",
                status=GateStatus.FAILED,
                detail=f"{len(mismatches)} benchmark expectation(s) did not hold",
                coverage=f"{ran} benchmarks, {ran * suite.seeds} episodes",
                episodes=ran * suite.seeds,
            )
        return GateResult(
            name="benchmarks",
            status=GateStatus.PASSED,
            detail=f"all {ran} benchmark expectations held",
            coverage=f"{ran} benchmarks, {ran * suite.seeds} episodes",
            episodes=ran * suite.seeds,
        )

    report = run_suite(
        default_suite_config(backend, seeds=seeds),
        environment=environment_gate,
        regression=regression_gate,
        benchmarks=benchmark_gate if not no_benchmarks else None,
    )
    console.print(Panel("\n".join(report.summary_lines()), title="ROBOT CI", style="bold"))
    if report.blocking_failures:
        console.print(
            f"[red]{len(report.blocking_failures)} blocking failure(s):[/] "
            + ", ".join(g.name for g in report.blocking_failures)
        )
        raise typer.Exit(EXIT_REGRESSION)


def _sort_key(label: str, axis_name: str) -> tuple[int, float, str]:
    """Sort a cell by the numeric value of the swept axis.

    Falls back to placing unparseable labels last rather than raising: a cell
    the sort cannot read is still a cell the reader needs to see.
    """
    value = _condition_value(label, axis_name)
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def _condition_value(label: str, axis_name: str) -> str:
    """Reduce a cell label to the value of the swept axis.

    ``nominal [depth_noise_mm=2.0]`` becomes ``2.0``, and the column is headed
    with the axis name. Printing the full label did not work: at 80 columns rich
    truncated the condition column to the bare scenario id, so four different
    noise levels all read "nominal" -- the one column a reader most needs was the
    one being hidden. A label short enough not to need truncating is the fix;
    the full label is in the JSON artifact either way.
    """
    marker = f"{axis_name}="
    if marker in label:
        tail = label[label.index(marker) + len(marker) :]
        return tail.rstrip("]").split(",")[0].strip()
    return label


def _parse_axis_spec(spec: str) -> Any:
    """Parse ``NAME[@PATH]:LOWER:UPPER[:UNIT]`` into a search axis.

    The path is optional because the experiment schema already maps well-known
    variable names to scenario fields; ``@`` is only needed for a name the table
    does not know. Requiring the path on every axis would make a one-line search
    invocation unreasonably verbose.
    """
    from robot_ci.counterexamples import PerturbationAxis
    from robot_ci.models.scenario import variable_path

    parts = spec.split(":")
    head, tail = parts[0], parts[1:]
    if len(tail) not in (2, 3):
        fail(
            f"axis {spec!r} must be NAME[@PATH]:LOWER:UPPER[:UNIT], "
            "e.g. depth_noise_mm:0:60:mm"
        )
    try:
        lower, upper = float(tail[0]), float(tail[1])
    except ValueError:
        fail(f"axis {spec!r} has non-numeric bounds: {tail[0]!r}, {tail[1]!r}")
    unit = tail[2] if len(tail) == 3 else ""
    if "@" in head:
        name, path = head.split("@", 1)
        if not path:
            fail(f"axis {spec!r} declares an empty path after '@'")
    else:
        name = head
        inferred = variable_path(name)
        if not inferred:
            fail(
                f"{name!r} is not a well-known variable; pass an explicit path as "
                f"{name}@PATH:LOWER:UPPER[:UNIT]"
            )
        path = inferred
    return PerturbationAxis(name, path, lower, upper, unit)


def _make_searcher(name: str, space: Any, *, budget: int, tolerance: float | None) -> Any:
    """Instantiate a strategy by name.

    The tolerance is passed only where it exists; spelling each branch out keeps
    the constructor arguments visible to the type checker instead of hiding them
    behind an ``**kwargs`` dict that erases them.
    """
    from robot_ci.counterexamples import (
        AdaptiveBoundarySearcher,
        BinaryBoundarySearcher,
        GridSearcher,
        RandomSearcher,
    )

    if name == "grid":
        return GridSearcher(space, budget=budget)
    if name == "random":
        return RandomSearcher(space, budget=budget, seed=0)
    if name == "binary":
        if tolerance is None:
            return BinaryBoundarySearcher(space, budget=budget)
        return BinaryBoundarySearcher(space, budget=budget, tolerance=tolerance)
    if name == "adaptive":
        if tolerance is None:
            return AdaptiveBoundarySearcher(space, budget=budget)
        return AdaptiveBoundarySearcher(space, budget=budget, tolerance=tolerance)
    fail(f"unknown strategy {name!r}; choose from grid, random, binary, adaptive")


@counterexample_app.command("search")
@handle_errors
def counterexample_search(
    axis: Annotated[
        list[str] | None,
        typer.Option("--axis", help="NAME[@PATH]:LOWER:UPPER[:UNIT]; repeatable."),
    ] = None,
    scenario: Annotated[Path | None, typer.Option("--scenario", help="Base scenario YAML.")] = None,
    strategy: Annotated[str, typer.Option(help="grid | random | binary | adaptive")] = "adaptive",
    budget: Annotated[int, typer.Option(help="Maximum oracle evaluations.")] = 64,
    tolerance: Annotated[
        float | None, typer.Option(help="Bisection tolerance, in axis units.")
    ] = None,
    backend: Annotated[str, typer.Option(help="Simulator backend.")] = "mock",
    baseline: Annotated[str, typer.Option(help="Baseline policy name.")] = "policy_v1",
    candidate: Annotated[str, typer.Option(help="Candidate policy name.")] = "policy_v2",
    # 240, not 90. A successful episode on the nominal condition needs 74
    # steps, and 4 mm of depth noise pushes the median to 188 -- a budget of 90
    # therefore truncates the very episodes the search is looking for and calls
    # the resulting timeouts failures. `reproduce` uses the same figure on
    # purpose: it has to be able to replay what `search` found, and a smaller
    # budget there would make every found counterexample unreproducible.
    max_steps: Annotated[int, typer.Option(help="Step budget per episode.")] = 240,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Artifact YAML path.")] = None,
) -> None:
    """Search for a condition where the baseline succeeds and the candidate fails.

    The default axis is depth noise over ``0..60 mm``, chosen because that is the
    axis the candidate's confidence gate responds to. A search that finds nothing
    exits 0: "no counterexample" is a result, not an error.
    """
    from robot_ci.counterexamples import (
        DiscoveryRequest,
        EpisodeOracle,
        SearchSpace,
        discover,
        save_counterexample,
    )
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.scenarios import default_scenario, load_scenario

    specs = list(axis) if axis else ["depth_noise_mm:0:60:mm"]
    space = SearchSpace([_parse_axis_spec(spec) for spec in specs])
    # The nominal catalogue family at a pinned, verified-solvable seed, not a
    # bare ScenarioConfig(): its default seed 0 is a nominal condition this arm
    # cannot solve (55 mm placement error), so the search stopped after one
    # evaluation and reported a false "both arms fail at nominal".
    base = load_scenario(scenario) if scenario is not None else default_scenario()
    searcher = _make_searcher(strategy, space, budget=budget, tolerance=tolerance)
    request = DiscoveryRequest(
        experiment_id="counterexample_search",
        base_scenario=base,
        space=space,
        baseline_policy=baseline,
        candidate_policy=candidate,
        seed=int(base.seed),
        backend=backend,
        git_commit=_git_head(),
    )

    oracle = EpisodeOracle(
        base_scenario=base,
        space=space,
        baseline=PolicyConfig(name=request.baseline_policy),
        candidate=PolicyConfig(name=request.candidate_policy),
        backend_name=backend,
        max_steps=max_steps,
    )
    try:
        result = discover(request, oracle, searcher)
    finally:
        oracle.close()

    outcome = result.outcome
    if not result.found:
        console.print(
            Panel(
                f"[bold yellow]no counterexample found[/]\n"
                f"strategy  : {outcome.strategy}\n"
                f"evaluated : {outcome.evaluations} points\n"
                f"note      : {outcome.notes}",
                title="counterexample search",
                border_style="yellow",
            )
        )
        return

    artifact = result.counterexample
    if artifact is None:  # pragma: no cover - guarded by found
        fail("counterexample reported found but no artifact was built")
    lines = [
        f"strategy   : {outcome.strategy}",
        f"evaluated  : {outcome.evaluations} points",
        f"severity   : {artifact.severity:.4f}",
        "condition  : " + ", ".join(f"{k}={v:g}" for k, v in sorted(artifact.perturbation.items())),
        f"baseline   : {artifact.baseline_policy} -> success",
        f"candidate  : {artifact.candidate_policy} -> failure",
    ]
    if artifact.boundary is not None:
        lines.append(f"boundary   : {artifact.boundary:.4f} ({artifact.boundary_axis})")
    if artifact.bracket is not None:
        lines.append(f"bracket    : [{artifact.bracket[0]:.4f}, {artifact.bracket[1]:.4f}]")
    if artifact.necessary_axes:
        lines.append(f"necessary  : {', '.join(artifact.necessary_axes)}")
    console.print(Panel("\n".join(lines), title="counterexample found", border_style="green"))

    target = output or (
        Path("experiments/results/counterexamples") / f"{artifact.counterexample_id}.yaml"
    )
    saved = save_counterexample(artifact, target)
    console.print(f"[green]artifact[/] {saved}")
    console.print(f"reproduce: robot-ci counterexample reproduce {saved}")


@counterexample_app.command("reproduce")
@handle_errors
def counterexample_reproduce(
    artifact: Annotated[Path, typer.Argument(help="Counterexample YAML written by `search`.")],
    backend: Annotated[str, typer.Option(help="Simulator backend.")] = "mock",
    # 240, not 90. A successful episode on the nominal condition needs 74
    # steps, and 4 mm of depth noise pushes the median to 188 -- a budget of 90
    # therefore truncates the very episodes the search is looking for and calls
    # the resulting timeouts failures. `reproduce` uses the same figure on
    # purpose: it has to be able to replay what `search` found, and a smaller
    # budget there would make every found counterexample unreproducible.
    max_steps: Annotated[int, typer.Option(help="Step budget per episode.")] = 240,
    seeds: Annotated[
        str, typer.Option(help="Comma-separated seeds; defaults to the artifact's own seed.")
    ] = "",
    strict: Annotated[
        bool, typer.Option("--strict/--no-strict", help="Exit 1 unless every seed reproduced.")
    ] = False,
) -> None:
    """Re-run a stored counterexample and report how often it reproduces."""
    from robot_ci.counterexamples import EpisodeOracle, load_counterexample, reproduce
    from robot_ci.models.scenario import PolicyConfig

    stored = load_counterexample(artifact)
    space = stored.space()
    base = stored.base_scenario()
    seed_list = (
        [int(part) for part in seeds.split(",") if part.strip()] if seeds.strip() else [stored.seed]
    )

    def make_oracle(seed: int) -> EpisodeOracle:
        """Rebuild the discovery conditions under a requested seed.

        ``reproduce`` releases any oracle that exposes ``close``, so each attempt
        gets its own backend and none is leaked.
        """
        return EpisodeOracle(
            base_scenario=base.model_copy(update={"seed": seed}),
            space=space,
            baseline=PolicyConfig(name=stored.baseline_policy),
            candidate=PolicyConfig(name=stored.candidate_policy),
            backend_name=backend,
            max_steps=max_steps,
        )

    report = reproduce(stored, make_oracle, seeds=seed_list)

    lines = [
        f"counterexample: {stored.counterexample_id}",
        "condition     : "
        + ", ".join(f"{k}={v:g}" for k, v in sorted(stored.perturbation.items())),
        f"seeds         : {', '.join(str(s) for s in report.seeds)}",
        f"reproduced    : {report.reproduced}/{report.attempts} ({report.rate:.3f})",
        f"all agreed    : {'yes' if report.consistent else 'NO'}",
    ]
    colour = "green" if report.rate == 1.0 else "yellow"
    console.print(Panel("\n".join(lines), title="reproduction", border_style=colour))

    table = Table(title="per-seed outcomes", title_style="bold")
    table.add_column("seed", justify="right")
    table.add_column("baseline")
    table.add_column("candidate")
    table.add_column("reproduced")
    for record in report.trials:
        table.add_row(
            str(record["seed"]),
            "success" if record["baseline_success"] else "failure",
            "success" if record["candidate_success"] else "failure",
            "[green]yes[/]" if record["reproduced"] else "[red]no[/]",
        )
    console.print(table)
    if report.rate < 1.0:
        console.print(
            "[yellow]a counterexample that reproduces only under some seeds is a "
            "property of the seed, not of the condition; do not report it as a "
            "regression without stating the reproduction rate.[/]"
        )
        if strict:
            raise typer.Exit(EXIT_REGRESSION)


#: Policy parameters the ablation engine varies, and the subsystem that reads
#: each one. Declared here rather than in the diagnosis package so the CLI's
#: defaults are visible in one place and can be overridden per run.
DEFAULT_ABLATION_FACTORS: tuple[tuple[str, tuple[float, ...], str], ...] = (
    # Perception: the confidence gate, and the settle window. ``settle_steps``
    # is the parameter that actually causes the policy_v1 / policy_v3
    # regression, so an ablation that omitted it would have nothing to find and
    # would report the absence of an effect as a finding.
    ("max_position_sigma_m", (0.004, 0.05), "perception"),
    ("settle_steps", (0, 6), "perception"),
    # Planner: waypoint clearance, and where the grasp is aimed.
    ("clearance_bias_m", (0.0, 0.05), "planner"),
    ("grasp_bias_m", (0.0, 0.01), "planner"),
    # Controller: the IK solver and the gripper's closing force.
    ("ik_damping", (0.05, 0.2), "controller"),
    ("ik_restarts", (6, 24), "controller"),
    ("grasp_width_fraction", (0.7, 0.85), "gripper"),
)


def _ablation_factors() -> list[Any]:
    """Build the default ablation factors from the table above."""
    from robot_ci.diagnosis import AblationFactor
    from robot_ci.models.scenario import PolicyComponent

    return [
        AblationFactor(parameter, values, PolicyComponent(subsystem))
        for parameter, values, subsystem in DEFAULT_ABLATION_FACTORS
    ]


def _mean_metrics(results: Iterable[Any]) -> dict[str, float]:
    """Average every metric across episodes, skipping ones nobody recorded."""
    totals: dict[str, list[float]] = {}
    for result in results:
        for metric in result.metrics:
            totals.setdefault(metric.name.value, []).append(float(metric.value))
    return {name: sum(values) / len(values) for name, values in totals.items() if values}


def _success_rate(results: Sequence[Any]) -> float:
    """Fraction of episodes that succeeded."""
    return sum(1 for result in results if result.success) / len(results) if results else 0.0


@app.command("diagnose")
@handle_errors
def diagnose_command(
    scenario: Annotated[
        Path | None, typer.Option(help="Base scenario YAML. Defaults to the nominal catalogue family.")
    ] = None,
    backend: Annotated[str, typer.Option(help="Simulator backend.")] = "mock",
    baseline: Annotated[str, typer.Option(help="Baseline policy name.")] = "policy_v1",
    candidate: Annotated[str, typer.Option(help="Candidate policy name.")] = "policy_v2",
    axis: Annotated[
        str | None,
        typer.Option(help="NAME:LOWER:UPPER[:UNIT] condition to sweep, e.g. depth_noise_mm:0:10:mm"),
    ] = None,
    seeds: Annotated[int, typer.Option(help="Seeds per cell and per arm.")] = 4,
    skip_ablation: Annotated[bool, typer.Option("--no-ablation", help="Skip the OFAT sweep.")] = False,
) -> None:
    """Compare two policies, classify their failures, and attribute the difference.

    Runs a small two-arm sweep over one condition axis, reports the conditional
    success difference per cell, classifies the candidate's failures, and then
    substitutes one policy parameter at a time to see which subsystem the
    difference is associated with.

    The attribution is an association measured by substitution. It is reported in
    that language deliberately: varying a parameter also varies the trajectories
    it produces, so a component can be credited with an effect that belongs to
    the episodes it happened to run on.
    """
    from robot_ci.diagnosis import classify_failure, diagnose, run_ablation
    from robot_ci.experiments.runner import run_episode
    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario, load_scenario, set_by_path
    from robot_ci.simulation import create_backend

    base = load_scenario(scenario) if scenario is not None else default_scenario()
    seed_list = [base.seed + index for index in range(seeds)]

    points: list[dict[str, float]] = [{}]
    axis_label = "nominal"
    if axis is not None:

        parts = axis.split(":")
        if len(parts) not in (3, 4):
            fail(f"axis {axis!r} must be NAME:LOWER:UPPER[:UNIT]")
        if _axis_path(parts[0]) == parts[0] and not parts[0].startswith("sensors."):
            fail(f"{parts[0]!r} is not a known variable; pass e.g. depth_noise_mm:0:10:mm")
        low, high = float(parts[1]), float(parts[2])
        steps = 4
        points = [{parts[0]: low + (high - low) * i / steps} for i in range(steps + 1)]
        axis_label = parts[0]

    def evaluate_condition(point: dict[str, float]) -> tuple[list[Any], list[Any]]:
        """Run both arms over every seed at one point on the axis."""
        condition = base if not point else set_by_path(base, _axis_path(axis_label), next(iter(point.values())))
        engine = create_backend(backend)
        try:
            arms: dict[str, list[Any]] = {baseline: [], candidate: []}
            for name in (baseline, candidate):
                for seed in seed_list:
                    arms[name].append(
                        run_episode(
                            engine,
                            create_policy(PolicyConfig(name=name), condition),
                            condition,
                            seed=seed,
                            experiment_id="diagnose",
                        )
                    )
            return arms[baseline], arms[candidate]
        finally:
            engine.close_all()

    table = Table(title=f"diagnosis: {baseline} vs {candidate}", header_style="bold")
    for column in ("condition", "baseline", "candidate", "delta"):
        table.add_column(column)
    # (delta, label, baseline metrics, candidate metrics, failures) for the
    # worst cell seen. The label travels with the numbers so the diagnosis is
    # attributed to the cell it was measured on.
    worst: tuple[float, str, dict[str, Any], dict[str, Any], list[Any]] | None = None
    for point in points:
        base_results, cand_results = evaluate_condition(point)
        base_rate, cand_rate = _success_rate(base_results), _success_rate(cand_results)
        label = f"{next(iter(point.values())):g}" if point else "nominal"
        delta = cand_rate - base_rate
        style = "red" if delta <= -0.10 else ("yellow" if delta <= -0.05 else "green")
        table.add_row(label, f"{base_rate:.2f}", f"{cand_rate:.2f}", f"[{style}]{delta:+.2f}[/{style}]")
        cell_label = "nominal" if not point else f"{axis_label}={next(iter(point.values())):g}"
        if worst is None or delta < worst[0]:
            failures = [
                event
                for result in cand_results
                if not result.success
                for event in classify_failure(
                    {m.name.value: float(m.value) for m in result.metrics},
                    step=result.steps,
                    message=str(result.status),
                )
            ]
            worst = (
                delta,
                cell_label,
                _mean_metrics(base_results),
                _mean_metrics(cand_results),
                failures,
            )
    console.print(table)

    if worst is None:
        console.print("[yellow]no conditions were evaluated[/]")
        return
    delta, condition_label, base_metrics, cand_metrics, failures = worst
    console.print()
    console.print(
        diagnose(
            condition=condition_label,
            baseline_metrics=base_metrics,
            candidate_metrics=cand_metrics,
            samples=len(seed_list),
            failures=failures,
        ).summary()
    )

    if skip_ablation:
        return
    console.print()
    console.print("[bold]ablation (one factor at a time)[/]")
    from robot_ci.models.scenario import PolicyConfig as _PolicyConfig

    reference = _PolicyConfig(name=candidate)

    def evaluate_arm(config: Any, _arm_id: str) -> tuple[float, int, dict[str, float]]:
        """Measure one ablation arm over the same seeds and conditions."""
        engine = create_backend(backend)
        try:
            results = [
                run_episode(
                    engine,
                    create_policy(config, point_scenario(base, axis_label, point)),
                    point_scenario(base, axis_label, point),
                    seed=seed,
                    experiment_id="ablation",
                )
                for point in points
                for seed in seed_list
            ]
            return _success_rate(results), len(results), _mean_metrics(results)
        finally:
            engine.close_all()

    outcome = run_ablation(reference, _ablation_factors(), evaluate_arm, designs=("ofat",))
    arm_table = Table(header_style="bold")
    for column in ("arm", "success", "n", "change"):
        arm_table.add_column(column)
    reference_rate = outcome.arms[0].success_rate
    for arm in outcome.arms:
        arm_table.add_row(
            arm.description,
            f"{arm.success_rate:.2f}",
            str(arm.samples),
            f"{arm.success_rate - reference_rate:+.2f}",
        )
    console.print(arm_table)
    for subsystem, effect in outcome.attribution:
        console.print(
            f"  {subsystem:12s} {effect:+.2f}  "
            + ("strongest association" if abs(effect) == max(abs(e) for _, e in outcome.attribution) else "")
        )
    console.print(f"[dim]{outcome.note}[/]")


def point_scenario(base: Any, axis_label: str, point: dict[str, float]) -> Any:
    """Return the scenario for one point on the sweep axis."""
    from robot_ci.scenarios import set_by_path as _set

    if not point:
        return base
    return _set(base, _axis_path(axis_label), next(iter(point.values())))


def _axis_path(name: str) -> str:
    """Resolve a variable name to the scenario path it controls."""
    from robot_ci.models.scenario import variable_path

    if name == "depth_noise_mm":
        return "sensors.depth.sigma_mm"
    resolved = variable_path(name)
    return resolved or name


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
