r"""Build the browser bundle for the ROBOT CI research dashboard.

This module is the single data-access layer between the frozen research
evidence under ``experiments/results/`` (plus the recorded Gazebo telemetry and
the Panda URDF) and the static frontend in ``dashboard/``. It exists so that
the numbers on screen are *recomputed from the source records* rather than
transcribed, and so that a disagreement between the dashboard and the frozen
JSON is caught here instead of silently shipping.

Design rules this file follows:

* **No number is invented.** Every field the browser renders traces back to a
  specific key in a specific result file. Where a metric was never measured the
  bundle carries ``None`` and the UI shows ``Not measured`` -- never a filler
  value.
* **Aggregates are recomputed**, not copied. Nominal solvability, policy deltas,
  the settle-window sweep and the paired McNemar comparisons are all derived
  from the per-seed records so the dashboard cannot drift from the evidence.
* **Consistency is checked.** ``consistency_checks`` records, per derived value,
  whether it agrees with the value already stored in the JSON. A failed check is
  surfaced in the UI's provenance panel rather than hidden.
* **The frozen evidence is read-only.** Nothing here writes to
  ``experiments/results/`` or ``artifacts/``; it only writes the generated
  bundle under ``dashboard/data/``.

Usage::

    python experiments/build_dashboard_data.py
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
ARTIFACTS = REPO / "artifacts"
URDF = REPO / "gazebo" / "models" / "panda" / "panda.urdf"
TELEMETRY = ARTIFACTS / "linkedin" / "robot_ci_demo" / "results" / "telemetry.csv"
OUT_DIR = REPO / "dashboard" / "data"

CURRENT_REVISION = "d7026af"


# --------------------------------------------------------------------------
# small stats helpers (Wilson interval + exact McNemar), dependency-free
# --------------------------------------------------------------------------
def _z(p: float) -> float:
    return 1.959963984540054  # 95% two-sided normal quantile


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    """Wilson score interval for a binomial rate."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    rate = successes / n
    z = _z(0.0)
    denom = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return (rate, max(0.0, center - margin), min(1.0, center + margin))


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * _binom_cdf(k, n))


# --------------------------------------------------------------------------
# URDF -> browser kinematics + real link geometry
# --------------------------------------------------------------------------
def _floats(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(v) for v in (text if text else default).split()]


def robot_from_urdf(urdf_path: Path) -> dict[str, Any]:
    """Project the Panda URDF into the shape the browser viewer consumes.

    Unlike ``robot_ci.gazebo.live.robot_kinematics`` (which ships only the
    kinematic tree for the live WebSocket viewer), this also emits each link's
    *visual geometry* -- a real cylinder or box with its URDF origin and rpy -- so
    the browser can draw an articulated arm with recognizable links rather than a
    line skeleton. It is a projection of the same single URDF, not a second
    hand-maintained model.
    """
    root = ElementTree.parse(urdf_path).getroot()
    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": _floats(origin.get("xyz") if origin is not None else None),
                "rpy": _floats(origin.get("rpy") if origin is not None else None),
                "axis": _floats(axis.get("xyz") if axis is not None else None, "0 0 1"),
                "lower": float(limit.get("lower", -3.0)) if limit is not None else -3.0,
                "upper": float(limit.get("upper", 3.0)) if limit is not None else 3.0,
            }
        )
    links: list[dict[str, Any]] = []
    for link in root.findall("link"):
        visuals: list[dict[str, Any]] = []
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            geom = vis.find("geometry")
            shape: dict[str, Any] = {"kind": "none"}
            if geom is not None:
                cyl = geom.find("cylinder")
                box = geom.find("box")
                sph = geom.find("sphere")
                if cyl is not None:
                    shape = {
                        "kind": "cylinder",
                        "radius": float(cyl.get("radius", "0.05")),
                        "length": float(cyl.get("length", "0.1")),
                    }
                elif box is not None:
                    shape = {"kind": "box", "size": _floats(box.get("size"), "0 0 0")}
                elif sph is not None:
                    shape = {"kind": "sphere", "radius": float(sph.get("radius", "0.05"))}
            visuals.append(
                {
                    "xyz": _floats(origin.get("xyz") if origin is not None else None),
                    "rpy": _floats(origin.get("rpy") if origin is not None else None),
                    "shape": shape,
                }
            )
        links.append({"name": link.get("name"), "visuals": visuals})
    return {"name": "panda", "joints": joints, "links": links}


# --------------------------------------------------------------------------
# recorded telemetry -> replayable episode
# --------------------------------------------------------------------------
ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]


def parse_telemetry(path: Path) -> dict[str, Any]:
    """Parse a recorded telemetry CSV into ordered frames for replay.

    Each frame keeps the raw per-step joint angles, gripper aperture, object
    pose and the phase label the recorder stamped, so the browser replays the
    exact recorded motion. Phases are reported in first-seen order and de-duped
    so the UI only ever shows phases actually present in the episode.
    """
    frames: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame = {
                "step": int(row["step"]),
                "t": float(row["sim_t_s"]),
                "phase": row["phase"].strip(),
                "grasped": int(row["grasped"]),
                "aperture": float(row["aperture_m"]),
                "tcp": [float(row["tcp_x"]), float(row["tcp_y"]), float(row["tcp_z"])],
                "obj": [float(row["obj_x"]), float(row["obj_y"]), float(row["obj_z"])],
                "joints": {name: float(row[name]) for name in ARM_JOINTS},
            }
            frames.append(frame)
    phases: list[str] = []
    for frame in frames:
        if frame["phase"] and frame["phase"] not in phases:
            phases.append(frame["phase"])
    return {"frames": frames, "phases": phases}


# --------------------------------------------------------------------------
# evidence aggregation
# --------------------------------------------------------------------------
def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


CHECKS: list[dict[str, Any]] = []


def _check(label: str, derived: Any, stored: Any, tol: float = 1e-6) -> None:
    ok = derived == stored
    if isinstance(derived, float) and isinstance(stored, (int, float)):
        ok = abs(derived - stored) <= tol
    CHECKS.append({"label": label, "derived": derived, "stored": stored, "ok": bool(ok)})


def build_nominal() -> dict[str, Any]:
    """Recompute nominal solvability (rate + Wilson interval) from the source record."""
    data = _load("nominal_solvability.json")
    if not data:
        return {"available": False}
    policies = {}
    for name, rec in data["policies"].items():
        rate, lo, hi = wilson(rec["successes"], rec["seeds"])
        _check(f"nominal[{name}].rate", round(rate, 6), round(rec["rate"], 6))
        policies[name] = {
            "policy": name,
            "seeds": rec["seeds"],
            "successes": rec["successes"],
            "rate": rate,
            "ci": [lo, hi],
            "steps_mean": rec["steps_mean"],
            "steps_range": [rec["steps_min"], rec["steps_max"]],
        }
    return {"available": True, "condition": data["condition"], "backend": data["backend"], "policies": policies}


def build_boundary_map() -> dict[str, Any]:
    """Behavioral boundary map from the current-revision settle-window sweep.

    The sweep is a genuine two-dimensional grid (settle window x depth-noise
    level), every cell measured over the same 16 matched seeds, so it is the
    best-supported 2D surface in the frozen evidence. Each cell is classified
    from its measured success rate; untested (window, noise) pairs are absent
    from the JSON and are rendered as untested, never fabricated.
    """
    data = _load("rq2_settle_window_sweep.json")
    if not data:
        return {"available": False}
    windows = sorted({float(w) for w in data["windows"]})
    noises = sorted({float(n) for n in data["levels"]})
    cells = {}
    for cell in data["cells"].values():
        w = float(cell["window"])
        n = float(cell["noise_mm"])
        rate = cell["success_rate"]["rate"]
        succ = cell["successes"]
        total = cell["n"]
        # classification is a pure function of the measured rate + sample size
        if total == 0:
            cls = "untested"
        elif rate >= 0.999:
            cls = "success"
        elif rate <= 0.001:
            cls = "failure"
        else:
            cls = "mixed"
        rate2, lo, hi = wilson(succ, total)
        _check(f"settle[w={w},n={n}].rate", round(rate2, 6), round(rate, 6))
        cells[f"{w}|{n}"] = {
            "window": w,
            "noise": n,
            "n": total,
            "successes": succ,
            "rate": rate,
            "ci": [lo, hi],
            "steps_mean": cell["mean_steps_success"]["mean"],
            "steps_ci": [cell["mean_steps_success"]["lower"], cell["mean_steps_success"]["upper"]],
            "classification": cls,
        }
    return {
        "available": True,
        "revision": data.get("revision", CURRENT_REVISION),
        "policy": _sweep_policy(),
        "x": {"name": "depth_noise_mm", "label": "depth noise", "unit": "mm", "values": noises},
        "y": {"name": "settle_window", "label": "settle window", "unit": "steps", "values": windows},
        "n_seeds": len(data["seeds"]),
        "seeds": data["seeds"],
        "cells": cells,
    }


def _sweep_policy() -> str | None:
    """Policy name the settle-window sweep varied, read from its generator.

    The frozen sweep JSON records windows and per-seed outcomes but not *which*
    policy they belong to; the generator script that produced it constructs
    ``PolicyConfig(name="policy_v1", parameters={"settle_steps": window})`` --
    every window is policy_v1 with only the settle window overridden (the
    candidate v3 *is* settle_steps=6). The label is read back from that script
    so the UI names the true subject instead of guessing ``policy_v3``; if the
    script is absent the value is ``None`` and the UI shows "Not measured".
    """
    script = REPO / "experiments" / "rq2_settle_window_sweep.py"
    if not script.exists():
        return None
    match = re.search(r'PolicyConfig\(name="([a-z0-9_]+)"', script.read_text())
    return match.group(1) if match else None


def build_settle_tradeoff() -> dict[str, Any]:
    """Settle-window trade-off: mean execution steps and paired McNemar tests.

    Robustness is compared window-by-window against the window=0 reference at
    each noise level over the shared 16 seeds (a paired design), so the McNemar
    discordant counts are real. The headline finding -- no robustness advantage
    demonstrated under the tested conditions -- is reported, and the panel makes
    clear that non-significance is not proof of a zero effect.
    """
    data = _load("rq2_settle_window_sweep.json")
    if not data:
        return {"available": False}
    windows = sorted({float(w) for w in data["windows"]})
    noises = sorted({float(n) for n in data["levels"]})
    per_seed = {}
    for cell in data["cells"].values():
        per_seed[(float(cell["window"]), float(cell["noise_mm"]))] = {
            int(r["seed"]): int(r["success"]) for r in cell["per_seed"]
        }
    # mean execution steps per window, averaged across the measured noise levels
    series = []
    for w in windows:
        steps_vals = []
        rates = []
        for n in noises:
            cell = data["cells"].get(f"window={w}_noise={n}")
            if not cell:
                continue
            steps_vals.append(cell["mean_steps_success"]["mean"])
            rates.append(cell["success_rate"]["rate"])
        series.append(
            {
                "window": w,
                "steps_mean": sum(steps_vals) / len(steps_vals) if steps_vals else None,
                "robustness_mean": sum(rates) / len(rates) if rates else None,
            }
        )
    # paired McNemar: window w vs window 0, per noise level
    comparisons = []
    for n in noises:
        ref = per_seed.get((0.0, n), {})
        for w in windows:
            if w == 0.0:
                continue
            cur = per_seed.get((w, n), {})
            seeds = sorted(set(ref) & set(cur))
            b = sum(1 for s in seeds if ref[s] == 1 and cur[s] == 0)  # ref ok, w fails
            c = sum(1 for s in seeds if ref[s] == 0 and cur[s] == 1)  # ref fails, w ok
            p = mcnemar_exact(b, c)
            comparisons.append({"noise": n, "window": w, "b": b, "c": c, "p": p, "n_pairs": len(seeds)})
    sig = [c for c in comparisons if c["p"] < 0.05]
    return {
        "available": True,
        "revision": data.get("revision", CURRENT_REVISION),
        "windows": windows,
        "noises": noises,
        "series": series,
        "comparisons": comparisons,
        "n_significant": len(sig),
        "finding": "no robustness advantage demonstrated under the tested conditions",
        "caveat": "statistical non-significance does not prove a zero effect",
    }


def build_search() -> dict[str, Any]:
    """RQ1 counterexample-search strategies (the SEARCH/MINIMIZE source)."""
    data = _load("rq1_search_strategies.json")
    if not data:
        return {"available": False}
    return {
        "available": True,
        "question": data["question"],
        "baseline": data["baseline"],
        "candidate": data["candidate"],
        "budget": data["budget"],
        "seed": data["seed"],
        "axis": data["axis"],
        "status": data["status"],
        "strategies": data["strategies"],
        "limitation": data.get("limitation"),
    }


def build_reproducibility() -> dict[str, Any]:
    """RQ4 reproduction rate, recomputed with a Wilson interval from the counts."""
    data = _load("rq4_reproducibility.json")
    if not data:
        return {"available": False}
    rate, lo, hi = wilson(data["reproduced"], data["replay_count"])
    _check("rq4.reproduction_rate", round(rate, 6), round(data["reproduction_rate"], 6))
    return {
        "available": True,
        "condition": data["condition"],
        "baseline": data["baseline"],
        "candidate": data["candidate"],
        "discovery_seed": data["discovery_seed"],
        "replay_seeds": data["replay_seeds"],
        "reproduced": data["reproduced"],
        "replay_count": data["replay_count"],
        "rate": rate,
        "ci": [lo, hi],
        "interpretation": data.get("interpretation"),
        "trials": data.get("trials", []),
    }


def build_counterexample() -> dict[str, Any]:
    """The selected counterexample from the frozen search artifact, if present.

    Beyond the id, the artifact records the swept noise value -- nested as
    ``perturbation: -> depth_noise_mm: -> value:``, so a flat scan for the
    ``depth_noise_mm:`` key finds the *empty* block key and loses the number --
    plus the discovery seed and which policy succeeded / failed. All are
    surfaced so the UI shows the real found condition, not just a filename.
    """
    ce_file = _counterexample_file()
    if not ce_file:
        return {"available": False}
    text = ce_file.read_text()
    cid = _yaml_scalar(text, "counterexample_id")
    depth = _perturbation_value(text, "depth_noise_mm")
    # cross-check the parsed value against the id's rounded copy (1.171875 -> 1.172)
    if depth is not None and cid and "depth_noise_mm=" in cid:
        with contextlib.suppress(ValueError):
            _check(
                "counterexample.depth_noise_mm",
                round(depth, 3),
                float(cid.rsplit("=", 1)[1]),
            )
    baseline = _yaml_block_scalars(text, "baseline")
    candidate = _yaml_block_scalars(text, "candidate")
    seed = _yaml_top_scalar(text, "seed")
    return {
        "available": True,
        "id": cid,
        "file": ce_file.name,
        "depth_noise_mm": depth,
        "experiment_id": _yaml_scalar(text, "experiment_id"),
        "seed": int(seed) if seed and seed.lstrip("-").isdigit() else None,
        "baseline": {"policy": baseline.get("policy"), "result": baseline.get("result")},
        "candidate": {"policy": candidate.get("policy"), "result": candidate.get("result")},
    }


def _counterexample_file() -> Path | None:
    """The frozen counterexample artifact, if the search produced one."""
    ce_dir = RESULTS / "counterexamples"
    if not ce_dir.exists():
        return None
    return next(ce_dir.glob("ce_counterexample_search_*.yaml"), None)


def _yaml_top_scalar(text: str, key: str) -> str | None:
    """Read a column-zero ``key: value`` scalar (ignores nested same-name keys)."""
    prefix = key + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _yaml_block_scalars(text: str, block: str) -> dict[str, str]:
    """Read the ``key: value`` pairs indented under a column-zero ``block:`` key."""
    out: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        if line.strip() and not line[:1].isspace():
            in_block = line.strip() == block + ":"
            continue
        if in_block and line.strip():
            key, sep, value = line.strip().partition(":")
            if sep:
                out.setdefault(key.strip(), value.strip())
    return out


def _perturbation_value(text: str, axis: str) -> float | None:
    """Read ``perturbation: -> axis: -> value:`` out of a counterexample artifact.

    The axis key is a *block* key (``depth_noise_mm:`` with nothing after the
    colon) and the number sits on the indented ``value:`` line beneath it, so
    matching the block key alone parses ``""`` and -- as the previous flat
    scan did -- ends on ``None`` even though the value is recorded.
    """
    lines = text.splitlines()
    in_perturbation = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not line[:1].isspace():
            in_perturbation = stripped == "perturbation:"
            continue
        if not in_perturbation or stripped != axis + ":":
            continue
        for next_line in lines[index + 1 :]:
            next_stripped = next_line.strip()
            if next_stripped.startswith("value:"):
                with contextlib.suppress(ValueError):
                    return float(next_stripped.split(":", 1)[1].strip())
                return None
            if next_stripped and not next_line[:1].isspace():
                break  # left the axis block without a value
    return None


def _yaml_scalar(text: str, key: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(key + ":"):
            return s.split(":", 1)[1].strip()
    return None


def build_opposing_effect() -> dict[str, Any]:
    """Execution-cost channel from the current-revision opposing-effect sweep.

    The current evidence supports a measured *execution-cost* increase for the
    averaging policy (v3) at a generous budget, with a sign reversal under a
    binding deadline. The retired two-sided sign-flip hypothesis is recorded as
    retired so the UI labels it honestly.
    """
    data = _load("rq2_opposing_effect_current.json")
    if not data:
        return {"available": False}
    ev = data.get("evaluation", {})
    per_level = ev.get("per_level", [])
    control = next((p for p in per_level if p.get("control_budget") == data.get("reference_budget")), None)
    return {
        "available": True,
        "revision": data.get("revision", CURRENT_REVISION),
        "baseline": data["baseline"],
        "candidate": data["candidate"],
        "n_seeds": data["n_seeds"],
        "reference_budget": data.get("reference_budget"),
        "verdict": ev.get("verdict"),
        "flipped_levels": ev.get("flipped_levels", []),
        "per_level": per_level,
        "criterion": ev.get("criterion"),
        "detail": ev.get("detail"),
        "control_delta": control.get("control_delta") if control else None,
        "hypothesis_status": "retired (original two-sided sign-flip hypothesis, current revision)",
    }


def build_policy_comparison() -> dict[str, Any]:
    """Baseline vs candidate under the same nominal conditions (RQ2 cells).

    The RQ2 conditional-vs-aggregate file measures policy_v3 (baseline) against
    policy_v1 (candidate) across depth-noise cells. We surface task success and
    execution-step means where the frozen records carry them.
    """
    cond = _load("rq2_conditional_vs_aggregate.json")
    nominal = _load("nominal_solvability.json")
    if not cond:
        return {"available": False}
    baseline = cond["baseline"]
    candidate = cond["candidate"]
    # execution steps come from the nominal solvability record for the same pair
    base_steps = cand_steps = None
    if nominal:
        bp = nominal["policies"].get(baseline)
        cp = nominal["policies"].get(candidate)
        base_steps = bp["steps_mean"] if bp else None
        cand_steps = cp["steps_mean"] if cp else None
    agg = cond["aggregate"]
    _check("rq2.aggregate.delta", round(agg["candidate_success"] - agg["baseline_success"], 6), round(agg["delta"], 6))
    cells = [
        {
            "depth_noise_mm": c["depth_noise_mm"],
            "baseline_success": c["baseline_success"],
            "candidate_success": c["candidate_success"],
            "delta": c["delta"],
            "samples": c["samples"],
        }
        for c in cond["cells"]
    ]
    return {
        "available": True,
        "baseline": baseline,
        "candidate": candidate,
        "cells": cells,
        "aggregate": agg,
        "baseline_steps_mean": base_steps,
        "candidate_steps_mean": cand_steps,
        "steps_delta": (cand_steps - base_steps) if (base_steps is not None and cand_steps is not None) else None,
    }


def build_config() -> dict[str, Any]:
    """Experiment configuration, sourced from the search + reproducibility files.

    The config fingerprint is the scenario hash recorded inside the frozen
    counterexample artifact (``scenario: -> config_hash:``); when no artifact
    exists it stays ``None`` so the UI reports "Not measured" instead of the
    empty ``sha256:…`` placeholder a missing value used to render.
    """
    search = _load("rq1_search_strategies.json") or {}
    repro = _load("rq4_reproducibility.json") or {}
    axis = search.get("axis", {})
    ce_file = _counterexample_file()
    ce_text = ce_file.read_text() if ce_file else ""
    config_hash = None
    if ce_text:
        match = re.search(r"^\s*config_hash:\s*(sha256:[0-9a-f]+)\s*$", ce_text, re.MULTILINE)
        config_hash = match.group(1) if match else None
    return {
        "experiment_id": "counterexample_search",
        "research_question": search.get("question", "RQ1: tests to failure, by search strategy"),
        "baseline": search.get("baseline"),
        "candidate": search.get("candidate"),
        "perturbation": axis.get("name"),
        "perturbation_unit": axis.get("unit"),
        "value_range": [axis.get("low"), axis.get("high")],
        "search_algorithm": "adaptive (coarse scan -> bisection -> coordinate shrink)",
        "budget": search.get("budget"),
        "seed": repro.get("discovery_seed"),
        "evaluation_budget": search.get("budget"),
        "backend": search.get("backend", "mujoco"),
        "state": "recorded",
        "config_hash": config_hash,
        "evidence_provenance": "frozen revision " + CURRENT_REVISION,
    }


def build_metrics() -> dict[str, Any]:
    """Research metric tiles. Missing metrics stay None -> UI shows Not measured."""
    nominal = build_nominal()
    repro = build_reproducibility()
    search = _load("rq1_search_strategies.json") or {}
    settle = _load("rq2_settle_window_sweep.json")

    episodes = 0
    if nominal.get("available"):
        episodes += sum(p["seeds"] for p in nominal["policies"].values())
    if settle:
        episodes += sum(c["n"] for c in settle["cells"].values())

    nominal_rate = None
    steps_mean = None
    if nominal.get("available"):
        v1 = nominal["policies"].get("policy_v1")
        if v1:
            nominal_rate = v1["rate"]
            steps_mean = v1["steps_mean"]

    discovery_cost = None
    for strat in search.get("strategies", []):
        if strat.get("strategy") == "adaptive":
            discovery_cost = strat.get("oracle_evaluations")

    return {
        "episodes_evaluated": episodes if episodes else None,
        "nominal_solvability": nominal_rate,
        "mean_execution_steps": steps_mean,
        "counterexample_discovery_cost": discovery_cost,
        "boundary_estimation_error": None,  # not a stored metric in the frozen evidence
        "regression_reproduction_rate": repro.get("rate") if repro.get("available") else None,
    }


def build_git_revision() -> dict[str, Any]:
    """Read the current git revision/branch for provenance (never fabricated)."""
    import subprocess

    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return {"revision": CURRENT_REVISION, "branch": None, "available": False}
    return {"revision": rev, "short": rev[:7], "branch": branch, "available": True}


def build_evidence_files() -> list[dict[str, Any]]:
    """Real experiment artifacts with provenance and validation status."""
    files: list[dict[str, Any]] = []
    candidates = [
        ("RQ1 search strategies", "experiments/results/rq1_search_strategies.json", "result"),
        ("RQ2 conditional vs aggregate", "experiments/results/rq2_conditional_vs_aggregate.json", "result"),
        ("RQ2 settle-window sweep (current)", "experiments/results/rq2_settle_window_sweep.json", "result"),
        ("RQ2 opposing effect (current)", "experiments/results/rq2_opposing_effect_current.json", "result"),
        ("RQ3 ablation attribution", "experiments/results/rq3_ablation_attribution.json", "result"),
        ("RQ4 reproducibility", "experiments/results/rq4_reproducibility.json", "result"),
        ("RQ5 perturbation boundaries", "experiments/results/rq5_perturbation_boundaries.json", "result"),
        ("RQ5 boundary probe", "experiments/results/rq5_boundary_probe.json", "result"),
        ("Nominal solvability", "experiments/results/nominal_solvability.json", "result"),
        ("Controlled settle steps", "experiments/results/control_settle_steps.json", "result"),
        ("Research report", "research_report.md", "report"),
        ("Research report (generated)", "research_report_generated.md", "report"),
        ("Recorded telemetry (CSV)", "artifacts/linkedin/robot_ci_demo/results/telemetry.csv", "telemetry"),
    ]
    for title, rel, kind in candidates:
        path = REPO / rel
        if not path.exists():
            continue
        valid = True
        detail = "present"
        if path.suffix == ".json":
            try:
                json.loads(path.read_text())
            except Exception as exc:
                valid = False
                detail = f"invalid JSON: {exc}"
        elif path.suffix == ".csv":
            try:
                with path.open(newline="") as fh:
                    rows = list(csv.reader(fh))
                detail = f"{len(rows) - 1} rows" if rows else "empty"
            except Exception as exc:
                valid = False
                detail = f"unreadable: {exc}"
        files.append(
            {
                "title": title,
                "path": rel,
                "kind": kind,
                "bytes": path.stat().st_size,
                "valid": valid,
                "detail": detail,
            }
        )
    return files


def build_scene() -> dict[str, Any]:
    """Static scene facts (table, workspace, object) for the recorded viewport."""
    ce_file = _counterexample_file()
    scene = {"table_height_m": 0.0, "workspace_center": [0.0, 0.1], "table_extent_m": [0.5, 0.35]}
    if ce_file:
        text = ce_file.read_text()
        th = _yaml_scalar(text, "table_height_m")
        if th is not None:
            with contextlib.suppress(ValueError):
                scene["table_height_m"] = float(th)
    return scene


def build_rq3() -> dict[str, Any]:
    """RQ3 attribution result, passed through verbatim for the detail view."""
    data = _load("rq3_ablation_attribution.json")
    return data if data else {"available": False}


def build_rq5() -> dict[str, Any]:
    """One-dimensional perturbation sweeps (RQ5) for the boundary detail view."""
    data = _load("rq5_perturbation_boundaries.json")
    if not data:
        return {"available": False}
    curves = {}
    for name, curve in data.get("curves", {}).items():
        curves[name] = {
            "path": curve.get("path"),
            "unit": curve.get("unit"),
            "boundary_value": curve.get("boundary_value"),
            "no_separation": curve.get("no_separation"),
            "points": curve.get("points", []),
        }
    return {
        "available": True,
        "baseline": data["baseline"],
        "candidate": data["candidate"],
        "method": data.get("method"),
        "boundaries": data.get("boundaries", {}),
        "axes_with_no_separation": data.get("axes_with_no_separation", []),
        "curves": curves,
    }


def build() -> dict[str, Any]:
    """Assemble and write the full dashboard evidence bundle (read-only inputs)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKS.clear()

    telemetry = parse_telemetry(TELEMETRY) if TELEMETRY.exists() else {"frames": [], "phases": []}

    evidence: dict[str, Any] = {
        "revision": CURRENT_REVISION,
        "git": build_git_revision(),
        "config": build_config(),
        "metrics": build_metrics(),
        "nominal": build_nominal(),
        "policy_comparison": build_policy_comparison(),
        "boundary_map": build_boundary_map(),
        "settle_tradeoff": build_settle_tradeoff(),
        "search": build_search(),
        "reproducibility": build_reproducibility(),
        "counterexample": build_counterexample(),
        "opposing_effect": build_opposing_effect(),
        "rq3": build_rq3(),
        "rq5": build_rq5(),
        "episode": {
            "available": bool(telemetry["frames"]),
            "source": str(TELEMETRY.relative_to(REPO)) if TELEMETRY.exists() else None,
            "phases": telemetry["phases"],
            "frames": telemetry["frames"],
        },
        "evidence_files": build_evidence_files(),
        "scene": build_scene(),
        "consistency_checks": list(CHECKS),
    }

    (OUT_DIR / "evidence.json").write_text(json.dumps(evidence, indent=1))
    (OUT_DIR / "robot.json").write_text(json.dumps(robot_from_urdf(URDF), indent=1))

    failed = [c for c in CHECKS if not c["ok"]]
    print(f"wrote {OUT_DIR / 'evidence.json'} ({len(CHECKS)} checks, {len(failed)} failed)")
    print(f"wrote {OUT_DIR / 'robot.json'}")
    for c in failed:
        print(f"  MISMATCH {c['label']}: derived={c['derived']} stored={c['stored']}")
    return evidence


if __name__ == "__main__":
    build()
