"""RQ2 follow-up (settle-window sweep): is the candidate's cost reducible?

The controlled re-run (``rq2_opposing_effect_current``) established that the
candidate (``policy_v3``) differs from the baseline (``policy_v1``) by exactly
one parameter, ``settle_steps`` (0 -> 6), and that this costs ~5 steps under
nominal conditions -- a cost that a tight deadline refuses first (the one-sided
deadline penalty). Mechanically, ``_settled_estimate`` holds position for
``settle_steps - 1`` ticks while it averages pose samples, so the cost is
*intrinsic to the averaging technique by construction*: any window ``w > 0``
must pay ~``w - 1`` hold ticks.

The open question this sweep decides is whether that cost is *all-or-nothing*
with the robustness benefit, or whether a **smaller** window keeps most of the
robustness while paying a smaller step penalty. If a cheaper equivalent of the
candidate exists, the deadline penalty that retired RQ2's two-sided hypothesis
could be engineered away without abandoning averaging.

PRE-REGISTERED DESIGN (fixed before any outcome is inspected)
-------------------------------------------------------------
* Windows: ``settle_steps`` in {0, 2, 3, 4, 6}. ``0`` is the baseline (no
  averaging); ``6`` is the published candidate; {2, 3, 4} probe intermediate
  windows. Each window is ``policy_v1`` with only ``settle_steps`` overridden,
  matching the validated ``control_settle_steps`` intervention.
* Noise grid: depth ``sigma_mm`` in {0.0, 1.0, 2.0, 4.0}.
* Seeds: 16 matched seeds (1..16), identical initial conditions, uncapped
  budget (``task.max_steps`` = 400, well above the ~155-162 slow-recovery
  ceiling) so step-cost is never censored.
* Every episode is recorded per-seed (success, steps, status).

PRE-REGISTERED ANALYSIS (the claims we will make, set in advance)
----------------------------------------------------------------
1. Cost curve (nominal, noise=0): the mean uncapped step-count of successful
   episodes is **monotone non-decreasing** in ``settle_steps``, with the
   per-unit slope ~1 hold tick. We report the mean and a 95% bootstrap CI per
   window rather than asserting a deterministic cost.
2. Robustness (under noise): for each window we report the per-level success
   rate with a 95% Wilson CI. The decisive question is whether **any**
   intermediate window (2/3/4) is statistically non-inferior to the full
   window (6) on robustness while costing strictly fewer steps. We define
   non-inferiority by overlapping Wilson CIs at every noise level (a
   descriptive test, reported as such -- n=16 per cell cannot resolve small
   differences, and we will say so rather than over-claim).
3. We do **not** tune ``settle_steps`` to maximise success; we report the full
   curve and let the reader see the cost/robustness trade-off. If no cheaper
   equivalent exists, we say the cost is intrinsic and the deadline penalty
   stands as the supported finding.

The result adjudicates the "next decisive step" left open by the RQ2 decision:
whether the candidate's step cost is a fixed tax of averaging or a tunable one.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend
from robot_ci.statistics.intervals import mean_ci, proportion_ci

#: settle_steps windows to probe. 0 = baseline (no averaging), 6 = candidate.
WINDOWS: tuple[float, ...] = (0.0, 2.0, 3.0, 4.0, 6.0)
#: Depth-noise grid, in mm of sigma.
LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
#: Matched seeds for paired, identical-initial-condition comparison.
SEEDS: tuple[int, ...] = tuple(range(1, 17))
#: Uncapped budget: above the ~162-step slow-recovery ceiling, so cost is real.
UNCAPPED_MAX_STEPS = 400
#: Sensor-noise path (depth sigma, in mm) matching the RQ2 re-run grid.
NOISE_PATH = "sensors.depth.sigma_mm"

OUTPUT = Path("experiments/results/rq2_settle_window_sweep.json")


def _code_revision() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"

def _run_cell(window: float, level: float) -> dict:
    """Run all seeds for one (settle_steps, noise) cell, uncapped, per-seed recorded."""
    cond = set_by_path(default_scenario(), NOISE_PATH, level)
    cond = set_by_path(cond, "task.max_steps", UNCAPPED_MAX_STEPS)
    engine = create_backend("mujoco")
    try:
        per_seed: list[dict] = []
        for seed in SEEDS:
            r = run_episode(
                engine,
                create_policy(
                    PolicyConfig(name="policy_v1", parameters={"settle_steps": window}),
                    cond,
                ),
                cond,
                seed=seed,
                experiment_id="rq2_settle_window_sweep",
            )
            per_seed.append(
                {
                    "seed": seed,
                    "success": int(bool(r.success)),
                    "steps": int(r.steps),
                    "status": str(r.status),
                }
            )
    finally:
        engine.close_all()

    n = len(per_seed)
    successes = sum(p["success"] for p in per_seed)
    rate = proportion_ci(successes, n)
    ok_steps = [p["steps"] for p in per_seed if p["success"]]
    cost = mean_ci(ok_steps) if ok_steps else None
    return {
        "window": window,
        "noise_mm": level,
        "n": n,
        "successes": successes,
        "success_rate": {
            "rate": rate.rate,
            "lower": rate.lower,
            "upper": rate.upper,
        },
        "mean_steps_success": (
            {"mean": cost.mean, "lower": cost.lower, "upper": cost.upper}
            if cost is not None
            else None
        ),
        "per_seed": per_seed,
    }


def main() -> int:
    """Sweep settle_steps windows x noise; cost curve + robustness, checkpointed."""
    cells: dict[str, dict] = {}
    for window in WINDOWS:
        for level in LEVELS:
            key = f"window={window}_noise={level}"
            cell = _run_cell(window, level)
            cells[key] = cell
            mr = cell["mean_steps_success"]
            step_s = (
                f"{mr['mean']:.1f} [{mr['lower']:.1f},{mr['upper']:.1f}]"
                if mr is not None
                else "n/a"
            )
            print(
                f"window={window} noise={level}: "
                f"succ {cell['successes']}/{cell['n']} "
                f"(rate {cell['success_rate']['rate']:.3f}) steps {step_s}"
            )
            # Incremental checkpoint so a long run is never lost.
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "revision": _code_revision(),
                "windows": list(WINDOWS),
                "levels": list(LEVELS),
                "seeds": list(SEEDS),
                "uncapped_max_steps": UNCAPPED_MAX_STEPS,
                "noise_path": NOISE_PATH,
                "cells": cells,
            }
            OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

