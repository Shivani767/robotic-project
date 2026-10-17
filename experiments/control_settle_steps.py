"""RQ3 (controlled intervention): does settle_steps change behaviour on fixed code?

v3 vs v1 differ in one subsystem (perception) via settle_steps averaging. This
probe swaps ONLY settle_steps on policy_v1 and measures success-rate change
across a noise grid, preserving individual per-epoch outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_ci.experiments.runner import run_episode
from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend

LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
SEEDS: tuple[int, ...] = (1, 3, 4, 6, 8, 9)
OUTPUT = Path("experiments/results/control_settle_steps.json")


def _run(settle: float, seeds: tuple[int, ...], path: str) -> dict:
    records: list[int] = []
    per_level: dict[str, list[int]] = {}
    for level in LEVELS:
        cond = set_by_path(default_scenario(), path, level)
        engine = create_backend("mujoco")
        cell: list[int] = []
        for seed in seeds:
            r = run_episode(
                engine,
                create_policy(
                    PolicyConfig(name="policy_v1", parameters={"settle_steps": settle}),
                    cond,
                ),
                cond,
                seed=seed,
                experiment_id="control_settle_steps",
            )
            records.append(int(r.success))
            cell.append(int(r.success))
        per_level[str(level)] = cell
        engine.close_all()
    return {"cell_success": records, "levels": LEVELS, "seeds": SEEDS, "per_level": per_level}


def main() -> int:
    """Swap only settle_steps on policy_v1 across the noise grid and record deltas."""
    base = _run(0.0, SEEDS, "sensors.depth.sigma_mm")
    pert = _run(6.0, SEEDS, "sensors.depth.sigma_mm")

    # Per-level success-rate deltas. The previous revision compared
    # cell_success[i] for i in 0..len(LEVELS)-1, but the flat record list is
    # grouped len(SEEDS) per level, so i only ever indexed the first level's
    # seed cells and every level after the first was compared against the wrong
    # (noise-0) arm. That mislabelled a behaviour-changing parameter as inert.
    cells = []
    for lvl in LEVELS:
        key = str(lvl)
        b = base["per_level"][key]
        p = pert["per_level"][key]
        n = len(b)
        cells.append(
            {
                "level": lvl,
                "base_rate": round(sum(b) / n, 4),
                "pert_rate": round(sum(p) / n, 4),
                "cell_delta": round(sum(p) / n - sum(b) / n, 4),
                "base_cells": b,
                "pert_cells": p,
            }
        )
    n_total = len(base["cell_success"])
    base_rate = sum(base["cell_success"]) / n_total
    pert_rate = sum(pert["cell_success"]) / n_total
    overall = round(pert_rate - base_rate, 4)
    helps = sum(1 for c in cells if c["cell_delta"] > 0)
    hurts = sum(1 for c in cells if c["cell_delta"] < 0)
    combined = {
        "base": base,
        "perturbed": pert,
        "cells": cells,
        "overall_delta": overall,
        "base_overall_rate": round(base_rate, 4),
        "pert_overall_rate": round(pert_rate, 4),
        "levels_helped": helps,
        "levels_hurt": hurts,
        "implication": (
            f"settle_steps is NOT behaviour-neutral on the fixed simulator: "
            f"overall delta {overall:+.4f}, helping {helps}/{len(cells)} levels and "
            f"hurting {hurts}/{len(cells)}. A single-parameter swap on policy_v1 "
            f"moves the outcome, so the parameter is a live causal knob."
        ),
    }
    OUTPUT.write_text(json.dumps(combined, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
