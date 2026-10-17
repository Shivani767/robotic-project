"""RQ2 opposing-effect re-run at the current revision (d7026af).

The previous RQ2 grid (``rq2_opposing_effect.json``) was calibrated against a
prior code revision whose nominal cost was baseline 76 / candidate 81 steps.
The grasp-aim-clearance fix raised the current nominal cost to v1 ~103-105 and
v3 ~108-110, so every tight budget in that grid (78-84) now sits below *both*
arms' cost and its sign flip cannot be reproduced here. That prior artifact is
left untouched as prior-revision evidence; this script writes a new, separately
keyed artifact for the current code.

Design, derived from measurement rather than copied from the stale grid:

* Budgets are centred on the two measured clusters -- nominal ~103-112 and
  slow-recovery ~155-162 -- resolved finely across the split region (104-112)
  where a deadline can refuse the candidate's +5-step settle cost while still
  admitting the baseline, plus the generous control (300).
* The generous 300-step runs double as the uncapped reference, so every
  censored (timeout) episode can be checked against what that same (arm, seed)
  needed when unconstrained. A deadline effect is real only if the censored arm
  would have succeeded given more steps; an episode that would have failed
  anyway is not evidence of a deadline.
* Both arms share identical seeds, scenario, and stopping criterion inside a
  cell, so any within-cell difference is the pair (policy, budget), never an
  evaluation-opportunity imbalance.
* Per-seed outcomes, steps, and terminal status are recorded, and a censored
  episode is labelled distinctly from a genuine task failure.

Sign-flip criterion (PRE-REGISTERED before any cell is inspected): the opposing
effect is a *sign reversal across channels*, and the honest unit of analysis is
the noise level. Within one noise column the candidate (policy_v3, which
averages pose samples) should beat the baseline when the budget is generous
(averaging absorbs noise) yet lose to it when a deadline binds (settling costs
~5 extra steps). So, per noise level, we compare the control cell (budget 300)
against the worst tight cell (budget <= 112):

    margin flip := control delta >= +0.05 AND worst tight delta <= -0.10
    strict flip := control delta  >    0  AND worst tight delta <    0

The margin form is the primary criterion (guards against a near-zero noise
floor); the strict sign-inversion is reported per level so the reader sees the
sensitivity to the threshold. The overall verdict is REPRODUCED iff any noise
level meets the margin criterion. A null is reported as measured, not as a
failure to find the expected result.

Usage:
    python experiments/rq2_opposing_effect_current.py [--seeds N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from robot_ci.experiments.runner import EpisodeResult, run_episode
from robot_ci.models.scenario import PolicyConfig, variable_path
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import create_backend
from robot_ci.statistics.intervals import proportion_ci

RESULTS = Path("experiments/results")

BASELINE = "policy_v1"
CANDIDATE = "policy_v3"

# Favourable channel: depth-sensor noise the candidate's averaging absorbs.
NOISE_MM = (0.0, 1.0, 2.0, 4.0)

# Adverse channel: a deadline, resolved across both measured clusters.
# 104-112 straddle the nominal split (v1 103-105 vs v3 108-110); 155-161 the
# slow-recovery split (v1 155-157 vs v3 159-162); 120 the inter-cluster gap;
# 300 the uncapped control and truncation reference.
STEP_BUDGETS = (104, 106, 108, 110, 112, 120, 155, 158, 161, 300)

SEEDS = tuple(range(1, 25))  # 24 matched seeds: enough to test seed robustness.

# Terminal statuses that mean "the deadline stopped it", not "the task failed".
CENSORED = frozenset({"timeout", "step_budget_exhausted"})


def _code_revision() -> str:
    """The commit these numbers are produced against, recorded not assumed."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance best-effort
        return "unknown"


def build(noise_mm: float, max_steps: int) -> Any:
    """One grid cell: depth noise and step deadline pinned."""
    scenario = set_by_path(default_scenario(), variable_path("depth_noise_mm"), noise_mm)
    return scenario.model_copy(
        update={"task": scenario.task.model_copy(update={"max_steps": max_steps})}
    )


def config_hash(noise_mm: float, max_steps: int) -> str:
    """Stable hash of the exact scenario a cell runs, for traceability."""
    blob = json.dumps(build(noise_mm, max_steps).model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_cell(engine: Any, noise_mm: float, budget: int, seeds) -> dict[str, list]:
    """Run both arms over every seed, paired by seed."""
    baseline: list[EpisodeResult] = []
    candidate: list[EpisodeResult] = []
    for arm, sink in ((BASELINE, baseline), (CANDIDATE, candidate)):
        for seed in seeds:
            scenario = build(noise_mm, budget)
            sink.append(
                run_episode(
                    engine,
                    create_policy(PolicyConfig(name=arm), scenario),
                    scenario,
                    seed=seed,
                    experiment_id="rq2_opposing_current",
                )
            )
    return {"baseline": baseline, "candidate": candidate}


def _summarise(
    results: list[EpisodeResult],
    n: int,
    reference: list[EpisodeResult] | None = None,
) -> dict[str, Any]:
    """Per-arm cell summary, with deadline-censoring separated from genuine stall.

    Censoring is detected by the *uncapped-reference cross-check*, NOT by the
    terminal ``status`` string. Verified on this code: the pick-and-place task
    never emits the ``timeout`` / ``step_budget_exhausted`` statuses -- on budget
    exhaustion it returns a plain ``failure`` -- so a status-based censoring count
    always reads zero and every deadline-refused episode would be mislabelled a
    genuine stall (a truncation artifact, exactly what the audit must avoid).

    A seed is *deadline_refused* iff it failed at this budget while the uncapped
    (300-step) reference for the same seed succeeded; it *would_fail_anyway* iff
    it failed at both. When no reference is supplied (the reference cell itself)
    every non-success is conservative and counted as would_fail_anyway.
    """
    ref_success = {r.seed: bool(r.success) for r in reference} if reference is not None else {}
    succ = sum(bool(r.success) for r in results)
    if reference is not None:
        deadline_refused = sum(
            1 for r in results if not r.success and ref_success.get(r.seed) is True
        )
        would_fail_anyway = sum(
            1 for r in results if not r.success and ref_success.get(r.seed) is False
        )
    else:
        deadline_refused = 0
        would_fail_anyway = sum(1 for r in results if not r.success)
    steps = [r.steps for r in results]
    return {
        "success": succ,
        "n": n,
        "rate": succ / n,
        "deadline_refused": deadline_refused,
        "would_fail_anyway": would_fail_anyway,
        "steps_mean": sum(steps) / len(steps),
        "steps_min": min(steps),
        "steps_max": max(steps),
        "per_seed": [
            {
                "seed": r.seed,
                "success": bool(r.success),
                "steps": r.steps,
                "status": r.status,
            }
            for r in results
        ],
    }


def _cell_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Within-cell (candidate - baseline) difference and its 95% CI on the delta."""
    n = baseline["n"]
    b_ci = proportion_ci(baseline["success"], n)
    c_ci = proportion_ci(candidate["success"], n)
    delta = c_ci.rate - b_ci.rate
    # Conservative SE for a difference of proportions from the two arms' widths.
    se = (b_ci.upper - b_ci.lower) / (2 * 1.96) + (c_ci.upper - c_ci.lower) / (2 * 1.96)
    se = max(se, 1e-9) / 2
    return {
        "delta": delta,
        "lower": delta - 1.96 * se,
        "upper": delta + 1.96 * se,
        "baseline_rate": b_ci.rate,
        "candidate_rate": c_ci.rate,
    }


def _attributable(results: list[EpisodeResult], reference: list[EpisodeResult]) -> int:
    """Censored episodes the uncapped reference shows would have succeeded.

    A deadline effect is real only if a censored episode would have succeeded
    given more steps. Episodes that would have failed anyway (stalls) are not
    evidence of a deadline. Returns the count of genuinely deadline-refused seeds.
    """
    ref_by_seed = {r.seed: r.success for r in reference}
    return sum(
        1
        for r in results
        if not r.success and ref_by_seed.get(r.seed) is True
    )

def _preregistered_verdict(cells: dict[str, Any]) -> dict[str, Any]:
    """Apply the per-noise-level criterion frozen in the docstring.

    The unit of analysis is the noise level. Within one noise column the
    candidate (policy_v3, averaging) should beat the baseline under a generous
    budget (averaging absorbs noise) yet lose under a binding deadline (settling
    costs ~5 extra steps). For each noise level we compare the control cell
    (budget >= 300) against the worst tight cell (budget <= 112):

        control delta >= +0.05  (averaging helps)  AND  worst tight delta <= -0.10

    The margin form is primary; the strict sign inversion (control > 0 AND
    worst tight < 0) is also recorded per level so the reader sees sensitivity
    to the thresholds. The overall verdict is REPRODUCED iff ANY noise level
    meets the margin criterion; every level is reported transparently so the
    conditions of any flip (or its absence) are explicit. A null is reported as
    measured, not as a failure to find the expected result.
    """
    per_level: list[dict[str, Any]] = []
    for noise in sorted({c["noise_mm"] for c in cells.values()}):
        column = [c for c in cells.values() if c["noise_mm"] == noise]
        controls = [c for c in column if c["budget"] >= 300]
        tights = [c for c in column if c["budget"] <= 112]
        if not controls or not tights:
            continue
        control = max(controls, key=lambda c: c["delta"]["delta"])
        worst_tight = min(tights, key=lambda c: c["delta"]["delta"])
        control_delta = control["delta"]["delta"]
        tight_delta = worst_tight["delta"]["delta"]
        per_level.append(
            {
                "noise_mm": noise,
                "control_budget": control["budget"],
                "control_delta": control_delta,
                "control_ci": [control["delta"]["lower"], control["delta"]["upper"]],
                "worst_tight_budget": worst_tight["budget"],
                "worst_tight_delta": tight_delta,
                "worst_tight_ci": [
                    worst_tight["delta"]["lower"],
                    worst_tight["delta"]["upper"],
                ],
                "candidate_deadline_refused": worst_tight["candidate_deadline_refused"],
                "margin_pass": control_delta >= +0.05 and tight_delta <= -0.10,
                "strict_sign_flip": control_delta > 0.0 and tight_delta < 0.0,
            }
        )
    flipped = [lv for lv in per_level if lv["margin_pass"]]
    reproduced = bool(flipped)
    if reproduced:
        where = ", ".join(f"{lv['noise_mm']:g} mm" for lv in flipped)
        detail = (
            f"Opposing effect REPRODUCED at {_code_revision()} at noise level(s): "
            f"{where}. Within those columns the candidate wins under the generous "
            f"budget (control delta >= +0.05) and loses under a binding deadline "
            f"(worst tight delta <= -0.10) -- a genuine sign reversal across channels."
        )
    else:
        shown = "; ".join(
            f"{lv['noise_mm']:g} mm (control {lv['control_delta']:+.3f}, "
            f"worst tight {lv['worst_tight_delta']:+.3f} "
            f"@ budget {lv['worst_tight_budget']})"
            for lv in per_level
        )
        detail = (
            f"NOT reproduced at {_code_revision()}: no noise column shows the "
            f"candidate winning under the generous budget while losing under a "
            f"binding deadline. Per level -- {shown}. The prior sign flip was a "
            f"prior-revision result and is retired for this code."
        )
    return {
        "criterion": (
            "per noise level: control delta >= +0.05 (budget >= 300) AND worst "
            "tight delta <= -0.10 (budget <= 112); strict sign inversion also "
            "recorded. REPRODUCED iff any noise level meets the margin criterion."
        ),
        "verdict": "reproduced" if reproduced else "not_reproduced",
        "per_level": per_level,
        "flipped_levels": [lv["noise_mm"] for lv in flipped],
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the controlled grid, checkpoint after every cell, then evaluate."""
    parser = argparse.ArgumentParser(description="RQ2 opposing-effect re-run (current revision)")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="number of matched seeds")
    args = parser.parse_args(argv)
    seeds = SEEDS[: args.seeds]

    revision = _code_revision()
    engine = create_backend("mujoco")
    started = time.time()
    cells: dict[str, Any] = {}
    out_path = RESULTS / "rq2_opposing_effect_current.json"
    reference = max(STEP_BUDGETS)

    payload: dict[str, Any] = {
        "experiment": "rq2_opposing_effect_current",
        "revision": revision,
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "noise_mm": list(NOISE_MM),
        "step_budgets": list(STEP_BUDGETS),
        "censored_statuses": sorted(CENSORED),
        "reference_budget": reference,
        "started_unix": int(started),
        "cells": cells,
    }

    print(f"RQ2 re-run at {revision}; {len(seeds)} seeds x {len(NOISE_MM)} noise x "
          f"{len(STEP_BUDGETS)} budgets", flush=True)
    try:
        for noise in NOISE_MM:
            # The reference cell (uncapped) is measured first and reused to
            # attribute every censored episode in this noise column.
            ref_pair = run_cell(engine, noise, reference, seeds)
            ref_key = f"noise={noise:g}_budget={reference}"
            ref_c = _summarise(ref_pair["candidate"], len(seeds))
            cells[ref_key] = {
                "noise_mm": noise,
                "budget": reference,
                "config_hash": config_hash(noise, reference),
                "baseline": _summarise(ref_pair["baseline"], len(seeds)),
                "candidate": ref_c,
                "delta": _cell_delta(
                    _summarise(ref_pair["baseline"], len(seeds)), ref_c
                ),
                "candidate_deadline_refused": 0,
                "is_reference": True,
            }
            for budget in (b for b in STEP_BUDGETS if b != reference):
                pair = run_cell(engine, noise, budget, seeds)
                # Pass the uncapped reference so deadline-censoring is separated
                # from genuine stall per arm (see ``_summarise``). Without it
                # ``deadline_refused``/``would_fail_anyway`` would always read 0.
                b_sum = _summarise(
                    pair["baseline"], len(seeds), ref_pair["baseline"]
                )
                c_sum = _summarise(
                    pair["candidate"], len(seeds), ref_pair["candidate"]
                )
                key = f"noise={noise:g}_budget={budget}"
                cells[key] = {
                    "noise_mm": noise,
                    "budget": budget,
                    "config_hash": config_hash(noise, budget),
                    "baseline": b_sum,
                    "candidate": c_sum,
                    "delta": _cell_delta(b_sum, c_sum),
                    "candidate_deadline_refused": _attributable(
                        pair["candidate"], ref_pair["candidate"]
                    ),
                    "is_reference": False,
                }
                d = cells[key]["delta"]
                print(
                    f"  {key}: base {b_sum['success']}/{len(seeds)} "
                    f"(refused {b_sum['deadline_refused']})  cand {c_sum['success']}/{len(seeds)} "
                    f"(refused {c_sum['deadline_refused']})  "
                    f"delta {d['delta']:+.3f} [{d['lower']:+.3f},{d['upper']:+.3f}]",
                    flush=True,
                )
                out_path.write_text(json.dumps(payload, indent=2))
    finally:
        engine.close_all()

    evaluation = _preregistered_verdict(cells)
    payload["evaluation"] = evaluation
    payload["wall_s"] = round(time.time() - started, 1)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nPRE-REGISTERED VERDICT: {evaluation['verdict']}")
    print(evaluation["detail"])
    print(f"wrote {out_path} ({payload['wall_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

