"""RQ2 follow-up: is the candidate's *uncapped* advantage at 1 mm real?

The controlled RQ2 re-run (``rq2_opposing_effect_current``) reproduced the
opposing-effect sign inversion at noise = 1 mm, but its **positive leg** was
fragile: at the uncapped 300-step control the baseline won 21/24 and the
candidate 23/24 (delta +0.083, 95% CI [-0.032, +0.199] -- straddling zero), and
the entire margin rested on just **two net discordant seeds** (3 candidate-only
wins vs 1 baseline-only win). An exact McNemar test on those 4 discordant pairs
is nowhere near significant (p ~= 0.63). That is too thin to support the claim
that "the candidate wins when unconstrained", which is the hinge of the whole
two-sided inversion.

This probe decides that one question and nothing else.

    Q: at noise = 1 mm, uncapped (budget 300), is the candidate (policy_v3,
       settle_steps = 6) *genuinely* more successful than the baseline
       (policy_v1, settle_steps = 0), or is the positive leg itself zero?

**Pre-registration (frozen before any outcome is inspected).**

Primary analysis -- the paired per-seed success difference (candidate minus
baseline), because the two arms are run on identical matched seeds and so the
paired contrast has far more power than two independent proportions. The
decisive inferential test is an **exact McNemar (sign) test on the discordant
pairs** (seeds where exactly one arm succeeded), two-sided, alpha = 0.05 --
the correct test when an effect is small and carried by few discordant seeds,
and the reason a margin-of-2 on 24 seeds could not be trusted.

Decision rule (primary):
  * reject H0 (McNemar p < 0.05) AND point estimate > 0
        -> the candidate's uncapped advantage is **demonstrated**; the
           two-sided inversion stands on both legs.
  * otherwise (p >= 0.05, or point estimate <= 0)
        -> the uncapped advantage is **not distinguished from zero**; retire the
           "candidate wins unconstrained" claim. RQ2 is then only a *one-sided
           deadline penalty*, not a genuine two-sided opposing effect.

Secondary (descriptive, reported alongside, never used as the gate): each arm's
Wilson rate CI, Cohen's h, and a bootstrap CI on the paired per-seed delta.

Conditions are imported unchanged from ``rq2_opposing_effect_current`` (same
``build``/``run_cell``, same depth-noise and max_steps plumbing) so this probe
measures exactly the control cell of the completed run, only with more seeds.

Raw per-seed records are written incrementally after every seed pair so a long
run cannot be lost. Nothing is overwritten; the completed RQ2 artifact stands.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from robot_ci.experiments.runner import EpisodeResult
from robot_ci.statistics.intervals import bootstrap_mean_ci, cohens_h, proportion_ci

RESULTS = Path("experiments/results")

BASELINE = "policy_v1"
CANDIDATE = "policy_v3"

# The single cell under test: the uncapped control at the one noise level where
# the inversion appeared. A step deadline of 300 is the truncation reference
# (well above both arms' slow-recovery cluster at ~155-162), i.e. genuinely
# unconstrained -- this is the leg whose sign the RQ2 verdict turned on.
NOISE_MM = 1.0
BUDGET = 300

# 150 matched seeds: ~6x the original 24, chosen so a McNemar test on the
# discordant pairs has enough of them to separate a real (candidate-favouring)
# asymmetry from the coin-flip the 24-seed margin could not resolve.
SEEDS = tuple(range(1, 151))

# Determinism provenance: the arms share the seed, so the paired contrast is
# valid only if the backend is reproducible per (seed, policy). Verified in the
# RQ4 audit (outcomes/steps bit-identical across repeats); re-asserted here by
# recording the code revision the run executes at.


def _load_rq2_module() -> Any:
    """Import the completed RQ2 script to reuse its exact cell construction.

    Guarantees the probe runs the *identical* scenario/noise/budget plumbing as
    the control cell it is adjudicating -- no silent condition drift.
    """
    path = Path(__file__).with_name("rq2_opposing_effect_current.py")
    spec = importlib.util.spec_from_file_location("rq2_current_mod", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _mcnemar_exact(cand_only: int, base_only: int) -> dict[str, Any]:
    """Exact (sign) test on discordant pairs.

    Under H0 each discordant seed is equally likely to favour either arm, so the
    number favouring the candidate is Binomial(n_discordant, 0.5). Uses SciPy's
    exact binomial test when present; falls back to a two-sided normal
    approximation otherwise (recorded via ``method``).
    """
    n_disc = cand_only + base_only
    if n_disc == 0:
        return {"p_value": 1.0, "n_discordant": 0, "method": "no-discordant-pairs"}
    try:
        from scipy.stats import binomtest

        p = float(binomtest(cand_only, n_disc, 0.5, alternative="two-sided").pvalue)
        return {"p_value": p, "n_discordant": n_disc, "method": "exact-binomial"}
    except ImportError:
        z = (cand_only - n_disc / 2.0) / (0.5 * (n_disc**0.5))
        p = 2.0 * (1.0 - 0.5 * (1.0 + _erf(abs(z) / (2**0.5))))
        return {"p_value": float(p), "n_discordant": n_disc, "method": "normal-approx"}


def _erf(x: float) -> float:
    import math

    return math.erf(x)


def main() -> None:
    """Adjudicate the RQ2 positive leg at noise=1 mm, uncapped, 150 matched seeds."""
    mod = _load_rq2_module()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "rq2_positive_leg_probe.json"
    started = time.time()
    config_hash = "unknown"

    per_seed: list[dict[str, Any]] = []
    engine = mod.create_backend("mujoco")
    try:
        for i, seed in enumerate(SEEDS, 1):
            row: dict[str, Any] = {"seed": seed}
            for arm in (BASELINE, CANDIDATE):
                scenario = mod.build(NOISE_MM, BUDGET)
                res: EpisodeResult = mod.run_episode(
                    engine,
                    mod.create_policy(mod.PolicyConfig(name=arm), scenario),
                    scenario,
                    seed=seed,
                    experiment_id="rq2_positive_leg_probe",
                )
                config_hash = res.config_hash or config_hash
                row[arm] = {
                    "success": bool(res.success),
                    "steps": int(res.steps),
                    "status": str(res.status),
                }
            per_seed.append(row)
            # Incremental checkpoint every 10 pairs so a long run can't be lost.
            if i % 10 == 0 or i == len(SEEDS):
                _checkpoint(out_path, per_seed, config_hash, started, partial=i < len(SEEDS))
                print(f"  {i}/{len(SEEDS)} seed pairs ({time.time() - started:.0f}s)", flush=True)
    finally:
        engine.close_all()

    base = [r[BASELINE]["success"] for r in per_seed]
    cand = [r[CANDIDATE]["success"] for r in per_seed]
    n = len(per_seed)
    base_wins = sum(base)
    cand_wins = sum(cand)
    # Paired per-seed delta (candidate minus baseline), one entry per seed.
    deltas = [float(c) - float(b) for c, b in zip(cand, base, strict=True)]
    cand_only = sum(1 for d in deltas if d > 0)
    base_only = sum(1 for d in deltas if d < 0)
    both = sum(1 for c, b in zip(cand, base, strict=True) if c and b)
    neither = sum(1 for c, b in zip(cand, base, strict=True) if not c and not b)

    base_ci = proportion_ci(base_wins, n)
    cand_ci = proportion_ci(cand_wins, n)
    paired_ci = bootstrap_mean_ci(deltas, n_resamples=20000, seed=0)
    mcn = _mcnemar_exact(cand_only, base_only)
    point = cand_ci.rate - base_ci.rate

    # Primary decision rule (frozen in the module docstring).
    demonstrated = mcn["p_value"] < 0.05 and point > 0
    verdict = (
        "candidate's uncapped advantage DEMONSTRATED (two-sided inversion stands)"
        if demonstrated
        else "uncapped advantage NOT distinguished from zero "
        "(RQ2 is a one-sided deadline penalty, not a two-sided opposing effect)"
    )

    evaluation = {
        "question": (
            "At noise=1 mm uncapped (budget 300), is the candidate (policy_v3, "
            "settle_steps=6) genuinely more successful than baseline (policy_v1)?"
        ),
        "noise_mm": NOISE_MM,
        "budget": BUDGET,
        "n_seeds": n,
        "baseline_rate": base_ci.rate,
        "baseline_ci": [base_ci.lower, base_ci.upper],
        "candidate_rate": cand_ci.rate,
        "candidate_ci": [cand_ci.lower, cand_ci.upper],
        "point_delta_candidate_minus_baseline": point,
        "paired_bootstrap_delta": {
            "mean": paired_ci.mean,
            "ci": [paired_ci.lower, paired_ci.upper],
            "method": paired_ci.method,
        },
        "cohens_h": cohens_h(cand_ci.rate, base_ci.rate),
        "discordant": {
            "candidate_only_wins": cand_only,
            "baseline_only_wins": base_only,
            "both_succeeded": both,
            "neither_succeeded": neither,
        },
        "mcnemar_exact": mcn,
        "primary_decision_rule": "reject H0 iff McNemar p < 0.05 AND point estimate > 0",
        "verdict": verdict,
        "verdict_flag": "DEMONSTRATED" if demonstrated else "NOT-DEMONSTRATED",
    }

    payload = {
        "experiment": "rq2_positive_leg_probe",
        "revision": _code_revision(),
        "config_hash": config_hash,
        "backend": "mujoco",
        "baseline_policy": BASELINE,
        "candidate_policy": CANDIDATE,
        "seeds": list(SEEDS),
        "duration_wall_s": round(time.time() - started, 1),
        "completed": True,
        "evaluation": evaluation,
        "per_seed": per_seed,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  baseline {base_wins}/{n} = {base_ci.rate:.3f} CI [{base_ci.lower:.3f}, {base_ci.upper:.3f}]")
    print(f"  candidate {cand_wins}/{n} = {cand_ci.rate:.3f} CI [{cand_ci.lower:.3f}, {cand_ci.upper:.3f}]")
    print(f"  paired delta = {paired_ci.mean:+.4f} CI [{paired_ci.lower:.3f}, {paired_ci.upper:.3f}]")
    print(
        f"  McNemar exact p = {mcn['p_value']:.4f} "
        f"({cand_only} cand-only vs {base_only} base-only of {mcn['n_discordant']} discordant)"
    )
    print(f"  VERDICT: {verdict}")
    print(f"wrote {out_path} ({time.time() - started:.1f}s)")


def _checkpoint(path: Path, per_seed: list[dict[str, Any]], config_hash: str, started: float, *, partial: bool) -> None:
    """Write a partial payload so incremental progress is durable."""
    payload = {
        "experiment": "rq2_positive_leg_probe",
        "revision": _code_revision(),
        "config_hash": config_hash,
        "backend": "mujoco",
        "noise_mm": NOISE_MM,
        "budget": BUDGET,
        "completed": not partial,
        "duration_wall_s": round(time.time() - started, 1),
        "per_seed": per_seed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

