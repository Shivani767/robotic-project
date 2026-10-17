"""Tests for the ROBOT CI dashboard data layer (``experiments/build_dashboard_data.py``).

These tests pin down the guarantees the dashboard depends on:

* parsing and schema validation of the frozen JSON and the telemetry CSV;
* metric aggregation (Wilson intervals, McNemar) and policy differences;
* heatmap classification of measured cells and of untested regions;
* provenance and current-revision labels;
* missing-metric handling (unavailable stays ``None``, never a filler number);
* replay timeline construction (ordered frames + only present phases);
* consistency between derived aggregates and the values stored in the JSON.

They read the *real* frozen evidence so that if an experiment is regenerated
and the dashboard's recomputation starts to disagree, these fail loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "results"
MODULE_PATH = REPO / "experiments" / "build_dashboard_data.py"


def _load_module():
    """Import the builder by path (it is a script, not a package module)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_dashboard_data", MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bld():
    return _load_module()


@pytest.fixture(scope="module")
def evidence(bld):
    return bld.build()


# --------------------------------------------------------------------------- #
# parsing + schema validation
# --------------------------------------------------------------------------- #
def test_robot_from_urdf_projects_real_panda(bld):
    robot = bld.robot_from_urdf(bld.URDF)
    joint_names = [j["name"] for j in robot["joints"]]
    assert "joint1" in joint_names and "joint7" in joint_names
    assert "left_finger" in joint_names and "right_finger" in joint_names
    for j in robot["joints"]:
        assert len(j["xyz"]) == 3 and len(j["axis"]) == 3
    named = {lnk["name"]: lnk for lnk in robot["links"]}
    assert "panda_link1" in named
    shapes = [v["shape"] for v in named["panda_link1"]["visuals"]]
    assert any(s["kind"] == "cylinder" for s in shapes)


def test_telemetry_parse_orders_frames_and_dedups_phases(bld):
    data = bld.parse_telemetry(bld.TELEMETRY)
    frames = data["frames"]
    assert frames, "telemetry produced no frames"
    steps = [f["step"] for f in frames]
    assert steps == sorted(steps), "frames must be in step order"
    for f in frames[:5]:
        assert set(f) >= {"step", "t", "phase", "grasped", "aperture", "tcp", "obj", "joints"}
        assert len(f["joints"]) == 7
        assert len(f["obj"]) == 3
    phases = data["phases"]
    assert len(phases) == len(set(phases))
    present = {f["phase"] for f in frames}
    assert present == set(phases)


def test_frozen_result_files_are_valid_json():
    for name in (
        "nominal_solvability.json",
        "rq1_search_strategies.json",
        "rq2_conditional_vs_aggregate.json",
        "rq2_settle_window_sweep.json",
        "rq4_reproducibility.json",
    ):
        json.loads((RESULTS / name).read_text())


# --------------------------------------------------------------------------- #
# metric aggregation + policy differences
# --------------------------------------------------------------------------- #
def test_wilson_interval_brackets_the_rate(bld):
    rate, lo, hi = bld.wilson(120, 120)
    assert lo <= rate <= hi
    assert hi <= 1.0
    assert lo > 0.9


def test_mcnemar_exact_is_symmetric_and_bounded(bld):
    assert bld.mcnemar_exact(0, 0) == 1.0
    p = bld.mcnemar_exact(4, 0)
    assert 0.0 <= p <= 1.0
    assert bld.mcnemar_exact(3, 1) == bld.mcnemar_exact(1, 3)


def test_policy_comparison_delta_matches_stored(evidence):
    pc = evidence["policy_comparison"]
    assert pc["available"]
    agg = pc["aggregate"]
    recomputed = agg["candidate_success"] - agg["baseline_success"]
    assert abs(recomputed - agg["delta"]) < 1e-9
    assert pc["baseline_steps_mean"] is not None
    assert pc["candidate_steps_mean"] is not None
    assert abs(pc["steps_delta"] - (pc["candidate_steps_mean"] - pc["baseline_steps_mean"])) < 1e-9


def test_nominal_metrics_recomputed_from_records(evidence):
    nominal = evidence["nominal"]
    assert nominal["available"]
    v1 = nominal["policies"]["policy_v1"]
    assert v1["successes"] == v1["seeds"] == 120
    assert v1["rate"] == pytest.approx(1.0)
    assert v1["ci"][0] <= v1["rate"] <= v1["ci"][1]


# --------------------------------------------------------------------------- #
# heatmap classification + untested regions
# --------------------------------------------------------------------------- #
def test_heatmap_cells_classified_from_measured_rate(evidence):
    bm = evidence["boundary_map"]
    assert bm["available"]
    assert bm["x"]["name"] == "depth_noise_mm"
    assert bm["y"]["name"] == "settle_window"
    for cell in bm["cells"].values():
        rate = cell["rate"]
        if cell["n"] == 0:
            assert cell["classification"] == "untested"
        elif rate >= 0.999:
            assert cell["classification"] == "success"
        elif rate <= 0.001:
            assert cell["classification"] == "failure"
        else:
            assert cell["classification"] == "mixed"
        assert cell["successes"] <= cell["n"]


def test_heatmap_every_cell_is_a_measured_grid_point(evidence):
    bm = evidence["boundary_map"]
    for key in bm["cells"]:
        w_s, n_s = key.split("|")
        assert float(w_s) in bm["y"]["values"]
        assert float(n_s) in bm["x"]["values"]
    assert len(bm["cells"]) == len(bm["y"]["values"]) * len(bm["x"]["values"]) == 20
    assert bm["n_seeds"] == 16


# --------------------------------------------------------------------------- #
# provenance + current-revision labels
# --------------------------------------------------------------------------- #
def test_settle_sweep_is_current_revision(evidence):
    st = evidence["settle_tradeoff"]
    assert st["available"]
    assert st["revision"] == evidence["revision"] == "d7026af"
    assert evidence["boundary_map"]["revision"] == "d7026af"


def test_opposing_effect_labels_hypothesis_retired(evidence):
    opp = evidence["opposing_effect"]
    assert opp["available"]
    assert "retired" in opp["hypothesis_status"]


def test_settle_finding_reports_no_advantage_with_caveat(evidence):
    st = evidence["settle_tradeoff"]
    assert "no robustness advantage" in st["finding"]
    assert "does not prove a zero effect" in st["caveat"]
    assert len(st["comparisons"]) == 16
    for c in st["comparisons"]:
        assert 0.0 <= c["p"] <= 1.0
        assert c["n_pairs"] == 16


# --------------------------------------------------------------------------- #
# missing-data handling
# --------------------------------------------------------------------------- #
def test_unavailable_metric_is_none_not_zero(evidence):
    assert evidence["metrics"]["boundary_estimation_error"] is None
    assert evidence["metrics"]["nominal_solvability"] is not None
    assert evidence["metrics"]["episodes_evaluated"] is not None


def test_every_evidence_file_validates(evidence):
    files = evidence["evidence_files"]
    assert files
    for f in files:
        assert f["valid"], f"{f['path']} failed validation: {f['detail']}"
        assert (REPO / f["path"]).exists()


# --------------------------------------------------------------------------- #
# consistency: dashboard values vs frozen artifacts
# --------------------------------------------------------------------------- #
def test_all_consistency_checks_pass(evidence):
    checks = evidence["consistency_checks"]
    assert checks, "no consistency checks were recorded"
    failed = [c for c in checks if not c["ok"]]
    assert not failed, f"dashboard disagrees with frozen JSON: {failed}"


def test_reproducibility_rate_matches_stored(evidence):
    r = evidence["reproducibility"]
    assert r["available"]
    stored = json.loads((RESULTS / "rq4_reproducibility.json").read_text())
    assert r["reproduced"] == stored["reproduced"]
    assert r["rate"] == pytest.approx(stored["reproduction_rate"])


# --------------------------------------------------------------------------- #
# replay timeline synchronisation
# --------------------------------------------------------------------------- #
def test_replay_frames_synchronized_with_timeline(evidence):
    ep = evidence["episode"]
    assert ep["available"]
    frames = ep["frames"]
    times = [f["t"] for f in frames]
    assert times == sorted(times), "sim time must be monotonic for replay"
    assert len(frames) > 1
    assert set(ep["phases"]) == {f["phase"] for f in frames}


# ---------------------------------------------------------------------------
# counterexample detail: nested perturbation value, seed, outcomes
# ---------------------------------------------------------------------------
def test_counterexample_depth_parsed_from_nested_block(evidence):
    """The swept value lives under perturbation:->depth_noise_mm:->value:.

    A flat scan hits the empty block key and loses it.
    """
    ce = evidence["counterexample"]
    assert ce["available"]
    assert ce["depth_noise_mm"] == pytest.approx(1.171875)
    # the id carries the same value rounded to 3 dp
    assert f"{ce['depth_noise_mm']:.3f}" in ce["id"]


def test_counterexample_seed_and_outcomes_are_surfaced(evidence):
    ce = evidence["counterexample"]
    assert ce["seed"] == 3
    assert ce["baseline"] == {"policy": "policy_v3", "result": "success"}
    assert ce["candidate"] == {"policy": "policy_v1", "result": "failure"}


def test_config_fingerprint_is_the_scenario_hash(evidence):
    """The fingerprint is the real sha256 from the artifact, not a placeholder."""
    fp = evidence["config"]["config_hash"]
    assert fp is not None and fp.startswith("sha256:") and len(fp) == 71
    assert fp == "sha256:e657dc8df59c050d10d5992fd697fb5032e833a498d19c131b3f22114c4651e8"


def test_boundary_map_names_the_swept_policy(evidence):
    """The sweep varied policy_v1's settle window, not policy_v3.

    The generator builds PolicyConfig(name="policy_v1", ...), so the label
    must come from evidence, not be guessed.
    """
    assert evidence["boundary_map"]["policy"] == "policy_v1"


# ---------------------------------------------------------------------------
# viewer wiring: every top-level evidence section is rendered somewhere
# ---------------------------------------------------------------------------
def test_viewer_renders_every_evidence_section(evidence):
    """Regression guard for 'aggregated but never displayed'.

    Each section the data layer produces must be consumed by viewer.js.
    """
    src = (REPO / "dashboard" / "viewer.js").read_text(encoding="utf-8")
    sections = [k for k in evidence if k not in {"revision"}]  # revision shown via git/label fallbacks
    missing = [k for k in sections if f"EVIDENCE.{k}" not in src]
    assert not missing, f"evidence sections never read by the viewer: {missing}"
