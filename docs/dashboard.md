# The research dashboard

**Verified locally** — built and screenshotted on this machine (macOS, headless
Chrome for the interaction checks); every number it shows is recomputed from the
frozen evidence under `experiments/results/`, and `tests/unit/test_dashboard_data.py`
guards that agreement.

The dashboard is a *recorded-experiment* console: a static three-column page
(Panda robot viewport, behavioral boundary map, dense lower charts) that
replays the frozen research record at revision `d7026af`. It is deliberately
not a live backend — the robot pane is labelled **TELEMETRY-DRIVEN
RECONSTRUCTION** because it animates recorded joint states, never a running
simulator.

## Run it

```bash
python experiments/build_dashboard_data.py   # regenerate dashboard/data/*.json
python dashboard/serve.py 8123              # → http://localhost:8123/dashboard/
```

`serve.py` roots at the repository so the Evidence panel's `open` links resolve
to the real artifacts; the default port is 8000, pass another as the first
argument. No build step, no framework: `index.html` + `style.css` + `viewer.js`
plus a vendored copy of three.js under `dashboard/vendor/`.

## Architecture

```
experiments/results/*.json   frozen evidence (read-only)
artifacts/.../telemetry.csv  recorded joint states for the replay
gazebo/.../panda.urdf        the one robot model, projected for the browser
        │
        ▼
experiments/build_dashboard_data.py   data layer: recompute + cross-check
        │                              (Wilson CIs, McNemar, 30 consistency checks)
        ▼
dashboard/data/evidence.json  +  robot.json
        │
        ▼
dashboard/viewer.js           renders every section; missing data → "Not measured"
```

Design rules the dashboard keeps:

* **Nothing is invented.** A metric not stored in the evidence renders as *Not
  measured* (e.g. boundary-estimation error). Untested heatmap regions are
  hatched, never interpolated.
* **Aggregates are recomputed**, not transcribed. `build_dashboard_data.py`
  derives solvability rates, policy deltas, and the paired McNemar tests from
  per-seed records, and records 30 consistency checks against the stored
  values; the viewer shows the pass count and an expandable per-check log.
* **Provenance is pinned.** Every panel cites the source artifact (e.g.
  `rq2_settle_window_sweep.json`), and the config fingerprint is the real
  `sha256` scenario hash from the counterexample artifact.
* **Negative results are shown as findings**, with their caveats attached
  (e.g. "no robustness advantage demonstrated" plus "non-significance is not
  proof of a zero effect", and the retired RQ2 hypothesis badge).

## Panels

| Column | Panel | Source |
|---|---|---|
| left | Experiment Configuration | `config` (search + repro + CE artifact) |
| left | Research Metrics | `metrics` (recomputed tiles) |
| left | Policy Comparison + nominal baselines | `policy_comparison`, `nominal` |
| left | Subsystem Attribution (RQ3) | `rq3` |
| left | Reproducibility + interpretation | `reproducibility` |
| center | Panda viewport + telemetry footer | `episode` + `robot.json` |
| center | Counterexample Discovery | `search`, `counterexample`, `opposing_effect` |
| center | Settle-Window Trade-off | `settle_tradeoff` |
| center | Perturbation Boundaries (RQ5) | `rq5` |
| right | Behavioral Boundary Map (click a cell) | `boundary_map` |
| right | Evidence & Artifacts + consistency log | `evidence_files`, `consistency_checks` |

## Verification

* `tests/unit/test_dashboard_data.py` — 22 tests: parsing, statistics,
  provenance (`d7026af`), counterexample detail, and a wiring regression test
  that fails if any evidence section stops being rendered.
* Headless-Chrome interaction checks (heatmap cell click, stage switching,
  playback controls, nav targets) were run during development; WebGL needs
  `--enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader`.

## Limitations

* The robot pane requires WebGL; where it is unavailable the 2D charts still
  render (the viewport is best-effort by design).
* The page is a static replay of one frozen revision — re-running experiments
  requires the CLI (`robot-ci`), not this dashboard.
* The recorded telemetry is one nominal episode (seed 3, 0 mm); the replay
  shows that episode, not a per-cell sample from the heatmap.