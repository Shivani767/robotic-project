# ROBOT CI demo walkthrough

A concise, staged walkthrough of the recorded research record at revision
`d7026af`. Every stage runs against frozen artifacts — no simulator, no live
robot, nothing fabricated. The full CLI stage-by-stage demo (with real captured
output) lives at `experiments/results/DEMO.md`; this document is the short
narrative version keyed to the dashboard screenshots in `docs/screenshots/`.

## Stage 0 — Open the frozen console

```bash
python experiments/build_dashboard_data.py   # recompute evidence.json + robot.json
python dashboard/serve.py 8123               # http://localhost:8123/dashboard/
```

The page is a read-only replay of one frozen revision. The mode badge reads
**RECORDED EXPERIMENT**; every number is recomputed from
`experiments/results/` and a missing metric renders as *Not measured*.

## 1. Panda replay (telemetry-driven reconstruction)

Open the center panel. The Panda arm animates from recorded joint-state
telemetry (one nominal episode, seed 3, 0 mm), not from a live camera or
simulator stream — that is why it is labelled *TELEMETRY-DRIVEN
RECONSTRUCTION*. Use the playback bar to play/pause, restart, scrub, and set
1× / 2× / 4× speed; the clock shows `t = 0.00 s` … `t = N.NN s`. The
⌖ button resets the orbit camera to `VIEW_HOME`. This is the visual evidence
that the arm approaches, grasps, lifts, transports and places — measured from
the recorded run, asserted nowhere.

## 2. Interactive behavioral boundary map

Click **Boundary Map** (right column). The heatmap plots success rate across
depth noise (x) × settle window (y): green = success, red = failure, orange =
mixed, hatched = untested. Click any cell to open its detail callout with the
exact rate and source. The boundary is read off where the candidate first
succeeds while the baseline fails — no curve fitting, no interpolation into
untested regions. This is `docs/screenshots/03-boundary-map.png`.

## 3. Counterexample discovery

Click **Discovery** (center). The searcher minimizes a counterexample over the
perturbation axis and reports the found setting, its minimality, and an honest
reproduction rate (36.4% — the difference between a behavioral boundary and a
lucky episode). Tests-to-failure alone would rank a 10×-worse counterexample as
best, so minimality is part of the finding, not an extra. This is
`docs/screenshots/04-counterexample-discovery.png`.

## 4. Subsystem attribution (RQ3)

Open the left **Subsystem Attribution** panel. Three methods — correlation
only, OFAT, interaction-aware — each name an implicated subsystem with a
`correct?` verdict. Here OFAT names the gripper because perception owns a second
opposing knob (`max_position_sigma_m`, −0.083) that dilutes the subsystem mean
to +0.0417, while the real lever `settle_steps` is live and favourable
(+0.167). The takeaway: subsystem-granular attribution is unsound when a
subsystem's knobs disagree in sign. This is `docs/screenshots/05-subsystem-attribution.png`.

## 5. Evidence provenance

Click **Logs** (right column) → **Evidence & Artifacts**. Every panel cites its
source artifact (e.g. `rq2_settle_window_sweep.json`), and the config
fingerprint is the real SHA-256 scenario hash from the counterexample
artifact. The consistency log shows the 30 checks that `build_dashboard_data.py`
runs against the stored values; all pass. This is
`docs/screenshots/06-evidence-provenance.png`.

## What the walkthrough deliberately does not claim

* The dashboard does not run a simulator — it replays one frozen revision.
* The Panda pane is a reconstruction of recorded telemetry, never a live camera.
* Rates are small-n (mostly 6–8 seeds per cell); directions reproduce, counts
  do not — always quote with `n`.
* Nothing here is projected, simulated on paper, or copied from a previous
  session; where an experiment was not run, the dashboard says *Not measured*.
