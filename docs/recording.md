# Recording experiments: video, images, ZIPs

This document covers how a live simulation run becomes a downloadable artifact
package, and how to get one from a MacBook with Docker.

## 1. One command

```bash
docker compose up
# open http://localhost:8000
```

That starts Gazebo Harmonic, the joint driver and the live browser viewer. To
generate the artifacts instead, run the suite headlessly:

```bash
docker compose run --rm gazebo-demo \
  python3 scripts/record_suite.py --out artifacts/visual_reports
```

Single condition:

```bash
docker compose run --rm gazebo-demo \
  python3 -m robot_ci.cli.main record one nominal_baseline
```

`record suite`, `record one` and `scripts/record_suite.py` are three doors into
the same `robot_ci.recording` package, so batch runs and single runs cannot drift
apart in behaviour.

## 2. What gets produced

```text
artifacts/visual_reports/
├── <condition>/
│   ├── experiment.mp4     1920x1080 @ 30 fps, H.264
│   ├── overview.png        task, policy, condition, outcome
│   ├── robot_view.png      3-D Panda from recorded joint states
│   ├── camera_view.png     one real frame from the Gazebo camera
│   ├── metrics.png         measured metrics + joint trajectory
│   ├── timeline.png        timestamped episode events
│   ├── manifest.json       config, seed, versions, recording metadata, hashes
│   ├── results.json        raw measured status, metrics, errors
│   └── README.md           what ran, how it was captured, limitations
├── <condition>.zip         the package above, verified to reopen
├── suite_summary.json      one row per suite entry, with its status
├── SUITE.md                the same table in Markdown
└── complete_experiment_suite.zip
```

These are generated media. They are `.gitignore`d on purpose: the manifest and
summary record the seed, commit and config hash needed to rebuild them.

## 3. The two views

The video is split into two panes, both drawn during the same episode:

| Pane | Source | Label |
|---|---|---|
| Left | `robot_ci.recording.render3d` — the Panda through the same URDF the browser viewer uses, from the joint angles Gazebo reported | `3D VIEW (recorded joint states)` |
| Right | the simulator's own RGB output, captured before any corruption stage | `CAMERA (simulator output)` |

They are separate streams of the same run and are labelled permanently so no one
has to guess which is footage. The 3-D view is a projection, not a render, and
says so; if the joints had not moved, the arm in it would not move either.

A restrained slate opens the video with the experiment's identity and terminal
status, and each frame carries the status, step count, simulation clock and
current phase.

## 4. Synchronisation

Every frame and joint sample carries **both** clocks: the simulation clock and
the host wall clock. The simulation clock is authoritative — Gazebo runs faster
or slower than real time — so the two are never interchanged.

`manifest.json` reports the offset rather than hiding it:

```json
"synchronisation": {
  "clocks": ["simulation clock", "host wall clock"],
  "clock_offset_s": -2.14,
  "note": "..."
}
```

Known limitation: video frames are written at a constant 30 fps regardless of
simulated speed, so a viewer gets a real-time clock while the `t` overlay reports
the simulated time the frame belongs to. The two are therefore offset by exactly
the factor the manifest records.

## 5. Honesty rules the pipeline enforces

* **No invented frames.** If the simulator produced no image, the camera pane
  says `NO CAMERA FRAME` and `camera_view.png` states that zero frames arrived.
  Nothing is drawn to fill the gap.
* **No promoted statuses.** `PASS` is only the task's own verdict. `FAIL`,
  `ERROR`, `NOT_RUN`, `NOT_SUPPORTED` and `NOT_IMPLEMENTED` stay distinct
  everywhere, including the ZIP index.
* **No ZIP without a run.** An entry that did not execute contributes a summary
  row, never a package directory, so the archive and the index cannot disagree.
* **Verified archives.** Each ZIP is reopened and integrity-checked after being
  written; a file missing from the archive raises.
* **Streaming encode.** Frames are piped to ffmpeg as they are composed, so a
  300-tick 1080p episode never materialises ~1.9 GB in memory.

## 6. Concurrency

Gazebo's transport bus is global to the container, and the live viewer bridge and
a recorded episode both use the same world name. Two runs at once would corrupt
each other's joint state, so `robot_ci.recording.launcher` takes a lock and
refuses a second start while one is in flight, and the dashboard's Start button
reports which run holds it.

## 7. Encoder

Encoding needs `ffmpeg`. It is looked up on `PATH` first, then inside the
`imageio-ffmpeg` wheel, which the demo image installs — so no system package and
no GPU are required. When neither exists the stills, JSON and ZIP are still
written and the package records `No video in this package` instead of implying
one exists.

## 8. Adding a condition

Conditions live in `robot_ci/recording/suite.py`. An entry declares a `build()`
that returns the `ScenarioConfig` for that condition; if the backend genuinely
cannot execute it, give it a `status` and a `reason` and leave `build` unset —
it will then be reported rather than attempted. The tests
(`tests/unit/test_recording.py`) check that perturbed conditions really move the
thing they claim to perturb, so a silently-ignored setting fails the build.