# Dashboard screenshot protocol (1920 × 1080)

These are the six polished screenshots for the research release. Each is a
static capture of the frozen research dashboard at revision `d7026af`.

> **Headless note.** This terminal has no display server, browser, or WebGL
> renderer (no Chrome, no Xvfb, no Playwright/Pyppeteer), so the screenshots
> cannot be captured here. Capture them on a machine with a display and a
> desktop browser, at a viewport of exactly **1920 × 1080**, then place the
> PNGs in this folder (`docs/screenshots/`).
>
> **Current status (this environment).** None of `01-overview.png` through
> `06-evidence-provenance.png` has been captured here, and none will be
> fabricated or committed as placeholders. They remain to be captured on a
> machine with a display, following this protocol, then added to
> `docs/screenshots/`. Do not commit generated or AI-upscaled images as if they
> were browser captures — the dashboard is the evidence, not an illustration.

## Prerequisites

```bash
# Regenerate the frozen evidence bundle (read-only sources → data/)
python experiments/build_dashboard_data.py

# Serve the dashboard (roots at repo root so evidence links resolve)
python dashboard/serve.py 8123   # → http://localhost:8123/dashboard/
```

Open `http://localhost:8123/dashboard/`, size the browser window to
**1920 × 1080**, wait for `evidence.json` to load and the Panda replay /
boundary map to draw, then capture each nav target below. The nav buttons are
`Overview`, `Discovery`, `Boundary Map`, `Policy Comparison`, `Results`, `Logs`;
clicking one scrolls that panel into center view.

## Shots to capture

| # | File | Nav target | What it shows |
|---|------|------------|---------------|
| 1 | `01-overview.png` | Overview | Full three-column console at rest: config, metrics, policy comparison, RQ3 attribution, reproducibility. |
| 2 | `02-panda-replay.png` | — (center) | Panda viewport + telemetry footer: telemetry-driven reconstruction of the recorded episode (labelled *not* a live camera), play/scrub/speed controls, `t = 0.00 s` clock. |
| 3 | `03-boundary-map.png` | Boundary Map | Right-column behavioral boundary heatmap (depth noise × settle window), one cell clicked so its cell-detail callout is visible; untested regions hatched, never interpolated. |
| 4 | `04-counterexample-discovery.png` | Discovery | Center counterexample search panel: found counterexample, minimality note, reproduction rate (36.4%), opposing-effect badge. |
| 5 | `05-subsystem-attribution.png` | — (left) | Subsystem Attribution (RQ3): correlation-only vs OFAT vs interaction-aware rows, implicated subsystem, `correct?` column. |
| 6 | `06-evidence-provenance.png` | Logs | Right-column Evidence & Artifacts + consistency check log: provenance pinned to source artifacts, fingerprint SHA-256, 30 consistency checks pass. |

## Filename convention

`NN-short-name.png` at 1920 × 1080, lossless PNG. Do not upscale, filter, or
retouch beyond a clean browser capture — the dashboard is the evidence, not an
illustration.

## Regenerating

If any evidence JSON changes, re-run `build_dashboard_data.py`, reload, and
re-capture the affected shots so the screenshots stay in sync with `d7026af`.
