r"""Build a single page that plays the rendered episodes.

``render_episode.py`` writes one GIF per episode. This collects every episode
sidecar under ``experiments/results/`` into one HTML page, so the clips can be
watched without hunting for filenames.

The page shows, for each episode, the clip and the condition that produced it.
That pairing is the point: a clip of an arm failing means nothing without the
noise level, the seed and the step budget that made it fail, and the numbers on
the page are the same ones the sidecar recorded, not retyped.

Usage:
    python experiments/make_episode_viewer.py [--out experiments/results/episodes.html]
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/results")

#: Fields shown next to each clip, in order. Everything here comes from the
#: sidecar the render wrote, so the page cannot drift from the artifact.
FIELDS: tuple[tuple[str, str], ...] = (
    ("policy", "policy"),
    ("seed", "seed"),
    ("scenario_id", "scenario id"),
    ("depth_noise_mm", "depth noise (mm)"),
    ("max_steps", "step budget"),
    ("status", "terminal status"),
    ("steps", "steps taken"),
    ("config_hash", "config hash"),
    ("native_resolution", "rendered at"),
    ("output_resolution", "displayed at"),
)


def _cell(value: Any) -> str:
    if isinstance(value, list):
        return "x".join(str(v) for v in value)
    return html.escape(str(value))


def _card(sidecar: dict[str, Any], directory: Path) -> str:
    gif = Path(str(sidecar.get("gif", ""))).name
    strip = Path(str(sidecar.get("filmstrip", ""))).name
    success = bool(sidecar.get("success"))
    classes = "ok" if success else "bad"
    verdict = "solved" if success else "failed"
    rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{_cell(sidecar.get(key, 'unknown'))}</td></tr>"
        for key, label in FIELDS
    )
    return f"""<section class="card {classes}">
  <h2>{html.escape(gif)} <span class="pill">{verdict}</span></h2>
  <figure>
    <img src="{html.escape(strip)}" alt="key frames of {html.escape(gif)}">
    <figcaption>Key frames: first, evenly spaced, last.</figcaption>
  </figure>
  <figure>
    <img src="{html.escape(gif)}" alt="{html.escape(str(sidecar.get("caption", gif)))}">
    <figcaption>{html.escape(str(sidecar.get("caption", "")))}. The clip itself is
    not committed; regenerate it with <code>experiments/render_episode.py</code>.</figcaption>
  </figure>
  <table>{rows}</table>
</section>"""


def build(results: Path, out: Path) -> tuple[Path, int]:
    """Write the viewer page. Returns the path and how many clips it shows."""
    sidecars = sorted(results.glob("episode_*.json"))
    cards: list[str] = []
    for path in sidecars:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        cards.append(_card(payload, results))

    body = "\n".join(cards) if cards else "<p>No rendered episodes yet.</p>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Rendered episodes &mdash; ROBOT CI</title>
<style>
body {{ font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       margin: 2rem auto; max-width: 60rem; padding: 0 1.5rem; color: #16181d; }}
h1 {{ font-size: 1.35rem; margin: 0 0 1.1rem; }}
.card {{ border: 1px solid #e3e6ea; border-radius: 8px; padding: 1rem 1.2rem;
        margin-bottom: 1.6rem; }}
.card.ok {{ border-left: 4px solid #2f9e5b; }}
.card.bad {{ border-left: 4px solid #c0392b; }}
h2 {{ font-size: .95rem; margin: 0 0 .7rem; font-family: ui-monospace, SFMono-Regular, monospace; }}
img {{ width: 100%; height: auto; border-radius: 4px; image-rendering: auto; }}
figure {{ margin: 0 0 .6rem; }}
figcaption {{ color: #5b6472; font-size: .78rem; margin-top: .25rem; }}
.pill {{ font-size: .72rem; padding: .1rem .5rem; border-radius: 9px;
        background: #eef1f5; color: #445; }}
table {{ border-collapse: collapse; width: 100%; margin-top: .7rem; font-size: .82rem; }}
th, td {{ border: 1px solid #e3e6ea; padding: .3rem .55rem; text-align: left; }}
th {{ background: #f6f7f9; font-weight: 600; width: 12rem; }}
.note {{ color: #5b6472; font-size: .85rem; margin-bottom: 1.4rem; }}
</style></head><body>
<h1>Rendered episodes</h1>
<p class="note">Each clip is one episode driven through the project's own runner and
rendered from the simulator's fixed workspace camera. The frames are enlarged for
viewing; the episode itself always ran at the scenario's native resolution.</p>
{body}
</body></html>
"""
    out.write_text(page)
    return out, len(cards)


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the rendered-episode viewer page.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.results / "episodes.html"
    written, count = build(args.results, out)
    print(f"viewer written : {written}")
    print(f"episodes shown : {count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
