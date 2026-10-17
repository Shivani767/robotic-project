#!/usr/bin/env python3
r"""Serve the ROBOT CI research dashboard for local inspection.

A tiny static file server rooted at the repository root, so that

* ``dashboard/index.html`` loads its ``data/*.json`` bundle, and
* the Evidence panel's ``open`` links (which point at ``../experiments/...``
  and ``../artifacts/...``) resolve to the real frozen artifacts.

This is a *recorded-experiment* server: it does not run a simulator and does
not expose any live control surface. To refresh the data bundle first run::

    python experiments/build_dashboard_data.py
    python dashboard/serve.py            # then open http://localhost:8000/dashboard/

It serves on port 8000 by default; pass a different port as the first argument.
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    """Static handler that logs each request to stderr (root = repository)."""

    def end_headers(self) -> None:
        """Disable caching so asset edits show up on plain reloads."""
        # Never let the browser cache dashboard assets: a stale viewer.js was
        # served after viewport fixes, making it look like "the same bug".
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """Emit one request line to stderr."""
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def main() -> None:
    """Start the threaded static server rooted at the repository."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT)))
    print(f"ROBOT CI dashboard → http://localhost:{port}/dashboard/")
    print("(serving repository root so evidence links resolve to real artifacts)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
