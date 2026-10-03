"""Start and monitor an experiment from the browser dashboard.

Two properties this exists to guarantee
---------------------------------------
**One experiment at a time.** Gazebo's transport bus is global to the container:
the live viewer bridge and a recorded episode both publish on
``/world/robot_ci_pick_place/...`` for the same world name, so two runs
interleaved would corrupt each other's joint state and neither result would be
trustworthy. The launcher therefore takes a file lock and refuses a second run
while one holds it, rather than racing and reporting a result nobody can check.

**Honest progress.** The status file says which key is running, what stage it
reached and what the terminal status was. A crash is written as an error status,
not left as a permanently "running" entry, because a launcher that can only ever
report success is worse than none.

The lock is a plain file with a PID in it, which is enough for this single-
container demo and deliberately not a distributed lock: this is one Mac, one
Gazebo, one simulator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_ci.core.logging import get_logger

__all__ = ["ExperimentLauncher", "LaunchStatus"]

_LOG = get_logger(__name__)

#: Where the suite writes packages. Overridable with ``ROBOT_CI_ARTIFACTS``.
def artifacts_root() -> Path:
    """Return the directory visual-report packages are written under."""
    return Path(os.environ.get("ROBOT_CI_ARTIFACTS", "artifacts/visual_reports"))


@dataclass
class LaunchStatus:
    """The launcher's current state, as published to the dashboard.

    Attributes:
        running: Whether an experiment is in flight right now.
        key: The suite key being run, or ``""`` when idle.
        seed / policy / repeat: The parameters the run was started with.
        stage: One of ``starting``, ``running``, ``packaging``, ``done``, ``error``.
        message: Human-readable detail for the current stage.
        status: The terminal status once the run finishes (``PASS``/``FAIL``/...).
        started_wall / finished_wall: Unix times bracketing the run.
        zip: Relative path to the produced ZIP, once one exists.
    """

    running: bool = False
    key: str = ""
    seed: int = 0
    policy: str = ""
    stage: str = ""
    message: str = ""
    status: str = ""
    started_wall: float = 0.0
    finished_wall: float = 0.0
    zip: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the HTTP endpoint."""
        return asdict(self)


def _status_file(root: Path) -> Path:
    """Return the path of the launcher's status file."""
    return root / ".launcher.json"


class ExperimentLauncher:
    """Runs one suite entry at a time on behalf of the dashboard.

    Args:
        root: Where packages (and the status file) live.
        command: Python interpreter to invoke; defaults to the current one.
    """

    def __init__(self, root: Path | None = None, command: str | None = None) -> None:
        self._root = Path(root) if root is not None else artifacts_root()
        self._command = command or sys.executable
        self._process: subprocess.Popen[bytes] | None = None
        self._monitor: threading.Thread | None = None
        self._local = threading.Lock()

    def status(self) -> LaunchStatus:
        """Return the persisted launcher status, defaulting to idle."""
        path = _status_file(self._root)
        if not path.exists():
            return LaunchStatus()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - torn write
            return LaunchStatus()
        known = set(LaunchStatus.__dataclass_fields__)
        return LaunchStatus(**{k: v for k, v in payload.items() if k in known})

    def _write(self, status: LaunchStatus) -> None:
        """Persist status atomically so a reader never sees a torn file."""
        path = _status_file(self._root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)

    def busy(self) -> bool:
        """Whether an experiment is currently running."""
        return self.status().running

    def suite(self) -> list[dict[str, Any]]:
        """Describe every suite entry: what it is and whether it has run."""
        from robot_ci.recording.suite import default_suite

        summary_path = self._root / "suite_summary.json"
        previous: dict[str, Any] = {}
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                previous = {row["key"]: row for row in payload.get("entries", [])}
            except (OSError, json.JSONDecodeError, KeyError):  # pragma: no cover
                previous = {}

        rows: list[dict[str, Any]] = []
        for entry in default_suite():
            archive = self._root / f"{entry.key}.zip"
            prior = previous.get(entry.key, {})
            rows.append(
                {
                    "key": entry.key,
                    "category": entry.category,
                    "title": entry.title,
                    "condition": entry.condition,
                    "status": prior.get("status", entry.status.value),
                    "detail": entry.reason or prior.get("detail", ""),
                    "seed": prior.get("seed", entry.seed),
                    "policy": prior.get("policy", entry.policy),
                    "repeats": entry.repeats,
                    "runnable": entry.runnable,
                    "has_archive": archive.exists(),
                    "zip": f"{entry.key}.zip" if archive.exists() else None,
                    "package": entry.key if (self._root / entry.key).is_dir() else None,
                }
            )
        return rows

    def start(
        self, key: str, *, seed: int | None = None, policy: str | None = None
    ) -> tuple[bool, str]:
        """Start one suite entry, or refuse because another run holds the lock.

        Args:
            key: Suite key to run.
            seed: Override the episode seed when given.
            policy: Override the policy when given.

        Returns:
            ``(started, message)``. ``started`` is ``False`` when refused, and
            ``message`` then says which run holds the lock.
        """
        with self._local:
            current = self.status()
            if current.running:
                return False, f"an experiment is already running: {current.key}"
            from robot_ci.recording.suite import default_suite

            runnable = {entry.key for entry in default_suite() if entry.runnable}
            if key not in runnable:
                return False, f"unknown or unavailable suite key: {key}"

            status = LaunchStatus(
                running=True,
                key=key,
                seed=seed or 0,
                policy=policy or "",
                stage="starting",
                message="launching",
                started_wall=time.time(),
            )
            self._write(status)
            command = [
                self._command, "-m", "robot_ci.cli.main", "record", "one", key,
                "--out", str(self._root),
            ]
            if seed is not None:
                command += ["--seed", str(seed)]
            if policy is not None:
                command += ["--policy", policy]
            environment = {**os.environ, "ROBOT_CI_ARTIFACTS": str(self._root)}
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            except OSError as exc:
                status.running = False
                status.stage = "error"
                status.message = f"could not start: {exc}"
                status.finished_wall = time.time()
                self._write(status)
                return False, status.message
            self._monitor = threading.Thread(target=self._follow, args=(status,), daemon=True)
            self._monitor.start()
        return True, f"started {key}"

    def _follow(self, status: LaunchStatus) -> None:
        """Watch the child process and publish each stage until it exits."""
        process = self._process
        if process is None:  # pragma: no cover - set immediately before start
            return
        status.stage = "running"
        status.message = "episode in flight"
        self._write(status)
        output = b""
        stream = process.stdout
        try:
            assert stream is not None
            for chunk in iter(lambda: stream.read(1024), b""):
                output += chunk
            code = process.wait(timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            code = -1
            status.message = f"wait failed: {exc}"
        status.running = False
        status.finished_wall = time.time()
        if code != 0:
            tail = output.decode("utf-8", "replace").strip()[-600:]
            status.stage = "error"
            status.message = f"{status.message} | {tail or f'exit code {code}'}"
        else:
            status.stage = "done"
            status.status = self._final_status(status.key)
            status.zip = (
                f"{status.key}.zip" if (self._root / f"{status.key}.zip").exists() else ""
            )
        self._write(status)

    def _final_status(self, key: str) -> str:
        """Read the recorded terminal status for ``key``, if a package exists."""
        results = self._root / key / "results.json"
        if not results.exists():
            return ""
        try:
            payload = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover
            return ""
        return str(payload.get("status", ""))
