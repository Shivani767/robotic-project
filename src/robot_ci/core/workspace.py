"""Workspace layout resolution.

No module in this package may hard-code an absolute path. All file access goes
through :class:`Workspace`, which is discovered once (explicitly, or by walking
up from a marker file) and then passed down as an argument.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from robot_ci.core.errors import WorkspaceError

__all__ = ["DirectoryLayout", "Workspace", "find_workspace_root", "get_workspace"]

#: Files whose presence marks the repository root.
_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".robot-ci-root")

_ENV_VAR = "ROBOT_CI_ROOT"


def find_workspace_root(start: str | Path | None = None) -> Path:
    """Walk upwards from ``start`` looking for a repository root marker.

    Args:
        start: Directory to start from. Defaults to the current working directory.

    Returns:
        Absolute path of the repository root.

    Raises:
        WorkspaceError: If no marker is found within the filesystem root.
    """
    origin = Path(start) if start is not None else Path.cwd()
    origin = origin.resolve()
    if origin.is_file():
        origin = origin.parent

    for candidate in (origin, *origin.parents):
        for marker in _ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate

    raise WorkspaceError(
        "could not locate the ROBOT CI repository root",
        code="workspace.root_not_found",
        start=str(origin),
        markers=list(_ROOT_MARKERS),
    )


@dataclass(frozen=True)
class DirectoryLayout:
    """Relative names of the directories ROBOT CI reads from and writes to.

    They are relative on purpose so the layout is portable and testable against
    a temporary root.
    """

    scenarios: str = "scenarios"
    experiments: str = "experiments"
    results: str = "experiments/results"
    logs: str = "experiments/results/logs"
    counterexamples: str = "experiments/results/counterexamples"
    reports: str = "experiments/results/reports"
    data: str = "experiments/results/data"
    docs: str = "docs"
    notebooks: str = "notebooks"
    dashboard: str = "dashboard"
    ros2_ws: str = "ros2_ws"

    def known_attributes(self) -> tuple[str, ...]:
        """Return every layout *attribute* name (e.g. ``"results"``).

        These are the keys accepted by :meth:`Workspace.dir_for`, which is why
        they are kept separate from the relative path values.
        """
        return tuple(vars(self))

    def relative_paths(self) -> tuple[str, ...]:
        """Return every relative directory path, deduplicated."""
        return tuple(dict.fromkeys(vars(self).values()))


@dataclass
class Workspace:
    """An absolute, validated view of the project directory tree.

    Attributes:
        root: Absolute path of the repository root.
        layout: Relative directory names.
    """

    root: Path
    layout: DirectoryLayout = field(default_factory=DirectoryLayout)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(
                f"workspace root is not a directory: {self.root}",
                code="workspace.root_missing",
            )

    @classmethod
    def discover(cls, start: str | Path | None = None) -> Workspace:
        """Build a workspace by walking up from ``start`` looking for markers."""
        return cls(root=find_workspace_root(start))

    @classmethod
    def from_env(cls) -> Workspace:
        """Build a workspace from ``ROBOT_CI_ROOT``, falling back to discovery."""
        env_value = os.environ.get(_ENV_VAR)
        if env_value:
            return cls(root=Path(env_value))
        return cls.discover()

    def path(self, *parts: str) -> Path:
        """Join ``parts`` onto the root."""
        return self.root.joinpath(*parts)

    def dir_for(self, name: str) -> Path:
        """Return the absolute path of a layout directory by attribute name.

        Args:
            name: A :class:`DirectoryLayout` attribute name, e.g. ``"results"``.

        Raises:
            WorkspaceError: If ``name`` is not a layout attribute.
        """
        if name not in self.layout.known_attributes():
            raise WorkspaceError(
                f"unknown layout directory: {name!r}",
                code="workspace.unknown_directory",
                known=list(self.layout.known_attributes()),
            )
        return self.path(getattr(self.layout, name))

    @property
    def scenarios_dir(self) -> Path:
        """Directory holding scenario YAML files."""
        return self.dir_for("scenarios")

    @property
    def results_dir(self) -> Path:
        """Directory holding all generated experiment artifacts."""
        return self.dir_for("results")

    def ensure_layout(self) -> list[Path]:
        """Create every layout directory that does not yet exist.

        Returns:
            The directories actually created (empty if all already existed).
        """
        created: list[Path] = []
        for name in self.layout.known_attributes():
            target = self.dir_for(name)
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                created.append(target)
        return created

    def ensure(self, path: str | Path) -> Path:
        """Create ``path`` (and parents) if needed and return it resolved."""
        target = Path(path)
        if not target.is_absolute():
            target = self.root / target
        target.mkdir(parents=True, exist_ok=True)
        return target

    def relative(self, path: str | Path) -> str:
        """Return ``path`` relative to the root, or absolute if it escapes it."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            return candidate.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return candidate.resolve().as_posix()

    @contextmanager
    def chdir(self, path: str | Path) -> Iterator[Path]:
        """Temporarily change the process working directory."""
        target = self.ensure(path)
        previous = Path.cwd()
        os.chdir(target)
        try:
            yield target
        finally:
            os.chdir(previous)

    def __fspath__(self) -> str:  # pragma: no cover - trivial
        return str(self.root)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.root)


_cached: Workspace | None = None


def get_workspace(*, refresh: bool = False) -> Workspace:
    """Return the process-wide workspace, discovering it on first use.

    Args:
        refresh: Force re-discovery. Used by tests that create a temp root.

    Raises:
        WorkspaceError: If no repository root can be located.
    """
    global _cached
    if _cached is None or refresh:
        _cached = Workspace.from_env()
    return _cached
