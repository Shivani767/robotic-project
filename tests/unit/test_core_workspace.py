"""Workspace discovery and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_ci.core.errors import WorkspaceError
from robot_ci.core.workspace import DirectoryLayout, Workspace, find_workspace_root, get_workspace


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A minimal repository root: just the marker file."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    return tmp_path


def test_discovers_root_from_marker(fake_root: Path) -> None:
    assert Workspace.discover(fake_root).root == fake_root.resolve()


def test_discovers_root_from_nested_directory(fake_root: Path) -> None:
    nested = fake_root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert Workspace.discover(nested).root == fake_root.resolve()


def test_discovers_root_from_a_file_path(fake_root: Path) -> None:
    marker = fake_root / "pyproject.toml"
    assert Workspace.discover(marker).root == fake_root.resolve()


def test_markerless_directory_raises(tmp_path: Path) -> None:
    """A directory with no marker anywhere up the tree must fail loudly."""
    with pytest.raises(WorkspaceError, match="could not locate"):
        Workspace.discover(tmp_path)


def test_custom_marker_is_honoured(tmp_path: Path) -> None:
    (tmp_path / ".robot-ci-root").write_text("", encoding="utf-8")
    assert find_workspace_root(tmp_path) == tmp_path.resolve()


def test_missing_root_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="not a directory"):
        Workspace(root=tmp_path / "nope")


def test_ensure_layout_creates_everything(fake_root: Path) -> None:
    workspace = Workspace.discover(fake_root)
    created = workspace.ensure_layout()
    assert created, "first call should create the directories"
    for name in workspace.layout.known_attributes():
        assert workspace.dir_for(name).is_dir()
    assert workspace.ensure_layout() == [], "second call should be a no-op"


def test_dir_for_rejects_unknown_names(fake_root: Path) -> None:
    workspace = Workspace.discover(fake_root)
    with pytest.raises(WorkspaceError, match="unknown layout directory"):
        workspace.dir_for("not_a_directory")


def test_convenience_properties(fake_root: Path) -> None:
    workspace = Workspace.discover(fake_root)
    assert workspace.scenarios_dir.name == "scenarios"
    assert workspace.results_dir.name == "results"


def test_relative_inside_and_outside_root(fake_root: Path, tmp_path: Path) -> None:
    workspace = Workspace.discover(fake_root)
    assert workspace.relative(workspace.path("docs")) == "docs"
    assert workspace.relative(tmp_path.parent).startswith("/")


def test_ensure_creates_intermediate_directories(fake_root: Path) -> None:
    workspace = Workspace.discover(fake_root)
    target = workspace.ensure("a/b/c")
    assert target.is_dir()


def test_chdir_restores_the_previous_directory(fake_root: Path) -> None:
    workspace = Workspace.discover(fake_root)
    before = Path.cwd()
    with workspace.chdir("scenarios") as target:
        assert Path.cwd() == target.resolve()
    assert Path.cwd() == before


def test_chdir_restores_on_exception(fake_root: Path) -> None:
    workspace = Workspace.discover(fake_root)
    before = Path.cwd()
    with pytest.raises(RuntimeError), workspace.chdir("docs"):
        raise RuntimeError("boom")
    assert Path.cwd() == before


def test_from_env_uses_the_environment_variable(fake_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_CI_ROOT", str(fake_root))
    assert Workspace.from_env().root == fake_root.resolve()


def test_from_env_falls_back_to_discovery(fake_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOT_CI_ROOT", raising=False)
    assert Workspace.from_env().root == Path.cwd().resolve()


def test_get_workspace_is_cached(fake_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROBOT_CI_ROOT", str(fake_root))
    first = get_workspace(refresh=True)
    assert get_workspace() is first


def test_layout_uses_relative_names_only() -> None:
    """No layout entry may be absolute, or the layout is not portable."""
    for value in DirectoryLayout().relative_paths():
        assert not Path(value).is_absolute()


def test_layout_paths_are_deduplicated() -> None:
    paths = DirectoryLayout().relative_paths()
    assert len(paths) == len(set(paths))
