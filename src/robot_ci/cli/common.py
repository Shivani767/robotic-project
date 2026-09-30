"""Shared CLI helpers: console, error handling and the phase gate."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from robot_ci.core.errors import (
    BackendUnavailableError,
    ConfigurationError,
    RobotCIError,
    ScenarioError,
    WorkspaceError,
)
from robot_ci.core.logging import configure_logging
from robot_ci.core.version import SCHEMA_VERSION, VERSION

__all__ = [
    "EXIT_BACKEND_UNAVAILABLE",
    "EXIT_CONFIG_ERROR",
    "EXIT_NOT_IMPLEMENTED",
    "EXIT_OK",
    "EXIT_REGRESSION",
    "EXIT_USAGE",
    "console",
    "error_console",
    "fail",
    "handle_errors",
    "not_implemented",
]

#: Process exit codes. Documented so CI can branch on them reliably.
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG_ERROR = 3
EXIT_BACKEND_UNAVAILABLE = 4
EXIT_NOT_IMPLEMENTED = 5
EXIT_REGRESSION = 1

console = Console()
error_console = Console(stderr=True)

T = TypeVar("T")


def handle_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Translate ROBOT CI exceptions into tidy CLI errors and exit codes.

    This keeps ``try/except`` out of every command body and guarantees that a
    failure is always reported with a stable exit code.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        configure_logging("WARNING")
        try:
            return func(*args, **kwargs)
        except BackendUnavailableError as exc:
            error_console.print(f"[bold red]backend unavailable[/] {exc.message}")
            raise typer.Exit(EXIT_BACKEND_UNAVAILABLE) from exc
        except (ConfigurationError, ScenarioError, WorkspaceError) as exc:
            error_console.print(f"[bold red]configuration error[/] {exc.message}")
            raise typer.Exit(EXIT_CONFIG_ERROR) from exc
        except RobotCIError as exc:
            error_console.print(f"[bold red]{type(exc).__name__}[/] {exc.message}")
            raise typer.Exit(EXIT_USAGE) from exc
        except KeyboardInterrupt:  # pragma: no cover - interactive
            error_console.print("[yellow]interrupted[/]")
            raise typer.Exit(130) from None

    return wrapper


def fail(message: str, *, code: int = EXIT_USAGE) -> NoReturn:
    """Print an error and exit with ``code``.

    Typed ``NoReturn`` because it always raises: that is what lets type
    checkers narrow a value guarded by ``if x is None: fail(...)``.
    """
    error_console.print(f"[bold red]error[/] {message}")
    raise typer.Exit(code)


def not_implemented(command: str, phase: int, detail: str = "") -> NoReturn:
    """Report a command that belongs to a later development phase.

    The project deliberately surfaces unimplemented phases instead of returning
    a fake success, so a user can never mistake a stub for a real result.

    Args:
        command: The command path, e.g. ``"robot-ci experiment run"``.
        phase: The development phase that will implement it.
        detail: Optional extra context, such as a workaround available today.

    Raises:
        typer.Exit: Always, with :data:`EXIT_NOT_IMPLEMENTED`.
    """
    message = f"'{command}' is scheduled for Phase {phase} and is not implemented yet."
    if detail:
        message += f" {detail}"
    error_console.print(f"[bold yellow]not implemented[/] {message}")
    raise typer.Exit(EXIT_NOT_IMPLEMENTED)


def version_table() -> Table:
    """Build the table printed by ``robot-ci version``."""
    table = Table(title="ROBOT CI", show_header=False, title_style="bold cyan")
    table.add_column("component")
    table.add_column("version")
    table.add_row("robot-ci", VERSION)
    table.add_row("schema", SCHEMA_VERSION)
    return table
