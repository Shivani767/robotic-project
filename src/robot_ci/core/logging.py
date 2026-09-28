"""Structured logging.

ROBOT CI writes two kinds of output:

* *human* output through :mod:`rich` (CLI tables, progress, reports);
* *machine* output as JSON Lines (``logs/experiments/<id>.jsonl``).

Both are configured from one place so that a test run never pollutes stdout and
an experiment run always produces machine-readable records.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any, TextIO

__all__ = [
    "JsonlLogger",
    "LogRecord",
    "configure_logging",
    "get_logger",
    "utc_now_iso",
]

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Safe to call before configuration."""
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Install a single stderr handler on the ``robot_ci`` logger.

    Args:
        level: Logging level name or numeric value.
        stream: Destination stream. Defaults to ``sys.stderr``.
        force: Reconfigure even if already configured.
    """
    global _configured
    if _configured and not force:
        return

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if not isinstance(level, int):  # pragma: no cover - defensive
        raise ValueError(f"invalid logging level: {level!r}")

    logger = logging.getLogger("robot_ci")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _configured = True


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class LogRecord(dict[str, Any]):
    """A single JSONL record.

    A ``dict`` subclass so that records are directly JSON-serializable and
    cheaply filterable, while still supporting attribute access for readability.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


class JsonlLogger:
    """Append-only JSON Lines writer for research logs.

    Every event is flushed immediately so that a crashed or interrupted
    experiment still leaves a usable partial log.

    Example:
        >>> import io
        >>> buffer = io.StringIO()
        >>> log = JsonlLogger(buffer)
        >>> log.event("episode_start", experiment_id="exp_1")
        {'timestamp': '...', 'event': 'episode_start', 'experiment_id': 'exp_1'}
        >>> log.close()
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        context: Mapping[str, Any] | None = None,
        own_stream: bool = False,
    ) -> None:
        """Initialize the logger.

        Args:
            stream: Destination text stream (a file handle or ``io.StringIO``).
            context: Fields merged into every record (experiment id, policy, ...).
            own_stream: When True, :meth:`close` also closes ``stream``. The
                default is False because the logger usually does not own its
                stream: closing a caller's handle behind their back is a bug.
        """
        self._stream = stream
        self._context: dict[str, Any] = dict(context or {})
        self._own_stream = bool(own_stream)
        self._closed = False
        self._count = 0

    @property
    def count(self) -> int:
        """Number of records written so far."""
        return self._count

    def bind(self, **fields: Any) -> JsonlLogger:
        """Return a child logger whose context is ``self``'s plus ``fields``.

        The child shares the same stream and does *not* own it; closing the
        parent closes the underlying handle for both.
        """
        return JsonlLogger(self._stream, context={**self._context, **fields}, own_stream=False)

    def event(self, event_type: str, /, **fields: Any) -> LogRecord:
        """Write one record.

        Args:
            event_type: Dot-separated event name, e.g. ``"episode.finish"``.
            **fields: Additional structured fields.

        Returns:
            The record that was written.
        """
        if self._closed:  # pragma: no cover - defensive
            raise ValueError("cannot write to a closed JsonlLogger")
        record = LogRecord(
            {
                "timestamp": utc_now_iso(),
                "event": event_type,
                **self._context,
                **fields,
            }
        )
        self._stream.write(json.dumps(record, default=str, sort_keys=False) + "\n")
        self._stream.flush()
        self._count += 1
        return record

    def close(self) -> None:
        """Flush the stream, and close it only if this logger owns it.

        Leaving a caller-supplied stream open is the default: a logger is
        frequently constructed over an already-open file handle that the caller
        continues to use (for appending a final summary line, say).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.flush()
        finally:
            if self._own_stream and not self._stream.closed:
                self._stream.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
