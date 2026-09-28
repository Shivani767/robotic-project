"""Structured logging and JSONL research logs."""

from __future__ import annotations

import io
import json
import logging

import pytest

from robot_ci.core.logging import (
    JsonlLogger,
    LogRecord,
    configure_logging,
    get_logger,
    utc_now_iso,
)


def test_utc_now_iso_has_a_z_suffix() -> None:
    assert utc_now_iso().endswith("Z")


def test_get_logger_is_namespaced() -> None:
    assert get_logger("robot_ci.test").name == "robot_ci.test"


def test_configure_logging_is_idempotent() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream, force=True)
    configure_logging("INFO", stream=stream)
    assert len(logging.getLogger("robot_ci").handlers) == 1


def test_configure_logging_writes_to_the_stream() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream, force=True)
    get_logger("robot_ci.demo").info("hello %s", "world")
    assert "hello world" in stream.getvalue()


def test_configure_logging_rejects_a_bad_level() -> None:
    with pytest.raises(ValueError, match="invalid logging level"):
        configure_logging(123.5, force=True)  # type: ignore[arg-type]


def test_jsonl_records_are_valid_json() -> None:
    buffer = io.StringIO()
    log = JsonlLogger(buffer)
    log.event("episode.start", episode_id="e1", seed=7)
    record = json.loads(buffer.getvalue().strip())
    assert record["event"] == "episode.start"
    assert record["episode_id"] == "e1"
    assert record["seed"] == 7
    assert record["timestamp"].endswith("Z")


def test_jsonl_context_is_merged_into_every_record() -> None:
    buffer = io.StringIO()
    log = JsonlLogger(buffer, context={"experiment_id": "exp_1"})
    log.event("a")
    log.event("b")
    for line in buffer.getvalue().strip().splitlines():
        assert json.loads(line)["experiment_id"] == "exp_1"


def test_bind_creates_a_child_without_mutating_the_parent() -> None:
    buffer = io.StringIO()
    parent = JsonlLogger(buffer, context={"policy": "v1"})
    child = parent.bind(policy="v2", step=3)
    child.event("child")
    parent.event("parent")
    lines = [json.loads(line) for line in buffer.getvalue().strip().splitlines()]
    assert lines[0]["policy"] == "v2" and lines[0]["step"] == 3
    assert lines[1]["policy"] == "v1" and "step" not in lines[1]


def test_event_returns_the_written_record() -> None:
    log = JsonlLogger(io.StringIO())
    record = log.event("x", value=1)
    assert isinstance(record, LogRecord)
    assert record["value"] == 1
    assert record["event"] == "x"


def test_count_tracks_records() -> None:
    log = JsonlLogger(io.StringIO())
    log.event("a")
    log.event("b")
    assert log.count == 2


def test_writing_after_close_raises() -> None:
    log = JsonlLogger(io.StringIO())
    log.close()
    with pytest.raises(ValueError, match="closed"):
        log.event("a")


def test_non_serializable_values_do_not_crash_the_log() -> None:
    """Research logs must survive an unexpected object in a field."""
    import numpy as np

    buffer = io.StringIO()
    log = JsonlLogger(buffer)
    log.event("x", array=np.arange(3))
    assert "array" in json.loads(buffer.getvalue())


def test_records_flush_immediately() -> None:
    """A crashed experiment must still leave a usable partial log."""
    buffer = io.StringIO()
    log = JsonlLogger(buffer)
    log.event("a")
    assert buffer.getvalue(), "record was not flushed"


def test_close_does_not_close_a_borrowed_stream() -> None:
    """Regression guard: the logger must not close a caller's handle."""
    buffer = io.StringIO()
    log = JsonlLogger(buffer)
    log.close()
    assert not buffer.closed
    buffer.write("still usable")


def test_close_closes_an_owned_stream() -> None:
    buffer = io.StringIO()
    log = JsonlLogger(buffer, own_stream=True)
    log.close()
    assert buffer.closed
