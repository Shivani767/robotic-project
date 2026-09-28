"""Deterministic seed derivation."""

from __future__ import annotations

import pytest

from robot_ci.core.seeding import (
    derive_seed,
    describe_seed,
    make_numpy_rng,
    make_python_rng,
    spawn_streams,
)


def test_same_labels_give_same_seed() -> None:
    assert derive_seed(42, "a", 1) == derive_seed(42, "a", 1)


def test_different_labels_give_different_seeds() -> None:
    assert derive_seed(42, "a") != derive_seed(42, "b")


def test_seed_is_stable_across_processes() -> None:
    """A literal, not a re-derivation: a regression guard on the hash function.

    If this ever changes, every seed recorded in an older artifact becomes
    unreproducible, so the value is pinned deliberately.
    """
    assert derive_seed(0, "robot_ci") == 4096978065922085741


def test_seed_in_range() -> None:
    for label in ("x", "y", 0, 1, 2, None):
        value = derive_seed(12345, label)
        assert 0 <= value < 2**63 - 1


def test_rejects_negative_master_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1, "a")


def test_rejects_bool_master_seed() -> None:
    with pytest.raises(TypeError):
        derive_seed(True, "a")  # type: ignore[arg-type]


def test_numpy_streams_are_reproducible() -> None:
    a = make_numpy_rng(5, "obs").normal(size=8)
    b = make_numpy_rng(5, "obs").normal(size=8)
    assert (a == b).all()


def test_numpy_streams_differ_across_labels() -> None:
    a = make_numpy_rng(5, "obs").normal(size=8)
    b = make_numpy_rng(5, "act").normal(size=8)
    assert not (a == b).all()


def test_python_streams_are_reproducible() -> None:
    assert [make_python_rng(3, "p").random() for _ in range(2)] == [
        make_python_rng(3, "p").random() for _ in range(2)
    ]


def test_spawn_streams_are_distinct() -> None:
    seeds = spawn_streams(1, 5, "rep")
    assert len(set(seeds)) == 5


def test_describe_seed_is_readable() -> None:
    assert describe_seed(42, "a", 3) == "42/a/3"
