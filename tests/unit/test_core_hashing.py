"""Canonical serialization and config hashing."""

from __future__ import annotations

import hashlib
import math
from enum import Enum
from pathlib import Path

from robot_ci.core.hashing import (
    canonical_json,
    hash_file,
    hash_payload,
    normalize_for_json,
)


def test_key_order_does_not_change_the_hash() -> None:
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_integral_floats_match_integers() -> None:
    assert hash_payload({"x": 1.0}) == hash_payload({"x": 1})


def test_non_integral_floats_differ() -> None:
    assert hash_payload({"x": 1.5}) != hash_payload({"x": 1.0})


def test_hash_has_the_documented_prefix() -> None:
    value = hash_payload({"a": 1})
    assert value.startswith("sha256:")
    assert len(value) == len("sha256:") + 64


def test_nested_structures_are_order_insensitive() -> None:
    left = {"outer": {"a": 1, "b": [1, 2, 3]}}
    right = {"outer": {"b": [1, 2, 3], "a": 1}}
    assert hash_payload(left) == hash_payload(right)


def test_list_order_is_significant() -> None:
    """Sequences are ordered by definition, unlike sets."""
    assert hash_payload([1, 2]) != hash_payload([2, 1])


def test_sets_are_order_insensitive() -> None:
    assert hash_payload({"s": {1, 2, 3}}) == hash_payload({"s": {3, 2, 1}})


def test_nan_is_encoded_not_dropped() -> None:
    """Dropping a NaN silently would make two different configs collide."""
    assert normalize_for_json(float("nan")) == "NaN"
    assert normalize_for_json(float("inf")) == "Infinity"
    assert normalize_for_json(float("-inf")) == "-Infinity"
    assert hash_payload({"v": math.nan}) == hash_payload({"v": math.nan})


class Colour(str, Enum):
    RED = "red"


def test_enums_are_encoded_by_value() -> None:
    assert hash_payload({"c": Colour.RED}) == hash_payload({"c": "red"})


def test_paths_are_posix_normalized() -> None:
    assert normalize_for_json(Path("/a/b")) == "/a/b"


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_hash_file_matches_payload_hash(tmp_path: Path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    expected = "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert hash_file(path) == expected
