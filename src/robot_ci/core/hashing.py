"""Canonical serialization and configuration hashing.

Every artifact this project writes records a ``config_hash`` so that a result
can be traced to the exact configuration that produced it. The hash is computed
over a *canonical* JSON encoding (sorted keys, no insignificant whitespace,
float normalization) so that logically identical configurations hash identically
regardless of YAML key order or formatting.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "HASH_ALGORITHM",
    "canonical_json",
    "hash_bytes",
    "hash_file",
    "hash_payload",
    "normalize_for_json",
]

HASH_ALGORITHM = "sha256"
_DIGEST_CHARS = 64  # sha256 hex digest length


def _normalize(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-safe, canonical form.

    The rules are intentionally explicit:

    * ``float`` values that are integral become ``int`` (``1.0`` and ``1`` are
      the same configuration and must hash the same).
    * non-finite floats become their string spelling, because JSON cannot
      represent them and silently dropping them would be worse.
    * mappings are emitted with sorted string keys.
    * sets and other unordered iterables are sorted before emission.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize(item) for item in value)
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]
    if hasattr(value, "model_dump"):  # Pydantic v2
        return _normalize(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):  # dataclasses / plain objects
        return _normalize(value.to_dict())
    return str(value)


def normalize_for_json(value: Any) -> Any:
    """Public wrapper around the canonicalization rules (useful in tests)."""
    return _normalize(value)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to canonical JSON.

    Args:
        value: Any JSON-compatible structure, dataclass, Pydantic model or path.

    Returns:
        Compact JSON with sorted keys and a trailing-free deterministic form.
    """
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def hash_bytes(data: bytes) -> str:
    """Return ``"sha256:<hex>"`` for raw bytes."""
    return f"{HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def hash_payload(value: Any) -> str:
    """Return ``"sha256:<hex>"`` for any canonicalizable payload.

    Example:
        >>> hash_payload({"a": 1, "b": 2.0}) == hash_payload({"b": 1, "a": 1.0})
        True
    """
    if not isinstance(hashlib.sha256().hexdigest(), str):  # pragma: no cover - sanity
        raise RuntimeError("hashlib backend is unusable")
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    if len(digest) != _DIGEST_CHARS:  # pragma: no cover - sanity
        raise RuntimeError("unexpected digest length")
    return f"{HASH_ALGORITHM}:{digest}"


def hash_file(path: str | Path) -> str:
    """Return ``"sha256:<hex>"`` for a file's raw bytes.

    Args:
        path: File to hash.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"{HASH_ALGORITHM}:{digest.hexdigest()}"
