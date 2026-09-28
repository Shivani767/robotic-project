"""Deterministic seed derivation.

Reproducibility is the backbone of this project. Every stochastic component
(scenario sampling, sensor corruption, policy tie-breaking, search proposals)
draws from a generator whose seed is *derived* from a small tuple of labels
rather than from a shared mutable global RNG. That makes any single episode
addressable in isolation, which is what counterexample reproduction requires.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

__all__ = [
    "SEED_MODULUS",
    "derive_seed",
    "make_numpy_rng",
    "make_python_rng",
    "spawn_streams",
]

#: Seeds are kept inside 63 bits so they survive a JSON round-trip unchanged and
#: can be passed to ``np.random.default_rng`` on every supported version.
SEED_MODULUS = 2**63 - 1


def derive_seed(master_seed: int, *labels: object) -> int:
    """Derive a stable child seed from a master seed and a path of labels.

    The derivation uses BLAKE2b over a canonical string, so it is stable across
    processes, platforms and Python versions (unlike :func:`hash`, which is
    randomized per process by ``PYTHONHASHSEED``).

    Args:
        master_seed: The experiment-level seed.
        *labels: Ordered, hashable labels identifying the derived stream, e.g.
            ``derive_seed(42, "scenario", "nominal", "replicate", 3)``.

    Returns:
        A non-negative seed in ``[0, SEED_MODULUS)``.

    Example:
        >>> derive_seed(42, "a") == derive_seed(42, "a")
        True
        >>> derive_seed(42, "a") == derive_seed(42, "b")
        False
    """
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError(f"master_seed must be int, got {type(master_seed).__name__}")
    if master_seed < 0:
        raise ValueError(f"master_seed must be non-negative, got {master_seed}")

    payload = "|".join([str(master_seed), *(str(label) for label in labels)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MODULUS


def make_numpy_rng(master_seed: int, *labels: object) -> np.random.Generator:
    """Return a NumPy :class:`~numpy.random.Generator` for a derived stream.

    Example:
        >>> a = make_numpy_rng(1, "x").normal(size=3)
        >>> b = make_numpy_rng(1, "x").normal(size=3)
        >>> bool((a == b).all())
        True
    """
    return np.random.default_rng(derive_seed(master_seed, *labels))


def make_python_rng(master_seed: int, *labels: object) -> random.Random:
    """Return a :class:`random.Random` for a derived stream."""
    return random.Random(derive_seed(master_seed, *labels))


def spawn_streams(master_seed: int, count: int, *labels: object) -> list[int]:
    """Return ``count`` distinct child seeds derived from one parent.

    Useful for a scenario-independent comparison: the same ``count`` seeds are
    handed to every policy so that baselines and the proposed method see
    identical randomness.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    return [derive_seed(master_seed, *labels, index) for index in range(count)]


def describe_seed(master_seed: int, *labels: object) -> str:
    """Return a human-readable, unambiguous description of a derived stream."""
    path: Sequence[object] = (master_seed, *labels)
    return "/".join(str(part) for part in path)
