"""YAML / JSON loading, saving and validation helpers.

All configuration enters the system through this module so that error messages
are uniform and so that a file's own hash can be recorded alongside the parsed
value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_ci.core.errors import ConfigurationError
from robot_ci.core.hashing import hash_file, hash_payload

__all__ = [
    "dumps_yaml",
    "load_config_file",
    "load_yaml",
    "write_json",
    "write_yaml",
]

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSON_SUFFIXES = {".json"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML (or JSON) mapping from disk.

    Args:
        path: File to read.

    Returns:
        The parsed mapping.

    Raises:
        ConfigurationError: If the file is missing, unparsable, or does not
            contain a mapping at the top level.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigurationError(
            f"configuration file not found: {file_path}",
            code="config.file_missing",
            path=str(file_path),
        )
    if file_path.suffix.lower() not in _YAML_SUFFIXES | _JSON_SUFFIXES:
        raise ConfigurationError(
            f"unsupported configuration extension: {file_path.suffix!r}",
            code="config.unsupported_extension",
            path=str(file_path),
            supported=sorted(_YAML_SUFFIXES | _JSON_SUFFIXES),
        )

    text = file_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"invalid YAML in {file_path}: {exc}",
            code="config.parse_error",
            path=str(file_path),
        ) from exc

    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigurationError(
            f"top level of {file_path} must be a mapping, got {type(data).__name__}",
            code="config.not_a_mapping",
            path=str(file_path),
        )
    return dict(data)


def load_config_file(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a configuration file together with the hash of its raw bytes.

    Recording the *file* hash (not just the parsed value) means an artifact can
    be traced back to the literal file on disk, including comments and ordering.

    Returns:
        ``(data, file_hash)`` where ``file_hash`` looks like ``"sha256:..."``.
    """
    file_path = Path(path)
    return load_yaml(file_path), hash_file(file_path)


def dumps_yaml(data: Mapping[str, Any]) -> str:
    """Serialize a mapping to block-style YAML with stable key order."""
    return yaml.safe_dump(
        dict(data),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> Path:
    """Write ``data`` to ``path`` as YAML, creating parent directories."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(dumps_yaml(data), encoding="utf-8")
    return file_path


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> Path:
    """Write ``data`` to ``path`` as UTF-8 JSON, creating parent directories."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, indent=indent, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return file_path


def payload_hash(data: Any) -> str:
    """Convenience re-export of :func:`robot_ci.core.hashing.hash_payload`."""
    return hash_payload(data)
