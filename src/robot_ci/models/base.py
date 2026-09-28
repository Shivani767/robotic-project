"""Base model configuration shared by every ROBOT CI schema.

Design rules enforced here:

* **Versioned** — every persisted model carries :data:`SCHEMA_VERSION`.
* **Strict** — unknown fields are rejected, so a typo in a YAML file is an
  error rather than a silently ignored setting.
* **Immutable where it matters** — scenario and sensor configuration use
  frozen models so that a running experiment cannot mutate the conditions it
  claimed to be testing.
* **No mutable defaults** — mutable defaults go through Pydantic factories.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from robot_ci.core.version import SCHEMA_VERSION

__all__ = ["RCI_BASE_CONFIG", "ModelT", "VersionedModel", "schema_version_field"]

RCI_BASE_CONFIG = ConfigDict(
    extra="forbid",
    validate_assignment=True,
    frozen=True,
    populate_by_name=True,
    str_strip_whitespace=True,
    use_enum_values=False,
    ser_json_timedelta="iso8601",
)

ModelT = TypeVar("ModelT", bound="VersionedModel")


class VersionedModel(BaseModel):
    """Base class for every persisted schema.

    Attributes:
        schema_version: Semantic version of the schema, defaulted to the
            package-wide :data:`robot_ci.core.SCHEMA_VERSION`.
    """

    model_config = RCI_BASE_CONFIG

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Semantic version of this schema, used to reject stale artifacts.",
    )

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the fields that participate in configuration hashing.

        ``schema_version`` is excluded on purpose: a *result* stored under two
        compatible schema versions should still hash the same when the
        experimental conditions are unchanged.
        """
        return self.model_dump(mode="json", exclude={"schema_version"})

    def upgraded_to(self, *, expected: type[ModelT]) -> ModelT:
        """Re-validate this instance against another model type.

        This is the schema-migration hook: an artifact written by an older
        version is loaded, its ``schema_version`` is inspected, and the payload
        is validated into the current model.

        Args:
            expected: The model type to validate into.

        Returns:
            A new instance of ``expected``.

        Raises:
            TypeError: If ``expected`` is not supplied.

        Example:
            >>> ScenarioConfig().upgraded_to(expected=ScenarioConfig).scenario_id
            'nominal'
        """
        if expected is None:  # pragma: no cover - programming error
            raise TypeError("`expected` model type is required")
        return expected.model_validate(self.model_dump(mode="python"))


def schema_version_field(description: str = "Schema version") -> Any:
    """Return a reusable ``schema_version`` field (for nested non-model types)."""
    return Field(default=SCHEMA_VERSION, description=description)
