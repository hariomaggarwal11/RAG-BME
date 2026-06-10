"""Shared validation helpers for the data models.

Field-level invariants from the design's Data Models section are enforced at
construction time so that an invalid model can never exist in memory.
"""

from __future__ import annotations

from numbers import Real
from typing import Optional


class ModelValidationError(ValueError):
    """Raised when a data model is constructed with a value that violates a
    field-level invariant defined in the design."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`ModelValidationError` when ``condition`` is falsy."""
    if not condition:
        raise ModelValidationError(message)


def require_int_in_range(
    value: object,
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Validate that ``value`` is an ``int`` within an inclusive range."""
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an int, got {type(value).__name__}",
    )
    ivalue = int(value)  # type: ignore[arg-type]
    if minimum is not None:
        require(ivalue >= minimum, f"{name} must be >= {minimum}, got {ivalue}")
    if maximum is not None:
        require(ivalue <= maximum, f"{name} must be <= {maximum}, got {ivalue}")
    return ivalue


def require_float_in_range(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Validate that ``value`` is a real number within an inclusive range."""
    require(
        isinstance(value, Real) and not isinstance(value, bool),
        f"{name} must be a number, got {type(value).__name__}",
    )
    fvalue = float(value)  # type: ignore[arg-type]
    require(
        minimum <= fvalue <= maximum,
        f"{name} must be in [{minimum}, {maximum}], got {fvalue}",
    )
    return fvalue


def require_non_empty_str(value: object, name: str) -> str:
    """Validate that ``value`` is a non-empty string."""
    require(isinstance(value, str), f"{name} must be a str, got {type(value).__name__}")
    require(len(value) > 0, f"{name} must be a non-empty string")
    return value  # type: ignore[return-value]
