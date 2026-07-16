"""Route Program validation records by explicit schema without guessing intent."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cascade_planner.application.execution_program_validations import (
    EXECUTION_PROGRAM_VALIDATION_SCHEMA,
)
from cascade_planner.application.mechanism_program_validations import (
    MECHANISM_PROGRAM_VALIDATION_SCHEMA,
)


def partition_program_validations(
    validations: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return biocatalytic/legacy, execution, and mechanism validation rows."""

    biocatalytic: list[dict[str, Any]] = []
    execution: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    for value in validations:
        row = dict(value)
        if row.get("schema_version") == EXECUTION_PROGRAM_VALIDATION_SCHEMA:
            execution.append(row)
        elif row.get("schema_version") == MECHANISM_PROGRAM_VALIDATION_SCHEMA:
            mechanism.append(row)
        else:
            biocatalytic.append(row)
    return biocatalytic, execution, mechanism


__all__ = ["partition_program_validations"]
