"""Access to packaged validation-report JSON Schema resources."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, cast

from ssys._validator.report import VALIDATION_REPORT_SCHEMA_VERSION

VALIDATION_REPORT_SCHEMA_RESOURCE = "validation-report-v1.schema.json"

__all__ = [
    "VALIDATION_REPORT_SCHEMA_RESOURCE",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "load_validation_report_schema",
]


def load_validation_report_schema() -> dict[str, Any]:
    """Load the JSON Schema for validation reports emitted by this package."""
    schema_text = (
        files("ssys.schemas")
        .joinpath(VALIDATION_REPORT_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return cast(dict[str, Any], json.loads(schema_text))
