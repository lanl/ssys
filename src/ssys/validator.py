"""Compatibility API for recast validation.

Implementation is split across ``ssys._validator`` modules; this module preserves
historical imports from ``ssys.validator``.
"""

from ssys._validator.common import (
    _canonicalize_expr_by_name,
    _is_dev_mode,
    _simplify_identity_difference,
    _substitute_symbols_by_name,
)
from ssys._validator.core import RecastValidator, validate_recast_pair
from ssys._validator.report import (
    EquivalenceTest,
    VALIDATION_REPORT_SCHEMA_VERSION,
    ValidationProfile,
    ValidationProfileSpec,
    ValidationReport,
    ValidationResult,
    _test_passed,
    validation_profile_choices,
)
from ssys._validator.serialization import validate_generated_output_roundtrip
from ssys.validation_schema import load_validation_report_schema

__all__ = [name for name in globals() if not name.startswith("__")]
