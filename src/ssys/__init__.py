"""
ssys: Exact algebraic recasting of ODEs to canonical S-system form.

This package provides tools to automatically transform arbitrary ordinary differential
equations (ODEs) into canonical S-system form using exact algebraic recasting. S-systems
are a power-law formalism from Biochemical Systems Theory that express dynamics as
differences of product terms with real-valued exponents.

Main components:
- parse_antimony_via_sbml: Parse Antimony model syntax into a symbolic ODE system
- recast_to_ssystem: Transform symbolic ODEs into canonical S-system form
- ssystem_to_antimony: Export recast S-system back to Antimony format
"""

from ssys.classification import (
    classify_result,
    classify_solver_requirement,
)
from ssys.formatting import ssystem_to_antimony
from ssys.parsing import parse_antimony_via_sbml, parse_sbml
from ssys.recasting import canonicalize_aux_names, recast_to_ssystem
from ssys.types import (
    ModelIR,
    RecastResult,
    SBMLParseError,
    SolverRequirement,
    SSysEquation,
    SymSystem,
    SystemClass,
)
from ssys.validation_schema import (
    VALIDATION_REPORT_SCHEMA_RESOURCE,
    VALIDATION_REPORT_SCHEMA_VERSION,
    load_validation_report_schema,
)
from ssys.validator import ValidationProfile, validation_profile_choices

__version__ = "0.6.1"
__release_date__ = "2026-07-05"
__release_maturity__ = "alpha"

__all__ = [
    "ModelIR",
    "SymSystem",
    "RecastResult",
    "SBMLParseError",
    "SSysEquation",
    "SolverRequirement",
    "SystemClass",
    "ValidationProfile",
    "VALIDATION_REPORT_SCHEMA_RESOURCE",
    "VALIDATION_REPORT_SCHEMA_VERSION",
    "parse_antimony_via_sbml",
    "parse_sbml",
    "recast_to_ssystem",
    "ssystem_to_antimony",
    "canonicalize_aux_names",
    "classify_result",
    "classify_solver_requirement",
    "load_validation_report_schema",
    "validation_profile_choices",
]
