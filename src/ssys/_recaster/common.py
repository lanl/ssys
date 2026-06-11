# mypy: ignore-errors
# ruff: noqa: F401,I001
"""Shared imports, constants, and compatibility symbols for recaster internals."""

import re
from dataclasses import dataclass

import sympy as sp

from ssys.classification import (
    classify_result,
    classify_solver_requirement,
    classify_sym_system_solver_requirement,
    classify_system,
)
from ssys.math_utils import (
    _expand_exps_through_factors,
    _exponents_match,
    _get_coefficient_sign,
    _is_term_monomial,
    expand_to_terms,
    product_expr,
)
from ssys.metadata import (
    _extract_sim_metadata,
    _extract_solver_requirement_metadata,
    _format_antimony_number,
    _format_sim_metadata_lines,
    _format_solver_metadata_lines,
    normalize_solver_requirement,
)
from ssys.types import (
    GMAEquation,
    ModelIR,
    Reaction,
    RecastResult,
    RecastStatus,
    SBMLParseError,
    SolverRequirement,
    SSysEquation,
    SymSystem,
    SystemClass,
)

arrow_pat = re.compile(r"<->|->")
prime_rule_pat = re.compile(r"^\s*\$?([A-Za-z_]\w*)\s*'\s*=\s*(.+)$")
func_def_pat = re.compile(r"^([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:=\s*(.+)$")
func_call_start_pat = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
simple_identifier_pat = re.compile(r"^[A-Za-z_]\w*$")
simple_numeric_literal_pat = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

EPS_INIT = 1e-6
EPS_SLACK = 1.0

ANTIMONY_RESERVED_KEYWORDS = frozenset({
    "compartment",
    "DNA",
    "RNA",
})

__all__ = [name for name in globals() if not name.startswith("__")]
