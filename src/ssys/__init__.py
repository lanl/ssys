"""
ssys: Exact algebraic recasting of ODEs to canonical S-system form.

This package provides tools to automatically transform arbitrary ordinary differential
equations (ODEs) into canonical S-system form using exact algebraic recasting. S-systems
are a power-law formalism from Biochemical Systems Theory that express dynamics as
differences of product terms with real-valued exponents.

Main components:
- parse_antimony: Parse Antimony model syntax into an intermediate representation
- build_sym_system: Build symbolic ODE system from parsed model
- recast_to_ssystem: Transform symbolic ODEs into canonical S-system form
- ssystem_to_antimony: Export recast S-system back to Antimony format
"""

from ssys.recaster import (
    ModelIR,
    RecastResult,
    SSysEquation,
    SymSystem,
    SystemClass,
    build_sym_system,
    canonicalize_aux_names,
    classify_result,
    parse_antimony,
    parse_sbml,
    recast_to_ssystem,
    ssystem_to_antimony,
)

__version__ = "0.5.5"

__all__ = [
    "ModelIR",
    "SymSystem",
    "RecastResult",
    "SSysEquation",
    "SystemClass",
    "parse_antimony",
    "parse_sbml",
    "build_sym_system",
    "recast_to_ssystem",
    "ssystem_to_antimony",
    "canonicalize_aux_names",
    "classify_result",
]
