"""Shared data types for ssys parsing, recasting, and formatting."""

from dataclasses import dataclass, field
from enum import Enum

import sympy as sp


class SolverRequirement(Enum):
    """Numerical backend required to validate or simulate a generated model."""

    ODE_ONLY = "ode_only"
    ODE_WITH_ASSIGNMENT_RULES = "ode_with_assignment_rules"
    DAE_REQUIRED = "dae_required"


class SBMLParseError(ValueError):
    """Structured SBML math parse/evaluation error."""

    def __init__(
        self,
        kind: str,
        formula: str | None,
        message: str,
        *,
        source: str,
        reaction_id: str | None = None,
        reaction_name: str | None = None,
        variable: str | None = None,
    ) -> None:
        self.kind = kind
        self.formula = formula
        self.message = message
        self.source = source
        self.reaction_id = reaction_id
        self.reaction_name = reaction_name
        self.variable = variable
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.kind == "kinetic_law":
            context = f"reaction {self.reaction_id or '<unnamed>'}"
            if self.reaction_name:
                context += f" ({self.reaction_name})"
            target = f"kinetic law in {context}"
        elif self.kind == "rate_rule":
            target = f"rate rule for variable {self.variable or '<unknown>'}"
        elif self.kind == "initial_assignment":
            target = f"initial assignment for symbol {self.variable or '<unknown>'}"
        else:
            target = self.kind

        formula = self.formula if self.formula not in (None, "") else "<none>"
        return f"Failed to parse SBML {target} in {self.source}: formula {formula!r}: {self.message}"


@dataclass
class SymSystem:
    vars: list[sp.Symbol]
    params: dict[str, float]
    odes: dict[sp.Symbol, sp.Expr]
    initials: dict[sp.Symbol, float]
    initial_exprs: dict[sp.Symbol, str] = field(default_factory=dict)
    assignment_rules: dict[str, str] = field(default_factory=dict)
    algebraic_constraints: list[str] = field(default_factory=list)
    compartments: dict[str, float] = field(default_factory=dict)
    sim_t_start: float | None = None
    sim_t_end: float | None = None
    sim_n_steps: int | None = None
    eps_init: float | None = None
    eps_slack: float | None = None
    antimony_text: str = ""
    solver_requirement: SolverRequirement = SolverRequirement.ODE_ONLY


@dataclass
class SSysEquation:
    var: sp.Symbol
    growth: tuple[sp.Expr, dict[sp.Symbol, float]]
    decay: tuple[sp.Expr, dict[sp.Symbol, float]]


@dataclass
class GMAEquation:
    """Generalized Mass Action equation with multiple production/degradation terms."""

    var: sp.Symbol
    production: list[tuple[sp.Expr, dict[sp.Symbol, float]]]
    degradation: list[tuple[sp.Expr, dict[sp.Symbol, float]]]


class RecastStatus(Enum):
    """Status of recasting operation."""

    CANONICAL_SSYSTEM = "canonical_ssystem"
    GMA = "gma"
    FAILED = "failed"


class SystemClass(Enum):
    """Classification of system form."""

    SSYSTEM = "S-system"
    CANONICAL_SSYSTEM = "Canonical S-system"
    GMA = "GMA"
    GMA_TIME_VARYING = "GMA with time-varying coefficients"
    GENERAL = "General"


@dataclass
class RecastResult:
    status: RecastStatus
    equations: list[SSysEquation]
    initials: dict[sp.Symbol, float]
    variables: list[sp.Symbol]
    factor_map: dict[sp.Symbol, list[sp.Symbol]] = field(default_factory=dict)
    gma_equations: list[GMAEquation] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    compartments: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    blockers: dict[str, list[str]] = field(default_factory=dict)
    auxiliary_defs: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)
    canonical_refusal_reason: str | None = None
    initial_exprs: dict[sp.Symbol, str] = field(default_factory=dict)
    assignment_rules: dict[str, str] = field(default_factory=dict)
    algebraic_constraints: list[str] = field(default_factory=list)
    solver_requirement: SolverRequirement = SolverRequirement.ODE_ONLY
    sim_t_start: float | None = None
    sim_t_end: float | None = None
    sim_n_steps: int | None = None
    eps_init: float | None = None
    eps_slack: float | None = None


__all__ = [
    "GMAEquation",
    "RecastResult",
    "RecastStatus",
    "SBMLParseError",
    "SSysEquation",
    "SolverRequirement",
    "SymSystem",
    "SystemClass",
]
