"""
Unified interface for solver backends.

ODE-only models use libRoadRunner/CVODE. Recast outputs that require DAE
simulation use the optional IDA/SUNDIALS backend by default. The projection
backend remains available as an explicit diagnostic fallback for simple
explicit manifolds.
"""

from typing import Any

import numpy as np

from ..recaster import ModelIR, SolverRequirement, normalize_solver_requirement


def _infer_solver_requirement(model_ir: ModelIR) -> SolverRequirement:
    configured = normalize_solver_requirement(getattr(model_ir, "solver_requirement", None))
    if configured is not None:
        return configured
    if getattr(model_ir, "algebraic_constraints", None):
        return SolverRequirement.DAE_REQUIRED
    if getattr(model_ir, "assignment_rules", None):
        return SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
    return SolverRequirement.ODE_ONLY


def _annotate_result(
    result: dict[str, Any], *, backend: str, requirement: SolverRequirement
) -> dict[str, Any]:
    result["backend"] = backend
    result["solver_requirement"] = requirement.value
    result.setdefault("unsupported_solver_requirement", False)
    return result


def _failure_result(
    *,
    message: str,
    backend: str,
    requirement: SolverRequirement,
    unsupported: bool = False,
) -> dict[str, Any]:
    return {
        "t": np.array([]),
        "y": np.array([]),
        "state_names": [],
        "success": False,
        "message": message,
        "backend": backend,
        "solver_requirement": requirement.value,
        "unsupported_solver_requirement": unsupported,
        "integrator_stats": {},
    }


def simulate_ode(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Simulate an ODE system using libRoadRunner.

    Args:
        model_ir: Parsed model intermediate representation
        t0: Start time
        t_end: End time
        n_points: Number of time points
        y0_override: Optional initial conditions override
        options: Backend-specific options

    Returns:
        Dictionary containing:
            - t: Time array (n_points,)
            - y: State array (n_points, n_states)
            - state_names: List of state variable names
            - success: True if simulation succeeded
            - message: Error message if failed (empty on success)
            - integrator_stats: Dict with step counts, etc.
    """
    if options is None:
        options = {}

    try:
        from .roadrunner_backend import simulate_with_roadrunner

        result = simulate_with_roadrunner(
            model_ir, t0, t_end, n_points, y0_override, options
        )
        return _annotate_result(
            result,
            backend="roadrunner_cvode",
            requirement=SolverRequirement.ODE_ONLY,
        )
    except ImportError as e:
        # libRoadRunner not available
        return _failure_result(
            message=f"libRoadRunner not available: {e}",
            backend="roadrunner_cvode",
            requirement=SolverRequirement.ODE_ONLY,
        )


def simulate_dae(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a DAE model using the selected DAE-capable backend."""
    if options is None:
        options = {}

    requirement = SolverRequirement.DAE_REQUIRED
    backend = options.get("dae_backend", options.get("backend", "ida_sundials"))

    if backend in {"projection", "dae_projection"}:
        return simulate_dae_projection(
            model_ir, t0, t_end, n_points, y0_override, options
        )

    if backend not in {"ida", "ida_sundials", "sundials_ida"}:
        return _failure_result(
            message=f"Unsupported DAE backend {backend!r}",
            backend=str(backend),
            requirement=requirement,
            unsupported=True,
        )

    try:
        from .ida_sundials_backend import simulate_with_ida_sundials
    except ImportError as e:
        return _failure_result(
            message=f"IDA/SUNDIALS backend unavailable: {e}",
            backend="ida_sundials",
            requirement=requirement,
            unsupported=True,
        )

    result = simulate_with_ida_sundials(
        model_ir, t0, t_end, n_points, y0_override, options
    )
    return _annotate_result(result, backend="ida_sundials", requirement=requirement)


def simulate_dae_projection(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate a DAE-required model with the explicit projection fallback."""
    requirement = SolverRequirement.DAE_REQUIRED
    try:
        from .dae_backend import simulate_with_dae_projection
    except ImportError as e:
        return _failure_result(
            message=f"DAE projection backend unavailable: {e}",
            backend="dae_projection",
            requirement=requirement,
            unsupported=True,
        )

    result = simulate_with_dae_projection(
        model_ir, t0, t_end, n_points, y0_override, options
    )
    return _annotate_result(result, backend="dae_projection", requirement=requirement)


def simulate_model(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: dict[str, float] | None = None,
    options: dict[str, Any] | None = None,
    solver_requirement: str | SolverRequirement | None = None,
) -> dict[str, Any]:
    """Select and run the backend required by a parsed model."""
    requirement = normalize_solver_requirement(solver_requirement) or _infer_solver_requirement(
        model_ir
    )

    if requirement == SolverRequirement.DAE_REQUIRED:
        return simulate_dae(model_ir, t0, t_end, n_points, y0_override, options)

    if requirement == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES:
        result = simulate_ode(model_ir, t0, t_end, n_points, y0_override, options)
        return _annotate_result(
            result,
            backend="roadrunner_cvode_assignment_rules",
            requirement=requirement,
        )

    if requirement == SolverRequirement.ODE_ONLY:
        return simulate_ode(model_ir, t0, t_end, n_points, y0_override, options)

    return _failure_result(
        message=f"Unsupported solver requirement: {requirement}",
        backend="none",
        requirement=SolverRequirement.DAE_REQUIRED,
        unsupported=True,
    )
