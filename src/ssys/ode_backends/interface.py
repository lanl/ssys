"""
Unified interface for ODE solver backends.

Uses libRoadRunner for ODE integration.
"""

from typing import Any

import numpy as np

from ..recaster import ModelIR


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

        return simulate_with_roadrunner(
            model_ir, t0, t_end, n_points, y0_override, options
        )
    except ImportError as e:
        # libRoadRunner not available
        return {
            "t": np.array([]),
            "y": np.array([]),
            "state_names": [],
            "success": False,
            "message": f"libRoadRunner not available: {e}",
            "integrator_stats": {},
        }
