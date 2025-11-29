"""
Unified interface for ODE solver backends.
"""

from typing import Dict, Optional, Any
import numpy as np
from ..recaster import ModelIR


def simulate_ode(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: Optional[Dict[str, float]] = None,
    backend: str = "roadrunner",
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simulate an ODE system using the specified backend.
    
    Args:
        model_ir: Parsed model intermediate representation
        t0: Start time
        t_end: End time
        n_points: Number of time points
        y0_override: Optional initial conditions override
        backend: Solver backend ('roadrunner' or 'rk4')
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
    
    if backend == "roadrunner":
        try:
            from .roadrunner_backend import simulate_with_roadrunner
            return simulate_with_roadrunner(
                model_ir, t0, t_end, n_points, y0_override, options
            )
        except ImportError as e:
            # libRoadRunner not available
            if options.get("fallback_to_rk4", True):
                from .rk4_backend import simulate_with_rk4
                return simulate_with_rk4(
                    model_ir, t0, t_end, n_points, y0_override, options
                )
            else:
                return {
                    "t": np.array([]),
                    "y": np.array([]),
                    "state_names": [],
                    "success": False,
                    "message": f"libRoadRunner not available: {e}",
                    "integrator_stats": {}
                }
    elif backend == "rk4":
        from .rk4_backend import simulate_with_rk4
        return simulate_with_rk4(
            model_ir, t0, t_end, n_points, y0_override, options
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Choose 'roadrunner' or 'rk4'."
        )
