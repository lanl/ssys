"""Symbolic lifting helpers for rational, composite, and time-dependent terms."""

from ssys.recaster import (
    AutonomousLiftResult,
    add_dummy_for_constants,
    find_composite_functions,
    find_rational_denominators,
    find_sqrt_of_sums,
    lift_composite_functions,
    lift_rational_functions,
    lift_squared_for_sqrt,
    lift_time_functions_to_autonomous,
)

__all__ = [
    "AutonomousLiftResult",
    "add_dummy_for_constants",
    "find_composite_functions",
    "find_rational_denominators",
    "find_sqrt_of_sums",
    "lift_composite_functions",
    "lift_rational_functions",
    "lift_squared_for_sqrt",
    "lift_time_functions_to_autonomous",
]
