"""
ODE solver backend for S-system simulation.

Uses libRoadRunner for ODE integration.
"""

from .interface import simulate_ode

__all__ = ["simulate_ode"]
