"""
ODE solver backend for S-system simulation.

Uses libRoadRunner for ODE integration.
"""

from .interface import simulate_dae, simulate_dae_projection, simulate_model, simulate_ode

__all__ = ["simulate_ode", "simulate_dae", "simulate_dae_projection", "simulate_model"]
