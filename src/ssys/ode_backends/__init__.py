"""
ODE solver backends for S-system simulation.

Provides a unified interface for different ODE solvers:
- libRoadRunner (default, production)
- RK4 (fallback, debugging)
"""

from .interface import simulate_ode

__all__ = ['simulate_ode']
