# ODE Solver Backends

This module provides a unified interface for simulating ODE systems with different solver backends.

## Overview

The `simulate_ode()` function provides a consistent API for running ODE simulations, with support for:
- **libRoadRunner** (default) - Production-quality CVODE integrator
- **RK4** (fallback) - Simple Runge-Kutta 4th order method

## Usage

```python
from src.ssys.recaster import parse_antimony
from src.ssys.ode_backends import simulate_ode

# Parse a model
model_ir = parse_antimony(antimony_text)

# Simulate with default (roadrunner) backend
result = simulate_ode(
    model_ir,
    t0=0.0,
    t_end=10.0,
    n_points=100,
    backend="roadrunner"
)

if result["success"]:
    t = result["t"]        # Time array
    y = result["y"]        # State trajectories
    names = result["state_names"]  # Variable names
```

## Backend Selection

### libRoadRunner (Recommended)

Uses CVODE integrator with adaptive step sizing. Best for:
- Production simulations
- Stiff systems
- High accuracy requirements

```python
result = simulate_ode(
    model_ir,
    t0=0.0,
    t_end=10.0,
    n_points=100,
    backend="roadrunner",
    options={
        "integrator": "cvode",
        "absolute_tolerance": 1e-9,
        "relative_tolerance": 1e-6,
        "maximum_num_steps": 20000,
        "log_solver_details": False
    }
)
```

**Installation**: `pip install libroadrunner`

### RK4 (Fallback)

Simple fixed-step RK4 integrator. Best for:
- Debugging
- Simple non-stiff systems
- When RoadRunner unavailable

```python
result = simulate_ode(
    model_ir,
    t0=0.0,
    t_end=10.0,
    n_points=100,
    backend="rk4"
)
```

**Note**: RK4 backend requires integration of existing RK4 code.

## Return Format

All backends return a dictionary with:

```python
{
    "t": np.ndarray,              # Time points (n_points,)
    "y": np.ndarray,              # States (n_points, n_variables)
    "state_names": List[str],     # Variable names
    "success": bool,              # True if completed
    "message": str,               # Error message if failed
    "integrator_stats": dict      # Step counts, etc.
}
```

## Error Handling

Simulations that fail return `success=False` with diagnostic info:

```python
result = simulate_ode(...)
if not result["success"]:
    print(f"Simulation failed: {result['message']}")
```

## Automatic Fallback

By default, if RoadRunner is unavailable, the system falls back to RK4:

```python
result = simulate_ode(
    model_ir, ...,
    backend="roadrunner",
    options={"fallback_to_rk4": True}  # Default
)
```

To disable fallback:

```python
result = simulate_ode(
    model_ir, ...,
    backend="roadrunner",
    options={"fallback_to_rk4": False}
)
```

## Architecture

```
ode_backends/
├── __init__.py           # Exports simulate_ode()
├── interface.py          # Main API and backend routing
├── roadrunner_backend.py # libRoadRunner implementation
├── rk4_backend.py        # RK4 implementation
└── README.md            # This file
```

## Development Status

**Phase 1 (MVP)**: ✅ Complete
- Unified interface
- RoadRunner backend
- RK4 placeholder
- Basic tests

**Phase 2**: ✅ Complete
- ✅ RK4 backend fully integrated
- ✅ Symbol identity handling
- ✅ All tests passing (2 passed, 1 skipped)
- Validator integration (planned for Phase 3)
- Performance benchmarks (planned for Phase 3)

**Phase 3** (Future):
- Validator trajectory testing integration
- Performance benchmarks (RK4 vs RoadRunner)
- Additional backends (scipy, Julia)
- Advanced features (sensitivity, steady state)
