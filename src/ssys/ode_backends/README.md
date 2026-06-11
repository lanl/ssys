# Solver Backends

This module provides the validation-time solver interface used by `ssys`.

## Backend Selection

- `ode_only`: uses `roadrunner_cvode`.
- `ode_with_assignment_rules`: uses `roadrunner_cvode_assignment_rules`.
- `dae_required`: uses `ida_sundials` by default.

The older `dae_projection` path is still available for diagnostics:

```python
simulate_model(model_ir, 0.0, 10.0, 101, options={"dae_backend": "dae_projection"})
```

Projection integrates the differential part and recomputes explicit algebraic
quantities afterward. It is not a general DAE solver and is not selected by
default for production validation.

## Optional DAE Dependency

IDA/SUNDIALS support is optional:

```bash
uv sync --extra dae
```

or, for an editable install:

```bash
uv pip install -e ".[dae]"
```

The `dae` extra currently uses `scikit-sundae`, which provides Python bindings
to SUNDIALS IDA. It is kept out of the base install because SUNDIALS-backed
binary wheels and supported Python/platform combinations can lag the pure Python
and RoadRunner dependency set. If the extra is missing, `dae_required` validation
returns `UNSUPPORTED` with the installation hint instead of passing.

## IDA Residual Form

The `ida_sundials` backend builds a residual system `F(t, y, ydot) = 0`:

- Differential states: `ydot_i - f_i(y, z, t) = 0`.
- Explicit assignment auxiliaries: `z_j - g_j(y, z, t) = 0`.
- Implicit algebraic constraints: `h_k(y, z, t) = 0`.

Explicit algebraic variables are initialized from their definitions when possible.
If a caller provides inconsistent algebraic initial conditions, the backend fails
unless `repair_consistent_initial_conditions=True` is set.

## Result Metadata

All backend results include:

- `backend`
- `solver_requirement`
- `unsupported_solver_requirement`
- `integrator_stats`

IDA results also record the selected package/version, tolerances, algebraic
indices, return status, solver message, initial residual norms, and algebraic
residual norms over the trajectory. Validator residual reports include absolute
and scaled norms for explicit algebraic definitions so large-magnitude
manifolds are checked against solver-relative accuracy instead of raw absolute
roundoff alone.
