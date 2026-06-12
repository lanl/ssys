"""Optional solver cross-checks for local release-candidate runs."""

from __future__ import annotations

import os
from importlib.util import find_spec

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssys.ode_backends import simulate_model
from ssys.types import ModelIR, SolverRequirement


def _require_solver_crosscheck_dependencies() -> None:
    missing = [
        name
        for name in ("antimony", "roadrunner", "sksundae")
        if find_spec(name) is None
    ]
    if not missing:
        return
    message = (
        "solver cross-checks require Antimony, libRoadRunner/CVODE, and "
        f"scikit-SUNDAE/IDA; missing: {', '.join(missing)}"
    )
    if os.environ.get("SSYS_REQUIRE_DAE_VALIDATION") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _ode_reference_model() -> ModelIR:
    model = ModelIR()
    model.species = {"X"}
    model.params = {"k": 0.3}
    model.initial = {"X": 2.0}
    model.explicit_rates = {"X": "-k*X"}
    model.solver_requirement = SolverRequirement.ODE_ONLY
    model.antimony_text = """
        model ode_reference()
          species X;
          X' = -k*X;
          k = 0.3;
          X = 2.0;
        end
    """
    return model


def _assignment_dae_model() -> ModelIR:
    model = ModelIR()
    model.species = {"X"}
    model.params = {"k": 0.3}
    model.initial = {"X": 2.0}
    model.explicit_rates = {"X": "-k*X"}
    model.assignment_rules = {"Y": "X"}
    model.solver_requirement = SolverRequirement.DAE_REQUIRED
    return model


def _implicit_constraint_dae_model() -> ModelIR:
    model = ModelIR()
    model.species = {"X", "Y"}
    model.params = {"k": 0.3}
    model.initial = {"X": 2.0, "Y": 2.0}
    model.explicit_rates = {"X": "-k*X"}
    model.algebraic_constraints = ["Y - X"]
    model.solver_requirement = SolverRequirement.DAE_REQUIRED
    return model


def _state_column(result: dict, name: str) -> np.ndarray:
    return np.asarray(result["y"][:, result["state_names"].index(name)], dtype=float)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    ("case_name", "dae_factory"),
    [
        pytest.param("assignment_rule", _assignment_dae_model, id="assignment-rule"),
        pytest.param(
            "implicit_constraint",
            _implicit_constraint_dae_model,
            id="implicit-constraint",
        ),
    ],
)
def test_roadrunner_cvode_and_ida_sundials_agree_on_small_dae_fixtures(
    case_name: str,
    dae_factory,
) -> None:
    _require_solver_crosscheck_dependencies()
    solver_options = {
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance": 1.0e-12,
        "max_num_steps": 200000,
        "maximum_num_steps": 200000,
        "repair_consistent_initial_conditions": True,
    }

    ode_result = simulate_model(
        _ode_reference_model(),
        t0=0.0,
        t_end=2.0,
        n_points=41,
        options=solver_options,
    )
    dae_result = simulate_model(
        dae_factory(),
        t0=0.0,
        t_end=2.0,
        n_points=41,
        options=solver_options,
    )

    assert ode_result["success"], ode_result["message"]
    assert dae_result["success"], dae_result["message"]
    assert ode_result["backend"] == "roadrunner_cvode"
    assert dae_result["backend"] == "ida_sundials"

    ode_x = _state_column(ode_result, "X")
    dae_x = _state_column(dae_result, "X")
    max_abs = float(np.max(np.abs(ode_x - dae_x)))
    max_rel = float(np.max(np.abs(ode_x - dae_x) / np.maximum(np.abs(ode_x), 1.0e-12)))

    assert_allclose(
        dae_x,
        ode_x,
        rtol=1.0e-5,
        atol=1.0e-7,
        err_msg=(
            f"backend_numerical_difference:{case_name}: "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        ),
    )
    for residual in dae_result.get("algebraic_residuals", {}).values():
        assert residual < 1.0e-7
