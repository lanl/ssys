"""
Tests for ODE solver backends.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ssys.ode_backends import simulate_ode
from ssys.ode_backends.dae_backend import (
    _evaluate_expr_over_trajectory,
    _project_variable,
    simulate_with_dae_projection,
)
from ssys.ode_backends.ida_sundials_backend import IDASundialsUnavailable
from ssys.ode_backends.interface import simulate_model
from ssys.ode_backends.roadrunner_backend import (
    _evaluate_complete_gamma_argument,
    _get_antimony_text,
    _replace_complete_gamma_calls,
    _set_initial_conditions,
    simulate_with_roadrunner,
)
from ssys.recaster import ModelIR, SolverRequirement, parse_antimony


def test_roadrunner_backend_success_and_initial_condition_override(monkeypatch):
    """RoadRunner backend parses, simulates, and applies state IC overrides."""
    model_ir = ModelIR()
    model_ir.species = {"X"}
    model_ir.explicit_rates = {"X": "-k*X"}
    model_ir.params = {"k": 0.5}
    model_ir.antimony_text = """
        model test()
            species X;
            X' = -k*X;
            k = 0.5;
            X = 1;
        end
    """
    model_ir.initials = {"X": 1.0, "k": 0.5, ("compartment", "cell"): 1.0}

    class FakeAntimony:
        def clearPreviousLoads(self):
            pass

        def loadAntimonyString(self, text):
            assert "model test()" in text
            return 0

        def getLastError(self):
            return ""

        def getMainModuleName(self):
            return "test"

        def getSBMLString(self, model_name):
            assert model_name == "test"
            return "<sbml/>"

    class FakeSimulationResult:
        colnames = ["time", "[X]"]

        def __init__(self):
            self.data = np.array([[0.0, 2.5], [1.0, 1.25]])

        def __getitem__(self, key):
            return self.data[key]

    class FakeIntegrator:
        def getNumSteps(self):
            return 4

        def getNumErrTestFails(self):
            return 0

    class FakeRoadRunner:
        last_instance = None

        def __init__(self, sbml):
            assert sbml == "<sbml/>"
            self.integrator = FakeIntegrator()
            self.values = {}
            FakeRoadRunner.last_instance = self

        def setIntegrator(self, name):
            self.integrator_name = name

        def resetToOrigin(self):
            self.values.clear()

        def getFloatingSpeciesIds(self):
            return ["X"]

        def __setitem__(self, key, value):
            self.values[key] = value

        def simulate(self, t0, t_end, n_points):
            assert (t0, t_end, n_points) == (0.0, 1.0, 2)
            assert self.values == {"[X]": 2.5}
            return FakeSimulationResult()

    monkeypatch.setitem(sys.modules, "antimony", FakeAntimony())
    monkeypatch.setitem(
        sys.modules,
        "roadrunner",
        SimpleNamespace(RoadRunner=FakeRoadRunner),
    )

    result = simulate_with_roadrunner(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        y0_override={"X": 2.5, "k": 99.0},
    )

    assert result["success"] is True
    assert result["state_names"] == ["X"]
    assert result["y"].tolist() == [[2.5], [1.25]]
    assert result["integrator_stats"]["n_steps"] == 4
    assert FakeRoadRunner.last_instance.integrator_name == "cvode"


def test_roadrunner_backend_reports_antimony_parser_failure(monkeypatch):
    """Antimony parser failures are returned as structured backend failures."""
    model_ir = ModelIR()
    model_ir.species = {"X"}
    model_ir.explicit_rates = {"X": "-k*X"}
    model_ir.antimony_text = "model bad("

    class FakeAntimony:
        def clearPreviousLoads(self):
            pass

        def loadAntimonyString(self, text):
            assert text == "model bad("
            return -1

        def getLastError(self):
            return "syntax error near '('"

    class UnusedRoadRunner:
        def __init__(self, *args, **kwargs):
            raise AssertionError("RoadRunner should not be constructed after parse failure")

    monkeypatch.setitem(sys.modules, "antimony", FakeAntimony())
    monkeypatch.setitem(
        sys.modules,
        "roadrunner",
        SimpleNamespace(RoadRunner=UnusedRoadRunner),
    )

    result = simulate_with_roadrunner(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert "Antimony parse error" in result["message"]
    assert "syntax error near" in result["message"]


def test_roadrunner_backend_reports_missing_antimony(monkeypatch):
    """Missing Antimony raises a clear dependency error after RoadRunner imports."""
    model_ir = ModelIR()
    monkeypatch.setitem(sys.modules, "roadrunner", SimpleNamespace(RoadRunner=object))
    monkeypatch.setitem(sys.modules, "antimony", None)

    with pytest.raises(ImportError, match="Antimony not installed"):
        simulate_with_roadrunner(model_ir, 0.0, 1.0, 2)


@pytest.mark.parametrize(
    ("module_name", "sbml", "message"),
    [
        ("", "<sbml/>", "Could not get Antimony module name"),
        ("test", "", "Antimony→SBML conversion failed"),
    ],
)
def test_roadrunner_backend_reports_antimony_conversion_failures(
    monkeypatch, module_name, sbml, message
):
    """Empty module names and SBML conversion failures are structured backend failures."""
    model_ir = ModelIR()
    model_ir.antimony_text = "model test() end"

    class FakeAntimony:
        def clearPreviousLoads(self):
            pass

        def loadAntimonyString(self, text):
            return 0

        def getLastError(self):
            return "conversion failed"

        def getMainModuleName(self):
            return module_name

        def getSBMLString(self, name):
            return sbml

    class UnusedRoadRunner:
        def __init__(self, *args, **kwargs):
            raise AssertionError("RoadRunner should not be constructed")

    monkeypatch.setitem(sys.modules, "antimony", FakeAntimony())
    monkeypatch.setitem(sys.modules, "roadrunner", SimpleNamespace(RoadRunner=UnusedRoadRunner))

    result = simulate_with_roadrunner(model_ir, 0.0, 1.0, 2)

    assert result["success"] is False
    assert message in result["message"]


def test_roadrunner_antimony_text_preprocessing_multiline_gamma_and_numeric_model():
    """RoadRunner preprocessing fixes numeric model names, continuations, and gamma constants."""
    model_ir = ModelIR()
    model_ir.antimony_text = """
model 24_decay()
  X' = -k*X // keep the next term
       + gamma(1/2); // comment supplies semicolon
  k = 1;
end
"""

    text = _get_antimony_text(model_ir)

    assert "model m_24_decay()" in text
    assert "X' = -k*X + 1.7724538509055159;" in text
    assert "**" not in text


def test_roadrunner_reconstructs_antimony_when_cached_text_missing():
    """ModelIR reconstruction includes species, parameters, reactions, and explicit rates."""
    model_ir = ModelIR()
    model_ir.species = {"S", "P"}
    model_ir.params = {"k": 0.5}
    model_ir.reactions = [SimpleNamespace(name="J0", lhs=[(1, "S")], rhs=[(1, "P")], rate_expr="k*S")]
    model_ir.explicit_rates = {"S": "-k*S"}

    text = _get_antimony_text(model_ir)

    assert "model recast_model()" in text
    assert "species S;" in text
    assert "k = 0.5;" in text
    assert "J0: S -> P; k*S;" in text
    assert "S' = -k*S;" in text


def test_roadrunner_initial_condition_warnings_cover_backend_failures():
    """Initial condition setup reports floating-species and assignment failures."""
    model_ir = ModelIR()
    model_ir.initials = {"X": 1.0}

    class FailingRoadRunner:
        def getFloatingSpeciesIds(self):
            raise RuntimeError("floating species unavailable")

        def resetToOrigin(self):
            pass

        def __setitem__(self, key, value):
            raise RuntimeError(f"cannot set {key}")

    warnings = _set_initial_conditions(FailingRoadRunner(), model_ir, {"X": 2.0})

    assert warnings[0]["stage"] == "floating_species"
    assert warnings[1]["stage"] == "model_initial"
    assert warnings[2]["stage"] == "override_initial"


def test_roadrunner_gamma_rewrite_accepts_numeric_allowlist():
    """Complete gamma rewrites allow arithmetic, pi, sqrt, and exponentiation only."""
    assert _evaluate_complete_gamma_argument("sqrt(4) + 1") == 3.0

    rewritten = _replace_complete_gamma_calls("k = gamma(1/2) + gamma(sqrt(4));")

    assert "1.7724538509055159" in rewritten
    assert "1.0" in rewritten
    assert "gamma(" not in rewritten


def test_roadrunner_gamma_rewrite_preserves_symbolic_and_incomplete_gamma():
    """Symbolic gamma and incomplete gamma calls are left for Antimony to handle."""
    text = "a = gamma(nu/2); b = gamma(1, 2);"

    assert _replace_complete_gamma_calls(text) == text


@pytest.mark.parametrize(
    "expr",
    [
        "gamma()",
        "gamma(sqrt(-1))",
        "gamma(__import__('os'))",
        "gamma(abs(1))",
        "gamma(1+)",
    ],
)
def test_roadrunner_gamma_rewrite_rejects_malformed_or_unsafe(expr):
    """Malformed complete gamma expressions fail closed with a useful diagnostic."""
    with pytest.raises(ValueError, match="Malformed complete gamma expression"):
        _replace_complete_gamma_calls(f"x = {expr};")


def test_simulate_ode_reports_missing_roadrunner_as_backend_failure(monkeypatch):
    """Missing RoadRunner is reported as a failed ODE backend, not an exception."""
    model_ir = ModelIR()
    model_ir.species = {"X"}
    model_ir.explicit_rates = {"X": "-X"}

    monkeypatch.setitem(sys.modules, "roadrunner", None)

    result = simulate_ode(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert result["backend"] == "roadrunner_cvode"
    assert "libRoadRunner not available" in result["message"]


def test_simulate_model_selects_assignment_rule_backend(monkeypatch):
    """ODE-with-assignment models use the ODE backend with explicit metadata."""
    model_ir = ModelIR()
    model_ir.solver_requirement = SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
    model_ir.assignment_rules = {"A": "S + 1"}

    def fake_simulate_ode(*args, **kwargs):
        return {
            "t": [],
            "y": [],
            "state_names": [],
            "success": True,
            "message": "",
            "integrator_stats": {},
        }

    monkeypatch.setattr("ssys.ode_backends.interface.simulate_ode", fake_simulate_ode)

    result = simulate_model(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is True
    assert result["solver_requirement"] == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES.value
    assert result["backend"] == "roadrunner_cvode_assignment_rules"


def test_dae_required_without_ida_dependency_fails_unsupported(monkeypatch):
    """DAE-required models fail unsupported when optional IDA bindings are absent."""
    model_ir = ModelIR()
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED
    model_ir.algebraic_constraints = ["X - 1"]

    def missing_ida():
        raise IDASundialsUnavailable("scikit-SUNDAE is not installed. uv sync --extra dae")

    monkeypatch.setattr(
        "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
        missing_ida,
    )

    result = simulate_model(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert result["unsupported_solver_requirement"] is True
    assert result["solver_requirement"] == SolverRequirement.DAE_REQUIRED.value
    assert result["backend"] == "ida_sundials"
    assert "uv sync --extra dae" in result["message"]


def test_projection_backend_is_explicit_dae_fallback():
    """The projection backend remains available only when explicitly requested."""
    model_ir = ModelIR()
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED
    model_ir.algebraic_constraints = ["X - 1"]

    result = simulate_model(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        options={"dae_backend": "dae_projection"},
    )

    assert result["success"] is False
    assert result["unsupported_solver_requirement"] is True
    assert result["backend"] == "dae_projection"
    assert "implicit algebraic constraints" in result["message"]


def test_dae_projection_evaluates_state_params_and_time():
    """Projection expressions can depend on states, parameters, and time."""
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([[1.0], [2.0], [3.0]])

    result = _evaluate_expr_over_trajectory(
        "K*X + time",
        t=t,
        y=y,
        state_names=["X"],
        params={"K": 2.0},
    )

    np.testing.assert_allclose(result, [2.0, 5.0, 8.0])


def test_dae_projection_projects_existing_and_new_variables():
    """Projection updates existing variables and appends missing algebraic variables."""
    t = np.array([0.0, 1.0])
    y = np.array([[1.0, 99.0], [2.0, 99.0]])
    state_names = ["X", "Y"]

    y, residual = _project_variable(
        name="Y",
        expr="X + 1",
        t=t,
        y=y,
        state_names=state_names,
        params={},
    )
    y, new_residual = _project_variable(
        name="Z",
        expr="Y + time",
        t=t,
        y=y,
        state_names=state_names,
        params={},
    )

    np.testing.assert_allclose(y[:, 1], [2.0, 3.0])
    np.testing.assert_allclose(y[:, 2], [2.0, 4.0])
    assert residual == pytest.approx(97.0)
    assert new_residual == pytest.approx(0.0)
    assert state_names == ["X", "Y", "Z"]


def test_dae_projection_applies_assignment_rules_and_auxiliary_defs(monkeypatch):
    """Projection backend records residuals for assignment and auxiliary definitions."""
    model_ir = ModelIR()
    model_ir.params = {"K": 1.0}
    model_ir.assignment_rules = {"Y": "X + K"}
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    def fake_roadrunner(*args, **kwargs):
        return {
            "success": True,
            "t": np.array([0.0, 1.0]),
            "y": np.array([[1.0], [2.0]]),
            "state_names": ["X"],
            "message": "",
            "backend": "roadrunner_cvode",
            "integrator_stats": {},
        }

    monkeypatch.setattr("ssys.ode_backends.dae_backend.simulate_with_roadrunner", fake_roadrunner)

    result = simulate_with_dae_projection(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        options={"auxiliary_defs": {"Z": "Y + time"}},
    )

    assert result["success"] is True
    assert result["backend"] == "dae_projection"
    assert result["state_names"] == ["X", "Y", "Z"]
    np.testing.assert_allclose(result["y"], [[1.0, 2.0, 2.0], [2.0, 3.0, 4.0]])
    assert result["algebraic_residuals"]["Y"] == pytest.approx(0.0)
    assert result["algebraic_residuals"]["Z"] == pytest.approx(0.0)


def test_dae_projection_preserves_base_failure_metadata(monkeypatch):
    """Projection backend reclassifies base ODE failures without hiding diagnostics."""
    model_ir = ModelIR()
    model_ir.assignment_rules = {"Y": "X + 1"}
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    def fake_roadrunner(*args, **kwargs):
        return {
            "success": False,
            "message": "base failed",
            "unsupported_solver_requirement": True,
        }

    monkeypatch.setattr("ssys.ode_backends.dae_backend.simulate_with_roadrunner", fake_roadrunner)

    result = simulate_with_dae_projection(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert result["backend"] == "dae_projection"
    assert result["unsupported_solver_requirement"] is True
    assert result["message"] == "base failed"


def test_ida_backend_enforces_explicit_assignment_auxiliary(monkeypatch):
    """IDA residuals include explicit assignment auxiliaries as algebraic states."""
    model_ir = ModelIR()
    model_ir.species = {"X"}
    model_ir.params = {"K": 1.0, "k": 0.5}
    model_ir.initial = {"X": 1.0}
    model_ir.explicit_rates = {"X": "-k*X"}
    model_ir.assignment_rules = {"Y_1": "K + X"}
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    class FakeIDA:
        kwargs = {}
        y0 = None

        def __init__(self, residual, **kwargs):
            self.residual = residual
            FakeIDA.kwargs = kwargs

        def solve(self, t_eval, y0, yp0):
            FakeIDA.y0 = np.asarray(y0, dtype=float)
            t = np.asarray(t_eval, dtype=float)
            x = np.exp(-0.5 * t)
            y = np.column_stack([x, 1.0 + x])
            yp = np.column_stack([-0.5 * x, -0.5 * x])
            residual_at_start = np.asarray(self.residual(t[0], y[0], yp[0]), dtype=float)
            assert abs(residual_at_start[1]) < 1.0e-12
            return {"success": True, "t": t, "y": y, "yp": yp, "status": 0, "message": "ok"}

    monkeypatch.setattr(
        "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
        lambda: SimpleNamespace(
            package="fake-sundials",
            version="1.0",
            solver_class=FakeIDA,
        ),
    )

    result = simulate_model(model_ir, t0=0.0, t_end=1.0, n_points=3)

    assert result["success"] is True
    assert result["backend"] == "ida_sundials"
    assert result["state_names"] == ["X", "Y_1"]
    assert FakeIDA.y0.tolist() == [1.0, 2.0]
    assert FakeIDA.kwargs["algebraic_idx"].tolist() == [1]
    assert result["algebraic_residuals"]["Y_1"] < 1.0e-12
    assert result["integrator_stats"]["package"] == "fake-sundials"
    assert result["integrator_stats"]["package_version"] == "1.0"


def test_ida_backend_enforces_ode_mode_lifted_auxiliary(monkeypatch):
    """ODE-mode lifted auxiliaries are treated as algebraic in IDA validation."""
    model_ir = ModelIR()
    model_ir.species = {"X", "Y_1"}
    model_ir.params = {"K": 1.0}
    model_ir.initial = {"X": 1.0, "Y_1": 2.0}
    model_ir.explicit_rates = {"X": "-X/Y_1", "Y_1": "-X/Y_1"}
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    class FakeIDA:
        kwargs = {}

        def __init__(self, residual, **kwargs):
            self.residual = residual
            FakeIDA.kwargs = kwargs

        def solve(self, t_eval, y0, yp0):
            t = np.asarray(t_eval, dtype=float)
            x = 1.0 / (1.0 + t)
            y = np.column_stack([x, 1.0 + x])
            yp = np.column_stack([-1.0 / (1.0 + t) ** 2, -1.0 / (1.0 + t) ** 2])
            residual_at_start = np.asarray(self.residual(t[0], y[0], yp[0]), dtype=float)
            assert abs(residual_at_start[1]) < 1.0e-12
            return {"success": True, "t": t, "y": y, "yp": yp, "status": 0, "message": "ok"}

    monkeypatch.setattr(
        "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
        lambda: SimpleNamespace(
            package="fake-sundials",
            version="1.0",
            solver_class=FakeIDA,
        ),
    )

    result = simulate_model(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=3,
        options={"auxiliary_defs": {"Y_1": "K + X"}},
    )

    assert result["success"] is True
    assert result["backend"] == "ida_sundials"
    assert FakeIDA.kwargs["algebraic_idx"].tolist() == [1]
    assert result["algebraic_residuals"]["Y_1"] < 1.0e-12


def test_ida_backend_handles_implicit_algebraic_constraint(monkeypatch):
    """IDA residuals include implicit algebraic constraints for algebraic slots."""
    model_ir = ModelIR()
    model_ir.species = {"X", "Z"}
    model_ir.initial = {"X": 1.0, "Z": 1.0}
    model_ir.explicit_rates = {"X": "-X + Z"}
    model_ir.algebraic_constraints = ["Z - X^2"]
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    class FakeIDA:
        kwargs = {}

        def __init__(self, residual, **kwargs):
            self.residual = residual
            FakeIDA.kwargs = kwargs

        def solve(self, t_eval, y0, yp0):
            t = np.asarray(t_eval, dtype=float)
            x = 1.0 / (1.0 + t)
            z = x**2
            y = np.column_stack([x, z])
            yp = np.column_stack([-1.0 / (1.0 + t) ** 2, -2.0 / (1.0 + t) ** 3])
            residual_at_start = np.asarray(self.residual(t[0], y[0], yp[0]), dtype=float)
            assert abs(residual_at_start[1]) < 1.0e-12
            return {"success": True, "t": t, "y": y, "yp": yp, "status": 0, "message": "ok"}

    monkeypatch.setattr(
        "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
        lambda: SimpleNamespace(
            package="fake-sundials",
            version="1.0",
            solver_class=FakeIDA,
        ),
    )

    result = simulate_model(model_ir, t0=0.0, t_end=1.0, n_points=3)

    assert result["success"] is True
    assert result["backend"] == "ida_sundials"
    assert result["state_names"] == ["X", "Z"]
    assert FakeIDA.kwargs["algebraic_idx"].tolist() == [1]
    assert result["algebraic_residuals"]["algebraic_constraint:1"] < 1.0e-12


def test_ida_backend_rejects_inconsistent_user_algebraic_ic(monkeypatch):
    """User-provided algebraic ICs fail closed unless explicit repair is requested."""
    model_ir = ModelIR()
    model_ir.species = {"X"}
    model_ir.params = {"K": 1.0}
    model_ir.initial = {"X": 1.0}
    model_ir.explicit_rates = {"X": "-X"}
    model_ir.assignment_rules = {"Y_1": "K + X"}
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    class FakeIDA:
        def __init__(self, *args, **kwargs):
            raise AssertionError("solver should not run with inconsistent ICs")

    monkeypatch.setattr(
        "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
        lambda: SimpleNamespace(
            package="fake-sundials",
            version="1.0",
            solver_class=FakeIDA,
        ),
    )

    result = simulate_model(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        y0_override={"X": 1.0, "Y_1": 99.0},
    )

    assert result["success"] is False
    assert result["unsupported_solver_requirement"] is False
    assert "repair_consistent_initial_conditions=True" in result["message"]
    assert result["initial_residual_norms"]["Y_1"] > 1.0


def test_ida_backend_repairs_user_algebraic_ic_when_requested(monkeypatch):
    """Explicit repair allows inconsistent assignment-rule IC overrides."""
    model_ir = ModelIR()
    model_ir.species = {"X"}
    model_ir.params = {"K": 1.0}
    model_ir.initial = {"X": 1.0}
    model_ir.explicit_rates = {"X": "-X"}
    model_ir.assignment_rules = {"Y_1": "K + X"}
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED

    class FakeIDA:
        y0 = None

        def __init__(self, residual, **kwargs):
            self.residual = residual

        def solve(self, t_eval, y0, yp0):
            FakeIDA.y0 = np.asarray(y0, dtype=float)
            t = np.asarray(t_eval, dtype=float)
            y = np.tile(FakeIDA.y0, (len(t), 1))
            yp = np.zeros_like(y)
            return {"success": True, "t": t, "y": y, "yp": yp, "status": 0, "message": "ok"}

    monkeypatch.setattr(
        "ssys.ode_backends.ida_sundials_backend._load_ida_binding",
        lambda: SimpleNamespace(
            package="fake-sundials",
            version="1.0",
            solver_class=FakeIDA,
        ),
    )

    result = simulate_model(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        y0_override={"X": 1.0, "Y_1": 99.0},
        options={"repair_consistent_initial_conditions": True},
    )

    assert result["success"] is True
    assert FakeIDA.y0.tolist() == [1.0, 2.0]


def test_simulate_ode_interface():
    """Test that simulate_ode interface works."""
    # Skip if roadrunner not installed
    pytest.importorskip("roadrunner", reason="Requires libRoadRunner installation")

    # Simple exponential decay model
    antimony_text = """
    model exp_decay()
        species S;
        S = 10;
        k = 0.1;

        J0: S -> ; k * S;
    end
    """

    model_ir = parse_antimony(antimony_text)

    result = simulate_ode(model_ir, t0=0.0, t_end=10.0, n_points=11)

    assert result["success"] is True
    assert len(result["t"]) == 11
    assert result["y"].shape[0] == 11
    assert len(result["state_names"]) > 0
    # Check decay: S(t=10) < S(t=0)
    assert result["y"][-1, 0] < result["y"][0, 0]


def test_roadrunner_not_installed_graceful():
    """Test graceful handling when roadrunner not available."""
    antimony_text = """
    model simple()
        species S;
        S = 1;
    end
    """

    model_ir = parse_antimony(antimony_text)

    # Try simulation - will succeed if roadrunner installed, fail gracefully if not
    result = simulate_ode(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert isinstance(result, dict)
    assert "success" in result
    assert "message" in result


def test_roadrunner_exp_decay():
    """Test roadrunner backend on exponential decay."""
    # Skip if roadrunner not installed
    pytest.importorskip("roadrunner", reason="Requires libRoadRunner installation")

    antimony_text = """
    model exp_decay()
        species S;
        S = 10;
        k = 0.1;

        J0: S -> ; k * S;
    end
    """

    model_ir = parse_antimony(antimony_text)

    result = simulate_ode(model_ir, t0=0.0, t_end=10.0, n_points=11)

    if result["success"]:
        # Check basic structure
        assert len(result["t"]) == 11
        assert result["y"].shape[0] == 11
        assert len(result["state_names"]) > 0

        # Check decay behavior (S should decrease)
        assert result["y"][0, 0] > result["y"][-1, 0]
