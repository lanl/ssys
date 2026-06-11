"""
Tests for ODE solver backends.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ssys.ode_backends import simulate_ode
from ssys.ode_backends.ida_sundials_backend import IDASundialsUnavailable
from ssys.ode_backends.interface import simulate_model
from ssys.ode_backends.roadrunner_backend import simulate_with_roadrunner
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
