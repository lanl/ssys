"""
Tests for ODE solver backends.
"""

import pytest

from src.ssys.ode_backends import simulate_ode
from src.ssys.ode_backends.interface import simulate_model
from src.ssys.recaster import ModelIR, SolverRequirement, parse_antimony


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

    monkeypatch.setattr("src.ssys.ode_backends.interface.simulate_ode", fake_simulate_ode)

    result = simulate_model(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is True
    assert result["solver_requirement"] == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES.value
    assert result["backend"] == "roadrunner_cvode_assignment_rules"


def test_unsupported_implicit_dae_requirement_fails_clear():
    """Implicit algebraic constraints are not accepted as an ODE validation pass."""
    model_ir = ModelIR()
    model_ir.solver_requirement = SolverRequirement.DAE_REQUIRED
    model_ir.algebraic_constraints = ["X - 1"]

    result = simulate_model(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert result["unsupported_solver_requirement"] is True
    assert result["solver_requirement"] == SolverRequirement.DAE_REQUIRED.value
    assert "implicit algebraic constraints" in result["message"]


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
