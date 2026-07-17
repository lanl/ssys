"""
Tests for ODE solver backends.
"""

import re
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from ssys.ode_backends import simulate_ode
from ssys.ode_backends.dae_backend import (
    _evaluate_expr_over_trajectory,
    _project_variable,
    simulate_with_dae_projection,
)
from ssys.ode_backends.ida_sundials_backend import IDASundialsUnavailable
from ssys.ode_backends.interface import (
    _infer_solver_requirement,
    simulate_dae,
    simulate_dae_projection,
    simulate_model,
)
from ssys.ode_backends.roadrunner_backend import (
    _evaluate_complete_gamma_argument,
    _get_antimony_text,
    _replace_complete_gamma_calls,
    _set_initial_conditions,
    simulate_with_roadrunner,
)
from ssys.recaster import SolverRequirement, SymSystem, parse_antimony_via_sbml


def _sympify_rate(expr, symbols):
    """Sympify an ODE right-hand side with explicit symbol locals.

    Passing locals keeps single-letter variable names such as ``S`` from
    resolving to SymPy singletons (``sympy.S``), matching how the parser and the
    backends build expressions.
    """
    if isinstance(expr, sp.Expr):
        return expr
    text = str(expr).replace("^", "**")
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    locals_map = {name: symbols.get(name, sp.Symbol(name)) for name in names}
    return sp.sympify(text, locals=locals_map)


def _sym_system(
    *,
    state_vars=(),
    params=None,
    odes=None,
    initials=None,
    assignment_rules=None,
    algebraic_constraints=(),
    antimony_text="",
    solver_requirement=SolverRequirement.ODE_ONLY,
):
    """Build a SymSystem fixture from convenient string-keyed inputs.

    ``state_vars``/``odes``/``initials`` are given by variable name; symbols and
    SymPy expressions are constructed here so the fixture mirrors what the parser
    produces. ``odes`` values may be strings (``^`` is normalized to ``**``).
    """
    odes = dict(odes or {})
    initials = dict(initials or {})
    names = sorted(dict.fromkeys([*state_vars, *odes, *initials]))
    symbols = {name: sp.Symbol(name) for name in names}
    return SymSystem(
        vars=list(symbols.values()),
        params=dict(params or {}),
        odes={symbols[name]: _sympify_rate(expr, symbols) for name, expr in odes.items()},
        initials={symbols[name]: value for name, value in initials.items()},
        assignment_rules=dict(assignment_rules or {}),
        algebraic_constraints=list(algebraic_constraints),
        antimony_text=antimony_text,
        solver_requirement=solver_requirement,
    )


def test_roadrunner_backend_success_and_initial_condition_override(monkeypatch):
    """RoadRunner backend parses, simulates, and applies state IC overrides."""
    model_ir = _sym_system(
        state_vars=["X"],
        odes={"X": "-k*X"},
        params={"k": 0.5},
        antimony_text="""
        model test()
            species X;
            X' = -k*X;
            k = 0.5;
            X = 1;
        end
    """,
    )
    # RoadRunner's IC setter tolerates string keys, non-species params, and tuple
    # compartment metadata; keep that mix to exercise all three skip branches.
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


def test_roadrunner_backend_non_cvode_records_initial_condition_warnings(monkeypatch, capsys):
    """Non-CVODE runs skip CVODE stats and still surface IC assignment warnings."""
    model_ir = _sym_system(
        state_vars=["X"],
        odes={"X": "-X"},
        initials={"X": 1.0},
        antimony_text="""
        model test()
            species X;
            X' = -X;
            X = 1;
        end
    """,
    )

    class FakeAntimony:
        def clearPreviousLoads(self):
            pass

        def loadAntimonyString(self, text):
            return 0

        def getLastError(self):
            return ""

        def getMainModuleName(self):
            return "test"

        def getSBMLString(self, model_name):
            return "<sbml/>"

    class FakeSimulationResult:
        colnames = ["time", "X"]

        def __init__(self):
            self.data = np.array([[0.0, 1.0], [1.0, 0.5]])

        def __getitem__(self, key):
            return self.data[key]

    class FakeRoadRunner:
        def __init__(self, sbml):
            self.integrator = SimpleNamespace()

        def setIntegrator(self, name):
            self.integrator_name = name

        def resetToOrigin(self):
            pass

        def getFloatingSpeciesIds(self):
            return ["X"]

        def __setitem__(self, key, value):
            raise RuntimeError(f"{key} is read-only")

        def simulate(self, t0, t_end, n_points):
            return FakeSimulationResult()

    monkeypatch.setitem(sys.modules, "antimony", FakeAntimony())
    monkeypatch.setitem(sys.modules, "roadrunner", SimpleNamespace(RoadRunner=FakeRoadRunner))

    result = simulate_with_roadrunner(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        options={"integrator": "gillespie", "log_solver_details": True},
    )

    assert result["success"] is True
    assert result["state_names"] == ["X"]
    assert "n_steps" not in result["integrator_stats"]
    assert result["integrator_stats"]["initial_condition_warnings"] == [
        {
            "stage": "model_initial",
            "species": "X",
            "message": "[X] is read-only",
        }
    ]
    assert "Integrator: gillespie" in capsys.readouterr().out


def test_roadrunner_backend_reports_antimony_parser_failure(monkeypatch):
    """Antimony parser failures are returned as structured backend failures."""
    model_ir = _sym_system(
        state_vars=["X"], odes={"X": "-k*X"}, antimony_text="model bad("
    )

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
    model_ir = _sym_system()
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
    model_ir = _sym_system(antimony_text="model test() end")

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
    model_ir = _sym_system(
        antimony_text="""
model 24_decay()
  X' = -k*X // keep the next term
       + gamma(1/2); // comment supplies semicolon
  k = 1;
end
"""
    )

    text = _get_antimony_text(model_ir)

    assert "model m_24_decay()" in text
    assert "X' = -k*X + 1.7724538509055159;" in text
    assert "**" not in text


def test_roadrunner_reconstructs_antimony_when_cached_text_missing():
    """SymSystem reconstruction includes species, parameters, and rate rules."""
    model_ir = _sym_system(
        state_vars=["S", "P"],
        params={"k": 0.5},
        odes={"S": "-k*S"},
    )

    text = _get_antimony_text(model_ir)

    assert "model recast_model()" in text
    assert "species S;" in text
    assert "species P;" in text
    assert "k = 0.5;" in text
    assert "S' = -S*k;" in text


def test_roadrunner_initial_condition_warnings_cover_backend_failures():
    """Initial condition setup reports floating-species and assignment failures."""
    model_ir = _sym_system(state_vars=["X"], initials={"X": 1.0})

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
    model_ir = _sym_system(state_vars=["X"], odes={"X": "-X"})

    monkeypatch.setitem(sys.modules, "roadrunner", None)

    result = simulate_ode(model_ir, t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert result["backend"] == "roadrunner_cvode"
    assert "libRoadRunner not available" in result["message"]


def test_solver_requirement_inference_uses_metadata_then_model_structure():
    """Backend selection inference follows metadata, DAE, assignment, then ODE-only order."""
    configured = _sym_system(solver_requirement=SolverRequirement.DAE_REQUIRED.value)
    algebraic = _sym_system(solver_requirement=None, algebraic_constraints=["X - 1"])
    assigned = _sym_system(solver_requirement=None, assignment_rules={"Y": "X + 1"})
    ode_only = _sym_system(solver_requirement=None)

    assert _infer_solver_requirement(configured) == SolverRequirement.DAE_REQUIRED
    assert _infer_solver_requirement(algebraic) == SolverRequirement.DAE_REQUIRED
    assert _infer_solver_requirement(assigned) == SolverRequirement.ODE_WITH_ASSIGNMENT_RULES
    assert _infer_solver_requirement(ode_only) == SolverRequirement.ODE_ONLY


def test_simulate_ode_passes_default_options_to_roadrunner(monkeypatch):
    """simulate_ode normalizes missing options and annotates successful ODE results."""
    model_ir = _sym_system()
    captured = {}

    def fake_roadrunner(model, t0, t_end, n_points, y0_override, options):
        captured.update({
            "model": model,
            "t0": t0,
            "t_end": t_end,
            "n_points": n_points,
            "y0_override": y0_override,
            "options": options,
        })
        return {
            "success": True,
            "t": np.array([0.0]),
            "y": np.array([[1.0]]),
            "state_names": ["X"],
            "message": "",
        }

    monkeypatch.setattr("ssys.ode_backends.roadrunner_backend.simulate_with_roadrunner", fake_roadrunner)

    result = simulate_ode(model_ir, t0=0.0, t_end=2.0, n_points=3)

    assert captured == {
        "model": model_ir,
        "t0": 0.0,
        "t_end": 2.0,
        "n_points": 3,
        "y0_override": None,
        "options": {},
    }
    assert result["success"] is True
    assert result["backend"] == "roadrunner_cvode"
    assert result["solver_requirement"] == SolverRequirement.ODE_ONLY.value
    assert result["unsupported_solver_requirement"] is False


def test_simulate_dae_rejects_unknown_backend():
    """Unknown DAE backend names are structured unsupported results."""
    result = simulate_dae(
        _sym_system(),
        t0=0.0,
        t_end=1.0,
        n_points=2,
        options={"backend": "not-a-backend"},
    )

    assert result["success"] is False
    assert result["unsupported_solver_requirement"] is True
    assert result["backend"] == "not-a-backend"
    assert "Unsupported DAE backend" in result["message"]


def test_simulate_dae_projection_reports_import_failure(monkeypatch):
    """Projection backend import failures are unsupported solver results."""
    monkeypatch.setitem(sys.modules, "ssys.ode_backends.dae_backend", None)

    result = simulate_dae_projection(_sym_system(), t0=0.0, t_end=1.0, n_points=2)

    assert result["success"] is False
    assert result["unsupported_solver_requirement"] is True
    assert result["backend"] == "dae_projection"
    assert "DAE projection backend unavailable" in result["message"]


def test_simulate_model_explicit_ode_only_override_ignores_dae_metadata(monkeypatch):
    """An explicit ODE-only request overrides model-level DAE metadata."""
    model_ir = _sym_system(algebraic_constraints=["X - 1"])

    def fake_simulate_ode(*args, **kwargs):
        return {
            "success": True,
            "t": np.array([0.0]),
            "y": np.array([[1.0]]),
            "state_names": ["X"],
            "message": "",
            "backend": "roadrunner_cvode",
            "solver_requirement": SolverRequirement.ODE_ONLY.value,
        }

    monkeypatch.setattr("ssys.ode_backends.interface.simulate_ode", fake_simulate_ode)

    result = simulate_model(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        solver_requirement=SolverRequirement.ODE_ONLY,
    )

    assert result["success"] is True
    assert result["backend"] == "roadrunner_cvode"
    assert result["solver_requirement"] == SolverRequirement.ODE_ONLY.value


def test_simulate_model_selects_assignment_rule_backend(monkeypatch):
    """ODE-with-assignment models use the ODE backend with explicit metadata."""
    model_ir = _sym_system(
        solver_requirement=SolverRequirement.ODE_WITH_ASSIGNMENT_RULES,
        assignment_rules={"A": "S + 1"},
    )

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
    model_ir = _sym_system(
        solver_requirement=SolverRequirement.DAE_REQUIRED,
        algebraic_constraints=["X - 1"],
    )

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
    model_ir = _sym_system(
        solver_requirement=SolverRequirement.DAE_REQUIRED,
        algebraic_constraints=["X - 1"],
    )

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


def test_dae_projection_evaluates_constants_symbols_and_scalar_broadcasts():
    """Projection expression evaluation broadcasts constants and parameter-only values."""
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([[1.0], [2.0], [3.0]])
    X = sp.Symbol("X")

    constant = _evaluate_expr_over_trajectory(
        "2.5",
        t=t,
        y=y,
        state_names=["X"],
        params={},
    )
    symbolic = _evaluate_expr_over_trajectory(
        X + 1,
        t=t,
        y=y,
        state_names=["X"],
        params={},
    )
    param_only = _evaluate_expr_over_trajectory(
        "K + 1",
        t=t,
        y=y,
        state_names=["X"],
        params={"K": 2.0},
    )

    np.testing.assert_allclose(constant, [2.5, 2.5, 2.5])
    np.testing.assert_allclose(symbolic, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(param_only, [3.0, 3.0, 3.0])


def test_dae_projection_reports_missing_expression_symbol():
    """Unknown algebraic symbols fail with the missing symbol name."""
    with pytest.raises(ValueError, match="missing symbol 'Y'"):
        _evaluate_expr_over_trajectory(
            "X + Y",
            t=np.array([0.0]),
            y=np.array([[1.0]]),
            state_names=["X"],
            params={},
        )


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


def test_dae_projection_skips_auxiliary_already_covered_by_assignment_rule(monkeypatch):
    """Assignment rules own duplicate auxiliary names during projection."""
    model_ir = _sym_system(
        assignment_rules={"Y": "X + 1"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
        options={"auxiliary_defs": {"Y": "X + 99"}},
    )

    assert result["success"] is True
    assert result["state_names"] == ["X", "Y"]
    np.testing.assert_allclose(result["y"][:, 1], [2.0, 3.0])
    assert result["algebraic_residuals"] == {"Y": 0.0}


def test_dae_projection_applies_assignment_rules_and_auxiliary_defs(monkeypatch):
    """Projection backend records residuals for assignment and auxiliary definitions."""
    model_ir = _sym_system(
        params={"K": 1.0},
        assignment_rules={"Y": "X + K"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
    model_ir = _sym_system(
        assignment_rules={"Y": "X + 1"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
    model_ir = _sym_system(
        state_vars=["X"],
        params={"K": 1.0, "k": 0.5},
        initials={"X": 1.0},
        odes={"X": "-k*X"},
        assignment_rules={"Y_1": "K + X"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
    model_ir = _sym_system(
        state_vars=["X", "Y_1"],
        params={"K": 1.0},
        initials={"X": 1.0, "Y_1": 2.0},
        odes={"X": "-X/Y_1", "Y_1": "-X/Y_1"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
    model_ir = _sym_system(
        state_vars=["X", "Z"],
        initials={"X": 1.0, "Z": 1.0},
        odes={"X": "-X + Z"},
        algebraic_constraints=["Z - X^2"],
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
    model_ir = _sym_system(
        state_vars=["X"],
        params={"K": 1.0},
        initials={"X": 1.0},
        odes={"X": "-X"},
        assignment_rules={"Y_1": "K + X"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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
    model_ir = _sym_system(
        state_vars=["X"],
        params={"K": 1.0},
        initials={"X": 1.0},
        odes={"X": "-X"},
        assignment_rules={"Y_1": "K + X"},
        solver_requirement=SolverRequirement.DAE_REQUIRED,
    )

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


def test_ida_private_helpers_cover_error_and_name_branches(monkeypatch):
    import ssys.ode_backends.ida_sundials_backend as ida

    compiled = ida._compile_expr("X + k", {"X", "k"})
    with pytest.raises(ValueError, match="missing value for symbol 'k'"):
        compiled.evaluate({"X": 1.0})

    assert ida._as_float(object(), default=2.5) == 2.5
    assert ida._is_zero_expr(object()) is False
    assert ida._sympify(sp.Symbol("X")) == sp.Symbol("X")
    assert ida._definition_mentions_state(object(), {"X"}) is True

    model_ir = SimpleNamespace(vars=[sp.Symbol("A")])
    names = ida._model_variable_names(
        model_ir,
        {"C": "1"},
        {"D": "A"},
        {"E": "A"},
        ["F + G + k + time + sin(H)"],
        {"k": 1.0},
    )

    assert names == ["A", "C", "D", "E", "F", "G", "H"]

    assert ida._ode_expressions(SimpleNamespace(odes={sp.Symbol("X"): "-X"})) == {"X": "-X"}
    assert ida._ode_expressions(SimpleNamespace(odes={})) == {}


def test_ida_binding_loader_reports_missing_and_unknown_version(monkeypatch):
    import ssys.ode_backends.ida_sundials_backend as ida

    monkeypatch.setitem(sys.modules, "sksundae", None)
    monkeypatch.setitem(sys.modules, "sksundae.ida", None)
    with pytest.raises(IDASundialsUnavailable, match="uv sync --extra dae"):
        ida._load_ida_binding()

    fake_package = types.ModuleType("sksundae")
    fake_ida_module = types.ModuleType("sksundae.ida")

    class FakeIDA:
        pass

    fake_ida_module.IDA = FakeIDA
    monkeypatch.setitem(sys.modules, "sksundae", fake_package)
    monkeypatch.setitem(sys.modules, "sksundae.ida", fake_ida_module)

    def missing_version(distribution_name):
        assert distribution_name == "scikit-sundae"
        raise ida.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(ida.metadata, "version", missing_version)

    binding = ida._load_ida_binding()

    assert binding.package == "scikit-sundae"
    assert binding.version == "unknown"
    assert binding.solver_class is FakeIDA


def test_ida_implicit_slot_selection_branches():
    import ssys.ode_backends.ida_sundials_backend as ida

    assert ida._select_implicit_slots(["X"], set(), {"X": "-X"}, [], {}) == []

    with pytest.raises(ida.UnsupportedDAESystem, match="one algebraic variable"):
        ida._select_implicit_slots(["X"], set(), {"X": "-X"}, ["Z - X"], {})

    selected = ida._select_implicit_slots(
        ["X", "Z", "W"],
        {"W"},
        {"X": "-X", "Z": "0", "W": "0"},
        ["Z - X"],
        {},
    )
    assert selected == ["Z"]

    preferred = ida._select_implicit_slots(
        ["X", "Z", "W"],
        set(),
        {"X": "-X", "Z": "0", "W": "0"},
        ["W - X"],
        {},
    )
    assert preferred == ["W"]


def test_ida_residual_system_helpers_cover_output_and_reconstructed_ydot():
    import ssys.ode_backends.ida_sundials_backend as ida

    model_ir = _sym_system(
        state_vars=["X"],
        params={"K": 1.0},
        initials={"X": 1.0},
        odes={"X": "-X"},
        assignment_rules={"Y": "X + K"},
    )

    system = ida._build_residual_system(model_ir, None, {})
    residual = ida._make_sksundae_residual(system)
    out = np.empty(len(system.equations), dtype=float)

    returned = residual(0.0, system.y0, system.ydot0, out)

    assert returned is None
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1.0e-12)

    t = np.array([0.0, 1.0])
    y = np.array([[1.0, 2.0], [0.5, 1.5]])
    residuals = ida._trajectory_algebraic_residuals(system, t, y, ydot=None)

    assert residuals["Y"] == pytest.approx(0.0)

    ode_only = ida._build_residual_system(
        _sym_system(state_vars=["X"], initials={"X": 1.0}, odes={"X": "-X"}),
        None,
        {},
    )
    assert ida._trajectory_algebraic_residuals(ode_only, t, y[:, :1], ydot=None) == {}


def test_ida_solution_array_extraction_accepts_supported_shapes_and_rejects_bad_shapes():
    import ssys.ode_backends.ida_sundials_backend as ida

    object_solution = SimpleNamespace(
        success=False,
        t=np.array([0.0, 1.0]),
        y=np.array([[1.0], [0.5]]),
        yp=np.array([[-1.0], [-0.5]]),
        status=-1,
        message="failed",
    )
    success, t, y, ydot, status, message = ida._extract_solution_arrays(object_solution)

    assert success is False
    assert status == -1
    assert message == "failed"
    np.testing.assert_allclose(t, np.array([0.0, 1.0]))
    np.testing.assert_allclose(y, np.array([[1.0], [0.5]]))
    np.testing.assert_allclose(ydot, np.array([[-1.0], [-0.5]]))

    values_solution = SimpleNamespace(
        values=SimpleNamespace(
            t=np.array([0.0, 1.0]),
            y=np.array([[1.0, 0.5]]),
            ydot=np.array([[-1.0, -0.5]]),
        ),
        flag=0,
        message="ok",
    )
    success, t, y, ydot, status, message = ida._extract_solution_arrays(values_solution)

    assert success is True
    assert status == 0
    assert message == "ok"
    np.testing.assert_allclose(y, np.array([[1.0], [0.5]]))
    np.testing.assert_allclose(ydot, np.array([[-1.0], [-0.5]]))

    with pytest.raises(RuntimeError, match="Unsupported IDA solution object"):
        ida._extract_solution_arrays(object())
    with pytest.raises(RuntimeError, match="expected a 2-D array"):
        ida._extract_solution_arrays({"t": [0.0], "y": [1.0]})
    with pytest.raises(RuntimeError, match="expected a 1-D array"):
        ida._extract_solution_arrays({"t": [[0.0]], "y": [[1.0]]})
    with pytest.raises(RuntimeError, match="incompatible time/state shapes"):
        ida._extract_solution_arrays({"t": [0.0, 1.0], "y": np.ones((3, 1))})
    with pytest.raises(RuntimeError, match="ydot with shape"):
        ida._extract_solution_arrays({"t": [0.0], "y": [[1.0]], "yp": [0.0]})


def test_ida_simulate_failure_paths_and_solver_options(monkeypatch):
    import ssys.ode_backends.ida_sundials_backend as ida

    def binding_for(solver_class):
        return SimpleNamespace(package="fake-sundials", version="1.0", solver_class=solver_class)

    class UnusedIDA:
        def __init__(self, *args, **kwargs):
            raise AssertionError("solver should not be constructed")

    monkeypatch.setattr(ida, "_load_ida_binding", lambda: binding_for(UnusedIDA))
    unsupported = ida.simulate_with_ida_sundials(_sym_system(), 0.0, 1.0, 2)

    assert unsupported["success"] is False
    assert unsupported["unsupported_solver_requirement"] is True
    assert "no model variables" in unsupported["message"]

    real_build_residual_system = ida._build_residual_system

    def bad_setup(*args, **kwargs):
        raise RuntimeError("bad setup")

    monkeypatch.setattr(ida, "_build_residual_system", bad_setup)
    setup_failed = ida.simulate_with_ida_sundials(_sym_system(), 0.0, 1.0, 2)

    assert setup_failed["success"] is False
    assert setup_failed["unsupported_solver_requirement"] is False
    assert "bad setup" in setup_failed["message"]

    monkeypatch.setattr(ida, "_build_residual_system", real_build_residual_system)

    model_ir = _sym_system(state_vars=["X"], initials={"X": 1.0}, odes={"X": "-X"})

    class FailingIDA:
        kwargs = {}

        def __init__(self, residual, **kwargs):
            FailingIDA.kwargs = kwargs

        def solve(self, t_eval, y0, yp0):
            return {"success": False, "t": t_eval, "y": np.array([[1.0], [0.5]]), "message": "bad"}

    monkeypatch.setattr(ida, "_load_ida_binding", lambda: binding_for(FailingIDA))
    failed = ida.simulate_with_ida_sundials(
        model_ir,
        0.0,
        1.0,
        2,
        options={"max_num_steps": 123},
    )

    assert failed["success"] is False
    assert "solver failed" in failed["message"]
    assert FailingIDA.kwargs["max_num_steps"] == 123

    class CrashingIDA:
        def __init__(self, residual, **kwargs):
            pass

        def solve(self, t_eval, y0, yp0):
            raise RuntimeError("linear solver failed")

    monkeypatch.setattr(ida, "_load_ida_binding", lambda: binding_for(CrashingIDA))
    crashed = ida.simulate_with_ida_sundials(model_ir, 0.0, 1.0, 2)

    assert crashed["success"] is False
    assert "linear solver failed" in crashed["message"]
    assert crashed["integrator_stats"]["solver_diagnostics"] == "linear solver failed"


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

    model_ir = parse_antimony_via_sbml(antimony_text)

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

    model_ir = parse_antimony_via_sbml(antimony_text)

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

    model_ir = parse_antimony_via_sbml(antimony_text)

    result = simulate_ode(model_ir, t0=0.0, t_end=10.0, n_points=11)

    if result["success"]:
        # Check basic structure
        assert len(result["t"]) == 11
        assert result["y"].shape[0] == 11
        assert len(result["state_names"]) > 0

        # Check decay behavior (S should decrease)
        assert result["y"][0, 0] > result["y"][-1, 0]
