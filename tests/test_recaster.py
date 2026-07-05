"""Tests for core recasting functionality."""

import re
import sys
from pathlib import Path

import pytest
import sympy as sp

import ssys
from ssys import (
    build_sym_system,
    parse_antimony,
    recast_to_ssystem,
    ssystem_to_antimony,
)
from ssys._recaster.templates import (
    _expand_function_calls,
    _find_next_template_call,
    _parse_function_args,
    _substitute_template_params,
    expand_antimony_function_templates,
)


def _minimal_sbml(
    *,
    species: str,
    compartments: str | None = None,
    function_definitions: str = "",
    parameters: str = "",
    reactions: str = "",
    rules: str = "",
    initial_assignments: str = "",
) -> str:
    compartment_block = compartments if compartments is not None else """
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" units="litre" constant="true"/>
    </listOfCompartments>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m" substanceUnits="mole" timeUnits="second" extentUnits="mole">
{compartment_block}
{function_definitions}
    <listOfSpecies>
{species}
    </listOfSpecies>
{parameters}
{reactions}
{rules}
{initial_assignments}
  </model>
</sbml>"""


class TestAntimonyParser:
    """Tests for Antimony parsing."""

    def test_parse_simple_reaction(self):
        """Test parsing a simple reaction."""
        text = """
        model simple
        A -> B; k*A
        k = 0.1
        A = 10
        B = 0
        end
        """
        ir = parse_antimony(text)
        assert "A" in ir.species
        assert "B" in ir.species
        assert "k" in ir.params
        assert len(ir.reactions) == 1

    def test_parse_explicit_ode(self):
        """Test parsing explicit ODE (X' = ...)."""
        text = """
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        ir = parse_antimony(text)
        assert "X" in ir.species
        assert "X" in ir.explicit_rates
        assert ir.explicit_rates["X"] == "-k*X"

    def test_legacy_parser_records_boundary_const_algebraic_and_symbolic_initials(self):
        """Small local parser constructs should preserve their diagnostic metadata."""
        from ssys.recaster import SolverRequirement

        text = """
        model parser_edges()
          J0: 2 S -> P; k*S
          $S' = -k*S
          const k = 0.5
          $B = 2.0
          0 = P - S
          X = k + 1
          obs := X + time
        end
        """

        ir = parse_antimony(text)

        assert ir.reactions[0].lhs == [(2, "S")]
        assert ir.boundary == {"S", "B"}
        assert ir.explicit_rates["S"] == "-k*S"
        assert ir.params["k"] == 0.5
        assert ir.algebraic_constraints == ["P - S"]
        assert ir.solver_requirement == SolverRequirement.DAE_REQUIRED
        assert ir.initial_exprs["X"] == "k + 1"
        assert ir.assignment_rules == {"obs": "X + time"}

    def test_parser_helper_branches_are_structured(self):
        """Tokenization, substitutions, and SBML helper branches have stable contracts."""
        from ssys._recaster.parsing import (
            _evaluate_initial_assignment,
            _iter_kinetic_law_local_parameters,
            _numeric_param_subs,
            _replace_formula_identifiers,
            _sanitize_sbml_identifier,
            _sympify_sbml_formula,
            _unique_identifier,
        )
        from ssys.recaster import SBMLParseError

        assert parse_antimony("$A + two B").reactions == []
        assert ssys.recaster.tokenize_species_side("$A + two B") == [
            (1, "A"),
            (1, "two B"),
        ]

        X, k = sp.symbols("X k")
        assert _numeric_param_subs(X + k, {}) == X + k
        assert sp.simplify(_numeric_param_subs(X + k, {"k": 2.0}) - (X + 2.0)) == 0
        assert _replace_formula_identifiers("k + kk + k_1", {"": "skip", "k": "J__k"}) == (
            "J__k + kk + k_1"
        )
        assert _sanitize_sbml_identifier(" 1 bad-id! ", fallback="fallback") == "_1_bad_id"
        assert _sanitize_sbml_identifier(" !!! ", fallback="fallback") == "fallback"
        assert _unique_identifier("k", {"k", "k_2"}) == "k_3"

        class LocalKineticLaw:
            def getNumLocalParameters(self):
                return 1

            def getLocalParameter(self, idx):
                assert idx == 0
                return "local"

            def getNumParameters(self):
                raise AssertionError("legacy parameter path should not run")

        class LegacyKineticLaw:
            def getNumParameters(self):
                return 1

            def getParameter(self, idx):
                assert idx == 0
                return "legacy"

        assert _iter_kinetic_law_local_parameters(LocalKineticLaw()) == ["local"]
        assert _iter_kinetic_law_local_parameters(LegacyKineticLaw()) == ["legacy"]

        all_syms = {"X": X, "k": k}
        with pytest.raises(SBMLParseError, match="missing math formula"):
            _sympify_sbml_formula(None, all_syms, source="inline", kind="rate_rule")
        with pytest.raises(SBMLParseError, match="unsupported function"):
            _sympify_sbml_formula("f(X)", all_syms, source="inline", kind="rate_rule")
        with pytest.raises(SBMLParseError, match="unknown identifier"):
            _sympify_sbml_formula("X + missing", all_syms, source="inline", kind="rate_rule")
        with pytest.raises(SBMLParseError, match="expected numeric expression"):
            _sympify_sbml_formula("[1, 2]", all_syms, source="inline", kind="rate_rule")

        scoped = sp.Symbol("reaction_10__k1")
        scoped_expr = _sympify_sbml_formula(
            "reaction_10__k1 * 5.9e-4 * X",
            {"X": X, "reaction_10__k1": scoped},
            source="inline",
            kind="rate_rule",
        )
        assert sp.simplify(scoped_expr - scoped * sp.Float("5.9e-4") * X) == 0

        assert _evaluate_initial_assignment(
            "k + 1",
            all_syms,
            {"k": 3.0},
            source="inline",
            variable="X",
            warn_initial_assignment_failures=False,
        ) == 4.0
        with pytest.warns(RuntimeWarning, match="initial assignment"):
            assert _evaluate_initial_assignment(
                "X + missing",
                all_syms,
                {},
                source="inline",
                variable="X",
                warn_initial_assignment_failures=True,
            ) is None

    def test_antimony_sbml_bridge_reports_library_failures(self, monkeypatch):
        """Antimony conversion failures are classified before SBML parsing."""
        from ssys.recaster import parse_antimony_via_sbml

        class ParseFailureAntimony:
            def clearPreviousLoads(self):
                pass

            def loadAntimonyString(self, text):
                return -1

            def getLastError(self):
                return "bad syntax"

        monkeypatch.setitem(sys.modules, "antimony", ParseFailureAntimony())
        with pytest.raises(ValueError, match="Antimony parsing error: bad syntax"):
            parse_antimony_via_sbml("not antimony")

        class NoModuleAntimony(ParseFailureAntimony):
            def loadAntimonyString(self, text):
                return 0

            def getMainModuleName(self):
                return ""

        monkeypatch.setitem(sys.modules, "antimony", NoModuleAntimony())
        with pytest.raises(ValueError, match="Antimony library failed: No module found"):
            parse_antimony_via_sbml("model m() end")

        class NoSbmlAntimony(NoModuleAntimony):
            def getMainModuleName(self):
                return "m"

            def getSBMLString(self, module_name):
                assert module_name == "m"
                return ""

            def getLastError(self):
                return "conversion failed"

        monkeypatch.setitem(sys.modules, "antimony", NoSbmlAntimony())
        with pytest.raises(ValueError, match="SBML conversion failed: conversion failed"):
            parse_antimony_via_sbml("model m() end")

    def test_parse_antimony_via_sbml_repairs_empty_compartment_initializer(self):
        """libAntimony can export an SBML compartment named compartment as unset."""
        from ssys.recaster import parse_antimony_via_sbml

        text = """
        model exported_compartment_keyword()
          species S, P;
          J0: S -> P; compartment_*k*S;
          S = 1;
          P = 0;
          k = 2;
          compartment_ = ;
        end
        """

        sym = parse_antimony_via_sbml(text)

        S = sp.Symbol("S", positive=True)
        P = sp.Symbol("P", positive=True)
        k = sp.Symbol("k", positive=True)
        compartment = sp.Symbol("compartment_", positive=True)
        expected = compartment * k * S
        assert sym.params["compartment_"] == 1.0
        assert sp.simplify(sym.odes[S] + expected) == 0
        assert sp.simplify(sym.odes[P] - expected) == 0

    def test_function_template_substitution_parenthesizes_expression_arguments(self):
        """Regression: f(x+1) must not become X + 1/(X + 2)."""
        text = """
        f(x) := x/(1+x)
        g(x) := f(x+1)
        X' = g(X)
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)

        X = sp.Symbol("X", positive=True)
        expected = (X + 1) / (X + 2)

        assert sp.simplify(sym.odes[X] - expected) == 0

    def test_nested_multi_argument_function_templates_still_expand(self):
        """Nested calls and multi-argument templates use the shared expander."""
        text = """
        f(x, y) := x/(1+y)
        g(x) := f(x+1, x*2)
        X' = g(X)
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)

        X = sp.Symbol("X", positive=True)
        expected = (X + 1) / (1 + 2 * X)

        assert sp.simplify(sym.odes[X] - expected) == 0

    def test_roadrunner_preprocessor_uses_shared_function_expansion(self):
        """RoadRunner backend expands legacy templates before Antimony parsing."""
        import antimony

        from ssys.ode_backends.roadrunner_backend import _expand_parametric_functions

        text = """
        model ftest
          f(x) := x/(1+x)
          g(x) := f(x+1)
          X' = g(X)
          X = 1
        end
        """
        expanded = _expand_parametric_functions(text)

        assert ":=" not in expanded
        antimony.clearPreviousLoads()
        assert antimony.loadAntimonyString(expanded) >= 0, antimony.getLastError()

    def test_function_template_helpers_reject_non_expandable_calls(self):
        """Template call discovery skips unknown, unbalanced, and wrong-arity calls."""
        templates = {"f": (["x"], "x + 1")}

        assert _find_next_template_call("g(X)", templates) is None
        assert _find_next_template_call("f(X", templates) is None
        assert _find_next_template_call("f(X, Y)", templates) is None
        assert _expand_function_calls("f(X)", templates, max_depth=0) == "f(X)"
        assert _expand_function_calls("f(X)", {}) == "f(X)"

    def test_function_template_argument_parser_handles_empty_and_nested_args(self):
        """Argument splitting preserves nested commas and drops empty argument lists."""
        assert _parse_function_args("") == []
        assert _parse_function_args("X, (A + B), f(Y, Z)") == [
            "X",
            "(A + B)",
            "f(Y, Z)",
        ]

    def test_function_template_substitution_is_simultaneous(self):
        """Inserted arguments must not be rewritten by later parameter substitutions."""
        body = "(kdsic_ + kdsic * Clb) * Sic1t"

        assert (
            _substitute_template_params(
                body,
                ["kdsic_", "kdsic", "Clb", "Sic1t"],
                ["kdsic", "kdsic_0", "Clb", "Sic1t"],
            )
            == "(kdsic + kdsic_0 * Clb) * Sic1t"
        )

    def test_function_template_expansion_preserves_comments_and_skips_model_boundaries(self):
        """Executable lines expand while model/end boundaries and comments are preserved."""
        text = """
        model templated()
          f(x) := x + 1 // keep template comment
          X' = f(X);
          note = f(2);
        end
        """

        expanded = expand_antimony_function_templates(text)

        assert "f(x) :=" not in expanded
        assert "// keep template comment" in expanded
        assert "model templated()" in expanded
        assert "X' = (X + 1);" in expanded
        assert "note = (2 + 1);" in expanded
        assert expanded.strip().endswith("end")

    def test_function_template_expansion_noops_without_templates(self):
        """Models without function-template definitions are returned unchanged."""
        text = "model plain()\n  X' = -X;\nend"

        assert expand_antimony_function_templates(text) == text


class TestSymbolicSystem:
    """Tests for symbolic ODE system building."""

    def test_build_system_from_reaction(self):
        """Test building symbolic system from reaction."""
        text = """
        A -> B; k*A
        k = 0.1
        A = 10.0
        B = 0.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)

        assert len(sym.vars) == 2
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        assert A in sym.odes
        assert B in sym.odes
        assert sym.params["k"] == 0.1

    def test_exponential_decay(self):
        """Test exponential decay model."""
        text = """
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        # Check that the ODE is correct
        assert X in sym.odes
        rhs = sp.simplify(sym.odes[X])
        expected = -k * X
        assert sp.simplify(rhs - expected) == 0


class TestRecasting:
    """Tests for S-system recasting."""

    def test_recast_exponential_decay(self):
        """Test recasting exponential decay to S-system."""
        text = """
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        # Should have one original variable
        X = sp.Symbol("X", positive=True)
        assert X in rec.factor_map

        # Should have one auxiliary
        assert len(rec.factor_map[X]) == 1

        # Should have one equation
        assert len(rec.equations) == 1

    def test_recast_two_term_ode(self):
        """Test that an already canonical S-system is preserved."""
        text = """
        X' = a*X - b*X^2
        a = 1.0
        b = 0.1
        X = 0.5
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        X = sp.Symbol("X", positive=True)
        assert rec.factor_map[X] == [X]
        assert rec.variables == [X]
        assert len(rec.equations) == 1
        assert ssys.classify_result(rec) == ssys.SystemClass.CANONICAL_SSYSTEM

    def test_recast_preserves_initial_conditions(self):
        """Test that initial conditions are preserved in recasting."""
        text = """
        X' = -k*X
        k = 0.5
        X = 2.5
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        X = sp.Symbol("X", positive=True)
        # Product of auxiliaries should equal original initial condition
        aux_list = rec.factor_map[X]
        product = 1.0
        for aux in aux_list:
            product *= rec.initials[aux]
        assert abs(product - 2.5) < 1e-9

    def test_canonical_naming(self):
        """Test that auxiliaries are named canonically (Z_1, Z_2, ...)."""
        text = """
        A' = k1*A + k3*A^3 - k2*A^2
        k1 = 1.0
        k2 = 0.5
        k3 = 0.2
        A = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        # Check that auxiliary names follow Z_1, Z_2 pattern (Z prefix avoids collision)
        var_names = [v.name for v in rec.variables]
        assert "Z_1" in var_names
        assert "Z_2" in var_names


class TestAntimonyExport:
    """Tests for exporting S-system to Antimony."""

    def test_export_simple_system(self):
        """Test exporting a simple recast system."""
        text = """
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)
        output = ssystem_to_antimony(rec, model_name="test_recast")

        assert "model test_recast" in output
        assert "end" in output
        # Should contain auxiliary variable definitions (Z_1 prefix to avoid collision)
        assert "Z_1" in output


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_rhs(self):
        """Test handling of zero RHS (X' = 0)."""
        text = """
        X' = 0
        X = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        # Should still create auxiliary variables
        X = sp.Symbol("X", positive=True)
        assert X in rec.factor_map
        assert len(rec.factor_map[X]) >= 1

    def test_constant_term(self):
        """Test handling of constant term in ODE."""
        text = """
        X' = 1.5
        X = 0.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        # Should create auxiliary for constant
        X = sp.Symbol("X", positive=True)
        assert X in rec.factor_map


class TestCompartmentPreservation:
    """Tests for compartment name preservation through recasting."""

    def test_symsystem_compartments_field(self):
        """Test that SymSystem stores compartments."""
        from ssys.recaster import SymSystem

        X = sp.Symbol("X", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -sp.Symbol("k") * X},
            initials={X: 1.0},
            compartments={"plasma": 1.0}
        )

        assert sym.compartments == {"plasma": 1.0}

    def test_compartment_propagation_through_recast(self):
        """Test compartments propagate through recast_to_ssystem."""
        from ssys.recaster import SymSystem, recast_to_ssystem

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
            compartments={"plasma": 1.0}
        )

        result = recast_to_ssystem(sym)

        assert result.compartments == {"plasma": 1.0}

    def test_compartment_in_antimony_output(self):
        """Test original compartment name appears in Antimony output."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
            compartments={"plasma": 1.0}
        )

        result = recast_to_ssystem(sym)
        output = ssystem_to_antimony(result, model_name="test")

        assert "compartment plasma = 1" in output
        assert "compartment cell" not in output

    def test_default_compartment_when_empty(self):
        """Test default compartment 'cell' used when no compartments."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
            compartments={}
        )

        result = recast_to_ssystem(sym)
        output = ssystem_to_antimony(result, model_name="test")

        assert "compartment cell = 1" in output

    def test_compartment_not_duplicated_in_params(self):
        """Test compartment names are filtered from params output."""
        from ssys.recaster import SymSystem, recast_to_ssystem

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)
        # plasma in both params and compartments
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5, "plasma": 1.0},
            odes={X: -k * X},
            initials={X: 1.0},
            compartments={"plasma": 1.0}
        )

        result = recast_to_ssystem(sym)

        # compartment should not appear in params (filtered out)
        assert "plasma" not in result.params


class TestEpsInitMetadata:
    """Tests for user-configurable EPS_INIT via @SIM metadata."""

    def test_all_sim_metadata_preserved_in_legacy_and_sbml_parsers(self):
        """Both parser modes should use shared @SIM metadata extraction."""
        from ssys.recaster import build_sym_system, parse_antimony, parse_antimony_via_sbml

        text = """
        model sim_metadata_test()
        X' = -k*X
        k = 0.5
        X = 1.0
        end
        // @SIM T_START=1 T_END=25 N_STEPS=123 EPS_INIT=1e-8 EPS_SLACK=1e-5
        """

        legacy_ir = parse_antimony(text)
        legacy_sym = build_sym_system(legacy_ir)
        sbml_sym = parse_antimony_via_sbml(text)

        for parsed in (legacy_ir, legacy_sym, sbml_sym):
            assert parsed.sim_t_start == 1.0
            assert parsed.sim_t_end == 25.0
            assert parsed.sim_n_steps == 123
            assert parsed.eps_init == 1e-8
            assert parsed.eps_slack == 1e-5

    def test_cohesive_module_surfaces_preserve_backward_compatible_imports(self):
        """New cohesive modules expose the same public objects as ssys.recaster."""
        from ssys import formatting, lifting, parsing, recaster, recasting

        assert parsing.parse_antimony is recaster.parse_antimony
        assert parsing.parse_antimony_via_sbml is recaster.parse_antimony_via_sbml
        assert lifting.lift_rational_functions is recaster.lift_rational_functions
        assert recasting.recast_to_ssystem is recaster.recast_to_ssystem
        assert formatting.ssystem_to_antimony is recaster.ssystem_to_antimony

    def test_extract_sim_metadata_eps_init(self):
        """Test parsing EPS_INIT from @SIM comment."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6
        X' = -k*X
        k = 0.5
        X = 0.0
        """
        t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(text)

        assert t_start == 0.0
        assert t_end == 100.0
        assert n_steps == 500
        assert eps_init == 1e-6
        assert eps_slack is None

    def test_extract_sim_metadata_eps_init_none(self):
        """Test that eps_init is None when not specified."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM T_START=0 T_END=100 N_STEPS=500
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(text)

        assert t_start == 0.0
        assert t_end == 100.0
        assert n_steps == 500
        assert eps_init is None
        assert eps_slack is None

    def test_extract_sim_metadata_eps_init_only(self):
        """Test parsing EPS_INIT alone without other metadata."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM EPS_INIT=1e-12
        X' = -k*X
        k = 0.5
        X = 0.0
        """
        t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(text)

        assert t_start is None
        assert t_end is None
        assert n_steps is None
        assert eps_init == 1e-12
        assert eps_slack is None

    def test_extract_sim_metadata_eps_slack(self):
        """Test parsing EPS_SLACK from @SIM comment."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM T_START=0 T_END=100 EPS_SLACK=1e-8
        X' = -k*X
        k = 0.5
        X = 0.0
        """
        t_start, t_end, n_steps, eps_init, eps_slack = _extract_sim_metadata(text)

        assert t_start == 0.0
        assert t_end == 100.0
        assert n_steps is None
        assert eps_init is None
        assert eps_slack == 1e-8

    def test_eps_init_propagation_through_parsing(self):
        """Test eps_init is propagated through parse_antimony_via_sbml."""
        from ssys.recaster import parse_antimony_via_sbml

        text = """
        model test
        X' = -k*X
        k = 0.5
        X = 0.0
        end
        // @SIM EPS_INIT=1e-8
        """
        sym = parse_antimony_via_sbml(text)

        assert sym.eps_init == 1e-8

    def test_eps_init_used_in_pool_construction(self):
        """Test user-specified eps_init is used for zero IC approximation."""
        from ssys.recaster import SymSystem, recast_to_ssystem

        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)
        k = sp.Symbol("k", positive=True)

        # Create system with zero IC and negative exponent (requires eps_init)
        # X' = k*Y - k*X^2  has two terms with different signs
        sym = SymSystem(
            vars=[X, Y],
            params={"k": 1.0},
            odes={
                X: k * Y - k * X**2,
                Y: k * X - k * Y,
            },
            initials={X: 0.0, Y: 1.0},  # X has zero IC
            eps_init=1e-6  # User-specified eps_init
        )

        result = recast_to_ssystem(sym)

        # The auxiliary variable for X with zero IC should use eps_init=1e-6
        # instead of the default 1e-9
        # Check that 1e-9 (default) is NOT used for any IC
        default_eps = 1e-9
        for var, val in result.initials.items():
            # Should NOT use default EPS_INIT (1e-9)
            assert abs(val - default_eps) > 1e-14 or val == 0.0, \
                "Default EPS_INIT should not be used when user specifies"

    def test_default_eps_init_when_not_specified(self):
        """Test default EPS_INIT is used when eps_init not specified."""
        from ssys.recaster import SymSystem, recast_to_ssystem

        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)
        k = sp.Symbol("k", positive=True)

        # Create system without eps_init specified
        sym = SymSystem(
            vars=[X, Y],
            params={"k": 1.0},
            odes={
                X: k * Y - k * X**2,
                Y: k * X - k * Y,
            },
            initials={X: 0.0, Y: 1.0},  # X has zero IC
            # eps_init not specified - should use default
        )

        result = recast_to_ssystem(sym)

        # The zero initial condition is exact because expanded exponents do not
        # require the zero-valued pool variable as a denominator.
        assert result.eps_init is None
        x_factors = result.factor_map[X]
        assert result.initials[x_factors[0]] == 0.0
        assert all(abs(value - 1e-6) > 1e-14 for value in result.initials.values())


class TestSbmlParserIcHandling:
    """Tests for SBML parser initial condition handling.

    When libSBML parses Antimony via SBML conversion, species initial
    conditions may end up in the params dict instead of the initials dict.
    These tests ensure the recaster handles this correctly.
    """

    def test_unparseable_kinetic_law_raises_structured_error(self, monkeypatch):
        """An unparseable reaction rate must not be silently dropped."""
        import libsbml

        from ssys.recaster import SBMLParseError, parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_bad" name="bad reaction" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> S </ci><cn> 1 </cn></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(species=species, reactions=reactions)

        monkeypatch.setattr(libsbml, "formulaToString", lambda _math: "bad +")

        with pytest.raises(SBMLParseError) as exc_info:
            parse_sbml_from_string(sbml)

        err = exc_info.value
        assert err.kind == "kinetic_law"
        assert err.reaction_id == "J_bad"
        assert err.reaction_name == "bad reaction"
        assert err.formula == "bad +"

    def test_unparseable_rate_rule_raises_structured_error(self, monkeypatch):
        """An unparseable rate rule must not leave the state derivative at zero."""
        import libsbml

        from ssys.recaster import SBMLParseError, parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        rules = """
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 1 </cn></math>
      </rateRule>
    </listOfRules>"""
        sbml = _minimal_sbml(species=species, rules=rules)

        monkeypatch.setattr(libsbml, "formulaToString", lambda _math: "bad +")

        with pytest.raises(SBMLParseError) as exc_info:
            parse_sbml_from_string(sbml)

        err = exc_info.value
        assert err.kind == "rate_rule"
        assert err.variable == "S"
        assert err.formula == "bad +"

    def test_unknown_formula_identifier_raises_structured_error(self, monkeypatch):
        """Model-derived formulas must not silently create undeclared symbols."""
        import libsbml

        from ssys.recaster import SBMLParseError, parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        rules = """
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> S </ci></math>
      </rateRule>
    </listOfRules>"""
        sbml = _minimal_sbml(species=species, rules=rules)

        monkeypatch.setattr(libsbml, "formulaToString", lambda _math: "S + missing_param")

        with pytest.raises(SBMLParseError) as exc_info:
            parse_sbml_from_string(sbml)

        err = exc_info.value
        assert err.kind == "rate_rule"
        assert err.variable == "S"
        assert "unknown identifier(s): missing_param" in err.message

    def test_unknown_formula_function_raises_structured_error(self, monkeypatch):
        """Unsupported function calls fail before SymPy parsing."""
        import libsbml

        from ssys.recaster import SBMLParseError, parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        rules = """
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> S </ci></math>
      </rateRule>
    </listOfRules>"""
        sbml = _minimal_sbml(species=species, rules=rules)

        monkeypatch.setattr(libsbml, "formulaToString", lambda _math: "unsupported(S)")

        with pytest.raises(SBMLParseError) as exc_info:
            parse_sbml_from_string(sbml)

        err = exc_info.value
        assert err.kind == "rate_rule"
        assert err.variable == "S"
        assert "unsupported function(s): unsupported" in err.message

    def test_sbml_function_definition_expands_in_kinetic_law(self):
        """Declared SBML FunctionDefinition helpers are expanded before parsing."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        function_definitions = """
    <listOfFunctionDefinitions>
      <functionDefinition id="rate_law">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci> k </ci></bvar>
            <bvar><ci> substrate </ci></bvar>
            <apply><times/><ci> k </ci><ci> substrate </ci></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_fn" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci> rate_law </ci><ci> k </ci><ci> S </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            function_definitions=function_definitions,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)
        s_sym = sp.Symbol("S", positive=True)
        p_sym = sp.Symbol("P", positive=True)
        k_sym = sp.Symbol("k", positive=True)

        assert sp.simplify(sym.odes[s_sym] + k_sym * s_sym) == 0
        assert sp.simplify(sym.odes[p_sym] - k_sym * s_sym) == 0

    def test_sbml_function_definition_preserves_prefix_like_arguments(self):
        """Regression for BIOMD0000001058-style function parameters."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="Q" compartment="cell" initialAmount="2" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        function_definitions = """
    <listOfFunctionDefinitions>
      <functionDefinition id="rate_law">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci> k_ </ci></bvar>
            <bvar><ci> k </ci></bvar>
            <bvar><ci> S </ci></bvar>
            <bvar><ci> Q </ci></bvar>
            <apply><times/>
              <apply><plus/><ci> k_ </ci><apply><times/><ci> k </ci><ci> Q </ci></apply></apply>
              <ci> S </ci>
            </apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
      <parameter id="k_0" value="2.0" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_fn" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci> rate_law </ci><ci> k </ci><ci> k_0 </ci><ci> S </ci><ci> Q </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            function_definitions=function_definitions,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)

        S = sp.Symbol("S", positive=True)
        P = sp.Symbol("P", positive=True)
        Q = sp.Symbol("Q", positive=True)
        k = sp.Symbol("k", positive=True)
        k_0 = sp.Symbol("k_0", positive=True)
        expected = (k + k_0 * Q) * S
        assert sp.simplify(sym.odes[S] + expected) == 0
        assert sp.simplify(sym.odes[P] - expected) == 0

    def test_sbml_function_definition_duplicate_parameters_keep_first_argument(self):
        """Regression for MODEL2003040001 duplicate lambda parameter names."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        function_definitions = """
    <listOfFunctionDefinitions>
      <functionDefinition id="rate_law">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <lambda>
            <bvar><ci> k </ci></bvar>
            <bvar><ci> k </ci></bvar>
            <bvar><ci> S </ci></bvar>
            <apply><times/><ci> k </ci><ci> S </ci></apply>
          </lambda>
        </math>
      </functionDefinition>
    </listOfFunctionDefinitions>"""
        parameters = """
    <listOfParameters>
      <parameter id="k_plus" value="0.5" constant="true"/>
      <parameter id="k_minus" value="24.0" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_fn" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><ci> rate_law </ci><ci> k_plus </ci><ci> k_minus </ci><ci> S </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            function_definitions=function_definitions,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)

        S = sp.Symbol("S", positive=True)
        P = sp.Symbol("P", positive=True)
        k_plus = sp.Symbol("k_plus", positive=True)
        expected = k_plus * S
        assert sp.simplify(sym.odes[S] + expected) == 0
        assert sp.simplify(sym.odes[P] - expected) == 0

    def test_malicious_formula_string_rejected_before_sympify(self, monkeypatch):
        """Malicious-looking formula text is reported as parser data, not evaluated."""
        import libsbml

        from ssys.recaster import SBMLParseError, parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        rules = """
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> S </ci></math>
      </rateRule>
    </listOfRules>"""
        sbml = _minimal_sbml(species=species, rules=rules)

        monkeypatch.setattr(
            libsbml,
            "formulaToString",
            lambda _math: "__import__('os').system('echo unsafe')",
        )

        with pytest.raises(SBMLParseError) as exc_info:
            parse_sbml_from_string(sbml)

        err = exc_info.value
        assert err.kind == "rate_rule"
        assert "__import__" in err.message

    def test_declared_species_I_is_not_sympy_imaginary_in_sbml_parser(self):
        """Reserved SymPy name I is a model symbol when declared by SBML."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="I" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_i" reversible="false">
        <listOfReactants>
          <speciesReference species="I" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> I </ci></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(species=species, reactions=reactions)

        sym = parse_sbml_from_string(sbml)

        I_sym = sp.Symbol("I", positive=True)
        P = sp.Symbol("P", positive=True)
        assert I_sym in sym.odes
        assert sp.simplify(sym.odes[I_sym] + I_sym) == 0
        assert sp.simplify(sym.odes[P] - I_sym) == 0

    def test_initial_assignment_evaluation_failure_raises_by_default(self):
        """InitialAssignment formulas fail closed unless warning mode is requested."""
        from ssys.recaster import SBMLParseError, parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        initial_assignments = """
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> missing_param </ci></math>
      </initialAssignment>
    </listOfInitialAssignments>"""
        sbml = _minimal_sbml(species=species, initial_assignments=initial_assignments)

        with pytest.raises(SBMLParseError) as exc_info:
            parse_sbml_from_string(sbml)

        err = exc_info.value
        assert err.kind == "initial_assignment"
        assert err.variable == "S"
        assert err.formula == "missing_param"

    def test_initial_assignment_failure_can_warn_for_exploratory_mode(self):
        """Exploratory mode keeps the previous default only when explicitly requested."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        initial_assignments = """
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> missing_param </ci></math>
      </initialAssignment>
    </listOfInitialAssignments>"""
        sbml = _minimal_sbml(species=species, initial_assignments=initial_assignments)

        with pytest.warns(RuntimeWarning, match="initial assignment"):
            sym = parse_sbml_from_string(sbml, warn_initial_assignment_failures=True)

        S = sp.Symbol("S", positive=True)
        assert sym.initials[S] == 0.0

    def test_initial_assignment_can_reference_declared_species_initial_value(self):
        """SBML InitialAssignments may be numeric expressions over declared species."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="template" compartment="cell" initialAmount="2.5" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="S" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        initial_assignments = """
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn> 2 </cn><ci> template </ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>"""
        sbml = _minimal_sbml(species=species, initial_assignments=initial_assignments)

        sym = parse_sbml_from_string(sbml)

        S = sp.Symbol("S", positive=True)
        assert sym.initials[S] == 5.0

    def test_formula_identifier_scan_ignores_scientific_notation(self, monkeypatch):
        """Scientific notation exponents are numeric literals, not identifiers."""
        import libsbml

        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_sci" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> S </ci></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(species=species, reactions=reactions)
        monkeypatch.setattr(libsbml, "formulaToString", lambda _math: "5.9e-4 * S")

        sym = parse_sbml_from_string(sbml)

        S = sp.Symbol("S", positive=True)
        P = sp.Symbol("P", positive=True)
        expected = sp.Float("5.9e-4") * S
        assert sp.simplify(sym.odes[S] + expected) == 0
        assert sp.simplify(sym.odes[P] - expected) == 0

    def test_declared_keyword_identifier_parses_as_model_symbol(self):
        """Valid SBML ids such as lambda are model symbols, not Python syntax."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="lambda" value="0.5" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J_lambda" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> lambda </ci><ci> S </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)

        S = sp.Symbol("S", positive=True)
        P = sp.Symbol("P", positive=True)
        lambda_sym = sp.Symbol("lambda", positive=True)
        assert sp.simplify(sym.odes[S] + lambda_sym * S) == 0
        assert sp.simplify(sym.odes[P] - lambda_sym * S) == 0

    def test_same_named_local_parameters_are_scoped_by_reaction(self):
        """Same local parameter ids in different reactions preserve distinct values."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="S" compartment="cell" initialAmount="1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="P" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> S </ci></apply>
          </math>
          <listOfLocalParameters>
            <localParameter id="k" value="1"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
      <reaction id="J1" reversible="false">
        <listOfReactants>
          <speciesReference species="P" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> P </ci></apply>
          </math>
          <listOfLocalParameters>
            <localParameter id="k" value="2"/>
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(species=species, reactions=reactions)

        sym = parse_sbml_from_string(sbml)

        assert sym.params["J0__k"] == 1.0
        assert sym.params["J1__k"] == 2.0
        assert "k" not in sym.params

        S = sp.Symbol("S", positive=True)
        P = sp.Symbol("P", positive=True)
        J0_k = sp.Symbol("J0__k", positive=True)
        J1_k = sp.Symbol("J1__k", positive=True)

        assert sp.simplify(sym.odes[S] + J0_k * S) == 0
        assert sp.simplify(sym.odes[P] - (J0_k * S - J1_k * P)) == 0

        numeric_subs = {J0_k: sym.params["J0__k"], J1_k: sym.params["J1__k"]}
        assert sp.simplify(sym.odes[P].subs(numeric_subs) - (S - 2 * P)) == 0

    def test_sbml_rules_and_initial_assignments_are_preserved(self):
        """SBML rate, assignment, algebraic, species, parameter, and compartment data survive."""
        from ssys.recaster import SolverRequirement, parse_sbml_from_string

        compartments = """
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" constant="true"/>
    </listOfCompartments>"""
        species = """
      <species id="S" compartment="cell" initialConcentration="1.5" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" constant="true"/>
      <parameter id="obs" constant="false"/>
    </listOfParameters>"""
        rules = """
    <listOfRules>
      <rateRule variable="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> k </ci></math>
      </rateRule>
      <assignmentRule variable="obs">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><ci> S </ci><ci> k </ci></apply>
        </math>
      </assignmentRule>
      <algebraicRule>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><ci> obs </ci><ci> S </ci></apply>
        </math>
      </algebraicRule>
    </listOfRules>"""
        initial_assignments = """
    <listOfInitialAssignments>
      <initialAssignment symbol="S">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 4 </cn></math>
      </initialAssignment>
      <initialAssignment symbol="k">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><divide/><cn> 1 </cn><cn> 7 </cn></apply>
        </math>
      </initialAssignment>
      <initialAssignment symbol="cell">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn> 2 </cn></math>
      </initialAssignment>
    </listOfInitialAssignments>"""
        sbml = _minimal_sbml(
            compartments=compartments,
            species=species,
            parameters=parameters,
            rules=rules,
            initial_assignments=initial_assignments,
        )

        sym = parse_sbml_from_string(sbml)

        S = sp.Symbol("S", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)
        assert sym.vars == [S, B]
        assert sym.initials[S] == 4.0
        assert sym.initials[B] == 0.0
        assert sym.params["k"] == pytest.approx(1.0 / 7.0)
        assert sym.compartments["cell"] == 2.0
        assert sp.simplify(sym.odes[S] - k) == 0
        assert sym.assignment_rules == {"obs": "S + k"}
        assert sym.algebraic_constraints == ["obs - S"]
        assert sym.solver_requirement == SolverRequirement.DAE_REQUIRED

    def test_sbml_assignment_rule_species_are_not_recast_as_independent_states(self):
        """Assignment-rule species are algebraic observables, not ODE states."""
        import antimony

        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="APLC" compartment="cell" initialConcentration="0.1" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="IP3" compartment="cell" initialConcentration="0.2" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="DG" compartment="cell" initialConcentration="0.2" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="PLC" compartment="cell" initialConcentration="0.3" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="1" constant="true"/>
      <parameter id="Cplc_total" value="10" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="PLC" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="APLC" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> PLC </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        rules = """
    <listOfRules>
      <assignmentRule variable="DG">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci> IP3 </ci></math>
      </assignmentRule>
      <assignmentRule variable="PLC">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><minus/><ci> Cplc_total </ci><ci> APLC </ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>"""
        sbml = _minimal_sbml(
            species=species,
            parameters=parameters,
            reactions=reactions,
            rules=rules,
        )

        sym = parse_sbml_from_string(sbml)

        assert [str(v) for v in sym.vars] == ["APLC", "IP3"]
        assert set(map(str, sym.odes)) == {"APLC", "IP3"}
        assert sym.assignment_rules == {"DG": "IP3", "PLC": "Cplc_total - APLC"}

        rec = recast_to_ssystem(sym)
        assert "DG" not in {str(v) for v in rec.factor_map}
        assert "PLC" not in {str(v) for v in rec.factor_map}

        output = ssystem_to_antimony(rec, model_name="assignment_rule_species")
        assert len(re.findall(r"^DG\s*:=", output, flags=re.MULTILINE)) == 1
        assert len(re.findall(r"^PLC\s*:=", output, flags=re.MULTILINE)) == 1
        assert "DG := IP3;" in output
        assert "PLC := Cplc_total - APLC;" in output
        assert not re.search(r"\bDG\s*:=\s*Z_", output)
        assert not re.search(r"\bPLC\s*:=\s*Z_", output)

        antimony.clearPreviousLoads()
        assert antimony.loadAntimonyString(output) >= 0, antimony.getLastError()

    def test_species_ic_in_params_used_for_auxiliary_ic(self):
        """Test that species ICs in params are used for auxiliary IC."""
        from ssys.recaster import SymSystem, lift_rational_functions

        X = sp.Symbol("X", positive=True)
        KM = sp.Symbol("KM", positive=True)

        # Simulate SBML parser behavior: species IC in params, not initials
        # When SBML parser puts ICs in params, initials is empty or missing
        sym = SymSystem(
            vars=[X],
            params={"X": 0.5, "KM": 1.0},  # X IC is in params!
            odes={X: 1 / (KM + X)},  # ODE with rational term
            initials={},  # Empty - SBML puts species ICs in params
        )

        result, aux_defs = lift_rational_functions(sym)

        # Y_1 = KM + X should have IC = 1.0 + 0.5 = 1.5
        Y_1 = sp.Symbol("Y_1", positive=True)
        assert Y_1 in result.initials
        assert abs(result.initials[Y_1] - 1.5) < 1e-10

    def test_species_ic_output_from_params(self):
        """Test that species ICs from params appear in Antimony output."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        KM = sp.Symbol("KM", positive=True)

        # Simulate SBML parser behavior: species IC in params
        sym = SymSystem(
            vars=[X],
            params={"X": 0.5, "KM": 1.0},  # X IC is in params!
            odes={X: 1 / (KM + X)},
            initials={},  # Empty initials - all ICs in params
        )

        result = recast_to_ssystem(sym)
        output = ssystem_to_antimony(result, model_name="test")

        # X = 0.5 should appear in output
        assert "X = 0.5" in output

    def test_mm_to_gma_initial_conditions(self):
        """Test Michaelis-Menten to GMA recasting has correct ICs."""
        from ssys.recaster import parse_antimony_via_sbml, recast_to_ssystem

        # Simplified MM model similar to MS2007_MM_to_GMA.ant
        text = """
        model test
        X' = Vmax/(KM + X) - k*X
        Vmax = 1.0
        KM = 0.5
        k = 0.1
        X = 0.5
        end
        """

        sym = parse_antimony_via_sbml(text)
        result = recast_to_ssystem(sym)

        # Should have auxiliary Y_1 = KM + X with IC = 0.5 + 0.5 = 1.0
        Y_1 = sp.Symbol("Y_1", positive=True)
        if Y_1 in result.initials:
            assert abs(result.initials[Y_1] - 1.0) < 1e-10

    def test_composite_function_ic_from_params(self):
        """Test that composite function auxiliaries use ICs from params.

        Regression test for bug where exp(log(Z)^2) with Z=2 produced
        wrong auxiliary ICs: Z_1=1, Z_2=0 (using Z=1 default) instead of
        Z_1≈1.617, Z_2≈0.693 (using Z=2 from params).

        Root cause: lift_composite_functions was only checking initials dict,
        but SBML parser puts species ICs in params dict.
        """
        import math

        from ssys.recaster import SymSystem, lift_composite_functions

        Z = sp.Symbol("Z", positive=True)
        k = sp.Symbol("k", positive=True)

        # Simulate SBML parser behavior: Z=2 in params, not initials
        # ODE: Z' = k * exp(log(Z)^2)
        sym = SymSystem(
            vars=[Z],
            params={"k": 0.001, "Z": 2.0},  # Z IC is in params!
            odes={Z: k * sp.exp(sp.log(Z)**2)},
            initials={},  # Empty - SBML puts species ICs in params
        )

        result, aux_defs = lift_composite_functions(sym)

        # Compute expected values
        Z_init = 2.0
        Z_2_expected = math.log(Z_init)  # ln(2) ≈ 0.693
        Z_1_expected = math.exp(Z_2_expected**2)  # exp((ln(2))^2) ≈ 1.617

        # Find auxiliaries by their definitions
        Z_1 = None  # exp(log(Z)^2)
        Z_2 = None  # log(Z)
        for aux, defn in aux_defs.items():
            defn_str = str(defn)
            if "log(Z)" in defn_str and "exp" not in defn_str:
                Z_2 = aux
            elif "exp" in defn_str:
                Z_1 = aux

        # Check auxiliaries were created
        assert Z_1 is not None, "Z_1 (exp(log(Z)^2)) auxiliary not found"
        assert Z_2 is not None, "Z_2 (log(Z)) auxiliary not found"

        # Check ICs are computed correctly from Z=2 (not Z=1 default)
        assert Z_1 in result.initials, f"Z_1 ({Z_1}) not in initials"
        assert Z_2 in result.initials, f"Z_2 ({Z_2}) not in initials"

        assert abs(result.initials[Z_1] - Z_1_expected) < 1e-6, \
            f"Z_1 IC wrong: got {result.initials[Z_1]}, expected {Z_1_expected}"
        assert abs(result.initials[Z_2] - Z_2_expected) < 1e-6, \
            f"Z_2 IC wrong: got {result.initials[Z_2]}, expected {Z_2_expected}"


class TestSbmlCompartmentVolumeScaling:
    """Non-unit compartment volumes must scale reaction ODEs and initial values.

    Regression for a silent-wrong-answer bug: reaction-derived ODEs were assembled
    as ``d(species)/dt = Σ stoich·kineticLaw`` (an amount rate) while the species
    symbol denotes concentration (hasOnlySubstanceUnits=false). The correct
    concentration ODE divides by the owning compartment size, and initial values
    supplied as amounts must be divided by that size. Both were omitted, so any
    model with a compartment size != 1 produced ODEs off by the compartment size.
    """

    _COMP_SIZE_2 = """
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="2" units="litre" constant="true"/>
    </listOfCompartments>"""

    def test_compartment_factor_idiom_cancels_and_initials_reconcile(self):
        """``compartment·k·A`` amount-rate / V gives the exact concentration ODE."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="A" compartment="cell" initialAmount="4" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> cell </ci><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            compartments=self._COMP_SIZE_2,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)

        # d[A]/dt = -(cell·k·A)/cell = -k·A ; the compartment factor cancels.
        assert sp.simplify(sym.odes[A] + k * A) == 0
        assert sp.simplify(sym.odes[B] - k * A) == 0

        # initialAmount=4 in a size-2 compartment ⇒ concentration 2.0.
        assert sym.initials[A] == 2.0
        assert sym.initials[B] == 0.0
        assert sym.compartments["cell"] == 2.0

    def test_bare_amount_rate_law_is_divided_by_volume(self):
        """A kinetic law without a compartment factor is an amount rate ÷ V."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="A" compartment="cell" initialConcentration="3" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            compartments=self._COMP_SIZE_2,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)
        cell = sp.Symbol("cell", positive=True)

        # No compartment factor in the law ⇒ ODE carries an explicit 1/V.
        assert sp.simplify(sym.odes[A] + k * A / cell) == 0
        assert sp.simplify(sym.odes[B] - k * A / cell) == 0
        # Numerically, with cell = 2: d[A]/dt = -k·A/2.
        subs = {cell: sym.params["cell"]}
        assert sp.simplify(sym.odes[A].subs(subs) + k * A / 2) == 0

        # initialConcentration is already in the symbol's unit: unchanged by V.
        assert sym.initials[A] == 3.0

    def test_substance_units_species_are_amounts_and_not_scaled(self):
        """hasOnlySubstanceUnits=true species stay amount-valued (no 1/V)."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="A" compartment="cell" initialConcentration="1" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            compartments=self._COMP_SIZE_2,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)

        # Amount-valued species: dAmount/dt = -k·A, no compartment division.
        assert sp.simplify(sym.odes[A] + k * A) == 0
        assert sp.simplify(sym.odes[B] - k * A) == 0
        # initialConcentration=1 in a size-2 compartment ⇒ amount 2.0.
        assert sym.initials[A] == 2.0
        assert sym.initials[B] == 0.0


def _max_trajectory_error_vs_roadrunner(sbml, *, t_end, n=11, amount_species=frozenset()):
    """Integrate ssys's parsed ODEs and compare to RoadRunner on the ORIGINAL SBML.

    ssys's own validators are common-mode blind to parser misinterpretation: they
    compare two systems that both flow from this parser. The independent ground
    truth is libRoadRunner loading the same SBML directly. This integrates the
    ODEs ssys parsed (params substituted) with scipy and returns the worst
    per-species absolute trajectory error against RoadRunner. Concentration
    species are read as ``[S]``; ``amount_species`` names are read as amounts.
    """
    rr = pytest.importorskip("roadrunner", reason="requires libRoadRunner")
    solve_ivp = pytest.importorskip("scipy.integrate", reason="requires scipy").solve_ivp
    import numpy as np

    from ssys.recaster import parse_sbml_from_string

    sym = parse_sbml_from_string(sbml)
    state = list(sym.vars)
    param_subs = {sp.Symbol(name, positive=True): val for name, val in sym.params.items()}
    rhs = [sp.sympify(sym.odes[s]).subs(param_subs) for s in state]
    free = set()
    for expr in rhs:
        free |= expr.free_symbols
    leftover = free - set(state)
    assert not leftover, f"parsed ODEs still reference non-state symbols: {leftover}"

    f = sp.lambdify(state, rhs, modules="numpy")
    y0 = [float(sym.initials[s]) for s in state]
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        lambda t, y: f(*y), (0.0, t_end), y0, t_eval=t_eval, rtol=1e-10, atol=1e-12
    )
    assert sol.success

    r = rr.RoadRunner(sbml)
    selections = [
        (s.name if s.name in amount_species else f"[{s.name}]") for s in state
    ]
    r.selections = ["time"] + selections
    result = r.simulate(0.0, t_end, n)
    cols = list(result.colnames)
    worst = 0.0
    for i, s in enumerate(state):
        col = s.name if s.name in amount_species else f"[{s.name}]"
        worst = max(worst, float(np.max(np.abs(sol.y[i] - result[:, cols.index(col)]))))
    return worst


class TestSbmlConversionFactor:
    """SBML L3 conversionFactor must scale each species' reaction-derived rate.

    Regression for a silent-wrong-answer bug: a species' (or the model's default)
    ``conversionFactor`` scales how its amount changes per unit reaction extent,
    ``d(amount_S)/dt = cf_S·Σ stoich·kineticLaw``, and was referenced nowhere in
    the parser. Any model declaring one integrated the unscaled rate. The factor
    is a per-species scalar, so it multiplies the whole reaction-derived ODE and
    composes with the STEP 5a compartment-volume division. Cross-checked against
    libRoadRunner on the original SBML.
    """

    _COMP_SIZE_2 = """
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="2" units="litre" constant="true"/>
    </listOfCompartments>"""

    def test_species_conversion_factor_scales_ode_and_composes_with_volume(self):
        """cf multiplies the amount rate; the compartment factor still cancels."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="A" compartment="cell" initialAmount="4" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false" conversionFactor="cf"/>
      <species id="B" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false" conversionFactor="cf"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.5" constant="true"/>
      <parameter id="cf" value="3" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> cell </ci><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            compartments=self._COMP_SIZE_2,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)
        cf = sp.Symbol("cf", positive=True)

        # d[A]/dt = cf·(-cell·k·A)/cell = -cf·k·A ; cell cancels, cf survives.
        assert sp.simplify(sym.odes[A] + cf * k * A) == 0
        assert sp.simplify(sym.odes[B] - cf * k * A) == 0
        assert sym.initials[A] == 2.0
        assert sym.params["cf"] == 3.0

        assert _max_trajectory_error_vs_roadrunner(sbml, t_end=4.0) < 1e-4

    def test_model_default_conversion_factor_with_per_species_override(self):
        """Model default applies to species without their own; per-species wins."""
        from ssys.recaster import parse_sbml_from_string

        # Model default cfM=4 applies to B; A overrides with cfA=2. Unit volume so
        # the factor is isolated from volume scaling.
        sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="m" substanceUnits="mole" timeUnits="second" extentUnits="mole" conversionFactor="cfM">
    <listOfCompartments>
      <compartment id="cell" spatialDimensions="3" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="5" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false" conversionFactor="cfA"/>
      <species id="B" compartment="cell" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.7" constant="true"/>
      <parameter id="cfA" value="2" constant="true"/>
      <parameter id="cfM" value="4" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)
        cfA = sp.Symbol("cfA", positive=True)
        cfM = sp.Symbol("cfM", positive=True)

        # A carries its own cfA; B inherits the model default cfM.
        assert sp.simplify(sym.odes[A] + cfA * k * A) == 0
        assert sp.simplify(sym.odes[B] - cfM * k * A) == 0

        assert _max_trajectory_error_vs_roadrunner(sbml, t_end=4.0) < 1e-4

    def test_substance_units_species_conversion_factor_is_not_volume_scaled(self):
        """hosu=true species: cf applies to the amount rate, no compartment division."""
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="A" compartment="cell" initialAmount="6" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false" conversionFactor="cf"/>
      <species id="B" compartment="cell" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false" conversionFactor="cf"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.4" constant="true"/>
      <parameter id="cf" value="3" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(
            species=species,
            compartments=self._COMP_SIZE_2,
            parameters=parameters,
            reactions=reactions,
        )

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)
        cf = sp.Symbol("cf", positive=True)

        # Amount-valued: dAmount/dt = -cf·k·A, no 1/V; initial amount unchanged.
        assert sp.simplify(sym.odes[A] + cf * k * A) == 0
        assert sp.simplify(sym.odes[B] - cf * k * A) == 0
        assert sym.initials[A] == 6.0

        assert (
            _max_trajectory_error_vs_roadrunner(
                sbml, t_end=4.0, amount_species=frozenset({"A", "B"})
            )
            < 1e-4
        )


class TestSbmlStoichiometry:
    """Constant stoichiometryMath and non-constant SpeciesReferences.

    Regression for a silent-wrong-answer bug: STEP 5 read only the static
    ``getStoichiometry()`` attribute and ignored L2 ``<stoichiometryMath>`` (which
    ``getStoichiometry()`` reports as 1.0). A stoichiometry that constant-folds is
    now used exactly; a genuinely variable one is rejected at the trust boundary
    (see tests/test_negative_corpus.py). Cross-checked against libRoadRunner.
    """

    @staticmethod
    def _l2_reaction_sbml(reactant_ref: str, product_ref: str, extra_params: str = "") -> str:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="m">
    <listOfCompartments><compartment id="cell" size="1"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="2" boundaryCondition="false"/>
      <species id="B" compartment="cell" initialConcentration="0" boundaryCondition="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="0.3"/>{extra_params}</listOfParameters>
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>{reactant_ref}</listOfReactants>
        <listOfProducts>{product_ref}</listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

    def test_constant_stoichiometry_math_is_folded(self):
        """A numeric <stoichiometryMath> (=2) is used instead of the 1.0 default."""
        from ssys.recaster import parse_sbml_from_string

        reactant = """
          <speciesReference species="A">
            <stoichiometryMath><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math></stoichiometryMath>
          </speciesReference>"""
        product = """<speciesReference species="B" stoichiometry="3"/>"""
        sbml = self._l2_reaction_sbml(reactant, product)

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)

        assert sp.simplify(sym.odes[A] + 2 * k * A) == 0
        assert sp.simplify(sym.odes[B] - 3 * k * A) == 0
        assert _max_trajectory_error_vs_roadrunner(sbml, t_end=2.0) < 1e-4

    def test_constant_stoichiometry_math_over_parameter_is_folded(self):
        """<stoichiometryMath> = 2*sc constant-folds over parameter sc=2 to 4."""
        from ssys.recaster import parse_sbml_from_string

        reactant = """
          <speciesReference species="A">
            <stoichiometryMath><math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply><times/><cn>2</cn><ci>sc</ci></apply>
            </math></stoichiometryMath>
          </speciesReference>"""
        product = """<speciesReference species="B" stoichiometry="1"/>"""
        sbml = self._l2_reaction_sbml(
            reactant, product, extra_params="""<parameter id="sc" value="2"/>"""
        )

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        k = sp.Symbol("k", positive=True)

        assert sp.simplify(sym.odes[A] + 4 * k * A) == 0
        assert _max_trajectory_error_vs_roadrunner(sbml, t_end=1.0) < 1e-4

    def test_non_constant_flagged_but_rule_free_stoichiometry_is_kept(self):
        """An L3 constant=false SpeciesReference with no rule folds to its attribute.

        Only a stoichiometry that a rule/initialAssignment actually drives is
        variable; merely marking it mutable does not, so it stays supported.
        """
        from ssys.recaster import parse_sbml_from_string

        species = """
      <species id="A" compartment="cell" initialConcentration="2" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialConcentration="0" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>"""
        parameters = """
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>"""
        reactions = """
    <listOfReactions>
      <reaction id="J0" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="2" constant="false"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="false"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci> k </ci><ci> A </ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>"""
        sbml = _minimal_sbml(species=species, parameters=parameters, reactions=reactions)

        sym = parse_sbml_from_string(sbml)
        A = sp.Symbol("A", positive=True)
        B = sp.Symbol("B", positive=True)
        k = sp.Symbol("k", positive=True)

        assert sp.simplify(sym.odes[A] + 2 * k * A) == 0
        assert sp.simplify(sym.odes[B] - k * A) == 0
        assert _max_trajectory_error_vs_roadrunner(sbml, t_end=2.0) < 1e-4


class TestSqrtSumIcComputation:
    """Tests for sqrt(sum) auxiliary IC computation.

    Regression test for bug where sqrt(1 + Z2^2) with Z2(0)=0.01 produced
    wrong auxiliary IC Z_1 ≈ 1.41421 (using fallback Z2=1) instead of
    Z_1 ≈ 1.00005 (correct: sqrt(1 + 0.01^2)).

    Root cause: Symbol object identity mismatch in lift_composite_functions.
    The symbols in sqrt_at_0.free_symbols were different Python objects from
    those in sym.initials, causing the substitution to fail silently.
    """

    def test_sqrt_sum_auxiliary_ic(self):
        """Test sqrt(sum) auxiliary uses correct IC from state variables."""
        import math

        from ssys.recaster import parse_antimony_via_sbml, recast_to_ssystem

        # Model with sqrt(1 + Z2^2) term
        text = """
        model test
        Z1 = 0.01
        Z2 = 0.01
        Z1' = Z2
        Z2' = (1 + Z2^2)^(1/2) * (25 - time)^(-1)
        end
        """

        sym = parse_antimony_via_sbml(text)
        result = recast_to_ssystem(sym)

        # sqrt(1 + 0.01^2) = sqrt(1.0001) ≈ 1.00005
        expected_sqrt_ic = math.sqrt(1 + 0.01**2)

        # Find auxiliary with IC close to expected value
        found_correct_ic = False
        for var, ic in result.initials.items():
            var_name = var.name if hasattr(var, 'name') else str(var)
            # Look for sqrt auxiliary (Z_n pattern, not original Z1/Z2)
            if var_name.startswith('Z_') and '_' in var_name:
                # Check if this is the sqrt auxiliary (IC ≈ 1.00005)
                if abs(ic - expected_sqrt_ic) < 1e-6:
                    found_correct_ic = True
                    break
                # Also check it's NOT using fallback value (≈1.41421)
                wrong_ic = math.sqrt(2)  # sqrt(1+1^2) if Z2=1 fallback
                assert abs(ic - wrong_ic) > 0.1, \
                    f"sqrt aux {var_name} has wrong IC {ic} (Z2=1 fallback)"

        assert found_correct_ic, \
            f"No auxiliary with expected sqrt IC ≈ {expected_sqrt_ic}. " \
            f"ICs: {[(str(k), v) for k, v in result.initials.items()]}"

    def test_squared_lifting_ic(self):
        """Test lift_squared_for_sqrt uses correct IC for u = X^2 + c."""
        from ssys.recaster import SymSystem, lift_squared_for_sqrt

        X = sp.Symbol("X", positive=True)

        # Create minimal system with sqrt(X^2 + 1) pattern
        # This triggers lift_squared_for_sqrt for u = X^2 + 1
        sym = SymSystem(
            vars=[X],
            params={},
            odes={X: sp.Integer(1)},  # X' = 1 (dummy ODE)
            initials={X: 0.5},  # X(0) = 0.5
        )

        # Create the sqrt expression and lift it
        sqrt_expr = sp.sqrt(X**2 + 1)  # sqrt(X^2 + 1)
        result = lift_squared_for_sqrt(sqrt_expr, aux_counter=1, sym=sym)

        assert result is not None, \
            "lift_squared_for_sqrt should handle sqrt(X^2 + c)"

        # u(0) = X(0)^2 + 1 = 0.5^2 + 1 = 1.25
        expected_u_ic = 0.5**2 + 1.0

        u_sym = result.new_vars[0]
        actual_u_ic = float(result.new_initials[u_sym])

        assert abs(actual_u_ic - expected_u_ic) < 1e-10, \
            f"u IC wrong: {actual_u_ic}, expected {expected_u_ic}"


class TestEpsInitFactorMapExpansion:
    """Tests for EPS_INIT factor_map expansion fix.

    Regression test for bug where vars_with_neg_exp was computed from
    intermediate exponents BEFORE factor_map expansion. This caused
    variables to incorrectly get EPS_INIT when their negative exponents
    actually cancel out after expansion.

    Example: if z = Z_5*Z_6*Z_7 and exponent dict has {z:1, Z_5:-1, Z_7:-1}
      After expansion: Z_5^1*Z_6^1*Z_7^1*Z_5^-1*Z_7^-1 = Z_6^1
      So Z_5 and Z_7 don't actually appear with negative exponents!
    """

    def test_canceling_negative_exponents_get_zero_ic(self):
        """Test variables with canceling negative exponents get zero IC."""
        from ssys.recaster import parse_antimony_via_sbml, recast_to_ssystem

        # Rössler-band model where z's pool terms cancel
        text = """
        model test
        x' = -y - z
        y' = a*y + x
        z' = b*z - c*z + x*z
        a = 0.343
        b = 1.82
        c = 9.75
        x = 0
        y = 1
        z = 0
        end
        """

        sym = parse_antimony_via_sbml(text)
        result = recast_to_ssystem(sym)

        # z maps to Z_5*Z_6*Z_7 (3 terms in z' equation)
        # The z^1 in Z_6's growth expands to Z_5*Z_6*Z_7
        # Combined with Z_5^-1 and Z_7^-1, this cancels out
        # So Z_5 should get 0.0 (original z IC), NOT 1e-6 (EPS_INIT)

        # Look up Z_5 by name (symbol object identity may differ)
        initials_by_name = {k.name: v for k, v in result.initials.items()}

        # Z_5 should have IC = 0.0, NOT EPS_INIT (1e-6)
        assert "Z_5" in initials_by_name, \
            f"Z_5 not found in initials: {list(initials_by_name.keys())}"
        z5_ic = initials_by_name["Z_5"]

        # Should be exactly 0.0, not EPS_INIT (1e-6)
        assert z5_ic == 0.0, \
            f"Z_5 IC should be 0.0 (canceling negatives), got {z5_ic}"

    def test_true_negative_exponent_gets_eps_init(self):
        """Test variables with TRUE negative exponents get EPS_INIT."""
        from ssys.recaster import EPS_INIT, parse_antimony_via_sbml, recast_to_ssystem

        # Same model - Z_1 should get EPS_INIT because it has
        # true negative exponents that don't cancel
        text = """
        model test
        x' = -y - z
        y' = a*y + x
        z' = b*z - c*z + x*z
        a = 0.343
        b = 1.82
        c = 9.75
        x = 0
        y = 1
        z = 0
        end
        """

        sym = parse_antimony_via_sbml(text)
        result = recast_to_ssystem(sym)

        # Z_1's decay has Z_2^-1 which doesn't cancel
        # So Z_1 should get EPS_INIT (x=0 and Z_1 has neg exp)

        # Look up Z_1 by name (symbol object identity may differ)
        initials_by_name = {k.name: v for k, v in result.initials.items()}

        # Z_1 should have EPS_INIT (true negative exponent)
        assert "Z_1" in initials_by_name
        z1_ic = initials_by_name["Z_1"]
        assert abs(z1_ic - EPS_INIT) < 1e-12, \
            f"Z_1 IC should be EPS_INIT ({EPS_INIT}), got {z1_ic}"


class TestSymbolicExponents:
    """Tests for handling symbolic exponents in pool construction.

    Regression test for bug where expand_exponents_via_factor_map failed
    with TypeError when exponents were symbolic expressions (parameters)
    rather than numeric values. isinstance(exp, sp.Expr) matched all SymPy
    expressions including symbolic ones like 'h', but float(exp) failed.
    """

    def test_symbolic_exponent_in_pool_construction(self):
        """Test pool construction handles symbolic exponents without crashing."""
        from ssys.recaster import parse_antimony_via_sbml, recast_to_ssystem

        # Model with symbolic exponent 'h' (Hill coefficient)
        # This caused TypeError before the fix because float(h) fails
        text = """
        model test
        X' = k * X^h + c * X - d * X
        k = 1.0
        c = 0.25
        d = 0.5
        h = 2.0
        X = 1.0
        end
        """

        sym = parse_antimony_via_sbml(text)

        result = recast_to_ssystem(sym)

        X = sp.Symbol("X", positive=True)
        h = sp.Symbol("h", positive=True)
        assert [var.name for var in result.factor_map[X]] == ["Z_1", "Z_2", "Z_3"]
        assert result.auxiliary_defs == {}
        assert any(h in eq.growth[1].values() for eq in result.equations)
        assert any(
            eq.growth[1].get(result.factor_map[X][1]) == -1.0
            for eq in result.equations
        )

    def test_symbolic_exponent_in_complex_model(self):
        """Test symbolic exponents in realistic model (bistable gene switch)."""
        from ssys.recaster import parse_antimony_via_sbml, recast_to_ssystem

        # Simplified bistable gene model with Hill function
        text = """
        model test
        // X regulated by Y with Hill kinetics
        X' = V_max * Y^h / (K^h + Y^h) - d * X
        Y' = X - d * Y

        V_max = 10.0
        K = 1.0
        h = 2.0
        d = 0.5
        X = 1.0
        Y = 1.0
        end
        """

        sym = parse_antimony_via_sbml(text)

        result = recast_to_ssystem(sym)

        X = sp.Symbol("X", positive=True)
        Y = sp.Symbol("Y", positive=True)
        h = sp.Symbol("h", positive=True)
        Y_1 = sp.Symbol("Y_1", positive=True)
        assert result.factor_map[X] == [X]
        assert result.factor_map[Y] == [Y]
        assert result.auxiliary_defs == {Y_1: sp.Symbol("K", positive=True) ** h + Y**h}
        assert any(h in exps.values() for eq in result.equations for exps in [eq.growth[1], eq.decay[1]])


class TestCanonicalModeFormatting:
    """Tests for canonical mode S-system formatting."""

    def test_assignment_rules_in_canonical_output(self):
        """Test that assignment rules appear in canonical Antimony output."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        T = sp.Symbol("T", positive=True)
        k = sp.Symbol("k", positive=True)

        # Create system with assignment rule (like time-dependent beta)
        sym = SymSystem(
            vars=[X, T],
            params={"k0": 0.5, "mu": 0.0},  # mu = 0 is the initial value
            odes={X: k * X, T: sp.Integer(1)},  # T' = 1 (clock)
            initials={X: 1.0, T: 0.0},
            assignment_rules={"mu": "k0 * (1 + T)"},  # mu depends on time
        )

        result = recast_to_ssystem(sym, mode='canonical')
        output = ssystem_to_antimony(result, model_name="test", mode='canonical')

        # Assignment rule should appear in output
        assert "mu :=" in output, "Assignment rule 'mu' should appear in canonical output"
        assert "k0" in output, "Assignment rule should reference k0 parameter"

    def test_identity_mapping_skipped(self):
        """Test that X := X identity mappings are skipped to prevent loop errors.

        Regression test for bug where canonical mode output X := X caused
        Antimony "Loop detected" error. Identity mappings should be skipped.
        """
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        # Simple system where X maps to itself (no pool construction needed)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
        )

        result = recast_to_ssystem(sym, mode='canonical')
        output = ssystem_to_antimony(result, model_name="test", mode='canonical')

        # Should NOT contain X := X (that causes loop error)
        # Use regex to match X followed by := and X (with any whitespace)
        import re
        has_identity_loop = re.search(r'\bX\s*:=\s*X\s*;', output)
        assert not has_identity_loop, \
            f"Output should NOT contain 'X := X' identity mapping: {output}"

    def test_eps_slack_in_canonical_output(self):
        """Test that eps_slack appears in canonical output and uses user value."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        # System with pure decay (needs epsilon slack)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},  # Pure decay, growth is 0
            initials={X: 1.0},
            eps_slack=1e-8,  # User-specified eps_slack
        )

        result = recast_to_ssystem(sym, mode='canonical')
        output = ssystem_to_antimony(result, model_name="test", mode='canonical')

        # Should contain epsilon declaration with user value
        assert "epsilon = 1e-08" in output, \
            f"Should have 'epsilon = 1e-08' in output: {output}"

        # Should also appear in @SIM metadata
        assert "EPS_SLACK=1e-08" in output, \
            f"Should have 'EPS_SLACK=1e-08' in @SIM metadata: {output}"

    def test_eps_slack_default_value_in_canonical(self):
        """Test that default eps_slack (1.0) is used when not specified."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        # System with pure decay (needs epsilon slack), no eps_slack specified
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},  # Pure decay, growth is 0
            initials={X: 1.0},
            # eps_slack not specified - should use default 1.0
        )

        result = recast_to_ssystem(sym, mode='canonical')
        output = ssystem_to_antimony(result, model_name="test", mode='canonical')

        # Should contain epsilon declaration with default value
        assert "epsilon = 1" in output, \
            f"Should have 'epsilon = 1' (default) in output: {output}"

    def test_observable_variables_in_canonical(self):
        """Test that observable variables are defined for non-trivial mappings."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        a = sp.Symbol("a", positive=True)
        b = sp.Symbol("b", positive=True)
        c = sp.Symbol("c", positive=True)

        # Multi-production ODE creates pool variables with a non-trivial mapping.
        sym = SymSystem(
            vars=[X],
            params={"a": 1.0, "b": 0.5, "c": 0.2},
            odes={X: a * X + c * X**3 - b * X**2},
            initials={X: 1.0},
        )

        result = recast_to_ssystem(sym, mode='canonical')
        output = ssystem_to_antimony(result, model_name="test", mode='canonical')

        # Should have observable definition for X (non-trivial mapping)
        # X := Z_1 * Z_2 (or similar)
        assert "X :=" in output, \
            f"Should have 'X :=' observable definition: {output}"
        assert "Observable" in output, \
            f"Should have 'Observable' comment section: {output}"

    def test_eps_slack_propagation_through_recast(self):
        """Test eps_slack is propagated through recast_to_ssystem."""
        from ssys.recaster import SymSystem, recast_to_ssystem

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
            eps_slack=1e-10,
        )

        result = recast_to_ssystem(sym, mode='canonical')

        assert result.eps_slack == 1e-10, \
            f"eps_slack should propagate to result: {result.eps_slack}"


class TestReservedKeywordSanitization:
    """Tests for Antimony reserved keyword sanitization.

    Regression tests for bug where compartment names like 'compartment' caused
    parsing errors: "unexpected 'compartment', expecting '$' or element name"
    """

    def test_sanitize_antimony_name_reserved_keyword(self):
        """Test that empirically-problematic keywords get _var suffix."""
        from ssys.recaster import _sanitize_antimony_name

        # Only keywords that have caused real-world parsing errors in BioModels
        assert _sanitize_antimony_name("compartment") == "compartment_var"
        assert _sanitize_antimony_name("DNA") == "DNA_var"
        assert _sanitize_antimony_name("RNA") == "RNA_var"

    def test_sanitize_antimony_name_reserved_function_identifiers(self):
        """Test that function names are sanitized when used as identifiers."""
        from ssys.recaster import _sanitize_antimony_name

        assert _sanitize_antimony_name("exp") == "exp_var"
        assert _sanitize_antimony_name("log") == "log_var"
        assert _sanitize_antimony_name("sin") == "sin_var"
        assert _sanitize_antimony_name("pow") == "pow_var"
        assert _sanitize_antimony_name("time") == "time"

    def test_sanitize_antimony_name_safe_names(self):
        """Test that safe names are not modified."""
        from ssys.recaster import _sanitize_antimony_name

        # Safe names should pass through unchanged
        assert _sanitize_antimony_name("X") == "X"
        assert _sanitize_antimony_name("cell") == "cell"
        assert _sanitize_antimony_name("k1") == "k1"
        assert _sanitize_antimony_name("my_compartment") == "my_compartment"

    def test_sanitize_antimony_name_case_insensitive(self):
        """Test that sanitization is case-insensitive."""
        from ssys.recaster import _sanitize_antimony_name

        # Antimony is case-insensitive for keywords
        assert _sanitize_antimony_name("COMPARTMENT") == "COMPARTMENT_var"
        assert _sanitize_antimony_name("Compartment") == "Compartment_var"
        assert _sanitize_antimony_name("rna") == "rna_var"

    def test_build_name_sanitization_map(self):
        """Test building sanitization map for multiple names."""
        from ssys.recaster import _build_name_sanitization_map

        names = {"X", "compartment", "k1", "DNA", "cell"}
        name_map = _build_name_sanitization_map(names)

        # Only empirically-problematic keywords should be in the map
        assert "compartment" in name_map
        assert "DNA" in name_map
        assert name_map["compartment"] == "compartment_var"
        assert name_map["DNA"] == "DNA_var"

        # Safe names should NOT be in the map
        assert "X" not in name_map
        assert "k1" not in name_map
        assert "cell" not in name_map

    def test_apply_name_sanitization_expression(self):
        """Test applying sanitization to expression strings."""
        from ssys.recaster import _apply_name_sanitization

        name_map = {"compartment": "compartment_var", "exp": "exp_var"}

        # Test expression sanitization
        result = _apply_name_sanitization("compartment + X", name_map)
        assert result == "compartment_var + X"

        # Test that partial matches are not replaced (word boundary)
        result = _apply_name_sanitization("compartmental", name_map)
        assert result == "compartmental"  # Not "compartment_varal"

        # Test that actual function calls are not rewritten.
        result = _apply_name_sanitization("exp(X) + exp", name_map)
        assert result == "exp(X) + exp_var"

    def test_compartment_sanitization_in_output(self):
        """Test that compartment 'compartment' is sanitized in Antimony output."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        k = sp.Symbol("k", positive=True)

        # System with compartment named 'compartment' (the reserved keyword)
        sym = SymSystem(
            vars=[X],
            params={"k": 0.5},
            odes={X: -k * X},
            initials={X: 1.0},
            compartments={"compartment": 1.0},  # Reserved keyword as name
        )

        result = recast_to_ssystem(sym)
        output = ssystem_to_antimony(result, model_name="test")

        # Should have CORRECT Antimony syntax:
        # "compartment compartment_var = 1" where:
        # - First "compartment" is the Antimony keyword (preserved)
        # - "compartment_var" is the sanitized identifier (renamed)
        assert "compartment compartment_var = 1" in output, \
            f"Should have 'compartment compartment_var = 1' (keyword + sanitized name): {output}"
        # Should NOT have the original invalid syntax "compartment compartment = 1"
        assert "compartment compartment = 1" not in output, \
            f"Should NOT have 'compartment compartment': {output}"
        # Should NOT corrupt the keyword: "compartment_var compartment_var"
        assert "compartment_var compartment_var" not in output, \
            f"Should NOT corrupt keyword to 'compartment_var compartment_var': {output}"

    def test_dna_keyword_sanitization_in_output(self):
        """Test that 'DNA' as a param name is sanitized in Antimony output."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        dna_param = sp.Symbol("DNA", positive=True)  # Reserved keyword

        # System with parameter named 'DNA' (empirically-problematic keyword)
        sym = SymSystem(
            vars=[X],
            params={"DNA": 0.5},  # Reserved keyword as param
            odes={X: -dna_param * X},
            initials={X: 1.0},
        )

        result = recast_to_ssystem(sym)
        output = ssystem_to_antimony(result, model_name="test")

        # Should have sanitized parameter name
        assert "DNA_var = 0.5" in output, \
            f"Should sanitize 'DNA' to 'DNA_var': {output}"


class TestDnaRnaSanitization:
    """Tests for DNA/RNA reserved keyword sanitization."""

    def test_dna_rna_keywords_sanitized(self):
        """Test that DNA and RNA are sanitized to DNA_var and RNA_var."""
        from ssys.recaster import _sanitize_antimony_name

        assert _sanitize_antimony_name("DNA") == "DNA_var"
        assert _sanitize_antimony_name("RNA") == "RNA_var"
        # Case insensitive
        assert _sanitize_antimony_name("dna") == "dna_var"
        assert _sanitize_antimony_name("Dna") == "Dna_var"

    def test_rna_keyword_sanitization_in_output(self):
        """Test that 'RNA' as a param name is sanitized in Antimony output."""
        from ssys.recaster import SymSystem, recast_to_ssystem, ssystem_to_antimony

        X = sp.Symbol("X", positive=True)
        rna_param = sp.Symbol("RNA", positive=True)  # Reserved keyword

        # System with parameter named 'RNA' (empirically-problematic keyword)
        sym = SymSystem(
            vars=[X],
            params={"RNA": 0.5},  # Reserved keyword as param
            odes={X: -rna_param * X},
            initials={X: 1.0},
        )

        result = recast_to_ssystem(sym)
        output = ssystem_to_antimony(result, model_name="test")

        # Should have sanitized parameter name
        assert "RNA_var = 0.5" in output, \
            f"Should sanitize 'RNA' to 'RNA_var': {output}"


class TestVersionConsistency:
    """Tests for version consistency between pyproject.toml and __init__.py."""

    def test_version_matches_pyproject(self):
        """Verify version in pyproject.toml matches __version__ in ssys."""
        # Read version from pyproject.toml using regex (works with Python 3.10+)
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        pyproject_text = pyproject_path.read_text()

        # Extract version from pyproject.toml
        pattern = r'^version\s*=\s*"([^"]+)"'
        match = re.search(pattern, pyproject_text, re.MULTILINE)
        assert match is not None, "Could not find version"
        pyproject_version = match.group(1)

        # Compare with ssys.__version__
        init_version = ssys.__version__

        assert pyproject_version == init_version, (
            f"Version mismatch: pyproject.toml has '{pyproject_version}'"
            f", ssys.__version__ has '{init_version}'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
