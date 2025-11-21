"""Tests for core recasting functionality."""

import pytest
import sympy as sp

from ssys import (
    parse_antimony,
    build_sym_system,
    recast_to_ssystem,
    ssystem_to_antimony,
)


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
        """Test recasting ODE with two terms."""
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
        # Two terms mean two auxiliaries
        assert len(rec.factor_map[X]) == 2
        assert len(rec.equations) == 2

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
        """Test that auxiliaries are named canonically (X_1, X_2, ...)."""
        text = """
        A' = k1*A - k2*A
        k1 = 1.0
        k2 = 0.5
        A = 1.0
        """
        ir = parse_antimony(text)
        sym = build_sym_system(ir)
        rec = recast_to_ssystem(sym)

        # Check that auxiliary names follow X_1, X_2 pattern
        var_names = [v.name for v in rec.variables]
        assert "X_1" in var_names
        assert "X_2" in var_names


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
        # Should contain auxiliary variable definitions
        assert "X_1" in output


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
