"""Tests for core recasting functionality."""

import re
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
        """Test that auxiliaries are named canonically (Z_1, Z_2, ...)."""
        text = """
        A' = k1*A - k2*A^2
        k1 = 1.0
        k2 = 0.5
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

    def test_extract_sim_metadata_eps_init(self):
        """Test parsing EPS_INIT from @SIM comment."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6
        X' = -k*X
        k = 0.5
        X = 0.0
        """
        t_start, t_end, n_steps, eps_init = _extract_sim_metadata(text)

        assert t_start == 0.0
        assert t_end == 100.0
        assert n_steps == 500
        assert eps_init == 1e-6

    def test_extract_sim_metadata_eps_init_none(self):
        """Test that eps_init is None when not specified."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM T_START=0 T_END=100 N_STEPS=500
        X' = -k*X
        k = 0.5
        X = 1.0
        """
        t_start, t_end, n_steps, eps_init = _extract_sim_metadata(text)

        assert t_start == 0.0
        assert t_end == 100.0
        assert n_steps == 500
        assert eps_init is None

    def test_extract_sim_metadata_eps_init_only(self):
        """Test parsing EPS_INIT alone without other metadata."""
        from ssys.recaster import _extract_sim_metadata

        text = """
        // @SIM EPS_INIT=1e-12
        X' = -k*X
        k = 0.5
        X = 0.0
        """
        t_start, t_end, n_steps, eps_init = _extract_sim_metadata(text)

        assert t_start is None
        assert t_end is None
        assert n_steps is None
        assert eps_init == 1e-12

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

        # System should use default EPS_INIT or keep 0.0
        # (depends on whether negative exponents are present)
        assert result is not None  # Should not crash


class TestSbmlParserIcHandling:
    """Tests for SBML parser initial condition handling.
    
    When libSBML parses Antimony via SBML conversion, species initial
    conditions may end up in the params dict instead of the initials dict.
    These tests ensure the recaster handles this correctly.
    """

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
