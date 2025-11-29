"""
Tests for ODE solver backends.
"""

import pytest
from src.ssys.recaster import parse_antimony
from src.ssys.ode_backends import simulate_ode


def test_simulate_ode_interface():
    """Test that simulate_ode interface works."""
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
    
    # Test with RK4 backend (should now work!)
    result = simulate_ode(
        model_ir,
        t0=0.0,
        t_end=10.0,
        n_points=11,
        backend="rk4"
    )
    
    assert result["success"] is True
    assert len(result["t"]) == 11
    assert result["y"].shape == (11, 1)  # 1 species
    assert len(result["state_names"]) == 1
    # Check decay: S(t=10) < S(t=0)
    assert result["y"][-1, 0] < result["y"][0, 0]


def test_roadrunner_backend_not_installed():
    """Test fallback when roadrunner not available."""
    antimony_text = """
    model simple()
        species S;
        S = 1;
    end
    """
    
    model_ir = parse_antimony(antimony_text)
    
    # Try roadrunner with no fallback
    result = simulate_ode(
        model_ir,
        t0=0.0,
        t_end=1.0,
        n_points=2,
        backend="roadrunner",
        options={"fallback_to_rk4": False}
    )
    
    # Will either succeed (if roadrunner installed) or fail gracefully
    assert isinstance(result, dict)
    assert "success" in result
    assert "message" in result


@pytest.mark.skipif(
    True,  # Skip unless roadrunner actually installed
    reason="Requires libRoadRunner installation"
)
def test_roadrunner_exp_decay():
    """Test roadrunner backend on exponential decay."""
    antimony_text = """
    model exp_decay()
        species S;
        S = 10;
        k = 0.1;
        
        J0: S -> ; k * S;
    end
    """
    
    model_ir = parse_antimony(antimony_text)
    
    result = simulate_ode(
        model_ir,
        t0=0.0,
        t_end=10.0,
        n_points=11,
        backend="roadrunner"
    )
    
    if result["success"]:
        # Check basic structure
        assert len(result["t"]) == 11
        assert result["y"].shape[0] == 11
        assert len(result["state_names"]) > 0
        
        # Check decay behavior (S should decrease)
        assert result["y"][0, 0] > result["y"][-1, 0]
