"""
libRoadRunner ODE solver backend.
"""

from typing import Dict, Optional, Any
import numpy as np
from ..recaster import ModelIR


def simulate_with_roadrunner(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: Optional[Dict[str, float]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simulate using libRoadRunner (CVODE integrator).
    
    Args:
        model_ir: Model intermediate representation
        t0: Start time
        t_end: End time
        n_points: Number of time points
        y0_override: Override initial conditions
        options: Solver options:
            - integrator: 'cvode' (default) or 'gillespie'
            - absolute_tolerance: float (default 1e-9)
            - relative_tolerance: float (default 1e-6)
            - maximum_num_steps: int (default 20000)
            - log_solver_details: bool (default False)
            
    Returns:
        Dictionary with simulation results
    """
    try:
        import roadrunner as rr
    except ImportError:
        raise ImportError(
            "libRoadRunner not installed. "
            "Install with: pip install libroadrunner"
        )
    
    try:
        import antimony
    except ImportError:
        raise ImportError(
            "Antimony not installed. "
            "Install with: pip install antimony (or pip install tellurium)"
        )
    
    if options is None:
        options = {}
    
    try:
        # Get or reconstruct Antimony text
        antimony_text = _get_antimony_text(model_ir)
        
        # Convert Antimony → SBML using antimony library
        # RoadRunner only understands SBML, not Antimony directly
        antimony.clearPreviousLoads()
        rc = antimony.loadAntimonyString(antimony_text)
        if rc < 0:
            raise RuntimeError(
                f"Antimony parse error: {antimony.getLastError()}"
            )
        
        model_name = antimony.getMainModuleName()
        if not model_name:
            raise RuntimeError("Could not get Antimony module name")
        
        sbml_str = antimony.getSBMLString(model_name)
        if not sbml_str:
            raise RuntimeError(
                f"Antimony→SBML conversion failed: {antimony.getLastError()}"
            )
        
        # Build RoadRunner model from SBML
        r = rr.RoadRunner(sbml_str)
        
        # Configure integrator
        integrator_name = options.get("integrator", "cvode")
        r.setIntegrator(integrator_name)
        
        if integrator_name == "cvode":
            r.integrator.absolute_tolerance = options.get(
                "absolute_tolerance", 1e-9
            )
            r.integrator.relative_tolerance = options.get(
                "relative_tolerance", 1e-6
            )
            r.integrator.maximum_num_steps = options.get(
                "maximum_num_steps", 20000
            )
        
        # Set initial conditions
        _set_initial_conditions(r, model_ir, y0_override)
        
        # Run simulation
        result = r.simulate(t0, t_end, n_points)
        
        # Extract results
        # result is array with shape (n_points, 1 + n_species)
        # Column 0 is time, rest are species
        t = result[:, 0]
        y = result[:, 1:]
        
        # Get species names from result column headers
        # CRITICAL: This ensures state_names match the actual column order
        # in the simulation result (getFloatingSpeciesIds() may not match!)
        col_names = list(result.colnames)  # e.g., ['time', 'S', 'I', 'R']
        state_names = col_names[1:]  # Skip 'time' column
        
        # Strip brackets if present (RoadRunner sometimes uses [species])
        state_names = [
            name.strip('[]') if name.startswith('[') else name 
            for name in state_names
        ]
        
        # Get integrator statistics
        stats = {}
        if integrator_name == "cvode":
            try:
                stats = {
                    "n_steps": r.integrator.getNumSteps(),
                    "n_failed_steps": r.integrator.getNumErrTestFails(),
                    "last_time": t[-1]
                }
            except Exception:
                # Some RoadRunner versions may not support these
                stats = {"last_time": t[-1]}
        
        if options.get("log_solver_details", False):
            print("RoadRunner simulation completed:")
            print(f"  Integrator: {integrator_name}")
            print(f"  Steps: {stats.get('n_steps', 'N/A')}")
            print(f"  Time range: [{t[0]}, {t[-1]}]")
        
        return {
            "t": t,
            "y": y,
            "state_names": list(state_names),
            "success": True,
            "message": "",
            "integrator_stats": stats
        }
        
    except Exception as e:
        # Simulation failed - return structured error
        return {
            "t": np.array([]),
            "y": np.array([]),
            "state_names": [],
            "success": False,
            "message": f"RoadRunner simulation failed: {str(e)}",
            "integrator_stats": {}
        }


def _get_antimony_text(model_ir: ModelIR) -> str:
    """
    Get Antimony text from ModelIR.
    
    Prefers cached text if available, otherwise reconstructs.
    Applies fixes for Antimony/RoadRunner compatibility:
    - Numeric model names: "model 24_name" → "model m_24_name"
    - Multi-line equations: join continuation lines
    - gamma() function: compute at Python level (Antimony only has incomplete gamma)
    """
    import re
    from math import gamma as math_gamma, pi, sqrt
    
    # Check if antimony_text was cached during parsing
    if hasattr(model_ir, 'antimony_text') and model_ir.antimony_text:
        text = model_ir.antimony_text
        
        # Fix 1: Numeric model names
        text = re.sub(r'^(model\s+)(\d)', r'\1m_\2', text, flags=re.MULTILINE)
        
        # Fix 2: Multi-line equations - join continuation lines
        # Handles: X3' = term1
        #          + term2
        #          - term3;
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if this is an ODE line without semicolon
            if "'" in line and '=' in line and not line.rstrip().endswith(';'):
                # Collect continuation lines
                combined = line.rstrip()
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    # Check if next line starts with operator
                    if next_line and next_line[0] in ['+', '-']:
                        combined += ' ' + next_line
                        i += 1
                        if next_line.rstrip().endswith(';'):
                            break
                    else:
                        break
                fixed_lines.append(combined)
            else:
                fixed_lines.append(line)
                i += 1
        text = '\n'.join(fixed_lines)
        
        # Fix 3: gamma() function - compute values and replace
        # Antimony's gamma() is incomplete gamma (needs 2+ args)
        # We need complete gamma Γ(x)
        def replace_gamma(match):
            """Replace gamma(expr) with computed value."""
            expr_str = match.group(1)
            try:
                # Try to evaluate the expression
                # Common cases in our models
                if 'nu' in expr_str:
                    # Can't evaluate at parse time if it contains variables
                    # For model 17, we'd need to handle this differently
                    # For now, keep original and let it fail gracefully
                    return match.group(0)
                else:
                    # Try direct evaluation
                    result = eval(expr_str, {"pi": pi, "sqrt": sqrt})
                    gamma_val = math_gamma(result)
                    return f"{gamma_val}"
            except Exception:
                # Can't evaluate - keep original
                return match.group(0)
        
        # Match gamma(expression) but not gamma(a, b, ...)
        text = re.sub(r'gamma\(([^,)]+)\)', replace_gamma, text)
        
        # Fix 4: Exponentiation syntax - convert ** to ^
        # Python/SymPy uses ** for exponentiation, Antimony uses ^
        text = text.replace('**', '^')
        
        return text
    
    # Otherwise, reconstruct from IR
    return _reconstruct_antimony(model_ir)


def _reconstruct_antimony(model_ir: ModelIR) -> str:
    """
    Reconstruct Antimony text from ModelIR.
    
    Note: This is a basic reconstruction. For production use,
    consider caching the original Antimony text in ModelIR.
    """
    lines = []
    lines.append("model recast_model()")
    lines.append("")
    
    # Species
    if model_ir.species:
        lines.append("  // Species")
        for species in model_ir.species:
            lines.append(f"  species {species};")
        lines.append("")
    
    # Parameters
    if model_ir.params:
        lines.append("  // Parameters")
        for param, value in model_ir.params.items():
            lines.append(f"  {param} = {value};")
        lines.append("")
    
    # Reactions
    if model_ir.reactions:
        lines.append("  // Reactions")
        for rxn in model_ir.reactions:
            lhs = " + ".join(rxn.reactants) if rxn.reactants else ""
            rhs = " + ".join(rxn.products) if rxn.products else ""
            arrow = "->" if not lhs else "-> " if not rhs else " -> "
            lines.append(
                f"  {rxn.id}: {lhs}{arrow}{rhs}; {rxn.rate_law};"
            )
        lines.append("")
    
    # Explicit rate rules (ODEs)
    if model_ir.explicit_rates:
        lines.append("  // ODEs")
        for species, rate_expr in model_ir.explicit_rates.items():
            lines.append(f"  {species}' = {rate_expr};")
        lines.append("")
    
    lines.append("end")
    return "\n".join(lines)


def _set_initial_conditions(
    r,
    model_ir: ModelIR,
    y0_override: Optional[Dict[str, float]]
):
    """
    Set initial conditions in RoadRunner model.
    
    Args:
        r: RoadRunner instance
        model_ir: Model IR with initial conditions
        y0_override: Optional overrides
    """
    # Reset to model-defined initial conditions
    r.resetToOrigin()
    
    # Apply model_ir initials
    if hasattr(model_ir, 'initials') and model_ir.initials:
        for species, value in model_ir.initials.items():
            try:
                # Use bracket notation for floating species
                r[f'[{species}]'] = value
            except Exception:
                # Fallback without brackets
                try:
                    r[species] = value
                except Exception as e:
                    print(f"Warning: Could not set {species} = {value}: {e}")
    
    # Apply overrides
    if y0_override:
        for species, value in y0_override.items():
            try:
                r[f'[{species}]'] = value
            except Exception:
                try:
                    r[species] = value
                except Exception as e:
                    print(
                        f"Warning: Could not override {species} = "
                        f"{value}: {e}"
                    )
