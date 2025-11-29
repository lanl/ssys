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
    
    if options is None:
        options = {}
    
    try:
        # Get or reconstruct Antimony text
        antimony_text = _get_antimony_text(model_ir)
        
        # Build RoadRunner model
        r = rr.RoadRunner()
        r.load(antimony_text)
        
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
        state_names = r.getFloatingSpeciesIds()
        
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
    """
    # Check if antimony_text was cached during parsing
    if hasattr(model_ir, 'antimony_text') and model_ir.antimony_text:
        return model_ir.antimony_text
    
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
