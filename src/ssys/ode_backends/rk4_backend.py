"""
RK4 ODE solver backend (fallback implementation).
"""

from typing import Dict, Optional, Any
import numpy as np
from sympy import lambdify
from ..recaster import ModelIR, build_sym_system


def simulate_with_rk4(
    model_ir: ModelIR,
    t0: float,
    t_end: float,
    n_points: int,
    y0_override: Optional[Dict[str, float]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simulate using RK4 integrator (4th order Runge-Kutta).
    
    Args:
        model_ir: Model intermediate representation
        t0: Start time
        t_end: End time
        n_points: Number of time points
        y0_override: Override initial conditions
        options: Solver options:
            - log_solver_details: bool (default False)
        
    Returns:
        Dictionary with simulation results
    """
    if options is None:
        options = {}
    
    try:
        # Build symbolic system
        sym_sys = build_sym_system(model_ir)
        
        # Get state variables in consistent order
        state_vars = sorted(sym_sys.odes.keys(), key=str)
        state_names = [str(v) for v in state_vars]
        
        # Build initial conditions
        y0 = []
        for var in state_vars:
            if y0_override and str(var) in y0_override:
                y0.append(float(y0_override[str(var)]))
            elif var in sym_sys.initials:
                # Get initial value (may already be float or symbolic)
                init_val = sym_sys.initials[var]
                if hasattr(init_val, 'subs'):
                    # Symbolic expression - substitute params
                    init_val = float(init_val.subs(model_ir.params))
                else:
                    # Already numeric
                    init_val = float(init_val)
                y0.append(init_val)
            else:
                y0.append(0.0)  # Default
        
        # Build RHS function
        f = _build_rhs_function(state_vars, sym_sys.odes, model_ir.params)
        
        # Run RK4
        t, y = _rk4_integrate(
            f,
            (t0, t_end),
            np.array(y0, dtype=float),
            n_points - 1  # n_steps = n_points - 1
        )
        
        if options.get("log_solver_details", False):
            print("RK4 simulation completed:")
            print(f"  Steps: {n_points - 1}")
            print(f"  Time range: [{t[0]}, {t[-1]}]")
        
        return {
            "t": t,
            "y": y,
            "state_names": state_names,
            "success": True,
            "message": "",
            "integrator_stats": {
                "n_steps": n_points - 1,
                "last_time": t[-1]
            }
        }
        
    except Exception as e:
        return {
            "t": np.array([]),
            "y": np.array([]),
            "state_names": [],
            "success": False,
            "message": f"RK4 simulation failed: {str(e)}",
            "integrator_stats": {}
        }


def _build_rhs_function(state_vars, odes, params):
    """
    Build numerical RHS function from symbolic ODEs.
    
    Args:
        state_vars: List of state variable symbols
        odes: Dict mapping state vars to their ODEs
        params: Dict of parameter values (string keys)
        
    Returns:
        Function f(t, y) -> dydt
    """
    # Build param substitution dict by matching symbol names
    # Find all parameter symbols used in ODEs
    all_syms = set()
    for ode_expr in odes.values():
        all_syms.update(ode_expr.free_symbols)
    
    # Match param symbols by name
    param_subs = {}
    for sym in all_syms:
        if str(sym) in params:
            param_subs[sym] = params[str(sym)]
    
    # Build lambdified function for each ODE
    rhs_funcs = []
    for var in state_vars:
        ode_expr = odes[var]
        # Substitute parameter values, then lambdify
        ode_numeric = ode_expr.subs(param_subs)
        func = lambdify(state_vars, ode_numeric, modules='numpy')
        rhs_funcs.append(func)
    
    def f(t, y):
        """RHS function: dydt = f(t, y)"""
        # Evaluate each ODE with current state
        dydt = np.array([func(*y) for func in rhs_funcs], dtype=float)
        return dydt
    
    return f


def _rk4_integrate(f, t_span, y0, n_steps):
    """
    4th order Runge-Kutta integrator.
    
    Args:
        f: RHS function f(t, y) -> dydt
        t_span: (t0, t_end) time interval
        y0: Initial conditions array
        n_steps: Number of integration steps
        
    Returns:
        t: Time array (n_steps + 1,)
        y: State array (n_steps + 1, n_vars)
    """
    t0, t1 = t_span
    t = np.linspace(t0, t1, n_steps + 1)
    h = (t1 - t0) / n_steps
    
    y = np.zeros((len(t), len(y0)), dtype=float)
    y[0] = np.array(y0, dtype=float)
    
    for i in range(n_steps):
        ti = t[i]
        yi = y[i]
        
        k1 = f(ti, yi)
        k2 = f(ti + 0.5*h, yi + 0.5*h*k1)
        k3 = f(ti + 0.5*h, yi + 0.5*h*k2)
        k4 = f(ti + h, yi + h*k3)
        
        y[i+1] = yi + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    return t, y
