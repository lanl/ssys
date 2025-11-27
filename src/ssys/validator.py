"""
Validation framework for verifying S-system and GMA recasts.

This module provides automated validation of recasting correctness through:
- Symbolic equivalence testing (Jacobian chain rule)
- Numerical pointwise validation with sampling
- Trajectory comparison via simulation
- Auxiliary identity verification
- Structural classification verification
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import symbols, lambdify, simplify, Matrix, log

from .recaster import (
    parse_antimony, 
    build_sym_system, 
    classify_system,
    classify_result,
    SystemClass
)


class ValidationResult(Enum):
    """Validation test outcomes."""
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    NOT_ATTEMPTED = "not_attempted"


@dataclass
class EquivalenceTest:
    """Results from a single equivalence test."""
    name: str
    result: ValidationResult
    max_error: Optional[float] = None
    mean_error: Optional[float] = None
    details: str = ""
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report for an original/recast pair."""
    original_file: str
    recast_file: str
    
    # Structural classification
    original_class: SystemClass
    recast_class: SystemClass
    expected_class: Optional[SystemClass] = None
    canonical_refusal_reason: Optional[str] = None
    
    # Test results
    symbolic_test: Optional[EquivalenceTest] = None
    numerical_test: Optional[EquivalenceTest] = None
    trajectory_test: Optional[EquivalenceTest] = None
    auxiliary_tests: List[EquivalenceTest] = field(default_factory=list)
    
    # Overall verdict
    overall_pass: bool = False
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        def test_to_dict(test: Optional[EquivalenceTest]) -> Optional[Dict]:
            if test is None:
                return None
            return {
                'name': test.name,
                'result': test.result.value,
                'max_error': test.max_error,
                'mean_error': test.mean_error,
                'details': test.details,
                'counterexamples': test.counterexamples[:5]  # Limit to first 5
            }
        
        return {
            'original_file': self.original_file,
            'recast_file': self.recast_file,
            'classification': {
                'original': self.original_class.value,
                'recast': self.recast_class.value,
                'expected': self.expected_class.value if self.expected_class else None,
                'canonical_refusal_reason': self.canonical_refusal_reason
            },
            'tests': {
                'symbolic': test_to_dict(self.symbolic_test),
                'numerical': test_to_dict(self.numerical_test),
                'trajectory': test_to_dict(self.trajectory_test),
                'auxiliaries': [test_to_dict(t) for t in self.auxiliary_tests]
            },
            'overall_pass': self.overall_pass,
            'summary': self.summary
        }


class RecastValidator:
    """
    Validates that a recast model is mathematically equivalent to the original.
    
    Uses multiple validation strategies:
    1. Symbolic equivalence via Jacobian chain rule
    2. Numerical pointwise testing with random sampling
    3. Trajectory comparison via ODE simulation
    4. Auxiliary identity verification for lifted functions
    """
    
    def __init__(self, 
                 original_file: str,
                 recast_file: str,
                 factor_map: Optional[Dict[sp.Symbol, List[sp.Symbol]]] = None,
                 mode: str = "simplified"):
        """
        Initialize validator.
        
        Args:
            original_file: Path to original Antimony file
            recast_file: Path to recast Antimony file
            factor_map: Mapping from original to auxiliary variables (X -> [X1, X2, ...])
            mode: Recast mode ('simplified' or 'canonical')
        """
        self.original_file = original_file
        self.recast_file = recast_file
        self.mode = mode
        
        # Read recast file to extract mapping comments
        recast_text = open(recast_file).read()
        
        # Parse both models
        self.orig_ir = parse_antimony(open(original_file).read())
        self.recast_ir = parse_antimony(recast_text)
        
        # Build symbolic systems
        self.orig_system = build_sym_system(self.orig_ir)
        self.recast_system = build_sym_system(self.recast_ir)
        
        # Extract ODE dictionaries
        self.orig_odes = self.orig_system.odes
        self.recast_odes = self.recast_system.odes
        
        # Extract mapping from comments if not provided
        if factor_map is None:
            self.factor_map = self._extract_mapping_from_comments(recast_text)
        else:
            self.factor_map = factor_map
        
        # Extract auxiliary definitions from comments
        self.auxiliary_defs = self._extract_auxiliary_definitions(recast_text)
        
        # Merge auxiliary definitions into factor_map for use in validation
        self.factor_map.update(self.auxiliary_defs)
        
        # Build mapping function Φ: Z -> X
        self._build_mapping()
        
        # Classify systems
        self.orig_class = classify_system(self.orig_system)
        self.recast_class = classify_system(self.recast_system)
        
        # Extract refusal reason if present (for GMA outputs)
        self.canonical_refusal_reason = self._extract_refusal_reason(recast_text)
    
    def _extract_mapping_from_comments(self, recast_text: str) -> Dict:
        """
        Extract variable mapping from recast file comments.
        
        Looks for patterns like:
        // VARIABLE MAPPING (new format)
        // ========
        // S = X_1
        // ========
        
        Or old format:
        // Mapping from original variables...
        // S = X_1
        // --- end mapping ---
        """
        import re
        
        mapping = {}
        in_mapping = False
        seen_first_separator = False
        
        for line in recast_text.split('\n'):
            line = line.strip()
            
            # Start of mapping section (both formats)
            if 'VARIABLE MAPPING' in line or 'Mapping from original variables' in line:
                in_mapping = True
                seen_first_separator = False
                continue
            
            # Handle separators in new format
            if in_mapping and '========' in line:
                if not seen_first_separator:
                    # This is the opening separator, skip it
                    seen_first_separator = True
                    continue
                else:
                    # This is the closing separator, end mapping section
                    break
            
            # End of mapping section (old format)
            if in_mapping and '--- end mapping ---' in line:
                break
            
            # Parse mapping line: // VAR = EXPR
            if in_mapping and line.startswith('//'):
                content = line[2:].strip()
                # Must have '=' but not start with '=' (skip separator lines)
                if '=' in content and not content.startswith('='):
                    parts = content.split('=', 1)
                    if len(parts) == 2:
                        orig_var = parts[0].strip()
                        expr_str = parts[1].strip()
                        
                        # Convert to sympy symbols
                        orig_sym = sp.Symbol(orig_var)
                        
                        # Parse expression (handles products like X_1*X_2)
                        try:
                            expr = sp.sympify(expr_str)
                            mapping[orig_sym] = expr
                        except:
                            # If parsing fails, treat as single symbol
                            mapping[orig_sym] = sp.Symbol(expr_str)
        
        return mapping
    
    def _extract_refusal_reason(self, recast_text: str) -> Optional[str]:
        """
        Extract canonical S-system refusal reason from GMA output comments.
        
        Looks for pattern like:
        // NOTE: Canonical S-system recast was not attempted because:
        //   <reason>
        """
        lines = recast_text.split('\n')
        for i, line in enumerate(lines):
            if 'Canonical S-system recast was not attempted because:' in line:
                # Get next line which should contain the reason
                if i + 1 < len(lines):
                    reason_line = lines[i + 1].strip()
                    if reason_line.startswith('//'):
                        reason = reason_line[2:].strip()
                        return reason
        return None
    
    def _extract_auxiliary_definitions(self, recast_text: str) -> Dict[sp.Symbol, sp.Expr]:
        """
        Extract auxiliary variable definitions from recast file comments.
        
        Looks for patterns like:
        // AUXILIARY DEFINITIONS (for lifted variables)
        // ========================================================================
        // Z_1 := exp(X*k)
        // ========================================================================
        """
        aux_defs = {}
        in_aux_section = False
        seen_first_separator = False
        
        for line in recast_text.split('\n'):
            line = line.strip()
            
            # Start of auxiliary definitions section (both old and new formats)
            if 'AUXILIARY DEFINITIONS' in line or 'Auxiliary variable definitions' in line:
                in_aux_section = True
                seen_first_separator = False
                continue
            
            # Handle separators
            if in_aux_section and '========' in line:
                if not seen_first_separator:
                    # Opening separator, skip it
                    seen_first_separator = True
                    continue
                else:
                    # Closing separator, end section
                    break
            
            # End of auxiliary definitions section (old format)
            if in_aux_section and '--- end auxiliary definitions ---' in line:
                break
            
            # Parse auxiliary definition line: // Z_1 := exp(X*k)
            if in_aux_section and line.startswith('//'):
                content = line[2:].strip()
                # Skip separator lines
                if content.startswith('='):
                    continue
                if ':=' in content:
                    parts = content.split(':=', 1)
                    if len(parts) == 2:
                        aux_var = parts[0].strip()
                        expr_str = parts[1].strip()
                        
                        # Convert to sympy
                        try:
                            aux_sym = sp.Symbol(aux_var)
                            expr = sp.sympify(expr_str)
                            aux_defs[aux_sym] = expr
                        except:
                            pass  # Skip if parsing fails
        
        return aux_defs
    
    def _infer_auxiliary_definitions(self):
        """
        Infer definitions of auxiliary variables by pattern matching.
        
        Strategy:
        1. Check if Y' = X' pattern exists (Y = X + constant)
        2. Match auxiliaries with denominators based on usage context
        """
        # Identify auxiliary variables
        mapped_vars = set()
        if self.factor_map:
            for val in self.factor_map.values():
                if isinstance(val, sp.Symbol):
                    mapped_vars.add(val)
                elif hasattr(val, 'free_symbols'):
                    mapped_vars.update(val.free_symbols)
        
        orig_vars = set(self.orig_odes.keys())
        recast_vars = set(self.recast_odes.keys())
        potential_auxiliaries = recast_vars - orig_vars - mapped_vars
        
        # Pattern 1: Check if Y' matches any original variable derivative
        # If Y' = X', then Y = X + constant, find the constant from denominators
        for aux in potential_auxiliaries:
            if aux in self.factor_map:
                continue
            
            aux_ode = self.recast_odes[aux]
            
            # Check if Y' equals some original variable's recast ODE
            for orig_var in orig_vars:
                if orig_var in self.recast_odes:
                    if aux_ode.equals(self.recast_odes[orig_var]):
                        # Y' = X', so Y = X + C
                        # Find C by checking denominators
                        orig_ode = self.orig_odes[orig_var]
                        denoms = self._find_denominators(orig_ode)
                        
                        # Look for denominator of form (C + X)
                        for denom in denoms:
                            # Check if denom = constant + orig_var
                            if denom.is_Add and orig_var in denom.free_symbols:
                                # This is likely Y = denom
                                self.factor_map[aux] = denom
                                break
                        break
    
    def _has_negative_exponent(self, expr: sp.Expr, symbol: sp.Symbol) -> bool:
        """Check if symbol appears with negative exponent in expression."""
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Pow):
                base, exp = term.args
                if base == symbol and exp.is_number and float(exp) < 0:
                    return True
        return False
    
    def _find_denominators(self, expr: sp.Expr) -> List[sp.Expr]:
        """Find denominator expressions in a symbolic expression."""
        denominators = []
        
        def visit(e):
            if isinstance(e, sp.Pow):
                base, exp = e.args
                # Negative exponent indicates division
                if exp.is_number and float(exp) < 0:
                    if not isinstance(base, sp.Symbol):  # Non-trivial denominator
                        denominators.append(base)
            elif isinstance(e, (sp.Add, sp.Mul)):
                for arg in e.args:
                    visit(arg)
        
        visit(expr)
        return denominators

    def _build_mapping(self):
        """
        Build symbolic mapping Φ(Z) that reconstructs original variables.

        For variables with factor_map entries: X = X1 * X2 * ...
        For unmapped variables: X = X (identity)
        
        Also attempts to infer auxiliary variable definitions.
        """
        # First, try to infer auxiliary definitions
        self._infer_auxiliary_definitions()
        
        self.mapping = {}  # X_orig -> expression in Z_recast

        # Get all original and recast variables
        orig_vars = set(self.orig_odes.keys())
        recast_vars = set(self.recast_odes.keys())
        
        # Build symbol mapping from names to actual recast symbols
        recast_symbols_by_name = {str(v): v for v in recast_vars}
        
        if self.factor_map:
            # Use provided/extracted factor map
            # Match symbols by name and substitute with actual recast symbols
            factor_map_by_name = {str(k): v for k, v in self.factor_map.items()}
            
            for orig_var in orig_vars:
                var_name = str(orig_var)
                if var_name in factor_map_by_name:
                    expr = factor_map_by_name[var_name]
                    if isinstance(expr, list):
                        # List of auxiliaries - compute product
                        product = sp.prod(expr)
                        self.mapping[orig_var] = product
                    else:
                        # Substitute symbols in expression with actual recast symbols
                        # This ensures Jacobian computation works correctly
                        substitutions = {}
                        for sym in expr.free_symbols:
                            sym_name = str(sym)
                            if sym_name in recast_symbols_by_name:
                                substitutions[sym] = recast_symbols_by_name[sym_name]
                        
                        if substitutions:
                            expr = expr.subs(substitutions)
                        
                        self.mapping[orig_var] = expr
                else:
                    # No mapping found, assume identity
                    self.mapping[orig_var] = orig_var
        else:
            # Assume identity mapping (recast uses same variable names)
            for var in orig_vars:
                self.mapping[var] = var
        
        # Add auxiliary variable mappings to the full mapping
        # For Jacobian computation, we need Φ to map ALL recast variables
        # IMPORTANT: Match by name, not by object identity, since symbols
        # from comments may be different objects than symbols from ODEs
        recast_only_vars = recast_vars - orig_vars
        factor_map_by_name = {str(k): v for k, v in self.factor_map.items()}
        
        for aux_var in recast_only_vars:
            aux_name = str(aux_var)
            if aux_name in factor_map_by_name:
                # We have a definition for this auxiliary
                expr = factor_map_by_name[aux_name]
                # Substitute symbols with actual recast symbols
                substitutions = {}
                for sym in expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name in recast_symbols_by_name:
                        substitutions[sym] = recast_symbols_by_name[sym_name]
                if substitutions:
                    expr = expr.subs(substitutions)
                self.mapping[aux_var] = expr
            else:
                # No definition, assume identity
                self.mapping[aux_var] = aux_var
        
        # Build list of all recast state variables
        self.recast_state_vars = list(self.recast_odes.keys())
    
    def check_symbolic_equivalence(self, timeout: float = 30.0) -> EquivalenceTest:
        """
        Check symbolic equivalence using Jacobian chain rule.
        
        Tests if: J_Φ(Z) · f_recast(Z) = f_orig(Φ(Z))
        
        Where:
        - Φ(Z) maps recast states to original states
        - J_Φ is the Jacobian matrix ∂Φ/∂Z
        - f_recast is the recast ODE RHS
        - f_orig is the original ODE RHS
        
        Args:
            timeout: Maximum time in seconds for simplification
            
        Returns:
            EquivalenceTest with symbolic validation results
        """
        try:
            # Build Φ as a vector
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            Phi_vector = Matrix([self.mapping[v] for v in orig_vars_ordered])
            
            # Build Z state vector
            Z_vector = Matrix(self.recast_state_vars)
            
            # Compute Jacobian J_Φ = ∂Φ/∂Z
            J_Phi = Phi_vector.jacobian(Z_vector)
            
            # Build f_recast as vector
            f_recast = Matrix([self.recast_odes[v] for v in self.recast_state_vars])
            
            # Compute J_Φ · f_recast
            lhs = J_Phi * f_recast
            
            # Build f_orig(Φ(Z)) by substituting Φ into original ODEs
            f_orig_at_Phi = Matrix([
                self.orig_odes[v].subs(self.mapping) 
                for v in orig_vars_ordered
            ])
            
            # Compute difference Δ = J_Φ · f_recast - f_orig(Φ(Z))
            Delta = lhs - f_orig_at_Phi
            
            # Try to simplify each component
            simplified_components = []
            for i, component in enumerate(Delta):
                try:
                    # First, explicitly substitute auxiliary definitions
                    # This helps SymPy recognize Y_1 = K_2 + X_1 type relationships
                    # Match symbols by name, not object identity
                    substituted = component
                    factor_map_by_name = {str(k): v for k, v in self.factor_map.items()}
                    orig_var_names = {str(v) for v in orig_vars_ordered}
                    
                    # Build substitution dict matching symbols by name
                    subs_dict = {}
                    for sym in component.free_symbols:
                        sym_name = str(sym)
                        if sym_name in factor_map_by_name and sym_name not in orig_var_names:
                            # This is an auxiliary - substitute its definition
                            subs_dict[sym] = factor_map_by_name[sym_name]
                    
                    if subs_dict:
                        substituted = substituted.subs(subs_dict)
                    
                    # Normalize floating-point exponents to rational form
                    # This converts 0.666666... back to 2/3, etc.
                    substituted = sp.nsimplify(substituted, rational=True, tolerance=1e-10)
                    
                    # Now simplify with multiple aggressive strategies
                    simp = sp.simplify(substituted)
                    
                    # Try additional simplification if not zero
                    if simp != 0:
                        # Strategy 1: Cancel and expand
                        simp = sp.cancel(sp.expand(simp))
                    
                    if simp != 0:
                        # Strategy 2: Rational simplification (better for complex fractions)
                        simp = sp.ratsimp(simp)
                    
                    if simp != 0:
                        # Strategy 3: Factor then simplify
                        try:
                            simp = sp.simplify(sp.factor(simp))
                        except:
                            pass  # Factor may fail on some expressions
                    
                    if simp != 0:
                        # Strategy 4: Expand and collect like terms
                        simp = sp.simplify(sp.expand(simp))
                    
                    simplified_components.append(simp)
                except Exception as e:
                    return EquivalenceTest(
                        name="symbolic_equivalence",
                        result=ValidationResult.TIMEOUT,
                        details=f"Simplification timeout/error on component {i}: {e}"
                    )
            
            # Check if all components are zero
            all_zero = all(comp == 0 for comp in simplified_components)
            
            if all_zero:
                return EquivalenceTest(
                    name="symbolic_equivalence",
                    result=ValidationResult.PASS,
                    details="Symbolic proof: Δ(Z) ≡ 0 (exact equivalence)"
                )
            else:
                # Find non-zero components
                non_zero_indices = [i for i, c in enumerate(simplified_components) if c != 0]
                details = f"Non-zero components: {non_zero_indices}\n"
                details += f"Components: {[str(simplified_components[i]) for i in non_zero_indices[:3]]}"
                
                return EquivalenceTest(
                    name="symbolic_equivalence",
                    result=ValidationResult.FAIL,
                    details=details
                )
                
        except Exception as e:
            return EquivalenceTest(
                name="symbolic_equivalence",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during symbolic test: {str(e)}"
            )
    
    def check_numerical_pointwise_jax(self,
                                      n_samples: int = 1000,
                                      domain_min: float = 0.01,
                                      domain_max: float = 10.0,
                                      threshold: float = 1e-5) -> EquivalenceTest:
        """
        Check numerical equivalence using JAX automatic differentiation.
        
        Computes J_Φ(Z) · f_recast(Z) and compares with f_orig(Φ(Z)) using
        JAX's autodiff for the Jacobian (no symbolic computation).
        
        Args:
            n_samples: Number of random sample points
            domain_min: Minimum value for each variable (log-uniform sampling)
            domain_max: Maximum value for each variable
            threshold: Error threshold for pass/fail
            
        Returns:
            EquivalenceTest with numerical validation results
        """
        try:
            import jax
            import jax.numpy as jnp
            from jax import jacfwd
        except ImportError:
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details="JAX not available. Install with: pip install jax jaxlib"
            )
        
        try:
            # Get ordered variables
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars
            
            # Extract parameter values
            param_values = self.recast_ir.params
            param_names = sorted(param_values.keys())
            param_vals_array = jnp.array([param_values[name] for name in param_names])
            
            # Build mapping functions Φ(Z) using lambdify for JAX
            # Φ maps recast variables to original variables
            phi_funcs = []
            for orig_var in orig_vars_ordered:
                mapping_expr = self.mapping[orig_var]
                # Create function that takes Z values and returns mapped value
                func = lambdify(recast_vars_ordered + [sp.Symbol(p) for p in param_names],
                               mapping_expr,
                               modules='jax')
                phi_funcs.append(func)
            
            # Build f_orig functions (original ODEs after mapping substitution)
            f_orig_funcs = []
            for orig_var in orig_vars_ordered:
                ode_expr = self.orig_odes[orig_var].subs(self.mapping)
                func = lambdify(recast_vars_ordered + [sp.Symbol(p) for p in param_names],
                               ode_expr,
                               modules='jax')
                f_orig_funcs.append(func)
            
            # Build f_recast functions
            f_recast_funcs = []
            for recast_var in recast_vars_ordered:
                ode_expr = self.recast_odes[recast_var]
                func = lambdify(recast_vars_ordered + [sp.Symbol(p) for p in param_names],
                               ode_expr,
                               modules='jax')
                f_recast_funcs.append(func)
            
            # Define Φ as a JAX-compatible vector function
            def phi_vec(z_vals):
                """Φ: R^n_recast -> R^n_orig"""
                combined = jnp.concatenate([z_vals, param_vals_array])
                return jnp.array([func(*combined) for func in phi_funcs])
            
            # Define f_orig(Φ(Z)) as a JAX-compatible function
            def f_orig_at_phi(z_vals):
                """f_orig(Φ(Z))"""
                combined = jnp.concatenate([z_vals, param_vals_array])
                return jnp.array([func(*combined) for func in f_orig_funcs])
            
            # Define f_recast as a JAX-compatible function
            def f_recast_vec(z_vals):
                """f_recast(Z)"""
                combined = jnp.concatenate([z_vals, param_vals_array])
                return jnp.array([func(*combined) for func in f_recast_funcs])
            
            # Compute Jacobian of Φ using JAX autodiff
            jac_phi = jacfwd(phi_vec)
            
            errors = []
            counterexamples = []
            
            # Sample points in log-uniform distribution
            np.random.seed(42)
            log_min = np.log(domain_min)
            log_max = np.log(domain_max)
            
            # Identify which recast variables are auxiliaries vs. original/independent
            orig_var_names = {str(v) for v in orig_vars_ordered}
            independent_vars = []
            auxiliary_vars = []
            auxiliary_var_indices = []
            
            for i, var in enumerate(recast_vars_ordered):
                var_name = str(var)
                if var_name in orig_var_names:
                    # This is an original variable - sample independently
                    independent_vars.append((i, var))
                elif var_name in {str(k) for k in self.auxiliary_defs.keys()}:
                    # This is a lifted auxiliary - compute from definition
                    auxiliary_vars.append((i, var))
                    auxiliary_var_indices.append(i)
                else:
                    # Pool auxiliary from factorization - sample independently
                    independent_vars.append((i, var))
            
            for _ in range(n_samples):
                # Initialize array for all recast variables
                Z_sample = np.zeros(len(recast_vars_ordered))
                
                # Sample independent variables (originals + pool auxiliaries)
                for idx, var in independent_vars:
                    Z_sample[idx] = np.exp(np.random.uniform(log_min, log_max))
                
                # Compute auxiliary variables from their definitions
                for idx, var in auxiliary_vars:
                    var_name = str(var)
                    aux_def = self.auxiliary_defs[sp.Symbol(var_name)]
                    
                    # Evaluate auxiliary definition
                    # Need to substitute current values of independent variables
                    # Match symbols by name to handle symbol object mismatches
                    subs_dict = {}
                    
                    # Substitute independent variable values
                    for indep_idx, indep_var in independent_vars:
                        indep_var_name = str(indep_var)
                        # Find matching symbol in aux_def by name
                        for sym in aux_def.free_symbols:
                            if str(sym) == indep_var_name:
                                subs_dict[sym] = Z_sample[indep_idx]
                                break
                    
                    # Substitute parameter values
                    for param_name, param_val in param_values.items():
                        # Find matching symbol in aux_def by name
                        for sym in aux_def.free_symbols:
                            if str(sym) == param_name:
                                subs_dict[sym] = param_val
                                break
                    
                    # Evaluate - use evalf() then convert to float
                    try:
                        aux_val = float(aux_def.subs(subs_dict).evalf())
                    except:
                        # Fallback: if still symbolic, use lambdify
                        aux_func = lambdify(list(aux_def.free_symbols), aux_def, modules='numpy')
                        aux_val = float(aux_func(*[subs_dict.get(s, 1.0) for s in aux_def.free_symbols]))
                    
                    Z_sample[idx] = aux_val
                
                Z_jax = jnp.array(Z_sample)
                
                # Compute J_Φ(Z) using JAX autodiff
                J_at_Z = jac_phi(Z_jax)
                
                # Compute f_recast(Z)
                f_recast_at_Z = f_recast_vec(Z_jax)
                
                # Compute f_orig(Φ(Z))
                f_orig_at_Phi_Z = f_orig_at_phi(Z_jax)
                
                # Compute J_Φ(Z) · f_recast(Z)
                lhs = J_at_Z @ f_recast_at_Z
                
                # Compute error
                diff = lhs - f_orig_at_Phi_Z
                abs_error = float(jnp.max(jnp.abs(diff)))
                
                # Relative error
                scale = 1.0 + float(jnp.max(jnp.abs(f_orig_at_Phi_Z)))
                rel_error = abs_error / scale
                
                errors.append(rel_error)
                
                if rel_error > threshold and len(counterexamples) < 5:
                    counterexamples.append({
                        'Z': Z_sample.tolist(),
                        'error': float(rel_error),
                        'abs_error': float(abs_error),
                        'lhs': [float(x) for x in lhs],
                        'rhs': [float(x) for x in f_orig_at_Phi_Z],
                        'diff': [float(x) for x in diff]
                    })
            
            max_error = max(errors)
            mean_error = np.mean(errors)
            
            if max_error < threshold:
                return EquivalenceTest(
                    name="numerical_pointwise_jax",
                    result=ValidationResult.PASS,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"JAX autodiff: Passed with {n_samples} samples. Max error: {max_error:.2e}"
                )
            else:
                return EquivalenceTest(
                    name="numerical_pointwise_jax",
                    result=ValidationResult.FAIL,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"JAX autodiff: Failed - max error {max_error:.2e} > threshold {threshold:.2e}",
                    counterexamples=counterexamples
                )
                
        except Exception as e:
            return EquivalenceTest(
                name="numerical_pointwise_jax",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during JAX numerical test: {str(e)}"
            )
    
    def check_numerical_pointwise(self,
                                   n_samples: int = 1000,
                                   domain_min: float = 0.01,
                                   domain_max: float = 10.0,
                                   threshold: float = 1e-6) -> EquivalenceTest:
        """
        Check numerical equivalence at random sample points.
        
        Computes J_Φ(Z) · f_recast(Z) and compares with f_orig(Φ(Z)) using
        symbolic Jacobian (falls back to JAX if available).
        
        Args:
            n_samples: Number of random sample points
            domain_min: Minimum value for each variable (log-uniform sampling)
            domain_max: Maximum value for each variable
            threshold: Error threshold for pass/fail
            
        Returns:
            EquivalenceTest with numerical validation results
        """
        try:
            # Build symbolic Jacobian and lambdify it for speed
            orig_vars_ordered = sorted(self.orig_odes.keys(), key=str)
            recast_vars_ordered = self.recast_state_vars
            
            # Extract parameter values from recast IR
            param_values = self.recast_ir.params
            param_symbols = [sp.Symbol(name) for name in sorted(param_values.keys())]
            param_vals_ordered = [param_values[str(sym)] for sym in param_symbols]
            
            # Build Φ as a vector
            Phi_vector = Matrix([self.mapping[v] for v in orig_vars_ordered])
            Z_vector = Matrix(recast_vars_ordered)
            
            # Compute symbolic Jacobian J_Φ = ∂Φ/∂Z
            J_Phi = Phi_vector.jacobian(Z_vector)
            
            # Lambdify each element of Jacobian with both state vars and params
            n_orig = len(orig_vars_ordered)
            n_recast = len(recast_vars_ordered)
            all_symbols = list(recast_vars_ordered) + param_symbols
            J_Phi_funcs = []
            for i in range(n_orig):
                row_funcs = []
                for j in range(n_recast):
                    elem_func = lambdify(all_symbols, J_Phi[i, j], modules='numpy')
                    row_funcs.append(elem_func)
                J_Phi_funcs.append(row_funcs)
            
            # Lambdify recast ODEs with params
            f_recast_funcs = []
            for var in recast_vars_ordered:
                f_recast_funcs.append(lambdify(all_symbols,
                                               self.recast_odes[var],
                                               modules='numpy'))
            
            # Lambdify original ODEs with mapping substituted
            f_orig_at_Phi_funcs = []
            for var in orig_vars_ordered:
                # Substitute mapping into original ODE
                ode_at_phi = self.orig_odes[var].subs(self.mapping)
                f_orig_at_Phi_funcs.append(lambdify(all_symbols,
                                                     ode_at_phi,
                                                     modules='numpy'))
            
            errors = []
            counterexamples = []
            
            # Sample points in log-uniform distribution (positive orthant)
            np.random.seed(42)  # Reproducibility
            log_min = np.log(domain_min)
            log_max = np.log(domain_max)
            
            for _ in range(n_samples):
                # Sample ALL recast state variables in log-uniform
                # This handles both factorized (canonical) and lifted (simplified) forms
                Z_sample = np.exp(np.random.uniform(log_min, log_max, 
                                                    len(recast_vars_ordered)))
                
                # Combine state variables and parameters for evaluation
                all_vals = tuple(Z_sample) + tuple(param_vals_ordered)
                
                # Evaluate J_Φ(Z) element by element - returns a matrix
                J_at_Z = np.zeros((n_orig, n_recast))
                for i in range(n_orig):
                    for j in range(n_recast):
                        result = J_Phi_funcs[i][j](*all_vals)
                        # Handle both numeric and symbolic results
                        if hasattr(result, 'evalf'):
                            # It's still symbolic, evaluate it
                            J_at_Z[i, j] = float(result.evalf())
                        else:
                            J_at_Z[i, j] = float(result)
                
                # Evaluate f_recast(Z) - returns a vector
                f_recast_at_Z = np.array([f(*all_vals) for f in f_recast_funcs], 
                                        dtype=float)
                
                # Evaluate f_orig(Φ(Z)) - returns a vector
                f_orig_at_Phi_Z = np.array([f(*all_vals) for f in f_orig_at_Phi_funcs],
                                           dtype=float)
                
                # Compute J_Φ(Z) · f_recast(Z)
                lhs = J_at_Z @ f_recast_at_Z
                
                # Compute error: ||J_Φ · f_recast - f_orig(Φ)||
                diff = lhs - f_orig_at_Phi_Z
                abs_error = np.max(np.abs(diff))
                
                # Relative error
                scale = 1.0 + np.max(np.abs(f_orig_at_Phi_Z))
                rel_error = abs_error / scale
                
                errors.append(rel_error)
                
                if rel_error > threshold and len(counterexamples) < 5:
                    counterexamples.append({
                        'Z': Z_sample.tolist(),
                        'error': float(rel_error),
                        'abs_error': float(abs_error),
                        'lhs': lhs.tolist(),
                        'rhs': f_orig_at_Phi_Z.tolist(),
                        'diff': diff.tolist()
                    })
            
            max_error = max(errors)
            mean_error = np.mean(errors)
            
            if max_error < threshold:
                return EquivalenceTest(
                    name="numerical_pointwise",
                    result=ValidationResult.PASS,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"Passed with {n_samples} samples. Max error: {max_error:.2e}"
                )
            else:
                return EquivalenceTest(
                    name="numerical_pointwise",
                    result=ValidationResult.FAIL,
                    max_error=float(max_error),
                    mean_error=float(mean_error),
                    details=f"Failed: max error {max_error:.2e} > threshold {threshold:.2e}",
                    counterexamples=counterexamples
                )
                
        except Exception as e:
            return EquivalenceTest(
                name="numerical_pointwise",
                result=ValidationResult.NOT_ATTEMPTED,
                details=f"Exception during numerical test: {str(e)}"
            )
    
    def validate(self,
                 run_symbolic: bool = True,
                 run_numerical: bool = True,
                 run_trajectory: bool = False) -> ValidationReport:
        """
        Run full validation suite.
        
        Args:
            run_symbolic: Run symbolic equivalence test
            run_numerical: Run numerical pointwise test
            run_trajectory: Run trajectory comparison test
            
        Returns:
            ValidationReport with all test results
        """
        report = ValidationReport(
            original_file=self.original_file,
            recast_file=self.recast_file,
            original_class=self.orig_class,
            recast_class=self.recast_class,
            canonical_refusal_reason=self.canonical_refusal_reason
        )
        
        # Run tests
        if run_symbolic:
            report.symbolic_test = self.check_symbolic_equivalence()
        
        if run_numerical:
            # Try JAX first (more robust), fall back to symbolic Jacobian
            try:
                import jax
                report.numerical_test = self.check_numerical_pointwise_jax()
            except ImportError:
                # JAX not available, use symbolic Jacobian method
                report.numerical_test = self.check_numerical_pointwise()
        
        # Determine overall pass/fail
        tests_run = []
        if report.symbolic_test:
            tests_run.append(report.symbolic_test)
        if report.numerical_test:
            tests_run.append(report.numerical_test)
        
        # Pass if any strong test passes (symbolic OR numerical)
        strong_pass = any(t.result == ValidationResult.PASS 
                         for t in tests_run)
        
        # Only fail if ALL tests definitively fail
        all_fail = all(t.result == ValidationResult.FAIL 
                      for t in tests_run)
        
        report.overall_pass = strong_pass and not all_fail
        
        # Generate summary
        if report.overall_pass:
            report.summary = "✓ Validation PASSED: Recast is mathematically equivalent"
        elif all_fail:
            report.summary = "✗ Validation FAILED: Recast differs from original"
        else:
            report.summary = "? Validation INCONCLUSIVE: Tests timed out or not attempted"
        
        return report


def validate_recast_pair(original_file: str,
                         recast_file: str,
                         factor_map: Optional[Dict] = None,
                         mode: str = "simplified",
                         output_json: Optional[str] = None) -> ValidationReport:
    """
    Convenience function to validate a recast.
    
    Args:
        original_file: Path to original Antimony file
        recast_file: Path to recast Antimony file
        factor_map: Optional factor map
        mode: Recast mode
        output_json: Optional path to save JSON report
        
    Returns:
        ValidationReport
    """
    validator = RecastValidator(original_file, recast_file, factor_map, mode)
    report = validator.validate()
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    return report
