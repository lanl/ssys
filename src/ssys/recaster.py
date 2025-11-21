
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import sympy as sp

arrow_pat = re.compile(r"<->|->")
prime_rule_pat = re.compile(r"^\s*\$?([A-Za-z_]\w*)\s*'\s*=\s*(.+)$")

EPS_INIT = 1e-9

def tokenize_species_side(side: str) -> List[Tuple[int, str]]:
    parts = [p.strip() for p in side.split('+') if p.strip()]
    result = []
    for p in parts:
        p = p.strip()
        if p.startswith("$"):
            p = p[1:].strip()
        toks = p.split()
        if len(toks) == 1:
            coeff = 1
            name = toks[0]
        else:
            try:
                coeff = int(sp.sympify(toks[0]))
                name = toks[1]
            except Exception:
                coeff = 1
                name = p
        result.append((coeff, name))
    return result

def _expand_exps_through_factors(exps, factor_map):
    """Return a new {sym: exp} dict where any original symbol present in factor_map
       is replaced by its factor variables, each receiving the same exponent."""
    new = {}
    for s, e in exps.items():
        if s in factor_map:
            for v in factor_map[s]:
                new[v] = new.get(v, 0.0) + float(e)
        else:
            new[s] = new.get(s, 0.0) + float(e)
    return new

def _numeric_param_subs(expr: sp.Expr, params: Dict[str, float]) -> sp.Expr:
    """Replace parameter symbols in expr with their numeric values from params."""
    if not params:
        return expr
    # Build a mapping only for symbols actually used in expr
    subs = {s: sp.Float(params[s.name]) for s in expr.free_symbols if s.name in params}
    return sp.simplify(expr.subs(subs)) if subs else expr

@dataclass
class Reaction:
    name: Optional[str]
    lhs: List[Tuple[int, str]]
    rhs: List[Tuple[int, str]]
    rate_expr: str

@dataclass
class ModelIR:
    species: Set[str] = field(default_factory=set)
    boundary: Set[str] = field(default_factory=set)
    params: Dict[str, float] = field(default_factory=dict)
    initial: Dict[str, float] = field(default_factory=dict)
    reactions: List[Reaction] = field(default_factory=list)
    explicit_rates: Dict[str, str] = field(default_factory=dict)
    raw_lines: List[str] = field(default_factory=list)

def parse_antimony(text: str) -> ModelIR:
    ir = ModelIR()
    ir.raw_lines = [ln.rstrip() for ln in text.splitlines()]

    for raw in ir.raw_lines:
        # strip inline comments
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("model ") or line.lower() == "end":
            continue

        # REACTIONS: keep the whole line (rate law is after ';')
        if ("->" in line) or ("<->" in line):
            s = line
            before_rate, rate_expr = s.split(";", 1)
            rate_expr = rate_expr.strip()
            if ":" in before_rate:
                rxn_name, stoich = before_rate.split(":", 1)
                rxn_name = rxn_name.strip()
                stoich = stoich.strip()
            else:
                rxn_name = None
                stoich = before_rate.strip()
            arrow = arrow_pat.search(stoich)
            if not arrow:
                continue
            lhs = stoich[:arrow.start()].strip()
            rhs = stoich[arrow.end():].strip()
            lhs_list = tokenize_species_side(lhs) if lhs else []
            rhs_list = tokenize_species_side(rhs) if rhs else []
            for _, nm in lhs_list + rhs_list:
                ir.species.add(nm)
            ir.reactions.append(Reaction(rxn_name, lhs_list, rhs_list, rate_expr))
            continue

        # NON-REACTION LINES: may contain multiple ';'-separated statements
        for stmt in [seg.strip() for seg in line.split(";") if seg.strip()]:
            # explicit rate rule: S' = ...
            m = prime_rule_pat.match(stmt)
            if m:
                sp_name = m.group(1)
                expr = m.group(2).strip()
                if stmt.strip().startswith("$"):
                    ir.boundary.add(sp_name)
                ir.species.add(sp_name)
                ir.explicit_rates[sp_name] = expr
                continue

            # parameter/initial assignment: X = 2.5
            if ("=" in stmt) and (":=" not in stmt):
                left, right = stmt.split("=", 1)
                left = left.strip()
                right = right.strip()
                try:
                    val = float(sp.sympify(right))
                except Exception:
                    val = None
                if left.startswith("$"):
                    left = left[1:].strip()
                    ir.boundary.add(left)
                ir.initial[left] = val if val is not None else 0.0
                continue

            # (Optional) ignore assignment rules ':=' for now, or treat as params
            if ":=" in stmt:
                left, right = stmt.split(":=", 1)
                left = left.strip()
                ir.params[left] = float(0.0)
                continue

    # promote non-species initializations to parameters
    for nm, val in list(ir.initial.items()):
        if nm not in ir.species:
            ir.params[nm] = val

    return ir

@dataclass
class SymSystem:
    vars: List[sp.Symbol]
    params: Dict[str, float]
    odes: Dict[sp.Symbol, sp.Expr]
    initials: Dict[sp.Symbol, float]

def build_sym_system(ir: ModelIR) -> SymSystem:
    var_syms: Dict[str, sp.Symbol] = {nm: sp.symbols(nm, positive=True) for nm in sorted(ir.species)}
    param_syms: Dict[str, sp.Symbol] = {}
    for nm, val in ir.params.items():
        if nm in var_syms:
            continue
        param_syms[nm] = sp.symbols(nm, positive=True)
    odes: Dict[sp.Symbol, sp.Expr] = {var_syms[nm]: sp.Integer(0) for nm in var_syms}
    for rxn in ir.reactions:
        rate = sp.sympify(rxn.rate_expr, locals={**var_syms, **param_syms})
        lhs_sto = {nm: coeff for coeff, nm in rxn.lhs}
        rhs_sto = {nm: coeff for coeff, nm in rxn.rhs}
        all_sp = set(lhs_sto) | set(rhs_sto)
        for nm in all_sp:
            if nm not in var_syms:
                continue
            net = rhs_sto.get(nm, 0) - lhs_sto.get(nm, 0)
            if nm in ir.boundary:
                continue
            odes[var_syms[nm]] += sp.Integer(net) * rate
    for nm, expr in ir.explicit_rates.items():
        if nm not in var_syms:
            continue
        odes[var_syms[nm]] += sp.sympify(expr, locals={**var_syms, **param_syms})
    initials: Dict[sp.Symbol, float] = {}
    for nm, sym in var_syms.items():
        initials[sym] = float(ir.initial.get(nm, 0.0))
    for nm, sym in param_syms.items():
        initials[sym] = float(ir.initial.get(nm, ir.params.get(nm, 0.0)))
    return SymSystem(vars=list(odes.keys()), params={k: float(v) for k, v in ir.params.items()}, odes=odes, initials=initials)

from dataclasses import dataclass

def expand_to_terms(expr: sp.Expr) -> List[sp.Expr]:
    expr = sp.expand(expr)
    if expr.is_Add:
        return list(expr.args)
    else:
        return [expr]

@dataclass
class SSysEquation:
    var: sp.Symbol
    growth: Tuple[float, Dict[sp.Symbol, float]]  # α, {sym: exponent}
    decay: Tuple[float, Dict[sp.Symbol, float]]   # β, {sym: exponent}

@dataclass
class RecastResult:
    equations: List[SSysEquation]
    initials: Dict[sp.Symbol, float]
    variables: List[sp.Symbol]
    factor_map: Dict[sp.Symbol, List[sp.Symbol]] = field(default_factory=dict)

# --- CANONICALIZE AUX NAMES: must be placed below the dataclasses ---
def canonicalize_aux_names(res: 'RecastResult', prefix: str = "X") -> 'RecastResult':
    """
    Rename every auxiliary variable to X_1, X_2, ... in first-appearance order.
    Updates equations, initials, variables, and factor_map consistently.
    """
    # 1) Determine aux order by first appearance in equations
    aux_order, seen = [], set()
    for eq in res.equations:
        if eq.var not in seen:
            aux_order.append(eq.var)
            seen.add(eq.var)

    # 2) Map old aux -> new canonical aux
    name_map = {old: sp.Symbol(f"{prefix}_{i}") for i, old in enumerate(aux_order, start=1)}

    def remap_exps(exps: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        out: Dict[sp.Symbol, float] = {}
        for s, e in exps.items():
            out[name_map.get(s, s)] = float(e)
        return out

    # 3) Remap equations (var and exponent maps)
    new_eqs: List[SSysEquation] = []
    for eq in res.equations:
        new_eqs.append(SSysEquation(
            var=name_map.get(eq.var, eq.var),
            growth=(float(eq.growth[0]), remap_exps(eq.growth[1])),
            decay =(float(eq.decay[0]),  remap_exps(eq.decay[1])),
        ))

    # 4) Remap initials (keys)
    new_initials = {name_map.get(s, s): float(v) for s, v in res.initials.items()}

    # 5) Remap factor_map (lists of auxiliaries)
    new_factor_map = {
        orig: [name_map.get(a, a) for a in aux_list]
        for orig, aux_list in res.factor_map.items()
    }

    # 6) Canonical variables list, in canonical order
    new_variables = [name_map[old] for old in aux_order]

    return RecastResult(
        equations=new_eqs,
        initials=new_initials,
        variables=new_variables,
        factor_map=new_factor_map,
    )
# --- end canonicalize_aux_names ---

def term_to_coeff_exps(term: sp.Expr) -> Tuple[float, Dict[sp.Symbol, float]]:
    term = sp.simplify(term)
    coeff = 1.0
    exps: Dict[sp.Symbol, float] = {}
    if term.is_Number:
        return float(term), exps
    if isinstance(term, sp.Symbol):
        exps[term] = exps.get(term, 0.0) + 1.0
        return coeff, exps
    if term.is_Mul:
        for f in term.args:
            if f.is_Number:
                coeff *= float(f)
            elif isinstance(f, sp.Symbol):
                exps[f] = exps.get(f, 0.0) + 1.0
            elif isinstance(f, sp.Pow):
                base, exp = f.args
                exps[base] = exps.get(base, 0.0) + float(exp)
            else:
                raise ValueError(f"Cannot monomialize factor: {f}")
        return coeff, exps
    if isinstance(term, sp.Pow):
        base, exp = term.args
        exp_num = sp.nsimplify(exp)
        if not exp_num.is_number:
            raise ValueError(f"Exponent must be numeric after parameter substitution, got {exp}")
        exps[base] = exps.get(base, 0.0) + float(exp_num)
        return coeff, exps
    raise ValueError(f"Unsupported term: {term}")

def product_expr(coeff: float, exps: Dict[sp.Symbol, float]) -> sp.Expr:
    expr = sp.Float(coeff)
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        expr *= s**sp.Float(e)
    return sp.simplify(expr)


def find_rational_denominators(expr: sp.Expr) -> Set[sp.Expr]:
    """
    Find all unique non-trivial denominators in an expression.
    Returns set of denominator expressions that need auxiliary variables.
    
    A denominator is "non-trivial" if it's not:
    - A constant
    - A single variable (already power-law)
    """
    denoms = set()
    
    def visit(e):
        if isinstance(e, sp.Pow):
            base, exp = e.args
            # Check for negative exponents (divisions)
            if exp.is_number and float(exp) < 0:
                # If base is not a simple symbol, it needs lifting
                if not isinstance(base, sp.Symbol):
                    denoms.add(base)
            # Recurse into base
            visit(base)
        elif isinstance(e, (sp.Add, sp.Mul)):
            for arg in e.args:
                visit(arg)
    
    visit(expr)
    return denoms


def find_composite_functions(expr: sp.Expr) -> Set[sp.Expr]:
    """
    Find all composite functions (non-algebraic functions) in an expression.
    Returns set of function applications that need auxiliary variables.
    
    A composite function is any sympy function application (exp, sin, log, etc.)
    that is not a simple algebraic operation (Add, Mul, Pow with numeric exponent).
    """
    functions = set()
    
    def visit(e):
        # Check if this is a function application
        if isinstance(e, sp.Function):
            # This is a function like exp(X), sin(X), etc.
            functions.add(e)
            # Also recurse into arguments
            for arg in e.args:
                visit(arg)
        elif isinstance(e, sp.Pow):
            # Recurse into base and exponent
            visit(e.args[0])
            if not e.args[1].is_number:
                visit(e.args[1])
        elif isinstance(e, (sp.Add, sp.Mul)):
            for arg in e.args:
                visit(arg)
        # Note: we don't add Symbol, Number, or Pow with numeric exponent
    
    visit(expr)
    return functions


def create_auxiliary_for_denominator(
    denom: sp.Expr,
    var_odes: Dict[sp.Symbol, sp.Expr],
    aux_counter: int,
    prefix: str = "W"
) -> Tuple[sp.Symbol, sp.Expr]:
    """
    Create auxiliary W = 1/denom and compute W' via chain rule.
    
    For W = 1/D(X):
        W' = -W^2 * dD/dt
    where dD/dt = sum_i (∂D/∂X_i) * dX_i/dt
    
    Returns: (W_symbol, W_ode)
    """
    # Create auxiliary symbol
    W = sp.symbols(f"{prefix}_{aux_counter}", positive=True)
    
    # Compute dD/dt using chain rule
    denom_prime = sp.Integer(0)
    for var, var_ode in var_odes.items():
        if var in denom.free_symbols:
            partial = sp.diff(denom, var)
            denom_prime += partial * var_ode
    
    # W' = -W^2 * dD/dt
    W_ode = -W**2 * denom_prime
    W_ode = sp.simplify(W_ode)
    
    return W, W_ode


def lift_rational_functions(sym: SymSystem) -> SymSystem:
    """
    Augment system with auxiliary variables for all rational terms.
    
    For each unique non-trivial denominator D(X):
    1. Create auxiliary W = 1/D
    2. Add ODE: W' = -W^2 * dD/dt (chain rule)
    3. Replace 1/D with W in all ODEs
    4. Set W(0) = 1/D(X(0))
    
    Returns augmented SymSystem with rational terms eliminated.
    """
    # Find all unique denominators across all ODEs
    all_denoms = set()
    for var, ode in sym.odes.items():
        denoms = find_rational_denominators(ode)
        all_denoms.update(denoms)
    
    if not all_denoms:
        # No rational functions to lift
        return sym
    
    # Create auxiliary symbols for each denominator
    denom_to_aux: Dict[sp.Expr, sp.Symbol] = {}
    aux_counter = 1
    
    for denom in sorted(all_denoms, key=str):
        W = sp.symbols(f"W_{aux_counter}", positive=True)
        denom_to_aux[denom] = W
        aux_counter += 1
    
    # Substitute auxiliaries in original ODEs
    # Replace D^(-1) with W for each denominator D
    new_odes: Dict[sp.Symbol, sp.Expr] = {}
    for var, ode in sym.odes.items():
        new_ode = ode
        for denom, W in denom_to_aux.items():
            # Replace denom^(-n) with W^n for any n
            new_ode = new_ode.replace(denom**(-1), W)
            # Handle other negative powers if present
            for n in range(2, 6):  # Check powers -2 through -5
                if denom**(-n) in new_ode.atoms():
                    new_ode = new_ode.replace(denom**(-n), W**n)
        new_odes[var] = sp.simplify(new_ode)
    
    # NOW compute W' using the LIFTED ODEs (not the original ones)
    new_aux_odes: Dict[sp.Symbol, sp.Expr] = {}
    for denom, W in denom_to_aux.items():
        # Compute dD/dt using the lifted ODEs
        denom_prime = sp.Integer(0)
        for var in sym.vars:
            if var in denom.free_symbols:
                partial = sp.diff(denom, var)
                # Use the NEW (lifted) ODE for var, not the original
                denom_prime += partial * new_odes[var]
        
        # W' = -W^2 * dD/dt
        W_ode = -W**2 * denom_prime
        new_aux_odes[W] = sp.simplify(W_ode)
    
    # Combine original and auxiliary ODEs
    combined_odes = {**new_odes, **new_aux_odes}
    
    # Compute initial conditions for auxiliaries
    new_initials = dict(sym.initials)
    for denom, W in denom_to_aux.items():
        # Evaluate denominator at t=0
        denom_at_0 = denom
        for var in sym.vars:
            if var in denom.free_symbols:
                denom_at_0 = denom_at_0.subs(var, sym.initials.get(var, 1.0))
        # W(0) = 1/D(X(0))
        try:
            W_init = float(1.0 / denom_at_0)
        except:
            W_init = 1.0  # Fallback if evaluation fails
        new_initials[W] = W_init
    
    # Create new variable list: keep original vars, add W auxiliaries
    # Original vars come first, then the new W auxiliaries
    new_vars = list(sym.vars) + list(denom_to_aux.values())
    
    return SymSystem(
        vars=new_vars,
        params=sym.params,
        odes=combined_odes,
        initials=new_initials
    )

def lift_composite_functions(sym: SymSystem) -> SymSystem:
    """
    Augment system with auxiliary variables for all composite functions.
    
    For each unique composite function f(X) (exp, sin, log, etc.):
    1. Create auxiliary Z = f(X)
    2. Add ODE: Z' = df/dX * X' (chain rule)
    3. Replace f(X) with Z in all ODEs
    4. Set Z(0) = f(X(0))
    
    This is a general implementation that works for any differentiable function
    that sympy knows how to differentiate.
    
    Returns augmented SymSystem with composite functions eliminated.
    """
    # Find all unique composite functions across all ODEs
    all_functions = set()
    for var, ode in sym.odes.items():
        funcs = find_composite_functions(ode)
        all_functions.update(funcs)
    
    if not all_functions:
        # No composite functions to lift
        return sym
    
    # Create auxiliary symbols for each function
    func_to_aux: Dict[sp.Expr, sp.Symbol] = {}
    aux_counter = 1
    
    for func in sorted(all_functions, key=str):
        Z = sp.symbols(f"Z_{aux_counter}", positive=True)
        func_to_aux[func] = Z
        aux_counter += 1
    
    # Substitute auxiliaries in original ODEs
    new_odes: Dict[sp.Symbol, sp.Expr] = {}
    for var, ode in sym.odes.items():
        new_ode = ode
        for func, Z in func_to_aux.items():
            new_ode = new_ode.replace(func, Z)
        new_odes[var] = sp.simplify(new_ode)
    
    # Compute Z' using chain rule with the LIFTED ODEs
    new_aux_odes: Dict[sp.Symbol, sp.Expr] = {}
    for func, Z in func_to_aux.items():
        # Compute df/dt using chain rule: df/dt = sum_i (∂f/∂X_i) * dX_i/dt
        func_prime = sp.Integer(0)
        for var in sym.vars:
            if var in func.free_symbols:
                partial = sp.diff(func, var)
                # Use the NEW (lifted) ODE for var
                func_prime += partial * new_odes[var]
        
        # Now substitute all composite functions with their auxiliaries in Z'
        Z_ode = func_prime
        for other_func, other_Z in func_to_aux.items():
            Z_ode = Z_ode.replace(other_func, other_Z)
        
        Z_ode = sp.simplify(Z_ode)
        new_aux_odes[Z] = Z_ode
    
    # Combine original and auxiliary ODEs
    combined_odes = {**new_odes, **new_aux_odes}
    
    # Compute initial conditions for auxiliaries
    new_initials = dict(sym.initials)
    for func, Z in func_to_aux.items():
        # Evaluate function at t=0
        func_at_0 = func
        for var in sym.vars:
            if var in func.free_symbols:
                func_at_0 = func_at_0.subs(var, sym.initials.get(var, 1.0))
        # Z(0) = f(X(0))
        try:
            Z_init = float(func_at_0)
        except:
            Z_init = 1.0  # Fallback if evaluation fails
        new_initials[Z] = Z_init
    
    # Create new variable list: keep original vars, add Z auxiliaries
    new_vars = list(sym.vars) + list(func_to_aux.values())
    
    return SymSystem(
        vars=new_vars,
        params=sym.params,
        odes=combined_odes,
        initials=new_initials
    )


def recast_to_ssystem(sym: 'SymSystem') -> 'RecastResult':
    """
    Canonical S-system recast using log-derivative 'pool' auxiliaries:
      For X' = sum_j s_j (signed monomials), create one aux V_j per term with
        V_j' = + u_j * (∏_{ℓ≠j} V_ℓ)^(-1)   if coeff > 0
        V_j' = 0 - u_j * (∏_{ℓ≠j} V_ℓ)^(-1) if coeff < 0
      where u_j = |coeff_j| * ∏ original^{exp}.
      Then X = ∏_j V_j, and d/dt(∏ V_j) = ∑ s_j exactly.
    
    Automatically lifts rational and composite functions before recasting.
    """
    # Track original variables before lifting
    original_vars = set(sym.vars)
    
    # Lift composite functions first (exp, sin, log, etc.)
    sym = lift_composite_functions(sym)
    
    # Then lift rational functions (1/(X+1), etc.)
    sym = lift_rational_functions(sym)
    
    # Identify lifted auxiliaries (those added during lifting)
    lifted_vars = set(sym.vars) - original_vars
    
    new_equations: List[SSysEquation] = []
    new_variables: List[sp.Symbol] = []
    new_initials: Dict[sp.Symbol, float] = dict(sym.initials)   # keep params and originals
    factor_map: Dict[sp.Symbol, List[sp.Symbol]] = {}

    for Xi in sorted(sym.vars, key=lambda s: s.name):
        # Lifted auxiliaries: keep as single variables (already power-law)
        if Xi in lifted_vars:
            # Lifted vars already have power-law ODEs from the lifting process
            # No need for pool construction - just convert to growth/decay form
            rhs = sp.simplify(sym.odes[Xi])
            rhs = _numeric_param_subs(rhs, sym.params)
            terms = expand_to_terms(rhs)
            
            # Separate positive and negative terms for growth/decay
            growth_terms = []
            decay_terms = []
            for t in terms:
                if t == 0:
                    continue
                coeff, exps = term_to_coeff_exps(t)
                if coeff >= 0:
                    growth_terms.append((coeff, exps))
                else:
                    decay_terms.append((abs(coeff), exps))
            
            # Combine growth terms (sum coefficients, average exponents)
            if growth_terms:
                g_coeff = sum(c for c, _ in growth_terms)
                g_exps = {}
                for c, e in growth_terms:
                    weight = c / g_coeff
                    for s, exp in e.items():
                        g_exps[s] = g_exps.get(s, 0.0) + weight * exp
            else:
                g_coeff, g_exps = 0.0, {}
            
            # Combine decay terms (sum coefficients, average exponents)
            if decay_terms:
                d_coeff = sum(c for c, _ in decay_terms)
                d_exps = {}
                for c, e in decay_terms:
                    weight = c / d_coeff
                    for s, exp in e.items():
                        d_exps[s] = d_exps.get(s, 0.0) + weight * exp
            else:
                d_coeff, d_exps = 0.0, {}
            
            # Add the lifted variable itself with its equation
            new_variables.append(Xi)
            new_equations.append(SSysEquation(Xi, (g_coeff, g_exps), (d_coeff, d_exps)))
            
            # Do NOT add to factor_map - lifted vars are not reconstructed
            continue
        
        # Original variables: apply pool construction
        # 1) decompose RHS into signed monomial terms over ORIGINAL symbols
        rhs = sp.simplify(sym.odes[Xi])
        rhs = _numeric_param_subs(rhs, sym.params)
        terms = expand_to_terms(rhs)
        mono_terms: List[Tuple[float, Dict[sp.Symbol, float]]] = []
        for t in terms:
            if t == 0:
                continue
            coeff, exps = term_to_coeff_exps(t)  # coeff may be ±
            mono_terms.append((coeff, exps))

        # Handle degenerate X' == 0
        if not mono_terms:
            V = sp.symbols(f"{Xi.name}_t1", positive=True)
            new_variables.append(V)
            new_initials[V] = 1.0
            new_equations.append(SSysEquation(V, (0.0, {}), (0.0, {})))
            factor_map[Xi] = [V]
            continue

        # 2) create one auxiliary per term
        V_list: List[sp.Symbol] = []
        for j in range(len(mono_terms)):
            Vj = sp.symbols(f"{Xi.name}_t{j+1}", positive=True)
            V_list.append(Vj)
            new_variables.append(Vj)
            new_initials.setdefault(Vj, 1.0)

        # 3) define each V_j' per the pool formula; EXCLUDE V_j from the denominator
        for j, (coeff, exps_orig) in enumerate(mono_terms):
            Vj = V_list[j]
            exps = dict(exps_orig)  # start with original-variable exponents

            # Multiply by (∏_{ℓ≠j} V_ℓ)^(-1)  → add -1 exponent for every V_k with k != j
            for k, Vk in enumerate(V_list):
                if k == j:    # exclude V_j itself!
                    continue
                exps[Vk] = exps.get(Vk, 0.0) - 1.0

            # Assign growth/decay by sign of coeff
            if coeff >= 0:
                new_equations.append(SSysEquation(
                    var=Vj,
                    growth=(abs(coeff), exps),
                    decay=(0.0, {})
                ))
            else:
                new_equations.append(SSysEquation(
                    var=Vj,
                    growth=(0.0, {}),
                    decay=(abs(coeff), exps)
                ))

        # 4) mapping X = ∏_j V_j and initial consistency at t=0
        factor_map[Xi] = list(V_list)
        xi0 = float(new_initials.get(Xi, 1.0))
        # Set the first aux to Xi(0), others to 1.0 (product equals Xi(0))
        if V_list:
            new_initials[V_list[0]] = xi0 if xi0 > 0.0 else EPS_INIT

    # 5) build result and canonicalize names to X_1, X_2, ...
    res = RecastResult(
        equations=new_equations,
        initials=new_initials,
        variables=new_variables,
        factor_map=factor_map,
    )
    return canonicalize_aux_names(res, prefix="X")

def product_to_antimony(coeff: float, exps: Dict[sp.Symbol, float]) -> str:
    parts: List[str] = []
    if coeff not in (0.0, 1.0):
        parts.append(f"{coeff:g}")
    elif coeff == 0.0:
        return "0"
    for s, e in sorted(exps.items(), key=lambda kv: str(kv[0])):
        if abs(e) < 1e-14:
            continue
        # Handle both Symbol and complex expressions
        s_name = s.name if hasattr(s, 'name') else str(s)
        if abs(e - 1.0) < 1e-14:
            parts.append(f"{s_name}")
        else:
            parts.append(f"{s_name}^{e:g}")
    if not parts:
        return "0"
    return "*".join(parts)

def ssystem_to_antimony(result, model_name: str = "recast") -> str:
    lines: List[str] = []
    lines.append(f"model {model_name}()")

    # --- mapping: original → product of auxiliaries ---
    lines.append("// Mapping from original variables to canonical auxiliaries (product form)")
    for orig in sorted(result.factor_map.keys(), key=lambda s: s.name):
        aux = result.factor_map[orig]
        rhs = "*".join(a.name for a in aux) if aux else "1"
        lines.append(f"// {orig.name} = {rhs}")
    lines.append("// --- end mapping ---")

    # initial assignments (as before)
    for s, v in sorted(result.initials.items(), key=lambda kv: kv[0].name):
        lines.append(f"{s.name} = {float(v):g}")

    # canonical S-system ODEs (aux-only exponents already expanded)
    for eq in result.equations:
        g_exps = _expand_exps_through_factors(eq.growth[1], result.factor_map)
        h_exps = _expand_exps_through_factors(eq.decay[1], result.factor_map)
        g = product_to_antimony(eq.growth[0], g_exps)
        h = product_to_antimony(eq.decay[0], h_exps)
        lines.append(f"{eq.var.name}' = {g} - {h}")

    lines.append("end")
    return "\n".join(lines)

def latex_odes(sym: 'SymSystem') -> str:
    lines = []
    for v in sorted(sym.odes.keys(), key=lambda s: s.name):
        rhs = sp.simplify(sym.odes[v])
        lines.append(r"\dot{%s} = %s" % (sp.latex(v), sp.latex(rhs)))
    return r"\\begin{aligned}" + r"\\\\\n".join(lines) + r"\\end{aligned}"

def latex_ssys(result: 'RecastResult') -> str:
    lines = []
    for eq in result.equations:
        g = product_expr(eq.growth[0], eq.growth[1])
        h = product_expr(eq.decay[0], eq.decay[1])
        lines.append(r"\dot{%s} = %s - %s" % (sp.latex(eq.var), sp.latex(g), sp.latex(h)))
    return r"\\begin{aligned}" + r"\\\\\n".join(lines) + r"\\end{aligned}"
