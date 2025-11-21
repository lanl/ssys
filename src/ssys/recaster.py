
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
    for s, e in sorted(exps.items(), key=lambda kv: kv[0].name):
        expr *= s**sp.Float(e)
    return sp.simplify(expr)

def recast_to_ssystem(sym: 'SymSystem') -> 'RecastResult':
    """
    Canonical S-system recast using log-derivative 'pool' auxiliaries:
      For X' = sum_j s_j (signed monomials), create one aux V_j per term with
        V_j' = + u_j * (∏_{ℓ≠j} V_ℓ)^(-1)   if coeff > 0
        V_j' = 0 - u_j * (∏_{ℓ≠j} V_ℓ)^(-1) if coeff < 0
      where u_j = |coeff_j| * ∏ original^{exp}.
      Then X = ∏_j V_j, and d/dt(∏ V_j) = ∑ s_j exactly.
    """
    new_equations: List[SSysEquation] = []
    new_variables: List[sp.Symbol] = []
    new_initials: Dict[sp.Symbol, float] = dict(sym.initials)   # keep params and originals
    factor_map: Dict[sp.Symbol, List[sp.Symbol]] = {}

    for Xi in sorted(sym.vars, key=lambda s: s.name):
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
    for s, e in sorted(exps.items(), key=lambda kv: kv[0].name):
        if abs(e) < 1e-14:
            continue
        if abs(e - 1.0) < 1e-14:
            parts.append(f"{s.name}")
        else:
            parts.append(f"{s.name}^{e:g}")
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

def product_expr(coeff: float, exps: Dict[sp.Symbol, float]) -> sp.Expr:
    expr = sp.Float(coeff)
    for s, e in sorted(exps.items(), key=lambda kv: kv[0].name):
        expr *= s**sp.Float(e)
    return sp.simplify(expr)

def latex_ssys(result: 'RecastResult') -> str:
    lines = []
    for eq in result.equations:
        g = product_expr(eq.growth[0], eq.growth[1])
        h = product_expr(eq.decay[0], eq.decay[1])
        lines.append(r"\dot{%s} = %s - %s" % (sp.latex(eq.var), sp.latex(g), sp.latex(h)))
    return r"\\begin{aligned}" + r"\\\\\n".join(lines) + r"\\end{aligned}"
