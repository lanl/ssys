"""Compatibility API for recasting, parsing, lifting, and formatting helpers.

Implementation is split across ``ssys._recaster`` modules; this module preserves
historical imports from ``ssys.recaster``.
"""

from ssys.classification import (
    classify_result,
    classify_solver_requirement,
    classify_sym_system_solver_requirement,
    classify_system,
)
from ssys.math_utils import (
    _expand_exps_through_factors,
    _exponents_match,
    _get_coefficient_sign,
    _is_term_monomial,
    expand_to_terms,
    product_expr,
)
from ssys.metadata import (
    _extract_sim_metadata,
    _extract_solver_requirement_metadata,
    _format_antimony_number,
    _format_sim_metadata_lines,
    _format_solver_metadata_lines,
    normalize_solver_requirement,
)
from ssys.types import (
    GMAEquation,
    NegativeInitialConditionError,
    RecastResult,
    RecastStatus,
    SBMLParseError,
    SolverRequirement,
    SSysEquation,
    SymSystem,
    SystemClass,
)
from ssys._recaster.common import EPS_INIT, EPS_SLACK
from ssys._recaster.algorithms import (
    _analyze_ode_terms,
    _direct_ssystem_recast,
    _gma_recast,
    _is_coefficient_positive,
    _pool_ssystem_recast,
    _requires_gma,
    _should_attempt_pool_construction,
    _validate_pool_result,
    canonicalize_aux_names,
    recast_to_ssystem,
    term_to_coeff_exps,
)
from ssys._recaster.antimony_formatting import (
    _failed_to_antimony,
    _format_factor,
    _format_symbolic_coeff,
    _ssystem_to_antimony_canonical,
    _ssystem_to_antimony_simplified,
    gma_to_antimony,
    product_to_antimony,
    ssystem_to_antimony,
)
from ssys._recaster.latex_formatting import _latex_power_law, latex_odes, latex_ssys
from ssys._recaster.lifting import (
    AutonomousLiftResult,
    _build_composite_inverse_mappings,
    _detect_exp_decay_pattern,
    _detect_harmonic_pattern,
    _detect_sqrt_of_squared_pattern,
    _detect_tanh_sigmoid_pattern,
    _is_composite_function_expr,
    _is_time_only_function,
    _requires_positivity_transform,
    add_dummy_for_constants,
    create_auxiliary_for_denominator,
    find_composite_functions,
    find_rational_denominators,
    find_sqrt_of_sums,
    lift_composite_functions,
    lift_exp_decay,
    lift_harmonic,
    lift_rational_functions,
    lift_squared_for_sqrt,
    lift_tanh_sigmoid,
    lift_time_functions_to_autonomous,
)
from ssys._recaster.names import (
    _apply_name_sanitization,
    _build_name_sanitization_map,
    _collect_antimony_names,
    _format_antimony_token,
    _sanitize_antimony_name,
)
from ssys._recaster.parsing import (
    _antimony_to_sympy_syntax,
    _evaluate_initial_assignment,
    _iter_kinetic_law_local_parameters,
    _numeric_param_subs,
    _parse_sbml_document,
    _preprocess_antimony_text,
    _reaction_scope_name,
    _replace_formula_identifiers,
    _sanitize_sbml_identifier,
    _sympify_sbml_formula,
    _sympy_to_antimony_syntax,
    _unique_identifier,
    _warn_or_raise_initial_assignment_error,
    parse_antimony_via_sbml,
    parse_sbml,
    parse_sbml_from_string,
)
from ssys._recaster.templates import (
    _expand_function_calls,
    _find_matching_paren,
    _find_next_template_call,
    _format_function_argument_for_substitution,
    _parse_function_args,
    _substitute_param,
    expand_antimony_function_templates,
)

__all__ = [name for name in globals() if not name.startswith("__")]
