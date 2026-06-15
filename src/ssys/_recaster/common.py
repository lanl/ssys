"""Shared constants and regex patterns for recaster internals."""

import re

arrow_pat = re.compile(r"<->|->")
prime_rule_pat = re.compile(r"^\s*\$?([A-Za-z_]\w*)\s*'\s*=\s*(.+)$")
func_def_pat = re.compile(r"^([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:=\s*(.+)$")
func_call_start_pat = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
simple_identifier_pat = re.compile(r"^[A-Za-z_]\w*$")
simple_numeric_literal_pat = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

EPS_INIT = 1e-6
EPS_SLACK = 1.0

ANTIMONY_RESERVED_KEYWORDS = frozenset({
    "abs",
    "acos",
    "and",
    "asin",
    "at",
    "atan",
    "ceil",
    "ceiling",
    "compartment",
    "const",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "csch",
    "DNA",
    "end",
    "eq",
    "exp",
    "ext",
    "factorial",
    "floor",
    "gamma",
    "geq",
    "gt",
    "inf",
    "leq",
    "ln",
    "log",
    "log10",
    "lt",
    "max",
    "min",
    "model",
    "nan",
    "neq",
    "not",
    "or",
    "piecewise",
    "pi",
    "pow",
    "RNA",
    "sec",
    "sech",
    "sin",
    "sinh",
    "species",
    "sqrt",
    "tan",
    "tanh",
    "var",
    "xor",
})

__all__ = [
    "ANTIMONY_RESERVED_KEYWORDS",
    "EPS_INIT",
    "EPS_SLACK",
    "arrow_pat",
    "func_call_start_pat",
    "func_def_pat",
    "prime_rule_pat",
    "simple_identifier_pat",
    "simple_numeric_literal_pat",
]
