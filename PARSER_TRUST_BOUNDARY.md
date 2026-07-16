# Parser Trust Boundary

This audit records the local parser threat model for ssys. It is intentionally
limited to work that can be verified from the source tree and local tests.

## Decision

ssys has one supported input trust mode for this release: **trusted-local
scientific model files**.

There is no hardened parser mode for arbitrary untrusted uploads, multi-tenant
services, browser-facing APIs, or adversarial resource-exhaustion testing. The
CLI and Python parsers must not be exposed directly to arbitrary user-submitted
Antimony or SBML text in security-sensitive environments.

## Parser Modes

| Mode | Entry points | Status | Trust boundary |
| --- | --- | --- | --- |
| SBML-first Antimony | `ssys.parse_antimony_via_sbml()`, `ssys-recast` | Only Antimony parser | Trusted local Antimony is parsed by the reference Antimony implementation, converted to SBML, read with libSBML, and then checked by ssys formula gates before SymPy conversion. |
| Direct SBML | `ssys.parse_sbml()`, `ssys.parse_sbml_from_string()` | Supported Python API | Trusted local SBML is read by libSBML and checked for supported features and formula identifiers/functions before symbolic conversion. |

No hardened parser mode or `parser="hardened"` option exists. If future hosted or
multi-tenant use is required, it needs a separate design with process isolation,
resource limits, dependency sandboxing, parser timeouts, and adversarial corpus
testing.

## Current Controls

- The CLI help and docs state that Antimony and SBML inputs are trusted
  scientific files, not safe untrusted uploads.
- SBML formula parsing rejects unknown identifiers and unsupported function
  calls before calling `sympy.sympify()`.
- SBML parsing rejects unsupported feature classes such as events, delays,
  constraints, invalid identifiers, duplicate rate rules, and malformed math
  with structured diagnostics.
- Initial-assignment failures are fail-closed by default. Warning mode is an
  explicit exploratory parser option, not release-grade validation behavior.
- Complete-gamma preprocessing in the RoadRunner backend uses a small Python AST
  allowlist for numeric arithmetic instead of `eval()`.
- Negative corpus tests assert that unsupported parser inputs fail before a
  successful recast artifact can be produced.

## Non-Goals

- No claim is made that C/C++ dependencies such as Antimony, libSBML, or
  RoadRunner are safe for adversarial inputs.
- No memory, CPU, recursion-depth, subprocess, or file-system sandbox is
  provided by ssys parsers.
- No hosted service, upload endpoint, or public documentation site is part of
  this local release.

## Local Audit Evidence

- `tests/test_negative_corpus.py`
- `tests/test_recaster.py::test_malicious_formula_string_rejected_before_sympify`
- `tests/test_public_api.py::test_cli_help_lists_stable_options`
- `tests/test_parser_trust_boundary.py`
- README `Input Trust Boundary`
- `CORRECTNESS_SPEC.md` supported/unsupported input classes
