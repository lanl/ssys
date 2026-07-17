# ssys — recast ODEs into canonical S‑system (or GMA) form
*Source: README.md | v0.7.0 | 2026-07-16*

[![PyPI version](https://img.shields.io/pypi/v/ssys.svg)](https://pypi.org/project/ssys/)
[![Python versions](https://img.shields.io/pypi/pyversions/ssys.svg)](https://pypi.org/project/ssys/)
[![License: MIT](https://img.shields.io/pypi/l/ssys.svg)](LICENSE)

ssys converts a system of ordinary differential equations into exact **S‑system**
or **GMA** (power‑law) form and verifies the result symbolically and numerically.
It reads and writes **Antimony**, can parse **SBML** directly, and can generate a
Jupyter notebook report comparing the original and recast models.

## Install

```bash
pip install ssys
```

Supported on **Python 3.10, 3.11, and 3.12**, and on Linux, macOS, and Windows
(all tested in CI). The install pulls in the SBML‑first parser and ODE simulation
stack (libRoadRunner, Antimony, python‑libsbml).

## Quickstart

### Recast one model

Antimony in, Antimony out — in Python, using only the stable top‑level API:

```python
import ssys

text = open("model.ant").read()
sym = ssys.parse_antimony_via_sbml(text)          # Antimony -> symbolic ODE system
result = ssys.recast_to_ssystem(sym, mode="simplified")
antimony = ssys.ssystem_to_antimony(result, model_name="model_recast", mode="simplified")

open("model_recast.ant", "w").write(antimony)
print("recast as:", ssys.classify_result(result, mode="simplified").value)
```

Already have SBML? Replace the first two lines with
`sym = ssys.parse_sbml("model.xml")`.

From the command line, the `ssys-recast` tool works on a **manifest** — a text
file with one `.ant` path per line. To recast a single model, point it at a
one‑line manifest:

```bash
printf '%s\n' model.ant > models.manifest
ssys-recast --manifest models.manifest --outdir out
# writes out/model_recast.ant and out/recast_report.ipynb
```

### Recast a batch and validate

```bash
ssys-recast --manifest test_models1/models.manifest --outdir out \
  --mode simplified --validate --validation-profile strict
```

`--validate` runs the correctness checks and writes a `*_validation.json` next to
each recast. `strict` is the release‑grade profile: a model counts as validated
only when every required check passes.

## What ssys does

- **S‑system** — at most two power‑law terms per equation, one production and one
  degradation: `Ẋᵢ = αᵢ · ∏ⱼ Xⱼ^gᵢⱼ − βᵢ · ∏ⱼ Xⱼ^hᵢⱼ`.
- **GMA** — each equation is a sum of power‑law monomials.
- Automatically **lifts** rational and composite terms (`exp`, `sin`, `log`,
  `1/(X+Y)`, …) into auxiliary variables, so a broad class of ODEs becomes exactly
  recastable — not just those already in power‑law form.
- Two output modes: `simplified` (default; preserves structure, allows
  single‑term equations) and `canonical` (strict two‑term form via ε‑splitting,
  following Savageau & Voit 1987).

See [RECASTING.md](RECASTING.md) for the theory and worked examples, and
[CORRECTNESS_SPEC.md](CORRECTNESS_SPEC.md) for the exact supported input/output
contract. The stable API surface is documented in [PUBLIC_API.md](PUBLIC_API.md).

## Validation

Validation is **fail‑closed**: `overall_pass` is true only when every required
check for the selected profile returns `pass`. Unsupported, not‑attempted,
timed‑out, inconclusive, and failed checks are never passes. Every result carries
a machine‑readable `reason` and the reports carry a `schema_version`; load the
packaged JSON Schema with `ssys.load_validation_report_schema()`.

| Profile | Intended use |
| --- | --- |
| `strict` | Release‑grade; the default for `--validate`. Roundtrip + parser + mapping + symbolic + numerical + trajectory + auxiliary identities. |
| `structural` | Fast smoke test: roundtrip, parser, and mapping only. |
| `symbolic` | Exact symbolic proof, no simulation. |
| `numerical` | Pointwise numerical support over sampled domains. |
| `trajectory` | Solver‑backed trajectory support. |

- **Symbolic** evidence is an exact algebraic proof — the Jacobian chain‑rule
  identity `J_Φ(Z)·f_recast(Z) = f_orig(Φ(Z))` simplifies to zero.
- **Numerical** evidence is pointwise support at deterministic random samples
  (ε = 10⁻⁵); it can find counterexamples but does not prove global equivalence.
- **Trajectory** evidence is solver‑backed behavioral support over a time grid
  (default 3% peak‑scaled threshold); exact claims require symbolic validation.

## Supported input

Antimony reactions (`A + B -> C; k*A*B`), initializations (`X = 2.5`), explicit
rate rules (`X' = ...`), boundary species (`$X`), positive parameters, elementary
functions (`exp`, `log`, `sin`, `cos`, `tan`, `sqrt`, `sinh`, `cosh`, `tanh`, …),
rational functions, and assignment rules (`Z := X + Y`). Simulation settings come
from `@SIM` comments, e.g. `// @SIM T_START=0 T_END=100 N_STEPS=500 EPS_INIT=1e-6`.

For SBML, species `hasOnlySubstanceUnits`, non‑unit and constant compartment
volumes, a species/model `conversionFactor`, and constant reaction stoichiometry
(including a `stoichiometryMath` that folds to a constant) are all honored.

Not yet supported: modules, events/piecewise functions, and non‑positive state
variables. At the SBML trust boundary these are rejected with a structured
`unsupported_feature` error rather than mis‑integrated: variable reaction
stoichiometry (a non‑constant `stoichiometryMath`, or a `speciesReference` id
driven by a rule/`InitialAssignment`) and time‑varying compartment volume (a
rate‑rule compartment, or an assignment‑rule compartment that does not fold to a
constant, holding a concentration species — its `-[S]·(dV/dt)/V` dilution term is
not modeled). S‑systems require positive states, so a zero initial condition is
ε‑regularized to a small positive `EPS_INIT` (default `1e-6`, configurable via
`@SIM`); this solves a slightly perturbed IVP near `t = 0`.

## Scope and trust boundary

ssys is **alpha** software. Treat the APIs, generated Antimony details, and the
validation‑report format as subject to change until the package leaves alpha.
Local artifact builds, validation reports, and benchmark evidence are the source
of truth for release claims.

**Trust boundary:** ssys treats Antimony and SBML inputs as trusted local
scientific model files, not as safe untrusted uploads. Do not expose the CLI or
parser directly to arbitrary user‑submitted model text in a multi‑tenant or
security‑sensitive service. See [PARSER_TRUST_BOUNDARY.md](PARSER_TRUST_BOUNDARY.md)
for the parser audit and threat model.

## Development and releases

Contributor setup, the full test suite, and the local release gates are covered in
[CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
In brief:

```bash
uv sync --extra dev                 # dev environment (add --extra dae for DAE validation)
uv run pytest -m "not slow"         # fast test suite
uv run ssys-recast --help
```

The four committed model sets (`test_models1/` through `test_models4/`, 117 models
in total) are described in [TEST_MODELS.md](TEST_MODELS.md). Releases are published
to PyPI automatically by `.github/workflows/publish.yml` via trusted publishing;
see [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the procedure.

## Troubleshooting

- **Missing RoadRunner/CVODE.** ODE and assignment‑rule trajectory validation
  need libRoadRunner. If it is unavailable, required trajectory checks report
  `unsupported` (never a pass). Re‑sync with `uv sync --extra dev`.
- **Missing DAE backend.** DAE‑required trajectory validation needs the `dae`
  extra (`uv sync --extra dev --extra dae`); missing IDA/SUNDIALS is reported as
  `unsupported`.
- **Parser errors.** `ssys-recast` parses Antimony through the reference
  implementation and SBML/libSBML (`ssys.parse_antimony_via_sbml`); unsupported
  features (events, delays, unknown functions, malformed input) are rejected before
  any artifact is written.
- **Numerical sampling failures.** `invalid_sampling_domain` means the model
  metadata gave no usable finite positive domain; `nonfinite_sample` means a
  sampled point hit a singular surface. Reports record the seed, sampled ranges,
  parameters, threshold, and counterexamples needed to reproduce the failure.
- **Trajectory mismatches** can indicate a recasting error, a domain/parameter
  issue, or a model that needs symbolic rather than solver‑backed evidence.
  Confirm with the `symbolic` or `numerical` profile before treating it as a bug.

## References

- Savageau, M. A., & Voit, E. O. (1987). Recasting nonlinear differential
  equations as S‑systems: a canonical nonlinear form. *Mathematical Biosciences*,
  87(1), 83–115.
- Smith, L. P., Bergmann, F. T., Chandran, D., & Sauro, H. M. (2009). Antimony: a
  modular model definition language. *Bioinformatics*, 25(18), 2452–2454.

## License and citation

MIT License — see [LICENSE](LICENSE) for the full text and the accompanying Triad
National Security, LLC / U.S. Government copyright notice. Produced at Los Alamos
National Laboratory under U.S. Government contract 89233218CNA000001 and released
under LANL reference **O5066**. © 2026 Triad National Security, LLC.

Please cite ssys using the metadata in [CITATION.cff](CITATION.cff).
