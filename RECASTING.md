# Recasting ODEs into S-system Canonical Form

This document explains the theory and practice of **exact algebraic recasting**—transforming ordinary differential equation (ODE) models into canonical S-system or Generalized Mass Action (GMA) form while preserving dynamics exactly.

## Contents

1. [What is Recasting?](#what-is-recasting)
2. [The Hierarchy of ODE Forms](#the-hierarchy-of-ode-forms)
3. [Recasting Rules](#recasting-rules)
4. [When Does Recasting Succeed?](#when-does-recasting-succeed)
5. [Worked Examples from the Test Collection](#worked-examples-from-the-test-collection)
6. [Benefits of Canonical Form](#benefits-of-canonical-form)
7. [References](#references)

---

## What is Recasting?

**Recasting** is an algebraic transformation that converts an arbitrary ODE system into an equivalent system with a specific canonical structure. The transformation is *exact*: the original and recast systems have identical dynamics for the original state variables.

### The S-system Canonical Form

An **S-system** is a system of ODEs where each equation has exactly two terms—a production term and a degradation term—both in power-law form:

$$\frac{dX_i}{dt} = \alpha_i \prod_j X_j^{g_{ij}} - \beta_i \prod_j X_j^{h_{ij}}$$

where:
- $\alpha_i, \beta_i > 0$ are rate constants
- $g_{ij}, h_{ij} \in \mathbb{R}$ are kinetic orders (exponents)
- $X_j > 0$ are state variables

This form was developed by M.A. Savageau as part of Biochemical Systems Theory (BST) in the 1960s–1980s.

### Historical Context

Savageau & Voit (1987) proved a remarkable result: **any ODE system composed of elementary functions can be exactly recast into S-system form**, given sufficient auxiliary variables. This means S-systems are not a narrow special case but a universal framework for nonlinear dynamics.

The test collection in `test_models2/` contains 28 examples from this foundational paper, demonstrating recasting across diverse domains: exponential decay, oscillators, chemical reactors, orbital mechanics, and boundary layer equations.

---

## The Hierarchy of ODE Forms

ODE systems can be classified by their right-hand-side structure:

```
General ODE  ⊃  GMA  ⊃  S-system  ⊃  Canonical S-system
```

### General ODE

Arbitrary combinations of elementary functions: polynomials, rationals, exp, log, sin, cos, and their compositions.

**Example** (Michaelis-Menten kinetics):
```
X' = Vmax·S/(Km + S) - k·X
```

*Test models*: Most models in `test_models4/` start as General ODEs (e.g., `Goldbeter1996`, `Kholodenko2000`).

### Generalized Mass Action (GMA)

A sum of power-law monomials:

$$\frac{dX_i}{dt} = \sum_k c_k \prod_j X_j^{e_{kj}}$$

**Example** (three-term sum):
```
X' = a·X - b·X·Y - c·X·Z
```

*Test models*: `m04_mass_action_branch`, `m22_Brusselator`, `S1987_B2_linear_chain`, `V1993_rossler_band`

### S-system

Two power-law terms per equation (production minus degradation), with zero terms allowed.

**Example** (logistic growth):
```
X' = r·X - (r/K)·X²
```

*Test models*: `m02_logistic` ↔ `S1987_A4_logistic`, `m03_Lotka_Volterra` ↔ `S1987_B1_lotka_volterra`

### Strict Canonical S-system

S-system form with both coefficients strictly positive (achieved via ε-splitting when needed).

**Example** (pure degradation with splitting):
```
Original:  X' = -k·X
Canonical: X' = ε·X - (k+ε)·X    (for small ε > 0)
```

*Test models*: `m07_SIR` demonstrates ε-splitting for reactions with only one natural direction. `m18_Goodwin` recasts to strict canonical form.

---

## Recasting Rules

Recasting proceeds by applying a set of algebraic rules to eliminate non-polynomial terms and reduce multi-term sums. Each rule introduces **auxiliary variables** with their own ODEs derived via the chain rule.

> **Note**: The rules below correspond directly to the lifting functions in `ssys`:
> - Rules 1–3: `lift_composite_functions()` and `lift_time_functions_to_autonomous()`
> - Rule 4: `lift_rational_functions()`
> - Rule 5: Product rule (implicit in term handling)
> - Rule 6: ε-splitting (in output formatting)
> - Rule 7: `add_dummy_for_constants()` (canonical mode only)

### Rule 1: Exponential Lifting

**Pattern**: $Y = e^{f(X)}$

**Transformation**: Introduce auxiliary $Y$ satisfying $Y' = f'(X) \cdot Y$

**Example**:
```
Original:    X' = a·exp(-b·t)
Lifted:      X' = a·E,  where E = exp(-b·t)
             E' = -b·E·T'  (and T' = 1 for explicit time)
```

**Test models demonstrating this rule**:
| Model | Original | Lifted |
|-------|----------|--------|
| `m01_exp_decay` ↔ `S1987_A1_exponential` | `X' = -k·X` | Already S-system |
| `m13_composite_func_decomp` ↔ `V1988b_exponential_ode` | `X' = exp(-X)` | General → S-system |
| `S1987_4C_exp_composition` | Exponential composition | General → S-system |
| `KPW2024_exponential_lifting` | Machine learning example | General → Canonical S-system |
| `m29_time_varying_beta` | SIR with exp-based step function | General → General (complex) |

### Rule 2: Logarithmic Lifting

**Pattern**: $Y = \log(X)$

**Transformation**: $Y' = X'/X = X' \cdot X^{-1}$

**Example**:
```
Original:    f(X) involves log(X)
Lifted:      Introduce L = log(X), then L' = X'/X
```

**Test models**:
| Model | Description |
|-------|-------------|
| `V1992_log_ode` | ODE with logarithmic term → General → GMA |
| `m24_scalarExpLog` | Combined exp/log dynamics → General → S-system |

### Rule 3: Trigonometric Lifting

**Pattern**: Functions involving $\sin(\theta)$ or $\cos(\theta)$

**Transformation**: Introduce coupled pair $S = \sin(\theta)$, $C = \cos(\theta)$ with:
```
S' = C · θ'
C' = -S · θ'
```

**Example** (cosine-driven growth):
```
Original:    X' = X·cos(t)
Lifted:      X' = X·C,  where C = cos(t), S = sin(t)
             C' = -S,   S' = C     (since t' = 1)
```

**Test models demonstrating trigonometric recasting**:
| Model | Application |
|-------|-------------|
| `m26_cos_growth` ↔ `S1987_A3_cos_growth` | Basic cosine dynamics |
| `m27_spiral` ↔ `S1987_A5_spiral` | Spiral curves (sin/cos parametric) |
| `m28_torus` ↔ `S1987_B4_torus` | Torus surface dynamics |
| `A2013_power_system_2machine` | Power system swing equations |
| `A2013_power_system_3machine` | 3-machine power network |
| `Z2022_pll_converter` | Phase-locked loop (PLL) dynamics |
| `DN2015b_sinx_recasting` | sin(x) recasting for inference |
| `V1988a_sin_exp_system` | Combined sin/exp nonlinearities |

### Rule 4: Sum Handling (Denominator Lifting)

**Pattern**: Rational expressions with sums in denominators: $\frac{f(X)}{g(X) + h(X)}$

**Transformation**: Introduce auxiliary $D = g(X) + h(X)$, then $D' = g'(X) + h'(X)$

**Example** (central t-distribution):
```
Original:    f' = -(ν+1)·t/(ν + t²)·f
Problem:     (ν + t²) is a sum in denominator

Lifted:      Introduce D = ν + t²
             D' = 2·t·t' = 2·t    (since t' = 1)
             f' = -(ν+1)·t·D⁻¹·f  (now power-law in D)
```

**Test models demonstrating sum handling**:
| Model | Sum Structure | Outcome |
|-------|---------------|---------|
| `m17_central_t` ↔ `RV1990_central_t_density` | ν + t² | General → S-system |
| `RV1990_central_chisquared` | χ² distribution | GMA → S-system |
| `RV1990_central_F` | F distribution | General → GMA |
| `m11_Monod_chemostat` ↔ `S1988_Monod_chemostat` | Km + S | General → GMA |
| `m14_Michaelis_Menten_prod_deg` ↔ `MS2007_MM_to_GMA` | Two MM terms | General → General |
| `m05_three_term_sum` ↔ `S1987_4D_sum_reduction` | Multi-term sum | GMA → S-system |
| `S1993_sum_radical` | Sum with radical | Canonical S-system → S-system |
| `S1993_mixed_terms` | Multiple +/− terms | GMA → S-system |

### Rule 5: Product Rule (Factoring)

**Pattern**: Product of variables that need separate dynamics

**Transformation**: If $Z = X \cdot Y$, then $Z' = X' \cdot Y + X \cdot Y'$

**Example** (Brusselator):
```
Original:    x' = A - (B+1)·x + x²·y
Introduce:   P = x·y (product auxiliary)
             P' = x'·y + x·y' (chain rule)
```

**Test models**:
| Model | Description |
|-------|-------------|
| `m22_Brusselator` ↔ `HBF1998_brusselator` | Classic chemical oscillator |
| `S1987_B5_rigid_body` | Euler equations |
| `V1988b_gma_to_ssystem` | GMA → S-system via product rule |

### Rule 6: ε-Splitting (Slack Variables)

**Pattern**: Single-term equation (only production or only degradation)

**Transformation**: Split using small $\varepsilon > 0$:
```
X' = -k·X                           (single term)
X' = ε·X - (k+ε)·X                  (canonical two-term)
```

The dynamics are identical since the ε terms cancel.

**Test models using ε-splitting**:
| Model | Reason for Splitting |
|-------|---------------------|
| `m07_SIR` | Infection/recovery have natural directions |
| `S1987_A1_exponential` | Pure decay |
| `S1987_B_binary` | Half-life transformation |
| Many `test_models1/` models | Comments show hand-crafted ε-splits |

### Rule 7: Clock State (Time-Dependent Systems)

**Pattern**: Explicit time dependence in ODEs (functions of `time`)

**Transformation**: Introduce clock state $T$ with $T' = 1$, $T(0) = 0$, then substitute `time` → `T` everywhere.

**Example**:
```
Original:    X' = exp(-k·time)·X
Clock lift:  X' = E·X,  where E = exp(-k·T)
             T' = 1
             E' = -k·E  (since dE/dt = -k·exp(-k·T) = -k·E)
```

This converts non-autonomous systems (time-dependent) to autonomous systems (state-dependent only).

**Test models with time-dependent dynamics**:
| Model | Time Dependence |
|-------|-----------------|
| `m29_time_varying_beta` | β(t) via tanh step functions |
| `m26_cos_growth` ↔ `S1987_A3_cos_growth` | cos(t) oscillation |
| `A2013_power_system_2machine` | sin(δ-θ) with time evolution |

### Rule 8: Constant Term Handling (Canonical Mode)

**Pattern**: Constant term in ODE (not involving any state variables)

**Transformation**: Introduce `dummy_const` variable where `dummy_const = 1` always.

**Example**:
```
Original:    X' = A - k·X        (A is a constant)
Transformed: X' = A·dummy_const^0 - k·X
```

Since `dummy_const^0 = 1` for all time, this preserves mathematical equivalence while expressing the constant in power-law form. This is used only in canonical mode for strict S-system representation.

**Note**: In simplified mode, `ssys` outputs constant terms directly (e.g., `T' = 1`) without dummy variables, which is valid GMA but not strict S-system.

---

## When Does Recasting Succeed?

Based on the test collection results (117 models), we can characterize when different outcomes occur:

### Achieves S-system Form (61 models)

Recasting reaches the two-term S-system form when:

1. **Model is already S-system** (19 models): No transformation needed
   - Examples: `m01_exp_decay`, `m02_logistic`, `m03_Lotka_Volterra`, `S1987_B5_rigid_body`

2. **GMA with pairable terms** (20 models): Terms can be combined via lifting
   - Examples: `m04_mass_action_branch`, `m22_Brusselator`, `S1987_E2_van_der_pol`

3. **General ODE with reducible structure** (22 models): Lifting produces ≤2 terms per equation
   - Examples: `m17_central_t`, `m26_cos_growth`, `S1987_D1-D5_orbit` (all 5 eccentricity variants)

### Stays at GMA Form (37 models)

Recasting stops at GMA (3+ terms per equation) when:

1. **Irreducible multi-term sums**: Cannot be factored into ≤2 terms
   - Examples: `m11_Monod_chemostat`, `m15_tryptophan_operon`, `m19_RM_predator_prey`

2. **Multiple independent saturation terms**: Each introduces an auxiliary but total exceeds 2
   - Examples: `m25_CSTR` ↔ `S1987_5_CSTR`, `S1987_E1_bessel`, `S1987_E3_duffing`

3. **Complex feedback structures**: GMA endemic infection models
   - Examples: `V1988a_endemic_infection`, `V1990_kemper_endemic`, `V1990_cooke_endemic`

### Remains General Form (19 models)

The model stays in General form when:

1. **Coupled rational functions**: Multiple Michaelis-Menten or Hill terms that cannot be fully decomposed
   - Examples: `m14_Michaelis_Menten_prod_deg`, `MS2007_tryptophan_operon`, `I1988_metabolic_pathway`

2. **Complex sigmoidal kinetics**: Nested saturations
   - Examples: `V2005_bistable_gene`, `P2011_branched_SC`, most of `test_models4/`

3. **Time-dependent constructs**: Complex coefficient variations
   - Examples: `m29_time_varying_beta` (exp-based step functions remain complex)

### Summary by Directory

| Directory | → S-system | → GMA | → General | Total |
|-----------|------------|-------|-----------|-------|
| `test_models1/` | 17 | 8 | 4 | 29 |
| `test_models2/` | 19 | 9 | 0 | 28 |
| `test_models3/` | 17 | 15 | 8 | 40 |
| `test_models4/` | 8 | 5 | 7 | 20 |
| **Total** | **61** | **37** | **19** | **117** |

Note: `test_models2/` (Savageau & Voit 1987 examples) achieves 100% recasting (no models remain General), demonstrating that the foundational examples were chosen to showcase successful transformations.

---

## Worked Examples from the Test Collection

### Example 1: Exponential Decay (Trivial)

**Model**: `m01_exp_decay.ant` ↔ `S1987_A1_exponential.ant`

```
X' = -k·X
```

**Analysis**: Already S-system form (single degradation term). Canonical form uses ε-splitting:
```
X' = ε·X - (k+ε)·X
```

**Classification**: S-system → S-system (no lifting needed)

---

### Example 2: Central t-Distribution (Sum Lifting)

**Model**: `m17_central_t.ant` ↔ `RV1990_central_t_density.ant`

**Original**:
```
t' = 1
f' = -(ν+1)·t/(ν + t²)·f
```

**Problem**: The term `(ν + t²)` is a sum in the denominator—not power-law.

**Recasting Steps**:
1. Introduce auxiliary: $D = \nu + t^2$
2. Derive its ODE: $D' = 2t \cdot t' = 2t$
3. Rewrite: $f' = -(\nu+1) \cdot t \cdot D^{-1} \cdot f$

**Recast System** (3 → 4 variables):
```
t' = 1
D' = 2·t
f' = (ν+1)·c·t⁻¹·f - (ν+1)·t·D⁻¹·f    (with shift c to avoid t=0)
```

**Classification**: General → S-system

---

### Example 3: Van der Pol Oscillator (GMA → S-system)

**Model**: `m10_van_der_Pol.ant` ↔ `S1987_E2_van_der_pol.ant`

**Original** (2nd order converted to 1st order):
```
x' = y
y' = μ·(1 - x²)·y - x
```

**Expanded**:
```
x' = y
y' = μ·y - μ·x²·y - x
```

**Problem**: Three terms in the y-equation (GMA form).

**Recasting**: Introduce auxiliaries to absorb the multi-term structure. The paper gives a 5-variable S-system using transformations like $X_1 = x + p$, $X_2 = y + q$ (shifted to ensure positivity).

**Classification**: GMA → S-system (requires careful auxiliary construction)

---

### Example 4: Monod Chemostat (General → GMA)

**Model**: `m11_Monod_chemostat.ant` ↔ `S1988_Monod_chemostat.ant`

**Original**:
```
S' = D·(S₀ - S) - (μmax·S/(Km + S))·X/Y
X' = (μmax·S/(Km + S))·X - D·X
```

**Problem**: Michaelis-Menten term `S/(Km + S)` is not power-law.

**Recasting**:
1. Introduce auxiliary: $M = K_m + S$
2. Then: $M' = S' = D(S_0 - S) - ...$
3. Rewrite MM term as: $S/M = S \cdot M^{-1}$ (power-law!)

**Result**: GMA form (but 3+ terms remain)

**Classification**: General → GMA (cannot reach S-system due to multi-term structure)

---

### Example 5: Brusselator (GMA → S-system)

**Model**: `m22_Brusselator.ant` ↔ `HBF1998_brusselator.ant`

**Original**:
```
x' = A - (B+1)·x + x²·y
y' = B·x - x²·y
```

**Recasting Strategy** (from Hernández-Bermejo et al. 1998):

The Brusselator can be recast to "unimonomial" form using auxiliary variables. The key insight is that introducing $z = x \cdot y$ (product auxiliary) and additional variables allows each equation to have a single nonlinear term.

**Classification**: GMA → S-system (elegant algebraic manipulation)

---

### Example 6: SIR Epidemic Model (ε-Splitting)

**Model**: `m07_SIR.ant`

**Original**:
```
S' = -β·S·I
I' = β·S·I - γ·I
R' = γ·I
```

**Problem**: The S and R equations have only one term each.

**Canonical Form** (ε-splitting):
```
S' = ε·S·I - (β+ε)·S·I       (net = -β·S·I)
I' = β·S·I - γ·I              (already two terms)
R' = (γ+ε)·I - ε·I            (net = γ·I)
```

**Related model**: `m29_time_varying_beta.ant` extends this with a time-varying β(t) using smooth step functions—demonstrating how to handle non-autonomous systems.

**Classification**: S-system → S-system (canonical via ε-splitting)

---

### Example 7: Two-Body Orbit Problem (Parameter Family)

**Models**: `S1987_D1_orbit_e0.1.ant` through `S1987_D5_orbit_e0.9.ant`

These 5 models represent the same orbital mechanics problem with different eccentricities (e = 0.1, 0.3, 0.5, 0.7, 0.9):

**Original** (radial coordinates):
```
r' = v_r
θ' = L/(m·r²)
v_r' = L²/(m·r³) - GM/r²
```

**Recasting**: The `1/r²` and `1/r³` terms are already power-law! The angular momentum dynamics lift cleanly.

**All 5 variants**: General → S-system

This family demonstrates that structural recasting outcomes are often independent of parameter values—only the equation structure matters.

---

## Benefits of Canonical Form

### Log-Linear Structure

In S-system form, taking logarithms converts multiplicative power-law dynamics into additive linear combinations:

$$\log(X_i') = \log(\alpha_i) + \sum_j g_{ij} \log(X_j) + \text{(correction terms)}$$

This enables:
- **Linear regression for parameter estimation**: The exponents $g_{ij}$ appear linearly in log-space
- **Algebraic steady-state analysis**: Setting $X_i' = 0$ yields linear equations in log-transformed variables
- **Sensitivity analysis**: Log-gains have direct interpretation as elasticity coefficients

### Identifiability

S-system parameters are often **structurally identifiable** from time-series data because the log-linear structure separates the roles of rate constants and exponents. This was exploited by Daniels & Nemenman (2015) for efficient network inference.

**Relevant test models**: The papers behind `DN2015_planetary_motion`, `DN2015b_michaelis_menten`, and `DN2015b_sinx_recasting` demonstrate S-system-based inference methods.

### Ground Truth for Structure Learning

Exactly recast models provide **benchmarks** for symbolic regression and structure-learning algorithms like SINDy. One can:

1. Take a model from `test_models4/` (e.g., `Selkov1968` glycolytic oscillator)
2. Recast it exactly with `ssys`
3. Simulate synthetic data
4. Test whether SINDy or other methods recover the known structure

### Numerical Methods

The uniform power-law structure of S-systems enables specialized numerical integrators. Some historical S-system solvers achieved faster integration than general ODE solvers by exploiting the monomial structure.

---

## References

- Savageau MA (1969). Biochemical systems analysis. I. Some mathematical properties of the rate law for the component enzymatic reactions. *J Theor Biol* 25:365–369.

- Savageau MA, Voit EO (1987). Recasting nonlinear differential equations as S-systems: a canonical nonlinear form. *Math Biosci* 87:83–115. **[Foundational paper; all `test_models2/` models derive from this work]**

- Voit EO (2013). Biochemical Systems Theory: A Review. *ISRN Biomathematics* 2013:897658.

- Hernández-Bermejo B, Fairén V, Brenig L (1998). Algebraic recasting of nonlinear systems of ODEs into universal formats. *J Phys A: Math Gen* 31:2415–2430. **[HBF1998_* models]**

- Daniels BC, Nemenman I (2015). Efficient inference of parsimonious phenomenological models of cellular dynamics using S-systems and alternating regression. *PLOS ONE* 10(3):e0119821. **[DN2015b_* models]**

- Brunton SL, Proctor JL, Kutz JN (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS* 113:3932–3937.

- Marin-Sanguino A, et al. (2007). Optimization of biotechnological systems through geometric programming. *Theor Biol Med Model* 4:38. **[MS2007_* models]**

- Rust PF, Voit EO (1990). Statistical densities, cumulatives, quantiles, and power obtained by S-system differential equations. *J Amer Statist Assoc* 85:572–578. **[RV1990_* models]**

See [TEST_MODELS.md](TEST_MODELS.md) for complete documentation of the 117-model test collection and full reference list.
