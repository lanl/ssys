# Recasting ODEs into S-system Canonical Form

This document explains the theory and practice of **exact algebraic recasting**—transforming ordinary differential equation (ODE) models into canonical S-system or Generalized Mass Action (GMA) form while preserving dynamics exactly on the invariant constraint manifold (given consistent initial conditions).

## Contents

1. [What is Recasting?](#what-is-recasting)
2. [The Hierarchy of ODE Forms](#the-hierarchy-of-ode-forms)
3. [Positivity Requirements](#positivity-requirements)
4. [Recasting Rules](#recasting-rules)
5. [When Does Recasting Succeed?](#when-does-recasting-succeed)
6. [Worked Examples from the Test Collection](#worked-examples-from-the-test-collection)
7. [Benefits of Canonical Form](#benefits-of-canonical-form)
8. [References](#references)

---

## What is Recasting?

**Recasting** is an algebraic transformation that converts an ODE system into a system with a specific canonical structure in higher dimension. The transformation is *exact*: the original and recast systems have identical dynamics for the original state variables **when restricted to the invariant constraint manifold** defined by the auxiliary variable definitions and consistent initial conditions.

### The S-system Canonical Form

An **S-system** is a system of ODEs where each equation has exactly two terms—a production term and a degradation term—both in power-law form:

$$\frac{dX_i}{dt} = \alpha_i \prod_j X_j^{g_{ij}} - \beta_i \prod_j X_j^{h_{ij}}$$

where:
- $\alpha_i, \beta_i \geq 0$ are rate constants (nonnegative)
- $g_{ij}, h_{ij} \in \mathbb{R}$ are kinetic orders (exponents)
- $X_j > 0$ are state variables

A **strict canonical S-system** additionally requires $\alpha_i, \beta_i > 0$ (strictly positive), achieved via ε-splitting when one coefficient would otherwise be zero. The Savageau & Voit (1987) theorem proves recasting to S-system form (allowing zero coefficients); strict canonical form is an optional convention.

This form was developed by M.A. Savageau as part of Biochemical Systems Theory (BST) in the 1960s–1980s.

### Historical Context

Savageau & Voit (1987) proved a foundational result: a broad class of ODE systems can be transformed via smooth change of variables into an S-system in higher dimension. Specifically, the theorem applies to ODEs whose right-hand sides are built from **elementary functions** (polynomials, rationals, exp, log, sin, cos, etc.) via sums, products, and nested compositions. ODEs involving special functions, integrals, or non-elementary constructs are not guaranteed by the theorem. However, SV87 notes that many special functions (including elliptic integrals and Bessel functions) can nonetheless be represented as S-systems—either because they satisfy ODEs that happen to be recastable, or via Taylor-series approximation where each truncation is an elementary function. The authors explicitly state their theorem provides only a *minimum estimate* of the range of systems that can be recast; the full range remains an open question.

**Important caveat**: The recast system is dynamically equivalent to the original *on an invariant constraint manifold* determined by the auxiliary variable definitions. Recasting introduces m−n constraints (where m is the new dimension and n the original), which must be satisfied by initial conditions. Trajectories are restricted to a reference manifold chosen by these initial conditions.

The test collection in `test_models2/` contains 28 examples from this foundational paper, demonstrating recasting across diverse domains: exponential decay, oscillators, chemical reactors, orbital mechanics, and boundary layer equations.

---

## The Hierarchy of ODE Forms

ODE systems can be classified by their right-hand-side structure:

```
General ODE  ⊃  GMA  ⊃  S-system  ⊃  Strict canonical S-system
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

Two power-law terms per equation (production minus degradation), with zero and constant terms allowed.

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

## Positivity Requirements

S-system variables must be **strictly positive** ($X_j > 0$) because the power-law terms $X_j^{g_{ij}}$ require positive bases when exponents are non-integer or negative. This is not merely a numerical convenience—it is structural.

### Translation to the Positive Orthant

When original variables can be zero or negative, recasting requires **translation** to ensure positivity:

1. **Variables that are always negative**: Use sign change $X \to -Z$
2. **Variables bounded below by $c < 0$**: Translate $X \to Z + c$ where $Z = X - c > 0$
3. **Variables bounded above by $c > 0$**: Define $Z = c - X > 0$
4. **Unbounded variables**: Replace $X$ with the difference of two positive variables, e.g., $X = X_{+} - X_{-}$ where both $X_+, X_- > 0$

Savageau & Voit (1987) explicitly use these translations in their examples (e.g., Van der Pol: "translate the variables with positive parameters $p$ and $q$").

**Important**: Selecting appropriate translation constants requires knowledge (or assumption) of bounds on the original variables over the integration interval. If the original variable $X$ can cross below $-p$ during evolution, a translation $X \to Z = X + p$ will fail (Z becomes non-positive). In practice, one either:
- Uses prior knowledge of variable ranges to select safe constants
- Validates a posteriori that variables remain positive throughout integration
- Uses the unbounded strategy (replacing $X$ with $X_+ - X_-$), which handles any range but doubles the variables

### Initial Condition Constraints

When auxiliary variables are introduced, their initial conditions must be **consistent with the definitions**. For example:
- If $D = K_m + S$ and $S(0) = S_0$, then $D(0) = K_m + S_0$
- If $E = \exp(-k \cdot T)$ and $T(0) = 0$, then $E(0) = 1$

These constraints define the **reference manifold** on which the recast system is equivalent to the original. Choosing inconsistent initial conditions produces spurious trajectories.

### Implications for `ssys`

The `ssys` tool supports some automatic transformations for positivity:
- **ε-splitting** (in `--mode canonical`): Ensures all α, β > 0 by adding slack terms
- **Denominator lifting** (Rule 4a): Introduces auxiliaries for rational expressions
- **Initial condition adjustment**: Can be configured to replace zero initial conditions with a small positive value via `EPS_INIT` comments in model files

**User responsibility**: Translation of non-positive variables to the positive orthant is *not* automatic. If the original model contains variables that can become zero or negative during integration, the user must pre-process the model with appropriate translations. If integration produces non-positive values in auxiliary variables, the recast system will produce incorrect results or fail.

### Numerical Drift Warning

The recast system is effectively a **constrained dynamical system**—auxiliary variables must satisfy their defining algebraic relations throughout integration. When integrating the lifted ODE in floating point, numerical errors can cause **drift off the constraint manifold**, leading to divergence between the recast and original systems.

**Remedies**:
1. **Back-substitution**: Periodically recompute auxiliary variables from their definitions (e.g., if $Y = \exp(X)$, reset $Y \leftarrow \exp(X)$ after each step)
2. **Projection**: Project the state onto the constraint manifold at each step
3. **DAE formulation**: Treat the system as a differential-algebraic equation with stabilized constraints
4. **Validation**: Compare trajectories of original and recast systems to detect drift

In our test runs on the 117-model collection (see [TEST_MODELS.md](TEST_MODELS.md)), naive integration with standard tolerances sufficed because integration intervals are short (typically T_END ≤ 200). You can verify this yourself by running `ssys validate` on any model, which compares original and recast trajectories and reports the maximum scaled error. For long-time integration or sensitive systems, constraint management may be necessary.

---

## Recasting Rules

Recasting proceeds by applying a set of algebraic rules to eliminate non-polynomial terms and reduce multi-term sums. Each rule introduces **auxiliary variables** with their own ODEs derived via the chain rule.

> **Note**: The rules below correspond directly to the lifting functions in `ssys`:
> - Rules 1–3: `lift_composite_functions()` and `lift_time_functions_to_autonomous()`
> - Rule 4: `lift_rational_functions()`
> - Rule 5: Product rule (implicit in term handling)
> - Rule 6: ε-splitting (in output formatting)
> - Rule 7: Clock state via `lift_time_functions_to_autonomous()`

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
| `m27_spiral` ↔ `S1987_A5_spiral` | Spiral curves (sin/cos parametric) → GMA |
| `m28_torus` ↔ `S1987_B4_torus` | Torus surface dynamics → GMA |
| `A2013_power_system_2machine` | Power system swing equations |
| `A2013_power_system_3machine` | 3-machine power network |
| `Z2022_pll_converter` | Phase-locked loop (PLL) dynamics |
| `DN2015b_sinx_recasting` | sin(x) recasting for inference |
| `V1988a_sin_exp_system` | Combined sin/exp nonlinearities |

### Rule 4: Sum Handling

Sum handling involves two distinct operations:

#### 4a: Decomposition (Denominator Lifting)

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

#### 4b: Canonical Sum Reduction (Product Splitting)

**Pattern**: Equation with $m$ terms (where $m > 2$) that needs reduction to S-system form

**Transformation** (from SV87 §4D): Replace a variable $X_i$ by the product of two new variables $X_{n+1} \cdot X_{n+2}$, then identify terms to separate the sum. This reduces the term count by one while increasing dimension by one.

**Example** (three-term sum → two-term):
```
Original:    X' = A·X^a - B·X^b - C·X^c       (3 terms)

Replace:     X = Y·Z  (product of two new variables)
Identify:    Y' = A·Y^a·Z^{a-1} - B·Y^b·Z^{b-1}
             Z' = -C·Y^{c-1}·Z^c

Result:      Two equations, each with ≤2 terms
```

This step is repeated until each equation has at most two terms. The initial conditions $Y(0)$ and $Z(0)$ must satisfy $Y(0) \cdot Z(0) = X(0)$, which defines the reference manifold. In multi-basin systems, different consistent initializations satisfying the product constraint correspond to different sections of the reference manifold, each with potentially distinct dynamics.

**Note**: This systematic reduction is more structured than simply "introducing a sum variable." The `ssys` implementation uses denominator lifting (4a) but does not currently implement the full canonical sum reduction (4b), which is why some GMA models do not reach S-system form under current `ssys`. Implementing Rule 4b would enable these models to reach strict S-system form at the cost of increased dimension.

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

**Caveat**: Introducing product auxiliaries is **not intrinsically simplifying**. If $X'$ and $Y'$ are multi-term expressions, then $P' = X' \cdot Y + X \cdot Y'$ produces even more terms. The product rule is useful for certain algebraic manipulations but may require subsequent sum reduction (Rule 4b) to reach S-system form.

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

**Important**: ε-splitting is an **optional post-processing convention** for ensuring both α and β coefficients are strictly positive. It is *not* part of Savageau & Voit's original theorem, which proves recasting to S-system form (allowing zero coefficients). The `ssys` tool uses ε-splitting only in `--mode canonical`.

**Numerical considerations**: While analytically benign (the ε terms cancel exactly), ε-splitting can introduce numerical artifacts:
- Very small ε may cause precision issues
- Moderately sized ε changes the apparent rate constants
- In stiff systems, ε-splitting may affect step-size selection

For most purposes, `--mode simplified` (default) produces equivalent results without these concerns.

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

---

## When Does Recasting Succeed?

Based on the test collection results (117 models), we can characterize when different outcomes occur:

### Achieves S-system Form (72 models)

Recasting reaches the two-term S-system form when:

1. **Model is already S-system**: No transformation needed
   - Examples: `m01_exp_decay`, `m02_logistic`, `m03_Lotka_Volterra`, `S1987_B5_rigid_body`

2. **GMA with pairable terms**: Terms can be combined via lifting
   - Examples: `m04_mass_action_branch`, `m22_Brusselator`, `S1987_E2_van_der_pol`

3. **General ODE with reducible structure**: Lifting produces ≤2 terms per equation
   - Examples: `m17_central_t`, `m13_composite_func_decomp`, `KPW2024_exponential_lifting`

### Stays at GMA Form (44 models)

Recasting stops at GMA (3+ terms per equation) when the current implementation does not reduce multi-term sums further:

1. **Multi-term sums not reduced**: Rule 4b (canonical sum reduction via product splitting) is not applied by `ssys`. These models *could* reach S-system form with increased dimension, but the implementation stops at GMA.
   - Examples: `m11_Monod_chemostat`, `m15_tryptophan_operon`, `m19_RM_predator_prey`

2. **Multiple independent saturation terms**: Each introduces an auxiliary, but the resulting equation still has 3+ terms
   - Examples: `m25_CSTR` ↔ `S1987_5_CSTR`, `S1987_E1_bessel`, `S1987_E3_duffing`

3. **Complex feedback structures**: GMA endemic infection models where multi-term structure persists
   - Examples: `V1988a_endemic_infection`, `V1990_kemper_endemic`, `V1990_cooke_endemic`

### GMA with Time-Varying Coefficients (1 model)

Some models achieve GMA structure but have coefficients that depend on time:

- **`m29_time_varying_beta`**: SIR model with β(t) via smooth step functions. The model is recast to power-law form, but the coefficient `beta_t` remains a function of clock state T via assignment rules, so it's not strict constant-coefficient GMA

### Summary by Directory

| Directory | → S-system | → GMA | → GMA (time-varying) | Total |
|-----------|------------|-------|----------------------|-------|
| `test_models1/` | 21 | 7 | 1 | 29 |
| `test_models2/` | 18 | 10 | 0 | 28 |
| `test_models3/` | 22 | 17 | 1 | 40 |
| `test_models4/` | 11 | 7 | 2 | 20 |
| **Total** | **72** | **41** | **4** | **117** |

**Classification Notes**:
- *S-system*: Successfully recast to canonical S-system form (1-2 monomial terms per equation)
- *GMA*: Generalized Mass Action form (multiple monomials with constant coefficients)
- *GMA (time-varying)*: Power-law structure but with coefficients that depend on clock state T

**Provenance**: These counts were computed by running `ssys recast --mode simplified` on all 117 models in `test_models{1,2,3,4}/` and classifying output based on the presence of "S-SYSTEM DYNAMICS" (S-system), "GMA" header (GMA), or time-dependent assignment rules in the recast `.ant` files. See [TEST_MODELS.md](TEST_MODELS.md) for per-model classifications.

Note: All 117 models successfully process through `ssys` — none remain in General form. While some models could theoretically reach strict S-system form via more complex transformations (canonical sum reduction, Rule 4b), `ssys` stops at GMA for multi-term equations.

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

**Result**: GMA form (3+ terms remain after denominator lifting)

**Classification**: General → GMA. In principle, reaching S-system form is possible via sum reduction (Rule 4b), which would further increase dimension; `ssys` does not currently implement this step.

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

**Original** (Cartesian coordinates):
```
Z1' = Z3
Z2' = Z4
Z3' = -Z1·(Z1² + Z2²)^(-3/2)
Z4' = -Z2·(Z1² + Z2²)^(-3/2)
```

**Recasting**: The term $(Z_1^2 + Z_2^2)^{-3/2}$ requires lifting via an auxiliary $Y = Z_1^2 + Z_2^2$. The resulting system has 5 variables but remains GMA because $Y' = 2Z_1 Z_3 + 2Z_2 Z_4$ introduces a multi-term sum.

**All 5 variants**: General → GMA

This family demonstrates that the algebraic recasting steps are symbolic/structural and independent of parameter values—only the equation structure determines which lifting rules apply. However, the *validity* of the recast system depends on trajectories staying in the assumed domain (positive orthant), which can depend on parameters and initial conditions. While Savageau & Voit (1987) show these orbit problems can be fully recast to S-system via more complex transformations (8 variables with product splitting), `ssys` stops at GMA.

---

## Benefits of Canonical Form

### Log-Linear Structure

Each *monomial term* in an S-system is log-linear:

$$\log\left(\alpha_i \prod_j X_j^{g_{ij}}\right) = \log(\alpha_i) + \sum_j g_{ij} \log(X_j)$$

**Note**: One cannot directly take the log of $X_i'$ because the net derivative (production minus degradation) can be negative. However, at **steady state**, production equals degradation:

$$\alpha_i \prod_j X_j^{g_{ij}} = \beta_i \prod_j X_j^{h_{ij}}$$

Taking logs yields linear equations in $\log(X_j)$:

$$\log(\alpha_i) + \sum_j g_{ij} \log(X_j) = \log(\beta_i) + \sum_j h_{ij} \log(X_j)$$

This enables:
- **Algebraic steady-state analysis**: Linear equations in log-transformed variables
- **Term-wise parameter estimation**: Methods like alternating regression exploit log-linearity of individual monomial terms rather than the net derivative
- **Sensitivity analysis**: Log-gains have direct interpretation as elasticity coefficients

### Parameter Estimation

The monomial log-linearity of S-systems enables convenient regression-style estimation of term parameters in certain settings. For example, alternating regression methods can separately fit production and degradation terms. However, whether parameters are actually identifiable depends on observability (which states are measured), experiment design, and potential model symmetries/degeneracies.

**Relevant test models**: The papers behind `DN2015_planetary_motion`, `DN2015b_michaelis_menten`, and `DN2015b_sinx_recasting` demonstrate S-system-based inference methods (Daniels & Nemenman 2015).

### Ground Truth for Structure Learning

Exactly recast models provide **benchmarks** for symbolic regression and structure-learning algorithms. One can:

1. Take a model from `test_models4/` (e.g., `Selkov1968` glycolytic oscillator)
2. Recast it exactly with `ssys`
3. Simulate synthetic data
4. Test whether data-driven methods recover the known structure


---

## References

- Savageau MA (1969). Biochemical systems analysis. I. Some mathematical properties of the rate law for the component enzymatic reactions. *J Theor Biol* 25:365–369.

- Savageau MA, Voit EO (1987). Recasting nonlinear differential equations as S-systems: a canonical nonlinear form. *Math Biosci* 87:83–115. **[Foundational paper; all `test_models2/` models derive from this work]**

- Voit EO (2013). Biochemical Systems Theory: A Review. *ISRN Biomathematics* 2013:897658.

- Hernández-Bermejo B, Fairén V, Brenig L (1998). Algebraic recasting of nonlinear systems of ODEs into universal formats. *J Phys A: Math Gen* 31:2415–2430. **[HBF1998_* models]**

- Daniels BC, Nemenman I (2015). Efficient inference of parsimonious phenomenological models of cellular dynamics using S-systems and alternating regression. *PLOS ONE* 10(3):e0119821. **[DN2015b_* models]**

- Marin-Sanguino A, et al. (2007). Optimization of biotechnological systems through geometric programming. *Theor Biol Med Model* 4:38. **[MS2007_* models]**

- Rust PF, Voit EO (1990). Statistical densities, cumulatives, quantiles, and power obtained by S-system differential equations. *J Amer Statist Assoc* 85:572–578. **[RV1990_* models]**

See [TEST_MODELS.md](TEST_MODELS.md) for complete documentation of the 117-model test collection and full reference list.
