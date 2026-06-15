---
title: 'ssys: Exact algebraic recasting of ODE models into S-system or GMA form'
tags:
  - systems biology
  - dynamical systems
  - S-systems
  - power-law formalism
  - symbolic computation
  - Antimony
  - SBML
authors:
  - name: William S. Hlavacek
    orcid: 0000-0003-4383-8711
    affiliation: 1
affiliations:
  - name: Theoretical Biology & Biophysics Group, Theoretical Division, Los Alamos National Laboratory, Los Alamos, NM 87545, USA
    index: 1
date: 2025-12-31
bibliography: paper.bib
header-includes:
  - \hyphenpenalty=10000
  - \exhyphenpenalty=10000
---

# Summary

`ssys` is a Python toolkit for exact algebraic transformation of ordinary differential equation (ODE) models into canonical S-system or Generalized Mass Action (GMA) form. Given a model in Antimony format (or SBML through the Python API), `ssys` produces a mathematically equivalent representation where each equation has one of the following forms:

**S-system form** (difference of two monomials):
$$\frac{dX_i}{dt} = \alpha_i \prod_j X_j^{g_{ij}} - \beta_i \prod_j X_j^{h_{ij}}$$

**GMA form** (sum of monomials):
$$\frac{dX_i}{dt} = \sum_k \gamma_{ik} \prod_j X_j^{f_{ijk}}$$

The transformation introduces auxiliary variables as needed to decompose a broad class of nonlinearities---including rational functions, exponentials, logarithms, and trigonometric functions (excluding events and delays)---into products of power-law terms. The recast is exact: the original and transformed systems have identical dynamics on the invariant constraint manifold defined by auxiliary variable definitions, given consistent initial conditions.

The software provides a command-line interface for batch processing Antimony models, a three-test validation suite (symbolic, numerical, trajectory), and generates Jupyter notebooks with LaTeX renderings and trajectory comparisons. It parses Antimony models through the reference Antimony-to-SBML conversion path and can parse SBML directly through libSBML in the Python API.

# Statement of need

S-systems and GMA systems are canonical ODE forms developed within Biochemical Systems Theory (BST) [@Savageau1976; @SavageauVoit1987]. Savageau & Voit (1987) proved that systems built from sums, products, and compositions of elementary functions can be recast into S-system form; their theorem provides a minimum estimate of the class of recastable systems. These canonical representations offer several advantages: steady-state equations become linear in log-space, enabling algebraic analysis of identifiability and sensitivity; the uniform power-law structure facilitates parameter estimation via linear regression techniques [@Daniels2015]; and exact recast models can serve as ground truth for validating structure-learning algorithms that infer governing equations from data [@Daniels2015b].

Despite the theoretical utility of S-systems and GMA systems, no general-purpose, open-source tool previously existed to perform exact recasting of arbitrary ODE models. `ssys` fills this gap by providing:

- **Exact transformation.** The recast is algebraically equivalent to the original on the constraint manifold, verified by symbolic differentiation and numerical comparison.
- **Broad applicability.** The tool handles rational functions, composite transcendental functions, and time-dependent coefficients through systematic lifting procedures. S-system variables must be strictly positive; variables that can be zero or negative require preprocessing (translation to the positive orthant).
- **Standard formats.** Input/output in Antimony format ensures interoperability with SBML-based workflows and model repositories such as BioModels [@Malik2020].
- **Validation infrastructure.** A three-test suite (symbolic Jacobian verification, pointwise numerical sampling, trajectory comparison) provides certificates of correctness.

The package enables researchers to convert published models into canonical form for analysis, create ground-truth benchmarks for structure-learning algorithms, and leverage the log-linear properties of S-systems for identifiability studies [@Villaverde2019Observability].

# State of the field

Antimony is a human-readable modeling language for systems biology [@Smith2009Antimony; @Heydarabadipour2026Antimony], with transparent interoperability with SBML [@Keating2020SBML]. The S-system and GMA formalisms are classical [@Savageau1976; @SavageauVoit1987; @VoitKemp2025], but general-purpose recasting tools are scarce. Classical BST workflows described by Voit [@Voit2000ComputationalAnalysis] focus on analyzing systems already expressed in S-system or GMA form. PLMaddon [@Vera2007PLMaddon] can generate power-law approximations via Taylor expansion, but such local approximations differ fundamentally from exact recasting with auxiliary variables. More recently, BSTModelKit.jl [@Vadhin2026BSTModelKit] provides Julia tools for constructing and analyzing BST models, though it emphasizes model construction and analysis from declarative specifications rather than recasting arbitrary ODE/SBML/Antimony models. To the best of our knowledge, no existing open-source tool performs exact algebraic recasting from arbitrary SBML/Antimony ODEs to S-system or GMA form with explicit constraint-manifold handling and validation certificates.

# Functionality

- **SBML-first parsing.** Antimony models are parsed through the reference Antimony-to-SBML conversion path, and SBML files can be parsed directly through the Python API; both paths produce SymPy [@Meurer2017SymPy] symbolic expressions for manipulation.
- **Function lifting.** Composite functions (exp, log, sin, cos, etc.) and rational denominators are lifted to auxiliary variables with chain-rule-derived ODEs, preserving exactness.
- **Sum splitting.** Sums of monomials are factored into products via pool-auxiliary construction, yielding the canonical two-term S-system form when feasible without excessive auxiliary-variable expansion.
- **Validation.** Three independent tests verify correctness: symbolic Jacobian chain-rule verification, numerical sampling at random positive points, and trajectory comparison via libRoadRunner simulation.
- **Output modes.** Simplified mode preserves zeros; canonical mode ensures strict two-term form with epsilon slack variables.

# Quality control

The package includes 712 pytest-collected test cases covering parsing, recasting, validation, and CLI functionality. Of these, 11 are integration-marked test cases, including 8 slow manifest tests that entail transformation and validation of 117 models across two output modes.

# BioModels database benchmark

To assess applicability to real-world models, we applied `ssys` to ODE models from the BioModels database [@Malik2020]. The benchmark suite (included in the repository under `biomodels_batch/`) operates in three phases: (1) fetch SBML models from BioModels, filtering for ODE-based models; (2) identify transformation candidates by applying filters that exclude models with unsupported features (discrete events, delay equations) or excessive complexity (species or reaction counts above specified thresholds); and (3) batch transform the candidates. After filtering 1,644 ODE models from BioModels, 978 candidates remained. Of these, the transformation algorithm completed for 848 (86.7%), and numerical validation confirmed mathematical equivalence for 738 transformed models. Among the validated models, 365 achieved meaningful structural canonization: 281 models in general form were converted to GMA, 77 models in GMA form were converted to S-system form, 2 models in general form were converted to S-system form, and 5 models in general form were converted directly to canonical S-system form. The remaining validated models were already in GMA, S-system, or canonical S-system form, or retained general form after exact rewriting. Transformation failures were primarily due to SBML parsing issues (46 models), recast-complexity limits (37), unsupported constructs (37), and processing timeouts (10; >60s per model). Validation was not obtained for 110 transformed models because of unsupported validation features (54), validation-complexity or non-finite sample limitations (44), SBML input loading or decoding failures in the validation wrapper (10), or validator parser failures (2). Benchmark scripts are in `biomodels_batch/` and detailed outputs are written to `biomodels_batch/results/` (see also `RESULTS.md`).

# Availability

`ssys` is available on GitHub at https://github.com/lanl/ssys. Installation requires Python >= 3.10, SymPy, NumPy, libRoadRunner, and the Antimony library. The repository includes comprehensive documentation: theoretical background and recasting rules (`RECASTING.md`), test model descriptions (`TEST_MODELS.md`), and usage instructions (`README.md`).

Installation: `uv pip install -e ".[dev]"`. A working example that recasts 29 models from the test suite with validation: `python recast_models.py test_models1`.

# Acknowledgments

This work was inspired by a phone call with Michael A. Savageau in early 2025. I will always be grateful to Mike for his kind and patient mentorship of me as a graduate student at the University of Michigan, his exemplary and inspiring scholarship, and his enduring support.

This work was supported by the National Institutes of Health (NIH) National Institute of General Medical Sciences (NIGMS) under grant R01GM111510 and by the U.S. Department of Energy through the Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy (Contract No. 89233218CNA000001).

# References
