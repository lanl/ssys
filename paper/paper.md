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
date: 2025-12-28
bibliography: paper.bib
header-includes:
  - \hyphenpenalty=10000
  - \exhyphenpenalty=10000
---

# Summary

`ssys` is a Python toolkit for exact algebraic transformation of ordinary differential equation (ODE) models into canonical S-system or Generalized Mass Action (GMA) form. Given a model in Antimony or SBML format, `ssys` produces a mathematically equivalent representation where each equation has one of the following forms:

**S-system form** (difference of two monomials):
$$\frac{dX_i}{dt} = \alpha_i \prod_j X_j^{g_{ij}} - \beta_i \prod_j X_j^{h_{ij}}$$

**GMA form** (sum of monomials):
$$\frac{dX_i}{dt} = \sum_k \gamma_{ik} \prod_j X_j^{f_{ijk}}$$

The transformation introduces auxiliary variables as needed to decompose a broad class of nonlinearities---including rational functions, exponentials, logarithms, and trigonometric functions (excluding events and delays)---into products of power-law terms. The recast is exact: the original and transformed systems have identical dynamics on the invariant constraint manifold defined by auxiliary variable definitions, given consistent initial conditions.

The software provides a command-line interface for batch processing, a three-test validation suite (symbolic, numerical, trajectory), and generates Jupyter notebooks with LaTeX renderings and trajectory comparisons. It parses models via the reference Antimony library and libRoadRunner, ensuring compatibility with standard SBML semantics.

# Statement of need

S-systems and GMA systems are canonical ODE forms developed within Biochemical Systems Theory [@Savageau1976; @SavageauVoit1987]. Savageau & Voit (1987) proved that systems built from sums, products, and compositions of elementary functions can be recast into S-system form; their theorem provides a minimum estimate of the class of recastable systems. These canonical representations offer several advantages: steady-state equations become linear in log-space, enabling algebraic analysis of identifiability and sensitivity; the uniform power-law structure facilitates parameter estimation via linear regression techniques [@Daniels2015]; and exact recast models can serve as ground truth for validating structure-learning algorithms that infer governing equations from data [@Daniels2015b].

Despite the theoretical utility of S-systems and GMA systems, no general-purpose, open-source tool previously existed to perform exact recasting of arbitrary ODE models. `ssys` fills this gap by providing:

- **Exact transformation.** The recast is algebraically equivalent to the original on the constraint manifold, verified by symbolic differentiation and numerical comparison.
- **Broad applicability.** The tool handles rational functions, composite transcendental functions, and time-dependent coefficients through systematic lifting procedures. S-system variables must be strictly positive; variables that can be zero or negative require preprocessing (translation to the positive orthant).
- **Standard formats.** Input/output in Antimony format ensures interoperability with SBML-based workflows and model repositories such as BioModels [@Malik2020].
- **Validation infrastructure.** A three-test suite (symbolic Jacobian verification, pointwise numerical sampling, trajectory comparison) provides certificates of correctness.

The package enables researchers to convert published models into canonical form for analysis, create ground-truth benchmarks for structure-learning algorithms, and leverage the log-linear properties of S-systems for identifiability studies [@Villaverde2019Observability].

# State of the field

Antimony is a widely used human-readable modeling language for systems biology [@Smith2009Antimony; @Smith2024Antimony], with transparent interoperability to SBML [@Keating2020SBML]. The S-system and GMA formalisms are classical [@Savageau1976; @Voit2013], but general-purpose recasting tools are scarce. Classical BST tools such as PLAS [@Voit1991PLAS] focus on analyzing systems already expressed in S-system or GMA form. PLMaddon [@Villaverde2007] can generate power-law approximations via Taylor expansion, but such local approximations differ fundamentally from exact recasting with auxiliary variables. More recently, BSTModelKit.jl [@Sund2025BSTModelKit] provides Julia tools for constructing and analyzing BST models, though it emphasizes model construction from network descriptions rather than recasting arbitrary ODEs. To the best of our knowledge, no existing open-source tool performs exact algebraic recasting from arbitrary SBML/Antimony ODEs to S-system or GMA form with explicit constraint-manifold handling and validation certificates. `ssys` aims to fill that gap.

# Functionality

- **SBML-first parsing.** Models are parsed via the Antimony library and libRoadRunner, then converted to SymPy [@Meurer2017SymPy] symbolic expressions for manipulation.
- **Function lifting.** Composite functions (exp, log, sin, cos, etc.) and rational denominators are lifted to auxiliary variables with chain-rule-derived ODEs, preserving exactness.
- **Sum splitting.** Sums of monomials are factored into products via pool-auxiliary construction, yielding the canonical two-term S-system form when feasible without excessive auxiliary-variable expansion.
- **Validation.** Three independent tests verify correctness: symbolic Jacobian chain-rule verification, numerical sampling at random positive points, and trajectory comparison via libRoadRunner simulation.
- **Output modes.** Simplified mode preserves zeros; canonical mode ensures strict two-term form with epsilon slack variables.

# Quality control

The package includes more than 200 unit tests covering parsing, recasting, validation, and CLI functionality. Integration tests entail transformation and validation of more than 100 models.

# BioModels database benchmark

To assess applicability to real-world models, we applied `ssys` to ODE models from the BioModels database [@Malik2020]. The benchmark suite (included in the repository under `biomodels_batch/`) operates in three phases: (1) fetch SBML models from BioModels, filtering for ODE-based models; (2) identify transformation candidates by applying filters that exclude models with unsupported features (discrete events, delay equations) or excessive complexity (species or reaction counts above specified thresholds); and (3) batch transform the candidates. After filtering 1,644 ODE models from BioModels, 978 candidates remained. Of these, the transformation algorithm completed without error for 895 (91%), producing output files. Numerical validation then confirmed mathematical equivalence between original and transformed models for 175 cases. The remainder could not be validated (the transformation completed but the output was not confirmed equivalent to the input). Most transformation failures were due to processing timeouts (greater than 60 s per model). Benchmark scripts are in `biomodels_batch/` and results are in `biomodels_batch/results/` (see also `RESULTS.md`).

# Availability

`ssys` is available on GitHub at https://github.com/lanl/ssys. Installation requires Python >= 3.10, SymPy, NumPy, libRoadRunner, and the Antimony library. The repository includes comprehensive documentation: theoretical background and recasting rules (`RECASTING.md`), test model descriptions (`TEST_MODELS.md`), and usage instructions (`README.md`).

Installation: `pip install -e ".[dev]"`. A working example that recasts 29 models from the test suite with validation: `python recast_models.py test_models1 --validate`.

# Acknowledgments

This work was inspired by a phone call with Michael A. Savageau in early 2025. I will always be grateful to Mike for his kind and patient mentorship of me as a graduate student at the University of Michigan, his exemplary and inspiring scholarship, and his enduring support.

This work was supported by the National Institutes of Health (NIH) National Institute of General Medical Sciences (NIGMS) under grant R01GM111510 and by the U.S. Department of Energy through the Los Alamos National Laboratory (LANL). LANL is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy (Contract No. 89233218CNA000001).

# References
