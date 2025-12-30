# Test Models from Recasting Literature

This folder contains test models extracted from published papers on exact recasting
of nonlinear differential equations into S-system canonical form.

## Sources

### Savageau & Voit (1987)
**"Recasting Nonlinear Differential Equations as S-Systems: A Canonical Nonlinear Form"**
*Mathematical Biosciences* 87:83-115

- Time horizon: t ∈ [0, 20] for all benchmark problems
- 28 examples covering single equations, small systems, moderate systems, orbit equations, and higher-order equations

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| S1987_4C_exp_composition.ant | Exponential composition | 1→3 | §4C |
| S1987_4D_sum_reduction.ant | Sum reduction example | 1→2 | §4D |
| S1987_5_CSTR.ant | Continuous stirred tank reactor | 2→8 | §5 |
| S1987_A1_exponential.ant | Exponential decay | 1 | Appendix A |
| S1987_A2_riccati.ant | Riccati equation | 1 | Appendix A |
| S1987_A3_cos_growth.ant | Cosine growth | 1→3 | Appendix A |
| S1987_A4_logistic.ant | Logistic growth | 1 | Appendix A |
| S1987_A5_spiral.ant | Spiral curve | 1→3 | Appendix A |
| S1987_B1_lotka_volterra.ant | Lotka-Volterra | 2 | Appendix A |
| S1987_B2_linear_chain.ant | Linear chemical chain | 3→4 | Appendix A |
| S1987_B3_nonlinear_chain.ant | Nonlinear chemical chain | 3→4 | Appendix A |
| S1987_B4_torus.ant | Torus integral surface | 3→8 | Appendix A |
| S1987_B5_rigid_body.ant | Euler rigid body equations | 3→4 | Appendix A |
| S1987_C1_decay_chain.ant | Radioactive decay chain | 10 | Appendix A |
| S1987_C2_decay_chain2.ant | Radioactive decay chain 2 | 10 | Appendix A |
| S1987_C3_parabolic_pde.ant | Parabolic PDE derived | 10→18 | Appendix A |
| S1987_D1_orbit_e0.1.ant | Orbit (e=0.1) | 4→8 | Appendix A |
| S1987_D2_orbit_e0.3.ant | Orbit (e=0.3) | 4→8 | Appendix A |
| S1987_D3_orbit_e0.5.ant | Orbit (e=0.5) | 4→8 | Appendix A |
| S1987_D4_orbit_e0.7.ant | Orbit (e=0.7) | 4→8 | Appendix A |
| S1987_D5_orbit_e0.9.ant | Orbit (e=0.9) | 4→8 | Appendix A |
| S1987_E1_bessel.ant | Bessel equation derived | 2→5 | Appendix A |
| S1987_E2_van_der_pol.ant | Van der Pol oscillator | 2→5 | Appendix A |
| S1987_E3_duffing.ant | Duffing equation | 2→6 | Appendix A |
| S1987_E4_falling_body.ant | Falling body | 2 | Appendix A |
| S1987_E5_implicit.ant | Implicit DE example | 2→4 | Appendix A |
| S1987_B_binary.ant | Half→Binary transformation | Appendix B |
| S1987_C_implicit_de.ant | Implicit DE transformation | Appendix C |

### Marin-Sanguino et al. (2007)
**"Optimization of biotechnological systems through geometric programming"**
*Theor. Biol. Med. Model.* 4:38

- 3 examples demonstrating GMA recasting for optimization

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| MS2007_MM_to_GMA.ant | Michaelis-Menten to GMA | 1→2 | Eq. 14-16 |
| MS2007_fermentation_yeast.ant | Anaerobic fermentation (S. cerevisiae) | 5+9 | Eq. 39-40 |
| MS2007_tryptophan_operon.ant | Tryptophan operon (E. coli) | 3→8 | Eq. 41-44 |

### Savageau (1988)
**"Introduction to S-Systems and the Underlying Power-Law Formalism"**
*Math. Comput. Modelling* 11:546-551

- 2 examples showing different recasting strategies for the same system

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| S1988_monod_chemostat_v1.ant | Monod chemostat (method 1) | 2→4 | Eq. 19-20 |
| S1988_monod_chemostat_v2.ant | Monod chemostat (method 2) | 2→5 | Eq. 21 |

### Voit (1988)
**"S-Systems as a Canonical Nonlinear Form"**
*Math. Comput. Modelling* 11:140-145

- 3 examples of exact recasting

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| V1988a_weibull_growth.ant | Weibull growth law | 1→2 | Eq. 2-4 |
| V1988a_sin_exp_system.ant | Sin/exp nonlinearities | 2→6 | Eq. 6-10 |
| V1988a_endemic_infection.ant | Endemic infection (Kemper/Cooke) | 3→7 | Eq. 14-15 |

### Voit (1988b)
**"New Nonlinear Methodologies for Modeling Molecular and Cellular Systems"**
*Proc. IFAC Modelling and Control in Biomedical Systems*, Venice, Italy

- 2 examples of exact recasting

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| V1988b_exponential_ode.ant | Exponential ODE | 1→2 | p. 223 |
| V1988b_gma_to_ssystem.ant | GMA to S-system (product rule) | 2→4 | p. 223 |

### Voit (1990)
**"S-System Modelling of Endemic Infections"**
*Comput. Math. Applic.* 20(4-6):161-173

- 3 examples of exact recasting (endemic infection models)

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| V1990_kemper_endemic.ant | Kemper endemic model (γ1=γ2) | 3→5 | Eq. 9 |
| V1990_cooke_endemic.ant | Cooke endemic model (γ1≠γ2) | 3→7 | Eq. 15 |
| V1990_endemic_exp_infection.ant | Exponential infection term | 2→3 | Eq. 39-41 |

### Voit (1992)
**"Symmetries of S-Systems"**
*Math. Biosci.* 109:19-31

- 2 examples of exact recasting (demonstrating symmetry analysis)

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| V1992_blasius_equation.ant | Blasius equation (boundary layer) | 3 | pp. 28-30 |
| V1992_log_ode.ant | ODE with logarithmic term | 2→4 | p. 28 |

### Voit (1993)
**"S-System Modelling of Complex Systems with Chaotic Input"**
*Environmetrics* 4(2):153-186

- 3 examples of exact recasting (chaotic systems)

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| V1993_rossler_band.ant | Rössler strange attractor (chaos) | 3→4 | Eq. 15 |
| V1993_forced_oscillator.ant | Forced Duffing-type oscillator | 2→7 | Appendix A |
| V1993_blue_sky.ant | Blue sky catastrophe | 2→8 | Appendix A |

### Voit (2005)
**"Smooth bistable S-systems"**
*IEE Proc. Syst. Biol.* 152(4):207-213

- 1 example of bistable system (sigmoidal switch with recasting)

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| V2005_bistable_gene.ant | Bistable gene expression with hysteresis | 4→6 | Eq. 4 |

### Daniels & Nemenman (2015)
**"Automated adaptive inference of phenomenological dynamical models"**
*Nature Communications* 6:8133

- 1 example of exact recasting (gravitational two-body problem)

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| DN2015_planetary_motion.ant | Planetary motion (radial 2-body) | 2 | Eq. 5-6 |

### Anghel, Milano & Papachristodoulou (2013)
**"Algorithmic Construction of Lyapunov Functions for Power System Stability Analysis"**
*IEEE Trans. Circuits Syst. I* 60(9):2533-2546

- 2 examples of trigonometric → polynomial DAE recasting (citing Savageau & Voit 1987)
- Power system swing equations with sin/cos nonlinearities

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| A2013_power_system_3machine.ant | 3-machine lossless power system | 4→6 | §V-A |
| A2013_power_system_2machine.ant | 2-machine with transfer conductances | 4→6 | §V-B |

### Zhang et al. (2022)
**"Domain of Attraction's Estimation for Grid Connected Converters with Phase-Locked Loop"**
*IEEE Trans. Power Systems* 37(2):1351-1362

- 1 example with **explicit citation of Savageau & Voit (1987)** [ref 30]
- Trig → polynomial recasting for SOS programming

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| Z2022_pll_converter.ant | Phase-locked loop converter | 2→4 | Eq. 7, 17-19 |

### Kramer, Peherstorfer & Willcox (2024)
**"Learning Nonlinear Reduced Models from Data with Operator Inference"**
*Annu. Rev. Fluid Mech.* 56:521-548

- 1 example with **explicit citation of Savageau & Voit (1987)** in Section 2.1.3
- Demonstrates "lifting" to quadratic form for machine learning

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| KPW2024_exponential_lifting.ant | Exponential ODE lifted to quadratic | 1→2 | Sidebar p. 526 |

### Papachristodoulou & Prajna (2005)
**"Analysis of Non-polynomial Systems using the Sum of Squares Decomposition"**
*In: Positive Polynomials in Control*, Springer

- 2 examples with **explicit citation of Savageau & Voit (1987)** as reference [14]
- Recasting for SOS/SDP stability analysis

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| PP2005_exponential_exact.ant | Exact exponential recasting (same dim) | 1→1 | p. 2 |
| PP2005_CSTR_arrhenius.ant | Diabatic CSTR with Arrhenius kinetics | 2→3 | §4.4 |

### Daniels & Nemenman (2015b)
**"Efficient Inference of Parsimonious Phenomenological Models of Cellular Dynamics Using S-Systems and Alternating Regression"**
*PLOS ONE* 10(3):e0119821

- 2 examples with **explicit citation of Savageau & Voit (1987)** as reference [17]
- Recasting for S-system parameter inference

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| DN2015b_michaelis_menten.ant | Michaelis-Menten kinetics | 1→2 | Eq. 6 |
| DN2015b_sinx_recasting.ant | sin(x) dynamics | 1→3 | Eq. 7 |

### Anguelov et al. (2018)
**"On the chemical meaning of some growth models possessing Gompertzian-type property"**
*Math. Methods Appl. Sci.* 41:8365-8376

- 1 example with **explicit citation of Savageau & Voit (1987)** as reference [7]
- Recasting of Gompertz growth model to reveal chemical interpretation

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| A2017_gompertz_recasting.ant | Gompertz growth model | 1→2 | Eq. SG |

### Savageau (1993)
**"Finding Multiple Roots of Nonlinear Algebraic Equations Using S-System Methodology"**
*Appl. Math. Comput.* 55:187-199

- 2 examples with **explicit citation of Savageau & Voit (1987)** as reference [12]
- Demonstrates "reductive recasting" for ODEs with multiple terms of same/both signs

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| S1993_sum_radical.ant | Radical term with sum | 1→2 | Eq. 4-7 |
| S1993_mixed_terms.ant | Multiple + and − terms | 2→5 | Eq. 15-19 |

### Irvine (1988)
**"Efficient Solution of Nonlinear Models Expressed in S-System Canonical Form"**
*Math. Comput. Modelling* 11:123-128

- 1 example referencing Savageau & Voit (1986/1987) recasting algorithm
- Demonstrates computational benefit: 9→19 variables, but 10x faster solution

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| I1988_metabolic_pathway.ant | Unbranched pathway with MM kinetics | 9→19 | Eq. 8-9 |

### Pozo et al. (2011)
**"Steady-state global optimization of metabolic non-linear dynamic models through recasting into power-law canonical models"**
*BMC Systems Biology* 5:137

- 2 examples with **explicit citation of Savageau & Voit (1987)** as reference [36]
- Recasting SC (Saturable and Cooperative) kinetics to GMA for global optimization

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| P2011_linear_pathway_MM.ant | Linear pathway + competitive inhibition | 2→6 | Eq. 20-23 |
| P2011_branched_SC.ant | Branched network with SC kinetics | 4 | Eq. 2, Fig. 1 |

### Rust & Voit (1990)
**"Statistical Densities, Cumulatives, Quantiles, and Power Obtained by S-System Differential Equations"**
*J. Amer. Statist. Assoc.* 85(410):572-578

Also: Voit & Rust (1990) *Biom. J.* 32(6):681-695

- 3 examples of probability distributions recast as S-systems
- Cites Savageau (1982) which cites Savageau & Voit (1987)
- Demonstrates handling of SUM terms in distribution expressions

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| RV1990_central_t_density.ant | Central t-distribution | 1→4 | Eq. 4 |
| RV1990_central_chisquared.ant | Central χ² distribution | 3 | Eq. 6 |
| RV1990_central_F.ant | Central F distribution | 1→4 | Eq. 10 |

### Hernández-Bermejo, Fairén & Brenig (1998)
**"Algebraic recasting of nonlinear systems of ODEs into universal formats"**
*J. Phys. A: Math. Gen.* 31:2415-2430

- 4 examples of recasting to quasipolynomial/Lotka-Volterra/unimonomial form
- Develops systematic algebraic framework using matrix operations
- References S&V framework indirectly via Brenig/Voit literature

| File | Description | Variables | Source |
|------|-------------|-----------|--------|
| HBF1998_morse_oscillator.ant | Morse oscillator (molecular physics) | 2→5 | §3.2, Eq. 17-25 |
| HBF1998_semiconductor_exciton.ant | Electron-hole/exciton oscillations | 2→6 | §3.3, Eq. 29-36 |
| HBF1998_brusselator.ant | Brusselator (Belousov-Zhabotinskii) | 2→4 | §4.2, Eq. 45-54 |
| HBF1998_three_wave.ant | Three-wave interaction (plasma physics) | 3→4 | §4.3, Eq. 55-61 |

## File Format

Each `.ant` file contains:
- Header with source citation and equation numbers
- Time horizon specification
- Initial conditions
- Original ODE system in Antimony format
- Recast S-system form in comments

## Usage

```python
import ssys
sym = ssys.parse_antimony("tests2/S1987_A4_logistic.ant")
result = ssys.recast_to_ssystem(sym, mode="simplified")
