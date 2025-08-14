]
# CNDO/2 Analytic Gradient Implementation

## Overview
This project implements an **analytic gradient module** for a CNDO/2 Self-Consistent Field (SCF) quantum chemistry program.  
The gradient computation enables **geometry optimization** by providing nuclear forces directly from the SCF solution, avoiding the cost of finite-difference derivatives.

The code extends a pre-existing CNDO/2 SCF engine with:
- Assembly of the **x** and **y** matrices from SCF outputs.
- Derivatives of contracted Gaussian overlap integrals (s and p functions) with respect to nuclear coordinates.
- Derivatives of γ<sub>AB</sub> integrals using translational invariance to minimize computation.
- Derivative of the nuclear repulsion term V<sub>nuc</sub>.
- Final gradient vector assembly for all atoms.

This implementation achieves **O(N²)** scaling for large molecules by exploiting symmetry and redundancy.

---

## Features
- **Analytic gradient evaluation** for unrestricted CNDO/2 SCF energies.
- Modular C++ code for:
  - Overlap matrix derivative computation.
  - γ<sub>AB</sub> derivative evaluation.
  - Gradient assembly per Equation:
    ```
    E_RA = Σ_{μ≠ν} x_{μν} s^RA_{μν} + Σ_{B≠A} y_{AB} γ^RA_{AB} + V^RA_nuc
    ```
- **Performance optimizations**:
  - Use of translational invariance to reduce derivative evaluations by 50%.
  - Symmetry-based reuse of computed integrals.
- **Validation tools** for comparing analytic gradients against finite-difference calculations.

---

## Mathematical Background
The gradient of the CNDO/2 SCF energy with respect to nuclear coordinates is given by:

