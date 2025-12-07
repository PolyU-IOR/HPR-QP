# HPRQP.jl Documentation

**A Julia implementation of the Halpern Peaceman-Rachford (HPR) method for solving quadratic programming (QP) problems on the GPU.**

## Overview

HPRQP.jl is a high-performance quadratic programming solver that leverages GPU acceleration to solve large-scale QP problems efficiently. It implements the Halpern Peaceman-Rachford splitting method with adaptive restart strategy and penalty parameter selection.

## Features

- ✅ **GPU Acceleration**: Native CUDA support for solving large-scale problems
- ✅ **CPU Support**: Support CPU mode when GPU is not available
- ✅ **Multiple Inputs**: 
  - Direct API with matrix inputs
  - MPS file format support
  - JuMP integration via MOI wrapper
- ✅ **Flexible Q Operators**: Support for sparse matrices, LASSO, QAP, and custom operators
- ✅ **Flexible Scaling**: Ruiz, Pock-Chambolle, and scalar scaling methods
- ✅ **Adaptive Algorithms**: Automatic restart strategy and penalty parameter selection

## Problem Formulation

HPRQP solves quadratic programming problems of the form:

```math
\begin{array}{ll}
\underset{x \in \mathbb{R}^n}{\min} \quad & \frac{1}{2} \langle x, Qx \rangle + \langle c, x \rangle \\
\text{s.t.} \quad & L \leq A x \leq U, \\
& l \leq x \leq u .
\end{array}
```

where:
- ``x \in \mathbb{R}^n`` is the decision variable
- ``Q \in \mathbb{R}^{n \times n}`` is a symmetric positive semidefinite matrix (or operator)
- ``c \in \mathbb{R}^n`` is the linear objective coefficient vector
- ``A \in \mathbb{R}^{m \times n}`` is the constraint matrix
- ``L, U \in \mathbb{R}^m`` are lower and upper bounds on constraints
- ``l, u \in \mathbb{R}^n`` are lower and upper bounds on variables

## Quick Start

### Installation

From GitHub (recommended for applications):
```julia
using Pkg
Pkg.add(url="https://github.com/PolyU-IOR/HPR-QP")
```

Locally (recommended for development):
```bash
git clone https://github.com/PolyU-IOR/HPR-QP.git
cd HPR-QP
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Simple Example

```julia
using HPRQP
using SparseArrays

# Define QP: min 0.5*x'*Q*x + c'*x s.t. Ax ≤ b, x ≥ 0
Q = sparse([2.0 0.5; 0.5 2.0])
A = sparse([-1.0 -2.0; -3.0 -1.0])
c = [-3.0, -5.0]
AL = [-10.0, -12.0]
AU = [Inf, Inf]
l = [0.0, 0.0]
u = [Inf, Inf]

# Build and solve
model = build_from_QAbc(Q, A, c, AL, AU, l, u)

params = HPRQP_parameters()
params.stoptol = 1e-9  # Set stopping tolerance

result = optimize(model, params)

println("Optimal value: ", result.primal_obj)
println("Solution: x = ", result.x)
```

### With JuMP

```julia
using JuMP, HPRQP

model = Model(HPRQP.Optimizer)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, x1^2 + x1*x2 + x2^2 - 3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

set_attribute(model, "stoptol", 1e-9)  # Set stopping tolerance

optimize!(model)
println("Objective: ", objective_value(model))
println("x1 = ", value(x1), ", x2 = ", value(x2))
```

## Documentation Contents

```@contents
Pages = [
    "getting_started.md",
    "guide/mps_files.md",
    "guide/direct_api.md",
    "guide/jump_integration.md",
    "guide/q_operators_overview.md",
    "guide/sparse_matrix_qp.md",
    "guide/lasso_problems.md",
    "guide/qap_problems.md",
    "api.md",
    "examples.md",
]
Depth = 2
```

## Citation

If you use HPRQP in your research, please cite:

```bibtex
@article{chen2025hpr,
  title={HPR-QP: An implementation of an HPR method for solving quadratic programming.},
  author={Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  journal={Mathematical Programming Computation},
  year={2025},
  publisher={Springer}
}
```

## License

HPRQP.jl is licensed under the MIT License. See [LICENSE](https://github.com/PolyU-IOR/HPR-QP/blob/main/LICENSE) for details.
