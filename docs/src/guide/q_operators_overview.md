# Q Operators Overview

HPRQP supports different representations of the quadratic term Q in the objective function. This flexibility allows you to leverage problem structure for improved performance and memory efficiency.

## What is a Q Operator?

In the quadratic programming problem:

```math
\min \quad \frac{1}{2} \langle x, Qx \rangle + \langle c, x \rangle
```

the quadratic term ``\frac{1}{2} \langle x, Qx \rangle`` is represented by a Q operator. Instead of always storing Q as an explicit matrix, HPRQP allows you to define how to compute ``Qx`` for any vector ``x``.

## Available Q Operators

HPRQP provides three built-in Q operators plus support for custom operators:

| Operator | Best For | Memory | Speed |
|----------|----------|--------|-------|
| [**Sparse Matrix**](sparse_matrix_qp.md) | General QP problems | ``O(nnz)`` | Fast |
| [**LASSO**](lasso_problems.md) | L1-regularized least squares | ``O(mn)`` | Very Fast |
| [**QAP**](qap_problems.md) | Quadratic assignment problems | ``O(n^2)`` | Fast |
| **Custom** | Specialized structures | User-defined | User-defined |

## Sparse Matrix Operator

**Use when:** You have a general sparse quadratic objective.

```julia
using HPRQP
using SparseArrays

# Define Q as a sparse matrix
Q = sparse([2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0])

# Build model (automatically uses SparseMatrixQOperator)
model = build_from_QAbc(Q, A, c, AL, AU, l, u)
```

**Characteristics:**
- General-purpose operator for any sparse positive semidefinite matrix
- Memory: ``O(\text{nnz}(Q))`` where nnz is number of non-zeros
- Computation: Sparse matrix-vector product
- Automatically selected when you pass a sparse matrix

[→ Full Guide: Sparse Matrix QP](sparse_matrix_qp.md)

---

## LASSO Operator

**Use when:** Solving L1-regularized least squares problems.

```math
\min \quad \frac{1}{2} \|Ax - b\|^2 + \lambda \|x\|_1
```

```julia
using HPRQP

# Problem data
A = randn(100, 50)  # Design matrix
b = randn(100)      # Observations
λ = 0.1             # Regularization parameter

# Create LASSO operator (Q = A'A implicitly)
Q_lasso = LASSOOperatorCPU(A, b, λ)

# Build model
model = build_from_QAbc(Q_lasso, A_constr, c, AL, AU, l, u)
```

**Characteristics:**
- Specialized for regression with L1 penalty
- Memory: Stores A and b, not A'A
- Computation: Two matrix-vector products instead of one
- More efficient than forming A'A explicitly

[→ Full Guide: LASSO Problems](lasso_problems.md)

---

## QAP Operator

**Use when:** Solving quadratic assignment problems or similar structured QPs.

```julia
using HPRQP

# QAP data: Flow and distance matrices
F = rand(n, n)  # Flow between facilities
D = rand(n, n)  # Distance between locations

# Create QAP operator
Q_qap = QAPOperatorCPU(F, D, n)

# Build model
model = build_from_QAbc(Q_qap, A, c, AL, AU, l, u)
```

**Characteristics:**
- Specialized for quadratic assignment structure
- Memory: Stores F and D matrices
- Computation: Exploits Kronecker product structure
- Efficient for large-scale QAP relaxations

[→ Full Guide: QAP Problems](qap_problems.md)

---

## Custom Q Operators

You can define your own Q operators for specialized problem structures.

**When to use:**
- Your problem has special structure not covered by built-in operators
- You want matrix-free implementations
- You need to integrate with external libraries

**Example:** Diagonal plus low-rank structure

```julia
# Define your operator type
struct DiagonalPlusLowRankCPU <: AbstractQOperatorCPU
    diag::Vector{Float64}
    U::Matrix{Float64}  # n × k matrix
end

# Implement the interface (see custom operator guide)
function to_gpu(Q::DiagonalPlusLowRankCPU)
    # Transfer to GPU...
end

function Qmap!(x, Qx, Q::DiagonalPlusLowRankGPU)
    # Compute Qx = (Diag + UU')x
    # Qx = diag .* x + U * (U' * x)
end
```

See `src/Q_operators/Q_operator_interface.jl` for the complete interface specification.

---

## Choosing the Right Operator

### Decision Tree

```
Do you have a least squares problem with L1 regularization?
├─ YES → Use LASSOOperator
└─ NO
    ├─ Is your problem a QAP or similar Kronecker structure?
    │   ├─ YES → Use QAPOperator
    │   └─ NO
    │       ├─ Do you have a general sparse matrix?
    │       │   ├─ YES → Use SparseMatrixOperator (default)
    │       │   └─ NO
    │       │       └─ Define CustomOperator for your structure
```

### Performance Comparison

For a problem with n=10,000 variables:

| Operator | Memory (GB) | Time/Iteration (ms) | Notes |
|----------|-------------|---------------------|-------|
| Sparse (1% density) | 0.8 | 5 | General purpose |
| LASSO (m=5000) | 0.4 | 8 | Avoids forming A'A |
| QAP | 1.6 | 12 | Exploits structure |
| Dense | 800 | 1000 | Impractical |

*Timings are approximate and depend on hardware*

## Operator Interface

All Q operators must implement:

```julia
# CPU version
struct MyQOperatorCPU <: AbstractQOperatorCPU
    # Your data fields
end

# GPU version  
mutable struct MyQOperatorGPU <: AbstractQOperator
    # GPU data fields
    temp::CuVector{Float64}  # Temporary storage
end

# Required methods:
to_gpu(Q::MyQOperatorCPU) -> MyQOperatorGPU
Qmap!(x::CuVector, Qx::CuVector, Q::MyQOperatorGPU) -> nothing
get_temp_size(Q::MyQOperatorCPU) -> Int
get_operator_name(::Type{MyQOperatorGPU}) -> String
get_problem_size(Q::MyQOperatorCPU) -> Int
```

## Examples

### Comparing Operators

```julia
using HPRQP
using SparseArrays
using BenchmarkTools

# Same problem, different representations
A = randn(100, 50)
b = randn(50)

# Method 1: Form Q = A'A explicitly
Q_explicit = sparse(A' * A)
model1 = build_from_QAbc(Q_explicit, ...)

# Method 2: Use LASSO operator  
Q_lasso = LASSOOperatorCPU(A, b, 0.0)
model2 = build_from_QAbc(Q_lasso, ...)

# Compare
@btime optimize(model1, params)  # Sparse matrix
@btime optimize(model2, params)  # LASSO operator
```

## See Also

- [Sparse Matrix QP](sparse_matrix_qp.md) - Using sparse matrices
- [LASSO Problems](lasso_problems.md) - L1-regularized least squares
- [QAP Problems](qap_problems.md) - Quadratic assignment problems
- [Direct API](direct_api.md) - Building models with Q operators
