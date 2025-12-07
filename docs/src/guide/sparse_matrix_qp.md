# Sparse Matrix QP

The sparse matrix Q operator is the default and most general way to represent quadratic objectives in HPRQP. It's suitable for any sparse positive semidefinite matrix.

## When to Use

Use the sparse matrix operator when:
- You have a general quadratic programming problem
- Your Q matrix is sparse (not dense)
- You don't have special structure (LASSO, QAP, etc.)
- You want the simplest, most straightforward approach

## Basic Usage

```julia
using HPRQP
using SparseArrays

# Define a sparse positive semidefinite matrix
Q = sparse([
    2.0  0.5  0.0  0.0;
    0.5  2.0  0.5  0.0;
    0.0  0.5  2.0  0.5;
    0.0  0.0  0.5  2.0
])

# Linear term
c = [1.0, 2.0, 3.0, 4.0]

# Constraints (simple box constraints for this example)
A = sparse(zeros(0, 4))
AL = Float64[]
AU = Float64[]
l = zeros(4)
u = fill(10.0, 4)

# Build and solve
model = build_from_QAbc(Q, A, c, AL, AU, l, u)

params = HPRQP_parameters()
result = optimize(model, params)
```

## Creating Sparse Q Matrices

### From Dense Matrices

```julia
using SparseArrays

# Convert dense to sparse
Q_dense = [2.0 0.5; 0.5 2.0]
Q_sparse = sparse(Q_dense)

# HPRQP will also convert automatically
model = build_from_QAbc(Q_dense, A, c, AL, AU, l, u)  # Works but issues warning
```

### Building Sparse Matrices Directly

```julia
using SparseArrays

n = 100

# Tridiagonal matrix
main_diag = fill(2.0, n)
off_diag = fill(0.5, n-1)
Q = spdiagm(0 => main_diag, 1 => off_diag, -1 => off_diag)

# Random sparse matrix (make it positive semidefinite)
H = sprandn(n, n, 0.1)  # 10% density
Q = H' * H  # Q = H'H is positive semidefinite
```

### Block Diagonal Structure

```julia
using SparseArrays
using LinearAlgebra

# Create block diagonal Q
n_blocks = 10
block_size = 5

blocks = [spdiagm(0 => rand(block_size)) for _ in 1:n_blocks]
Q = blockdiag(blocks...)
```

## Ensuring Positive Semidefiniteness

Q must be positive semidefinite for convex QP. Here are ways to ensure this:

### Method 1: Form Q = H'H

```julia
H = sprandn(n, n, 0.1)
Q = H' * H  # Always positive semidefinite
```

### Method 2: Add Regularization

```julia
Q_indefinite = ... # Some matrix
λ = 1e-6
Q = Q_indefinite + λ * I  # Make positive definite
```

### Method 3: Symmetric Part of A'A

```julia
A = randn(m, n)
Q = sparse((A' * A + (A' * A)') / 2)  # Ensure symmetry
```

## Performance Considerations

### Sparsity Pattern

The performance depends heavily on sparsity:

```julia
using SparseArrays

n = 1000

# Diagonal (very sparse)
Q_diag = spdiagm(0 => rand(n))
println("Diagonal nnz: ", nnz(Q_diag))  # 1000

# Tridiagonal  
Q_tri = spdiagm(0 => rand(n), 1 => rand(n-1), -1 => rand(n-1))
println("Tridiagonal nnz: ", nnz(Q_tri))  # ~3000

# Banded
bandwidth = 10
Q_band = spdiagm([k => rand(n - abs(k)) for k in -bandwidth:bandwidth]...)
println("Banded nnz: ", nnz(Q_band))  # ~20,000

# Random sparse
density = 0.01
H = sprandn(n, n, density)
Q_random = H' * H
println("Random nnz: ", nnz(Q_random))  # ~100,000
```

### Dense vs Sparse Threshold

As a rule of thumb:
- **Sparse** (< 10% density): Use sparse matrix operator
- **Dense** (> 50% density): Consider if problem can be reformulated
- **Medium** (10-50% density): Test both, sparse usually better

```julia
using SparseArrays

n = 1000
density = 0.05  # 5% density

H = sprandn(n, n, density)
Q = H' * H

println("Matrix size: ", n, "×", n)
println("Density: ", nnz(Q) / (n^2) * 100, "%")
println("Memory (sparse): ", Base.summarysize(Q) / 1e6, " MB")
println("Memory (dense): ", 8 * n^2 / 1e6, " MB")
```

## Common Patterns

### Portfolio Optimization

```julia
using SparseArrays
using LinearAlgebra

# Covariance matrix (often sparse for large portfolios)
n_assets = 100

# Block diagonal covariance (assets grouped by sector)
sector_sizes = [20, 30, 25, 25]
sectors = []
for s in sector_sizes
    Σ_sector = rand(s, s)
    Σ_sector = (Σ_sector + Σ_sector') / 2  # Symmetric
    Σ_sector += 2I  # Positive definite
    push!(sectors, Σ_sector)
end

Q = 2 * blockdiag([sparse(s) for s in sectors]...)  # Factor of 2 for QP form

# Expected returns
μ = rand(n_assets)

# Build QP: min 0.5*x'Qx - μ'x subject to sum(x) = 1, x >= 0
A = sparse(ones(1, n_assets))
AL = [1.0]
AU = [1.0]
c = -μ
l = zeros(n_assets)
u = ones(n_assets)

model = build_from_QAbc(Q, A, c, AL, AU, l, u)
```

### Support Vector Machine (Dual)

```julia
using SparseArrays

# SVM dual: min 0.5*α'Qα - 1'α where Q[i,j] = y[i]*y[j]*K[i,j]
m = 1000  # Training samples

y = rand([-1, 1], m)  # Labels
K = exp.(-0.1 * (rand(m) .- rand(m)').^2)  # RBF kernel (dense)

# Q is often dense for kernel methods, but can sparsify
Q_full = sparse([(y[i] * y[j] * K[i,j]) for i in 1:m, j in 1:m])

# Sparsify by thresholding small entries
threshold = 1e-3
Q = sparse(Q_full .* (abs.(Q_full) .> threshold))

println("Sparsified to ", nnz(Q) / length(Q) * 100, "% density")
```

### Regularized Least Squares (Non-LASSO)

```julia
using SparseArrays

# Ridge regression: min ||Ax - b||² + λ||Dx||²
# where D is a regularization matrix (e.g., finite differences)

m, n = 100, 50
A = randn(m, n)
b = randn(m)
λ = 0.1

# First-order finite difference operator
D = spdiagm(0 => ones(n-1), 1 => -ones(n-1))

# Q = A'A + λD'D
Q = sparse(A' * A + λ * (D' * D))

c = -A' * b
```

## Troubleshooting

### Matrix Not Positive Semidefinite

If you get convergence issues or unbounded solutions:

```julia
using LinearAlgebra

# Check if Q is positive semidefinite
eigvals_Q = eigvals(Matrix(Q))
if minimum(eigvals_Q) < -1e-10
    @warn "Q has negative eigenvalues, not positive semidefinite"
    
    # Fix by adding regularization
    λ = abs(minimum(eigvals_Q)) + 1e-6
    Q = Q + λ * I
end
```

### Out of Memory

For very large sparse matrices:

```julia
# Use GPU to offload memory
params = HPRQP_parameters()
params.use_gpu = true

# Or reduce problem size through preprocessing
```

### Slow Performance

```julia
# Check sparsity
println("Sparsity: ", nnz(Q) / length(Q) * 100, "%")

# If too dense, consider reformulation or different Q operator
```

## See Also

- [Q Operators Overview](q_operators_overview.md) - Choosing the right operator
- [LASSO Problems](lasso_problems.md) - Alternative for least squares
- [Direct API](direct_api.md) - Building models with matrices
