# LASSO Problems

The LASSO (Least Absolute Shrinkage and Selection Operator) is a regression method that performs both variable selection and regularization. HPRQP provides a specialized operator for efficiently solving LASSO problems.

## Problem Formulation

The LASSO problem is:

```math
\min_{x} \quad \frac{1}{2} \|Ax - b\|^2 + \lambda \|x\|_1
```

where:
- ``A \in \mathbb{R}^{m \times n}`` is the design matrix (features)
- ``b \in \mathbb{R}^m`` is the response vector
- ``x \in \mathbb{R}^n`` is the coefficient vector to solve for
- ``\lambda > 0`` is the regularization parameter

This is equivalent to the QP:

```math
\min_{x} \quad \frac{1}{2} x^T (A^T A) x - (A^T b)^T x + \lambda \|x\|_1
```

## Why Use the LASSO Operator?

Instead of explicitly forming ``Q = A^T A``:

**Benefits:**
- ✅ **Memory efficient**: Stores ``A`` instead of ``A^T A``
- ✅ **Numerically stable**: Avoids conditioning issues from forming ``A^T A``
- ✅ **Faster**: Two matrix-vector products ``A^T(Ax)`` instead of one with ``A^T A``

**When ``m = 1000, n = 500``:**
- Sparse matrix operator: stores ``A^T A`` (potentially ``n^2 = 250000`` entries)
- LASSO operator: stores ``A`` (only ``mn = 500000`` entries, but structured)

## Basic Usage

```julia
using HPRQP

# Problem data
m, n = 100, 50
A = randn(m, n)  # Design matrix
b = randn(m)     # Observations
λ = 0.1          # Regularization parameter

# Create LASSO operator
Q_lasso = LASSOOperatorCPU(A, b, λ)

# No additional constraints (or add your own)
A_constr = sparse(zeros(0, n))
AL = Float64[]
AU = Float64[]
c = zeros(n)  # Linear term is handled by LASSO operator
l = fill(-Inf, n)
u = fill(Inf, n)

# Build and solve
model = build_from_QAbc(Q_lasso, A_constr, c, AL, AU, l, u)

params = HPRQP_parameters()
params.use_gpu = true
result = optimize(model, params)

# Analyze sparsity
sparsity = sum(abs.(result.x) .> 1e-6)
println("Non-zero coefficients: ", sparsity, " / ", n)
println("Sparsity: ", (1 - sparsity/n) * 100, "%")
```

## Complete Example with Constraints

```julia
using HPRQP
using SparseArrays

# Generate synthetic data
m, n = 200, 100
A = randn(m, n)
x_true = sparsevec([1, 5, 10, 25, 50], randn(5), n)  # Sparse truth
noise = 0.1 * randn(m)
b = A * x_true + noise

λ = 0.5  # Regularization

# Create LASSO operator
Q_lasso = LASSOOperatorCPU(A, b, λ)

# Add non-negativity constraint
A_constr = sparse(zeros(0, n))
AL = Float64[]
AU = Float64[]
c = zeros(n)
l = zeros(n)  # x >= 0 (non-negative LASSO)
u = fill(Inf, n)

# Solve
model = build_from_QAbc(Q_lasso, A_constr, c, AL, AU, l, u)

params = HPRQP_parameters()
result = optimize(model, params)

# Compare to true solution
x_recovered = result.x
println("Recovery error: ", norm(x_recovered - x_true))
println("Support recovery: ", sum((abs.(x_recovered) .> 1e-3) .& (abs.(x_true) .> 0)))
```

## Choosing the Regularization Parameter λ

The regularization parameter λ controls the sparsity-accuracy trade-off:

- **Small λ**: Less regularization, more non-zero coefficients, better fit
- **Large λ**: More regularization, sparser solution, worse fit

### Cross-Validation

```julia
using HPRQP

function lasso_cv(A, b, λ_values; n_folds=5)
    m, n = size(A)
    fold_size = div(m, n_folds)
    cv_errors = zeros(length(λ_values))
    
    for (i, λ) in enumerate(λ_values)
        fold_errors = zeros(n_folds)
        
        for fold in 1:n_folds
            # Split data
            test_idx = (fold-1)*fold_size+1 : fold*fold_size
            train_idx = setdiff(1:m, test_idx)
            
            A_train = A[train_idx, :]
            b_train = b[train_idx]
            A_test = A[test_idx, :]
            b_test = b[test_idx]
            
            # Solve LASSO on training set
            Q_lasso = LASSOOperatorCPU(A_train, b_train, λ)
            model = build_from_QAbc(Q_lasso, sparse(zeros(0,n)), 
                                    zeros(n), Float64[], Float64[], 
                                    fill(-Inf,n), fill(Inf,n))
            
            params = HPRQP_parameters()
            params.verbose = false
            result = optimize(model, params)
            
            # Test error
            fold_errors[fold] = norm(A_test * result.x - b_test)^2 / length(test_idx)
        end
        
        cv_errors[i] = mean(fold_errors)
    end
    
    best_idx = argmin(cv_errors)
    return λ_values[best_idx], cv_errors
end

# Use it
λ_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
λ_best, errors = lasso_cv(A, b, λ_values)
println("Best λ: ", λ_best)
```

## Elastic Net Variation

Elastic net combines L1 and L2 regularization:

```math
\min_{x} \quad \frac{1}{2} \|Ax - b\|^2 + \lambda_1 \|x\|_1 + \frac{\lambda_2}{2} \|x\|^2
```

```julia
# Elastic net as modified LASSO
λ1 = 0.1  # L1 penalty
λ2 = 0.05  # L2 penalty

# Augment A and b for L2 penalty
A_aug = vcat(A, sqrt(λ2) * I(n))
b_aug = vcat(b, zeros(n))

Q_elastic = LASSOOperatorCPU(A_aug, b_aug, λ1)

model = build_from_QAbc(Q_elastic, A_constr, c, AL, AU, l, u)
```

## Applications

### Feature Selection

```julia
# Select important features from high-dimensional data
m, n = 100, 1000  # More features than samples!

A = randn(m, n)
x_sparse = sparsevec(rand(1:n, 10), randn(10), n)  # Only 10 important features
b = A * x_sparse + 0.1 * randn(m)

# Use LASSO to identify important features
λ = 0.5
Q_lasso = LASSOOperatorCPU(A, b, λ)
model = build_from_QAbc(Q_lasso, sparse(zeros(0,n)), zeros(n), 
                        Float64[], Float64[], fill(-Inf,n), fill(Inf,n))

params = HPRQP_parameters()
result = optimize(model, params)

# Identify selected features
threshold = 1e-4
selected_features = findall(abs.(result.x) .> threshold)
println("Selected ", length(selected_features), " out of ", n, " features")
```

### Compressed Sensing

```julia
# Recover sparse signal from compressed measurements
n = 1000  # Signal dimension
m = 200   # Number of measurements (m << n)
k = 20    # Sparsity level

# Measurement matrix (random Gaussian)
A = randn(m, n) / sqrt(m)

# Sparse signal
x_true = sparsevec(rand(1:n, k), randn(k), n)

# Compressed measurements
b = A * x_true

# Recover using LASSO
λ = 0.01
Q_lasso = LASSOOperatorCPU(A, b, λ)
model = build_from_QAbc(Q_lasso, sparse(zeros(0,n)), zeros(n),
                        Float64[], Float64[], fill(-Inf,n), fill(Inf,n))

params = HPRQP_parameters()
result = optimize(model, params)

println("Recovery error: ", norm(result.x - x_true) / norm(x_true))
```

### Time Series (Trend Filtering)

```julia
# LASSO with finite difference regularization
T = 200  # Time points
y = sin.(range(0, 2π, length=T)) + 0.2 * randn(T)

# Identity observation
A = Matrix(1.0I, T, T)
b = y

# But add constraint on differences (trend filtering)
# This is approximated using LASSO on transformed space

λ = 0.1
Q_lasso = LASSOOperatorCPU(A, b, λ)
model = build_from_QAbc(Q_lasso, sparse(zeros(0,T)), zeros(T),
                        Float64[], Float64[], fill(-Inf,T), fill(Inf,T))

result = optimize(model, HPRQP_parameters())
x_smooth = result.x

using Plots
plot(y, label="Noisy", alpha=0.5)
plot!(x_smooth, label="LASSO smoothed", linewidth=2)
```

## Performance Tips

1. **GPU Acceleration**: LASSO operator benefits significantly from GPU for large problems
   ```julia
   params = HPRQP_parameters()
   params.use_gpu = true
   ```

2. **Scaling**: Standardize features for better numerical behavior
   ```julia
   # Standardize columns of A
   A_mean = mean(A, dims=1)
   A_std = std(A, dims=1)
   A_scaled = (A .- A_mean) ./ A_std
   ```

3. **Warm-start**: Solve for sequence of λ values using warm-start
   ```julia
   λ_path = [1.0, 0.5, 0.1, 0.05, 0.01]
   solutions = []
   
   for λ in λ_path
       Q_lasso = LASSOOperatorCPU(A, b, λ)
       model = build_from_QAbc(Q_lasso, ...)
       
       params = HPRQP_parameters()
       if !isempty(solutions)
           params.initial_x = solutions[end]  # Warm-start
       end
       
       result = optimize(model, params)
       push!(solutions, result.x)
   end
   ```

## See Also

- [Q Operators Overview](q_operators_overview.md) - Understanding Q operators
- [Sparse Matrix QP](sparse_matrix_qp.md) - Alternative general approach
- [Direct API](direct_api.md) - Building models
