using SparseArrays
using LinearAlgebra
using Random
import HPRQP

# ============================================================================
# LASSO Regression Demo using build_from_Ab_lambda
# ============================================================================
#
# This demo shows how to solve LASSO regression problems:
#     min  0.5 ||A*x - b||₂² + λ ||x||₁
#
# using the build_from_Ab_lambda function which creates an operator-based
# QP formulation optimized for LASSO problems.
# ============================================================================

println("="^70)
println("LASSO REGRESSION DEMO")
println("="^70)

# ============================================================================
# Example: Underdetermined LASSO (m < n) - sparse recovery
# ============================================================================

println("\n\n" * "="^70)
println("Example : Underdetermined LASSO Problem (Sparse Recovery)")
println("="^70)

Random.seed!(1)

m = 50
n = 200

println("\nProblem setup:")
println("  Number of samples (m):  ", m)
println("  Number of features (n): ", n)
println("  System is underdetermined (m < n)")

# Very sparse ground truth
A = sprandn(m, n, 0.1)
x_true = sprandn(n, 0.05)  # Only 5% non-zero
b = Vector(A * x_true) + 0.001 * randn(m)  # Low noise

println("  True sparsity:          ", nnz(x_true), "/", n)

# Higher lambda for sparse recovery
lambda = 0.1 * norm(A' * b, Inf)
println("  Lambda:                 ", lambda)

# Build and solve
model = HPRQP.build_from_Ab_lambda(A, b, lambda)

params = HPRQP.HPRQP_parameters()
params.stoptol = 1e-8
params.warm_up = true
params.use_gpu = false

println("\nSolving LASSO problem...")
result_cpu = HPRQP.optimize(model, params)

params.use_gpu = true
result_gpu = HPRQP.optimize(model, params)

println("\nResults:")
println("  Objective value (CPU): ", result_cpu.primal_obj)
println("  Objective value (GPU): ", result_gpu.primal_obj)
println("  Residuals (CPU):      ",  result_cpu.residuals)
println("  Residuals (GPU):      ",  result_gpu.residuals)
println("  Iterations (CPU):      ", result_cpu.iter)
println("  Iterations (GPU):      ", result_gpu.iter)