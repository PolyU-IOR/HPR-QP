# Examples

Complete, runnable examples demonstrating HPRQP usage. More examples coming soon!

For detailed guides on each input method, see:
- [Direct API Guide](guide/direct_api.md)
- [JuMP Integration Guide](guide/jump_integration.md)  
- [MPS Files Guide](guide/mps_files.md)
- [Q Operators Overview](guide/q_operators_overview.md)

## Example 1: Direct API - Basic QP

Solve a simple 2-variable QP problem using matrices.

```julia
using HPRQP
using SparseArrays

# Problem:
# min  0.5*(2x₁² + x₁x₂ + 2x₂²) - 3x₁ - 5x₂
# s.t.  x₁ + 2x₂ ≤ 10
#      3x₁ +  x₂ ≤ 12
#      x₁, x₂ ≥ 0

# Quadratic and linear terms
Q = sparse([2.0 0.5; 0.5 2.0])
c = [-3.0, -5.0]

# Standard form: AL ≤ Ax ≤ AU
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
l = [0.0, 0.0]
u = [Inf, Inf]

# Build and solve
model = build_from_QAbc(Q, A, c, AL, AU, l, u)

params = HPRQP_parameters()
params.use_gpu = false

result = optimize(model, params)

println("Status: ", result.status)
println("Objective: ", result.primal_obj)
println("Solution: x₁ = ", result.x[1], ", x₂ = ", result.x[2])
```

## Example 2: MPS Files

Read and solve a QP problem from an MPS file.

```julia
using HPRQP

# Build model from file
model = build_from_mps("qp_problem.mps")

# Configure parameters
params = HPRQP_parameters()
params.stoptol = 1e-6
params.use_gpu = true
params.verbose = true

# Solve
result = optimize(model, params)

if result.status == "OPTIMAL"
    println("✓ Optimal solution found!")
    println("  Objective: ", result.primal_obj)
    println("  Time: ", result.time, " seconds")
end
```

## Example 3: JuMP Integration

Build and solve using JuMP's modeling language.

```julia
using JuMP, HPRQP

model = Model(HPRQP.Optimizer)
set_optimizer_attribute(model, "stoptol", 1e-4)

@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, x1^2 + 0.5*x1*x2 + x2^2 - 3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

optimize!(model)

println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
println("x1 = ", value(x1), ", x2 = ", value(x2))
```

## Example 4: LASSO Problem

Solve a LASSO regression problem using the specialized LASSO operator.

```julia
using HPRQP
using SparseArrays

# LASSO problem: min 0.5||Ax - b||² + λ||x||₁
# Reformulated as QP with Q = A'A

m, n = 100, 50
A_data = randn(m, n)
b = randn(m)
λ = 0.1

# Build LASSO operator
Q_lasso = LASSOOperatorCPU(A_data, b, λ)

# No additional linear constraints (just bounds)
A = sparse(zeros(0, n))
AL = Float64[]
AU = Float64[]
c = zeros(n)
l = fill(-Inf, n)
u = fill(Inf, n)

# Build and solve
model = build_from_QAbc(Q_lasso, A, c, AL, AU, l, u)

params = HPRQP_parameters()
params.use_gpu = true

result = optimize(model, params)

println("LASSO solution found!")
println("Sparsity: ", sum(abs.(result.x) .> 1e-6), " / ", n)
```

## Example 5: Sparse Matrix QP

Solve a large sparse QP problem.

```julia
using HPRQP
using SparseArrays

# Generate a random sparse positive semidefinite matrix
n = 1000
density = 0.01
H = sprandn(n, n, density)
Q = H' * H  # Make it positive semidefinite

c = randn(n)

# Simple box constraints
A = sparse(zeros(0, n))
AL = Float64[]
AU = Float64[]
l = fill(-1.0, n)
u = fill(1.0, n)

# Build and solve
model = build_from_QAbc(Q, A, c, AL, AU, l, u)

params = HPRQP_parameters()
params.use_gpu = true
params.stoptol = 1e-4

result = optimize(model, params)

println("Solved large sparse QP!")
println("Objective: ", result.primal_obj)
println("Time: ", result.time, " seconds")
```

## Example 6: Using Warm-Start

Solve related problems with warm-start.

```julia
using HPRQP
using SparseArrays

Q = sparse([2.0 0.5; 0.5 2.0])
A = sparse([1.0 2.0; 3.0 1.0])
c = [-3.0, -5.0]
AL = [-Inf, -Inf]
AU = [10.0, 12.0]
l = [0.0, 0.0]
u = [Inf, Inf]

# First solve
model = build_from_QAbc(Q, A, c, AL, AU, l, u)
params = HPRQP_parameters()
result1 = optimize(model, params)

# Solve modified problem with warm-start
AU_new = [11.0, 12.0]
model2 = build_from_QAbc(Q, A, c, AL, AU_new, l, u)
params.initial_x = result1.x
params.initial_y = result1.y
result2 = optimize(model2, params)

println("Warm-start improved convergence!")
```

## Example 7: Auto-Save Feature

Enable auto-save for long optimizations.

```julia
using HPRQP

model = build_from_mps("large_qp_problem.mps")

params = HPRQP_parameters()
params.time_limit = 3600
params.auto_save = true
params.save_filename = "best_qp_solution.h5"

result = optimize(model, params)
```

## Example 8: Reading Auto-Saved Results

```julia
using HDF5

h5open("best_qp_solution.h5", "r") do file
    x_best = read(file, "x")
    y_best = read(file, "y")
    println("Best solution found at iteration: ", read(file, "iter"))
end
```

## Example 9: Portfolio Optimization

A classic quadratic programming application.

```julia
using JuMP, HPRQP
using Statistics

# Historical returns for 5 assets
returns = [0.12, 0.10, 0.08, 0.15, 0.09]
n_assets = length(returns)

# Covariance matrix (simplified)
Σ = [0.04 0.01 0.01 0.02 0.01;
     0.01 0.03 0.01 0.01 0.01;
     0.01 0.01 0.02 0.01 0.00;
     0.02 0.01 0.01 0.05 0.02;
     0.01 0.01 0.00 0.02 0.03]

# Risk aversion parameter
γ = 2.0

model = Model(HPRQP.Optimizer)
set_silent(model)

@variable(model, w[1:n_assets] >= 0)  # Portfolio weights
@constraint(model, sum(w) == 1)       # Fully invested

# Objective: maximize return - γ * risk
@objective(model, Max, 
    sum(returns[i] * w[i] for i in 1:n_assets) - 
    γ * sum(Σ[i,j] * w[i] * w[j] for i in 1:n_assets, j in 1:n_assets)
)

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("Optimal Portfolio:")
    for i in 1:n_assets
        if value(w[i]) > 1e-4
            println("  Asset $i: ", round(value(w[i])*100, digits=2), "%")
        end
    end
    println("Expected return: ", sum(returns .* value.(w)))
end
```

## More Examples

*More examples will be added in future releases.*
