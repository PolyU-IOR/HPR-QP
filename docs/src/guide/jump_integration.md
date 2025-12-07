# JuMP Integration

HPRQP integrates seamlessly with JuMP through the MathOptInterface (MOI), allowing you to use HPRQP as a backend solver for JuMP models with quadratic objectives.

## Basic Usage

```julia
using JuMP
using HPRQP

model = Model(HPRQP.Optimizer)

@variable(model, x >= 0)
@variable(model, y >= 0)
@objective(model, Min, x^2 + 0.5*x*y + y^2 - 3x - 5y)
@constraint(model, x + 2y <= 10)
@constraint(model, 3x + y <= 12)

optimize!(model)

println("Optimal value: ", objective_value(model))
println("x = ", value(x), ", y = ", value(y))
```

## Setting Solver Attributes

### Using `set_optimizer_attribute`

```julia
model = Model(HPRQP.Optimizer)

# Standard MOI attributes
set_silent(model)                      # Suppress output
set_time_limit_sec(model, 3600.0)     # 1 hour time limit

# HPRQP-specific attributes
set_optimizer_attribute(model, "stoptol", 1e-6)
set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "device_number", 0)
set_optimizer_attribute(model, "use_Ruiz_scaling", true)
set_optimizer_attribute(model, "warm_up", true)
set_optimizer_attribute(model, "max_iter", 100000)

# New features: warm-start and auto-save
set_optimizer_attribute(model, "initial_x", x0)  # Warm-start primal
set_optimizer_attribute(model, "initial_y", y0)  # Warm-start dual
set_optimizer_attribute(model, "auto_save", true)
set_optimizer_attribute(model, "save_filename", "qp_optimization.h5")
```

!!! tip "Parameter Reference"
    For detailed explanations of all parameters, see the [Parameters](parameters.md) guide.

## Querying Results

### Termination Status

```julia
optimize!(model)

status = termination_status(model)

if status == MOI.OPTIMAL
    println("Optimal solution found!")
elseif status == MOI.TIME_LIMIT
    println("Time limit reached")
elseif status == MOI.ITERATION_LIMIT
    println("Iteration limit reached")
end
```

### Objective and Solutions

```julia
if has_values(model)
    obj_val = objective_value(model)
    x_val = value(x)
    y_val = value(y)
    
    println("Objective: $obj_val")
    println("x = $x_val, y = $y_val")
end
```

### Solve Time

```julia
time = solve_time(model)
println("Solved in $time seconds")
```

## Silent Mode

Suppress all solver output:

```julia
model = Model(HPRQP.Optimizer)
set_silent(model)

# Build model...
optimize!(model)
# No output from solver
```

Or equivalently:

```julia
model = Model(HPRQP.Optimizer)
set_optimizer_attribute(model, "verbose", false)
optimize!(model)
```

## Common Patterns

### Portfolio Optimization

```julia
using JuMP, HPRQP
using LinearAlgebra

# Historical returns for assets
returns = [0.12, 0.10, 0.08, 0.15, 0.09]
n_assets = length(returns)

# Covariance matrix
Σ = [0.04 0.01 0.01 0.02 0.01;
     0.01 0.03 0.01 0.01 0.01;
     0.01 0.01 0.02 0.01 0.00;
     0.02 0.01 0.01 0.05 0.02;
     0.01 0.01 0.00 0.02 0.03]

# Risk aversion
γ = 2.0

model = Model(HPRQP.Optimizer)
set_silent(model)

@variable(model, w[1:n_assets] >= 0)
@constraint(model, sum(w) == 1)

# Mean-variance objective
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
end
```

### Ridge Regression

```julia
using JuMP, HPRQP

# Data: X is m×n feature matrix, y is m-vector of responses
m, n = 100, 20
X = randn(m, n)
y = randn(m)
λ = 0.1  # Regularization parameter

model = Model(HPRQP.Optimizer)
set_silent(model)

@variable(model, β[1:n])

# Ridge objective: ||Xβ - y||² + λ||β||²
@objective(model, Min, 
    sum((sum(X[i,j]*β[j] for j in 1:n) - y[i])^2 for i in 1:m) +
    λ * sum(β[j]^2 for j in 1:n)
)

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    β_opt = value.(β)
    println("Ridge regression coefficients: ", β_opt)
end
```

### Quadratic Assignment Problem (Relaxation)

```julia
using JuMP, HPRQP

# QAP: Assign n facilities to n locations
n = 5

# Flow matrix (traffic between facilities)
F = rand(n, n)
F = F + F'  # Make symmetric

# Distance matrix (distance between locations)
D = rand(n, n)
D = D + D'  # Make symmetric

model = Model(HPRQP.Optimizer)
set_silent(model)

# Relaxation: x[i,j] ∈ [0,1] instead of {0,1}
@variable(model, 0 <= x[i=1:n, j=1:n] <= 1)

# Each facility to exactly one location
for i in 1:n
    @constraint(model, sum(x[i,j] for j in 1:n) == 1)
end

# Each location gets exactly one facility
for j in 1:n
    @constraint(model, sum(x[i,j] for i in 1:n) == 1)
end

# QAP objective
@objective(model, Min,
    sum(F[i,k] * D[j,l] * x[i,j] * x[k,l] 
        for i in 1:n, j in 1:n, k in 1:n, l in 1:n)
)

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("QAP relaxation value: ", objective_value(model))
end
```

### Support Vector Machine (SVM)

```julia
using JuMP, HPRQP

# Binary classification data
m = 50  # Number of samples
n = 10  # Number of features
X = randn(m, n)
y = rand([-1, 1], m)  # Binary labels
C = 1.0  # Regularization parameter

model = Model(HPRQP.Optimizer)
set_silent(model)

@variable(model, w[1:n])
@variable(model, b)
@variable(model, ξ[1:m] >= 0)  # Slack variables

# Soft-margin SVM
@constraint(model, [i=1:m], y[i] * (sum(X[i,j]*w[j] for j in 1:n) + b) >= 1 - ξ[i])

@objective(model, Min, 
    0.5 * sum(w[j]^2 for j in 1:n) + C * sum(ξ[i] for i in 1:m)
)

optimize!(model)

if termination_status(model) == MOI.OPTIMAL
    println("SVM trained successfully!")
    w_opt = value.(w)
    b_opt = value(b)
end
```

## Reading MPS Files with JuMP

You can read MPS files and solve them with HPRQP via JuMP:

```julia
using JuMP, HPRQP

model = read_from_file("qp_problem.mps")
set_optimizer(model, HPRQP.Optimizer)

# Set attributes
set_optimizer_attribute(model, "stoptol", 1e-6)

# Solve
optimize!(model)

println("Status: ", termination_status(model))
if has_values(model)
    println("Objective: ", objective_value(model))
end
```

## Warm-Starting with JuMP

```julia
using JuMP, HPRQP

# Solve initial problem
model1 = Model(HPRQP.Optimizer)
@variable(model1, x >= 0)
@variable(model1, y >= 0)
@objective(model1, Min, x^2 + y^2 - 4x - 6y)
@constraint(model1, x + y <= 5)

optimize!(model1)
x_sol = value(x)
y_sol = value(y)

# Solve modified problem with warm-start
model2 = Model(HPRQP.Optimizer)
@variable(model2, x >= 0)
@variable(model2, y >= 0)
@objective(model2, Min, x^2 + y^2 - 4x - 6y)
@constraint(model2, x + y <= 6)  # Modified constraint

# Warm-start from previous solution
set_optimizer_attribute(model2, "initial_x", [x_sol, y_sol])

optimize!(model2)
```

## See Also

- [Parameters](parameters.md) - All available solver parameters
- [Examples](../examples.md) - More complete examples
- [Direct API](direct_api.md) - Lower-level matrix interface
