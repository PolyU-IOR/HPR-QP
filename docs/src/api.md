# API Reference

Complete API documentation for HPRQP. For detailed guides, see:
- [Parameters Guide](guide/parameters.md) - Detailed parameter explanations
- [Output & Results](guide/output_results.md) - Understanding solver output

## Main Functions

```@docs
build_from_mps
build_from_QAbc
build_from_mat
build_from_ABST
build_from_Ab_lambda
optimize
HPRQP.Optimizer
```

## Main Types

### HPRQP_parameters

Solver parameters struct. Create with default values using `HPRQP_parameters()`.

**Key Parameters:**
- `stoptol::Float64`: Convergence tolerance (default: 1e-6)
- `max_iter::Int`: Maximum iterations (default: 1000000)
- `use_gpu::Bool`: Enable GPU acceleration (default: true)
- `verbose::Bool`: Print iteration logs (default: true)
- `warm_up::Bool`: Run warmup solve to eliminate JIT overhead (default: false)
- `time_limit::Float64`: Maximum solve time in seconds (default: Inf)

See [Parameters Guide](guide/parameters.md) for complete list.

### HPRQP_results

Solution results struct returned by `optimize()`.

**Main Fields:**
- `status::String`: Solution status ("OPTIMAL", "MAX_ITER", "TIME_LIMIT")
- `x::Vector{Float64}`: Primal solution vector
- `y::Vector{Float64}`: Dual solution for constraints
- `z::Vector{Float64}`: Dual solution for bounds
- `primal_obj::Float64`: Primal objective value
- `dual_obj::Float64`: Dual objective value
- `iter::Int`: Total iterations
- `time::Float64`: Solve time in seconds

See [Output & Results Guide](guide/output_results.md) for complete list.

## Q Operators

The following Q operator types are available for specialized problems:

- `LASSO_Q_operator_cpu`: For LASSO (L1-regularized least squares) problems
- `QAP_Q_operator_cpu`: For quadratic assignment problems
- Sparse matrices (`SparseMatrixCSC`): For general quadratic programming

See the [Q Operators Guide](guide/q_operators_overview.md) for detailed information.

## Quick Reference

### Solving Problems

**Direct API (Matrix Form):**
```julia
# Step 1: Build the model
model = build_from_QAbc(Q, A, c, AL, AU, l, u)

# Step 2: Set parameters
params = HPRQP_parameters()
params.stoptol = 1e-6

# Step 3: Optimize
result = optimize(model, params)
```

**MPS Files:**
```julia
# Step 1: Build the model from file
model = build_from_mps("problem.mps")

# Step 2: Set parameters
params = HPRQP_parameters()

# Step 3: Optimize
result = optimize(model, params)
```

**JuMP:**
```julia
model = Model(HPRQP.Optimizer)
# ... add variables and constraints ...
optimize!(model)
```

### Common Parameter Settings

```julia
params = HPRQP_parameters()
params.stoptol = 1e-6           # Convergence tolerance
params.use_gpu = true           # Enable GPU
params.verbose = false          # Silent mode
params.time_limit = 3600        # Time limit (seconds)
params.warm_up = true           # Enable warmup for accurate timing
params.initial_x = x0           # Initial primal solution
params.initial_y = y0           # Initial dual solution
params.auto_save = true         # Auto-save best solution
params.save_filename = "opt.h5" # HDF5 file for auto-save
```

### Accessing Results

```julia
result.status         # "OPTIMAL", "MAX_ITER", or "TIME_LIMIT"
result.primal_obj     # Primal objective value
result.x              # Primal solution vector
result.y              # Dual solution vector (constraints)
result.z              # Dual solution vector (bounds)
result.iter           # Total iterations
result.time           # Solve time (seconds)
result.residuals      # Final residual (max of primal, dual, gap)
```

## See Also

- [User Guide](guide/input_overview.md) - Comprehensive usage guides
- [Examples](examples.md) - Complete working examples
- [Q Operators](guide/q_operators_overview.md) - Understanding Q operators
