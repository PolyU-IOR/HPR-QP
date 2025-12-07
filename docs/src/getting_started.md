# Getting Started

## Prerequisites

Before using HPRQP, ensure you have:

- **Julia** (version 1.10 or higher recommended)
- **CUDA** (optional, for GPU acceleration)
  - CUDA Toolkit 12.4 or higher for best compatibility

## Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/PolyU-IOR/HPR-QP")
```

### Development Installation

To install for development:

```bash
git clone https://github.com/PolyU-IOR/HPR-QP.git
cd HPR-QP
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Verifying CUDA Installation

If you plan to use GPU acceleration, verify CUDA is working:

```julia
using CUDA
CUDA.versioninfo()
```

If CUDA is not available, HPRQP will automatically fall back to CPU mode.

## First Example: Solving a Simple QP

Let's solve a basic quadratic programming problem:

```math
\begin{array}{ll}
\min \quad & \frac{1}{2}(2x_1^2 + x_1x_2 + 2x_2^2) - 3x_1 - 5x_2 \\
\text{s.t.} \quad & x_1 + 2x_2 \leq 10 \\
& 3x_1 + x_2 \leq 12 \\
& x_1, x_2 \geq 0
\end{array}
```

### Using the Direct API

```julia
using HPRQP
using SparseArrays

# Define the quadratic objective: 0.5*x'*Q*x + c'*x
Q = sparse([2.0 0.5; 0.5 2.0])
c = [-3.0, -5.0]

# Convert constraints to standard form: AL ≤ Ax ≤ AU
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
l = [0.0, 0.0]
u = [Inf, Inf]

# Step 1: Build the model
model = build_from_QAbc(Q, A, c, AL, AU, l, u)

# Step 2: Set up parameters
params = HPRQP_parameters()
params.use_gpu = false      # Use CPU for this small problem
params.stoptol = 1e-4       # Convergence tolerance
params.time_limit = 60      # Maximum 60 seconds
params.verbose = true       # Print solver output

# Step 3: Solve
result = optimize(model, params)

# Check results
println("Status: ", result.status)
println("Optimal value: ", result.primal_obj)
println("Solution: x₁ = ", result.x[1], ", x₂ = ", result.x[2])
println("Iterations: ", result.iter)
println("Solve time: ", result.time, " seconds")
```

### Using JuMP

The same problem using the JuMP modeling interface:

```julia
using JuMP
using HPRQP

model = Model(HPRQP.Optimizer)

# Set optimizer attributes
set_optimizer_attribute(model, "stoptol", 1e-4)
set_optimizer_attribute(model, "use_gpu", false)
set_optimizer_attribute(model, "verbose", true)

# Define variables and constraints
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)
@objective(model, Min, x1^2 + 0.5*x1*x2 + x2^2 - 3x1 - 5x2)
@constraint(model, x1 + 2x2 <= 10)
@constraint(model, 3x1 + x2 <= 12)

# Solve
optimize!(model)

# Get results
println("Status: ", termination_status(model))
println("Optimal value: ", objective_value(model))
println("Solution: x₁ = ", value(x1), ", x₂ = ", value(x2))
println("Solve time: ", solve_time(model), " seconds")
```

### Solve from MPS Files

```julia
using HPRQP

# Build model from file (for QP problems in MPS format)
model = build_from_mps("model.mps")

# Set parameters
params = HPRQP_parameters()

# Solve
result = optimize(model, params)
```

## Next Steps

- Learn about [solving MPS files](guide/mps_files.md)
- Explore the [Direct API](guide/direct_api.md) in detail
- See more [JuMP integration examples](guide/jump_integration.md)
- Understand [Q Operators](guide/q_operators_overview.md) for different problem types
- Check out the full [API Reference](api.md)
- Browse additional [Examples](examples.md)

## Performance Tips

1. **JIT Compilation**: The first run will be slow due to Julia's JIT compilation. For benchmarking, run the solver twice or use a warm-up phase.

2. **GPU Usage**: For large problems (typically > 10,000 variables/constraints), GPU acceleration can provide significant speedups.

3. **Scaling**: The default scaling methods (Ruiz and Pock-Chambolle) improve numerical stability. Keep them enabled unless you have specific reasons to disable them.

4. **Tolerance**: The default `stoptol = 1e-4` is relatively loose. Increase for more accurate solutions in critical applications.

5. **Q Operator Selection**: Choose the appropriate Q operator for your problem structure (sparse matrix, LASSO, QAP) for best performance.
