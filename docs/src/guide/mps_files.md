# Solving MPS Files

HPRQP can directly read and solve quadratic programming problems in MPS (Mathematical Programming System) format, a widely-used industry standard format.

## Quick Start

```julia
using HPRQP

# Step 1: Build model from MPS file
model = build_from_mps("path/to/qp_problem.mps")

# Step 2: Configure solver parameters
params = HPRQP_parameters()
params.stoptol = 1e-4

# Step 3: Optimize
result = optimize(model, params)
```

## Working with MPS Files

### Basic Usage

```julia
using HPRQP

# Build the model
model = build_from_mps("qp_model.mps")

# Set up parameters
params = HPRQP_parameters()

# Solve
result = optimize(model, params)

if result.status == "OPTIMAL"
    println("Found optimal solution!")
    println("Objective: ", result.primal_obj)
    println("Solution vector: ", result.x)
else
    println("Solver stopped with status: ", result.status)
end
```

### With Custom Parameters

```julia
# Build model
model = build_from_mps("large_qp_problem.mps")

# Configure parameters
params = HPRQP_parameters()
params.stoptol = 1e-9          # Higher accuracy
params.time_limit = 3600       # 1 hour time limit
params.use_gpu = true          # Enable GPU
params.verbose = true          # Show progress
params.warm_up = true          # Enable warmup for accurate timing

# Solve
result = optimize(model, params)
```

!!! tip "Parameter Reference"
    For detailed explanations of all parameters, see the [Parameters](parameters.md) guide.

## MPS Format for QP Problems

MPS files can encode quadratic programming problems using the QUADOBJ section:

```
NAME          EXAMPLE_QP
ROWS
 N  OBJ
 L  CON1
 L  CON2
COLUMNS
    X1        OBJ       -3.0
    X1        CON1       1.0
    X1        CON2       3.0
    X2        OBJ       -5.0
    X2        CON1       2.0
    X2        CON2       1.0
RHS
    RHS1      CON1      10.0
    RHS1      CON2      12.0
BOUNDS
 LO BND1      X1         0.0
 LO BND1      X2         0.0
QUADOBJ
    X1        X1         2.0
    X1        X2         0.5
    X2        X2         2.0
ENDATA
```

The QUADOBJ section defines the Q matrix where each entry specifies a quadratic term.

## Common QP Sources

### MAROS and MESZAROS

QP test set from Maros and Meszaros:
- Contains various QP problems
- Available online from QP benchmarking repositories
- Standard benchmark for QP solvers

### CUTEst

Constrained and Unconstrained Testing Environment:
- Download from: https://github.com/JuliaSmoothOptimizers/CUTEst.jl
- Contains QP and general nonlinear problems
- Can export to MPS format

### Custom QP Problems

Generate your own MPS files from Julia:

```julia
using JuMP

# Build a QP model
model = Model()
@variable(model, x[1:n])
@objective(model, Min, 0.5*sum(Q[i,j]*x[i]*x[j] for i in 1:n, j in 1:n) + sum(c[i]*x[i] for i in 1:n))
# ... add constraints ...

# Export to MPS
write_to_file(model, "my_qp.mps")
```

## Performance Tips

1. **GPU vs CPU**: 
   - Use GPU for large problems (> 10,000 variables/constraints)
   - Use CPU for small to medium problems or when GPU is unavailable

2. **Tolerance**:
   - Use `1e-6` or `1e-8` for high-accuracy requirements
   - Default `1e-4` is suitable for most applications

3. **Time Limits**:
   - Set reasonable time limits for batch processing
   - Default is 3600 seconds (1 hour)

4. **Scaling**:
   - Keep scaling enabled for better numerical stability
   - Disable only if you have pre-scaled data

5. **Q Operator**:
   - For problems with special structure (LASSO, QAP), consider using specialized operators
   - Sparse matrix operator is used by default for MPS files

## Troubleshooting

### File Not Found

```julia
# Use absolute path or check current directory
using Pkg
pwd()  # Check current directory

model = build_from_mps("/full/path/to/problem.mps")
```

### Parsing Errors

If MPS file has format issues:
- Ensure QUADOBJ section is properly formatted
- Check that all variable names are consistent
- Verify bounds are correctly specified

### Memory Issues

For very large MPS files:
```julia
params = HPRQP_parameters()
params.use_gpu = true  # Offload to GPU memory
```

## See Also

- [Parameters](parameters.md) - Complete guide to solver parameters and configuration
- [Output & Results](output_results.md) - Understanding and interpreting solver results
- [Direct API](direct_api.md) - Alternative matrix-based input method
