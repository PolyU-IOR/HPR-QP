# Solver Parameters

HPRQP provides extensive customization through the `HPRQP_parameters` type. This guide explains all available parameters and their effects on solver behavior.

## Parameter Summary Table

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `stoptol` | `1e-6` | Stopping tolerance for convergence checks. |
| `sigma` | `-1 (auto)` | Initial value of the σ parameter used in the algorithm. |
| `max_iter` | `typemax(Int32)` | Maximum number of iterations allowed. |
| `sigma_fixed` | `false` | Whether σ is fixed throughout the optimization process. |
| `time_limit` | `3600.0` | Maximum allowed runtime (seconds) for the algorithm. |
| `eig_factor` | `1.05` | Factor used to scale the maximum eigenvalue estimation. |
| `check_iter` | `100` | Frequency (in iterations) to check for convergence or perform other checks. |
| `warm_up` | `false` | Determines if a warm-up phase is performed before main execution. |
| `spmv_mode_Q` | `"auto"` | Mode for Q matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized", "operator"). |
| `spmv_mode_A` | `"auto"` | Mode for A matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized"). |
| `print_frequency` | `-1 (auto)` | Frequency (in iterations) for printing progress or logging information. |
| `device_number` | `0` | GPU device number (e.g., 0, 1, 2, 3). |
| `use_Ruiz_scaling` | `true` | Whether to apply Ruiz scaling to the problem data. |
| `use_bc_scaling` | `false` | Whether to apply bc scaling. (For QAP and LASSO, only this scaling is applicable) |
| `use_l2_scaling` | `false` | Whether to apply L2-norm based scaling. |
| `use_Pock_Chambolle_scaling` | `true` | Whether to apply Pock-Chambolle scaling to the problem data. |
| `problem_type` | `"QP"` | Type of problem being solved (e.g., "QP", "LASSO", "QAP"). |
| `lambda` | `0.0` | Regularization parameter for LASSO problems. |
| `initial_x` | `nothing` | Initial primal solution for warm-start. |
| `initial_y` | `nothing` | Initial dual solution for warm-start. |
| `auto_save` | `false` | Automatically save best x, y, z, w, and sigma during optimization. |
| `save_filename` | `"hprqp_autosave.h5"` | Filename for auto-save HDF5 file. |
| `verbose` | `true` | Enable verbose output during optimization. |
| `use_gpu` | `true` | Whether to use GPU acceleration (requires CUDA). |


## Creating Parameters

```julia
using HPRQP

# Create with default values
params = HPRQP_parameters()

# Customize as needed
params.stoptol = 1e-6
params.use_gpu = true
params.verbose = true
params.time_limit = 1800.0
```

## Parameter Details

### Convergence Control

- **`stoptol`**: Stopping tolerance for convergence checks. Lower values require higher accuracy but may take longer.
  ```julia
  params.stoptol = 1e-9  # High accuracy
  params.stoptol = 1e-4  # Faster, less accurate
  ```

- **`max_iter`**: Maximum number of iterations. Set lower to prevent excessive computation time.
  ```julia
  params.max_iter = 100000
  ```

- **`time_limit`**: Maximum runtime in seconds before the solver stops.
  ```julia
  params.time_limit = 3600.0  # 1 hour
  ```

- **`check_iter`**: How often (in iterations) to check convergence and update statistics.
  ```julia
  params.check_iter = 100
  ```

### Algorithm Parameters

- **`sigma`**: Initial value of the σ parameter. When set to `-1`, it's automatically computed.
  ```julia
  params.sigma = -1  # Auto-compute (recommended)
  params.sigma = 0.5 # Manual value
  ```

- **`sigma_fixed`**: Whether σ remains constant or adapts during optimization.
  ```julia
  params.sigma_fixed = false  # Adaptive (default)
  params.sigma_fixed = true   # Fixed
  ```

- **`eig_factor`**: Scaling factor for maximum eigenvalue estimation.
  ```julia
  params.eig_factor = 1.05
  ```

### Matrix-Vector Multiplication Modes

- **`spmv_mode_Q`**: Controls how Q matrix-vector products are computed.
  - `"auto"`: Automatically select best method
  - `"CUSPARSE"`: Use CUDA sparse matrix operations
  - `"customized"`: Use custom kernels
  - `"operator"`: Use operator interface (for LASSO/QAP)

- **`spmv_mode_A`**: Controls how A matrix-vector products are computed.
  - `"auto"`: Automatically select best method
  - `"CUSPARSE"`: Use CUDA sparse matrix operations
  - `"customized"`: Use custom kernels

### Scaling Options

Scaling improves numerical stability and convergence:

- **`use_Ruiz_scaling`**: Apply Ruiz equilibration to balance matrix rows/columns.
  ```julia
  params.use_Ruiz_scaling = true
  ```

- **`use_bc_scaling`**: Scale the objective vector (c) and constraint bounds (b). **Required for QAP and LASSO problems.**
  ```julia
  params.use_bc_scaling = false  # Default for standard QP
  params.use_bc_scaling = true   # Required for QAP/LASSO
  ```

- **`use_l2_scaling`**: Apply L2-norm based scaling.
  ```julia
  params.use_l2_scaling = false
  ```

- **`use_Pock_Chambolle_scaling`**: Apply Pock-Chambolle diagonal scaling.
  ```julia
  params.use_Pock_Chambolle_scaling = true
  ```

### GPU Configuration

- **`use_gpu`**: Enable GPU acceleration (requires CUDA).
  ```julia
  params.use_gpu = true   # Use GPU (faster for large problems)
  params.use_gpu = false  # CPU only
  ```

- **`device_number`**: Select which GPU to use (0-indexed).
  ```julia
  params.device_number = 0  # First GPU
  params.device_number = 1  # Second GPU
  ```

- **`warm_up`**: Perform warm-up to ensure accurate timing (accounts for JIT compilation).
  ```julia
  params.warm_up = false  # Default
  params.warm_up = true   # For benchmarking
  ```

### Output Control

- **`verbose`**: Enable detailed solver output.
  ```julia
  params.verbose = true   # Show progress
  params.verbose = false  # Silent
  ```

- **`print_frequency`**: How often to print iteration information. `-1` means automatic.
  ```julia
  params.print_frequency = -1   # Auto
  params.print_frequency = 100  # Every 100 iterations
  ```

### Warm-Start

- **`initial_x`**: Initial primal solution vector.
  ```julia
  params.initial_x = x0  # From previous solve
  ```

- **`initial_y`**: Initial dual solution vector.
  ```julia
  params.initial_y = y0  # From previous solve
  ```

### Auto-Save Feature

- **`auto_save`**: Automatically save the best solution during optimization.
  ```julia
  params.auto_save = true
  ```

- **`save_filename`**: HDF5 filename for auto-saved solutions.
  ```julia
  params.save_filename = "my_problem_autosave.h5"
  ```

### Problem Type

- **`problem_type`**: Specifies the type of problem being solved.
  - `"QP"`: Standard quadratic programming
  - `"LASSO"`: LASSO regression problems
  - `"QAP"`: Quadratic assignment problems

- **`lambda`**: Regularization parameter for LASSO problems.
  ```julia
  params.problem_type = "LASSO"
  params.lambda = 0.1
  ```

## Common Configurations

### High Accuracy
```julia
params = HPRQP_parameters()
params.stoptol = 1e-9
params.time_limit = 7200.0
params.max_iter = 500000
```

### Fast Approximate Solutions
```julia
params = HPRQP_parameters()
params.stoptol = 1e-4
params.time_limit = 300.0
params.check_iter = 200
```

### CPU-Only (No GPU Available)
```julia
params = HPRQP_parameters()
params.use_gpu = false
params.verbose = true
```

### LASSO Problems
```julia
params = HPRQP_parameters()
params.problem_type = "LASSO"
params.lambda = 0.1
params.use_bc_scaling = true  # Required for LASSO
params.spmv_mode_Q = "operator"
```

### QAP Problems
```julia
params = HPRQP_parameters()
params.problem_type = "QAP"
params.use_bc_scaling = true  # Required for QAP
params.spmv_mode_Q = "operator"
```

### With Auto-Save and Warm-Start
```julia
params = HPRQP_parameters()
params.auto_save = true
params.save_filename = "my_solution.h5"
params.initial_x = x0
params.initial_y = y0
```

### Silent/Batch Processing
```julia
params = HPRQP_parameters()
params.verbose = false
params.warm_up = false
```

### Debugging/Analysis
```julia
params = HPRQP_parameters()
params.verbose = true
params.print_frequency = 10
params.check_iter = 10
params.max_iter = 1000
```