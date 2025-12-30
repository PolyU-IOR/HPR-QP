# HPR-QP: A GPU Solver for Convex Composite Quadratic Programming in Julia

[![CI](https://github.com/PolyU-IOR/HPR-QP/actions/workflows/CI.yml/badge.svg)](https://github.com/PolyU-IOR/HPR-QP/actions/workflows/CI.yml)
[![Documentation](https://github.com/PolyU-IOR/HPR-QP/actions/workflows/Documentation.yml/badge.svg)](https://github.com/PolyU-IOR/HPR-QP/actions/workflows/Documentation.yml)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://PolyU-IOR.github.io/HPR-QP/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://PolyU-IOR.github.io/HPR-QP/dev/)

> **HPR-QP: A dual Halpern Peaceman–Rachford (HPR) method for solving large-scale convex composite quadratic programming (CCQP).**

---

## 🎉 What's New in v0.1.1

This version represents a major architectural overhaul with significant improvements:

### **1. Unified Architecture**
- **Single codebase** for all problem types (standard QP, QAP, LASSO)
- Modular **Q operator system** for extensibility: easily add custom problem types

### **2. CPU & GPU Support**
- **Full CPU implementation** in addition to GPU acceleration
- Automatic device selection via `use_gpu` parameter

### **3. JuMP Integration**
- **Native JuMP/MathOptInterface (MOI) support** for easy modeling
- Use HPR-QP directly as a JuMP optimizer

### **4. Warm-Start Capability**
- Initialize via `initial_x` and `initial_y` parameters
- Resume optimization from previous (auto-saved) solutions

### **5. Auto-Save Feature**
- Automatically save best solution during optimization (`auto_save=true`)
- Resume from saved states for long-running problems

---

## CCQP Problem Formulation

<div align="center">

$$
\begin{array}{ll}
\underset{x \in \mathbb{R}^n}{\min} \quad \frac{1}{2}\langle x,Qx \rangle + \langle c, x \rangle +\phi(x)\\
\text{s.t.} \quad \quad \quad \quad \quad   l \leq  A x \leq u,
\end{array}
$$

</div>

- $Q$ is a positive semidefinite self-adjoint linear operator;
- $Q$'s matrix representation may not be computable in large-scale instances, such as QAP relaxation and LASSO problems;
- $\phi$ is a proper closed convex function.
---

# Getting Started

## Prerequisites

Before using HPR-QP, make sure the following dependencies are installed:

- **Julia** (Recommended version: `1.10.4`)
- **CUDA** (Required for GPU acceleration; install the appropriate version for your GPU and Julia)
- Required Julia packages

> To install the required Julia packages and build the HPR-QP environment, run:
```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

> To verify that CUDA is properly installed and working with Julia, run:
```julia
using CUDA
CUDA.versioninfo()
```

---

## Usage 1: Test Instances in MPS (MAT for QAP and LASSO) Format

### Setting Data and Result Paths

> Before running the scripts, please modify **`run_single_file.jl`** or **`run_dataset.jl`** in the demo directory to specify the data path and result path according to your setup.

### Running a Single Instance

To test the script on a single instance:

```bash
julia --project demo/run_single_file.jl
```

### Running All Instances in a Directory

To process all files in a directory:

```bash
julia --project demo/run_dataset.jl
```

### Note

> **QAP Instances:**  
> The `.mat` file for QAP should include the matrices **$A$**, **$B$**, **$S$**, and **$T$**.  
> For details, refer to Section 4.5 of the paper.  
> See [`HPR-QP_QAP_LASSO/demo/demo_QAP.jl`](HPR-QP_QAP_LASSO/demo/demo_QAP.jl) for an example of generating such files.
>
> **LASSO Instances:**  
> The `.mat` file for LASSO should contain the matrix **$A$**, vector **$b$**.

---

## Usage 2: Define Your CQP Model in Julia Scripts

### Example 1: Build and Export a CQP Model Using JuMP

This example demonstrates how to construct a CQP model using the JuMP modeling language in Julia and export it to MPS format for use with the HPR-QP solver.

```bash
julia --project demo/demo_JuMP.jl
```

The script:
- Builds a CQP model.
- Uses HPR-QP to solve the CQP instance.

> **Remark:** If the model may be infeasible or unbounded, you can use HiGHS to check it.

```julia
using JuMP, HiGHS
## read a model from file (or create in other ways)
mps_file_path = "xxx" # your file path
model = read_from_file(mps_file_path)
## set HiGHS as the optimizer
set_optimizer(model, HiGHS.Optimizer)
## solve it
optimize!(model)
```

---

### Example 2: Define a CQP Instance Directly in Julia

This example demonstrates how to construct and solve a CQP problem directly in Julia without relying on JuMP.

```bash
julia --project demo/demo_QAbc.jl
```

---

### Example 3: Generate a Random LASSO Instance in Julia

This example demonstrates how to construct and solve a random LASSO instance.

```bash
julia --project demo/demo_LASSO.jl
```


---

## Note on First-Time Execution Performance

You may notice that solving a single instance — or the first instance in a dataset — appears slow. This is due to Julia’s Just-In-Time (JIT) compilation, which compiles code on first execution.

> **💡 Tip for Better Performance:**  
> To reduce repeated compilation overhead, it’s recommended to run scripts from an **IDE like VS Code** or the **Julia REPL** in the terminal.

#### Start Julia REPL with the project environment:

```bash
julia --project
```

Then, at the Julia REPL, run demo/demo_QAbc.jl (or other scripts):

```julia
include("demo/demo_QAbc.jl")
```

> **CAUTION:**  
> If you encounter the error message:  
> `Error: Error during loading of extension AtomixCUDAExt of Atomix, use Base.retry_load_extensions() to retry`.
>
> Don’t panic — this is usually a transient issue. Simply wait a few moments; the extension typically loads successfully on its own.

---

## Parameters

Below is a list of the parameters in HPR-QP along with their default values and usage:

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Default Value</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>stoptol</code></td><td><code>1e-6</code></td><td>Stopping tolerance for convergence checks.</td></tr>
    <tr><td><code>sigma</code></td><td><code>-1 (auto)</code></td><td>Initial value of the σ parameter used in the algorithm.</td></tr>
    <tr><td><code>max_iter</code></td><td><code>typemax(Int32)</code></td><td>Maximum number of iterations allowed.</td></tr>
    <tr><td><code>sigma_fixed</code></td><td><code>false</code></td><td>Whether σ is fixed throughout the optimization process.</td></tr>
    <tr><td><code>time_limit</code></td><td><code>3600.0</code></td><td>Maximum allowed runtime (seconds) for the algorithm.</td></tr>
    <tr><td><code>eig_factor</code></td><td><code>1.05</code></td><td>Factor used to scale the maximum eigenvalue estimation.</td></tr>
    <tr><td><code>check_iter</code></td><td><code>100</code></td><td>Frequency (in iterations) to check for convergence or perform other checks.</td></tr>
    <tr><td><code>warm_up</code></td><td><code>false</code></td><td>Determines if a warm-up phase is performed before main execution.</td></tr>
    <tr><td><code>spmv_mode_Q</code></td><td><code>"auto"</code></td><td>Mode for Q matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized", "operator").</td></tr>
    <tr><td><code>spmv_mode_A</code></td><td><code>"auto"</code></td><td>Mode for A matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized").</td></tr>
    <tr><td><code>print_frequency</code></td><td><code>-1 (auto)</code></td><td>Frequency (in iterations) for printing progress or logging information.</td></tr>
    <tr><td><code>device_number</code></td><td><code>0</code></td><td>GPU device number (e.g., 0, 1, 2, 3).</td></tr>
    <tr><td><code>use_Ruiz_scaling</code></td><td><code>true</code></td><td>Whether to apply Ruiz scaling to the problem data.</td></tr>
    <tr><td><code>use_bc_scaling</code></td><td><code>false</code></td><td>Whether to apply bc scaling. (For QAP and LASSO, only this scaling is applicable)</td></tr>
    <tr><td><code>use_l2_scaling</code></td><td><code>false</code></td><td>Whether to apply L2-norm based scaling.</td></tr>
    <tr><td><code>use_Pock_Chambolle_scaling</code></td><td><code>true</code></td><td>Whether to apply Pock-Chambolle scaling to the problem data.</td></tr>
    <tr><td><code>problem_type</code></td><td><code>"QP"</code></td><td>Type of problem being solved (e.g., "QP", "LASSO", "QAP").</td></tr>
    <tr><td><code>lambda</code></td><td><code>0.0</code></td><td>Regularization parameter for LASSO problems.</td></tr>
    <tr><td><code>initial_x</code></td><td><code>nothing</code></td><td>Initial primal solution for warm-start.</td></tr>
    <tr><td><code>initial_y</code></td><td><code>nothing</code></td><td>Initial dual solution for warm-start.</td></tr>
    <tr><td><code>auto_save</code></td><td><code>false</code></td><td>Automatically save best x, y, z, w, and sigma during optimization.</td></tr>
    <tr><td><code>save_filename</code></td><td><code>"hprqp_autosave.h5"</code></td><td>Filename for auto-save HDF5 file.</td></tr>
    <tr><td><code>verbose</code></td><td><code>true</code></td><td>Enable verbose output during optimization.</td></tr>
    <tr><td><code>use_gpu</code></td><td><code>true</code></td><td>Whether to use GPU acceleration (requires CUDA).</td></tr>
  </tbody>
</table>

---

# Result Explanation

After solving an instance, you can access the result variables as shown below:

```julia
# Example from /demo/demo_QAbc.jl
println("Objective value: ", result.primal_obj)
println("x1 = ", result.x[1])
println("x2 = ", result.x[2])
```

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Variable</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Iteration Counts</b></td><td><code>iter</code></td><td>Total number of iterations performed by the algorithm.</td></tr>
    <tr><td></td><td><code>iter_4</code></td><td>Number of iterations required to achieve an accuracy of 1e-4.</td></tr>
    <tr><td></td><td><code>iter_6</code></td><td>Number of iterations required to achieve an accuracy of 1e-6.</td></tr>
    <tr><td><b>Time Metrics</b></td><td><code>time</code></td><td>Total time in seconds taken by the algorithm.</td></tr>
    <tr><td></td><td><code>time_4</code></td><td>Time in seconds taken to achieve an accuracy of 1e-4.</td></tr>
    <tr><td></td><td><code>time_6</code></td><td>Time in seconds taken to achieve an accuracy of 1e-6.</td></tr>
    <tr><td></td><td><code>power_time</code></td><td>Time in seconds used by the power method.</td></tr>
    <tr><td><b>Objective Values</b></td><td><code>primal_obj</code></td><td>The primal objective value obtained.</td></tr>
    <tr><td></td><td><code>gap</code></td><td>The gap between the primal and dual objective values.</td></tr>
    <tr><td><b>Residuals</b></td><td><code>residuals</code></td><td>Relative residuals of the primal feasibility, dual feasibility, and duality gap.</td></tr>
    <tr><td><b>Algorithm Status</b></td><td><code>status</code></td><td>The final status of the algorithm:<br/>- <code>OPTIMAL</code>: Found optimal solution<br/>- <code>MAX_ITER</code>: Max iterations reached<br/>- <code>TIME_LIMIT</code>: Time limit reached</td></tr>
    <tr><td><b>Solution Vectors</b></td><td><code>x</code></td><td>The final solution vector <code>x</code>.</td></tr>
    <tr><td></td><td><code>y</code></td><td>The final solution vector <code>y</code>.</td></tr>
    <tr><td></td><td><code>z</code></td><td>The final solution vector <code>z</code>.</td></tr>
    <tr><td></td><td><code>w</code></td><td>The final solution vector <code>w</code>.</td></tr>
  </tbody>
</table>

---

## Citation

```bibtex
@article{chen2025hpr,
  title={HPR-QP: A dual Halpern Peaceman-Rachford method for solving large-scale convex composite quadratic programming},
  author={Chen, Kaihuang and Sun, Defeng and Yuan, Yancheng and Zhang, Guojun and Zhao, Xinyuan},
  journal={arXiv preprint arXiv:2507.02470},
  year={2025}
}
```
