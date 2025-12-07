# HPR-QP: A GPU Solver for Convex Composite Quadratic Programming in Julia

> **HPR-QP: A dual Halpern Peaceman–Rachford (HPR) method for solving large-scale convex composite quadratic programming (CCQP).**

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

## Numerical Results

- **HPR-QP** is implemented in Julia and leverages CUDA for GPU acceleration
- [**PDQP**](https://github.com/jinwen-yang/PDQP.jl) (GPU, downloaded in April 2025)
- [**SCS**](https://github.com/jump-dev/SCS.jl) (GPU, version 2.1.0) is written in C/C++ with a Julia interface. GPU acceleration is enabled via its indirect solver, which performs all matrix operations on the GPU
- [**CuClarabel**](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel) (GPU, version 0.10.0)
- [**Gurobi**](https://www.gurobi.com/) (CPU, version 12.0.2, academic license) is executed on CPU using the barrier method
- All benchmarks were conducted on a SuperServer SYS-420GP-TNR with an NVIDIA A100-SXM4-80GB GPU, Intel Xeon Platinum 8338C CPU @ 2.60 GHz, and 256 GB RAM

### Maros-Mészáros Data Set (137 Instances, Tolerance $10^{-6}$ and $10^{-8}$)

| **Solver**   | **SGM10<br/>($10^{-6}$)** | **Solved<br/>($10^{-6}$)** | **SGM10<br/>($10^{-8}$)** | **Solved<br/>($10^{-8}$)** |
|:------------ | ------------------------: | -------------------------: | ------------------------: | -------------------------: |
| **HPR-QP**   | 10.5                      | 129                        | 12.6                      | 128                        |
| **PDQP**     | 33.1                      | 125                        | 42.5                      | 124                        |
| **SCS**      | 126.0                     | 103                        | 165.0                     | 93                         |
| **CuClarabel** | 3.7                     | 130                        | 7.8                       | 124                        |
| **Gurobi**   | 0.4                       | 137                        | 1.2                       | 135                        |

---

### QAP Relaxations (36 Instances, Tolerance $10^{-6}$ and $10^{-8}$)

| **Solver**   | **SGM10<br/>($10^{-6}$)** | **Solved<br/>($10^{-6}$)** | **SGM10<br/>($10^{-8}$)** | **Solved<br/>($10^{-8}$)** |
|:------------ | ------------------------: | -------------------------: | ------------------------: | -------------------------: |
| **HPR-QP**   | 1.8                       | 36                         | 4.7                       | 36                         |
| **PDQP**     | 124.1                     | 23                         | 149.4                     | 23                         |
| **SCS**      | 11.3                      | 36                         | 86.0                      | 36                         |
| **CuClarabel** | 13.6                    | 33                         | 114.9                     | 22                         |
| **Gurobi**   | 24.8                      | 36                         | 26.8                      | 36                         |

---

### LASSO Problems (11 Instances, Tolerance $10^{-8}$)

Abbreviations: **T** = time-limit, **F** = failure (e.g., unbounded or infeasible).

| **Instance**         | **HPR-QP** | **PDQP** | **SCS** | **CuClarabel** | **Gurobi** |
|:---------------------| ----------:| --------:| -------:| -------------:| ----------:|
| abalone7             | 10.5       | 372.5    | T       | 24.4          | 127.3      |
| bodyfat7             | 1.2        | 33.3     | T       | 2.2           | 30.8       |
| E2006.test           | 0.2        | 1.3      | T       | 15.4          | 9.0        |
| E2006.train          | 0.7        | 1.9      | F       | 116.0         | 277.8      |
| housing7             | 22.6       | 123.3    | T       | 5.7           | 125.9      |
| log1p.E2006.test     | 7.0        | 1416.9   | T       | 196.0         | 137.0      |
| log1p.E2006.train    | 17.3       | 2983.2   | T       | 361.0         | 878.8      |
| mpg7                 | 0.6        | 18.1     | 2000.0  | 0.3           | 1.2        |
| pyrim5               | 49.1       | 410.6    | T       | 3.5           | 35.9       |
| space_ga9            | 0.6        | 62.7     | 1210.0  | 6.7           | 38.1       |
| triazines4           | 401.3      | 3533.3   | T       | 26.0          | 843.1      |
| **SGM10 (Time)**     | **13.2**   | **161.8**| **3091.0** | **26.1**   | **91.2**   |

---

# Getting Started

## First Step — Pick the Right Solver for Your Problem
If you need to solve a LASSO problem (with an $\ell_1$ regularizer) or a QAP instance where the matrix form of $Q$ is unavailable, please refer to the HPR-QP_QAP_LASSO module.


```shell
cd HPR-QP_QAP_LASSO
```

otherwise, for convex QP (COP) problems where the matrix form of $Q$ is available, please refer to the HPR-QP module.

```shell
cd HPR-QP
```

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
- Saves the model as an MPS file.
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
    <tr><td><code>warm_up</code></td><td><code>false</code></td><td>Determines if a warm-up phase is performed before main execution.</td></tr>
    <tr><td><code>time_limit</code></td><td><code>3600</code></td><td>Maximum allowed runtime (seconds) for the algorithm.</td></tr>
    <tr><td><code>stoptol</code></td><td><code>1e-6</code></td><td>Stopping tolerance for convergence checks.</td></tr>
    <tr><td><code>device_number</code></td><td><code>0</code></td><td>GPU device number (only relevant if <code>use_gpu</code> is true).</td></tr>
    <tr><td><code>max_iter</code></td><td><code>typemax(Int32)</code></td><td>Maximum number of iterations allowed.</td></tr>
    <tr><td><code>sigma</code></td><td><code>-1 (auto)</code></td><td>Initial value of the σ parameter used in the algorithm.</td></tr>
    <tr><td><code>sigma_fixed</code></td><td><code>false</code></td><td>Whether σ is fixed throughout the optimization process.</td></tr>
    <tr><td><code>check_iter</code></td><td><code>100</code></td><td>Number of iterations to check residuals.</td></tr>
    <tr><td><code>use_Ruiz_scaling</code></td><td><code>true</code></td><td>Whether to apply Ruiz scaling.</td></tr>
    <tr><td><code>use_Pock_Chambolle_scaling</code></td><td><code>true</code></td><td>Whether to use the Pock-Chambolle scaling.</td></tr>
    <tr><td><code>use_l2_scaling</code></td><td><code>true</code></td><td>Whether to use the Pock-Chambolle scaling.</td></tr>
    <tr><td><code>use_bc_scaling</code></td><td><code>true</code></td><td>Whether to use the scaling for b and c. (For QAP and LASSO, only this scaling is applicable)</td></tr>
    <tr><td><code>print_frequency</code></td><td><code>-1 (auto)</code></td><td>Print the log every <code>print_frequency</code> iterations.</td></tr>
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

## Documentation

Comprehensive documentation is available in the `docs/` directory:

📚 **[View Full Documentation](docs/README.md)**

### Quick Links

- **[Getting Started](docs/src/getting_started.md)** - Installation and first examples
- **[User Guide](docs/src/guide/input_overview.md)** - Detailed usage guides
  - [Direct API](docs/src/guide/direct_api.md) - Matrix-based input
  - [JuMP Integration](docs/src/guide/jump_integration.md) - Modeling interface
  - [MPS Files](docs/src/guide/mps_files.md) - File-based input
  - [Q Operators](docs/src/guide/q_operators_overview.md) - QAP, LASSO, and custom operators
  - [Parameters](docs/src/guide/parameters.md) - Solver configuration
  - [Output & Results](docs/src/guide/output_results.md) - Interpreting results
- **[API Reference](docs/src/api.md)** - Complete API documentation
- **[Examples](docs/src/examples.md)** - Collection of examples

### Building Documentation Locally

To build the HTML documentation:

```bash
cd docs
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project make.jl
```

The generated documentation will be in `docs/build/`.

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
