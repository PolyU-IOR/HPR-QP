# Parallel HPR-QP / HPR-QCQP (C/CUDA)

A CUDA/C implementation of the Halpern Peaceman--Rachford (HPR) method for
quadratic programming and the supported diagonal convex QCQP input format.
This HPR-QP branch imports the September handoff's C/CUDA source snapshot,
identified there as `31afb2ac5c6315fa5348363cd5570e1b4e77c4e4`, and makes its
documented parallel profile the defaults. The original iteration kernels,
scaling routines, controllers and stopping checks are retained. Coupling-moment
spectral estimation includes the conditional numerical repair described below.

## Install

For a fresh setup:

```sh
make
./build-cuda/solve_qp model.mps
```

Install the shared library:

```sh
make install
```

CUDA must be installed for GPU builds. For the reference CUDA 12.8 installation,
use `make CUDA_ROOT=/usr/local/cuda-12.8 GPU_SM=90` and make its runtime libraries
discoverable, for example with `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`.
The original Makefile detects CUDA and builds GPU targets when available.
This source export does not include the separate Julia/Python wrapper packages.

## solve_qp

The executable is `solve_qp`.

GPU example:

```sh
./build-cuda/solve_qp model.mps \
  --device-number 0
```

No profile flags are needed. The native API, executable and dataset launcher
use parallel updates, `moment_selector`, QC majorizer `auto`, adaptive-restart
tau in `[0.025, 40]`, adaptive sigma, both SOCP policies enabled, CR10 followed
by the existing scaling passes, and global b/c normalization off. GPU use is
enabled, tolerance is `1e-8`, checks occur every 100 updates, and the native
time limit is 1000 seconds. Explicit CLI options override these defaults.

Dataset example (the GPU executable is selected by default):

```sh
python3 scripts/run_dataset.py /path/to/mps results/run --device-number 0
```

The dataset summary counts only finite native `OPTIMAL` results satisfying the
requested tolerance. Its native-time SGM10 charges nonoptimal cases the requested
time limit; this is not a complete-process wall-clock metric. Completed native
timeouts remain recorded and are not automatically rerun when resuming a folder.

Usage:

```sh
./build/solve_qp [mps_path] [--method sgs|parallel] [--parallel-majorizer worst_case|coupling_power|two_moment|moment_selector] [--qc-majorizer auto|pairwise|joint_f|scalar_b] [--tau-mode unit|fixed|balanced|adaptive_restart] [--tau X] [--tau-min X] [--tau-max X] [--sigma-fixed 0|1] [--max-iter N] [--stoptol X] [--check-iter N] [--print-frequency N] [--time-limit X] [--verbose 0|1] [--use-gpu 0|1] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1] [--use-cr-scaling 0|1] [--cr-iterations N] [--use-bc-scaling 0|1]
```

CPU example:

```sh
./build/solve_qp model.mps --use-gpu 0
```

CPU solving supports QP; the QCQP solve path requires the GPU executable.

Parameters:

- `mps_path`: input `.mps` file path; defaults to `model.mps`
- `--method sgs|parallel`: choose the original sGS path or the parallel update path
- `--parallel-majorizer worst_case|coupling_power|two_moment|moment_selector`: select the parallel majorizer; `moment_selector` uses the note's second-moment test
- `--qc-majorizer auto|pairwise|joint_f|scalar_b`: select the QC block majorizer
- `--tau-mode unit|fixed|balanced|adaptive_restart`: select the parallel metric rule
- `--tau`, `--tau-min`, `--tau-max`: set the fixed value or adaptive bounds for `tau`
- `--sigma-fixed 0|1`: disable or enable sigma adaptation
- `--max-iter N`: maximum iteration count
- `--stoptol X`: stopping tolerance
- `--check-iter N`: interval for residual and restart checks
- `--print-frequency N`: iteration log interval; use `-1` for adaptive printing
- `--time-limit X`: time limit in seconds
- `--verbose 0|1`: enable or disable solver logging
- `--use-gpu 0|1`: enable or disable GPU solving
- `--device-number N`: CUDA device index when GPU solving is enabled
- `--spmv-mode-q auto|cusparse|customized`: SpMV backend for `Q`
- `--spmv-mode-a auto|cusparse|customized`: SpMV backend for `A` and `A^T`
- `--use-scaling 0|1`: enable or disable problem scaling
- `--use-cr-scaling 0|1`: apply A-only Curtis--Reid scaling before the existing scaling passes
- `--cr-iterations N`: number of Curtis--Reid alternating row/column passes; the validated setting is `10`
- `--use-bc-scaling 0|1`: enable global b/c normalization after the existing scaling passes

The default profile can also be written explicitly:

```sh
./build-cuda/solve_qp INSTANCE.mps \
  --use-gpu 1 --device-number 0 --method parallel \
  --parallel-majorizer moment_selector --qc-majorizer auto \
  --tau-mode adaptive_restart --tau-min 0.025 --tau-max 40 \
  --sigma-fixed 0 --use-socp-restart-policy 1 \
  --use-socp-sigma-policy 1 --check-iter 100 \
  --use-scaling 1 --use-cr-scaling 1 --cr-iterations 10 \
  --use-bc-scaling 0 --stoptol 1e-8 --time-limit 1000
```

This profile applies `CR -> Ruiz -> PC` with global b/c normalization off.
It retains the existing double-precision spectral estimation and metric/control
implementation. `moment_selector` uses the second-moment test on eligible QP
paths; QC majorizer `auto` retains the original QCQP selection rule. The other
majorizers and legacy sGS method remain explicitly selectable. No CUDA
fast-math option is used.

The moment estimator normally retains the original power-iteration probe. If
the first coupling moment is annihilated by that probe, or its absolute stopping
floor masks a failed relative convergence test, it restarts once with a fixed
nonsmooth probe and recomputes the moment pair with relative stopping checks.
This prevents a spurious near-zero coupling estimate from invalidating the
parallel metric on structured problems such as QPLIB_8500. It is automatic on
CPU and GPU and needs no additional option. The normal probe, the `AA^T` and
`Q` estimators, the safety factor, and all solver thresholds remain unchanged.
Power-iteration estimates are still numerical estimates, not certified spectral
upper bounds; this repair does not guarantee convergence within the time limit.

The selectable Curtis--Reid path follows PolyU-IOR/HPR-QP commit
`8023de0426e096d0f540b8126460f2b9d7c1a355`: it performs A-only
log-geometric equilibration, then the existing 10 Ruiz passes and one
Pock--Chambolle pass. CPU and CUDA implementations use `double`, the largest
eigenvalues are still computed by the existing power method, and the build
does not enable fast-math. Curtis--Reid scaling is enabled with ten passes by
default; `--use-cr-scaling 0` selects the path without that preprocessing.

The handoff's historical table used an earlier August source snapshot and a
3600-second cap. This branch's default 1000-second cap does not claim to
reproduce that table. Original Git history was not included in the handoff;
the import and defaults commits use their actual current dates.

Set `HPRQP_PROFILE_SETUP=1` to print opt-in phase timings for MPS parsing and
model construction, CUDA context/upload/allocation, scaling, spectral and
workspace setup, kernel selection, the solver loop, and cleanup.  The
profiling probes are disabled by default and do not change solver arithmetic.

For large parallel problems with diagonal `Q` and average `A` row and column
lengths no greater than one warp, the solver selects the exact fused row path
analytically.  For each warp-aligned thread count in `{128, 256, 512}`, it
queries CUDA occupancy and maximizes
`threads * active_blocks_per_SM * SM_count`, breaking ties toward the larger
thread block.  Thus the cooperative grid adapts to the active GPU (including
A100, H100, B200, and MIG configurations) without timed trial launches.  Other
matrix structures retain the measured fallback selector.

Output fields:

- `status`
- `iter`
- `time`
- `objective`
- `residual`
- `gap`
- `lambda_max_A` (parallel method)
- `lambda_max_Q` (parallel method)
- `lambda_max_AQA` (parallel method)
- `lambda_max_AQ2A` (parallel method)
- `zeta_AQ` (parallel method)
- `tau` (parallel method)
- `d_y` (parallel method)
- `d_w` (parallel method)
- `iter_4`
- `time_4`
- `iter_6`
- `time_6`
