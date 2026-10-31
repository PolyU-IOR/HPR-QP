# HPR-QP-C

A CUDA/C implementation of the Halpern Peaceman--Rachford (HPR) method for solving quadratic programming (QP) problems on GPUs.

## Install

For a fresh setup:

```sh
make clean && make
```

Install the shared library:

```sh
make install
```

Wrapper documentation:

- [Julia wrapper API](julia/README.md)
- [Python wrapper API](python/README.md)

## solve_qp

The executable is `solve_qp`.

GPU example:

```sh
./build-cuda/solve_qp model.mps \
  --method parallel \
  --parallel-majorizer coupling_power \
  --tau-mode adaptive_restart \
  --use-cr-scaling 1 \
  --cr-iterations 10 \
  --use-gpu 1 \
  --device-number 0 \
  --stoptol 1e-8 \
  --check-iter 100 \
  --time-limit 1000
```

Usage:

```sh
./build/solve_qp [mps_path] [--method sgs|parallel] [--parallel-majorizer worst_case|coupling_power|two_moment|moment_selector] [--qc-majorizer auto|pairwise|joint_f|scalar_b] [--tau-mode unit|fixed|balanced|adaptive_restart] [--tau X] [--tau-min X] [--tau-max X] [--sigma-fixed 0|1] [--max-iter N] [--stoptol X] [--check-iter N] [--print-frequency N] [--time-limit X] [--verbose 0|1] [--use-gpu 0|1] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1] [--use-cr-scaling 0|1] [--cr-iterations N] [--use-bc-scaling 0|1]
```

CPU example:

```sh
./build/solve_qp model.mps --stoptol 1e-6 --time-limit 3600
```

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
- `--use-bc-scaling 0|1`: enable the final Pock--Chambolle pass

The note-aligned QCQP profile is selected explicitly so legacy defaults remain
unchanged:

```sh
./build-cuda/solve_qp INSTANCE.mps \
  --use-gpu 1 --method parallel \
  --parallel-majorizer coupling_power --qc-majorizer pairwise \
  --tau-mode adaptive_restart --tau-min 0.025 --tau-max 40 \
  --sigma-fixed 0 --use-socp-restart-policy 1 \
  --use-socp-sigma-policy 0 --check-iter 150 \
  --use-scaling 1 --use-cr-scaling 1 --cr-iterations 10 \
  --use-bc-scaling 1 --stoptol 1e-8
```

This profile applies `CR -> Ruiz -> PC`. It estimates
`lambda_max(A Q A^T)` by the existing double-precision power method, updates
the three pairwise QCQP shapes only at accepted restarts, refreshes sigma in
the new metric with the one-decade safeguard, and reseeds restart history after
every metric change. The optional `two_moment` and `moment_selector` modes also
estimate `lambda_max(A Q^2 A^T)`. No CUDA fast-math option is used.

The selectable Curtis--Reid path follows PolyU-IOR/HPR-QP commit
`8023de0426e096d0f540b8126460f2b9d7c1a355`: it performs A-only
log-geometric equilibration, then the existing 10 Ruiz passes and one
Pock--Chambolle pass. CPU and CUDA implementations use `double`, the largest
eigenvalues are still computed by the existing power method, and the build
does not enable fast-math. Curtis--Reid scaling is off by default so the
pre-existing path remains selectable; pass the two flags above to enable the
validated CR10 configuration.

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
