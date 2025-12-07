# HPR-QP TODO (short)

This file summarizes the current status and next high-priority work items for the HPR-QP repository.

Last updated: 2025-12-05

## Completed (high-impact)
- Unified `Qmap!` interface for sparse matrices and operator Q (LASSO/QAP). ✅
- Replaced many specialized kernels with 5 unified GPU kernels; added separate `spmv_mode_Q` and `spmv_mode_A`. ✅
- Benchmarking now uses the real unified update functions and handles operator cases (LASSO no-A, QAP with A). ✅
- Add concise auto-save (HDF5) support for best-so-far solutions (opt-in via params), including w variables. ✅
- CUSPARSE preprocessing and buffer size allocation for A, AT, and Q matrices (when Q is sparse). ✅

## Next priorities (short list)
1. Unify operator-based updates (remove `problem_type` branching).
   - Design and add a small projection interface (per-operator `project_x!`) or equivalent.
2. Add optional warm-start support (initial_x / initial_y) and wire into workspace allocation.

## Nice-to-have / future
- Complete Documenter.jl docs and examples.
- Consider deprecating the `noC` (no box constraints) special-case code if unused.
- Small performance kernels and profiling on large problems.
- Gradually migrate kernel functions to use preprocessed CUSPARSE operations.

## Contribution notes (short)
- Follow Conventional Commits for changes to `src/` (e.g. `feat:`, `fix:`, `perf:`).
- Keep commits focused and small; prefer one feature per PR.
- Exclude backup files (e.g. `utils.jl.backup`) from commits.

If you want, I can commit this simplified TODO for you (or leave it unstaged).
