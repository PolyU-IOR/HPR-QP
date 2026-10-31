# Contributing to HPR-QP-C

This document describes the expected workflow for contributions to HPR-QP-C. The project includes a core C/CUDA solver, a C binding API, and Julia and Python wrappers, so changes should keep the relevant build, test, and documentation paths in sync.

## Getting Started

1. Fork the repository if you are contributing through GitHub.
2. Clone your fork or local checkout:

   ```sh
   git clone https://github.com/YOUR_USERNAME/HPR-QP-C.git
   cd HPR-QP-C
   ```

3. Create a focused branch for the change:

   ```sh
   git checkout -b feature/short-description
   ```

## Development Setup

### Prerequisites

- CMake 3.16+
- A C11 compiler and a C++17 compiler
- Optional: a CUDA toolkit with `nvcc` in `PATH` or exposed through `CUDA_ROOT` for GPU builds
- Optional: Julia for the Julia wrapper and tests
- Optional: Python 3 for the Python wrapper and tests

### Core Build

For a normal local build, use the top-level Makefile:

```sh
make clean && make
```

This auto-detects whether CUDA is available and builds the matching CPU-only or CPU+GPU targets.

Useful explicit targets:

```sh
make build
make test
make shared
make install
make gpu-build
make gpu-test
```

### Wrapper Setup

Julia wrapper:

```sh
cd julia
make setup
make test
```

Python wrapper:

```sh
cd python
make setup
make test
```

## Testing Expectations

Run the tests that cover the area you changed before opening a pull request.

- Core solver or shared library changes: `make test`
- CUDA-specific changes: `make gpu-test`
- Julia wrapper changes: `cd julia && make test`
- Python wrapper changes: `cd python && make test`
- CLI or example changes: run the relevant demo or `solve_qp` command directly

If your change affects both the core solver and a wrapper, test both layers.

## Reporting Bugs

When filing a bug report, include:

- A clear summary of the problem
- Steps to reproduce it
- The expected behavior and the actual behavior
- Your platform details, including compiler and CUDA information when relevant
- The exact command you ran and any error output
- A small `.mps` or array-based reproducer when possible

## Pull Requests

Before opening a pull request:

1. Keep the change focused and avoid unrelated refactors.
2. Add or update tests when behavior changes.
3. Update user-facing documentation when APIs, build steps, or examples change.
4. Keep C, Julia, and Python surfaces aligned when the shared binding or solver behavior changes.
5. Use clear commit messages.

Example commit prefixes:

```text
feat: add GPU control regression for diagonal Q
fix: correct binding result ownership on error
docs: update python wrapper setup instructions
test: add solver regression for unchecked iterations
refactor: split gpu eigs setup from control path
```

## Code Style

- Follow the existing style in the surrounding code instead of introducing a new one.
- Prefer small, single-purpose functions and keep public API changes deliberate.
- Add comments only where the control flow or math would otherwise be hard to follow.
- Keep naming consistent across the solver, binding API, and wrapper layers.
- Avoid changing generated build artifacts or unrelated formatting in the same pull request.

## Documentation

- Update [README.md](README.md) when top-level build, install, or CLI usage changes.
- Update [julia/README.md](julia/README.md) when the Julia wrapper surface changes.
- Update [python/README.md](python/README.md) when the Python wrapper surface changes.
- Update [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) when the port status or major milestones change.

## License

By contributing to HPR-QP-C, you agree that your contributions will be licensed under the MIT License used by this repository.