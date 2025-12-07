# HPRQP Documentation

This directory contains the documentation for HPRQP.jl built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

## Building the Documentation Locally

To build the documentation on your local machine:

```bash
# From the root directory of HPR-QP
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The generated documentation will be in `docs/build/`.

To view it:
```bash
# Linux/Mac
open docs/build/index.html

# Or use Python's simple HTTP server
cd docs/build
python3 -m http.server 8000
# Then open http://localhost:8000 in your browser
```

## Documentation Structure

- `src/index.md` - Homepage and overview
- `src/getting_started.md` - Installation and first steps
- `src/guide/` - User guides for different interfaces and Q operators
  - **Input Methods:**
    - `input_overview.md` - Overview of input methods
    - `direct_api.md` - Using the direct matrix API
    - `jump_integration.md` - JuMP integration
    - `mps_files.md` - Solving MPS files
  - **Q Operators:**
    - `q_operators_overview.md` - Understanding Q operators
    - `sparse_matrix_qp.md` - Sparse matrix Q operator
    - `lasso_problems.md` - LASSO operator for L1-regularized least squares
    - `qap_problems.md` - QAP operator for quadratic assignment problems
  - `parameters.md` - Solver parameters
  - `output_results.md` - Understanding results
- `src/api.md` - API reference
- `src/examples.md` - Complete examples

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when you:
- Push to `main` or `master` branch
- Create a new tag/release

The deployed documentation will be available at:
https://PolyU-IOR.github.io/HPR-QP/

## Adding New Pages

1. Create a new `.md` file in `src/` or `src/guide/`
2. Add it to the `pages` list in `make.jl`
3. Rebuild the documentation

## Writing Documentation

Documenter.jl supports:
- Markdown syntax
- LaTeX math with `$` and `$$`
- Julia code blocks with syntax highlighting
- Cross-references
- Docstring integration with `@docs`

For more information, see the [Documenter.jl documentation](https://documenter.juliadocs.org/).
