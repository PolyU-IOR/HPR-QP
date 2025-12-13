# ============================================================================
# Unified Operations for CPU and GPU
# ============================================================================
#
# This file provides unified operation interfaces that dispatch based on array types.
# Pattern follows the Q operator interface (Qmap!) where the same function name
# dispatches to different implementations based on GPU/CPU types.
#
# Benefits:
# - Algorithm code calls one function (e.g., unified_dot)
# - Julia's multiple dispatch selects the right implementation
# - Each implementation is optimized for its device
# - Zero-cost abstraction (no runtime overhead)
#
# ============================================================================

using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE

# ============================================================================
# Dot Product Operations
# ============================================================================

"""
    unified_dot(x, y)

Compute dot product of two vectors, dispatching to appropriate implementation
based on vector types (CPU Vector or GPU CuVector).

# Examples
```julia
# CPU version
x_cpu = [1.0, 2.0, 3.0]
y_cpu = [4.0, 5.0, 6.0]
result = unified_dot(x_cpu, y_cpu)  # Uses LinearAlgebra.dot

# GPU version  
x_gpu = CuVector([1.0, 2.0, 3.0])
y_gpu = CuVector([4.0, 5.0, 6.0])
result = unified_dot(x_gpu, y_gpu)  # Uses CUDA.dot
```
"""
@inline unified_dot(x::Vector{T}, y::Vector{T}) where {T} = dot(x, y)
@inline unified_dot(x::CuVector{T}, y::CuVector{T}) where {T} = CUDA.dot(x, y)

# ============================================================================
# Norm Operations
# ============================================================================

"""
    unified_norm(x, p=2)

Compute p-norm of a vector, dispatching to appropriate implementation
based on vector type (CPU Vector or GPU CuVector).

# Arguments
- `x`: Vector (CPU or GPU)
- `p`: Norm type (default: 2 for Euclidean norm)

# Examples
```julia
# CPU version
x_cpu = [3.0, 4.0]
result = unified_norm(x_cpu)  # Uses LinearAlgebra.norm

# GPU version
x_gpu = CuVector([3.0, 4.0])
result = unified_norm(x_gpu)  # Uses CUDA.norm
```
"""
@inline unified_norm(x::Vector{T}, p::Real=2) where {T} = norm(x, p)
@inline unified_norm(x::CuVector{T}, p::Real=2) where {T} = CUDA.norm(x, p)

# ============================================================================
# Matrix-Vector Multiplication
# ============================================================================

"""
    unified_mul!(y, A, x)
    unified_mul!(y, A, x, spmv_A)
    unified_mul!(x, AT, y, spmv_AT)

Compute matrix-vector product y = A * x in-place, dispatching to appropriate
implementation based on matrix/vector types.

For CPU: Uses LinearAlgebra.mul!
For GPU: Uses CUDA.CUSPARSE.mv! with optimized CSR algorithm
For GPU with preprocessed structures: Uses CUSPARSE.cusparseSpMV with 
    preprocessed matrix descriptors and buffers for better performance

# Arguments
- `y`: Output vector (modified in-place)
- `A`: Sparse matrix (CSC for CPU, CSR for GPU)
- `x`: Input vector
- `spmv_A` (optional): Preprocessed CUSPARSE_spmv_A structure for A operations
- `spmv_AT` (optional): Preprocessed CUSPARSE_spmv_AT structure for AT operations

# Examples
```julia
# CPU version
y_cpu = zeros(m)
unified_mul!(y_cpu, A_cpu, x_cpu)  # Uses mul!

# GPU version (without preprocessing)
y_gpu = CUDA.zeros(m)
unified_mul!(y_gpu, A_gpu, x_gpu)  # Uses CUSPARSE.mv!

# GPU version (with preprocessing)
ws = allocate_workspace(...)
prepare_workspace_spmv!(ws, qp)
unified_mul!(y_gpu, A_gpu, x_gpu, ws.spmv_A)  # Uses preprocessed structures
unified_mul!(x_gpu, AT_gpu, y_gpu, ws.spmv_AT)  # Uses preprocessed structures
```
"""
@inline function unified_mul!(y::Vector{T}, A::SparseMatrixCSC{T,I}, x::Vector{T}) where {T,I}
    mul!(y, A, x)
end

@inline function unified_mul!(y::CuVector{T}, A::CuSparseMatrixCSR{T,I}, x::CuVector{T}) where {T,I}
    CUDA.CUSPARSE.mv!('N', one(T), A, x, zero(T), y, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

# Overload with preprocessed CUSPARSE structure for A * x
@inline function unified_mul!(y::CuVector{T}, A::CuSparseMatrixCSR{T,I}, x::CuVector{T}, spmv::CUSPARSE_spmv_A) where {T,I}
    # Use preprocessed matrix descriptor and buffer, but create descriptors for current x and y
    desc_x = CUDA.CUSPARSE.CuDenseVectorDescriptor(x)
    desc_y = CUDA.CUSPARSE.CuDenseVectorDescriptor(y)
    CUDA.CUSPARSE.cusparseSpMV(spmv.handle, spmv.operator, spmv.alpha,
        spmv.desc_A, desc_x, spmv.beta, desc_y,
        spmv.compute_type, spmv.alg, spmv.buf)
end

# Overload with preprocessed CUSPARSE structure for AT * y (transpose)
@inline function unified_mul!(x::CuVector{T}, AT::CuSparseMatrixCSR{T,I}, y::CuVector{T}, spmv::CUSPARSE_spmv_AT) where {T,I}
    # Use preprocessed matrix descriptor and buffer, but create descriptors for current y and x
    desc_y = CUDA.CUSPARSE.CuDenseVectorDescriptor(y)
    desc_x = CUDA.CUSPARSE.CuDenseVectorDescriptor(x)
    CUDA.CUSPARSE.cusparseSpMV(spmv.handle, spmv.operator, spmv.alpha,
        spmv.desc_AT, desc_y, spmv.beta, desc_x,
        spmv.compute_type, spmv.alg, spmv.buf)
end

# ============================================================================
# Elementwise Operations
# ============================================================================

"""
    unified_clamp!(x, lb, ub)

Clamp vector x element-wise to be within [lb, ub] bounds, in-place.
Dispatches based on vector type.

# Arguments
- `x`: Vector to clamp (modified in-place)
- `lb`: Lower bound (scalar or vector)
- `ub`: Upper bound (scalar or vector)
"""
@inline function unified_clamp!(x::Vector{T}, lb, ub) where {T}
    clamp!(x, lb, ub)
end

@inline function unified_clamp!(x::CuVector{T}, lb, ub) where {T}
    # CUDA.jl's clamp! works on CuVector
    clamp!(x, lb, ub)
end

# ============================================================================
# Scalar Operations with Vectors
# ============================================================================

"""
    unified_axpy!(a, x, y)

Compute y = a*x + y in-place, dispatching based on vector types.
Uses BLAS axpy! for both CPU and GPU.

# Arguments
- `a`: Scalar coefficient
- `x`: Input vector
- `y`: Output vector (modified in-place)
"""
@inline function unified_axpy!(a::T, x::Vector{T}, y::Vector{T}) where {T}
    BLAS.axpy!(a, x, y)
end

@inline function unified_axpy!(a::T, x::CuVector{T}, y::CuVector{T}) where {T}
    CUDA.CUBLAS.axpy!(length(x), a, x, 1, y, 1)
end

"""
    unified_axpby!(a, x, b, y)

Compute y = a*x + b*y in-place, dispatching based on vector types.
Uses BLAS axpby! for both CPU and GPU.

# Arguments
- `a`: Scalar coefficient for x
- `x`: First input vector
- `b`: Scalar coefficient for y
- `y`: Second input/output vector (modified in-place)
"""
@inline function unified_axpby!(a::T, x::Vector{T}, b::T, y::Vector{T}) where {T}
    BLAS.axpby!(a, x, b, y)
end

@inline function unified_axpby!(a::T, x::CuVector{T}, b::T, y::CuVector{T}) where {T}
    CUDA.CUBLAS.axpby!(length(x), a, x, 1, b, y, 1)
end

# ============================================================================
# Vector Copying
# ============================================================================

"""
    unified_copyto!(dest, src)

Copy src to dest, dispatching based on vector types.

# Arguments
- `dest`: Destination vector (modified in-place)
- `src`: Source vector
"""
@inline function unified_copyto!(dest::Vector{T}, src::Vector{T}) where {T}
    copyto!(dest, src)
end

@inline function unified_copyto!(dest::CuVector{T}, src::CuVector{T}) where {T}
    copyto!(dest, src)
end

# ============================================================================
# Vector Fill Operations
# ============================================================================

"""
    unified_fill!(x, val)

Fill vector x with scalar value val, dispatching based on vector type.

# Arguments
- `x`: Vector to fill (modified in-place)
- `val`: Scalar value to fill with
"""
@inline function unified_fill!(x::Vector{T}, val::T) where {T}
    fill!(x, val)
end

@inline function unified_fill!(x::CuVector{T}, val::T) where {T}
    fill!(x, val)
end

# ============================================================================
# Maximum/Minimum Operations
# ============================================================================

"""
    unified_maximum(x)

Find maximum element in vector x, dispatching based on vector type.

# Arguments
- `x`: Vector (CPU or GPU)
"""
@inline unified_maximum(x::Vector{T}) where {T} = maximum(x)
@inline unified_maximum(x::CuVector{T}) where {T} = maximum(x)

"""
    unified_minimum(x)

Find minimum element in vector x, dispatching based on vector type.

# Arguments
- `x`: Vector (CPU or GPU)
"""
@inline unified_minimum(x::Vector{T}) where {T} = minimum(x)
@inline unified_minimum(x::CuVector{T}) where {T} = minimum(x)

# ============================================================================
# Sum/Mean Operations
# ============================================================================

"""
    unified_sum(x)

Compute sum of elements in vector x, dispatching based on vector type.

# Arguments
- `x`: Vector (CPU or GPU)
"""
@inline unified_sum(x::Vector{T}) where {T} = sum(x)
@inline unified_sum(x::CuVector{T}) where {T} = sum(x)

"""
    unified_mean(x)

Compute mean of elements in vector x, dispatching based on vector type.

# Arguments
- `x`: Vector (CPU or GPU)
"""
@inline unified_mean(x::Vector{T}) where {T} = sum(x) / length(x)
@inline unified_mean(x::CuVector{T}) where {T} = sum(x) / length(x)

# ============================================================================
# Helper Functions for Type Conversion
# ============================================================================

"""
    to_cpu(x)

Transfer data from GPU to CPU if needed, or return as-is if already on CPU.

# Arguments
- `x`: Vector or matrix (CPU or GPU)
"""
@inline to_cpu(x::Vector) = x
@inline to_cpu(x::CuVector) = Vector(x)
@inline to_cpu(x::SparseMatrixCSC) = x
@inline to_cpu(x::CuSparseMatrixCSR) = SparseMatrixCSC(x)
@inline to_cpu(x::T) where {T<:Number} = x  # Scalars pass through

# Note: to_gpu is already defined in the codebase for Q operators
# Following the same pattern here for completeness of documentation

# ============================================================================
# Specialized Algorithm Functions
# ============================================================================

"""
    unified_golden_Q_diag(a, b, Q, c, d, tempv; lo, hi, tol, maxiter)

Golden-section search for minimizing the objective function in sigma update
when Q is diagonal. Dispatches to GPU or CPU implementation based on types.

This function is used in the adaptive sigma update for diagonal Q matrices.

# Arguments
- `a`, `b`: Scalar coefficients
- `Q`: Diagonal Q vector (CPU Vector or GPU CuVector)
- `c`, `d`: Vectors used in objective computation
- `tempv`: Temporary workspace vector
- `lo`, `hi`: Search bounds
- `tol`: Convergence tolerance
- `maxiter`: Maximum iterations
"""
function unified_golden_Q_diag(a::Float64, b::Float64,
    Q::Vector{Float64}, c::Vector{Float64},
    d::Vector{Float64}, tempv::Vector{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using CPU operations
    function f_cpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * dot(c, tempv)
    end

    # Golden section search
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_cpu(x1)
    f2 = f_cpu(x2)

    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_cpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_cpu(x1)
        end
        iter += 1
    end

    return (f1 < f2) ? x1 : x2
end

function unified_golden_Q_diag(a::Float64, b::Float64,
    Q::CuArray{Float64}, c::CuArray{Float64},
    d::CuArray{Float64}, tempv::CuArray{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using GPU operations
    function f_gpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * CUDA.dot(c, tempv)
    end

    # Golden section search
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_gpu(x1)
    f2 = f_gpu(x2)

    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_gpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_gpu(x1)
        end
        iter += 1
    end

    return (f1 < f2) ? x1 : x2
end

"""
    unified_update_Q_factors!(fact2, fact, fact1, fact_M, diag_Q, sigma)

Update Q-related factors for diagonal Q matrices. Used in sigma updates.
Dispatches to GPU kernel or CPU loop based on vector types.

Computes:
- fact2[i] = 1 / (1 + sigma * diag_Q[i])
- fact[i] = sigma * fact2[i]
- fact1[i] = sigma * diag_Q[i] * fact2[i]
- fact_M[i] = sigma^2 * fact2[i]

# Arguments
- `fact2`, `fact`, `fact1`, `fact_M`: Output factor vectors (modified in-place)
- `diag_Q`: Diagonal elements of Q matrix
- `sigma`: Current sigma parameter value
"""
function unified_update_Q_factors!(
    fact2::Vector{Float64},
    fact::Vector{Float64},
    fact1::Vector{Float64},
    fact_M::Vector{Float64},
    diag_Q::Vector{Float64},
    sigma::Float64
)
    s2 = sigma * sigma
    @. fact2 = 1.0 / (1.0 + sigma * diag_Q)
    @. fact = sigma * fact2
    @. fact1 = sigma * diag_Q * fact2
    @. fact_M = s2 * fact2
    return
end

function unified_update_Q_factors!(
    fact2::CuVector{Float64},
    fact::CuVector{Float64},
    fact1::CuVector{Float64},
    fact_M::CuVector{Float64},
    diag_Q::CuVector{Float64},
    sigma::Float64
)
    s2 = sigma * sigma
    @. fact2 = 1.0 / (1.0 + sigma * diag_Q)
    @. fact = sigma * fact2
    @. fact1 = sigma * diag_Q * fact2
    @. fact_M = s2 * fact2
    return
end

# ============================================================================
# Unified Ones Operations (for initialization)
# ============================================================================

"""
    unified_ones(T, n)

Create a vector of ones, dispatching to CPU or GPU based on context.
"""
@inline unified_ones(::Type{Vector{T}}, n::Int) where {T} = ones(T, n)
@inline unified_ones(::Type{CuVector{T}}, n::Int) where {T} = CUDA.ones(T, n)

# Helper to get the vector type from an existing vector
@inline unified_ones_like(x::Vector{T}) where {T} = ones(T, length(x))
@inline unified_ones_like(x::CuVector{T}) where {T} = CUDA.ones(T, length(x))

# ============================================================================
# Unified Zeros Operations (for initialization)
# ============================================================================

"""
    unified_zeros(T, n)

Create a vector of zeros, dispatching to CPU or GPU based on context.
"""
@inline unified_zeros(::Type{Vector{T}}, n::Int) where {T} = zeros(T, n)
@inline unified_zeros(::Type{CuVector{T}}, n::Int) where {T} = CUDA.zeros(T, n)

# Helper to get the vector type from an existing vector
@inline unified_zeros_like(x::Vector{T}) where {T} = zeros(T, length(x))
@inline unified_zeros_like(x::CuVector{T}) where {T} = CUDA.zeros(T, length(x))

# ============================================================================
# Notes
# ============================================================================
#
# Design Philosophy:
# ------------------
# This follows the same pattern as the Q operator interface:
#   - Single function name (e.g., Qmap!, unified_dot)
#   - Multiple dispatch on types
#   - Device-optimized implementations
#   - Zero runtime overhead
#
# Usage in Algorithm Code:
# ------------------------
# Instead of:
#   if use_gpu
#       result = CUDA.dot(x, y)
#   else
#       result = dot(x, y)
#   end
#
# Simply write:
#   result = unified_dot(x, y)
#
# Julia automatically dispatches to the correct implementation based on types.
#
# Adding New Operations:
# ----------------------
# To add a new unified operation:
# 1. Define CPU version: @inline unified_op(x::Vector, ...) = ...
# 2. Define GPU version: @inline unified_op(x::CuVector, ...) = ...
# 3. Add docstring with examples
# 4. Algorithm code can call unified_op without knowing device
#
# ============================================================================

# ============================================================================
# Workspace Allocation Helpers
# ============================================================================

"""
    allocate_vector(::Type{<:HPRQP_workspace_cpu}, T, n)
    allocate_vector(::Type{<:HPRQP_workspace_gpu}, T, n)

Allocate a zero-initialized vector of length n, dispatching to CPU or GPU
based on workspace type.

# Examples
```julia
# CPU allocation
v = allocate_vector(HPRQP_workspace_cpu, Float64, 100)  # Returns Vector{Float64}

# GPU allocation
v = allocate_vector(HPRQP_workspace_gpu, Float64, 100)  # Returns CuVector{Float64}
```
"""
@inline allocate_vector(::Type{HPRQP_workspace_cpu}, ::Type{T}, n::Int) where {T} = zeros(T, n)
@inline allocate_vector(::Type{HPRQP_workspace_gpu}, ::Type{T}, n::Int) where {T} = CUDA.zeros(T, n)

"""
    allocate_saved_state(::Type{<:HPRQP_workspace_cpu})
    allocate_saved_state(::Type{<:HPRQP_workspace_gpu})

Allocate appropriate saved state struct based on workspace type.
"""
@inline allocate_saved_state(::Type{HPRQP_workspace_cpu}) = HPRQP_saved_state_cpu()
@inline allocate_saved_state(::Type{HPRQP_workspace_gpu}) = HPRQP_saved_state_gpu()

"""
    convert_to_device(::Type{<:HPRQP_workspace_cpu}, v)
    convert_to_device(::Type{<:HPRQP_workspace_gpu}, v)

Convert a vector to appropriate device array based on workspace type.
For CPU, returns the vector unchanged. For GPU, converts to CuVector.
"""
@inline convert_to_device(::Type{HPRQP_workspace_cpu}, v::Vector{T}) where {T} = v
@inline convert_to_device(::Type{HPRQP_workspace_gpu}, v::Vector{T}) where {T} = CuVector(v)
@inline convert_to_device(::Type{HPRQP_workspace_gpu}, v::CuVector{T}) where {T} = v
@inline convert_to_device(::Type{HPRQP_workspace_cpu}, v::CuVector{T}) where {T} = Vector(v)

"""
    fill_vector(::Type{<:HPRQP_workspace_cpu}, value, n)
    fill_vector(::Type{<:HPRQP_workspace_gpu}, value, n)

Create a vector filled with a specific value, dispatching to CPU or GPU
based on workspace type.
"""
@inline fill_vector(::Type{HPRQP_workspace_cpu}, value::T, n::Int) where {T} = fill(value, n)
@inline fill_vector(::Type{HPRQP_workspace_gpu}, value::T, n::Int) where {T} = CUDA.fill(value, n)

# ============================================================================
# CUSPARSE Preprocessing Helpers
# ============================================================================

"""
    prepare_workspace_spmv!(ws::HPRQP_workspace_cpu, qp)

No-op for CPU workspace - CUSPARSE preprocessing not needed.
"""
function prepare_workspace_spmv!(ws::HPRQP_workspace_cpu, qp::QP_info_cpu, verbose::Bool=false)
    # CPU doesn't need CUSPARSE preprocessing
    return nothing
end

"""
    prepare_workspace_spmv!(ws::HPRQP_workspace_gpu, qp)

Prepare CUSPARSE SpMV structures for GPU workspace.
Handles preprocessing for A, AT, and Q matrices.
"""
function prepare_workspace_spmv!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, verbose::Bool=false)
    m = ws.m
    n = ws.n

    # Prepare CUSPARSE SpMV structures for A and AT (if m > 0)
    if m > 0
        ws.spmv_A, ws.spmv_AT = prepare_spmv_A!(qp.A, qp.AT, ws.x_bar, ws.x_hat, ws.dx, ws.Ax,
            ws.y_bar, ws.y, ws.ATy)
        if verbose
            println("Preprocess CUSPARSE SpMV structures for A and AT.")
        end
    else
        ws.spmv_A = nothing
        ws.spmv_AT = nothing
    end


    # Prepare CUSPARSE SpMV structure for Q 
    ws.spmv_Q = nothing
    if supports_cusparse_preprocessing(qp.Q)
        # Operator supports preprocessing - call its prepare function
        # This stores CUSPARSE structures inside the operator itself
        ws.spmv_Q = prepare_spmv_Q!(qp.Q, ws.w, ws.Qw)
        if verbose
            println("Preprocess CUSPARSE SpMV structure for Q.")
        end
    end

    return nothing
end

# ============================================================================
# Problem Info Helper Functions
# ============================================================================

"""
    get_Q_nnz(Q)

Get the number of non-zero elements in Q matrix.
Dispatches based on Q type (SparseMatrixCSC for CPU, CuSparseMatrixCSR for GPU).
"""
get_Q_nnz(Q::SparseMatrixCSC) = nnz(Q)
get_Q_nnz(Q::CuSparseMatrixCSR) = length(Q.nzVal)

"""
    get_A_nnz(A)

Get the number of non-zero elements in A matrix.
Dispatches based on A type (SparseMatrixCSC for CPU, CuSparseMatrixCSR for GPU).
"""
get_A_nnz(A::SparseMatrixCSC) = nnz(A)
get_A_nnz(A::CuSparseMatrixCSR) = length(A.nzVal)

"""
    is_q_operator(Q)

Check if Q is an abstract Q operator (LASSO, QAP, or custom).
Returns true for AbstractQOperator (GPU) or AbstractQOperatorCPU (CPU).
"""
is_q_operator(Q::AbstractQOperator) = true
is_q_operator(Q::AbstractQOperatorCPU) = true
is_q_operator(Q::Union{SparseMatrixCSC, CuSparseMatrixCSR}) = false

"""
    get_Q_nzvals(Q)

Get the array of non-zero values in Q matrix.
Dispatches based on Q type (SparseMatrixCSC for CPU, CuSparseMatrixCSR for GPU).
"""
get_Q_nzvals(Q::SparseMatrixCSC) = Q.nzval
get_Q_nzvals(Q::CuSparseMatrixCSR) = Q.nzVal

"""
    compute_lambda_max_Q(Q, ws, eig_factor, Q_is_diag)

Compute the maximum eigenvalue of Q matrix.
Dispatches based on Q type (AbstractQOperator, SparseMatrixCSC, or CuSparseMatrixCSR).

For Q operators: Uses power iteration with eigenvalue factor.
For sparse matrices: 
  - If Q is diagonal: Returns maximum of diagonal values
  - If Q is non-diagonal: Uses power iteration with eigenvalue factor
  - If Q is empty: Returns 0.0
"""
function compute_lambda_max_Q(Q::Union{AbstractQOperator, AbstractQOperatorCPU}, ws, eig_factor)
    return power_iteration_Q(ws) * eig_factor
end

function compute_lambda_max_Q(Q::Union{CuSparseMatrixCSR, SparseMatrixCSC}, ws, eig_factor)
    nz_vals = get_Q_nzvals(Q)
    if length(nz_vals) > 0
        if !ws.Q_is_diag
            return power_iteration_Q(ws) * eig_factor
        else
            return maximum(nz_vals)
        end
    else
        return 0.0
    end
end

# Helper function to get workspace type from qp type
workspace_type(::QP_info_cpu) = HPRQP_workspace_cpu
workspace_type(::QP_info_gpu) = HPRQP_workspace_gpu