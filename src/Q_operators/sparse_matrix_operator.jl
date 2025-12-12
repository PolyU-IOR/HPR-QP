# ============================================================================
# Sparse Matrix Q Operator (Standard QP)
# ============================================================================
#
# Standard quadratic programming with explicit Q matrix
# Q is represented as a sparse matrix (SparseMatrixCSC on CPU, CuSparseMatrixCSR on GPU)
#
# ============================================================================

# For sparse matrices, implement to_gpu conversion
# Handle both Int and Int32 indices
function to_gpu(Q::SparseMatrixCSC{Float64,Int32})
    return CuSparseMatrixCSR(Q)
end

function to_gpu(Q::SparseMatrixCSC{Float64,Int})
    return CuSparseMatrixCSR(Q)
end

get_operator_name(::Type{CuSparseMatrixCSR{Float64,Int32}}) = "SparseMatrix"
get_problem_size(Q::CuSparseMatrixCSR{Float64,Int32}) = size(Q, 1)
get_problem_size(Q::SparseMatrixCSC{Float64,Int}) = size(Q, 1)
get_problem_size(Q::SparseMatrixCSC{Float64,Int32}) = size(Q, 1)

# Q operator mapping for sparse matrix Q (GPU)
# With optional preprocessed CUSPARSE structure
@inline function Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, Q::CuSparseMatrixCSR{Float64,Int32}, 
                      spmv_Q::Union{Nothing, CUSPARSE_spmv_Q}=nothing)
    if spmv_Q !== nothing
        # Use preprocessed CUSPARSE SpMV
        desc_x = CUDA.CUSPARSE.CuDenseVectorDescriptor(x)
        desc_Qx = CUDA.CUSPARSE.CuDenseVectorDescriptor(Qx)
        CUDA.CUSPARSE.cusparseSpMV(spmv_Q.handle, spmv_Q.operator, spmv_Q.alpha,
            spmv_Q.desc_Q, desc_x, spmv_Q.beta, desc_Qx,
            spmv_Q.compute_type, spmv_Q.alg, spmv_Q.buf)
    else
        # Fallback to standard CUSPARSE
        CUDA.CUSPARSE.mv!('N', 1, Q, x, 0, Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
end

# Q operator mapping for sparse matrix Q (CPU)
@inline function Qmap!(x::Vector{Float64}, Qx::Vector{Float64}, Q::SparseMatrixCSC{Float64,Int32})
    mul!(Qx, Q, x)
end

@inline function Qmap!(x::Vector{Float64}, Qx::Vector{Float64}, Q::SparseMatrixCSC{Float64,Int})
    mul!(Qx, Q, x)
end

# ============================================================================
# CUSPARSE Preprocessing for Sparse Matrix Q
# ============================================================================
supports_cusparse_preprocessing(Q::CuSparseMatrixCSR{Float64,Int32}) = Q.nnz > 0

"""
    prepare_spmv_Q!(Q, w, w_bar, w_hat, Qw)

Prepare CUSPARSE SpMV operations for Q matrix (when Q is a sparse matrix, not an operator).
Allocates buffers and performs preprocessing (for CUDA >= 12.4).

# Arguments
- `Q::CuSparseMatrixCSR`: The Q matrix in CSR format
- `w, w_bar, w_hat::CuVector{Float64}`: Dense vectors for Q operations
- `Qw::CuVector{Float64}`: Output vector for Q*w

# Returns
- `spmv_Q::CUSPARSE_spmv_Q`: CUSPARSE structure for Q matrix operations
"""
function prepare_spmv_Q!(Q::CuSparseMatrixCSR{Float64,Int32}, 
                        w::CuVector{Float64},
                        Qw::CuVector{Float64})
    # Create matrix and vector descriptors
    desc_Q = CUDA.CUSPARSE.CuSparseMatrixDescriptor(Q, 'O')
    desc_w = CUDA.CUSPARSE.CuDenseVectorDescriptor(w)
    desc_Qw = CUDA.CUSPARSE.CuDenseVectorDescriptor(Qw)
    
    CUSPARSE_handle = CUDA.CUSPARSE.handle()
    ref_one = Ref{Float64}(one(Float64))
    ref_zero = Ref{Float64}(zero(Float64))
    
    # Prepare Q SpMV
    sz_Q = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_Q, desc_w, ref_zero,
        desc_Qw, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_Q)
    buf_Q = CUDA.CuArray{UInt8}(undef, sz_Q[])
    
    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_Q, desc_w, ref_zero, desc_Qw,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_Q)
    end
    
    spmv_Q = CUSPARSE_spmv_Q(CUSPARSE_handle, 'N', ref_one, desc_Q, desc_w,
        ref_zero, desc_Qw, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_Q)
    
    return spmv_Q
end
