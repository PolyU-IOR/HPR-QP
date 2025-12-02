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

# Q operator mapping for sparse matrix Q (standard case)
@inline function Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, Q::CuSparseMatrixCSR{Float64,Int32})
    CUDA.CUSPARSE.mv!('N', 1, Q, x, 0, Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end
