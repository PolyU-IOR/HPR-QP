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
