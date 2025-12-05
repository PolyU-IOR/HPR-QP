# ============================================================================
# LASSO Q Operator
# ============================================================================
#
# Objective: min 0.5*||Ax - b||² + lambda*||x||₁
# Reformulated as QP: min 0.5*x'Qx + c'x + lambda*||x||₁
# Q operator: Q(x) = A'*(A*x) where A is the data matrix
# Note: lambda is stored in QP_info_cpu.lambda, not in the operator
#
# ============================================================================

# ============================================================================
# CUSPARSE Structures for LASSO Operator
# ============================================================================

# CUSPARSE SpMV structure for LASSO operator's internal A matrix
mutable struct CUSPARSE_spmv_LASSO_A
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_A::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_x::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_temp::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# CUSPARSE SpMV structure for LASSO operator's internal AT matrix
mutable struct CUSPARSE_spmv_LASSO_AT
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_AT::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_temp::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_Qx::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# ============================================================================
# LASSO Operator Structures
# ============================================================================

# CPU version: stores problem data on CPU
# Note: lambda is stored in QP_info_cpu.lambda, not here
struct LASSO_Q_operator_cpu <: AbstractQOperatorCPU
    A::SparseMatrixCSC{Float64,Int}
end

# GPU version: stores problem data on GPU
# Q(x) = A'*(A*x) where A is the data matrix
mutable struct LASSO_Q_operator_gpu <: AbstractQOperator
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    temp::CuVector{Float64}  # Temporary storage for A*x
    # Optional CUSPARSE preprocessing structures
    spmv_A::Union{CUSPARSE_spmv_LASSO_A, Nothing}
    spmv_AT::Union{CUSPARSE_spmv_LASSO_AT, Nothing}
end

# Interface implementations for LASSO
get_temp_size(Q::LASSO_Q_operator_cpu) = size(Q.A, 1)  # m rows
get_operator_name(::Type{LASSO_Q_operator_gpu}) = "LASSO"
get_problem_size(Q::LASSO_Q_operator_cpu) = size(Q.A, 2)  # n columns
get_problem_size(Q::LASSO_Q_operator_gpu) = size(Q.A, 2)  # n columns

function to_gpu(Q_cpu::LASSO_Q_operator_cpu)
    m = size(Q_cpu.A, 1)
    return LASSO_Q_operator_gpu(
        CuSparseMatrixCSR(Q_cpu.A),
        CuSparseMatrixCSR(Q_cpu.A'),
        CUDA.zeros(Float64, m),
        nothing,  # spmv_A will be initialized by prepare_operator_spmv!
        nothing   # spmv_AT will be initialized by prepare_operator_spmv!
    )
end

# Q operator mapping for LASSO problem: Q(x) = A'*(A*x)
@inline function Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, Q::LASSO_Q_operator_gpu)
    # Use preprocessed CUSPARSE if available, otherwise fallback to standard
    if Q.spmv_A !== nothing
        # Preprocessed path: A * x -> temp
        desc_x = CUDA.CUSPARSE.CuDenseVectorDescriptor(x)
        desc_temp = CUDA.CUSPARSE.CuDenseVectorDescriptor(Q.temp)
        CUDA.CUSPARSE.cusparseSpMV(Q.spmv_A.handle, Q.spmv_A.operator, Q.spmv_A.alpha,
            Q.spmv_A.desc_A, desc_x, Q.spmv_A.beta, desc_temp,
            Q.spmv_A.compute_type, Q.spmv_A.alg, Q.spmv_A.buf)
    else
        CUDA.CUSPARSE.mv!('N', 1, Q.A, x, 0, Q.temp, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    
    if Q.spmv_AT !== nothing
        # Preprocessed path: AT * temp -> Qx
        desc_temp = CUDA.CUSPARSE.CuDenseVectorDescriptor(Q.temp)
        desc_Qx = CUDA.CUSPARSE.CuDenseVectorDescriptor(Qx)
        CUDA.CUSPARSE.cusparseSpMV(Q.spmv_AT.handle, Q.spmv_AT.operator, Q.spmv_AT.alpha,
            Q.spmv_AT.desc_AT, desc_temp, Q.spmv_AT.beta, desc_Qx,
            Q.spmv_AT.compute_type, Q.spmv_AT.alg, Q.spmv_AT.buf)
    else
        CUDA.CUSPARSE.mv!('N', 1, Q.AT, Q.temp, 0, Qx, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
end

# Enable CUSPARSE preprocessing for LASSO operator
supports_cusparse_preprocessing(Q::LASSO_Q_operator_gpu) = true

# Prepare CUSPARSE preprocessing for LASSO operator's internal A and AT matrices
function prepare_operator_spmv!(Q::LASSO_Q_operator_gpu, x::CuVector{Float64}, Qx::CuVector{Float64})
    # Create descriptors
    desc_A = CUDA.CUSPARSE.CuSparseMatrixDescriptor(Q.A, 'O')
    desc_AT = CUDA.CUSPARSE.CuSparseMatrixDescriptor(Q.AT, 'O')
    desc_x = CUDA.CUSPARSE.CuDenseVectorDescriptor(x)
    desc_temp = CUDA.CUSPARSE.CuDenseVectorDescriptor(Q.temp)
    desc_Qx = CUDA.CUSPARSE.CuDenseVectorDescriptor(Qx)
    
    CUSPARSE_handle = CUDA.CUSPARSE.handle()
    ref_one = Ref{Float64}(one(Float64))
    ref_zero = Ref{Float64}(zero(Float64))
    
    # Prepare A SpMV (x -> temp)
    sz_A = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x, ref_zero,
        desc_temp, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_A)
    buf_A = CUDA.CuArray{UInt8}(undef, sz_A[])
    
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x, ref_zero,
            desc_temp, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)
    end
    
    Q.spmv_A = CUSPARSE_spmv_LASSO_A(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x, ref_zero,
        desc_temp, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)
    
    # Prepare AT SpMV (temp -> Qx)
    sz_AT = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_temp, ref_zero,
        desc_Qx, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_AT)
    buf_AT = CUDA.CuArray{UInt8}(undef, sz_AT[])
    
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_temp, ref_zero,
            desc_Qx, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)
    end
    
    Q.spmv_AT = CUSPARSE_spmv_LASSO_AT(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_temp, ref_zero,
        desc_Qx, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)
    
    return nothing
end
