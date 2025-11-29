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
        CUDA.zeros(Float64, m)
    )
end
