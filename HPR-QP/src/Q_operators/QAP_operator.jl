# ============================================================================
# QAP (Quadratic Assignment Problem) Q Operator
# ============================================================================
#
# Objective: min trace(AXB + SX + XT) subject to X being a permutation matrix
# Reformulated as QP: min x'Qx + c'x where x = vec(X)
# Q operator: Q(X) = 2*(AXB - SX - XT) where X is n×n reshaped from x
#
# ============================================================================

# CPU version: stores problem data on CPU
struct QAP_Q_operator_cpu <: AbstractQOperatorCPU
    A::Matrix{Float64}
    B::Matrix{Float64}
    S::Matrix{Float64}
    T::Matrix{Float64}
    n::Int
end

# GPU version: stores problem data on GPU
# Q(X) = 2*(AXB - SX - XT) where X is n×n reshaped from vector x of length n²
mutable struct QAP_Q_operator_gpu <: AbstractQOperator
    A::CuMatrix{Float64}
    B::CuMatrix{Float64}
    S::CuMatrix{Float64}
    T::CuMatrix{Float64}
    n::Int
    temp::CuVector{Float64}  # Temporary storage for intermediate computation
end

# Interface implementations for QAP
get_temp_size(Q::QAP_Q_operator_cpu) = Q.n^2
get_operator_name(::Type{QAP_Q_operator_gpu}) = "QAP"
get_problem_size(Q::QAP_Q_operator_cpu) = Q.n^2
get_problem_size(Q::QAP_Q_operator_gpu) = Q.n^2

function to_gpu(Q_cpu::QAP_Q_operator_cpu)
    return QAP_Q_operator_gpu(
        CuMatrix(Q_cpu.A),
        CuMatrix(Q_cpu.B),
        CuMatrix(Q_cpu.S),
        CuMatrix(Q_cpu.T),
        Q_cpu.n,
        CUDA.zeros(Float64, get_temp_size(Q_cpu))
    )
end
