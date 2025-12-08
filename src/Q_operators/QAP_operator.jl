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

# Q operator mapping for QAP problem: Q(X) = 2*(AXB - SX - XT) - GPU version
# where X is n×n reshaped from vector x of length n²
@inline function Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, Q::QAP_Q_operator_gpu)
    n = Q.n
    X = reshape(x, n, n)
    QX = reshape(Qx, n, n)
    TMP = reshape(Q.temp, n, n)
    mul!(TMP, Q.A, X)
    mul!(QX, TMP, Q.B, 2.0, 0.0)
    mul!(QX, Q.S, X, -2.0, 1.0)
    mul!(QX, X, Q.T, -2.0, 1.0)
end

# Q operator mapping for QAP problem: Q(X) = 2*(AXB - SX - XT) - CPU version
# where X is n×n reshaped from vector x of length n²
@inline function Qmap!(x::Vector{Float64}, Qx::Vector{Float64}, Q::QAP_Q_operator_cpu)
    n = Q.n
    X = reshape(x, n, n)
    QX = reshape(Qx, n, n)
    # TMP = A * X
    TMP = Q.A * X
    # QX = 2*(TMP * B - S*X - X*T)
    mul!(QX, TMP, Q.B, 2.0, 0.0)
    mul!(QX, Q.S, X, -2.0, 1.0)
    mul!(QX, X, Q.T, -2.0, 1.0)
end
