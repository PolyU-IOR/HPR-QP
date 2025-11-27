using SparseArrays
using LinearAlgebra
import HPRQP

# min 0.5 <x,Qx> + <c,x>
# s.t. AL <= Ax <= AU
#      l <= x <= u


# Example 1
# min x1^2 + x2^2 -3x1 - 5x2
# s.t. -x1 - 2x2 >= -10
#      -3x1 - x2 >= -12
#      x1 >= 0, x2 >= 0
#      x1 <= Inf, x2 <= Inf

Q = SparseMatrixCSC{Float64, Int32}([2 0; 0 2])
A = SparseMatrixCSC{Float64, Int32}([-1 -2; -3 -1])
AL = Vector{Float64}([-10, -12])
AU = Vector{Float64}([Inf, Inf])
# empty A, AL, AU
# A = SparseMatrixCSC{Float64, Int32}(undef, 0, 2)
# AL = Vector{Float64}(zeros(0))
# AU = Vector{Float64}(zeros(0))
c = Vector{Float64}([-3, -5])
l = Vector{Float64}([0.0, 0.0])
u = Vector{Float64}([Inf, Inf])
obj_constant = 0.0

params = HPRQP.HPRQP_parameters()
params.max_iter = typemax(Int32)
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0 
params.warm_up = true

result = HPRQP.run_qp(Q, c, A, AL, AU, l, u, obj_constant, params)

println("Objective value: ", result.primal_obj)
println("x1 = ", result.x[1])
println("x2 = ", result.x[2])
