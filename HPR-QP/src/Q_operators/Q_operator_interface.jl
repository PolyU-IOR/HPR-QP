# ============================================================================
# Custom Q Operator Interface
# ============================================================================
# 
# To define a custom Q operator, users must implement the following interface:
#
# 1. Define CPU struct <: AbstractQOperatorCPU (problem-specific data on CPU)
#    struct MyQOperatorCPU <: AbstractQOperatorCPU
#        # Your data fields (matrices, parameters, etc.)
#    end
#
# 2. Define GPU struct <: AbstractQOperator (problem-specific data on GPU)
#    mutable struct MyQOperatorGPU <: AbstractQOperator
#        # Your GPU data fields (CuArrays, etc.)
#        temp::CuVector{Float64}  # Temporary storage if needed
#    end
#
# 3. Implement required interface functions:
#    - to_gpu(cpu_data::MyQOperatorCPU) -> MyQOperatorGPU
#    - Qmap!(x::CuVector, Qx::CuVector, Q::MyQOperatorGPU) -> nothing
#    - get_temp_size(Q_cpu::MyQOperatorCPU) -> Int  # Size of temp vector
#    - get_operator_name(::Type{MyQOperatorGPU}) -> String  # For logging
#    - get_problem_size(Q::MyQOperatorCPU) -> Int  # Problem dimension
#
# 4. Optional: For eigenvalue estimation
#    - power_iteration_Q(Q::MyQOperatorGPU; max_iter, tol) -> Float64
#
# See CUSTOM_Q_OPERATOR_GUIDE.md for detailed examples.
#
# ============================================================================

# Abstract types for Q operators
abstract type AbstractQOperator end      # GPU operators
abstract type AbstractQOperatorCPU end   # CPU operators

# ============================================================================
# Interface Functions (must be implemented by users for custom operators)
# ============================================================================

"""
    to_gpu(Q_cpu::AbstractQOperatorCPU) -> AbstractQOperator

Transfer Q operator data from CPU to GPU and allocate temporary storage.
Users must implement this for their custom operator types.

# Arguments
- `Q_cpu`: CPU version of operator data

# Returns
- GPU version of the operator with allocated temp storage
"""
function to_gpu end

"""
    get_temp_size(Q_cpu) -> Int

Return the size of temporary storage needed for Qmap! operation.
Return 0 if no temporary storage is needed.

# Arguments
- `Q_cpu`: CPU version of operator data

# Returns
- Size of temporary vector needed
"""
function get_temp_size end

"""
    get_operator_name(::Type{<:AbstractQOperator}) -> String

Return a descriptive name for the operator type (used in logging).

# Arguments
- Operator type

# Returns
- String name (e.g., "QAP", "LASSO", "CustomOperator")
"""
function get_operator_name end

"""
    get_problem_size(Q) -> Int

Return the problem dimension (size of x vector) for the operator.

# Arguments
- `Q`: CPU or GPU operator

# Returns
- Problem dimension n
"""
function get_problem_size end

# Union type for Q on GPU (sparse matrix or operator)
const QType = Union{CuSparseMatrixCSR{Float64,Int32}, AbstractQOperator}

# Union type for Q on CPU (sparse matrix or CPU operator)
# Accept both Int and Int32 indices for compatibility
const QTypeCPU = Union{SparseMatrixCSC{Float64,Int}, SparseMatrixCSC{Float64,Int32}, AbstractQOperatorCPU}
