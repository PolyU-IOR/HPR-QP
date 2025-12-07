# MOI Wrapper for HPRQP
# Based on the structure of HPRLP's MOI wrapper

import MathOptInterface as MOI

# Supported scalar sets
const SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

"""
    Optimizer()

Create a new HPRQP Optimizer object.

Set optimizer attributes using `MOI.RawOptimizerAttribute` or
`JuMP.set_optimizer_attribute`.

## Example

```julia
using JuMP, HPRQP
model = JuMP.Model(HPRQP.Optimizer)
set_optimizer_attribute(model, "stoptol", 1e-6)
set_optimizer_attribute(model, "use_gpu", true)
set_optimizer_attribute(model, "device_number", 0)
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    params::HPRQP_parameters
    results::Union{Nothing, HPRQP_results}
    silent::Bool
    cache::Union{Nothing, OptimizerCache}
    index_map::Union{Nothing, MOI.IndexMap}
    obj_sense::MOI.OptimizationSense  # Track objective sense for result conversion
    
    function Optimizer()
        return new(HPRQP_parameters(), nothing, false, nothing, nothing, MOI.MIN_SENSE)
    end
end

# ====================
#   Utility functions
# ====================

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

# ====================
#   Empty functions
# ====================

function MOI.is_empty(model::Optimizer)
    return model.results === nothing
end

function MOI.empty!(model::Optimizer)
    model.results = nothing
    model.cache = nothing
    model.index_map = nothing
    model.obj_sense = MOI.MIN_SENSE  # Reset to default
    return
end

# ====================
#   Solver attributes
# ====================

MOI.get(::Optimizer, ::MOI.SolverName) = "HPRQP"

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    return "0.1.1"  # Update this to match your package version
end

# HPRQP does not support incremental interface - requires copy_to
MOI.supports_incremental_interface(::Optimizer) = false

# Silent mode
MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

# Time limit
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    model.params.time_limit = Float64(value)
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    model.params.time_limit = 3600.0  # Default value
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return model.params.time_limit
end

# Number of threads (not supported)
MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = false

# Raw optimizer attributes for HPRQP parameters
const SUPPORTED_PARAMETERS = (
    "stoptol",
    "sigma",
    "max_iter",
    "sigma_fixed",
    "time_limit",
    "eig_factor",
    "check_iter",
    "warm_up",
    "spmv_mode_Q",
    "spmv_mode_A",
    "print_frequency",
    "device_number",
    "use_Ruiz_scaling",
    "use_bc_scaling",
    "use_l2_scaling",
    "use_Pock_Chambolle_scaling",
    "initial_x",
    "initial_y",
    "auto_save",
    "save_filename",
    "verbose",
)

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return param.name in SUPPORTED_PARAMETERS
end

function MOI.set(model::Optimizer, param::MOI.RawOptimizerAttribute, value)
    name = String(param.name)
    if name == "stoptol"
        model.params.stoptol = Float64(value)
    elseif name == "sigma"
        model.params.sigma = Float64(value)
    elseif name == "max_iter"
        model.params.max_iter = Int(value)
    elseif name == "sigma_fixed"
        model.params.sigma_fixed = Bool(value)
    elseif name == "time_limit"
        model.params.time_limit = Float64(value)
    elseif name == "eig_factor"
        model.params.eig_factor = Float64(value)
    elseif name == "check_iter"
        model.params.check_iter = Int(value)
    elseif name == "warm_up"
        model.params.warm_up = Bool(value)
    elseif name == "spmv_mode_Q"
        model.params.spmv_mode_Q = String(value)
    elseif name == "spmv_mode_A"
        model.params.spmv_mode_A = String(value)
    elseif name == "print_frequency"
        model.params.print_frequency = Int(value)
    elseif name == "device_number"
        model.params.device_number = Int32(value)
    elseif name == "use_Ruiz_scaling"
        model.params.use_Ruiz_scaling = Bool(value)
    elseif name == "use_bc_scaling"
        model.params.use_bc_scaling = Bool(value)
    elseif name == "use_l2_scaling"
        model.params.use_l2_scaling = Bool(value)
    elseif name == "use_Pock_Chambolle_scaling"
        model.params.use_Pock_Chambolle_scaling = Bool(value)
    elseif name == "initial_x"
        model.params.initial_x = value === nothing ? nothing : Vector{Float64}(value)
    elseif name == "initial_y"
        model.params.initial_y = value === nothing ? nothing : Vector{Float64}(value)
    elseif name == "auto_save"
        model.params.auto_save = Bool(value)
    elseif name == "save_filename"
        model.params.save_filename = String(value)
    elseif name == "verbose"
        model.params.verbose = Bool(value)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end

function MOI.get(model::Optimizer, param::MOI.RawOptimizerAttribute)
    name = String(param.name)
    if name == "stoptol"
        return model.params.stoptol
    elseif name == "sigma"
        return model.params.sigma
    elseif name == "max_iter"
        return model.params.max_iter
    elseif name == "sigma_fixed"
        return model.params.sigma_fixed
    elseif name == "time_limit"
        return model.params.time_limit
    elseif name == "eig_factor"
        return model.params.eig_factor
    elseif name == "check_iter"
        return model.params.check_iter
    elseif name == "warm_up"
        return model.params.warm_up
    elseif name == "spmv_mode_Q"
        return model.params.spmv_mode_Q
    elseif name == "spmv_mode_A"
        return model.params.spmv_mode_A
    elseif name == "print_frequency"
        return model.params.print_frequency
    elseif name == "device_number"
        return model.params.device_number
    elseif name == "use_Ruiz_scaling"
        return model.params.use_Ruiz_scaling
    elseif name == "use_bc_scaling"
        return model.params.use_bc_scaling
    elseif name == "use_l2_scaling"
        return model.params.use_l2_scaling
    elseif name == "use_Pock_Chambolle_scaling"
        return model.params.use_Pock_Chambolle_scaling
    elseif name == "initial_x"
        return model.params.initial_x
    elseif name == "initial_y"
        return model.params.initial_y
    elseif name == "auto_save"
        return model.params.auto_save
    elseif name == "save_filename"
        return model.params.save_filename
    elseif name == "verbose"
        return model.params.verbose
    end
    throw(MOI.UnsupportedAttribute(param))
end

# ========================================
#   Supported constraints and objectives
# ========================================

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,MOI.ScalarAffineFunction{Float64}}},
    ::Type{<:SCALAR_SETS},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

# Support linear objective
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

# Support quadratic objective
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return true
end

# =======================
#   `copy_to` function
# =======================

function MOI.copy_to(dest::Optimizer, src::OptimizerCache)
    # This is called with the cache directly
    @assert MOI.is_empty(dest)
    dest.cache = src
    dest.index_map = _index_map(src)
    return dest.index_map
end

function MOI.copy_to(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache},
)
    # Throw error if there are unsupported constraints
    MOI.Utilities.throw_unsupported(src)
    return MOI.copy_to(dest, src.model)
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    # For general MOI models, copy to cache first
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    
    # Copy from cache to optimizer
    MOI.copy_to(dest, cache)
    
    return index_map
end

# Helper function to create index map from cache
function _index_map(src::OptimizerCache)
    index_map = MOI.IndexMap()
    # Map variables (1-indexed)
    for (i, x) in enumerate(MOI.get(src, MOI.ListOfVariableIndices()))
        index_map[x] = MOI.VariableIndex(i)
    end
    # Map constraints
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        _index_map_constraints(src, index_map, F, S)
    end
    return index_map
end

function _index_map_constraints(
    src::OptimizerCache,
    index_map,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{S},
) where {S<:SCALAR_SETS}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},S}())
        row = MOI.Utilities.rows(src.constraints, ci)
        index_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S}(row)
    end
    return
end

function _index_map_constraints(
    src::OptimizerCache,
    index_map,
    ::Type{MOI.VariableIndex},
    ::Type{S},
) where {S<:SCALAR_SETS}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex,S}())
        col = index_map[MOI.VariableIndex(ci.value)].value
        index_map[ci] = MOI.ConstraintIndex{MOI.VariableIndex,S}(col)
    end
    return
end

# ===============================
#   Optimize and post-optimize
# ===============================

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    # Extract constraint matrix A and bounds
    A = src.constraints.coefficients
    row_bounds = src.constraints.constants
    
    # Get objective function - check if it's linear or quadratic
    obj_function_type = MOI.get(src, MOI.ObjectiveFunctionType())
    
    # Initialize Q, c, and obj_constant
    n = A.n
    c = zeros(n)
    obj_constant = 0.0
    Q = spzeros(n, n)
    
    if obj_function_type == MOI.ScalarQuadraticFunction{Float64}
        # Quadratic objective
        obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
        
        # Extract linear part
        for term in obj.affine_terms
            c[term.variable.value] += term.coefficient
        end
        
        # Extract quadratic part
        # JuMP/MOI represents quadratic objectives as: sum of (coef * x_i * x_j) terms
        # For x^2, JuMP stores it as coefficient=2.0 in MOI (not 1.0)
        # For x*y, JuMP stores it as coefficient=1.0 for the cross term
        #
        # HPRQP expects Q such that the objective is: 0.5 * x'Qx + c'x
        # So if MOI has coefficient c for x_i * x_j, we need:
        #   - Diagonal (i==j): MOI has c*x_i^2, HPRQP needs Q_ii such that 0.5*Q_ii*x_i^2 = c*x_i^2, so Q_ii = 2c
        #   - Off-diagonal (i!=j): MOI has c*x_i*x_j, HPRQP needs Q_ij+Q_ji such that 0.5*(Q_ij+Q_ji)*x_i*x_j = c*x_i*x_j
        #                         so Q_ij = Q_ji = c
        #
        # BUT: JuMP actually represents x^2 with coefficient 2.0 in MOI (from the expansion)
        # And x*y with coefficient 1.0
        # So actually: For diagonal, coefficient is already 2*original, for off-diagonal it's just the original
        # Therefore:
        #   - Diagonal: coef is already doubled by JuMP, so Q_ii = coef (no additional factor)
        #   - Off-diagonal: Q_ij = Q_ji = 2*coef (to account for 0.5 factor in HPRQP)
        I_idx = Int[]
        J_idx = Int[]
        V_vals = Float64[]
        
        for term in obj.quadratic_terms
            i = term.variable_1.value
            j = term.variable_2.value
            coef = term.coefficient
            
            if i == j
                # Diagonal term: MOI already has 2*original_coef from JuMP's expansion
                # HPRQP needs Q_ii for 0.5*Q_ii*x_i^2
                # So Q_ii = coef (which is already 2*original)
                push!(I_idx, i)
                push!(J_idx, j)
                push!(V_vals, coef)
            else
                # Off-diagonal term: MOI has original_coef for x_i*x_j
                # HPRQP needs Q_ij and Q_ji such that 0.5*(Q_ij + Q_ji)*x_i*x_j equals the term
                # So Q_ij = Q_ji = 2*coef
                push!(I_idx, i)
                push!(J_idx, j)
                push!(V_vals, 2.0 * coef)
                
                push!(I_idx, j)
                push!(J_idx, i)
                push!(V_vals, 2.0 * coef)
            end
        end
        
        if !isempty(I_idx)
            Q = sparse(I_idx, J_idx, V_vals, n, n)
        end
        
        obj_constant = obj.constant
        
    elseif obj_function_type == MOI.ScalarAffineFunction{Float64}
        # Linear objective (Q = 0)
        obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        for term in obj.terms
            c[term.variable.value] += term.coefficient
        end
        obj_constant = obj.constant
    else
        error("Unsupported objective function type: $obj_function_type")
    end
    
    # Handle objective sense
    sense = MOI.get(src, MOI.ObjectiveSense())
    dest.obj_sense = sense  # Store for later use in result retrieval
    if sense == MOI.MAX_SENSE
        Q = -Q
        c = -c
        obj_constant = -obj_constant
    end
    
    # Extract variable bounds
    l = src.variables.lower
    u = src.variables.upper
    
    # Extract constraint bounds
    AL = row_bounds.lower
    AU = row_bounds.upper
    
    # Convert to standard Julia SparseMatrixCSC with Int32 indices
    A_sparse = SparseMatrixCSC{Float64, Int32}(A.m, A.n, 
                                                 convert(Vector{Int32}, A.colptr), 
                                                 convert(Vector{Int32}, A.rowval), 
                                                 A.nzval)
    Q_sparse = SparseMatrixCSC{Float64, Int32}(Q)
    
    # Set verbose: if silent mode is set, disable verbose regardless of parameter
    if dest.silent
        dest.params.verbose = false
    end
    
    # Build model and optimize
    model = build_from_QAbc(Q_sparse, c, A_sparse, AL, AU, l, u, obj_constant, verbose=dest.params.verbose)
    dest.results = optimize(model, dest.params)
    
    return
end

function MOI.optimize!(model::Optimizer)
    # Extract from stored cache
    if model.cache === nothing
        error("No problem has been loaded. Use JuMP.Model(HPRQP.Optimizer) and build your model first.")
    end
    
    MOI.optimize!(model, model.cache)
    return
end

# Solve time
function MOI.get(model::Optimizer, ::MOI.SolveTimeSec)
    if model.results === nothing
        return 0.0
    end
    return model.results.time
end

# Objective value
function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available. Call optimize! first.")
    end
    value = model.results.primal_obj
    # Convert back if this was a maximization problem
    if model.obj_sense == MOI.MAX_SENSE
        value = -value
    end
    return value
end

# Number of variables
function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    if model.results === nothing
        return 0
    end
    return length(model.results.x)
end

# ===============================
#   Termination and Result Status
# ===============================

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.results === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    
    if model.results.status == "OPTIMAL"
        return MOI.OPTIMAL
    elseif model.results.status == "MAX_ITER"
        return MOI.ITERATION_LIMIT
    elseif model.results.status == "TIME_LIMIT"
        return MOI.TIME_LIMIT
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.results === nothing
        return "OPTIMIZE_NOT_CALLED"
    end
    return model.results.status
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.results === nothing
        return 0
    end
    # HPRQP always returns a result if it has run
    return model.results.status == "OPTIMAL" ? 1 : 0
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    if model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.status == "OPTIMAL"
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    if model.results === nothing
        return MOI.NO_SOLUTION
    end
    if model.results.status == "OPTIMAL"
        return MOI.FEASIBLE_POINT
    end
    return MOI.UNKNOWN_RESULT_STATUS
end

# ===================
#   Primal solution
# ===================

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return model.results.x[x.value]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    xs::Vector{MOI.VariableIndex},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    return [model.results.x[x.value] for x in xs]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    # This would require storing the constraint matrix and computing Ax
    # For now, return a fallback
    return MOI.Utilities.get_fallback(model, attr, c)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    return MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(c.value))
end

# =================
#   Dual solution
# =================

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:SCALAR_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # Return dual for row constraint
    dual_val = model.results.y[c.value]
    # Adjust sign if maximization
    if model.obj_sense == MOI.MAX_SENSE
        dual_val = -dual_val
    end
    return dual_val
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # Return reduced cost (should be non-positive for upper bounds)
    dual_val = min(0.0, model.results.z[c.value])
    # Adjust sign if maximization
    if model.obj_sense == MOI.MAX_SENSE
        dual_val = -dual_val
    end
    return dual_val
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # Return reduced cost (should be non-negative for lower bounds)
    dual_val = max(0.0, model.results.z[c.value])
    # Adjust sign if maximization
    if model.obj_sense == MOI.MAX_SENSE
        dual_val = -dual_val
    end
    return dual_val
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{
        MOI.VariableIndex,
        <:Union{MOI.Interval{Float64},MOI.EqualTo{Float64}},
    },
)
    MOI.check_result_index_bounds(model, attr)
    if model.results === nothing
        error("No results available.")
    end
    # For interval and equality constraints, return the full reduced cost
    dual_val = model.results.z[c.value]
    # Adjust sign if maximization
    if model.obj_sense == MOI.MAX_SENSE
        dual_val = -dual_val
    end
    return dual_val
end
