"""
Demo: Using HPRQP with JuMP Interface

This demo shows how to:
1. Build and solve a simple QP problem using JuMP with HPRQP
2. Read and solve a QP problem from an MPS file using JuMP
"""

using JuMP
import HPRQP

println("="^70)
println("Demo: HPRQP with JuMP Interface")
println("="^70)

# ============================================================================
# Example 1: Simple QP Problem with JuMP
# ============================================================================
println("\n[Example 1] Solving a Simple QP Problem with JuMP")
println("-"^70)

# Create a JuMP model with HPRQP optimizer
model = Model(HPRQP.Optimizer)

# Set solver parameters using JuMP's set_optimizer_attribute
set_optimizer_attribute(model, "stoptol", 1e-8)
set_optimizer_attribute(model, "warm_up", true)

# Define variables
@variable(model, x1 >= 0)
@variable(model, x2 >= 0)

# Define quadratic objective: min -3x1 - 5x2 + x1^2 + x2^2
@objective(model, Min, -3x1 - 5x2 + x1^2 + x2^2)

# Add linear constraints
@constraint(model, con1, 1x1 + 2x2 <= 10)
@constraint(model, con2, 3x1 + 1x2 <= 12)

println("Problem formulation:")
println(model)

# Solve the problem
println("\nSolving...")
optimize!(model)

# Get the results
status = termination_status(model)
obj_value = objective_value(model)
x1_val = value(x1)
x2_val = value(x2)

println("✓ Solution found!")
println("  Status: ", status)
println("  Objective value: ", obj_value)
println("  x1 = ", x1_val)
println("  x2 = ", x2_val)

# ============================================================================
# Example 2: Reading and Solving MPS File with JuMP
# ============================================================================
println("\n[Example 2] Reading and Solving AUG2D.mps with JuMP")
println("-"^70)

# Check if the MPS file exists
mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")

if isfile(mps_file)
    # Create a new JuMP model with HPRQP optimizer
    model_mps = read_from_file(mps_file)
    set_optimizer(model_mps, HPRQP.Optimizer)
    
    # Set solver parameters
    set_optimizer_attribute(model_mps, "stoptol", 1e-8)
    set_optimizer_attribute(model_mps, "warm_up", false)
    
    println("Reading MPS file: ", mps_file)
    
    println("✓ MPS file loaded successfully")
    println("  Number of variables: ", num_variables(model_mps))
    
    # Solve the problem
    println("\nSolving AUG2D problem...")
    optimize!(model_mps)
    
    # Get the results
    status = termination_status(model_mps)
    obj_value = objective_value(model_mps)
    time_elapsed = solve_time(model_mps)
    
    println("✓ Solution found!")
    println("  Status: ", status)
    println("  Objective value: ", obj_value)
    println("  Solve time: ", time_elapsed, " seconds")
    
    # Get some solution values (first few variables)
    println("\n  First 5 variable values:")
    vars = all_variables(model_mps)
    for i in 1:min(5, length(vars))
        println("    ", vars[i], " = ", value(vars[i]))
    end
else
    @warn "AUG2D.mps file not found at: $mps_file"
    println("Skipping Example 2")
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Summary:")
println("="^70)
println("""
The JuMP interface provides a high-level way to work with HPRQP:

1. Use Model(HPRQP.Optimizer) to create a JuMP model with HPRQP
2. Set solver parameters with set_optimizer_attribute()
3. Define variables, objectives, and constraints using JuMP macros
4. Solve with optimize!()
5. Read MPS files directly with read_from_file()

For more JuMP examples, visit: https://jump.dev/JuMP.jl/stable/
""")
println("="^70)
