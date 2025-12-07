"""
Demo: Using HPRQP with Warmstart via JuMP Interface

This demo shows how to use initial_x and initial_y parameters
to provide warmstart solutions to HPRQP.
"""

using JuMP
import HPRQP

println("="^70)
println("Demo: HPRQP with Warmstart (initial_x and initial_y)")
println("="^70)

# ============================================================================
# Example 1: Solving QP without warmstart
# ============================================================================
println("\n[Example 1] Solving QP without warmstart")
println("-"^70)

# Create a JuMP model with HPRQP optimizer
model1 = Model(HPRQP.Optimizer)
set_optimizer_attribute(model1, "stoptol", 1e-8)
set_optimizer_attribute(model1, "verbose", true)

# Define variables
@variable(model1, x1 >= 0)
@variable(model1, x2 >= 0)

# Define quadratic objective: min -3x1 - 5x2 + x1^2 + x2^2
@objective(model1, Min, -3x1 - 5x2 + x1^2 + x2^2)

# Add linear constraints
@constraint(model1, con1, 1x1 + 2x2 <= 10)
@constraint(model1, con2, 3x1 + 1x2 <= 12)

println("Solving...")
optimize!(model1)

# Get the results
x1_val = value(x1)
x2_val = value(x2)
time1 = solve_time(model1)

println("✓ Solution without warmstart:")
println("  x1 = ", x1_val)
println("  x2 = ", x2_val)
println("  Solve time: ", time1, " seconds")

# ============================================================================
# Example 2: Solving same QP with warmstart
# ============================================================================
println("\n[Example 2] Solving same QP with warmstart (initial_x and initial_y)")
println("-"^70)

# Create a new JuMP model with HPRQP optimizer
model2 = Model(HPRQP.Optimizer)
set_optimizer_attribute(model2, "stoptol", 1e-8)
set_optimizer_attribute(model2, "verbose", true)

# Provide warmstart solutions from previous solve
# For this demo, we use the solution from Example 1
initial_x_vals = [x1_val, x2_val]
initial_y_vals = [dual(con1), dual(con2)]  # dual variables from constraints

set_optimizer_attribute(model2, "initial_x", initial_x_vals)
set_optimizer_attribute(model2, "initial_y", initial_y_vals)

# Define the same problem
@variable(model2, y1 >= 0)
@variable(model2, y2 >= 0)
@objective(model2, Min, -3y1 - 5y2 + y1^2 + y2^2)
@constraint(model2, c1, 1y1 + 2y2 <= 10)
@constraint(model2, c2, 3y1 + 1y2 <= 12)

println("Solving with warmstart...")
optimize!(model2)

# Get the results
y1_val = value(y1)
y2_val = value(y2)
time2 = solve_time(model2)

println("✓ Solution with warmstart:")
println("  y1 = ", y1_val)
println("  y2 = ", y2_val)
println("  Solve time: ", time2, " seconds")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("Summary:")
println("="^70)
println("  Time without warmstart: ", time1, " seconds")
println("  Time with warmstart: ", time2, " seconds")
println("\nNote: Warmstart can significantly reduce solve time for similar problems")
println("or when solving a sequence of related optimization problems.")
println("="^70)
