using JuMP
import HPRQP
model = Model()

function simple_example(model)
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)

    # add a quadratic term to the objective
    @objective(model, Min, -3x1 + -5x2 + x1^2 + x2^2)


    @constraint(model, 1x1 + 2x2 <= 10)
    @constraint(model, 3x1 + 1x2 <= 12)
end

function HPRQP_solve()
    params = HPRQP.HPRQP_parameters()
    params.max_iter = typemax(Int32)
    params.time_limit = 3600
    params.stoptol = 1e-8
    params.device_number = 0 
    params.warm_up = true

    result = HPRQP.run_single("model.mps", params)

    # if maximize, then the objective value is the negative of the result
    if MOI.get(model, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        println("Maximizing, the objective value is the negative of the result")
        result.primal_obj = -result.primal_obj
    end

    println("Objective value: ", result.primal_obj)
    println("x1 = ", result.x[1])
    println("x2 = ", result.x[2])
end

# For more examples, please refer to the JuMP documentation: https://jump.dev/JuMP.jl/stable/tutorials/linear/introduction/
simple_example(model)

# Export the model to an MPS file
write_to_file(model, "model.mps")

# Solve the model using HPRQP
HPRQP_solve()
