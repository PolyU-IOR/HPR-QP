import HPRQP

params = HPRQP.HPRQP_parameters()
# params.max_iter = 300
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0 
params.warm_up = false

file = "/home/chenkaihuang/Data/QP_data/MM_mps/BOYD1.mps"
model = HPRQP.build_from_mps(file)
result = HPRQP.optimize(model, params)

println("Objective value: ", result.primal_obj)
