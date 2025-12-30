import HPRQP

params = HPRQP.HPRQP_parameters()
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0 
params.warm_up = false
params.problem_type = "QP"

file = "model.mps"  # Path to your MPS file
model = HPRQP.build_from_mps(file)
result = HPRQP.optimize(model, params)

params.problem_type = "LASSO"
file = "./data/E2006.test.mat"  # Path to your MAT file for LASSO
model_lasso = HPRQP.build_from_mat(file, problem_type="LASSO")
result_lasso = HPRQP.optimize(model_lasso, params)

params.problem_type = "QAP"
file = "./data/esc64a.mat"  # Path to your QAP data file
model_qap = HPRQP.build_from_mat(file, problem_type="QAP")
result_qap = HPRQP.optimize(model_qap, params)

println(" ✓ All problems solved successfully!")