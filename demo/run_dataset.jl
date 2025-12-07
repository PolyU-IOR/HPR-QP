import HPRQP

params = HPRQP.HPRQP_parameters()
params.max_iter = typemax(Int32)
params.time_limit = 3600
params.stoptol = 1e-8
params.warm_up = true
params.device_number = 0 

data_path = "xxx" # Path to the directory containing the dataset files
result_path = "xxx" # Path to save the results

HPRQP.run_dataset(data_path, result_path, params)
