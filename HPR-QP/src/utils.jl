# Read data from a mps file
function read_mps(file::String)
    if file[end-3:end] == ".mps" || file[end-4:end] == ".MPS"
        io = open(file)
        qp = Logging.with_logger(Logging.NullLogger()) do
            QPSReader.readqps(io, mpsformat=:free)
        end
        close(io)
    else
        error("Unsupported file format. Please provide a .mps file.")
    end
    # constraint matrix
    A = sparse(qp.arows, qp.acols, qp.avals, qp.ncon, qp.nvar)
    lcon = qp.lcon
    ucon = qp.ucon

    # quadratic part
    Q = sparse(qp.qrows, qp.qcols, qp.qvals, qp.nvar, qp.nvar)
    # the Q matrix is not symmetric, so we need to symmetrize it
    diag_Q = diag(Q)
    Q = Q + Q' - Diagonal(diag_Q)

    # linear part
    c = qp.c
    c0 = qp.c0

    # bounds
    lvar = qp.lvar
    uvar = qp.uvar

    return Q, c, A, lcon, ucon, lvar, uvar, c0
end

# Formulate the QP problem without the C constraints (l ≤ x ≤ u)
function qp_formulation_noC(Q::SparseMatrixCSC,
    c::Vector{Float64},
    A::SparseMatrixCSC,
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    c0::Float64=0.0)
    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)

    A = vcat(A, spdiagm(ones(length(l))))
    AL = vcat(AL, l)
    AU = vcat(AU, u)

    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((AL .== -Inf) .& (AU .== Inf)))

    if length(del_row) > 0
        keep_rows = setdiff(1:size(A, 1), del_row)
        A = A[keep_rows, :]
        AL = AL[keep_rows]
        AU = AU[keep_rows]
        println("Deleted ", length(del_row), " rows of A (empty or free bounds).")
    end


    idxE = findall(AL .== AU)
    idxG = findall((AL .> -Inf) .& (AU .== Inf))
    idxL = findall((AL .== -Inf) .& (AU .< Inf))
    idxB = findall((AL .> -Inf) .& (AU .< Inf))
    idxB = setdiff(idxB, idxE)

    # check dimension of Q, c, A, l, u, AL, AU
    println("problem information: nRow = ", size(A, 1), ", nCol = ", size(A, 2), ", nnz Q = ", nnz(Q), ", nnz A = ", nnz(A))
    println("                     number of equalities = ", length(idxE))
    println("                     number of inequalities = ", length(idxG) + length(idxL) + length(idxB))
    @assert size(Q, 1) == size(Q, 2)
    @assert size(Q, 1) == length(c)
    @assert size(A, 2) == length(c)
    @assert length(l) == length(u)
    @assert length(c) == size(Q, 1)
    @assert length(AL) == length(AU)
    @assert length(AL) == size(A, 1)

    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, c0, [], false, true)

    # Return the modified qp
    return standard_qp
end

# Formulate the QP problem with the C constraints (l ≤ x ≤ u)
function qp_formulation(Q::SparseMatrixCSC,
    c::Vector{Float64},
    A::SparseMatrixCSC,
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    c0::Float64=0.0)
    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)

    # rows that AL and AU are -Inf and Inf
    del_row = union(del_row, findall((AL .== -Inf) .& (AU .== Inf)))

    if length(del_row) > 0
        keep_rows = setdiff(1:size(A, 1), del_row)
        A = A[keep_rows, :]
        AL = AL[keep_rows]
        AU = AU[keep_rows]
        println("Deleted ", length(del_row), " rows of A that are all zeros.")
    end

    idxE = findall(AL .== AU)
    idxG = findall((AL .> -Inf) .& (AU .== Inf))
    idxL = findall((AL .== -Inf) .& (AU .< Inf))
    idxB = findall((AL .> -Inf) .& (AU .< Inf))
    idxB = setdiff(idxB, idxE)

    # check dimension of Q, c, A, l, u, AL, AU
    println("problem information: nRow = ", size(A, 1), ", nCol = ", size(A, 2), ", nnz Q = ", nnz(Q), ", nnz A = ", nnz(A))
    println("                     number of equalities = ", length(idxE))
    println("                     number of inequalities = ", length(idxG) + length(idxL) + length(idxB))
    @assert size(Q, 1) == size(Q, 2)
    @assert size(Q, 1) == length(c)
    @assert size(A, 2) == length(c)
    @assert length(l) == length(u)
    @assert length(l) == size(Q, 1)
    @assert length(AL) == length(AU)
    @assert length(AL) == size(A, 1)


    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, c0, [], false, false)

    # Return the modified qp
    return standard_qp
end

# Scale the QP problem according to the Ruiz scaling method, Pock-Chambolle scaling, or L2 scaling.
function scaling!(qp::QP_info_cpu, params::HPRQP_parameters)
    m, n = size(qp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    # Preallocate temporary arrays
    temp_norm_A_row = zeros(m)
    temp_norm_A_col = zeros(n)
    temp_norm_Q = zeros(n)
    DR = spdiagm(temp_norm_A_row)
    DC = spdiagm(temp_norm_A_col)

    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info = Scaling_info_cpu(copy(qp.l), copy(qp.u), row_norm, col_norm, 1, 1, 1, 1, norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf), norm(qp.c, Inf))

    if params.use_Ruiz_scaling
        for _ in 1:10
            temp_norm_Q .= maximum(abs, qp.Q, dims=1)[1, :]

            temp_norm_A_col .= maximum(abs, qp.A, dims=1)[1, :]
            temp_norm_A_col .= sqrt.(max.(temp_norm_A_col, temp_norm_Q))
            temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

            temp_norm_A_row .= sqrt.(maximum(abs, qp.A, dims=2)[:, 1])
            temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

            row_norm .*= temp_norm_A_row
            col_norm .*= temp_norm_A_col


            DR .= spdiagm(1.0 ./ temp_norm_A_row)
            DC .= spdiagm(1.0 ./ temp_norm_A_col)

            qp.Q .= DC * qp.Q * DC
            qp.c ./= temp_norm_A_col
            qp.A .= DR * qp.A * DC

            qp.AL ./= temp_norm_A_row
            qp.AU ./= temp_norm_A_row
            qp.l .*= temp_norm_A_col
            qp.u .*= temp_norm_A_col
        end
    end

    if params.use_bc_scaling
        b_scale = 1 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
        c_scale = 1 + norm(qp.c)
        println("b_scale: ", b_scale)
        println("c_scale: ", c_scale)
        qp.Q .*= b_scale / c_scale
        qp.AL ./= b_scale
        qp.AU ./= b_scale
        qp.c ./= c_scale
        qp.l ./= b_scale
        qp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    if params.use_l2_scaling
        # compute the l2 norm of the rows and columns of A
        temp_norm_Q .= sum(t -> t^2, qp.Q, dims=1)[1, :]
        temp_norm_A_col .= sum(t -> t^2, qp.A, dims=1)[1, :]
        temp_norm_A_col .= sqrt.(temp_norm_A_col .+ temp_norm_Q)
        temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

        temp_norm_A_row .= sqrt.(sum(t -> t^2, qp.A, dims=2)[:, 1])
        temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

        row_norm .*= temp_norm_A_row
        col_norm .*= temp_norm_A_col

        DR .= spdiagm(1.0 ./ temp_norm_A_row)
        DC .= spdiagm(1.0 ./ temp_norm_A_col)

        qp.Q .= DC * qp.Q * DC
        qp.c ./= temp_norm_A_col
        qp.A .= DR * qp.A * DC
        qp.x0 .*= temp_norm_A_col
        qp.y0 .*= temp_norm_A_row

        qp.AL ./= temp_norm_A_row
        qp.AU ./= temp_norm_A_row
        qp.l .*= temp_norm_A_col
        qp.u .*= temp_norm_A_col
    end

    if params.use_Pock_Chambolle_scaling
        temp_norm_Q .= sum(abs, qp.Q, dims=1)[1, :]
        temp_norm_A_col .= sum(abs, qp.A, dims=1)[1, :]
        temp_norm_A_col .= sqrt.(temp_norm_A_col .+ temp_norm_Q)
        temp_norm_A_col[iszero.(temp_norm_A_col)] .= 1.0

        temp_norm_A_row .= sqrt.(sum(abs, qp.A, dims=2)[:, 1])
        temp_norm_A_row[iszero.(temp_norm_A_row)] .= 1.0

        row_norm .*= temp_norm_A_row
        col_norm .*= temp_norm_A_col

        DR .= spdiagm(1.0 ./ temp_norm_A_row)
        DC .= spdiagm(1.0 ./ temp_norm_A_col)

        qp.Q .= DC * qp.Q * DC
        qp.c ./= temp_norm_A_col
        qp.A .= DR * qp.A * DC

        qp.AL ./= temp_norm_A_row
        qp.AU ./= temp_norm_A_row
        qp.l .*= temp_norm_A_col
        qp.u .*= temp_norm_A_col
    end

    temp_norm_Q .= sum(abs, qp.Q, dims=1)[1, :]

    # the diagonal matrix of the Q matrix
    diag_Q = diag(qp.Q)
    Q_is_diag = (sum(temp_norm_Q .== diag_Q) == length(temp_norm_Q))
    qp.Q_is_diag = Q_is_diag
    qp.diag_Q = diag_Q


    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(qp.c)
    qp.AT = qp.A'
    # eliminate the numerical error cause the asymmetry of Q
    qp.Q = (qp.Q + transpose(qp.Q)) / 2
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    return scaling_info
end

function mean(x::Vector{Float64})
    return sum(x) / length(x)
end

# Power iteration method to find the largest eigenvalue of a matrix AAT using GPU
function power_iteration_A_gpu(A::CuSparseMatrixCSR, AT::CuSparseMatrixCSR, max_iterations=5000, tolerance=1e-4)
    seed = 1
    m, n = size(A)
    z = CuVector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, m)
    ATq = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, AT, q, 0, ATq, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        CUDA.CUSPARSE.mv!('N', 1, A, ATq, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = CUDA.norm(q) / (CUDA.norm(z) + lambda_max)
        if error < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# Power iteration method to find the largest eigenvalue of a matrix Q using GPU
function power_iteration_Q_gpu(Q::CuSparseMatrixCSR, max_iterations=5000, tolerance=1e-4)
    seed = 1
    n, n = size(Q)
    z = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        CUDA.CUSPARSE.mv!('N', 1, Q, q, 0, z, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        lambda_max = CUDA.dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = CUDA.norm(q) / (CUDA.norm(z) + lambda_max)

        if error < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# Run the HPR-QP solver on a given file with specified parameters
function run_file(FILE_NAME::String, params::HPRQP_parameters)
    CUDA.device!(params.device_number)
    t_start = time()
    println("READING FILE ... ", FILE_NAME)
    Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps(FILE_NAME)
    read_time = time() - t_start
    println(@sprintf("READING FILE time: %.2f seconds", read_time))

    t_start = time()
    setup_start = time()
    println("FORMULATING QP ...")
    number_empty_lu = sum((lvar .== -Inf) .& (uvar .== Inf))
    if (number_empty_lu > 0.8 * length(lvar))
        println("put C into K")
        standard_qp = qp_formulation_noC(Q, c, A, lcon, ucon, lvar, uvar, c0)
    else
        standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
    end
    t_end = time()
    println(@sprintf("FORMULATING QP time: %.2f seconds", t_end - t_start))

    t_start = time()
    println("SCALING QP ... ")
    scaling_info = scaling!(standard_qp, params)
    t_end = time()
    println(@sprintf("SCALING time: %.2f seconds", t_end - t_start))
    if standard_qp.Q_is_diag
        println("Q is diagonal: true")
    else
        println("Q is diagonal: false")
    end

    t_start_gpu = time()
    CUDA.synchronize()
    println("COPY TO GPU ...")
    standard_qp_gpu = QP_info_gpu(
        CuSparseMatrixCSR(standard_qp.Q),
        CuVector(standard_qp.c),
        CuSparseMatrixCSR(standard_qp.A),
        CuSparseMatrixCSR(standard_qp.A'),
        CuVector(standard_qp.AL),
        CuVector(standard_qp.AU),
        CuVector(standard_qp.l),
        CuVector(standard_qp.u),
        standard_qp.obj_constant,
        CuVector(standard_qp.diag_Q),
        standard_qp.Q_is_diag,
        standard_qp.noC,
    )

    scaling_info_gpu = Scaling_info_gpu(
        CuVector(scaling_info.l_org),
        CuVector(scaling_info.u_org),
        CuVector(scaling_info.row_norm),
        CuVector(scaling_info.col_norm),
        scaling_info.b_scale,
        scaling_info.c_scale,
        scaling_info.norm_b,
        scaling_info.norm_c,
        scaling_info.norm_b_org,
        scaling_info.norm_c_org)

    CUDA.synchronize()
    t_end_gpu = time()
    println(@sprintf("COPY TO GPU time: %.2f seconds", t_end_gpu - t_start_gpu))
    setup_time = time() - setup_start

    results = solve(standard_qp_gpu, scaling_info_gpu, params)

    println(@sprintf("Total time: %.2fs", read_time + setup_time + results.time),
        @sprintf("  read time = %.2fs", read_time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))

    return results

end

# Run the dataset of QP problems from a specified directory and save the results to a CSV file
function run_dataset(data_path::String, result_path::String, params::HPRQP_parameters)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = result_path * "HPRQP_result.csv"

    # redirect the output to a file
    log_path = result_path * "HPRQP_log.txt"

    if !isdir(result_path)
        mkdir(result_path)
    end

    io = open(log_path, "a")

    # if csv file exists, read the existing results, where each column is an any array
    if isfile(csv_file)
        result_table = CSV.read(csv_file, DataFrame)
        namelist = Vector{Any}(result_table.name[1:end-2])
        iterlist = Vector{Any}(result_table.iter[1:end-2])
        timelist = Vector{Any}(result_table.alg_time[1:end-2])
        reslist = Vector{Any}(result_table.res[1:end-2])
        objlist = Vector{Any}(result_table.primal_obj[1:end-2])
        statuslist = Vector{Any}(result_table.status[1:end-2])
        iter4list = Vector{Any}(result_table.iter_4[1:end-2])
        time3list = Vector{Any}(result_table.time_4[1:end-2])
        iter6list = Vector{Any}(result_table.iter_6[1:end-2])
        time6list = Vector{Any}(result_table.time_6[1:end-2])
        powerlist = Vector{Any}(result_table.power_time[1:end-2])
    else
        namelist = []
        iterlist = []
        timelist = []
        reslist = []
        objlist = []
        statuslist = []
        iter4list = []
        time3list = []
        iter6list = []
        time6list = []
        powerlist = []
    end

    warm_up_done = false
    for i = 1:length(files)
        file = files[i]
        if occursin(".mps", file) && !(file in namelist)
            FILE_NAME = data_path * file
            if params.warm_up && !warm_up_done
                max_iter = params.max_iter
                params.max_iter = 200
                warm_up_done = true
                println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                params.max_iter = max_iter
                all_time = time() - t_start_all
                println("warm up time: ", all_time)
                println("warm up ends ----------------------------------------------------------------------------------------------------------")
            end
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                results = run_file(FILE_NAME, params)
                all_time = time() - t_start_all
                println("main run ends----------------------------------------------------------------------------------------------------------")


                println("iter = ", results.iter,
                    @sprintf("  time = %3.2e", results.time),
                    @sprintf("  residual = %3.2e", results.residuals),
                    @sprintf("  primal_obj = %3.15e", results.primal_obj),
                )

                push!(namelist, file)
                push!(iterlist, results.iter)
                push!(timelist, min(results.time, params.time_limit))
                push!(reslist, results.residuals)
                push!(objlist, results.primal_obj)
                push!(statuslist, results.output_type)
                push!(iter4list, results.iter_4)
                push!(time3list, min(results.time_4, params.time_limit))
                push!(iter6list, results.iter_6)
                push!(time6list, min(results.time_6, params.time_limit))
                push!(powerlist, results.power_time)

            end

            result_table = DataFrame(name=namelist,
                iter=iterlist,
                alg_time=timelist,
                res=reslist,
                primal_obj=objlist,
                status=statuslist,
                iter_4=iter4list,
                time_4=time3list,
                iter_6=iter6list,
                time_6=time6list,
                power_time=powerlist,
            )

            # compute the shifted geometric mean of the algorithm_time, put it in the last row
            geomean_time = exp(mean(log.(timelist .+ 10.0))) - 10.0
            geomean_time_4 = exp(mean(log.(time3list .+ 10.0))) - 10.0
            geomean_time_6 = exp(mean(log.(time6list .+ 10.0))) - 10.0
            geomean_iter = exp(mean(log.(iterlist .+ 10.0))) - 10.0
            geomean_iter_4 = exp(mean(log.(iter4list .+ 10.0))) - 10.0
            geomean_iter_6 = exp(mean(log.(iter6list .+ 10.0))) - 10.0
            push!(result_table, ["SGM10", geomean_iter, geomean_time, "", "", "", geomean_iter_4, geomean_time_4, geomean_iter_6, geomean_time_6, ""])

            # count the number of solved instances, termlist = "OPTIMAL" means solved
            solved = count(x -> x < params.time_limit, timelist)
            solved_3 = count(x -> x < params.time_limit, time3list)
            solved_6 = count(x -> x < params.time_limit, time6list)
            push!(result_table, ["solved", "", solved, "", "", "", "", solved_3, "", solved_6, ""])

            CSV.write(csv_file, result_table)
        end
    end

    close(io)
end

# it's used in demo_QAbc.jl
function run_qp(Q::SparseMatrixCSC,
    c::Vector{Float64},
    A::SparseMatrixCSC,
    lcon::Vector{Float64},
    ucon::Vector{Float64},
    lvar::Vector{Float64},
    uvar::Vector{Float64},
    c0::Float64,
    params)
    CUDA.device!(params.device_number)
    t_start = time()
    setup_start = time()
    println("Formulating QP...")
    number_empty_lu = sum((lvar .== -Inf) .& (uvar .== Inf))
    if (number_empty_lu > 0.8 * length(lvar))
        standard_qp = qp_formulation_noC(Q, c, A, lcon, ucon, lvar, uvar, c0)
        println("QP formulation without C")
    else
        standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
        println("QP formulation with C")
    end
    t_end = time()
    println("QP formulation time = ", t_end - t_start, " seconds")

    t_start = time()
    scaling_info = scaling!(standard_qp, params)
    t_end = time()
    println("scaling time = ", t_end - t_start, " seconds")
    if standard_qp.Q_is_diag
        println("Q is diagonal")
    end

    t_start_gpu = time()
    CUDA.synchronize()
    println("copy to GPU starts...")
    standard_qp_gpu = QP_info_gpu(
        CuSparseMatrixCSR(standard_qp.Q),
        CuVector(standard_qp.c),
        CuSparseMatrixCSR(standard_qp.A),
        CuSparseMatrixCSR(standard_qp.A'),
        CuVector(standard_qp.AL),
        CuVector(standard_qp.AU),
        CuVector(standard_qp.l),
        CuVector(standard_qp.u),
        standard_qp.obj_constant,
        CuVector(standard_qp.diag_Q),
        standard_qp.Q_is_diag,
        standard_qp.noC,
    )

    scaling_info_gpu = Scaling_info_gpu(
        CuVector(scaling_info.l_org),
        CuVector(scaling_info.u_org),
        CuVector(scaling_info.row_norm),
        CuVector(scaling_info.col_norm),
        scaling_info.b_scale,
        scaling_info.c_scale,
        scaling_info.norm_b,
        scaling_info.norm_c,
        scaling_info.norm_b_org,
        scaling_info.norm_c_org)

    CUDA.synchronize()
    t_end_gpu = time()
    println("copy to GPU ends, time = ", t_end_gpu - t_start_gpu, " seconds")
    setup_time = time() - setup_start

    results = solve(standard_qp_gpu, scaling_info_gpu, params)
    println(@sprintf("Total time: %.2fs", setup_time + results.time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))

    return results
end