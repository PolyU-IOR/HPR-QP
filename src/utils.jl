"""
    validate_gpu_parameters!(params::HPRQP_parameters)

Validates GPU-related parameters and adjusts settings if GPU is requested but not available.

# Arguments
- `params::HPRQP_parameters`: The solver parameters to validate

# Behavior
- If `use_gpu=true` but CUDA is not functional, sets `use_gpu=false` and warns user
- If `use_gpu=true` but device_number is invalid, sets `use_gpu=false` and warns user
- Validates that device_number is within valid range [0, num_devices-1]
"""
function validate_gpu_parameters!(params::HPRQP_parameters)
    if params.use_gpu
        # Check if CUDA is functional
        if !CUDA.functional()
            @warn "GPU requested but CUDA is not functional. Falling back to CPU execution."
            params.use_gpu = false
            return
        end
        
        # Check if device_number is valid
        num_devices = length(CUDA.devices())
        if params.device_number < 0 || params.device_number >= num_devices
            @warn "Invalid GPU device number $(params.device_number). Valid range is [0, $(num_devices-1)]. Falling back to CPU execution."
            params.use_gpu = false
            return
        end
    end
end

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

# Formulate the QP problem with the C constraints (l ≤ x ≤ u)
function qp_formulation(Q::SparseMatrixCSC,
    c::Vector{Float64},
    A::SparseMatrixCSC,
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    c0::Float64=0.0)

    # ====================================================================
    # Input Validation
    # ====================================================================

    # Check Q matrix properties
    m_Q, n_Q = size(Q)
    if m_Q != n_Q
        error("Q matrix must be square. Got size ($m_Q, $n_Q).")
    end
    n = n_Q

    # Check Q is symmetric (within tolerance)
    if nnz(Q) > 0
        Q_diff = Q - Q'
        if norm(Q_diff, Inf) > 1e-10
            @warn "Q matrix is not symmetric (max deviation: $(norm(Q_diff, Inf))). Symmetrizing Q = 0.5*(Q + Q')."
            Q = 0.5 * (Q + Q')
            dropzeros!(Q)
        end
    end

    # Check vector dimensions
    if length(c) != n
        error("Dimension mismatch: Q is $n×$n but c has length $(length(c)).")
    end
    if length(l) != n
        error("Dimension mismatch: Q is $n×$n but l has length $(length(l)).")
    end
    if length(u) != n
        error("Dimension mismatch: Q is $n×$n but u has length $(length(u)).")
    end

    # Check A matrix dimensions
    m_A, n_A = size(A)
    if n_A != n
        error("Dimension mismatch: Q is $n×$n but A has $n_A columns.")
    end
    if length(AL) != m_A
        error("Dimension mismatch: A has $m_A rows but AL has length $(length(AL)).")
    end
    if length(AU) != m_A
        error("Dimension mismatch: A has $m_A rows but AU has length $(length(AU)).")
    end

    # Check bound consistency
    infeasible_bounds = findall(l .> u)
    if !isempty(infeasible_bounds)
        error("Infeasible variable bounds: l > u at indices: $(infeasible_bounds[1:min(5, length(infeasible_bounds))]) $(length(infeasible_bounds) > 5 ? "..." : "")")
    end

    infeasible_constraints = findall(AL .> AU)
    if !isempty(infeasible_constraints)
        error("Infeasible constraint bounds: AL > AU at rows: $(infeasible_constraints[1:min(5, length(infeasible_constraints))]) $(length(infeasible_constraints) > 5 ? "..." : "")")
    end

    # Check for NaN or Inf in problem data (except bounds which can be ±Inf)
    if any(isnan, Q.nzval) || any(isinf, Q.nzval)
        error("Q matrix contains NaN or Inf values.")
    end
    if any(isnan, c) || any(isinf, c)
        error("c vector contains NaN or Inf values.")
    end
    if any(isnan, A.nzval) || any(isinf, A.nzval)
        error("A matrix contains NaN or Inf values.")
    end
    if any(isnan.(AL) .& isfinite.(AL)) || any(isnan.(AU) .& isfinite.(AU))
        error("Constraint bounds AL or AU contain NaN values.")
    end
    if any(isnan.(l) .& isfinite.(l)) || any(isnan.(u) .& isfinite.(u))
        error("Variable bounds l or u contain NaN values.")
    end

    # ====================================================================
    # Problem Preprocessing
    # ====================================================================

    # Remove the rows of A that are all zeros
    abs_A = abs.(A)
    del_row = findall(sum(abs_A, dims=2)[:, 1] .== 0)    # rows that AL and AU are -Inf and Inf
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
    # println("problem information: nRow = ", size(A, 1), ", nCol = ", size(A, 2), ", nnz Q = ", nnz(Q), ", nnz A = ", nnz(A))
    # println("                     number of equalities = ", length(idxE))
    # println("                     number of inequalities = ", length(idxG) + length(idxL) + length(idxB))
    @assert size(Q, 1) == size(Q, 2)
    @assert size(Q, 1) == length(c)
    @assert size(A, 2) == length(c)
    @assert length(l) == length(u)
    @assert length(l) == size(Q, 1)
    @assert length(AL) == length(AU)
    @assert length(AL) == size(A, 1)


    standard_qp = QP_info_cpu(Q, c, A, A', AL, AU, l, u, c0, [], false, 0.0)

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

# CPU-based scaling function for the QP problem (similar to GPU version)
function scaling_cpu!(qp::QP_info_cpu, params::HPRQP_parameters)
    m, n = size(qp.A)

    # Check if Q is a sparse matrix (not an operator)
    # If Q is an operator (QAP/LASSO), we skip ALL scaling
    Q_is_operator = isa(qp.Q, AbstractQOperatorCPU)

    if Q_is_operator
        if params.verbose
            println("Q is an operator - skipping ALL scaling")
        end
        # Return minimal scaling info with no scaling applied
        row_norm = ones(Float64, m)
        col_norm = ones(Float64, n)

        AL_nInf = copy(qp.AL)
        AU_nInf = copy(qp.AU)
        AL_nInf[qp.AL.==-Inf] .= 0.0
        AU_nInf[qp.AU.==Inf] .= 0.0
        norm_b_org = m > 0 ? norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf) : 0.0
        norm_c_org = norm(qp.c, Inf)

        scaling_info = Scaling_info_cpu(
            copy(qp.l), copy(qp.u),
            row_norm, col_norm,
            1.0, 1.0, 1.0, 1.0,
            norm_b_org, norm_c_org
        )

        scaling_info.norm_b = m > 0 ? norm(max.(abs.(AL_nInf), abs.(AU_nInf))) : 0.0
        scaling_info.norm_c = norm(qp.c)

        return scaling_info
    end

    # For sparse Q, proceed with normal scaling
    # Initialize scaling vectors
    row_norm = ones(Float64, m)
    col_norm = ones(Float64, n)

    # Compute original norms for scaling info
    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    norm_b_org = norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf)
    norm_c_org = norm(qp.c, Inf)

    # Initialize scaling info
    scaling_info = Scaling_info_cpu(
        copy(qp.l), copy(qp.u),
        row_norm, col_norm,
        1.0, 1.0, 1.0, 1.0,
        norm_b_org, norm_c_org
    )

    # Temporary vectors for scaling
    temp_row_norm = ones(Float64, m)
    temp_col_norm = ones(Float64, n)

    # Ruiz scaling
    if params.use_Ruiz_scaling
        for _ in 1:10
            # Compute column-wise max of |Q| and |A| combined
            temp_col_norm .= vec(maximum(abs, qp.A, dims=1))
            temp_norm_Q = vec(maximum(abs, qp.Q, dims=1))
            temp_col_norm .= sqrt.(max.(temp_col_norm, temp_norm_Q))
            temp_col_norm[iszero.(temp_col_norm)] .= 1.0

            # Compute row-wise max of |A|
            if m > 0
                temp_row_norm .= sqrt.(vec(maximum(abs, qp.A, dims=2)))
                temp_row_norm[iszero.(temp_row_norm)] .= 1.0
            end

            # Update cumulative norms
            row_norm .*= temp_row_norm
            col_norm .*= temp_col_norm

            # Scale Q: Q = DC * Q * DC
            DC = spdiagm(1.0 ./ temp_col_norm)
            qp.Q = DC * qp.Q * DC

            # Scale A: A = DR * A * DC
            if m > 0
                DR = spdiagm(1.0 ./ temp_row_norm)
                qp.A = DR * qp.A * DC
            end

            # Scale objective and constraint bounds
            qp.c ./= temp_col_norm

            # Scale constraint bounds
            if m > 0
                qp.AL ./= temp_row_norm
                qp.AU ./= temp_row_norm
            end

            # Scale variable bounds
            qp.l .*= temp_col_norm
            qp.u .*= temp_col_norm
        end
    end

    # Pock-Chambolle scaling
    if params.use_Pock_Chambolle_scaling
        # Compute column-wise sum of |Q| and |A| combined
        temp_col_norm .= vec(sum(abs, qp.A, dims=1))
        temp_norm_Q = vec(sum(abs, qp.Q, dims=1))
        temp_col_norm .= sqrt.(temp_col_norm .+ temp_norm_Q)
        temp_col_norm[iszero.(temp_col_norm)] .= 1.0

        # Compute row-wise sum of |A|
        if m > 0
            temp_row_norm .= sqrt.(vec(sum(abs, qp.A, dims=2)))
            temp_row_norm[iszero.(temp_row_norm)] .= 1.0
        end

        # Update cumulative norms
        row_norm .*= temp_row_norm
        col_norm .*= temp_col_norm

        # Scale Q: Q = DC * Q * DC
        DC = spdiagm(1.0 ./ temp_col_norm)
        qp.Q = DC * qp.Q * DC

        # Scale A: A = DR * A * DC
        if m > 0
            DR = spdiagm(1.0 ./ temp_row_norm)
            qp.A = DR * qp.A * DC
        end

        # Scale objective and bounds
        qp.c ./= temp_col_norm
        if m > 0
            qp.AL ./= temp_row_norm
            qp.AU ./= temp_row_norm
        end
        qp.l .*= temp_col_norm
        qp.u .*= temp_col_norm
    end

    # b and c scaling (disabled for now, same as GPU)
    if params.use_bc_scaling && false
        AL_nInf = copy(qp.AL)
        AU_nInf = copy(qp.AU)
        AL_nInf[qp.AL.==-Inf] .= 0.0
        AU_nInf[qp.AU.==Inf] .= 0.0
        b_scale = 1 + norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
        c_scale = 1 + norm(qp.c)

        if params.verbose
            println("b_scale: ", b_scale)
            println("c_scale: ", c_scale)
        end

        # Scale Q: Q *= b_scale / c_scale
        if Q_is_sparse
            scale_factor = b_scale / c_scale
            qp.Q = qp.Q .* scale_factor
        end

        if m > 0
            qp.AL ./= b_scale
            qp.AU ./= b_scale
        end
        qp.c ./= c_scale
        qp.l ./= b_scale
        qp.u ./= b_scale

        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    qp.AT = qp.A'

    # Update diagonal of Q if Q is sparse
    # Extract diagonal from Q matrix
    diag_Q_cpu = diag(qp.Q)
    qp.diag_Q = diag_Q_cpu

    # Check if Q is diagonal
    temp_norm_Q = vec(sum(abs, qp.Q, dims=1))
    Q_is_diag = (sum(temp_norm_Q .== abs.(diag_Q_cpu)) == length(temp_norm_Q))
    qp.Q_is_diag = Q_is_diag

    # Symmetrize Q to eliminate numerical errors
    qp.Q = (qp.Q + transpose(qp.Q)) / 2
    # Compute final norms
    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(qp.c)

    # Store the cumulative scaling norms
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm

    return scaling_info
end

# Wrapper function for CPU scaling (similar to scale_on_gpu for consistency)
function scale_on_cpu!(qp::QP_info_cpu, params::HPRQP_parameters)
    if params.verbose
        println("SCALING QP ON CPU ...")
    end
    t_start = time()

    scaling_info_cpu = scaling_cpu!(qp, params)
    CUDA.synchronize()

    scaling_time = time() - t_start
    if params.verbose
        println(@sprintf("CPU SCALING time: %.2f seconds", scaling_time))
    end

    return scaling_info_cpu, scaling_time
end

# GPU-based scaling function for the QP problem
function scaling_gpu!(qp::QP_info_gpu, params::HPRQP_parameters)
    m = size(qp.A, 1)
    n = size(qp.A, 2)

    # Check if Q is a sparse matrix (not an operator)
    # If Q is an operator (QAP/LASSO), we skip ALL scaling
    Q_is_operator = isa(qp.Q, AbstractQOperator)

    if Q_is_operator
        println("Q is an operator - skipping ALL scaling")
        # Return minimal scaling info with no scaling applied
        row_norm = CUDA.ones(Float64, m)
        col_norm = CUDA.ones(Float64, n)

        AL_nInf = copy(qp.AL)
        AU_nInf = copy(qp.AU)
        AL_nInf[qp.AL.==-Inf] .= 0.0
        AU_nInf[qp.AU.==Inf] .= 0.0
        norm_b_org = m > 0 ? CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf) : 0.0
        norm_c_org = CUDA.norm(qp.c, Inf)

        scaling_info = Scaling_info_gpu(
            copy(qp.l), copy(qp.u),
            row_norm, col_norm,
            1.0, 1.0, 1.0, 1.0,
            norm_b_org, norm_c_org
        )

        scaling_info.norm_b = m > 0 ? CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf))) : 0.0
        scaling_info.norm_c = CUDA.norm(qp.c)

        return scaling_info
    end

    # For sparse Q, proceed with normal scaling
    # Initialize scaling vectors on GPU
    row_norm = CUDA.ones(Float64, m)
    col_norm = CUDA.ones(Float64, n)

    # Compute original norms for scaling info
    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    norm_b_org = CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf)
    norm_c_org = CUDA.norm(qp.c, Inf)

    # Initialize scaling info
    scaling_info = Scaling_info_gpu(
        copy(qp.l), copy(qp.u),
        row_norm, col_norm,
        1.0, 1.0, 1.0, 1.0,
        norm_b_org, norm_c_org
    )

    # Get CSR matrix components for A and AT
    A_rowPtr = qp.A.rowPtr
    A_colVal = qp.A.colVal
    A_nzVal = qp.A.nzVal
    AT_rowPtr = qp.AT.rowPtr
    AT_colVal = qp.AT.colVal
    AT_nzVal = qp.AT.nzVal

    # Temporary vectors for scaling
    temp_row_norm = CUDA.ones(Float64, m)
    temp_col_norm = CUDA.ones(Float64, n)

    Q_rowPtr = qp.Q.rowPtr
    Q_colVal = qp.Q.colVal
    Q_nzVal = qp.Q.nzVal

    # Ruiz scaling - only if Q is a sparse matrix
    if params.use_Ruiz_scaling
        for _ in 1:10
            # Compute column-wise max of |Q| and |A| combined
            @cuda threads = 256 blocks = ceil(Int, n / 256) compute_row_max_abs_with_Q_kernel!(
                AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
            )
            CUDA.synchronize()

            # Compute row-wise max of |A|
            if m > 0
                @cuda threads = 256 blocks = ceil(Int, m / 256) compute_row_max_abs_kernel!(
                    A_rowPtr, A_nzVal, temp_row_norm, m
                )
                CUDA.synchronize()
            end

            # Update cumulative norms
            row_norm .*= temp_row_norm
            col_norm .*= temp_col_norm

            # Scale Q if it's a sparse matrix
            # Scale Q: Q = DC * Q * DC (both rows and cols by temp_col_norm)
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_rows_csr_kernel!(
                Q_rowPtr, Q_nzVal, temp_col_norm, n
            )
            CUDA.synchronize()

            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_csr_cols_kernel!(
                Q_rowPtr, Q_colVal, Q_nzVal, temp_col_norm, n
            )
            CUDA.synchronize()

            # Scale A: A = DR * A * DC (rows by temp_row_norm, cols by temp_col_norm)
            if m > 0
                @cuda threads = 256 blocks = ceil(Int, m / 256) scale_rows_csr_kernel!(
                    A_rowPtr, A_nzVal, temp_row_norm, m
                )
                CUDA.synchronize()

                @cuda threads = 256 blocks = ceil(Int, m / 256) scale_csr_cols_kernel!(
                    A_rowPtr, A_colVal, A_nzVal, temp_col_norm, m
                )
                CUDA.synchronize()

                # Scale AT: AT = DC * AT * DR (rows by temp_col_norm, cols by temp_row_norm)
                @cuda threads = 256 blocks = ceil(Int, n / 256) scale_rows_csr_kernel!(
                    AT_rowPtr, AT_nzVal, temp_col_norm, n
                )
                CUDA.synchronize()

                @cuda threads = 256 blocks = ceil(Int, n / 256) scale_csr_cols_kernel!(
                    AT_rowPtr, AT_colVal, AT_nzVal, temp_row_norm, n
                )
                CUDA.synchronize()
            end

            # Scale objective and constraint bounds
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(
                qp.c, temp_col_norm, n
            )
            CUDA.synchronize()

            # Scale constraint bounds
            if m > 0
                @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
                    qp.AL, temp_row_norm, m
                )
                @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
                    qp.AU, temp_row_norm, m
                )
                CUDA.synchronize()
            end

            # Scale variable bounds
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
                qp.l, temp_col_norm, n
            )
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
                qp.u, temp_col_norm, n
            )
            CUDA.synchronize()
        end
    end

    # Pock-Chambolle scaling - only if Q is a sparse matrix
    if params.use_Pock_Chambolle_scaling
        # Compute column-wise sum of |Q| and |A| combined
        @cuda threads = 256 blocks = ceil(Int, n / 256) compute_col_sum_abs_with_Q_kernel!(
            AT_rowPtr, AT_nzVal, Q_rowPtr, Q_nzVal, temp_col_norm, n
        )
        CUDA.synchronize()

        # Compute row-wise sum of |A|
        if m > 0
            @cuda threads = 256 blocks = ceil(Int, m / 256) compute_row_sum_abs_kernel!(
                A_rowPtr, A_nzVal, temp_row_norm, m
            )
            CUDA.synchronize()
        end

        # Update cumulative norms
        row_norm .*= temp_row_norm
        col_norm .*= temp_col_norm

        # Scale Q if it's a sparse matrix
        # Scale Q: Q = DC * Q * DC
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_rows_csr_kernel!(
            Q_rowPtr, Q_nzVal, temp_col_norm, n
        )
        CUDA.synchronize()

        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_csr_cols_kernel!(
            Q_rowPtr, Q_colVal, Q_nzVal, temp_col_norm, n
        )
        CUDA.synchronize()

        # Scale A: A = DR * A * DC
        if m > 0
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_rows_csr_kernel!(
                A_rowPtr, A_nzVal, temp_row_norm, m
            )
            CUDA.synchronize()

            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_csr_cols_kernel!(
                A_rowPtr, A_colVal, A_nzVal, temp_col_norm, m
            )
            CUDA.synchronize()

            # Scale AT: AT = DC * AT * DR
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_rows_csr_kernel!(
                AT_rowPtr, AT_nzVal, temp_col_norm, n
            )
            CUDA.synchronize()

            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_csr_cols_kernel!(
                AT_rowPtr, AT_colVal, AT_nzVal, temp_row_norm, n
            )
            CUDA.synchronize()
        end

        # Scale objective and bounds
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_div_kernel!(
            qp.c, temp_col_norm, n
        )
        if m > 0
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
                qp.AL, temp_row_norm, m
            )
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_div_kernel!(
                qp.AU, temp_row_norm, m
            )
        end
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
            qp.l, temp_col_norm, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_mul_kernel!(
            qp.u, temp_col_norm, n
        )
        CUDA.synchronize()
    end

    # b and c scaling, dont use it for now
    if params.use_bc_scaling && false
        AL_nInf = copy(qp.AL)
        AU_nInf = copy(qp.AU)
        AL_nInf[qp.AL.==-Inf] .= 0.0
        AU_nInf[qp.AU.==Inf] .= 0.0
        b_scale = 1 + CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
        c_scale = 1 + CUDA.norm(qp.c)

        println("b_scale: ", b_scale)
        println("c_scale: ", c_scale)

        # Scale Q: Q *= b_scale / c_scale
        if Q_is_sparse
            scale_factor = b_scale / c_scale
            @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_mul_kernel!(
                Q_nzVal, scale_factor, length(Q_nzVal)
            )
            CUDA.synchronize()
        end

        if m > 0
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_scalar_div_kernel!(
                qp.AL, b_scale, m
            )
            @cuda threads = 256 blocks = ceil(Int, m / 256) scale_vector_scalar_div_kernel!(
                qp.AU, b_scale, m
            )
        end
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_div_kernel!(
            qp.c, c_scale, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_div_kernel!(
            qp.l, b_scale, n
        )
        @cuda threads = 256 blocks = ceil(Int, n / 256) scale_vector_scalar_div_kernel!(
            qp.u, b_scale, n
        )
        CUDA.synchronize()

        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    # Update diagonal of Q if Q is sparse
    # Extract diagonal from GPU Q matrix
    diag_Q = CUDA.zeros(Float64, n)
    # This is a simple approach - extract diagonal elements
    Q_cpu = SparseMatrixCSC(qp.Q)
    diag_Q_cpu = diag(Q_cpu)
    qp.diag_Q = CuVector(diag_Q_cpu)

    # Check if Q is diagonal
    temp_norm_Q = sum(abs, Q_cpu, dims=1)[1, :]
    Q_is_diag = (sum(temp_norm_Q .== abs.(diag_Q_cpu)) == length(temp_norm_Q))
    qp.Q_is_diag = Q_is_diag

    # Symmetrize Q to eliminate numerical errors
    Q_cpu = (Q_cpu + transpose(Q_cpu)) / 2
    qp.Q = CuSparseMatrixCSR(Q_cpu)

    # Compute final norms
    AL_nInf = copy(qp.AL)
    AU_nInf = copy(qp.AU)
    AL_nInf[qp.AL.==-Inf] .= 0.0
    AU_nInf[qp.AU.==Inf] .= 0.0
    scaling_info.norm_b = CUDA.norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = CUDA.norm(qp.c)

    # Store the cumulative scaling norms
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm

    return scaling_info
end

function mean(x::Vector{Float64})
    return sum(x) / length(x)
end

# ==================== Build Functions (Public API) ====================

"""
    build_from_mps(filename::String; verbose::Bool=true)

Build a QP model from an MPS file.

This function reads a QP problem from an MPS file and returns a CPU-based model
that can be solved with `optimize()` or `solve()`.

# Arguments
- `filename::String`: Path to the MPS file
- `verbose::Bool`: Whether to print progress information (default: true)

# Returns
- `QP_info_cpu`: QP model ready to be solved

# Example
```julia
using HPRQP

model = build_from_mps("problem.mps")
params = HPRQP_parameters()
result = optimize(model, params)
```

See also: [`build_from_QAbc`](@ref), [`build_from_mat`](@ref), [`optimize`](@ref)
"""
function build_from_mps(filename::String; verbose::Bool=true)
    t_start = time()
    if verbose
        println("READING FILE ... ", filename)
    end
    Q, c, A, lcon, ucon, lvar, uvar, c0 = read_mps(filename)
    read_time = time() - t_start
    if verbose
        println(@sprintf("READING FILE time: %.2f seconds", read_time))
    end

    t_start = time()
    if verbose
        println("FORMULATING QP ...")
    end
    standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, c0)
    if verbose
        println("QP formulation with C")
    end
    if verbose
        println(@sprintf("FORMULATING QP time: %.2f seconds", time() - t_start))
    end

    return standard_qp
end

"""
    build_from_QAbc(Q, c, A, AL, AU, l, u, obj_constant=0.0)

Build a QP model from matrix form.

This function creates a QP problem from the standard form:
    min  0.5 <x,Qx> + <c,x> + obj_constant
    s.t. AL <= Ax <= AU
         l <= x <= u

Accepts both sparse and dense matrices for Q and A. Dense matrices will be automatically 
converted to sparse format for efficient computation.

# Arguments
- `Q::Union{SparseMatrixCSC, Matrix{Float64}}`: Quadratic objective matrix (n × n). Can be sparse or dense.
- `c::Vector{Float64}`: Linear objective coefficients (length n)
- `A::Union{SparseMatrixCSC, Matrix{Float64}}`: Constraint matrix (m × n). Can be sparse or dense.
- `AL::Vector{Float64}`: Lower bounds for constraints Ax (length m)
- `AU::Vector{Float64}`: Upper bounds for constraints Ax (length m)
- `l::Vector{Float64}`: Lower bounds for variables x (length n)
- `u::Vector{Float64}`: Upper bounds for variables x (length n)
- `obj_constant::Float64`: Constant term in objective function (default: 0.0)

# Returns
- `QP_info_cpu`: QP model ready to be solved

# Example
```julia
using SparseArrays, HPRQP

# Example 1: Sparse matrices
Q = sparse([2.0 0.0; 0.0 2.0])
c = [-3.0, -5.0]
A = sparse([-1.0 -2.0; -3.0 -1.0])
AL = [-10.0, -12.0]
AU = [Inf, Inf]
l = [0.0, 0.0]
u = [Inf, Inf]

model = build_from_QAbc(Q, c, A, AL, AU, l, u)
params = HPRQP_parameters()
result = optimize(model, params)

# Example 2: Dense matrices (automatically converted)
n = 10
Q = zeros(n, n)  # Empty or dense Q matrix
Q[1,1] = 2.0
c = ones(n)
A = ones(5, n)  # Dense constraint matrix
AL = -Inf * ones(5)
AU = ones(5)
l = zeros(n)
u = ones(n)
model = build_from_QAbc(Q, c, A, AL, AU, l, u)
```

See also: [`build_from_mps`](@ref), [`build_from_mat`](@ref), [`optimize`](@ref)
"""
function build_from_QAbc(Q::Union{SparseMatrixCSC,Matrix{Float64}},
    c::Vector{Float64},
    A::Union{SparseMatrixCSC,Matrix{Float64}},
    AL::Vector{Float64},
    AU::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    obj_constant::Float64=0.0;
    verbose::Bool=true)

    # Convert dense matrices to sparse if needed
    if Q isa Matrix{Float64}
        if verbose
            println("Converting dense Q matrix to sparse format...")
        end
        Q = sparse(Q)
    end

    if A isa Matrix{Float64}
        if verbose
            println("Converting dense A matrix to sparse format...")
        end
        A = sparse(A)
    end

    # Create copies to avoid modifying the input
    Q = copy(Q)
    c = copy(c)
    A = copy(A)
    lcon = copy(AL)
    ucon = copy(AU)
    lvar = copy(l)
    uvar = copy(u)

    t_start = time()
    if verbose
        println("FORMULATING QP ...")
    end
    standard_qp = qp_formulation(Q, c, A, lcon, ucon, lvar, uvar, obj_constant)
    if verbose
        println("QP formulation with C")
    end
    if verbose
        println(@sprintf("FORMULATING QP time: %.2f seconds", time() - t_start))
    end

    return standard_qp
end

"""
    build_from_mat(filename::String; problem_type::String="QAP", lambda::Float64=1.0, verbose::Bool=true)

Build a QP model from a MAT file (for QAP or LASSO problems).

This function reads a QAP (Quadratic Assignment Problem) or LASSO problem from a .mat file.
Note: This function stores metadata that will be used to create operator-based models during solve().

# Arguments
- `filename::String`: Path to the MAT file
- `problem_type::String`: Type of problem - "QAP" or "LASSO" (default: "QAP")
- `lambda::Float64`: Regularization parameter for LASSO (default: 1.0)
- `verbose::Bool`: Whether to print progress information (default: true)

# Returns
- `Tuple`: (metadata_dict, problem_type) containing problem data

# Example
```julia
using HPRQP

model_info, prob_type = build_from_mat("qap_problem.mat", problem_type="QAP")
params = HPRQP_parameters()
params.problem_type = prob_type
result = optimize((model_info, prob_type), params)
```

See also: [`build_from_mps`](@ref), [`build_from_QAbc`](@ref), [`optimize`](@ref)
"""
function build_from_mat(filename::String; problem_type::String="QAP", verbose::Bool=true)
    t_start = time()
    if verbose
        println("READING FILE ... ", filename)
    end

    if problem_type == "QAP"
        # Read QAP data from .mat file
        QAPdata = matread(filename)
        A = Matrix{Float64}(QAPdata["A"])
        B = Matrix{Float64}(QAPdata["B"])
        S = Matrix{Float64}(QAPdata["S"])
        T = Matrix{Float64}(QAPdata["T"])

        read_time = time() - t_start
        if verbose
            println(@sprintf("READING FILE time: %.2f seconds", read_time))
        end

        # Use the new build_from_ABST function
        return build_from_ABST(A, B, S, T; verbose=verbose)

    elseif problem_type == "LASSO"
        # Read LASSO data from .mat file
        data = matread(filename)
        A_lasso = sparse(data["A"])
        b = vec(data["b"])

        read_time = time() - t_start
        if verbose
            println(@sprintf("READING FILE time: %.2f seconds", read_time))
        end

        # Set lambda based on ||A'b||_inf (common heuristic)
        lambda = 0.001 * norm(A_lasso' * b, Inf)
        if verbose
            println("Auto-selected lambda = ", lambda)
        end

        # Use the new build_from_Ab_lambda function
        return build_from_Ab_lambda(A_lasso, b, lambda; verbose=verbose)

    else
        error("Unknown problem_type: $problem_type. Supported types are 'QAP' and 'LASSO'.")
    end
end

"""
    build_from_ABST(A, B, S, T; verbose::Bool=true)

Build a QP model for Quadratic Assignment Problem (QAP) from matrices A, B, S, T.

This function creates a QAP problem in the standard form used by HPR-QP:
    min  <vec(X), Q*vec(X)>
    s.t. X*e = e, X'*e = e  (doubly stochastic constraints)
         X >= 0

Where Q(X) = 2*(A*X*B - S*X - X*T) is represented as a matrix-free operator using
the CUSTOM_Q_OPERATOR API.

# Arguments
- `A::Matrix{Float64}`: Distance matrix for facility locations (n × n)
- `B::Matrix{Float64}`: Flow matrix between facilities (n × n)
- `S::Matrix{Float64}`: Linear term for rows (n × n)
- `T::Matrix{Float64}`: Linear term for columns (n × n)
- `verbose::Bool`: Whether to print progress information (default: true)

# Returns
- `QP_info_cpu`: QP model ready to be solved with operator-based Q

# Example
```julia
using HPRQP

# Define QAP data matrices
n = 10
A = rand(n, n)
B = rand(n, n)
S = zeros(n, n)
T = zeros(n, n)

model = build_from_ABST(A, B, S, T)
params = HPRQP_parameters()
result = optimize(model, params)
```

See also: [`build_from_Ab_lambda`](@ref), [`build_from_mat`](@ref), [`build_from_QAbc`](@ref), [`optimize`](@ref)
"""
function build_from_ABST(A::Matrix{Float64}, B::Matrix{Float64},
    S::Matrix{Float64}, T::Matrix{Float64};
    verbose::Bool=true)
    t_start = time()

    # Validate dimensions
    n = size(A, 1)
    @assert size(A) == (n, n) "A must be square"
    @assert size(B) == (n, n) "B must be square and same size as A"
    @assert size(S) == (n, n) "S must be square and same size as A"
    @assert size(T) == (n, n) "T must be square and same size as A"

    if verbose
        println("FORMULATING QAP PROBLEM ...")
        println("QAP problem information: nRow = ", 2 * n, ", nCol = ", n^2)
    end

    # Create constraint matrix: each row is assigned exactly once, each column exactly once
    # Constraints: X*e = e, X'*e = e (doubly stochastic)
    ee = ones(Float64, n)
    Id = spdiagm(ones(Float64, n))
    A_constraint = sparse(vcat(kron(ee', Id), kron(Id, ee')))

    # Create the CPU operator struct (will be transferred to GPU via to_gpu interface)
    Q_cpu = QAP_Q_operator_cpu(A, B, S, T, n, zeros(Float64, n^2))

    # Create QP_info_cpu with Q operator stored directly in Q field
    # Use to_gpu(qp.Q) to transfer to GPU
    qp = QP_info_cpu(
        Q_cpu,  # Q operator (CPU version, use to_gpu to transfer)
        zeros(Float64, n^2),  # c vector (all zeros for QAP, linear term is in S and T)
        SparseMatrixCSC{Float64,Int32}(A_constraint),  # Constraint matrix
        SparseMatrixCSC{Float64,Int32}(A_constraint'),  # AT (transpose of constraint matrix)
        ones(Float64, 2 * n),  # AL = b (equality constraints)
        ones(Float64, 2 * n),  # AU = b (equality constraints)
        zeros(Float64, n^2),  # l (lower bounds = 0)
        fill(Inf, n^2),  # u (upper bounds = Inf)
        0.0,  # obj_constant
        Float64[],  # diag_Q (empty for operator)
        false,  # Q_is_diag
        0.0,  # lambda (not used for QAP)
    )

    if verbose
        println(@sprintf("FORMULATING QAP time: %.2f seconds", time() - t_start))
        println("Note: Operator-based Q will be created on GPU during solve()")
    end

    return qp
end

"""
    build_from_Ab_lambda(A, b, lambda; verbose::Bool=true)

Build a QP model for LASSO regression from data matrix A, target vector b, and regularization λ.

This function creates a LASSO problem in the standard form:
    min  0.5 ||A*x - b||₂² + λ ||x||₁

Which is reformulated as a QP with operator-based Q:
    min  0.5 <x, Q*x> + <c, x> + constant
    s.t. (no constraints on x, handled via proximal operator for L1)

Where Q = A'*A is represented as a matrix-free operator using the CUSTOM_Q_OPERATOR API.

# Arguments
- `A::SparseMatrixCSC{Float64}`: Data matrix (m × n)
- `b::Vector{Float64}`: Target vector (length m)
- `lambda::Float64`: Regularization parameter (must be positive)
- `verbose::Bool`: Whether to print progress information (default: true)

# Returns
- `QP_info_cpu`: QP model ready to be solved with operator-based Q

# Example
```julia
using HPRQP, SparseArrays

# Define LASSO data
m, n = 100, 50
A = sprandn(m, n, 0.1)
b = randn(m)
lambda = 0.01 * norm(A' * b, Inf)

model = build_from_Ab_lambda(A, b, lambda)
params = HPRQP_parameters()
result = optimize(model, params)
```

See also: [`build_from_ABST`](@ref), [`build_from_mat`](@ref), [`build_from_QAbc`](@ref), [`optimize`](@ref)
"""
function build_from_Ab_lambda(A::SparseMatrixCSC{Float64}, b::Vector{Float64},
    lambda::Float64; verbose::Bool=true)
    t_start = time()

    # Validate inputs
    m, n = size(A)
    @assert length(b) == m "Length of b must match number of rows in A"
    @assert lambda > 0 "Lambda must be positive"

    if verbose
        println("FORMULATING LASSO PROBLEM ...")
        println("LASSO problem information: m = ", m, ", n = ", n)
        println("Regularization lambda = ", lambda)
    end

    # Compute linear term c = -A'*b
    c = -A' * b

    # Objective constant: 0.5 * ||b||²
    obj_constant = 0.5 * norm(b)^2

    # Create the CPU operator struct (will be transferred to GPU via to_gpu interface)
    Q_cpu = LASSO_Q_operator_cpu(A, transpose(A), Vector{Float64}(b))

    # Create QP_info_cpu with Q operator stored directly in Q field
    # Use to_gpu(qp.Q) to transfer to GPU
    qp = QP_info_cpu(
        Q_cpu,  # Q operator (CPU version, use to_gpu to transfer)
        c,  # c vector = -A'*b
        SparseMatrixCSC{Float64,Int32}(spzeros(0, n)),  # No constraints (empty matrix)
        SparseMatrixCSC{Float64,Int32}(spzeros(n, 0)),  # AT (transpose of empty constraint matrix)
        Float64[],  # AL (empty)
        Float64[],  # AU (empty)
        -Inf * ones(Float64, n),  # l (lower bounds = -Inf, no box constraints)
        Inf * ones(Float64, n),  # u (upper bounds = Inf, no box constraints)
        obj_constant,  # obj_constant = 0.5 * ||b||²
        Float64[],  # diag_Q (empty for operator)
        false,  # Q_is_diag
        lambda,  # lambda for LASSO regularization
    )

    if verbose
        println(@sprintf("FORMULATING LASSO time: %.2f seconds", time() - t_start))
        println("Note: Operator-based Q will be created on GPU during solve()")
    end

    return qp
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
# Supports both CuSparseMatrixCSR and AbstractQOperator (LASSO, QAP, custom operators)
function power_iteration_Q_gpu(Q::Union{CuSparseMatrixCSR, AbstractQOperator}, max_iterations=5000, tolerance=1e-4)
    seed = 1
    n = get_problem_size(Q)
    z = CuVector(randn(Random.MersenneTwister(seed), n)) .+ 1e-8 # Initial random vector
    q = CUDA.zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= CUDA.norm(q)
        Qmap!(q, z, Q)  # Uses operator's Qmap! and temp storage
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

# Run the dataset of QP problems from a specified directory and save the results to a CSV file
function run_dataset(data_path::String, result_path::String, params::HPRQP_parameters)

    files = readdir(data_path)

    # Specify the path and filename for the CSV file
    csv_file = joinpath(result_path, "HPRQP_result.csv")

    # redirect the output to a file
    log_path = joinpath(result_path, "HPRQP_log.txt")

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

    for i = 1:length(files)
        file = files[i]
        if occursin(".mps", file) && !(file in namelist)
            FILE_NAME = joinpath(data_path, file)
            println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))

            redirect_stdout(io) do
                println(@sprintf("solving the problem %d", i), @sprintf(": %s", file))
                println("main run starts: ----------------------------------------------------------------------------------------------------------")
                t_start_all = time()
                model = build_from_mps(FILE_NAME, verbose=true)
                results = optimize(model, params)
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
                push!(statuslist, results.status)
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

# ==================== QAP and LASSO Support Functions ====================

# Scaling function for operator-based QP problems (QAP/LASSO)
function scaling!(qp::QP_info_gpu, params::HPRQP_parameters)
    m, n = size(qp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    AL_nInf = Array(qp.AL)
    AU_nInf = Array(qp.AU)
    AL_nInf[AL_nInf.==-Inf] .= 0.0
    AU_nInf[AU_nInf.==Inf] .= 0.0
    c_arr = Array(qp.c)
    l_arr = Array(qp.l)

    scaling_info = Scaling_info_gpu(CuVector(copy(l_arr)), CuVector(copy(l_arr)),
        CuVector(row_norm), CuVector(col_norm),
        1.0, 1.0, 1.0, 1.0,
        norm(max.(abs.(AL_nInf), abs.(AU_nInf)), Inf),
        norm(c_arr, Inf))

    if params.use_bc_scaling
        b_norm = norm(min.(AL_nInf, AU_nInf))
        c_norm = norm(c_arr)
        b_scale = max(1.0, b_norm, c_norm)
        c_scale = b_scale

        qp.AL ./= b_scale
        qp.AU ./= b_scale
        qp.c ./= c_scale
        if length(qp.lambda) > 0
            qp.lambda ./= c_scale
        end
        qp.l ./= b_scale
        qp.u ./= b_scale
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    AL_nInf = Array(qp.AL)
    AU_nInf = Array(qp.AU)
    AL_nInf[AL_nInf.==-Inf] .= 0.0
    AU_nInf[AU_nInf.==Inf] .= 0.0
    c_arr = Array(qp.c)
    scaling_info.norm_b = norm(max.(abs.(AL_nInf), abs.(AU_nInf)))
    scaling_info.norm_c = norm(c_arr)
    scaling_info.row_norm = CuVector(row_norm)
    scaling_info.col_norm = CuVector(col_norm)

    return scaling_info
end

# ============================================================================
# CPU Power Iteration Functions
# ============================================================================

# Power iteration for CPU: A matrix (constraint matrix)
function power_iteration_A_cpu(A::SparseMatrixCSC, AT::SparseMatrixCSC,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8  # Initial random vector
    q = zeros(Float64, m)
    ATq = zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0

    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        mul!(ATq, AT, q)
        mul!(z, A, ATq)
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q  # error vector
        error = norm(q) / (norm(z) + lambda_max)

        if error < tolerance
            return lambda_max
        end
    end

    println("Power iteration (A) did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# Power iteration for CPU: Q sparse matrix
function power_iteration_Q_cpu(Q::Union{SparseMatrixCSC, AbstractQOperatorCPU}, max_iterations=5000, tolerance=1e-4)
    seed = 1
    n = get_problem_size(Q)
    z = randn(Random.MersenneTwister(seed), n) .+ 1e-8 # Initial random vector
    q = zeros(Float64, n)
    lambda_max = 1.0
    error = 1.0
    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        Qmap!(q, z, Q)  # Uses operator's Qmap! and temp storage
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q        # error 
        error = norm(q) / (norm(z) + lambda_max)

        if error < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", error)
    return lambda_max
end

# ============================================================================
# CUSPARSE SpMV Preprocessing and Buffer Allocation
# ============================================================================

"""
    prepare_spmv_A!(A, AT, x_bar, x_hat, dx, Ax, y_bar, y, ATy)

Prepare CUSPARSE SpMV operations for A and AT matrices.
Allocates buffers and performs preprocessing (for CUDA >= 12.4).

# Arguments
- `A::CuSparseMatrixCSR`: The constraint matrix in CSR format
- `AT::CuSparseMatrixCSR`: The transpose of A in CSR format
- `x_bar, x_hat, dx::CuVector{Float64}`: Dense vectors for A operations
- `Ax::CuVector{Float64}`: Output vector for A*x
- `y_bar, y::CuVector{Float64}`: Dense vectors for AT operations
- `ATy::CuVector{Float64}`: Output vector for AT*y

# Returns
- `(spmv_A, spmv_AT)`: Tuple of CUSPARSE_spmv_A and CUSPARSE_spmv_AT structures
"""
function prepare_spmv_A!(A::CuSparseMatrixCSR{Float64,Int32},
    AT::CuSparseMatrixCSR{Float64,Int32},
    x_bar::CuVector{Float64},
    x_hat::CuVector{Float64},
    dx::CuVector{Float64},
    Ax::CuVector{Float64},
    y_bar::CuVector{Float64},
    y::CuVector{Float64},
    ATy::CuVector{Float64})
    # Create matrix and vector descriptors
    desc_A = CUDA.CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
    desc_x_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(x_bar)
    desc_x_hat = CUDA.CUSPARSE.CuDenseVectorDescriptor(x_hat)
    desc_dx = CUDA.CUSPARSE.CuDenseVectorDescriptor(dx)
    desc_Ax = CUDA.CUSPARSE.CuDenseVectorDescriptor(Ax)

    desc_AT = CUDA.CUSPARSE.CuSparseMatrixDescriptor(AT, 'O')
    desc_y_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(y_bar)
    desc_y = CUDA.CUSPARSE.CuDenseVectorDescriptor(y)
    desc_ATy = CUDA.CUSPARSE.CuDenseVectorDescriptor(ATy)

    CUSPARSE_handle = CUDA.CUSPARSE.handle()
    ref_one = Ref{Float64}(one(Float64))
    ref_zero = Ref{Float64}(zero(Float64))

    # Prepare A SpMV
    sz_A = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, ref_zero,
        desc_Ax, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_A)
    buf_A = CUDA.CuArray{UInt8}(undef, sz_A[])

    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, ref_zero, desc_Ax,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)
    end

    spmv_A = CUSPARSE_spmv_A(CUSPARSE_handle, 'N', ref_one, desc_A, desc_x_bar, desc_x_hat, desc_dx,
        ref_zero, desc_Ax, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_A)

    # Prepare AT SpMV
    sz_AT = Ref{Csize_t}(0)
    CUDA.CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, ref_zero,
        desc_ATy, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, sz_AT)
    buf_AT = CUDA.CuArray{UInt8}(undef, sz_AT[])

    # Only call preprocess for CUDA >= 12.4
    if CUDA.CUSPARSE.version() >= v"12.4"
        CUDA.CUSPARSE.cusparseSpMV_preprocess(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, ref_zero, desc_ATy,
            Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)
    end

    spmv_AT = CUSPARSE_spmv_AT(CUSPARSE_handle, 'N', ref_one, desc_AT, desc_y_bar, desc_y,
        ref_zero, desc_ATy, Float64, CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2, buf_AT)

    return spmv_A, spmv_AT
end

# Note: prepare_spmv_Q! is now in Q_operators/sparse_matrix_operator.jl

# ============================================================================
# Helper Functions for CUSPARSE SpMV Operations
# ============================================================================

"""
    spmv_A_operation!(ws, vec_in, vec_out)

Perform A * vec_in -> vec_out using preprocessed CUSPARSE if available.
Falls back to standard CUSPARSE.mv! if preprocessing not available.

# Note
This uses ws.spmv_A which contains the preprocessed buffer and descriptors.
The descriptor for vec_in is determined by which descriptor matches the input vector.
"""
function spmv_A_operation!(ws::HPRQP_workspace_gpu, vec_in::CuVector{Float64}, vec_out::CuVector{Float64})
    if ws.spmv_A !== nothing
        # Use preprocessed CUSPARSE spmv - need to determine which descriptor to use
        # The spmv_A struct has desc_x_bar, desc_x_hat, desc_dx
        # We'll use cusparseSpMV with the appropriate descriptor
        # For simplicity, create a temporary descriptor for the input vector
        desc_in = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_in)
        desc_out = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_out)
        CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha,
            ws.spmv_A.desc_A, desc_in, ws.spmv_A.beta, desc_out,
            ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
    else
        CUDA.CUSPARSE.mv!('N', 1, ws.A, vec_in, 0, vec_out, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
end

"""
    spmv_AT_operation!(ws, vec_in, vec_out)

Perform AT * vec_in -> vec_out using preprocessed CUSPARSE if available.
Falls back to standard CUSPARSE.mv! if preprocessing not available.
"""
function spmv_AT_operation!(ws::HPRQP_workspace_gpu, vec_in::CuVector{Float64}, vec_out::CuVector{Float64})
    if ws.spmv_AT !== nothing
        desc_in = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_in)
        desc_out = CUDA.CUSPARSE.CuDenseVectorDescriptor(vec_out)
        CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha,
            ws.spmv_AT.desc_AT, desc_in, ws.spmv_AT.beta, desc_out,
            ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
    else
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, vec_in, 0, vec_out, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
end

# ============================================================================
# MOI Cache Definition for JuMP Integration
# ============================================================================

MOI.Utilities.@product_of_sets(
    QPSets,
    MOI.EqualTo{T},
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.Interval{T},
)

# Define the cache type with MatrixOfConstraints for QP (similar to HPRLP)
const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.OneBasedIndexing,
        },
        MOI.Utilities.Hyperrectangle{Float64},
        QPSets{Float64},
    },
}
