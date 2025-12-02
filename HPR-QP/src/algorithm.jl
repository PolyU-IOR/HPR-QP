### This file contains the main algorithm for solving quadratic programming problems using the HPR-QP method on GPU.

# The package is used to solve convex quadratic programming (QP) with HPR method in the paper 
# HPR-QP: A dual Halpern Peaceman–Rachford method for solving large scale convex composite quadratic programming
# The package is developed by Kaihuang Chen · Defeng Sun · Yancheng Yuan · Guojun Zhang · Xinyuan Zhao.

# Quadratic Programming (QP) problem formulation:
# 
#     minimize    (1/2) x' Q x + c' x
#     subject to  AL ≤ Ax ≤ AU
#                   l ≤ x ≤ u
#
# where:
#   - Q is a symmetric positive semidefinite matrix (n x n)
#   - c is a vector (n)
#   - A is a constraint matrix (m x n)
#   - l, u are vectors (m), lower and upper bounds for constraints
#   - x is the variable vector (n)
#

# This function computes the residuals for the HPR-QP algorithm on GPU.
function compute_residuals_gpu(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int,
)
    ### Objective values
    # Use unified Qmap! function (supports both operators and sparse matrices via dispatch)
    Qmap!(ws.x_bar, ws.Qx, qp.Q)
    res.primal_obj_bar = sc.b_scale * sc.c_scale *
                         (CUDA.dot(ws.c, ws.x_bar) + 0.5 * CUDA.dot(ws.x_bar, ws.Qx)) + qp.obj_constant

    # Dual objective: always include z'x term (for bounds/L1), conditionally add y'(Ax-b) term
    res.dual_obj_bar = sc.b_scale * sc.c_scale *
                       (-0.5 * CUDA.dot(ws.x_bar, ws.Qx) + CUDA.dot(ws.z_bar, ws.x_bar)) + qp.obj_constant
    if ws.m > 0
        res.dual_obj_bar += sc.b_scale * sc.c_scale * CUDA.dot(ws.y_bar, ws.s)
    end

    # Add L1 norm term for LASSO problems
    if params.problem_type == "LASSO" && length(ws.lambda) > 0
        ws.tempv .= ws.lambda .* ws.x_bar
        l1_norm = CUDA.norm(ws.tempv, 1)
        res.primal_obj_bar += sc.b_scale * sc.c_scale * l1_norm
        res.dual_obj_bar += sc.b_scale * sc.c_scale * l1_norm
    end

    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1.0 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))

    ### Dual residuals
    # For LASSO, we need to use compute_Rd_gpu! even when noC=true to include z term (subdifferential of L1)
    if qp.noC
        compute_Rd_noC_gpu!(ws, sc)
    else
        compute_Rd_gpu!(ws, sc)
    end
    res.err_Rd_org_bar = CUDA.norm(ws.Rd, Inf) / (1.0 + maximum([sc.norm_c_org, CUDA.norm(ws.ATdy, Inf), CUDA.norm(ws.Qx, Inf)]))

    ### Rp
    if ws.m > 0
        compute_Rp_gpu!(ws, sc)
        res.err_Rp_org_bar = CUDA.norm(ws.Rp, Inf) / (1.0 + max(sc.norm_b_org, CUDA.norm(ws.Ax, Inf)))
    else
        res.err_Rp_org_bar = 0.0
    end

    if iter == 0
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_err_lu_kernel!(ws.dx, ws.x_bar, ws.l, ws.u, sc.col_norm, sc.b_scale, ws.n)
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, CUDA.norm(ws.dx, Inf))
    end
    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)

    # Save best values if auto_save is enabled
    if params.auto_save
        if iter == 0 || res.KKTx_and_gap_org_bar < max(ws.saved_state.save_err_Rp, ws.saved_state.save_err_Rd, ws.saved_state.save_rel_gap)
            ws.saved_state.save_x .= ws.x_bar
            ws.saved_state.save_y .= ws.y_bar
            ws.saved_state.save_z .= ws.z_bar
            ws.saved_state.save_w .= ws.w_bar
            ws.saved_state.save_sigma = ws.sigma
            ws.saved_state.save_iter = iter
            ws.saved_state.save_err_Rp = res.err_Rp_org_bar
            ws.saved_state.save_err_Rd = res.err_Rd_org_bar
            ws.saved_state.save_primal_obj = res.primal_obj_bar
            ws.saved_state.save_dual_obj = res.dual_obj_bar
            ws.saved_state.save_rel_gap = res.rel_gap_bar
        end
    end

end

# This function saves the current and best-so-far state to an HDF5 file
# It is called whenever the log is printed (if auto_save is enabled)
# It saves:
#   - Current solution (x_bar, y_bar, z_bar, w_bar) - scaled to original problem
#   - Best solution so far (save_x, save_y, save_z, save_w) - scaled to original problem
#   - Current and best sigma values
#   - Current and best residuals, objectives, and iteration numbers
#   - Current iteration number and elapsed time
#   - All solver parameters including initial solutions
function save_state_to_hdf5(
    filename::String,
    ws::HPRQP_workspace_gpu,
    sc::Scaling_info_gpu,
    residuals::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int,
    t_start_alg::Float64,
)
    # Convert GPU arrays to CPU
    x_bar = Vector(ws.x_bar)
    y_bar = Vector(ws.y_bar)
    z_bar = Vector(ws.z_bar)
    w_bar = Vector(ws.w_bar)
    save_x = Vector(ws.saved_state.save_x)
    save_y = Vector(ws.saved_state.save_y)
    save_z = Vector(ws.saved_state.save_z)
    save_w = Vector(ws.saved_state.save_w)
    col_norm = Vector(sc.col_norm)
    row_norm = Vector(sc.row_norm)

    # Scale the variables (same as in collect_results)
    x_bar_scaled = sc.b_scale * (x_bar ./ col_norm)
    w_bar_scaled = sc.b_scale * (w_bar ./ col_norm)
    save_x_scaled = sc.b_scale * (save_x ./ col_norm)
    save_w_scaled = sc.b_scale * (save_w ./ col_norm)
    
    if ws.m > 0
        y_bar_scaled = sc.c_scale * (y_bar ./ row_norm)
        save_y_scaled = sc.c_scale * (save_y ./ row_norm)
        z_bar_scaled = sc.c_scale * (z_bar ./ col_norm)
        save_z_scaled = sc.c_scale * (save_z ./ col_norm)
    else
        # For problems without constraints (m=0), y and dual variables are empty or not used
        y_bar_scaled = Float64[]
        save_y_scaled = Float64[]
        z_bar_scaled = sc.c_scale * (z_bar ./ col_norm)
        save_z_scaled = sc.c_scale * (save_z ./ col_norm)
    end

    # Create or open HDF5 file
    if isfile(filename)
        rm(filename, force=true)
    end
    h5open(filename, "w") do file
        # Save current iteration info
        file["current/iteration"] = iter
        file["current/time_elapsed"] = time() - t_start_alg
        file["current/timestamp"] = string(Dates.now())

        # Save current solution (scaled)
        file["current/x_org"] = x_bar_scaled
        file["current/w_org"] = w_bar_scaled
        if ws.m > 0
            file["current/y_org"] = y_bar_scaled
        end
        file["current/z_org"] = z_bar_scaled
        file["current/sigma"] = ws.sigma

        # Save current residuals
        file["current/err_Rp"] = residuals.err_Rp_org_bar
        file["current/err_Rd"] = residuals.err_Rd_org_bar
        file["current/primal_obj"] = residuals.primal_obj_bar
        file["current/dual_obj"] = residuals.dual_obj_bar
        file["current/rel_gap"] = residuals.rel_gap_bar
        file["current/KKTx_and_gap"] = residuals.KKTx_and_gap_org_bar

        # Save best solution so far (scaled)
        file["best/x_org"] = save_x_scaled
        file["best/w_org"] = save_w_scaled
        if ws.m > 0
            file["best/y_org"] = save_y_scaled
        end
        file["best/z_org"] = save_z_scaled
        file["best/sigma"] = ws.saved_state.save_sigma
        file["best/iteration"] = ws.saved_state.save_iter

        # Save best residuals
        file["best/err_Rp"] = ws.saved_state.save_err_Rp
        file["best/err_Rd"] = ws.saved_state.save_err_Rd
        file["best/primal_obj"] = ws.saved_state.save_primal_obj
        file["best/dual_obj"] = ws.saved_state.save_dual_obj
        file["best/rel_gap"] = ws.saved_state.save_rel_gap

        # Save parameters
        file["parameters/stoptol"] = params.stoptol
        file["parameters/sigma"] = params.sigma
        file["parameters/max_iter"] = params.max_iter
        file["parameters/sigma_fixed"] = params.sigma_fixed
        file["parameters/time_limit"] = params.time_limit
        file["parameters/eig_factor"] = params.eig_factor
        file["parameters/check_iter"] = params.check_iter
        file["parameters/warm_up"] = params.warm_up
        file["parameters/spmv_mode_Q"] = params.spmv_mode_Q
        file["parameters/spmv_mode_A"] = params.spmv_mode_A
        file["parameters/print_frequency"] = params.print_frequency
        file["parameters/device_number"] = params.device_number
        file["parameters/use_Ruiz_scaling"] = params.use_Ruiz_scaling
        file["parameters/use_bc_scaling"] = params.use_bc_scaling
        file["parameters/use_l2_scaling"] = params.use_l2_scaling
        file["parameters/use_Pock_Chambolle_scaling"] = params.use_Pock_Chambolle_scaling
        file["parameters/problem_type"] = params.problem_type
        file["parameters/lambda"] = params.lambda
        file["parameters/auto_save"] = params.auto_save

        # Save initial solutions if provided
        if params.initial_x !== nothing
            file["parameters/initial_x"] = params.initial_x
        end
        if params.initial_y !== nothing
            file["parameters/initial_y"] = params.initial_y
        end
    end
end

# This function updates the penalty parameter (sigma) based on the current state of the algorithm.
function update_sigma(params::HPRQP_parameters,
    restart_info::HPRQP_restart,
    ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    Q_is_diag::Bool,
    noC::Bool,
)
    if ~params.sigma_fixed && (restart_info.restart_flag >= 1)
        sigma_old = ws.sigma
        if noC
            axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
            axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
            Qmap!(ws.dw, ws.dQw, qp.Q)

            primal_move = CUDA.dot(ws.dx, ws.dx)
            dual_move = 0.0

            if ws.m > 0
                axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
                dual_move += ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
            end

            if Q_is_diag
                # dual_move already includes A term if ws.m > 0
            else
                dual_move += ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw) - CUDA.dot(ws.dQw, ws.dQw)
            end

            primal_move = max(primal_move, 1e-12)
            dual_move = max(dual_move, 1e-12)
            sigma_new = sqrt(primal_move / dual_move)
            fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
            ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))
        else
            axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
            axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
            # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
            Qmap!(ws.dw, ws.dQw, qp.Q)

            a = 0.0
            b = CUDA.dot(ws.dx, ws.dx)
            c = 0.0
            d = 0.0

            if ws.m > 0
                axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
                axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.last_ATy, ws.ATdy, ws.n)
                # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
                Qmap!(ws.ATdy, ws.QATdy, qp.Q)
                a = ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy) - 2 * CUDA.dot(ws.dQw, ws.ATdy)
            end

            if Q_is_diag
                # if Q_is_diag
                a += CUDA.norm(ws.dQw)^2
            else
                a += ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
                if ws.m > 0
                    c = CUDA.dot(ws.ATdy, ws.QATdy)
                    d = ws.lambda_max_Q
                end
            end
            a = max(a, 1e-12)
            b = max(b, 1e-12)
            if Q_is_diag
                if ws.m > 0
                    sigma_new = golden_Q_diag(a, b, ws.diag_Q, ws.ATdy, ws.QATdy, ws.tempv; lo=1e-12, hi=1e12, tol=1e-13)
                else
                    # No constraints: simplified sigma update for diagonal Q
                    sigma_new = sqrt(b / a)
                end
            else
                # min a * x + b / x + c * x^2 / (1 + d * x)
                if ws.m > 0
                    sigma_new = golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
                else
                    # No constraints: simplified sigma update
                    sigma_new = sqrt(b / a)
                end
            end
            fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
            ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))
        end

        # update Q factors if sigma changes
        if Q_is_diag
            if abs(sigma_old - ws.sigma) > 1e-15
                update_Q_factors_gpu!(
                    ws.fact2, ws.fact, ws.fact1, ws.fact_M,
                    ws.diag_Q, ws.sigma
                )
            end
        end
    end

end

# This function checks whether a restart is needed based on the current state of the algorithm.
function check_restart(restart_info::HPRQP_restart,
    iter::Int,
    check_iter::Int,
)
    restart_info.restart_flag = 0
    if restart_info.first_restart
        if iter == check_iter
            restart_info.first_restart = false
            restart_info.restart_flag = 1
            restart_info.weighted_norm = restart_info.current_gap
        end
    else
        if rem(iter, check_iter) == 0

            if restart_info.current_gap <= 0.2 * restart_info.last_gap
                restart_info.sufficient += 1
                restart_info.restart_flag = 1
            end

            if (restart_info.current_gap <= 0.8 * restart_info.last_gap) && (restart_info.current_gap > 1.00 * restart_info.save_gap)
                restart_info.necessary += 1
                restart_info.restart_flag = 2
            end

            if restart_info.current_gap / restart_info.weighted_norm > 1e-1
                fact = 0.5
            else
                fact = 0.2
            end

            if restart_info.inner >= fact * iter
                restart_info.long += 1
                restart_info.restart_flag = 3
            end
            restart_info.save_gap = restart_info.current_gap
        end
    end
end

# This function performs the restart for the HPR-QP algorithm on GPU.
function do_restart(restart_info::HPRQP_restart, ws::HPRQP_workspace_gpu, qp::QP_info_gpu, noC::Bool)
    if restart_info.restart_flag > 0
        ws.x .= ws.x_bar
        ws.y .= ws.y_bar
        ws.w .= ws.w_bar
        ws.last_x .= ws.x_bar
        ws.last_y .= ws.y_bar
        ws.last_w .= ws.w_bar
        if !noC
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            ws.last_ATy .= ws.ATy_bar
            ws.ATy .= ws.ATy_bar
        else
            # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q)
            ws.Qw .= ws.Qw_bar
            ws.last_Qw .= ws.Qw_bar
        end
        restart_info.last_gap = restart_info.current_gap
        restart_info.save_gap = Inf
        restart_info.times += 1
        restart_info.inner = 0
    end
end

# This function checks the stopping criteria for the HPR-QP algorithm on GPU.
function check_break(residuals::HPRQP_residuals,
    iter::Int,
    t_start_alg::Float64,
    params::HPRQP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "MAX_ITER"
    end

    if time() - t_start_alg > params.time_limit
        return "TIME_LIMIT"
    end

    return "CONTINUE"
end

# This function collects the results from the HPR-QP algorithm on GPU and prepares them for output.
function collect_results_gpu!(
    ws::HPRQP_workspace_gpu,
    residuals::HPRQP_residuals,
    sc::Scaling_info_gpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
)
    results = HPRQP_results()
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    ### copy the results to the CPU ### 
    results.w = Vector(sc.b_scale * (ws.w_bar ./ sc.col_norm))
    results.x = Vector(sc.b_scale * (ws.x_bar ./ sc.col_norm))
    results.y = Vector(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    results.z = Vector(sc.c_scale * (ws.z_bar .* sc.col_norm))
    return results
end

# This function allocates the workspace for the HPR-QP algorithm on GPU.
function allocate_workspace_gpu(qp::QP_info_gpu,
    params::HPRQP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64,
    scaling_info::Scaling_info_gpu,
)
    ws = HPRQP_workspace_gpu()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n
    if params.sigma == -1
        norm_b = scaling_info.norm_b
        norm_c = scaling_info.norm_c
        if norm_c > 1e-16 && norm_b > 1e-16 && norm_b < 1e16 && norm_c < 1e16
            ws.sigma = norm_b / norm_c
        else
            ws.sigma = 1.0
        end
    elseif params.sigma > 0
        ws.sigma = params.sigma
    else
        error("Invalid sigma value: ", params.sigma, ". It should be a positive number or -1 for automatic.")
    end
    # println("initial sigma = ", ws.sigma)
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q
    ws.diag_Q = qp.diag_Q
    ws.w = CUDA.zeros(Float64, n)
    ws.w_hat = CUDA.zeros(Float64, n)
    ws.w_bar = CUDA.zeros(Float64, n)
    ws.dw = CUDA.zeros(Float64, n)
    ws.x = CUDA.zeros(Float64, n)
    ws.x_hat = CUDA.zeros(Float64, n)
    ws.x_bar = CUDA.zeros(Float64, n)
    ws.dx = CUDA.zeros(Float64, n)
    ws.y = CUDA.zeros(Float64, m)
    ws.y_hat = CUDA.zeros(Float64, m)
    ws.y_bar = CUDA.zeros(Float64, m)
    ws.dy = CUDA.zeros(Float64, m)
    ws.s = CUDA.zeros(Float64, m)
    ws.z_bar = CUDA.zeros(Float64, n)
    ws.Q = qp.Q
    ws.A = qp.A
    ws.AT = qp.AT
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u
    if ws.m > 0
        ws.AL[ws.AL.==-Inf] .= -1e20
        ws.AU[ws.AU.==Inf] .= 1e20
    end
    ws.l[ws.l.==-Inf] .= -1e20
    ws.u[ws.u.==Inf] .= 1e20
    ws.Rp = CUDA.zeros(Float64, m)
    ws.Rd = CUDA.zeros(Float64, n)
    ws.ATy = CUDA.zeros(Float64, n)
    ws.ATy_bar = CUDA.zeros(Float64, n)
    ws.ATdy = CUDA.zeros(Float64, n)
    ws.QATdy = CUDA.zeros(Float64, n)
    ws.Ax = CUDA.zeros(Float64, m)
    ws.Qw = CUDA.zeros(Float64, n)
    ws.Qw_bar = CUDA.zeros(Float64, n)
    ws.Qw_hat = CUDA.zeros(Float64, n)
    ws.Qx = CUDA.zeros(Float64, n)
    ws.dQw = CUDA.zeros(Float64, n)
    ws.last_x = CUDA.zeros(Float64, n)
    ws.last_y = CUDA.zeros(Float64, m)
    ws.last_Qw = CUDA.zeros(Float64, n)
    ws.last_ATy = CUDA.zeros(Float64, n)
    ws.last_w = CUDA.zeros(Float64, n)
    ws.tempv = CUDA.zeros(Float64, n)
    ws.fact1 = CUDA.zeros(Float64, n)
    ws.fact2 = CUDA.zeros(Float64, n)
    ws.fact = CUDA.zeros(Float64, n)
    ws.fact_M = CUDA.zeros(Float64, n)

    # Copy lambda for LASSO problems
    ws.lambda = qp.lambda

    # Initialize saved_state for auto_save feature
    if params.auto_save
        ws.saved_state = HPRQP_saved_state_gpu()
        ws.saved_state.save_x = CUDA.zeros(Float64, n)
        ws.saved_state.save_y = CUDA.zeros(Float64, m)
        ws.saved_state.save_z = CUDA.zeros(Float64, n)
        ws.saved_state.save_w = CUDA.zeros(Float64, n)
        ws.saved_state.save_sigma = ws.sigma
        ws.saved_state.save_iter = 0
        ws.saved_state.save_err_Rp = Inf
        ws.saved_state.save_err_Rd = Inf
        ws.saved_state.save_primal_obj = Inf
        ws.saved_state.save_dual_obj = Inf
        ws.saved_state.save_rel_gap = Inf
    end

    # Initialize with user-provided initial x if available
    if params.initial_x !== nothing
        # Copy initial_x to GPU first
        initial_x_gpu = CuArray(params.initial_x)

        # Scale x on GPU: inverse of x_result = b_scale * (x_bar / col_norm)
        # So x_bar = x_input * col_norm / b_scale
        scaled_x = initial_x_gpu .* scaling_info.col_norm ./ scaling_info.b_scale
        
        # Initialize x and related variables
        ws.x .= scaled_x
        ws.x_bar .= scaled_x
        ws.last_x .= scaled_x
        
        # Initialize w and related variables with the same initial x
        ws.w .= scaled_x
        ws.w_bar .= scaled_x
        ws.last_w .= scaled_x
    end

    # Initialize with user-provided initial y if available
    if params.initial_y !== nothing
        # Copy initial_y to GPU first
        ws.y .= CuArray(params.initial_y)

        # Scale y on GPU: inverse of y_result = c_scale * (y_bar / row_norm)
        # So y_bar = y_input * row_norm / c_scale
        ws.y .= ws.y .* scaling_info.row_norm ./ scaling_info.c_scale
        ws.y_bar .= ws.y
        ws.last_y .= ws.y

        # Compute ATy_bar from y_bar
        if m > 0
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            ws.ATy .= ws.ATy_bar
            ws.last_ATy .= ws.ATy_bar
        end
    end

    return ws
end

# This function initializes the restart information for the HPR-QP algorithm.
function initialize_restart()
    restart_info = HPRQP_restart()
    restart_info.first_restart = true
    restart_info.save_gap = Inf
    restart_info.current_gap = Inf
    restart_info.last_gap = Inf
    restart_info.inner = 0
    restart_info.times = 0
    restart_info.sufficient = 0
    restart_info.necessary = 0
    restart_info.long = 0
    restart_info.ratio = 0
    restart_info.restart_flag = 0
    restart_info.weighted_norm = Inf
    return restart_info
end

function print_step(iter::Int)
    return max(10^floor(log10(iter)) / 10, 10)
end

# This function updates the variables in the HPR-QP algorithm, when Q is diagonal, there's no proximal term on w;
# when the problem is formulated without l≤x≤u, the update is w->x->y.
function main_update!(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    spmv_mode_Q::String,
    spmv_mode_A::String,
    restart_info::HPRQP_restart,
)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    Halpern_fact2 = 1.0 - Halpern_fact1

    # Handle operator-based Q (QAP/LASSO) within main_update
    if spmv_mode_Q == "operator"
        # Use operator-specific updates with configurable A mode
        if isa(qp.Q, LASSO_Q_operator_gpu)
            # LASSO update with soft-thresholding (no A matrix for LASSO)
            update_zxw_LASSO_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        elseif isa(qp.Q, QAP_Q_operator_gpu)
            # QAP update using unified kernels (operator mode for Q, configurable mode for A)
            unified_update_zxw1_gpu!(ws, qp, Halpern_fact1, Halpern_fact2;
                spmv_mode_Q="operator", spmv_mode_A=spmv_mode_A, is_diag_Q=false)
            unified_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2;
                spmv_mode_Q="operator", spmv_mode_A=spmv_mode_A)
            unified_update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2;
                spmv_mode_Q="operator", spmv_mode_A=spmv_mode_A, is_diag_Q=false)
        else
            error("Unknown Q operator type: $(typeof(qp.Q))")
        end
        return
    end

    # Standard sparse matrix Q case
    if qp.noC
        # For noC case, use spmv_mode_Q for Q operations and spmv_mode_A for A operations
        if spmv_mode_Q == "customized" && spmv_mode_A == "customized"
            if qp.Q_is_diag
                cust_update_w_noC_diagQ_gpu!(ws, Halpern_fact1, Halpern_fact2)
                cust_update_x_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
                cust_update_y_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
            else
                cust_update_w_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
                cust_update_x_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
                cust_update_y_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
            end
        else
            if qp.Q_is_diag
                update_w_noC_diagQ_gpu!(ws, Halpern_fact1, Halpern_fact2)
                update_x_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
                update_y_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
            else
                update_w_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
                update_x_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
                update_y_noC_gpu!(ws, Halpern_fact1, Halpern_fact2)
            end
        end
    else
        if length(qp.Q.nzVal) > 0
            # Standard case with Q matrix - use unified kernels with separate Q and A modes
            unified_update_zxw1_gpu!(ws, qp, Halpern_fact1, Halpern_fact2;
                spmv_mode_Q=spmv_mode_Q, spmv_mode_A=spmv_mode_A, is_diag_Q=qp.Q_is_diag)
            unified_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2;
                spmv_mode_Q=spmv_mode_Q, spmv_mode_A=spmv_mode_A)
            unified_update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2;
                spmv_mode_Q=spmv_mode_Q, spmv_mode_A=spmv_mode_A, is_diag_Q=qp.Q_is_diag)
        else
            # Empty Q case (linear program) - use unified kernels with A mode only
            unified_update_zx_gpu!(ws, Halpern_fact1, Halpern_fact2; spmv_mode_A=spmv_mode_A)
            unified_update_y_noQ_gpu!(ws, Halpern_fact1, Halpern_fact2; spmv_mode_A=spmv_mode_A)
        end
    end
end

# This function computes the M norm for the HPR-QP algorithm on GPU.
function compute_M_norm_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Q_is_diag::Bool, noC::Bool)
    if noC
        M_1 = 1 / ws.sigma * CUDA.dot(ws.dx, ws.dx)
        M_2 = 0.0

        # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
        Qmap!(ws.dw, ws.dQw, qp.Q)

        if ws.m > 0
            M_1 += ws.sigma * ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            M_1 += 2 * CUDA.dot(ws.ATdy, ws.dx)
        end

        if !Q_is_diag
            M_2 = ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
            M_2 -= ws.sigma * CUDA.dot(ws.dQw, ws.dQw)
        end
        M_norm = max(M_2, 0) + max(M_1, 0)
        if min(M_1, M_2) < -1e-8
            println("M_1 = $M_1,M_2 = $M_2, negative M norm due to numerical instability, consider increasing eig_factor")
        end
    else
        # Initialize M terms
        M_1 = 0.0
        M_2 = 1 / ws.sigma * CUDA.dot(ws.dx, ws.dx)
        M_3 = 0.0

        # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
        Qmap!(ws.dw, ws.dQw, qp.Q)
        M_2 -= 2 * CUDA.dot(ws.dQw, ws.dx)

        # Add constraint-related terms if constraints exist
        if ws.m > 0
            M_1 = ws.sigma * ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            Qmap!(ws.ATdy, ws.QATdy, qp.Q)
            M_1 -= 2 * ws.sigma * CUDA.dot(ws.dQw, ws.ATdy)
            M_2 += 2 * CUDA.dot(ws.ATdy, ws.dx)

            if Q_is_diag
                ws.ATdy .*= ws.fact_M
                M_3 = CUDA.dot(ws.ATdy, ws.QATdy) # sGS term
                M_1 += ws.sigma * CUDA.dot(ws.dQw, ws.dQw)
            else
                M_3 = (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * CUDA.dot(ws.ATdy, ws.QATdy)  # sGS term
                M_1 += ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
            end
        else
            # No constraints case: only add Q-related term to M_1
            if !Q_is_diag
                M_1 = ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
            end
        end

        M_2 += max(M_1, 0)
        M_norm = max(M_2, 0) + max(M_3, 0)
        if min(M_1, M_2, M_3) < -1e-8
            println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3, negative M norm due to numerical instability, consider increasing eig_factor")
        end
    end
    return sqrt(M_norm)
end

# ==================== Helper Functions for Solver ====================

# Transfer model data from CPU to GPU
function transfer_to_gpu(model::QP_info_cpu, params::HPRQP_parameters)
    if params.verbose
        println("COPY TO GPU ...")
    end
    t_start = time()
    CUDA.synchronize()

    n = length(model.c)

    # Transfer Q to GPU using unified to_gpu interface
    # Works for both sparse matrices and CPU operators (QAP, LASSO, custom)
    Q_gpu = to_gpu(model.Q)

    # Determine lambda vector for GPU
    # For LASSO problems, replicate lambda value for each variable
    # For other problems, use empty vector
    if isa(model.Q, LASSO_Q_operator_cpu)
        lambda_gpu = CuVector(fill(model.lambda, n))
    else
        lambda_gpu = CuVector{Float64}([])
    end

    # Create QP_info_gpu
    qp = QP_info_gpu(
        Q_gpu,
        CuVector(model.c),
        CuSparseMatrixCSR(model.A),
        CuSparseMatrixCSR(model.AT),
        CuVector(model.AL),
        CuVector(model.AU),
        CuVector(model.l),
        CuVector(model.u),
        model.obj_constant,
        CuVector(model.diag_Q),
        model.Q_is_diag,
        model.noC,
        lambda_gpu,
    )

    CUDA.synchronize()
    transfer_time = time() - t_start
    if params.verbose
        println(@sprintf("COPY TO GPU time: %.2f seconds", transfer_time))
    end

    return qp, transfer_time
end

# Perform scaling on GPU
function scale_on_gpu!(qp::QP_info_gpu, params::HPRQP_parameters)
    if params.verbose
        println("SCALING QP ON GPU ...")
    end
    t_start = time()

    scaling_info_gpu = scaling_gpu!(qp, params)
    CUDA.synchronize()

    scaling_time = time() - t_start
    if params.verbose
        println(@sprintf("GPU SCALING time: %.2f seconds", scaling_time))
    end

    return scaling_info_gpu, scaling_time
end

# Print QP problem information
function print_problem_info(qp::QP_info_gpu, params::HPRQP_parameters)
    if !params.verbose
        return
    end
    
    m, n = size(qp.A)

    println("="^80)
    println("QP PROBLEM INFORMATION")
    println("="^80)

    # Determine QP type
    # Determine problem type using the interface
    qp_type = if isa(qp.Q, AbstractQOperator)
        get_operator_name(typeof(qp.Q))
    elseif isa(qp.Q, CuSparseMatrixCSR)
        if length(qp.Q.nzVal) > 0
            "QP (Quadratic Program - Non-empty Q)"
        else
            "LP (Linear Program - Empty Q)"
        end
    else
        "Unknown QP Type"
    end
    println("Problem Type: $qp_type")

    # Q matrix information
    if isa(qp.Q, CuSparseMatrixCSR)
        q_size = size(qp.Q, 1)
        q_nnz = length(qp.Q.nzVal)
        println("Q Matrix: $(q_size)×$(q_size), nnz = $q_nnz")
        if q_nnz > 0
            println("Q is Diagonal: $(qp.Q_is_diag)")
        end
    elseif isa(qp.Q, AbstractQOperator)
        op_name = get_operator_name(typeof(qp.Q))
        println("Q Operator: $op_name operator (implicit matrix)")
    end

    # Constraint matrix information
    if m > 0
        a_nnz = length(qp.A.nzVal)
        println("A Matrix: $(m)×$(n), nnz = $a_nnz")
    else
        println("A Matrix: No constraints (unconstrained)")
    end

    println()
end

# Print solver parameters
function print_solver_params(params::HPRQP_parameters, qp::QP_info_gpu, spmv_mode_Q::String, spmv_mode_A::String)
    if !params.verbose
        return
    end
    
    m = size(qp.A, 1)
    n = size(qp.A, 2)
    
    println("="^80)
    println("SOLVER PARAMETERS:")
    println("  Problem size: m = ", m, ", n = ", n)
    println("  Device: GPU (device $(params.device_number))")
    println("  Stop tolerance: ", params.stoptol)
    println("  Max iterations: ", params.max_iter)
    println("  Time limit: ", params.time_limit, " seconds")
    println("  Check interval: ", params.check_iter)
    println("  Print frequency: ", params.print_frequency == -1 ? "Adaptive" : params.print_frequency)
    println("  Eigenvalue factor: ", params.eig_factor)
    println("  Sigma fixed: ", params.sigma_fixed)
    println("  SpMV mode Q: ", spmv_mode_Q, params.spmv_mode_Q == "auto" ? " (auto-detected)" : "")
    println("  SpMV mode A: ", spmv_mode_A, params.spmv_mode_A == "auto" ? " (auto-detected)" : "")
    println("  Scaling options:")
    println("    Ruiz scaling: ", params.use_Ruiz_scaling ? "Enabled" : "Disabled")
    println("    Pock-Chambolle scaling: ", params.use_Pock_Chambolle_scaling ? "Enabled" : "Disabled")
    println("    b/c scaling: ", params.use_bc_scaling ? "Enabled" : "Disabled")
    println("    L2 scaling: ", params.use_l2_scaling ? "Enabled" : "Disabled")
    
    if params.warm_up
        println("  Warm-up: Enabled (avoids JIT compilation overhead)")
    else
        println("  Warm-up: Disabled")
        println("    ⚠ WARNING: First run of each function may be slower due to JIT compilation.")
        println("    ⚠ Consider enabling warm_up for more accurate timing measurements.")
    end
    
    if params.initial_x !== nothing
        println("  Initial x: Provided (length ", length(params.initial_x), ")")
    end
    if params.initial_y !== nothing
        println("  Initial y: Provided (length ", length(params.initial_y), ")")
    end
    
    if params.auto_save
        # Calculate estimated memory for auto_save
        memory_bytes = (n + m + 2*n) * 16  # x, y, z, w (8 bytes per Float64, 2 copies)
        memory_mb = memory_bytes / (1024 * 1024)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        
        println("  Auto-save: ENABLED")
        println("    ⚠ WARNING: Auto-save will write to disk at each print iteration.")
        println("    ⚠ This may consume significant I/O bandwidth and slightly reduce speed.")
        if memory_gb >= 1.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f GB", memory_gb))
        elseif memory_mb >= 1.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f MB", memory_mb))
        elseif memory_bytes >= 1024.0
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f KB", memory_bytes / 1024))
        else
            println(@sprintf("    ⚠ Estimated memory for saved state: %.2f bytes", memory_bytes))
        end
        println("    Save file: ", params.save_filename)
    else
        println("  Auto-save: Disabled")
    end
    println("="^80)
end

# Estimate maximum eigenvalues using power iteration
function estimate_eigenvalues(qp::QP_info_gpu, params::HPRQP_parameters)
    if params.verbose
        println("ESTIMATING MAXIMUM EIGENVALUES ...")
    end
    t_start = time()
    CUDA.synchronize()

    m = size(qp.A, 1)

    # Estimate lambda_max_A
    if m > 0
        lambda_max_A = power_iteration_A_gpu(qp.A, qp.AT) * params.eig_factor
    else
        lambda_max_A = 0.0
    end

    # Estimate lambda_max_Q based on Q type
    # Compute largest eigenvalue of Q using power iteration or direct computation
    if isa(qp.Q, AbstractQOperator)
        # Use the unified power_iteration_Q interface for all operators
        lambda_max_Q = power_iteration_Q(qp.Q) * params.eig_factor
    elseif isa(qp.Q, CuSparseMatrixCSR)
        if length(qp.Q.nzVal) > 0
            if !qp.Q_is_diag
                lambda_max_Q = power_iteration_Q_gpu(qp.Q) * params.eig_factor
            else
                lambda_max_Q = maximum(qp.Q.nzVal)
            end
        else
            lambda_max_Q = 0.0
        end
    else
        error("Unsupported Q type: ", typeof(qp.Q))
    end

    CUDA.synchronize()
    power_time = time() - t_start

    if params.verbose
        println(@sprintf("ESTIMATING MAXIMUM EIGENVALUES time = %.2f seconds", power_time))
        # println(@sprintf("estimated maximum eigenvalue of AAT = %.2e", lambda_max_A))
        # println(@sprintf("estimated maximum eigenvalue of Q = %.2e", lambda_max_Q))
    end

    return lambda_max_A, lambda_max_Q, power_time
end

# Determine SpMV mode based on problem structure
# Check if we should print at this iteration
function should_print(iter::Int, params::HPRQP_parameters, t_start_alg::Float64, max_iter::Int)
    if params.print_frequency == -1
        return ((rem(iter, print_step(iter)) == 0) || (iter == max_iter) ||
                (time() - t_start_alg > params.time_limit))
    elseif params.print_frequency > 0
        return ((rem(iter, params.print_frequency) == 0) || (iter == max_iter) ||
                (time() - t_start_alg > params.time_limit))
    else
        error("Invalid print_frequency: ", params.print_frequency,
            ". It should be a positive integer or -1 for automatic printing.")
    end
end

# Check and record tolerance milestones
function check_tolerance_milestones!(residuals::HPRQP_residuals,
    iter::Int,
    t_start_alg::Float64,
    iter_4::Ref{Int}, time_4::Ref{Float64}, first_4::Ref{Bool},
    iter_6::Ref{Int}, time_6::Ref{Float64}, first_6::Ref{Bool})
    if residuals.KKTx_and_gap_org_bar < 1e-4 && first_4[]
        time_4[] = time() - t_start_alg
        iter_4[] = iter
        first_4[] = false
        println("KKT < 1e-4 at iter = ", iter)
    end
    if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6[]
        time_6[] = time() - t_start_alg
        iter_6[] = iter
        first_6[] = false
        println("KKT < 1e-6 at iter = ", iter)
    end
end

# ==================== Public API Functions ====================

"""
    optimize(model::QP_info_cpu, params::HPRQP_parameters)

Solve a QP model with optional warm-up phase.

This is the main entry point for solving QP problems. It handles:
1. Optional warm-up phase to avoid JIT compilation overhead
2. Calls solve() which does scaling, GPU transfer, and optimization

# Arguments
- `model::QP_info_cpu`: QP model built from build_from_mps(), build_from_QAbc(), etc.
- `params::HPRQP_parameters`: Solver parameters

# Returns
- `HPRQP_results`: Solution results

# Example
```julia
using HPRQP

model = build_from_mps("problem.mps")
params = HPRQP_parameters()
params.stoptol = 1e-8
params.warm_up = true
result = optimize(model, params)
```

See also: [`build_from_mps`](@ref), [`build_from_QAbc`](@ref), [`solve`](@ref)
"""
function optimize(model::QP_info_cpu, params::HPRQP_parameters)
    # Setup: GPU device
    CUDA.device!(params.device_number)

    # Handle warmup if requested
    if params.warm_up
        if params.verbose
            println("="^80)
            println("WARM UP PHASE")
            println("  ℹ Running warmup to avoid JIT compilation overhead in main solve")
            println("="^80)
        end
        t_start_warmup = time()

        # Save original max_iter and verbose
        original_max_iter = params.max_iter
        original_verbose = params.verbose
        params.max_iter = 200
        params.verbose = false

        # Run warmup solve
        solve(model, params)

        # Restore original parameters
        params.max_iter = original_max_iter
        params.verbose = original_verbose

        warmup_time = time() - t_start_warmup
        if params.verbose
            println(@sprintf("Warmup time: %.2f seconds", warmup_time))
            println("="^80)
            println()
        end
    end

    # Main solve
    if params.verbose
        println("="^80)
        println("MAIN SOLVE")
        println("="^80)
    end

    # Run the main algorithm (scaling and GPU transfer happen inside solve)
    results = solve(model, params)

    if params.verbose
        println("="^80)
    end

    return results
end

# Helper function: Initialize solver state with workspace, residuals, and restart info
function initialize_solver_state(qp::QP_info_gpu, params::HPRQP_parameters,
    lambda_max_A::Float64, lambda_max_Q::Float64,
    scaling_info_gpu::Scaling_info_gpu)
    residuals = HPRQP_residuals()
    restart_info = initialize_restart()
    ws = allocate_workspace_gpu(qp, params, lambda_max_A, lambda_max_Q, scaling_info_gpu)
    return ws, residuals, restart_info
end

# Helper function: Benchmark A operations only
function benchmark_A_operations(ws::HPRQP_workspace_gpu, verbose::Bool=true)
    if verbose
        println("  Benchmarking A operations...")
    end

    # Save workspace state
    saved_y = copy(ws.y)
    saved_y_hat = copy(ws.y_hat)
    saved_y_bar = copy(ws.y_bar)
    saved_w = copy(ws.w)
    saved_w_hat = copy(ws.w_hat)
    saved_w_bar = copy(ws.w_bar)

    # Test unified update with CUSPARSE mode for A (uses y and w2 which involve A operations)
    CUDA.@sync begin end
    t_A_cusparse_start = time()
    for _ in 1:10
        unified_update_y_gpu!(ws, 0.5, 0.5; spmv_mode_Q="CUSPARSE", spmv_mode_A="CUSPARSE")
        unified_update_w2_gpu!(ws, 0.5, 0.5; spmv_mode_Q="CUSPARSE", spmv_mode_A="CUSPARSE", is_diag_Q=false)
    end
    CUDA.@sync begin end
    t_A_cusparse = time() - t_A_cusparse_start

    # Restore state
    copyto!(ws.y, saved_y)
    copyto!(ws.y_hat, saved_y_hat)
    copyto!(ws.y_bar, saved_y_bar)
    copyto!(ws.w, saved_w)
    copyto!(ws.w_hat, saved_w_hat)
    copyto!(ws.w_bar, saved_w_bar)

    # Test unified update with customized mode for A (uses y and w2 which involve A operations)
    CUDA.@sync begin end
    t_A_custom_start = time()
    for _ in 1:10
        unified_update_y_gpu!(ws, 0.5, 0.5; spmv_mode_Q="customized", spmv_mode_A="customized")
        unified_update_w2_gpu!(ws, 0.5, 0.5; spmv_mode_Q="customized", spmv_mode_A="customized", is_diag_Q=false)
    end
    CUDA.@sync begin end
    t_A_custom = time() - t_A_custom_start

    # Restore state
    copyto!(ws.y, saved_y)
    copyto!(ws.y_hat, saved_y_hat)
    copyto!(ws.y_bar, saved_y_bar)
    copyto!(ws.w, saved_w)
    copyto!(ws.w_hat, saved_w_hat)
    copyto!(ws.w_bar, saved_w_bar)

    spmv_mode_A = (t_A_cusparse < t_A_custom) ? "CUSPARSE" : "customized"
    if verbose
        println("    A: CUSPARSE=$(round(t_A_cusparse*1000, digits=2))ms, customized=$(round(t_A_custom*1000, digits=2))ms → $(spmv_mode_A)")
    end

    return spmv_mode_A
end

# Helper function: Determine SPMV mode based on problem structure
# Returns a tuple (spmv_mode_Q, spmv_mode_A) for separate control of Q and A operations
# Note: A is always a sparse matrix (never an operator), only Q can be an operator
function determine_spmv_mode(qp::QP_info_gpu, params::HPRQP_parameters, ws::HPRQP_workspace_gpu)
    spmv_mode_Q = params.spmv_mode_Q
    spmv_mode_A = params.spmv_mode_A

    # For operator-based Q, always force operator mode for Q
    # (operator Q cannot use CUSPARSE or customized sparse matrix kernels)
    if isa(qp.Q, AbstractQOperator)
        # Force operator mode regardless of user setting
        if params.verbose
            if params.spmv_mode_Q != "auto" && params.spmv_mode_Q != "operator"
                println("Warning: Q is AbstractQOperator, forcing spmv_mode_Q = operator (user setting '$(params.spmv_mode_Q)' ignored)")
            else
                println("Q is AbstractQOperator → spmv_mode_Q = operator")
            end
        end
        spmv_mode_Q = "operator"

        # Benchmark A operations if auto mode and A matrix exists
        if ws.m > 0
            if params.spmv_mode_A == "auto"
                spmv_mode_A = benchmark_A_operations(ws, params.verbose)
            else
                if params.verbose
                    println("A operations: spmv_mode_A = $(spmv_mode_A) (user-specified)")
                end
            end
        else
            # No A matrix (e.g., LASSO unconstrained problem)
            if params.verbose
                println("No A matrix (unconstrained problem) → spmv_mode_A not applicable")
            end
            spmv_mode_A = "auto"  # Placeholder, won't be used
        end

        return (spmv_mode_Q, spmv_mode_A)
    end

    # For sparse matrix Q, benchmark Q and/or A based on auto mode settings
    if isa(qp.Q, CuSparseMatrixCSR)
        if params.verbose
            println("Auto-detecting SPMV modes via benchmarking...")
        end

        # Benchmark Q if auto mode
        if params.spmv_mode_Q == "auto" && length(qp.Q.nzVal) > 0
            if params.verbose
                println("  Benchmarking Q operations...")
            end

            # Save workspace state
            saved_z = copy(ws.z_bar)
            saved_x = copy(ws.x)
            saved_x_hat = copy(ws.x_hat)
            saved_x_bar = copy(ws.x_bar)
            saved_w = copy(ws.w)
            saved_w_hat = copy(ws.w_hat)
            saved_w_bar = copy(ws.w_bar)

            # Test unified update with CUSPARSE mode for Q (zxw1 involves Q operations)
            CUDA.@sync begin end
            t_Q_cusparse_start = time()
            for _ in 1:10
                unified_update_zxw1_gpu!(ws, qp, 0.5, 0.5; spmv_mode_Q="CUSPARSE", spmv_mode_A="CUSPARSE", is_diag_Q=false)
            end
            CUDA.@sync begin end
            t_Q_cusparse = time() - t_Q_cusparse_start

            # Restore state
            copyto!(ws.z_bar, saved_z)
            copyto!(ws.x, saved_x)
            copyto!(ws.x_hat, saved_x_hat)
            copyto!(ws.x_bar, saved_x_bar)
            copyto!(ws.w, saved_w)
            copyto!(ws.w_hat, saved_w_hat)
            copyto!(ws.w_bar, saved_w_bar)

            # Test unified update with customized mode for Q (zxw1 involves Q operations)
            CUDA.@sync begin end
            t_Q_custom_start = time()
            for _ in 1:10
                unified_update_zxw1_gpu!(ws, qp, 0.5, 0.5; spmv_mode_Q="customized", spmv_mode_A="customized", is_diag_Q=false)
            end
            CUDA.@sync begin end
            t_Q_custom = time() - t_Q_custom_start

            # Restore state
            copyto!(ws.z_bar, saved_z)
            copyto!(ws.x, saved_x)
            copyto!(ws.x_hat, saved_x_hat)
            copyto!(ws.x_bar, saved_x_bar)
            copyto!(ws.w, saved_w)
            copyto!(ws.w_hat, saved_w_hat)
            copyto!(ws.w_bar, saved_w_bar)

            spmv_mode_Q = (t_Q_cusparse < t_Q_custom) ? "CUSPARSE" : "customized"
            if params.verbose
                println("    Q: CUSPARSE=$(round(t_Q_cusparse*1000, digits=2))ms, customized=$(round(t_Q_custom*1000, digits=2))ms → $(spmv_mode_Q)")
            end
        else
            if params.verbose
                println("  Q operations: spmv_mode_Q = $(spmv_mode_Q) (user-specified)")
            end
        end

        # Benchmark A if auto mode
        if params.spmv_mode_A == "auto" && ws.m > 0
            spmv_mode_A = benchmark_A_operations(ws, params.verbose)
        else
            if params.verbose
                println("  A operations: spmv_mode_A = $(spmv_mode_A) (user-specified)")
            end
        end

        if params.verbose
            println("  Selected: spmv_mode_Q = $(spmv_mode_Q), spmv_mode_A = $(spmv_mode_A)")
        end
        return (spmv_mode_Q, spmv_mode_A)
    end

    # Default return
    return (spmv_mode_Q, spmv_mode_A)
end

# Helper function: Setup iteration loop variables and Q factors
function setup_iteration_loop!(qp::QP_info_gpu, ws::HPRQP_workspace_gpu)
    # Initialize tracking variables for milestones
    iter_4, time_4 = 0, 0.0
    iter_6, time_6 = 0, 0.0
    first_4, first_6 = true, true

    # Update Q factors for diagonal Q
    if isa(qp.Q, CuSparseMatrixCSR) && qp.Q_is_diag
        update_Q_factors_gpu!(ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, ws.sigma)
    end

    return iter_4, time_4, iter_6, time_6, first_4, first_6
end

# Helper function: Determine if iteration should print
function should_print(iter::Int, params::HPRQP_parameters, t_start_alg::Float64)
    if params.print_frequency == -1
        return (rem(iter, print_step(iter)) == 0) ||
               (iter == params.max_iter) ||
               (time() - t_start_alg > params.time_limit)
    elseif params.print_frequency > 0
        return (rem(iter, params.print_frequency) == 0) ||
               (iter == params.max_iter) ||
               (time() - t_start_alg > params.time_limit)
    else
        error("Invalid print_frequency: ", params.print_frequency,
            ". It should be a positive integer or -1 for automatic printing.")
    end
end

# Helper function: Print iteration log
function print_iteration_log(iter::Int, residuals::HPRQP_residuals,
    ws::HPRQP_workspace_gpu, t_start_alg::Float64)
    println(@sprintf("%5.0f    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f",
        iter,
        residuals.err_Rp_org_bar,
        residuals.err_Rd_org_bar,
        residuals.primal_obj_bar,
        residuals.dual_obj_bar,
        residuals.rel_gap_bar,
        ws.sigma,
        time() - t_start_alg))
end

# Helper function: Update milestone tracking for KKT thresholds
function update_milestone_tracking!(residuals::HPRQP_residuals, iter::Int,
    t_start_alg::Float64,
    iter_4::Int, time_4::Float64, first_4::Bool,
    iter_6::Int, time_6::Float64, first_6::Bool,
    verbose::Bool)
    if residuals.KKTx_and_gap_org_bar < 1e-4 && first_4
        time_4 = time() - t_start_alg
        iter_4 = iter
        first_4 = false
        if verbose
            println("KKT < 1e-4 at iter = ", iter)
        end
    end

    if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6
        time_6 = time() - t_start_alg
        iter_6 = iter
        first_6 = false
        if verbose
            println("KKT < 1e-6 at iter = ", iter)
        end
    end

    return iter_4, time_4, first_4, iter_6, time_6, first_6
end

# Helper function: Handle termination and collect results
function handle_termination(status::String, residuals::HPRQP_residuals,
    ws::HPRQP_workspace_gpu, scaling_info_gpu::Scaling_info_gpu,
    iter::Int, t_start_alg::Float64, power_time::Float64,
    setup_time::Float64, iter_4::Int, time_4::Float64,
    iter_6::Int, time_6::Float64, verbose::Bool)
    # Print termination message
    if verbose
        if status == "OPTIMAL"
            println("The instance is solved, the accuracy is ", residuals.KKTx_and_gap_org_bar)
        elseif status == "MAX_ITER"
            println("The maximum number of iterations is reached, the accuracy is ",
                residuals.KKTx_and_gap_org_bar)
        elseif status == "TIME_LIMIT"
            println("The time limit is reached, the accuracy is ", residuals.KKTx_and_gap_org_bar)
        end
    end

    # Collect results
    results = collect_results_gpu!(ws, residuals, scaling_info_gpu, iter,
        t_start_alg, power_time)
    results.status = status
    results.time_4 = time_4 == 0.0 ? results.time : time_4
    results.iter_4 = iter_4 == 0 ? iter : iter_4
    results.time_6 = time_6 == 0.0 ? results.time : time_6
    results.iter_6 = iter_6 == 0 ? iter : iter_6

    # Print solution summary
    if verbose
        println()
        println("="^80)
        println("SOLUTION SUMMARY")
        println("="^80)
        println(@sprintf("Status: %s", status))
        println(@sprintf("Iterations: %d", iter))
        println(@sprintf("Time: %.2f seconds", results.time))
        println(@sprintf("Primal Objective: %.12e", residuals.primal_obj_bar))
        println(@sprintf("Dual Objective: %.12e", residuals.dual_obj_bar))
        println(@sprintf("Primal Residual: %.6e", residuals.err_Rp_org_bar))
        println(@sprintf("Dual Residual: %.6e", residuals.err_Rd_org_bar))
        println(@sprintf("Relative Gap: %.6e", residuals.rel_gap_bar))
        println("="^80)
        println(@sprintf("Total time: %.2fs  (setup = %.2fs, solve = %.2fs)", 
            setup_time + results.time, setup_time, results.time))
        println("="^80)
    end

    return results
end

# Helper function: Perform main iteration step (update and norm computation)
function perform_iteration_step!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu,
    params::HPRQP_parameters, restart_info::HPRQP_restart,
    spmv_mode_Q::String, spmv_mode_A::String, iter::Int, check_iter::Int)
    # Main update - now handles both operator and sparse matrix Q within main_update!
    main_update!(ws, qp, spmv_mode_Q, spmv_mode_A, restart_info)

    # Compute M norm for restart decision
    if restart_info.restart_flag > 0
        restart_info.last_gap = compute_M_norm_gpu!(ws, qp, qp.Q_is_diag, qp.noC)
    end

    if rem(iter + 1, check_iter) == 0
        restart_info.current_gap = compute_M_norm_gpu!(ws, qp, qp.Q_is_diag, qp.noC)
    end

    restart_info.inner += 1
end

# This function is the main solver function for the HPR-QP algorithm on GPU.
# It handles GPU transfer, scaling on GPU, and optimization.
function solve(model::QP_info_cpu, params::HPRQP_parameters)
    setup_start = time()

    # Step 1: Transfer to GPU
    qp, transfer_time = transfer_to_gpu(model, params)

    # Step 2: Scaling on GPU
    scaling_info_gpu, scaling_time = scale_on_gpu!(qp, params)

    setup_time = time() - setup_start

    # Step 3: Estimate eigenvalues (needed for workspace allocation)
    t_start_alg = time()
    lambda_max_A, lambda_max_Q, power_time = estimate_eigenvalues(qp, params)

    # Step 4: Initialize solver state (needed for determine_spmv_mode)
    ws, residuals, restart_info = initialize_solver_state(qp, params, lambda_max_A,
        lambda_max_Q, scaling_info_gpu)

    # Step 5: Determine SPMV mode (before printing so we can show the actual mode)
    spmv_mode_Q, spmv_mode_A = determine_spmv_mode(qp, params, ws)

    # Step 6: Print problem information
    print_problem_info(qp, params)
    print_solver_params(params, qp, spmv_mode_Q, spmv_mode_A)

    # Step 7: Setup iteration loop
    iter_4, time_4, iter_6, time_6, first_4, first_6 = setup_iteration_loop!(qp, ws)

    if params.verbose
        println("HPRQP SOLVER starts...")
        println(" iter     errRp        errRd         p_obj           d_obj          gap        sigma       time")
    end

    check_iter = params.check_iter

    # Main iteration loop
    for iter = 0:params.max_iter
        # Determine if we should print at this iteration
        print_yes = should_print(iter, params, t_start_alg)

        # Compute residuals if needed
        if rem(iter, check_iter) == 0 || print_yes
            residuals.is_updated = true
            compute_residuals_gpu(ws, qp, scaling_info_gpu, residuals, params, iter)
        else
            residuals.is_updated = false
        end

        # Check termination criteria
        status = check_break(residuals, iter, t_start_alg, params)

        # Check and perform restart if needed
        check_restart(restart_info, iter, check_iter)

        # Update sigma parameter
        update_sigma(params, restart_info, ws, qp, qp.Q_is_diag, qp.noC)

        # Perform restart
        do_restart(restart_info, ws, qp, qp.noC)

        # Print iteration log
        if (print_yes || (status != "CONTINUE")) && params.verbose
            print_iteration_log(iter, residuals, ws, t_start_alg)
        end
            
        # Save to HDF5 if auto_save is enabled
        if (print_yes || (status != "CONTINUE")) && params.auto_save
            try
                save_state_to_hdf5(params.save_filename, ws, scaling_info_gpu, residuals, params, iter, t_start_alg)
            catch e
                if params.verbose
                    println("Warning: Failed to save to HDF5 file: ", e)
                end
            end
        end

        # Update milestone tracking
        iter_4, time_4, first_4, iter_6, time_6, first_6 =
            update_milestone_tracking!(residuals, iter, t_start_alg,
                iter_4, time_4, first_4, iter_6, time_6, first_6, params.verbose)

        # Handle termination
        if status != "CONTINUE"
            return handle_termination(status, residuals, ws, scaling_info_gpu,
                iter, t_start_alg, power_time, setup_time,
                iter_4, time_4, iter_6, time_6, params.verbose)
        end

        # Perform main iteration step
        perform_iteration_step!(ws, qp, params, restart_info, spmv_mode_Q, spmv_mode_A, iter, check_iter)
    end
end
