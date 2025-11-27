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
    # Use Qmap! for operator-based Q, or CUSPARSE for sparse matrix Q
    if isa(qp.Q, AbstractQOperator)
        Qmap!(ws.x_bar, ws.Qx, ws.temp1, qp.Q)
    else
        CUSPARSE.mv!('N', 1, ws.Q, ws.x_bar, 0, ws.Qx, 'O', CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
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
    if qp.noC && params.problem_type != "LASSO"
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

    if (params.problem_type == "QAP" && iter == 0) || (params.problem_type == "QP" && iter == 0)
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_err_lu_kernel!(ws.dx, ws.x_bar, ws.l, ws.u, sc.col_norm, sc.b_scale, ws.n)
        res.err_Rp_org_bar = max(res.err_Rp_org_bar, CUDA.norm(ws.dx, Inf))
    end
    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)

end

# This function updates sigma for operator-based Q (QAP/LASSO)
function update_sigma_operator(params::HPRQP_parameters,
    restart_info::HPRQP_restart,
    ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
)
    if ~params.sigma_fixed && (restart_info.restart_flag >= 1)
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
        if ws.m > 0
            axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
            axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.last_ATy, ws.ATdy, ws.n)
            Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
            a += ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy) - 2 * CUDA.dot(ws.dQw, ws.ATdy)
        end
        axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
        Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)

        b += CUDA.dot(ws.dx, ws.dx)
        a += ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
        if ws.m > 0
            c += CUDA.dot(ws.ATdy, ws.QATdy)
            d += ws.lambda_max_Q
        end
        a = max(a, 1e-12)
        b = max(b, 1e-12)
        if ws.m > 0
            sigma_new = golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
        else
            sigma_new = sqrt(b / a)
        end
        fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
        ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))
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
            if ws.m > 0
                axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
            end
            axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
            CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)

            primal_move = CUDA.dot(ws.dx, ws.dx)
            if Q_is_diag
                dual_move = ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
            else
                dual_move = ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy) + ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw) - CUDA.dot(ws.dQw, ws.dQw)
            end

            primal_move = max(primal_move, 1e-12)
            dual_move = max(dual_move, 1e-12)
            sigma_new = sqrt(primal_move / dual_move)
            fact = exp(-restart_info.current_gap / restart_info.weighted_norm)
            ws.sigma = exp(fact * log(sigma_new) + (1 - fact) * log(ws.sigma))
        else
            axpby_gpu!(1.0, ws.x_bar, -1.0, ws.last_x, ws.dx, ws.n)
            if ws.m > 0
                axpby_gpu!(1.0, ws.y_bar, -1.0, ws.last_y, ws.dy, ws.m)
                axpby_gpu!(1.0, ws.ATy_bar, -1.0, ws.last_ATy, ws.ATdy, ws.n)
                # Handle operator-based Q vs sparse matrix Q
                if isa(qp.Q, AbstractQOperator)
                    Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
                else
                    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
                end
            end
            axpby_gpu!(1.0, ws.w_bar, -1.0, ws.last_w, ws.dw, ws.n)
            # Handle operator-based Q vs sparse matrix Q
            if isa(qp.Q, AbstractQOperator)
                Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)
            else
                CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            end
            a = ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy) - 2 * CUDA.dot(ws.dQw, ws.ATdy)
            b = CUDA.dot(ws.dx, ws.dx)
            if Q_is_diag
                # if Q_is_diag
                a += CUDA.norm(ws.dQw)^2
            else
                a += ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
                c = CUDA.dot(ws.ATdy, ws.QATdy)
                d = ws.lambda_max_Q
            end
            a = max(a, 1e-12)
            b = max(b, 1e-12)
            if Q_is_diag
                sigma_new = golden_Q_diag(a, b, ws.diag_Q, ws.ATdy, ws.QATdy, ws.tempv; lo=1e-12, hi=1e12, tol=1e-13)
            else
                # min a * x + b / x + c * x^2 / (1 + d * x)
                sigma_new = golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
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
    restart::Int=-1,
)
    restart_info.restart_flag = 0
    if restart > 0
        if rem(iter, restart) == 0 && iter > 0
            restart_info.restart_flag = 1
        end
        return
    end
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
            # Handle operator-based Q vs sparse matrix Q
            if isa(qp.Q, AbstractQOperator)
                Qmap!(ws.w_bar, ws.Qw_bar, ws.temp1, qp.Q)
            else
                CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            end
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
    println("initial sigma = ", ws.sigma)
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

    # Allocate temp1 for Q operator computations
    if isa(qp.Q, LASSO_Q_operator_gpu)
        ws.temp1 = CUDA.zeros(Float64, qp.Q.A.dims[1])
    elseif isa(qp.Q, QAP_Q_operator_gpu)
        ws.temp1 = CUDA.zeros(Float64, n)
    else
        ws.temp1 = CUDA.zeros(Float64, n)  # Default for sparse matrix Q
    end

    # Copy lambda for LASSO problems
    ws.lambda = qp.lambda

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

# This function updates the variables for operator-based Q (QAP/LASSO)
function main_update_operator!(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    params::HPRQP_parameters,
    restart_info::HPRQP_restart,
)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    Halpern_fact2 = 1.0 - Halpern_fact1

    if params.problem_type == "LASSO"
        # LASSO update with soft-thresholding
        Qmap!(ws.w, ws.Qw, ws.temp1, qp.Q)
        fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
        fact1 = 1.0 - fact2
        @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw_LASSO_kernel!(ws.lambda, ws.dw, ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
    elseif params.problem_type == "QAP"
        # QAP update with box constraints
        update_zxw1_QAP_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        update_y_QAP_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        update_w2_QAP_gpu!(ws, Halpern_fact1, Halpern_fact2)
    end
end

# This function computes the M norm for operator-based Q
function compute_M_norm_operator_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu)
    M_1 = 0.0
    M_2 = 1 / ws.sigma * CUDA.dot(ws.dx, ws.dx)
    Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)
    M_1 += ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
    M_2 -= 2 * CUDA.dot(ws.dQw, ws.dx)
    M_3 = 0.0

    if ws.m > 0
        M_1 += ws.sigma * ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
        M_1 -= 2 * ws.sigma * CUDA.dot(ws.dQw, ws.ATdy)
        M_2 += 2 * CUDA.dot(ws.ATdy, ws.dx)
        M_3 += (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * CUDA.dot(ws.ATdy, ws.QATdy)
    end
    M_2 += max(M_1, 0)
    M_norm = max(M_2, 0) + max(M_3, 0)
    if min(M_1, M_2, M_3) < -1e-8
        println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3, negative M norm due to numerical instability, consider increasing eig_factor")
    end
    return sqrt(M_norm)
end

# This function updates the variables in the HPR-QP algorithm, when Q is diagonal, there's no proximal term on w;
# when the problem is formulated without l≤x≤u, the update is w->x->y.
function main_update!(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    spmv_mode::String,
    restart_info::HPRQP_restart,
)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    Halpern_fact2 = 1.0 - Halpern_fact1
    if qp.noC
        if spmv_mode == "customized"
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
            # Standard case with Q matrix - use unified kernels
            unified_update_zxw1_gpu!(ws, Halpern_fact1, Halpern_fact2; 
                                     spmv_mode=spmv_mode, is_diag_Q=qp.Q_is_diag)
            unified_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2; spmv_mode=spmv_mode)
            unified_update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2; 
                                   spmv_mode=spmv_mode, is_diag_Q=qp.Q_is_diag)
        else
            # Empty Q case (linear program) - use unified kernels
            unified_update_zx_gpu!(ws, Halpern_fact1, Halpern_fact2; spmv_mode=spmv_mode)
            unified_update_y_noQ_gpu!(ws, Halpern_fact1, Halpern_fact2; spmv_mode=spmv_mode)
        end
    end
end

# This function computes the M norm for the HPR-QP algorithm on GPU.
function compute_M_norm_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Q_is_diag::Bool, noC::Bool)
    if noC
        M_1 = ws.sigma * ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
        M_1 += 1 / ws.sigma * CUDA.dot(ws.dx, ws.dx)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        # Handle operator-based Q vs sparse matrix Q
        if isa(qp.Q, AbstractQOperator)
            Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)
        else
            CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
        M_1 += 2 * CUDA.dot(ws.ATdy, ws.dx)
        M_2 = 0
        if !Q_is_diag
            M_2 = ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
            M_2 -= ws.sigma * CUDA.dot(ws.dQw, ws.dQw)
        end
        M_norm = max(M_2, 0) + max(M_1, 0)
        if min(M_1, M_2) < -1e-8
            println("M_1 = $M_1,M_2 = $M_2, negative M norm due to numerical instability, consider increasing eig_factor")
        end
    else
        M_1 = ws.sigma * ws.lambda_max_A * CUDA.dot(ws.dy, ws.dy)
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.dy, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        # Handle operator-based Q vs sparse matrix Q
        if isa(qp.Q, AbstractQOperator)
            Qmap!(ws.ATdy, ws.QATdy, ws.temp1, qp.Q)
            Qmap!(ws.dw, ws.dQw, ws.temp1, qp.Q)
        else
            CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.ATdy, 0, ws.QATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
            CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.dw, 0, ws.dQw, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
        M_1 -= 2 * ws.sigma * CUDA.dot(ws.dQw, ws.ATdy)
        M_2 = 1 / ws.sigma * CUDA.dot(ws.dx, ws.dx)
        M_2 += 2 * CUDA.dot(ws.ATdy, ws.dx)
        M_2 -= 2 * CUDA.dot(ws.dQw, ws.dx)
        if Q_is_diag
            ws.ATdy .*= ws.fact_M
            M_3 = CUDA.dot(ws.ATdy, ws.QATdy) # sGS term
            M_1 += ws.sigma * CUDA.dot(ws.dQw, ws.dQw)
        else
            M_3 = (ws.sigma * ws.sigma) / (1 + ws.sigma * ws.lambda_max_Q) * CUDA.dot(ws.ATdy, ws.QATdy)  # sGS term
            M_1 += ws.sigma * ws.lambda_max_Q * CUDA.dot(ws.dw, ws.dQw)
        end
        M_2 += max(M_1, 0)
        M_norm = max(M_2, 0) + max(M_3, 0)
        if min(M_1, M_2, M_3) < -1e-8
            println("M_1 = $M_1,M_2 = $M_2,M_3 = $M_3, negative M norm due to numerical instability, consider increasing eig_factor")
        end
    end
    return sqrt(M_norm)
end

# This function is the main solver function for the HPR-QP algorithm on GPU.
function solve(qp::QP_info_gpu, scaling_info::Scaling_info_gpu, params::HPRQP_parameters)
    m, n = size(qp.A)

    # Print QP Information
    println("="^80)
    println("QP PROBLEM INFORMATION")
    println("="^80)
    
    # Determine QP type
    qp_type = if isa(qp.Q, QAP_Q_operator_gpu)
        "QAP (Quadratic Assignment Problem)"
    elseif isa(qp.Q, LASSO_Q_operator_gpu)
        "LASSO (Least Absolute Shrinkage and Selection Operator)"
    elseif isa(qp.Q, CuSparseMatrixCSR)
        if length(qp.Q.nzVal) > 0
            qp.Q_is_diag ? "QP with Diagonal Q" : "QP with Sparse Q"
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
    elseif isa(qp.Q, QAP_Q_operator_gpu)
        println("Q Operator: QAP operator (implicit matrix)")
    elseif isa(qp.Q, LASSO_Q_operator_gpu)
        println("Q Operator: LASSO operator (implicit matrix)")
    end
    
    # Constraint matrix information
    if m > 0
        a_nnz = length(qp.A.nzVal)
        println("A Matrix: $(m)×$(n), nnz = $a_nnz")
        # println("Constraint Type: $(qp.noC ? "No C matrix (direct A)" : "With C matrix")")
    else
        println("A Matrix: No constraints (unconstrained)")
    end
    
    println()
    println("="^80)
    println("SOLVER PARAMETERS")
    println("="^80)
    println("Max Iterations: $(params.max_iter)")
    println("Time Limit: $(params.time_limit)s")
    println("Stop Tolerance: $(params.stoptol)")
    println("Eigenvalue Factor: $(params.eig_factor)")
    println("Sigma Fixed: $(params.sigma_fixed)")
    if !params.sigma_fixed
        println("  Sigma Update: adaptive (based on restart)")
        println("  Check Iteration: every $(params.check_iter) iterations")
        if params.restart > 0
            println("  Restart Frequency: every $(params.restart) iterations")
        else
            println("  Restart Strategy: adaptive")
        end
    end
    println("SpMV Mode: $(params.spmv_mode)")
    if isa(qp.Q, CuSparseMatrixCSR) && length(qp.Q.nzVal) > 0
        println("  (Controls kernel fusion for Q*w and A*x operations)")
    end
    println("Print Frequency: $(params.print_frequency == -1 ? "adaptive" : "every $(params.print_frequency) iterations")")
    println("Device Number: $(params.device_number)")
    println("Warm Up: $(params.warm_up)")
    println("="^80)
    println()

    CUDA.synchronize()
    t_start_alg = time()
    ### power iteration to estimate lambda_max ###
    println("ESTIMATING MAXIMUM EIGENVALUES ...")
    if m > 0
        lambda_max_A = power_iteration_A_gpu(qp.A, qp.AT) * params.eig_factor
    else
        lambda_max_A = 0.0
    end

    # Handle different Q types for eigenvalue estimation
    if isa(qp.Q, QAP_Q_operator_gpu)
        lambda_max_Q = power_iteration_Q_QAP_gpu(qp.Q) * params.eig_factor
    elseif isa(qp.Q, LASSO_Q_operator_gpu)
        lambda_max_Q = power_iteration_Q_LASSO_gpu(qp.Q) * params.eig_factor
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
    power_time = time() - t_start_alg
    println(@sprintf("ESTIMATING MAXIMUM EIGENVALUES time = %.2f seconds", power_time))
    println(@sprintf("estimated maximum eigenvalue of AAT = %.2e", lambda_max_A))
    println(@sprintf("estimated maximum eigenvalue of Q = %.2e", lambda_max_Q))

    ### Initialization ###
    residuals = HPRQP_residuals()
    restart_info = initialize_restart()

    ws = allocate_workspace_gpu(qp, params, lambda_max_A, lambda_max_Q, scaling_info)

    iter_4 = 0
    time_4 = 0.0
    iter_6 = 0
    time_6 = 0.0
    first_4 = true
    first_6 = true

    # Determine spmv_mode (only relevant for sparse matrix Q)
    spmv_mode = params.spmv_mode
    if isa(qp.Q, CuSparseMatrixCSR) && params.spmv_mode == "auto"
        if ws.m > 0
            max_nnz_A_row = maximum(vec(sum(t -> (abs(t) > 0), qp.A, dims=2)))
            max_nnz_A_col = maximum(vec(sum(t -> (abs(t) > 0), qp.A, dims=1)))
        else
            max_nnz_A_row = 0
            max_nnz_A_col = 0
        end
        max_nnz_Q_col = maximum(vec(sum(t -> (abs(t) > 0), qp.Q, dims=1)))
        if (max_nnz_A_row > max(0.01 * n, 500)) || (max_nnz_A_col > max(0.01 * m, 500)) || (max_nnz_Q_col > max(0.01 * n, 500))
            spmv_mode = "CUSPARSE"
        else
            spmv_mode = "customized"
        end

        if m > 1e5 && n > 1e5
            spmv_mode = "CUSPARSE"
        end
    elseif isa(qp.Q, AbstractQOperator)
        spmv_mode = "operator"  # Use operator mode for Q operators
    end
    println("SPMV mode = ", spmv_mode)

    if isa(qp.Q, CuSparseMatrixCSR) && qp.Q_is_diag
        update_Q_factors_gpu!(
            ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, ws.sigma
        )
    end

    println("HPRQP SOLVER starts...")
    println(" iter     errRp        errRd         p_obj           d_obj          gap        sigma       time")

    check_iter = params.check_iter
    for iter = 0:params.max_iter
        if params.print_frequency == -1
            print_yes = ((rem(iter, print_step(iter)) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        elseif params.print_frequency > 0
            print_yes = ((rem(iter, params.print_frequency) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        else
            error("Invalid print_frequency: ", params.print_frequency, ". It should be a positive integer or -1 for automatic printing.")
        end

        if rem(iter, check_iter) == 0 || print_yes
            residuals.is_updated = true
            compute_residuals_gpu(ws, qp, scaling_info, residuals, params, iter)
        else
            residuals.is_updated = false
        end

        ### check break ###
        status = check_break(residuals, iter, t_start_alg, params)

        ### check restart ###
        check_restart(restart_info, iter, check_iter, params.restart)

        ### update sigma - use operator version for QAP/LASSO ###
        if isa(qp.Q, AbstractQOperator)
            update_sigma_operator(params, restart_info, ws, qp)
        else
            update_sigma(params, restart_info, ws, qp, qp.Q_is_diag, qp.noC)
        end

        ### restart if needed ###
        do_restart(restart_info, ws, qp, qp.noC)

        ### print the log ##
        if print_yes || (status != "CONTINUE")
            print(@sprintf("%5.0f", iter),
                @sprintf("    %3.2e", residuals.err_Rp_org_bar),
                @sprintf("    %3.2e", residuals.err_Rd_org_bar),
                @sprintf("    %7.6e", residuals.primal_obj_bar),
                @sprintf("    %7.6e", residuals.dual_obj_bar),
                @sprintf("    %3.2e", residuals.rel_gap_bar))
            # end
            print(@sprintf("    %3.2e", ws.sigma),
                @sprintf("    %6.2f", time() - t_start_alg))
            println()
        end


        ### collect results and return ###
        if residuals.KKTx_and_gap_org_bar < 1e-4 && first_4
            time_4 = time() - t_start_alg
            iter_4 = iter
            first_4 = false
            println("KKT < 1e-4 at iter = ", iter)
        end
        if residuals.KKTx_and_gap_org_bar < 1e-6 && first_6
            time_6 = time() - t_start_alg
            iter_6 = iter
            first_6 = false
            println("KKT < 1e-6 at iter = ", iter)
        end
        if status != "CONTINUE"
            if status == "OPTIMAL"
                println("The instance is solved, the accuracy is ", residuals.KKTx_and_gap_org_bar)
            elseif status == "MAX_ITER"
                println("The maximum number of iterations is reached, the accuracy is ", residuals.KKTx_and_gap_org_bar)
            elseif status == "TIME_LIMIT"
                println("The time limit is reached, the accuracy is ", residuals.KKTx_and_gap_org_bar)
            end
            results = collect_results_gpu!(ws, residuals, scaling_info, iter, t_start_alg, power_time)
            results.status = status
            results.time_4 = time_4 == 0.0 ? results.time : time_4
            results.iter_4 = iter_4 == 0 ? iter : iter_4
            results.time_6 = time_6 == 0.0 ? results.time : time_6
            results.iter_6 = iter_6 == 0 ? iter : iter_6
            return results
        end

        ## main update - use operator-specific update for QAP/LASSO, standard update otherwise
        if spmv_mode == "operator"
            main_update_operator!(ws, qp, params, restart_info)
        else
            main_update!(ws, qp, spmv_mode, restart_info)
        end

        # Compute M norm for restart decision
        if restart_info.restart_flag > 0
            if isa(qp.Q, AbstractQOperator)
                restart_info.last_gap = compute_M_norm_operator_gpu!(ws, qp)
            else
                restart_info.last_gap = compute_M_norm_gpu!(ws, qp, qp.Q_is_diag, qp.noC)
            end
        end
        if rem(iter + 1, check_iter) == 0
            if isa(qp.Q, AbstractQOperator)
                restart_info.current_gap = compute_M_norm_operator_gpu!(ws, qp)
            else
                restart_info.current_gap = compute_M_norm_gpu!(ws, qp, qp.Q_is_diag, qp.noC)
            end
        end
        restart_info.inner += 1
    end
end