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

# ============================================================================
# Unified Algorithm Functions (CPU and GPU)
# ============================================================================
# These functions use multiple dispatch based on workspace type to handle
# both CPU and GPU implementations. They follow the Q operator pattern.
# ============================================================================

"""
    compute_residuals!(ws, qp, sc, res, params, iter)

Compute residuals for the HPR-QP algorithm. Unified function that dispatches
to appropriate implementation based on workspace type.

Uses unified operations (unified_dot, unified_norm) that dispatch based on array types.
GPU-specific operations (kernels) are handled via dispatch on workspace type.
"""
function compute_residuals!(
    ws::HPRQP_workspace,
    qp::HPRQP_QP_info,
    sc::HPRQP_scaling,
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int
)
    ### Objective values
    # Use unified Qmap! function (already supports both operators and sparse matrices via dispatch)
    # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
    if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRQP_workspace_gpu)
        Qmap!(ws.x_bar, ws.Qx, qp.Q, ws.spmv_Q)
    else
        Qmap!(ws.x_bar, ws.Qx, qp.Q)
    end

    # Compute primal and dual objectives using unified_dot
    res.primal_obj_bar = sc.b_scale * sc.c_scale *
                         (unified_dot(ws.c, ws.x_bar) + 0.5 * unified_dot(ws.x_bar, ws.Qx)) + qp.obj_constant

    # Dual objective: always include z'x term (for bounds/L1), conditionally add y'(Ax-b) term
    res.dual_obj_bar = sc.b_scale * sc.c_scale *
                       (-0.5 * unified_dot(ws.x_bar, ws.Qx) + unified_dot(ws.z_bar, ws.x_bar)) + qp.obj_constant
    if ws.m > 0
        res.dual_obj_bar += sc.b_scale * sc.c_scale * unified_dot(ws.y_bar, ws.s)
    end

    # Add L1 norm term for LASSO problems
    if params.problem_type == "LASSO" && length(ws.lambda) > 0
        ws.tempv .= ws.lambda .* ws.x_bar
        l1_norm = unified_norm(ws.tempv, 1)
        res.primal_obj_bar += sc.b_scale * sc.c_scale * l1_norm
        res.dual_obj_bar += sc.b_scale * sc.c_scale * l1_norm
    end

    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) /
                      (1.0 + max(abs(res.primal_obj_bar), abs(res.dual_obj_bar)))

    ### Dual residuals
    compute_Rd!(ws, sc)
    res.err_Rd_org_bar = unified_norm(ws.Rd, Inf) /
                         (1.0 + maximum([sc.norm_c_org, unified_norm(ws.ATdy, Inf), unified_norm(ws.Qx, Inf)]))

    ### Primal residuals
    if ws.m > 0
        compute_Rp!(ws, sc)
        res.err_Rp_org_bar = unified_norm(ws.Rp, Inf) /
                             (1.0 + max(sc.norm_b_org, unified_norm(ws.Ax, Inf)))
    else
        res.err_Rp_org_bar = 0.0
    end

    # Compute bounds violations at iteration 0 (device-specific via dispatch)
    if iter == 0
        compute_bounds_violation!(ws, sc, res)
    end

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar)

    # Save best values if auto_save is enabled
    if params.auto_save
        if iter == 0 || res.KKTx_and_gap_org_bar < max(ws.saved_state.save_err_Rp,
            ws.saved_state.save_err_Rd,
            ws.saved_state.save_rel_gap)
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

# GPU-specific bounds violation using kernel
function compute_bounds_violation!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu, res::HPRQP_residuals)
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_err_lu_kernel!(ws.dx, ws.x_bar, ws.l, ws.u, sc.col_norm, sc.b_scale, ws.n)
    end
    res.err_Rp_org_bar = max(res.err_Rp_org_bar, unified_norm(ws.dx, Inf))
end

# CPU-specific bounds violation using loop
function compute_bounds_violation!(ws::HPRQP_workspace_cpu, sc::Scaling_info_cpu, res::HPRQP_residuals)
    # Compute bounds violations: max(l - x, 0) for lower, max(x - u, 0) for upper
    lower_violation = max.(ws.l .- ws.x_bar, 0.0)
    upper_violation = max.(ws.x_bar .- ws.u, 0.0)
    ws.dx .= (lower_violation .+ upper_violation) .* (sc.b_scale ./ sc.col_norm)
    res.err_Rp_org_bar = max(res.err_Rp_org_bar, unified_norm(ws.dx, Inf))
end

# ============================================================================
# Legacy GPU-specific version (kept for backward compatibility, calls unified version)
# ============================================================================

# This function computes the residuals for the HPR-QP algorithm on GPU.
function compute_residuals_gpu!(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    sc::Scaling_info_gpu,
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int,
)
    # Call unified version
    compute_residuals!(ws, qp, sc, res, params, iter)
end


# ============================================================================
# Unified save_state_to_hdf5! function
# ============================================================================

"""
    save_state_to_hdf5!(filename, ws, sc, residuals, params, iter, t_start_alg)

Unified implementation of HDF5 state saving for both GPU and CPU workspaces.

Saves the current algorithm state, best solution found so far, and algorithm
parameters to an HDF5 file. This function uses multiple dispatch via the
`to_cpu` helper to automatically transfer GPU arrays to CPU for file I/O
while being a no-op for CPU workspaces.

# Arguments
- `filename::String`: Path to HDF5 file to create/overwrite
- `ws`: Workspace (HPRQP_workspace_gpu or HPRQP_workspace_cpu)
- `sc`: Scaling information (Scaling_info_gpu or Scaling_info_cpu)
- `residuals::HPRQP_residuals`: Current residuals
- `params::HPRQP_parameters`: Algorithm parameters
- `iter::Int`: Current iteration number
- `t_start_alg::Float64`: Algorithm start time

# File Structure
The HDF5 file contains three main groups:
- `current/`: Current iteration state (solution, residuals, iteration info)
- `best/`: Best solution found so far (solution, residuals)
- `parameters/`: Algorithm parameters and settings

All solution vectors are saved in their original (unscaled) space.
"""
function save_state_to_hdf5!(
    filename::String,
    ws::Union{HPRQP_workspace_gpu,HPRQP_workspace_cpu},
    sc::Union{Scaling_info_gpu,Scaling_info_cpu},
    residuals::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int,
    t_start_alg::Float64,
)
    # Transfer arrays to CPU (no-op for CPU workspace)
    x_bar = to_cpu(ws.x_bar)
    y_bar = to_cpu(ws.y_bar)
    z_bar = to_cpu(ws.z_bar)
    w_bar = to_cpu(ws.w_bar)
    save_x = to_cpu(ws.saved_state.save_x)
    save_y = to_cpu(ws.saved_state.save_y)
    save_z = to_cpu(ws.saved_state.save_z)
    save_w = to_cpu(ws.saved_state.save_w)
    col_norm = to_cpu(sc.col_norm)
    row_norm = to_cpu(sc.row_norm)

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
        file["best/KKTx_and_gap"] = max(ws.saved_state.save_err_Rp, ws.saved_state.save_err_Rd, ws.saved_state.save_rel_gap)

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

    if params.verbose
        println(@sprintf("State saved to %s at iteration %d", filename, iter))
    end
end

# ============================================================================
# Unified Algorithm Functions (Block 2 Refactoring)
# ============================================================================

"""
    update_sigma!(params, restart_info, ws, qp, residuals)

Unified implementation of adaptive sigma update for both GPU and CPU.
Updates the primal penalty parameter sigma based on the progress of the algorithm.

This function uses multiple dispatch and unified operations to work with both
GPU and CPU workspaces without code duplication.

# Arguments
- `params::HPRQP_parameters`: Algorithm parameters
- `restart_info::HPRQP_restart`: Restart tracking information
- `ws::HPRQP_workspace`: Workspace (GPU or CPU)
- `qp::HPRQP_QP_info`: Problem data (GPU or CPU)
- `residuals::HPRQP_residuals`: Current residual values
"""
function update_sigma!(params::HPRQP_parameters,
    restart_info::HPRQP_restart,
    ws::HPRQP_workspace,
    qp::HPRQP_QP_info,
    residuals::HPRQP_residuals,
)
    if ~params.sigma_fixed && (restart_info.restart_flag >= 1)
        sigma_old = ws.sigma

        # Compute differences: dx = x_bar - last_x, dw = w_bar - last_w
        ws.dx .= ws.x_bar .- ws.last_x
        ws.dw .= ws.w_bar .- ws.last_w

        # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
        # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRQP_workspace_gpu)
            Qmap!(ws.dw, ws.dQw, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.dw, ws.dQw, qp.Q)
        end

        a = 0.0
        b = unified_dot(ws.dx, ws.dx)
        c = 0.0
        d = 0.0

        if ws.m > 0
            ws.dy .= ws.y_bar .- ws.last_y
            ws.ATdy .= ws.ATy_bar .- ws.last_ATy

            # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
            # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
            if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRQP_workspace_gpu)
                Qmap!(ws.ATdy, ws.QATdy, qp.Q, ws.spmv_Q)
            else
                Qmap!(ws.ATdy, ws.QATdy, qp.Q)
            end
            a = ws.lambda_max_A * unified_dot(ws.dy, ws.dy) - 2 * unified_dot(ws.dQw, ws.ATdy)
        end

        if ws.Q_is_diag
            a += unified_norm(ws.dQw)^2
        else
            a += ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
            if ws.m > 0
                c = unified_dot(ws.ATdy, ws.QATdy)
                d = ws.lambda_max_Q
            end
        end

        a = max(a, 1e-12)
        b = max(b, 1e-12)

        # Estimate optimal sigma
        if ws.Q_is_diag
            if ws.m > 0
                sigma_estimation = unified_golden_Q_diag(a, b, ws.diag_Q, ws.ATdy, ws.QATdy, ws.tempv;
                    lo=1e-12, hi=1e12, tol=1e-13)
            else
                # No constraints: simplified sigma update for diagonal Q
                sigma_estimation = sqrt(b / a)
            end
        else
            # min a * x + b / x + c * x^2 / (1 + d * x)
            if ws.m > 0
                sigma_estimation = golden(a, b, c, d; lo=1e-12, hi=1e12, tol=1e-13)
            else
                # No constraints: simplified sigma update
                sigma_estimation = sqrt(b / a)
            end
        end

        # Compute adaptive sigma with gap-based blending
        fact = exp(-0.05 * (restart_info.current_gap / restart_info.best_gap))
        temp_1 = max(min(residuals.err_Rd_org_bar, residuals.err_Rp_org_bar),
            min(residuals.rel_gap_bar, restart_info.current_gap))
        sigma_cand = exp(fact * log(sigma_estimation) + (1 - fact) * log(restart_info.best_sigma))

        # Compute scaling factor κ based on infeasibility ratio
        if temp_1 > 9e-10
            κ = 1.0
        elseif temp_1 > 5e-10
            ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
            κ = clamp(sqrt(ratio_infeas_org), 1e-2, 100.0)
        else
            ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
            κ = clamp(ratio_infeas_org, 1e-2, 100.0)
        end
        ws.sigma = κ * sigma_cand

        # Update Q factors if sigma changes (only for diagonal Q)
        if ws.Q_is_diag
            if abs(sigma_old - ws.sigma) > 1e-15
                unified_update_Q_factors!(
                    ws.fact2, ws.fact, ws.fact1, ws.fact_M,
                    ws.diag_Q, ws.sigma
                )
            end
        end
    end
end

# ============================================================================
# Legacy GPU Wrapper (calls unified version)
# ============================================================================

# This function updates the penalty parameter (sigma) based on the current state of the algorithm.
function update_sigma_gpu!(params::HPRQP_parameters,
    restart_info::HPRQP_restart,
    ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    residuals::HPRQP_residuals,
)
    # Call unified implementation
    update_sigma!(params, restart_info, ws, qp, residuals)
end

# This function checks whether a restart is needed based on the current state of the algorithm.
function check_restart(restart_info::HPRQP_restart,
    iter::Int,
    check_iter::Int,
    sigma::Float64,
)
    restart_info.restart_flag = 0
    if restart_info.first_restart
        if iter == check_iter
            restart_info.first_restart = false
            restart_info.restart_flag = 1
            restart_info.weighted_norm = restart_info.current_gap
            restart_info.best_gap = restart_info.current_gap
            restart_info.best_sigma = sigma
        end
    else
        if rem(iter, check_iter) == 0
            if restart_info.current_gap < 0
                restart_info.current_gap = 1e-6
                println("current_gap < 0")
            end

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

            # Update best_gap and best_sigma if current gap is better
            if restart_info.best_gap > restart_info.current_gap
                restart_info.best_gap = restart_info.current_gap
                restart_info.best_sigma = sigma
            end

            restart_info.save_gap = restart_info.current_gap
        end
    end
end

"""
    do_restart!(restart_info, ws, qp)

Unified implementation of restart operation for both GPU and CPU.

Performs a restart by resetting the algorithm state to the current averaged iterates.
This operation is triggered when the restart_flag is set (>0) by check_restart.

# What Happens During Restart
1. Set current iterates (x, w, y) to averaged iterates (x̄, w̄, ȳ)
2. Set last iterates (last_x, last_w, last_y) to averaged iterates
3. Recompute ATy_bar using the new y_bar
4. Update restart tracking information

# Arguments
- `restart_info::HPRQP_restart`: Restart tracking structure (modified in-place)
- `ws::HPRQP_workspace`: Workspace with iterate vectors (modified in-place)
- `qp::HPRQP_QP_info`: Problem data (used for matrix A)

# Why Restart?
Restarting helps when:
- Progress stalls (long periods without gap reduction)
- Gap reduces sufficiently (sufficient condition)
- Gap increases after recent progress (necessary condition)

# See Also
- `check_restart`: Determines when to restart based on gap progress
"""
function do_restart!(restart_info::HPRQP_restart,
    ws::HPRQP_workspace,
    qp::HPRQP_QP_info)
    if restart_info.restart_flag > 0
        # Reset current iterates to averaged iterates
        ws.x .= ws.x_bar
        ws.w .= ws.w_bar
        ws.last_x .= ws.x_bar
        ws.last_w .= ws.w_bar

        # Handle constraint-related variables if constraints exist
        if ws.m > 0
            ws.y .= ws.y_bar
            ws.last_y .= ws.y_bar

            # Compute AT*y_bar using unified matrix-vector multiplication
            unified_mul!(ws.ATy_bar, ws.AT, ws.y_bar)

            ws.last_ATy .= ws.ATy_bar
            ws.ATy .= ws.ATy_bar
        end

        if ws.noC
            ws.Qw .= ws.Qw_bar
            ws.last_Qw .= ws.Qw_bar
        end

        # Update restart tracking information
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

"""
    collect_results!(ws, qp, sc, residuals, iter, t_start_alg, power_time)

Unified implementation for collecting and scaling final results from both GPU and CPU solvers.

This function:
1. Creates a new HPRQP_results object
2. Copies timing and iteration information
3. Scales and de-normalizes solution vectors (x, w, y, z)
4. Transfers GPU data to CPU if needed

# Arguments
- `ws::HPRQP_workspace`: Workspace containing solution vectors
- `qp::HPRQP_QP_info`: Problem data (used for dimension checking)
- `sc::HPRQP_scaling`: Scaling information for de-normalization
- `residuals::HPRQP_residuals`: Final residual values
- `iter::Int`: Final iteration count
- `t_start_alg::Float64`: Algorithm start time
- `power_time::Float64`: Total power measurement time (GPU only, default 0.0)

# Returns
- `HPRQP_results`: Results object with scaled solution and metadata

# Scaling Operations
- `x = b_scale * (x_bar ./ col_norm)`: Primal variable x
- `w = b_scale * (w_bar ./ col_norm)`: Slack variable w  
- `y = c_scale * (y_bar ./ row_norm)`: Dual variable y
- `z = c_scale * (z_bar .* col_norm)`: Reduced cost z

# Note on GPU Transfer
For GPU workspaces, `to_cpu()` automatically transfers CuArrays to Arrays.
For CPU workspaces, it's a no-op that returns the arrays unchanged.
"""
function collect_results!(
    ws::HPRQP_workspace,
    qp::Union{HPRQP_QP_info,Nothing},
    sc::HPRQP_scaling,
    residuals::HPRQP_residuals,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64=0.0
)
    results = HPRQP_results()
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar

    # Scale solution and transfer to CPU if needed
    # to_cpu() is a no-op for CPU arrays, transfers GPU arrays to CPU
    results.w = to_cpu(sc.b_scale * (ws.w_bar ./ sc.col_norm))
    results.x = to_cpu(sc.b_scale * (ws.x_bar ./ sc.col_norm))

    # Handle dual variables (may be empty for unconstrained problems)
    if ws.m > 0
        results.y = to_cpu(sc.c_scale * (ws.y_bar ./ sc.row_norm))
    else
        results.y = Float64[]
    end

    results.z = to_cpu(sc.c_scale * (ws.z_bar .* sc.col_norm))

    return results
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
    # Call unified implementation (qp not needed but kept for signature compatibility)
    return collect_results!(ws, nothing, sc, residuals, iter, t_start_alg, power_time)
end

# ============================================================================
# CPU Algorithm Functions
# ============================================================================

# CPU version of compute residuals
function compute_residuals_cpu!(ws::HPRQP_workspace_cpu,
    qp::QP_info_cpu,
    sc::Scaling_info_cpu,
    res::HPRQP_residuals,
    params::HPRQP_parameters,
    iter::Int)
    # Call unified version
    compute_residuals!(ws, qp, sc, res, params, iter)
end

"""
    compute_M_norm!(ws, qp)

Unified implementation of M-norm computation for both GPU and CPU.
Computes the norm of the M matrix used in convergence analysis.

The M-norm is a weighted norm that combines primal, dual, and constraint-related
terms based on the current iterate differences and problem structure.

# Arguments
- `ws::HPRQP_workspace`: Workspace (GPU or CPU)
- `qp::HPRQP_QP_info`: Problem data (GPU or CPU)

# Returns
- Float64: The computed M-norm value
"""
function compute_M_norm!(ws::HPRQP_workspace, qp::HPRQP_QP_info)
    # Initialize M terms
    M_1 = 0.0
    M_2 = (1.0 / ws.sigma) * unified_dot(ws.dx, ws.dx)
    M_3 = 0.0

    # Use unified Qmap! function (dispatch handles operator vs sparse matrix)
    # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
    if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRQP_workspace_gpu)
        Qmap!(ws.dw, ws.dQw, qp.Q, ws.spmv_Q)
    else
        Qmap!(ws.dw, ws.dQw, qp.Q)
    end
    M_2 -= 2.0 * unified_dot(ws.dQw, ws.dx)

    # Add constraint-related terms if constraints exist
    if ws.m > 0
        M_1 = ws.sigma * ws.lambda_max_A * unified_dot(ws.dy, ws.dy)
        unified_mul!(ws.ATdy, ws.AT, ws.dy)
        # Pass spmv_Q for GPU sparse matrices to use preprocessed CUSPARSE
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32}) && isa(ws, HPRQP_workspace_gpu)
            Qmap!(ws.ATdy, ws.QATdy, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.ATdy, ws.QATdy, qp.Q)
        end
        M_1 -= 2.0 * ws.sigma * unified_dot(ws.dQw, ws.ATdy)
        M_2 += 2.0 * unified_dot(ws.ATdy, ws.dx)

        if ws.Q_is_diag
            ws.ATdy .*= ws.fact_M
            M_3 = unified_dot(ws.ATdy, ws.QATdy) # sGS term
            M_1 += ws.sigma * unified_dot(ws.dQw, ws.dQw)
        else
            M_3 = (ws.sigma^2) / (1.0 + ws.sigma * ws.lambda_max_Q) * unified_dot(ws.ATdy, ws.QATdy)  # sGS term
            M_1 += ws.sigma * ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
        end
    elseif !ws.Q_is_diag
        # No constraints case: only add Q-related term to M_1
        M_1 = ws.sigma * ws.lambda_max_Q * unified_dot(ws.dw, ws.dQw)
    end

    M_2 += max(M_1, 0.0)
    M_norm = max(M_2, 0.0) + max(M_3, 0.0)

    # Check for numerical instability
    if min(M_1, M_2, M_3) < -1e-8
        println("M_1 = $M_1, M_2 = $M_2, M_3 = $M_3, negative M norm due to numerical instability, consider increasing eig_factor")
    end

    return sqrt(M_norm)
end

# CPU version of compute M norm
function compute_M_norm_cpu!(ws::HPRQP_workspace_cpu, qp::QP_info_cpu)
    # Call unified implementation
    return compute_M_norm!(ws, qp)
end

# ============================================================================
# Unified Workspace Allocation
# ============================================================================

"""
    allocate_workspace(qp, params, lambda_max_A, lambda_max_Q, scaling_info, diag_Q, Q_is_diag)

Unified workspace allocation function that works for both CPU and GPU.
Dispatches to appropriate implementation based on qp type.

This replaces the separate allocate_workspace_cpu and allocate_workspace_gpu functions,
reducing code duplication and ensuring consistency.
"""
function allocate_workspace(
    qp::HPRQP_QP_info,
    params::HPRQP_parameters,
    lambda_max_A::Float64,
    lambda_max_Q::Float64,
    scaling_info::HPRQP_scaling,
    diag_Q::Vector{Float64},
    Q_is_diag::Bool
)
    # Determine workspace type from qp type
    WS = workspace_type(qp)

    # Create workspace
    ws = WS()
    m, n = size(qp.A)
    ws.m = m
    ws.n = n

    # Allocate vectors using type-dispatched helper
    ws.w = allocate_vector(WS, Float64, n)
    ws.w_hat = allocate_vector(WS, Float64, n)
    ws.w_bar = allocate_vector(WS, Float64, n)
    ws.dw = allocate_vector(WS, Float64, n)
    ws.x = allocate_vector(WS, Float64, n)
    ws.x_hat = allocate_vector(WS, Float64, n)
    ws.x_bar = allocate_vector(WS, Float64, n)
    ws.dx = allocate_vector(WS, Float64, n)
    ws.y = allocate_vector(WS, Float64, m)
    ws.y_hat = allocate_vector(WS, Float64, m)
    ws.y_bar = allocate_vector(WS, Float64, m)
    ws.dy = allocate_vector(WS, Float64, m)
    ws.z_bar = allocate_vector(WS, Float64, n)

    # Assign QP problem data
    ws.Q = qp.Q
    ws.A = qp.A
    ws.AT = qp.AT
    ws.AL = qp.AL
    ws.AU = qp.AU
    ws.c = qp.c
    ws.l = qp.l
    ws.u = qp.u

    # Allocate work vectors
    ws.Rp = allocate_vector(WS, Float64, m)
    ws.Rd = allocate_vector(WS, Float64, n)
    ws.Ax = allocate_vector(WS, Float64, m)
    ws.ATy = allocate_vector(WS, Float64, n)
    ws.ATy_bar = allocate_vector(WS, Float64, n)
    ws.ATdy = allocate_vector(WS, Float64, n)
    ws.QATdy = allocate_vector(WS, Float64, n)
    ws.s = allocate_vector(WS, Float64, m)
    ws.Qw = allocate_vector(WS, Float64, n)
    ws.Qw_hat = allocate_vector(WS, Float64, n)
    ws.Qw_bar = allocate_vector(WS, Float64, n)
    ws.Qx = allocate_vector(WS, Float64, n)
    ws.dQw = allocate_vector(WS, Float64, n)
    ws.last_x = allocate_vector(WS, Float64, n)
    ws.last_y = allocate_vector(WS, Float64, m)
    ws.last_Qw = allocate_vector(WS, Float64, n)
    ws.last_w = allocate_vector(WS, Float64, n)
    ws.last_ATy = allocate_vector(WS, Float64, n)
    ws.tempv = allocate_vector(WS, Float64, n)

    # Set Q properties
    ws.Q_is_diag = Q_is_diag
    ws.diag_Q = convert_to_device(WS, diag_Q)

    # Allocate factorization vectors
    ws.fact1 = allocate_vector(WS, Float64, n)
    ws.fact2 = allocate_vector(WS, Float64, n)
    ws.fact = allocate_vector(WS, Float64, n)
    ws.fact_M = allocate_vector(WS, Float64, n)

    # Set eigenvalue bounds
    ws.lambda_max_A = lambda_max_A
    ws.lambda_max_Q = lambda_max_Q

    # Compute sigma
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

    # Compute factors for diagonal Q
    if ws.Q_is_diag
        # Use broadcasting for GPU-compatible computation
        temp = 1.0 .+ ws.sigma .* ws.diag_Q
        ws.fact1 .= 1.0 ./ temp
        ws.fact2 .= (ws.sigma .* ws.diag_Q) ./ temp
        ws.fact_M .= (ws.sigma^2 .* ws.diag_Q) ./ temp
    end

    # Convert scalar lambda to vector (same for both CPU and GPU)
    ws.lambda = fill_vector(WS, qp.lambda, n)

    # Set to_check flag
    ws.to_check = true

    # Initialize saved state if auto_save enabled
    if params.auto_save
        ws.saved_state = allocate_saved_state(WS)
        ws.saved_state.save_x = allocate_vector(WS, Float64, n)
        ws.saved_state.save_y = allocate_vector(WS, Float64, m)
        ws.saved_state.save_z = allocate_vector(WS, Float64, n)
        ws.saved_state.save_w = allocate_vector(WS, Float64, n)
        ws.saved_state.save_sigma = ws.sigma
        ws.saved_state.save_iter = 0
        ws.saved_state.save_err_Rp = Inf
        ws.saved_state.save_err_Rd = Inf
        ws.saved_state.save_primal_obj = Inf
        ws.saved_state.save_dual_obj = Inf
        ws.saved_state.save_rel_gap = Inf
    end

    return ws
end

# ============================================================================
# CPU Main Update Functions
# ============================================================================

# Main update function for CPU - dispatches to appropriate update based on problem type
"""  
    main_update_cpu!(ws, qp, restart_info)

CPU-specific main iteration update for the HPR-QP algorithm.

This function performs the core primal-dual update step using CPU-optimized loops.
It remains separate from GPU version because it calls device-specific kernels/loops.

# Algorithm Overview
The update follows the Halpern iteration scheme:
  - Compute Halpern averaging factors: α₁ = 1/(k+2), α₂ = 1 - α₁
  - Update primal variables (z, x, w) using proximal operators
  - Update dual variables (y) via gradient ascent
  - Apply Halpern averaging to produce (x̄, w̄, ȳ)

# Three Update Paths:

1. **LASSO Operator**:
   - Q is a LASSO regularization operator
   - Uses soft-thresholding proximal operator
   - Calls: update_zxw_LASSO_cpu! with precomputed factors

2. **Standard QP (Sparse Matrix or QAP) with non-empty Q**:
   - Q is sparse matrix or QAP operator
   - Three-step update process:
     * Step 1: Update z, x, w (without dual correction)
     * Step 2: Update dual y variables  
     * Step 3: Complete w update (add dual correction)
   - Handles diagonal Q with precomputed factor vectors
   - Handles non-diagonal Q with scalar factors

3. **Empty Q (Linear Program)**: 
   - No Q matrix present (LP instead of QP)
   - Simplified updates without proximal operator for Q
   - Calls: unified_update_zx_cpu!, unified_update_y_noQ_cpu!

# Arguments
- `ws::HPRQP_workspace_cpu`: CPU workspace containing all iterate vectors
- `qp::QP_info_cpu`: Problem data (Q, A, b, c, etc.)
- `restart_info::HPRQP_restart`: Restart tracking (provides iteration count)

# Implementation Notes
- **Why CPU-specific**: Calls CPU loop functions instead of GPU kernels
- **Cannot be unified**: Device-specific execution paths (loops vs kernels)
- **Diagonal Q optimization**: Uses precomputed factor vectors when Q is diagonal
- **Operator dispatch**: Different update logic for LASSO/QAP/sparse matrix Q
- **Structure matches GPU**: Same branching logic as main_update_gpu! for consistency

# See Also
- `main_update_gpu!`: GPU version using CUDA kernels
- `update_zxw1_cpu!`, `update_y_cpu!`, `update_w2_cpu!`: Standard QP updates
- `update_zxw_LASSO_cpu!`: LASSO-specific update
"""
function main_update_cpu!(ws::HPRQP_workspace_cpu, qp::QP_info_cpu, restart_info::HPRQP_restart)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    Halpern_fact2 = 1.0 - Halpern_fact1

    # Handle operator-based Q (QAP/LASSO) within main_update
    if isa(qp.Q, LASSO_Q_operator_cpu)
        # LASSO update with soft-thresholding (no A matrix for LASSO)
        update_zxw_LASSO_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        return
    end
    if isa(qp.Q, QAP_Q_operator_cpu) || (isa(qp.Q, SparseMatrixCSC) && length(qp.Q.nzval) > 0)
        if ws.noC
            # noC case (no constraint c): use unified kernels that combine z, x, w updates
            unified_update_zxw_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            unified_update_y_cpu!(ws, Halpern_fact1, Halpern_fact2)
            return
        else
            # Standard case with Q matrix - use unified kernels with separate Q and A modes
            update_zxw1_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            update_y_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            update_w2_cpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        end
    else
        # Empty Q case (linear program) - use unified kernels with A mode only
        unified_update_zx_cpu!(ws, Halpern_fact1, Halpern_fact2)
        unified_update_y_noQ_cpu!(ws, Halpern_fact1, Halpern_fact2)
    end
end

# ============================================================================
# CPU Sigma Update
# ============================================================================

function update_sigma_cpu!(params::HPRQP_parameters,
    restart_info::HPRQP_restart,
    ws::HPRQP_workspace_cpu,
    qp::QP_info_cpu,
    residuals::HPRQP_residuals,
)
    # Call unified implementation
    update_sigma!(params, restart_info, ws, qp, residuals)
end

# ============================================================================
# CPU Collect Results  
# ============================================================================

function collect_results_cpu!(ws::HPRQP_workspace_cpu, qp::QP_info_cpu, scaling_info::Scaling_info_cpu, results::HPRQP_results)
    # Call unified implementation with dummy timing values (results object already has timing)
    # This maintains backward compatibility where results object is passed in
    temp_results = collect_results!(ws, qp, scaling_info, HPRQP_residuals(),
        results.iter, 0.0, 0.0)

    # Copy solution vectors to the provided results object
    results.x = temp_results.x
    results.w = temp_results.w
    results.y = temp_results.y
    results.z = temp_results.z
end

# ============================================================================
# CPU Helper Functions (restart, termination, etc.)
# ============================================================================


# ============================================================================
# Unified handle_termination function
# ============================================================================

"""
    handle_termination(status, residuals, ws, scaling_info, iter, t_start_alg, 
                      power_time, setup_time, iter_4, time_4, iter_6, time_6, verbose)

Unified termination handler that works for both GPU and CPU workspaces.

Collects final results, prints solution summary, and returns HPRQP_results.
Automatically handles GPU->CPU transfer when needed via collect_results!.

# Arguments
- `status::String`: Termination status ("OPTIMAL", "MAX_ITER", "TIME_LIMIT")
- `residuals::HPRQP_residuals`: Final residuals
- `ws::HPRQP_workspace`: Workspace (GPU or CPU)
- `scaling_info::HPRQP_scaling`: Scaling information (GPU or CPU)
- `iter::Int`: Final iteration number
- `t_start_alg::Float64`: Algorithm start time
- `power_time::Float64`: Time spent in power iteration
- `setup_time::Float64`: Setup time
- `iter_4, time_4`: Milestone tracking for 1e-4 accuracy
- `iter_6, time_6`: Milestone tracking for 1e-6 accuracy
- `verbose::Bool`: Whether to print output

# Returns
- `HPRQP_results`: Final results structure
"""
function handle_termination(
    status::String,
    residuals::HPRQP_residuals,
    ws::HPRQP_workspace,
    scaling_info::HPRQP_scaling,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
    setup_time::Float64,
    iter_4::Int,
    time_4::Float64,
    iter_6::Int,
    time_6::Float64,
    verbose::Bool
)
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

    # Collect results using unified function (handles GPU->CPU transfer automatically)
    results = collect_results!(ws, nothing, scaling_info, residuals, iter,
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

# CPU version of print_problem_info
function print_problem_info(qp::HPRQP_QP_info, ws::HPRQP_workspace, params::HPRQP_parameters)
    if !params.verbose
        return
    end

    m, n = size(qp.A)

    println("="^80)
    println("QP PROBLEM INFORMATION")
    println("="^80)

    # Determine QP type using helper functions
    qp_type = if is_q_operator(qp.Q)
        get_operator_name(typeof(qp.Q))
    else
        # Q is a sparse matrix
        if get_Q_nnz(qp.Q) > 0
            "QP (Quadratic Program - Non-empty Q)"
        else
            "LP (Linear Program - Empty Q)"
        end
    end
    println("Problem Type: $qp_type")

    # Q matrix information
    if is_q_operator(qp.Q)
        op_name = get_operator_name(typeof(qp.Q))
        println("Q Operator: $op_name operator (implicit matrix)")
    else
        # Q is a sparse matrix
        q_size = size(qp.Q, 1)
        q_nnz = get_Q_nnz(qp.Q)
        println("Q Matrix: $(q_size)×$(q_size), nnz = $q_nnz")
        if q_nnz > 0
            println("Q is Diagonal: $(ws.Q_is_diag)")
        end
    end

    # Constraint matrix information
    if m > 0
        a_nnz = get_A_nnz(qp.A)
        println("A Matrix: $(m)×$(n), nnz = $a_nnz")
    else
        println("A Matrix: No constraints (unconstrained)")
    end

    println()
end

# ============================================================================
# GPU Algorithm Functions  
# ============================================================================

# This function initializes the restart information for the HPR-QP algorithm.
function initialize_restart()
    restart_info = HPRQP_restart()
    restart_info.first_restart = true
    restart_info.save_gap = Inf
    restart_info.current_gap = Inf
    restart_info.last_gap = Inf
    restart_info.best_gap = Inf
    restart_info.best_sigma = 1.0
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
"""  
    main_update_gpu!(ws, qp, spmv_mode_Q, spmv_mode_A, restart_info)

GPU-specific main iteration update for the HPR-QP algorithm.

This function performs the core primal-dual update step using CUDA kernels.
It remains separate from CPU version because it dispatches to GPU-optimized kernels.

# Algorithm Overview
The update follows the Halpern iteration scheme:
  - Compute Halpern averaging factors: α₁ = 1/(k+2), α₂ = 1 - α₁  
  - Update primal variables (z, x, w) using proximal operators
  - Update dual variables (y) via gradient ascent
  - Apply Halpern averaging to produce (x̄, w̄, ȳ)

# Execution Modes

**Operator Mode** (`spmv_mode_Q == "operator"`):
  - Q is a structured operator (LASSO or QAP)
  - LASSO: Soft-thresholding proximal with no constraints
  - QAP: Uses unified kernels with operator-based Q multiplication

**Sparse Matrix Mode**:
  - Q is a sparse matrix (standard QP)
  - **Empty Q** (linear program): Simplified updates without Q terms
  - **Non-empty Q**: Three-step update with configurable SpMV modes

# SpMV Mode Parameters
- `spmv_mode_Q`: How to multiply Q matrix ("CSR", "operator", etc.)
- `spmv_mode_A`: How to multiply A matrix ("CSR", "CSC", "hybrid", etc.)
- Different modes optimize memory access patterns on GPU

# Three Update Paths:

1. **LASSO Operator** (soft-thresholding):
   - Calls: update_zxw_LASSO_gpu!
   - No constraint matrix A (unconstrained regularization)

2. **QAP/Other Operators**:
   - Calls: unified_update_zxw1_gpu!, unified_update_y_gpu!, unified_update_w2_gpu!
   - Uses operator mode for Q, configurable mode for A

3. **Sparse Matrix Q**:
   - **Empty Q** (LP): unified_update_zx_gpu!, unified_update_y_noQ_gpu!
   - **Non-empty Q** (QP): Three-step update with configurable SpMV modes
   - Handles diagonal Q optimization with precomputed factors

# Arguments  
- `ws::HPRQP_workspace_gpu`: GPU workspace containing all iterate vectors
- `qp::QP_info_gpu`: Problem data on GPU (Q, A, b, c, etc.)
- `spmv_mode_Q::String`: Sparse matrix-vector product mode for Q
- `spmv_mode_A::String`: Sparse matrix-vector product mode for A
- `restart_info::HPRQP_restart`: Restart tracking (provides iteration count)

# Implementation Notes
- **Why GPU-specific**: Launches CUDA kernels instead of CPU loops
- **Cannot be unified**: Fundamentally different execution model (parallel vs serial)
- **compute_full flag**: Derived from ws.to_check, controls full vs partial kernel execution
- **Diagonal Q optimization**: Uses precomputed factor vectors for diagonal Q
- **SpMV mode selection**: Allows runtime selection of optimal sparse multiplication

# Performance Considerations
- Kernel launch overhead is amortized over large problem dimensions
- Different SpMV modes trade off memory bandwidth vs computation
- Diagonal Q path uses vectorized operations instead of SpMV

# See Also
- `main_update_cpu!`: CPU version using loops
- `unified_update_zxw1_gpu!`, `unified_update_y_gpu!`, `unified_update_w2_gpu!`: Standard QP kernels
- `update_zxw_LASSO_gpu!`: LASSO-specific kernel
"""
function main_update_gpu!(ws::HPRQP_workspace_gpu,
    qp::QP_info_gpu,
    restart_info::HPRQP_restart,
)
    Halpern_fact1 = 1.0 / (restart_info.inner + 2.0)
    Halpern_fact2 = 1.0 - Halpern_fact1

    # Handle operator-based Q (QAP/LASSO) within main_update
    if isa(qp.Q, LASSO_Q_operator_gpu)
        # LASSO update with soft-thresholding (no A matrix for LASSO)
        update_zxw_LASSO_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
        return
    end
    if isa(qp.Q, QAP_Q_operator_gpu) || length(qp.Q.nzVal) > 0
        if ws.noC
            unified_update_zxw_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            unified_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2)
            return
        else
            # Standard case with Q matrix - use unified kernels with separate Q and A modes
            unified_update_zxw1_gpu!(ws, qp, Halpern_fact1, Halpern_fact2)
            unified_update_y_gpu!(ws, Halpern_fact1, Halpern_fact2)
            unified_update_w2_gpu!(ws, Halpern_fact1, Halpern_fact2)
        end
    else
        # Empty Q case (linear program) - use unified kernels with A mode only
        unified_update_zx_gpu!(ws, Halpern_fact1, Halpern_fact2)
        unified_update_y_noQ_gpu!(ws, Halpern_fact1, Halpern_fact2)
    end
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

    # Lambda is kept as scalar in QP_info_gpu (matching CPU)
    # It will be converted to CuVector in allocate_workspace_gpu using CUDA.fill
    lambda_scalar = model.lambda

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
        lambda_scalar,
    )

    CUDA.synchronize()
    transfer_time = time() - t_start
    if params.verbose
        println(@sprintf("COPY TO GPU time: %.2f seconds", transfer_time))
    end

    return qp, transfer_time
end

# Unified model preparation function
# Handles both GPU transfer and CPU copy with consistent interface
function prepare_model(model::QP_info_cpu, params::HPRQP_parameters)
    if params.use_gpu
        return transfer_to_gpu(model, params)
    else
        # CPU: work on a copy to avoid modifying original, no transfer time
        return deepcopy(model), 0.0
    end
end

# Print solver parameters
function print_solver_params(params::HPRQP_parameters, qp::Union{QP_info_gpu,QP_info_cpu}, spmv_mode_Q::String="", spmv_mode_A::String="")
    if !params.verbose
        return
    end

    m = size(qp.A, 1)
    n = size(qp.A, 2)

    println("="^80)
    println("SOLVER PARAMETERS:")
    println("  Problem size: m = ", m, ", n = ", n)
    println("  Device: ", params.use_gpu ? "GPU (device $(params.device_number))" : "CPU")
    println("  Stop tolerance: ", params.stoptol)
    println("  Max iterations: ", params.max_iter)
    println("  Time limit: ", params.time_limit, " seconds")
    println("  Check interval: ", params.check_iter)
    println("  Print frequency: ", params.print_frequency == -1 ? "Adaptive" : params.print_frequency)
    println("  Eigenvalue factor: ", params.eig_factor)
    println("  Sigma fixed: ", params.sigma_fixed)
    # Only print SpMV modes if they're non-empty (GPU sets them, CPU leaves empty)
    if !isempty(spmv_mode_Q)
        println("  SpMV mode Q: ", spmv_mode_Q, params.spmv_mode_Q == "auto" ? " (auto-detected)" : "")
        println("  SpMV mode A: ", spmv_mode_A, params.spmv_mode_A == "auto" ? " (auto-detected)" : "")
    end
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
        memory_bytes = (n + m + 2 * n) * 16  # x, y, z, w (8 bytes per Float64, 2 copies)
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
function estimate_eigenvalues(qp::HPRQP_QP_info, params::HPRQP_parameters, ws::HPRQP_workspace)
    if params.verbose
        println("ESTIMATING MAXIMUM EIGENVALUES ...")
    end
    t_start = time()
    CUDA.synchronize()

    m = size(qp.A, 1)

    # Estimate lambda_max_A using preprocessed SpMV structures from workspace
    if m > 0
        lambda_max_A = power_iteration_A(ws) * params.eig_factor
    else
        lambda_max_A = 0.0
    end

    # Estimate lambda_max_Q based on Q type using unified dispatch
    lambda_max_Q = compute_lambda_max_Q(qp.Q, ws, params.eig_factor)

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

# ==================== Helper Function for CPU/GPU Dispatch ====================

# Helper function to select GPU or CPU function implementations
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

See also: [`build_from_mps`](@ref), [`build_from_QAbc`](@ref)
"""
function optimize(model::QP_info_cpu, params::HPRQP_parameters)
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

    return results
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
    ws.spmv_mode_Q = "CUSPARSE"  # Dummy value
    ws.spmv_mode_A = "CUSPARSE"  # Test CUSPARSE mode for A
    CUDA.@sync begin end
    t_A_cusparse_start = time()
    for _ in 1:30
        unified_update_y_gpu!(ws, 0.5, 0.5)
        unified_update_w2_gpu!(ws, 0.5, 0.5)
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
    ws.spmv_mode_Q = "customized"  # Dummy value
    ws.spmv_mode_A = "customized"  # Test customized mode for A
    CUDA.@sync begin end
    t_A_custom_start = time()
    for _ in 1:30
        unified_update_y_gpu!(ws, 0.5, 0.5)
        unified_update_w2_gpu!(ws, 0.5, 0.5)
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
            ws.spmv_mode_Q = "CUSPARSE"  # Ensure spmv_mode_Q is set for the test
            ws.spmv_mode_A = "CUSPARSE"  # Dummy value for A
            CUDA.@sync begin end
            t_Q_cusparse_start = time()
            for _ in 1:30
                unified_update_zxw1_gpu!(ws, qp, 0.5, 0.5)
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
            ws.spmv_mode_Q = "customized"  # Ensure spmv_mode_Q is set for the test
            ws.spmv_mode_A = "customized"  # Dummy value for A
            CUDA.@sync begin end
            t_Q_custom_start = time()
            for _ in 1:30
                unified_update_zxw1_gpu!(ws, qp, 0.5, 0.5)
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

# CPU stub - no SpMV mode selection needed for CPU
# CPU always uses standard LinearAlgebra operations, no benchmarking required
function determine_spmv_mode(qp::QP_info_cpu, params::HPRQP_parameters, ws::HPRQP_workspace_cpu)
    return ("", "")  # CPU doesn't use SpMV modes
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

"""
    process_initial_points!(ws, qp, params, scaling_info, m)

Process initial primal and dual points if provided in parameters.
Scales and assigns initial values to workspace variables.

# Arguments
- `ws`: Workspace containing solver state
- `qp`: QP problem information
- `params`: Solver parameters (containing initial_x and initial_y)
- `scaling_info`: Scaling information for the problem
- `m`: Number of constraints
"""
function process_initial_points!(
    ws::HPRQP_workspace,
    qp::HPRQP_QP_info,
    params::HPRQP_parameters,
    scaling_info::HPRQP_scaling,
    m::Int
)
    # Process initial_x if provided
    if params.initial_x !== nothing
        # Convert to device array and scale
        WS = workspace_type(qp)
        initial_x_device = convert_to_device(WS, params.initial_x)
        scaled_x = initial_x_device .* scaling_info.col_norm ./ scaling_info.b_scale

        ws.x .= scaled_x
        ws.x_bar .= scaled_x
        ws.last_x .= scaled_x
        ws.w .= scaled_x
        ws.w_bar .= scaled_x
        ws.last_w .= scaled_x
    end

    # Process initial_y if provided (depends on lambda_max_A)
    if params.initial_y !== nothing
        # Warning: may have bug that quit with wrong result when we have initial points (<z,x> not equals to support function)
        # Convert to device array and scale
        WS = workspace_type(qp)
        ws.y .= convert_to_device(WS, params.initial_y)
        ws.y .= ws.y .* scaling_info.row_norm ./ scaling_info.c_scale
        ws.y_bar .= ws.y
        ws.last_y .= ws.y

        # Compute ATy_bar from y_bar
        if m > 0
            unified_mul!(ws.ATy_bar, ws.AT, ws.y_bar)
            ws.ATy .= ws.ATy_bar
            ws.last_ATy .= ws.ATy_bar
        end

        # Compute z_bar from projection: z_bar = (x_bar - z_raw) / sigma
        # where z_raw = x_bar + sigma * (-Qx + ATy - c)
        Qmap!(ws.x_bar, ws.Qx, qp.Q)
        tmp = .-ws.Qx .+ ws.ATy_bar .- ws.c
        z_raw = ws.x_bar .+ ws.sigma .* tmp
        ws.z_bar .= (ws.x_bar .- z_raw) ./ ws.sigma

        # Compute s for dual objective: s = proj_{[AL,AU]}(Ax - lambda_max_A * sigma * y)
        if m > 0
            unified_mul!(ws.Ax, ws.A, ws.x_bar)
            fact1 = ws.lambda_max_A * ws.sigma
            ws.s .= min.(max.(ws.Ax .- fact1 .* ws.y, ws.AL), ws.AU)
        end
    end
end

# CPU version of print_iteration_log
function print_iteration_log(iter::Int, residuals::HPRQP_residuals,
    ws::HPRQP_workspace, t_start_alg::Float64)
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

# Helper function: Perform main iteration step (update and norm computation)
# GPU version
function perform_iteration_step!(ws::HPRQP_workspace, qp::HPRQP_QP_info,
    params::HPRQP_parameters, restart_info::HPRQP_restart,
    iter::Int, check_iter::Int)
    # Main update - now handles both operator and sparse matrix Q within main_update_gpu!
    if isa(ws, HPRQP_workspace_gpu) && isa(qp, QP_info_gpu)
        main_update_gpu!(ws, qp, restart_info)
    else
        main_update_cpu!(ws, qp, restart_info)
    end
    # Compute M norm for restart decision
    if restart_info.restart_flag > 0
        restart_info.last_gap = compute_M_norm!(ws, qp)
    end

    if rem(iter + 1, check_iter) == 0
        restart_info.current_gap = compute_M_norm!(ws, qp)
    end

    restart_info.inner += 1
end

# This function is the main solver function for the HPR-QP algorithm.
# It handles GPU transfer/CPU setup, scaling, and optimization.
function solve(model::QP_info_cpu, params::HPRQP_parameters)
    setup_start = time()

    # Validate GPU parameters before attempting GPU operations
    validate_gpu_parameters!(params)
    # Setup: GPU device (only if using GPU)
    if params.use_gpu
        CUDA.device!(params.device_number)
    end

    # Setup: GPU transfer and scaling
    diag_Q, Q_is_diag = nothing, false
    qp, transfer_time = prepare_model(model, params)

    scaling_info = scaling!(qp, params)

    diag_Q, Q_is_diag = check_Q_diagonal(qp)

    # Get problem dimensions
    m, n = size(qp.A)

    # Initialize workspace and solver state
    residuals = HPRQP_residuals()
    restart_info = initialize_restart()

    # Allocate workspace
    ws = allocate_workspace(qp, params, 0.0, 0.0, scaling_info, diag_Q, Q_is_diag)

    # Prepare CUSPARSE SpMV structures (GPU-only, no-op for CPU)
    prepare_workspace_spmv!(ws, qp, params.verbose)

    setup_time = time() - setup_start
    t_start_alg = time()

    # Estimate eigenvalues using power_iteration
    ws.lambda_max_A, ws.lambda_max_Q, power_time = estimate_eigenvalues(qp, params, ws)

    # Process initial points if provided
    process_initial_points!(ws, qp, params, scaling_info, m)

    ws.spmv_mode_Q, ws.spmv_mode_A = determine_spmv_mode(qp, params, ws)

    # Initialize best_sigma with the initial sigma value
    restart_info.best_sigma = ws.sigma
    print_problem_info(qp, ws, params)
    print_solver_params(params, qp, ws.spmv_mode_Q, ws.spmv_mode_A)

    # Setup iteration tracking
    iter_4, time_4 = 0, 0.0
    iter_6, time_6 = 0, 0.0
    first_4, first_6 = true, true

    # Update Q factors for diagonal Q
    if ws.Q_is_diag
        unified_update_Q_factors!(ws.fact2, ws.fact, ws.fact1, ws.fact_M,
            ws.diag_Q, ws.sigma)
    end

    if params.verbose
        println("HPRQP SOLVER starts", params.use_gpu ? "..." : " (CPU mode)...")
        println(" iter     errRp        errRd         p_obj           d_obj          gap        sigma       time")
    end

    check_iter = params.check_iter

    number_empty_lu = sum((model.l .== -Inf) .& (model.u .== Inf))
    if (number_empty_lu > 0.8 * length(model.l))
        ws.noC = true
    else
        ws.noC = false
    end

    # Main iteration loop
    for iter = 0:params.max_iter
        # Determine if we should print at this iteration
        print_yes = should_print(iter, params, t_start_alg, params.max_iter)

        # Compute residuals if needed
        if rem(iter, check_iter) == 0 || print_yes
            residuals.is_updated = true
            compute_residuals!(ws, qp, scaling_info, residuals, params, iter)
        else
            residuals.is_updated = false
        end

        # Check termination criteria
        status = check_break(residuals, iter, t_start_alg, params)

        # Check and perform restart if needed
        check_restart(restart_info, iter, check_iter, ws.sigma)
        # Update sigma parameter (dispatches to GPU or CPU version)
        update_sigma!(params, restart_info, ws, qp, residuals)

        # Perform restart
        do_restart!(restart_info, ws, qp)

        # Print iteration log
        if (print_yes || (status != "CONTINUE")) && params.verbose
            print_iteration_log(iter, residuals, ws, t_start_alg)
        end

        # Save to HDF5 if auto_save is enabled
        if (print_yes || (status != "CONTINUE")) && params.auto_save
            try
                save_state_to_hdf5!(params.save_filename, ws, scaling_info, residuals, params, iter, t_start_alg)
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

        # Handle termination using unified function (dispatches based on workspace type)
        if status != "CONTINUE"
            return handle_termination(status, residuals, ws, scaling_info,
                iter, t_start_alg, power_time, setup_time,
                iter_4, time_4, iter_6, time_6, params.verbose)
        end

        next_iter = iter + 1
        ws.to_check = (rem(next_iter, check_iter) == 0) || (restart_info.restart_flag > 0)
        if params.print_frequency == -1
            ws.to_check = ws.to_check || (rem(next_iter, print_step(next_iter)) == 0)
        elseif params.print_frequency > 0
            ws.to_check = ws.to_check || (rem(next_iter, params.print_frequency) == 0)
        end

        # Perform main iteration step
        perform_iteration_step!(ws, qp, params, restart_info, iter, check_iter)
    end
end
