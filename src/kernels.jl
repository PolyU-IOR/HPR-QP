## Q operator mapping functions for implicit Q representations
## 
## Qmap! implementations are now in their respective operator files:
##   - QAP_operator.jl: Qmap! for QAP_Q_operator_gpu
##   - LASSO_operator.jl: Qmap! for LASSO_Q_operator_gpu
##   - sparse_matrix_operator.jl: Qmap! for CuSparseMatrixCSR
##
## Users can add custom operators by implementing Qmap! in their operator files:
##   @inline function Qmap!(x::CuVector{Float64}, Qx::CuVector{Float64}, Q::MyOperatorGPU)
##       # Your implementation: Qx .= Q * x
##   end

# ============================================================================
# Unified Kernel Wrapper Functions (CPU and GPU)
# ============================================================================
# These functions dispatch based on workspace type, following the Q operator pattern.
# GPU versions call CUDA kernels, CPU versions use optimized loops.
# ============================================================================

"""
    compute_Rd!(ws, sc)

Compute dual residual Rd = (Qx + c - A'y - z) * scale.
Unified function that dispatches based on workspace type.
"""
function compute_Rd!(ws::HPRQP_workspace, sc::HPRQP_scaling)
    # Compute A'y if constraints exist
    if ws.m > 0
        unified_mul!(ws.ATdy, ws.AT, ws.y_bar)
    end
    # Compute scaled residual (device-specific implementation via dispatch)
    _compute_Rd_impl!(ws, sc)
end

# GPU implementation using kernel
function _compute_Rd_impl!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
    end
end

# CPU implementation using loop
function _compute_Rd_impl!(ws::HPRQP_workspace_cpu, sc::Scaling_info_cpu)
    Rd = ws.Rd
    ATdy = ws.ATdy
    z_bar = ws.z_bar
    Qx = ws.Qx
    c = ws.c
    col_norm = sc.col_norm

    @simd for i in eachindex(Rd)
        @inbounds begin
            scale_fact = col_norm[i] * sc.c_scale
            Rd[i] = (Qx[i] + c[i] - ATdy[i] - z_bar[i]) * scale_fact
            Qx[i] *= scale_fact
            ATdy[i] *= scale_fact
        end
    end
end

"""
    compute_Rp!(ws, sc)

Compute primal residual Rp = proj(Ax, [AL, AU]) - Ax and scale Ax.
Unified function that dispatches based on workspace type.
"""
function compute_Rp!(ws::HPRQP_workspace, sc::HPRQP_scaling)
    # Compute Ax = A * x_bar
    unified_mul!(ws.Ax, ws.A, ws.x_bar)
    # Compute residual and scale (device-specific implementation via dispatch)
    _compute_Rp_impl!(ws, sc)
end

# GPU implementation using kernel
function _compute_Rp_impl!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    threads, blocks = gpu_launch_config(ws.m)
    if threads > 0
        @cuda threads = threads blocks = blocks compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, sc.row_norm, sc.b_scale, ws.m)
    end
end

# CPU implementation using loop
function _compute_Rp_impl!(ws::HPRQP_workspace_cpu, sc::Scaling_info_cpu)
    AL = ws.AL
    AU = ws.AU
    Ax = ws.Ax
    Rp = ws.Rp
    row_norm = sc.row_norm
    b_scale = sc.b_scale

    @simd for i in eachindex(Rp)
        @inbounds begin
            ax_i = Ax[i]
            AL_i = AL[i]
            AU_i = AU[i]
            # Project onto [AL_i, AU_i]
            ax_proj = min(max(ax_i, AL_i), AU_i)
            # Correction: ax_proj - ax_i
            corr = ax_proj - ax_i
            scale_fact = row_norm[i] * b_scale
            Rp[i] = corr * scale_fact
            Ax[i] = ax_i * scale_fact
        end
    end
end

# ============================================================================
# GPU Kernel Definitions and Launch Configurations
# ============================================================================
#
# This section contains CUDA kernels that execute on the GPU. These kernels
# MUST remain GPU-specific and should NOT be unified with CPU implementations.
#
# The kernels are organized into the following categories:
#
# 1. LASSO Update Kernels (lines ~125-265)
#    - update_zxw_LASSO_kernel_full!/partial!: LASSO-specific updates with L1 soft-thresholding
#    - update_zxw_LASSO_gpu!: Wrapper function to launch LASSO kernels
#
# 2. Unified Standard QP Kernels (lines ~270-690)
#    - unified_update_zxw1_kernel_full!/partial!: Standard QP variable updates
#    - compute_tempv_unified_kernel!: Compute temporary vectors for updates
#    - unified_update_y_kernel_full!/partial!: Dual variable y updates
#    - unified_update_w2_kernel_full!/partial!: Dual variable w updates
#    - Wrapper functions: unified_update_zxw1_gpu!, unified_update_y_gpu!, unified_update_w2_gpu!
#
# 3. LP-Specific Kernels (lines ~690-1000)
#    - unified_update_zx_gpu!: Update for problems with empty Q (LP problems)
#    - unified_update_y_noQ_gpu!: Dual updates for LP problems
#
# 4. Golden Section Search and Factor Updates (lines ~1000-1160)
#    - golden_Q_diag: GPU version of golden section search for sigma tuning
#    - update_Q_factors_kernel!/gpu!: Update scaling factors for diagonal Q
#
# 5. Scaling and Utility Kernels (lines ~1240-1570)
#    - compute_Rd_kernel!/compute_Rp_kernel!: Residual computation kernels
#    - axpby_gpu!: GPU vector operation (y = a*x + b*y)
#    - compute_row/col_max/sum kernels: CSR matrix statistics
#    - scale_* kernels: Various scaling operations on CSR matrices and vectors
#
# Key Design Principles:
# - Each kernel is optimized for GPU parallelism with minimal thread divergence
# - Kernels use CUDA.@fastmath for performance where numerical stability permits
# - Full vs Partial kernel variants control which intermediate values are stored
# - Custom vs cuSPARSE SpMV modes allow flexible performance tuning
# - Val{} type parameters enable compile-time specialization without runtime overhead
#
# These kernels are called from wrapper functions that handle:
# - Thread/block configuration via gpu_launch_config()
# - Device memory management
# - Kernel parameter setup
# - Integration with the overall solver algorithm
#
# NOTE: Do NOT attempt to unify these with CPU implementations. The CPU versions
# use fundamentally different execution models (vectorized loops vs parallel kernels).
# ============================================================================

## LASSO-specific update kernel with soft-thresholding
const DEFAULT_KERNEL_THREADS = 256

@inline function gpu_launch_config(length::Int)
    @assert length >= 0 "kernel launch length must be non-negative"
    if length == 0
        return 0, 0
    end
    threads = min(DEFAULT_KERNEL_THREADS, max(32, 32 * cld(length, 32)))
    blocks = cld(length, threads)
    return threads, blocks
end

# Full version: computes all intermediate values
CUDA.@fastmath @inline function update_zxw_LASSO_kernel_full!(
    last_w::CuDeviceVector{Float64},
    lambda::CuDeviceVector{Float64},
    dw::CuDeviceVector{Float64},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    sigma::Float64,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_i = Qw[i]
        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        w_i = w[i]
        last_w_i = last_w[i]

        tmp = -(qw_i) + atyi - c_i
        z_raw = x_i + sigma * tmp

        # Soft-thresholding for L1 regularization
        lambda_sigma = lambda[i] * sigma
        abs_z = abs(z_raw)
        shrink = max(abs_z - lambda_sigma, 0.0)
        x_bar_i = copysign(shrink, z_raw)
        x_bar[i] = x_bar_i

        x_hat_i = 2.0 * x_bar_i - x_i
        x_hat[i] = x_hat_i
        dx[i] = x_bar_i - x_i
        z_bar[i] = (x_bar_i - z_raw) / sigma

        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)
        x[i] = x_new

        w_bar_i = muladd(fact1, w_i, fact2 * x_hat_i)
        w_bar[i] = w_bar_i
        two_w_bar_minus_w = 2.0 * w_bar_i - w_i
        w_new = muladd(Halpern_fact2, two_w_bar_minus_w, Halpern_fact1 * last_w_i)
        w[i] = w_new
        dw[i] = w_bar_i - w_i
    end
    return
end

# Partial version: skips some intermediate writes
CUDA.@fastmath @inline function update_zxw_LASSO_kernel_partial!(
    last_w::CuDeviceVector{Float64},
    lambda::CuDeviceVector{Float64},
    dw::CuDeviceVector{Float64},
    dx::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    sigma::Float64,
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_i = Qw[i]
        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        w_i = w[i]
        last_w_i = last_w[i]

        tmp = -(qw_i) + atyi - c_i
        z_raw = x_i + sigma * tmp

        # Soft-thresholding for L1 regularization
        lambda_sigma = lambda[i] * sigma
        abs_z = abs(z_raw)
        shrink = max(abs_z - lambda_sigma, 0.0)
        x_bar_i = copysign(shrink, z_raw)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_hat[i] = x_hat_i

        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)
        x[i] = x_new

        w_bar_i = muladd(fact1, w_i, fact2 * x_hat_i)
        w_bar[i] = w_bar_i
        two_w_bar_minus_w = 2.0 * w_bar_i - w_i
        w_new = muladd(Halpern_fact2, two_w_bar_minus_w, Halpern_fact1 * last_w_i)
        w[i] = w_new
    end
    return
end

# Wrapper function for LASSO update
function update_zxw_LASSO_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu,
    Halpern_fact1::Float64, Halpern_fact2::Float64)
    # LASSO operator handles preprocessing internally via Q.spmv_A and Q.spmv_AT
    Qmap!(ws.w, ws.Qw, qp.Q)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    threads, blocks = gpu_launch_config(ws.n)
    threads == 0 && return

    # Choose kernel based on compute_full flag - no recompilation overhead
    if ws.to_check
        @cuda threads = threads blocks = blocks update_zxw_LASSO_kernel_full!(ws.last_w,
            ws.lambda, ws.dw, ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat,
            ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.sigma, fact1, fact2,
            Halpern_fact1, Halpern_fact2, ws.n)
    else
        @cuda threads = threads blocks = blocks update_zxw_LASSO_kernel_partial!(ws.last_w,
            ws.lambda, ws.dw, ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat,
            ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.sigma, fact1, fact2,
            Halpern_fact1, Halpern_fact2, ws.n)
    end
end

## normal z x w1 y w2 kernels (customized and unified)

# Fully unified kernel that handles all variants in a single implementation:
# 
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute Q*w inline (better for small problems, avoids cuSPARSE overhead)
#   - false: Use pre-computed Qw from cuSPARSE (better for large problems)
#
# Q Matrix Structure (Q_is_diag):
#   - true:  Q is diagonal, use element-wise fact1_vec[i] and fact2_vec[i]
#   - false: Q is general, use scalar fact1_scalar and fact2_scalar
#
# Key optimizations:
#   1. Kernel fusion: combines SpMV + update operations in one kernel
#   2. Minimal branching: conditionals are uniform across warps (no divergence)
#   3. Adaptive: can switch strategies based on problem characteristics
#
# Note: tempv computation is separated into compute_tempv_unified_kernel! for clarity
#
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_zxw1_kernel_full!(::Val{UseCustom}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_val = if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = acc
            acc
        else
            Qw[i]
        end

        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_val + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end

        dx_val = x_bar_i - x_i
        dx[i] = dx_val
        x_bar[i] = x_bar_i
        z_bar[i] = (x_bar_i - z_raw) / sigma
        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_zxw1_kernel_partial!(::Val{UseCustom}, ::Val{IsDiag},
    dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_val = if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = acc
            acc
        else
            Qw[i]
        end

        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_val + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end

        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
    end
    return
end

# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_zxw_kernel_full!(::Val{UseCustom}, ::Val{IsDiag},
    last_w::CuDeviceVector{Float64}, dw::CuDeviceVector{Float64}, dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_val = if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = acc
            acc
        else
            Qw[i]
        end

        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        last_w_i = last_w[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_val + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end

        w_new = muladd(Halpern_fact2, w_bar_i, Halpern_fact1 * last_w_i)

        dx_val = x_bar_i - x_i
        dx[i] = dx_val
        x_bar[i] = x_bar_i
        z_bar[i] = (x_bar_i - z_raw) / sigma
        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
        w[i] = w_new
        dw[i] = w_bar_i - w_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_zxw_kernel_partial!(::Val{UseCustom}, ::Val{IsDiag},
    last_w::CuDeviceVector{Float64}, dw::CuDeviceVector{Float64}, dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64,
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        qw_val = if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = acc
            acc
        else
            Qw[i]
        end

        atyi = ATy[i]
        c_i = c[i]
        x_i = x[i]
        last_x_i = last_x[i]
        last_w_i = last_w[i]
        l_i = l[i]
        u_i = u[i]
        w_i = w[i]

        tmp = -qw_val + atyi - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)

        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

        w_bar_i = if IsDiag
            fact1_i = fact1_vec[i]
            fact2_i = fact2_vec[i]
            muladd(fact1_i, w_i, fact2_i * x_hat_i)
        else
            muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
        end
        w_new = muladd(Halpern_fact2, w_bar_i, Halpern_fact1 * last_w_i)

        x[i] = x_new
        x_hat[i] = x_hat_i
        w_bar[i] = w_bar_i
        w[i] = w_new
    end
    return
end

# Unified tempv computation kernel
# Computes: tempv = x_hat + sigma * (Qw - Qw_bar)
# where Qw_bar can be computed inline (custom) or pre-computed (cuSPARSE)
CUDA.@fastmath @inline function compute_tempv_unified_kernel!(::Val{UseCustom},
    tempv::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32},
    colValQ::CuDeviceVector{Int32},
    nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    sigma::Float64,
    n::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w_bar[colValQ[k]]
            end
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            tempv[i] = x_hat_i + sigma * (qw_i - acc)
        else
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            qw_bar_i = Qw_bar[i]
            tempv[i] = x_hat_i + sigma * (qw_i - qw_bar_i)
        end
    end
    return
end

# Unified tempv computation kernel
# Computes: tempv = x_hat + sigma * (Qw - Qw_bar)
# where Qw_bar can be computed inline (custom) or pre-computed (cuSPARSE)
CUDA.@fastmath @inline function compute_tempv_zxwy_unified_kernel!(::Val{UseCustom},
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    last_Qw::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32},
    colValQ::CuDeviceVector{Int32},
    nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    sigma::Float64,
    n::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        if UseCustom
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            acc = 0.0
            @inbounds for k in startQ:stopQ
                acc += nzValQ[k] * w_bar[colValQ[k]]
            end
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            qw_bar_i = Qw_bar[i]
            tempv[i] = x_hat_i + sigma * (qw_i - acc)
            qw_hat_i = 2.0 * qw_bar_i - qw_i
            Qw[i] = muladd(Halpern_fact2, qw_hat_i, Halpern_fact1 * last_Qw[i])
        else
            x_hat_i = x_hat[i]
            qw_i = Qw[i]
            qw_bar_i = Qw_bar[i]
            tempv[i] = x_hat_i + sigma * (qw_i - qw_bar_i)
        end
    end
    return
end

# Unified update_y kernel
# Combines A*tempv computation with y update in a single kernel
#
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute A*tempv inline (better for small problems)
#   - false: Use pre-computed Ax from cuSPARSE (better for large problems)
#
# Key optimization: Fuses A*tempv SpMV with y update to reduce memory traffic
#
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_y_kernel_full!(::Val{UseCustom},
    dy::CuDeviceVector{Float64},
    rowPtrA::CuDeviceVector{Int32},
    colValA::CuDeviceVector{Int32},
    nzValA::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    m::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        Ax_val = if UseCustom
            startA = rowPtrA[i]
            stopA = rowPtrA[i+1] - 1
            acc = 0.0
            @inbounds for k in startA:stopA
                acc += nzValA[k] * tempv[colValA[k]]
            end
            Ax[i] = acc
            acc
        else
            Ax[i]
        end

        y_i = y[i]
        last_y_i = last_y[i]
        AL_i = AL[i]
        AU_i = AU[i]

        s_raw = Ax_val - fact1 * y_i
        s_proj = min(max(s_raw, AL_i), AU_i)
        # Original ternary correction: (s_raw < AL_i) ? (AL_i - s_raw) : ((s_raw > AU_i) ? (AU_i - s_raw) : 0.0)
        corr = s_proj - s_raw
        y_bar_i = fact2 * corr
        y_new = Halpern_fact1 * last_y_i + Halpern_fact2 * (2.0 * y_bar_i - y_i)

        s[i] = s_proj
        dy_i = y_bar_i - y_i
        dy[i] = dy_i
        y_bar[i] = y_bar_i
        y[i] = y_new
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_y_kernel_partial!(::Val{UseCustom},
    dy::CuDeviceVector{Float64},
    rowPtrA::CuDeviceVector{Int32},
    colValA::CuDeviceVector{Int32},
    nzValA::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    last_y::CuDeviceVector{Float64},
    s::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    fact1::Float64,
    fact2::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    m::Int) where {UseCustom}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        Ax_val = if UseCustom
            startA = rowPtrA[i]
            stopA = rowPtrA[i+1] - 1
            acc = 0.0
            @inbounds for k in startA:stopA
                acc += nzValA[k] * tempv[colValA[k]]
            end
            acc
        else
            Ax[i]
        end

        y_i = y[i]
        last_y_i = last_y[i]
        AL_i = AL[i]
        AU_i = AU[i]

        s_raw = Ax_val - fact1 * y_i
        s_proj = min(max(s_raw, AL_i), AU_i)
        corr = s_proj - s_raw
        y_bar_i = fact2 * corr
        y_new = Halpern_fact1 * last_y_i + Halpern_fact2 * (2.0 * y_bar_i - y_i)

        y_bar[i] = y_bar_i
        y[i] = y_new
    end
    return
end

# Unified update_w2 kernel
# Combines AT*y_bar computation with w2 update in a single kernel
#
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute AT*y_bar inline (better for small problems)
#   - false: Use pre-computed ATy_bar from cuSPARSE (better for large problems)
#
# Q Matrix Structure (Q_is_diag):
#   - true:  Q is diagonal, use element-wise fact_vec[i]
#   - false: Q is general, use scalar fact_scalar
#
# Key optimization: Fuses AT*y_bar SpMV with w2 update to reduce memory traffic
#
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_w2_kernel_full!(::Val{UseCustom}, ::Val{IsDiag},
    dw::CuDeviceVector{Float64},
    ATdy::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    ATy_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64},
    last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    fact_scalar::Float64,
    fact_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_bar_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y_bar[colValAT[k]]
            end
            ATy_bar[i] = acc
            acc
        else
            ATy_bar[i]
        end

        fact = if IsDiag
            fact_vec[i]
        else
            fact_scalar
        end

        w_i = w[i]
        w_bar_i = w_bar[i]
        ATy_i = ATy[i]
        last_w_i = last_w[i]
        last_ATy_i = last_ATy[i]

        w_bar_new = w_bar_i + fact * (ATy_bar_val - ATy_i)
        w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)
        ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_val - ATy_i)

        w[i] = w_new
        ATy[i] = ATy_new
        w_bar[i] = w_bar_new
        dw_i = w_bar_new - w_i
        ATdy_i = ATy_bar_val - ATy_i
        dw[i] = dw_i
        ATdy[i] = ATdy_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_w2_kernel_partial!(::Val{UseCustom}, ::Val{IsDiag},
    dw::CuDeviceVector{Float64},
    ATdy::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y_bar::CuDeviceVector{Float64},
    ATy_bar::CuDeviceVector{Float64},
    w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    last_w::CuDeviceVector{Float64},
    last_ATy::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    fact_scalar::Float64,
    fact_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom,IsDiag}
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_bar_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y_bar[colValAT[k]]
            end
            acc
        else
            ATy_bar[i]
        end

        fact = if IsDiag
            fact_vec[i]
        else
            fact_scalar
        end

        w_i = w[i]
        w_bar_i = w_bar[i]
        ATy_i = ATy[i]
        last_w_i = last_w[i]
        last_ATy_i = last_ATy[i]

        w_bar_new = w_bar_i + fact * (ATy_bar_val - ATy_i)
        w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)
        ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_val - ATy_i)

        w[i] = w_new
        ATy[i] = ATy_new
    end
    return
end

# Unified wrapper for update_zxw1 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_zxw_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu,
    Halpern_fact1::Float64, Halpern_fact2::Float64)
    # Determine whether to use custom inline SpMV for Q operations
    use_custom_spmv_Q = (ws.spmv_mode_Q == "customized")

    # Prepare scalar factors for regular Q
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    # Pre-computed diagonal scaling factors (ignored when Q is not diagonal)
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2

    # Get Q matrix structure (use dummy vectors for operators)
    # For operators, rowPtr/colVal/nzVal won't be accessed since use_custom_spmv_Q will be false
    if isa(qp.Q, CuSparseMatrixCSR)
        rowPtrQ, colValQ, nzValQ = qp.Q.rowPtr, qp.Q.colVal, qp.Q.nzVal
    else
        # Dummy vectors for operators (won't be accessed in kernel)
        rowPtrQ, colValQ, nzValQ = ws.A.rowPtr, ws.A.colVal, ws.A.nzVal
    end

    # Use Qmap! for Q*w (works for both sparse matrices and operators)
    # if !use_custom_spmv_Q
    #     # Pass ws.spmv_Q for sparse matrix Q, operators handle their own preprocessing
    #     if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
    #         Qmap!(ws.w, ws.Qw, qp.Q, ws.spmv_Q)
    #     else
    #         Qmap!(ws.w, ws.Qw, qp.Q)
    #     end
    # end

    # Step 1: Update z, x, w1
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_zxw_kernel_full!(
                Val(use_custom_spmv_Q), Val(ws.Q_is_diag),
                ws.last_w, ws.dw, ws.dx, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_zxw_kernel_partial!(
                Val(use_custom_spmv_Q), Val(ws.Q_is_diag),
                ws.last_w, ws.dw, ws.dx, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, ws.n)
        end
    end

    # Step 2: Compute tempv for subsequent use in update_y
    # Use Qmap! for Q*w_bar (works for both sparse matrices and operators)
    if !use_custom_spmv_Q
        # Pass ws.spmv_Q for sparse matrix Q, operators handle their own preprocessing
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q)
        end
    end

    if threads > 0
        @cuda threads = threads blocks = blocks compute_tempv_zxwy_unified_kernel!(
            Val(use_custom_spmv_Q),
            Halpern_fact1, Halpern_fact2,
            ws.last_Qw,
            ws.tempv, rowPtrQ, colValQ, nzValQ,
            ws.w_bar, ws.x_hat, ws.Qw, ws.Qw_bar, ws.sigma, ws.n)
    end
end

# Unified wrapper that handles all cases: regular/diagonal Q, custom/cuSPARSE SpMV
# Unified wrapper for update_zxw1 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_zxw1_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu,
    Halpern_fact1::Float64, Halpern_fact2::Float64)
    # Determine whether to use custom inline SpMV for Q operations
    use_custom_spmv_Q = (ws.spmv_mode_Q == "customized")

    # Prepare scalar factors for regular Q
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    # Pre-computed diagonal scaling factors (ignored when Q is not diagonal)
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2

    # Get Q matrix structure (use dummy vectors for operators)
    # For operators, rowPtr/colVal/nzVal won't be accessed since use_custom_spmv_Q will be false
    if isa(qp.Q, CuSparseMatrixCSR)
        rowPtrQ, colValQ, nzValQ = qp.Q.rowPtr, qp.Q.colVal, qp.Q.nzVal
    else
        # Dummy vectors for operators (won't be accessed in kernel)
        rowPtrQ, colValQ, nzValQ = ws.A.rowPtr, ws.A.colVal, ws.A.nzVal
    end

    # Use Qmap! for Q*w (works for both sparse matrices and operators)
    if !use_custom_spmv_Q
        # Pass ws.spmv_Q for sparse matrix Q, operators handle their own preprocessing
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
            Qmap!(ws.w, ws.Qw, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.w, ws.Qw, qp.Q)
        end
    end

    # Step 1: Update z, x, w1
    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_zxw1_kernel_full!(
                Val(use_custom_spmv_Q), Val(ws.Q_is_diag),
                ws.dx, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_zxw1_kernel_partial!(
                Val(use_custom_spmv_Q), Val(ws.Q_is_diag),
                ws.dx, rowPtrQ, colValQ, nzValQ,
                ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c,
                ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
                Halpern_fact1, Halpern_fact2, ws.n)
        end
    end

    # Step 2: Compute tempv for subsequent use in update_y
    # Use Qmap! for Q*w_bar (works for both sparse matrices and operators)
    if !use_custom_spmv_Q
        # Pass ws.spmv_Q for sparse matrix Q, operators handle their own preprocessing
        if isa(qp.Q, CuSparseMatrixCSR{Float64,Int32})
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q, ws.spmv_Q)
        else
            Qmap!(ws.w_bar, ws.Qw_bar, qp.Q)
        end
    end

    if threads > 0
        @cuda threads = threads blocks = blocks compute_tempv_unified_kernel!(
            Val(use_custom_spmv_Q),
            ws.tempv, rowPtrQ, colValQ, nzValQ,
            ws.w_bar, ws.x_hat, ws.Qw, ws.Qw_bar, ws.sigma, ws.n)
    end
end

# Unified wrapper for update_y that handles both custom and cuSPARSE SpMV
# Unified wrapper for update_y that handles both custom and cuSPARSE SpMV
function unified_update_y_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (ws.spmv_mode_A == "customized")

    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    # Only compute A*tempv via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        # Use preprocessed CUSPARSE if available
        if ws.spmv_A !== nothing
            desc_tempv = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.tempv)
            desc_Ax = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.Ax)
            CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha,
                ws.spmv_A.desc_A, desc_tempv, ws.spmv_A.beta, desc_Ax,
                ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
        else
            CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
    end

    if ws.m > 0
        threads_A, blocks_A = gpu_launch_config(ws.m)
        if threads_A > 0
            # Choose kernel based on to_check flag - no recompilation overhead
            if ws.to_check
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_full!(
                    Val(use_custom_spmv_A),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.tempv, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
            else
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_partial!(
                    Val(use_custom_spmv_A),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.tempv, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
            end
        end
    end
end

# Unified wrapper for update_w2 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
# Unified wrapper for update_w2 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (ws.spmv_mode_A == "customized")

    # Prepare scalar factor for regular Q
    fact_scalar = ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q)

    # Pre-computed vector factor for diagonal Q (ignored otherwise)
    fact_vec = ws.fact

    # Only compute AT*y_bar via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        # Use preprocessed CUSPARSE if available
        if ws.spmv_AT !== nothing
            desc_y_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.y_bar)
            desc_ATy_bar = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.ATy_bar)
            CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha,
                ws.spmv_AT.desc_AT, desc_y_bar, ws.spmv_AT.beta, desc_ATy_bar,
                ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
        else
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
    end

    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_w2_kernel_full!(
                Val(use_custom_spmv_A), Val(ws.Q_is_diag),
                ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y_bar, ws.ATy_bar, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy,
                fact_scalar, fact_vec, Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_w2_kernel_partial!(
                Val(use_custom_spmv_A), Val(ws.Q_is_diag),
                ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y_bar, ws.ATy_bar, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy,
                fact_scalar, fact_vec, Halpern_fact1, Halpern_fact2, ws.n)
        end
    end
end

## Unified kernels for empty Q case (Q.nzVal has length 0 - linear program)

# Unified update_zx kernel - handles both custom inline AT*y and cuSPARSE
# Full version: computes all intermediate values
CUDA.@fastmath @inline function unified_update_zx_kernel_full!(::Val{UseCustom},
    dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom}

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y[colValAT[k]]
            end
            acc
        else
            ATy[i]
        end

        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        c_i = c[i]

        tmp = ATy_val - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)
        x_hat_i = 2.0 * x_bar_i - x_i
        dx_val = x_bar_i - x_i
        z_bar_i = (x_bar_i - z_raw) / sigma
        x_new = Halpern_fact1 * last_x_i + Halpern_fact2 * x_hat_i

        dx[i] = dx_val
        x_bar[i] = x_bar_i
        z_bar[i] = z_bar_i
        x[i] = x_new
        x_hat[i] = x_hat_i
    end
    return
end

# Partial version: skips intermediate writes
CUDA.@fastmath @inline function unified_update_zx_kernel_partial!(::Val{UseCustom},
    dx::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32},
    colValAT::CuDeviceVector{Int32},
    nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    sigma::Float64,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64,
    n::Int) where {UseCustom}

    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        ATy_val = if UseCustom
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            acc = 0.0
            @inbounds for k in startAT:stopAT
                acc += nzValAT[k] * y[colValAT[k]]
            end
            acc
        else
            ATy[i]
        end

        x_i = x[i]
        last_x_i = last_x[i]
        l_i = l[i]
        u_i = u[i]
        c_i = c[i]

        tmp = ATy_val - c_i
        z_raw = x_i + sigma * tmp
        x_bar_i = min(max(z_raw, l_i), u_i)
        x_hat_i = 2.0 * x_bar_i - x_i
        x_new = Halpern_fact1 * last_x_i + Halpern_fact2 * x_hat_i

        x[i] = x_new
        x_hat[i] = x_hat_i
    end
    return
end

# Unified wrapper for update_zx
function unified_update_zx_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    use_custom_spmv_A = (ws.spmv_mode_A == "customized")

    # Only compute AT*y via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        # Use preprocessed CUSPARSE if available
        if ws.spmv_AT !== nothing
            desc_y = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.y)
            desc_ATy = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.ATy)
            CUDA.CUSPARSE.cusparseSpMV(ws.spmv_AT.handle, ws.spmv_AT.operator, ws.spmv_AT.alpha,
                ws.spmv_AT.desc_AT, desc_y, ws.spmv_AT.beta, desc_ATy,
                ws.spmv_AT.compute_type, ws.spmv_AT.alg, ws.spmv_AT.buf)
        else
            CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
    end

    threads, blocks = gpu_launch_config(ws.n)
    if threads > 0
        # Choose kernel based on to_check flag - no recompilation overhead
        if ws.to_check
            @cuda threads = threads blocks = blocks unified_update_zx_kernel_full!(
                Val(use_custom_spmv_A),
                ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                ws.c, ws.l, ws.u, ws.sigma,
                Halpern_fact1, Halpern_fact2, ws.n)
        else
            @cuda threads = threads blocks = blocks unified_update_zx_kernel_partial!(
                Val(use_custom_spmv_A),
                ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
                ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
                ws.c, ws.l, ws.u, ws.sigma,
                Halpern_fact1, Halpern_fact2, ws.n)
        end
    end
end

# Unified update_y_noQ - uses x_hat directly instead of tempv
function unified_update_y_noQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (ws.spmv_mode_A == "customized")

    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    # Only compute A*x_hat via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        # Use preprocessed CUSPARSE if available
        if ws.spmv_A !== nothing
            desc_x_hat = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.x_hat)
            desc_Ax = CUDA.CUSPARSE.CuDenseVectorDescriptor(ws.Ax)
            CUDA.CUSPARSE.cusparseSpMV(ws.spmv_A.handle, ws.spmv_A.operator, ws.spmv_A.alpha,
                ws.spmv_A.desc_A, desc_x_hat, ws.spmv_A.beta, desc_Ax,
                ws.spmv_A.compute_type, ws.spmv_A.alg, ws.spmv_A.buf)
        else
            CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
        end
    end

    if ws.m > 0
        threads_A, blocks_A = gpu_launch_config(ws.m)
        if threads_A > 0
            # Reuse unified_update_y_kernel but pass x_hat instead of tempv
            # Choose kernel based on to_check flag - no recompilation overhead
            if ws.to_check
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_full!(
                    Val(use_custom_spmv_A),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
            else
                @cuda threads = threads_A blocks = blocks_A unified_update_y_kernel_partial!(
                    Val(use_custom_spmv_A),
                    ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
                    ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
                    fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
            end
        end
    end
end


CUDA.@fastmath @inline function cust_compute_r2_kernel!(rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    sigma::Float64, x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        tempv[i] = x_hat[i] + sigma * (Qw[i] - qr1)
    end
    return
end


## kernels used to update sigma

@inline function f_dev(x, a, b, c, d)
    return a * x + b / x + c * x^2 / (1 + d * x)
end

function golden(
    a_p::Float64, b_p::Float64, c_p::Float64, d_p::Float64;
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200
)
    # golden ratio constant
    φ = (sqrt(5.0) - 1.0) / 2.0
    a = lo
    b = hi
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    f_c = f_dev(c, a_p, b_p, c_p, d_p)
    f_d = f_dev(d, a_p, b_p, c_p, d_p)

    for i in 1:maxiter
        if f_d < f_c
            a, c, f_c = c, d, f_d
            d = a + φ * (b - a)
            f_d = f_dev(d, a_p, b_p, c_p, d_p)
        else
            b, d, f_d = d, c, f_c
            c = b - φ * (b - a)
            f_c = f_dev(c, a_p, b_p, c_p, d_p)
        end
        if (b - a) < tol
            break
        end
    end

    x_sol = 0.5 * (a + b)
    return x_sol
end



# Golden-section search for minimizing 
# f(x) = a*x + b/x + x^2 * dot(c, (I + x*Q) \ d)
# GPU‑enabled golden‐section search for 
# f(x) = a*x + b/x + x^2 * dot(c, d ./ (1 + x*Q))
function golden_Q_diag(a::Float64, b::Float64, Q::CuArray{Float64}, c::CuArray{Float64}, d::CuArray{Float64}, tempv::CuArray{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using GPU operations, reusing tempv
    function f_gpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * CUDA.dot(c, tempv)
    end

    # Initialize bracket
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_gpu(x1)
    f2 = f_gpu(x2)

    # Main golden‐section loop
    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_gpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_gpu(x1)
        end
        iter += 1
    end

    return (lo + hi) / 2
end

#############################
# CUDA kernel to update all four factors in one pass
#############################
function update_Q_factors_kernel!(
    fact2::CuDeviceVector{Float64},
    fact::CuDeviceVector{Float64},
    fact1::CuDeviceVector{Float64},
    fact_M::CuDeviceVector{Float64},
    diag_Q::CuDeviceVector{Float64},
    sigma::Float64,
    s2::Float64,
    N::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        v = diag_Q[i]
        t2 = 1.0 / (1.0 + sigma * v)
        fact2[i] = t2
        fact[i] = sigma * t2
        fact1[i] = sigma * v * t2
        fact_M[i] = s2 * t2
    end
    return
end

#############################
# High-level wrapper to launch the above kernel
#############################
function update_Q_factors_gpu!(
    fact2::CuVector{Float64},
    fact::CuVector{Float64},
    fact1::CuVector{Float64},
    fact_M::CuVector{Float64},
    diag_Q::CuVector{Float64},
    sigma::Float64
)
    N = length(diag_Q)
    s2 = sigma * sigma
    threads = 256
    blocks = cld(N, threads)
    @cuda threads = threads blocks = blocks update_Q_factors_kernel!(
        fact2, fact, fact1, fact_M,
        diag_Q, sigma, s2, N
    )
    return
end

#############################
# CPU version of golden_Q_diag
#############################
function golden_Q_diag_cpu(a::Float64, b::Float64, Q::Vector{Float64}, c::Vector{Float64}, d::Vector{Float64}, tempv::Vector{Float64};
    lo::Float64=eps(Float64),
    hi::Float64=1e12,
    tol::Float64=1e-12,
    maxiter::Int=200)
    φ = (sqrt(5.0) - 1.0) / 2.0

    # Objective using CPU operations, reusing tempv
    function f_cpu(x)
        @. tempv = d / (1.0 + x * Q)
        return a * x + b / x + x^2 * dot(c, tempv)
    end

    # Initialize bracket
    x1 = hi - φ * (hi - lo)
    x2 = lo + φ * (hi - lo)
    f1 = f_cpu(x1)
    f2 = f_cpu(x2)

    # Main golden‐section loop
    iter = 0
    while abs(hi - lo) > tol * max(1.0, abs(lo)) && iter < maxiter
        if f1 > f2
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + φ * (hi - lo)
            f2 = f_cpu(x2)
        else
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - φ * (hi - lo)
            f1 = f_cpu(x1)
        end
        iter += 1
    end

    return (lo + hi) / 2
end

#############################
# CPU version of update_Q_factors
#############################
function update_Q_factors_cpu!(
    fact2::Vector{Float64},
    fact::Vector{Float64},
    fact1::Vector{Float64},
    fact_M::Vector{Float64},
    diag_Q::Vector{Float64},
    sigma::Float64
)
    N = length(diag_Q)
    s2 = sigma * sigma
    for i in 1:N
        v = diag_Q[i]
        t2 = 1.0 / (1.0 + sigma * v)
        fact2[i] = t2
        fact[i] = sigma * t2
        fact1[i] = sigma * v * t2
        fact_M[i] = s2 * t2
    end
    return
end

## kernels to compute residuals

CUDA.@fastmath @inline function compute_Rd_kernel!(ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Qx::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= n
        qx_i = Qx[i]
        c_i = c[i]
        atyi = ATy[i]
        z_i = z[i]
        scale_fact = col_norm[i] * c_scale
        rd_i = (qx_i + c_i - atyi - z_i) * scale_fact
        Rd[i] = rd_i
        Qx[i] = qx_i * scale_fact
        ATy[i] = atyi * scale_fact
    end
    return
end

function compute_Rd_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    # Call unified version
    compute_Rd!(ws, sc)
end

CUDA.@fastmath @inline function compute_Rp_kernel!(Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= m
        ax_i = Ax[i]
        AL_i = AL[i]
        AU_i = AU[i]
        ax_proj = min(max(ax_i, AL_i), AU_i)
        # Original ternary correction: (ax_i < AL_i) ? (AL_i - ax_i) : ((ax_i > AU_i) ? (AU_i - ax_i) : 0.0)
        corr = ax_proj - ax_i
        scale_fact = row_norm[i] * b_scale
        Rp[i] = corr * scale_fact
        Ax[i] = ax_i * scale_fact
    end
    return
end

function compute_Rp_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    # Call unified version
    compute_Rp!(ws, sc)
end

CUDA.@fastmath @inline function compute_err_lu_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        x_i = x[i]
        l_i = l[i]
        u_i = u[i]
        lower_violation = max(l_i - x_i, 0.0)
        upper_violation = max(x_i - u_i, 0.0)
        # Original ternary: (x_i < l_i) ? (l_i - x_i) : ((x_i > u_i) ? (x_i - u_i) : 0.0)
        corr = lower_violation + upper_violation
        dx[i] = corr * (b_scale / col_norm[i])
    end
    return
end

CUDA.@fastmath @inline function axpby_kernel!(a::Float64, x::CuDeviceVector{Float64},
    b::Float64, y::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    @inbounds if i <= n
        z[i] = muladd(a, x[i], b * y[i])
    end
    return
end

function axpby_gpu!(a::Float64, x::CuArray{Float64},
    b::Float64, y::CuArray{Float64},
    z::CuArray{Float64}, n::Int)
    threads, blocks = gpu_launch_config(n)
    if threads > 0
        @cuda threads = threads blocks = blocks axpby_kernel!(a, x, b, y, z, n)
    end
end

# GPU kernels for scaling operations

# Kernel to compute row-wise maximum of absolute values for CSR matrix
function compute_row_max_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            max_val = 0.0
            for k in start_idx:end_idx
                val = abs(nzVal[k])
                max_val = max(max_val, val)
            end
            sqrt_max = sqrt(max_val)
            mask = Float64(max_val > 0.0)
            # Original ternary: max_val > 0.0 ? sqrt(max_val) : 1.0
            row_norm[i] = muladd(mask, sqrt_max - 1.0, 1.0)
        end
    end
    return
end

# Kernel to compute column-wise maximum of absolute values for CSR matrix (operates on AT in CSR format)
function compute_col_max_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            max_val = 0.0
            for k in start_idx:end_idx
                val = abs(nzVal[k])
                max_val = max(max_val, val)
            end
            sqrt_max = sqrt(max_val)
            mask = Float64(max_val > 0.0)
            # Original ternary: max_val > 0.0 ? sqrt(max_val) : 1.0
            col_norm[i] = muladd(mask, sqrt_max - 1.0, 1.0)
        end
    end
    return
end

# Kernel to compute row-wise sum of absolute values for CSR matrix
function compute_row_sum_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            sqrt_sum = sqrt(sum_val)
            mask = Float64(sum_val > 0.0)
            # Original ternary: sum_val > 0.0 ? sqrt(sum_val) : 1.0
            row_norm[i] = muladd(mask, sqrt_sum - 1.0, 1.0)
        end
    end
    return
end

# Kernel to compute column-wise sum of absolute values for CSR matrix (operates on AT in CSR format)
function compute_col_sum_abs_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            sqrt_sum = sqrt(sum_val)
            mask = Float64(sum_val > 0.0)
            # Original ternary: sum_val > 0.0 ? sqrt(sum_val) : 1.0
            col_norm[i] = muladd(mask, sqrt_sum - 1.0, 1.0)
        end
    end
    return
end

# Kernel to compute row-wise maximum of absolute values including Q diagonal
function compute_row_max_abs_with_Q_kernel!(A_rowPtr::CuDeviceVector{Int32},
    A_nzVal::CuDeviceVector{Float64},
    Q_rowPtr::CuDeviceVector{Int32},
    Q_nzVal::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            # Compute max for Q (column-wise, using rowPtr since Q is also in CSR)
            start_idx = Q_rowPtr[i]
            end_idx = Q_rowPtr[i+1] - 1
            max_val_Q = 0.0
            for k in start_idx:end_idx
                val = abs(Q_nzVal[k])
                max_val_Q = max(max_val_Q, val)
            end

            # Compute max for A (column-wise, using AT stored in CSR)
            start_idx = A_rowPtr[i]
            end_idx = A_rowPtr[i+1] - 1
            max_val_A = 0.0
            for k in start_idx:end_idx
                val = abs(A_nzVal[k])
                max_val_A = max(max_val_A, val)
            end

            max_val = max(max_val_Q, max_val_A)
            sqrt_max = sqrt(max_val)
            mask = Float64(max_val > 0.0)
            # Original ternary: max_val > 0.0 ? sqrt(max_val) : 1.0
            row_norm[i] = muladd(mask, sqrt_max - 1.0, 1.0)
        end
    end
    return
end

# Kernel to compute column-wise sum including Q diagonal
function compute_col_sum_abs_with_Q_kernel!(A_rowPtr::CuDeviceVector{Int32},
    A_nzVal::CuDeviceVector{Float64},
    Q_rowPtr::CuDeviceVector{Int32},
    Q_nzVal::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            # Compute sum for Q (column-wise)
            start_idx = Q_rowPtr[i]
            end_idx = Q_rowPtr[i+1] - 1
            sum_val_Q = 0.0
            for k in start_idx:end_idx
                sum_val_Q += abs(Q_nzVal[k])
            end

            # Compute sum for A (column-wise, using AT)
            start_idx = A_rowPtr[i]
            end_idx = A_rowPtr[i+1] - 1
            sum_val_A = 0.0
            for k in start_idx:end_idx
                sum_val_A += abs(A_nzVal[k])
            end

            sum_val = sum_val_Q + sum_val_A
            sqrt_sum = sqrt(sum_val)
            mask = Float64(sum_val > 0.0)
            # Original ternary: sum_val > 0.0 ? sqrt(sum_val) : 1.0
            col_norm[i] = muladd(mask, sqrt_sum - 1.0, 1.0)
        end
    end
    return
end

# Kernel to check if a CSR matrix is diagonal
# For each row i, check if it has exactly one non-zero at column i
function check_diagonal_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    is_diag::CuDeviceVector{Bool},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            nnz_in_row = end_idx - start_idx + 1
            
            # Check: exactly one non-zero AND it's at column index i
            if nnz_in_row == 1
                col_idx = colVal[start_idx]
                is_diag[i] = (col_idx == i)
            elseif nnz_in_row == 0
                # Empty row is considered diagonal (zero on diagonal)
                is_diag[i] = true
            else
                # More than one non-zero: not diagonal
                is_diag[i] = false
            end
        end
    end
    return
end

# Kernel to extract diagonal elements from CSR matrix
function extract_diagonal_csr_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    diag::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            
            # Search for diagonal element in this row
            diag[i] = 0.0
            for k in start_idx:end_idx
                if colVal[k] == i
                    diag[i] = nzVal[k]
                    break
                end
            end
        end
    end
    return
end

# Kernel to scale rows of CSR matrix by 1.0 / row_scale
function scale_rows_csr_kernel!(rowPtr::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    row_scale::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            scale = 1.0 / row_scale[i]
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            for k in start_idx:end_idx
                nzVal[k] *= scale
            end
        end
    end
    return
end

# Kernel to scale columns of CSR matrix by column indices
function scale_csr_cols_kernel!(rowPtr::CuDeviceVector{Int32},
    colVal::CuDeviceVector{Int32},
    nzVal::CuDeviceVector{Float64},
    col_scale::CuDeviceVector{Float64},
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= m
        @inbounds begin
            start_idx = rowPtr[i]
            end_idx = rowPtr[i+1] - 1
            for k in start_idx:end_idx
                col_idx = colVal[k]
                nzVal[k] /= col_scale[col_idx]
            end
        end
    end
    return
end

# Kernel to scale a vector by another vector element-wise (v[i] /= scale[i])
function scale_vector_div_kernel!(v::CuDeviceVector{Float64},
    scale::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] /= scale[i]
    end
    return
end

# Kernel to scale a vector by another vector element-wise (v[i] *= scale[i])
function scale_vector_mul_kernel!(v::CuDeviceVector{Float64},
    scale::CuDeviceVector{Float64},
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] *= scale[i]
    end
    return
end

# Kernel to scale a vector by a scalar (v[i] /= scalar)
function scale_vector_scalar_div_kernel!(v::CuDeviceVector{Float64},
    scalar::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] /= scalar
    end
    return
end

# Kernel to scale a vector by a scalar (v[i] *= scalar)
function scale_vector_scalar_mul_kernel!(v::CuDeviceVector{Float64},
    scalar::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        @inbounds v[i] *= scalar
    end
    return
end

# ============================================================================
# CPU Loop-Based Update Functions
# ============================================================================
#
# This section contains CPU implementations that mirror the GPU kernel logic.
# These functions MUST remain separate from GPU kernels due to fundamentally
# different execution models.
#
# GPU Kernels vs CPU Loops:
# - GPU: Massively parallel execution across thousands of threads
# - CPU: Vectorized serial/SIMD loops optimized for cache locality
#
# The CPU functions are organized to match GPU kernel categories:
#
# 1. Standard QP Updates (lines ~1576-1770)
#    - update_zxw1_cpu!: CPU version of unified_update_zxw1_kernel
#      Mirrors GPU logic but uses @simd loops instead of parallel threads
#      Computes z, x, w updates with box projection for standard QP
#
# 2. LASSO Updates (lines ~1671-1765)
#    - update_zxw_LASSO_cpu!: CPU version of update_zxw_LASSO_kernel
#      Implements L1 soft-thresholding for LASSO problems
#      Uses @simd loops for vectorization
#
# 3. Unified LP Updates (lines ~1766-1930)
#    - unified_update_zx_cpu!: CPU version for problems with empty Q (LP)
#    - unified_update_y_noQ_cpu!: CPU dual updates for LP problems
#      These mirror the GPU LP kernels but use CPU-optimized loops
#
# 4. Standard Update Subroutines (lines ~1928-2090)
#    - update_y_cpu!: CPU version of unified_update_y_kernel
#      Dual variable y updates with projection
#    - update_w2_cpu!: CPU version of unified_update_w2_kernel
#      Dual variable w updates with AT*y_bar computation
#
# Key CPU Optimization Techniques:
# - @simd macro: Enables SIMD vectorization for compatible loops
# - @inbounds: Removes bounds checking for performance (use carefully)
# - @fastmath: Relaxes floating-point semantics for speed (use where safe)
# - Loop fusion: Combines multiple operations in single pass
# - Cache-friendly memory access patterns
#
# Relationship to GPU Kernels:
# - Algorithmic logic is identical to GPU kernels
# - Implementation differs to exploit CPU architecture:
#   * Sequential iteration vs parallel GPU threads
#   * CPU SIMD vs GPU warps
#   * Cache hierarchy vs GPU shared memory
# - Both produce bit-identical numerical results (within floating-point tolerance)
#
# Design Rationale:
# - Keeping separate implementations allows architecture-specific optimization
# - CPU code can use standard Julia broadcasting and SIMD
# - GPU code uses CUDA-specific features (shared memory, warp primitives)
# - Attempting to unify would sacrifice performance on both platforms
#
# Calling Convention:
# - These functions are called through device-agnostic wrappers in algorithm.jl
# - Dispatch on workspace type (HPRQP_workspace_cpu) selects CPU versions
# - Parameters and semantics match GPU versions for consistency
#
# NOTE: When modifying algorithmic logic, changes must be synchronized between
# GPU kernels and CPU loops to maintain numerical equivalence.
# ============================================================================

# CPU version of update_zxw for standard QP (non-LASSO)
function update_zxw1_cpu!(ws::HPRQP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Compute Qw
    Qmap!(ws.w, ws.Qw, ws.Q)

    # Determine Q type and compute factors
    Q_is_diag = ws.Q_is_diag
    # Prepare scalar factors for regular Q
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar

    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
    dx = ws.dx
    w = ws.w
    w_bar = ws.w_bar
    dw = ws.dw
    last_x = ws.last_x
    last_w = ws.last_w
    Qw = ws.Qw
    ATy = ws.ATy
    c = ws.c
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    fact1_vec = ws.fact1
    fact2_vec = ws.fact2

    if ws.to_check
        @simd for i in eachindex(x)
            @inbounds begin
                qw_val = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                l_i = l[i]
                u_i = u[i]
                w_i = w[i]

                tmp = -qw_val + atyi - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                w_bar_i = if Q_is_diag
                    muladd(fact1_vec[i], w_i, fact2_vec[i] * x_hat_i)
                else
                    muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                end

                dx[i] = x_bar_i - x_i
                x_bar[i] = x_bar_i
                z_bar[i] = (x_bar_i - z_raw) / sigma
                x[i] = x_new
                x_hat[i] = x_hat_i
                w_bar[i] = w_bar_i
            end
        end
    else
        @simd for i in eachindex(x)
            @inbounds begin
                qw_val = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                l_i = l[i]
                u_i = u[i]
                w_i = w[i]

                tmp = -qw_val + atyi - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                w_bar_i = if Q_is_diag
                    muladd(fact1_vec[i], w_i, fact2_vec[i] * x_hat_i)
                else
                    muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                end

                x[i] = x_new
                x_hat[i] = x_hat_i
                w_bar[i] = w_bar_i
            end
        end
    end
end

# CPU version of update_zxw for LASSO problems
function update_zxw_LASSO_cpu!(ws::HPRQP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Compute fact scalars for LASSO
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar
    
    # Compute Qw (for LASSO, Q is the data matrix operations)
    Qmap!(ws.w, ws.Qw, ws.Q)

    lambda = ws.lambda
    dw = ws.dw
    dx = ws.dx
    w_bar = ws.w_bar
    w = ws.w
    z_bar = ws.z_bar
    x_bar = ws.x_bar
    x_hat = ws.x_hat
    last_x = ws.last_x
    x = ws.x
    Qw = ws.Qw
    ATy = ws.ATy
    c = ws.c
    sigma = ws.sigma
    last_w = ws.last_w

    if ws.to_check
        @simd for i in eachindex(x)
            @inbounds begin
                qw_i = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                w_i = w[i]

                tmp = -qw_i + atyi - c_i
                z_raw = x_i + sigma * tmp

                # Soft-thresholding for L1 regularization
                lambda_sigma = lambda[i] * sigma
                abs_z = abs(z_raw)
                shrink = max(abs_z - lambda_sigma, 0.0)
                x_bar_i = copysign(shrink, z_raw)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                dx[i] = x_bar_i - x_i
                x_bar[i] = x_bar_i
                z_bar[i] = (x_bar_i - z_raw) / sigma
                x[i] = x_new
                x_hat[i] = x_hat_i

                w_bar_i = muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                w_bar[i] = w_bar_i
                two_w_bar_minus_w = 2.0 * w_bar_i - w_i
                w_new = muladd(Halpern_fact2, two_w_bar_minus_w, Halpern_fact1 * last_w[i])
                w[i] = w_new
                dw[i] = w_bar_i - w_i
            end
        end
    else
        @simd for i in eachindex(x)
            @inbounds begin
                qw_i = Qw[i]
                atyi = ATy[i]
                c_i = c[i]
                x_i = x[i]
                last_x_i = last_x[i]
                w_i = w[i]

                tmp = -qw_i + atyi - c_i
                z_raw = x_i + sigma * tmp

                # Soft-thresholding
                lambda_sigma = lambda[i] * sigma
                abs_z = abs(z_raw)
                shrink = max(abs_z - lambda_sigma, 0.0)
                x_bar_i = copysign(shrink, z_raw)

                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)

                x[i] = x_new
                x_hat[i] = x_hat_i

                w_bar_i = muladd(fact1_scalar, w_i, fact2_scalar * x_hat_i)
                w_bar[i] = w_bar_i
                two_w_bar_minus_w = 2.0 * w_bar_i - w_i
                w_new = muladd(Halpern_fact2, two_w_bar_minus_w, Halpern_fact1 * last_w[i])
                w[i] = w_new
            end
        end
    end
end

# ============================================================================
# CPU Functions for noQ case (empty Q matrix - linear program)
# ============================================================================

# CPU version of unified_update_zx - for noQ case (empty Q)
# Updates z and x variables without Q matrix operations
function unified_update_zx_cpu!(ws::HPRQP_workspace_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    
    # Compute AT*y
    mul!(ws.ATy, ws.AT, ws.y)
    
    x = ws.x
    x_bar = ws.x_bar
    x_hat = ws.x_hat
    z_bar = ws.z_bar
    dx = ws.dx
    last_x = ws.last_x
    ATy = ws.ATy
    c = ws.c
    l = ws.l
    u = ws.u
    sigma = ws.sigma
    
    if ws.to_check
        @simd for i in eachindex(x)
            @inbounds begin
                x_i = x[i]
                last_x_i = last_x[i]
                ATy_i = ATy[i]
                c_i = c[i]
                l_i = l[i]
                u_i = u[i]
                
                # Compute z_raw = x + sigma * (AT*y - c)
                tmp = ATy_i - c_i
                z_raw = x_i + sigma * tmp
                
                # Project onto bounds [l, u]
                x_bar_i = min(max(z_raw, l_i), u_i)
                
                # Compute x_hat = 2*x_bar - x (for Peaceman-Rachford)
                x_hat_i = 2.0 * x_bar_i - x_i
                
                # Compute z_bar (dual variable for bounds)
                z_bar_i = (x_bar_i - z_raw) / sigma
                
                # Halpern averaging: x_new = alpha*last_x + (1-alpha)*x_hat
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)
                
                # Store results
                dx[i] = x_bar_i - x_i
                x_bar[i] = x_bar_i
                z_bar[i] = z_bar_i
                x[i] = x_new
                x_hat[i] = x_hat_i
            end
        end
    else
        @simd for i in eachindex(x)
            @inbounds begin
                x_i = x[i]
                last_x_i = last_x[i]
                ATy_i = ATy[i]
                c_i = c[i]
                l_i = l[i]
                u_i = u[i]
                
                tmp = ATy_i - c_i
                z_raw = x_i + sigma * tmp
                x_bar_i = min(max(z_raw, l_i), u_i)
                x_hat_i = 2.0 * x_bar_i - x_i
                x_new = muladd(Halpern_fact2, x_hat_i, Halpern_fact1 * last_x_i)
                
                x[i] = x_new
                x_hat[i] = x_hat_i
            end
        end
    end
end

# CPU version of unified_update_y_noQ - for noQ case (empty Q)
# Updates y variable using x_hat directly (no tempv needed since Q is empty)
function unified_update_y_noQ_cpu!(ws::HPRQP_workspace_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    
    if ws.m == 0
        return
    end
    
    # Compute A*x_hat (no Q corrections needed for noQ case)
    mul!(ws.Ax, ws.A, ws.x_hat)
    
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    
    y = ws.y
    y_bar = ws.y_bar
    dy = ws.dy
    last_y = ws.last_y
    s = ws.s
    Ax = ws.Ax
    AL = ws.AL
    AU = ws.AU
    
    if ws.to_check
        @simd for i in eachindex(y)
            @inbounds begin
                y_i = y[i]
                last_y_i = last_y[i]
                Ax_i = Ax[i]
                AL_i = AL[i]
                AU_i = AU[i]
                
                # Compute s_raw = Ax - fact1*y
                s_raw = Ax_i - fact1 * y_i
                
                # Project onto constraint bounds [AL, AU]
                s_proj = min(max(s_raw, AL_i), AU_i)
                
                # Compute correction
                corr = s_proj - s_raw
                
                # Compute y_bar
                y_bar_i = fact2 * corr
                
                # Compute y_hat = 2*y_bar - y
                y_hat_i = 2.0 * y_bar_i - y_i
                
                # Halpern averaging: y_new = alpha*last_y + (1-alpha)*y_hat
                y_new = muladd(Halpern_fact2, y_hat_i, Halpern_fact1 * last_y_i)
                
                # Store results
                s[i] = s_proj
                dy[i] = y_bar_i - y_i
                y_bar[i] = y_bar_i
                y[i] = y_new
            end
        end
    else
        @simd for i in eachindex(y)
            @inbounds begin
                y_i = y[i]
                last_y_i = last_y[i]
                Ax_i = Ax[i]
                AL_i = AL[i]
                AU_i = AU[i]
                
                s_raw = Ax_i - fact1 * y_i
                s_proj = min(max(s_raw, AL_i), AU_i)
                corr = s_proj - s_raw
                y_bar_i = fact2 * corr
                y_hat_i = 2.0 * y_bar_i - y_i
                y_new = muladd(Halpern_fact2, y_hat_i, Halpern_fact1 * last_y_i)
                y[i] = y_new
            end
        end
    end
end

# ============================================================================
# CPU Update Functions with Q Matrix (Standard QP Updates)
# ============================================================================
#
# These functions handle updates for problems with non-empty Q matrices.
# They mirror the GPU unified_update_y and unified_update_w2 kernels.
#
# - update_y_cpu!: Computes dual variable y updates with A*tempv computation
#   where tempv = x_hat + sigma*(Qw - Qw_bar). Mirrors unified_update_y_kernel.
#
# - update_w2_cpu!: Computes dual variable w updates with AT*y_bar computation.
#   Mirrors unified_update_w2_kernel.
#
# Both functions use:
# - @simd loops for vectorization
# - Broadcasting for vector operations (.= syntax)
# - mul! for efficient matrix-vector products
# - Conditional execution based on ws.to_check flag
# ============================================================================

# CPU version of update_y with Q matrix
# Mirrors unified_update_y_kernel - computes y updates for standard QP
function update_y_cpu!(ws::HPRQP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    if ws.m == 0
        return
    end

    # Compute Qw_hat
    Qmap!(ws.w_bar, ws.Qw_bar, ws.Q)

    # Compute A * (x_hat + sigma * (Qw - Qw_bar))
    ws.tempv .= ws.x_hat .+ ws.sigma .* (ws.Qw - ws.Qw_bar)
    mul!(ws.Ax, ws.A, ws.tempv)

    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1

    y = ws.y
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    AL = ws.AL
    AU = ws.AU
    Ax = ws.Ax
    last_y = ws.last_y
    dy = ws.dy
    s = ws.s

    if ws.to_check
        @simd for i in eachindex(y)
            @inbounds begin
                yi = y[i]
                s_raw = Ax[i] - fact1 * yi
                s_proj = min(max(s_raw, AL[i]), AU[i])
                corr = s_proj - s_raw
                yb = fact2 * corr
                yh = 2.0 * yb - yi
                y_new = muladd(Halpern_fact2, yh, Halpern_fact1 * last_y[i])

                s[i] = s_proj
                dy[i] = yb - yi
                y_bar[i] = yb
                y[i] = y_new
            end
        end
    else
        @simd for i in eachindex(y)
            @inbounds begin
                yi = y[i]
                s_raw = Ax[i] - fact1 * yi
                s_proj = min(max(s_raw, AL[i]), AU[i])
                corr = s_proj - s_raw
                yb = fact2 * corr
                yh = 2.0 * yb - yi
                y_new = muladd(Halpern_fact2, yh, Halpern_fact1 * last_y[i])

                y_bar[i] = yb
                y[i] = y_new
            end
        end
    end
end

# CPU version of update_w2 - completes the w update using y_bar
# Mirrors unified_update_w2_kernel - computes AT*y_bar and updates w
function update_w2_cpu!(ws::HPRQP_workspace_cpu,
    qp::QP_info_cpu,
    Halpern_fact1::Float64,
    Halpern_fact2::Float64)
    # Compute ATy_bar
    mul!(ws.ATy_bar, ws.AT, ws.y_bar)

    # Determine Q type
    Q_is_diag = ws.Q_is_diag
    
    # Determine the fact scalar for w update
    fact_scalar = ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q)

    w = ws.w
    w_bar = ws.w_bar
    dw = ws.dw
    ATy = ws.ATy
    ATy_bar = ws.ATy_bar
    ATdy = ws.ATdy
    last_w = ws.last_w
    last_ATy = ws.last_ATy
    fact_vec = ws.fact

    if ws.to_check
        @simd for i in eachindex(w)
            @inbounds begin
                w_i = w[i]
                w_bar_i = w_bar[i]
                ATy_i = ATy[i]
                ATy_bar_i = ATy_bar[i]
                last_w_i = last_w[i]
                last_ATy_i = last_ATy[i]

                fact = if Q_is_diag
                    fact_vec[i]
                else
                    fact_scalar
                end

                # Second part of w_bar update: add the AT*y_bar correction
                w_bar_new = w_bar_i + fact * (ATy_bar_i - ATy_i)

                # Complete w update with Halpern averaging
                w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)

                # Complete ATy update with Halpern averaging
                ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_i - ATy_i)

                w[i] = w_new
                ATy[i] = ATy_new
                w_bar[i] = w_bar_new
                dw[i] = w_bar_new - w_i
                ATdy[i] = ATy_bar_i - ATy_i
            end
        end
    else
        @simd for i in eachindex(w)
            @inbounds begin
                w_i = w[i]
                w_bar_i = w_bar[i]
                ATy_i = ATy[i]
                ATy_bar_i = ATy_bar[i]
                last_w_i = last_w[i]
                last_ATy_i = last_ATy[i]

                fact = if Q_is_diag
                    fact_vec[i]
                else
                    fact_scalar
                end

                # Second part of w_bar update: add the AT*y_bar correction
                w_bar_new = w_bar_i + fact * (ATy_bar_i - ATy_i)

                # Complete w update with Halpern averaging
                w_new = Halpern_fact1 * last_w_i + Halpern_fact2 * (2.0 * w_bar_new - w_i)

                # Complete ATy update with Halpern averaging
                ATy_new = Halpern_fact1 * last_ATy_i + Halpern_fact2 * (2.0 * ATy_bar_i - ATy_i)

                w[i] = w_new
                ATy[i] = ATy_new
            end
        end
    end
end

# CPU version of compute_Rd
function compute_Rd_cpu!(ws::HPRQP_workspace_cpu, sc::Scaling_info_cpu)
    # Call unified version
    compute_Rd!(ws, sc)
end

# CPU version of compute_Rp
function compute_Rp_cpu!(ws::HPRQP_workspace_cpu, sc::Scaling_info_cpu)
    # Call unified version
    compute_Rp!(ws, sc)
end