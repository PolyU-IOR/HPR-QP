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

## LASSO-specific update kernel with soft-thresholding
CUDA.@fastmath @inline function update_zxw_LASSO_kernel!(lambda::CuDeviceVector{Float64},
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
        tmp = -Qw[i] + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        # Soft-thresholding for L1 regularization
        lambda_sigma = lambda[i] * sigma
        x_bar[i] = (z_bar[i] < -lambda_sigma) ? (z_bar[i] + lambda_sigma) : ((z_bar[i] > lambda_sigma) ? (z_bar[i] - lambda_sigma) : 0.0)
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        tempv[i] = x_hat[i] + sigma * Qw[i]

        w_bar[i] = fact1 * w[i] + fact2 * x_hat[i]
        w[i] = Halpern_fact1 * w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        dw[i] = w_bar[i] - w[i]
    end
    return
end

# Wrapper function for LASSO update
function update_zxw_LASSO_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    Qmap!(ws.w, ws.Qw, qp.Q)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_zxw_LASSO_kernel!(ws.lambda, ws.dw, ws.dx, ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

## normal z x w1 y w2 kernels (customized and unified)

# Fully unified kernel that handles all variants in a single implementation:
# 
# SpMV Strategy (use_custom_spmv):
#   - true:  Compute Q*w inline (better for small problems, avoids cuSPARSE overhead)
#   - false: Use pre-computed Qw from cuSPARSE (better for large problems)
#
# Q Matrix Structure (is_diag_Q):
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
CUDA.@fastmath @inline function unified_update_zxw1_kernel!(dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    z_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    last_x::CuDeviceVector{Float64}, x::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64}, ATy::CuDeviceVector{Float64}, c::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, l::CuDeviceVector{Float64}, u::CuDeviceVector{Float64},
    sigma::Float64, fact1_scalar::Float64, fact2_scalar::Float64, 
    fact1_vec::CuDeviceVector{Float64}, fact2_vec::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64, 
    use_custom_spmv::Bool, is_diag_Q::Bool, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        # Conditionally compute SpMV inline or use pre-computed value
        qw_val = if use_custom_spmv
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            qr1 = 0.0
            @inbounds for k in startQ:stopQ
                qr1 += nzValQ[k] * w[colValQ[k]]
            end
            Qw[i] = qr1
            qr1
        else
            Qw[i]  # Already computed by cuSPARSE
        end
        
        tmp = -qw_val + ATy[i] - c[i]
        z_bar[i] = x[i] + sigma * tmp

        x_bar[i] = z_bar[i] < (l[i]) ? l[i] : (z_bar[i] > (u[i]) ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma

        dx[i] = x_bar[i] - x[i]

        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]

        # w_bar computation: use vector or scalar fact1/fact2 depending on Q type
        if is_diag_Q
            w_bar[i] = fact1_vec[i] * w[i] + fact2_vec[i] * x_hat[i]
        else
            w_bar[i] = fact1_scalar * w[i] + fact2_scalar * x_hat[i]
        end
    end
    return
end

# Unified tempv computation kernel
# Computes: tempv = x_hat + sigma * (Qw - Qw_bar)
# where Qw_bar can be computed inline (custom) or pre-computed (cuSPARSE)
CUDA.@fastmath @inline function compute_tempv_unified_kernel!(
    tempv::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32},
    colValQ::CuDeviceVector{Int32},
    nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    sigma::Float64,
    use_custom_spmv::Bool,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        if use_custom_spmv
            # Compute Qw_bar inline
            startQ = rowPtrQ[i]
            stopQ = rowPtrQ[i+1] - 1
            qr1 = 0.0
            @inbounds for k in startQ:stopQ
                qr1 += nzValQ[k] * w_bar[colValQ[k]]
            end
            tempv[i] = x_hat[i] + sigma * (Qw[i] - qr1)
        else
            # Use pre-computed Qw_bar (from cuSPARSE)
            tempv[i] = x_hat[i] + sigma * (Qw[i] - Qw_bar[i])
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
CUDA.@fastmath @inline function unified_update_y_kernel!(
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
    use_custom_spmv::Bool,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= m
        # Conditionally compute A*tempv inline or use pre-computed value
        Ax_val = if use_custom_spmv
            startA = rowPtrA[i]
            stopA = rowPtrA[i+1] - 1
            Ai = 0.0
            @inbounds for k in startA:stopA
                Ai += nzValA[k] * tempv[colValA[k]]
            end
            Ax[i] = Ai
            Ai
        else
            Ax[i]  # Already computed by cuSPARSE
        end
        
        s[i] = Ax_val - fact1 * y[i]
        y_bar[i] = s[i] < (AL[i]) ? (AL[i] - s[i]) : (s[i] > (AU[i]) ? (AU[i] - s[i]) : 0.0)
        s[i] = s[i] + y_bar[i]
        y_bar[i] = fact2 * y_bar[i]
        dy[i] = y_bar[i] - y[i]
        y[i] = Halpern_fact1 * last_y[i] + Halpern_fact2 * (2 * y_bar[i] - y[i])
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
# Q Matrix Structure (is_diag_Q):
#   - true:  Q is diagonal, use element-wise fact_vec[i]
#   - false: Q is general, use scalar fact_scalar
#
# Key optimization: Fuses AT*y_bar SpMV with w2 update to reduce memory traffic
#
CUDA.@fastmath @inline function unified_update_w2_kernel!(
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
    use_custom_spmv::Bool,
    is_diag_Q::Bool,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        # Conditionally compute AT*y_bar inline or use pre-computed value
        ATy_bar_val = if use_custom_spmv
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            sAT = 0.0
            @inbounds for k in startAT:stopAT
                sAT += nzValAT[k] * y_bar[colValAT[k]]
            end
            ATy_bar[i] = sAT
            sAT
        else
            ATy_bar[i]  # Already computed by cuSPARSE
        end
        
        # Use vector or scalar fact depending on Q type
        fact = is_diag_Q ? fact_vec[i] : fact_scalar
        
        w_bar[i] = w_bar[i] + fact * (ATy_bar_val - ATy[i])
        dw[i] = w_bar[i] - w[i]
        ATdy[i] = ATy_bar_val - ATy[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
        ATy[i] = Halpern_fact1 * last_ATy[i] + Halpern_fact2 * (2 * ATy_bar_val - ATy[i])
    end
    return
end

# Unified wrapper that handles all cases: regular/diagonal Q, custom/cuSPARSE SpMV
# Unified wrapper for update_zxw1 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_zxw1_gpu!(ws::HPRQP_workspace_gpu, qp::QP_info_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64; 
                                   spmv_mode_Q::String="CUSPARSE", spmv_mode_A::String="CUSPARSE", is_diag_Q::Bool=false)
    # Determine whether to use custom inline SpMV for Q operations
    use_custom_spmv_Q = (spmv_mode_Q == "customized")
    
    # Prepare scalar factors for regular Q
    fact2_scalar = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1_scalar = 1.0 - fact2_scalar
    
    # For diagonal Q, use pre-computed vector factors; for regular Q, pass dummy vectors
    fact1_vec = is_diag_Q ? ws.fact1 : ws.dx  # dummy if not diagonal
    fact2_vec = is_diag_Q ? ws.fact2 : ws.dx  # dummy if not diagonal
    
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
        Qmap!(ws.w, ws.Qw, qp.Q)
    end
    
    # Step 1: Update z, x, w1
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) unified_update_zxw1_kernel!(
        ws.dx, rowPtrQ, colValQ, nzValQ,
        ws.w_bar, ws.w, ws.z_bar, ws.x_bar, ws.x_hat, ws.last_x, ws.x, ws.Qw, ws.ATy, ws.c, 
        ws.tempv, ws.l, ws.u, ws.sigma, fact1_scalar, fact2_scalar, fact1_vec, fact2_vec,
        Halpern_fact1, Halpern_fact2, use_custom_spmv_Q, is_diag_Q, ws.n)
    
    # Step 2: Compute tempv for subsequent use in update_y
    # Use Qmap! for Q*w_bar (works for both sparse matrices and operators)
    if !use_custom_spmv_Q
        Qmap!(ws.w_bar, ws.Qw_bar, qp.Q)
    end
    
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_tempv_unified_kernel!(
        ws.tempv, rowPtrQ, colValQ, nzValQ,
        ws.w_bar, ws.x_hat, ws.Qw, ws.Qw_bar, ws.sigma, use_custom_spmv_Q, ws.n)
end

# Unified wrapper for update_y that handles both custom and cuSPARSE SpMV
# Unified wrapper for update_y that handles both custom and cuSPARSE SpMV
function unified_update_y_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64;
                                spmv_mode_Q::String="CUSPARSE", spmv_mode_A::String="CUSPARSE")
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (spmv_mode_A == "customized")
    
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    
    # Only compute A*tempv via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.tempv, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) unified_update_y_kernel!(
            ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
            ws.tempv, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
            fact1, fact2, Halpern_fact1, Halpern_fact2, use_custom_spmv_A, ws.m)
    end
end

# Unified wrapper for update_w2 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
# Unified wrapper for update_w2 that handles both custom and cuSPARSE SpMV, and regular/diagonal Q
function unified_update_w2_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64;
                                 spmv_mode_Q::String="CUSPARSE", spmv_mode_A::String="CUSPARSE", is_diag_Q::Bool=false)
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (spmv_mode_A == "customized")
    
    # Prepare scalar factor for regular Q
    fact_scalar = ws.sigma / (1.0 + ws.sigma * ws.lambda_max_Q)
    
    # For diagonal Q, use pre-computed vector factor; for regular Q, pass dummy vector
    fact_vec = is_diag_Q ? ws.fact : ws.dx  # dummy if not diagonal
    
    # Only compute AT*y_bar via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATy_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) unified_update_w2_kernel!(
        ws.dw, ws.ATdy, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
        ws.y_bar, ws.ATy_bar, ws.w, ws.w_bar, ws.last_w, ws.last_ATy, ws.ATy,
        fact_scalar, fact_vec, Halpern_fact1, Halpern_fact2,
        use_custom_spmv_A, is_diag_Q, ws.n)
end

## Unified kernels for empty Q case (Q.nzVal has length 0 - linear program)

# Unified update_zx kernel - handles both custom inline AT*y and cuSPARSE
CUDA.@fastmath @inline function unified_update_zx_kernel!(
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
    use_custom_spmv::Bool,
    n::Int)
    
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    @inbounds if i <= n
        # Compute AT*y either inline or use pre-computed value
        ATy_val = if use_custom_spmv
            # Custom inline computation of AT*y
            startAT = rowPtrAT[i]
            stopAT = rowPtrAT[i+1] - 1
            sAT = 0.0
            for k in startAT:stopAT
                sAT += nzValAT[k] * y[colValAT[k]]
            end
            sAT
        else
            # Use pre-computed ATy from cuSPARSE
            ATy[i]
        end
        
        # Update z and x (no w since Q=0)
        tmp = ATy_val - c[i]
        z_bar[i] = x[i] + sigma * tmp
        
        # Project onto [l, u]
        x_bar[i] = z_bar[i] < l[i] ? l[i] : (z_bar[i] > u[i] ? u[i] : z_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        z_bar[i] = (x_bar[i] - z_bar[i]) / sigma
        
        dx[i] = x_bar[i] - x[i]
        
        # Halpern averaging
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
    end
    return
end

# Unified wrapper for update_zx
function unified_update_zx_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64;
                                 spmv_mode_A::String="CUSPARSE")
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (spmv_mode_A == "customized")
    
    # Only compute AT*y via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) unified_update_zx_kernel!(
        ws.dx, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal,
        ws.y, ws.ATy, ws.z_bar, ws.x_bar, ws.x_hat, ws.x, ws.last_x,
        ws.c, ws.l, ws.u, ws.sigma,
        Halpern_fact1, Halpern_fact2, use_custom_spmv_A, ws.n)
end

# Unified update_y_noQ - uses x_hat directly instead of tempv
function unified_update_y_noQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64;
                                    spmv_mode_A::String="CUSPARSE")
    # Determine whether to use custom inline SpMV for A operations
    use_custom_spmv_A = (spmv_mode_A == "customized")
    
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    
    # Only compute A*x_hat via cuSPARSE if not using custom SpMV for A
    if !use_custom_spmv_A
        CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    
    if ws.m > 0
        # Reuse unified_update_y_kernel but pass x_hat instead of tempv
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) unified_update_y_kernel!(
            ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal,
            ws.x_hat, ws.Ax, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU,
            fact1, fact2, Halpern_fact1, Halpern_fact2, use_custom_spmv_A, ws.m)
    end
end


function cust_compute_r2_kernel!(rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    sigma::Float64, x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
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


## w x y normal update kernels, when without C

CUDA.@fastmath @inline function update_w_noC_kernel!(dw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::Float64, fact2::Float64,
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1 * w[i] + fact2 * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function update_w_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w_noC_kernel!(ws.dw, ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function update_x_noC_kernel!(dx::CuDeviceVector{Float64},
    x_bar::CuDeviceVector{Float64}, x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64}, last_Qw::CuDeviceVector{Float64},
    sigma::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        x_bar[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
        x_hat[i] = 2 * x_bar[i] - x[i]
        dx[i] = x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])
    end
    return
end

function update_x_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.Q, ws.w_bar, 0, ws.Qw_bar, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_x_noC_kernel!(ws.dx, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end

function update_y_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_hat, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) update_y_kernel!(ws.dy, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, ws.Ax, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

## w x y normal update kernels, when without C (customized)

CUDA.@fastmath @inline function cust_update_w_noC_kernel!(dw::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::Float64, fact2::Float64,
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = sAT

        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1 * w[i] + fact2 * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function cust_update_w_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact2 = 1.0 / (1.0 + ws.sigma * ws.lambda_max_Q)
    fact1 = 1.0 - fact2
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w_noC_kernel!(ws.dw, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

CUDA.@fastmath @inline function cust_update_x_noC_kernel!(dx::CuDeviceVector{Float64},
    rowPtrQ::CuDeviceVector{Int32}, colValQ::CuDeviceVector{Int32}, nzValQ::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, x_bar::CuDeviceVector{Float64},
    x_hat::CuDeviceVector{Float64},
    tempv::CuDeviceVector{Float64}, last_x::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    Qw_bar::CuDeviceVector{Float64},
    last_Qw::CuDeviceVector{Float64},
    sigma::Float64, Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startQ = rowPtrQ[i]
        stopQ = rowPtrQ[i+1] - 1
        qr1 = 0.0
        @inbounds for k in startQ:stopQ
            qr1 += nzValQ[k] * w_bar[colValQ[k]]
        end
        Qw_bar[i] = qr1

        x_bar[i] = tempv[i] + sigma * (Qw[i] - Qw_bar[i])
        dx[i] = x_bar[i] - x[i]
        x_hat[i] = 2 * x_bar[i] - x[i]
        x[i] = Halpern_fact1 * last_x[i] + Halpern_fact2 * x_hat[i]
        Qw[i] = Halpern_fact1 * last_Qw[i] + Halpern_fact2 * (2 * Qw_bar[i] - Qw[i])
    end
    return
end

function cust_update_x_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_x_noC_kernel!(ws.dx, ws.Q.rowPtr, ws.Q.colVal, ws.Q.nzVal, ws.w_bar, ws.x_bar, ws.x_hat, ws.tempv, ws.last_x, ws.x, ws.Qw, ws.Qw_bar, ws.last_Qw, ws.sigma, Halpern_fact1, Halpern_fact2, ws.n)
end

function cust_update_y_noC_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    fact1 = ws.lambda_max_A * ws.sigma
    fact2 = 1.0 / fact1
    if ws.m > 0
        @cuda threads = 256 blocks = ceil(Int, ws.m / 256) cust_update_y_kernel!(ws.dy, ws.A.rowPtr, ws.A.colVal, ws.A.nzVal, ws.x_hat, ws.y_bar, ws.y, ws.last_y, ws.s, ws.AL, ws.AU, fact1, fact2, Halpern_fact1, Halpern_fact2, ws.m)
    end
end

# Update w, x, y kernels for no C case with diagonal Q

CUDA.@fastmath @inline function update_w_noC_diagQ_kernel!(dw::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1[i] * w[i] + fact2[i] * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function update_w_noC_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y, 0, ws.ATy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) update_w_noC_diagQ_kernel!(ws.dw, ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
end

# Update w, x, y kernels for no C case with diagonal Q (customized)

CUDA.@fastmath @inline function cust_update_w_noC_diagQ_kernel!(dw::CuDeviceVector{Float64},
    rowPtrAT::CuDeviceVector{Int32}, colValAT::CuDeviceVector{Int32}, nzValAT::CuDeviceVector{Float64},
    y::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64}, w::CuDeviceVector{Float64},
    w_bar::CuDeviceVector{Float64}, last_w::CuDeviceVector{Float64},
    ATy::CuDeviceVector{Float64}, Qw::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64}, tempv::CuDeviceVector{Float64},
    sigma::Float64, fact1::CuDeviceVector{Float64}, fact2::CuDeviceVector{Float64},
    Halpern_fact1::Float64, Halpern_fact2::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        startAT = rowPtrAT[i]
        stopAT = rowPtrAT[i+1] - 1
        sAT = 0.0
        @inbounds for k in startAT:stopAT
            sAT += nzValAT[k] * y[colValAT[k]]
        end
        ATy[i] = sAT

        tempv[i] = x[i] + sigma * (ATy[i] - Qw[i] - c[i])
        w_bar[i] = fact1[i] * w[i] + fact2[i] * tempv[i]
        dw[i] = w_bar[i] - w[i]
        w[i] = Halpern_fact1 * last_w[i] + Halpern_fact2 * (2 * w_bar[i] - w[i])
    end
    return
end

function cust_update_w_noC_diagQ_gpu!(ws::HPRQP_workspace_gpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) cust_update_w_noC_diagQ_kernel!(ws.dw, ws.AT.rowPtr, ws.AT.colVal, ws.AT.nzVal, ws.y,
        ws.x, ws.w, ws.w_bar, ws.last_w, ws.ATy, ws.Qw, ws.c, ws.tempv, ws.sigma, ws.fact1, ws.fact2, Halpern_fact1, Halpern_fact2, ws.n)
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

## kernels to compute residuals

function compute_Rd_kernel!(ATy::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Qx::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qx[i] + c[i] - ATy[i] - z[i]
        scale_fact = col_norm[i] * c_scale
        Rd[i] *= scale_fact
        Qx[i] *= scale_fact
        ATy[i] *= scale_fact
    end
    return
end

function compute_Rd_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    if ws.m > 0
        CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    end
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_kernel!(ws.ATdy, ws.z_bar, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
end

function compute_Rd_noC_kernel!(ATy::CuDeviceVector{Float64},
    c::CuDeviceVector{Float64},
    Qx::CuDeviceVector{Float64},
    Rd::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    c_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        Rd[i] = Qx[i] + c[i] - ATy[i]
        scale_fact = col_norm[i] * c_scale
        Rd[i] *= scale_fact
        Qx[i] *= scale_fact
        ATy[i] *= scale_fact
    end
    return
end

function compute_Rd_noC_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.AT, ws.y_bar, 0, ws.ATdy, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.n / 256) compute_Rd_noC_kernel!(ws.ATdy, ws.c, ws.Qx, ws.Rd, sc.col_norm, sc.c_scale, ws.n)
end

function compute_Rp_kernel!(Rp::CuDeviceVector{Float64},
    AL::CuDeviceVector{Float64},
    AU::CuDeviceVector{Float64},
    Ax::CuDeviceVector{Float64},
    row_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    m::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= m
        Rp[i] = (Ax[i] < AL[i]) ? (AL[i] - Ax[i]) : (Ax[i] > AU[i] ? (AU[i] - Ax[i]) : 0.0)
        scale_fact = row_norm[i] * b_scale
        Rp[i] *= scale_fact
        Ax[i] *= scale_fact
    end
    return
end

function compute_Rp_gpu!(ws::HPRQP_workspace_gpu, sc::Scaling_info_gpu)
    CUDA.CUSPARSE.mv!('N', 1, ws.A, ws.x_bar, 0, ws.Ax, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    @cuda threads = 256 blocks = ceil(Int, ws.m / 256) compute_Rp_kernel!(ws.Rp, ws.AL, ws.AU, ws.Ax, sc.row_norm, sc.b_scale, ws.m)
end

function compute_err_lu_kernel!(dx::CuDeviceVector{Float64},
    x::CuDeviceVector{Float64},
    l::CuDeviceVector{Float64},
    u::CuDeviceVector{Float64},
    col_norm::CuDeviceVector{Float64},
    b_scale::Float64,
    n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if i <= n
        dx[i] = (x[i] < l[i]) ? (l[i] - x[i]) : ((x[i] > u[i]) ? (x[i] - u[i]) : 0.0)
        dx[i] *= b_scale / col_norm[i]
    end
    return
end

function axpby_kernel!(a::Float64, x::CuDeviceVector{Float64},
    b::Float64, y::CuDeviceVector{Float64},
    z::CuDeviceVector{Float64}, n::Int)
    i = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if i <= n
        @inbounds z[i] = a * x[i] + b * y[i]
    end
    return
end

function axpby_gpu!(a::Float64, x::CuArray{Float64},
    b::Float64, y::CuArray{Float64},
    z::CuArray{Float64}, n::Int)
    @cuda threads = 256 blocks = ceil(Int, n / 256) axpby_kernel!(a, x, b, y, z, n)
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
            end_idx = rowPtr[i + 1] - 1
            max_val = 0.0
            for k in start_idx:end_idx
                val = abs(nzVal[k])
                max_val = max(max_val, val)
            end
            row_norm[i] = max_val > 0.0 ? sqrt(max_val) : 1.0
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
            end_idx = rowPtr[i + 1] - 1
            max_val = 0.0
            for k in start_idx:end_idx
                val = abs(nzVal[k])
                max_val = max(max_val, val)
            end
            col_norm[i] = max_val > 0.0 ? sqrt(max_val) : 1.0
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
            end_idx = rowPtr[i + 1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            row_norm[i] = sum_val > 0.0 ? sqrt(sum_val) : 1.0
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
            end_idx = rowPtr[i + 1] - 1
            sum_val = 0.0
            for k in start_idx:end_idx
                sum_val += abs(nzVal[k])
            end
            col_norm[i] = sum_val > 0.0 ? sqrt(sum_val) : 1.0
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
            end_idx = Q_rowPtr[i + 1] - 1
            max_val_Q = 0.0
            for k in start_idx:end_idx
                val = abs(Q_nzVal[k])
                max_val_Q = max(max_val_Q, val)
            end
            
            # Compute max for A (column-wise, using AT stored in CSR)
            start_idx = A_rowPtr[i]
            end_idx = A_rowPtr[i + 1] - 1
            max_val_A = 0.0
            for k in start_idx:end_idx
                val = abs(A_nzVal[k])
                max_val_A = max(max_val_A, val)
            end
            
            max_val = max(max_val_Q, max_val_A)
            row_norm[i] = max_val > 0.0 ? sqrt(max_val) : 1.0
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
            end_idx = Q_rowPtr[i + 1] - 1
            sum_val_Q = 0.0
            for k in start_idx:end_idx
                sum_val_Q += abs(Q_nzVal[k])
            end
            
            # Compute sum for A (column-wise, using AT)
            start_idx = A_rowPtr[i]
            end_idx = A_rowPtr[i + 1] - 1
            sum_val_A = 0.0
            for k in start_idx:end_idx
                sum_val_A += abs(A_nzVal[k])
            end
            
            sum_val = sum_val_Q + sum_val_A
            col_norm[i] = sum_val > 0.0 ? sqrt(sum_val) : 1.0
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
            end_idx = rowPtr[i + 1] - 1
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
            end_idx = rowPtr[i + 1] - 1
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