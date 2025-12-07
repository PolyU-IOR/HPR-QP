# ============================================================================
# Q Operator Types and Interfaces
# ============================================================================
# Q operators are defined in separate files under Q_operators/
# Each operator type has its own file for better organization

include("Q_operators/Q_operator_interface.jl")  # Base interface and abstract types
include("Q_operators/QAP_operator.jl")           # QAP operator
include("Q_operators/LASSO_operator.jl")         # LASSO operator
include("Q_operators/sparse_matrix_operator.jl") # Standard sparse matrix QP

# ============================================================================
# Problem Data Structures
# ============================================================================

# This struct stores the problem data.
mutable struct QP_info_cpu
    """
        Q::QTypeCPU
            The Q matrix/operator. Can be:
            - SparseMatrixCSC{Float64,Int32}: Standard QP with explicit Q matrix
            - AbstractQOperatorCPU: Operator-based Q (QAP, LASSO, custom)
            Use to_gpu(qp.Q) to transfer to GPU.

        c::Vector{Float64}
            The linear coefficient vector in the objective function.

        A::SparseMatrixCSC{Float64,Int32}
            The constraint matrix in CSC format.

        AT::SparseMatrixCSC{Float64,Int32}
            The transpose of the constraint matrix `A` in CSC format.

        AL::Vector{Float64}
            The lower bounds for the linear constraints.

        AU::Vector{Float64}
            The upper bounds for the linear constraints.

        l::Vector{Float64}
            The lower bounds for the decision variables.

        u::Vector{Float64}
            The upper bounds for the decision variables.

        obj_constant::Float64
            The constant term in the objective function.

        diag_Q::Vector{Float64}
            The diagonal elements of the matrix `Q`.

    Q_is_diag::Bool
        Indicates whether the matrix `Q` is diagonal.
        
        lambda::Float64
            Regularization parameter for LASSO problems. Ignored for other problems.
    """
    Q::QTypeCPU  # Sparse matrix or CPU operator (QAP/LASSO/custom)
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    AL::Vector{Float64}
    AU::Vector{Float64}
    l::Vector{Float64}
    u::Vector{Float64}
    obj_constant::Float64
    diag_Q::Vector{Float64}
    Q_is_diag::Bool
    lambda::Float64  # Regularization parameter for LASSO (0.0 for other problems)
end

# This struct stores the problem data for GPU computations.
# Q can be either a sparse matrix or a Q operator (for QAP/LASSO problems)
mutable struct QP_info_gpu
    Q::QType  # Union{CuSparseMatrixCSR{Float64,Int32}, AbstractQOperator}
    c::CuVector{Float64}
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    obj_constant::Float64
    diag_Q::CuVector{Float64}
    Q_is_diag::Bool
    lambda::CuVector{Float64}  # Regularization parameter for LASSO problems
end

# This struct stores the scaling information.
mutable struct Scaling_info_cpu
    l_org::Vector{Float64}
    u_org::Vector{Float64}
    row_norm::Vector{Float64}
    col_norm::Vector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
end

# This struct stores the scaling information for GPU computations.
mutable struct Scaling_info_gpu
    l_org::CuVector{Float64}
    u_org::CuVector{Float64}
    row_norm::CuVector{Float64}
    col_norm::CuVector{Float64}
    b_scale::Float64
    c_scale::Float64
    norm_b::Float64
    norm_c::Float64
    norm_b_org::Float64
    norm_c_org::Float64
end

# This struct contains parameters for the HPR-QP solver.
mutable struct HPRQP_parameters
    """
        stoptol::Float64
            Stopping tolerance for the algorithm; determines convergence accuracy.
        sigma::Float64
            Initial penalty parameter used in the algorithm.
        max_iter::Int
            Maximum number of iterations allowed.
        sigma_fixed::Bool
            Indicates whether the regularization parameter `sigma` is fixed during optimization.
        time_limit::Float64
            Maximum allowed runtime in seconds.
        eig_factor::Float64
            Factor used to scale the maximum eigenvalue estimation.
        check_iter::Int
            Frequency (in iterations) to check for convergence or perform other checks.
        warm_up::Bool
            If true, enables a warm-up phase before the main algorithm starts.
        spmv_mode_Q::String
            Mode for Q matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized", "operator").
        spmv_mode_A::String
            Mode for A matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized").
        print_frequency::Int
            Frequency (in iterations) for printing progress or logging information.
        device_number::Int32
            Identifier for the computational device (e.g., GPU device number 0 1 2 3).
        use_Ruiz_scaling::Bool
            If true, applies Ruiz scaling to the problem data.
        use_bc_scaling::Bool
            If true, applies bc scaling.
        use_l2_scaling::Bool
            If true, applies L2-norm based scaling.
        use_Pock_Chambolle_scaling::Bool
            If true, applies Pock-Chambolle scaling to the problem data.
        problem_type::String
            Type of problem being solved (e.g., "QP", "LASSO", "QAP").
        lambda::Float64
            Regularization parameter for LASSO problems.
        initial_x::Union{Vector{Float64},Nothing}
            Initial primal solution (default: nothing).
        initial_y::Union{Vector{Float64},Nothing}
            Initial dual solution (default: nothing).
        auto_save::Bool
            Automatically save best x, y, z, w, and sigma during optimization (default: false).
        save_filename::String
            Filename for auto-save HDF5 file (default: "hprqp_autosave.h5").
        verbose::Bool
            Enable verbose output (default: true).
    """
    stoptol::Float64
    sigma::Float64
    max_iter::Int
    sigma_fixed::Bool
    time_limit::Float64
    eig_factor::Float64
    check_iter::Int
    warm_up::Bool
    spmv_mode_Q::String
    spmv_mode_A::String
    print_frequency::Int
    device_number::Int32
    # scaling
    use_Ruiz_scaling::Bool
    use_bc_scaling::Bool
    use_l2_scaling::Bool
    use_Pock_Chambolle_scaling::Bool
    # problem type and regularization
    problem_type::String
    lambda::Float64
    # warm-start
    initial_x::Union{Vector{Float64},Nothing}
    initial_y::Union{Vector{Float64},Nothing}
    # auto-save
    auto_save::Bool
    save_filename::String
    # verbose output
    verbose::Bool
    HPRQP_parameters() = new(1e-6, -1, typemax(Int32), false, 3600.0, 1.05, 100, false, "auto", "auto", -1, 0, true, false, false, true, "QP", 0.0, nothing, nothing, false, "hprqp_autosave.h5", true)
end

# This struct stores the residuals and other metrics during the HPR-QP algorithm.
mutable struct HPRQP_residuals
    is_updated::Bool
    err_Rp_org_bar::Float64
    err_Rd_org_bar::Float64
    KKTx_and_gap_org_bar::Float64
    primal_obj_bar::Float64
    rel_gap_bar::Float64
    dual_obj_bar::Float64

    # Define a default constructor
    HPRQP_residuals() = new()
end

# This struct stores the results of the HPR-QP algorithm.
mutable struct HPRQP_results
    iter::Int                # Total number of iterations performed.
    iter_4::Int              # Number of iterations to get 1e-4 (if applicable).
    iter_6::Int              # Number of iterations to get 1e-6 (if applicable).
    time::Float64            # Total computation time (seconds).
    time_4::Float64          # Computation time spent to get 1e-4 (seconds).
    time_6::Float64          # Computation time spent to get 1e-6 (seconds).
    power_time::Float64      # Time spent on eigenvalue estimation (seconds).
    primal_obj::Float64      # Final value of the primal objective function.
    residuals::Float64       # Final value of the residuals.
    gap::Float64             # Final duality gap.
    status::String      # Status or type of output (e.g., "OPTIMAL", "MAX_ITER", "TIME_LIMIT").
    x::Vector{Float64}       # Solution vector for the primal variables.
    y::Vector{Float64}       # Solution vector for the dual variables (equality/inequality constraints).
    z::Vector{Float64}       # Solution vector for the dual variables (bounds).
    w::Vector{Float64}       # Auxiliary variable vector.
    HPRQP_results() = new()
end

# ============================================================================
# CUSPARSE SpMV Structures (for buffer management and preprocessing)
# ============================================================================

# CUSPARSE SpMV structure for A matrix operations
mutable struct CUSPARSE_spmv_A
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_A::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_x_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_x_hat::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_dx::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_Ax::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# CUSPARSE SpMV structure for AT (transpose of A) matrix operations
mutable struct CUSPARSE_spmv_AT
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_AT::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_y_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_y::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_ATy::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# CUSPARSE SpMV structure for Q matrix operations (when Q is a sparse matrix)
mutable struct CUSPARSE_spmv_Q
    handle::CUDA.CUSPARSE.cusparseHandle_t
    operator::Char
    alpha::Ref{Float64}
    desc_Q::CUDA.CUSPARSE.CuSparseMatrixDescriptor
    desc_w::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_w_bar::CUDA.CUSPARSE.CuDenseVectorDescriptor
    desc_w_hat::CUDA.CUSPARSE.CuDenseVectorDescriptor
    beta::Ref{Float64}
    desc_Qw::CUDA.CUSPARSE.CuDenseVectorDescriptor
    compute_type::DataType
    alg::CUDA.CUSPARSE.cusparseSpMVAlg_t
    buf::CuArray{UInt8}
end

# Note: Operator-specific CUSPARSE structures are defined in their respective operator files
# e.g., CUSPARSE_spmv_LASSO_A and CUSPARSE_spmv_LASSO_AT are in Q_operators/LASSO_operator.jl

# This struct stores the best-so-far state for auto-save feature
mutable struct HPRQP_saved_state_gpu
    # Best x found so far (GPU)
    save_x::CuVector{Float64}
    
    # Best y found so far (GPU)
    save_y::CuVector{Float64}
    
    # Best z found so far (GPU)
    save_z::CuVector{Float64}
    
    # Best w found so far (GPU)
    save_w::CuVector{Float64}
    
    # Best sigma value
    save_sigma::Float64
    
    # Iteration when best state was saved
    save_iter::Int
    
    # Primal residual at best state
    save_err_Rp::Float64
    
    # Dual residual at best state
    save_err_Rd::Float64
    
    # Primal objective at best state
    save_primal_obj::Float64
    
    # Dual objective at best state
    save_dual_obj::Float64
    
    # Relative gap at best state
    save_rel_gap::Float64
    
    # Default constructor
    HPRQP_saved_state_gpu() = new()
end

# This struct stores the workspace for the HPR-QP algorithm on the GPU.
mutable struct HPRQP_workspace_gpu
    w::CuVector{Float64}
    w_hat::CuVector{Float64}
    w_bar::CuVector{Float64}
    dw::CuVector{Float64}
    x::CuVector{Float64}
    x_hat::CuVector{Float64}
    x_bar::CuVector{Float64}
    dx::CuVector{Float64}
    y::CuVector{Float64}
    y_hat::CuVector{Float64}
    y_bar::CuVector{Float64}
    dy::CuVector{Float64}
    z_bar::CuVector{Float64}
    Q::QType  # Can be sparse matrix or Q operator
    A::CuSparseMatrixCSR{Float64,Int32}
    AT::CuSparseMatrixCSR{Float64,Int32}
    AL::CuVector{Float64}
    AU::CuVector{Float64}
    c::CuVector{Float64}
    l::CuVector{Float64}
    u::CuVector{Float64}
    Rp::CuVector{Float64}
    Rd::CuVector{Float64}
    m::Int
    n::Int
    sigma::Float64
    lambda_max_A::Float64
    lambda_max_Q::Float64
    Ax::CuVector{Float64}
    ATy::CuVector{Float64}
    ATy_bar::CuVector{Float64}
    ATdy::CuVector{Float64}
    QATdy::CuVector{Float64}
    s::CuVector{Float64}
    Qw::CuVector{Float64}
    Qw_hat::CuVector{Float64}
    Qw_bar::CuVector{Float64}
    Qx::CuVector{Float64}
    dQw::CuVector{Float64}
    last_x::CuVector{Float64}
    last_y::CuVector{Float64}
    last_Qw::CuVector{Float64}
    last_w::CuVector{Float64}
    last_ATy::CuVector{Float64}
    tempv::CuVector{Float64}
    diag_Q::CuVector{Float64}
    fact1::CuVector{Float64}
    fact2::CuVector{Float64}
    fact::CuVector{Float64}
    fact_M::CuVector{Float64}
    lambda::CuVector{Float64}  # Regularization parameter for LASSO
    # CUSPARSE SpMV structures for preprocessed matrix operations
    spmv_A::Union{CUSPARSE_spmv_A, Nothing}  # For A matrix operations (nothing if m=0)
    spmv_AT::Union{CUSPARSE_spmv_AT, Nothing}  # For AT matrix operations (nothing if m=0)
    spmv_Q::Union{CUSPARSE_spmv_Q, Nothing}  # For Q matrix operations (nothing if Q is operator)
    # Saved state for auto_save feature
    saved_state::HPRQP_saved_state_gpu
    HPRQP_workspace_gpu() = new()
end

# This struct stores the restart information for the HPR-QP algorithm.
mutable struct HPRQP_restart
    restart_flag::Int
    first_restart::Bool
    last_gap::Float64
    current_gap::Float64
    save_gap::Float64
    inner::Int
    step::Int
    sufficient::Int
    necessary::Int
    long::Int
    ratio::Int
    times::Int

    weighted_norm::Float64
    best_gap::Float64
    best_sigma::Float64

    HPRQP_restart() = new()
end