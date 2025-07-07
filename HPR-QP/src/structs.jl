# This struct stores the problem data.
mutable struct QP_info_cpu
    """
        Q::SparseMatrixCSC{Float64,Int32}
            The quadratic coefficient matrix in compressed sparse column (CSC) format.

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

        noC::Bool
            Indicates whether there are no l≤x≤u constraints (`C` is empty).
    """
    Q::SparseMatrixCSC{Float64,Int32}
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
    noC::Bool
end

# This struct stores the problem data for GPU computations.
mutable struct QP_info_gpu
    Q::CuSparseMatrixCSR{Float64,Int32}
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
    noC::Bool
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
        spmv_mode::String
            Mode for sparse matrix-vector multiplication (e.g., "auto", "CUSPARSE", "customized").
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
    """
    stoptol::Float64
    sigma::Float64
    max_iter::Int
    sigma_fixed::Bool
    time_limit::Float64
    eig_factor::Float64
    check_iter::Int
    warm_up::Bool
    spmv_mode::String
    print_frequency::Int
    device_number::Int32
    # scaling
    use_Ruiz_scaling::Bool
    use_bc_scaling::Bool
    use_l2_scaling::Bool
    use_Pock_Chambolle_scaling::Bool
    HPRQP_parameters() = new(1e-6, -1, typemax(Int32), false, 3600.0, 1.05, 100, false, "auto", -1, 0, true, false, false, true)
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
    output_type::String      # Status or type of output (e.g., "OPTIMAL", "MAX_ITER", "TIME_LIMIT").
    x::Vector{Float64}       # Solution vector for the primal variables.
    y::Vector{Float64}       # Solution vector for the dual variables (equality/inequality constraints).
    z::Vector{Float64}       # Solution vector for the dual variables (bounds).
    w::Vector{Float64}       # Auxiliary variable vector.
    HPRQP_results() = new()
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
    Q::CuSparseMatrixCSR{Float64,Int32}
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

    HPRQP_restart() = new()
end