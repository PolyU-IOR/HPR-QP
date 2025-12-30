using Test
using HPRQP
using SparseArrays
using LinearAlgebra
using HDF5

# Test Configuration:
# - All tests are configured to ensure convergence (OPTIMAL status)
# - max_iter = 50000-100000 (depending on problem type)
# - time_limit = 600-1800 seconds (10-30 minutes)
# - stoptol = 1e-6
# - verbose = false (no output during model building)
# - print_frequency = -1 (no intermediate solver output)
# - Primal objectives are printed for all data folder instances

@testset "HPRQP.jl Tests" begin
    
    @testset "Input Validation" begin
        # Test that qp_formulation validates inputs correctly
        
        # Helper to create valid test data
        function create_valid_qp(n=3, m=2)
            Q = sparse([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0][1:n, 1:n])
            c = zeros(n)
            A = sparse([1.0 1.0 0.0; 0.0 1.0 1.0][1:m, 1:n])
            AL = -Inf * ones(m)
            AU = ones(m)
            l = zeros(n)
            u = Inf * ones(n)
            return Q, c, A, AL, AU, l, u
        end
        
        @testset "Valid input accepted" begin
            Q, c, A, AL, AU, l, u = create_valid_qp()
            qp = HPRQP.qp_formulation(Q, c, A, AL, AU, l, u)
            @test qp !== nothing
        end
        
        @testset "Non-square Q rejected" begin
            Q, c, A, AL, AU, l, u = create_valid_qp()
            Q_bad = sparse(ones(3, 4))
            @test_throws ErrorException HPRQP.qp_formulation(Q_bad, c, A, AL, AU, l, u)
        end
        
        @testset "Dimension mismatch rejected" begin
            Q, c, A, AL, AU, l, u = create_valid_qp()
            c_bad = zeros(4)
            @test_throws ErrorException HPRQP.qp_formulation(Q, c_bad, A, AL, AU, l, u)
        end
        
        @testset "Infeasible bounds rejected" begin
            Q, c, A, AL, AU, l, u = create_valid_qp()
            l_bad = ones(3)
            u_bad = zeros(3)  # l > u
            @test_throws ErrorException HPRQP.qp_formulation(Q, c, A, AL, AU, l_bad, u_bad)
        end
        
        @testset "NaN in data rejected" begin
            Q, c, A, AL, AU, l, u = create_valid_qp()
            c_nan = [NaN, 0.0, 0.0]
            @test_throws ErrorException HPRQP.qp_formulation(Q, c_nan, A, AL, AU, l, u)
        end
        
        @testset "Empty constraints accepted" begin
            Q, c, _, _, _, l, u = create_valid_qp(3, 0)
            A_empty = spzeros(0, 3)
            AL_empty = Float64[]
            AU_empty = Float64[]
            qp = HPRQP.qp_formulation(Q, c, A_empty, AL_empty, AU_empty, l, u)
            @test qp !== nothing
        end
        
        @testset "Dense Q matrix accepted" begin
            # Test that build_from_QAbc accepts dense Q matrix
            n, m = 5, 3
            Q_dense = Matrix{Float64}(I, n, n) * 2.0  # Dense identity matrix
            c = ones(n)
            A_dense = ones(m, n)  # Dense constraint matrix
            AL = zeros(m)
            AU = fill(n/2, m)
            l = zeros(n)
            u = ones(n)
            
            # Should accept dense matrices and convert to sparse
            model = build_from_QAbc(Q_dense, c, A_dense, AL, AU, l, u; verbose=false)
            @test model !== nothing
            @test model.Q isa SparseMatrixCSC  # Should be converted to sparse
            @test model.A isa SparseMatrixCSC  # Should be converted to sparse
        end
        
        @testset "Empty (zero) Q matrix accepted" begin
            # Test that build_from_QAbc accepts empty Q matrix (all zeros)
            n, m = 5, 3
            Q_empty = zeros(n, n)  # Empty Q matrix - LP problem
            c = ones(n)
            A = sparse(ones(m, n))
            AL = zeros(m)
            AU = fill(n/2, m)
            l = zeros(n)
            u = ones(n)
            
            # Should accept empty Q and convert to sparse
            model = build_from_QAbc(Q_empty, c, A, AL, AU, l, u; verbose=false)
            @test model !== nothing
            @test model.Q isa SparseMatrixCSC
            @test nnz(model.Q) == 0  # Should have no nonzeros
        end
        
        @testset "Mixed sparse and dense matrices" begin
            # Test mixing sparse Q with dense A
            n, m = 4, 2
            Q_sparse = sparse(Matrix{Float64}(I, n, n))
            c = ones(n)
            A_dense = ones(m, n)
            AL = zeros(m)
            AU = ones(m) * 2
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q_sparse, c, A_dense, AL, AU, l, u; verbose=false)
            @test model !== nothing
            @test model.Q isa SparseMatrixCSC
            @test model.A isa SparseMatrixCSC
        end
    end
    
    @testset "QP from MPS file" begin
        # Test loading and solving a QP from MPS file
        @testset "Basic MPS solve (model.mps)" begin
            # Use the model.mps file in the root directory
            mps_file = joinpath(@__DIR__, "..", "model.mps")
            if isfile(mps_file)
                model = build_from_mps(mps_file; verbose=false)
                
                # Create solver parameters - ensure convergence
                params = HPRQP_parameters()
                params.max_iter = 50000
                params.stoptol = 1e-6
                params.time_limit = 600.0  # 10 minutes
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                # Solve
                result = optimize(model, params)
                
                # Check that we got a result
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.iter >= 0
                @test result.time > 0
                @test result.residuals < 1e-6
                
                # Check solution vectors have correct dimensions
                @test length(result.x) > 0
                @test length(result.w) > 0
                
                println("Basic MPS solve (model.mps): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "model.mps file not found, skipping MPS test"
            end
        end
        
        # Test warm-start functionality with model.mps
        @testset "Warm-start test (model.mps)" begin
            mps_file = joinpath(@__DIR__, "..", "model.mps")
            if isfile(mps_file)
                model = build_from_mps(mps_file; verbose=false)
                
                # First, solve without warm-start to get the optimal solution
                params_cold = HPRQP_parameters()
                params_cold.max_iter = 50000
                params_cold.stoptol = 1e-6
                params_cold.time_limit = 600.0
                params_cold.warm_up = false
                params_cold.verbose = false
                params_cold.print_frequency = -1
                
                result_cold = optimize(model, params_cold)
                
                @test result_cold !== nothing
                @test result_cold.status == "OPTIMAL"
                @test result_cold.residuals < 1e-6
                
                println("Warm-start test - Cold-start: Status=$(result_cold.status), Objective=$(result_cold.primal_obj)")
                
                # Save optimal solution
                x_opt = copy(result_cold.x)
                y_opt = copy(result_cold.y)
                
                # Test 1: Warm-start with only x
                params_x = HPRQP_parameters()
                params_x.max_iter = 50000
                params_x.stoptol = 1e-6
                params_x.time_limit = 600.0
                params_x.warm_up = false
                params_x.verbose = false
                params_x.print_frequency = -1
                params_x.initial_x = x_opt
                
                result_x = optimize(model, params_x)
                
                @test result_x !== nothing
                @test result_x.status == "OPTIMAL"
                @test result_x.residuals < 1e-6
                @test result_x.iter <= result_cold.iter  # Should converge in same or fewer iterations
                
                println("Warm-start (x only): Status=$(result_x.status), Objective=$(result_x.primal_obj)")
                
                # Test 2: Warm-start with only y
                params_y = HPRQP_parameters()
                params_y.max_iter = 50000
                params_y.stoptol = 1e-6
                params_y.time_limit = 600.0
                params_y.warm_up = false
                params_y.verbose = false
                params_y.print_frequency = -1
                params_y.initial_y = y_opt
                
                result_y = optimize(model, params_y)
                
                @test result_y !== nothing
                @test result_y.status == "OPTIMAL"
                @test result_y.residuals < 1e-6
                # Note: Providing only y may not help much since x is still random
                
                println("Warm-start (y only): Status=$(result_y.status), Objective=$(result_y.primal_obj)")
                
                # Test 3: Warm-start with both x and y (should converge in 0 iterations or very few)
                params_xy = HPRQP_parameters()
                params_xy.max_iter = 50000
                params_xy.stoptol = 1e-6
                params_xy.time_limit = 600.0
                params_xy.warm_up = false
                params_xy.verbose = false
                params_xy.print_frequency = -1
                params_xy.initial_x = x_opt
                params_xy.initial_y = y_opt
                
                result_xy = optimize(model, params_xy)
                
                @test result_xy !== nothing
                @test result_xy.status == "OPTIMAL"
                @test result_xy.residuals < 1e-6
                @test result_xy.iter == 0  # Should converge very quickly (ideally 0 iterations)
                
                println("Warm-start (x and y): Status=$(result_xy.status), Objective=$(result_xy.primal_obj)")
            else
                @warn "model.mps file not found, skipping warm-start test"
            end
        end
        
        # Test AUG2D.mps from data folder
        @testset "AUG2D.mps from data folder" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            if isfile(mps_file)
                model = build_from_mps(mps_file; verbose=false)
                
                # Create solver parameters - ensure convergence
                params = HPRQP_parameters()
                params.max_iter = 50000
                params.stoptol = 1e-6
                params.time_limit = 600.0  # 10 minutes
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                # Solve
                result = optimize(model, params)
                
                # Check that we got a result
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.residuals < 1e-6
                
                println("AUG2D.mps: Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "AUG2D.mps file not found, skipping test"
            end
        end
    end
    
    @testset "QP from matrices (QAbc)" begin
        # Test solving QP problems with dense and empty Q matrices
        @testset "Solve with dense Q matrix" begin
            n, m = 10, 5
            # Create a small dense QP
            Q_dense = Matrix{Float64}(I, n, n) * 2.0
            c = -ones(n)
            A_dense = ones(m, n)
            AL = zeros(m)
            AU = fill(n/2, m)
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q_dense, c, A_dense, AL, AU, l, u; verbose=false)
            
            params = HPRQP_parameters()
            params.max_iter = 10000
            params.stoptol = 1e-6
            params.time_limit = 120.0
            params.warm_up = false
            params.verbose = false
            params.print_frequency = -1
            
            result = optimize(model, params)
            
            @test result !== nothing
            @test result.status == "OPTIMAL"
            @test result.residuals < 1e-6
            @test length(result.x) == n
            
            println("Dense Q matrix solve: Status=$(result.status), Objective=$(result.primal_obj)")
        end
        
        @testset "Solve with empty (zero) Q matrix - LP" begin
            # This is essentially an LP problem (Q = 0)
            n, m = 8, 4
            Q_empty = zeros(n, n)  # Empty Q - linear programming
            c = -ones(n)  # Minimize -sum(x), equivalent to maximize sum(x)
            A = ones(m, n)
            AL = fill(2.0, m)
            AU = fill(5.0, m)
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q_empty, c, A, AL, AU, l, u; verbose=false)
            
            params = HPRQP_parameters()
            params.max_iter = 10000
            params.stoptol = 1e-6
            params.time_limit = 120.0
            params.warm_up = false
            params.verbose = false
            params.print_frequency = -1
            
            result = optimize(model, params)
            
            @test result !== nothing
            @test result.status == "OPTIMAL"
            @test result.residuals < 1e-6
            @test length(result.x) == n
            
            println("Empty Q matrix (LP) solve: Status=$(result.status), Objective=$(result.primal_obj)")
        end
    end
    
    @testset "Parameter validation" begin
        # Test that solver handles different parameter configurations
        @testset "Parameter settings" begin
            # Create a tiny problem
            n, m = 5, 2
            Q = sparse(Matrix{Float64}(I, n, n))
            c = ones(n)
            A = sparse(ones(m, n))
            lcon = zeros(m)
            ucon = fill(n/2, m)
            lvar = zeros(n)
            uvar = ones(n)
            
            model = build_from_QAbc(Q, c, A, lcon, ucon, lvar, uvar; verbose=false)
            
            # Test with different tolerances
            for tol in [1e-3, 1e-5]
                params = HPRQP_parameters()
                params.max_iter = 50000
                params.stoptol = tol
                params.time_limit = 600.0
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                result = optimize(model, params)
                @test result !== nothing
                @test result.status == "OPTIMAL"
            end
            
            println("✓ Parameter validation test passed")
        end
    end
    
    @testset "Result structure" begin
        # Test that result structure contains expected fields
        @testset "Result fields" begin
            n, m = 4, 2
            Q = sparse(Matrix{Float64}(I, n, n))
            c = ones(n)
            A = sparse(ones(m, n))
            lcon = zeros(m)
            ucon = ones(m) * 2
            lvar = zeros(n)
            uvar = ones(n)
            
            model = build_from_QAbc(Q, c, A, lcon, ucon, lvar, uvar; verbose=false)
            
            params = HPRQP_parameters()
            params.max_iter = 50000
            params.stoptol = 1e-6
            params.time_limit = 600.0
            params.warm_up = false
            params.verbose = false
            params.print_frequency = -1
            
            result = optimize(model, params)
            
            # Check result has all expected fields
            @test hasfield(typeof(result), :x)
            @test hasfield(typeof(result), :y)
            @test hasfield(typeof(result), :z)
            @test hasfield(typeof(result), :w)
            @test hasfield(typeof(result), :iter)
            @test hasfield(typeof(result), :time)
            @test hasfield(typeof(result), :residuals)
            @test hasfield(typeof(result), :primal_obj)
            @test hasfield(typeof(result), :gap)
            @test hasfield(typeof(result), :status)
            
            println("✓ Result structure test passed")
        end
    end
    
    @testset "LASSO Problem" begin
        # Test LASSO problem from .mat file
        @testset "LASSO from E2006.test.mat" begin
            lasso_file = joinpath(@__DIR__, "..", "data", "E2006.test.mat")
            
            if isfile(lasso_file)
                # Build LASSO model from .mat file
                model = build_from_mat(lasso_file; problem_type="LASSO", verbose=false)
                
                # Create solver parameters - ensure convergence
                params = HPRQP_parameters()
                params.max_iter = 100000  # LASSO may need many iterations
                params.time_limit = 1800.0  # 30 minutes
                params.stoptol = 1e-6
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                params.problem_type = "LASSO"
                params.use_bc_scaling = false
                params.use_l2_scaling = false
                params.use_Pock_Chambolle_scaling = false
                params.use_Ruiz_scaling = false
                
                # Solve
                result = optimize(model, params)
                
                # Check that we got a result
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.iter >= 0
                @test result.time > 0
                @test result.residuals < 1e-6
                @test !isnan(result.residuals)  # No NaN values
                
                # Check solution vectors have correct dimensions
                @test length(result.x) > 0
                @test length(result.w) > 0
                
                println("LASSO (E2006.test.mat): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "E2006.test.mat file not found, skipping LASSO test"
            end
        end
    end
    
    @testset "QAP Problem" begin
        # Test QAP problem from .mat file
        @testset "QAP from esc64a.mat" begin
            qap_file = joinpath(@__DIR__, "..", "data", "esc64a.mat")
            
            if isfile(qap_file)
                # Build QAP model from .mat file
                model = build_from_mat(qap_file; problem_type="QAP", verbose=false)
                
                # Create solver parameters - ensure convergence
                params = HPRQP_parameters()
                params.max_iter = 100000
                params.time_limit = 1800.0  # 30 minutes
                params.stoptol = 1e-6
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                params.problem_type = "QAP"
                
                # Solve
                result = optimize(model, params)
                
                # Check that we got a result
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.iter >= 0
                @test result.time > 0
                @test result.residuals < 1e-6
                @test !isnan(result.residuals)  # No NaN values
                
                # Check solution vectors have correct dimensions
                @test length(result.x) > 0
                @test length(result.w) > 0
                
                println("QAP (esc64a.mat): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "esc64a.mat file not found, skipping QAP test"
            end
        end
    end
    
    @testset "Auto-save feature" begin
        @testset "HDF5 auto-save test" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            if isfile(mps_file)
                model = build_from_mps(mps_file; verbose=false)
                
                # Create solver parameters with auto-save enabled
                params = HPRQP_parameters()
                params.max_iter = 5000
                params.stoptol = 1e-4
                params.time_limit = 120.0
                params.check_iter = 100
                params.verbose = false
                params.print_frequency = 500  # Save less frequently in tests
                params.warm_up = false
                
                # Enable auto-save
                params.auto_save = true
                test_filename = "test_autosave_runtests.h5"
                params.save_filename = test_filename
                
                # Remove old file if exists
                if isfile(test_filename)
                    rm(test_filename)
                end
                
                result = optimize(model, params)
                
                # Check solver results
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.residuals < 1e-4
                
                # Check that HDF5 file was created
                @test isfile(test_filename)
                
                # Read and verify HDF5 file contents
                h5open(test_filename, "r") do file
                    # Check current state exists
                    @test haskey(file, "current/iteration")
                    @test haskey(file, "current/x_org")
                    @test haskey(file, "current/w_org")
                    @test haskey(file, "current/z_org")
                    @test haskey(file, "current/y_org")
                    @test haskey(file, "current/sigma")
                    @test haskey(file, "current/err_Rp")
                    @test haskey(file, "current/err_Rd")
                    @test haskey(file, "current/primal_obj")
                    @test haskey(file, "current/dual_obj")
                    @test haskey(file, "current/rel_gap")
                    
                    # Check best solution exists
                    @test haskey(file, "best/iteration")
                    @test haskey(file, "best/x_org")
                    @test haskey(file, "best/w_org")
                    @test haskey(file, "best/z_org")
                    @test haskey(file, "best/y_org")
                    @test haskey(file, "best/sigma")
                    @test haskey(file, "best/err_Rp")
                    @test haskey(file, "best/err_Rd")
                    
                    # Check parameters saved
                    @test haskey(file, "parameters/stoptol")
                    @test haskey(file, "parameters/auto_save")
                    @test read(file, "parameters/auto_save") == true
                    
                    # Verify dimensions
                    x_best = read(file, "best/x_org")
                    w_best = read(file, "best/w_org")
                    z_best = read(file, "best/z_org")
                    y_best = read(file, "best/y_org")
                    
                    @test length(x_best) > 0
                    @test length(w_best) > 0
                    @test length(z_best) > 0
                    @test length(y_best) > 0
                    @test length(x_best) == length(w_best)
                    @test length(x_best) == length(z_best)
                    
                    # Check that best solution has reasonable values
                    best_err_Rp = read(file, "best/err_Rp")
                    best_err_Rd = read(file, "best/err_Rd")
                    @test best_err_Rp < 1e-4
                    @test best_err_Rd < 1e-4
                    
                    # Verify iteration counter
                    current_iter = read(file, "current/iteration")
                    best_iter = read(file, "best/iteration")
                    @test current_iter >= 0
                    @test best_iter >= 0
                    @test best_iter <= current_iter
                end
                
                # Clean up test file
                rm(test_filename)
                
                println("Auto-save (AUG2D.mps): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "AUG2D.mps file not found, skipping auto-save test"
            end
        end
    end
    
    @testset "CPU Mode Tests with Data Files" begin
        @testset "CPU - MPS file (AUG2D.mps)" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            
            if isfile(mps_file)
                # Build model from MPS file
                model = build_from_mps(mps_file; verbose=false)
                
                # Configure parameters for CPU
                params = HPRQP_parameters()
                params.use_gpu = false
                params.max_iter = 100000
                params.stoptol = 1e-6
                params.time_limit = 1200.0
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                
                result = optimize(model, params)
                
                @test result !== nothing
                @test result.status in ["OPTIMAL", "MAX_ITER"]
                if result.status == "OPTIMAL"
                    @test result.residuals < 1e-6
                end
                
                println("CPU - AUG2D.mps: Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "AUG2D.mps file not found, skipping CPU MPS test"
            end
        end
        
        @testset "CPU - LASSO (E2006.test.mat)" begin
            lasso_file = joinpath(@__DIR__, "..", "data", "E2006.test.mat")
            
            if isfile(lasso_file)
                # Build LASSO model from .mat file
                model = build_from_mat(lasso_file; problem_type="LASSO", verbose=false)
                
                # Configure parameters for CPU
                params = HPRQP_parameters()
                params.use_gpu = false
                params.max_iter = 100000
                params.stoptol = 1e-6
                params.time_limit = 1800.0
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                params.problem_type = "LASSO"
                
                result = optimize(model, params)
                
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.residuals < 1e-6
                
                println("CPU - LASSO (E2006.test.mat): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "E2006.test.mat file not found, skipping CPU LASSO test"
            end
        end
        
        @testset "CPU - QAP (esc64a.mat)" begin
            qap_file = joinpath(@__DIR__, "..", "data", "esc64a.mat")
            
            if isfile(qap_file)
                # Build QAP model from .mat file
                model = build_from_mat(qap_file; problem_type="QAP", verbose=false)
                
                # Configure parameters for CPU
                params = HPRQP_parameters()
                params.use_gpu = false
                params.max_iter = 80000
                params.stoptol = 1e-6
                params.time_limit = 1200.0
                params.warm_up = false
                params.verbose = false
                params.print_frequency = -1
                params.problem_type = "QAP"
                
                result = optimize(model, params)
                
                @test result !== nothing
                @test result.status == "OPTIMAL"
                @test result.residuals < 1e-6
                
                println("CPU - QAP (esc64a.mat): Status=$(result.status), Objective=$(result.primal_obj)")
            else
                @warn "esc64a.mat file not found, skipping CPU QAP test"
            end
        end
    end
    
    @testset "SpMV Mode Tests (GPU)" begin
        # Test different combinations of spmv_mode_Q and spmv_mode_A for AUG2D.mps and AUG3DCQP.mps
        
        @testset "AUG2D.mps - SpMV mode combinations" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG2D.mps")
            
            if isfile(mps_file)
                # Test all 4 combinations: CUSPARSE/CUSPARSE, CUSPARSE/customized, customized/CUSPARSE, customized/customized
                test_combinations = [
                    ("CUSPARSE", "CUSPARSE", "CUSPARSE Q + CUSPARSE A"),
                    ("CUSPARSE", "customized", "CUSPARSE Q + customized A"),
                    ("customized", "CUSPARSE", "customized Q + CUSPARSE A"),
                    ("customized", "customized", "customized Q + customized A")
                ]
                
                for (spmv_Q, spmv_A, desc) in test_combinations
                    @testset "$desc" begin
                        model = build_from_mps(mps_file; verbose=false)
                        
                        params = HPRQP_parameters()
                        params.use_gpu = true
                        params.spmv_mode_Q = spmv_Q
                        params.spmv_mode_A = spmv_A
                        params.max_iter = 50000
                        params.stoptol = 1e-6
                        params.time_limit = 600.0
                        params.warm_up = false
                        params.verbose = false
                        params.print_frequency = -1
                        
                        result = optimize(model, params)
                        
                        @test result !== nothing
                        @test result.status == "OPTIMAL"
                        @test result.residuals < 1e-6
                        
                        println("AUG2D.mps ($desc): Status=$(result.status), Objective=$(result.primal_obj)")
                    end
                end
            else
                @warn "AUG2D.mps file not found, skipping SpMV mode tests"
            end
        end
        
        @testset "AUG3DCQP.mps - SpMV mode combinations" begin
            mps_file = joinpath(@__DIR__, "..", "data", "AUG3DCQP.mps")
            
            if isfile(mps_file)
                # Test all 4 combinations: CUSPARSE/CUSPARSE, CUSPARSE/customized, customized/CUSPARSE, customized/customized
                test_combinations = [
                    ("CUSPARSE", "CUSPARSE", "CUSPARSE Q + CUSPARSE A"),
                    ("CUSPARSE", "customized", "CUSPARSE Q + customized A"),
                    ("customized", "CUSPARSE", "customized Q + CUSPARSE A"),
                    ("customized", "customized", "customized Q + customized A")
                ]
                
                for (spmv_Q, spmv_A, desc) in test_combinations
                    @testset "$desc" begin
                        model = build_from_mps(mps_file; verbose=false)
                        
                        params = HPRQP_parameters()
                        params.use_gpu = true
                        params.spmv_mode_Q = spmv_Q
                        params.spmv_mode_A = spmv_A
                        params.max_iter = 50000
                        params.stoptol = 1e-6
                        params.time_limit = 600.0
                        params.warm_up = false
                        params.verbose = false
                        params.print_frequency = -1
                        
                        result = optimize(model, params)
                        
                        @test result !== nothing
                        @test result.status == "OPTIMAL"
                        @test result.residuals < 1e-6
                        
                        println("AUG3DCQP.mps ($desc): Status=$(result.status), Objective=$(result.primal_obj)")
                    end
                end
            else
                @warn "AUG3DCQP.mps file not found, skipping SpMV mode tests"
            end
        end
    end
    
    @testset "GPU Validation" begin
        # Test GPU availability checking and automatic fallback to CPU
        
        @testset "GPU validation function exists" begin
            # Test that the validate_gpu_parameters! function is defined
            @test isdefined(HPRQP, :validate_gpu_parameters!)
        end
        
        @testset "CPU mode works correctly" begin
            # Create a simple test problem
            n, m = 10, 5
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = 10 * ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test with explicit CPU mode
            params = HPRQP_parameters()
            params.use_gpu = false
            params.max_iter = 100
            params.verbose = false
            
            result = optimize(model, params)
            @test result !== nothing
            @test result.status in ["OPTIMAL", "MAX_ITER"]
        end
        
        @testset "GPU parameter validation" begin
            using CUDA
            
            # Create a simple test problem
            n, m = 10, 5
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = 10 * ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test with invalid GPU device number
            params = HPRQP_parameters()
            params.use_gpu = true
            params.device_number = 999  # Invalid device number
            params.max_iter = 100
            params.verbose = false
            
            # Should automatically fall back to CPU without error
            result = optimize(model, params)
            @test result !== nothing
            @test result.status in ["OPTIMAL", "MAX_ITER"]
            
            # After validation, use_gpu should be false if GPU is unavailable or device is invalid
            if !CUDA.functional() || params.device_number >= length(CUDA.devices())
                # GPU should have been disabled
                @test params.use_gpu == false || CUDA.functional()
            end
        end
        
        @testset "Default GPU behavior" begin
            using CUDA
            
            # Create a simple test problem
            n, m = 10, 5
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = 10 * ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test default parameters (should use GPU if available)
            params = HPRQP_parameters()
            params.max_iter = 100
            params.verbose = false
            initial_use_gpu = params.use_gpu
            
            result = optimize(model, params)
            @test result !== nothing
            @test result.status in ["OPTIMAL", "MAX_ITER"]
            
            # If CUDA is not functional, use_gpu should be false after optimization
            if !CUDA.functional()
                @test params.use_gpu == false
            end
        end
        
        @testset "Parameter printing shows correct device" begin
            # Create a simple test problem
            n, m = 5, 3
            Q = sparse(1.0I, n, n)
            c = ones(n)
            A = sparse(rand(m, n))
            AL = -ones(m)
            AU = ones(m)
            l = zeros(n)
            u = ones(n)
            
            model = build_from_QAbc(Q, c, A, AL, AU, l, u; verbose=false)
            
            # Test with CPU mode - just verify it runs without error
            # (actual output checking would require more complex test setup)
            params = HPRQP_parameters()
            params.use_gpu = false
            params.max_iter = 10
            params.verbose = false  # Keep verbose off for clean test output
            
            result = optimize(model, params)
            
            # Verify the result is valid and use_gpu is correctly set
            @test result !== nothing
            @test params.use_gpu == false
            @test result.status in ["OPTIMAL", "MAX_ITER"]
        end
    end
end
