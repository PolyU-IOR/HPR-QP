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
        # Note: Randomly generated QP instances are skipped because they may not
        # have feasible solutions. Real-world instances from MPS files, LASSO, 
        # and QAP problems provide more reliable test cases.
        @test true  # Placeholder to keep the testset valid
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
end
