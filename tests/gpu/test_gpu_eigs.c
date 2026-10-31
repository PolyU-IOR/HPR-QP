#include "test_support.h"
#include "test_eigen_retry.h"

static int run_retry_case(double q, int zero_coupling)
{
    int exit_code=0;
    HPRQPModel model;
    HPRQPParams params;
    HPRQPGPUContext context;
    HPRQPGPUProblem problem;
    HPRQPGPUWorkspace workspace;
    double expected;
    model_init(&model); gpu_context_init(&context);
    gpu_problem_init(&problem); gpu_workspace_init(&workspace);
    test_params_set_no_scaling(&params);
    params.method=HPRQP_METHOD_PARALLEL;
    params.parallel_majorizer_mode=HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR;
    TEST_ASSERT_OK(test_build_eigen_retry_case(q,zero_coupling,&model));
    TEST_ASSERT_OK(gpu_context_open(&context,0));
    TEST_ASSERT_OK(gpu_problem_upload(&problem,&model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&workspace,&problem,&params));
    TEST_ASSERT_OK(gpu_workspace_prepare_spmv(&workspace,&problem,0));
    TEST_ASSERT_OK(gpu_estimate_eigenvalues(&workspace,&problem,&params));
    expected=6*q*params.eig_factor;
    TEST_ASSERT(zero_coupling
                ? workspace.lambda_max_AQA==0 && workspace.lambda_max_AQ2A==0
                : fabs(workspace.lambda_max_AQA/expected-1)<1e-8 &&
                  fabs(workspace.lambda_max_AQ2A/(expected*q)-1)<1e-8,
                "gpu moment retry spectrum q=%.17g zero=%d: AQA=%.17g AQ2A=%.17g",
                q,zero_coupling,workspace.lambda_max_AQA,workspace.lambda_max_AQ2A);
cleanup:
    gpu_workspace_free(&workspace); gpu_problem_free(&problem);
    gpu_context_free(&context); model_free(&model);
    return exit_code;
}

static int run_case(HPRQPTestModelCase case_id,
                    HPRQPMethod method,
                    const char *label)
{
    int exit_code = 0;
    HPRQPMatrixCSR cpu_Q = {0};
    HPRQPMatrixCSR cpu_A = {0};
    HPRQPMatrixCSR gpu_Q = {0};
    HPRQPMatrixCSR gpu_A = {0};
    HPRQPModel cpu_model;
    HPRQPModel gpu_model;
    HPRQPWorkspace cpu_workspace;
    HPRQPParams params;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;

    model_init(&cpu_model);
    model_init(&gpu_model);
    workspace_init(&cpu_workspace);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    test_params_set_no_scaling(&params);
    params.method = method;

    TEST_ASSERT_OK(test_build_model_case(case_id, &cpu_Q, &cpu_A, &cpu_model));
    TEST_ASSERT_OK(test_build_model_case(case_id, &gpu_Q, &gpu_A, &gpu_model));
    TEST_ASSERT_OK(workspace_allocate(&cpu_model, &params, &cpu_workspace));

    TEST_ASSERT_OK(gpu_context_open(&context, 0));
    TEST_ASSERT_OK(gpu_problem_upload(&gpu_problem, &gpu_model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params));
    TEST_ASSERT_OK(gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0));
    TEST_ASSERT_OK(gpu_estimate_eigenvalues(&gpu_workspace, &gpu_problem, &params));

    TEST_ASSERT(gpu_workspace.Q_is_diag == cpu_workspace.Q_is_diag &&
                test_nearly_equal_rel(gpu_workspace.lambda_max_A, cpu_workspace.lambda_max_A, 1e-6) &&
                test_nearly_equal_rel(gpu_workspace.lambda_max_Q, cpu_workspace.lambda_max_Q, 1e-4),
                "gpu eigenvalue estimates are wrong for %s: A=%g expected=%g Q=%g expected=%g",
                label,
                gpu_workspace.lambda_max_A,
                cpu_workspace.lambda_max_A,
                gpu_workspace.lambda_max_Q,
                cpu_workspace.lambda_max_Q);
    if (method == HPRQP_METHOD_PARALLEL) {
        TEST_ASSERT(test_nearly_equal_rel(gpu_workspace.lambda_max_AQA,
                                          cpu_workspace.lambda_max_AQA,
                                          1e-4) &&
                        test_nearly_equal_rel(gpu_workspace.zeta_AQ,
                                              cpu_workspace.zeta_AQ,
                                              1e-4) &&
                        gpu_workspace.zeta_AQ >= 0.0 &&
                        gpu_workspace.zeta_AQ <= 1.0,
                    "gpu AQA estimate is wrong for %s: AQA=%g expected=%g zeta=%g expected=%g",
                    label,
                    gpu_workspace.lambda_max_AQA,
                    cpu_workspace.lambda_max_AQA,
                    gpu_workspace.zeta_AQ,
                    cpu_workspace.zeta_AQ);
    }

cleanup:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&cpu_workspace);
    model_free(&gpu_model);
    model_free(&cpu_model);
    matrix_csr_free(&gpu_A);
    matrix_csr_free(&gpu_Q);
    matrix_csr_free(&cpu_A);
    matrix_csr_free(&cpu_Q);
    return exit_code;
}

int main(void)
{
    if (!gpu_backend_available()) {
        printf("test_gpu_eigs skipped\n");
        return 0;
    }

    if (run_retry_case(1,0) || run_retry_case(1e-8,0) ||
        run_retry_case(1e-12,0) || run_retry_case(1,1)) return 1;

    if (run_case(HPRQP_TEST_CASE_STANDARD, HPRQP_METHOD_SGS, "diag") != 0) {
        return 1;
    }
    if (run_case(HPRQP_TEST_CASE_DENSE_Q, HPRQP_METHOD_SGS, "dense-q") != 0) {
        return 1;
    }
    if (run_case(HPRQP_TEST_CASE_PARALLEL_SINGULAR,
                 HPRQP_METHOD_PARALLEL,
                 "parallel-singular-q") != 0) {
        return 1;
    }

    printf("test_gpu_eigs passed\n");
    return 0;
}
