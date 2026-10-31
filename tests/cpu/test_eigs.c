#include "test_support.h"
#include "test_eigen_retry.h"

static int run_retry_case(double q, int zero_coupling)
{
    int exit_code=0;
    HPRQPModel model;
    HPRQPWorkspace workspace;
    HPRQPParams params;
    double expected;
    model_init(&model); workspace_init(&workspace);
    test_params_set_no_scaling(&params);
    params.method=HPRQP_METHOD_PARALLEL;
    params.parallel_majorizer_mode=HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR;
    TEST_ASSERT_OK(test_build_eigen_retry_case(q,zero_coupling,&model));
    TEST_ASSERT_OK(workspace_allocate(&model,&params,&workspace));
    expected=6*q*params.eig_factor;
    TEST_ASSERT(zero_coupling
                ? workspace.lambda_max_AQA==0 && workspace.lambda_max_AQ2A==0
                : fabs(workspace.lambda_max_AQA/expected-1)<1e-8 &&
                  fabs(workspace.lambda_max_AQ2A/(expected*q)-1)<1e-8,
                "moment retry spectrum q=%.17g zero=%d: AQA=%.17g AQ2A=%.17g",
                q,zero_coupling,workspace.lambda_max_AQA,workspace.lambda_max_AQ2A);
cleanup:
    workspace_free(&workspace); model_free(&model);
    return exit_code;
}

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPWorkspace workspace;
    HPRQPParams params;

    if (run_retry_case(1,0) || run_retry_case(1e-8,0) ||
        run_retry_case(1e-12,0) || run_retry_case(1,1)) return 1;

    model_init(&model);
    workspace_init(&workspace);
    test_params_set_no_scaling(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_DENSE_Q, &Q, &A, &model));
    TEST_ASSERT_OK(workspace_allocate(&model, &params, &workspace));

    TEST_ASSERT(test_nearly_equal_abs(workspace.lambda_max_A, 5.25, 1e-8) &&
                test_nearly_equal_abs(workspace.lambda_max_Q, 3.15, 1e-3),
                "eigenvalue estimates are wrong: A=%f Q=%f",
                workspace.lambda_max_A,
                workspace.lambda_max_Q);

cleanup:
    workspace_free(&workspace);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    if (exit_code == 0) {
        printf("test_eigs passed\n");
    }
    return exit_code;
}
