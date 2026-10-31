#include "test_support.h"

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPWorkspace workspace;
    HPRQPParams params;
    double expected_sigma;

    model_init(&model);
    workspace_init(&workspace);
    test_params_set_no_scaling(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_STANDARD, &Q, &A, &model));
    TEST_ASSERT_OK(workspace_allocate(&model, &params, &workspace));

    TEST_ASSERT(workspace.m == 1 && workspace.n == 2 && workspace.Q_is_diag,
                "workspace dimensions or diagonal detection are wrong");

    expected_sigma = 2.0 / sqrt(5.0);
    TEST_ASSERT(test_nearly_equal_abs(workspace.sigma, expected_sigma, 1e-12) &&
                test_nearly_equal_abs(workspace.fact[0], expected_sigma / (1.0 + 2.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact[1], expected_sigma / (1.0 + 4.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact1[0], (2.0 * expected_sigma) / (1.0 + 2.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact1[1], (4.0 * expected_sigma) / (1.0 + 4.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact2[0], 1.0 / (1.0 + 2.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact2[1], 1.0 / (1.0 + 4.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact_M[0], (expected_sigma * expected_sigma) / (1.0 + 2.0 * expected_sigma), 1e-12) &&
                test_nearly_equal_abs(workspace.fact_M[1], (expected_sigma * expected_sigma) / (1.0 + 4.0 * expected_sigma), 1e-12),
                "workspace sigma or factors are wrong");

    TEST_ASSERT(test_nearly_equal_abs(workspace.lambda_max_A, 2.1, 1e-12) &&
                test_nearly_equal_abs(workspace.lambda_max_Q, 4.0, 1e-12),
                "workspace eigenvalue estimates are wrong");

    TEST_ASSERT(workspace.Q == &model.Q && workspace.A == &model.A && workspace.AT == &model.AT,
                "workspace model bindings are wrong");

cleanup:
    workspace_free(&workspace);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    if (exit_code == 0) {
        printf("test_workspace passed\n");
    }
    return exit_code;
}