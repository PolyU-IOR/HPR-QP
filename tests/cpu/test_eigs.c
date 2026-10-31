#include "test_support.h"

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPWorkspace workspace;
    HPRQPParams params;

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