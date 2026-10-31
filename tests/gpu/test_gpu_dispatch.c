#include "test_support.h"

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPParams params;
    HPRQPResults results;
    HPRQPStatus status;

    model_init(&model);
    test_params_set_no_scaling(&params);
    results_init(&results);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_DISPATCH_1D, &Q, &A, &model));

    params.use_gpu = 1;
    params.verbose = 0;
    params.device_number = 0;

    status = solve(&model, &params, &results);
    TEST_ASSERT(status == HPRQP_STATUS_OPTIMAL && results.status == HPRQP_STATUS_OPTIMAL &&
                results.iter == 0 && results.primal_obj == 0.0 &&
                results.residuals == 0.0 && results.gap == 0.0 &&
                results.x != NULL && results.x[0] == 0.0,
                "gpu solve dispatch behavior is wrong");

cleanup:
    results_free(&results);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    if (exit_code == 0) {
        printf("test_gpu_dispatch passed\n");
    }
    return exit_code;
}