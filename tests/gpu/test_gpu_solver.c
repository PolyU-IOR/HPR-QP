#include "test_support.h"

typedef enum TestCaseKind {
    TEST_CASE_STANDARD = 0,
    TEST_CASE_NOC = 1,
    TEST_CASE_EMPTY_A = 2,
    TEST_CASE_TALL_DIAG = 3,
    TEST_CASE_PARALLEL_SINGULAR = 4,
} TestCaseKind;

static HPRQPStatus build_model_for_case(TestCaseKind kind,
                                        HPRQPMatrixCSR *Q,
                                        HPRQPMatrixCSR *A,
                                        HPRQPModel *model)
{
    switch (kind) {
    case TEST_CASE_STANDARD:
        return test_build_model_case(HPRQP_TEST_CASE_STANDARD, Q, A, model);
    case TEST_CASE_NOC:
        return test_build_model_case(HPRQP_TEST_CASE_STANDARD_NOC, Q, A, model);
    case TEST_CASE_EMPTY_A:
        return test_build_model_case(HPRQP_TEST_CASE_EMPTY_A, Q, A, model);
    case TEST_CASE_TALL_DIAG:
        return test_build_model_case(HPRQP_TEST_CASE_TALL_DIAG, Q, A, model);
    case TEST_CASE_PARALLEL_SINGULAR:
        return test_build_model_case(HPRQP_TEST_CASE_PARALLEL_SINGULAR,
                                     Q,
                                     A,
                                     model);
    }

    return HPRQP_STATUS_INVALID_ARGUMENT;
}

static int run_case(TestCaseKind kind, const char *label)
{
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPParams cpu_params;
    HPRQPParams gpu_params;
    HPRQPResults cpu_results;
    HPRQPResults gpu_results;
    HPRQPStatus status;

    model_init(&model);
    test_params_set_no_scaling(&cpu_params);
    test_params_set_no_scaling(&gpu_params);
    results_init(&cpu_results);
    results_init(&gpu_results);

    status = build_model_for_case(kind, &Q, &A, &model);
    if (status != HPRQP_STATUS_OK) {
        return 1;
    }

    cpu_params.print_frequency = 0;
    cpu_params.check_iter = 1;
    cpu_params.max_iter = 4;
    cpu_params.stoptol = 0.0;
    cpu_params.verbose = 0;
    if (kind == TEST_CASE_PARALLEL_SINGULAR) {
        cpu_params.method = HPRQP_METHOD_PARALLEL;
        cpu_params.tau_mode = HPRQP_TAU_MODE_FIXED;
        cpu_params.tau = 0.75;
        cpu_params.sigma_fixed = 1;
    }

    gpu_params = cpu_params;
    gpu_params.use_gpu = 1;

    status = solve(&model, &cpu_params, &cpu_results);
    if (status != HPRQP_STATUS_MAX_ITER) {
        fprintf(stderr, "cpu solve failed for %s with status=%d\n", label, (int)status);
        goto fail;
    }
    status = solve(&model, &gpu_params, &gpu_results);
    if (status != HPRQP_STATUS_MAX_ITER) {
        fprintf(stderr, "gpu solve failed for %s with status=%d\n", label, (int)status);
        goto fail;
    }

    if (cpu_results.status != gpu_results.status ||
        cpu_results.iter != gpu_results.iter ||
        !test_nearly_equal_rel(cpu_results.primal_obj, gpu_results.primal_obj, 1e-8) ||
        !test_nearly_equal_rel(cpu_results.residuals, gpu_results.residuals, 1e-8) ||
        !test_nearly_equal_rel(cpu_results.gap, gpu_results.gap, 1e-8)) {
        fprintf(stderr, "cpu/gpu solve mismatch for %s\n", label);
        goto fail;
    }
    for (size_t i = 0; i < model.Q.nrow; ++i) {
        if (!test_nearly_equal_rel(cpu_results.x[i], gpu_results.x[i], 1e-8) ||
            !test_nearly_equal_rel(cpu_results.w[i], gpu_results.w[i], 1e-8) ||
            !test_nearly_equal_rel(cpu_results.z[i], gpu_results.z[i], 1e-8)) {
            fprintf(stderr,
                    "cpu/gpu primal state mismatch for %s at index %zu\n",
                    label,
                    i);
            goto fail;
        }
    }
    for (size_t i = 0; i < model.A.nrow; ++i) {
        if (!test_nearly_equal_rel(cpu_results.y[i], gpu_results.y[i], 1e-8)) {
            fprintf(stderr,
                    "cpu/gpu dual solve mismatch for %s at index %zu\n",
                    label,
                    i);
            goto fail;
        }
    }
    if (kind == TEST_CASE_PARALLEL_SINGULAR &&
        (cpu_results.method != HPRQP_METHOD_PARALLEL ||
         gpu_results.method != HPRQP_METHOD_PARALLEL ||
         !test_nearly_equal_rel(cpu_results.lambda_max_AQA,
                                gpu_results.lambda_max_AQA,
                                1e-4) ||
         !test_nearly_equal_rel(cpu_results.zeta_AQ,
                                gpu_results.zeta_AQ,
                                1e-4))) {
        fprintf(stderr, "cpu/gpu parallel scalar mismatch for %s\n", label);
        goto fail;
    }

    results_free(&gpu_results);
    results_free(&cpu_results);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return 0;

fail:
    results_free(&gpu_results);
    results_free(&cpu_results);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return 1;
}

int main(void)
{
    if (run_case(TEST_CASE_STANDARD, "standard") != 0) {
        return 1;
    }
    if (run_case(TEST_CASE_NOC, "noc") != 0) {
        return 1;
    }
    if (run_case(TEST_CASE_EMPTY_A, "empty-a") != 0) {
        return 1;
    }
    if (run_case(TEST_CASE_TALL_DIAG, "tall-diag") != 0) {
        return 1;
    }
    if (run_case(TEST_CASE_PARALLEL_SINGULAR, "parallel-singular-q") != 0) {
        return 1;
    }

    printf("test_gpu_solver passed\n");
    return 0;
}
