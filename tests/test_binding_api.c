#include "hprqp_binding.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifndef HPRQP_SOURCE_DIR
#define HPRQP_SOURCE_DIR "."
#endif

static int nearly_equal(double lhs, double rhs)
{
    double scale = fabs(lhs) > fabs(rhs) ? fabs(lhs) : fabs(rhs);
    if (scale < 1.0) {
        scale = 1.0;
    }
    return fabs(lhs - rhs) <= 1e-6 * scale;
}

static int test_read_from_mps_and_max_iter_setter(void)
{
    HPRQPBindingModel *model = NULL;
    HPRQPBindingParams *params = NULL;
    HPRQPBindingResults *results = NULL;
    HPRQPStatus status;

    status = hprqp_model_create(&model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_model_create failed: %d\n", status);
        return 1;
    }

    status = hprqp_params_create(&params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_create failed: %d\n", status);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_results_create(&results);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_results_create failed: %d\n", status);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_model_read_from_mps(model, HPRQP_SOURCE_DIR "/tests/data/simple_qp.mps");
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_model_read_from_mps failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_model_num_variables(model) != 2 || hprqp_model_num_constraints(model) != 2) {
        fprintf(stderr, "unexpected model dimensions from MPS\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_verbose(params, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_verbose failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_max_iter(params, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_max_iter failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_use_cr_scaling(params, 1);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_use_cr_scaling failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_cr_iterations(params, 10);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_cr_iterations failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_optimize(model, params, results);
    if (status != HPRQP_STATUS_MAX_ITER) {
        fprintf(stderr, "expected MAX_ITER from binding optimize, got %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_results_status(results) != HPRQP_STATUS_MAX_ITER) {
        fprintf(stderr, "binding results status mismatch\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_results_num_variables(results) != 2 || hprqp_results_num_constraints(results) != 2) {
        fprintf(stderr, "binding results dimensions mismatch\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_results_x_data(results) == NULL || hprqp_results_y_data(results) == NULL) {
        fprintf(stderr, "expected solution vectors after MAX_ITER solve\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    hprqp_results_destroy(results);
    hprqp_params_destroy(params);
    hprqp_model_destroy(model);
    return 0;
}

static int test_build_from_qabc_csr_and_optimize(void)
{
    static const size_t q_row_ptr[] = {0, 1, 2};
    static const size_t q_col_ind[] = {0, 1};
    static const double q_values[] = {2.0, 2.0};
    static const double c[] = {-4.0, -2.0};
    static const double l[] = {-INFINITY, -INFINITY};
    static const double u[] = {INFINITY, INFINITY};
    HPRQPBindingModel *model = NULL;
    HPRQPBindingParams *params = NULL;
    HPRQPBindingResults *results = NULL;
    const double *x;
    HPRQPStatus status;

    status = hprqp_model_create(&model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_model_create failed: %d\n", status);
        return 1;
    }

    status = hprqp_params_create(&params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_create failed: %d\n", status);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_results_create(&results);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_results_create failed: %d\n", status);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_model_build_from_qabc_csr(model,
                                             2,
                                             2,
                                             2,
                                             q_row_ptr,
                                             q_col_ind,
                                             q_values,
                                             0,
                                             2,
                                             0,
                                             NULL,
                                             NULL,
                                             NULL,
                                             c,
                                             NULL,
                                             NULL,
                                             l,
                                             u,
                                             0.0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_model_build_from_qabc_csr failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_defaults(params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_defaults failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_verbose(params, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_verbose failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_params_set_stoptol(params, 1e-8);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_set_stoptol failed: %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    status = hprqp_optimize(model, params, results);
    if (status != HPRQP_STATUS_OPTIMAL) {
        fprintf(stderr, "expected OPTIMAL from binding optimize, got %d\n", status);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_results_num_variables(results) != 2 || hprqp_results_num_constraints(results) != 0) {
        fprintf(stderr, "unexpected binding result sizes for QAbc solve\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    x = hprqp_results_x_data(results);
    if (x == NULL) {
        fprintf(stderr, "missing x solution vector\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (!nearly_equal(x[0], 2.0) || !nearly_equal(x[1], 1.0)) {
        fprintf(stderr, "unexpected solution from binding API: [%g, %g]\n", x[0], x[1]);
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (!nearly_equal(hprqp_results_objective(results), -5.0)) {
        fprintf(stderr, "unexpected objective from binding API: %g\n", hprqp_results_objective(results));
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_results_y_data(results) != NULL) {
        fprintf(stderr, "expected no constraint-dual vector for empty-A model\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    if (hprqp_results_z_data(results) == NULL || hprqp_results_w_data(results) == NULL) {
        fprintf(stderr, "expected primal-side z and w vectors for empty-A model\n");
        hprqp_results_destroy(results);
        hprqp_params_destroy(params);
        hprqp_model_destroy(model);
        return 1;
    }

    hprqp_results_destroy(results);
    hprqp_params_destroy(params);
    hprqp_model_destroy(model);
    return 0;
}

int main(void)
{
    if (test_read_from_mps_and_max_iter_setter() != 0) {
        return 1;
    }
    if (test_build_from_qabc_csr_and_optimize() != 0) {
        return 1;
    }
    return 0;
}
