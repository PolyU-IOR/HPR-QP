#include "test_support.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

typedef struct ExpectedSnapshot {
    size_t iter;
    double residual;
    double gap;
    double primal_obj;
    double x[2];
    double y[1];
    double z[2];
    double w[2];
} ExpectedSnapshot;

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

static size_t expected_adaptive_print_step(size_t iter)
{
    size_t step = 10;

    if (iter == 0) {
        return step;
    }

    while (iter / step >= 100 && step <= SIZE_MAX / 10) {
        step *= 10;
    }

    return step;
}

static int expected_should_check_next_iter(const HPRQPParams *params,
                                           const HPRQPRestart *restart_info,
                                           size_t next_iter)
{
    int to_check = 0;

    if (params->check_iter > 0 && next_iter % params->check_iter == 0) {
        to_check = 1;
    }
    if (restart_info->restart_flag > 0) {
        to_check = 1;
    }
    if (params->print_frequency == -1 && next_iter % expected_adaptive_print_step(next_iter) == 0) {
        to_check = 1;
    }
    if (params->print_frequency > 0 && next_iter % (size_t)params->print_frequency == 0) {
        to_check = 1;
    }

    return to_check;
}

static int expected_should_print_current_iter(const HPRQPParams *params,
                                              size_t iter,
                                              size_t max_iter)
{
    if (params->print_frequency == -1 &&
        (iter % expected_adaptive_print_step(iter) == 0 || iter == max_iter)) {
        return 1;
    }
    if (params->print_frequency > 0 &&
        (iter % (size_t)params->print_frequency == 0 || iter == max_iter)) {
        return 1;
    }

    return 0;
}

static int compute_expected_reported_state(const HPRQPModel *model,
                                           const HPRQPParams *params,
                                           size_t max_iter,
                                           ExpectedSnapshot *expected)
{
    HPRQPModel working_model;
    HPRQPScaling scaling;
    HPRQPWorkspace workspace;
    HPRQPResiduals residuals;
    HPRQPRestart restart_info;
    HPRQPStatus status;
    size_t iter;
    int print_yes;

    if (model == NULL || params == NULL || expected == NULL) {
        return 0;
    }

    model_init(&working_model);
    scaling_init(&scaling);
    workspace_init(&workspace);
    residuals_init(&residuals);
    restart_init(&restart_info);

    status = model_copy(&working_model, model);
    if (status != HPRQP_STATUS_OK) {
        return 0;
    }
    status = scale_problem(&working_model, params, &scaling);
    if (status != HPRQP_STATUS_OK) {
        model_free(&working_model);
        return 0;
    }
    status = workspace_allocate(&working_model, params, &workspace);
    if (status != HPRQP_STATUS_OK) {
        scaling_free(&scaling);
        model_free(&working_model);
        return 0;
    }
    status = residuals_compute(&workspace, &working_model, &scaling, &residuals, 0);
    if (status != HPRQP_STATUS_OK) {
        workspace_free(&workspace);
        scaling_free(&scaling);
        model_free(&working_model);
        return 0;
    }

    expected->iter = 0;
    expected->residual = residuals.KKTx_and_gap_org_bar;
    expected->gap = residuals.rel_gap_bar;
    expected->primal_obj = residuals.primal_obj_bar;
    expected->x[0] = workspace.x_bar[0];
    expected->x[1] = workspace.x_bar[1];
    expected->y[0] = workspace.y_bar[0];
    expected->z[0] = workspace.z_bar[0];
    expected->z[1] = workspace.z_bar[1];
    expected->w[0] = workspace.w_bar[0];
    expected->w[1] = workspace.w_bar[1];
    restart_info.best_sigma = workspace.sigma;

    for (iter = 0;; ++iter) {
        print_yes = 0;

        if (iter > 0) {
            print_yes = expected_should_print_current_iter(params, iter, max_iter);
            if ((params->check_iter > 0 && iter % params->check_iter == 0) || print_yes) {
                status = residuals_compute(&workspace, &working_model, &scaling, &residuals, iter);
                if (status != HPRQP_STATUS_OK) {
                    workspace_free(&workspace);
                    scaling_free(&scaling);
                    model_free(&working_model);
                    return 0;
                }

                expected->residual = residuals.KKTx_and_gap_org_bar;
                expected->gap = residuals.rel_gap_bar;
                expected->primal_obj = residuals.primal_obj_bar;
            }
        }

        if (residuals.KKTx_and_gap_org_bar < params->stoptol || iter == max_iter) {
            break;
        }

        restart_check(&restart_info, iter, params->check_iter, workspace.sigma);
        status = sigma_update(params, &restart_info, &workspace, &residuals);
        if (status != HPRQP_STATUS_OK) {
            workspace_free(&workspace);
            scaling_free(&scaling);
            model_free(&working_model);
            return 0;
        }
        status = restart_apply(&restart_info, &workspace);
        if (status != HPRQP_STATUS_OK) {
            workspace_free(&workspace);
            scaling_free(&scaling);
            model_free(&working_model);
            return 0;
        }

        workspace.to_check = expected_should_check_next_iter(params, &restart_info, iter + 1);
        status = iteration_step(&workspace, restart_info.inner);
        if (status != HPRQP_STATUS_OK) {
            workspace_free(&workspace);
            scaling_free(&scaling);
            model_free(&working_model);
            return 0;
        }
        if (restart_info.restart_flag > 0 ||
            (params->check_iter > 0 && (iter + 1) % params->check_iter == 0)) {
            restart_post_iteration(&restart_info,
                                   iter,
                                   params->check_iter,
                                   compute_m_norm(&workspace));
        }
        restart_info.inner += 1;
    }

    expected->iter = iter;
    expected->x[0] = workspace.x_bar[0];
    expected->x[1] = workspace.x_bar[1];
    expected->y[0] = workspace.y_bar[0];
    expected->z[0] = workspace.z_bar[0];
    expected->z[1] = workspace.z_bar[1];
    expected->w[0] = workspace.w_bar[0];
    expected->w[1] = workspace.w_bar[1];

    workspace_free(&workspace);
    scaling_free(&scaling);
    model_free(&working_model);
    return 1;
}

int main(void)
{
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR Q_free;
    HPRQPMatrixCSR A_free;
    HPRQPMatrixCSR Q_zero;
    HPRQPMatrixCSR A_zero;
    HPRQPModel model;
    HPRQPModel free_model;
    HPRQPModel zero_model;
    HPRQPParams params;
    HPRQPResults results;
    HPRQPStatus status;
    ExpectedSnapshot expected;
    model_init(&model);
    model_init(&free_model);
    model_init(&zero_model);
    results_init(&results);
    test_params_set_no_scaling(&params);
    params.use_gpu = 0;
    params.max_iter = 1;

    status = test_build_model_case(HPRQP_TEST_CASE_STANDARD, &Q, &A, &model);
    if (status != HPRQP_STATUS_OK) {
        return 1;
    }

    status = solve(&model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER) {
        fprintf(stderr, "solve should stop on max iter\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    if (!compute_expected_reported_state(&model, &params, 1, &expected) ||
        results.status != HPRQP_STATUS_MAX_ITER || results.iter != expected.iter) {
        fprintf(stderr, "solver status metadata are wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    if (!nearly_equal(results.primal_obj, expected.primal_obj) || !nearly_equal(results.gap, expected.gap) ||
        !nearly_equal(results.residuals, expected.residual)) {
        fprintf(stderr, "solver diagnostics are wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    if (results.x == NULL || results.y == NULL || results.z == NULL || results.w == NULL ||
        !nearly_equal(results.x[0], expected.x[0]) || !nearly_equal(results.x[1], expected.x[1]) ||
        !nearly_equal(results.y[0], expected.y[0]) || !nearly_equal(results.z[0], expected.z[0]) ||
        !nearly_equal(results.z[1], expected.z[1]) || !nearly_equal(results.w[0], expected.w[0]) ||
        !nearly_equal(results.w[1], expected.w[1])) {
        fprintf(stderr, "solver iterate collection is wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    if (results.iter_4 != results.iter || results.iter_6 != results.iter ||
        !nearly_equal(results.time_4, results.time) || !nearly_equal(results.time_6, results.time)) {
        fprintf(stderr, "solver milestone defaults are wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.max_iter = 10;
    params.stoptol = 1.0;

    status = solve(&model, &params, &results);
    if (status != HPRQP_STATUS_OPTIMAL || results.status != HPRQP_STATUS_OPTIMAL || results.iter != 0) {
        fprintf(stderr, "solver should stop immediately on loose tolerance\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.stoptol = 0.0;
    params.max_iter = 2;
    params.check_iter = 1;

    status = solve(&model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER || results.status != HPRQP_STATUS_MAX_ITER ||
        !isfinite(results.residuals) || !isfinite(results.gap) || results.x == NULL || results.y == NULL) {
        fprintf(stderr, "solver multi-iteration control path is wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    if (!compute_expected_reported_state(&model, &params, 2, &expected) || results.iter != expected.iter ||
        !nearly_equal(results.residuals, expected.residual) || !nearly_equal(results.gap, expected.gap) ||
        !nearly_equal(results.primal_obj, expected.primal_obj) || !nearly_equal(results.x[0], expected.x[0]) ||
        !nearly_equal(results.x[1], expected.x[1]) || !nearly_equal(results.y[0], expected.y[0]) ||
        !nearly_equal(results.z[0], expected.z[0]) || !nearly_equal(results.z[1], expected.z[1]) ||
        !nearly_equal(results.w[0], expected.w[0]) || !nearly_equal(results.w[1], expected.w[1])) {
        fprintf(stderr, "solver reported iteration state is wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.max_iter = 4;
    params.check_iter = 3;
    params.print_frequency = 0;

    status = solve(&model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER || results.status != HPRQP_STATUS_MAX_ITER ||
        !compute_expected_reported_state(&model, &params, 4, &expected) || results.iter != expected.iter ||
        !nearly_equal(results.residuals, expected.residual) || !nearly_equal(results.gap, expected.gap) ||
        !nearly_equal(results.primal_obj, expected.primal_obj) || !nearly_equal(results.x[0], expected.x[0]) ||
        !nearly_equal(results.x[1], expected.x[1]) || !nearly_equal(results.y[0], expected.y[0]) ||
        !nearly_equal(results.z[0], expected.z[0]) || !nearly_equal(results.z[1], expected.z[1]) ||
        !nearly_equal(results.w[0], expected.w[0]) || !nearly_equal(results.w[1], expected.w[1])) {
        fprintf(stderr, "solver delayed-check scheduling is wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.max_iter = 12;
    params.check_iter = 100;
    params.print_frequency = -1;

    status = solve(&model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER || results.status != HPRQP_STATUS_MAX_ITER ||
        !compute_expected_reported_state(&model, &params, 12, &expected) || results.iter != expected.iter ||
        !nearly_equal(results.residuals, expected.residual) || !nearly_equal(results.gap, expected.gap) ||
        !nearly_equal(results.primal_obj, expected.primal_obj) || !nearly_equal(results.x[0], expected.x[0]) ||
        !nearly_equal(results.x[1], expected.x[1]) || !nearly_equal(results.y[0], expected.y[0]) ||
        !nearly_equal(results.z[0], expected.z[0]) || !nearly_equal(results.z[1], expected.z[1]) ||
        !nearly_equal(results.w[0], expected.w[0]) || !nearly_equal(results.w[1], expected.w[1])) {
        fprintf(stderr, "solver adaptive print scheduling is wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    status = test_build_model_case(HPRQP_TEST_CASE_STANDARD_NOC, &Q_free, &A_free, &free_model);
    if (status != HPRQP_STATUS_OK) {
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.max_iter = 2;
    params.check_iter = 1;
    params.print_frequency = 0;
    params.stoptol = 0.0;

    status = solve(&free_model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER || results.status != HPRQP_STATUS_MAX_ITER ||
        !compute_expected_reported_state(&free_model, &params, 2, &expected) || results.iter != expected.iter ||
        !nearly_equal(results.residuals, expected.residual) || !nearly_equal(results.gap, expected.gap) ||
        !nearly_equal(results.primal_obj, expected.primal_obj) || !nearly_equal(results.x[0], expected.x[0]) ||
        !nearly_equal(results.x[1], expected.x[1]) || !nearly_equal(results.y[0], expected.y[0]) ||
        !nearly_equal(results.z[0], expected.z[0]) || !nearly_equal(results.z[1], expected.z[1]) ||
        !nearly_equal(results.w[0], expected.w[0]) || !nearly_equal(results.w[1], expected.w[1])) {
        fprintf(stderr, "solver noC path is wrong\n");
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.max_iter = 4;
    params.check_iter = 3;
    params.print_frequency = 0;

    status = solve(&free_model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER || results.status != HPRQP_STATUS_MAX_ITER ||
        !compute_expected_reported_state(&free_model, &params, 4, &expected) || results.iter != expected.iter ||
        !nearly_equal(results.residuals, expected.residual) || !nearly_equal(results.gap, expected.gap) ||
        !nearly_equal(results.primal_obj, expected.primal_obj) || !nearly_equal(results.x[0], expected.x[0]) ||
        !nearly_equal(results.x[1], expected.x[1]) || !nearly_equal(results.y[0], expected.y[0]) ||
        !nearly_equal(results.z[0], expected.z[0]) || !nearly_equal(results.z[1], expected.z[1]) ||
        !nearly_equal(results.w[0], expected.w[0]) || !nearly_equal(results.w[1], expected.w[1])) {
        fprintf(stderr, "solver noC delayed-check scheduling is wrong\n");
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    status = test_build_model_case(HPRQP_TEST_CASE_ZERO_1D, &Q_zero, &A_zero, &zero_model);
    if (status != HPRQP_STATUS_OK) {
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    results_init(&results);
    params.max_iter = 5;
    params.check_iter = 1;
    params.stoptol = 1e-8;

    status = solve(&zero_model, &params, &results);
    if (status != HPRQP_STATUS_OPTIMAL || results.status != HPRQP_STATUS_OPTIMAL ||
        results.iter != 0 || results.iter_4 != 0 || results.iter_6 != 0 ||
        results.time_4 < 0.0 || results.time_6 < 0.0 || results.time_4 > results.time || results.time_6 > results.time) {
        fprintf(stderr, "solver milestone tracking is wrong\n");
        model_free(&zero_model);
        matrix_csr_free(&A_zero);
        matrix_csr_free(&Q_zero);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        results_free(&results);
        return 1;
    }

    results_free(&results);
    model_free(&zero_model);
    matrix_csr_free(&Q_zero);
    matrix_csr_free(&A_zero);
    model_free(&free_model);
    matrix_csr_free(&Q_free);
    matrix_csr_free(&A_free);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    printf("test_solver passed\n");
    return 0;
}
