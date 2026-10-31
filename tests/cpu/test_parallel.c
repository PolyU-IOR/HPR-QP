#include "test_support.h"

#include "shared/parallel.h"

#include <math.h>
#include <string.h>

static double project_interval(double value, double lower, double upper)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPModel model;
    HPRQPWorkspace workspace;
    HPRQPParams params;
    HPRQPResiduals residuals;
    HPRQPRestart restart_info;
    HPRQPResults results;
    double x0[3] = {0.20, -0.10, 0.30};
    double y0[2] = {0.15, -0.25};
    double w0[3] = {0.40, 1.70, -0.20};
    double anchor_x[3] = {-0.15, 0.05, 0.20};
    double anchor_y[2] = {0.05, 0.10};
    double anchor_w[3] = {0.10, -2.00, 0.30};
    double expected_qw[3];
    double expected_aty[3];
    double expected_x_bar[3];
    double expected_x_hat[3];
    double expected_w_bar[3];
    double expected_y_bar[2];
    double expected_ax[2];
    double expected_qw_bar[3];
    double expected_aty_bar[3];
    double expected_metric_sq;
    double metric_before;
    double expected_tau;
    size_t i;
    const double alpha = 0.2;
    const double beta = 0.8;

    model_init(&model);
    workspace_init(&workspace);
    results_init(&results);
    residuals_init(&residuals);
    test_params_set_no_scaling(&params);
    params.method = HPRQP_METHOD_PARALLEL;
    params.parallel_majorizer_mode = HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER;
    params.tau_mode = HPRQP_TAU_MODE_FIXED;
    params.tau = 1.6;
    params.use_gpu = 0;
    params.verbose = 0;

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_PARALLEL_SINGULAR,
                                         &Q,
                                         &A,
                                         &model));
    TEST_ASSERT_OK(workspace_allocate(&model, &params, &workspace));
    TEST_ASSERT(workspace.lambda_max_AQA >= 0.0 &&
                    workspace.lambda_max_AQA <=
                        workspace.lambda_max_A * workspace.lambda_max_Q,
                "invalid AQA spectral estimate");
    TEST_ASSERT(workspace.zeta_AQ >= 0.0 && workspace.zeta_AQ <= 1.0,
                "zeta_AQ is outside [0,1]");

    workspace.sigma = 0.7;
    workspace.lambda_max_A = 3.15;
    workspace.lambda_max_Q = 2.1;
    workspace.lambda_max_AQA = workspace.lambda_max_A *
                               workspace.lambda_max_Q;
    workspace.lambda_max_AQ2A = workspace.lambda_max_Q *
                                workspace.lambda_max_AQA;
    workspace.zeta_AQ = 1.0;
    workspace.tau = 1.6;
    workspace.parallel_majorizer_mode =
        HPRQP_PARALLEL_MAJORIZER_WORST_CASE;
    TEST_ASSERT_OK(parallel_compute_coefficients(
        workspace.m,
        workspace.sigma,
        workspace.lambda_max_A,
        workspace.lambda_max_Q,
        workspace.lambda_max_AQA,
        workspace.lambda_max_AQ2A,
        workspace.zeta_AQ,
        workspace.tau,
        workspace.parallel_majorizer_mode,
        &workspace.d_y,
        &workspace.d_w,
        &workspace.parallel_y_rho,
        &workspace.parallel_y_inv,
        &workspace.parallel_w_gamma,
        &workspace.parallel_w_inv));

    memcpy(workspace.x, x0, sizeof(x0));
    memcpy(workspace.y, y0, sizeof(y0));
    memcpy(workspace.w, w0, sizeof(w0));
    memcpy(workspace.last_x, anchor_x, sizeof(anchor_x));
    memcpy(workspace.last_y, anchor_y, sizeof(anchor_y));
    memcpy(workspace.last_w, anchor_w, sizeof(anchor_w));
    workspace.to_check = 1;

    TEST_ASSERT_OK(matrix_csr_matvec(&model.Q, w0, expected_qw));
    TEST_ASSERT_OK(matrix_csr_matvec(&model.AT, y0, expected_aty));
    for (i = 0; i < 3; ++i) {
        double z_raw = x0[i] + workspace.sigma *
                                   (-expected_qw[i] + expected_aty[i] -
                                    model.c[i]);
        expected_x_bar[i] = project_interval(z_raw,
                                             model.l[i],
                                             model.u[i]);
        expected_x_hat[i] = 2.0 * expected_x_bar[i] - x0[i];
        expected_w_bar[i] =
            (workspace.parallel_w_gamma * w0[i] + expected_x_hat[i]) *
            workspace.parallel_w_inv;
    }
    TEST_ASSERT_OK(matrix_csr_matvec(&model.A,
                                     expected_x_hat,
                                     expected_ax));
    for (i = 0; i < 2; ++i) {
        double s_raw = expected_ax[i] - workspace.parallel_y_rho * y0[i];
        double s_proj = project_interval(s_raw, model.AL[i], model.AU[i]);
        expected_y_bar[i] = (s_proj - s_raw) * workspace.parallel_y_inv;
    }
    TEST_ASSERT_OK(matrix_csr_matvec(&model.Q,
                                     expected_w_bar,
                                     expected_qw_bar));
    TEST_ASSERT_OK(matrix_csr_matvec(&model.AT,
                                     expected_y_bar,
                                     expected_aty_bar));

    TEST_ASSERT_OK(iteration_step_parallel(&workspace, 3));
    for (i = 0; i < 3; ++i) {
        TEST_ASSERT(test_nearly_equal_rel(workspace.x_bar[i],
                                          expected_x_bar[i],
                                          2e-14),
                    "parallel x_bar mismatch at %zu", i);
        TEST_ASSERT(test_nearly_equal_rel(workspace.w_bar[i],
                                          expected_w_bar[i],
                                          2e-14),
                    "parallel w_bar mismatch at %zu", i);
        TEST_ASSERT(test_nearly_equal_rel(workspace.x[i],
                                          alpha * anchor_x[i] +
                                              beta * expected_x_hat[i],
                                          2e-14),
                    "parallel Halpern x mismatch at %zu", i);
        TEST_ASSERT(test_nearly_equal_rel(
                        workspace.w[i],
                        alpha * anchor_w[i] +
                            beta * (2.0 * expected_w_bar[i] - w0[i]),
                        2e-14),
                    "parallel Halpern w mismatch at %zu", i);
        TEST_ASSERT(test_nearly_equal_rel(workspace.Qw_bar[i],
                                          expected_qw_bar[i],
                                          2e-14),
                    "parallel Qw_bar mismatch at %zu", i);
        TEST_ASSERT(test_nearly_equal_rel(workspace.ATy_bar[i],
                                          expected_aty_bar[i],
                                          2e-14),
                    "parallel ATy_bar mismatch at %zu", i);
    }
    for (i = 0; i < 2; ++i) {
        TEST_ASSERT(test_nearly_equal_rel(workspace.y_bar[i],
                                          expected_y_bar[i],
                                          2e-14),
                    "parallel y_bar mismatch at %zu", i);
        TEST_ASSERT(test_nearly_equal_rel(
                        workspace.y[i],
                        alpha * anchor_y[i] +
                            beta * (2.0 * expected_y_bar[i] - y0[i]),
                        2e-14),
                    "parallel Halpern y mismatch at %zu", i);
    }

    expected_metric_sq =
        workspace.sigma * workspace.d_y * workspace.lambda_max_A *
            vec_dot(workspace.dy, workspace.dy, workspace.m) +
        workspace.sigma * workspace.d_w * workspace.lambda_max_Q *
            vec_dot(workspace.dw, workspace.dQw, workspace.n) +
        2.0 * (vec_dot(workspace.ATdy, workspace.dx, workspace.n) -
               vec_dot(workspace.dQw, workspace.dx, workspace.n)) +
        vec_dot(workspace.dx, workspace.dx, workspace.n) /
            workspace.sigma;
    metric_before = compute_m_norm_parallel(&workspace);
    TEST_ASSERT(test_nearly_equal_rel(metric_before,
                                      sqrt(expected_metric_sq),
                                      2e-14),
                "parallel M metric mismatch");
    workspace.dw[1] += 11.0;
    TEST_ASSERT_OK(matrix_csr_matvec(&model.Q,
                                     workspace.dw,
                                     workspace.dQw));
    TEST_ASSERT(test_nearly_equal_rel(compute_m_norm_parallel(&workspace),
                                      metric_before,
                                      2e-14),
                "singular-Q shadow observable changed in Ker(Q)");

    params.tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
    params.sigma_fixed = 1;
    workspace.parallel_majorizer_mode =
        HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER;
    workspace.zeta_AQ = 0.25;
    workspace.tau = 1.6;
    workspace.dy[0] = 3.0;
    workspace.dy[1] = 4.0;
    workspace.dw[0] = 2.0;
    workspace.dw[1] = 0.0;
    workspace.dw[2] = -1.0;
    TEST_ASSERT_OK(matrix_csr_matvec(&model.Q,
                                     workspace.dw,
                                     workspace.dQw));
    expected_tau = sqrt(
        workspace.lambda_max_Q *
            vec_dot(workspace.dw, workspace.dQw, workspace.n) /
        (workspace.lambda_max_A *
         vec_dot(workspace.dy, workspace.dy, workspace.m)));
    restart_init(&restart_info);
    restart_info.restart_flag = 1;
    TEST_ASSERT_OK(parallel_update_metric(&params,
                                          &restart_info,
                                          &workspace,
                                          &residuals));
    TEST_ASSERT(test_nearly_equal_rel(workspace.tau,
                                      expected_tau,
                                      2e-14),
                "adaptive tau mismatch");
    TEST_ASSERT(restart_info.metric_reseed_pending == 1,
                "coupling metric change did not request a reseed");

    workspace_free(&workspace);
    workspace_init(&workspace);
    params.sigma_fixed = 0;
    params.max_iter = 20000;
    params.check_iter = 10;
    params.stoptol = 1e-7;
    params.tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
    TEST_ASSERT(solve(&model, &params, &results) == HPRQP_STATUS_OPTIMAL,
                "parallel solver did not converge on the singular-Q case");
    TEST_ASSERT(results.residuals < params.stoptol,
                "parallel solver returned an invalid residual");
    TEST_ASSERT(results.zeta_AQ >= 0.0 && results.zeta_AQ <= 1.0,
                "parallel result omitted zeta_AQ");

cleanup:
    results_free(&results);
    workspace_free(&workspace);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    if (exit_code == 0) {
        printf("test_parallel passed\n");
    }
    return exit_code;
}
