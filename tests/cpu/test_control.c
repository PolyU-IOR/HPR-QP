#include "test_support.h"

#include <math.h>
#include <stdio.h>

static double diag_sigma_objective(double x,
                                   double a,
                                   double b,
                                   const double *diag_q,
                                   const double *c,
                                   const double *d,
                                   size_t n)
{
    size_t i;
    double dot_value = 0.0;
    for (i = 0; i < n; ++i) {
        dot_value += c[i] * (d[i] / (1.0 + x * diag_q[i]));
    }
    return a * x + b / x + x * x * dot_value;
}

static double diag_sigma_search(double a,
                                double b,
                                const double *diag_q,
                                const double *c,
                                const double *d,
                                size_t n)
{
    double left = 1e-12;
    double right = 1e12;
    double phi = (sqrt(5.0) - 1.0) / 2.0;
    double x1 = right - phi * (right - left);
    double x2 = left + phi * (right - left);
    double f1 = diag_sigma_objective(x1, a, b, diag_q, c, d, n);
    double f2 = diag_sigma_objective(x2, a, b, diag_q, c, d, n);
    size_t iter = 0;

    while (fabs(right - left) > 1e-12 * fmax(1.0, fabs(left)) && iter < 200) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + phi * (right - left);
            f2 = diag_sigma_objective(x2, a, b, diag_q, c, d, n);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - phi * (right - left);
            f1 = diag_sigma_objective(x1, a, b, diag_q, c, d, n);
        }
        iter += 1;
    }

    return f1 < f2 ? x1 : x2;
}

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

int main(void)
{
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR Q_nd;
    HPRQPMatrixCSR A_nd;
    HPRQPModel model;
    HPRQPModel model_nd;
    HPRQPWorkspace workspace;
    HPRQPWorkspace workspace_nd;
    HPRQPRestart restart_info;
    HPRQPRestart restart_info_nd;
    HPRQPResiduals residuals;
    HPRQPParams params;
    HPRQPStatus status;
    double old_sigma;
    double old_sigma_nd;
    double m_norm;
    double a_diag;
    double b_diag;
    double dqw0;
    double dqw1;
    double expected_sigma_diag;

    model_init(&model);
    model_init(&model_nd);
    workspace_init(&workspace);
    workspace_init(&workspace_nd);
    restart_init(&restart_info);
    restart_init(&restart_info_nd);
    residuals_init(&residuals);
    test_params_set_no_scaling(&params);

    status = test_build_model_case(HPRQP_TEST_CASE_STANDARD, &Q, &A, &model);
    if (status != HPRQP_STATUS_OK) {
        return 1;
    }

    status = workspace_allocate(&model, &params, &workspace);
    if (status != HPRQP_STATUS_OK) {
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    status = iteration_step(&workspace, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "iteration step failed\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    status = test_build_model_case(HPRQP_TEST_CASE_CONTROL_NONDIAG, &Q_nd, &A_nd, &model_nd);
    if (status != HPRQP_STATUS_OK) {
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    status = workspace_allocate(&model_nd, &params, &workspace_nd);
    if (status != HPRQP_STATUS_OK) {
        model_free(&model_nd);
        matrix_csr_free(&A_nd);
        matrix_csr_free(&Q_nd);
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    status = iteration_step(&workspace_nd, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "non-diagonal iteration step failed\n");
        workspace_free(&workspace_nd);
        model_free(&model_nd);
        matrix_csr_free(&A_nd);
        matrix_csr_free(&Q_nd);
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    m_norm = compute_m_norm(&workspace);
    if (!(m_norm > 0.0) || !isfinite(m_norm)) {
        fprintf(stderr, "M norm is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    restart_info_nd.first_restart = 0;
    restart_info_nd.inner = 0;
    restart_info_nd.last_gap = 10.0;
    restart_info_nd.current_gap = 1.0;
    restart_info_nd.save_gap = 2.0;
    restart_info_nd.best_gap = 1.0;
    restart_info_nd.best_sigma = workspace_nd.sigma;
    restart_info_nd.weighted_norm = 1.0;
    restart_info_nd.restart_flag = 1;
    residuals.err_Rd_org_bar = 0.15;
    residuals.err_Rp_org_bar = 0.1;
    residuals.rel_gap_bar = 0.05;
    old_sigma_nd = workspace_nd.sigma;
    status = sigma_update(&params, &restart_info_nd, &workspace_nd, &residuals);
    if (status != HPRQP_STATUS_OK || !(workspace_nd.sigma > 0.0) || !isfinite(workspace_nd.sigma) ||
        nearly_equal(workspace_nd.sigma, old_sigma_nd)) {
        fprintf(stderr, "non-diagonal sigma update is wrong\n");
        workspace_free(&workspace_nd);
        model_free(&model_nd);
        matrix_csr_free(&A_nd);
        matrix_csr_free(&Q_nd);
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    restart_init(&restart_info);
    restart_info.current_gap = m_norm;
    restart_check(&restart_info, 1, 1, 2.5);
    if (!restart_info.restart_flag || restart_info.first_restart ||
        !nearly_equal(restart_info.weighted_norm, m_norm) || !nearly_equal(restart_info.best_gap, m_norm) ||
        !nearly_equal(restart_info.best_sigma, 2.5)) {
        fprintf(stderr, "first restart initialization is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    restart_info.first_restart = 0;
    restart_info.inner = 0;
    restart_info.last_gap = 10.0 * m_norm;
    restart_info.current_gap = m_norm;
    restart_info.save_gap = 2.0 * m_norm;
    restart_info.best_gap = m_norm;
    restart_info.best_sigma = workspace.sigma;
    restart_info.weighted_norm = m_norm;
    restart_check(&restart_info, 1, 1, workspace.sigma);
    if (restart_info.restart_flag != 1) {
        fprintf(stderr, "restart check did not trigger sufficient restart\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    residuals.err_Rd_org_bar = 0.2;
    residuals.err_Rp_org_bar = 0.1;
    residuals.rel_gap_bar = 0.05;
    old_sigma = workspace.sigma;
    restart_info.best_gap = INFINITY;
    restart_info.current_gap = 1.0;

    dqw0 = (workspace.w_bar[0] - workspace.last_w[0]) * workspace.diag_Q[0];
    dqw1 = (workspace.w_bar[1] - workspace.last_w[1]) * workspace.diag_Q[1];
    a_diag = workspace.lambda_max_A * (workspace.y_bar[0] - workspace.last_y[0]) * (workspace.y_bar[0] - workspace.last_y[0]);
    a_diag -= 2.0 * (dqw0 * workspace.ATdy[0] + dqw1 * workspace.ATdy[1]);
    a_diag += dqw0 * dqw0 + dqw1 * dqw1;
    b_diag = (workspace.x_bar[0] - workspace.last_x[0]) * (workspace.x_bar[0] - workspace.last_x[0]) +
             (workspace.x_bar[1] - workspace.last_x[1]) * (workspace.x_bar[1] - workspace.last_x[1]);
    a_diag = a_diag > 1e-12 ? a_diag : 1e-12;
    b_diag = b_diag > 1e-12 ? b_diag : 1e-12;
    expected_sigma_diag = diag_sigma_search(a_diag,
                                            b_diag,
                                            workspace.diag_Q,
                                            workspace.ATdy,
                                            workspace.QATdy,
                                            workspace.n);
    status = sigma_update(&params, &restart_info, &workspace, &residuals);
    if (status != HPRQP_STATUS_OK || !(workspace.sigma > 0.0) || nearly_equal(workspace.sigma, old_sigma) ||
        fabs(workspace.sigma - expected_sigma_diag) > 1e-8) {
        fprintf(stderr, "sigma update is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    restart_info.restart_flag = 1;
    restart_info.inner = 0;
    restart_info.last_gap = 0.0;
    restart_info.current_gap = 0.0;
    restart_post_iteration(&restart_info, 0, 1, compute_m_norm(&workspace));
    restart_info.inner += 1;
    if (!(restart_info.last_gap > 0.0) || !(restart_info.current_gap > 0.0) || restart_info.inner != 1) {
        fprintf(stderr, "post-iteration restart bookkeeping is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    restart_info.restart_flag = 0;
    restart_info.inner = 0;
    restart_info.last_gap = 123.0;
    restart_info.current_gap = 456.0;
    restart_post_iteration(&restart_info, 0, 5, 0.0);
    restart_info.inner += 1;
    if (!nearly_equal(restart_info.last_gap, 123.0) || !nearly_equal(restart_info.current_gap, 456.0) || restart_info.inner != 1) {
        fprintf(stderr, "post-iteration no-op bookkeeping is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    restart_info.restart_flag = 1;
    restart_info.last_gap = 2.0;
    restart_info.current_gap = 9.0;
    status = restart_post_metric_socp(&restart_info,
                                      &workspace,
                                      0,
                                      5,
                                      3.0);
    if (status != HPRQP_STATUS_OK ||
        !nearly_equal(restart_info.last_gap, 2.0) ||
        !nearly_equal(restart_info.current_gap, 9.0)) {
        fprintf(stderr, "SOCP restart bookkeeping did not retain the smaller restart gap\n");
        workspace_free(&workspace);
        workspace_free(&workspace_nd);
        model_free(&model);
        model_free(&model_nd);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        matrix_csr_free(&Q_nd);
        matrix_csr_free(&A_nd);
        return 1;
    }

    status = restart_post_metric_socp(&restart_info,
                                      &workspace,
                                      4,
                                      5,
                                      1.5);
    if (status != HPRQP_STATUS_OK ||
        !nearly_equal(restart_info.last_gap, 1.5) ||
        !nearly_equal(restart_info.current_gap, 1.5)) {
        fprintf(stderr, "SOCP restart bookkeeping did not retain the smaller new gap\n");
        workspace_free(&workspace);
        workspace_free(&workspace_nd);
        model_free(&model);
        model_free(&model_nd);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        matrix_csr_free(&Q_nd);
        matrix_csr_free(&A_nd);
        return 1;
    }

    restart_info.restart_flag = 1;
    restart_info.inner = 1;

    workspace.noC = 1;
    workspace.Qw_bar[0] = 7.0;
    workspace.Qw_bar[1] = 11.0;
    workspace.Qw[0] = -1.0;
    workspace.Qw[1] = -2.0;
    workspace.last_Qw[0] = -3.0;
    workspace.last_Qw[1] = -4.0;

    status = restart_apply(&restart_info, &workspace);
    if (status != HPRQP_STATUS_OK || restart_info.inner != 0 || restart_info.times != 1 ||
        !nearly_equal(workspace.x[0], workspace.x_bar[0]) || !nearly_equal(workspace.w[1], workspace.w_bar[1]) ||
        !nearly_equal(workspace.y[0], workspace.y_bar[0]) ||
        !nearly_equal(workspace.Qw[0], workspace.Qw_bar[0]) || !nearly_equal(workspace.Qw[1], workspace.Qw_bar[1]) ||
        !nearly_equal(workspace.last_Qw[0], workspace.Qw_bar[0]) || !nearly_equal(workspace.last_Qw[1], workspace.Qw_bar[1])) {
        fprintf(stderr, "restart apply is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    workspace_free(&workspace);
    workspace_free(&workspace_nd);
    model_free(&model);
    model_free(&model_nd);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    matrix_csr_free(&Q_nd);
    matrix_csr_free(&A_nd);
    printf("test_control passed\n");
    return 0;
}
