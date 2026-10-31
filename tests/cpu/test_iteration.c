#include "test_support.h"

#include <math.h>
#include <stdio.h>

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

int main(void)
{
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR Q_free;
    HPRQPMatrixCSR A_free;
    HPRQPModel model;
    HPRQPModel free_model;
    HPRQPWorkspace workspace;
    HPRQPParams params;
    HPRQPStatus status;
    double expected_sigma;
    double expected_y_bar;
    double expected_w_bar0;
    double expected_w_bar1;
    double expected_x_bar0_free;
    double expected_x_bar1_free;
    double expected_w_bar0_free;
    double expected_w_bar1_free;
    double expected_qw0_free;
    double expected_qw1_free;
    double expected_tempv0_free;
    double expected_tempv1_free;
    double expected_y_bar_free;
    model_init(&model);
    model_init(&free_model);
    workspace_init(&workspace);
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

    expected_sigma = 2.0 / sqrt(5.0);
    expected_y_bar = 2.0 / (2.1 * expected_sigma);
    expected_w_bar0 = (expected_sigma / (1.0 + 2.0 * expected_sigma)) * expected_y_bar;
    expected_w_bar1 = (expected_sigma / (1.0 + 4.0 * expected_sigma)) * expected_y_bar;

    status = iteration_step(&workspace, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "iteration step failed\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.x_bar[0], 0.0) || !nearly_equal(workspace.x_bar[1], 0.0) ||
        !nearly_equal(workspace.z_bar[0], 2.0) || !nearly_equal(workspace.z_bar[1], 1.0)) {
        fprintf(stderr, "x/z update is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.y_bar[0], expected_y_bar) || !nearly_equal(workspace.y[0], expected_y_bar) || !nearly_equal(workspace.s[0], 2.0)) {
        fprintf(stderr, "y update is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.w_bar[0], expected_w_bar0) || !nearly_equal(workspace.w_bar[1], expected_w_bar1) ||
        !nearly_equal(workspace.w[0], expected_w_bar0) || !nearly_equal(workspace.w[1], expected_w_bar1)) {
        fprintf(stderr, "w update is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.ATy[0], expected_y_bar) || !nearly_equal(workspace.ATy[1], expected_y_bar) ||
        !nearly_equal(workspace.ATdy[0], expected_y_bar) || !nearly_equal(workspace.ATdy[1], expected_y_bar)) {
        fprintf(stderr, "ATy update is wrong\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    workspace_free(&workspace);
    workspace_init(&workspace);
    status = workspace_allocate(&model, &params, &workspace);
    if (status != HPRQP_STATUS_OK) {
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    workspace.to_check = 0;
    workspace.z_bar[0] = -9.0;
    workspace.z_bar[1] = -8.0;
    workspace.s[0] = -6.0;
    workspace.dx[0] = -7.0;
    workspace.dx[1] = -7.0;
    workspace.dy[0] = -7.0;
    workspace.dw[0] = -7.0;
    workspace.dw[1] = -7.0;
    workspace.ATdy[0] = -7.0;
    workspace.ATdy[1] = -7.0;

    status = iteration_step(&workspace, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "iteration step failed in no-check mode\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.dx[0], -7.0) || !nearly_equal(workspace.dy[0], -7.0) ||
        !nearly_equal(workspace.dw[0], -7.0) || !nearly_equal(workspace.ATdy[0], -7.0) ||
        !nearly_equal(workspace.z_bar[0], -9.0) || !nearly_equal(workspace.z_bar[1], -8.0) ||
        !nearly_equal(workspace.s[0], -6.0)) {
        fprintf(stderr, "no-check mode should preserve diagnostic deltas\n");
        workspace_free(&workspace);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    workspace_free(&workspace);
    workspace_init(&workspace);

    status = test_build_model_case(HPRQP_TEST_CASE_STANDARD_NOC, &Q_free, &A_free, &free_model);
    if (status != HPRQP_STATUS_OK) {
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    status = workspace_allocate(&free_model, &params, &workspace);
    if (status != HPRQP_STATUS_OK) {
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!workspace.noC) {
        fprintf(stderr, "noC detection is wrong\n");
        workspace_free(&workspace);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    expected_x_bar0_free = -2.0 * expected_sigma;
    expected_x_bar1_free = -expected_sigma;
    expected_w_bar0_free = -(4.0 * expected_sigma) / (1.0 + 2.0 * expected_sigma);
    expected_w_bar1_free = -(2.0 * expected_sigma) / (1.0 + 4.0 * expected_sigma);
    expected_qw0_free = 2.0 * expected_w_bar0_free;
    expected_qw1_free = 4.0 * expected_w_bar1_free;
    expected_tempv0_free = -4.0 * expected_sigma - expected_sigma * expected_qw0_free;
    expected_tempv1_free = -2.0 * expected_sigma - expected_sigma * expected_qw1_free;
    expected_y_bar_free = (2.0 - (expected_tempv0_free + expected_tempv1_free)) / (2.1 * expected_sigma);

    status = iteration_step(&workspace, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "iteration step failed in noC mode\n");
        workspace_free(&workspace);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.x_bar[0], expected_x_bar0_free) || !nearly_equal(workspace.x_bar[1], expected_x_bar1_free) ||
        !nearly_equal(workspace.z_bar[0], 0.0) || !nearly_equal(workspace.z_bar[1], 0.0)) {
        fprintf(stderr, "noC x/z update is wrong\n");
        workspace_free(&workspace);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.w_bar[0], expected_w_bar0_free) || !nearly_equal(workspace.w_bar[1], expected_w_bar1_free) ||
        !nearly_equal(workspace.w[0], expected_w_bar0_free) || !nearly_equal(workspace.w[1], expected_w_bar1_free) ||
        !nearly_equal(workspace.Qw[0], expected_qw0_free) || !nearly_equal(workspace.Qw[1], expected_qw1_free) ||
        !nearly_equal(workspace.Qw_bar[0], expected_qw0_free) || !nearly_equal(workspace.Qw_bar[1], expected_qw1_free)) {
        fprintf(stderr, "noC w/Qw update is wrong\n");
        workspace_free(&workspace);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.tempv[0], expected_tempv0_free) || !nearly_equal(workspace.tempv[1], expected_tempv1_free) ||
        !nearly_equal(workspace.y_bar[0], expected_y_bar_free) || !nearly_equal(workspace.y[0], expected_y_bar_free) ||
        !nearly_equal(workspace.s[0], 2.0)) {
        fprintf(stderr, "noC tempv/y update is wrong\n");
        workspace_free(&workspace);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(workspace.ATy[0], 0.0) || !nearly_equal(workspace.ATy[1], 0.0) ||
        !nearly_equal(workspace.dy[0], expected_y_bar_free) || !nearly_equal(workspace.dw[0], expected_w_bar0_free) ||
        !nearly_equal(workspace.dw[1], expected_w_bar1_free)) {
        fprintf(stderr, "noC diagnostic state is wrong\n");
        workspace_free(&workspace);
        model_free(&free_model);
        matrix_csr_free(&A_free);
        matrix_csr_free(&Q_free);
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    workspace_free(&workspace);
    model_free(&free_model);
    matrix_csr_free(&Q_free);
    matrix_csr_free(&A_free);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    printf("test_iteration passed\n");
    return 0;
}