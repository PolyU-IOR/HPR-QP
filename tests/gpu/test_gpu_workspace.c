#include "test_support.h"

#include <string.h>

static void seed_workspace_state(HPRQPWorkspace *workspace)
{
    workspace->sigma = 1.25;
    workspace->lambda_max_A = 2.5;
    workspace->lambda_max_Q = 4.0;
    workspace->Q_is_diag = 1;
    workspace->noC = 0;
    workspace->to_check = 1;

    workspace->x[0] = 0.4;
    workspace->x[1] = -0.2;
    workspace->x_bar[0] = 0.5;
    workspace->x_bar[1] = -0.3;
    workspace->x_hat[0] = 0.7;
    workspace->x_hat[1] = 0.9;
    workspace->z_bar[0] = 0.8;
    workspace->z_bar[1] = 0.6;
    workspace->dx[0] = -0.1;
    workspace->dx[1] = 0.2;
    workspace->w[0] = 0.15;
    workspace->w[1] = -0.35;
    workspace->w_bar[0] = 0.45;
    workspace->w_bar[1] = -0.55;
    workspace->dw[0] = 0.05;
    workspace->dw[1] = 0.08;
    workspace->Qw[0] = 1.1;
    workspace->Qw[1] = -1.2;
    workspace->Qw_bar[0] = 1.3;
    workspace->Qw_bar[1] = -1.4;
    workspace->Qx[0] = 2.1;
    workspace->Qx[1] = 2.2;
    workspace->ATy[0] = -0.4;
    workspace->ATy[1] = 0.6;
    workspace->ATy_bar[0] = -0.7;
    workspace->ATy_bar[1] = 0.9;
    workspace->ATdy[0] = 0.12;
    workspace->ATdy[1] = -0.18;
    workspace->dQw[0] = 0.22;
    workspace->dQw[1] = 0.33;
    workspace->QATdy[0] = -0.27;
    workspace->QATdy[1] = 0.19;
    workspace->tempv[0] = 0.31;
    workspace->tempv[1] = -0.29;
    workspace->last_x[0] = -0.11;
    workspace->last_x[1] = 0.13;
    workspace->last_w[0] = -0.17;
    workspace->last_w[1] = 0.21;
    workspace->last_Qw[0] = 0.41;
    workspace->last_Qw[1] = -0.43;
    workspace->last_ATy[0] = 0.24;
    workspace->last_ATy[1] = -0.26;
    workspace->fact1[0] = 0.51;
    workspace->fact1[1] = 0.52;
    workspace->fact2[0] = 0.61;
    workspace->fact2[1] = 0.62;
    workspace->fact[0] = 0.71;
    workspace->fact[1] = 0.72;
    workspace->fact_M[0] = 0.81;
    workspace->fact_M[1] = 0.82;
    workspace->diag_Q[0] = 2.0;
    workspace->diag_Q[1] = 4.0;

    workspace->y[0] = -0.14;
    workspace->y_bar[0] = 0.23;
    workspace->dy[0] = -0.19;
    workspace->s[0] = 0.44;
    workspace->Rp[0] = 0.54;
    workspace->Ax[0] = -0.64;
    workspace->last_y[0] = 0.74;
}

static int compare_workspace_state(const HPRQPWorkspace *actual, const HPRQPWorkspace *expected)
{
    size_t i;

    if (!test_nearly_equal_abs(actual->sigma, expected->sigma, 1e-12) ||
        !test_nearly_equal_abs(actual->lambda_max_A, expected->lambda_max_A, 1e-12) ||
        !test_nearly_equal_abs(actual->lambda_max_Q, expected->lambda_max_Q, 1e-12) ||
        actual->Q_is_diag != expected->Q_is_diag ||
        actual->noC != expected->noC) {
        fprintf(stderr,
                "workspace scalar mismatch: sigma=%g expected=%g lambdaA=%g expected=%g lambdaQ=%g expected=%g Qdiag=%d expected=%d noC=%d expected=%d\n",
                actual->sigma,
                expected->sigma,
                actual->lambda_max_A,
                expected->lambda_max_A,
                actual->lambda_max_Q,
                expected->lambda_max_Q,
                actual->Q_is_diag,
                expected->Q_is_diag,
                actual->noC,
                expected->noC);
        return 0;
    }

    for (i = 0; i < actual->n; ++i) {
        if (!test_nearly_equal_abs(actual->x[i], expected->x[i], 1e-12) ||
            !test_nearly_equal_abs(actual->x_bar[i], expected->x_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->x_hat[i], expected->x_hat[i], 1e-12) ||
            !test_nearly_equal_abs(actual->z_bar[i], expected->z_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dx[i], expected->dx[i], 1e-12) ||
            !test_nearly_equal_abs(actual->w[i], expected->w[i], 1e-12) ||
            !test_nearly_equal_abs(actual->w_bar[i], expected->w_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dw[i], expected->dw[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Qw[i], expected->Qw[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Qw_bar[i], expected->Qw_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Qx[i], expected->Qx[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATy[i], expected->ATy[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATy_bar[i], expected->ATy_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATdy[i], expected->ATdy[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dQw[i], expected->dQw[i], 1e-12) ||
            !test_nearly_equal_abs(actual->QATdy[i], expected->QATdy[i], 1e-12) ||
            !test_nearly_equal_abs(actual->tempv[i], expected->tempv[i], 1e-12) ||
            !test_nearly_equal_abs(actual->last_x[i], expected->last_x[i], 1e-12) ||
            !test_nearly_equal_abs(actual->last_w[i], expected->last_w[i], 1e-12) ||
            !test_nearly_equal_abs(actual->last_Qw[i], expected->last_Qw[i], 1e-12) ||
            !test_nearly_equal_abs(actual->last_ATy[i], expected->last_ATy[i], 1e-12)) {
            fprintf(stderr,
                    "workspace n-state mismatch at %zu: x=%g expected=%g x_bar=%g expected=%g x_hat=%g expected=%g z_bar=%g expected=%g dx=%g expected=%g w=%g expected=%g w_bar=%g expected=%g dw=%g expected=%g Qw=%g expected=%g Qw_bar=%g expected=%g Qx=%g expected=%g ATy=%g expected=%g ATy_bar=%g expected=%g ATdy=%g expected=%g dQw=%g expected=%g QATdy=%g expected=%g tempv=%g expected=%g last_x=%g expected=%g last_w=%g expected=%g last_Qw=%g expected=%g last_ATy=%g expected=%g\n",
                    i,
                    actual->x[i], expected->x[i],
                    actual->x_bar[i], expected->x_bar[i],
                    actual->x_hat[i], expected->x_hat[i],
                    actual->z_bar[i], expected->z_bar[i],
                    actual->dx[i], expected->dx[i],
                    actual->w[i], expected->w[i],
                    actual->w_bar[i], expected->w_bar[i],
                    actual->dw[i], expected->dw[i],
                    actual->Qw[i], expected->Qw[i],
                    actual->Qw_bar[i], expected->Qw_bar[i],
                    actual->Qx[i], expected->Qx[i],
                    actual->ATy[i], expected->ATy[i],
                    actual->ATy_bar[i], expected->ATy_bar[i],
                    actual->ATdy[i], expected->ATdy[i],
                    actual->dQw[i], expected->dQw[i],
                    actual->QATdy[i], expected->QATdy[i],
                    actual->tempv[i], expected->tempv[i],
                    actual->last_x[i], expected->last_x[i],
                    actual->last_w[i], expected->last_w[i],
                    actual->last_Qw[i], expected->last_Qw[i],
                    actual->last_ATy[i], expected->last_ATy[i]);
            return 0;
        }
    }

    for (i = 0; i < actual->m; ++i) {
        if (!test_nearly_equal_abs(actual->y[i], expected->y[i], 1e-12) ||
            !test_nearly_equal_abs(actual->y_bar[i], expected->y_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dy[i], expected->dy[i], 1e-12) ||
            !test_nearly_equal_abs(actual->s[i], expected->s[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Rp[i], expected->Rp[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Ax[i], expected->Ax[i], 1e-12) ||
            !test_nearly_equal_abs(actual->last_y[i], expected->last_y[i], 1e-12)) {
            fprintf(stderr,
                    "workspace m-state mismatch at %zu: y=%g expected=%g y_bar=%g expected=%g dy=%g expected=%g s=%g expected=%g Rp=%g expected=%g Ax=%g expected=%g last_y=%g expected=%g\n",
                    i,
                    actual->y[i], expected->y[i],
                    actual->y_bar[i], expected->y_bar[i],
                    actual->dy[i], expected->dy[i],
                    actual->s[i], expected->s[i],
                    actual->Rp[i], expected->Rp[i],
                    actual->Ax[i], expected->Ax[i],
                    actual->last_y[i], expected->last_y[i]);
            return 0;
        }
    }

    return 1;
}

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPWorkspace cpu_workspace;
    HPRQPWorkspace roundtrip_workspace;
    HPRQPParams params;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;

    if (!gpu_backend_available()) {
        printf("test_gpu_workspace skipped\n");
        return 0;
    }

    model_init(&model);
    workspace_init(&cpu_workspace);
    workspace_init(&roundtrip_workspace);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    test_params_set_no_scaling(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_STANDARD, &Q, &A, &model));
    TEST_ASSERT_OK(workspace_allocate(&model, &params, &cpu_workspace));
    TEST_ASSERT_OK(workspace_allocate(&model, &params, &roundtrip_workspace));

    TEST_ASSERT_OK(gpu_context_open(&context, 0));
    TEST_ASSERT_OK(gpu_problem_upload(&gpu_problem, &model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params));

    TEST_ASSERT(gpu_workspace.n == model.Q.nrow && gpu_workspace.m == model.A.nrow &&
                gpu_workspace.empty_A == 0 &&
                gpu_workspace.blocks_n > 0U && gpu_workspace.blocks_m > 0U,
                "gpu workspace allocation metadata are wrong");

    TEST_ASSERT_OK(gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0));
    TEST_ASSERT(gpu_workspace.spmv_prepared && gpu_workspace.spmv_Q.prepared &&
                gpu_workspace.spmv_A.prepared && gpu_workspace.spmv_AT.prepared,
                "gpu workspace SpMV preparation is wrong");

    seed_workspace_state(&cpu_workspace);
    TEST_ASSERT_OK(gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace));
    TEST_ASSERT_OK(gpu_vector_copy_from_host(&gpu_workspace.dx, cpu_workspace.dx, cpu_workspace.n));
    TEST_ASSERT_OK(gpu_vector_copy_from_host(&gpu_workspace.dw, cpu_workspace.dw, cpu_workspace.n));
    TEST_ASSERT_OK(gpu_vector_copy_from_host(&gpu_workspace.dy, cpu_workspace.dy, cpu_workspace.m));
    TEST_ASSERT_OK(gpu_workspace_copy_main_state_to_cpu(&gpu_workspace, &roundtrip_workspace));
    TEST_ASSERT_OK(gpu_workspace_copy_history_state_to_cpu(&gpu_workspace, &roundtrip_workspace));
    TEST_ASSERT_OK(gpu_workspace_copy_check_state_to_cpu(&gpu_workspace, &roundtrip_workspace));

    TEST_ASSERT(compare_workspace_state(&roundtrip_workspace, &cpu_workspace),
                "gpu workspace sync/copy roundtrip is wrong");

cleanup:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&roundtrip_workspace);
    workspace_free(&cpu_workspace);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    if (exit_code == 0) {
        printf("test_gpu_workspace passed\n");
    }
    return exit_code;
}