#include "test_support.h"

#include <cstdio>
#include <cstring>

static void seed_workspace_state(HPRQPWorkspace *workspace)
{
    workspace->x[0] = 0.4;
    workspace->x[1] = 0.8;
    workspace->w[0] = -0.2;
    workspace->w[1] = 0.3;
    workspace->last_x[0] = 0.1;
    workspace->last_x[1] = -0.05;
    workspace->last_w[0] = -0.1;
    workspace->last_w[1] = 0.07;
    workspace->last_Qw[0] = 0.01;
    workspace->last_Qw[1] = -0.02;
    workspace->last_ATy[0] = 0.03;
    workspace->last_ATy[1] = -0.04;
    if (workspace->m > 0) {
        workspace->y[0] = 0.15;
        workspace->last_y[0] = 0.02;
    }
    workspace->to_check = 1;
}

static void copy_workspace_state(HPRQPWorkspace *dst, const HPRQPWorkspace *src)
{
    std::memcpy(dst->x, src->x, src->n * sizeof(double));
    std::memcpy(dst->w, src->w, src->n * sizeof(double));
    std::memcpy(dst->last_x, src->last_x, src->n * sizeof(double));
    std::memcpy(dst->last_w, src->last_w, src->n * sizeof(double));
    std::memcpy(dst->last_Qw, src->last_Qw, src->n * sizeof(double));
    std::memcpy(dst->last_ATy, src->last_ATy, src->n * sizeof(double));
    if (src->m > 0) {
        std::memcpy(dst->y, src->y, src->m * sizeof(double));
        std::memcpy(dst->last_y, src->last_y, src->m * sizeof(double));
    }
    dst->to_check = src->to_check;
}

static int compare_workspace_state(const HPRQPWorkspace *actual,
                                   const HPRQPWorkspace *expected,
                                   const char *label,
                                   int expect_qw_bar)
{
    for (size_t i = 0; i < actual->n; ++i) {
        if (!test_nearly_equal_abs(actual->x[i], expected->x[i], 1e-12) ||
            !test_nearly_equal_abs(actual->x_bar[i], expected->x_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->w[i], expected->w[i], 1e-12) ||
            !test_nearly_equal_abs(actual->w_bar[i], expected->w_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Qw[i], expected->Qw[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATy[i], expected->ATy[i], 1e-12) ||
            !test_nearly_equal_abs(actual->ATy_bar[i], expected->ATy_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dx[i], expected->dx[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dw[i], expected->dw[i], 1e-12)) {
            std::fprintf(stderr, "GPU iteration mismatch at index %zu for %s\n", i, label);
            std::fprintf(stderr,
                         "  x=%g expected=%g x_bar=%g expected=%g w=%g expected=%g w_bar=%g expected=%g\n",
                         actual->x[i], expected->x[i], actual->x_bar[i], expected->x_bar[i],
                         actual->w[i], expected->w[i], actual->w_bar[i], expected->w_bar[i]);
            std::fprintf(stderr,
                         "  Qw=%g expected=%g Qw_bar=%g expected=%g ATy=%g expected=%g ATy_bar=%g expected=%g\n",
                         actual->Qw[i], expected->Qw[i], actual->Qw_bar[i], expected->Qw_bar[i],
                         actual->ATy[i], expected->ATy[i], actual->ATy_bar[i], expected->ATy_bar[i]);
            std::fprintf(stderr,
                         "  dx=%g expected=%g dw=%g expected=%g\n",
                         actual->dx[i], expected->dx[i], actual->dw[i], expected->dw[i]);
            return 1;
        }
        if (actual->method == HPRQP_METHOD_PARALLEL &&
            (!test_nearly_equal_abs(actual->x_hat[i], expected->x_hat[i], 1e-12) ||
             !test_nearly_equal_abs(actual->z_bar[i], expected->z_bar[i], 1e-12) ||
             !test_nearly_equal_abs(actual->Qw_bar[i], expected->Qw_bar[i], 1e-12) ||
             !test_nearly_equal_abs(actual->dQw[i], expected->dQw[i], 1e-12) ||
             !test_nearly_equal_abs(actual->ATdy[i], expected->ATdy[i], 1e-12))) {
            std::fprintf(stderr,
                         "GPU parallel image-state mismatch at index %zu for %s\n",
                         i,
                         label);
            return 1;
        }
        if (expect_qw_bar && !test_nearly_equal_abs(actual->Qw_bar[i], expected->Qw_bar[i], 1e-12)) {
            std::fprintf(stderr, "GPU iteration Qw_bar mismatch at index %zu for %s\n", i, label);
            std::fprintf(stderr,
                         "  Qw_bar=%g expected=%g\n",
                         actual->Qw_bar[i], expected->Qw_bar[i]);
            return 1;
        }
    }

    for (size_t i = 0; i < actual->m; ++i) {
        if (!test_nearly_equal_abs(actual->y[i], expected->y[i], 1e-12) ||
            !test_nearly_equal_abs(actual->y_bar[i], expected->y_bar[i], 1e-12) ||
            !test_nearly_equal_abs(actual->Ax[i], expected->Ax[i], 1e-12) ||
            !test_nearly_equal_abs(actual->dy[i], expected->dy[i], 1e-12)) {
            std::fprintf(stderr, "GPU dual-state mismatch at index %zu for %s\n", i, label);
            return 1;
        }
        if (actual->method == HPRQP_METHOD_PARALLEL &&
            !test_nearly_equal_abs(actual->s[i], expected->s[i], 1e-12)) {
            std::fprintf(stderr,
                         "GPU parallel projected constraint mismatch at index %zu for %s\n",
                         i,
                         label);
            return 1;
        }
    }

    return 0;
}

typedef struct ModeCase {
    HPRQPTestModelCase case_id;
    HPRQPMethod method;
    HPRQPSpmvMode mode_q;
    HPRQPSpmvMode mode_a;
    const char *label;
    int expect_qw_bar;
} ModeCase;

static int run_mode_case(const ModeCase *test_case)
{
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPParams params;
    HPRQPWorkspace cpu_workspace;
    HPRQPWorkspace expected_workspace;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPStatus status;
    int expect_qw_bar;

    model_init(&model);
    workspace_init(&cpu_workspace);
    workspace_init(&expected_workspace);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    test_params_set_no_scaling(&params);
    params.verbose = 0;
    params.spmv_mode_q = test_case->mode_q;
    params.spmv_mode_a = test_case->mode_a;
    params.method = test_case->method;
    params.tau_mode = HPRQP_TAU_MODE_FIXED;
    params.tau = 0.75;
    expect_qw_bar = test_case->expect_qw_bar;

    status = test_build_model_case(test_case->case_id, &Q, &A, &model);
    if (status != HPRQP_STATUS_OK) {
        return 1;
    }

    status = workspace_allocate(&model, &params, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = workspace_allocate(&model, &params, &expected_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }

    seed_workspace_state(&cpu_workspace);
    copy_workspace_state(&expected_workspace, &cpu_workspace);

    status = gpu_context_open(&context, 0);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_problem_upload(&gpu_problem, &model);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    gpu_workspace.spmv_mode_q = test_case->mode_q;
    gpu_workspace.spmv_mode_a = test_case->mode_a;
    status = gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }

    status = iteration_step(&expected_workspace, 0);
    if (status != HPRQP_STATUS_OK) {
        std::fprintf(stderr, "cpu iteration_step failed for %s\n", test_case->label);
        goto fail;
    }

    status = gpu_iteration_step(&gpu_workspace, &gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        std::fprintf(stderr, "gpu_iteration_step failed with status=%d for %s\n", (int)status, test_case->label);
        goto fail;
    }
    if (test_case->method == HPRQP_METHOD_PARALLEL) {
        double cpu_metric = compute_m_norm_parallel(&expected_workspace);
        double gpu_metric = 0.0;

        status = gpu_compute_m_norm_parallel_value(&gpu_workspace,
                                                   &gpu_metric);
        if (status != HPRQP_STATUS_OK ||
            !test_nearly_equal_rel(cpu_metric, gpu_metric, 1e-11)) {
            std::fprintf(stderr,
                         "GPU parallel metric mismatch for %s: %g expected %g (status=%d)\n",
                         test_case->label,
                         gpu_metric,
                         cpu_metric,
                         (int)status);
            goto fail;
        }
    }
    status = gpu_workspace_copy_main_state_to_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = gpu_workspace_copy_check_state_to_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }

    if (compare_workspace_state(&cpu_workspace, &expected_workspace, test_case->label, expect_qw_bar) != 0) {
        goto fail;
    }

    if (test_case->method == HPRQP_METHOD_PARALLEL) {
        HPRQPRestart cpu_restart;
        HPRQPRestart gpu_restart;
        HPRQPResiduals residuals;

        restart_init(&cpu_restart);
        restart_init(&gpu_restart);
        residuals_init(&residuals);
        cpu_restart.restart_flag = 1;
        gpu_restart.restart_flag = 1;
        params.tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
        params.sigma_fixed = 1;

        status = parallel_update_metric(&params,
                                        &cpu_restart,
                                        &expected_workspace,
                                        &residuals);
        if (status != HPRQP_STATUS_OK) {
            std::fprintf(stderr,
                         "CPU parallel metric update failed for %s\n",
                         test_case->label);
            goto fail;
        }
        status = gpu_parallel_update_metric_value(&gpu_workspace,
                                                  &gpu_problem,
                                                  &params,
                                                  &gpu_restart,
                                                  &residuals);
        if (status != HPRQP_STATUS_OK ||
            !test_nearly_equal_rel(gpu_workspace.tau,
                                   expected_workspace.tau,
                                   1e-11) ||
            !test_nearly_equal_rel(gpu_workspace.d_y,
                                   expected_workspace.d_y,
                                   1e-11) ||
            !test_nearly_equal_rel(gpu_workspace.d_w,
                                   expected_workspace.d_w,
                                   1e-11) ||
            gpu_restart.metric_reseed_pending !=
                cpu_restart.metric_reseed_pending) {
            std::fprintf(stderr,
                         "GPU adaptive parallel metric update mismatch for %s\n",
                         test_case->label);
            goto fail;
        }

        status = restart_apply_parallel(&cpu_restart, &expected_workspace);
        if (status != HPRQP_STATUS_OK) {
            goto fail;
        }
        status = gpu_restart_apply_parallel_value(&gpu_workspace);
        if (status != HPRQP_STATUS_OK) {
            goto fail;
        }
        status = gpu_workspace_copy_main_state_to_cpu(&gpu_workspace,
                                                       &cpu_workspace);
        if (status != HPRQP_STATUS_OK) {
            goto fail;
        }
        status = gpu_workspace_copy_history_state_to_cpu(&gpu_workspace,
                                                          &cpu_workspace);
        if (status != HPRQP_STATUS_OK) {
            goto fail;
        }
        for (size_t i = 0; i < cpu_workspace.n; ++i) {
            if (!test_nearly_equal_abs(cpu_workspace.x[i],
                                       expected_workspace.x[i],
                                       1e-12) ||
                !test_nearly_equal_abs(cpu_workspace.w[i],
                                       expected_workspace.w[i],
                                       1e-12) ||
                !test_nearly_equal_abs(cpu_workspace.last_x[i],
                                       expected_workspace.last_x[i],
                                       1e-12) ||
                !test_nearly_equal_abs(cpu_workspace.last_w[i],
                                       expected_workspace.last_w[i],
                                       1e-12) ||
                !test_nearly_equal_abs(cpu_workspace.last_Qw[i],
                                       expected_workspace.last_Qw[i],
                                       1e-12)) {
                std::fprintf(stderr,
                             "GPU parallel restart mismatch at index %zu for %s\n",
                             i,
                             test_case->label);
                goto fail;
            }
        }
    }

    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&expected_workspace);
    workspace_free(&cpu_workspace);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return 0;

fail:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&expected_workspace);
    workspace_free(&cpu_workspace);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return 1;
}

static int run_persistent_batch_case(
    HPRQPTestModelCase case_id,
    HPRQPGPUParallelPersistentKernel kernel,
    unsigned int q_group_width,
    size_t batch_count,
    const char *label)
{
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPParams params;
    HPRQPWorkspace initial_workspace;
    HPRQPWorkspace expected_workspace;
    HPRQPWorkspace actual_workspace;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPStatus status;
    int exit_code = 1;

    model_init(&model);
    workspace_init(&initial_workspace);
    workspace_init(&expected_workspace);
    workspace_init(&actual_workspace);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    test_params_set_no_scaling(&params);
    params.verbose = 0;
    params.method = HPRQP_METHOD_PARALLEL;
    params.parallel_majorizer_mode = HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER;
    params.tau_mode = HPRQP_TAU_MODE_FIXED;
    params.tau = 0.75;
    params.spmv_mode_q = HPRQP_SPMV_MODE_CUSTOMIZED;
    params.spmv_mode_a = HPRQP_SPMV_MODE_CUSTOMIZED;

    status = test_build_model_case(case_id,
                                   &Q,
                                   &A,
                                   &model);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = workspace_allocate(&model, &params, &initial_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = workspace_allocate(&model, &params, &expected_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = workspace_allocate(&model, &params, &actual_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }

    seed_workspace_state(&initial_workspace);
    copy_workspace_state(&expected_workspace, &initial_workspace);
    copy_workspace_state(&actual_workspace, &initial_workspace);
    for (size_t inner_iter = 0U; inner_iter < batch_count; ++inner_iter) {
        expected_workspace.to_check =
            inner_iter + 1U == batch_count ? 1 : 0;
        status = iteration_step(&expected_workspace, inner_iter);
        if (status != HPRQP_STATUS_OK) {
            std::fprintf(stderr,
                         "CPU persistent batch reference failed at %zu "
                         "for %s\n",
                         inner_iter,
                         label);
            goto cleanup;
        }
    }

    status = gpu_context_open(&context, 0);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = gpu_problem_upload(&gpu_problem, &model);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = gpu_workspace_sync_from_cpu(&gpu_workspace,
                                         &actual_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    if (kernel == HPRQP_GPU_PARALLEL_PERSISTENT_GROUP) {
        /* Production selection also prepares the compact nonempty-Q rows. */
        status = gpu_workspace_select_spmv_modes(&gpu_workspace,
                                                 &gpu_problem,
                                                 &params);
        if (status != HPRQP_STATUS_OK) {
            goto cleanup;
        }
    }
    gpu_workspace.spmv_mode_q = HPRQP_SPMV_MODE_CUSTOMIZED;
    gpu_workspace.custom_spmv_q_kernel =
        kernel == HPRQP_GPU_PARALLEL_PERSISTENT_GROUP
            ? HPRQP_GPU_CUSTOM_SPMV_WARP
            : HPRQP_GPU_CUSTOM_SPMV_ROW;
    gpu_workspace.parallel_persistent_enabled = 1;
    gpu_workspace.parallel_persistent_kernel = kernel;
    gpu_workspace.parallel_persistent_threads = 128U;
    gpu_workspace.parallel_persistent_blocks = 1U;
    gpu_workspace.parallel_persistent_use_col16 = 0;
    gpu_workspace.parallel_persistent_q_group_width = q_group_width;
    gpu_workspace.to_check = 1;

    status = gpu_iteration_batch_parallel(&gpu_workspace,
                                          &gpu_problem,
                                          0U,
                                          batch_count);
    if (status != HPRQP_STATUS_OK) {
        std::fprintf(stderr,
                     "GPU persistent batch failed with status=%d for %s\n",
                     (int)status,
                     label);
        goto cleanup;
    }
    status = gpu_workspace_copy_main_state_to_cpu(&gpu_workspace,
                                                   &actual_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    status = gpu_workspace_copy_check_state_to_cpu(&gpu_workspace,
                                                    &actual_workspace);
    if (status != HPRQP_STATUS_OK) {
        goto cleanup;
    }
    if (compare_workspace_state(&actual_workspace,
                                &expected_workspace,
                                label,
                                1) != 0) {
        goto cleanup;
    }

    exit_code = 0;

cleanup:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&actual_workspace);
    workspace_free(&expected_workspace);
    workspace_free(&initial_workspace);
    model_free(&model);
    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    return exit_code;
}

int main(void)
{
    static const ModeCase kCases[] = {
        {HPRQP_TEST_CASE_GPU_ITERATION_STANDARD, HPRQP_METHOD_SGS, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "standard-cusparse", 1},
        {HPRQP_TEST_CASE_GPU_ITERATION_STANDARD, HPRQP_METHOD_SGS, HPRQP_SPMV_MODE_CUSTOMIZED, HPRQP_SPMV_MODE_CUSTOMIZED, "standard-customized", 0},
        {HPRQP_TEST_CASE_STANDARD_NOC, HPRQP_METHOD_SGS, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "noc-cusparse", 1},
        {HPRQP_TEST_CASE_STANDARD_NOC, HPRQP_METHOD_SGS, HPRQP_SPMV_MODE_CUSTOMIZED, HPRQP_SPMV_MODE_CUSTOMIZED, "noc-customized", 1},
        {HPRQP_TEST_CASE_EMPTY_A, HPRQP_METHOD_SGS, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "empty-a", 1},
        {HPRQP_TEST_CASE_EMPTY_A, HPRQP_METHOD_SGS, HPRQP_SPMV_MODE_CUSTOMIZED, HPRQP_SPMV_MODE_CUSPARSE, "empty-a-custom-q", 0},
        {HPRQP_TEST_CASE_PARALLEL_SINGULAR, HPRQP_METHOD_PARALLEL, HPRQP_SPMV_MODE_CUSPARSE, HPRQP_SPMV_MODE_CUSPARSE, "parallel-singular-q", 1},
    };

    if (!gpu_backend_available()) {
        std::printf("test_gpu_iteration skipped\n");
        return 0;
    }

    for (size_t i = 0; i < sizeof(kCases) / sizeof(kCases[0]); ++i) {
        if (run_mode_case(&kCases[i]) != 0) {
            return 1;
        }
    }
    if (run_persistent_batch_case(
            HPRQP_TEST_CASE_EMPTY_A,
            HPRQP_GPU_PARALLEL_PERSISTENT_ROW,
            32U,
            100U,
            "parallel-empty-a-persistent-row-batch") != 0) {
        return 1;
    }
    if (run_persistent_batch_case(
            HPRQP_TEST_CASE_GPU_ITERATION_STANDARD,
            HPRQP_GPU_PARALLEL_PERSISTENT_ROW,
            32U,
            100U,
            "parallel-standard-persistent-row-batch") != 0) {
        return 1;
    }
    if (run_persistent_batch_case(
            HPRQP_TEST_CASE_GPU_ITERATION_STANDARD,
            HPRQP_GPU_PARALLEL_PERSISTENT_ROW,
            32U,
            3U,
            "parallel-standard-persistent-row-odd-batch") != 0) {
        return 1;
    }
    if (run_persistent_batch_case(
            HPRQP_TEST_CASE_GPU_ITERATION_STANDARD,
            HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE,
            32U,
            100U,
            "parallel-standard-persistent-row-pipeline") != 0) {
        return 1;
    }
    if (run_persistent_batch_case(
            HPRQP_TEST_CASE_GPU_ITERATION_STANDARD,
            HPRQP_GPU_PARALLEL_PERSISTENT_GROUP,
            64U,
            100U,
            "parallel-standard-persistent-warp-group") != 0) {
        return 1;
    }

    std::printf("test_gpu_iteration passed\n");
    return 0;
}
