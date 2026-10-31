#include "test_support.h"

#include <string.h>

static void copy_workspace_state(HPRQPWorkspace *dst, const HPRQPWorkspace *src)
{
    dst->sigma = src->sigma;
    dst->lambda_max_A = src->lambda_max_A;
    dst->lambda_max_Q = src->lambda_max_Q;
    dst->Q_is_diag = src->Q_is_diag;
    dst->noC = src->noC;
    dst->to_check = src->to_check;

    memcpy(dst->x, src->x, src->n * sizeof(double));
    memcpy(dst->x_bar, src->x_bar, src->n * sizeof(double));
    memcpy(dst->x_hat, src->x_hat, src->n * sizeof(double));
    memcpy(dst->z_bar, src->z_bar, src->n * sizeof(double));
    memcpy(dst->dx, src->dx, src->n * sizeof(double));
    memcpy(dst->w, src->w, src->n * sizeof(double));
    memcpy(dst->w_bar, src->w_bar, src->n * sizeof(double));
    memcpy(dst->dw, src->dw, src->n * sizeof(double));
    memcpy(dst->Qw, src->Qw, src->n * sizeof(double));
    memcpy(dst->Qw_bar, src->Qw_bar, src->n * sizeof(double));
    memcpy(dst->Qx, src->Qx, src->n * sizeof(double));
    memcpy(dst->ATy, src->ATy, src->n * sizeof(double));
    memcpy(dst->ATy_bar, src->ATy_bar, src->n * sizeof(double));
    memcpy(dst->ATdy, src->ATdy, src->n * sizeof(double));
    memcpy(dst->dQw, src->dQw, src->n * sizeof(double));
    memcpy(dst->QATdy, src->QATdy, src->n * sizeof(double));
    memcpy(dst->tempv, src->tempv, src->n * sizeof(double));
    memcpy(dst->last_x, src->last_x, src->n * sizeof(double));
    memcpy(dst->last_w, src->last_w, src->n * sizeof(double));
    memcpy(dst->last_Qw, src->last_Qw, src->n * sizeof(double));
    memcpy(dst->last_ATy, src->last_ATy, src->n * sizeof(double));
    memcpy(dst->fact1, src->fact1, src->n * sizeof(double));
    memcpy(dst->fact2, src->fact2, src->n * sizeof(double));
    memcpy(dst->fact, src->fact, src->n * sizeof(double));
    memcpy(dst->fact_M, src->fact_M, src->n * sizeof(double));
    memcpy(dst->diag_Q, src->diag_Q, src->n * sizeof(double));

    if (src->m > 0) {
        memcpy(dst->y, src->y, src->m * sizeof(double));
        memcpy(dst->y_bar, src->y_bar, src->m * sizeof(double));
        memcpy(dst->dy, src->dy, src->m * sizeof(double));
        memcpy(dst->s, src->s, src->m * sizeof(double));
        memcpy(dst->Rp, src->Rp, src->m * sizeof(double));
        memcpy(dst->Ax, src->Ax, src->m * sizeof(double));
        memcpy(dst->last_y, src->last_y, src->m * sizeof(double));
    }
}

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR std_cpu_Q = {0};
    HPRQPMatrixCSR std_cpu_A = {0};
    HPRQPMatrixCSR std_gpu_Q = {0};
    HPRQPMatrixCSR std_gpu_A = {0};
    HPRQPMatrixCSR nd_cpu_Q = {0};
    HPRQPMatrixCSR nd_cpu_A = {0};
    HPRQPMatrixCSR nd_gpu_Q = {0};
    HPRQPMatrixCSR nd_gpu_A = {0};
    HPRQPModel std_cpu_model;
    HPRQPModel std_gpu_model;
    HPRQPModel nd_cpu_model;
    HPRQPModel nd_gpu_model;
    HPRQPWorkspace std_cpu_workspace;
    HPRQPWorkspace expected_restart_workspace;
    HPRQPWorkspace restart_roundtrip_workspace;
    HPRQPWorkspace nd_cpu_workspace;
    HPRQPScaling std_gpu_scaling;
    HPRQPScaling nd_gpu_scaling;
    HPRQPResiduals residuals;
    HPRQPRestart restart_info;
    HPRQPRestart restart_info_nd;
    HPRQPRestart restart_policy;
    HPRQPParams params;
    HPRQPGPUContext std_context;
    HPRQPGPUProblem std_gpu_problem;
    HPRQPGPUWorkspace std_gpu_workspace;
    HPRQPGPUContext nd_context;
    HPRQPGPUProblem nd_gpu_problem;
    HPRQPGPUWorkspace nd_gpu_workspace;
    double diag_q_host[2] = {0.0, 0.0};
    double fact1_host[2] = {0.0, 0.0};
    double fact2_host[2] = {0.0, 0.0};
    double fact_host[2] = {0.0, 0.0};
    double fact_m_host[2] = {0.0, 0.0};
    double cpu_m_norm;
    double gpu_m_norm;
    double sigma_value = 0.0;

    if (!gpu_backend_available()) {
        printf("test_gpu_control skipped\n");
        return 0;
    }

    model_init(&std_cpu_model);
    model_init(&std_gpu_model);
    model_init(&nd_cpu_model);
    model_init(&nd_gpu_model);
    workspace_init(&std_cpu_workspace);
    workspace_init(&expected_restart_workspace);
    workspace_init(&restart_roundtrip_workspace);
    workspace_init(&nd_cpu_workspace);
    scaling_init(&std_gpu_scaling);
    scaling_init(&nd_gpu_scaling);
    residuals_init(&residuals);
    restart_init(&restart_info);
    restart_init(&restart_info_nd);
    restart_init(&restart_policy);
    gpu_context_init(&std_context);
    gpu_problem_init(&std_gpu_problem);
    gpu_workspace_init(&std_gpu_workspace);
    gpu_context_init(&nd_context);
    gpu_problem_init(&nd_gpu_problem);
    gpu_workspace_init(&nd_gpu_workspace);
    test_params_set_no_scaling(&params);

    /* Type 3 uses only the 15%-of-total-iterations long window.  The removed
     * 10,000-iteration shortcut would restart the first probe below. */
    restart_policy.first_restart = 0;
    restart_policy.inner = 10000U;
    restart_policy.last_gap = 4.0;
    restart_policy.current_gap = 2.8;
    restart_policy.save_gap = 3.0;
    restart_check_socp(&restart_policy, 100000U, 100U, 1.0, 1.0);
    TEST_ASSERT(restart_policy.restart_flag == 0,
                "SOCP Type-3 shortcut was not removed");
    restart_policy.inner = 15000U;
    restart_check_socp(&restart_policy, 100000U, 100U, 1.0, 1.0);
    TEST_ASSERT(restart_policy.restart_flag == 3,
                "SOCP Type-3 15-percent window did not trigger");

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_STANDARD, &std_cpu_Q, &std_cpu_A, &std_cpu_model));
    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_STANDARD, &std_gpu_Q, &std_gpu_A, &std_gpu_model));
    TEST_ASSERT_OK(workspace_allocate(&std_cpu_model, &params, &std_cpu_workspace));
    TEST_ASSERT_OK(workspace_allocate(&std_cpu_model, &params, &expected_restart_workspace));
    TEST_ASSERT_OK(workspace_allocate(&std_cpu_model, &params, &restart_roundtrip_workspace));

    TEST_ASSERT_OK(gpu_context_open(&std_context, 0));
    TEST_ASSERT_OK(gpu_problem_upload(&std_gpu_problem, &std_gpu_model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&std_gpu_workspace, &std_gpu_problem, &params));
    TEST_ASSERT_OK(gpu_problem_scale(&std_gpu_problem, &std_gpu_workspace, &params, &std_gpu_scaling));
    TEST_ASSERT_OK(gpu_workspace_prepare_spmv(&std_gpu_workspace, &std_gpu_problem, 0));
    TEST_ASSERT_OK(gpu_workspace_setup(&std_gpu_workspace, &std_gpu_problem, &params));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&std_gpu_workspace.diag_Q, diag_q_host, 2));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&std_gpu_workspace.fact1, fact1_host, 2));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&std_gpu_workspace.fact2, fact2_host, 2));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&std_gpu_workspace.fact, fact_host, 2));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&std_gpu_workspace.fact_M, fact_m_host, 2));

    TEST_ASSERT(test_nearly_equal_rel(std_gpu_workspace.sigma, std_cpu_workspace.sigma, 1e-10) &&
                test_nearly_equal_rel(std_gpu_workspace.lambda_max_A, std_cpu_workspace.lambda_max_A, 1e-10) &&
                test_nearly_equal_rel(std_gpu_workspace.lambda_max_Q, std_cpu_workspace.lambda_max_Q, 1e-10) &&
                std_gpu_workspace.Q_is_diag == std_cpu_workspace.Q_is_diag &&
                std_gpu_workspace.noC == std_cpu_workspace.noC,
                "gpu workspace setup scalars are wrong");

    TEST_ASSERT(test_nearly_equal_rel(diag_q_host[0], std_cpu_workspace.diag_Q[0], 1e-10) &&
                test_nearly_equal_rel(diag_q_host[1], std_cpu_workspace.diag_Q[1], 1e-10) &&
                test_nearly_equal_rel(fact1_host[0], std_cpu_workspace.fact1[0], 1e-10) &&
                test_nearly_equal_rel(fact1_host[1], std_cpu_workspace.fact1[1], 1e-10) &&
                test_nearly_equal_rel(fact2_host[0], std_cpu_workspace.fact2[0], 1e-10) &&
                test_nearly_equal_rel(fact2_host[1], std_cpu_workspace.fact2[1], 1e-10) &&
                test_nearly_equal_rel(fact_host[0], std_cpu_workspace.fact[0], 1e-10) &&
                test_nearly_equal_rel(fact_host[1], std_cpu_workspace.fact[1], 1e-10) &&
                test_nearly_equal_rel(fact_m_host[0], std_cpu_workspace.fact_M[0], 1e-10) &&
                test_nearly_equal_rel(fact_m_host[1], std_cpu_workspace.fact_M[1], 1e-10),
                "gpu workspace setup factors are wrong");

    TEST_ASSERT_OK(iteration_step(&std_cpu_workspace, 0));
    TEST_ASSERT_OK(gpu_iteration_step(&std_gpu_workspace, &std_gpu_problem, 0));
    cpu_m_norm = compute_m_norm(&std_cpu_workspace);
    TEST_ASSERT_OK(gpu_compute_m_norm_value(&std_gpu_workspace, &std_gpu_problem, &gpu_m_norm));
    TEST_ASSERT(test_nearly_equal_rel(gpu_m_norm, cpu_m_norm, 1e-8),
                "gpu M-norm value is wrong");

    restart_info.restart_flag = 1;
    restart_info.inner = 1;
    std_cpu_workspace.noC = 1;
    std_cpu_workspace.Qw_bar[0] = 7.0;
    std_cpu_workspace.Qw_bar[1] = 11.0;
    std_cpu_workspace.Qw[0] = -1.0;
    std_cpu_workspace.Qw[1] = -2.0;
    std_cpu_workspace.last_Qw[0] = -3.0;
    std_cpu_workspace.last_Qw[1] = -4.0;

    copy_workspace_state(&expected_restart_workspace, &std_cpu_workspace);
    TEST_ASSERT_OK(gpu_workspace_sync_from_cpu(&std_gpu_workspace, &std_cpu_workspace));
    TEST_ASSERT_OK(restart_apply(&restart_info, &expected_restart_workspace));
    TEST_ASSERT_OK(gpu_restart_apply_value(&std_gpu_workspace, &std_gpu_problem));
    TEST_ASSERT_OK(gpu_workspace_copy_main_state_to_cpu(&std_gpu_workspace, &restart_roundtrip_workspace));
    TEST_ASSERT_OK(gpu_workspace_copy_history_state_to_cpu(&std_gpu_workspace, &restart_roundtrip_workspace));

    TEST_ASSERT(test_nearly_equal_abs(restart_roundtrip_workspace.x[0], expected_restart_workspace.x[0], 1e-12) &&
                test_nearly_equal_abs(restart_roundtrip_workspace.w[1], expected_restart_workspace.w[1], 1e-12) &&
                test_nearly_equal_abs(restart_roundtrip_workspace.y[0], expected_restart_workspace.y[0], 1e-12) &&
                test_nearly_equal_abs(restart_roundtrip_workspace.Qw[0], expected_restart_workspace.Qw[0], 1e-12) &&
                test_nearly_equal_abs(restart_roundtrip_workspace.Qw[1], expected_restart_workspace.Qw[1], 1e-12) &&
                test_nearly_equal_abs(restart_roundtrip_workspace.last_Qw[0], expected_restart_workspace.last_Qw[0], 1e-12) &&
                test_nearly_equal_abs(restart_roundtrip_workspace.last_Qw[1], expected_restart_workspace.last_Qw[1], 1e-12),
                "gpu restart apply is wrong");

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_CONTROL_NONDIAG, &nd_cpu_Q, &nd_cpu_A, &nd_cpu_model));
    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_CONTROL_NONDIAG, &nd_gpu_Q, &nd_gpu_A, &nd_gpu_model));
    TEST_ASSERT_OK(workspace_allocate(&nd_cpu_model, &params, &nd_cpu_workspace));
    TEST_ASSERT_OK(iteration_step(&nd_cpu_workspace, 0));

    TEST_ASSERT_OK(gpu_context_open(&nd_context, 0));
    TEST_ASSERT_OK(gpu_problem_upload(&nd_gpu_problem, &nd_gpu_model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&nd_gpu_workspace, &nd_gpu_problem, &params));
    TEST_ASSERT_OK(gpu_problem_scale(&nd_gpu_problem, &nd_gpu_workspace, &params, &nd_gpu_scaling));
    TEST_ASSERT_OK(gpu_workspace_prepare_spmv(&nd_gpu_workspace, &nd_gpu_problem, 0));
    TEST_ASSERT_OK(gpu_workspace_setup(&nd_gpu_workspace, &nd_gpu_problem, &params));
    TEST_ASSERT_OK(gpu_workspace_sync_from_cpu(&nd_gpu_workspace, &nd_cpu_workspace));

    restart_info_nd.first_restart = 0;
    restart_info_nd.inner = 0;
    restart_info_nd.last_gap = 10.0;
    restart_info_nd.current_gap = 1.0;
    restart_info_nd.save_gap = 2.0;
    restart_info_nd.best_gap = 1.0;
    restart_info_nd.best_sigma = nd_cpu_workspace.sigma;
    restart_info_nd.weighted_norm = 1.0;
    restart_info_nd.restart_flag = 1;
    residuals.err_Rd_org_bar = 0.15;
    residuals.err_Rp_org_bar = 0.1;
    residuals.rel_gap_bar = 0.05;

    TEST_ASSERT_OK(sigma_update(&params, &restart_info_nd, &nd_cpu_workspace, &residuals));
    TEST_ASSERT_OK(gpu_sigma_update_value(&nd_gpu_workspace,
                                          &nd_gpu_problem,
                                          &params,
                                          &restart_info_nd,
                                          &residuals,
                                          &sigma_value));
    TEST_ASSERT(test_nearly_equal_rel(sigma_value, nd_cpu_workspace.sigma, 1e-8),
                "gpu sigma update is wrong");

cleanup:
    gpu_workspace_free(&nd_gpu_workspace);
    gpu_problem_free(&nd_gpu_problem);
    gpu_context_free(&nd_context);
    gpu_workspace_free(&std_gpu_workspace);
    gpu_problem_free(&std_gpu_problem);
    gpu_context_free(&std_context);
    scaling_free(&nd_gpu_scaling);
    scaling_free(&std_gpu_scaling);
    workspace_free(&nd_cpu_workspace);
    workspace_free(&restart_roundtrip_workspace);
    workspace_free(&expected_restart_workspace);
    workspace_free(&std_cpu_workspace);
    model_free(&nd_gpu_model);
    model_free(&nd_cpu_model);
    model_free(&std_gpu_model);
    model_free(&std_cpu_model);
    matrix_csr_free(&nd_gpu_A);
    matrix_csr_free(&nd_gpu_Q);
    matrix_csr_free(&nd_cpu_A);
    matrix_csr_free(&nd_cpu_Q);
    matrix_csr_free(&std_gpu_A);
    matrix_csr_free(&std_gpu_Q);
    matrix_csr_free(&std_cpu_A);
    matrix_csr_free(&std_cpu_Q);
    if (exit_code == 0) {
        printf("test_gpu_control passed\n");
    }
    return exit_code;
}
