#include "test_support.h"

static void seed_first_state(HPRQPWorkspace *workspace)
{
    workspace->x_bar[0] = 1.0;
    workspace->x_bar[1] = 1.0;
    workspace->y_bar[0] = 0.5;
    workspace->z_bar[0] = 0.2;
    workspace->z_bar[1] = 0.3;
    workspace->s[0] = 0.25;
}

static void seed_second_state(HPRQPWorkspace *workspace)
{
    workspace->x_bar[0] = 11.0;
    workspace->x_bar[1] = 1.0;
    workspace->y_bar[0] = 0.0;
    workspace->z_bar[0] = 0.0;
    workspace->z_bar[1] = 0.0;
    workspace->s[0] = 0.0;
}

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR cpu_Q = {0};
    HPRQPMatrixCSR cpu_A = {0};
    HPRQPMatrixCSR gpu_Q = {0};
    HPRQPMatrixCSR gpu_A = {0};
    HPRQPModel cpu_model;
    HPRQPModel gpu_model;
    HPRQPScaling cpu_scaling;
    HPRQPScaling gpu_scaling;
    HPRQPWorkspace cpu_workspace;
    HPRQPResiduals gpu_residuals;
    HPRQPParams params;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;

    if (!gpu_backend_available()) {
        printf("test_gpu_residuals skipped\n");
        return 0;
    }

    model_init(&cpu_model);
    model_init(&gpu_model);
    scaling_init(&cpu_scaling);
    scaling_init(&gpu_scaling);
    workspace_init(&cpu_workspace);
    residuals_init(&gpu_residuals);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    test_params_set_no_scaling(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_RESIDUALS, &cpu_Q, &cpu_A, &cpu_model));
    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_RESIDUALS, &gpu_Q, &gpu_A, &gpu_model));
    TEST_ASSERT_OK(scale_problem(&cpu_model, &params, &cpu_scaling));
    TEST_ASSERT_OK(workspace_allocate(&cpu_model, &params, &cpu_workspace));

    TEST_ASSERT_OK(gpu_context_open(&context, 0));
    TEST_ASSERT_OK(gpu_problem_upload(&gpu_problem, &gpu_model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params));
    TEST_ASSERT_OK(gpu_problem_scale(&gpu_problem, &gpu_workspace, &params, &gpu_scaling));
    TEST_ASSERT_OK(gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0));

    seed_first_state(&cpu_workspace);
    TEST_ASSERT_OK(gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace));
    TEST_ASSERT_OK(gpu_compute_residuals_value(&gpu_workspace, &gpu_problem, &gpu_residuals, 1));
    TEST_ASSERT(gpu_residuals.is_updated &&
                test_nearly_equal_abs(gpu_residuals.primal_obj_bar, 3.0, 1e-8) &&
                test_nearly_equal_abs(gpu_residuals.dual_obj_bar, -1.0, 1e-8) &&
                test_nearly_equal_abs(gpu_residuals.err_Rd_org_bar, 0.46, 1e-8) &&
                test_nearly_equal_abs(gpu_residuals.err_Rp_org_bar, 0.0, 1e-8) &&
                test_nearly_equal_abs(gpu_residuals.rel_gap_bar, 1.0, 1e-8) &&
                test_nearly_equal_abs(gpu_residuals.KKTx_and_gap_org_bar, 1.0, 1e-8),
                "gpu residual values are wrong on iter-1 case: primal=%g dual=%g rd=%g rp=%g gap=%g kkt=%g",
                gpu_residuals.primal_obj_bar,
                gpu_residuals.dual_obj_bar,
                gpu_residuals.err_Rd_org_bar,
                gpu_residuals.err_Rp_org_bar,
                gpu_residuals.rel_gap_bar,
                gpu_residuals.KKTx_and_gap_org_bar);

    seed_second_state(&cpu_workspace);
    TEST_ASSERT_OK(gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace));
    TEST_ASSERT_OK(gpu_compute_residuals_value(&gpu_workspace, &gpu_problem, &gpu_residuals, 0));
    TEST_ASSERT(test_nearly_equal_abs(gpu_residuals.err_Rp_org_bar, 1.0, 1e-8),
                "iteration-zero variable-bound safeguard is wrong");
    TEST_ASSERT_OK(gpu_compute_residuals_value(&gpu_workspace, &gpu_problem, &gpu_residuals, 1));
    TEST_ASSERT(test_nearly_equal_abs(gpu_residuals.err_Rp_org_bar, 0.0, 1e-8),
                "variable bounds must leave eta_primal after projection begins");

cleanup:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&cpu_workspace);
    scaling_free(&gpu_scaling);
    scaling_free(&cpu_scaling);
    model_free(&gpu_model);
    model_free(&cpu_model);
    matrix_csr_free(&gpu_A);
    matrix_csr_free(&gpu_Q);
    matrix_csr_free(&cpu_A);
    matrix_csr_free(&cpu_Q);
    if (exit_code == 0) {
        printf("test_gpu_residuals passed\n");
    }
    return exit_code;
}
