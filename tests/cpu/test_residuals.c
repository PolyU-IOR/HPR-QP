#include "test_support.h"

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPScaling scaling;
    HPRQPWorkspace workspace;
    HPRQPResiduals residuals;
    HPRQPParams params;

    model_init(&model);
    scaling_init(&scaling);
    workspace_init(&workspace);
    residuals_init(&residuals);
    test_params_set_no_scaling(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_RESIDUALS, &Q, &A, &model));
    TEST_ASSERT_OK(scale_problem(&model, &params, &scaling));
    TEST_ASSERT_OK(workspace_allocate(&model, &params, &workspace));

    workspace.x_bar[0] = 1.0;
    workspace.x_bar[1] = 1.0;
    workspace.y_bar[0] = 0.5;
    workspace.z_bar[0] = 0.2;
    workspace.z_bar[1] = 0.3;
    workspace.s[0] = 0.25;

    TEST_ASSERT_OK(residuals_compute(&workspace, &model, &scaling, &residuals, 1));

    TEST_ASSERT(residuals.is_updated &&
                test_nearly_equal_abs(residuals.primal_obj_bar, 3.0, 1e-12) &&
                test_nearly_equal_abs(residuals.dual_obj_bar, -2.375, 1e-12),
                "objective values are wrong");

    TEST_ASSERT(test_nearly_equal_abs(residuals.err_Rd_org_bar, 0.46, 1e-12) &&
                test_nearly_equal_abs(residuals.err_Rp_org_bar, 0.0, 1e-12) &&
                test_nearly_equal_abs(residuals.rel_gap_bar, 1.34375, 1e-12) &&
                test_nearly_equal_abs(residuals.KKTx_and_gap_org_bar, 1.34375, 1e-12),
                "residual values are wrong");

    workspace.x_bar[0] = 11.0;
    workspace.x_bar[1] = 1.0;
    workspace.y_bar[0] = 0.0;
    workspace.z_bar[0] = 0.0;
    workspace.z_bar[1] = 0.0;
    workspace.s[0] = 0.0;

    TEST_ASSERT_OK(residuals_compute(&workspace, &model, &scaling, &residuals, 0));
    TEST_ASSERT(test_nearly_equal_abs(residuals.err_Rp_org_bar, 1.0, 1e-12),
                "bound violation residual is wrong");

cleanup:
    workspace_free(&workspace);
    scaling_free(&scaling);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    if (exit_code == 0) {
        printf("test_residuals passed\n");
    }
    return exit_code;
}