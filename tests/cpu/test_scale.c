#include "test_support.h"

int main(void)
{
    int exit_code = 0;
    HPRQPMatrixCSR Q = {0};
    HPRQPMatrixCSR A = {0};
    HPRQPModel model;
    HPRQPScaling scaling;
    HPRQPParams params;
    double max_row_abs = 0.0;
    double max_col0_abs = 0.0;
    double max_col1_abs = 0.0;
    double cr_row_norm[1] = {0.0};
    double cr_col_norm[2] = {0.0, 0.0};

    model_init(&model);
    scaling_init(&scaling);
    params_set_defaults(&params);

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_SCALE, &Q, &A, &model));
    TEST_ASSERT_OK(curtis_reid_norms_compute(&model.A,
                                             &model.AT,
                                             10,
                                             cr_row_norm,
                                             cr_col_norm));
    TEST_ASSERT(test_nearly_equal_rel(cr_row_norm[0], 20.0, 1e-12) &&
                test_nearly_equal_rel(cr_col_norm[0], 0.8, 1e-12) &&
                test_nearly_equal_rel(cr_col_norm[1], 1.25, 1e-12),
                "CR10 norms do not match the reference recurrence");

    cr_row_norm[0] = 0.0;
    cr_col_norm[0] = 0.0;
    cr_col_norm[1] = 0.0;
    TEST_ASSERT_OK(curtis_reid_norms_compute(&model.A,
                                             &model.AT,
                                             20,
                                             cr_row_norm,
                                             cr_col_norm));
    TEST_ASSERT(test_nearly_equal_rel(cr_row_norm[0], 20.0, 1e-12) &&
                test_nearly_equal_rel(cr_col_norm[0], 0.8, 1e-12) &&
                test_nearly_equal_rel(cr_col_norm[1], 1.25, 1e-12),
                "CR20 norms do not match the reference recurrence");

    {
        size_t a_row_ptr[] = {0, 2};
        size_t a_col_ind[] = {0, 1};
        double a_values[] = {16.0, 0.0};
        size_t at_row_ptr[] = {0, 1, 2};
        size_t at_col_ind[] = {0, 0};
        double at_values[] = {16.0, 0.0};
        HPRQPMatrixCSR stored_zero_A = {1, 2, 2, a_row_ptr, a_col_ind, a_values, 0};
        HPRQPMatrixCSR stored_zero_AT = {2, 1, 2, at_row_ptr, at_col_ind, at_values, 0};
        double stored_zero_row_norm[1] = {0.0};
        double stored_zero_col_norm[2] = {0.0, 0.0};

        TEST_ASSERT_OK(curtis_reid_norms_compute(&stored_zero_A,
                                                 &stored_zero_AT,
                                                 1,
                                                 stored_zero_row_norm,
                                                 stored_zero_col_norm));
        TEST_ASSERT(stored_zero_row_norm[0] > 0.0 &&
                    stored_zero_row_norm[0] < 1e-20 &&
                    stored_zero_col_norm[0] > 1e20 &&
                    stored_zero_col_norm[1] > 0.0 &&
                    stored_zero_col_norm[1] < 1e-20,
                    "stored zeros do not follow the Julia 1e-300 floor semantics");
    }

    params.use_cr_scaling = 1;
    params.cr_iterations = 10;
    TEST_ASSERT_OK(scale_problem(&model, &params, &scaling));

    TEST_ASSERT(scaling.row_norm[0] > 0.0 && scaling.col_norm[0] > 0.0 && scaling.col_norm[1] > 0.0,
                "scaling norms are non-positive");

    max_row_abs = fabs(model.A.values[0]) > fabs(model.A.values[1]) ? fabs(model.A.values[0]) : fabs(model.A.values[1]);
    max_col0_abs = fabs(model.A.values[0]) > fabs(model.Q.values[0]) ? fabs(model.A.values[0]) : fabs(model.Q.values[0]);
    max_col1_abs = fabs(model.A.values[1]) > fabs(model.Q.values[1]) ? fabs(model.A.values[1]) : fabs(model.Q.values[1]);

    TEST_ASSERT(max_row_abs <= 1.0 + 1e-10 && max_col0_abs <= 1.0 + 1e-10 && max_col1_abs <= 1.0 + 1e-10,
                "scaled matrix magnitudes are too large");

    TEST_ASSERT(matrix_csr_is_valid(&model.AT) && model.AT.nrow == 2 && model.AT.ncol == 1 && model.AT.nnz == 2,
                "scaled AT is wrong");

    TEST_ASSERT(scaling.norm_b > 0.0 && scaling.norm_c > 0.0 &&
                scaling.norm_b_org > 0.0 && scaling.norm_c_org > 0.0,
                "scaling norms metadata are wrong");

    TEST_ASSERT(test_nearly_equal_abs(scaling.l_org[0], 0.0, 1e-10) &&
                test_nearly_equal_abs(scaling.l_org[1], -1.0, 1e-10) &&
                test_nearly_equal_abs(scaling.u_org[0], 10.0, 1e-10) &&
                test_nearly_equal_abs(scaling.u_org[1], 3.0, 1e-10),
                "original bound copies are wrong");

cleanup:
    scaling_free(&scaling);
    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    if (exit_code == 0) {
        printf("test_scale passed\n");
    }
    return exit_code;
}
