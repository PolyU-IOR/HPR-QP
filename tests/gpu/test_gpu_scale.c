#include "test_support.h"

#include <cuda_runtime.h>

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
    HPRQPParams params;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    double row_norm_host[3] = {0.0, 0.0, 0.0};
    double col_norm_host[2] = {0.0, 0.0};
    double q_values_host[2] = {0.0, 0.0};
    double a_values_host[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double at_values_host[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double c_host[2] = {0.0, 0.0};
    double al_host[3] = {0.0, 0.0, 0.0};
    double au_host[3] = {0.0, 0.0, 0.0};
    double l_host[2] = {0.0, 0.0};
    double u_host[2] = {0.0, 0.0};
    size_t i;

    if (!gpu_backend_available()) {
        printf("test_gpu_scale skipped\n");
        return 0;
    }

    model_init(&cpu_model);
    model_init(&gpu_model);
    scaling_init(&cpu_scaling);
    scaling_init(&gpu_scaling);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    params_set_defaults(&params);
    params.use_cr_scaling = 1;
    params.cr_iterations = 10;

    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_TALL_DIAG, &cpu_Q, &cpu_A, &cpu_model));
    TEST_ASSERT_OK(test_build_model_case(HPRQP_TEST_CASE_TALL_DIAG, &gpu_Q, &gpu_A, &gpu_model));
    TEST_ASSERT_OK(scale_problem(&cpu_model, &params, &cpu_scaling));

    TEST_ASSERT_OK(gpu_context_open(&context, 0));
    TEST_ASSERT_OK(gpu_problem_upload(&gpu_problem, &gpu_model));
    TEST_ASSERT_OK(gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params));
    TEST_ASSERT_OK(gpu_problem_scale(&gpu_problem, &gpu_workspace, &params, &gpu_scaling));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_workspace.row_norm, row_norm_host, 3));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_workspace.col_norm, col_norm_host, 2));
    TEST_ASSERT(cudaMemcpy(q_values_host,
                           gpu_problem.Q.values,
                           sizeof(q_values_host),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "failed to download scaled Q");
    TEST_ASSERT(cudaMemcpy(a_values_host,
                           gpu_problem.A.values,
                           sizeof(a_values_host),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "failed to download scaled A");
    TEST_ASSERT(cudaMemcpy(at_values_host,
                           gpu_problem.AT.values,
                           sizeof(at_values_host),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "failed to download scaled AT");
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_problem.c, c_host, 2));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_problem.AL, al_host, 3));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_problem.AU, au_host, 3));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_problem.l, l_host, 2));
    TEST_ASSERT_OK(gpu_vector_copy_to_host(&gpu_problem.u, u_host, 2));

    TEST_ASSERT(test_nearly_equal_rel(cpu_scaling.norm_b, gpu_scaling.norm_b, 1e-10) &&
                test_nearly_equal_rel(cpu_scaling.norm_c, gpu_scaling.norm_c, 1e-10) &&
                test_nearly_equal_rel(cpu_scaling.norm_b_org, gpu_scaling.norm_b_org, 1e-10) &&
                test_nearly_equal_rel(cpu_scaling.norm_c_org, gpu_scaling.norm_c_org, 1e-10),
                "gpu scaling norms are wrong");

    TEST_ASSERT(test_nearly_equal_abs(gpu_workspace.scale_b_scale, cpu_scaling.b_scale, 1e-12) &&
                test_nearly_equal_abs(gpu_workspace.scale_c_scale, cpu_scaling.c_scale, 1e-12) &&
                test_nearly_equal_rel(gpu_workspace.scale_norm_b_org, cpu_scaling.norm_b_org, 1e-10) &&
                test_nearly_equal_rel(gpu_workspace.scale_norm_c_org, cpu_scaling.norm_c_org, 1e-10),
                "gpu scaling metadata upload is wrong");

    TEST_ASSERT(test_nearly_equal_rel(col_norm_host[0], cpu_scaling.col_norm[0], 1e-10) &&
                test_nearly_equal_rel(col_norm_host[1], cpu_scaling.col_norm[1], 1e-10),
                "gpu column scaling vector is wrong");

    for (i = 0; i < 3; ++i) {
        TEST_ASSERT(test_nearly_equal_rel(row_norm_host[i], cpu_scaling.row_norm[i], 1e-10),
                    "gpu row scaling vector is wrong at index %zu",
                    i);
    }

    for (i = 0; i < 2; ++i) {
        TEST_ASSERT(test_nearly_equal_rel(q_values_host[i], cpu_model.Q.values[i], 1e-10) &&
                    test_nearly_equal_rel(c_host[i], cpu_model.c[i], 1e-10) &&
                    test_nearly_equal_rel(l_host[i], cpu_model.l[i], 1e-10) &&
                    test_nearly_equal_rel(u_host[i], cpu_model.u[i], 1e-10),
                    "gpu scaled Q/vector data do not match CPU at index %zu",
                    i);
    }
    for (i = 0; i < 6; ++i) {
        TEST_ASSERT(test_nearly_equal_rel(a_values_host[i], cpu_model.A.values[i], 1e-10) &&
                    test_nearly_equal_rel(at_values_host[i], cpu_model.AT.values[i], 1e-10),
                    "gpu scaled A/AT data do not match CPU at index %zu",
                    i);
    }
    for (i = 0; i < 3; ++i) {
        TEST_ASSERT(test_nearly_equal_rel(al_host[i], cpu_model.AL[i], 1e-10) &&
                    test_nearly_equal_rel(au_host[i], cpu_model.AU[i], 1e-10),
                    "gpu scaled constraint bounds do not match CPU at index %zu",
                    i);
    }

cleanup:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    scaling_free(&gpu_scaling);
    scaling_free(&cpu_scaling);
    model_free(&gpu_model);
    model_free(&cpu_model);
    matrix_csr_free(&gpu_A);
    matrix_csr_free(&gpu_Q);
    matrix_csr_free(&cpu_A);
    matrix_csr_free(&cpu_Q);
    if (exit_code == 0) {
        printf("test_gpu_scale passed\n");
    }
    return exit_code;
}
