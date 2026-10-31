#include "hprqp_binding.h"

#include <math.h>
#include <stdio.h>

static int nearly_equal(double lhs, double rhs, double tol)
{
    double scale = fmax(1.0, fmax(fabs(lhs), fabs(rhs)));

    return fabs(lhs - rhs) <= tol * scale;
}

int main(void)
{
    int exit_code = 0;
    static const size_t q_row_ptr[] = {0, 1, 2};
    static const size_t q_col_ind[] = {0, 1};
    static const double q_values[] = {2.0, 4.0};
    static const size_t a_row_ptr[] = {0, 2};
    static const size_t a_col_ind[] = {0, 1};
    static const double a_values[] = {1.0, 1.0};
    static const double c[] = {2.0, 1.0};
    static const double AL[] = {2.0};
    static const double AU[] = {2.0};
    static const double l[] = {0.0, 0.0};
    static const double u[] = {10.0, 10.0};
    HPRQPBindingModel *model = NULL;
    HPRQPBindingParams *cpu_params = NULL;
    HPRQPBindingParams *gpu_params = NULL;
    HPRQPBindingResults *cpu_results = NULL;
    HPRQPBindingResults *gpu_results = NULL;
    const double *cpu_x;
    const double *gpu_x;
    HPRQPStatus status;

    if (!hprqp_gpu_backend_available()) {
        printf("test_binding_api_gpu skipped: gpu backend unavailable\n");
        return 0;
    }

    status = hprqp_model_create(&model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_model_create failed: %s\n", hprqp_status_name(status));
        return 1;
    }

    status = hprqp_model_build_from_qabc_csr(model,
                                             2,
                                             2,
                                             2,
                                             q_row_ptr,
                                             q_col_ind,
                                             q_values,
                                             1,
                                             2,
                                             2,
                                             a_row_ptr,
                                             a_col_ind,
                                             a_values,
                                             c,
                                             AL,
                                             AU,
                                             l,
                                             u,
                                             0.0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_model_build_from_qabc_csr failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    status = hprqp_params_create(&cpu_params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_create cpu failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    status = hprqp_params_create(&gpu_params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_params_create gpu failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    status = hprqp_results_create(&cpu_results);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_results_create cpu failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    status = hprqp_results_create(&gpu_results);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "hprqp_results_create gpu failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    if (hprqp_params_set_verbose(cpu_params, 0) != HPRQP_STATUS_OK ||
        hprqp_params_set_verbose(gpu_params, 0) != HPRQP_STATUS_OK ||
        hprqp_params_set_use_gpu(cpu_params, 0) != HPRQP_STATUS_OK ||
        hprqp_params_set_use_gpu(gpu_params, 1) != HPRQP_STATUS_OK ||
        hprqp_params_set_use_cr_scaling(cpu_params, 1) != HPRQP_STATUS_OK ||
        hprqp_params_set_use_cr_scaling(gpu_params, 1) != HPRQP_STATUS_OK ||
        hprqp_params_set_cr_iterations(cpu_params, 10) != HPRQP_STATUS_OK ||
        hprqp_params_set_cr_iterations(gpu_params, 10) != HPRQP_STATUS_OK ||
        hprqp_params_set_device_number(gpu_params, 0) != HPRQP_STATUS_OK) {
        fprintf(stderr, "failed to configure binding params for cpu/gpu comparison\n");
        exit_code = 1;
        goto cleanup;
    }

    status = hprqp_optimize(model, cpu_params, cpu_results);
    if (status != HPRQP_STATUS_OPTIMAL) {
        fprintf(stderr, "cpu binding solve failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    status = hprqp_optimize(model, gpu_params, gpu_results);
    if (status != HPRQP_STATUS_OPTIMAL) {
        fprintf(stderr, "gpu binding solve failed: %s\n", hprqp_status_name(status));
        exit_code = 1;
        goto cleanup;
    }

    cpu_x = hprqp_results_x_data(cpu_results);
    gpu_x = hprqp_results_x_data(gpu_results);
    if (cpu_x == NULL || gpu_x == NULL) {
        fprintf(stderr, "binding cpu/gpu comparison missing x vectors\n");
        exit_code = 1;
        goto cleanup;
    }

    if (!nearly_equal(cpu_x[0], gpu_x[0], 1e-6) ||
        !nearly_equal(cpu_x[1], gpu_x[1], 1e-6) ||
        !nearly_equal(hprqp_results_objective(cpu_results), hprqp_results_objective(gpu_results), 1e-6) ||
        !nearly_equal(hprqp_results_residual(cpu_results), hprqp_results_residual(gpu_results), 1e-6) ||
        !nearly_equal(hprqp_results_gap(cpu_results), hprqp_results_gap(gpu_results), 1e-6)) {
        fprintf(stderr, "binding gpu solve diverged from cpu reference\n");
        exit_code = 1;
        goto cleanup;
    }

cleanup:
    hprqp_results_destroy(gpu_results);
    hprqp_results_destroy(cpu_results);
    hprqp_params_destroy(gpu_params);
    hprqp_params_destroy(cpu_params);
    hprqp_model_destroy(model);
    if (exit_code == 0) {
        printf("test_binding_api_gpu passed\n");
    }
    return exit_code;
}
