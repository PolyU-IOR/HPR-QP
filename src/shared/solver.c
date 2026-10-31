#include "hprqp.h"

#include "shared/build.h"
#include "cpu/control.h"
#include "cpu/iteration.h"
#include "cpu/residuals.h"
#include "cpu/scale.h"
#include "cpu/workspace.h"
#include "shared/parallel.h"
#include "solver_backend.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_small_matrix(const char *name, const HPRQPMatrixCSR *matrix)
{
    size_t row;

    printf("%s=\n", name);
    for (row = 0; row < matrix->nrow; ++row) {
        size_t idx = matrix->row_ptr[row];
        size_t next = matrix->row_ptr[row + 1];
        size_t col = 0;

        while (col < matrix->ncol) {
            double value = 0.0;
            if (idx < next && matrix->col_ind[idx] == col) {
                value = matrix->values[idx];
                idx += 1;
            }
            printf(col + 1 == matrix->ncol ? "%.15g" : "%.15g ", value);
            col += 1;
        }
        printf("\n");
    }
}

static void print_small_vector(const char *name, const double *values, size_t n)
{
    size_t i;

    printf("%s=[", name);
    for (i = 0; i < n; ++i) {
        if (i > 0) {
            printf(", ");
        }
        printf("%.15g", values[i]);
    }
    printf("]\n");
}

static HPRQPStatus cpu_compute_residuals(HPRQPSolverBackendContext *context,
                                         HPRQPResiduals *residuals,
                                         size_t iter)
{
    return residuals_compute(context->workspace, context->model, context->scaling, residuals, iter);
}

static HPRQPStatus cpu_update_sigma(HPRQPSolverBackendContext *context,
                                    HPRQPRestart *restart_info,
                                    const HPRQPResiduals *residuals)
{
    if (context->params->method == HPRQP_METHOD_PARALLEL) {
        return parallel_update_metric(context->params,
                                      restart_info,
                                      context->workspace,
                                      residuals);
    }
    return sigma_update(context->params, restart_info, context->workspace, residuals);
}

static HPRQPStatus cpu_copy_output_iterates(HPRQPSolverBackendContext *context,
                                            double *x_bar,
                                            double *w_bar,
                                            double *z_bar,
                                            double *y_bar)
{
    const HPRQPScaling *scaling;
    size_t i;

    if (context == NULL || context->workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    scaling = context->scaling;
    if (scaling == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (context->workspace->n > 0) {
        if (x_bar == NULL || w_bar == NULL || z_bar == NULL) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        for (i = 0; i < context->workspace->n; ++i) {
            double col_norm = scaling->col_norm != NULL ? scaling->col_norm[i] : 1.0;
            x_bar[i] = scaling->b_scale * (context->workspace->x_bar[i] / col_norm);
            w_bar[i] = scaling->b_scale * (context->workspace->w_bar[i] / col_norm);
            z_bar[i] = scaling->c_scale * (context->workspace->z_bar[i] * col_norm);
        }
    }
    if (context->workspace->m > 0) {
        if (y_bar == NULL) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        for (i = 0; i < context->workspace->m; ++i) {
            double row_norm = scaling->row_norm != NULL ? scaling->row_norm[i] : 1.0;
            y_bar[i] = scaling->c_scale * (context->workspace->y_bar[i] / row_norm);
        }
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus cpu_restart_apply(HPRQPSolverBackendContext *context,
                                     HPRQPRestart *restart_info)
{
    if (context->params->method == HPRQP_METHOD_PARALLEL) {
        return restart_apply_parallel(restart_info, context->workspace);
    }
    return restart_apply(restart_info, context->workspace);
}

static double cpu_compute_m_norm(HPRQPSolverBackendContext *context)
{
    if (context->params->method == HPRQP_METHOD_PARALLEL) {
        return compute_m_norm_parallel(context->workspace);
    }
    return compute_m_norm(context->workspace);
}

static HPRQPStatus cpu_iteration_step(HPRQPSolverBackendContext *context, size_t inner_iter)
{
    return iteration_step(context->workspace, inner_iter);
}

static const HPRQPSolverBackendOps CPU_BACKEND_OPS = {
    cpu_compute_residuals,
    cpu_copy_output_iterates,
    cpu_update_sigma,
    cpu_restart_apply,
    cpu_compute_m_norm,
    cpu_iteration_step,
    NULL,
};

void params_set_defaults(HPRQPParams *params)
{
    if (params == NULL) {
        return;
    }

    memset(params, 0, sizeof(*params));
    params->stoptol = 1e-8;
    params->max_iter = (size_t)INT_MAX;
    params->use_gpu = 1;
    params->device_number = 0;
    params->time_limit = 1000.0;
    params->check_iter = 100;
    params->print_frequency = -1;
    params->use_scaling = 1;
    params->use_cr_scaling = 1;
    params->cr_iterations = 10;
    params->use_bc_scaling = 0;
    params->verbose = 1;
    params->spmv_mode_q = HPRQP_SPMV_MODE_AUTO;
    params->spmv_mode_a = HPRQP_SPMV_MODE_AUTO;
    params->method = HPRQP_METHOD_PARALLEL;
    params->parallel_majorizer_mode = HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR;
    params->qc_majorizer_mode = HPRQP_QC_MAJORIZER_AUTO;
    params->tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
    params->tau = 1.0;
    params->tau_min = 1.0 / 40.0;
    params->tau_max = 40.0;
    params->eig_factor = 1.05;
    params->metric_negative_tolerance = 1e-10;
    params->parallel_adaptation_energy_floor = 1e-12;
    params->sigma_fixed = 0;
    params->initial_sigma = -1.0;
    params->use_socp_restart_policy = 1;
    params->use_socp_sigma_policy = 1;
}

void results_init(HPRQPResults *results)
{
    if (results == NULL) {
        return;
    }
    memset(results, 0, sizeof(*results));
}

void results_free(HPRQPResults *results)
{
    if (results == NULL) {
        return;
    }
    free(results->x);
    free(results->y);
    free(results->z);
    free(results->w);
    free(results->qc_dual);
    results_init(results);
}

static HPRQPStatus solve_cpu(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results)
{
    HPRQPModel working_model;
    HPRQPScaling scaling;
    HPRQPWorkspace workspace;
    HPRQPSolverBackendContext context;
    HPRQPStatus status;
    HPRQPStatus return_status;

    if (model == NULL || params == NULL || results == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (model->qc.count > 0) {
        /* QCQP execution is a CUDA-only path in HPR-QCQP-C. */
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    if (parallel_validate_params(params) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_init(&working_model);
    scaling_init(&scaling);
    workspace_init(&workspace);
    results_init(results);
    return_status = HPRQP_STATUS_OK;

    status = model_copy(&working_model, model);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }

    if (params->verbose) {
        print_model_info(model);
        print_params_info(params);
    }

    status = scale_problem(&working_model, params, &scaling);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    if (params->verbose && working_model.Q.ncol <= 4 && working_model.A.nrow <= 4) {
        print_small_matrix("scaled_Q", &working_model.Q);
        print_small_matrix("scaled_A", &working_model.A);
        print_small_vector("scaled_c", working_model.c, working_model.Q.ncol);
        print_small_vector("scaled_AL", working_model.AL, working_model.A.nrow);
        print_small_vector("scaled_AU", working_model.AU, working_model.A.nrow);
        print_small_vector("scaled_l", working_model.l, working_model.Q.ncol);
        print_small_vector("scaled_u", working_model.u, working_model.Q.ncol);
    }

    status = workspace_allocate(&working_model, params, &workspace);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }

    context.workspace = &workspace;
    context.model = &working_model;
    context.scaling = &scaling;
    context.params = params;
    context.data = NULL;

    return_status = solve_with_backend(&context, &CPU_BACKEND_OPS, "CPU mode", results);

cleanup:
    workspace_free(&workspace);
    scaling_free(&scaling);
    model_free(&working_model);
    return return_status;
}

HPRQPStatus solve(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results)
{
    if (params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (parallel_validate_params(params) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (params->use_gpu) {
        return solve_gpu(model, params, results);
    }

    return solve_cpu(model, params, results);
}
