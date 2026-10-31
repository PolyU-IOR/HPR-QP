#include "internal/hprqp_internal.h"

#include "solver_backend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef struct HPRQPGPUSolveContext {
    HPRQPGPUProblem *gpu_problem;
    HPRQPGPUWorkspace *gpu_workspace;
} HPRQPGPUSolveContext;

static void gpu_sync_workspace_scalars(HPRQPWorkspace *workspace,
                                       const HPRQPGPUWorkspace *gpu_workspace)
{
    workspace->sigma = gpu_workspace->sigma;
    workspace->lambda_max_A = gpu_workspace->lambda_max_A;
    workspace->lambda_max_Q = gpu_workspace->lambda_max_Q;
    workspace->lambda_max_AQA = gpu_workspace->lambda_max_AQA;
    workspace->lambda_max_AQ2A = gpu_workspace->lambda_max_AQ2A;
    workspace->zeta_AB = gpu_workspace->zeta_AB;
    workspace->zeta_AQ = gpu_workspace->zeta_AQ;
    workspace->zeta_BQ = gpu_workspace->zeta_BQ;
    workspace->tau_AB = gpu_workspace->tau_AB;
    workspace->tau_AQ = gpu_workspace->tau_AQ;
    workspace->tau_BQ = gpu_workspace->tau_BQ;
    workspace->tau = gpu_workspace->tau;
    workspace->tau_initial = gpu_workspace->tau_initial;
    workspace->d_y = gpu_workspace->d_y;
    workspace->d_s = gpu_workspace->d_s;
    workspace->d_w = gpu_workspace->d_w;
    workspace->parallel_y_rho = gpu_workspace->parallel_y_rho;
    workspace->parallel_y_inv = gpu_workspace->parallel_y_inv;
    workspace->parallel_w_gamma = gpu_workspace->parallel_w_gamma;
    workspace->parallel_w_inv = gpu_workspace->parallel_w_inv;
    workspace->metric_negative_tolerance =
        gpu_workspace->metric_negative_tolerance;
    workspace->method = gpu_workspace->method;
    workspace->parallel_majorizer_mode =
        gpu_workspace->parallel_majorizer_mode;
    workspace->Q_is_diag = gpu_workspace->Q_is_diag;
    workspace->eigenvalues_initialized =
        gpu_workspace->eigenvalues_initialized;
    workspace->noC = gpu_workspace->noC;
}

static HPRQPStatus gpu_compute_residuals(HPRQPSolverBackendContext *context,
                                         HPRQPResiduals *residuals,
                                         size_t iter)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;

    return gpu_compute_residuals_value(gpu_context->gpu_workspace,
                                       gpu_context->gpu_problem,
                                       residuals,
                                       iter);
}

static HPRQPStatus gpu_update_sigma(HPRQPSolverBackendContext *context,
                                    HPRQPRestart *restart_info,
                                    const HPRQPResiduals *residuals)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    HPRQPWorkspace *workspace = context->workspace;
    HPRQPStatus status;
    double sigma_value;
    if (restart_info->restart_flag < 1) {
        return HPRQP_STATUS_OK;
    }

    if (context->params->method == HPRQP_METHOD_PARALLEL) {
        status = gpu_parallel_update_metric_value(gpu_context->gpu_workspace,
                                                  gpu_context->gpu_problem,
                                                  context->params,
                                                  restart_info,
                                                  residuals);
    } else {
        status = gpu_sigma_update_value(gpu_context->gpu_workspace,
                                        gpu_context->gpu_problem,
                                        context->params,
                                        restart_info,
                                        residuals,
                                        &sigma_value);
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    gpu_sync_workspace_scalars(workspace, gpu_context->gpu_workspace);

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_copy_output_iterates(HPRQPSolverBackendContext *context,
                                            double *x_bar,
                                            double *w_bar,
                                            double *z_bar,
                                            double *y_bar)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    HPRQPGPUWorkspace *gpu_workspace;


    gpu_workspace = gpu_context->gpu_workspace;
    return gpu_workspace_copy_output_unscaled_to_host(gpu_workspace, x_bar, w_bar, z_bar, y_bar);
}

static HPRQPStatus gpu_restart_apply(HPRQPSolverBackendContext *context,
                                     HPRQPRestart *restart_info)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    HPRQPStatus status;

    if (restart_info->restart_flag <= 0) {
        return HPRQP_STATUS_OK;
    }

    if (context->params->method == HPRQP_METHOD_PARALLEL) {
        status = gpu_restart_apply_parallel_value(gpu_context->gpu_workspace);
    } else {
        status = gpu_restart_apply_value(gpu_context->gpu_workspace,
                                         gpu_context->gpu_problem);
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    restart_info->last_gap =
        context->params->method == HPRQP_METHOD_PARALLEL &&
                restart_info->metric_reseed_pending
            ? INFINITY
            : restart_info->current_gap;
    restart_info->save_gap = INFINITY;
    restart_info->times += 1;
    restart_info->inner = 0;
    return HPRQP_STATUS_OK;
}

static double gpu_compute_m_norm(HPRQPSolverBackendContext *context)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;
    double value;

    if (context->params->method == HPRQP_METHOD_PARALLEL) {
        return gpu_compute_m_norm_parallel_value(gpu_context->gpu_workspace,
                                                 &value) == HPRQP_STATUS_OK
                   ? value
                   : INFINITY;
    }
    return gpu_compute_m_norm_value(gpu_context->gpu_workspace,
                                    gpu_context->gpu_problem,
                                    &value) == HPRQP_STATUS_OK
               ? value
               : INFINITY;
}

static HPRQPStatus gpu_iteration_step_backend(HPRQPSolverBackendContext *context, size_t inner_iter)
{
    HPRQPGPUSolveContext *gpu_context = (HPRQPGPUSolveContext *)context->data;

    gpu_context->gpu_workspace->to_check = context->workspace->to_check;

    return gpu_iteration_step(gpu_context->gpu_workspace,
                              gpu_context->gpu_problem,
                              inner_iter);
}

static HPRQPStatus gpu_iteration_batch_backend(
    HPRQPSolverBackendContext *context,
    size_t inner_iter,
    size_t batch_count)
{
    HPRQPGPUSolveContext *gpu_context =
        (HPRQPGPUSolveContext *)context->data;

    if (context->params->method != HPRQP_METHOD_PARALLEL ||
        !gpu_context->gpu_workspace->parallel_persistent_enabled) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    gpu_context->gpu_workspace->to_check = context->workspace->to_check;
    return gpu_iteration_batch_parallel(gpu_context->gpu_workspace,
                                        gpu_context->gpu_problem,
                                        inner_iter,
                                        batch_count);
}

static const HPRQPSolverBackendOps GPU_BACKEND_OPS = {
    gpu_compute_residuals,
    gpu_copy_output_iterates,
    gpu_update_sigma,
    gpu_restart_apply,
    gpu_compute_m_norm,
    gpu_iteration_step_backend,
    gpu_iteration_batch_backend,
};

HPRQPStatus solve_gpu(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results)
{
    HPRQPScaling scaling;
    HPRQPWorkspace workspace;
    HPRQPGPUContext context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPGPUSolveContext gpu_solve_context;
    HPRQPSolverBackendContext backend_context;
    HPRQPStatus status;
    HPRQPStatus return_status;
    const bool profile_setup = std::getenv("HPRQP_PROFILE_SETUP") != nullptr;
    const auto profile_start =
        profile_setup ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
    auto profile_last = profile_start;
    double context_open_seconds = 0.0;
    double problem_upload_seconds = 0.0;
    double workspace_allocate_seconds = 0.0;
    double problem_scale_seconds = 0.0;
    double prepare_spmv_seconds = 0.0;
    double workspace_setup_seconds = 0.0;
    double select_kernels_seconds = 0.0;
    double solver_loop_seconds = 0.0;
    double *host_qc_constant = NULL;
    double *host_qc_cr_shape = NULL;

    if (model == NULL || params == NULL || results == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (model->qc.count > 0 && params->method != HPRQP_METHOD_PARALLEL) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    scaling_init(&scaling);
    workspace_init(&workspace);
    results_init(results);
    gpu_context_init(&context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    return_status = HPRQP_STATUS_OK;

    if (params->verbose) {
        print_model_info(model);
        print_params_info(params);
    }

    status = gpu_context_open(&context, params->device_number);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        context_open_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_problem_upload(&gpu_problem, model);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        problem_upload_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_workspace_allocate(&gpu_workspace, &gpu_problem, params);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        workspace_allocate_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_problem_scale(&gpu_problem, &gpu_workspace, params, &scaling);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        problem_scale_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, params->verbose);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        prepare_spmv_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    status = gpu_workspace_setup(&gpu_workspace, &gpu_problem, params);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        workspace_setup_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    if (std::getenv("HPRQP_DIAG_HPR_METRIC_SCALING") != nullptr) {
        double metric_damping =
            std::atof(std::getenv("HPRQP_DIAG_HPR_METRIC_SCALING"));
        if (!(metric_damping > 0.0) || metric_damping > 1.0) {
            metric_damping = 1.0;
        }
        status = gpu_problem_hpr_metric_scale(&gpu_problem,
                                              &gpu_workspace,
                                              &scaling,
                                              metric_damping);
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }
        /* Sparsity and device pointers are unchanged; recompute only the
         * numerical spectral quantities and the QC majorizer. */
        status = gpu_workspace_setup(&gpu_workspace, &gpu_problem, params);
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }
    }
    workspace.m = gpu_workspace.m;
    workspace.n = gpu_workspace.n;
    gpu_sync_workspace_scalars(&workspace, &gpu_workspace);
    workspace.to_check = 1;

    status = gpu_workspace_select_spmv_modes(&gpu_workspace, &gpu_problem, params);
    if (status == HPRQP_STATUS_OK) {
        status = gpu_workspace_prepare_selected_spmv_graphs(
            &gpu_workspace, &gpu_problem, params->verbose);
    }
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        select_kernels_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }

    gpu_solve_context.gpu_problem = &gpu_problem;
    gpu_solve_context.gpu_workspace = &gpu_workspace;

    backend_context.workspace = &workspace;
    backend_context.model = model;
    backend_context.scaling = &scaling;
    backend_context.params = params;
    backend_context.data = &gpu_solve_context;

    return_status = solve_with_backend(&backend_context, &GPU_BACKEND_OPS, "GPU mode", results);
    if (model->qc.count > 0U && results->x != NULL) {
        results->qc_dual = (double *)calloc(model->qc.cone_dim,
                                           sizeof(double));
        if (results->qc_dual == NULL) {
            results_free(results);
            return_status = HPRQP_STATUS_ALLOCATION_FAILED;
            goto cleanup;
        }
        status = cudaMemcpy(results->qc_dual,
                            gpu_workspace.qc_dual_bar.values,
                            model->qc.cone_dim * sizeof(double),
                            cudaMemcpyDeviceToHost) == cudaSuccess
                     ? HPRQP_STATUS_OK
                     : HPRQP_STATUS_NOT_IMPLEMENTED;
        if (status != HPRQP_STATUS_OK) {
            results_free(results);
            return_status = status;
            goto cleanup;
        }
        host_qc_constant = (double *)calloc(model->qc.count,
                                            sizeof(double));
        host_qc_cr_shape = (double *)calloc(model->qc.count,
                                            sizeof(double));
        if (host_qc_constant == NULL || host_qc_cr_shape == NULL ||
            cudaMemcpy(host_qc_constant, gpu_problem.qc_constant.values,
                       model->qc.count * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(host_qc_cr_shape, gpu_workspace.qc_cr_shape.values,
                       model->qc.count * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            results_free(results);
            return_status = host_qc_constant == NULL ||
                                    host_qc_cr_shape == NULL
                                ? HPRQP_STATUS_ALLOCATION_FAILED
                                : HPRQP_STATUS_NOT_IMPLEMENTED;
            goto cleanup;
        }
        for (size_t cone = 0; cone < model->qc.count; ++cone) {
            const size_t begin = 2U * cone +
                                 model->qc.diagonal.row_ptr[cone];
            const size_t end = 2U * (cone + 1U) +
                               model->qc.diagonal.row_ptr[cone + 1U];
            const double factor = host_qc_constant[cone] *
                                  gpu_workspace.scale_b_scale *
                                  gpu_workspace.scale_c_scale;
            const double shape = host_qc_cr_shape[cone];
            const size_t tail_begin = begin + 2U;
            results->qc_dual[begin] *= factor;
            results->qc_dual[begin + 1U] *= factor * shape;
            for (size_t i = tail_begin; i < end; ++i) {
                results->qc_dual[i] *= factor * sqrt(shape);
            }
        }
        results->qc_cone_dim = model->qc.cone_dim;
    }
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        solver_loop_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }

cleanup:
    free(host_qc_constant);
    free(host_qc_cr_shape);
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&context);
    workspace_free(&workspace);
    scaling_free(&scaling);
    if (profile_setup) {
        const auto profile_done = std::chrono::steady_clock::now();
        const double cleanup_seconds =
            std::chrono::duration<double>(profile_done - profile_last).count();
        const double total_seconds =
            std::chrono::duration<double>(profile_done - profile_start).count();
        std::fprintf(stderr,
                     "HPRQP_PROFILE gpu: context_open=%.6f s, "
                     "problem_upload=%.6f s, workspace_allocate=%.6f s, "
                     "problem_scale=%.6f s, prepare_spmv=%.6f s, "
                     "workspace_setup=%.6f s, select_kernels=%.6f s, "
                     "solver_loop=%.6f s, cleanup=%.6f s, total=%.6f s\n",
                     context_open_seconds,
                     problem_upload_seconds,
                     workspace_allocate_seconds,
                     problem_scale_seconds,
                     prepare_spmv_seconds,
                     workspace_setup_seconds,
                     select_kernels_seconds,
                     solver_loop_seconds,
                     cleanup_seconds,
                     total_seconds);
    }
    return return_status;
}
