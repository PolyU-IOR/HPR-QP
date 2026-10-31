#include "internal/hprqp_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TraceContext {
    HPRQPModel model;
    HPRQPScaling scaling;
    HPRQPWorkspace cpu_workspace;
    HPRQPWorkspace gpu_shadow;
    HPRQPGPUContext gpu_context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPParams params;
    int gpu_host_state_dirty;
} TraceContext;

static const char *status_name(HPRQPStatus status)
{
    switch (status) {
    case HPRQP_STATUS_OK:
        return "OK";
    case HPRQP_STATUS_OPTIMAL:
        return "OPTIMAL";
    case HPRQP_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case HPRQP_STATUS_ALLOCATION_FAILED:
        return "ALLOCATION_FAILED";
    case HPRQP_STATUS_NOT_IMPLEMENTED:
        return "NOT_IMPLEMENTED";
    case HPRQP_STATUS_CONTINUE:
        return "CONTINUE";
    case HPRQP_STATUS_MAX_ITER:
        return "MAX_ITER";
    case HPRQP_STATUS_TIME_LIMIT:
        return "TIME_LIMIT";
    default:
        return "UNKNOWN";
    }
}

static int parse_size_arg(const char *value, size_t *out)
{
    char *endptr;
    unsigned long long parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtoull(value, &endptr, 10);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = (size_t)parsed;
    return 1;
}

static int parse_int_arg(const char *value, int *out)
{
    char *endptr;
    long parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtol(value, &endptr, 10);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

static int parse_spmv_mode_arg(const char *value, HPRQPSpmvMode *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "auto") == 0) {
        *out = HPRQP_SPMV_MODE_AUTO;
        return 1;
    }
    if (strcmp(value, "cusparse") == 0) {
        *out = HPRQP_SPMV_MODE_CUSPARSE;
        return 1;
    }
    if (strcmp(value, "customized") == 0) {
        *out = HPRQP_SPMV_MODE_CUSTOMIZED;
        return 1;
    }
    return 0;
}

static void print_usage(const char *argv0)
{
    printf("usage: %s [mps_path] [--max-iter N] [--check-iter N] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1]\n",
           argv0);
}

static int should_check_next_iter(const HPRQPParams *params,
                                  const HPRQPRestart *restart_info,
                                  size_t next_iter)
{
    if (params == NULL || restart_info == NULL) {
        return 0;
    }
    if (params->check_iter > 0 && next_iter % params->check_iter == 0) {
        return 1;
    }
    if (restart_info->restart_flag > 0) {
        return 1;
    }
    return 0;
}

static HPRQPStatus gpu_sync_sigma_and_factors(HPRQPGPUWorkspace *gpu_workspace,
                                              const HPRQPWorkspace *cpu_workspace)
{
    HPRQPStatus status;

    if (gpu_workspace == NULL || cpu_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_workspace->sigma = cpu_workspace->sigma;
    gpu_workspace->lambda_max_A = cpu_workspace->lambda_max_A;
    gpu_workspace->lambda_max_Q = cpu_workspace->lambda_max_Q;
    gpu_workspace->Q_is_diag = cpu_workspace->Q_is_diag;

    if (cpu_workspace->Q_is_diag) {
        status = gpu_vector_copy_from_host(&gpu_workspace->fact1, cpu_workspace->fact1, cpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_vector_copy_from_host(&gpu_workspace->fact2, cpu_workspace->fact2, cpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_vector_copy_from_host(&gpu_workspace->fact, cpu_workspace->fact, cpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_vector_copy_from_host(&gpu_workspace->fact_M, cpu_workspace->fact_M, cpu_workspace->n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_compute_residuals_trace(TraceContext *trace,
                                               HPRQPResiduals *residuals,
                                               size_t iter)
{
    HPRQPStatus status;

    if (trace == NULL || residuals == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (!trace->gpu_host_state_dirty) {
        status = gpu_workspace_copy_main_state_to_cpu(&trace->gpu_workspace,
                                  &trace->gpu_shadow);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    return residuals_compute(&trace->gpu_shadow,
                             &trace->model,
                             &trace->scaling,
                             residuals,
                             iter);
}

static HPRQPStatus gpu_apply_restart_trace(TraceContext *trace, HPRQPRestart *restart_info)
{
    HPRQPStatus status;

    if (trace == NULL || restart_info == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (restart_info->restart_flag <= 0) {
        return HPRQP_STATUS_OK;
    }

    status = restart_apply(restart_info, &trace->gpu_shadow);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    trace->gpu_host_state_dirty = 1;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_iteration_step_trace(TraceContext *trace, size_t inner_iter)
{
    HPRQPStatus status;

    if (trace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (trace->gpu_shadow.to_check) {
        if (!trace->gpu_host_state_dirty) {
            status = gpu_workspace_copy_main_state_to_cpu(&trace->gpu_workspace,
                                                          &trace->gpu_shadow);
            if (status != HPRQP_STATUS_OK) {
                return status;
            }
            status = gpu_workspace_copy_history_state_to_cpu(&trace->gpu_workspace,
                                                             &trace->gpu_shadow);
            if (status != HPRQP_STATUS_OK) {
                return status;
            }
        }

        status = iteration_step(&trace->gpu_shadow, inner_iter);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        trace->gpu_host_state_dirty = 1;
        return HPRQP_STATUS_OK;
    }

    if (trace->gpu_host_state_dirty) {
        status = gpu_workspace_sync_from_cpu(&trace->gpu_workspace, &trace->gpu_shadow);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        trace->gpu_host_state_dirty = 0;
    }

    return gpu_iteration_step(&trace->gpu_workspace, &trace->gpu_problem, inner_iter);
}

static void trace_context_init(TraceContext *trace)
{
    if (trace == NULL) {
        return;
    }

    model_init(&trace->model);
    scaling_init(&trace->scaling);
    workspace_init(&trace->cpu_workspace);
    workspace_init(&trace->gpu_shadow);
    gpu_context_init(&trace->gpu_context);
    gpu_problem_init(&trace->gpu_problem);
    gpu_workspace_init(&trace->gpu_workspace);
    params_set_defaults(&trace->params);
    trace->params.verbose = 0;
    trace->gpu_host_state_dirty = 0;
}

static void trace_context_free(TraceContext *trace)
{
    if (trace == NULL) {
        return;
    }

    gpu_workspace_free(&trace->gpu_workspace);
    gpu_problem_free(&trace->gpu_problem);
    gpu_context_free(&trace->gpu_context);
    workspace_free(&trace->gpu_shadow);
    workspace_free(&trace->cpu_workspace);
    scaling_free(&trace->scaling);
    model_free(&trace->model);
}

static HPRQPStatus trace_context_setup(TraceContext *trace, const char *mps_path)
{
    HPRQPStatus status;

    if (trace == NULL || mps_path == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = build_model_from_mps(mps_path, &trace->model);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = scale_problem(&trace->model, &trace->params, &trace->scaling);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = workspace_allocate(&trace->model, &trace->params, &trace->cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = workspace_allocate(&trace->model, &trace->params, &trace->gpu_shadow);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_context_open(&trace->gpu_context, trace->params.device_number);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_problem_upload(&trace->gpu_problem, &trace->model);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_workspace_allocate(&trace->gpu_workspace, &trace->gpu_problem, &trace->params);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_workspace_upload_scaling(&trace->gpu_workspace, &trace->scaling);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_workspace_prepare_spmv(&trace->gpu_workspace, &trace->gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_workspace_sync_from_cpu(&trace->gpu_workspace, &trace->cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_workspace_select_spmv_modes(&trace->gpu_workspace, &trace->gpu_problem, &trace->params);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    trace->gpu_host_state_dirty = 0;
    return HPRQP_STATUS_OK;
}

static void print_trace_header(void)
{
    printf("iter  cpu_flag gpu_flag  cpu_gap        gpu_gap        cpu_best       gpu_best       cpu_sigma      gpu_sigma      cpu_res        gpu_res\n");
}

static void print_trace_row(size_t iter,
                            const HPRQPRestart *cpu_restart,
                            const HPRQPRestart *gpu_restart,
                            const HPRQPWorkspace *cpu_workspace,
                            const HPRQPWorkspace *gpu_shadow,
                            const HPRQPResiduals *cpu_residuals,
                            const HPRQPResiduals *gpu_residuals)
{
    printf("%5zu %9d %8d  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e  %.6e\n",
           iter,
           cpu_restart->restart_flag,
           gpu_restart->restart_flag,
           cpu_restart->current_gap,
           gpu_restart->current_gap,
           cpu_restart->best_gap,
           gpu_restart->best_gap,
           cpu_workspace->sigma,
           gpu_shadow->sigma,
           cpu_residuals->KKTx_and_gap_org_bar,
           gpu_residuals->KKTx_and_gap_org_bar);
}

int main(int argc, char **argv)
{
    TraceContext trace;
    HPRQPRestart cpu_restart;
    HPRQPRestart gpu_restart;
    HPRQPResiduals cpu_residuals;
    HPRQPResiduals gpu_residuals;
    const char *mps_path = "model.mps";
    HPRQPStatus status;
    int argi = 1;
    size_t iter;

    trace_context_init(&trace);
    restart_init(&cpu_restart);
    restart_init(&gpu_restart);
    residuals_init(&cpu_residuals);
    residuals_init(&gpu_residuals);

    while (argi < argc) {
        if (strcmp(argv[argi], "--max-iter") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &trace.params.max_iter)) {
                print_usage(argv[0]);
                trace_context_free(&trace);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--check-iter") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &trace.params.check_iter)) {
                print_usage(argv[0]);
                trace_context_free(&trace);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--device-number") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &trace.params.device_number)) {
                print_usage(argv[0]);
                trace_context_free(&trace);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-q") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &trace.params.spmv_mode_q)) {
                print_usage(argv[0]);
                trace_context_free(&trace);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-a") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &trace.params.spmv_mode_a)) {
                print_usage(argv[0]);
                trace_context_free(&trace);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-scaling") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &trace.params.use_scaling)) {
                print_usage(argv[0]);
                trace_context_free(&trace);
                return 1;
            }
            argi += 2;
        } else if (argv[argi][0] != '-') {
            mps_path = argv[argi];
            argi += 1;
        } else {
            print_usage(argv[0]);
            trace_context_free(&trace);
            return 1;
        }
    }

    if (trace.params.check_iter == 0) {
        fprintf(stderr, "trace_restart_compare requires --check-iter > 0\n");
        trace_context_free(&trace);
        return 1;
    }

    status = trace_context_setup(&trace, mps_path);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "setup failed for '%s': %s\n", mps_path, status_name(status));
        trace_context_free(&trace);
        return 1;
    }

    status = residuals_compute(&trace.cpu_workspace, &trace.model, &trace.scaling, &cpu_residuals, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "initial cpu residuals failed: %s\n", status_name(status));
        trace_context_free(&trace);
        return 1;
    }
    status = gpu_compute_residuals_trace(&trace, &gpu_residuals, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "initial gpu residuals failed: %s\n", status_name(status));
        trace_context_free(&trace);
        return 1;
    }

    printf("restart trace\n");
    printf("  model: %s\n", mps_path);
    printf("  max_iter: %zu\n", trace.params.max_iter);
    printf("  check_iter: %zu\n", trace.params.check_iter);
    printf("  spmv_mode_q: %d\n", (int)trace.gpu_workspace.spmv_mode_q);
    printf("  spmv_mode_a: %d\n", (int)trace.gpu_workspace.spmv_mode_a);
    print_trace_header();
    print_trace_row(0,
                    &cpu_restart,
                    &gpu_restart,
                    &trace.cpu_workspace,
                    &trace.gpu_shadow,
                    &cpu_residuals,
                    &gpu_residuals);

    for (iter = 0; iter < trace.params.max_iter; ++iter) {
        if (iter > 0 && iter % trace.params.check_iter == 0) {
            status = residuals_compute(&trace.cpu_workspace, &trace.model, &trace.scaling, &cpu_residuals, iter);
            if (status != HPRQP_STATUS_OK) {
                fprintf(stderr, "cpu residuals failed at iter %zu: %s\n", iter, status_name(status));
                trace_context_free(&trace);
                return 1;
            }
            status = gpu_compute_residuals_trace(&trace, &gpu_residuals, iter);
            if (status != HPRQP_STATUS_OK) {
                fprintf(stderr, "gpu residuals failed at iter %zu: %s\n", iter, status_name(status));
                trace_context_free(&trace);
                return 1;
            }
        }

        restart_check(&cpu_restart, iter, trace.params.check_iter, trace.cpu_workspace.sigma);
        restart_check(&gpu_restart, iter, trace.params.check_iter, trace.gpu_shadow.sigma);

        status = sigma_update(&trace.params, &cpu_restart, &trace.cpu_workspace, &cpu_residuals);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "cpu sigma_update failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }
        status = sigma_update(&trace.params, &gpu_restart, &trace.gpu_shadow, &gpu_residuals);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "gpu sigma_update failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }
        status = gpu_sync_sigma_and_factors(&trace.gpu_workspace, &trace.gpu_shadow);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "gpu sigma sync failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }

        status = restart_apply(&cpu_restart, &trace.cpu_workspace);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "cpu restart_apply failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }
        status = gpu_apply_restart_trace(&trace, &gpu_restart);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "gpu restart_apply failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }

        if (iter > 0 && iter % trace.params.check_iter == 0) {
            print_trace_row(iter,
                            &cpu_restart,
                            &gpu_restart,
                            &trace.cpu_workspace,
                            &trace.gpu_shadow,
                            &cpu_residuals,
                            &gpu_residuals);
        }

        trace.cpu_workspace.to_check = should_check_next_iter(&trace.params, &cpu_restart, iter + 1);
        trace.gpu_shadow.to_check = should_check_next_iter(&trace.params, &gpu_restart, iter + 1);
        trace.gpu_workspace.to_check = trace.gpu_shadow.to_check;

        status = iteration_step(&trace.cpu_workspace, cpu_restart.inner);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "cpu iteration_step failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }
        status = gpu_iteration_step_trace(&trace, gpu_restart.inner);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "gpu iteration_step failed at iter %zu: %s\n", iter, status_name(status));
            trace_context_free(&trace);
            return 1;
        }

        if (cpu_restart.restart_flag > 0 ||
            (trace.params.check_iter > 0 && (iter + 1) % trace.params.check_iter == 0)) {
            restart_post_iteration(&cpu_restart,
                                   iter,
                                   trace.params.check_iter,
                                   compute_m_norm(&trace.cpu_workspace));
        }
        cpu_restart.inner += 1;

        if (gpu_restart.restart_flag > 0 ||
            (trace.params.check_iter > 0 && (iter + 1) % trace.params.check_iter == 0)) {
            if (!trace.gpu_host_state_dirty) {
                status = gpu_workspace_copy_check_state_to_cpu(&trace.gpu_workspace,
                                                               &trace.gpu_shadow);
                if (status != HPRQP_STATUS_OK) {
                    fprintf(stderr, "gpu sync for m_norm failed at iter %zu: %s\n", iter, status_name(status));
                    trace_context_free(&trace);
                    return 1;
                }
                trace.gpu_host_state_dirty = 0;
            }
        }
        if (gpu_restart.restart_flag > 0 ||
            (trace.params.check_iter > 0 && (iter + 1) % trace.params.check_iter == 0)) {
            restart_post_iteration(&gpu_restart,
                                   iter,
                                   trace.params.check_iter,
                                   compute_m_norm(&trace.gpu_shadow));
        }
        gpu_restart.inner += 1;
    }

    trace_context_free(&trace);
    return 0;
}