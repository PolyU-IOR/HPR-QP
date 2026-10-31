#include "internal/hprqp_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct DiffStats {
    const char *name;
    size_t index;
    double cpu_value;
    double gpu_value;
    double max_abs_diff;
} DiffStats;

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

static int parse_double_arg(const char *value, double *out)
{
    char *endptr;
    double parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtod(value, &endptr);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = parsed;
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
    printf("usage: %s [mps_path] [--steps N] [--tolerance X] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1]\n",
           argv0);
}

static void diff_stats_init(DiffStats *stats)
{
    if (stats == NULL) {
        return;
    }
    stats->name = "none";
    stats->index = 0;
    stats->cpu_value = 0.0;
    stats->gpu_value = 0.0;
    stats->max_abs_diff = 0.0;
}

static void update_diff_stats(DiffStats *stats,
                              const char *name,
                              const double *cpu_values,
                              const double *gpu_values,
                              size_t length)
{
    size_t i;

    if (stats == NULL || name == NULL || cpu_values == NULL || gpu_values == NULL) {
        return;
    }

    for (i = 0; i < length; ++i) {
        double diff = fabs(cpu_values[i] - gpu_values[i]);
        if (diff > stats->max_abs_diff) {
            stats->name = name;
            stats->index = i;
            stats->cpu_value = cpu_values[i];
            stats->gpu_value = gpu_values[i];
            stats->max_abs_diff = diff;
        }
    }
}

static DiffStats compare_workspace_state(const HPRQPWorkspace *cpu_workspace,
                                         const HPRQPWorkspace *gpu_shadow)
{
    DiffStats stats;

    diff_stats_init(&stats);
    if (cpu_workspace == NULL || gpu_shadow == NULL) {
        return stats;
    }

    update_diff_stats(&stats, "x", cpu_workspace->x, gpu_shadow->x, cpu_workspace->n);
    update_diff_stats(&stats, "x_bar", cpu_workspace->x_bar, gpu_shadow->x_bar, cpu_workspace->n);
    update_diff_stats(&stats, "x_hat", cpu_workspace->x_hat, gpu_shadow->x_hat, cpu_workspace->n);
    update_diff_stats(&stats, "w", cpu_workspace->w, gpu_shadow->w, cpu_workspace->n);
    update_diff_stats(&stats, "w_bar", cpu_workspace->w_bar, gpu_shadow->w_bar, cpu_workspace->n);
    update_diff_stats(&stats, "Qw", cpu_workspace->Qw, gpu_shadow->Qw, cpu_workspace->n);
    update_diff_stats(&stats, "Qw_bar", cpu_workspace->Qw_bar, gpu_shadow->Qw_bar, cpu_workspace->n);
    update_diff_stats(&stats, "ATy", cpu_workspace->ATy, gpu_shadow->ATy, cpu_workspace->n);
    update_diff_stats(&stats, "ATy_bar", cpu_workspace->ATy_bar, gpu_shadow->ATy_bar, cpu_workspace->n);
    update_diff_stats(&stats, "tempv", cpu_workspace->tempv, gpu_shadow->tempv, cpu_workspace->n);

    if (cpu_workspace->m > 0) {
        update_diff_stats(&stats, "y", cpu_workspace->y, gpu_shadow->y, cpu_workspace->m);
        update_diff_stats(&stats, "y_bar", cpu_workspace->y_bar, gpu_shadow->y_bar, cpu_workspace->m);
        update_diff_stats(&stats, "Ax", cpu_workspace->Ax, gpu_shadow->Ax, cpu_workspace->m);
    }

    return stats;
}

static void print_diff_summary(size_t step, const DiffStats *stats)
{
    if (stats == NULL) {
        return;
    }

    printf("step=%zu first_drift_vector=%s index=%zu cpu=%.17g gpu=%.17g abs_diff=%.17g\n",
           step,
           stats->name,
           stats->index,
           stats->cpu_value,
           stats->gpu_value,
           stats->max_abs_diff);
}

int main(int argc, char **argv)
{
    HPRQPModel model;
    HPRQPScaling scaling;
    HPRQPWorkspace cpu_workspace;
    HPRQPWorkspace gpu_shadow;
    HPRQPGPUContext gpu_context;
    HPRQPGPUProblem gpu_problem;
    HPRQPGPUWorkspace gpu_workspace;
    HPRQPParams params;
    HPRQPStatus status;
    const char *mps_path = "model.mps";
    size_t steps = 100;
    double tolerance = 1e-12;
    int argi = 1;
    size_t step;

    model_init(&model);
    scaling_init(&scaling);
    workspace_init(&cpu_workspace);
    workspace_init(&gpu_shadow);
    gpu_context_init(&gpu_context);
    gpu_problem_init(&gpu_problem);
    gpu_workspace_init(&gpu_workspace);
    params_set_defaults(&params);
    params.verbose = 0;

    while (argi < argc) {
        if (strcmp(argv[argi], "--steps") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &steps)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--tolerance") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &tolerance)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--device-number") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.device_number)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-q") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &params.spmv_mode_q)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-a") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &params.spmv_mode_a)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-scaling") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_scaling)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (argv[argi][0] != '-') {
            mps_path = argv[argi];
            argi += 1;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    status = build_model_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "failed to load MPS file '%s' with status %s\n", mps_path, status_name(status));
        return 1;
    }

    status = scale_problem(&model, &params, &scaling);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "scale_problem failed: %s\n", status_name(status));
        goto fail;
    }

    status = workspace_allocate(&model, &params, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "workspace_allocate cpu failed: %s\n", status_name(status));
        goto fail;
    }
    status = workspace_allocate(&model, &params, &gpu_shadow);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "workspace_allocate gpu shadow failed: %s\n", status_name(status));
        goto fail;
    }

    status = gpu_context_open(&gpu_context, params.device_number);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_context_open failed: %s\n", status_name(status));
        goto fail;
    }
    status = gpu_problem_upload(&gpu_problem, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_problem_upload failed: %s\n", status_name(status));
        goto fail;
    }
    status = gpu_workspace_allocate(&gpu_workspace, &gpu_problem, &params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_workspace_allocate failed: %s\n", status_name(status));
        goto fail;
    }
    status = gpu_workspace_upload_scaling(&gpu_workspace, &scaling);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_workspace_upload_scaling failed: %s\n", status_name(status));
        goto fail;
    }
    status = gpu_workspace_prepare_spmv(&gpu_workspace, &gpu_problem, 0);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_workspace_prepare_spmv failed: %s\n", status_name(status));
        goto fail;
    }
    status = gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_workspace_sync_from_cpu failed: %s\n", status_name(status));
        goto fail;
    }
    status = gpu_workspace_select_spmv_modes(&gpu_workspace, &gpu_problem, &params);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_workspace_select_spmv_modes failed: %s\n", status_name(status));
        goto fail;
    }

    cpu_workspace.to_check = 0;
    status = gpu_workspace_sync_from_cpu(&gpu_workspace, &cpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "gpu_workspace_sync_from_cpu failed after disabling checks: %s\n", status_name(status));
        goto fail;
    }

    printf("unchecked iteration diagnostic\n");
    printf("  model: %s\n", mps_path);
    printf("  steps: %zu\n", steps);
    printf("  tolerance: %.3e\n", tolerance);
    printf("  spmv_mode_q: %d\n", (int)gpu_workspace.spmv_mode_q);
    printf("  spmv_mode_a: %d\n", (int)gpu_workspace.spmv_mode_a);

    for (step = 0; step < steps; ++step) {
        DiffStats stats;

        status = iteration_step(&cpu_workspace, step);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "cpu iteration_step failed at step %zu: %s\n", step, status_name(status));
            goto fail;
        }
        status = gpu_iteration_step(&gpu_workspace, &gpu_problem, step);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "gpu_iteration_step failed at step %zu: %s\n", step, status_name(status));
            goto fail;
        }
        status = gpu_workspace_copy_main_state_to_cpu(&gpu_workspace,
                                                      &gpu_shadow);
        if (status != HPRQP_STATUS_OK) {
            fprintf(stderr, "gpu_workspace_copy_main_state_to_cpu failed at step %zu: %s\n", step, status_name(status));
            goto fail;
        }

        stats = compare_workspace_state(&cpu_workspace, &gpu_shadow);
        if (stats.max_abs_diff > tolerance) {
            print_diff_summary(step + 1, &stats);
            goto success;
        }
    }

    printf("no drift above tolerance after %zu unchecked steps\n", steps);

success:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&gpu_context);
    workspace_free(&gpu_shadow);
    workspace_free(&cpu_workspace);
    scaling_free(&scaling);
    model_free(&model);
    return 0;

fail:
    gpu_workspace_free(&gpu_workspace);
    gpu_problem_free(&gpu_problem);
    gpu_context_free(&gpu_context);
    workspace_free(&gpu_shadow);
    workspace_free(&cpu_workspace);
    scaling_free(&scaling);
    model_free(&model);
    return 1;
}