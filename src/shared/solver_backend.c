#include "hprqp.h"

#include "cpu/control.h"
#include "cpu/residuals.h"
#include "solver_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

static double current_time_seconds(void)
{
    struct timeval tv;

    if (gettimeofday(&tv, NULL) != 0) {
        return 0.0;
    }

    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static double elapsed_seconds(double start_time)
{
    double now = current_time_seconds();

    if (now <= 0.0 || start_time <= 0.0) {
        return 0.0;
    }

    return now - start_time;
}

typedef struct IterSnapshot {
    int valid;
    size_t iter;
    double time;
    HPRQPResiduals residuals;
    double *x_bar;
    double *w_bar;
    double *z_bar;
    double *y_bar;
} IterSnapshot;

static void snapshot_init(IterSnapshot *snapshot)
{
    if (snapshot == NULL) {
        return;
    }
    memset(snapshot, 0, sizeof(*snapshot));
}

static void snapshot_free(IterSnapshot *snapshot)
{
    if (snapshot == NULL) {
        return;
    }
    free(snapshot->x_bar);
    free(snapshot->w_bar);
    free(snapshot->z_bar);
    free(snapshot->y_bar);
    snapshot_init(snapshot);
}

static HPRQPStatus snapshot_allocate(IterSnapshot *snapshot, size_t n, size_t m)
{
    if (snapshot == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    snapshot->x_bar = n > 0 ? (double *)calloc(n, sizeof(double)) : NULL;
    snapshot->w_bar = n > 0 ? (double *)calloc(n, sizeof(double)) : NULL;
    snapshot->z_bar = n > 0 ? (double *)calloc(n, sizeof(double)) : NULL;
    snapshot->y_bar = m > 0 ? (double *)calloc(m, sizeof(double)) : NULL;

    if ((n > 0 && (snapshot->x_bar == NULL || snapshot->w_bar == NULL || snapshot->z_bar == NULL)) ||
        (m > 0 && snapshot->y_bar == NULL)) {
        snapshot_free(snapshot);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    return HPRQP_STATUS_OK;
}

static void snapshot_update(IterSnapshot *snapshot,
                            const HPRQPResiduals *residuals,
                            size_t iter,
                            double elapsed,
                            const double *x_bar,
                            const double *w_bar,
                            const double *z_bar,
                            const double *y_bar,
                            size_t n,
                            size_t m)
{
    if (snapshot == NULL || residuals == NULL) {
        return;
    }

    snapshot->valid = 1;
    snapshot->iter = iter;
    snapshot->time = elapsed;
    snapshot->residuals = *residuals;
    if (n > 0 && x_bar != NULL && w_bar != NULL && z_bar != NULL) {
        memcpy(snapshot->x_bar, x_bar, n * sizeof(double));
        memcpy(snapshot->w_bar, w_bar, n * sizeof(double));
        memcpy(snapshot->z_bar, z_bar, n * sizeof(double));
    }
    if (m > 0 && y_bar != NULL) {
        memcpy(snapshot->y_bar, y_bar, m * sizeof(double));
    }
}

static size_t adaptive_print_step(size_t iter)
{
    size_t step = 10;

    if (iter == 0) {
        return step;
    }

    while (iter / step >= 100 && step <= SIZE_MAX / 10) {
        step *= 10;
    }

    return step;
}

static int should_print_current_iter(const HPRQPParams *params,
                                     size_t iter,
                                     double elapsed,
                                     size_t max_iter)
{
    if (params->print_frequency == -1 &&
        (iter % adaptive_print_step(iter) == 0 || iter == max_iter || elapsed > params->time_limit)) {
        return 1;
    }
    if (params->print_frequency > 0 &&
        (iter % (size_t)params->print_frequency == 0 || iter == max_iter || elapsed > params->time_limit)) {
        return 1;
    }

    return 0;
}

static int should_check_next_iter(const HPRQPParams *params,
                                  const HPRQPRestart *restart_info,
                                  size_t next_iter)
{
    int to_check = 0;

    if (params->check_iter > 0 && next_iter % params->check_iter == 0) {
        to_check = 1;
    }
    if (restart_info->restart_flag > 0) {
        to_check = 1;
    }
    if (params->print_frequency == -1 && next_iter % adaptive_print_step(next_iter) == 0) {
        to_check = 1;
    }
    if (params->print_frequency > 0 && next_iter % (size_t)params->print_frequency == 0) {
        to_check = 1;
    }

    return to_check;
}

static void update_milestones(const HPRQPResiduals *residuals,
                              size_t iter,
                              double elapsed,
                              HPRQPResults *results,
                              int *first_4,
                              int *first_6)
{
    if (*first_4 && residuals->KKTx_and_gap_org_bar < 1e-4) {
        results->iter_4 = iter;
        results->time_4 = elapsed;
        *first_4 = 0;
    }
    if (*first_6 && residuals->KKTx_and_gap_org_bar < 1e-6) {
        results->iter_6 = iter;
        results->time_6 = elapsed;
        *first_6 = 0;
    }
}

static HPRQPStatus check_termination_status(const HPRQPResiduals *residuals,
                                            size_t iter,
                                            double elapsed,
                                            const HPRQPParams *params)
{
    /* fmax can mask a NaN in one component.  Never certify such a point. */
    if (residuals == NULL || params == NULL ||
        !isfinite(residuals->err_Rp_org_bar) ||
        !isfinite(residuals->err_QC_org_bar) ||
        !isfinite(residuals->err_Rd_org_bar) ||
        !isfinite(residuals->primal_obj_bar) ||
        !isfinite(residuals->dual_obj_bar) ||
        !isfinite(residuals->rel_gap_bar) ||
        !isfinite(residuals->KKTx_and_gap_org_bar)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (residuals->KKTx_and_gap_org_bar < params->stoptol) {
        return HPRQP_STATUS_OPTIMAL;
    }
    if (iter == params->max_iter) {
        return HPRQP_STATUS_MAX_ITER;
    }
    if (elapsed > params->time_limit) {
        return HPRQP_STATUS_TIME_LIMIT;
    }

    return HPRQP_STATUS_CONTINUE;
}

static void print_iteration_log(size_t iter,
                                const HPRQPResiduals *residuals,
                                const HPRQPWorkspace *workspace,
                                int has_qc,
                                double elapsed)
{
    if (has_qc) {
        printf("%5zu    %3.2e    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f\n",
               iter,
               residuals->err_Rp_org_bar,
               residuals->err_QC_org_bar,
               residuals->err_Rd_org_bar,
               residuals->primal_obj_bar,
               residuals->dual_obj_bar,
               residuals->rel_gap_bar,
               workspace->sigma,
               elapsed);
        return;
    }
    printf("%5zu    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f\n",
           iter,
           residuals->err_Rp_org_bar,
           residuals->err_Rd_org_bar,
           residuals->primal_obj_bar,
           residuals->dual_obj_bar,
           residuals->rel_gap_bar,
           workspace->sigma,
           elapsed);
}

static const char *spmv_mode_name(HPRQPSpmvMode mode)
{
    switch (mode) {
    case HPRQP_SPMV_MODE_AUTO:
        return "auto";
    case HPRQP_SPMV_MODE_CUSPARSE:
        return "cusparse";
    case HPRQP_SPMV_MODE_CUSTOMIZED:
        return "customized";
    default:
        return "unknown";
    }
}

static const char *method_name(HPRQPMethod method)
{
    return method == HPRQP_METHOD_PARALLEL ? "parallel" : "sgs";
}

static const char *parallel_majorizer_name(HPRQPParallelMajorizerMode mode)
{
    if (mode == HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER) return "coupling_power";
    if (mode == HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT) return "two_moment";
    if (mode == HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR) return "moment_selector";
    return "worst_case";
}

static const char *qc_majorizer_name(HPRQPQCMajorizerMode mode)
{
    if (mode == HPRQP_QC_MAJORIZER_PAIRWISE) return "pairwise";
    if (mode == HPRQP_QC_MAJORIZER_JOINT_F) return "joint_f";
    if (mode == HPRQP_QC_MAJORIZER_SCALAR_B) return "scalar_b";
    if (mode == HPRQP_QC_MAJORIZER_DIAG_BQ) return "diag_bq";
    return "auto";
}

static const char *tau_mode_name(HPRQPTauMode mode)
{
    switch (mode) {
    case HPRQP_TAU_MODE_UNIT:
        return "unit";
    case HPRQP_TAU_MODE_FIXED:
        return "fixed";
    case HPRQP_TAU_MODE_BALANCED:
        return "balanced";
    case HPRQP_TAU_MODE_ADAPTIVE_RESTART:
        return "adaptive_restart";
    default:
        return "unknown";
    }
}

void print_model_info(const HPRQPModel *model)
{
    if (model == NULL) {
        return;
    }

    printf("Model info:\n");
    printf("  variables: %zu\n", model->Q.ncol);
    printf("  constraints: %zu\n", model->A.nrow);
    printf("  nnz(Q): %zu\n", model->Q.nnz);
    printf("  nnz(A): %zu\n", model->A.nnz);
    printf("  quadratic constraints: %zu\n", model->qc.count);
    if (model->qc.count > 0U) {
        printf("  rotated-cone dimension: %zu\n", model->qc.cone_dim);
    }
}

void print_params_info(const HPRQPParams *params)
{
    if (params == NULL) {
        return;
    }

    printf("Parameter info:\n");
    printf("  stoptol: %.12g\n", params->stoptol);
    printf("  use_gpu: %s\n", params->use_gpu ? "on" : "off");
    printf("  device_number: %d\n", params->device_number);
    printf("  max_iter: %zu\n", params->max_iter);
    printf("  time_limit: %.12g\n", params->time_limit);
    printf("  check_iter: %zu\n", params->check_iter);
    printf("  print_frequency: %d\n", params->print_frequency);
    printf("  use_scaling: %s\n", params->use_scaling ? "on" : "off");
    printf("  use_cr_scaling: %s\n", params->use_cr_scaling ? "on" : "off");
    printf("  cr_iterations: %zu\n", params->cr_iterations);
    printf("  b/c scaling: %s\n", params->use_bc_scaling ? "on" : "off");
    printf("  initial sigma: %.12g\n", params->initial_sigma);
    printf("  SOCP restart policy: %s\n",
           params->use_socp_restart_policy ? "on" : "off");
    printf("  SOCP sigma policy: %s\n",
           params->use_socp_sigma_policy ? "on" : "off");
    printf("  verbose: %s\n", params->verbose ? "on" : "off");
    printf("  spmv mode Q: %s\n", spmv_mode_name(params->spmv_mode_q));
    printf("  spmv mode A: %s\n", spmv_mode_name(params->spmv_mode_a));
    printf("  method: %s\n", method_name(params->method));
    if (params->method == HPRQP_METHOD_PARALLEL) {
        printf("  parallel majorizer: %s\n", parallel_majorizer_name(params->parallel_majorizer_mode));
        printf("  QC majorizer: %s\n", qc_majorizer_name(params->qc_majorizer_mode));
        printf("  tau mode: %s\n", tau_mode_name(params->tau_mode));
        printf("  tau bounds: [%.12g, %.12g]\n", params->tau_min, params->tau_max);
        printf("  eig factor: %.12g\n", params->eig_factor);
        printf("  adaptation energy floor: %.12g\n", params->parallel_adaptation_energy_floor);
    }
}

static HPRQPStatus allocate_results_vectors(HPRQPResults *results, size_t n, size_t m)
{
    results->x = n > 0 ? (double *)calloc(n, sizeof(double)) : NULL;
    results->w = n > 0 ? (double *)calloc(n, sizeof(double)) : NULL;
    results->z = n > 0 ? (double *)calloc(n, sizeof(double)) : NULL;
    results->y = m > 0 ? (double *)calloc(m, sizeof(double)) : NULL;

    if ((n > 0 && (results->x == NULL || results->w == NULL || results->z == NULL)) ||
        (m > 0 && results->y == NULL)) {
        results_free(results);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus collect_results(const HPRQPWorkspace *workspace,
                                   const HPRQPScaling *scaling,
                                   const HPRQPResiduals *residuals,
                                   size_t iter,
                                   double elapsed,
                                   HPRQPStatus status,
                                   const double *x_bar,
                                   const double *w_bar,
                                   const double *z_bar,
                                   const double *y_bar,
                                   HPRQPResults *results)
{
    HPRQPStatus alloc_status;
    size_t iter_4 = results->iter_4;
    size_t iter_6 = results->iter_6;
    double time_4 = results->time_4;
    double time_6 = results->time_6;

    results_free(results);
    alloc_status = allocate_results_vectors(results, workspace->n, workspace->m);
    if (alloc_status != HPRQP_STATUS_OK) {
        return alloc_status;
    }

    results->iter = iter;
    results->time = elapsed;
    results->iter_4 = iter_4 == 0 ? iter : iter_4;
    results->iter_6 = iter_6 == 0 ? iter : iter_6;
    results->time_4 = time_4 == 0.0 ? elapsed : time_4;
    results->time_6 = time_6 == 0.0 ? elapsed : time_6;
    results->primal_obj = residuals->primal_obj_bar;
    results->residuals = residuals->KKTx_and_gap_org_bar;
    results->primal_residual = residuals->err_Rp_org_bar;
    results->qc_residual = residuals->err_QC_org_bar;
    results->dual_residual = residuals->err_Rd_org_bar;
    results->gap = residuals->rel_gap_bar;
    results->method = workspace->method;
    results->lambda_max_A = workspace->lambda_max_A;
    results->lambda_max_Q = workspace->lambda_max_Q;
    results->lambda_max_AQA = workspace->lambda_max_AQA;
    results->lambda_max_AQ2A = workspace->lambda_max_AQ2A;
    results->zeta_AB = workspace->zeta_AB;
    results->zeta_AQ = workspace->zeta_AQ;
    results->zeta_BQ = workspace->zeta_BQ;
    results->tau_AB = workspace->tau_AB;
    results->tau_AQ = workspace->tau_AQ;
    results->tau_BQ = workspace->tau_BQ;
    results->tau = workspace->tau;
    results->d_y = workspace->d_y;
    results->d_s = workspace->d_s;
    results->d_w = workspace->d_w;
    results->status = status;

    (void)scaling;

    if (workspace->n > 0) {
        memcpy(results->x, x_bar, workspace->n * sizeof(double));
        memcpy(results->w, w_bar, workspace->n * sizeof(double));
        memcpy(results->z, z_bar, workspace->n * sizeof(double));
    }
    if (workspace->m > 0) {
        memcpy(results->y, y_bar, workspace->m * sizeof(double));
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus solve_with_backend(HPRQPSolverBackendContext *context,
                               const HPRQPSolverBackendOps *ops,
                               const char *mode_label,
                               HPRQPResults *results)
{
    HPRQPResiduals residual_state;
    HPRQPRestart restart_info;
    IterSnapshot best_snapshot;
    IterSnapshot current_snapshot;
    double start_time;
    size_t iter;
    HPRQPStatus status;
    HPRQPStatus return_status;
    HPRQPStatus final_status;
    size_t result_iter;
    double result_time;
    const HPRQPResiduals *result_residuals;
    const double *result_x_bar;
    const double *result_w_bar;
    const double *result_z_bar;
    const double *result_y_bar;
    int collect_output = 0;
    int first_4 = 1;
    int first_6 = 1;
    int print_yes = 0;
    double iter_elapsed = 0.0;
    size_t step_count;
    size_t target_iter;
    const int use_best_snapshot = 0;
    const int trace_restart = getenv("HPRQP_TRACE_RESTART") != NULL;

    if (context == NULL || ops == NULL || results == NULL || context->workspace == NULL ||
        context->model == NULL || context->scaling == NULL || context->params == NULL ||
        ops->compute_residuals == NULL || ops->copy_output_iterates == NULL ||
        ops->update_sigma == NULL || ops->restart_apply == NULL ||
        ops->compute_m_norm == NULL || ops->iteration_step == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    residuals_init(&residual_state);
    restart_init(&restart_info);
    snapshot_init(&best_snapshot);
    snapshot_init(&current_snapshot);
    start_time = current_time_seconds();
    results_init(results);
    return_status = HPRQP_STATUS_OK;

    if (use_best_snapshot) {
        status = snapshot_allocate(&best_snapshot, context->workspace->n, context->workspace->m);
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }
    }
    status = snapshot_allocate(&current_snapshot, context->workspace->n, context->workspace->m);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }

    status = ops->compute_residuals(context, &residual_state, 0);
    if (status != HPRQP_STATUS_OK) {
        return_status = status;
        goto cleanup;
    }
    if (use_best_snapshot) {
        status = ops->copy_output_iterates(context,
                                           best_snapshot.x_bar,
                                           best_snapshot.w_bar,
                                           best_snapshot.z_bar,
                                           best_snapshot.y_bar);
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }
        snapshot_update(&best_snapshot,
                        &residual_state,
                        0,
                        elapsed_seconds(start_time),
                        best_snapshot.x_bar,
                        best_snapshot.w_bar,
                        best_snapshot.z_bar,
                        best_snapshot.y_bar,
                        context->workspace->n,
                        context->workspace->m);
    }
    update_milestones(&residual_state, 0, elapsed_seconds(start_time), results, &first_4, &first_6);
    if (context->params->verbose) {
        printf("HPRQP SOLVER starts (%s)...\n", mode_label != NULL ? mode_label : "backend mode");
        if (context->model->qc.count > 0U) {
            printf(" iter     errRp        errQC        errRd         p_obj           d_obj          gap        sigma       time\n");
        } else {
            printf(" iter     errRp        errRd         p_obj           d_obj          gap        sigma       time\n");
        }
        print_iteration_log(0, &residual_state, context->workspace,
                            context->model->qc.count > 0U,
                            elapsed_seconds(start_time));
    }

    final_status = HPRQP_STATUS_MAX_ITER;
    restart_info.best_sigma = context->workspace->sigma;
    iter = 0;
    for (;;) {
        print_yes = 0;
        iter_elapsed = elapsed_seconds(start_time);

        if (iter > 0) {
            print_yes = should_print_current_iter(context->params, iter, iter_elapsed, context->params->max_iter);
            if ((context->params->check_iter > 0 && iter % context->params->check_iter == 0) || print_yes) {
                ops->compute_residuals(context, &residual_state, iter);
                if (use_best_snapshot &&
                    (!best_snapshot.valid || residual_state.KKTx_and_gap_org_bar < best_snapshot.residuals.KKTx_and_gap_org_bar)) {
                    ops->copy_output_iterates(context,
                                              best_snapshot.x_bar,
                                              best_snapshot.w_bar,
                                              best_snapshot.z_bar,
                                              best_snapshot.y_bar);
                    snapshot_update(&best_snapshot,
                                    &residual_state,
                                    iter,
                                    elapsed_seconds(start_time),
                                    best_snapshot.x_bar,
                                    best_snapshot.w_bar,
                                    best_snapshot.z_bar,
                                    best_snapshot.y_bar,
                                    context->workspace->n,
                                    context->workspace->m);
                }
                update_milestones(&residual_state, iter, elapsed_seconds(start_time), results, &first_4, &first_6);
            } else {
                residual_state.is_updated = 0;
            }
        }

        final_status = check_termination_status(&residual_state, iter, iter_elapsed, context->params);

        if (context->model->qc.count > 0U &&
            context->params->use_socp_restart_policy) {
            restart_check_socp(&restart_info,
                               iter,
                               context->params->check_iter,
                               context->workspace->sigma,
                               residual_state.KKTx_and_gap_org_bar);
        } else {
            restart_check(&restart_info,
                          iter,
                          context->params->check_iter,
                          context->workspace->sigma);
        }

        {
            const double sigma_before = context->workspace->sigma;
            status = ops->update_sigma(context, &restart_info, &residual_state);
            if (trace_restart && restart_info.restart_flag > 0 &&
                status == HPRQP_STATUS_OK) {
                fprintf(stderr,
                        "HPRQP_RESTART iter=%zu type=%d inner=%zu "
                        "fixed_gap=%.17g last_gap=%.17g save_gap=%.17g "
                        "kkt=%.17g sigma_before=%.17g sigma_after=%.17g\n",
                        iter, restart_info.restart_flag, restart_info.inner,
                        restart_info.current_gap, restart_info.last_gap,
                        restart_info.save_gap,
                        residual_state.KKTx_and_gap_org_bar,
                        sigma_before, context->workspace->sigma);
            }
        }
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }

        status = ops->restart_apply(context, &restart_info);
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }

        if (context->params->verbose && (print_yes || final_status != HPRQP_STATUS_CONTINUE)) {
            print_iteration_log(iter, &residual_state, context->workspace,
                                context->model->qc.count > 0U,
                                elapsed_seconds(start_time));
        }

        if (final_status != HPRQP_STATUS_CONTINUE) {
            break;
        }

        step_count = 1U;
        if (ops->iteration_batch != NULL &&
            restart_info.restart_flag == 0 &&
            context->params->print_frequency >= 0) {
            size_t available = 128U;

            if (context->params->check_iter > 0) {
                const size_t remainder = iter % context->params->check_iter;
                const size_t to_check =
                    remainder == 0U
                        ? context->params->check_iter
                        : context->params->check_iter - remainder;
                if (available > to_check) {
                    available = to_check;
                }
            }
            if (context->params->print_frequency > 0) {
                const size_t print_frequency =
                    (size_t)context->params->print_frequency;
                const size_t remainder = iter % print_frequency;
                const size_t to_print =
                    remainder == 0U
                        ? print_frequency
                        : print_frequency - remainder;
                if (available > to_print) {
                    available = to_print;
                }
            }
            if (context->params->max_iter > iter &&
                available > context->params->max_iter - iter) {
                available = context->params->max_iter - iter;
            }
            if (available > 1U) {
                step_count = available;
            }
        }
        target_iter = iter + step_count;
        context->workspace->to_check =
            should_check_next_iter(context->params,
                                   &restart_info,
                                   target_iter);

        if (step_count > 1U) {
            status = ops->iteration_batch(context,
                                          restart_info.inner,
                                          step_count);
            if (status == HPRQP_STATUS_NOT_IMPLEMENTED) {
                step_count = 1U;
                target_iter = iter + 1U;
                context->workspace->to_check =
                    should_check_next_iter(context->params,
                                           &restart_info,
                                           target_iter);
                status = ops->iteration_step(context,
                                             restart_info.inner);
            }
        } else {
            status = ops->iteration_step(context, restart_info.inner);
        }
        if (status != HPRQP_STATUS_OK) {
            return_status = status;
            goto cleanup;
        }


        if (restart_info.restart_flag > 0 ||
            (context->params->check_iter > 0 &&
             target_iter % context->params->check_iter == 0)) {
            double gap;

            gap = ops->compute_m_norm(context);
            if (context->model->qc.count > 0U &&
                context->params->use_socp_restart_policy) {
                status = restart_post_metric_socp(
                    &restart_info,
                    context->workspace,
                    target_iter - 1U,
                    context->params->check_iter,
                    gap);
            } else {
                status = restart_post_metric(&restart_info,
                                             context->workspace,
                                             target_iter - 1U,
                                             context->params->check_iter,
                                             gap);
            }
            if (status != HPRQP_STATUS_OK) {
                return_status = status;
                goto cleanup;
            }
        }
        restart_info.inner += step_count;
        iter = target_iter;
    }

    result_iter = iter;
    result_time = elapsed_seconds(start_time);
    result_residuals = &residual_state;
    result_x_bar = current_snapshot.x_bar;
    result_w_bar = current_snapshot.w_bar;
    result_z_bar = current_snapshot.z_bar;
    result_y_bar = current_snapshot.y_bar;

    if (use_best_snapshot && final_status != HPRQP_STATUS_OPTIMAL && best_snapshot.valid) {
        result_iter = best_snapshot.iter;
        result_time = best_snapshot.time;
        result_residuals = &best_snapshot.residuals;
        result_x_bar = best_snapshot.x_bar;
        result_w_bar = best_snapshot.w_bar;
        result_z_bar = best_snapshot.z_bar;
        result_y_bar = best_snapshot.y_bar;
    }

    collect_output = 1;

cleanup:
    if (collect_output) {
        if (!(use_best_snapshot && final_status != HPRQP_STATUS_OPTIMAL && best_snapshot.valid)) {
            status = ops->copy_output_iterates(context,
                                               current_snapshot.x_bar,
                                               current_snapshot.w_bar,
                                               current_snapshot.z_bar,
                                               current_snapshot.y_bar);
            if (status != HPRQP_STATUS_OK) {
                return_status = status;
                snapshot_free(&current_snapshot);
                snapshot_free(&best_snapshot);
                return return_status;
            }
        }
        status = collect_results(context->workspace,
                                 context->scaling,
                                 result_residuals,
                                 result_iter,
                                 result_time,
                                 final_status,
                                 result_x_bar,
                                 result_w_bar,
                                 result_z_bar,
                                 result_y_bar,
                                 results);
        return_status = status == HPRQP_STATUS_OK ? final_status : status;
    }

    snapshot_free(&current_snapshot);
    snapshot_free(&best_snapshot);
    return return_status;
}
